"""Remove everything deploy.py created, and nothing else.

Every deletion is filtered by the resource name or the ManagedBy tag this project sets.
Secrets are left alone: the deploy key and token are credentials a human created in
GitHub, and deleting them here would silently break the GitHub side.
"""
from __future__ import annotations

import logging

from botocore.exceptions import ClientError

log = logging.getLogger("teardown")

NAME = "bedrock-samples-pr-agent"
RUNTIME_NAME = "bedrock_samples_pr_agent"


def teardown(session, account: str) -> None:
    def attempt(label: str, fn) -> None:
        try:
            fn()
            log.info("  removed %s", label)
        except ClientError as exc:
            log.info("  %s: %s", label, exc.response["Error"]["Code"])
        except Exception as exc:  # noqa: BLE001
            log.info("  %s: %s", label, type(exc).__name__)

    # Test hosts first: they cost money by the second.
    ec2 = session.client("ec2")
    got = ec2.describe_instances(Filters=[
        {"Name": f"tag:ManagedBy", "Values": [NAME]},
        {"Name": "instance-state-name",
         "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    ids = [i["InstanceId"] for r in got["Reservations"] for i in r["Instances"]]
    if ids:
        attempt(f"instances {ids}", lambda: ec2.terminate_instances(InstanceIds=ids))

    sch = session.client("scheduler")
    attempt(f"schedule {NAME}-daily",
            lambda: sch.delete_schedule(Name=f"{NAME}-daily"))

    ac = session.client("bedrock-agentcore-control")
    # A page holds 10 runtimes, so read every page: the account has more than that.
    runtimes = [r for page in ac.get_paginator("list_agent_runtimes").paginate()
                for r in page.get("agentRuntimes", [])]
    for r in runtimes:
        if r["agentRuntimeName"] == RUNTIME_NAME:
            attempt(
                f"runtime {RUNTIME_NAME}",
                lambda rid=r["agentRuntimeId"]: ac.delete_agent_runtime(agentRuntimeId=rid),
            )

    ecr = session.client("ecr")
    attempt(f"ECR repo {NAME}",
            lambda: ecr.delete_repository(repositoryName=NAME, force=True))

    ddb = session.client("dynamodb")
    attempt(f"table {NAME}-state",
            lambda: ddb.delete_table(TableName=f"{NAME}-state"))

    iam = session.client("iam")
    for role in (f"{NAME}-runtime", f"{NAME}-testhost", f"{NAME}-scheduler"):
        if role == f"{NAME}-testhost":
            attempt(
                f"instance profile {role}",
                lambda r=role: (
                    iam.remove_role_from_instance_profile(InstanceProfileName=r, RoleName=r),
                    iam.delete_instance_profile(InstanceProfileName=r),
                ),
            )
        try:
            for p in iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]:
                iam.detach_role_policy(RoleName=role, PolicyArn=p["PolicyArn"])
            for p in iam.list_role_policies(RoleName=role)["PolicyNames"]:
                iam.delete_role_policy(RoleName=role, PolicyName=p)
        except ClientError:
            pass
        attempt(f"role {role}", lambda r=role: iam.delete_role(RoleName=r))

    log.info("  secrets left in place on purpose (they pair with GitHub credentials)")
