#!/usr/bin/env python3
"""Prove the agent's IAM policy allows what it must and refuses what it must not.

    python3 verify_policy.py

Every check is a pair: one call that must be allowed and one that must be denied. A
policy that allows everything passes an allow-only test suite, so an allow-only test
suite is not a test of anything. This is the same rule the container's own tests follow.

It exists because of a real failure. The first version of the policy conditioned
`ec2:RunInstances` on `ec2:InstanceType` against `Resource: "*"`. That key only exists on
the instance resource, so the volume and network-interface parts of the same call were
denied and every launch failed with UnauthorizedOperation -- which reads as "not allowed
to launch instances" rather than "that condition does not apply to this resource".
`simulate-principal-policy` would have shown it in a second.

Uses the IAM policy simulator, so nothing is created, launched, sent or billed.
"""
from __future__ import annotations

import sys

import boto3

from settings import SES_SENDERS

REGION = "us-east-1"
NAME = "bedrock-samples-pr-agent"

session = boto3.Session(region_name=REGION)
ACCOUNT = session.client("sts").get_caller_identity()["Account"]
ROLE = f"arn:aws:iam::{ACCOUNT}:role/{NAME}-runtime"
TESTHOST = f"arn:aws:iam::{ACCOUNT}:role/{NAME}-testhost"
iam = session.client("iam")

failures: list[str] = []


def decision(action: str, arn: str, ctx: dict[str, str] | None = None, role: str = ROLE) -> str:
    kwargs: dict = {"PolicySourceArn": role, "ActionNames": [action], "ResourceArns": [arn]}
    if ctx:
        kwargs["ContextEntries"] = [
            {"ContextKeyName": k, "ContextKeyType": "string", "ContextKeyValues": [v]}
            for k, v in ctx.items()
        ]
    return iam.simulate_principal_policy(**kwargs)["EvaluationResults"][0]["EvalDecision"]


def expect(label: str, want_allowed: bool, action: str, arn: str,
           ctx: dict[str, str] | None = None, role: str = ROLE) -> None:
    got = decision(action, arn, ctx, role)
    ok = (got == "allowed") if want_allowed else (got != "allowed")
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:52} {got}")
    if not ok:
        failures.append(f"{label}: wanted {'allow' if want_allowed else 'deny'}, got {got}")


def main() -> int:
    print(f"simulating {ROLE}\n")

    print("EC2: may launch a small test host, and only a small one")
    for size in ("t4g.small", "t4g.medium"):
        expect(f"RunInstances {size}", True, "ec2:RunInstances",
               f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*", {"ec2:InstanceType": size})
    for size in ("m5.24xlarge", "p5.48xlarge", "p6-b300.48xlarge"):
        expect(f"RunInstances {size} refused", False, "ec2:RunInstances",
               f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*", {"ec2:InstanceType": size})
    # Without these a launch fails even when the instance itself is permitted.
    for res in (f"arn:aws:ec2:{REGION}::image/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT}:volume/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT}:network-interface/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT}:subnet/*",
                f"arn:aws:ec2:{REGION}:{ACCOUNT}:security-group/*"):
        expect(f"RunInstances on {res.rsplit(':', 1)[-1]}", True, "ec2:RunInstances", res,
               {"ec2:InstanceType": "t4g.small"})

    print("\nEC2: may terminate only what it created")
    instance = f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/i-0123456789abcdef0"
    expect("Terminate its own host", True, "ec2:TerminateInstances", instance,
           {"ec2:ResourceTag/ManagedBy": NAME})
    expect("Terminate another owner's host refused", False, "ec2:TerminateInstances",
           instance, {"ec2:ResourceTag/ManagedBy": "someone-else"})
    expect("Terminate an untagged host refused", False, "ec2:TerminateInstances", instance)

    print("\nSES: may email only the verified identities")
    for addr in SES_SENDERS:
        expect(f"SendEmail to {addr}", True, "ses:SendEmail",
               f"arn:aws:ses:{REGION}:{ACCOUNT}:identity/{addr}")
    expect("SendEmail to a stranger refused", False, "ses:SendEmail",
           f"arn:aws:ses:{REGION}:{ACCOUNT}:identity/stranger@example.com")

    print("\nSecrets Manager: may read only its own")
    expect("Read its own token", True, "secretsmanager:GetSecretValue",
           f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:{NAME}/github-token-AbCdEf")
    expect("Read another team's secret refused", False, "secretsmanager:GetSecretValue",
           f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:other-team/prod-db-AbCdEf")

    print("\nDynamoDB: may write only its own ledger")
    expect("Write its own table", True, "dynamodb:PutItem",
           f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{NAME}-state")
    expect("Write another table refused", False, "dynamodb:PutItem",
           f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/some-other-table")

    print("\nIAM: must not be able to widen its own permissions")
    expect("Attach a policy to itself refused", False, "iam:AttachRolePolicy", ROLE)
    expect("Edit its own inline policy refused", False, "iam:PutRolePolicy", ROLE)
    expect("Create a new role refused", False, "iam:CreateRole",
           f"arn:aws:iam::{ACCOUNT}:role/anything")

    print("\nTest host: may run the notebooks on both endpoints, and nothing wider")
    project = f"arn:aws:bedrock-mantle:{REGION}:{ACCOUNT}:project/default"
    expect("mantle CreateInference", True, "bedrock-mantle:CreateInference", project, role=TESTHOST)
    expect("mantle calls with a bearer token", True, "bedrock-mantle:CallWithBearerToken", "*",
           role=TESTHOST)
    expect("runtime Converse", True, "bedrock:Converse",
           f"arn:aws:bedrock:{REGION}::foundation-model/amazon.nova-micro-v1:0", role=TESTHOST)
    expect("runtime calls with a bearer token", True, "bedrock:CallWithBearerToken", "*",
           role=TESTHOST)
    expect("set account data retention refused", False,
           "bedrock-mantle:PutAccountDataRetention", "*", role=TESTHOST)
    expect("start a fine-tuning job refused", False, "bedrock-mantle:CreateFineTuningJob",
           project, role=TESTHOST)
    expect("create an IAM user refused", False, "iam:CreateUser",
           f"arn:aws:iam::{ACCOUNT}:user/anything", role=TESTHOST)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nevery allow held and every deny held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
