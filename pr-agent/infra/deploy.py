#!/usr/bin/env python3
"""Provision and deploy the Bedrock samples PR agent. Idempotent.

    python3 deploy.py --step all          everything, in dependency order
    python3 deploy.py --step iam          just the roles
    python3 deploy.py --step image        build and push the container
    python3 deploy.py --step runtime      create or update the AgentCore runtime
    python3 deploy.py --step schedule     the daily EventBridge rule
    python3 deploy.py --step teardown     remove everything this created

Written as boto3 rather than CDK deliberately: the CLI's CDK path needs an npm registry
this machine cannot reach, and an explicit script makes the IAM policy -- the part that
actually matters for an autonomous agent -- readable in one place.

Every step is safe to re-run. Nothing here deletes a resource it did not create.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import subprocess
import sys
import time

import boto3
from botocore.exceptions import ClientError

from settings import INTERNAL_HOST_SUFFIXES, REGION, SES_SENDERS

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("deploy")

NAME = "bedrock-samples-pr-agent"
RUNTIME_NAME = "bedrock_samples_pr_agent"  # AgentCore requires underscores
TABLE = f"{NAME}-state"
ECR_REPO = NAME
AGENT_ROLE = f"{NAME}-runtime"
TESTHOST_ROLE = f"{NAME}-testhost"
SCHEDULER_ROLE = f"{NAME}-scheduler"
SENDERS = SES_SENDERS  # empty: deploy without email notifications
SCHEDULE_NAME = f"{NAME}-daily"
MODEL = "us.anthropic.claude-opus-5"
TAGS = {"ManagedBy": NAME, "Project": "sample-per-model-bedrock"}

session = boto3.Session(region_name=REGION)
ACCOUNT = session.client("sts").get_caller_identity()["Account"]


def _ok(msg: str) -> None:
    log.info("  %s", msg)


# ------------------------------------------------------------------------------ IAM
def trust(service: str) -> str:
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": service},
            "Action": "sts:AssumeRole",
            # Confused-deputy guard: only this account's AgentCore resources may
            # assume the agent role.
            "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT}}
            if service == "bedrock-agentcore.amazonaws.com" else {},
        }],
    })


def agent_policy() -> str:
    """What the agent may do.

    Bedrock and AgentCore are broad because the agent's job is to probe the service --
    it cannot know in advance which model or API a new announcement will involve. The
    rest is deliberately narrow, and the two that could cost real money or reach real
    people are fenced:

    * EC2 RunInstances is restricted to one small instance type, and terminate is
      restricted by tag, so the agent cannot terminate anything it did not create.
    * SES is restricted to the two verified identities.
    """
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockAndAgentCore",
                "Effect": "Allow",
                # `bedrock-mantle` is a separate IAM service namespace, not a prefix of
                # `bedrock`. Without it the mantle catalogue answers 403 access_denied on
                # `bedrock-mantle:ListModels` while every `bedrock:*` call succeeds, which
                # reads as a broken tool rather than a missing grant. The first selftest
                # against this policy failed exactly there.
                "Action": ["bedrock:*", "bedrock-mantle:*", "bedrock-agentcore:*"],
                "Resource": "*",
            },
            {
                "Sid": "ReadItsOwnSecrets",
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue"],
                "Resource": f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:{NAME}/*",
            },
            {
                "Sid": "Checkpoint",
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Query"],
                "Resource": f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{TABLE}",
            },
            {
                "Sid": "NotifyOnlyVerifiedIdentities",
                "Effect": "Allow",
                "Action": ["ses:SendEmail", "ses:SendRawEmail"],
                "Resource": [
                    f"arn:aws:ses:{REGION}:{ACCOUNT}:identity/{addr}" for addr in SENDERS
                ] or [f"arn:aws:ses:{REGION}:{ACCOUNT}:identity/none.invalid"],  # no senders: grant nothing real
            },
            {
                # Reading verification status is what lets notify_tools drop an
                # unconfirmed recipient instead of losing the whole message: in sandbox
                # mode SES rejects the entire send if any one recipient is unverified.
                # Without this grant the read is denied, the code falls back to trying
                # every address, and one unclicked confirmation link silences the
                # notification completely -- which is what happened on the first two
                # cycles. Not resource-scopable; SES only accepts "*" for these.
                "Sid": "ReadIdentityVerificationStatus",
                "Effect": "Allow",
                "Action": [
                    "ses:GetIdentityVerificationAttributes",
                    "ses:ListIdentities",
                    "ses:GetAccount",
                ],
                "Resource": "*",
            },
            # RunInstances is authorised against several resource types at once, and
            # `ec2:InstanceType` only exists on the instance itself. Putting that condition
            # on `Resource: "*"` therefore denies the volume and network-interface parts of
            # the same call, and the whole launch fails with UnauthorizedOperation -- which
            # reads as "not allowed to launch instances" rather than "the condition does
            # not apply to this resource". So it is split: the size limit is enforced where
            # the key exists, and the supporting resources are allowed without it.
            {
                "Sid": "LaunchOnlySmallInstances",
                "Effect": "Allow",
                "Action": ["ec2:RunInstances"],
                "Resource": f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*",
                "Condition": {
                    "StringEquals": {"ec2:InstanceType": ["t4g.small", "t4g.medium"]},
                },
            },
            {
                "Sid": "LaunchSupportingResources",
                "Effect": "Allow",
                "Action": ["ec2:RunInstances"],
                "Resource": [
                    f"arn:aws:ec2:{REGION}::image/*",
                    f"arn:aws:ec2:{REGION}:{ACCOUNT}:volume/*",
                    f"arn:aws:ec2:{REGION}:{ACCOUNT}:network-interface/*",
                    f"arn:aws:ec2:{REGION}:{ACCOUNT}:subnet/*",
                    f"arn:aws:ec2:{REGION}:{ACCOUNT}:security-group/*",
                    f"arn:aws:ec2:{REGION}:{ACCOUNT}:key-pair/*",
                ],
            },
            {
                "Sid": "TagOnLaunch",
                "Effect": "Allow",
                "Action": ["ec2:CreateTags"],
                "Resource": f"arn:aws:ec2:{REGION}:{ACCOUNT}:*/*",
                "Condition": {"StringEquals": {"ec2:CreateAction": "RunInstances"}},
            },
            {
                "Sid": "TerminateOnlyItsOwn",
                "Effect": "Allow",
                "Action": ["ec2:TerminateInstances", "ec2:StopInstances"],
                "Resource": f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*",
                "Condition": {"StringEquals": {f"ec2:ResourceTag/ManagedBy": NAME}},
            },
            {
                "Sid": "DescribeAndPassProfile",
                "Effect": "Allow",
                "Action": [
                    "ec2:DescribeInstances", "ec2:DescribeImages",
                    "ec2:DescribeInstanceStatus", "ec2:DescribeTags",
                    "iam:PassRole", "ssm:GetParameter",
                ],
                "Resource": "*",
            },
            {
                "Sid": "RunCommandsOnItsOwnHost",
                "Effect": "Allow",
                "Action": [
                    "ssm:SendCommand", "ssm:GetCommandInvocation",
                    "ssm:ListCommandInvocations", "ssm:DescribeInstanceInformation",
                ],
                "Resource": "*",
            },
            {
                "Sid": "Observability",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                    "logs:DescribeLogStreams", "cloudwatch:PutMetricData",
                    "xray:PutTraceSegments", "xray:PutSpans",
                    "xray:PutTelemetryRecords", "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                ],
                "Resource": "*",
            },
            {
                "Sid": "PullItsOwnImage",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken", "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability",
                ],
                "Resource": "*",
            },
        ],
    })


def ensure_role(name: str, service: str, policy: str | None, managed: list[str]) -> str:
    iam = session.client("iam")
    arn = f"arn:aws:iam::{ACCOUNT}:role/{name}"
    try:
        iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=trust(service),
            Description=f"{NAME}: {name}",
            Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()],
        )
        _ok(f"created role {name}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=trust(service))
        _ok(f"role {name} exists; trust policy refreshed")
    if policy:
        iam.put_role_policy(RoleName=name, PolicyName=f"{name}-inline", PolicyDocument=policy)
        _ok(f"inline policy attached to {name}")
    for m in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=m)
    return arn


def step_iam() -> None:
    log.info("IAM")
    ensure_role(AGENT_ROLE, "bedrock-agentcore.amazonaws.com", agent_policy(), [])
    ensure_role(
        TESTHOST_ROLE, "ec2.amazonaws.com", json.dumps({
            "Version": "2012-10-17",
            # What the notebooks call when the host runs them. Every action is named, so
            # a new service action cannot widen the grant; names are from the Service
            # Authorization Reference for bedrock and bedrock-mantle.
            "Statement": [
                {
                    "Sid": "TestHostCallsBedrockRuntime",
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream",
                               "bedrock:Converse", "bedrock:ConverseStream",
                               "bedrock:CallWithBearerToken",
                               "bedrock:ListFoundationModels", "bedrock:GetFoundationModel",
                               "bedrock:ListInferenceProfiles", "bedrock:GetInferenceProfile",
                               "bedrock:ApplyGuardrail", "bedrock:CreateGuardrail",
                               "bedrock:GetGuardrail", "bedrock:ListGuardrails",
                               "bedrock:DeleteGuardrail"],
                    "Resource": "*",
                },
                {
                    "Sid": "TestHostCallsBedrockMantle",
                    "Effect": "Allow",
                    "Action": ["bedrock-mantle:CreateInference", "bedrock-mantle:GetInference",
                               "bedrock-mantle:DeleteInference", "bedrock-mantle:CancelInference",
                               "bedrock-mantle:CountTokens", "bedrock-mantle:GetModel",
                               "bedrock-mantle:ListModels", "bedrock-mantle:CreateProject",
                               "bedrock-mantle:GetProject", "bedrock-mantle:ListProjects",
                               "bedrock-mantle:UpdateProject", "bedrock-mantle:ArchiveProject",
                               "bedrock-mantle:TagResource", "bedrock-mantle:UntagResource",
                               "bedrock-mantle:ListTagsForResource",
                               "bedrock-mantle:ListFiles", "bedrock-mantle:GetFile",
                               "bedrock-mantle:ListFineTuningJobs",
                               "bedrock-mantle:GetFineTuningJob"],
                    "Resource": "arn:aws:bedrock-mantle:*:*:project/*",
                },
                {
                    "Sid": "TestHostMantleAccountScope",
                    "Effect": "Allow",
                    "Action": ["bedrock-mantle:CallWithBearerToken",
                               "bedrock-mantle:GetAccountDataRetention",
                               "bedrock-websearch:InvokeSearch", "bedrock-websearch:InvokeFetch",
                               "cloudwatch:ListMetrics", "servicequotas:ListServiceQuotas"],
                    "Resource": "*",
                },
            ],
        }),
        ["arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"],
    )
    iam = session.client("iam")
    try:
        iam.create_instance_profile(InstanceProfileName=TESTHOST_ROLE)
        iam.add_role_to_instance_profile(
            InstanceProfileName=TESTHOST_ROLE, RoleName=TESTHOST_ROLE
        )
        _ok(f"created instance profile {TESTHOST_ROLE}")
        time.sleep(12)  # instance-profile propagation
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        _ok(f"instance profile {TESTHOST_ROLE} exists")


# -------------------------------------------------------------------------- storage
def step_state() -> None:
    log.info("DynamoDB")
    ddb = session.client("dynamodb")
    try:
        ddb.create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"},
                       {"AttributeName": "sk", "KeyType": "RANGE"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"},
                                  {"AttributeName": "sk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
            SSESpecification={"Enabled": True},
            Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()],
        )
        ddb.get_waiter("table_exists").wait(TableName=TABLE)
        _ok(f"created table {TABLE}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceInUseException":
            raise
        _ok(f"table {TABLE} exists")
    # Point-in-time recovery: the run history is an audit trail of what an autonomous
    # agent did to a public repository, so losing it to a fat-fingered delete is not fine.
    try:
        session.client("dynamodb").update_continuous_backups(
            TableName=TABLE,
            PointInTimeRecoverySpecification={"PointInTimeRecoveryEnabled": True},
        )
        _ok("point-in-time recovery enabled")
    except ClientError as exc:
        log.warning("  PITR: %s", exc.response["Error"]["Code"])


def step_ses() -> None:
    log.info("SES (sandbox)")
    if not SENDERS:
        _ok("no SES_SENDERS configured; email notifications are off")
        return
    ses = session.client("ses")
    verified = set(ses.list_identities()["Identities"])
    for addr in SENDERS:
        if addr in verified:
            attrs = ses.get_identity_verification_attributes(Identities=[addr])
            status = attrs["VerificationAttributes"].get(addr, {}).get("VerificationStatus")
            _ok(f"{addr}: {status}")
            if status != "Success":
                ses.verify_email_identity(EmailAddress=addr)
                _ok(f"{addr}: re-sent confirmation")
        else:
            ses.verify_email_identity(EmailAddress=addr)
            _ok(f"{addr}: confirmation email sent -- MUST be clicked before mail works")


# ---------------------------------------------------------------------------- image
def step_image() -> str:
    log.info("Container image")
    ecr = session.client("ecr")
    try:
        ecr.create_repository(
            repositoryName=ECR_REPO,
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
            tags=[{"Key": k, "Value": v} for k, v in TAGS.items()],
        )
        _ok(f"created ECR repo {ECR_REPO}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "RepositoryAlreadyExistsException":
            raise
        _ok(f"ECR repo {ECR_REPO} exists")

    registry = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
    uri = f"{registry}/{ECR_REPO}:latest"
    token = ecr.get_authorization_token()["authorizationData"][0]["authorizationToken"]
    user, password = base64.b64decode(token).decode().split(":", 1)

    def run(cmd: list[str], **kw) -> None:
        log.info("  $ %s", " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
        proc = subprocess.run(cmd, **kw)  # noqa: S603
        if proc.returncode != 0:
            raise SystemExit(f"command failed: {' '.join(cmd)}")

    run(["finch", "login", "--username", user, "--password-stdin", registry],
        input=password.encode())
    # AgentCore Runtime requires linux/arm64.
    run(["finch", "build", "--platform", "linux/arm64", "-t", uri, "."],
        cwd="../container")
    run(["finch", "push", uri])
    _ok(f"pushed {uri}")
    return uri


# -------------------------------------------------------------------------- runtime
def _find_runtime(ac) -> dict | None:
    """The runtime summary named RUNTIME_NAME, or None.

    A page holds 10 runtimes and the account has more, so read every page. Reading only
    the first would miss the runtime and make step_runtime create a duplicate.
    """
    for page in ac.get_paginator("list_agent_runtimes").paginate():
        for r in page.get("agentRuntimes", []):
            if r["agentRuntimeName"] == RUNTIME_NAME:
                return r
    return None


def step_runtime(image_uri: str | None = None) -> str:
    log.info("AgentCore Runtime")
    ac = session.client("bedrock-agentcore-control")
    uri = image_uri or f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}:latest"
    env = {
        "AWS_REGION": REGION,
        "AGENT_MODEL": MODEL,
        "TARGET_REPO": "aws-samples/sample-per-model-bedrock",
        "INTERNAL_HOST_SUFFIXES": INTERNAL_HOST_SUFFIXES,
        "CHECKPOINT_TABLE": TABLE,
        "SSH_KEY_SECRET": f"{NAME}/github-ssh-key",
        "GH_TOKEN_SECRET": f"{NAME}/github-token",
        "SES_SENDER": SENDERS[0] if SENDERS else "",
        "SES_RECIPIENTS": ",".join(SENDERS),
        "TESTHOST_INSTANCE_PROFILE": TESTHOST_ROLE,
        "AGENT_WORKDIR": "/app/work",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        # Spans and structured logs to AgentCore Observability. Metrics arrive by
        # default; these two need asking for.
        "AGENT_OBSERVABILITY_ENABLED": "true",
        "OTEL_PYTHON_DISTRO": "aws_distro",
        "OTEL_PYTHON_CONFIGURATOR": "aws_configurator",
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_SERVICE_NAME": RUNTIME_NAME,
    }
    common = dict(
        agentRuntimeArtifact={"containerConfiguration": {"containerUri": uri}},
        roleArn=f"arn:aws:iam::{ACCOUNT}:role/{AGENT_ROLE}",
        networkConfiguration={"networkMode": "PUBLIC"},
        protocolConfiguration={"serverProtocol": "HTTP"},
        environmentVariables=env,
        # A cycle can legitimately take hours: it runs notebooks and waits on EC2. The
        # 8-hour ceiling is the service maximum for a microVM session.
        lifecycleConfiguration={"idleRuntimeSessionTimeout": 1800, "maxLifetime": 28800},
    )
    existing = _find_runtime(ac)
    if existing:
        rid = existing["agentRuntimeId"]
        ac.update_agent_runtime(agentRuntimeId=rid, **common)
        _ok(f"updated runtime {RUNTIME_NAME} ({rid})")
    else:
        created = ac.create_agent_runtime(
            agentRuntimeName=RUNTIME_NAME,
            description="Monitors Amazon Bedrock for changes and opens sample PRs",
            tags=TAGS,
            **common,
        )
        rid = created["agentRuntimeId"]
        _ok(f"created runtime {RUNTIME_NAME} ({rid})")

    for _ in range(40):
        state = ac.get_agent_runtime(agentRuntimeId=rid)["status"]
        if state in ("READY", "CREATE_FAILED", "UPDATE_FAILED"):
            _ok(f"status {state}")
            if state != "READY":
                raise SystemExit(f"runtime is {state}; check CloudWatch logs")
            break
        time.sleep(15)
    arn = ac.get_agent_runtime(agentRuntimeId=rid)["agentRuntimeArn"]
    print(f"\nRUNTIME_ARN={arn}")
    return arn


# ------------------------------------------------------------------------- schedule
def step_schedule(runtime_arn: str | None = None, at_iso: str | None = None) -> None:
    log.info("EventBridge Scheduler")
    ac = session.client("bedrock-agentcore-control")
    if not runtime_arn:
        existing = _find_runtime(ac)
        if existing is None:
            raise SystemExit(f"no runtime named {RUNTIME_NAME}; run deploy.py --step runtime")
        runtime_arn = existing["agentRuntimeArn"]
    arn = runtime_arn
    role = ensure_role(
        SCHEDULER_ROLE, "scheduler.amazonaws.com",
        json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["bedrock-agentcore:InvokeAgentRuntime"],
                "Resource": [arn, f"{arn}/*"],
            }],
        }),
        [],
    )
    sch = session.client("scheduler")
    # A one-off test gets its own name. Reusing SCHEDULE_NAME would overwrite the daily
    # schedule with a rule that fires once and then never again -- and the agent would
    # look healthy right up until the day nobody noticed it had stopped.
    name = SCHEDULE_NAME if not at_iso else f"{SCHEDULE_NAME}-oneoff-test"
    expr = f"at({at_iso})" if at_iso else "cron(0 9 * * ? *)"
    # The universal target speaks the API's wire shape, and two details cost a test cycle
    # each to discover:
    #
    # * Field names are PascalCase, not the camelCase boto3 uses. Getting that wrong gives
    #   "Request payload is missing the following field(s): AgentRuntimeArn, Payload",
    #   which reads as if the fields were absent rather than misspelled.
    # * `Payload` is a blob in the API, but the scheduler passes this string through
    #   verbatim -- it does NOT base64-decode it. Sending base64 delivered the literal
    #   text `eyJtb2RlIjogImRhaWx5In0=` to the container, which failed to parse as JSON
    #   before the entrypoint was ever called. So it is the plain JSON string.
    #
    # The session id comes from the scheduler's own execution id, so every run gets a
    # distinct session. Reusing one id would land consecutive days in the same microVM
    # session and inherit yesterday's working directory.
    target = {
        "Arn": "arn:aws:scheduler:::aws-sdk:bedrockagentcore:invokeAgentRuntime",
        "RoleArn": role,
        "Input": json.dumps({
            "AgentRuntimeArn": arn,
            "RuntimeSessionId": "scheduled-<aws.scheduler.execution-id>",
            "Payload": json.dumps({"mode": "daily"}),
        }),
        # No retries. A cycle that failed halfway may have pushed a branch, and a blind
        # replay would either duplicate work or open a second pull request. The
        # notification and the checkpoint are how a failure gets attention.
        "RetryPolicy": {"MaximumRetryAttempts": 0},
    }
    kwargs = dict(
        Name=name,
        ScheduleExpression=expr,
        ScheduleExpressionTimezone="Australia/Sydney",
        FlexibleTimeWindow={"Mode": "OFF"},
        Target=target,
        Description=("One-off test of the scheduler path" if at_iso
                     else "Daily Bedrock samples maintenance cycle"),
        State="ENABLED",
    )
    try:
        sch.create_schedule(**kwargs)
        _ok(f"created schedule {name}: {expr} Australia/Sydney")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConflictException":
            raise
        sch.update_schedule(**kwargs)
        _ok(f"updated schedule {name}: {expr} Australia/Sydney")
    if at_iso:
        _ok("one-off schedule: EventBridge deletes an at() schedule after it fires "
            "only if ActionAfterCompletion is set, so remove it by hand when done: "
            f"aws scheduler delete-schedule --name {name}")


def step_observability() -> None:
    log.info("Observability")
    # Transaction Search is the one-time account setting that lets spans reach the
    # CloudWatch GenAI observability pages. Without it, traces are collected and not
    # displayed, which looks like an instrumentation bug for a long time.
    #
    # Both calls below reject a no-op change with InvalidRequestException, so whether
    # the call succeeded says nothing about whether the account is configured. Read the
    # state back and report that instead -- the desired end state is the fact worth
    # printing, not the API's opinion of this particular request.
    xray = session.client("xray")
    for call in (
        lambda: xray.update_trace_segment_destination(Destination="CloudWatchLogs"),
        lambda: xray.update_indexing_rule(
            Name="Default", Rule={"Probabilistic": {"DesiredSamplingPercentage": 100}}
        ),
    ):
        try:
            call()
        except ClientError as exc:
            log.debug("  %s", exc.response["Error"]["Message"])

    dest = xray.get_trace_segment_destination()
    _ok(f"segment destination: {dest.get('Destination')} ({dest.get('Status')})")
    if dest.get("Destination") != "CloudWatchLogs" or dest.get("Status") != "ACTIVE":
        raise SystemExit(
            "Transaction Search is not active, so spans will be collected but not shown "
            "on the CloudWatch GenAI observability pages. Fix this before relying on "
            "traces to debug a run."
        )
    for rule in xray.get_indexing_rules().get("IndexingRules", []):
        prob = rule.get("Rule", {}).get("Probabilistic", {})
        # ActualSamplingPercentage is absent until the account has been sampling for a
        # while, so read Desired -- it is what was configured, which is what this step
        # is responsible for.
        pct = prob.get("DesiredSamplingPercentage")
        _ok(f"indexing rule {rule.get('Name')}: desired sampling {pct}%")
        if rule.get("Name") == "Default" and pct != 100.0:
            raise SystemExit(
                f"Default indexing rule is sampling {pct}%, so some spans from a run "
                "will be missing. This agent runs once a day; sample all of it."
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--step",
        default="all",
        choices=["all", "iam", "state", "ses", "image", "runtime", "schedule",
                 "observability", "teardown"],
    )
    ap.add_argument("--at", help="One-off schedule time, ISO e.g. 2026-09-08T05:30:00")
    args = ap.parse_args()

    log.info("account %s region %s", ACCOUNT, REGION)
    if args.step == "teardown":
        from teardown import teardown  # noqa: PLC0415
        teardown(session, ACCOUNT)
        return
    if args.step in ("all", "iam"):
        step_iam()
    if args.step in ("all", "state"):
        step_state()
    if args.step in ("all", "ses"):
        step_ses()
    if args.step in ("all", "observability"):
        step_observability()
    uri = None
    if args.step in ("all", "image"):
        uri = step_image()
    arn = None
    if args.step in ("all", "runtime"):
        arn = step_runtime(uri)
    if args.step in ("all", "schedule"):
        step_schedule(arn, args.at)
    log.info("done")


if __name__ == "__main__":
    main()
