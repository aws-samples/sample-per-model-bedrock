"""A disposable EC2 host for running the samples the way a customer would.

Why this exists at all: the agent's own container is an ARM64 image with a pinned
dependency set. A notebook that imports `openai` and hits a Region-specific endpoint
should be proven on a clean machine with the repository's own requirements.txt, not
inside the agent. So the agent gets a throwaway host, reached over SSM -- no inbound
ports, no SSH key, no public IP needed.

The part that matters operationally is that it always goes away:

* Every instance is tagged, so `testhost_list_orphans` can find one that leaked.
* `testhost_launch` terminates any prior instance from a previous cycle before
  starting a new one, so a crashed run cannot accumulate hosts.
* An hour-long self-terminate is baked into user-data as a dead man's switch, so the
  host dies even if the agent, the session, and the container all disappear.

Three layers because the failure being defended against -- an agent that crashes
mid-cycle and silently leaves a running instance -- is invisible until the bill.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import Annotated, Any

import boto3
from botocore.exceptions import ClientError
from claude_agent_sdk import tool

log = logging.getLogger("pr-agent.testhost")
REGION = os.environ.get("AWS_REGION", "us-east-1")
TAG_KEY = "ManagedBy"
TAG_VALUE = "bedrock-samples-pr-agent"
INSTANCE_TYPE = os.environ.get("TESTHOST_TYPE", "t4g.small")
PROFILE_NAME = os.environ.get("TESTHOST_INSTANCE_PROFILE", "bedrock-samples-pr-agent-testhost")
MAX_LIFETIME_MIN = 60

# Dead man's switch. Runs regardless of what the agent does afterwards.
USER_DATA = f"""#!/bin/bash
set -x
shutdown -P +{MAX_LIFETIME_MIN} &
dnf install -y python3.12 python3.12-pip git >/dev/null 2>&1 || \
  yum install -y python3.12 python3.12-pip git >/dev/null 2>&1
echo READY > /var/tmp/testhost-ready
"""


def _ec2():
    return boto3.client("ec2", region_name=REGION)


def _ssm():
    return boto3.client("ssm", region_name=REGION)


def _text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(payload)}]}


def _find_managed(states: tuple[str, ...] = ("pending", "running", "stopping", "stopped")) -> list[str]:
    got = _ec2().describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [TAG_VALUE]},
            {"Name": "instance-state-name", "Values": list(states)},
        ]
    )
    return [
        i["InstanceId"]
        for r in got.get("Reservations", [])
        for i in r.get("Instances", [])
    ]


def terminate_all() -> list[str]:
    """Terminate every instance this agent owns. Safe to call at any time."""
    ids = _find_managed()
    if ids:
        _ec2().terminate_instances(InstanceIds=ids)
        log.info("terminated %s", ids)
    return ids


def _latest_al2023_arm64() -> str:
    """Resolve the AMI from SSM rather than hardcoding an ID that ages."""
    param = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
    return _ssm().get_parameter(Name=param)["Parameter"]["Value"]


@tool(
    "testhost_launch",
    "Launch a single small EC2 instance for testing samples against Bedrock, reachable "
    "over SSM (no SSH, no public ports). Terminates any host left over from a previous "
    "cycle first. The host self-terminates after an hour whatever happens. Returns the "
    "instance id once SSM can reach it. Call testhost_terminate when you are done -- do "
    "not leave it running.",
    {},
)
async def testhost_launch(args: dict[str, Any]) -> dict[str, Any]:
    try:
        leaked = terminate_all()
        prefix = f"terminated {len(leaked)} leftover host(s) first; " if leaked else ""
        ami = _latest_al2023_arm64()
        run = _ec2().run_instances(
            ImageId=ami,
            InstanceType=INSTANCE_TYPE,
            MinCount=1,
            MaxCount=1,
            IamInstanceProfile={"Name": PROFILE_NAME},
            UserData=USER_DATA,
            InstanceInitiatedShutdownBehavior="terminate",
            MetadataOptions={"HttpTokens": "required"},
            BlockDeviceMappings=[{
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": 20, "VolumeType": "gp3", "Encrypted": True,
                        "DeleteOnTermination": True},
            }],
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": TAG_KEY, "Value": TAG_VALUE},
                    {"Key": "Name", "Value": f"{TAG_VALUE}-testhost"},
                    {"Key": "AutoTerminateAfterMinutes", "Value": str(MAX_LIFETIME_MIN)},
                ],
            }],
        )
        iid = run["Instances"][0]["InstanceId"]
        log.info("launched %s (%s, %s)", iid, INSTANCE_TYPE, ami)

        # Wait for SSM, not just for 'running': the instance is useless until the
        # agent can send it a command.
        deadline = time.time() + 420
        while time.time() < deadline:
            time.sleep(15)
            info = _ssm().describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": [iid]}]
            )
            if info.get("InstanceInformationList"):
                return _text(
                    f"{prefix}instance {iid} ready ({INSTANCE_TYPE}, {ami}). "
                    f"Use testhost_run to execute commands, then testhost_terminate."
                )
        terminate_all()
        return _text(
            f"ERROR: {iid} did not register with SSM within 7 minutes; it has been "
            f"terminated. Check the instance profile {PROFILE_NAME} and VPC endpoints."
        )
    except ClientError as exc:
        return _text(f"ERROR {exc.response['Error']['Code']}: {exc.response['Error']['Message']}")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "testhost_run",
    "Run a shell script on the test host over SSM and return stdout and stderr. Use it "
    "to clone the repo, install requirements.txt, and execute the sample you changed. "
    "Scripts run as root in /root with a 15 minute timeout. Keep each call to one "
    "logical step so a failure tells you which step failed.",
    {
        "instance_id": Annotated[str, "Instance id from testhost_launch."],
        "script": Annotated[str, "Bash script to run. Multi-line is fine."],
    },
)
async def testhost_run(args: dict[str, Any]) -> dict[str, Any]:
    iid = (args.get("instance_id") or "").strip()
    script = args.get("script") or ""
    if not iid or not script:
        return _text("ERROR: instance_id and script are both required")
    if iid not in _find_managed(("running",)):
        return _text(f"ERROR: {iid} is not a running host owned by this agent")
    try:
        send = _ssm().send_command(
            InstanceIds=[iid],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [script], "executionTimeout": ["900"]},
            TimeoutSeconds=900,
        )
        cid = send["Command"]["CommandId"]
        deadline = time.time() + 960
        while time.time() < deadline:
            time.sleep(10)
            try:
                inv = _ssm().get_command_invocation(CommandId=cid, InstanceId=iid)
            except ClientError as exc:
                if exc.response["Error"]["Code"] == "InvocationDoesNotExist":
                    continue
                raise
            if inv["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
                return _text(
                    f"status={inv['Status']} exit={inv.get('ResponseCode')}\n"
                    f"--- stdout ---\n{(inv.get('StandardOutputContent') or '')[:8000]}\n"
                    f"--- stderr ---\n{(inv.get('StandardErrorContent') or '')[:4000]}"
                )
        return _text(f"ERROR: command {cid} did not finish within 16 minutes")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "testhost_terminate",
    "Terminate every test host this agent owns. Call it as soon as testing is done, and "
    "again before you finish the cycle even if you think you already did -- it is "
    "idempotent and costs nothing when there is nothing to terminate.",
    {},
)
async def testhost_terminate(args: dict[str, Any]) -> dict[str, Any]:
    try:
        ids = terminate_all()
        return _text(f"terminated: {ids or 'nothing was running'}")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "testhost_list_orphans",
    "List any test host still alive, with its age. Use it at the start and end of a "
    "cycle to confirm nothing leaked from a previous run.",
    {},
)
async def testhost_list_orphans(args: dict[str, Any]) -> dict[str, Any]:
    try:
        got = _ec2().describe_instances(
            Filters=[
                {"Name": f"tag:{TAG_KEY}", "Values": [TAG_VALUE]},
                {"Name": "instance-state-name",
                 "Values": ["pending", "running", "stopping", "stopped"]},
            ]
        )
        rows = [
            f"{i['InstanceId']} {i['State']['Name']} launched {i['LaunchTime'].isoformat()}"
            for r in got.get("Reservations", []) for i in r.get("Instances", [])
        ]
        return _text("\n".join(rows) if rows else "no test hosts alive")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


testhost_terminate.raw = terminate_all  # type: ignore[attr-defined]
