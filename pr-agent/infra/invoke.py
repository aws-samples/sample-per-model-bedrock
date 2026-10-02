#!/usr/bin/env python3
"""Invoke the agent by hand, and read what it did.

    python3 invoke.py selftest            synchronous; prints the result
    python3 invoke.py daily               starts a background cycle, returns at once
    python3 invoke.py logs                tail the runtime log group
    python3 invoke.py logs --since 2h
    python3 invoke.py runs                the checkpoint ledger, newest first
    python3 invoke.py observability       prove logs, metrics and traces are arriving

The read timeout is set high because `InvokeAgentRuntime` holds the connection for the
whole synchronous call. botocore defaults to 60 seconds, which is shorter than a selftest
takes -- the first manual invocation of this agent timed out client-side while the agent
carried on working, which looks like a failure and is not one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import boto3
from botocore.config import Config

from settings import REGION

NAME = "bedrock-samples-pr-agent"
RUNTIME_NAME = "bedrock_samples_pr_agent"
TABLE = f"{NAME}-state"

session = boto3.Session(region_name=REGION)


def runtime_arn() -> str:
    ac = session.client("bedrock-agentcore-control")
    # A page holds 10 runtimes, so read every page: the account has more than that.
    for page in ac.get_paginator("list_agent_runtimes").paginate():
        for r in page.get("agentRuntimes", []):
            if r["agentRuntimeName"] == RUNTIME_NAME:
                return r["agentRuntimeArn"]
    raise SystemExit(f"no runtime named {RUNTIME_NAME}; run deploy.py --step runtime")


def cmd_invoke(mode: str) -> int:
    arn = runtime_arn()
    # A session id must be at least 33 characters.
    sid = (uuid.uuid4().hex + uuid.uuid4().hex)[:40]
    client = session.client(
        "bedrock-agentcore",
        config=Config(read_timeout=900, connect_timeout=20, retries={"max_attempts": 0}),
    )
    print(f"invoking {mode} on {arn.rsplit('/', 1)[-1]}  session={sid}")
    started = time.time()
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=arn,
        runtimeSessionId=sid,
        payload=json.dumps({"mode": mode}).encode(),
    )
    raw = resp["response"].read().decode()
    print(f"--- {time.time() - started:.0f}s")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(raw[:8000])
        return 1

    final = data.pop("final", "")
    note = data.pop("note", "")
    print(json.dumps(data, indent=2))
    if note:
        print(f"\n{note}")
    if final:
        print(f"\n{'=' * 78}\n{final}")
    if mode == "daily":
        print("\nFollow it with:  python3 invoke.py logs --since 10m")
    return 0 if data.get("ok") else 1


def cmd_logs(since: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400}
    try:
        seconds = int(since[:-1]) * units[since[-1]]
    except (ValueError, KeyError, IndexError):
        raise SystemExit(f"--since must look like 30m, 2h or 1d, got {since!r}") from None

    rid = runtime_arn().rsplit("/", 1)[-1]
    # The group is the runtime id plus the endpoint name. `runtime-logs` is the log
    # STREAM prefix inside it, not part of the group name -- getting that wrong gives
    # a ResourceNotFoundException that reads like "the agent never ran".
    group = f"/aws/bedrock-agentcore/runtimes/{rid}-DEFAULT"
    logs = session.client("logs")
    start = int((time.time() - seconds) * 1000)
    print(f"{group}  (last {since})\n")
    seen = 0
    try:
        paginator = logs.get_paginator("filter_log_events")
        for page in paginator.paginate(logGroupName=group, startTime=start):
            for event in page.get("events", []):
                stamp = datetime.fromtimestamp(
                    event["timestamp"] / 1000, timezone.utc
                ).strftime("%H:%M:%S")
                print(f"{stamp}  {event['message'].rstrip()}")
                seen += 1
    except logs.exceptions.ResourceNotFoundException:
        print(
            f"log group does not exist yet. It is created by the first invocation, so "
            f"either the agent has not run or the name is wrong. Groups matching the "
            f"runtime:"
        )
        for g in logs.describe_log_groups(
            logGroupNamePrefix="/aws/bedrock-agentcore/runtimes/"
        ).get("logGroups", []):
            print(f"  {g['logGroupName']}")
        return 1
    if not seen:
        print(f"(no events in the last {since})")
    return 0


def cmd_observability() -> int:
    """Report each of the three pillars from the service, not from configuration.

    Checking that the OTEL environment variables are set proves nothing: they were set
    correctly for several cycles while no span was ever emitted, because the container
    launched `python app.py` instead of `opentelemetry-instrument python app.py` and the
    distro never activated. So this reads actual data back.

    Note the namespace: `AWS/Bedrock-AgentCore`, with the hyphen. `bedrock-agentcore`
    exists as a name and is always empty, which looks like "no metrics" rather than
    "wrong namespace".
    """
    import json as _json

    rid = runtime_arn().rsplit("/", 1)[-1]
    problems: list[str] = []

    print("logs")
    group = f"/aws/bedrock-agentcore/runtimes/{rid}-DEFAULT"
    logs = session.client("logs")
    try:
        got = logs.filter_log_events(
            logGroupName=group, startTime=int((time.time() - 86400) * 1000), limit=1
        )
        n = len(got.get("events", []))
        print(f"  {'ok  ' if n else 'FAIL'}  {group}  {'events present' if n else 'EMPTY'}")
        if not n:
            problems.append("no log events in the last 24h")
    except logs.exceptions.ResourceNotFoundException:
        print(f"  FAIL  {group} does not exist")
        problems.append("log group missing")

    print("\nmetrics (AWS/Bedrock-AgentCore, service-published)")
    cw = session.client("cloudwatch")
    found = set()
    for page in cw.get_paginator("list_metrics").paginate(Namespace="AWS/Bedrock-AgentCore"):
        for m in page["Metrics"]:
            if any(rid in str(d["Value"]) for d in m.get("Dimensions", [])):
                found.add(m["MetricName"])
    print(f"  {'ok  ' if found else 'FAIL'}  {len(found)} metric(s): {', '.join(sorted(found)) or 'none'}")
    if not found:
        problems.append("no metrics for this runtime")

    print("\nmetrics (ApplicationSignals, from the OTEL distro)")
    sig = set()
    for page in cw.get_paginator("list_metrics").paginate(Namespace="ApplicationSignals"):
        for m in page["Metrics"]:
            if any(RUNTIME_NAME in str(d["Value"]) for d in m.get("Dimensions", [])):
                sig.add(m["MetricName"])
    print(f"  {'ok  ' if sig else 'warn'}  {len(sig)} metric(s): {', '.join(sorted(sig)) or 'none'}")
    if not sig:
        problems.append("no ApplicationSignals metrics -- the OTEL distro may not be active")

    print("\ntraces")
    xray = session.client("xray")
    dest = xray.get_trace_segment_destination()
    ok_dest = dest.get("Destination") == "CloudWatchLogs" and dest.get("Status") == "ACTIVE"
    print(f"  {'ok  ' if ok_dest else 'FAIL'}  Transaction Search: "
          f"{dest.get('Destination')} ({dest.get('Status')})")
    if not ok_dest:
        problems.append("Transaction Search is not active, so spans are collected but not shown")
    # `aws/spans` is account-wide and large -- 681 MB when this was written, shared with
    # every other instrumented service. `filter_log_events` gives up on a wide window
    # before it finds a match, so asking for 24 hours returns nothing while asking for 30
    # minutes returns spans. Widening the window makes this check LESS reliable, which is
    # the opposite of the intuition. So it scans a short window, and a miss is reported as
    # "not found in this window" rather than as proof that nothing was emitted.
    window_minutes = 120
    spans: list[str] = []
    try:
        got = logs.filter_log_events(
            logGroupName="aws/spans",
            startTime=int((time.time() - window_minutes * 60) * 1000),
            filterPattern=f'"{RUNTIME_NAME}"', limit=5,
        )
        for e in got.get("events", []):
            doc = _json.loads(e["message"])
            attrs = (doc.get("resource") or {}).get("attributes") or {}
            if attrs.get("service.name") == RUNTIME_NAME:
                spans.append(doc.get("name", "?"))
        if spans:
            print(f"  ok    {len(spans)} span(s) in the last {window_minutes}m, "
                  f"service.name={RUNTIME_NAME}, e.g. {spans[:3]}")
        else:
            print(f"  ?     no spans found in the last {window_minutes}m. This is NOT "
                  "proof none were emitted: aws/spans is account-wide and the filter "
                  "gives up on a wide scan. Run a selftest and try again within a few "
                  "minutes.")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  aws/spans: {type(exc).__name__}: {exc}")
        problems.append("could not read aws/spans")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nlogs, metrics and traces are all arriving")
    return 0


def cmd_runs() -> int:
    from boto3.dynamodb.conditions import Key

    table = session.resource("dynamodb").Table(TABLE)
    got = table.query(
        KeyConditionExpression=Key("pk").eq("run"), ScanIndexForward=False, Limit=20
    )
    items = got.get("Items", [])
    if not items:
        print(f"{TABLE} has no run records yet")
        return 0
    print(f"{len(items)} most recent run(s), newest first\n")
    for item in items:
        print(f"{item.get('finished_at')}  {item.get('status')}")
        print(f"  {str(item.get('summary'))[:400]}")
        if item.get("pr_url"):
            print(f"  PR: {item['pr_url']}")
        if item.get("findings"):
            print(f"  findings: {str(item['findings'])[:300]}")
        print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command",
                    choices=["selftest", "daily", "logs", "runs", "observability"])
    ap.add_argument("--since", default="30m", help="For logs: 30m, 2h, 1d")
    args = ap.parse_args()

    if args.command in ("selftest", "daily"):
        return cmd_invoke(args.command)
    if args.command == "logs":
        return cmd_logs(args.since)
    if args.command == "observability":
        return cmd_observability()
    return cmd_runs()


if __name__ == "__main__":
    sys.exit(main())
