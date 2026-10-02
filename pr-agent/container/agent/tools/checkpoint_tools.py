"""Run-to-run memory, in DynamoDB.

An AgentCore session is ephemeral, so anything the agent needs to remember between
days has to be written down. This is deliberately not AgentCore Memory: what matters
here is a small, strongly-typed, queryable audit record of what an autonomous agent did
to a public repository -- closer to a ledger than to conversational recall.

Two item shapes in one table:

    pk="checkpoint", sk="latest"          the pointer the next run reads first
    pk="run",        sk=<ISO timestamp>   one immutable record per cycle
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Annotated, Any

import boto3
# Imported explicitly rather than reached through `boto3.dynamodb`: that attribute only
# exists once a dynamodb resource has been constructed, so `recent_runs` would raise
# AttributeError if it ever ran first.
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from claude_agent_sdk import tool

# The only statuses the ledger accepts. A free-text status makes the run history
# unqueryable, which defeats the point of keeping one.
STATUSES = ("success", "no-change", "error", "partial")

log = logging.getLogger("pr-agent.checkpoint")

TABLE = os.environ.get("CHECKPOINT_TABLE", "bedrock-samples-pr-agent-state")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def _table():
    return boto3.resource("dynamodb", region_name=REGION).Table(TABLE)


def read_checkpoint() -> dict[str, Any]:
    """What the previous cycle recorded. Never raises: a first run has no checkpoint."""
    try:
        got = _table().get_item(Key={"pk": "checkpoint", "sk": "latest"})
        item = got.get("Item")
        if not item:
            return {
                "first_run": True,
                "summary": "no previous run recorded",
                "how_to_bound_discovery": (
                    "first_run means this agent has no prior cycle, NOT that the "
                    "repository is untouched -- humans commit to it too. Derive your "
                    "discovery window from the repository instead: use "
                    "github_recent_changes and the HEAD commit date, and say in your "
                    "checkpoint that the window was chosen that way."
                ),
            }
        return {k: (float(v) if hasattr(v, "as_tuple") else v) for k, v in item.items()}
    except ClientError as exc:
        log.warning("checkpoint read failed: %s", exc.response["Error"]["Code"])
        return {"first_run": True, "summary": "checkpoint unavailable", "error": str(exc)}


def write_checkpoint(
    status: str,
    summary: str,
    session_id: str = "",
    pr_url: str = "",
    findings: str = "",
) -> dict[str, Any]:
    """Record this cycle, and move the `latest` pointer.

    Writes the immutable run record FIRST. If the process dies between the two writes,
    the history is still complete and only the pointer is stale -- the safe direction,
    because a stale pointer makes the next run re-examine ground it has already covered,
    whereas a missing record loses the evidence entirely.
    """
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if status not in STATUSES:
        # Recorded, not rejected. Losing the record of a cycle is worse than recording
        # one with an odd status, so the value is normalised and the original preserved.
        summary = f"[status was {status!r}, not one of {STATUSES}] {summary}"
        status = "partial"
    record = {
        "status": status,
        "summary": summary[:3500],
        "session_id": session_id,
        "pr_url": pr_url,
        "findings": findings[:2000],
        "finished_at": now,
    }
    try:
        table = _table()
        table.put_item(Item={"pk": "run", "sk": now, **record})
        table.put_item(Item={"pk": "checkpoint", "sk": "latest", **record})
        log.info("checkpoint written status=%s pr=%s", status, pr_url or "none")
        return {"ok": True, **record}
    except ClientError as exc:
        log.error("checkpoint write failed: %s", exc)
        return {"ok": False, "error": str(exc), **record}


def recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    """The last few cycles, newest first, so the agent can see its own pattern."""
    try:
        got = _table().query(
            KeyConditionExpression=Key("pk").eq("run"),
            ScanIndexForward=False,
            Limit=limit,
        )
        return got.get("Items", [])
    except ClientError as exc:
        log.warning("recent runs unavailable: %s", exc)
        return []


@tool(
    "checkpoint_read",
    "Read what the previous cycle did: when it finished, whether it succeeded, what it "
    "changed, and any pull request it opened. Call this FIRST in every cycle -- it is "
    "how you avoid re-reporting a change you already handled. Also returns the last 10 "
    "runs so you can see whether a failure is new or recurring.",
    {},
)
async def checkpoint_read(args: dict[str, Any]) -> dict[str, Any]:
    latest = read_checkpoint()
    history = recent_runs(10)
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"latest checkpoint:\n{latest}\n\n"
                    f"last {len(history)} run(s):\n"
                    + "\n".join(
                        f"  {r.get('finished_at')}  {str(r.get('status')):<8} "
                        f"{str(r.get('summary'))[:120]}"
                        for r in history
                    )
                ),
            }
        ]
    }


@tool(
    "checkpoint_write",
    "Record the outcome of this cycle. Call this LAST, exactly once, whatever happened "
    "-- including when you found nothing to do, and including when something failed. "
    "The next cycle reads it to decide where to start. status must be one of: "
    "'success', 'no-change', 'error', 'partial'. Write a summary a human can act on, "
    "not 'completed successfully'.",
    {
        "status": Annotated[str, "success | no-change | error | partial"],
        "summary": Annotated[str, "What you did and why, in a few sentences."],
        "pr_url": Annotated[str, "Pull request URL if you opened one, else empty."],
        "findings": Annotated[str, "Anything the next run should know about."],
    },
)
async def checkpoint_write(args: dict[str, Any]) -> dict[str, Any]:
    out = write_checkpoint(
        status=args.get("status", "partial"),
        summary=args.get("summary", ""),
        session_id=os.environ.get("BEDROCK_AGENTCORE_SESSION_ID", ""),
        pr_url=args.get("pr_url", ""),
        findings=args.get("findings", ""),
    )
    return {"content": [{"type": "text", "text": str(out)}]}


# app.py calls these directly, outside the model loop.
checkpoint_read.raw = read_checkpoint  # type: ignore[attr-defined]
checkpoint_write.raw = write_checkpoint  # type: ignore[attr-defined]
