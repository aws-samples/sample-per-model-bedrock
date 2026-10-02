"""The daily email.

SES stays in sandbox mode, so the agent can only reach addresses that have confirmed a
verification email. That is a feature here rather than a limitation: an autonomous agent
with unrestricted outbound email is a worse idea than one that can reach two addresses
its owner opted into.

The sandbox has one sharp edge worth knowing. SES rejects the entire message if any
single recipient is unverified, so one unclicked confirmation link silences the
notification for everyone -- which happened on the first real cycle. `send_email`
therefore filters to the verified set and names the addresses it could not reach, because
a partly delivered notification beats a silent one.

The notification is written for someone reading it on a phone: what happened, whether it
needs them, and the link. Detail belongs in the pull request, not the email.
"""
from __future__ import annotations

import logging
import os
from typing import Annotated, Any

import boto3
from botocore.exceptions import ClientError
from claude_agent_sdk import tool

log = logging.getLogger("pr-agent.notify")

REGION = os.environ.get("AWS_REGION", "us-east-1")
SENDER = os.environ.get("SES_SENDER", "")
RECIPIENTS = [
    r.strip()
    for r in os.environ.get(
        "SES_RECIPIENTS", ""
    ).split(",")
    if r.strip()
]


def _text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(payload)}]}


def verified_recipients(ses) -> tuple[list[str], dict[str, str]]:
    """Which configured recipients SES will actually accept, and why not for the rest.

    In sandbox mode SES rejects the whole message if any single recipient is unverified,
    so one address whose confirmation link was never clicked silences the notification
    for everyone. That happened on this agent's first real cycle: the run finished, and
    nobody was told. Filtering to the verified set means a half-delivered notification
    beats none, and the skipped addresses are named in the result so the gap is visible
    rather than silent.
    """
    try:
        attrs = ses.get_identity_verification_attributes(Identities=RECIPIENTS)
        status = {
            addr: attrs["VerificationAttributes"].get(addr, {}).get(
                "VerificationStatus", "NotFound"
            )
            for addr in RECIPIENTS
        }
    except ClientError as exc:
        # Cannot tell, so try them all: a send that fails is a clearer signal than a
        # send that was silently skipped.
        log.warning("could not read verification status: %s", exc)
        return list(RECIPIENTS), {}
    good = [a for a, s in status.items() if s == "Success"]
    skipped = {a: s for a, s in status.items() if s != "Success"}
    return good, skipped


def send_email(subject: str, body: str) -> dict[str, Any]:
    """Send to the verified recipients. Returns what happened, never raises."""
    if not SENDER or not RECIPIENTS:
        return {"ok": False, "error": "email notifications are not configured "
                "(SES_SENDER and SES_RECIPIENTS are empty), so nothing was sent."}
    ses = boto3.client("ses", region_name=REGION)
    good, skipped = verified_recipients(ses)
    if not good:
        log.error("no verified recipients; nothing sent. status=%s", skipped)
        return {
            "ok": False,
            "error": "no configured recipient is a verified SES identity, so nothing "
                     "could be sent. Report this in your checkpoint -- a cycle nobody "
                     "hears about is indistinguishable from a cycle that never ran.",
            "recipient_status": skipped,
        }
    if skipped:
        body = (
            f"{body}\n\n---\nNot delivered to {sorted(skipped)} "
            f"(SES verification status {skipped}). In sandbox mode every recipient must "
            "confirm the verification email before they can receive anything.\n"
        )
    try:
        resp = ses.send_email(
            Source=SENDER,
            Destination={"ToAddresses": good},
            Message={
                "Subject": {"Data": subject[:240], "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body[:80000], "Charset": "UTF-8"}},
            },
        )
        log.info("email sent %s to %s (skipped %s)", resp["MessageId"], good,
                 sorted(skipped) or "none")
        return {
            "ok": True,
            "message_id": resp["MessageId"],
            "to": good,
            "not_delivered": skipped,
        }
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        hint = ""
        if code in ("MessageRejected", "MailFromDomainNotVerified"):
            hint = (
                " In SES sandbox mode both the sender and every recipient must be a "
                "verified identity. If a recipient has not clicked the confirmation "
                "link yet, this will keep failing until they do."
            )
        log.error("email failed %s: %s", code, exc)
        return {"ok": False, "error": f"{code}: {exc.response['Error']['Message']}{hint}"}


@tool(
    "notify_email",
    "Email the outcome of this cycle to the maintainers. Send exactly one per cycle, "
    "including when you found nothing to do and including when something failed -- a "
    "silent day is indistinguishable from a broken agent. Subject should say the "
    "outcome, not 'daily report'. Body: what changed and why, the PR link if there is "
    "one, the scanner verdict, and anything needing a human decision. Keep it short; "
    "detail belongs in the PR.",
    {
        "subject": Annotated[str, "States the outcome, e.g. 'PR #58: Claude Fable 5.2 added'."],
        "body": Annotated[str, "Plain text. Short. Link to the PR if there is one."],
    },
)
async def notify_email(args: dict[str, Any]) -> dict[str, Any]:
    subject = (args.get("subject") or "").strip()
    body = (args.get("body") or "").strip()
    if not subject or not body:
        return _text("ERROR: subject and body are both required")
    if subject.lower() in ("daily report", "update", "report"):
        return _text(
            "ERROR: the subject must state the outcome so it is readable in a notification "
            "list. Try 'No Bedrock changes needing a sample' or 'PR #58: <what changed>'."
        )
    prefix = "[bedrock-samples-agent] "
    return _text(send_email(prefix + subject, body))


notify_email.raw = send_email  # type: ignore[attr-defined]
