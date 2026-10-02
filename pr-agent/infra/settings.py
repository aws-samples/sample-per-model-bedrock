"""Settings shared by the infra scripts.

Values come from the environment, or from `infra/agent.env` (KEY=VALUE lines, gitignored;
copy `agent.env.example`). A variable already set in the environment wins. AWS credentials
come from the default chain: set AWS_PROFILE to pick a profile.
"""
from __future__ import annotations

import os

_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.env")


def _load(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load(_ENV_FILE)


def _list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


REGION = os.environ.get("AWS_REGION", "us-east-1")
# Verified SES identities the agent may send from and to; the first one is the sender.
# Leave empty to deploy without email notifications.
SES_SENDERS = _list("SES_SENDERS")
# Hosts internal to your organization, which the agent must never read because it
# publishes to a public repository. Comma separated suffixes, e.g. corp.example.com.
INTERNAL_HOST_SUFFIXES = os.environ.get("INTERNAL_HOST_SUFFIXES", "")
