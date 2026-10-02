#!/usr/bin/env python3
"""Store the agent's GitHub token, after proving it actually works.

    python3 set_github_token.py

Paste the token when prompted. It is read without echo, never printed, never written to
disk, and never passed on a command line where it would land in shell history.

The validation matters more than the storage. A token that cannot open a pull request
fails at step 8 of the daily cycle -- after discovery, verification and the security gate
have all run -- so the cost of finding out late is a wasted cycle. This checks the three
things the agent actually needs before it replaces anything:

    1. the token authenticates at all
    2. it can see the target repository
    3. it can write to it, and read pull requests

There is no GitHub API for creating a fine-grained token, so the token itself has to come
from the web UI:

    Settings -> Developer settings -> Personal access tokens -> Fine-grained tokens
      Repository access : Only select repositories -> aws-samples/sample-per-model-bedrock
      Permissions       : Pull requests  Read and write
                          Contents       Read and write
"""
from __future__ import annotations

import getpass
import json
import sys
import urllib.error
import urllib.request

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
NAME = "bedrock-samples-pr-agent"
SECRET = f"{NAME}/github-token"
REPO = "aws-samples/sample-per-model-bedrock"

DESCRIPTION = (
    "GitHub token the agent uses to open pull requests. Should be a fine-grained PAT "
    f"scoped to {REPO} with Pull requests: read/write and Contents: read/write. Set with "
    "infra/set_github_token.py, which validates it before storing."
)


def api(token: str, path: str) -> tuple[int, dict | list | None]:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": NAME,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except json.JSONDecodeError:
            return exc.code, None


def validate(token: str) -> list[str]:
    """Everything wrong with this token. Empty means it will work."""
    problems: list[str] = []

    status, _ = api(token, "/rate_limit")
    if status != 200:
        problems.append(
            f"the token does not authenticate (HTTP {status} on /rate_limit). Check it was "
            "pasted whole -- fine-grained tokens are long and easy to truncate."
        )
        return problems  # nothing else is meaningful

    status, repo = api(token, f"/repos/{REPO}")
    if status != 200 or not isinstance(repo, dict):
        problems.append(
            f"the token cannot see {REPO} (HTTP {status}). If it is fine-grained, check it "
            "lists that repository under Repository access."
        )
        return problems

    perms = repo.get("permissions") or {}
    if not perms.get("push"):
        problems.append(
            f"the token cannot write to {REPO} (permissions={perms}). Contents needs "
            "Read and write."
        )

    status, _ = api(token, f"/repos/{REPO}/pulls?state=open&per_page=1")
    if status != 200:
        problems.append(
            f"the token cannot read pull requests (HTTP {status}). Pull requests needs "
            "Read and write."
        )

    return problems


def post(token: str, path: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        method="POST",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": NAME,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.status, ""
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, (json.loads(exc.read() or b"{}") or {}).get("message", "")
        except json.JSONDecodeError:
            return exc.code, ""


# Creating a ref whose SHA is forty zeros can never succeed: no such object exists. So the
# status code isolates authorisation from everything else, and nothing is created either
# way. 403 means the token has no write access to that repository; 422 means it does and
# only the payload was rejected.
IMPOSSIBLE_REF = {"ref": "refs/heads/pr-agent-scope-probe", "sha": "0" * 40}


def reach(token: str) -> list[str]:
    """Measure where the token can write, by trying to write.

    An earlier version of this counted `permissions.push` over `/user/repos` and called
    that the token's reach. That field describes the **authenticated user's** access to
    each repository, not the token's, so a correctly scoped fine-grained PAT reported
    "can write to 53 repositories" and looked broken. The field could not disagree with
    the claim being made from it, which is the defect this whole project exists to avoid.

    So this probes instead. Two calls: one repository the token should be able to write,
    and one it should not.
    """
    notes: list[str] = []

    status, message = post(token, f"/repos/{REPO}/git/refs", IMPOSSIBLE_REF)
    if status == 422:
        notes.append(f"ok    can write to {REPO} (422 on an impossible SHA = authorised)")
    elif status == 403:
        notes.append(f"FAIL  cannot write to {REPO}: {message}")
    else:
        notes.append(f"?     unexpected HTTP {status} writing to {REPO}: {message}")

    # Any other repository the token can see. If it cannot see one, that is already the
    # tightest possible scope and there is nothing to check.
    status, repos = api(
        token, "/user/repos?affiliation=owner,collaborator,organization_member&per_page=100"
    )
    others = [
        r["full_name"] for r in (repos if isinstance(repos, list) else [])
        if r.get("full_name") and r["full_name"] != REPO
    ]
    if not others:
        notes.append("ok    the token can see no other repository")
        return notes

    status, message = post(token, f"/repos/{others[0]}/git/refs", IMPOSSIBLE_REF)
    if status == 403:
        notes.append(f"ok    refused write to {others[0]} -- scope is limited as intended")
    elif status == 422:
        notes.append(
            f"WARN  the token CAN write to {others[0]}, which it does not need. It is "
            "scoped more widely than one repository."
        )
    else:
        notes.append(f"?     unexpected HTTP {status} on {others[0]}: {message}")
    notes.append(
        f"note  it can LIST {len(others) + 1} repositories. That is metadata only and is "
        "normal for a fine-grained PAT; the write probes above are what bound it."
    )
    return notes


def main() -> int:
    print(__doc__.splitlines()[0])
    print(f"\nTarget secret : {SECRET}")
    print(f"Target repo   : {REPO}\n")

    token = getpass.getpass("Paste the token (input hidden, then Enter): ").strip()
    if not token:
        print("nothing entered; leaving the existing secret untouched")
        return 1

    print("\nvalidating before storing anything...")
    problems = validate(token)
    if problems:
        print(f"\n{len(problems)} problem(s) -- the secret was NOT changed:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("  ok    authenticates")
    print(f"  ok    can see and write {REPO}")
    print("  ok    can read pull requests")
    for line in reach(token):
        print(f"  {line}")

    sm = boto3.Session(region_name=REGION).client("secretsmanager")
    try:
        sm.put_secret_value(SecretId=SECRET, SecretString=token)
        sm.update_secret(SecretId=SECRET, Description=DESCRIPTION)
        action = "updated"
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        sm.create_secret(Name=SECRET, Description=DESCRIPTION, SecretString=token)
        action = "created"

    # Read it back. A write that reported success and stored something else would fail at
    # step 8 tomorrow, which is exactly what this script exists to prevent.
    stored = sm.get_secret_value(SecretId=SECRET)["SecretString"]
    if stored != token:
        print("\nFAIL: the stored value does not match what was entered")
        return 1
    print(f"\n{action} {SECRET}, and read it back unchanged")
    print("The next cycle will use it. To try it now:  python3 invoke.py selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
