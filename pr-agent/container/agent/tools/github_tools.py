"""Git and GitHub, narrowed to what an autonomous agent should be able to do.

The repository this writes to is public and carries the AWS name, so the boundary
matters more than the convenience. Enforced here, in code, not in the prompt:

* Pushes go to `bot/` branches only. `_assert_safe_branch` rejects main and anything
  outside that prefix, so a confused agent cannot push over the trunk.
* There is no merge tool, and no force push. A human reviews every change.
* Commits are authored as the agent, not as a person, so `git log` shows plainly
  which changes were machine-made.

Two credentials, two jobs: the SSH deploy key moves git objects, and the token calls
the pull-request API. A deploy key cannot open a pull request, which is why both exist.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any

from claude_agent_sdk import tool

log = logging.getLogger("pr-agent.github")

REPO_SLUG = os.environ.get("TARGET_REPO", "aws-samples/sample-per-model-bedrock")
WORKDIR = Path(os.environ.get("AGENT_WORKDIR", "/app/work"))
CLONE = WORKDIR / REPO_SLUG.split("/")[-1]
BRANCH_PREFIX = "bot/bedrock-update-"
AUTHOR_NAME = "bedrock-samples-pr-agent"
AUTHOR_EMAIL = "bedrock-samples-pr-agent@users.noreply.github.com"


class UnsafeOperation(RuntimeError):
    """Raised when the agent asks for something outside its remit."""


def _assert_safe_branch(branch: str) -> None:
    if not branch.startswith(BRANCH_PREFIX):
        raise UnsafeOperation(
            f"refusing to operate on {branch!r}: this agent may only touch branches "
            f"named {BRANCH_PREFIX}*"
        )
    if branch in ("main", "master") or "/.." in branch or branch.endswith("/.."):
        raise UnsafeOperation(f"refusing to operate on {branch!r}")


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    """Run one git command. Never through a shell, so nothing is word-split."""
    proc = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=str(cwd or CLONE),
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[-800:]}"
        )
    return (proc.stdout or "").strip()


def _api(method: str, path: str, body: dict | None = None) -> dict[str, Any]:
    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        raise UnsafeOperation(
            "no GitHub token available, so the pull-request API cannot be called. The "
            "branch may already be pushed; report that and stop rather than retrying."
        )
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        method=method,
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": AUTHOR_NAME,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise RuntimeError(f"GitHub API {method} {path} -> {exc.code}: {detail}") from exc


def _text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(payload)}]}


# --------------------------------------------------------------------------- tools
@tool(
    "github_clone",
    "Clone (or refresh) the samples repository into the working directory and report "
    "HEAD. Idempotent: safe to call more than once in a cycle. Returns the clone path, "
    "which is where you should read and edit files.",
    {},
)
async def github_clone(args: dict[str, Any]) -> dict[str, Any]:
    try:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        if (CLONE / ".git").is_dir():
            _git("fetch", "--all", "--prune")
            _git("checkout", "main")
            _git("reset", "--hard", "origin/main")
        else:
            _git(
                "clone", f"git@github.com:{REPO_SLUG}.git", str(CLONE),
                cwd=WORKDIR,
            )
        _git("config", "user.name", AUTHOR_NAME)
        _git("config", "user.email", AUTHOR_EMAIL)
        head = _git("log", "-1", "--pretty=%H %s")
        count = _git("rev-list", "--count", "HEAD")
        return _text(
            f"clone_path={CLONE}\nhead={head}\ncommits={count}\n"
            f"branch={_git('rev-parse', '--abbrev-ref', 'HEAD')}"
        )
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "github_recent_changes",
    "Commits merged to main since a date, so you can see what changed in the repository "
    "while you were not looking. Pass an ISO date such as 2026-09-01; leave empty for "
    "the last 14 days.",
    {"since": Annotated[str, "ISO date, e.g. 2026-09-01. Empty for last 14 days."]},
)
async def github_recent_changes(args: dict[str, Any]) -> dict[str, Any]:
    since = (args.get("since") or "").strip() or "14.days.ago"
    try:
        out = _git("log", f"--since={since}", "--pretty=%h %ad %an %s", "--date=short")
        files = _git("diff", "--stat", f"@{{{since}}}..HEAD", check=False)
        return _text(f"commits since {since}:\n{out or '  (none)'}\n\nfiles:\n{files}")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "github_list_open_prs",
    "Work already in flight: every open pull request with its branch, title and changed "
    "files, AND every bot branch that was pushed but has no pull request. Check this "
    "before starting: an open PR may already cover the change you were about to make, in "
    "which case say so and stop rather than opening a duplicate. A pushed bot branch with "
    "no PR means a previous cycle did the work and could not deliver it -- inspect that "
    "branch and finish it rather than starting again.",
    {},
)
async def github_list_open_prs(args: dict[str, Any]) -> dict[str, Any]:
    out: list[str] = []
    try:
        prs = _api("GET", f"/repos/{REPO_SLUG}/pulls?state=open&per_page=50")
        pr_branches = {pr["head"]["ref"] for pr in prs}
        if prs:
            out.append("open pull requests:")
            for pr in prs:
                files = _api(
                    "GET", f"/repos/{REPO_SLUG}/pulls/{pr['number']}/files?per_page=100"
                )
                names = ", ".join(f["filename"] for f in files[:12])
                out.append(
                    f"  #{pr['number']} [{pr['head']['ref']}] {pr['title']}\n"
                    f"      by {pr['user']['login']} updated {pr['updated_at']}\n"
                    f"      files: {names}{' ...' if len(files) > 12 else ''}"
                )
        else:
            out.append("open pull requests: none")

        # A branch pushed without a PR is the signature of a cycle that did the work and
        # failed on the last step. Without this the next cycle cannot tell that from a
        # clean slate, and would redo the work on a fresh branch. A branch whose PR was
        # merged or closed is finished, and the repository keeps merged branches, so
        # every branch that has ever headed a PR is excluded, not only the open ones.
        for page in range(1, 6):
            closed = _api("GET", f"/repos/{REPO_SLUG}/pulls?state=closed&per_page=100&page={page}")
            pr_branches |= {pr["head"]["ref"] for pr in closed}
            if len(closed) < 100:
                break
        branches = _api("GET", f"/repos/{REPO_SLUG}/branches?per_page=100")
        orphans = [
            b for b in branches
            if b["name"].startswith(BRANCH_PREFIX) and b["name"] not in pr_branches
        ]
        if orphans:
            out.append("\nbot branches pushed with NO pull request -- unfinished work:")
            for b in orphans:
                commit = _api("GET", f"/repos/{REPO_SLUG}/commits/{b['commit']['sha']}")
                msg = (commit.get("commit", {}).get("message") or "").splitlines()[0]
                when = commit.get("commit", {}).get("committer", {}).get("date", "?")
                out.append(f"  {b['name']}  {b['commit']['sha'][:8]}  {when}  {msg[:90]}")
            out.append(
                "  Read these before doing anything new. If one carries the change you "
                "were about to make, check it out, verify it still holds against the "
                "service, and open the pull request instead of starting again."
            )
        return _text("\n".join(out))
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "github_create_branch",
    "Create a working branch off current main. The slug is appended to "
    f"'{BRANCH_PREFIX}' -- use something descriptive and lowercase, such as "
    "'claude-fable-5-2' or 'nova-3-region-expansion'. Pushing anywhere else is refused.",
    {"slug": Annotated[str, "Short lowercase kebab-case description of the change."]},
)
async def github_create_branch(args: dict[str, Any]) -> dict[str, Any]:
    slug = re.sub(r"[^a-z0-9-]+", "-", (args.get("slug") or "").lower()).strip("-")
    if not slug:
        return _text("ERROR: slug is required")
    branch = f"{BRANCH_PREFIX}{slug}"
    try:
        _assert_safe_branch(branch)
        _git("checkout", "main")
        _git("reset", "--hard", "origin/main")
        _git("checkout", "-B", branch)
        return _text(f"on branch {branch} at {_git('log', '-1', '--pretty=%h %s')}")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "github_commit",
    "Stage everything and commit. Write a real commit message: a subject line under 72 "
    "characters saying what changed, a blank line, then a body explaining WHY and what "
    "evidence supports it -- which announcement, which probe, which output. Never "
    "'update notebooks'.",
    {"message": Annotated[str, "Full commit message: subject, blank line, body."]},
)
async def github_commit(args: dict[str, Any]) -> dict[str, Any]:
    message = (args.get("message") or "").strip()
    if len(message.splitlines()[0]) > 72:
        return _text("ERROR: subject line exceeds 72 characters; shorten it")
    if len(message) < 40:
        return _text("ERROR: commit message is too thin; explain what changed and why")
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        _assert_safe_branch(branch)
        _git("add", "-A")
        if not _git("status", "--porcelain"):
            return _text("nothing to commit; the working tree is clean")
        _git("commit", "-m", message)
        return _text(f"committed {_git('log', '-1', '--pretty=%h %s')} on {branch}")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "github_push",
    "Push the current bot/ branch to GitHub. Refuses any other branch, and never "
    "force-pushes.",
    {},
)
async def github_push(args: dict[str, Any]) -> dict[str, Any]:
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        _assert_safe_branch(branch)
        _git("push", "--set-upstream", "origin", branch)
        return _text(f"pushed {branch}")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "github_open_pr",
    "Open a pull request from the current bot/ branch into main. This agent cannot "
    "merge and cannot push to main -- a human reviews every change. Write the body for "
    "that reviewer: what changed, what evidence justifies it (with the announcement URL "
    "and the measurements from your verification run), how it was verified, and what you "
    "deliberately did not do.",
    {
        "title": Annotated[str, "Under 72 characters, says what changed."],
        "body": Annotated[str, "Markdown. Evidence, verification, and scope."],
        "draft": Annotated[str, "'true' to open as draft (default), 'false' otherwise."],
    },
)
async def github_open_pr(args: dict[str, Any]) -> dict[str, Any]:
    title = (args.get("title") or "").strip()
    body = (args.get("body") or "").strip()
    if not title or len(body) < 200:
        return _text(
            "ERROR: a title and a substantial body are required. The body is what a "
            "reviewer reads to decide whether to trust the change."
        )
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        _assert_safe_branch(branch)
        if _git("status", "--porcelain"):
            return _text("ERROR: uncommitted changes present; commit them first")
        draft = str(args.get("draft", "true")).lower() != "false"
        pr = _api(
            "POST",
            f"/repos/{REPO_SLUG}/pulls",
            {"title": title, "head": branch, "base": "main", "body": body,
             "draft": draft, "maintainer_can_modify": True},
        )
        log.info("opened PR #%s %s", pr.get("number"), pr.get("html_url"))
        return _text(
            f"opened {'draft ' if draft else ''}PR #{pr.get('number')}: "
            f"{pr.get('html_url')}"
        )
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")
