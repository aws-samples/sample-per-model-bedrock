"""AgentCore Runtime entrypoint for the Bedrock samples PR agent.

Wakes on a schedule, looks for changes to Amazon Bedrock that a code sample should
reflect, and opens a pull request against the public samples repository.

Three things about the shape of this file are deliberate.

**The agent is given tools, not a script.** The daily cycle is described in a system
prompt and a set of skills, and the loop that runs it is a single `query()`. The
alternative -- orchestrating ten steps in Python and calling the model at each one --
was rejected because the interesting part of this job is judgement: whether an
announcement needs a sample at all, whether a notebook's output really demonstrates
what its prose claims. Python can sequence steps; it cannot make that call.

**Every side effect goes through a tool.** The agent could shell out to `git` and
`aws`, and the Claude Code CLI would let it. It gets narrow tools instead, so that a
pull request always carries the same branch naming, an EC2 test host is always
terminated, and a notification always records the same fields. Standardising the
side effects is what makes an autonomous agent auditable.

**It can open a pull request and nothing more.** No merge, no push to main, no force
push. The repository is public and the reviewer is human by design.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    query,
)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
log = logging.getLogger("pr-agent")

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL = os.environ.get("AGENT_MODEL", "us.anthropic.claude-opus-5")
WORKDIR = Path(os.environ.get("AGENT_WORKDIR", "/app/work"))
REPO_SLUG = os.environ.get("TARGET_REPO", "aws-samples/sample-per-model-bedrock")
SSH_SECRET = os.environ.get("SSH_KEY_SECRET", "bedrock-samples-pr-agent/github-ssh-key")
TOKEN_SECRET = os.environ.get("GH_TOKEN_SECRET", "bedrock-samples-pr-agent/github-token")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "400"))
MAX_BUDGET_USD = float(os.environ.get("AGENT_MAX_BUDGET_USD", "40"))
HERE = Path(__file__).resolve().parent

app = BedrockAgentCoreApp()


# --------------------------------------------------------------------------- setup
def _secret(name: str) -> str | None:
    """A secret's value, or None when it is not set up yet.

    None rather than an exception: the agent is deliberately able to run a
    read-only cycle before the GitHub token exists, so that the first deployment can
    be verified without waiting on a human to create credentials.
    """
    try:
        sm = boto3.client("secretsmanager", region_name=REGION)
        return sm.get_secret_value(SecretId=name)["SecretString"]
    except Exception as exc:  # noqa: BLE001
        log.warning("secret %s unavailable: %s", name, type(exc).__name__)
        return None


def install_ssh_key() -> bool:
    """Write the deploy key where git will find it. Returns whether it is usable."""
    key = _secret(SSH_SECRET)
    if not key:
        return False
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    key_path = ssh_dir / "id_ed25519"
    key_path.write_text(key if key.endswith("\n") else key + "\n")
    key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    # Pin GitHub's host key rather than disabling verification. StrictHostKeyChecking=no
    # would accept any host presenting itself as github.com, which is the whole attack
    # this setting exists to prevent.
    (ssh_dir / "known_hosts").write_text(GITHUB_HOST_KEYS)
    (ssh_dir / "config").write_text(
        "Host github.com\n"
        "  User git\n"
        "  IdentityFile ~/.ssh/id_ed25519\n"
        "  IdentitiesOnly yes\n"
        "  StrictHostKeyChecking yes\n"
    )
    log.info("deploy key installed at %s", key_path)
    return True


# github.com host keys, so StrictHostKeyChecking can stay on.
GITHUB_HOST_KEYS = """\
github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl
github.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=
github.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCj7ndNxQowgcQnjshcLrqPEiiphnt+VTTvDP6mHBL9j1aNUkY4Ue1gvwnGLVlOhGeYrnZaMgRK6+PKCUXaDbC7qtbW8gIkhL7aGCsOr/C56SJMy/BCZfxd1nWzAOxSDPgVsmerOBYfNqltV9/hWCqBywINIR+5dIg6JTJ72pcEpEjcYgXkE2YEFXV1JHnsKgbLWNlhScqb2UmyRkQyytRLtL+38TGxkxCflmO+5Z8CSSNY7GidjMIZ7Q4zMjA2n1nGrlTDkzwDCsw+wqFPGQA179cnfGWOWRVruj16z6XyvxvjJwbz0wQZ75XK5tKSb7FNyeIEs4TT4jk+S4dhPeAUC5y+bDYirYgM4GC7uEnztnZyaVWQ7B381AK4Qdrwt51ZqExKbQpTUNn+EjqoTwvqNj4kqx5QUCI0ThS/YkOxJCXmPUWZbhjpCg56i+2aB6CmK2JGhn57K5mj0MNdBXA4/WnwH6XoPWJzK5Nyu2zB3nAZp+S5hpQs+p1vN1/wsjk=
"""


def gh_token() -> str | None:
    """The token used for the pull-request API call. Not needed for git itself."""
    token = _secret(TOKEN_SECRET)
    if token:
        token = token.strip()
        # Accept a JSON secret as well as a bare string, because both shapes get
        # created by hand and a confusing failure here costs a whole daily cycle.
        if token.startswith("{"):
            try:
                token = json.loads(token).get("token", "").strip()
            except json.JSONDecodeError:
                pass
    return token or None


# --------------------------------------------------------------------------- prompt
def build_prompt(mode: str, checkpoint: dict[str, Any]) -> str:
    """The turn's instruction. The standing orders live in the system prompt."""
    if mode == "selftest":
        return (
            "Self-test only. Do not modify any repository and do not open a pull "
            "request. Do these four things and report the result of each in a short "
            "markdown table:\n"
            "1. Call `bedrock_list_models` for us-east-1 and report how many models "
            "came back.\n"
            "2. Call `checkpoint_read` and report what the last run recorded.\n"
            "3. Call `github_clone` and report the HEAD commit subject.\n"
            "4. Call `scan_run` with scanner='bandit' on the cloned repository and "
            "report the finding count.\n"
            "Then state plainly whether the environment is fit to run a real cycle, "
            "and name anything that is not working."
        )
    last = checkpoint.get("finished_at", "never")
    summary = checkpoint.get("summary", "no previous run recorded")
    return (
        f"Run the daily cycle. The previous cycle finished at {last} and recorded: "
        f"{summary}\n\n"
        "Follow the ten steps in your system prompt in order. Do not skip the "
        "verification steps even if the change looks trivial -- a sample that runs but "
        "prints something its prose does not claim is the defect this repository "
        "exists to avoid.\n\n"
        "If, after the discovery step, there is nothing that warrants a change to the "
        "samples, that is a valid and common outcome. Record it in the checkpoint, "
        "send the notification saying so, and stop. Do not invent work, and do not "
        "open a pull request to demonstrate activity."
    )


def load_system_prompt() -> str:
    text = (HERE / "agent" / "system_prompt.md").read_text()
    return text.replace("{{REPO_SLUG}}", REPO_SLUG).replace("{{REGION}}", REGION)


# ----------------------------------------------------------------------- the cycle
async def run_cycle(mode: str, session_id: str) -> dict[str, Any]:
    from agent.tools import build_tool_servers, checkpoint_read

    started = time.time()
    have_ssh = install_ssh_key()
    token = gh_token()
    # Into this process's own environment, not only into the options below.
    #
    # `ClaudeAgentOptions.env` configures the Claude Code CLI *subprocess*. The in-process
    # MCP servers run here, inside app.py, and `github_tools._api` reads GH_TOKEN from
    # this process's environ -- so passing it only through the options left the
    # pull-request API with no credential while SSH push worked fine, because SSH reads a
    # file. That combination is confusing from the outside: the agent could commit and
    # push a whole change and then fail on the last step of the cycle.
    if token:
        os.environ["GH_TOKEN"] = token
    checkpoint = checkpoint_read.raw()

    log.info(
        "cycle start mode=%s session=%s ssh=%s token=%s model=%s",
        mode, session_id, have_ssh, bool(token), MODEL,
    )

    servers = build_tool_servers()
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=load_system_prompt(),
        mcp_servers=servers,
        # Skills carry the long-form procedure for each step. Keeping them out of the
        # system prompt keeps the standing orders short enough to actually be followed.
        setting_sources=["project"],
        skills="all",
        cwd=str(HERE),
        add_dirs=[str(WORKDIR)],
        permission_mode="bypassPermissions",
        max_turns=MAX_TURNS,
        max_budget_usd=MAX_BUDGET_USD,
        effort="high",
        env={
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": REGION,
            "ANTHROPIC_MODEL": MODEL,
            "GH_TOKEN": token or "",
            "TARGET_REPO": REPO_SLUG,
            "AGENT_WORKDIR": str(WORKDIR),
        },
        # NotebookEdit is withheld deliberately. It edits a cell's source without
        # running it, which in this repository produces a notebook whose committed
        # output no longer matches its code -- the exact defect the samples exist to
        # avoid. The agent edits notebooks as JSON and re-runs them instead.
        disallowed_tools=["NotebookEdit"],
    )

    transcript: list[str] = []
    tools_used: list[str] = []
    result: ResultMessage | None = None

    async for message in query(prompt=build_prompt(mode, checkpoint), options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    transcript.append(block.text)
                    log.info("agent: %s", block.text[:400].replace("\n", " "))
                elif isinstance(block, ToolUseBlock):
                    tools_used.append(block.name)
                    log.info("tool: %s", block.name)
                elif isinstance(block, ThinkingBlock):
                    log.debug("thinking: %d chars", len(block.thinking or ""))
        elif isinstance(message, ResultMessage):
            result = message

    elapsed = round(time.time() - started, 1)
    outcome = {
        "mode": mode,
        "session_id": session_id,
        "seconds": elapsed,
        "turns": getattr(result, "num_turns", None),
        "cost_usd": getattr(result, "total_cost_usd", None),
        "is_error": bool(getattr(result, "is_error", False)),
        "tools_used": sorted(set(tools_used)),
        "tool_calls": len(tools_used),
        "final": (transcript[-1][:4000] if transcript else ""),
    }
    log.info(
        "cycle end turns=%s cost=%s error=%s seconds=%s tools=%d",
        outcome["turns"], outcome["cost_usd"], outcome["is_error"], elapsed,
        outcome["tool_calls"],
    )
    return outcome


def runtime_session_id() -> str:
    """The session id AgentCore is actually using for this invocation.

    It arrives as a request header and is exposed through the runtime's request context,
    not through the environment -- an earlier version read a
    `BEDROCK_AGENTCORE_SESSION_ID` environment variable that does not exist, so every
    run recorded an invented `local-<epoch>` id while the logs and traces carried the
    real one. That silently broke the only link between a ledger row and its trace.

    Must be called on the request thread, before any background work starts: the context
    is request-scoped and a worker thread cannot see it.
    """
    try:
        from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

        got = BedrockAgentCoreContext.get_session_id()
        if got:
            return got
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read the runtime session id: %s", type(exc).__name__)
    # Running outside a request, e.g. `python app.py` locally. Marked as such rather than
    # dressed up as a real session id.
    return f"nosession-{int(time.time())}"


def _run_and_record(mode: str, session_id: str) -> dict[str, Any]:
    """One cycle, with a checkpoint written whatever happens."""
    try:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        outcome = asyncio.run(run_cycle(mode, session_id))
        return {"ok": not outcome["is_error"], **outcome}
    except Exception as exc:  # noqa: BLE001
        # A crashed cycle must still leave a trace and a checkpoint, or tomorrow's run
        # has no way to know today's failed.
        log.error("cycle failed: %s\n%s", exc, traceback.format_exc())
        try:
            from agent.tools import checkpoint_write
            checkpoint_write.raw(
                status="error",
                summary=f"cycle raised {type(exc).__name__}: {exc}"[:900],
                session_id=session_id,
            )
        except Exception:  # noqa: BLE001
            log.error("could not record the failure in the checkpoint")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@app.entrypoint
def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    """AgentCore calls this once per invocation.

    Payload: {"mode": "daily" | "selftest"}. EventBridge sends "daily".

    `daily` returns immediately and runs the cycle on a background thread, registered
    with `add_async_task` so the runtime reports HealthyBusy and does not reap the
    session while it works. `InvokeAgentRuntime` is synchronous, and a cycle that runs
    notebooks and waits on an EC2 host can take an hour -- far longer than the caller
    will hold a connection. The first manual invocation of this agent proved that by
    timing out client-side at 60 seconds while the agent carried on working.

    `selftest` stays synchronous. It takes well under a minute, and the whole point of
    it is to read the result inline.
    """
    mode = (payload or {}).get("mode", "daily")
    session_id = (payload or {}).get("session_id") or runtime_session_id()
    # Publish it so the checkpoint tool records the same id the logs and traces carry.
    # Correlating a ledger row with the run that produced it is the entire reason the id
    # is stored, and an invented `local-<epoch>` correlates with nothing.
    os.environ["BEDROCK_AGENTCORE_SESSION_ID"] = session_id
    if mode not in ("daily", "selftest"):
        return {"ok": False, "error": f"unknown mode {mode!r}; expected daily|selftest"}

    if mode == "selftest":
        return _run_and_record(mode, session_id)

    task_id = app.add_async_task("daily_cycle", {"session_id": session_id})

    def worker() -> None:
        try:
            outcome = _run_and_record(mode, session_id)
            log.info("background cycle finished: ok=%s", outcome.get("ok"))
        finally:
            # In a finally block on purpose: a task that is never completed leaves the
            # runtime reporting HealthyBusy for the rest of the session's eight hours,
            # which looks like a hung agent and blocks nothing usefully.
            app.complete_async_task(task_id)

    threading.Thread(target=worker, name="daily-cycle", daemon=False).start()
    log.info("daily cycle started as async task %s", task_id)
    return {
        "ok": True,
        "accepted": True,
        "mode": mode,
        "task_id": task_id,
        "session_id": session_id,
        "note": (
            "The cycle runs in the background and reports HealthyBusy until it finishes. "
            "Read its outcome from the checkpoint table or the runtime logs, not from "
            "this response."
        ),
    }


if __name__ == "__main__":
    app.run()
