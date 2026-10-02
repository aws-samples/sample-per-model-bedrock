#!/usr/bin/env python3
"""Prove the image can do its job. Run at build time; the build fails if it cannot.

This exists because of two failures that already happened in this project:

* A scanner was installed into a virtual environment that was not on the invoking
  user's PATH. It exited 127, its empty output parsed as zero findings, and a
  repository passed a gate that never ran.
* A dependency was pinned to a version pair that could not co-exist. That one at
  least failed loudly, but it failed after several minutes of image layers.

So every claim this image makes about itself is checked here against the thing that
would disagree with it. The expected command list is imported from `scan_tools`, the
module that actually invokes them -- not read back from `uv tool list`, which would
only tell us what got installed and could never catch "the scanner we need is absent".

Exit code 0 means every check passed. Anything else lists what failed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
failures: list[str] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        return
    print(f"  ok    {label}{f'  [{detail}]' if detail else ''}")


def first_line(cmd: list[str], timeout: int = 180) -> str:
    proc = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, timeout=timeout
    )
    out = (proc.stdout or proc.stderr or "").strip().splitlines()
    if proc.returncode != 0 and not out:
        raise RuntimeError(f"exit {proc.returncode} with no output")
    return out[0][:70] if out else f"exit {proc.returncode}"


# ---------------------------------------------------------------- runtime imports
def runtime_imports() -> str:
    """The exact symbols app.py imports, not just the top-level packages.

    `import claude_agent_sdk` succeeding says nothing about whether `ThinkingBlock`
    or `ResultMessage` exist in the pinned version. Import what is actually used.
    """
    import bedrock_agentcore.runtime  # noqa: F401
    import boto3  # noqa: F401
    from claude_agent_sdk import (  # noqa: F401
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
        create_sdk_mcp_server,
        query,
        tool,
    )

    import claude_agent_sdk

    return f"claude_agent_sdk {getattr(claude_agent_sdk, '__version__', '?')}"


def options_fields() -> str:
    """Every ClaudeAgentOptions keyword app.py passes must be a real field.

    An unknown keyword is a TypeError at the first invocation, which on a daily
    schedule means a wasted day. Checked against the dataclass, not against the docs.
    """
    import dataclasses

    from claude_agent_sdk import ClaudeAgentOptions

    used = {
        "model", "system_prompt", "mcp_servers", "setting_sources", "skills", "cwd",
        "add_dirs", "permission_mode", "max_turns", "max_budget_usd", "effort", "env",
        "disallowed_tools",
    }
    available = {f.name for f in dataclasses.fields(ClaudeAgentOptions)}
    missing = sorted(used - available)
    if missing:
        raise RuntimeError(
            f"ClaudeAgentOptions has no field(s) {missing}; app.py would raise TypeError. "
            f"Available: {sorted(available)}"
        )
    return f"{len(used)} options fields present"


def tool_servers() -> str:
    """Building the servers exercises every @tool decorator and its schema."""
    sys.path.insert(0, str(HERE))
    from agent.tools import build_tool_servers

    servers = build_tool_servers()
    if not servers:
        raise RuntimeError("build_tool_servers() returned nothing")
    return f"{len(servers)} servers"


def system_prompt() -> str:
    text = (HERE / "agent" / "system_prompt.md").read_text()
    if "{{REPO_SLUG}}" not in text:
        raise RuntimeError("system_prompt.md no longer has the {{REPO_SLUG}} placeholder")
    return f"{len(text.splitlines())} lines"


def skills() -> str:
    found = sorted(p.parent.name for p in (HERE / ".claude" / "skills").glob("*/SKILL.md"))
    if not found:
        raise RuntimeError("no skills found under .claude/skills/*/SKILL.md")
    return f"{len(found)}: {', '.join(found)}"


def prompts_name_real_tools() -> str:
    """Every tool the system prompt and skills tell the agent to call must exist.

    A prompt that names a tool which was renamed or removed does not fail loudly. The
    agent reads the instruction, cannot find the tool, and improvises -- most likely with
    freehand shell, which is exactly what the narrow tools exist to prevent. So the
    instructions and the code are cross-checked here, and the two come from different
    places: names are scraped from the markdown, and the truth is the built servers.
    """
    import re

    sys.path.insert(0, str(HERE))
    from agent.tools import build_tool_servers  # noqa: F401
    import agent.tools as T

    real = {
        obj.name
        for mod in (T.github_tools, T.bedrock_tools, T.ec2_tools, T.scan_tools,
                    T.checkpoint_tools, T.notify_tools, T.source_tools)
        for obj in (getattr(mod, n) for n in dir(mod))
        if hasattr(obj, "name") and hasattr(obj, "handler")
    }
    pattern = re.compile(
        r"\b(github_[a-z_]+|bedrock_[a-z_]+|testhost_[a-z_]+|scan_[a-z]+"
        r"|checkpoint_[a-z]+|notify_email|sources_[a-z]+)\b"
    )
    docs = [HERE / "agent" / "system_prompt.md", *(HERE / ".claude" / "skills").glob("*/SKILL.md")]
    referenced: dict[str, str] = {}
    for doc in docs:
        for name in pattern.findall(doc.read_text()):
            referenced.setdefault(name, doc.name if doc.name != "SKILL.md" else doc.parent.name)
    missing = {n: where for n, where in referenced.items() if n not in real}
    if missing:
        raise RuntimeError(
            f"instructions name tools that do not exist: {missing}. Available: {sorted(real)}"
        )
    return f"{len(referenced)} of {len(real)} tools referenced"


# --------------------------------------------------------------------- executables
def scanner_commands() -> list[str]:
    sys.path.insert(0, str(HERE))
    from agent.tools.scan_tools import REQUIRED_COMMANDS

    return list(REQUIRED_COMMANDS)


def main() -> int:
    print("verifying image")
    check("runtime imports", runtime_imports)
    check("ClaudeAgentOptions fields", options_fields)
    check("tool servers build", tool_servers)
    check("system prompt", system_prompt)
    check("skills", skills)
    check("prompts name real tools", prompts_name_real_tools)
    check("git", lambda: first_line(["git", "--version"]))
    check("ssh", lambda: first_line(["ssh", "-V"]))
    check("claude cli", lambda: first_line(["claude", "--version"]))

    for cmd in scanner_commands():
        def probe(c: str = cmd) -> str:
            if not shutil.which(c):
                raise RuntimeError(
                    f"{c!r} is not on PATH for user {Path.home().name!r}. An absent "
                    "scanner exits 127 and its empty output parses as zero findings."
                )
            return first_line([c, "--version"])

        check(f"scanner {cmd}", probe)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
