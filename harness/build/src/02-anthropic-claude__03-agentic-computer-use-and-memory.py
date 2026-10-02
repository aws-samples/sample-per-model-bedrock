# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Claude agentic tools on Amazon Bedrock Mantle — computer use, memory, compaction
#
# The Anthropic-defined tool families: **computer use** (GUI automation), **bash**
# and **text editor**, the **memory** tool, and **context management** (compaction).
# These are what let a Claude agent run for a long time without drowning in its own
# context.
#
# **Prerequisites:** `01-messages-api-core.ipynb` (Messages API basics) and
# `02-thinking-tools-and-caching.ipynb` (thinking, tools, caching).
#
# ## ⚠️ Beta services and real risk
# Computer use is a **Beta Service** under the AWS Service Terms. It carries risks
# that ordinary chat APIs do not:
#
# - Run it in a **dedicated VM or container with minimal privileges**.
# - Do not give it access to sensitive accounts or data.
# - Restrict its network reach to the domains it needs.
# - Keep a **human in the loop** for anything consequential.
#
# Anything the model can see can influence it — this is a prompt-injection surface.
# **This notebook never executes a real GUI action.** It shows the protocol and
# simulates the environment, which is what you want when learning it.
#
# ## Region
# `us-east-1` — the only Region with the full Claude set (as of August 2026). Regional
# availability changes over time; verify with `GET /v1/models` for your Region.
#
# ## Self-contained, but see also
# - `01-messages-api-core.ipynb` · `02-thinking-tools-and-caching.ipynb`
# - **Auth, the three URL paths** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
#
# ## Prerequisites
# ```bash
# pip install -r ../requirements.txt
# ```
#
# Needs anthropic, aws-bedrock-token-generator.
#
# `requirements.txt` pins the exact versions this collection was tested
# against. An unpinned install resolves whatever is current, which may be
# untested or compromised (OWASP LLM03, Supply Chain).

# %%
import json
import sys

sys.path.insert(0, "../_shared")
from mantle import err, post

REGION = "us-east-1"

OPUS5 = "anthropic.claude-opus-5"
OPUS48 = "anthropic.claude-opus-4-8"
SONNET5 = "anthropic.claude-sonnet-5"
HAIKU45 = "anthropic.claude-haiku-4-5"

PREFIX = "/anthropic/v1"
AV = {"anthropic-version": "2023-06-01"}

# Beta features are opted into with the anthropic-beta HEADER on mantle.
# (On bedrock-runtime the equivalent goes in the body as "anthropic_beta".)
COMPUTER_USE_BETA = "computer-use-2025-11-24"
CONTEXT_MGMT_BETA = "context-management-2025-06-27"


def claude_text(payload: dict) -> str:
    return "".join(
        b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"
    )


def tool_uses(payload: dict) -> list:
    return [b for b in payload.get("content", []) if b.get("type") == "tool_use"]


print("endpoint:", f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}")

# %% [markdown]
# ## 1. How beta tools are enabled on mantle
#
# Two things must line up: the **beta header** and a **tool `type` that matches
# that beta version**. Get either wrong and you get a 400.

# %%
computer_tool = {
    "type": "computer_20251124",  # must match the beta version below
    "name": "computer",
    "display_width_px": 1024,
    "display_height_px": 768,
}

probes = [
    ("correct: header + tool", {COMPUTER_USE_BETA: True}, [computer_tool]),
    ("tool without beta header", {}, [computer_tool]),
    ("header without the tool", {COMPUTER_USE_BETA: True}, None),
]
for label, beta, tools in probes:
    headers = dict(AV)
    if beta:
        headers["anthropic-beta"] = COMPUTER_USE_BETA
    body = {
        "model": OPUS48,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Take a screenshot."}],
    }
    if tools:
        body["tools"] = tools
    code, data = post(f"{PREFIX}/messages", body, region=REGION, headers=headers)
    print(f"  {label:26} -> HTTP {code} {'' if code == 200 else err(data)[:70]}")

# %% [markdown]
# ## 2. Which models support computer use?
#
# Support varies by model. Probe before you build.

# %%
print(f"{'model':32} {'computer use':>14}")
print("-" * 50)
for model in (OPUS5, OPUS48, SONNET5, HAIKU45):
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": model,
            "max_tokens": 200,
            "tools": [computer_tool],
            "messages": [{"role": "user", "content": "Take a screenshot."}],
        },
        region=REGION,
        headers={**AV, "anthropic-beta": COMPUTER_USE_BETA},
    )
    verdict = "supported" if code == 200 else f"{code}"
    print(f"{model:32} {verdict:>14}")
    if code != 200:
        print(f"      {err(data)[:80]}")

# %% [markdown]
# ## 3. The computer-use protocol
#
# The model returns a `tool_use` block describing an **action** — `screenshot`,
# `left_click`, `type`, `scroll`, `key`. Your application performs it and returns a
# `tool_result`. Nothing happens unless your code makes it happen, which is where
# your safety controls belong.

# %%
code, data = post(
    f"{PREFIX}/messages",
    {
        "model": OPUS48,
        "max_tokens": 600,
        "tools": [computer_tool],
        "messages": [
            {
                "role": "user",
                "content": (
                    "Take a screenshot of the desktop so we can " "see what's open."
                ),
            }
        ],
    },
    region=REGION,
    headers={**AV, "anthropic-beta": COMPUTER_USE_BETA},
)
print("HTTP", code, "| stop_reason:", data.get("stop_reason"))
print("content block types:", [b.get("type") for b in data.get("content", [])])
for block in tool_uses(data):
    print(f"\ntool: {block['name']}")
    print("input:", json.dumps(block["input"], indent=2))

# %% [markdown]
# ## 4. A simulated computer-use loop
#
# We return **synthetic** screenshots and results. This exercises the full protocol
# — including the human-approval gate — without automating anything real.

# %%
import base64
import struct
import zlib


def make_png(width: int, height: int, rgb: tuple) -> bytes:
    """Solid-colour PNG standing in for a screenshot."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


FAKE_SCREENSHOT = base64.b64encode(make_png(128, 96, (40, 44, 52))).decode()

# Actions we are willing to perform without asking a human first.
# %% [markdown]
# The approval gate. A computer-use agent can click anything on screen, so decide
# up front which actions run unattended and which need a human (OWASP LLM06,
# Excessive Agency).

# %%
AUTO_APPROVED = {"screenshot", "cursor_position"}


def perform_action(action: dict) -> dict:
    """Simulate a GUI action. A real implementation would drive a sandboxed VM."""
    kind = action.get("action")
    if kind not in AUTO_APPROVED:
        # THE HUMAN-IN-THE-LOOP GATE. In a real system, block here for approval.
        return {
            "refused": True,
            "reason": f"action '{kind}' requires human approval in this environment",
        }
    if kind == "screenshot":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": FAKE_SCREENSHOT,
            },
        }
    return {"ok": True, "action": kind}


# %% [markdown]
# The loop, bounded by `max_rounds`.


# %%
def computer_use_loop(task: str, model: str = OPUS48, max_rounds: int = 4) -> list:
    conversation = [{"role": "user", "content": task}]
    trace = []
    for round_no in range(max_rounds):
        code, data = post(
            f"{PREFIX}/messages",
            {
                "model": model,
                "max_tokens": 900,
                "tools": [computer_tool],
                "messages": conversation,
            },
            region=REGION,
            headers={**AV, "anthropic-beta": COMPUTER_USE_BETA},
        )
        if code != 200:
            trace.append({"round": round_no + 1, "error": err(data)[:100]})
            break
        blocks = tool_uses(data)
        if not blocks:
            trace.append({"round": round_no + 1, "final": claude_text(data)[:200]})
            break
        conversation.append({"role": "assistant", "content": data["content"]})
        results = []
        for block in blocks:
            outcome = perform_action(block["input"])
            trace.append(
                {
                    "round": round_no + 1,
                    "action": block["input"].get("action"),
                    "outcome": (
                        "screenshot returned"
                        if outcome.get("type") == "image"
                        else json.dumps(outcome)[:80]
                    ),
                }
            )
            if outcome.get("type") == "image":
                content = [outcome]  # image block back to Claude
            else:
                content = json.dumps(outcome)
            results.append(
                {"type": "tool_result", "tool_use_id": block["id"], "content": content}
            )
        conversation.append({"role": "user", "content": results})
    return trace


# %% [markdown]
# Run it and watch each action pass through the gate.

# %%
for step in computer_use_loop("Take a screenshot, then describe what you can see."):
    print(json.dumps(step))

# %% [markdown]
# Notice the gate: any action outside `AUTO_APPROVED` comes back as a refusal, and
# the model has to react to that. That is the shape of a safe integration — the
# model proposes, your code decides.

# %% [markdown]
# ## 5. Bash and text-editor tools
#
# The same pattern with different tool types. These pair with the computer-use beta.

# %%
bash_tool = {"type": "bash_20250124", "name": "bash"}
editor_tool = {"type": "text_editor_20250124", "name": "str_replace_editor"}

for label, tool, prompt in [
    ("bash", bash_tool, "List the files in the current directory."),
    ("text editor", editor_tool, "Open /tmp/notes.txt and show me its contents."),
]:
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": OPUS48,
            "max_tokens": 500,
            "tools": [tool],
            "messages": [{"role": "user", "content": prompt}],
        },
        region=REGION,
        headers={**AV, "anthropic-beta": COMPUTER_USE_BETA},
    )
    print(f"{label:14} -> HTTP {code} | stop={data.get('stop_reason')}")
    for block in tool_uses(data):
        print(f"   {block['name']}: {json.dumps(block['input'])[:130]}")
    if code != 200:
        print("   ", err(data)[:100])

# %% [markdown]
# **Never execute these blindly.** A `bash` command or a file edit proposed by a
# model is untrusted input. Allow-list what you will run, and sandbox it.

# %% [markdown]
# ## 6. The memory tool
#
# Gives Claude a directory it can read and write, so information survives beyond the
# context window. Opt in with the **context-management** beta.

# %%
memory_tool = {"type": "memory_20250818", "name": "memory"}

code, data = post(
    f"{PREFIX}/messages",
    {
        "model": OPUS48,
        "max_tokens": 700,
        "tools": [memory_tool],
        "messages": [
            {
                "role": "user",
                "content": "Remember that our deploy tool is CodeDeploy and our "
                "primary Region is us-east-1.",
            }
        ],
    },
    region=REGION,
    headers={**AV, "anthropic-beta": CONTEXT_MGMT_BETA},
)
print("HTTP", code, "| stop_reason:", data.get("stop_reason"))
print("content block types:", [b.get("type") for b in data.get("content", [])])
for block in tool_uses(data):
    print(f"\nmemory command: {json.dumps(block['input'], indent=2)[:400]}")

# %% [markdown]
# ### A simulated memory backend
#
# The model issues filesystem-style commands; you implement the store. Here it is
# an in-memory dict — in production it would be a scoped directory, S3 prefix, or
# database, isolated per user.

# %%
MEMORY_STORE: dict[str, str] = {}


def memory_backend(command: dict) -> dict:
    """Minimal memory implementation: view / create / str_replace / insert."""
    action = command.get("command")
    path = command.get("path", "")
    if action == "view":
        if path.rstrip("/") in ("", "/memories"):
            return {"entries": sorted(MEMORY_STORE)}
        return {
            "path": path,
            "content": MEMORY_STORE.get(path, ""),
            "exists": path in MEMORY_STORE,
        }
    if action == "create":
        MEMORY_STORE[path] = command.get("file_text", "")
        return {"ok": True, "path": path, "bytes": len(MEMORY_STORE[path])}
    if action == "str_replace":
        existing = MEMORY_STORE.get(path, "")
        MEMORY_STORE[path] = existing.replace(
            command.get("old_str", ""), command.get("new_str", "")
        )
        return {"ok": True, "path": path}
    if action == "insert":
        MEMORY_STORE[path] = MEMORY_STORE.get(path, "") + command.get("insert_line", "")
        return {"ok": True, "path": path}
    return {"error": f"unsupported command {action}"}


def memory_session(prompt: str, model: str = OPUS48, max_rounds: int = 5) -> str:
    conversation = [{"role": "user", "content": prompt}]
    for _ in range(max_rounds):
        code, data = post(
            f"{PREFIX}/messages",
            {
                "model": model,
                "max_tokens": 900,
                "tools": [memory_tool],
                "messages": conversation,
            },
            region=REGION,
            headers={**AV, "anthropic-beta": CONTEXT_MGMT_BETA},
        )
        if code != 200:
            return f"HTTP {code}: {err(data)[:120]}"
        blocks = tool_uses(data)
        if not blocks:
            return claude_text(data)
        conversation.append({"role": "assistant", "content": data["content"]})
        results = []
        for block in blocks:
            outcome = memory_backend(block["input"])
            print(
                f"   memory: {block['input'].get('command'):12} "
                f"{block['input'].get('path', ''):28} -> {json.dumps(outcome)[:70]}"
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": json.dumps(outcome),
                }
            )
        conversation.append({"role": "user", "content": results})
    return "(max rounds reached)"


print("--- session 1: store facts ---")
print(
    "reply:",
    memory_session(
        "Remember: our deploy tool is CodeDeploy, primary Region us-east-1. "
        "Save it to memory, then confirm briefly."
    )[:200],
)
print("\nstore now holds:", {k: v[:60] for k, v in MEMORY_STORE.items()})

# %%
print("--- session 2: fresh conversation, recall from memory ---")
print(
    "reply:",
    memory_session("Check your memory directory and tell me which deploy tool we use.")[
        :250
    ],
)

# %% [markdown]
# The second session shares **no conversation history** with the first — only the
# memory store. That is the point: state outlives the context window.

# %% [markdown]
# ## 7. Context management (compaction)
#
# Long tool-using agents accumulate enormous tool-result history. `context_management`
# lets Claude clear old tool calls so the context stays affordable.

# %%
code, data = post(
    f"{PREFIX}/messages",
    {
        "model": OPUS48,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": "Reply OK"}],
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
    },
    region=REGION,
    headers={**AV, "anthropic-beta": CONTEXT_MGMT_BETA},
)
print("context_management ->", code, "|", claude_text(data)[:60] or err(data)[:80])

# %%
# Build a conversation with several rounds of bulky tool results, then compact it.
bulky_tool = {
    "name": "fetch_log",
    "description": "Fetch a log extract.",
    "input_schema": {
        "type": "object",
        "properties": {"service": {"type": "string"}},
        "required": ["service"],
    },
}


def fetch_log(service: str) -> dict:
    # Deliberately verbose, like a real log tool.
    return {
        "service": service,
        "lines": [
            f"{service} INFO handled request {i} in {10 + i}ms" for i in range(40)
        ],
    }


conversation = [
    {
        "role": "user",
        "content": "Check the logs for api, worker and scheduler, then tell "
        "me which is slowest.",
    }
]
for _ in range(4):
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": OPUS48,
            "max_tokens": 900,
            "tools": [bulky_tool],
            "messages": conversation,
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        },
        region=REGION,
        headers={**AV, "anthropic-beta": CONTEXT_MGMT_BETA},
    )
    if code != 200:
        print("HTTP", code, err(data)[:110])
        break
    usage = data.get("usage", {})
    blocks = tool_uses(data)
    print(
        f"round: input_tokens={usage.get('input_tokens'):>6} "
        f"tool_calls={len(blocks)}"
    )
    if not blocks:
        print("\nfinal:", claude_text(data)[:220])
        break
    conversation.append({"role": "assistant", "content": data["content"]})
    conversation.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": b["id"],
                    "content": json.dumps(fetch_log(**b["input"])),
                }
                for b in blocks
            ],
        }
    )

# %% [markdown]
# Watch `input_tokens` across rounds. Without compaction it grows with every bulky
# tool result; with `clear_tool_uses` Claude can drop stale ones. Combine this with
# prompt caching (see `02-thinking-tools-and-caching.ipynb`) and a long agent run
# stays affordable.

# %% [markdown]
# ## 8. All three together: a long-running agent shape
#
# Memory for durable facts, compaction for transient tool noise, caching for the
# static instructions.

# %%
AGENT_SYSTEM = (
    "You are an operations assistant. Use the memory tool to persist "
    "durable facts about the environment. Use fetch_log for logs. "
    "Be concise."
) * 40  # padded so caching is worthwhile

code, data = post(
    f"{PREFIX}/messages",
    {
        "model": OPUS48,
        "max_tokens": 900,
        # Static instructions, cached with a 1-hour TTL.
        "system": [
            {
                "type": "text",
                "text": AGENT_SYSTEM,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "tools": [memory_tool, bulky_tool],
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        "messages": [
            {
                "role": "user",
                "content": "What deploy tool do we use? Check memory first.",
            }
        ],
    },
    region=REGION,
    headers={**AV, "anthropic-beta": CONTEXT_MGMT_BETA},
)
usage = data.get("usage", {})
print("HTTP", code)
print(
    "cache_write:",
    usage.get("cache_creation_input_tokens"),
    "| cache_read:",
    usage.get("cache_read_input_tokens"),
)
print("blocks:", [b.get("type") for b in data.get("content", [])])
for block in tool_uses(data):
    print(f"   {block['name']}: {json.dumps(block['input'])[:110]}")

# %% [markdown]
# ## 9. Safety checklist before you ship any of this
#
# | Control | Why |
# |---|---|
# | Dedicated VM / container, least privilege | Contains mistakes and attacks |
# | No sensitive credentials in reach | Anything visible can be exfiltrated |
# | Domain allow-list for network access | Limits exposure to malicious content |
# | Action allow-list in your executor | The model proposes; you decide |
# | Human approval for consequential actions | Payments, deletions, consent clicks |
# | Per-user memory isolation | One user's memory must never leak into another's |
# | Audit log of every executed action | You will need it |
# | Inform users and obtain consent | Required, and the right thing to do |

# %%
# The gate from §4, restated as the pattern to copy.
DANGEROUS = {"left_click", "type", "key", "middle_click", "double_click", "scroll"}


def gated_executor(action: dict, human_approves=lambda a: False) -> dict:
    kind = action.get("action")
    if kind in DANGEROUS and not human_approves(action):
        return {"refused": True, "reason": "awaiting human approval"}
    return perform_action(action)


print("screenshot   ->", json.dumps(gated_executor({"action": "screenshot"}))[:70])
print(
    "left_click   ->",
    json.dumps(gated_executor({"action": "left_click", "coordinate": [100, 200]})),
)
print(
    "left_click ✓ ->",
    json.dumps(gated_executor({"action": "left_click"}, human_approves=lambda a: True)),
)

# %% [markdown]
# ## Gotchas — Claude agentic tools on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | Beta opt-in | `anthropic-beta` **header** on mantle (body field on runtime) |
# | Version pairing | Tool `type` must match the beta version, e.g. `computer_20251124` |
# | Model support | Computer use is not on every Claude model — probe first |
# | Memory beta | Uses `context-management-2025-06-27`, not the computer-use beta |
# | You implement the backend | Memory and GUI actions do nothing unless your code acts |
# | Untrusted proposals | `bash` / editor / click actions are untrusted input — allow-list |
# | Memory isolation | Scope the store per user, or you leak across tenants |
# | Compaction + caching | Combine `clear_tool_uses` with cached system prompts |
# | Beta Service terms | Computer use is Beta under the AWS Service Terms |
#
# ## Where next
# - `01-messages-api-core.ipynb` · `02-thinking-tools-and-caching.ipynb`
# - Server-side tools on another family:
#   `../01-openai-gpt/05-server-side-tools-and-fine-tuning.ipynb`
