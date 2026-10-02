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
# # Claude on Amazon Bedrock Mantle — thinking, tools, and prompt caching
#
# The three capabilities that make Claude worth its price on `bedrock-mantle`:
# extended/adaptive thinking, tool use with forced schemas, and prompt caching with
# a one-hour TTL option.
#
# **Prerequisite:** `01-messages-api-core.ipynb` (auth, the Messages API, the
# `content` block trap, `temperature` deprecation).
#
# **Models used:** `claude-opus-5`, `claude-sonnet-5`, `claude-opus-4-8`,
# `claude-haiku-4-5` — interleaved, because their thinking support differs.
#
# ## Region
# `us-east-1` is the only Region with the full Claude set (as of August 2026;
# verify with `GET /v1/models`). `us-west-2` has
# `haiku-4-5` only; `us-east-2` and `eu-central-1` have no Anthropic models at all.
#
# ## Self-contained, but see also
# - **Messages API basics** → `01-messages-api-core.ipynb`
# - **Agentic tool families (computer use, memory)** →
#   `03-agentic-computer-use-and-memory.ipynb`
# - **Auth, the three URL paths** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Workspaces / cost attribution / retention** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
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
import time

sys.path.insert(0, "../_shared")
from mantle import err, post

REGION = "us-east-1"

OPUS5 = "anthropic.claude-opus-5"
SONNET5 = "anthropic.claude-sonnet-5"
OPUS48 = "anthropic.claude-opus-4-8"
HAIKU45 = "anthropic.claude-haiku-4-5"

PREFIX = "/anthropic/v1"
AV = {"anthropic-version": "2023-06-01"}  # required header on mantle


def claude_text(payload: dict) -> str:
    """TEXT blocks only — reasoning models put a `thinking` block first."""
    return "".join(
        b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"
    )


def thinking_text(payload: dict) -> str:
    """The thinking blocks, when the model returns them."""
    return "".join(
        b.get("thinking", "")
        for b in payload.get("content", [])
        if b.get("type") == "thinking"
    )


print("endpoint:", f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}")

# %% [markdown]
# ## 1. Adaptive thinking — and which models have it
#
# `thinking: {"type": "adaptive"}` lets Claude decide how much to think per
# request. It is **not** available on every model: `haiku-4-5` rejects it. Probe
# before you depend on it.

# %%
print(f"{'model':32} {'adaptive thinking':>19}")
print("-" * 54)
for model in (OPUS5, SONNET5, OPUS48, HAIKU45):
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": model,
            "max_tokens": 64,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "Reply OK"}],
        },
        region=REGION,
        headers=AV,
    )
    verdict = "supported" if code == 200 else f"{code}"
    print(f"{model:32} {verdict:>19}")
    if code != 200:
        print(f"      {err(data)[:80]}")

# %% [markdown]
# ## 2. Reading a thinking response
#
# With thinking enabled the `content` array carries `thinking` blocks alongside
# `text`. This is why `content[0].text` is unsafe — covered in `01`, and it matters
# even more here.

# %%
PUZZLE = (
    "A farmer must cross a river with a wolf, a goat and a cabbage. The boat "
    "holds the farmer plus one item. The wolf eats the goat if left alone "
    "together; the goat eats the cabbage. Give the shortest sequence."
)

code, data = post(
    f"{PREFIX}/messages",
    {
        "model": SONNET5,
        "max_tokens": 3000,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": PUZZLE}],
    },
    region=REGION,
    headers=AV,
)
print("HTTP", code)
print("content block types:", [b.get("type") for b in data.get("content", [])])
print("\n=== THINKING (truncated) ===")
print(thinking_text(data)[:500] or "(none returned)")
print("\n=== ANSWER ===")
print(claude_text(data)[:500])
print("\nusage:", json.dumps(data.get("usage", {})))

# %% [markdown]
# ## 3. Thinking is incompatible with some parameters
#
# When thinking is on, sampling knobs and forced tool use are rejected. That is
# documented Anthropic behaviour, and worth seeing rather than discovering.

# %%
combinations = [
    ("thinking alone", {"thinking": {"type": "adaptive"}}),
    ("thinking + temperature", {"thinking": {"type": "adaptive"}, "temperature": 0.5}),
    ("thinking + top_p", {"thinking": {"type": "adaptive"}, "top_p": 0.9}),
]
for label, extra in combinations:
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": SONNET5,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Reply OK"}],
            **extra,
        },
        region=REGION,
        headers=AV,
    )
    print(f"  {label:26} -> HTTP {code} {'' if code == 200 else err(data)[:70]}")

# %% [markdown]
# Note `temperature` is already rejected on `sonnet-5` for its own reasons (see
# `01`), so on frontier models do not send sampling parameters at all.

# %% [markdown]
# ## 4. Streaming a thinking response
#
# Thinking and answer text arrive as distinct block types, so you can render a
# collapsible "thinking" panel.

# %%
from mantle import stream_lines

events, thinking_chars, text_chars = {}, 0, 0
print("--- live (thinking dimmed) ---")
for line in stream_lines(
    f"{PREFIX}/messages",
    {
        "model": SONNET5,
        "max_tokens": 2000,
        "stream": True,
        "thinking": {"type": "adaptive"},
        "messages": [
            {"role": "user", "content": "Why is 1/0 undefined? Two sentences."}
        ],
    },
    region=REGION,
    headers=AV,
):
    if not line.startswith("data: "):
        continue
    try:
        event = json.loads(line[6:])
    except json.JSONDecodeError:
        continue
    etype = event.get("type", "")
    events[etype] = events.get(etype, 0) + 1
    if etype == "content_block_delta":
        delta = event.get("delta", {})
        if delta.get("type") == "thinking_delta":
            chunk = delta.get("thinking", "")
            thinking_chars += len(chunk)
            print("\033[2m" + chunk + "\033[0m", end="", flush=True)
        elif delta.get("type") == "text_delta":
            chunk = delta.get("text", "")
            text_chars += len(chunk)
            print(chunk, end="", flush=True)

print(f"\n\nthinking chars: {thinking_chars} | answer chars: {text_chars}")
print("--- event types ---")
for name, count in sorted(events.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4}  {name}")

# %% [markdown]
# ## 5. Tool use
#
# The Messages API uses `input_schema` (not `parameters`), and tool results go back
# as a `tool_result` block inside a **user** turn.

# %%
RATES = {("EUR", "USD"): 1.09, ("USD", "SGD"): 1.34}


def convert(amount: float, source: str, target: str) -> dict:
    rate = RATES.get((source.upper(), target.upper()))
    return {
        "amount": amount,
        "from": source.upper(),
        "to": target.upper(),
        "rate": rate,
        "converted": round(amount * rate, 2) if rate else None,
    }


convert_tool = {
    "name": "convert_currency",
    "description": "Convert an amount between two currencies.",
    "input_schema": {  # note: input_schema, not parameters
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "source": {"type": "string"},
            "target": {"type": "string"},
        },
        "required": ["amount", "source", "target"],
    },
}

history = [{"role": "user", "content": "How much is 500 EUR in USD?"}]
code, first = post(
    f"{PREFIX}/messages",
    {"model": SONNET5, "max_tokens": 600, "tools": [convert_tool], "messages": history},
    region=REGION,
    headers=AV,
)
print("stop_reason:", first.get("stop_reason"))
tool_uses = [b for b in first.get("content", []) if b.get("type") == "tool_use"]
print("tool_use blocks:", [(b["name"], b["input"]) for b in tool_uses])

# %%
if tool_uses:
    # Echo the assistant turn, then return results in a USER turn.
    history.append({"role": "assistant", "content": first["content"]})
    results = []
    for block in tool_uses:
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block["id"],  # ties result to request
                "content": json.dumps(convert(**block["input"])),
            }
        )
    history.append({"role": "user", "content": results})

    code, final = post(
        f"{PREFIX}/messages",
        {
            "model": SONNET5,
            "max_tokens": 400,
            "tools": [convert_tool],
            "messages": history,
        },
        region=REGION,
        headers=AV,
    )
    print("final answer:", claude_text(final)[:250])

# %% [markdown]
# ## 6. Forced tools = structured output
#
# Claude's `output_config.format` is **rejected on mantle** (proved in `01`), so a
# forced tool is the way to get schema-enforced JSON. The result arrives already
# parsed as a dict — no string parsing, no trailing-character risk.

# %%
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string"},
    },
    "required": ["risk_level", "concerns", "recommendation"],
}

emit_tool = {
    "name": "emit_analysis",
    "description": "Return the structured risk analysis.",
    "input_schema": ANALYSIS_SCHEMA,
}


def structured_analysis(text: str, model: str = SONNET5, attempts: int = 3) -> dict:
    """Forced-tool structured output with a retry.

    Forcing a tool is honoured almost always, not always — handle the turn that
    comes back as prose instead of indexing blindly.
    """
    for attempt in range(attempts):
        code, data = post(
            f"{PREFIX}/messages",
            {
                "model": model,
                "max_tokens": 900,
                "tools": [emit_tool],
                "tool_choice": {"type": "tool", "name": "emit_analysis"},
                "messages": [{"role": "user", "content": text}],
            },
            region=REGION,
            headers=AV,
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        blocks = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
        if blocks:
            if attempt:
                print(f"(succeeded on attempt {attempt + 1})")
            return blocks[0]["input"]  # already a dict
        print(f"attempt {attempt + 1}: no tool_use block — retrying")
    raise RuntimeError("model would not emit the forced tool")


print(
    json.dumps(
        structured_analysis(
            "Review this clause: 'The supplier accepts unlimited liability for all "
            "losses.'"
        ),
        indent=2,
    )
)

# %% [markdown]
# ## 7. Prompt caching
#
# Mark a cacheable prefix with `cache_control` on a content block. Order matters:
# checkpoints are processed **tools → system → messages**, and changing an earlier
# section invalidates the later ones.

# %%
HANDBOOK = (
    """
PLATFORM HANDBOOK (excerpt)

1. Tiers. Priority for latency-critical customer paths. Standard by default.
   Flex for evaluations and batch-shaped work that tolerates queueing.
2. Retries. Exponential backoff with jitter. 429 and 5xx are transient and must
   be retried. Other 4xx indicate a client defect and must not be retried.
3. Retention. Workloads handling regulated data run with zero data retention and
   replay conversation history client-side.
4. Observability. Only client-error metrics are published; track server errors
   from client-side telemetry.
5. Attribution. Every workload runs under its own tagged project. Untagged usage
   is charged to a shared pool.
"""
    * 8
)  # comfortably over the per-checkpoint token minimum

print(f"handbook ≈ {len(HANDBOOK) // 4} tokens")


def cached_ask(question: str, model: str = SONNET5, ttl: str | None = None) -> dict:
    cache_control = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": model,
            "max_tokens": 300,
            "system": [
                {"type": "text", "text": HANDBOOK, "cache_control": cache_control}
            ],
            "messages": [{"role": "user", "content": question}],
        },
        region=REGION,
        headers=AV,
    )
    if code != 200:
        raise RuntimeError(f"HTTP {code}: {err(data)}")
    usage = data.get("usage", {})
    return {
        "text": claude_text(data),
        "input": usage.get("input_tokens"),
        "cache_write": usage.get("cache_creation_input_tokens"),
        "cache_read": usage.get("cache_read_input_tokens"),
    }


cold = cached_ask("Which tier suits evaluations? One line.")
print("call 1 (cold):", {k: v for k, v in cold.items() if k != "text"})
print("   answer:", cold["text"][:90])

warm = cached_ask("What must untagged usage expect? One line.")
print("\ncall 2 (warm):", {k: v for k, v in warm.items() if k != "text"})
print("   answer:", warm["text"][:90])

if warm["cache_read"]:
    print(
        f"\n{warm['cache_read']} tokens read from cache — the handbook was paid for "
        "once"
    )

# %% [markdown]
# ## 8. The one-hour TTL
#
# Claude supports a **1-hour** cache TTL as well as the default 5 minutes. Use it
# when follow-up requests may arrive more than 5 minutes apart — a long agent
# side-quest, or a user who takes their time replying.

# %%
for ttl in (None, "5m", "1h"):
    label = ttl or "default (5m)"
    try:
        result = cached_ask("Name one retry rule. One line.", ttl=ttl)
        print(
            f"  ttl={label:12} -> write={result['cache_write']} "
            f"read={result['cache_read']}"
        )
    except RuntimeError as exc:
        print(f"  ttl={label:12} -> {exc}")

# %% [markdown]
# **Constraint:** if you mix TTLs in one request, longer-TTL entries must appear
# **before** shorter ones. And note cache hits are not charged against your rate
# limit, so caching buys throughput headroom too.

# %% [markdown]
# ## 9. Caching tools and system together
#
# The agentic pattern: tool definitions and instructions are static, the
# conversation grows. Cache the static part once.


# %%
def agent_turn(question: str, model: str = SONNET5) -> dict:
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": model,
            "max_tokens": 500,
            # Checkpoint order is tools -> system -> messages.
            "tools": [
                {**convert_tool, "cache_control": {"type": "ephemeral"}},
            ],
            "system": [
                {
                    "type": "text",
                    "text": HANDBOOK,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": question}],
        },
        region=REGION,
        headers=AV,
    )
    if code != 200:
        raise RuntimeError(f"HTTP {code}: {err(data)}")
    usage = data.get("usage", {})
    return {
        "input": usage.get("input_tokens"),
        "write": usage.get("cache_creation_input_tokens"),
        "read": usage.get("cache_read_input_tokens"),
        "text": claude_text(data),
    }


print(f"{'turn':6} {'input':>8} {'write':>8} {'read':>8}  answer")
print("-" * 84)
for i, question in enumerate(
    [
        "Which tier for a nightly eval job?",
        "Which tier for live chat?",
        "Do we retry a 400?",
    ],
    1,
):
    r = agent_turn(question)
    print(
        f"{i:>6} {r['input']:>8} {r['write'] or 0:>8} {r['read'] or 0:>8}  "
        f"{' '.join(r['text'].split())[:40]!r}"
    )
print("\nTurn 1 writes the prefix; later turns read it.")

# %% [markdown]
# ## 10. Count tokens before you spend them
#
# `count_tokens` is mantle-only and includes system prompts and tool definitions —
# exactly the parts people forget when budgeting.

# %%
for label, body in [
    ("bare prompt", {"messages": [{"role": "user", "content": "Hi"}]}),
    (
        "+ handbook system",
        {"system": HANDBOOK, "messages": [{"role": "user", "content": "Hi"}]},
    ),
    (
        "+ handbook + tool",
        {
            "system": HANDBOOK,
            "tools": [convert_tool],
            "messages": [{"role": "user", "content": "Hi"}],
        },
    ),
]:
    code, data = post(
        f"{PREFIX}/messages/count_tokens",
        {"model": SONNET5, **body},
        region=REGION,
        headers=AV,
    )
    print(f"  {label:20} -> {data.get('input_tokens')} tokens")

# %% [markdown]
# ## 11. Compare thinking across models
#
# Same puzzle, different models, measuring what thinking costs.

# %%
QUESTION = (
    "Two trains 120km apart approach at 40km/h and 60km/h. A bird flies "
    "between them at 80km/h. How far does the bird fly before they meet?"
)

print(f"{'model':32} {'mode':10} {'in':>6} {'out':>6} {'latency':>9}  answer")
print("-" * 104)
for model in (HAIKU45, OPUS48, SONNET5):
    for mode in ("plain", "adaptive"):
        body = {
            "model": model,
            "max_tokens": 2500,
            "messages": [{"role": "user", "content": QUESTION}],
        }
        if mode == "adaptive":
            body["thinking"] = {"type": "adaptive"}
        started = time.perf_counter()
        code, data = post(f"{PREFIX}/messages", body, region=REGION, headers=AV)
        elapsed = time.perf_counter() - started
        if code != 200:
            print(
                f"{model:32} {mode:10} {'-':>6} {'-':>6} {'-':>9}  "
                f"HTTP {code}: {err(data)[:30]}"
            )
            continue
        usage = data.get("usage", {})
        answer = " ".join(claude_text(data).split())
        print(
            f"{model:32} {mode:10} {usage.get('input_tokens', 0):>6} "
            f"{usage.get('output_tokens', 0):>6} {elapsed:>8.2f}s  {answer[:34]!r}"
        )

# %% [markdown]
# ## 12. Production shape

# %%
code, workspace = post(
    "/v1/organization/projects",  # control plane is always /v1
    {
        "name": "claude-thinking-samples",
        "tags": {"Application": "ClaudeThinkingDemo", "Environment": "Demo"},
    },
    region=REGION,
)
workspace_id = workspace.get("id")
print("workspace:", code, workspace_id)


class ClaudeClient:
    """Production shape: caching, optional thinking, forced-tool JSON, attribution."""

    ADAPTIVE_MODELS = (OPUS5, SONNET5, OPUS48)  # haiku-4-5 rejects adaptive

    def __init__(
        self, model=SONNET5, region=REGION, workspace=None, system=None, cache_ttl="1h"
    ):
        self.model, self.region, self.workspace = model, region, workspace
        self.system, self.cache_ttl = system, cache_ttl

    def _headers(self):
        headers = dict(AV)
        if self.workspace:
            headers["anthropic-workspace"] = self.workspace
        return headers

    def _system_blocks(self):
        if not self.system:
            return None
        return [
            {
                "type": "text",
                "text": self.system,
                "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl},
            }
        ]

    def ask(self, question, *, thinking=False, max_tokens=1000, schema=None):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": question}],
        }
        # Never send temperature: deprecated on frontier Claude models.
        if self.system:
            body["system"] = self._system_blocks()
        if thinking and self.model in self.ADAPTIVE_MODELS:
            body["thinking"] = {"type": "adaptive"}
        if schema:
            # output_config.format is rejected on mantle — force a tool instead.
            body["tools"] = [
                {
                    "name": "emit",
                    "description": "Return the result.",
                    "input_schema": schema,
                }
            ]
            body["tool_choice"] = {"type": "tool", "name": "emit"}
        code, data = post(
            f"{PREFIX}/messages", body, region=self.region, headers=self._headers()
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        if schema:
            blocks = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
            if not blocks:
                raise RuntimeError("model did not emit the forced tool")
            return blocks[0]["input"]
        return claude_text(data)


bot = ClaudeClient(system=HANDBOOK, workspace=workspace_id)
print("plain     :", bot.ask("Which tier for batch evals? One line.")[:110])
print(
    "thinking  :", bot.ask("Do we retry a 503? Explain briefly.", thinking=True)[:110]
)
print(
    "structured:",
    json.dumps(
        bot.ask(
            "Assess: 'Unlimited liability for the supplier.'", schema=ANALYSIS_SCHEMA
        )
    ),
)

# %%
code, archived = post(
    f"/v1/organization/projects/{workspace_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — thinking, tools and caching on Claude
#
# | Gotcha | Detail |
# |---|---|
# | Adaptive thinking | **Not on `haiku-4-5`** — 400. Gate per model |
# | `content[0]` | Thinking blocks come first — filter by block type |
# | Thinking + sampling | `temperature`/`top_p` rejected alongside thinking |
# | Tool schema key | `input_schema`, not `parameters` |
# | Tool results | Go back in a **user** turn as `tool_result` blocks |
# | `output_config.format` | Rejected on mantle — force a tool instead |
# | Forced tool | Best-effort; handle the prose turn |
# | Cache order | tools → system → messages; earlier edits invalidate later |
# | Mixed TTLs | Longer TTL entries must precede shorter ones |
# | Cache hits and quota | Not charged against your rate limit |
# | `count_tokens` | mantle-only; counts system + tools |
#
# ## Next
# - `03-agentic-computer-use-and-memory.ipynb` — computer use, memory, compaction
# - Other caching model: `../01-openai-gpt/04-prompt-caching-and-cost.ipynb`
