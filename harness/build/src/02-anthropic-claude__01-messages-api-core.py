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
# # Anthropic Claude on Amazon Bedrock Mantle — Messages API core
#
# Claude on the `bedrock-mantle` endpoint speaks the **Anthropic-native Messages
# API**. This notebook covers the fundamentals; `02-thinking-tools-and-caching.ipynb`
# covers extended thinking, tool use, and prompt caching, and
# `03-agentic-computer-use-and-memory.ipynb` covers the agentic tool families.
#
# **Models covered**
#
# | Model ID | Notes |
# |---|---|
# | `anthropic.claude-opus-5` | Frontier |
# | `anthropic.claude-sonnet-5` | Balanced frontier |
# | `anthropic.claude-opus-4-8` | Previous-generation Opus |
# | `anthropic.claude-opus-4-7` | Earlier Opus |
# | `anthropic.claude-haiku-4-5` | Fast and cheap; different parameter support |
# | `anthropic.claude-fable-5` | Specialised variant (may be gated by data-retention mode — see §4b) |
#
# ## Which API? Messages — and *only* Messages
# On `bedrock-mantle`, Claude models do **not** serve the Responses API or Chat
# Completions. Both return 400. We prove that in §2.
#
# ## Region matters here more than anywhere else
# `us-east-1` is the only Region carrying the full Claude set (as of August 2026;
# verify with `GET /v1/models` per Region). `us-west-2` has just
# `claude-haiku-4-5`, and `us-east-2` / `eu-central-1` carry **no Anthropic models
# at all**. This notebook pins `us-east-1`.
#
# ## Self-contained, but see also
# - **Auth (SigV4 (AWS Signature Version 4) + short-term keys), the three URL paths,
#   model discovery** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Projects/Workspaces, cost attribution, data retention / ZDR (zero data retention),
#   CloudWatch** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
# - **Quotas, retry/backoff, service tiers, TTFT (time-to-first-token)** →
#   `../00-foundations/03-scaling-tiers-and-latency.ipynb`
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

REGION = "us-east-1"  # full Claude set as of Aug 2026; verify per Region

OPUS5 = "anthropic.claude-opus-5"
SONNET5 = "anthropic.claude-sonnet-5"
OPUS48 = "anthropic.claude-opus-4-8"
HAIKU45 = "anthropic.claude-haiku-4-5"
OPUS47 = "anthropic.claude-opus-4-7"

# Claude has its own path prefix on mantle — neither /v1 nor /openai/v1.
PREFIX = "/anthropic/v1"
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws/anthropic"
print("base URL:", BASE_URL)

# The anthropic-version header is REQUIRED on every mantle Messages request.
# (On bedrock-runtime the equivalent goes in the body as "anthropic_version".)
AV = {"anthropic-version": "2023-06-01"}


def claude_text(payload: dict) -> str:
    """Concatenate the TEXT blocks of a Messages response, ignoring thinking blocks.

    Never index content[0] blindly: reasoning-capable models put a `thinking`
    block first. Section 2b demonstrates this.
    """
    return "".join(
        b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"
    )


# %% [markdown]
# ## 1. First call
#
# Auth is a short-term Bedrock API key minted from ambient IAM credentials —
# valid ≤12 h, not refreshable, Region-pinned. (`../00-foundations/01` shows the
# self-refreshing provider and the SigV4 alternative.)

# %%
code, data = post(
    f"{PREFIX}/messages",
    {
        "model": SONNET5,
        "max_tokens": 300,  # REQUIRED on the Messages API, unlike OpenAI APIs
        "messages": [
            {
                "role": "user",
                "content": ("Explain idempotency in two sentences."),
            }
        ],
    },
    region=REGION,
    headers=AV,
)
print("HTTP", code)
print(claude_text(data))
print("\nusage:", json.dumps(data["usage"]))
print("stop_reason:", data["stop_reason"])

# %% [markdown]
# Note `max_tokens` is **mandatory** here. Omit it and the request is rejected —
# the OpenAI-compatible APIs default it for you, Messages does not.

# %%
code, data = post(
    f"{PREFIX}/messages",
    {"model": SONNET5, "messages": [{"role": "user", "content": "Hi"}]},
    region=REGION,
    headers=AV,
)
print(f"omitting max_tokens -> HTTP {code}: {err(data)[:110]}")

# %% [markdown]
# ## 2. Why Messages and not the OpenAI-compatible APIs
#
# Probe all three surfaces so the 400s are visible rather than asserted:

# %%
for label, path, body in [
    (
        "Messages",
        f"{PREFIX}/messages",
        {
            "model": SONNET5,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply OK"}],
        },
    ),
    (
        "Responses /v1",
        "/v1/responses",
        {"model": SONNET5, "input": "Reply OK", "max_output_tokens": 16},
    ),
    (
        "Responses /openai/v1",
        "/openai/v1/responses",
        {"model": SONNET5, "input": "Reply OK", "max_output_tokens": 16},
    ),
    (
        "ChatCompletions /v1",
        "/v1/chat/completions",
        {
            "model": SONNET5,
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_tokens": 16,
        },
    ),
]:
    code, resp = post(path, body, region=REGION, headers=AV)
    print(f"  {label:22} -> HTTP {code} {'' if code == 200 else err(resp)[:60]}")

# %% [markdown]
# So on mantle, Claude is Messages-only. Practical consequences:
#
# - No `previous_response_id`; you manage conversation history yourself.
# - Structured output uses tools, not `response_format` / `text.format` (§7).
# - Cost attribution uses the `anthropic-workspace` header, not `OpenAI-Project`.

# %% [markdown]
# ## 2b. Read the content array properly — never assume `content[0]` is text
#
# Reasoning-capable Claude models return a **`thinking` block before the text
# block**. `content[0].text` therefore raises or returns nothing on those models.
# Always filter by block type.

# %%
for model in (SONNET5, OPUS5):
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": model,
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": "What problem does a write-ahead log solve?",
                }
            ],
        },
        region=REGION,
        headers=AV,
    )
    kinds = [b.get("type") for b in data.get("content", [])]
    print(f"{model:30} blocks={kinds}")
    print(f"    text via helper: {claude_text(data).strip()[:80]!r}")
    first = data["content"][0]
    naive_verdict = "work" if first.get("type") == "text" else "FAIL"
    print(
        f"    content[0]['type'] = {first.get('type')!r} "
        f"-> naive content[0]['text'] would {naive_verdict}"
    )

# %% [markdown]
# ## 3. The Anthropic SDK
#
# Point the official SDK at mantle by overriding `base_url`. Note the base URL
# omits the `/v1` — the SDK appends it.

# %%
import anthropic
from aws_bedrock_token_generator import provide_token

client = anthropic.Anthropic(
    api_key=provide_token(region=REGION),  # fresh token; do not cache for hours
    base_url=BASE_URL,
)

message = client.messages.create(
    model=SONNET5,
    max_tokens=250,
    system="You are terse and precise.",
    messages=[
        {
            "role": "user",
            "content": ("What problem does a write-ahead log solve?"),
        }
    ],
)
# Filter by block type rather than indexing [0] — see section 2b.
print("".join(b.text for b in message.content if b.type == "text"))
print("\nmodel echoed:", message.model, "| stop:", message.stop_reason)

# %% [markdown]
# ## 4. `temperature` is deprecated on newer Claude models
#
# This is the parameter trap for this family. Frontier models reject `temperature`
# outright; `haiku-4-5` still accepts it. Never share a sampling config across
# Claude generations.

# %%
print(f"{'model':32} {'temperature=0.5':>16}")
print("-" * 50)
for model in (OPUS5, SONNET5, OPUS48, OPUS47, HAIKU45):
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": model,
            "max_tokens": 16,
            "temperature": 0.5,
            "messages": [{"role": "user", "content": "Reply OK"}],
        },
        region=REGION,
        headers=AV,
    )
    verdict = "accepted" if code == 200 else f"{code}: {err(data)[:40]}"
    print(f"{model:32} {verdict:>16}")

# %% [markdown]
# The fix is to omit it — the newer models are tuned to run without
# sampling knobs.

# %%
for model in (OPUS5, HAIKU45):
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": model,
            "max_tokens": 60,
            "messages": [{"role": "user", "content": "One word: capital of Japan?"}],
        },
        region=REGION,
        headers=AV,
    )
    print(f"{model:32} HTTP {code} -> {claude_text(data).strip()[:40]!r}")

# %% [markdown]
# ## 4b. A model can be gated by your data-retention mode
#
# Some models are only offered under specific data-retention modes. If your
# account's mode is not in the model's `allowed_modes`, inference returns 400 and
# `GET /v1/models/{id}` reports `status: unavailable` with a reason. Check before
# you debug your request shape.

# %%
code, retention = post("/v1/data_retention", None, region=REGION, method="GET")
print("account data_retention:", retention)

for mid in (SONNET5, "anthropic.claude-fable-5"):
    code, info = post(f"/v1/models/{mid}", None, region=REGION, method="GET")
    dr = info.get("data_retention", {})
    print(f"\n{mid}")
    print(f"   status       : {info.get('status')}")
    print(f"   allowed_modes: {dr.get('allowed_modes')}")
    if info.get("status_reason"):
        print(f"   reason       : {info['status_reason'][:110]}")

# %% [markdown]
# So `allowed_modes` **is** returned for some models — treat it as authoritative
# when present. Changing account or project retention mode changes which models you
# can call (see `../00-foundations/02`).

# %% [markdown]
# ## 5. Streaming
#
# Messages streams **named SSE (server-sent events) events** — `message_start`,
# `content_block_delta`,
# `message_stop` — which is a different shape from the OpenAI APIs' `data: {…}`
# frames.

# %%
event_counts = {}
print("--- live stream ---")
with client.messages.stream(
    model=SONNET5,
    max_tokens=300,
    messages=[{"role": "user", "content": "List three properties of a good API."}],
) as stream:
    for event in stream:
        event_counts[event.type] = event_counts.get(event.type, 0) + 1
        if event.type == "text":
            print(event.text, end="", flush=True)

print("\n\n--- event types ---")
for name, count in sorted(event_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4}  {name}")

# %%
# The raw SSE, for anyone not using the SDK.
from mantle import stream_lines

shown = 0
for line in stream_lines(
    f"{PREFIX}/messages",
    {
        "model": HAIKU45,
        "max_tokens": 40,
        "stream": True,
        "messages": [{"role": "user", "content": "Count to three."}],
    },
    region=REGION,
    headers=AV,
):
    print(line[:110])
    shown += 1
    if shown >= 8:
        print("…")
        break

# %% [markdown]
# ## 6. Multi-turn, system prompts, and prefill
#
# Messages must alternate user/assistant and **start with `user`**. The system
# prompt is a separate top-level parameter, not a message.

# %%
history = [{"role": "user", "content": "Name a NoSQL database."}]
first = client.messages.create(
    model=HAIKU45,
    max_tokens=100,
    system="Answer with the name only, no punctuation.",
    messages=history,
)
answer = "".join(b.text for b in first.content if b.type == "text").strip()
print("assistant:", answer)

history += [
    {"role": "assistant", "content": answer},
    {
        "role": "user",
        "content": "Now name its main consistency trade-off in one sentence.",
    },
]
second = client.messages.create(model=HAIKU45, max_tokens=150, messages=history)
print("assistant:", "".join(b.text for b in second.content if b.type == "text").strip())
print(f"\ninput tokens grew: {first.usage.input_tokens} -> {second.usage.input_tokens}")

# %%
# Prefill: seed the assistant turn to constrain the format. Claude continues from
# your partial text — very effective for forcing JSON or a fixed prefix.
prefilled = client.messages.create(
    model=HAIKU45,
    max_tokens=150,
    messages=[
        {"role": "user", "content": "Give three AWS Regions in the US."},
        {"role": "assistant", "content": "1. us-east-1"},  # prefill
    ],
)
print(
    "continuation:",
    "".join(b.text for b in prefilled.content if b.type == "text")[:150],
)

# %% [markdown]
# ## 7. Structured output — use tools, not `output_config`
#
# The Messages API has an `output_config.format` structured-output parameter, but
# it is **rejected on mantle**. Verify, then use the portable alternative.

# %%
code, data = post(
    f"{PREFIX}/messages",
    {
        "model": SONNET5,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Describe France."}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"capital": {"type": "string"}},
                },
            }
        },
    },
    region=REGION,
    headers=AV,
)
print(f"output_config.format -> HTTP {code}: {err(data)[:120]}")
print("\n=> For structured outputs with Claude either force a tool (below),")
print("   or use the Converse API on the bedrock-runtime endpoint.")

# %%
# The portable route: a forced tool whose input schema IS your output schema.
country_tool = {
    "name": "emit_country",
    "description": "Return structured facts about a country.",
    "input_schema": {
        "type": "object",
        "properties": {
            "country": {"type": "string"},
            "capital": {"type": "string"},
            "population_millions": {"type": "number"},
        },
        "required": ["country", "capital", "population_millions"],
    },
}

structured = client.messages.create(
    model=SONNET5,
    max_tokens=400,
    tools=[country_tool],
    tool_choice={"type": "tool", "name": "emit_country"},  # force it
    messages=[{"role": "user", "content": "Describe France."}],
)
tool_blocks = [b for b in structured.content if b.type == "tool_use"]
print("stop_reason:", structured.stop_reason)
print(json.dumps(tool_blocks[0].input, indent=2))

# %% [markdown]
# Because the schema is enforced at the tool-call layer, the result is already a
# dict — no string parsing, and no risk of trailing characters breaking a JSON
# parse (a real problem with some other mantle families).

# %% [markdown]
# ## 8. Count tokens before you spend them
#
# `count_tokens` is available on **mantle only** — it is not on `bedrock-runtime`.
# Useful for pre-flight cost estimation and for deciding whether a prompt fits.

# %%
for label, text in [
    ("short", "Hello world"),
    ("medium", "Explain distributed consensus. " * 20),
    ("long", "Explain distributed consensus. " * 200),
]:
    code, data = post(
        f"{PREFIX}/messages/count_tokens",
        {"model": OPUS5, "messages": [{"role": "user", "content": text}]},
        region=REGION,
        headers=AV,
    )
    print(f"  {label:8} chars={len(text):6} -> input_tokens={data.get('input_tokens')}")

# %%
# It also counts system prompts and tool definitions, which is where budgets
# usually go missing.
code, bare = post(
    f"{PREFIX}/messages/count_tokens",
    {"model": OPUS5, "messages": [{"role": "user", "content": "Hi"}]},
    region=REGION,
    headers=AV,
)
code, with_extras = post(
    f"{PREFIX}/messages/count_tokens",
    {
        "model": OPUS5,
        "system": "You are a meticulous assistant that always cites sources.",
        "tools": [country_tool],
        "messages": [{"role": "user", "content": "Hi"}],
    },
    region=REGION,
    headers=AV,
)
print(f"bare prompt          : {bare['input_tokens']} tokens")
print(f"+ system + 1 tool def: {with_extras['input_tokens']} tokens")
print(
    f"overhead             : {with_extras['input_tokens'] - bare['input_tokens']} "
    "tokens"
)

# %% [markdown]
# ## 9. Vision
#
# Images go in the `content` array as `source` blocks (base64 or URL).

# %%
import base64
import struct
import zlib


def make_png(width: int, height: int, rgb: tuple) -> bytes:
    """Solid-colour PNG with no third-party dependencies."""
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


green_b64 = base64.b64encode(make_png(64, 64, (20, 160, 60))).decode()

vision = client.messages.create(
    model=SONNET5,
    max_tokens=60,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": green_b64,
                    },
                },
                {"type": "text", "text": "What colour is this image? One word."},
            ],
        }
    ],
)
print(
    "model sees:",
    repr("".join(b.text for b in vision.content if b.type == "text").strip()),
)

# %% [markdown]
# ## 10. Compare the models
#
# Same prompt across generations. Note we omit `temperature` so the call is valid
# on every model (see §4).

# %%
task = "In one sentence, why is exactly-once delivery hard in distributed systems?"

print(f"{'model':34} {'latency':>9} {'in':>5} {'out':>5}  answer")
print("-" * 112)
for model in (HAIKU45, OPUS47, OPUS48, SONNET5, OPUS5):
    started = time.perf_counter()
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": model,
            "max_tokens": 160,
            "messages": [{"role": "user", "content": task}],
        },
        region=REGION,
        headers=AV,
    )
    elapsed = time.perf_counter() - started
    if code != 200:
        print(f"{model:34} {'-':>9} {'-':>5} {'-':>5}  HTTP {code}: {err(data)[:40]}")
        continue
    text = " ".join(claude_text(data).split())
    usage = data["usage"]
    print(
        f"{model:34} {elapsed:>8.2f}s {usage['input_tokens']:>5} "
        f"{usage['output_tokens']:>5}  {text[:46]!r}"
    )

# %% [markdown]
# ## 11. Regional reality check
#
# Claude's Region footprint on mantle is narrow. Verify before you deploy.

# %%
from mantle import list_models

print(f"{'region':14} {'anthropic models':>17}")
print("-" * 34)
for region in ("us-east-1", "us-east-2", "us-west-2", "eu-central-1"):
    try:
        claude = [m for m in list_models(region) if m.startswith("anthropic.")]
    except (RuntimeError, OSError) as exc:
        print(f"{region:14} error: {type(exc).__name__}")
        continue
    short_names = ", ".join(c.split(".")[-1] for c in claude)
    print(f"{region:14} {len(claude):>17}  {short_names or '(none)'}")

# %% [markdown]
# ## 12. Cost attribution with Workspaces
#
# Same underlying resource as OpenAI-API "Projects", but referenced with the
# `anthropic-workspace` header. (Details: `../00-foundations/02`.)

# %%
code, workspace = post(
    "/v1/organization/projects",  # control plane is always /v1
    {
        "name": "claude-samples",
        "tags": {"Application": "ClaudeDemo", "Environment": "Demo"},
    },
    region=REGION,
)
workspace_id = workspace.get("id")
print("workspace:", code, workspace_id)

code, data = post(
    f"{PREFIX}/messages",
    {
        "model": HAIKU45,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply OK"}],
    },
    region=REGION,
    headers={**AV, "anthropic-workspace": workspace_id},
)
print("attributed call ->", code)

code, archived = post(
    f"/v1/organization/projects/{workspace_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — Claude on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | Messages only | Responses **and** Chat Completions both 400 on mantle |
# | Path prefix | `/anthropic/v1`, not `/v1` or `/openai/v1` |
# | `anthropic-version` | Required **header** on mantle (body field on runtime) |
# | `max_tokens` | Mandatory — omitting it is a 400 |
# | `temperature` | **Deprecated → 400** on opus-5/sonnet-5/opus-4-8; fine on haiku-4-5 |
# | `output_config.format` | Rejected on mantle — force a tool, or use Converse on runtime |
# | Region | Full set only in us-east-1; haiku-only in us-west-2; none in us-east-2 / eu-central-1 |
# | Attribution header | `anthropic-workspace`, not `OpenAI-Project` |
# | `count_tokens` | mantle-only; counts system + tool definitions too |
# | Streaming shape | Named SSE events, not `data: {…}` frames |
# | `content[0]` | Reasoning models lead with a `thinking` block — filter by type |
# | Retention gating | A model can be `unavailable` under your mode; check `allowed_modes` |
#
# ## Next
# - `02-thinking-tools-and-caching.ipynb` — adaptive thinking, tool loops, prompt
#   caching
# - `03-agentic-computer-use-and-memory.ipynb` — computer use, memory, compaction
# - Other families: `../01-openai-gpt/`, `../03-google-gemma/`, `../04-qwen/`
