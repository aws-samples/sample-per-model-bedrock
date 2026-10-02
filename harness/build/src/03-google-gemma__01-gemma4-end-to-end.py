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
# # Gemma 4 on Amazon Bedrock Mantle — end to end
#
# Google DeepMind's Gemma 4 family (Apache 2.0) on the `bedrock-mantle` endpoint:
# simple inference → streaming → reasoning → stateful chat → tool use →
# structured JSON → multimodal → production hardening.
#
# **Three variants, interleaved throughout this notebook:**
#
# | Model ID | Architecture | Params | Context |
# |---|---|---|---|
# | `google.gemma-4-31b` | Dense | 30.7B | 256K |
# | `google.gemma-4-26b-a4b` | Mixture-of-Experts | 25.2B total / 3.8B active | 256K |
# | `google.gemma-4-e2b` | Dense (Per-Layer Embeddings) | 5.1B total / 2.3B effective | 128K |
#
# **Gemma 4 is `bedrock-mantle`-only.** There is no `bedrock-runtime` support —
# `invoke_model` and `converse` return errors for these model IDs.
#
# ## Self-contained, but see also
# This notebook stands alone. For deeper background:
# - **Auth, the three URL paths, model discovery** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Projects, cost attribution, data retention/ZDR, CloudWatch** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
# - **Quotas, retries, service tiers, TTFT (time-to-first-token) benchmarking** →
#   `../00-foundations/03-scaling-tiers-and-latency.ipynb`
#
# ## Prerequisites
# ```bash
# pip install -r ../requirements.txt
# ```
#
# Needs openai, aws-bedrock-token-generator.
#
# `requirements.txt` pins the exact versions this collection was tested
# against. An unpinned install resolves whatever is current, which may be
# untested or compromised (OWASP LLM03, Supply Chain).
#
# You also need AWS credentials with `bedrock-mantle:CreateInference` and
# `bedrock-mantle:CallWithBearerToken` (the managed policy
# `AmazonBedrockMantleInferenceAccess` grants both).

# %%
import base64
import json
import sys
import time

sys.path.insert(0, "../_shared")
from mantle import (
    err,
    function_calls,
    parse_json_lenient,
    post,
    redact_ids,
    response_text,
    stream_lines,
)

# Gemma 4 is available in ALL FOUR mantle Regions — the only family that is.
REGION = "us-east-1"

DENSE = "google.gemma-4-31b"
MOE = "google.gemma-4-26b-a4b"
COMPACT = "google.gemma-4-e2b"

# NOTE the "/openai" prefix. Gemma 4's model card calls this out explicitly:
# its paths differ from the bare /v1 used by most other mantle models.
PREFIX = "/openai/v1"
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}"
print("base URL:", BASE_URL)

# %% [markdown]
# ## 1. Simple inference with the Responses API
#
# The Responses API is AWS's recommended surface for new applications, and for
# Gemma 4 specifically it is the **only** way to read reasoning output (see §3).
#
# Auth here uses a short-term Bedrock API key minted from ambient IAM credentials.
# (Full explanation, including the SigV4 (AWS Signature Version 4) alternative that
# needs no key at all, is
# in `../00-foundations/01`.)

# %%
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

# Build the client from a FRESH token. Tokens last <=12h and cannot be refreshed,
# so don't construct one at import time and reuse it for hours.
client = OpenAI(api_key=provide_token(region=REGION), base_url=BASE_URL)

response = client.responses.create(
    model=DENSE,
    input="Explain what a mixture-of-experts model is, in two sentences.",
    max_output_tokens=200,
)
print(response.output_text)

# %% [markdown]
# ## 2. Sampling parameters — the Gemma 4 trap
#
# This is the single most surprising thing about Gemma 4 on the Responses API:
#
# 1. **`temperature` accepts only its default value, `1.0`.** Any other value —
#    including `0.0` and `0.7` — is rejected with
#    *"Unsupported parameter: 'temperature' is not supported with this model."*
# 2. **`top_p` is rejected outright**, even though AWS's own launch blog recommends
#    `top_p=0.95`. That guidance applies to Chat Completions, not Responses.
#
# So on Responses you effectively cannot tune sampling for this model. Sweep the
# values and see for yourself:

# %%
print(f"{'temperature':>12} {'HTTP':>6}  detail")
print("-" * 74)
for value in (0.0, 0.2, 0.5, 0.7, 0.9, 1.0, 1.5):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": DENSE,
            "input": "Reply with exactly: OK",
            "max_output_tokens": 16,
            "temperature": value,
        },
        region=REGION,
    )
    print(f"{value:>12} {code:>6}  {'' if code == 200 else err(data)[:52]}")

code, data = post(
    f"{PREFIX}/responses",
    {"model": DENSE, "input": "Reply OK", "max_output_tokens": 16, "top_p": 0.95},
    region=REGION,
)
print(f"\n  top_p=0.95        -> HTTP {code} {err(data)[:70]}")

# %% [markdown]
# Only `1.0` passes — which is Gemma 4's documented default. The pattern across
# mantle's Responses API is that a model accepts `temperature` **only at its own
# default** (Grok's default is `0.7`, so Grok rejects `1.0` and accepts `0.7`).
#
# `1.0` also happens to be the value AWS and Google recommend anyway
# (greedy decoding at `temperature=0` sends Gemma 4 into repetition loops involving
# reserved vocabulary tokens), so the constraint pushes you towards the right
# setting — but it means **you cannot lower the temperature for extraction tasks**
# on this API.
#
# The simplest safe approach on Responses: **omit both parameters** and accept the
# defaults.

# %%
# Chat Completions is different: it accepts the full temperature range.
print("Chat Completions accepts what Responses rejects:")
for value in (0.0, 0.7, 1.0):
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": DENSE,
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_tokens": 16,
            "temperature": value,
        },
        region=REGION,
    )
    detail = "" if code == 200 else err(data)[:60]
    print(f"  temperature={value:<4} -> HTTP {code} {detail}")
print("\n=> If you need to control sampling on Gemma 4, use Chat Completions.")

# %%
# Another undocumented constraint: max_output_tokens has a MINIMUM of 16.
for n in (8, 15, 16):
    code, data = post(
        f"{PREFIX}/responses",
        {"model": DENSE, "input": "Hi", "max_output_tokens": n},
        region=REGION,
    )
    detail = "" if code == 200 else err(data)[:70]
    print(f"  max_output_tokens={n:3} -> HTTP {code} {detail}")

# %% [markdown]
# ## 3. Reasoning — and why the API choice matters
#
# All three variants have built-in reasoning. The critical detail from the model
# card: reasoning effort is honoured on **both** Responses and Chat Completions,
# and the model does the extended thinking either way — but **only the Responses
# API returns the reasoning content**. On Chat Completions you pay for those
# tokens and never see them.

# %%
resp = client.responses.create(
    model=DENSE,
    input=(
        "A train leaves at 3pm travelling 60 km/h. Another leaves an hour later "
        "at 90 km/h from the same station on the same track. When does the second "
        "catch the first?"
    ),
    reasoning={"effort": "high"},
    max_output_tokens=1200,
)

reasoning_blocks = []
for item in resp.output:
    if item.type == "reasoning":
        for block in item.content:
            text = getattr(block, "text", "")
            if text:
                reasoning_blocks.append(text)

print("=== REASONING (visible only on the Responses API) ===")
print(("\n".join(reasoning_blocks))[:700] or "(none returned)")
print("\n=== FINAL ANSWER ===")
print(resp.output_text[:400])
print("\nreasoning tokens:", resp.usage.output_tokens_details.reasoning_tokens)

# %%
# Which effort values are valid? Probe rather than assume.
for effort in ("none", "minimal", "low", "medium", "high"):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": DENSE,
            "input": "2+2?",
            "max_output_tokens": 32,
            "reasoning": {"effort": effort},
        },
        region=REGION,
    )
    print(f"  effort={effort:8} -> HTTP {code} {'' if code == 200 else err(data)[:60]}")

# %% [markdown]
# `minimal` is rejected; the valid ladder is `none` / `low` / `medium` / `high`.
#
# **Variant-specific advice:** for `gemma-4-e2b`, set `effort="high"`. The smallest
# variant reasons extensively by default, and high effort keeps that thinking in
# the dedicated reasoning channel instead of leaking into the final answer.

# %%
for model in (COMPACT, DENSE):
    r = client.responses.create(
        model=model,
        input="If 3 shirts dry in 4 hours, how long for 9 shirts on the same line?",
        reasoning={"effort": "high"},
        max_output_tokens=600,
    )
    reasoning_tokens = r.usage.output_tokens_details.reasoning_tokens
    print(
        f"{model:26} reasoning_tokens={reasoning_tokens:5}  "
        f"answer={r.output_text[:90]!r}"
    )

# %% [markdown]
# ## 4. Streaming
#
# Reasoning and answer text arrive on **separate event types** — that's what lets
# you render a "thinking…" panel distinct from the answer.

# %%
stream = client.responses.create(
    model=DENSE,
    input="List three properties of a good distributed queue.",
    reasoning={"effort": "low"},
    max_output_tokens=400,
    stream=True,
)

event_counts = {}
print("--- live stream ---")
for event in stream:
    event_counts[event.type] = event_counts.get(event.type, 0) + 1
    if event.type == "response.reasoning_text.delta":
        print("\033[2m" + event.delta + "\033[0m", end="", flush=True)
    elif event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print("\n\n--- event types seen ---")
for name, count in sorted(event_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4}  {name}")

# %% [markdown]
# ## 5. Multi-turn: two approaches
#
# ### (a) Send the history yourself
# `input` accepts the same role/content array that Chat Completions calls
# `messages`. Fully stateless — nothing is retained server-side.

# %%
conversation = [
    {"role": "system", "content": "You are terse. Answer in one short sentence."},
    {"role": "user", "content": "What is a MoE model?"},
]
first = client.responses.create(model=MOE, input=conversation, max_output_tokens=120)
print("assistant:", first.output_text)

conversation += [
    {"role": "assistant", "content": first.output_text},
    {"role": "user", "content": "And why is it cheaper to run?"},
]
second = client.responses.create(model=MOE, input=conversation, max_output_tokens=120)
print("assistant:", second.output_text)

# %% [markdown]
# **Important for Gemma 4:** append only the *final answers* to history, never the
# reasoning items. AWS warns that replaying prior reasoning back to the model
# degrades later turns. Keep reasoning in your logs, strip it from `input`.

# %% [markdown]
# ### (b) Server-side state with `previous_response_id`
# Bedrock rebuilds the context for you. Cheaper on input tokens for long chats —
# but it requires `store=True`, which retains input and output for **30 days**
# in-Region (encrypted, project-scoped).

# %%
turn1 = client.responses.create(
    model=DENSE,
    input="My favourite database is DynamoDB. Reply with just: noted.",
    max_output_tokens=32,
    store=True,
)
print("turn 1 id:", redact_ids(turn1.id))

turn2 = client.responses.create(
    model=DENSE,
    input="What is my favourite database?",
    previous_response_id=turn1.id,
    max_output_tokens=48,
)
print("turn 2   :", turn2.output_text)

# %%
# The privacy/convenience trade-off, made concrete.
private = client.responses.create(
    model=DENSE, input="Secret: 42. Reply: ok.", max_output_tokens=16, store=False
)
code, data = post(
    f"{PREFIX}/responses",
    {
        "model": DENSE,
        "input": "What was the secret?",
        "previous_response_id": private.id,
        "max_output_tokens": 32,
    },
    region=REGION,
)
print(f"chaining from a store=False response -> HTTP {code}")
print("message:", err(data)[:100])
print("\n=> Choose: server-side state (store=True) OR zero retention, not both.")

# %% [markdown]
# ## 6. Retrieve, and run in the background
#
# Stored responses can be fetched later, and long jobs can run detached.

# %%
code, fetched = post(
    f"{PREFIX}/responses/{turn1.id}", None, region=REGION, method="GET"
)
print(f"GET  -> {code} status={fetched.get('status')}")

bg = client.responses.create(
    model=DENSE,
    input="Write a short paragraph about idempotency in distributed systems.",
    max_output_tokens=300,
    background=True,
    store=True,
)
print("background job:", redact_ids(bg.id), "status:", bg.status)

for _ in range(30):
    time.sleep(2)
    code, polled = post(
        f"{PREFIX}/responses/{bg.id}", None, region=REGION, method="GET"
    )
    if polled.get("status") in ("completed", "failed", "cancelled"):
        break
print("final status:", polled.get("status"))
print("text:", response_text(polled)[:200])

# %%
# Tidy up the stored responses we created.
for rid in (turn1.id, bg.id):
    code, _ = post(f"{PREFIX}/responses/{rid}", None, region=REGION, method="DELETE")
    print(f"DELETE {rid[:28]}… -> {code}")

# %% [markdown]
# ## 7. A short detour to Chat Completions
#
# Gemma 4 supports Chat Completions too (same `/openai/v1` prefix). Reach for it
# when you have existing OpenAI-shaped code, or want the simpler stateless model.
#
# Two differences worth seeing side by side: `messages` instead of `input`,
# `max_tokens` instead of `max_output_tokens` — and **no reasoning content**.

# %%
cc = client.chat.completions.create(
    model=DENSE,
    messages=[
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Why is idempotency useful? One sentence."},
    ],
    max_tokens=120,
    temperature=1.0,
)
print("content:", cc.choices[0].message.content)
print("\nusage:", cc.usage.model_dump_json())
print("\nAsk for reasoning on Chat Completions and you still pay for it,")
print("but the OpenAI CC spec has nowhere to return it:")

code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": DENSE,
        "messages": [{"role": "user", "content": "Tricky: 17*23?"}],
        "max_tokens": 300,
        "reasoning_effort": "high",
    },
    region=REGION,
)
choice = data.get("choices", [{}])[0].get("message", {})
print("  HTTP", code, "| keys in message:", sorted(choice.keys()))
print("  usage:", json.dumps(data.get("usage", {})))

# %%
# top_p IS accepted on Chat Completions for Gemma 4 — unlike Responses.
# This is exactly where the AWS blog's top_p=0.95 recommendation applies.
code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": DENSE,
        "messages": [{"role": "user", "content": "Reply OK"}],
        "max_tokens": 16,
        "temperature": 1.0,
        "top_p": 0.95,
    },
    region=REGION,
)
print(f"Chat Completions with top_p=0.95 -> HTTP {code}")
print("(Responses rejected the same parameter in section 2.)")

# %% [markdown]
# ## 8. Tool use (function calling)
#
# Gemma 4 has native function calling. Note the **flat** Responses tool shape —
# `name`/`description`/`parameters` at the top level, unlike Chat Completions which
# nests them under `"function"`.


# %%
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Stand-in for a real weather API."""
    table = {"seattle": 12, "singapore": 31, "berlin": 8}
    celsius = table.get(location.split(",")[0].strip().lower(), 20)
    value = celsius if unit == "celsius" else round(celsius * 9 / 5 + 32)
    return {"location": location, "temperature": value, "unit": unit, "sky": "cloudy"}


weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City, e.g. Seattle"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}

conv = [{"role": "user", "content": "What's the weather in Seattle?"}]
first = client.responses.create(
    model=DENSE,
    input=conv,
    tools=[weather_tool],
    tool_choice="auto",
    max_output_tokens=300,
)

calls = [i for i in first.output if i.type == "function_call"]
print("tool calls requested:", [(c.name, c.arguments) for c in calls])

for call in calls:
    args = json.loads(call.arguments)
    result = get_weather(**args)
    # Echo the call, then its result. Note: no reasoning items replayed.
    conv.append(
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
        }
    )
    conv.append(
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result),
        }
    )

final = client.responses.create(
    model=DENSE, input=conv, tools=[weather_tool], max_output_tokens=200
)
print("\nfinal answer:", final.output_text)

# %% [markdown]
# ### One tool call per turn
#
# Gemma 4's model card states parallel tool calls are not supported. It does not
# error — it just quietly does one. Design your loop to iterate rather than
# expecting a batch.

# %%
tool_a = {
    "type": "function",
    "name": "tool_a",
    "description": "Records a value for A.",
    "parameters": {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    },
}
tool_b = {
    "type": "function",
    "name": "tool_b",
    "description": "Records a value for B.",
    "parameters": {
        "type": "object",
        "properties": {"y": {"type": "string"}},
        "required": ["y"],
    },
}

multi = client.responses.create(
    model=DENSE,
    input="Call tool_a with x='1' AND tool_b with y='2'. Both, right now.",
    tools=[tool_a, tool_b],
    max_output_tokens=300,
)
issued = [i.name for i in multi.output if i.type == "function_call"]
print(f"asked for 2 tool calls, model issued {len(issued)}: {issued}")
print("=> iterate; do not assume a batch.")

# %% [markdown]
# ## 9. Structured JSON output
#
# Two routes, and an important caveat about the first one.

# %%
# (a) Native strict schema via text.format
schema = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "typed": {"type": "boolean"},
        "year_created": {"type": "integer"},
    },
    "required": ["language", "typed", "year_created"],
    "additionalProperties": False,
}

code, data = post(
    f"{PREFIX}/responses",
    {
        "model": DENSE,
        "input": "Describe the Rust programming language.",
        "max_output_tokens": 200,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "lang",
                "schema": schema,
                "strict": True,
            }
        },
    },
    region=REGION,
)
raw = response_text(data)
print("native json_schema ->", code)
print("raw output:", repr(raw))

# %% [markdown]
# ### ⚠️ "Strict" is not reliably strict on Gemma 4
#
# Gemma 4 intermittently appends characters **after** a well-formed JSON object —
# a stray `\n}`, or occasionally unrelated text. The object itself is correct, but
# `json.loads()` on the whole string raises. In repeated testing this happened in
# roughly half of runs.
#
# Never call bare `json.loads()` on Gemma 4 structured output in production.

# %%
runs, invalid = 6, 0
for i in range(runs):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": DENSE,
            "input": "Describe the Rust programming language.",
            "max_output_tokens": 200,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "lang",
                    "schema": schema,
                    "strict": True,
                }
            },
        },
        region=REGION,
    )
    text = response_text(data)
    try:
        json.loads(text)
        verdict = "parses"
    except json.JSONDecodeError:
        verdict = "FAILS json.loads"
        invalid += 1
    print(f"  run {i + 1}: {verdict:18} {text!r}")

print(f"\n{invalid}/{runs} runs would crash a naive json.loads()")

# %%
# The fix: extract the first balanced JSON object. mantle.parse_json_lenient()
# does exactly this, and is safe to use on every model.

for sample in [
    '{"language":"Rust","typed":true,"year_created":2010}',
    '{"language":"Rust","typed":true,"year_created":2010}\n}',
    '{"language":"Rust","typed":true,"year_created":2010}\ntrailing text',
    '```json\n{"language":"Rust","typed":true,"year_created":2010}\n```',
]:
    print(f"  {parse_json_lenient(sample)}   <- from {sample[:52]!r}")

# %%
# Re-run the real call and parse it safely.
code, data = post(
    f"{PREFIX}/responses",
    {
        "model": DENSE,
        "input": "Describe the Rust programming language.",
        "max_output_tokens": 200,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "lang",
                "schema": schema,
                "strict": True,
            }
        },
    },
    region=REGION,
)
parsed = parse_json_lenient(response_text(data))
print("parsed safely:")
print(json.dumps(parsed, indent=2))
expected = {"language", "typed", "year_created"}
if set(parsed) != expected:
    raise ValueError(
        f"schema mismatch: expected {sorted(expected)}, got {sorted(parsed)}"
    )
print("schema honoured exactly:", sorted(parsed))

# %%
# (b) Forced tool call — the arguments ARE the output. More portable: it works on
# models that lack native structured output, and enum constraints are respected.
emit = {
    "type": "function",
    "name": "emit_profile",
    "description": "Return the analysis. Use 'unknown' if unsure.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["summary", "sentiment", "confidence"],
    },
}

forced = client.responses.create(
    model=DENSE,
    input="Review: 'The battery life is superb but the screen scratches easily.'",
    tools=[emit],
    tool_choice={"type": "function", "name": "emit_profile"},  # must call it
    max_output_tokens=300,
)
call = next(i for i in forced.output if i.type == "function_call")
print("forced-tool output:")
# parse_json_lenient again: tool arguments can carry the same trailing garbage.
print(json.dumps(parse_json_lenient(call.arguments), indent=2))

# %% [markdown]
# ### Schema keywords
#
# You may read that Gemma 4 rejects JSON-Schema constraint keywords like
# `minLength` and `pattern`. On the current `/openai/v1` Responses path they are
# **accepted** — verify against your own path and model before adding a sanitiser.

# %%
constrained = {
    "type": "function",
    "name": "emit",
    "description": "Emit a code.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "minLength": 3, "pattern": "^[A-Z]"}},
        "required": ["code"],
    },
}
code, data = post(
    f"{PREFIX}/responses",
    {
        "model": DENSE,
        "input": "Emit code 'ABC'.",
        "max_output_tokens": 100,
        "tools": [constrained],
    },
    region=REGION,
)
print(
    f"schema with minLength + pattern -> HTTP {code} "
    f"{'accepted' if code == 200 else err(data)[:70]}"
)

# %% [markdown]
# ## 10. Multimodal — image input
#
# All three variants take text + image. Constraints from the model card and
# testing:
#
# - Base64 data URLs or `s3://` URLs. **Arbitrary `https://` image URLs are not
#   supported.**
# - Total request body max **3.5 MB**.
# - Put the image **before** the text (Google's recommended ordering).
# - Very small images are rejected as an unsupported format — use realistic sizes.

# %%
import struct
import zlib


def make_png(width: int, height: int, rgb: tuple) -> bytes:
    """Build a solid-colour PNG without external dependencies."""
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


crimson_png = make_png(64, 64, (220, 20, 60))
data_url = "data:image/png;base64," + base64.b64encode(crimson_png).decode()
print("data URL bytes:", len(data_url))

vision = client.responses.create(
    model=DENSE,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": data_url},  # image FIRST
                {"type": "input_text", "text": "What colour is this image? One word."},
            ],
        }
    ],
    max_output_tokens=32,
)
print("model sees:", repr(vision.output_text.strip()))

# %%
# A 1x1 pixel PNG is rejected — worth knowing so you don't chase a phantom bug.
tiny = "data:image/png;base64," + base64.b64encode(make_png(1, 1, (255, 0, 0))).decode()
code, data = post(
    f"{PREFIX}/responses",
    {
        "model": DENSE,
        "max_output_tokens": 16,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": tiny},
                    {"type": "input_text", "text": "Colour?"},
                ],
            }
        ],
    },
    region=REGION,
)
print(f"1x1 PNG -> HTTP {code}: {err(data)[:80]}")

# %% [markdown]
# There is also an undocumented ceiling of roughly **32 images per request**.
# Payloads far below the 3.5 MB size limit can still fail with
# `400 / "Engine bad request"` once you exceed it. Batch large image sets.

# %% [markdown]
# ## 11. Choosing a variant — a like-for-like comparison
#
# Same prompt, all three variants, measuring latency and tokens.

# %%
task = "In one sentence, explain why eventual consistency is a useful trade-off."

print(f"{'model':26} {'latency':>9} {'reason tok':>11} {'out tok':>8}  answer")
print("-" * 104)
for model in (COMPACT, MOE, DENSE):
    started = time.perf_counter()
    r = client.responses.create(
        model=model, input=task, reasoning={"effort": "low"}, max_output_tokens=250
    )
    elapsed = time.perf_counter() - started
    details = r.usage.output_tokens_details
    print(
        f"{model:26} {elapsed:>8.2f}s {details.reasoning_tokens:>11} "
        f"{r.usage.output_tokens:>8}  {r.output_text.strip()[:44]!r}"
    )

# %% [markdown]
# | If your workload is… | Choose | Why |
# |---|---|---|
# | Reasoning- or coding-heavy | `gemma-4-31b` | Largest dense variant, 256K context |
# | Cost-sensitive at high throughput | `gemma-4-26b-a4b` | MoE (mixture-of-experts): ~4B-class cost, larger knowledge capacity |
# | Latency-sensitive, on-device-style | `gemma-4-e2b` | Smallest and fastest; set `effort="high"` |
#
# All three share one API surface, so you can develop once and switch by model ID.

# %% [markdown]
# ## 12. Production hardening
#
# Retries, cost attribution, and privacy in one place.
# (Background: `../00-foundations/03-scaling-tiers-and-latency.ipynb`.)

# %%
# Attribute usage to a project for cost tracking (see ../00-foundations/02).
code, project = post(
    "/v1/organization/projects",
    {
        "name": "gemma4-samples",
        "tags": {"Application": "Gemma4Demo", "Environment": "Demo"},
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)

code, data = post(
    f"{PREFIX}/responses",
    {
        "model": DENSE,
        "input": "Reply OK",
        "max_output_tokens": 16,
        "service_tier": "flex",
        "store": False,
    },
    region=REGION,
    headers={"OpenAI-Project": project_id},
)
print(f"attributed call -> HTTP {code} resolved tier={data.get('service_tier')}")


# %%
class Gemma4Client:
    """Production-shaped wrapper: fresh token, right params, retries, attribution."""

    def __init__(self, model=DENSE, region=REGION, tier="default", project=None):
        self.model, self.region, self.tier, self.project = model, region, tier, project

    def ask(self, prompt, *, effort="low", max_output_tokens=512, structured=None):
        body = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max(16, max_output_tokens),  # API minimum is 16
            "temperature": 1.0,  # the ONLY value Responses accepts
            # top_p deliberately omitted: rejected on the Responses API
            "reasoning": {"effort": effort},
            "service_tier": self.tier,
            "store": False,  # no 30-day retention
        }
        if structured:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "out",
                    "schema": structured,
                    "strict": True,
                }
            }
        headers = {"OpenAI-Project": self.project} if self.project else None
        # post() retries 429/5xx with exponential backoff — mantle has no RPM
        # quota and sheds load under regional pressure.
        code, data = post(
            f"{PREFIX}/responses", body, region=self.region, headers=headers
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        return data


gemma = Gemma4Client(model=MOE, tier="flex", project=project_id)
out = gemma.ask(
    "Name the capital of Japan.",
    structured={
        "type": "object",
        "properties": {"capital": {"type": "string"}},
        "required": ["capital"],
        "additionalProperties": False,
    },
)
print("structured:", parse_json_lenient(response_text(out)))
print("usage:", json.dumps(out.get("usage", {})))

# %%
# Clean up the demo project.
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived project:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — Gemma 4 on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | `bedrock-mantle` only | No `bedrock-runtime` support for these model IDs |
# | Path prefix | `/openai/v1`, **not** the bare `/v1` most mantle models use |
# | `temperature` | On Responses, **only `1.0` is accepted** — every other value 400s |
# | `top_p` | **400 on Responses**; accepted on Chat Completions |
# | Tuning sampling | Not possible on Responses — switch to Chat Completions |
# | `max_output_tokens` | Minimum **16**; smaller values 400 |
# | `reasoning.effort` | `none`/`low`/`medium`/`high`; **`minimal` is rejected** |
# | Reasoning visibility | Returned on Responses only; billed-but-hidden on Chat Completions |
# | Reasoning replay | Never append reasoning items to history — degrades quality |
# | Parallel tool calls | Unsupported; model silently issues one |
# | `store=False` | Blocks `previous_response_id` chaining (404) |
# | Images | base64 or `s3://` only; ≤3.5 MB body; ~32 images max; 1×1 PNG rejected |
# | e2b reasoning | Set `effort="high"` to stop thinking leaking into the answer |
# | Regions | The only family in all four mantle Regions |
#
# ## Where next
# - Same-API neighbours: `../01-openai-gpt/` (web search, caching), `../11-xai-grok/`
# - Different API shape: `../04-qwen/` (Chat Completions), `../02-anthropic-claude/`
#   (Messages)
# - Cross-cutting: `../99-cross-cutting/`
