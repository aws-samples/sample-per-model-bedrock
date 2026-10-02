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
# # Qwen3 on Amazon Bedrock Mantle
#
# Alibaba's Qwen3 family on the `bedrock-mantle` endpoint — a wide range of sizes plus
# dedicated coder and vision-language variants. This notebook covers the general-purpose
# models end to end; `02-qwen3-coder-and-vision.ipynb` covers the specialists.
#
# **Models covered in this notebook**
#
# | Model ID | Notes |
# |---|---|
# | `qwen.qwen3-32b` | Dense 32B — the reliable default; also fine-tunable on mantle |
# | `qwen.qwen3-235b-a22b-2507` | MoE (mixture-of-experts) 235B total / 22B active — highest quality |
# | `qwen.qwen3-next-80b-a3b-instruct` | MoE 80B total / 3B active — cost-efficient |
#
# ## Which API? Chat Completions.
# This family is served by the **OpenAI-compatible Chat Completions API** on the
# `bedrock-mantle` endpoint, at the bare `/v1` path. The Responses API returns
# **400 "does not support this API"** for these models — we prove that in §2 rather
# than asking you to take it on trust.
#
# ## Self-contained, but see also
# Everything you need is here. For deeper background on shared mechanics:
# - **Auth (SigV4 (AWS Signature Version 4) + short-term API keys), the three URL paths,
#   model discovery** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Projects, cost attribution, data retention / ZDR (zero data retention), CloudWatch
#   namespace** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
# - **Quotas, retry/backoff, service tiers, TTFT (time-to-first-token) measurement** →
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

# %%
import json
import sys
import time

sys.path.insert(0, "../_shared")
from mantle import err, parse_json_lenient, post, ttft

REGION = "us-east-1"

DENSE_32B = "qwen.qwen3-32b"
MOE_235B = "qwen.qwen3-235b-a22b-2507"
MOE_80B = "qwen.qwen3-next-80b-a3b-instruct"


# Chat-Completions families live at the BARE /v1 path — not /openai/v1
# (that prefix is only for gemma-4, gpt-5.x and grok). See ../00-foundations/01.
PREFIX = "/v1"
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}"
print("base URL:", BASE_URL)
print("models  :", [DENSE_32B, MOE_235B, MOE_80B])

# %% [markdown]
# ## 1. First call
#
# Auth is a short-term Bedrock API key minted from your ambient IAM credentials.
# It expires within 12 hours and **cannot be refreshed** — mint a new one instead.
# (`../00-foundations/01` shows the self-refreshing provider and the SigV4
# alternative that needs no key at all.)

# %%
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

# Build the client from a FRESH token — don't construct one at import time and
# reuse it for hours, because the baked-in key expires.
client = OpenAI(api_key=provide_token(region=REGION), base_url=BASE_URL)

completion = client.chat.completions.create(
    model=DENSE_32B,
    messages=[
        {
            "role": "user",
            "content": (
                "Explain the difference between a dense and a sparse (MoE) "
                "transformer, in two "
                "sentences."
            ),
        }
    ],
    max_tokens=250,
)
# `content` can be None when the model spends the whole budget reasoning: the call
# succeeds with finish_reason="length" and no text. Check before printing -- this is
# the single most common surprise on this endpoint.
choice = completion.choices[0]
answer = choice.message.content or ""
if answer:
    print(answer)
else:
    print(f"(no text: finish_reason={choice.finish_reason!r} — raise max_tokens)")
print("\nusage:", completion.usage.model_dump_json())

# %% [markdown]
# ## 2. Why Chat Completions and not Responses
#
# AWS recommends the Responses API for new applications in general — but
# availability is per-model. Probe both surfaces so the 400 is visible:

# %%
for api_name, path, body in [
    (
        "Chat Completions",
        f"{PREFIX}/chat/completions",
        {
            "model": DENSE_32B,
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_tokens": 16,
        },
    ),
    (
        "Responses (/v1)",
        f"{PREFIX}/responses",
        {"model": DENSE_32B, "input": "Reply OK", "max_output_tokens": 16},
    ),
    (
        "Responses (/openai/v1)",
        "/openai/v1/responses",
        {"model": DENSE_32B, "input": "Reply OK", "max_output_tokens": 16},
    ),
]:
    code, data = post(path, body, region=REGION)
    print(f"  {api_name:24} -> HTTP {code} {'' if code == 200 else err(data)[:64]}")

# %% [markdown]
# Concrete consequences of being Chat-Completions-only:
#
# - **You own the conversation history.** There is no `previous_response_id`
#   server-side state on this API — send the full `messages` array each turn.
# - **Reasoning content is not returned.** `reasoning_effort` is accepted and the
#   model does think, but the OpenAI Chat Completions schema has nowhere to put the
#   trace, so you pay for those tokens without seeing them.
# - Structured output uses `response_format`, not `text.format`.

# %% [markdown]
# ## 3. Sampling parameters
#
# This family accepts both `temperature` and `top_p`. That is *not* universal on
# mantle — Gemma 4 rejects `top_p`, and Grok rejects `temperature` — so never share
# one sampling config across families.

# %%
for label, extra in [
    ("temperature=0.7", {"temperature": 0.7}),
    ("temperature=0.0", {"temperature": 0.0}),
    ("top_p=0.95", {"top_p": 0.95}),
    ("both", {"temperature": 0.7, "top_p": 0.95}),
    ("max_tokens=1", {"max_tokens": 1}),
]:
    body = {
        "model": DENSE_32B,
        "messages": [{"role": "user", "content": "Reply OK"}],
        "max_tokens": 16,
    }
    body.update(extra)
    code, data = post(f"{PREFIX}/chat/completions", body, region=REGION)
    print(f"  {label:18} -> HTTP {code} {'' if code == 200 else err(data)[:60]}")

# %% [markdown]
# Note `max_tokens=1` is accepted here. The Responses API enforces a minimum of
# 16 — another reason the two surfaces are not interchangeable.

# %% [markdown]
# ## 4. Streaming
#
# Chat Completions streams `data: {...}` SSE (server-sent events) frames carrying
# `choices[0].delta.content`, terminated by `data: [DONE]`.

# %%
stream = client.chat.completions.create(
    model=DENSE_32B,
    messages=[
        {
            "role": "user",
            "content": ("List four practical uses of long-context language models."),
        }
    ],
    max_tokens=300,
    stream=True,
)
chunks = 0
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        chunks += 1
        print(delta, end="", flush=True)
print(f"\n\n[{chunks} content deltas received]")

# %% [markdown]
# ## 5. Multi-turn — you manage the history
#
# No server-side state on this API. Append each turn yourself.

# %%
messages = [
    {"role": "system", "content": "You are concise. Two sentences maximum."},
    {"role": "user", "content": "What is quantisation in the context of LLM serving?"},
]
first = client.chat.completions.create(
    model=DENSE_32B, messages=messages, max_tokens=200
)
print("assistant:", first.choices[0].message.content)

messages.append({"role": "assistant", "content": first.choices[0].message.content})
messages.append(
    {
        "role": "user",
        "content": "Which quantisation format is most common for GPU inference?",
    }
)

second = client.chat.completions.create(
    model=DENSE_32B, messages=messages, max_tokens=200
)
print("\nassistant:", second.choices[0].message.content)
print(
    f"\ninput tokens grew: {first.usage.prompt_tokens} -> {second.usage.prompt_tokens}"
)

# %% [markdown]
# That growth is the cost of client-side history. Families on the Responses API can
# avoid it with `previous_response_id` (see `../03-google-gemma/`), at the price of
# 30-day server-side retention.

# %% [markdown]
# ## 6. Reasoning effort
#
# `reasoning_effort` is accepted. The trace is not returned — but the token count
# moves, which is how you can tell the model really is thinking harder.

# %%
print(f"{'effort':10} {'status':>7} {'completion tokens':>18}")
print("-" * 38)
for effort in ("none", "low", "medium", "high"):
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": DENSE_32B,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "A shop sells pens at 3 for $2. How much for 17 pens? Show "
                        "your "
                        "working."
                    ),
                }
            ],
            "max_tokens": 400,
            "reasoning_effort": effort,
        },
        region=REGION,
    )
    tokens = (data.get("usage") or {}).get("completion_tokens", "-")
    print(f"  {effort:8} {code:>7} {tokens!s:>18}")

# %% [markdown]
# ### Where the reasoning text actually goes
#
# Two things surprise people here, and both are visible in the output below.
#
# **1. Budget for the thinking.** At `reasoning_effort="high"` this model spent
# ~2,000 tokens reasoning before writing a single character of answer. With
# `max_tokens=500` you get **HTTP 200, `finish_reason="length"`, and an empty
# `content`** — no error, just nothing usable. That is why the loop below starts at
# 500 and escalates: watch the first two rows produce nothing.
#
# **2. The reasoning is returned in a non-standard field.** Older Qwen builds wrapped
# it in `<think>...</think>` *inside* `content`, and much of the internet still tells
# you to strip those tags. On `bedrock-mantle` today the trace arrives as
# **`choices[0].message.reasoning`** — a sibling of `content`, not part of it — so
# `content` is already clean and there is nothing to strip. Do not rely on either
# shape: check which one you actually received.

# %%
for budget in (500, 1500, 3000):
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": DENSE_32B,
            "messages": [
                {"role": "user", "content": "What is 17 * 23? Think step by step."}
            ],
            "max_tokens": budget,
            "reasoning_effort": "high",
        },
        region=REGION,
    )
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    print(
        f"max_tokens={budget:5} HTTP {code} finish={choice.get('finish_reason')!r:9} "
        f"content={len(content):5} reasoning={len(reasoning):5} "
        f"<think> tags={'<think>' in content}"
    )

print("\nfields on the message object:", sorted(message))
print("answer   :", " ".join(content.split())[:120] or "(empty)")
print("reasoning:", " ".join(reasoning.split())[:120] or "(none returned)")

# %% [markdown]
# **Read this defensively in your own code.** Handle both shapes, and never assume
# `content` is non-empty:
#
# ```python
# message = response["choices"][0]["message"]
# answer = message.get("content") or ""
# trace = message.get("reasoning") or ""            # mantle's Qwen shape today
# if "<think>" in answer:                          # older/self-hosted builds
#     import re
#     trace = trace or answer
#     answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()
# if not answer:                                   # budget ran out mid-thought
#     ...retry with a larger max_tokens
# ```


# %% [markdown]
# ## 7. Tool use (function calling)
#
# Chat Completions nests the schema under `"function"` — unlike the Responses API,
# which puts `name`/`parameters` at the top level. Same concept, different shape.
# %%
def lookup_inventory(sku: str, warehouse: str = "main") -> dict:
    """Stand-in for a real inventory service."""
    stock = {"A-100": 42, "B-200": 0, "C-300": 7}
    return {
        "sku": sku,
        "warehouse": warehouse,
        "quantity": stock.get(sku.upper(), 0),
        "in_stock": stock.get(sku.upper(), 0) > 0,
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_inventory",
            "description": "Look up stock level for a SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "SKU code, e.g. A-100"},
                    "warehouse": {"type": "string", "enum": ["main", "overflow"]},
                },
                "required": ["sku"],
            },
        },
    }
]

convo = [{"role": "user", "content": "Do we have SKU A-100 in stock?"}]

# `tool_choice="auto"` means the model MAY call a tool, not that it will. A
# reasoning-capable model can spend the budget thinking and return
# finish_reason="length" with no tool_calls. Retry instead of assuming.
msg = None
for attempt in range(1, 4):
    first = client.chat.completions.create(
        model=DENSE_32B, messages=convo, tools=tools, tool_choice="auto", max_tokens=800
    )
    choice = first.choices[0]
    print(
        f"attempt {attempt}: finish_reason={choice.finish_reason!r} "
        f"tool_calls={len(choice.message.tool_calls or [])}"
    )
    if choice.message.tool_calls:
        msg = choice.message
        break
if msg is None:
    raise RuntimeError("no tool call after 3 attempts - raise max_tokens")
print(
    "tool_calls:",
    [(c.function.name, c.function.arguments) for c in (msg.tool_calls or [])],
)

if msg.tool_calls:
    convo.append(msg.model_dump(exclude_none=True))
    for call in msg.tool_calls:
        args = parse_json_lenient(call.function.arguments)
        result = lookup_inventory(**args)
        convo.append(
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
        )
    final = client.chat.completions.create(
        model=DENSE_32B, messages=convo, tools=tools, max_tokens=200
    )
    print("\nfinal answer:", final.choices[0].message.content)

# %% [markdown]
# ### Forcing a specific tool
#
# `tool_choice` can compel a named function. This is the most portable route to
# strict structured output: the arguments *are* your JSON.
#
# **But treat it as best-effort, not a guarantee.** In repeated testing about 1
# call in 10 ignored the forced choice and returned prose with
# `finish_reason="stop"`. Always check for the tool call and retry.

# %%
emit = [
    {
        "type": "function",
        "function": {
            "name": "emit_review",
            "description": "Return the structured review analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative"],
                    },
                    "would_recommend": {"type": "boolean"},
                },
                "required": ["summary", "sentiment", "would_recommend"],
            },
        },
    }
]


def emit_review(prompt, model=DENSE_32B, attempts=3):
    """Forced tool call, with a retry.

    IMPORTANT: forcing `tool_choice` is honoured *almost* always, not always.
    In repeated testing roughly 1 call in 10 came back with finish_reason="stop"
    and prose instead of a tool call. Production code must handle that, so this
    helper retries rather than indexing [0] and hoping.
    """
    for attempt in range(attempts):
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=emit,
            tool_choice={"type": "function", "function": {"name": "emit_review"}},
            max_tokens=300,
        )
        choice = completion.choices[0]
        calls = choice.message.tool_calls or []
        if calls:
            if attempt:
                print(f"(succeeded on attempt {attempt + 1})")
            # parse_json_lenient, not json.loads: some models append characters
            # after a well-formed object even in strict modes.
            return parse_json_lenient(calls[0].function.arguments)
        print(
            f"attempt {attempt + 1}: no tool call "
            f"(finish_reason={choice.finish_reason}) — retrying"
        )
    raise RuntimeError("model would not emit the forced tool call")


review = emit_review("Review: 'Fast delivery, but the packaging arrived crushed.'")
print(json.dumps(review, indent=2))


# %% [markdown]
# ## 8. Structured output with `response_format`
#
# Two variants: loose `json_object`, and schema-enforced `json_schema`.

# %% [markdown]
# ### Budget enough tokens, or you get nothing
#
# A reasoning-capable model may spend most of its budget thinking before it emits
# the opening brace. If `max_tokens` runs out first you get **HTTP 200 with empty
# content** and `finish_reason="length"` - not an error, just nothing usable.
# Always check `finish_reason` before parsing.


# %%
def json_object_call(prompt, max_tokens, model=DENSE_32B):
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
        region=REGION,
    )
    choice = (data.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content") or ""
    return code, choice.get("finish_reason"), content


PROMPT = "Give the capital and population of France as JSON."
for budget in (64, 600):
    code, finish, content = json_object_call(PROMPT, budget)
    print(
        f"max_tokens={budget:4} HTTP {code} finish={finish!s:8} "
        f"content_len={len(content)}"
    )
    if finish == "length" and not content.strip():
        print("    -> truncated before any JSON was emitted; raise max_tokens")
    elif content.strip():
        print("    ->", parse_json_lenient(content))

# %%
schema = {
    "type": "object",
    "properties": {
        "country": {"type": "string"},
        "capital": {"type": "string"},
        "population_millions": {"type": "number"},
    },
    "required": ["country", "capital", "population_millions"],
    "additionalProperties": False,
}

code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": DENSE_32B,
        "messages": [{"role": "user", "content": "Describe France."}],
        "max_tokens": 250,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "country", "strict": True, "schema": schema},
        },
    },
    region=REGION,
)
choice = (data.get("choices") or [{}])[0]
content = choice.get("message", {}).get("content")  # may be None!
print("json_schema ->", code, "| finish_reason:", choice.get("finish_reason"))
print("raw:", repr((content or "")[:160]))

if choice.get("finish_reason") == "length":
    # Reasoning consumed the budget before the object closed. Retry bigger.
    print("truncated - retrying with a larger budget")
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": DENSE_32B,
            "messages": [{"role": "user", "content": "Describe France."}],
            "max_tokens": 2000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "country", "strict": True, "schema": schema},
            },
        },
        region=REGION,
    )
    choice = (data.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content")
    print("retry finish_reason:", choice.get("finish_reason"))

parsed = parse_json_lenient(content or "")
print("parsed:", json.dumps(parsed, indent=2))
missing = {"country", "capital"} - set(parsed)
if missing:
    raise ValueError(f"model omitted required keys: {sorted(missing)} in {parsed}")
print("required keys present: country, capital")

# %% [markdown]
# **Always parse leniently.** Even in strict mode, some mantle models append
# characters after a valid object (Gemma 4 does this in ~half of runs), which makes
# a bare `json.loads()` raise on output that is otherwise fine.

# %% [markdown]
# ## 9. Compare the models in this family
#
# Same prompt across three architectures: a dense 32B, a large MoE, and a cost-efficient
# MoE. Watch latency against quality.

# %%
task = (
    "In one sentence, why does a mixture-of-experts model cost less to serve than a "
    "dense model of the same total size?"
)

print(f"{'model':44} {'latency':>9} {'out tok':>8}  answer")
print("-" * 108)
for model in [DENSE_32B, MOE_80B, MOE_235B]:
    started = time.perf_counter()
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": task}],
            "max_tokens": 160,
        },
        region=REGION,
    )
    elapsed = time.perf_counter() - started
    if code != 200:
        print(f"{model:44} {'-':>9} {'-':>8}  HTTP {code}: {err(data)[:40]}")
        continue
    text = (data["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")
    print(
        f"{model:44} {elapsed:>8.2f}s "
        f"{data['usage']['completion_tokens']:>8}  {text[:44]!r}"
    )

# %% [markdown]
# ## 10. Latency: TTFT and throughput
#
# TTFT is dominated by *prefill* (the model reading your prompt) plus queue time.
# Service tiers trade cost against queue priority — they mostly separate under
# contention, so single samples on an idle account look flat.
# (`../00-foundations/03` has the full treatment.)

# %%
print(f"{'tier':10} {'TTFT (s)':>10} {'total (s)':>10} {'frames/s':>10}")
print("-" * 44)
for tier in ("default", "flex", "priority"):
    m = ttft(
        f"{PREFIX}/chat/completions",
        {
            "model": DENSE_32B,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "List four practical uses of long-context language models."
                    ),
                }
            ],
            "max_tokens": 200,
            "service_tier": tier,
        },
        region=REGION,
    )
    if m.get("error"):
        print(f"{tier:10} {m['error']:>32}  (tier not supported by this model)")
    else:
        print(
            f"{tier:10} {m['ttft_s']:>10.3f} {m['total_s']:>10.3f} "
            f"{m['frames_per_s']:>10.1f}"
        )

# %% [markdown]
# ## 11. Production hardening
#
# Retries, cost attribution, and privacy. Mantle has **no RPM quota** — throttling
# is token-based, and most models here have no published TPM (tokens per minute) quota
# at all, so
# capacity is fair-share. That makes retry-with-backoff mandatory, not optional.

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "qwen3-samples",
        "tags": {"Application": "Qwen3Demo", "Environment": "Demo"},
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)

code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": DENSE_32B,
        "messages": [{"role": "user", "content": "Reply OK"}],
        "max_tokens": 16,
    },
    region=REGION,
    headers={"OpenAI-Project": project_id},  # cost attribution
)
print("attributed call ->", code)


# %%
class Qwen3Client:
    """Production-shaped wrapper for this family on bedrock-mantle."""

    def __init__(self, model=DENSE_32B, region=REGION, tier="default", project=None):
        self.model, self.region, self.tier, self.project = model, region, tier, project

    def chat(self, messages, *, max_tokens=512, tools=None, schema=None, effort=None):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "service_tier": self.tier,
        }
        if tools:
            body["tools"] = tools
        if effort:
            body["reasoning_effort"] = effort
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True, "schema": schema},
            }
        headers = {"OpenAI-Project": self.project} if self.project else None
        # post() retries 429 + 5xx with exponential backoff and jitter.
        code, data = post(
            f"{PREFIX}/chat/completions", body, region=self.region, headers=headers
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        return data

    @staticmethod
    def _choice(data: dict) -> dict:
        """First choice, without assuming the list is non-empty."""
        return (data.get("choices") or [{}])[0]

    def json(self, prompt, schema, *, max_tokens=512, **kw):
        """Structured call that survives a truncated first attempt.

        A reasoning-capable model can spend its whole budget thinking and return
        HTTP 200 with EMPTY content and finish_reason="length". Parsing that raises.
        So check finish_reason and escalate the budget once before giving up.
        """
        messages = [{"role": "user", "content": prompt}]
        for budget in (max_tokens, max_tokens * 4):
            data = self.chat(messages, schema=schema, max_tokens=budget, **kw)
            choice = self._choice(data)
            content = choice.get("message", {}).get("content") or ""
            if content.strip():
                return parse_json_lenient(content)
            if choice.get("finish_reason") != "length":
                break  # empty for some other reason - escalating will not help
        raise RuntimeError(
            f"no content after budget escalation to {max_tokens * 4} tokens "
            f"(finish_reason={self._choice(data).get('finish_reason')!r})"
        )


bot = Qwen3Client(tier="flex", project=project_id)
out = bot.json(
    "Name the largest ocean and its average depth in metres.",
    {
        "type": "object",
        "properties": {"ocean": {"type": "string"}, "avg_depth_m": {"type": "number"}},
        "required": ["ocean", "avg_depth_m"],
        "additionalProperties": False,
    },
)
print("structured result:", out)

# %%
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived demo project:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — Qwen3 on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | Path prefix | Bare `/v1`, **not** `/openai/v1` (that's gemma-4 / gpt-5.x / grok) |
# | Responses API | Returns **400** for this family — Chat Completions only |
# | History | No `previous_response_id`; you send `messages` every turn |
# | Reasoning trace | `reasoning_effort` works but the trace is never returned |
# | Strict JSON | Parse leniently — models can append text after a valid object |
# | Sampling | `temperature` **and** `top_p` both fine here; not true family-wide |
# | `max_tokens` | 1 is valid here; Responses API demands ≥16 |
# | `content` can be `None` | Check `finish_reason` before slicing/parsing content |
# | Quotas | No RPM quota; most models have no published TPM — retry with backoff |
# | `reserved` tier | Rejected as a parameter; arranged via your account team |
# | CloudWatch | Metrics land in `AWS/BedrockMantle`, not `AWS/Bedrock` |
# | `<think>` blocks | Qwen3 can emit reasoning inline in `content` — strip it |
# | Fine-tuning | `qwen3-32b` is one of only two mantle-fine-tunable models (us-west-2) |
#
# ## Where next
# - Same API shape: `../05-deepseek/`, `../06-zai-glm/`, `../09-minimax/`
# - Different API shape: `../03-google-gemma/` (Responses),
#   `../02-anthropic-claude/` (Messages), `../01-openai-gpt/` (web search, caching)
# - Shared mechanics: `../00-foundations/`
