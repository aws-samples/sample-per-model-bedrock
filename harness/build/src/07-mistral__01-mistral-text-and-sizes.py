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
# # Mistral AI on Amazon Bedrock Mantle — text models and the size ladder
#
# Mistral has the widest size range of any family on `bedrock-mantle`: from a 3B
# Ministral up to a 675B Mistral Large 3. That size spread makes it a good family
# for demonstrating **cost-aware model routing** — send each request to the
# cheapest model that can actually do the job.
#
# **Models covered in this notebook**
#
# | Model ID | Notes |
# |---|---|
# | `mistral.mistral-large-3-675b-instruct` | Flagship, 675B |
# | `mistral.ministral-3-14b-instruct` | Mid-size |
# | `mistral.ministral-3-8b-instruct` | Small |
# | `mistral.ministral-3-3b-instruct` | Smallest, lowest latency |
# | `mistral.magistral-small-2509` | Reasoning-oriented variant |
#
# `02-devstral-and-voxtral.ipynb` covers the coding model (Devstral 2) and the
# audio-capable Voxtral variants.
#
# ## Which API? Chat Completions.
# This family is served by the **OpenAI-compatible Chat Completions API** at the
# bare `/v1` path. The Responses API returns **400** for these models — proven in §2.
#
# ## Self-contained, but see also
# - **Auth (SigV4 (AWS Signature Version 4) + short-term keys), the three URL paths,
#   model discovery** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Projects, cost attribution, data retention / ZDR (zero data retention),
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
from mantle import err, list_models, parse_json_lenient, post, ttft

REGION = "us-east-1"

LARGE3 = "mistral.mistral-large-3-675b-instruct"
M14B = "mistral.ministral-3-14b-instruct"
M8B = "mistral.ministral-3-8b-instruct"
M3B = "mistral.ministral-3-3b-instruct"
MAGISTRAL = "mistral.magistral-small-2509"

LADDER = [M3B, M8B, M14B, LARGE3]

# Chat-Completions families live at the BARE /v1 path — not /openai/v1
# (that prefix is only for gemma-4, gpt-5.x and grok). See ../00-foundations/01.
PREFIX = "/v1"
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}"
print("base URL:", BASE_URL)
print("size ladder:", LADDER)

# %% [markdown]
# ## 1. First call
#
# Auth is a short-term Bedrock API key minted from ambient IAM credentials —
# valid ≤12 h, **not refreshable**, Region-pinned. (`../00-foundations/01` shows the
# self-refreshing provider and the SigV4 alternative that needs no key.)

# %%
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

# Build from a FRESH token; don't construct once at import and reuse for hours.
client = OpenAI(api_key=provide_token(region=REGION), base_url=BASE_URL)

completion = client.chat.completions.create(
    model=LARGE3,
    messages=[
        {
            "role": "user",
            "content": (
                "Explain why model size is not the only quality "
                "factor, in two sentences."
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

# %%
for label, path, body in [
    (
        "Chat Completions",
        f"{PREFIX}/chat/completions",
        {
            "model": LARGE3,
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_tokens": 16,
        },
    ),
    (
        "Responses (/v1)",
        f"{PREFIX}/responses",
        {"model": LARGE3, "input": "Reply OK", "max_output_tokens": 16},
    ),
    (
        "Responses (/openai/v1)",
        "/openai/v1/responses",
        {"model": LARGE3, "input": "Reply OK", "max_output_tokens": 16},
    ),
]:
    # Short timeout, single attempt: a wrong path does not always fail fast.
    code, data = post(path, body, region=REGION, attempts=1, timeout=45)
    shown = code if code != -1 else "stalled"
    print(f"  {label:24} -> {shown} {'' if code == 200 else err(data)[:60]}")

# %% [markdown]
# Consequences of being Chat-Completions-only:
#
# - **You own the conversation history** — no `previous_response_id`.
# - **Reasoning content is not returned.** `reasoning_effort` is accepted and the
#   model does think, but the Chat Completions schema has nowhere to put the trace.
# - Structured output uses `response_format`, not `text.format`.

# %% [markdown]
# ## 3. Sampling parameters
#
# This family accepts both `temperature` and `top_p`. That is *not* universal on
# mantle: Gemma 4 rejects `top_p`, Grok rejects `temperature`, and newer Claude
# models reject `temperature` too. Never share one sampling config across families.

# %%
for label, extra in [
    ("temperature=0.7", {"temperature": 0.7}),
    ("temperature=0.0", {"temperature": 0.0}),
    ("top_p=0.95", {"top_p": 0.95}),
    ("both", {"temperature": 0.7, "top_p": 0.95}),
    ("max_tokens=1", {"max_tokens": 1}),
]:
    body = {
        "model": LARGE3,
        "messages": [{"role": "user", "content": "Reply OK"}],
        "max_tokens": 16,
    }
    body.update(extra)
    code, data = post(f"{PREFIX}/chat/completions", body, region=REGION)
    print(f"  {label:18} -> HTTP {code} {'' if code == 200 else err(data)[:60]}")

# %% [markdown]
# `max_tokens=1` is valid here; the Responses API enforces a minimum of 16. One
# more reason the two surfaces are not interchangeable.

# %% [markdown]
# ## 4. Streaming

# %%
stream = client.chat.completions.create(
    model=M8B,
    messages=[
        {
            "role": "user",
            "content": ("List four uses for a small language model."),
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
print(f"\n\n[{chunks} content deltas]")

# %% [markdown]
# ## 5. Multi-turn — you manage the history

# %%
messages = [
    {"role": "system", "content": "You are concise. Two sentences maximum."},
    {"role": "user", "content": "What is quantisation?"},
]
first = client.chat.completions.create(model=M14B, messages=messages, max_tokens=200)
print("assistant:", first.choices[0].message.content)

messages.append({"role": "assistant", "content": first.choices[0].message.content})
messages.append({"role": "user", "content": "What quality cost does it usually carry?"})

second = client.chat.completions.create(model=M14B, messages=messages, max_tokens=200)
print("\nassistant:", second.choices[0].message.content)
print(
    f"\ninput tokens grew: {first.usage.prompt_tokens} -> {second.usage.prompt_tokens}"
)

# %% [markdown]
# ## 6. Reasoning effort, and the Magistral variant
#
# `reasoning_effort` is accepted family-wide. `magistral-small-2509` is the
# reasoning-oriented member — worth comparing against a similarly sized Ministral.

# %%
print(f"{'model':40} {'effort':8} {'completion tokens':>18}")
print("-" * 70)
PUZZLE = (
    "A rope burns in 60 minutes unevenly. How do you time 30 minutes with two ropes?"
)
for model in (M8B, MAGISTRAL):
    for effort in ("low", "high"):
        code, data = post(
            f"{PREFIX}/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": PUZZLE}],
                "max_tokens": 700,
                "reasoning_effort": effort,
            },
            region=REGION,
        )
        tokens = (data.get("usage") or {}).get("completion_tokens", "-")
        print(f"{model:40} {effort:8} {tokens!s:>18}")

# %%
# What Magistral actually says on a reasoning task.
code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": MAGISTRAL,
        "messages": [{"role": "user", "content": PUZZLE}],
        "max_tokens": 800,
        "reasoning_effort": "high",
    },
    region=REGION,
)
choice = (data.get("choices") or [{}])[0]
print("finish_reason:", choice.get("finish_reason"))
print((choice.get("message", {}).get("content") or "")[:500])

# %% [markdown]
# ## 7. Tool use (function calling)
#
# Chat Completions nests the schema under `"function"` — unlike the Responses API,
# which puts `name`/`parameters` at the top level.


# %%
def convert_currency(amount: float, source: str, target: str) -> dict:
    """Stand-in for an FX service."""
    rates = {("EUR", "USD"): 1.09, ("USD", "EUR"): 0.92, ("EUR", "SGD"): 1.46}
    rate = rates.get((source.upper(), target.upper()))
    return {
        "amount": amount,
        "from": source.upper(),
        "to": target.upper(),
        "rate": rate,
        "converted": round(amount * rate, 2) if rate else None,
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "source": {"type": "string", "description": "ISO code, e.g. EUR"},
                    "target": {"type": "string", "description": "ISO code, e.g. USD"},
                },
                "required": ["amount", "source", "target"],
            },
        },
    }
]

convo = [{"role": "user", "content": "How much is 250 EUR in USD?"}]

# `tool_choice="auto"` means the model MAY call a tool, not that it will. Retry
# rather than assuming the first response carries a call.
msg = None
for attempt in range(1, 4):
    first = client.chat.completions.create(
        model=LARGE3, messages=convo, tools=tools, tool_choice="auto", max_tokens=800
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
        convo.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(convert_currency(**args)),
            }
        )
    final = client.chat.completions.create(
        model=LARGE3, messages=convo, tools=tools, max_tokens=200
    )
    print("\nfinal answer:", final.choices[0].message.content)

# %% [markdown]
# ### Small models are where tool use starts to fail
#
# This is the practical reason to care about the size ladder. Ask every model to
# make the same tool call and count who gets it right.

# %%
print(f"{'model':40} {'tool call?':>11}  arguments")
print("-" * 92)
for model in LADDER:
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "How much is 250 EUR in USD?"}],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 300,
        },
        region=REGION,
    )
    if code != 200:
        print(f"{model:40} {'HTTP ' + str(code):>11}")
        continue
    choice = (data.get("choices") or [{}])[0]
    calls = choice.get("message", {}).get("tool_calls") or []
    if calls:
        print(f"{model:40} {'yes':>11}  {calls[0]['function']['arguments'][:44]}")
    else:
        text = (choice.get("message", {}).get("content") or "").strip()
        print(f"{model:40} {'no':>11}  prose: {text[:40]!r}")

# %% [markdown]
# ## 8. Structured output with `response_format`


# %%
def json_object_call(prompt, max_tokens, model=LARGE3):
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


for budget in (64, 600):
    code, finish, content = json_object_call(
        "Give the capital and population of France as JSON.", budget
    )
    print(f"max_tokens={budget:4} HTTP {code} finish={finish!s:8} len={len(content)}")
    if finish == "length" and not content.strip():
        print("    -> truncated before any JSON; raise max_tokens")
    elif content.strip():
        print("    ->", parse_json_lenient(content))

# %% [markdown]
# **Always check `finish_reason` before parsing.** A reasoning-capable model can
# spend its whole budget thinking and return HTTP 200 with empty content.

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
        "model": LARGE3,
        "messages": [{"role": "user", "content": "Describe France."}],
        "max_tokens": 700,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "country", "strict": True, "schema": schema},
        },
    },
    region=REGION,
)
choice = (data.get("choices") or [{}])[0]
content = choice.get("message", {}).get("content")
print("json_schema ->", code, "| finish_reason:", choice.get("finish_reason"))
print("raw:", repr((content or "")[:160]))
parsed = parse_json_lenient(content or "")
print("parsed:", json.dumps(parsed, indent=2))
missing = {"country", "capital"} - set(parsed)
if missing:
    raise ValueError(f"model omitted required keys: {sorted(missing)} in {parsed}")
print("required keys present: country, capital")

# %% [markdown]
# Parse **leniently**: even in strict mode some mantle models append characters
# after a valid object, which makes a bare `json.loads()` raise on usable output.

# %% [markdown]
# ## 9. Cost-aware routing across the size ladder
#
# The pattern that justifies a family this wide: try the cheapest model, validate
# the result against a schema, escalate only on failure.

# %%
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_amount": {"type": "number"},
        "currency": {"type": "string"},
        "due_date": {"type": "string"},
    },
    "required": ["invoice_number", "total_amount", "currency", "due_date"],
    "additionalProperties": False,
}

INVOICE = (
    "INVOICE INV-2026-0042\nIssued: 2026-07-01\nDue: 2026-07-31\n"
    "Subtotal: 1,200.00 EUR\nVAT (19%): 228.00 EUR\nTotal: 1,428.00 EUR"
)


def try_extract(model: str):
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": f"Extract the invoice fields.\n\n{INVOICE}"}
            ],
            "max_tokens": 700,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice",
                    "strict": True,
                    "schema": EXTRACTION_SCHEMA,
                },
            },
        },
        region=REGION,
    )
    if code != 200:
        return None, f"HTTP {code}"
    choice = (data.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content") or ""
    if choice.get("finish_reason") == "length" and not content.strip():
        return None, "truncated"
    try:
        parsed = parse_json_lenient(content)
    except ValueError as exc:
        return None, f"unparseable ({exc.__class__.__name__})"
    # Validation gate: right keys AND the total we expect.
    if set(parsed) != set(EXTRACTION_SCHEMA["properties"]):
        return None, "wrong keys"
    if abs(float(parsed["total_amount"]) - 1428.0) > 0.01:
        return None, f"wrong total: {parsed['total_amount']}"
    return parsed, "ok"


print(f"{'model':40} {'verdict':>16}  result")
print("-" * 100)
chosen = None
for model in LADDER:  # cheapest first
    result, verdict = try_extract(model)
    print(f"{model:40} {verdict:>16}  {json.dumps(result)[:44] if result else ''}")
    if result and chosen is None:
        chosen = (model, result)

print(f"\ncheapest model that passed the gate: {chosen[0] if chosen else 'none'}")
if chosen:
    print("extracted:", json.dumps(chosen[1], indent=2))

# %% [markdown]
# ## 10. Latency across the ladder

# %%
task = "In one sentence, what is a mixture-of-experts model?"
print(f"{'model':40} {'latency':>9} {'out tok':>8}  answer")
print("-" * 104)
for model in LADDER + [MAGISTRAL]:
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
        print(f"{model:40} {'-':>9} {'-':>8}  HTTP {code}: {err(data)[:34]}")
        continue
    text = " ".join((data["choices"][0]["message"]["content"] or "").split())
    print(
        f"{model:40} {elapsed:>8.2f}s {data['usage']['completion_tokens']:>8}  "
        f"{text[:38]!r}"
    )

# %%
print(f"{'tier':10} {'TTFT (s)':>10} {'total (s)':>10} {'frames/s':>10}")
print("-" * 44)
for tier in ("default", "flex", "priority"):
    m = ttft(
        f"{PREFIX}/chat/completions",
        {
            "model": M8B,
            "messages": [{"role": "user", "content": task}],
            "max_tokens": 200,
            "service_tier": tier,
        },
        region=REGION,
    )
    if m.get("error"):
        print(f"{tier:10} {m['error']:>32}  (not supported by this model)")
    else:
        print(
            f"{tier:10} {m['ttft_s']:>10.3f} {m['total_s']:>10.3f} "
            f"{m['frames_per_s']:>10.1f}"
        )

# %% [markdown]
# ## 11. Regional availability
#
# Note `mistral-large-3` is **absent from eu-central-1** while the Ministrals are
# present — so a EU-resident deployment cannot use the flagship as a fallback.

# %%
regions = ("us-east-1", "us-east-2", "us-west-2", "eu-central-1")
inventory = {}
for region in regions:
    try:
        inventory[region] = set(list_models(region))
    except (RuntimeError, OSError) as exc:
        inventory[region] = set()
        print(f"{region}: {type(exc).__name__}")

print(f"{'model':40} " + "  ".join(f"{r:>13}" for r in regions))
print("-" * 100)
for model in LADDER + [MAGISTRAL]:
    cells = "  ".join(
        f"{('yes' if model in inventory[r] else '-'):>13}" for r in regions
    )
    print(f"{model:40} {cells}")

# %% [markdown]
# ## 12. Production hardening

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "mistral-samples",
        "tags": {"Application": "MistralDemo", "Environment": "Demo"},
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)


class MistralRouter:
    """Cheapest-model-first router with a schema validation gate."""

    def __init__(self, ladder=LADDER, region=REGION, tier="default", project=None):
        self.ladder, self.region, self.tier, self.project = (
            ladder,
            region,
            tier,
            project,
        )

    def extract(self, prompt: str, schema: dict, validate=None, max_tokens=700):
        headers = {"OpenAI-Project": self.project} if self.project else None
        for model in self.ladder:
            # post() retries 429/5xx with exponential backoff and jitter.
            code, data = post(
                f"{PREFIX}/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "service_tier": self.tier,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "out",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                },
                region=self.region,
                headers=headers,
            )
            if code != 200:
                continue
            choice = (data.get("choices") or [{}])[0]
            content = choice.get("message", {}).get("content") or ""
            if not content.strip():  # e.g. finish_reason == "length"
                continue
            try:
                parsed = parse_json_lenient(content)
            except ValueError:
                continue
            if validate and not validate(parsed):
                continue
            return {"model": model, "result": parsed}
        raise RuntimeError("no model in the ladder produced a valid result")


router = MistralRouter(project=project_id)
out = router.extract(
    f"Extract the invoice fields.\n\n{INVOICE}",
    EXTRACTION_SCHEMA,
    validate=lambda d: abs(float(d.get("total_amount", 0)) - 1428.0) < 0.01,
)
print("served by:", out["model"])
print("result   :", json.dumps(out["result"]))

# %%
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — Mistral on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | Path prefix | Bare `/v1`, **not** `/openai/v1` |
# | Responses API | Returns 400 for this family — Chat Completions only |
# | Wrong path | Not always a fast failure — always set a client-side timeout |
# | History | No `previous_response_id`; send `messages` every turn |
# | Reasoning trace | `reasoning_effort` works but the trace is never returned |
# | `content` can be `None` | Check `finish_reason` before slicing/parsing |
# | Strict JSON | Parse leniently — models can append text after a valid object |
# | Small models | 3B/8B often skip tool calls — validate, then escalate |
# | Region | `mistral-large-3` absent from eu-central-1; Ministrals present |
# | Quotas | No RPM quota; most models have no published TPM (tokens per minute) — retry with backoff |
#
# ## Where next
# - `02-devstral-and-voxtral.ipynb` — coding and audio-capable variants
# - Same API shape: `../04-qwen/`, `../05-deepseek/`, `../06-zai-glm/`
# - Different API shape: `../03-google-gemma/` (Responses), `../02-anthropic-claude/`
#   (Messages)
