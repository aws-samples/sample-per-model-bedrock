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
# # xAI Grok 4.3 on Amazon Bedrock Mantle
#
# Grok 4.3 on the `bedrock-mantle` endpoint — a reasoning-first model with
# always-on thinking, strong tool use, and a large context window. It is aimed at
# document-heavy enterprise work: contract review, case-law research, credit
# agreement analysis, financial Q&A.
#
# **Models covered**
#
# | Model ID | Notes |
# |---|---|
# | `xai.grok-4.3` | Reasoning-first; configurable effort (none/low/medium/high) |
#
# ## Both OpenAI-compatible APIs, on the `/openai/v1` path
# Grok serves **Responses and Chat Completions**, both under `/openai/v1` — the
# same prefix as `gemma-4` and `gpt-5.x`, *not* the bare `/v1` used by most
# open-weight families.
#
# ## Two things to know before you start
# 1. **Reasoning is always on**, and it spends roughly **400 output tokens before
#    any answer text appears**. A tight `max_output_tokens` therefore returns
#    HTTP 200 with `status: "incomplete"` and an **empty string**. Budget ≥1000.
# 2. Grok **rejects `temperature`** at any value but **accepts `top_p`** — the
#    inverse of Gemma 4. Its model card also documents non-standard defaults:
#    `temperature=0.7`, `top_p=0.95`, `max_completion_tokens=131072`.
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
from mantle import err, list_models, parse_json_lenient, post, response_text, ttft

# Grok is in us-east-1 / us-east-2 / us-west-2 but NOT eu-central-1 (see §9).
REGION = "us-east-1"
GROK = "xai.grok-4.3"

# Same prefix as gemma-4 and gpt-5.x — not the bare /v1.
PREFIX = "/openai/v1"
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}"
print("base URL:", BASE_URL)

# %% [markdown]
# ## 1. First call
#
# Auth is a short-term Bedrock API key minted from ambient IAM credentials —
# valid ≤12 h, not refreshable, Region-pinned. (`../00-foundations/01` shows the
# self-refreshing provider and the SigV4 alternative that needs no key.)

# %%
from aws_bedrock_token_generator import provide_token
from openai import APITimeoutError, OpenAI

# Build from a FRESH token; don't construct once at import and reuse for hours.
# max_retries: bedrock-mantle returns transient 500/503 under load and the SDK
# does NOT retry by default. The `post()` helper used elsewhere already retries.
client = OpenAI(
    api_key=provide_token(region=REGION),
    base_url=BASE_URL,
    max_retries=5,
    timeout=180.0,
)

resp = client.responses.create(
    model=GROK,
    input="Explain what a reasoning-first model is, in two sentences.",
    max_output_tokens=1000,  # >=1000: reasoning spends ~400 before any text
)
print(resp.output_text)
print("\nusage:", resp.usage.model_dump_json())

# %% [markdown]
# ## 2. Both APIs work — and both live under `/openai/v1`
#
# Probe every combination so the working set is explicit.

# %%
# Use a short timeout and a single attempt: one of these combinations does not
# return a clean 400 but simply STALLS (see the note below), so a naive probe with
# retries can hang for many minutes.
print(f"{'API':22} {'/openai/v1':>14} {'/v1':>18}")
print("-" * 58)
for label, suffix, body in [
    (
        "Responses",
        "/responses",
        {"model": GROK, "input": "Reply OK", "max_output_tokens": 16},
    ),
    (
        "Chat Completions",
        "/chat/completions",
        {
            "model": GROK,
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_tokens": 16,
        },
    ),
]:
    cells = []
    for prefix in ("/openai/v1", "/v1"):
        started = time.perf_counter()
        code, data = post(
            prefix + suffix, body, region=REGION, attempts=1, timeout=30
        )  # no retries, short timeout
        elapsed = time.perf_counter() - started
        cells.append(
            f"{code} ({elapsed:.0f}s)" if code != -1 else f"stalled ({elapsed:.0f}s)"
        )
    print(f"{label:22} {cells[0]:>14} {cells[1]:>18}")

# %% [markdown]
# ### Two lessons here
#
# 1. Grok is served on **`/openai/v1`** for both APIs. The bare `/v1` does not work.
# 2. `/v1/chat/completions` returns a clean, fast **400** ("isn't supported on this
#    route"), but **`/v1/responses` does not respond at all — it stalls**. A wrong
#    path is therefore not always a fast failure.
#
# Practical consequence: **always set a client-side timeout**, and do not retry
# indefinitely on a stall. A retry loop with a 240 s timeout and 5 attempts turns
# one bad path into a 20-minute hang.

# %%
# The clean 400 from the wrong route, for reference.
code, data = post(
    "/v1/chat/completions",
    {
        "model": GROK,
        "messages": [{"role": "user", "content": "Reply OK"}],
        "max_tokens": 16,
    },
    region=REGION,
    attempts=1,
    timeout=30,
)
print(f"/v1/chat/completions -> HTTP {code}: {err(data)[:100]}")

# %% [markdown]
# Both APIs are available on `/openai/v1` and both 400 on the bare `/v1`. Prefer
# **Responses**: it is the only surface that returns reasoning content, and Grok is
# a reasoning-first model, so that matters more here than usual.

# %% [markdown]
# ## 3. Sampling: `temperature` is rejected, `top_p` is not
#
# This is the headline gotcha, and it runs the opposite way to Gemma 4. Compare
# both models against both parameters, at two temperature values:

# %%
print(f"{'model':24} {'temp=0.7':>10} {'temp=1.0':>10} {'top_p=0.95':>12}")
print("-" * 60)
for model in (GROK, "google.gemma-4-31b"):
    results = []
    for extra in ({"temperature": 0.7}, {"temperature": 1.0}, {"top_p": 0.95}):
        code, data = post(
            f"{PREFIX}/responses",
            {"model": model, "input": "Reply OK", "max_output_tokens": 16, **extra},
            region=REGION,
        )
        results.append("ok" if code == 200 else f"{code}")
    print(f"{model:24} {results[0]:>10} {results[1]:>10} {results[2]:>12}")

# %% [markdown]
# ### The actual rule: each model accepts only *its own default*
#
# | | `temperature=0.7` | `temperature=1.0` | `top_p=0.95` |
# |---|---|---|---|
# | **Grok 4.3** | **accepted** | rejected | **accepted** |
# | **Gemma 4** | rejected | **accepted** | rejected |
#
# This looks arbitrary until you read the model cards. Grok's documents
# non-standard defaults — **`temperature=0.7`, `top_p=0.95`** — while Gemma 4 uses
# the spec default of `1.0`. On the Responses API each model accepts `temperature`
# **only at its own documented default**, and rejects every other value.
#
# In other words: you are not tuning sampling on this API, you are echoing the
# default back. The safest port is to **omit both parameters entirely** unless the
# model is gpt-oss- or qwen-class, which accept the full range.

# %%
# The exact error, so it is searchable.
code, data = post(
    f"{PREFIX}/responses",
    {"model": GROK, "input": "Reply OK", "max_output_tokens": 16, "temperature": 0.7},
    region=REGION,
)
# Sweep the values: only Grok's own documented default (0.7) is accepted.
print(f"{'temperature':>12} {'HTTP':>6}  detail")
print("-" * 70)
for value in (0.0, 0.5, 0.7, 1.0, 1.5):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GROK,
            "input": "Reply OK",
            "max_output_tokens": 16,
            "temperature": value,
        },
        region=REGION,
    )
    print(f"{value:>12} {code:>6}  {'' if code == 200 else err(data)[:48]}")
print("\n=> 0.7 is Grok's documented default. Safest: omit temperature entirely.")

# %%
# max_output_tokens minimum is 16 on the Responses API.
for n in (8, 16):
    code, data = post(
        f"{PREFIX}/responses",
        {"model": GROK, "input": "Hi", "max_output_tokens": n},
        region=REGION,
    )
    detail = "" if code == 200 else err(data)[:70]
    print(f"max_output_tokens={n:3} -> HTTP {code} {detail}")

# %% [markdown]
# ## 4. Reasoning — always on, effort configurable
#
# Grok reasons on every request rather than treating thinking as optional. That
# makes it behave consistently across multi-step agent loops, at the cost of
# spending reasoning tokens even on easy questions.

# %%
for effort in ("none", "minimal", "low", "medium", "high"):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GROK,
            "input": "2+2?",
            "max_output_tokens": 32,
            "reasoning": {"effort": effort},
        },
        region=REGION,
    )
    print(f"  effort={effort:8} -> HTTP {code} {'' if code == 200 else err(data)[:60]}")

# %%
# Each row is a SINGLE sample, so read the ordering, not the absolute numbers -
# Grok's wall-clock time swings widely with endpoint load.
print(f"{'effort':8} {'reasoning tok':>14} {'output tok':>11} {'latency':>9}")
print("-" * 46)
for effort in ("none", "low", "medium"):
    started = time.perf_counter()
    try:
        r = client.with_options(timeout=180.0).responses.create(
            model=GROK,
            input="What is 47 * 89? Answer with the number only.",
            reasoning={"effort": effort},
            max_output_tokens=1500,
        )
    except APITimeoutError:
        # A timeout here is a load signal, not a bug. Report and keep the table
        # going rather than aborting the notebook.
        print(f"{effort:8} {'timed out after 180s (endpoint under load)':>38}")
        continue
    elapsed = time.perf_counter() - started
    print(
        f"{effort:8} {r.usage.output_tokens_details.reasoning_tokens:>14} "
        f"{r.usage.output_tokens:>11} {elapsed:>8.2f}s"
    )
print("\nHigher effort spends more reasoning tokens and more wall-clock time.")
print("'high' is omitted here: it can exceed 15 minutes for one call under load.")

# %% [markdown]
# ### Set a client timeout on reasoning calls
#
# Grok's wall-clock time varies a lot with load: the same request has taken 45s and
# well over 15 minutes on different days. A reasoning call with no client timeout is
# therefore a latent hang in your application, not just a slow cell.
#
# The SDK's default timeout is generous, so set your own — and pick `effort` for the
# job rather than reaching for `high` by default. `medium` answers this question just
# as well and costs a fraction of the wait.

# %%
# Escalate the budget until the answer appears. Grok spends its reasoning tokens
# FIRST, so an under-budgeted call returns HTTP 200, status="incomplete", and an
# empty string -- reasoning consumed everything before any answer text.
#
# `timeout=` matters as much as the budget: bound the wait explicitly, because a
# reasoning call has no natural upper bound on latency.
grok = client.with_options(timeout=240.0)
QUESTION = (
    "A contract says payment is due 'within 30 days of invoice receipt, "
    "excluding public holidays'. The invoice arrives 1 December. Name two "
    "ambiguities a reviewer should flag."
)

print(f"{'budget':>8} {'status':>12} {'reasoning tok':>14} {'answer chars':>13}")
print("-" * 52)
reasoned = None
for budget in (1200, 2500, 4000):
    attempt = grok.responses.create(
        model=GROK,
        input=QUESTION,
        reasoning={"effort": "medium"},  # 'high' can take many minutes under load
        max_output_tokens=budget,
    )
    used = attempt.usage.output_tokens_details.reasoning_tokens
    text = attempt.output_text or ""
    print(f"{budget:>8} {attempt.status:>12} {used:>14} {len(text):>13}")
    if text.strip():
        reasoned = attempt
        break

if reasoned is None:
    raise RuntimeError(
        "no answer text even at 4000 tokens - reasoning consumed the whole budget"
    )

print("\noutput item types:", [i.type for i in reasoned.output])
print("=== ANSWER ===")
print(reasoned.output_text[:500])

# %% [markdown]
# ## 4b. Budget for reasoning, or you get an empty answer
#
# This is the single most important operational fact about Grok. Because reasoning
# always runs first, a small `max_output_tokens` is consumed entirely by thinking.
# The request succeeds — **HTTP 200** — but `status` is `incomplete` and the text is
# empty. Sweep the budget and watch where text starts appearing:

# %%
print(f"{'max_output_tokens':>18} {'status':>12} {'reason tok':>11} {'chars':>7}")
print("-" * 54)
for budget in (400, 1000, 2000):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GROK,
            "input": "List three contract risks.",
            "reasoning": {"effort": "low"},
            "max_output_tokens": budget,
        },
        region=REGION,
    )
    details = (data.get("usage") or {}).get("output_tokens_details") or {}
    text = response_text(data)
    print(
        f"{budget:>18} {str(data.get('status')):>12} "
        f"{details.get('reasoning_tokens'):>11} {len(text):>7}"
    )

# %% [markdown]
# At 400 tokens the whole budget goes to reasoning and you get nothing back. The
# practical rules:
#
# - **Always check `status`** (or that the text is non-empty) before using a result.
# - Budget **≥1000** output tokens for anything you expect an answer from.
# - `reasoning={"effort": "none"}` disables thinking, which frees the whole budget
#   for the answer — useful for simple extraction where you do not need reasoning.

# %%
for effort in ("low", "none"):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GROK,
            "input": "List three contract risks.",
            "reasoning": {"effort": effort},
            "max_output_tokens": 1000,
        },
        region=REGION,
    )
    details = (data.get("usage") or {}).get("output_tokens_details") or {}
    print(
        f"effort={effort:5} reasoning_tokens={details.get('reasoning_tokens'):>4} "
        f"-> {response_text(data)[:70]!r}"
    )

# %% [markdown]
# ## 4c. Reasoning content is encrypted
#
# Unlike Gemma 4 or gpt-5.x, Grok does not return readable reasoning text. Its model
# card documents that the trace is **encrypted**, retrievable with
# `include: ["reasoning.encrypted_content"]`, and can be passed back on later turns
# to carry reasoning context forward.

# %%
code, data = post(
    f"{PREFIX}/responses",
    {
        "model": GROK,
        "input": "Why is the sky blue? One sentence.",
        "reasoning": {"effort": "low"},
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": 1000,
    },
    region=REGION,
)
print("HTTP", code, "| items:", [i.get("type") for i in data.get("output", [])])
for item in data.get("output", []):
    if item.get("type") == "reasoning":
        encrypted = item.get("encrypted_content")
        print(f"   reasoning item keys : {sorted(item.keys())}")
        print(
            f"   encrypted_content   : {'present' if encrypted else 'absent'}"
            f"{f' ({len(encrypted)} chars)' if encrypted else ''}"
        )
print("answer:", response_text(data)[:150])

# %% [markdown]
# ## 5. Streaming
#
# Reasoning and answer text arrive as separate event types.

# %%
stream = client.responses.create(
    model=GROK,
    input="List three risks in a credit agreement review.",
    reasoning={"effort": "low"},
    max_output_tokens=1200,
    stream=True,
)
counts = {}
print("--- live ---")
for event in stream:
    counts[event.type] = counts.get(event.type, 0) + 1
    if event.type == "response.reasoning_text.delta":
        print("\033[2m" + event.delta + "\033[0m", end="", flush=True)
    elif event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print("\n\n--- event types ---")
for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4}  {name}")

# %% [markdown]
# ## 6. Multi-turn: history array vs server-side state

# %%
conversation = [
    {"role": "system", "content": "You are a precise legal-operations assistant."},
    {"role": "user", "content": "What is a force majeure clause?"},
]
first = client.responses.create(model=GROK, input=conversation, max_output_tokens=1000)
print("assistant:", first.output_text[:180])

conversation += [
    {"role": "assistant", "content": first.output_text},
    {"role": "user", "content": "Name one event commonly excluded from it."},
]
second = client.responses.create(model=GROK, input=conversation, max_output_tokens=1000)
print("\nassistant:", second.output_text[:180])

# %% [markdown]
# Append only **final answers** to history — not reasoning items. Replaying a
# model's own reasoning degrades later turns.

# %%
# Server-side state needs store=True, which retains input+output for 30 days
# in-Region (see ../00-foundations/02 for the retention/ZDR controls).
turn1 = client.responses.create(
    model=GROK,
    input="Our governing law is Singapore. Reply: noted.",
    max_output_tokens=1000,
    store=True,
)
turn2 = client.responses.create(
    model=GROK,
    input="Which governing law did I state?",
    previous_response_id=turn1.id,
    max_output_tokens=1000,
)
print("chained recall:", turn2.output_text[:140])

unstored = client.responses.create(
    model=GROK, input="Confidential. Reply ok.", max_output_tokens=1000, store=False
)
code, data = post(
    f"{PREFIX}/responses",
    {
        "model": GROK,
        "input": "What did I say?",
        "previous_response_id": unstored.id,
        "max_output_tokens": 1000,
    },
    region=REGION,
)
print(f"chaining from store=False -> HTTP {code}: {err(data)[:80]}")

# %% [markdown]
# ## 7. Tool use and structured output
#
# Grok has strong tool use, which is the main reason to pick it for agent loops.
# Note the **flat** Responses tool shape.


# %%
def get_filing(company: str, year: int) -> dict:
    """Stand-in for a filings database."""
    table = {("acme", 2025): {"revenue_musd": 812, "net_margin_pct": 11.4}}
    hit = table.get((company.lower(), year))
    return {"company": company, "year": year, "found": bool(hit), **(hit or {})}


filing_tool = {
    "type": "function",
    "name": "get_filing",
    "description": "Fetch headline financials for a company and year.",
    "parameters": {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "year": {"type": "integer"},
        },
        "required": ["company", "year"],
    },
}

# NOTE: we use the retrying `post()` helper rather than the bare SDK here.
# bedrock-mantle returns transient 500s under load, and the SDK does not retry
# them — a tool loop that runs several calls will eventually hit one.
convo = [{"role": "user", "content": "What was Acme's 2025 net margin?"}]
code, data = post(
    f"{PREFIX}/responses",
    {
        "model": GROK,
        "input": convo,
        "tools": [filing_tool],
        "tool_choice": "auto",
        "max_output_tokens": 1200,
    },
    region=REGION,
)
print("HTTP", code)
calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
print("tool calls:", [(c["name"], c["arguments"]) for c in calls])

for call in calls:
    args = parse_json_lenient(call["arguments"])
    result = get_filing(**args)
    convo.append(
        {
            "type": "function_call",
            "call_id": call["call_id"],
            "name": call["name"],
            "arguments": call["arguments"],
        }
    )
    convo.append(
        {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": json.dumps(result),
        }
    )

code, final = post(
    f"{PREFIX}/responses",
    {"model": GROK, "input": convo, "tools": [filing_tool], "max_output_tokens": 1000},
    region=REGION,
)
print("\nfinal answer:", response_text(final)[:220])

# %%
# Strict JSON via the native schema route.
schema = {
    "type": "object",
    "properties": {
        "clause_type": {"type": "string"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "rationale": {"type": "string"},
    },
    "required": ["clause_type", "risk", "rationale"],
    "additionalProperties": False,
}

code, data = post(
    f"{PREFIX}/responses",
    {
        "model": GROK,
        "input": "Classify: 'Either party may terminate immediately without notice.'",
        "max_output_tokens": 1500,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "clause",
                "schema": schema,
                "strict": True,
            }
        },
    },
    region=REGION,
)
raw = response_text(data)
print("HTTP", code, "| raw:", repr(raw[:160]))
# parse_json_lenient rather than json.loads: models can append characters after a
# valid object, and an empty body is possible if the token budget runs out first.
if raw.strip():
    print(json.dumps(parse_json_lenient(raw), indent=2))
else:
    print("empty output — raise max_output_tokens (reasoning consumed the budget)")

# %% [markdown]
# **Watch the budget on a reasoning-first model.** Grok thinks before it writes, so
# a tight `max_output_tokens` can be exhausted before any JSON is emitted, giving
# HTTP 200 with an empty string. Always check for content before parsing.

# %%
print(f"{'max_output_tokens':>18} {'chars returned':>16}")
print("-" * 36)
for budget in (400, 1000, 2000):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GROK,
            "input": "Classify: 'Termination without notice.'",
            "max_output_tokens": budget,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "clause",
                    "schema": schema,
                    "strict": True,
                }
            },
        },
        region=REGION,
    )
    print(f"{budget:>18} {len(response_text(data)):>16}")

# %% [markdown]
# ## 8. Web Search is not available here
#
# Bedrock's built-in Web Search tool is currently **OpenAI GPT only**. Attempting
# it on Grok is an explicit 400 — worth showing so you don't plan around it.
# (See `../01-openai-gpt/02-web-search-and-grounding.ipynb`.)

# %%
code, data = post(
    f"{PREFIX}/responses",
    {
        "model": GROK,
        "input": "What happened in the news today?",
        "max_output_tokens": 1000,
        "tools": [{"type": "web_search"}],
    },
    region=REGION,
)
print(f"web_search on Grok -> HTTP {code}")
print("message:", err(data)[:130])

# %% [markdown]
# ## 9. Regional availability

# %%
regions = ("us-east-1", "us-east-2", "us-west-2", "eu-central-1")
print(f"{'region':14} {'grok-4.3':>10}")
print("-" * 26)
for region in regions:
    try:
        present = GROK in list_models(region)
    except (RuntimeError, OSError) as exc:
        print(f"{region:14} {type(exc).__name__}")
        continue
    print(f"{region:14} {('yes' if present else '-'):>10}")

# %% [markdown]
# ## 10. Latency and service tiers

# %%
print(f"{'tier':10} {'TTFT (s)':>10} {'total (s)':>10} {'frames/s':>10}")
print("-" * 44)
for tier in ("default", "flex", "priority"):
    m = ttft(
        f"{PREFIX}/responses",
        {
            "model": GROK,
            "input": "List three contract review checkpoints.",
            "max_output_tokens": 1000,
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
# Tiers govern queue priority and mostly separate under contention, so single
# samples on an idle account look flat. See `../00-foundations/03`.

# %% [markdown]
# ## 11. A worked enterprise example
#
# The workload Grok is positioned for: read a document, extract structured
# findings, flag risk.

# %%
CONTRACT = """
SERVICES AGREEMENT (extract)
4.1 The Supplier shall deliver the Services with reasonable skill and care.
4.2 Either party may terminate this Agreement immediately upon written notice
    if the other party commits a material breach that remains unremedied for
    fourteen (14) days.
7.3 The Supplier's total liability shall not exceed the fees paid in the three
    (3) months preceding the claim.
9.1 This Agreement is governed by the laws of Singapore.
"""

findings_schema = {
    "type": "object",
    "properties": {
        "governing_law": {"type": "string"},
        "liability_cap": {"type": "string"},
        "termination_notice_days": {"type": "integer"},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "overall_risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "governing_law",
        "liability_cap",
        "termination_notice_days",
        "concerns",
        "overall_risk",
    ],
    "additionalProperties": False,
}

code, data = post(
    f"{PREFIX}/responses",
    {
        "model": GROK,
        "input": [
            {
                "role": "system",
                "content": "You are a contract reviewer. Be conservative.",
            },
            {
                "role": "user",
                "content": f"Review this extract and return findings.\n{CONTRACT}",
            },
        ],
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 2000,  # generous: reasoning runs before the JSON
        "text": {
            "format": {
                "type": "json_schema",
                "name": "findings",
                "schema": findings_schema,
                "strict": True,
            }
        },
        "store": False,
    },
    region=REGION,
)
raw = response_text(data)
print("HTTP", code, "| chars:", len(raw))
if raw.strip():
    print(json.dumps(parse_json_lenient(raw), indent=2))
else:
    print("empty — raise max_output_tokens further")

# %% [markdown]
# ## 12. Production hardening

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "grok-samples",
        "tags": {"Application": "GrokDemo", "Environment": "Demo"},
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)


class GrokClient:
    """Production shape: no temperature, generous budget, retries, attribution."""

    def __init__(self, region=REGION, tier="default", project=None):
        self.model, self.region, self.tier, self.project = GROK, region, tier, project

    def ask(self, prompt, *, effort="low", max_output_tokens=1500, schema=None):
        body = {
            "model": self.model,
            "input": prompt,
            # Reasoning-first: budget generously or you get an empty answer.
            # Floor of 1000: reasoning spends ~400 tokens before any text.
            "max_output_tokens": max(1000, max_output_tokens),
            "reasoning": {"effort": effort},
            "service_tier": self.tier,
            "store": False,  # opt out of the 30-day retention default
            # temperature deliberately omitted: rejected by this model.
            "top_p": 0.95,  # Grok's documented default; accepted
        }
        if schema:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "out",
                    "schema": schema,
                    "strict": True,
                }
            }
        headers = {"OpenAI-Project": self.project} if self.project else None
        # post() retries 429/5xx with exponential backoff.
        code, data = post(
            f"{PREFIX}/responses", body, region=self.region, headers=headers
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        text = response_text(data)
        if schema:
            if not text.strip():
                raise RuntimeError("empty output — raise max_output_tokens")
            return parse_json_lenient(text)
        return text


grok = GrokClient(project=project_id)

print("plain      :", grok.ask("Name one benefit of always-on reasoning.")[:140])

RISK_SCHEMA = {
    "type": "object",
    "properties": {"risk": {"type": "string", "enum": ["low", "medium", "high"]}},
    "required": ["risk"],
    "additionalProperties": False,
}
print(
    "structured :",
    grok.ask(
        "Classify the risk of: 'Unlimited liability for the supplier.'",
        schema=RISK_SCHEMA,
        max_output_tokens=1500,
    ),
)

# %%
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — Grok 4.3 on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | **`temperature`** | Accepted **only at Grok's own default `0.7`** — other values 400 |
# | Per-model defaults | Each model accepts `temperature` only at *its* documented default (Grok 0.7, Gemma 4 1.0) |
# | Path prefix | `/openai/v1`. Bare `/v1` chat/completions 400s fast, but **`/v1/responses` stalls** |
# | Always set timeouts | A stalling path plus a retry loop = a multi-minute hang |
# | Transient 500s | The service returns them under load; the SDK does **not** retry — wrap your calls |
# | Reasoning always on | Spends ~400 tokens first — budget **≥1000** or you get empty output |
# | `status: "incomplete"` | HTTP 200 + empty string when reasoning eats the budget — check `status` |
# | Reasoning is encrypted | Not human-readable; use `include:["reasoning.encrypted_content"]` |
# | Documented defaults | `temperature=0.7`, `top_p=0.95`, `max_completion_tokens=131072` |
# | `max_output_tokens` | Minimum **16** |
# | `reasoning.effort` | `none`/`low`/`medium`/`high`; **`minimal` rejected** |
# | `web_search` | **Not supported** — OpenAI GPT models only |
# | `store=False` | Blocks `previous_response_id` chaining (404) |
# | Region | Absent from eu-central-1 |
#
# ## Where next
# - Same path prefix: `../03-google-gemma/`, `../01-openai-gpt/`
# - Web Search: `../01-openai-gpt/02-web-search-and-grounding.ipynb`
# - Different API shape: `../04-qwen/` (Chat Completions), `../02-anthropic-claude/`
#   (Messages)
