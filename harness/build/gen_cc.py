#!/usr/bin/env python3
"""Generate the Chat-Completions-family notebooks from one tested template.

Every family gets a self-contained notebook: same arc, family-specific models,
prose, and extras. Duplication across notebooks is deliberate — a reader who
lands on one family's notebook from a search engine must not need any other file.
"""
import pathlib

SRC = pathlib.Path(__file__).parent / "src"

HEADER = '''# ---
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
'''


TOOLS_SECTION = '''# %% [markdown]
# ## 7. Tool use (function calling)
#
# Chat Completions nests the schema under `"function"` — unlike the Responses API,
# which puts `name`/`parameters` at the top level. Same concept, different shape.
# %%
def lookup_inventory(sku: str, warehouse: str = "main") -> dict:
    """Stand-in for a real inventory service."""
    stock = {"A-100": 42, "B-200": 0, "C-300": 7}
    return {"sku": sku, "warehouse": warehouse,
            "quantity": stock.get(sku.upper(), 0),
            "in_stock": stock.get(sku.upper(), 0) > 0}


tools = [{
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
}]

convo = [{"role": "user", "content": "Do we have SKU A-100 in stock?"}]
first = client.chat.completions.create(
    model=__PRIMARY__, messages=convo, tools=tools, tool_choice="auto", max_tokens=300
)
msg = first.choices[0].message
print("finish_reason:", first.choices[0].finish_reason)
print("tool_calls:", [(c.function.name, c.function.arguments) for c in (msg.tool_calls or [])])

if msg.tool_calls:
    convo.append(msg.model_dump(exclude_none=True))
    for call in msg.tool_calls:
        args = parse_json_lenient(call.function.arguments)
        result = lookup_inventory(**args)
        convo.append({"role": "tool", "tool_call_id": call.id,
                       "content": json.dumps(result)})
    final = client.chat.completions.create(
        model=__PRIMARY__, messages=convo, tools=tools, max_tokens=200
    )
    print("\\nfinal answer:", final.choices[0].message.content)

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
emit = [{
    "type": "function",
    "function": {
        "name": "emit_review",
        "description": "Return the structured review analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                "would_recommend": {"type": "boolean"},
            },
            "required": ["summary", "sentiment", "would_recommend"],
        },
    },
}]

def emit_review(prompt, model=__PRIMARY__, attempts=3):
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
        print(f"attempt {attempt + 1}: no tool call "
              f"(finish_reason={choice.finish_reason}) — retrying")
    raise RuntimeError("model would not emit the forced tool call")


review = emit_review("Review: 'Fast delivery, but the packaging arrived crushed.'")
print(json.dumps(review, indent=2))

'''

NO_TOOLS_SECTION = '''# %% [markdown]
# ## 7. Tool use — not available on this model
#
# This family **rejects tool definitions**. Its model card lists "Client-side tool
# calling: Not Supported", and the API returns a 400. We show it rather than
# leaving you to discover it.
#
# The consequence: the usual "forced tool call for strict JSON" trick is
# unavailable here, so `response_format` (next section) is your only structured
# output route.
'''

def build(cfg) -> str:
    p = []
    a = p.append

    # ---------------- intro ----------------
    a(HEADER)
    a(f'''
# %% [markdown]
# # {cfg["title"]} on Amazon Bedrock Mantle
#
# {cfg["blurb"]}
#
# **Models covered in this notebook**
#
# | Model ID | Notes |
# |---|---|
''')
    for mid, note in cfg["models_table"]:
        a(f"# | `{mid}` | {note} |\n")
    a(f'''#
# ### Which API? Chat Completions.
# This family is served by the **OpenAI-compatible Chat Completions API** on the
# `bedrock-mantle` endpoint, at the bare `/v1` path. The Responses API returns
# **400 "does not support this API"** for these models — we prove that in §2 rather
# than asking you to take it on trust.
#
# ### Self-contained, but see also
# Everything you need is here. For deeper background on shared mechanics:
# - **Auth (SigV4 + short-term API keys), the three URL paths, model discovery** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Projects, cost attribution, data retention / ZDR, CloudWatch namespace** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
# - **Quotas, retry/backoff, service tiers, TTFT measurement** →
#   `../00-foundations/03-scaling-tiers-and-latency.ipynb`
#
# ### Prerequisites
# ```bash
# pip install openai aws-bedrock-token-generator
# ```
# AWS credentials with `bedrock-mantle:CreateInference` and
# `bedrock-mantle:CallWithBearerToken` — both granted by the managed policy
# `AmazonBedrockMantleInferenceAccess`.

# %%
import json
import sys
import time

sys.path.insert(0, "../_shared")
from mantle import err, parse_json_lenient, post, ttft

REGION = "{cfg["region"]}"

{cfg["model_consts"]}
# Chat-Completions families live at the BARE /v1 path — not /openai/v1
# (that prefix is only for gemma-4, gpt-5.x and grok). See ../00-foundations/01.
PREFIX = "/v1"
BASE_URL = f"https://bedrock-mantle.{{REGION}}.api.aws{{PREFIX}}"
print("base URL:", BASE_URL)
print("models  :", {cfg["model_list_expr"]})

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
    model={cfg["primary"]},
    messages=[{{"role": "user", "content": "{cfg["first_prompt"]}"}}],
    max_tokens=250,
)
print(completion.choices[0].message.content)
print("\\nusage:", completion.usage.model_dump_json())

# %% [markdown]
# ## 2. Why Chat Completions and not Responses
#
# AWS recommends the Responses API for new applications in general — but
# availability is per-model. Probe both surfaces so the 400 is visible:

# %%
for api_name, path, body in [
    ("Chat Completions", f"{{PREFIX}}/chat/completions",
     {{"model": {cfg["primary"]}, "messages": [{{"role": "user", "content": "Reply OK"}}],
      "max_tokens": 16}}),
    ("Responses (/v1)", f"{{PREFIX}}/responses",
     {{"model": {cfg["primary"]}, "input": "Reply OK", "max_output_tokens": 16}}),
    ("Responses (/openai/v1)", "/openai/v1/responses",
     {{"model": {cfg["primary"]}, "input": "Reply OK", "max_output_tokens": 16}}),
]:
    code, data = post(path, body, region=REGION)
    print(f"  {{api_name:24}} -> HTTP {{code}} {{'' if code == 200 else err(data)[:64]}}")

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
    ("temperature=0.7", {{"temperature": 0.7}}),
    ("temperature=0.0", {{"temperature": 0.0}}),
    ("top_p=0.95", {{"top_p": 0.95}}),
    ("both", {{"temperature": 0.7, "top_p": 0.95}}),
    ("max_tokens=1", {{"max_tokens": 1}}),
]:
    body = {{"model": {cfg["primary"]},
            "messages": [{{"role": "user", "content": "Reply OK"}}], "max_tokens": 16}}
    body.update(extra)
    code, data = post(f"{{PREFIX}}/chat/completions", body, region=REGION)
    print(f"  {{label:18}} -> HTTP {{code}} {{'' if code == 200 else err(data)[:60]}}")

# %% [markdown]
# Note `max_tokens=1` is accepted here. The Responses API enforces a minimum of
# 16 — another reason the two surfaces are not interchangeable.

# %% [markdown]
# ## 4. Streaming
#
# Chat Completions streams `data: {{...}}` SSE frames carrying
# `choices[0].delta.content`, terminated by `data: [DONE]`.

# %%
stream = client.chat.completions.create(
    model={cfg["primary"]},
    messages=[{{"role": "user", "content": "{cfg["stream_prompt"]}"}}],
    max_tokens=300,
    stream=True,
)
chunks = 0
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        chunks += 1
        print(delta, end="", flush=True)
print(f"\\n\\n[{{chunks}} content deltas received]")

# %% [markdown]
# ## 5. Multi-turn — you manage the history
#
# No server-side state on this API. Append each turn yourself.

# %%
{cfg["system_msg_note"]}messages = [
{cfg["system_msg_line"]}    {{"role": "user", "content": "{cfg["turn1"]}"}},
]
first = client.chat.completions.create(model={cfg["primary"]}, messages=messages, max_tokens=200)
print("assistant:", first.choices[0].message.content)

messages.append({{"role": "assistant", "content": first.choices[0].message.content}})
messages.append({{"role": "user", "content": "{cfg["turn2"]}"}})

second = client.chat.completions.create(model={cfg["primary"]}, messages=messages, max_tokens=200)
print("\\nassistant:", second.choices[0].message.content)
print(f"\\ninput tokens grew: {{first.usage.prompt_tokens}} -> {{second.usage.prompt_tokens}}")

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
print(f"{{'effort':10}} {{'status':>7}} {{'completion tokens':>18}}")
print("-" * 38)
for effort in ("none", "low", "medium", "high"):
    code, data = post(
        f"{{PREFIX}}/chat/completions",
        {{"model": {cfg["primary"]},
         "messages": [{{"role": "user", "content": "{cfg["reasoning_prompt"]}"}}],
         "max_tokens": 400, "reasoning_effort": effort}},
        region=REGION,
    )
    tokens = (data.get("usage") or {{}}).get("completion_tokens", "-")
    print(f"  {{effort:8}} {{code:>7}} {{tokens!s:>18}}")

# %%
{cfg["reasoning_extra"]}
{cfg["tools_section"]}
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
def json_object_call(prompt, max_tokens, model={cfg["primary"]}):
    code, data = post(
        f"{{PREFIX}}/chat/completions",
        {{"model": model, "messages": [{{"role": "user", "content": prompt}}],
         "max_tokens": max_tokens, "response_format": {{"type": "json_object"}}}},
        region=REGION,
    )
    choice = (data.get("choices") or [{{}}])[0]
    content = choice.get("message", {{}}).get("content") or ""
    return code, choice.get("finish_reason"), content


PROMPT = "Give the capital and population of France as JSON."
for budget in (64, 600):
    code, finish, content = json_object_call(PROMPT, budget)
    print(f"max_tokens={{budget:4}} HTTP {{code}} finish={{finish!s:8}} "
          f"content_len={{len(content)}}")
    if finish == "length" and not content.strip():
        print("    -> truncated before any JSON was emitted; raise max_tokens")
    elif content.strip():
        print("    ->", parse_json_lenient(content))

# %%
schema = {{
    "type": "object",
    "properties": {{
        "country": {{"type": "string"}},
        "capital": {{"type": "string"}},
        "population_millions": {{"type": "number"}},
    }},
    "required": ["country", "capital", "population_millions"],
    "additionalProperties": False,
}}

code, data = post(
    f"{{PREFIX}}/chat/completions",
    {{"model": {cfg["primary"]},
     "messages": [{{"role": "user", "content": "Describe France."}}],
     "max_tokens": 250,
     "response_format": {{"type": "json_schema",
                          "json_schema": {{"name": "country", "strict": True,
                                           "schema": schema}}}}}},
    region=REGION,
)
choice = (data.get("choices") or [{{}}])[0]
content = choice.get("message", {{}}).get("content")   # may be None!
print("json_schema ->", code, "| finish_reason:", choice.get("finish_reason"))
print("raw:", repr((content or "")[:160]))

if choice.get("finish_reason") == "length":
    # Reasoning consumed the budget before the object closed. Retry bigger.
    print("truncated - retrying with a larger budget")
    code, data = post(
        f"{{PREFIX}}/chat/completions",
        {{"model": {cfg["primary"]},
         "messages": [{{"role": "user", "content": "Describe France."}}],
         "max_tokens": 2000,
         "response_format": {{"type": "json_schema",
                              "json_schema": {{"name": "country", "strict": True,
                                               "schema": schema}}}}}},
        region=REGION,
    )
    choice = (data.get("choices") or [{{}}])[0]
    content = choice.get("message", {{}}).get("content")
    print("retry finish_reason:", choice.get("finish_reason"))

parsed = parse_json_lenient(content or "")
print("parsed:", json.dumps(parsed, indent=2))
assert set(parsed) >= {{"country", "capital"}}, parsed

# %% [markdown]
# **Always parse leniently.** Even in strict mode, some mantle models append
# characters after a valid object (Gemma 4 does this in ~half of runs), which makes
# a bare `json.loads()` raise on output that is otherwise fine.

# %% [markdown]
# ## 9. Compare the models in this family
#
# {cfg["compare_blurb"]}

# %%
task = "{cfg["compare_task"]}"

print(f"{{'model':44}} {{'latency':>9}} {{'out tok':>8}}  answer")
print("-" * 108)
for model in {cfg["compare_models"]}:
    started = time.perf_counter()
    code, data = post(
        f"{{PREFIX}}/chat/completions",
        {{"model": model, "messages": [{{"role": "user", "content": task}}],
         "max_tokens": 160}},
        region=REGION,
    )
    elapsed = time.perf_counter() - started
    if code != 200:
        print(f"{{model:44}} {{'-':>9}} {{'-':>8}}  HTTP {{code}}: {{err(data)[:40]}}")
        continue
    text = (data["choices"][0]["message"]["content"] or "").strip().replace("\\n", " ")
    print(f"{{model:44}} {{elapsed:>8.2f}}s "
          f"{{data['usage']['completion_tokens']:>8}}  {{text[:44]!r}}")

# %% [markdown]
# ## 10. Latency: TTFT and throughput
#
# TTFT is dominated by *prefill* (the model reading your prompt) plus queue time.
# Service tiers trade cost against queue priority — they mostly separate under
# contention, so single samples on an idle account look flat.
# (`../00-foundations/03` has the full treatment.)

# %%
print(f"{{'tier':10}} {{'TTFT (s)':>10}} {{'total (s)':>10}} {{'frames/s':>10}}")
print("-" * 44)
for tier in ("default", "flex", "priority"):
    m = ttft(
        f"{{PREFIX}}/chat/completions",
        {{"model": {cfg["primary"]},
         "messages": [{{"role": "user", "content": "{cfg["stream_prompt"]}"}}],
         "max_tokens": 200, "service_tier": tier}},
        region=REGION,
    )
    if m.get("error"):
        print(f"{{tier:10}} {{m['error']:>32}}  (tier not supported by this model)")
    else:
        print(f"{{tier:10}} {{m['ttft_s']:>10.3f}} {{m['total_s']:>10.3f}} "
              f"{{m['frames_per_s']:>10.1f}}")

# %% [markdown]
# ## 11. Production hardening
#
# Retries, cost attribution, and privacy. Mantle has **no RPM quota** — throttling
# is token-based, and most models here have no published TPM quota at all, so
# capacity is fair-share. That makes retry-with-backoff mandatory, not optional.

# %%
code, project = post(
    "/v1/organization/projects",
    {{"name": "{cfg["project_name"]}",
     "tags": {{"Application": "{cfg["project_tag"]}", "Environment": "Demo"}}}},
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)

code, data = post(
    f"{{PREFIX}}/chat/completions",
    {{"model": {cfg["primary"]},
     "messages": [{{"role": "user", "content": "Reply OK"}}], "max_tokens": 16}},
    region=REGION,
    headers={{"OpenAI-Project": project_id}},   # cost attribution
)
print("attributed call ->", code)

# %%
class {cfg["class_name"]}:
    """Production-shaped wrapper for this family on bedrock-mantle."""

    def __init__(self, model={cfg["primary"]}, region=REGION, tier="default", project=None):
        self.model, self.region, self.tier, self.project = model, region, tier, project

    def chat(self, messages, *, max_tokens=512, tools=None, schema=None, effort=None):
        body = {{
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "service_tier": self.tier,
        }}
        if tools:
            body["tools"] = tools
        if effort:
            body["reasoning_effort"] = effort
        if schema:
            body["response_format"] = {{
                "type": "json_schema",
                "json_schema": {{"name": "out", "strict": True, "schema": schema}},
            }}
        headers = {{"OpenAI-Project": self.project}} if self.project else None
        # post() retries 429 + 5xx with exponential backoff and jitter.
        code, data = post(f"{{PREFIX}}/chat/completions", body,
                          region=self.region, headers=headers)
        if code != 200:
            raise RuntimeError(f"HTTP {{code}}: {{err(data)}}")
        return data

    def json(self, prompt, schema, **kw):
        data = self.chat([{{"role": "user", "content": prompt}}], schema=schema, **kw)
        return parse_json_lenient(data["choices"][0]["message"]["content"])


bot = {cfg["class_name"]}(tier="flex", project=project_id)
out = bot.json(
    "Name the largest ocean and its average depth in metres.",
    {{"type": "object",
      "properties": {{"ocean": {{"type": "string"}}, "avg_depth_m": {{"type": "number"}}}},
      "required": ["ocean", "avg_depth_m"], "additionalProperties": False}},
)
print("structured result:", out)

# %%
code, archived = post(f"/v1/organization/projects/{{project_id}}/archive", {{}}, region=REGION)
print("archived demo project:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — {cfg["title"]} on bedrock-mantle
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
# | Quotas | No RPM quota; most models have no published TPM — retry with backoff |
# | `reserved` tier | Rejected as a parameter; arranged via your account team |
# | CloudWatch | Metrics land in `AWS/BedrockMantle`, not `AWS/Bedrock` |
{cfg["extra_gotchas"]}#
# ## Where next
# - Same API shape: {cfg["siblings"]}
# - Different API shape: `../03-google-gemma/` (Responses),
#   `../02-anthropic-claude/` (Messages), `../01-openai-gpt/` (web search, caching)
# - Shared mechanics: `../00-foundations/`
''')
    return "".join(p)


FAMILIES = [
    dict(
        stem="04-qwen__01-qwen3-core-and-tools",
        title="Qwen3",
        blurb=("Alibaba's Qwen3 family on the `bedrock-mantle` endpoint — a wide "
               "range of sizes plus dedicated coder and vision-language variants. "
               "This notebook covers the general-purpose models end to end; "
               "`02-qwen3-coder-and-vision.ipynb` covers the specialists."),
        models_table=[
            ("qwen.qwen3-32b", "Dense 32B — the reliable default; also fine-tunable on mantle"),
            ("qwen.qwen3-235b-a22b-2507", "MoE 235B total / 22B active — highest quality"),
            ("qwen.qwen3-next-80b-a3b-instruct", "MoE 80B total / 3B active — cost-efficient"),
        ],
        region="us-east-1",
        model_consts=('DENSE_32B = "qwen.qwen3-32b"\n'
                      'MOE_235B = "qwen.qwen3-235b-a22b-2507"\n'
                      'MOE_80B = "qwen.qwen3-next-80b-a3b-instruct"\n\n'),
        model_list_expr="[DENSE_32B, MOE_235B, MOE_80B]",
        primary="DENSE_32B",
        first_prompt="Explain the difference between a dense and a sparse (MoE) transformer, in two sentences.",
        stream_prompt="List four practical uses of long-context language models.",
        turn1="What is quantisation in the context of LLM serving?",
        turn2="Which quantisation format is most common for GPU inference?",
        reasoning_prompt="A shop sells pens at 3 for $2. How much for 17 pens? Show your working.",
        reasoning_extra=(
            '# Qwen3 models often emit an explicit <think>...</think> block in the CONTENT\n'
            '# when reasoning hard. Strip it before showing the answer to a user.\n'
            'import re\n\n'
            'code, data = post(\n'
            '    f"{PREFIX}/chat/completions",\n'
            '    {"model": DENSE_32B,\n'
            '     "messages": [{"role": "user", "content": "What is 17 * 23? Think step by step."}],\n'
            '     "max_tokens": 500, "reasoning_effort": "high"},\n'
            '    region=REGION,\n'
            ')\n'
            'raw = data["choices"][0]["message"]["content"] or ""\n'
            'cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()\n'
            'print("has <think> block:", "<think>" in raw)\n'
            'print("raw first 120 chars:", repr(raw[:120]))\n'
            'print("cleaned answer   :", repr(cleaned[:160]))\n'),
        compare_blurb=("Same prompt across three architectures: a dense 32B, a large "
                       "MoE, and a cost-efficient MoE. Watch latency against quality."),
        compare_task="In one sentence, why does a mixture-of-experts model cost less to serve than a dense model of the same total size?",
        compare_models="[DENSE_32B, MOE_80B, MOE_235B]",
        project_name="qwen3-samples",
        project_tag="Qwen3Demo",
        class_name="Qwen3Client",
        extra_gotchas=("# | `<think>` blocks | Qwen3 can emit reasoning inline in `content` — strip it |\n"
                       "# | Fine-tuning | `qwen3-32b` is one of only two mantle-fine-tunable models (us-west-2) |\n"),
        tools_section=TOOLS_SECTION,
        system_msg_note="",
        system_msg_line='    {"role": "system", "content": "You are concise. Two sentences maximum."},\n',
        siblings="`../05-deepseek/`, `../06-zai-glm/`, `../09-minimax/`",
    ),
    dict(
        stem="05-deepseek__01-deepseek-v3-reasoning",
        title="DeepSeek V3",
        blurb=("DeepSeek's V3 family on the `bedrock-mantle` endpoint — strong "
               "reasoning and mathematics at open-weight pricing. Two generations "
               "are available, and this notebook interleaves them so you can see "
               "what changed."),
        models_table=[
            ("deepseek.v3.2", "Latest generation — the default choice"),
            ("deepseek.v3.1", "Previous generation — useful for regression comparison"),
        ],
        region="us-east-1",
        model_consts=('V32 = "deepseek.v3.2"\nV31 = "deepseek.v3.1"\n\n'),
        model_list_expr="[V32, V31]",
        primary="V32",
        first_prompt="Explain what chain-of-thought prompting does, in two sentences.",
        stream_prompt="List four techniques for making language-model maths more reliable.",
        turn1="What is the difference between a proof and a heuristic argument?",
        turn2="Give an example of each in one line.",
        reasoning_prompt="If a bat and ball cost $1.10 together and the bat costs $1 more than the ball, how much is the ball? Show your working.",
        reasoning_extra=(
            '# DeepSeek is reasoning-forward, so effort has a large effect on both\n'
            '# token spend and answer quality. Compare a genuinely tricky question.\n'
            'PUZZLE = ("Three people check into a hotel room costing $30 and pay $10 each. "\n'
            '          "The manager realises it should cost $25 and sends $5 back via the "\n'
            '          "bellhop, who keeps $2 and returns $1 each. Now each paid $9 = $27, "\n'
            '          "plus the bellhop\'s $2 = $29. Where is the missing dollar?")\n\n'
            'for effort in ("low", "high"):\n'
            '    code, data = post(\n'
            '        f"{PREFIX}/chat/completions",\n'
            '        {"model": V32, "messages": [{"role": "user", "content": PUZZLE}],\n'
            '         "max_tokens": 600, "reasoning_effort": effort},\n'
            '        region=REGION,\n'
            '    )\n'
            '    text = (data["choices"][0]["message"]["content"] or "").strip()\n'
            '    print(f"--- effort={effort} | completion_tokens="\n'
            '          f"{data[\'usage\'][\'completion_tokens\']} ---")\n'
            '    print(text[:320], "\\n")\n'),
        compare_blurb="Same question on both generations, to see the delta directly.",
        compare_task="In one sentence, why is greedy decoding sometimes worse than sampling?",
        compare_models="[V32, V31]",
        project_name="deepseek-samples",
        project_tag="DeepSeekDemo",
        class_name="DeepSeekClient",
        extra_gotchas=("# | Reasoning-forward | High effort can spend many tokens — cap `max_tokens` |\n"
                       "# | Region | Absent from eu-central-1 |\n"),
        tools_section=TOOLS_SECTION,
        system_msg_note="",
        system_msg_line='    {"role": "system", "content": "You are concise. Two sentences maximum."},\n',
        siblings="`../04-qwen/`, `../06-zai-glm/`, `../08-moonshot-kimi/`",
    ),
    dict(
        stem="06-zai-glm__01-glm-family",
        title="Z.AI GLM",
        blurb=("Z.AI's GLM family on the `bedrock-mantle` endpoint, spanning a "
               "flagship model and a low-latency Flash variant. The size spread makes "
               "this a good family for demonstrating a cost-aware escalation pattern."),
        models_table=[
            ("zai.glm-5", "Flagship — highest quality"),
            ("zai.glm-4.7", "Previous flagship"),
            ("zai.glm-4.7-flash", "Low latency / low cost; also the widest Region coverage"),
            ("zai.glm-4.6", "Earlier generation"),
        ],
        region="us-east-1",
        model_consts=('GLM5 = "zai.glm-5"\nGLM47 = "zai.glm-4.7"\n'
                      'FLASH = "zai.glm-4.7-flash"\nGLM46 = "zai.glm-4.6"\n\n'),
        model_list_expr="[GLM5, GLM47, FLASH, GLM46]",
        primary="GLM5",
        first_prompt="Explain what an inference engine does, in two sentences.",
        stream_prompt="List four ways to reduce language-model serving cost.",
        turn1="What is speculative decoding?",
        turn2="What is the main risk of using it?",
        reasoning_prompt="A tank fills in 6 hours with tap A and 4 hours with tap B. How long with both open? Show your working.",
        reasoning_extra=(
            '# A cost-aware escalation pattern: try Flash first, escalate only if the\n'
            '# cheap model fails a validation gate. This is the practical reason to\n'
            '# care that a family spans several price points.\n'
            'SCHEMA = {"type": "object",\n'
            '          "properties": {"answer_hours": {"type": "number"},\n'
            '                         "confident": {"type": "boolean"}},\n'
            '          "required": ["answer_hours", "confident"],\n'
            '          "additionalProperties": False}\n\n'
            'def attempt(model):\n'
            '    code, data = post(\n'
            '        f"{PREFIX}/chat/completions",\n'
            '        {"model": model,\n'
            '         "messages": [{"role": "user",\n'
            '                       "content": "Tap A fills a tank in 6h, tap B in 4h. "\n'
            '                                  "How many hours with both open?"}],\n'
            '         "max_tokens": 300,\n'
            '         "response_format": {"type": "json_schema",\n'
            '                             "json_schema": {"name": "ans", "strict": True,\n'
            '                                             "schema": SCHEMA}}},\n'
            '        region=REGION,\n'
            '    )\n'
            '    if code != 200:\n'
            '        return None\n'
            '    try:\n'
            '        return parse_json_lenient(data["choices"][0]["message"]["content"])\n'
            '    except ValueError:\n'
            '        return None\n\n'
            'result = attempt(FLASH)\n'
            'print("flash  ->", result)\n'
            'if not result or not result.get("confident"):\n'
            '    print("escalating to the flagship...")\n'
            '    result = attempt(GLM5)\n'
            '    print("glm-5  ->", result)\n'
            'print("\\nfinal:", result)\n'),
        compare_blurb=("Flash versus the flagships on one prompt — the latency and "
                       "token gap is the whole argument for tiered routing."),
        compare_task="In one sentence, what is KV-cache reuse and why does it help?",
        compare_models="[FLASH, GLM46, GLM47, GLM5]",
        project_name="glm-samples",
        project_tag="GLMDemo",
        class_name="GLMClient",
        extra_gotchas=("# | Flash vs flagship | Large latency/cost gap — good escalation candidate |\n"
                       "# | Region | Only `glm-4.6` and `glm-4.7-flash` reach eu-central-1 |\n"),
        tools_section=TOOLS_SECTION,
        system_msg_note="",
        system_msg_line='    {"role": "system", "content": "You are concise. Two sentences maximum."},\n',
        siblings="`../04-qwen/`, `../05-deepseek/`, `../09-minimax/`",
    ),
    dict(
        stem="08-moonshot-kimi__01-kimi-k2",
        title="Moonshot AI Kimi K2",
        blurb=("Moonshot AI's Kimi K2 models on the `bedrock-mantle` endpoint. Both "
               "variants are built for long-context and agentic work, and one is "
               "explicitly a thinking model."),
        models_table=[
            ("moonshotai.kimi-k2.5", "Latest generation — general purpose"),
            ("moonshotai.kimi-k2-thinking", "Reasoning-specialised variant"),
        ],
        region="us-east-1",
        model_consts=('K25 = "moonshotai.kimi-k2.5"\nTHINKING = "moonshotai.kimi-k2-thinking"\n\n'),
        model_list_expr="[K25, THINKING]",
        primary="K25",
        first_prompt="Explain why long context windows matter for agents, in two sentences.",
        stream_prompt="List four failure modes of long-running AI agents.",
        turn1="What is a tool-use loop?",
        turn2="What usually causes one to run away?",
        reasoning_prompt="I have 12 socks: 6 black, 4 blue, 2 grey. Drawing blind, how many must I take to guarantee a matching pair? Explain.",
        reasoning_extra=(
            '# The two variants split the work differently: k2-thinking is tuned for\n'
            '# reasoning, k2.5 for general use. Same prompt, both models, compare spend.\n'
            'RIDDLE = ("A rope ladder hangs over the side of a ship with rungs 30cm apart. "\n'
            '          "The tide rises 15cm per hour. After 3 hours, how many rungs are "\n'
            '          "underwater if 2 were underwater at the start? Explain.")\n\n'
            'for model in (K25, THINKING):\n'
            '    code, data = post(\n'
            '        f"{PREFIX}/chat/completions",\n'
            '        {"model": model, "messages": [{"role": "user", "content": RIDDLE}],\n'
            '         "max_tokens": 600, "reasoning_effort": "high"},\n'
            '        region=REGION,\n'
            '    )\n'
            '    if code != 200:\n'
            '        print(f"{model}: HTTP {code} {err(data)[:60]}")\n'
            '        continue\n'
            '    text = (data["choices"][0]["message"]["content"] or "").strip()\n'
            '    print(f"--- {model} | completion_tokens="\n'
            '          f"{data[\'usage\'][\'completion_tokens\']} ---")\n'
            '    print(text[:300], "\\n")\n'),
        compare_blurb="General-purpose versus reasoning-specialised, on one prompt.",
        compare_task="In one sentence, why do agents need a step budget?",
        compare_models="[K25, THINKING]",
        project_name="kimi-samples",
        project_tag="KimiDemo",
        class_name="KimiClient",
        extra_gotchas=("# | Thinking variant | `kimi-k2-thinking` spends more tokens by design |\n"
                       "# | Region | Absent from eu-central-1 |\n"),
        tools_section=TOOLS_SECTION,
        system_msg_note="",
        system_msg_line='    {"role": "system", "content": "You are concise. Two sentences maximum."},\n',
        siblings="`../05-deepseek/`, `../09-minimax/`, `../04-qwen/`",
    ),
    dict(
        stem="09-minimax__01-minimax-m2",
        title="MiniMax M2",
        blurb=("MiniMax's M2 line on the `bedrock-mantle` endpoint. Three "
               "generations are live simultaneously, which makes this the best "
               "family in the collection for demonstrating a version-migration test."),
        models_table=[
            ("minimax.minimax-m2.5", "Latest generation"),
            ("minimax.minimax-m2.1", "Intermediate generation"),
            ("minimax.minimax-m2", "Original release"),
        ],
        region="us-east-1",
        model_consts=('M25 = "minimax.minimax-m2.5"\nM21 = "minimax.minimax-m2.1"\n'
                      'M2 = "minimax.minimax-m2"\n\n'),
        model_list_expr="[M25, M21, M2]",
        primary="M25",
        first_prompt="Explain what makes a model good at tool use, in two sentences.",
        stream_prompt="List four signs that an agent's tool schema is badly designed.",
        turn1="Why do agents benefit from typed tool schemas?",
        turn2="What happens when the schema is too loose?",
        reasoning_prompt="A car travels 60km at 30km/h then 60km at 60km/h. What is the average speed? Show your working.",
        reasoning_extra=(
            '# Version migration: run one fixed prompt + schema across all three\n'
            '# generations and diff the results. This is the check to run before\n'
            '# switching a production model ID.\n'
            'MIGRATION_SCHEMA = {\n'
            '    "type": "object",\n'
            '    "properties": {"average_speed_kmh": {"type": "number"},\n'
            '                   "method": {"type": "string"}},\n'
            '    "required": ["average_speed_kmh", "method"],\n'
            '    "additionalProperties": False,\n'
            '}\n'
            'PROMPT = ("A car drives 60km at 30km/h then 60km at 60km/h. "\n'
            '          "What is the average speed over the whole trip?")\n\n'
            'print(f"{\'model\':28} {\'answer\':>10}  method")\n'
            'print("-" * 78)\n'
            'for model in (M2, M21, M25):\n'
            '    code, data = post(\n'
            '        f"{PREFIX}/chat/completions",\n'
            '        {"model": model, "messages": [{"role": "user", "content": PROMPT}],\n'
            '         "max_tokens": 400,\n'
            '         "response_format": {"type": "json_schema",\n'
            '                             "json_schema": {"name": "speed", "strict": True,\n'
            '                                             "schema": MIGRATION_SCHEMA}}},\n'
            '        region=REGION,\n'
            '    )\n'
            '    if code != 200:\n'
            '        print(f"{model:28} HTTP {code} {err(data)[:40]}")\n'
            '        continue\n'
            '    try:\n'
            '        got = parse_json_lenient(data["choices"][0]["message"]["content"])\n'
            '        print(f"{model:28} {got.get(\'average_speed_kmh\')!s:>10}  "\n'
            '              f"{str(got.get(\'method\'))[:40]}")\n'
            '    except ValueError as exc:\n'
            '        print(f"{model:28} unparseable: {exc}")\n'
            'print("\\n(The correct answer is 40 km/h — harmonic, not arithmetic, mean.)")\n'),
        compare_blurb="All three generations on one prompt — the migration diff.",
        compare_task="In one sentence, why is a harmonic mean the right average for speeds?",
        compare_models="[M2, M21, M25]",
        project_name="minimax-samples",
        project_tag="MiniMaxDemo",
        class_name="MiniMaxClient",
        extra_gotchas=("# | Three live generations | Pin an exact model ID in production |\n"
                       "# | Agentic focus | M2 is tuned for tool use — lean on typed schemas |\n"),
        tools_section=TOOLS_SECTION,
        system_msg_note="",
        system_msg_line='    {"role": "system", "content": "You are concise. Two sentences maximum."},\n',
        siblings="`../04-qwen/`, `../06-zai-glm/`, `../08-moonshot-kimi/`",
    ),
    dict(
        stem="10-nvidia-nemotron__01-nemotron-nano-and-super",
        title="NVIDIA Nemotron",
        blurb=("NVIDIA's Nemotron family on the `bedrock-mantle` endpoint — a set of "
               "small 'Nano' models plus a large 'Super', which together give a clean "
               "cost/quality curve to reason about. One Nano variant is also "
               "vision-language capable."),
        models_table=[
            ("nvidia.nemotron-super-3-120b", "Largest — highest quality"),
            ("nvidia.nemotron-nano-3-30b", "Mid-size Nano"),
            ("nvidia.nemotron-nano-12b-v2", "Small, vision-language capable"),
            ("nvidia.nemotron-nano-9b-v2", "Smallest — lowest latency"),
        ],
        region="us-east-1",
        model_consts=('SUPER = "nvidia.nemotron-super-3-120b"\n'
                      'NANO30 = "nvidia.nemotron-nano-3-30b"\n'
                      'NANO12 = "nvidia.nemotron-nano-12b-v2"\n'
                      'NANO9 = "nvidia.nemotron-nano-9b-v2"\n\n'),
        model_list_expr="[SUPER, NANO30, NANO12, NANO9]",
        primary="SUPER",
        first_prompt="Explain why smaller models are useful at the edge, in two sentences.",
        stream_prompt="List four trade-offs when choosing a smaller language model.",
        turn1="What is knowledge distillation?",
        turn2="What is usually lost in the process?",
        reasoning_prompt="A rectangle's perimeter is 24cm and its area is 35cm². Find its sides. Show your working.",
        reasoning_extra=(
            '# nemotron-nano-12b-v2 is vision-language capable. Chat Completions takes\n'
            '# images as `image_url` content blocks (base64 data URL or s3:// URI —\n'
            '# arbitrary https:// URLs are not supported).\n'
            'import base64, struct, zlib\n\n'
            'def make_png(width, height, rgb):\n'
            '    """Solid-colour PNG, no third-party dependencies."""\n'
            '    raw = b"".join(b"\\x00" + bytes(rgb) * width for _ in range(height))\n'
            '    def chunk(tag, payload):\n'
            '        body = tag + payload\n'
            '        return (struct.pack(">I", len(payload)) + body\n'
            '                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))\n'
            '    return (b"\\x89PNG\\r\\n\\x1a\\n"\n'
            '            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))\n'
            '            + chunk(b"IDAT", zlib.compress(raw))\n'
            '            + chunk(b"IEND", b""))\n\n'
            'blue = "data:image/png;base64," + base64.b64encode(\n'
            '    make_png(64, 64, (30, 60, 200))).decode()\n\n'
            'code, data = post(\n'
            '    f"{PREFIX}/chat/completions",\n'
            '    {"model": NANO12, "max_tokens": 40,\n'
            '     "messages": [{"role": "user", "content": [\n'
            '         {"type": "image_url", "image_url": {"url": blue}},\n'
            '         {"type": "text", "text": "What colour is this image? One word."}]}]},\n'
            '    region=REGION,\n'
            ')\n'
            'if code == 200:\n'
            '    print("nano-12b-v2 sees:",\n'
            '          repr((data["choices"][0]["message"]["content"] or "").strip()))\n'
            'else:\n'
            '    print(f"HTTP {code}: {err(data)[:90]}")\n'),
        compare_blurb=("Nano through Super on one prompt — this is the cost/quality "
                       "curve you would use to pick a model for an edge-ish workload."),
        compare_task="In one sentence, what is the main benefit of a smaller model in production?",
        compare_models="[NANO9, NANO12, NANO30, SUPER]",
        project_name="nemotron-samples",
        project_tag="NemotronDemo",
        class_name="NemotronClient",
        extra_gotchas=("# | Vision | Only `nemotron-nano-12b-v2` takes images in this family |\n"
                       "# | Size spread | Nano models trade quality for latency — validate outputs |\n"),
        tools_section=TOOLS_SECTION,
        system_msg_note="",
        system_msg_line='    {"role": "system", "content": "You are concise. Two sentences maximum."},\n',
        siblings="`../04-qwen/`, `../07-mistral/`, `../05-deepseek/`",
    ),
    dict(
        stem="12-writer-palmyra__01-palmyra-vision",
        title="Writer Palmyra Vision",
        blurb=("Writer's Palmyra Vision model on the `bedrock-mantle` endpoint. This "
               "is a deliberately instructive case: it is vision-capable but **does "
               "not support tool calling**, so it shows both a specialised strength "
               "and how to work around a real limitation."),
        models_table=[
            ("writer.palmyra-vision-7b", "Vision-language, 7B. Tool calling NOT supported"),
        ],
        region="us-east-1",
        model_consts=('VISION = "writer.palmyra-vision-7b"\n\n'),
        model_list_expr="[VISION]",
        primary="VISION",
        first_prompt="Explain what a vision-language model does, in two sentences.",
        stream_prompt="List four business uses for document image understanding.",
        turn1="What is OCR?",
        turn2="How does a vision-language model differ from plain OCR?",
        reasoning_prompt="If a chart shows sales of 10, 20 and 60 for three months, what is the growth rate month over month? Show your working.",
        reasoning_extra=(
            '# THE LIMITATION, demonstrated rather than asserted: Palmyra Vision\n'
            '# rejects tool definitions. Its model card lists "Client-side tool\n'
            '# calling: Not Supported", and the API agrees.\n'
            'probe_tool = [{"type": "function", "function": {\n'
            '    "name": "noop", "description": "does nothing",\n'
            '    "parameters": {"type": "object",\n'
            '                   "properties": {"x": {"type": "string"}},\n'
            '                   "required": ["x"]}}}]\n\n'
            'code, data = post(\n'
            '    f"{PREFIX}/chat/completions",\n'
            '    {"model": VISION, "messages": [{"role": "user", "content": "Call noop."}],\n'
            '     "max_tokens": 64, "tools": probe_tool},\n'
            '    region=REGION,\n'
            ')\n'
            'print(f"tools on palmyra-vision -> HTTP {code}")\n'
            'print("message:", err(data)[:150])\n'
            'print("\\n=> No forced-tool trick here. Use response_format instead (see below).")\n'),
        compare_blurb="Only one model in this family, so we compare prompting strategies instead of models.",
        compare_task="In one sentence, when is a specialised vision model preferable to a general multimodal one?",
        compare_models="[VISION]",
        project_name="palmyra-samples",
        project_tag="PalmyraDemo",
        class_name="PalmyraClient",
        extra_gotchas=("# | **Tool calling** | **Rejected with 400** — matches the model card |\n"
                       "# | Structured output | `response_format` works; use it instead of forced tools |\n"),
        tools_section=NO_TOOLS_SECTION,
        system_msg_note=("# NOTE: this model requires strictly alternating user/assistant roles\n"
                 "# and rejects a leading `system` message with a 400. Put any instruction\n"
                 "# into the user turn instead.\n"),
        system_msg_line="",
        siblings="`../04-qwen/` (qwen3-vl), `../10-nvidia-nemotron/` (nano-12b-v2)",
    ),
]


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    for cfg in FAMILIES:
        path = SRC / f"{cfg['stem']}.py"
        text = build(cfg).replace('__PRIMARY__', cfg['primary'])
        path.write_text(text)
        print("wrote", path.name)


if __name__ == "__main__":
    main()
