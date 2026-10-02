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
# # Tools and structured output — OpenAI GPT on Amazon Bedrock Mantle
#
# Client-side tool calling and strict JSON on the Responses API. This is the
# machinery behind most agent loops: define tools, let the model choose, execute,
# feed results back, repeat.
#
# **Models used here:** `openai.gpt-5.6-sol` (frontier, `/openai/v1`) and
# `openai.gpt-oss-120b` (open-weight, bare `/v1`).
#
# ## What this notebook covers
# - The Responses tool shape and the full call/result loop
# - `tool_choice`: auto, required, and forcing a specific function
# - Parallel tool calls
# - Strict JSON two ways — native `text.format` and the forced-tool trick
# - Robust parsing, and the failure modes that actually happen in production
#
# ## Self-contained, but see also
# - **Core Responses API for this family** → `01-responses-api-core.ipynb`
# - **Server-side tools (Lambda MCP (Model Context Protocol), notes/tasks)** →
#   `05-server-side-tools-and-fine-tuning.ipynb`
# - **Auth, the three URL paths** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
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

sys.path.insert(0, "../_shared")
from mantle import err, parse_json_lenient, post, response_text

REGION = "us-east-1"
SOL = "openai.gpt-5.6-sol"
OSS120 = "openai.gpt-oss-120b"

GPT5_PREFIX = "/openai/v1"  # gpt-5.x
OSS_PREFIX = "/v1"  # gpt-oss

from aws_bedrock_token_generator import provide_token
from openai import OpenAI

client = OpenAI(
    api_key=provide_token(region=REGION),
    base_url=f"https://bedrock-mantle.{REGION}.api.aws{GPT5_PREFIX}",
)
print("endpoint:", f"https://bedrock-mantle.{REGION}.api.aws{GPT5_PREFIX}")

# %% [markdown]
# ## 1. The Responses tool shape
#
# `name`, `description` and `parameters` sit at the **top level** of the tool
# object. Chat Completions nests them under `"function"` — a common source of
# confusion when porting code between the two APIs.

# %%
RESPONSES_SHAPE = {
    "type": "function",
    "name": "get_stock",  # top level
    "description": "Look up stock for a SKU.",
    "parameters": {
        "type": "object",
        "properties": {"sku": {"type": "string"}},
        "required": ["sku"],
    },
}

CHAT_COMPLETIONS_SHAPE = {
    "type": "function",
    "function": {  # nested
        "name": "get_stock",
        "description": "Look up stock for a SKU.",
        "parameters": {
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
        },
    },
}

print("Responses API tool:")
print(json.dumps(RESPONSES_SHAPE, indent=2))
print("\nChat Completions tool (same tool, different nesting):")
print(json.dumps(CHAT_COMPLETIONS_SHAPE, indent=2))

# %% [markdown]
# ## 2. A complete tool loop
#
# Four steps: send tools → model requests a call → you execute → send the result
# back. The `call_id` is what ties a result to its request.

# %%
INVENTORY = {"A-100": 42, "B-200": 0, "C-300": 7}
SHIPPING = {"standard": 4.99, "express": 12.50}


def get_stock(sku: str) -> dict:
    qty = INVENTORY.get(sku.upper(), 0)
    return {"sku": sku.upper(), "quantity": qty, "in_stock": qty > 0}


def get_shipping(method: str) -> dict:
    return {"method": method, "cost_usd": SHIPPING.get(method.lower())}


TOOLS = [
    {
        "type": "function",
        "name": "get_stock",
        "description": "Look up stock level for a SKU.",
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "SKU code, e.g. A-100"}
            },
            "required": ["sku"],
        },
    },
    {
        "type": "function",
        "name": "get_shipping",
        "description": "Look up the cost of a shipping method.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["standard", "express"]}
            },
            "required": ["method"],
        },
    },
]

DISPATCH = {"get_stock": get_stock, "get_shipping": get_shipping}


# %% [markdown]
# The loop. `max_rounds` bounds it: an agent with no bound can spin forever on a
# task it cannot finish (OWASP LLM10, Unbounded Consumption).


# %%
def run_tool_loop(
    question: str, model: str = SOL, max_rounds: int = 5, prefix: str = GPT5_PREFIX
) -> dict:
    """Full client-side tool loop. Returns the answer and a trace of calls."""
    conversation = [{"role": "user", "content": question}]
    trace = []
    for round_no in range(max_rounds):
        code, data = post(
            f"{prefix}/responses",
            {
                "model": model,
                "input": conversation,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_output_tokens": 600,
                "store": False,
            },
            region=REGION,
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")

        calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
        if not calls:
            return {
                "answer": response_text(data),
                "trace": trace,
                "rounds": round_no + 1,
            }

        for call in calls:
            # parse_json_lenient rather than json.loads: arguments occasionally
            # carry trailing characters after a valid object.
            args = parse_json_lenient(call["arguments"] or "{}")
            fn = DISPATCH.get(call["name"])
            result = fn(**args) if fn else {"error": f"unknown tool {call['name']}"}
            trace.append(
                {
                    "round": round_no + 1,
                    "tool": call["name"],
                    "args": args,
                    "result": result,
                }
            )
            # Echo the call, then its result. call_id links the two.
            conversation.append(
                {
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                }
            )
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(result),
                }
            )
    return {"answer": "(max rounds reached)", "trace": trace, "rounds": max_rounds}


result = run_tool_loop("Is SKU A-100 in stock, and what does express shipping cost?")
# %% [markdown]
# Run it and inspect the trace.

# %%
print("rounds:", result["rounds"])
for step in result["trace"]:
    print(
        f"  round {step['round']}: {step['tool']}({step['args']}) -> {step['result']}"
    )
print("\nanswer:", result["answer"][:300])

# %% [markdown]
# **Note what we did not append:** reasoning items. Send tool calls and their
# outputs, plus final answers — never the model's own reasoning traces. Replaying
# reasoning degrades later turns.

# %% [markdown]
# ## 3. `tool_choice`
#
# Three modes, in increasing order of compulsion.

# %%
QUESTION = "Do we have C-300 in stock?"
print(f"{'model':22} {'tool_choice':30} {'HTTP':>6} {'calls':>6}")
print("-" * 70)
for model, prefix in ((SOL, GPT5_PREFIX), (OSS120, OSS_PREFIX)):
    for choice in (
        "auto",
        "none",
        "required",
        {"type": "function", "name": "get_stock"},
    ):
        code, data = post(
            f"{prefix}/responses",
            {
                "model": model,
                "input": QUESTION,
                "tools": TOOLS,
                "tool_choice": choice,
                "max_output_tokens": 400,
                "store": False,
            },
            region=REGION,
        )
        calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
        label = choice if isinstance(choice, str) else "{'type':'function',...}"
        print(f"{model:22} {label:30} {code:>6} {len(calls):>6}")
        if code != 200:
            print(f"      {err(data)[:96]}")

# %% [markdown]
# ### `tool_choice` support is per model — and this one bites
#
# - **gpt-5.6** supports the full set: `auto`, `none`, `required`, and a named
#   function.
# - **gpt-oss** supports **only `auto`**. `none` and `required` are explicit 400s
#   ("Supported options: [\"auto\"]"), and the *named-function* form returns
#   **HTTP 200 while quietly ignoring the constraint* — the model answers in prose
#   with no tool call at all.
#
# That last case is the dangerous one: a 200 that silently did not do what you
# asked. Never assume a forced tool call succeeded; always check for the call.

# %%
# Forcing one specific function by name.
code, data = post(
    f"{GPT5_PREFIX}/responses",
    {
        "model": SOL,
        "input": "What about shipping?",
        "tools": TOOLS,
        "tool_choice": {"type": "function", "name": "get_shipping"},
        "max_output_tokens": 400,
        "store": False,
    },
    region=REGION,
)
calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
print(f"forced get_shipping -> HTTP {code} | calls={[c['name'] for c in calls]}")
if calls:
    print("arguments:", calls[0]["arguments"])

# %% [markdown]
# ## 4. Parallel tool calls
#
# Unlike some families (Gemma 4 issues one call per turn), gpt-5.6 accepts
# `parallel_tool_calls` and can request several at once.

# %%
code, data = post(
    f"{GPT5_PREFIX}/responses",
    {
        "model": SOL,
        "input": (
            "Check stock for A-100 and B-200, and get express "
            "shipping cost. All of it."
        ),
        "tools": TOOLS,
        "parallel_tool_calls": True,
        "max_output_tokens": 600,
        "store": False,
    },
    region=REGION,
)
calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
print(f"HTTP {code} | {len(calls)} call(s) requested in one turn:")
for call in calls:
    print(f"   {call['name']}({call['arguments']})")

# %%
# max_tool_calls caps how many the model may make in total.
code, data = post(
    f"{GPT5_PREFIX}/responses",
    {
        "model": SOL,
        "input": "Check stock for A-100, B-200 and C-300, plus both shipping methods.",
        "tools": TOOLS,
        "max_tool_calls": 2,
        "max_output_tokens": 600,
        "store": False,
    },
    region=REGION,
)
calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
print(f"max_tool_calls=2 -> {len(calls)} call(s) requested")

# %% [markdown]
# ## 5. Strict JSON via `text.format`
#
# The native route: give a JSON Schema and set `strict: True`.

# %%
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "topics": {"type": "array", "items": {"type": "string"}},
        "would_recommend": {"type": "boolean"},
    },
    "required": ["summary", "sentiment", "topics", "would_recommend"],
    "additionalProperties": False,
}

REVIEW = (
    "Battery life is superb and it charges fast, but the screen scratches "
    "far too easily and support took a week to reply."
)

code, data = post(
    f"{GPT5_PREFIX}/responses",
    {
        "model": SOL,
        "input": f"Analyse this review:\n{REVIEW}",
        "max_output_tokens": 700,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "review",
                "schema": REVIEW_SCHEMA,
                "strict": True,
            }
        },
        "store": False,
    },
    region=REGION,
)
raw = response_text(data)
print("HTTP", code, "| raw:", repr(raw[:120]))
parsed = parse_json_lenient(raw)
print(json.dumps(parsed, indent=2))
expected = set(REVIEW_SCHEMA["properties"])
if set(parsed) != expected:
    raise ValueError(
        f"schema mismatch: expected {sorted(expected)}, got {sorted(parsed)}"
    )
print("schema honoured exactly:", sorted(parsed))

# %% [markdown]
# ## 6. Strict JSON via a forced tool call
#
# The portable alternative: define a tool whose *input schema is your output
# schema*, then force it. This works on models that lack native structured output,
# and the result arrives already parsed at the tool layer.

# %%
EMIT_TOOL = {
    "type": "function",
    "name": "emit_review",
    "description": "Return the structured review analysis.",
    "parameters": REVIEW_SCHEMA,
}


def emit_structured(
    text: str, model: str = SOL, attempts: int = 3, prefix: str = GPT5_PREFIX
) -> dict:
    """Forced-tool structured output, with a retry.

    Forcing tool_choice is honoured *almost* always. Production code should handle
    the occasional turn that returns prose instead, rather than indexing [0].
    """
    for attempt in range(attempts):
        code, data = post(
            f"{prefix}/responses",
            {
                "model": model,
                "input": f"Analyse this review:\n{text}",
                "tools": [EMIT_TOOL],
                "tool_choice": {"type": "function", "name": "emit_review"},
                "max_output_tokens": 700,
                "store": False,
            },
            region=REGION,
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
        if calls:
            if attempt:
                print(f"(succeeded on attempt {attempt + 1})")
            return parse_json_lenient(calls[0]["arguments"])
        print(f"attempt {attempt + 1}: no tool call — retrying")
    raise RuntimeError("model would not emit the forced tool call")


print(json.dumps(emit_structured(REVIEW), indent=2))

# %% [markdown]
# ### Which route should you use?
#
# | | `text.format` | forced tool |
# |---|---|---|
# | Availability | gpt-5.x, gemma-4, grok | anywhere **constrained `tool_choice`** works |
# | Result arrives as | string to parse | arguments (already structured) |
# | Enum enforcement | yes | yes |
# | Extra round trip | no | no |
# | Fails when… | model lacks support | model only supports `tool_choice:"auto"` (e.g. gpt-oss) |
#
# Use `text.format` when the model supports it, and keep the forced-tool version as
# your portable fallback.

# %% [markdown]
# ## 7. The same, on gpt-oss (bare `/v1`)
#
# Everything above works on the open-weight models too — only the path changes.

# %%
result = run_tool_loop("Is B-200 in stock?", model=OSS120, prefix=OSS_PREFIX)
print("rounds:", result["rounds"])
for step in result["trace"]:
    print(f"   {step['tool']}({step['args']}) -> {step['result']}")
print("answer:", result["answer"][:200])

# %%
# On gpt-oss the forced-tool trick does NOT work: tool_choice only supports
# "auto", and the named form is accepted-but-ignored. Show that honestly.
try:
    print(
        json.dumps(
            emit_structured(REVIEW, model=OSS120, prefix=OSS_PREFIX, attempts=2),
            indent=2,
        )
    )
except RuntimeError as exc:
    print("forced tool on gpt-oss ->", exc)
    print("\nFallback for gpt-oss: ask for JSON in the prompt and parse leniently.")
    code, data = post(
        f"{OSS_PREFIX}/responses",
        {
            "model": OSS120,
            "input": (
                "Analyse this review and reply with ONLY a JSON object with keys "
                f"summary, sentiment, topics, would_recommend.\n{REVIEW}"
            ),
            "max_output_tokens": 700,
            "store": False,
        },
        region=REGION,
    )
    raw = response_text(data)
    print("HTTP", code, "| raw:", repr(raw[:120]))
    if raw.strip():
        print(json.dumps(parse_json_lenient(raw), indent=2))

# %% [markdown]
# ## 8. Failure modes worth handling
#
# Three things that actually happen in production.

# %%
# (a) The model invents arguments that don't match your schema. Validate.
code, data = post(
    f"{GPT5_PREFIX}/responses",
    {
        "model": SOL,
        "input": "Check stock for the blue widget please.",
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_output_tokens": 400,
        "store": False,
    },
    region=REGION,
)
calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
print("(a) invented arguments:")
for call in calls:
    args = parse_json_lenient(call["arguments"] or "{}")
    known = args.get("sku", "").upper() in INVENTORY
    print(f"    {call['name']}({args}) -> sku known to us? {known}")
print("    => always validate arguments against your own data before executing")


# %%
# (b) A tool that raises. Return a structured error rather than crashing the loop —
# the model can then apologise or try a different approach.
def risky_tool(sku: str) -> dict:
    if sku.upper() not in INVENTORY:
        raise KeyError(sku)
    return get_stock(sku)


def safe_call(name: str, args: dict) -> dict:
    try:
        return risky_tool(**args) if name == "get_stock" else DISPATCH[name](**args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "recoverable": True}


print("(b) tool error surfaced as data:")
print("   ", safe_call("get_stock", {"sku": "Z-999"}))

# %%
# (c) A tight token budget can truncate before any JSON is emitted — HTTP 200 with
# an empty string. Check before parsing.
print("(c) budget truncation:")
for budget in (16, 700):
    code, data = post(
        f"{GPT5_PREFIX}/responses",
        {
            "model": SOL,
            "input": f"Analyse: {REVIEW}",
            "max_output_tokens": budget,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "review",
                    "schema": REVIEW_SCHEMA,
                    "strict": True,
                }
            },
            "store": False,
        },
        region=REGION,
    )
    text = response_text(data)
    status = data.get("status")
    print(
        f"    max_output_tokens={budget:4} HTTP {code} status={status!s:12} "
        f"chars={len(text)}"
    )
print("    => check for content before calling any parser")

# %% [markdown]
# ## 9. An agent with validation and retries
#
# Putting the pieces together.


# %%
class ToolAgent:
    """Client-side tool agent with validation, error surfacing and round limits."""

    def __init__(
        self, model=SOL, region=REGION, tools=None, dispatch=None, project=None
    ):
        self.model, self.region, self.project = model, region, project
        self.prefix = GPT5_PREFIX if model.startswith("openai.gpt-5.") else OSS_PREFIX
        self.tools = tools or TOOLS
        self.dispatch = dispatch or DISPATCH

    def _execute(self, name, args):
        fn = self.dispatch.get(name)
        if not fn:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as exc:  # never let a tool kill the loop
            return {"error": f"{type(exc).__name__}: {exc}"}

    def run(self, question, max_rounds=6, max_output_tokens=600):
        conversation = [{"role": "user", "content": question}]
        headers = {"OpenAI-Project": self.project} if self.project else None
        for _ in range(max_rounds):
            code, data = post(
                f"{self.prefix}/responses",
                {
                    "model": self.model,
                    "input": conversation,
                    "tools": self.tools,
                    "tool_choice": "auto",
                    "max_output_tokens": max(16, max_output_tokens),
                    "store": False,
                },
                region=self.region,
                headers=headers,
            )
            if code != 200:
                raise RuntimeError(f"HTTP {code}: {err(data)}")
            calls = [
                i for i in data.get("output", []) if i.get("type") == "function_call"
            ]
            if not calls:
                return response_text(data)
            for call in calls:
                try:
                    args = parse_json_lenient(call["arguments"] or "{}")
                except ValueError:
                    args = {}
                output = self._execute(call["name"], args)
                conversation.append(
                    {
                        "type": "function_call",
                        "call_id": call["call_id"],
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                )
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps(output),
                    }
                )
        return "(max rounds reached)"


agent = ToolAgent()
print(
    agent.run(
        "Which of A-100, B-200 and C-300 are in stock, and what is the "
        "cheapest shipping?"
    )[:400]
)

# %% [markdown]
# ## Gotchas — tools and structured output
#
# | Gotcha | Detail |
# |---|---|
# | Tool shape differs by API | Responses = flat; Chat Completions = nested under `function` |
# | `call_id` | Must match between `function_call` and `function_call_output` |
# | Reasoning replay | Never append reasoning items to the conversation |
# | `tool_choice` support | **gpt-oss: `auto` only.** `none`/`required` 400; named form 200-but-ignored |
# | Forced `tool_choice` | Best-effort even where supported — handle the prose turn |
# | `json.loads` | Use lenient parsing; arguments can carry trailing characters |
# | Invented arguments | Validate against your own data before executing |
# | Tool exceptions | Return structured errors so the model can recover |
# | Tight budgets | Truncation gives HTTP 200 + empty output — check before parsing |
# | Parallel calls | Supported on gpt-5.6; Gemma 4 does one per turn |
#
# ## Next
# - `04-prompt-caching-and-cost.ipynb` — cache the tool definitions above
# - `05-server-side-tools-and-fine-tuning.ipynb` — let Bedrock run the tools
# - `02-web-search-and-grounding.ipynb` — a server-side tool you don't implement
