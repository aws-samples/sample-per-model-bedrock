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
# # Prompt caching on Amazon Bedrock Mantle — OpenAI GPT models
#
# Prompt caching cuts both latency and cost when a long, stable prefix repeats
# across requests — system instructions, tool definitions, a reference document.
# The cached prefix skips *prefill*, which is the part of inference that dominates
# time-to-first-token.
#
# **Two generations, two mechanisms:**
#
# | Models | Mechanism |
# |---|---|
# | `gpt-5.6-sol` / `-terra` / `-luna` | **Explicit** breakpoints you place yourself |
# | `gpt-5.5` and earlier (`gpt-5.4`) | **Automatic** prefix caching, no code change |
#
# ## Why bother
# - Cache **reads** are billed at a large discount versus uncached input.
# - Cache **writes** carry a small premium over an uncached input token on gpt-5.6,
#   and are free on gpt-5.5 and earlier.
# - Cached input tokens **do not count against your input-token-per-minute (TPM)
#   quota** — so caching buys throughput headroom as well as money.
#
# **Verify the rates before you model costs.** Cache read/write multipliers are
# per-model and change over time; the authoritative source is the [Amazon Bedrock
# pricing page](https://aws.amazon.com/bedrock/pricing/). This notebook measures
# *token counts* from the API's own `usage` fields — which is the part you can
# verify — and uses illustrative multipliers only to show the shape of the
# arithmetic.
#
# ## Self-contained, but see also
# - **Core Responses API for this family** → `01-responses-api-core.ipynb`
# - **Auth, the three URL paths** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Quotas and why cached tokens matter for them** →
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
from mantle import err, post, response_text

REGION = "us-east-1"
PREFIX = "/openai/v1"  # gpt-5.x path

SOL = "openai.gpt-5.6-sol"  # explicit breakpoints
GPT55 = "openai.gpt-5.5"  # automatic caching
GPT54 = "openai.gpt-5.4"  # automatic caching

print("endpoint:", f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}")

# %% [markdown]
# ## 1. Build a realistic cacheable prefix
#
# The minimum prefix length for a cache entry is **1,024 tokens**. Anything shorter
# is silently not cached — no error, just no benefit. So a toy prompt will not
# demonstrate anything; we need a real one.

# %%
REFERENCE_DOC = (
    """
INTERNAL PLATFORM HANDBOOK (excerpt)

1. Service tiers
Every inference request may specify a service tier. Priority is for
customer-facing latency-critical paths. Standard is the default for everyday
workloads. Flex is discounted and appropriate for evaluations, batch-shaped work,
and agent side-quests that tolerate queueing.

2. Retry policy
All clients must implement exponential backoff with jitter. HTTP 429 indicates
throttling; 5xx indicates capacity shedding. Both are transient and must be
retried. Other 4xx responses indicate a client defect and must not be retried.

3. Data handling
Requests must set an explicit data-retention posture. Workloads handling
regulated data must run with zero data retention. Conversation state must not be
stored server-side for those workloads; history is replayed by the client.

4. Observability
Metrics are published to a dedicated namespace. Only client-error metrics are
available; server errors must be tracked from client-side telemetry. Traces are
sampled at one percent in production and one hundred percent in staging.

5. Cost attribution
Every workload must run under its own project, tagged with application,
environment, owner, and cost centre. Untagged usage is charged to a shared pool
and reclaimed at the end of each quarter.

6. Capacity ramp
New workloads must ramp concurrency over minutes, not seconds. A cold start from
zero to peak concurrency will be shed. Steady-state traffic with a stable prefix
benefits from prompt caching and should be structured accordingly.
"""
    * 6
)  # repeat to comfortably clear the 1,024-token minimum

approx_tokens = len(REFERENCE_DOC) // 4
print(f"reference doc: {len(REFERENCE_DOC)} chars ≈ {approx_tokens} tokens")
print(f"clears the 1,024-token minimum: {approx_tokens > 1024}")

# %% [markdown]
# ## 2. Explicit cache breakpoints (gpt-5.6)
#
# Mark where the reusable prefix ends with
# `"prompt_cache_breakpoint": {"mode": "explicit"}` on a content block, and set
# `prompt_cache_options` on the request.
#
# `prompt_cache_options.mode`:
# - `implicit` (default) — an automatic breakpoint on the latest message, **plus**
#   any explicit ones you add.
# - `explicit` — only your breakpoints count. If you provide none, the request does
#   no caching and incurs no write charge.


# %%
def cached_call(
    question: str, model: str = SOL, mode: str = "explicit", ttl: str = "30m"
) -> dict:
    """One request whose stable prefix is the handbook, with a cache breakpoint."""
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": model,
            "max_output_tokens": 200,
            "prompt_cache_options": {"mode": mode, "ttl": ttl},
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": REFERENCE_DOC,
                            # Everything up to and including this block is cacheable.
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        },
                        # Anything after the breakpoint can vary freely without
                        # invalidating the cached prefix.
                        {"type": "input_text", "text": question},
                    ],
                }
            ],
            "store": False,
        },
        region=REGION,
    )
    if code != 200:
        raise RuntimeError(f"HTTP {code}: {err(data)}")
    usage = data.get("usage", {})
    details = usage.get("input_tokens_details", {})
    return {
        "text": response_text(data),
        "input_tokens": usage.get("input_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "cache_write_tokens": details.get("cache_write_tokens"),
    }


first = cached_call(
    "Per the handbook, which tier suits evaluations? Answer in one line."
)
print("call 1 (cold):")
print(f"   input_tokens      = {first['input_tokens']}")
print(f"   cached_tokens     = {first['cached_tokens']}")
print(f"   cache_write_tokens= {first['cache_write_tokens']}")
print(f"   answer            = {first['text'][:100]!r}")

# %% [markdown]
# The first call **writes** the cache: `cache_write_tokens` is large,
# `cached_tokens` is zero. Now repeat with the same prefix but a different question.

# %%
second = cached_call("Per the handbook, what must untagged usage expect? One line.")
print("call 2 (warm):")
print(f"   input_tokens      = {second['input_tokens']}")
print(f"   cached_tokens     = {second['cached_tokens']}")
print(f"   cache_write_tokens= {second['cache_write_tokens']}")
print(f"   answer            = {second['text'][:100]!r}")

print("\ninterpretation:")
if second["cached_tokens"]:
    pct = 100 * second["cached_tokens"] / max(second["input_tokens"], 1)
    print(
        f"   {second['cached_tokens']} of {second['input_tokens']} input tokens "
        f"({pct:.0f}%) were served from cache"
    )
    print("   cache_write_tokens == 0 means a full prefix match — maximum saving")

# %% [markdown]
# ## 3. Read the numbers as money and as quota
#
# The multipliers below are **illustrative placeholders**, not published prices —
# substitute the current values from the [Amazon Bedrock pricing
# page](https://aws.amazon.com/bedrock/pricing/) for your model before you use this
# to forecast spend. What the API *does* give you authoritatively is the token
# counts, which is what the cells above read.
#
# The shape of the result holds regardless of the exact rates: a cached prefix pays
# for itself after a small number of reuses, and every reuse after that is cheap.


# %%
# ILLUSTRATIVE multipliers, expressed relative to one uncached input token.
# These are NOT published prices. Replace them with the current per-model rates
# from https://aws.amazon.com/bedrock/pricing/ before forecasting real spend.
UNCACHED_UNIT = 1.00
CACHE_WRITE_UNIT = 1.25  # placeholder: a small premium over uncached
CACHE_READ_UNIT = 0.10  # placeholder: a large discount


def cost_units(input_tokens, cached, written):
    """Relative input cost in 'uncached token' units, using the placeholders above."""
    uncached = (input_tokens or 0) - (cached or 0) - (written or 0)
    return (
        max(uncached, 0) * UNCACHED_UNIT
        + (written or 0) * CACHE_WRITE_UNIT
        + (cached or 0) * CACHE_READ_UNIT
    )


baseline = (first["input_tokens"] or 0) * UNCACHED_UNIT  # if nothing were cached
call1 = cost_units(
    first["input_tokens"], first["cached_tokens"], first["cache_write_tokens"]
)
call2 = cost_units(
    second["input_tokens"], second["cached_tokens"], second["cache_write_tokens"]
)

print(f"{'scenario':34} {'relative input cost':>20}")
print("-" * 58)
print(f"{'no caching (per call)':34} {baseline:>20.0f}")
print(f"{'call 1 — cache write':34} {call1:>20.0f}")
print(f"{'call 2 — cache read':34} {call2:>20.0f}")
print(
    f"\nsteady-state saving per call: "
    f"{100 * (1 - call2 / max(baseline, 1)):.0f}% of input cost"
)
print("\nAnd cached tokens do NOT count against your input-TPM quota,")
print("so the same change also buys throughput headroom.")

# %% [markdown]
# ## 4. `implicit` vs `explicit` mode
#
# In `explicit` mode, no breakpoint means no caching at all — useful when you want
# full control and want to avoid automatic breakpoints consuming write slots.


# %%
def no_breakpoint_call(mode: str) -> dict:
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": SOL,
            "max_output_tokens": 120,
            "prompt_cache_options": {"mode": mode},
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": REFERENCE_DOC},
                        {
                            "type": "input_text",
                            "text": "Name the default tier. One word.",
                        },
                    ],
                }
            ],
            "store": False,
        },
        region=REGION,
    )
    details = (data.get("usage") or {}).get("input_tokens_details", {})
    return {
        "code": code,
        "cached": details.get("cached_tokens"),
        "written": details.get("cache_write_tokens"),
    }


for mode in ("explicit", "implicit"):
    r = no_breakpoint_call(mode)
    print(
        f"mode={mode:9} (no explicit breakpoint) -> "
        f"cached={r['cached']} written={r['written']}"
    )

# %% [markdown]
# `implicit` places an automatic breakpoint for you, so caching happens without
# any request changes. `explicit` with no breakpoint does nothing — deliberate,
# and the safer default for agent loops where you want to control write slots.

# %% [markdown]
# ## 5. Prefix order is everything
#
# The cache matches on an **exact prefix**. Static content must come first; anything
# that varies must come last. Put the variable part first and you get a cache miss
# on every call.


# %%
def ordered_call(static_first: bool) -> dict:
    question = f"Question {time.time():.0f}: name one retry rule. One line."
    blocks = (
        [
            {
                "type": "input_text",
                "text": REFERENCE_DOC,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
            {"type": "input_text", "text": question},
        ]
        if static_first
        else [
            {
                "type": "input_text",
                "text": question,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
            {"type": "input_text", "text": REFERENCE_DOC},
        ]
    )
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": SOL,
            "max_output_tokens": 120,
            "prompt_cache_options": {"mode": "explicit"},
            "input": [{"role": "user", "content": blocks}],
            "store": False,
        },
        region=REGION,
    )
    details = (data.get("usage") or {}).get("input_tokens_details", {})
    return details.get("cached_tokens") or 0


# Warm each arrangement, then measure the second call.
ordered_call(True)
warm_correct = ordered_call(True)
ordered_call(False)
warm_wrong = ordered_call(False)

print(f"static prefix first  -> cached_tokens on repeat = {warm_correct}")
print(f"variable text first  -> cached_tokens on repeat = {warm_wrong}")
print("\n=> Static content first, variable content last. Always.")

# %% [markdown]
# ## 6. Automatic caching on gpt-5.5 and earlier
#
# No parameters at all: the system caches eligible prefixes of ≥1,024 tokens by
# exact match, and there is **no cache-write fee** on these models.


# %%
def auto_cached_call(model: str, question: str) -> dict:
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": model,
            "max_output_tokens": 150,
            # No prompt_cache_options, no breakpoints — nothing to configure.
            "input": REFERENCE_DOC + "\n\n" + question,
            "store": False,
        },
        region=REGION,
    )
    usage = data.get("usage", {})
    details = usage.get("input_tokens_details", {})
    return {
        "code": code,
        "input": usage.get("input_tokens"),
        "cached": details.get("cached_tokens"),
        "written": details.get("cache_write_tokens"),
    }


for model in (GPT55, GPT54):
    cold = auto_cached_call(model, "Name the default tier. One word.")
    warm = auto_cached_call(model, "Name one observability rule. One line.")
    print(f"{model}")
    print(
        f"   call 1: input={cold['input']} cached={cold['cached']} "
        f"written={cold['written']}"
    )
    print(
        f"   call 2: input={warm['input']} cached={warm['cached']} "
        f"written={warm['written']}"
    )

# %% [markdown]
# ## 7. Does caching actually reduce latency?
#
# It should, because it skips prefill. Measure it rather than assuming.


# %%
def timed_call(warm_up: bool) -> float:
    if warm_up:
        cached_call("Warm the cache. One word.")
    started = time.perf_counter()
    cached_call(f"Unique question {time.time():.0f}. One line.")
    return time.perf_counter() - started


cold_latency = timed_call(warm_up=False)
warm_latency = timed_call(warm_up=True)
print(f"latency with a cold-ish cache : {cold_latency:.2f}s")
print(f"latency with a warm cache     : {warm_latency:.2f}s")
print("\nNote: single samples are noisy. The mechanism is sound (prefill is skipped),")
print("but measure over many calls before quoting a number.")

# %% [markdown]
# ## 8. TTL and eviction
#
# The default TTL is **30 minutes** on gpt-5.6, set via `prompt_cache_options.ttl`.
# That is long enough to cover the burst of calls a single agent run generates. A
# cache entry that is not read within its TTL expires.

# %%
for ttl in ("30m", "1h", "5m"):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": SOL,
            "max_output_tokens": 60,
            "prompt_cache_options": {"mode": "explicit", "ttl": ttl},
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": REFERENCE_DOC,
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        },
                        {"type": "input_text", "text": "Reply OK."},
                    ],
                }
            ],
            "store": False,
        },
        region=REGION,
    )
    print(f"  ttl={ttl:4} -> HTTP {code} {'' if code == 200 else err(data)[:70]}")

# %% [markdown]
# ## 9. The agentic pattern this is really for
#
# In an agent loop the system instructions and tool definitions never change while
# the conversation grows. Put those before the breakpoint and every subsequent turn
# reads them from cache.

# %%
AGENT_INSTRUCTIONS = (
    REFERENCE_DOC
    + """
You are a platform assistant. Follow the handbook exactly. Prefer the cheapest
tier that satisfies the requirement. Always state which handbook section supports
your answer.
"""
)

TOOL_CATALOGUE = json.dumps(
    [
        {
            "name": "get_tier",
            "description": "Look up the recommended service tier.",
            "parameters": {
                "type": "object",
                "properties": {"workload": {"type": "string"}},
                "required": ["workload"],
            },
        },
        {
            "name": "get_retry_policy",
            "description": "Return the retry policy for a status code.",
            "parameters": {
                "type": "object",
                "properties": {"status": {"type": "integer"}},
                "required": ["status"],
            },
        },
    ],
    indent=2,
)

turns = [
    "Which tier for a nightly evaluation job?",
    "And for a customer-facing chat endpoint?",
    "Should we retry an HTTP 400?",
]

print(f"{'turn':6} {'input':>8} {'cached':>8} {'written':>9}  answer")
print("-" * 92)
for i, question in enumerate(turns, 1):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": SOL,
            "max_output_tokens": 250,
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "input": [
                {
                    "role": "user",
                    "content": [
                        # STATIC: instructions + tool catalogue, cached once.
                        {"type": "input_text", "text": AGENT_INSTRUCTIONS},
                        {
                            "type": "input_text",
                            "text": TOOL_CATALOGUE,
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        },
                        # VARIABLE: this turn's question.
                        {"type": "input_text", "text": question},
                    ],
                }
            ],
            "store": False,
        },
        region=REGION,
    )
    usage = data.get("usage", {})
    details = usage.get("input_tokens_details", {})
    answer = " ".join(response_text(data).split())
    print(
        f"{i:>6} {usage.get('input_tokens', 0):>8} "
        f"{details.get('cached_tokens', 0):>8} "
        f"{details.get('cache_write_tokens', 0):>9}  {answer[:44]!r}"
    )

print("\nTurn 1 writes the prefix; turns 2 and 3 read it. The instruction block is")
print("paid for once, not once per turn.")

# %% [markdown]
# ## 10. Which models support caching at all?
#
# Sending `prompt_cache_options` to a model that does not support it is an error —
# so gate the parameter on the model, don't send it blindly.

# %%
candidates = [
    (SOL, PREFIX),
    (GPT55, PREFIX),
    (GPT54, PREFIX),
    ("openai.gpt-oss-120b", "/v1"),
    ("google.gemma-4-31b", "/openai/v1"),
]
print(f"{'model':28} {'prompt_cache_options':>22}")
print("-" * 54)
for model, prefix in candidates:
    code, data = post(
        f"{prefix}/responses",
        {
            "model": model,
            "input": "Reply OK",
            "max_output_tokens": 16,
            "prompt_cache_options": {"mode": "explicit"},
        },
        region=REGION,
    )
    print(f"{model:28} {('accepted' if code == 200 else str(code)):>22}")
    if code != 200:
        print(f"      {err(data)[:80]}")

# %% [markdown]
# ## 11. A caching client
#
# Production shape: model-aware caching, correct block order, usage reporting.


# %%
class CachingClient:
    """Responses client that caches a fixed prefix and reports cache efficiency."""

    EXPLICIT_MODELS = ("openai.gpt-5.6",)

    def __init__(self, model=SOL, region=REGION, static_prefix="", ttl="30m"):
        self.model, self.region, self.ttl = model, region, ttl
        self.static_prefix = static_prefix
        self.supports_explicit = model.startswith(self.EXPLICIT_MODELS)
        self.totals = {"input": 0, "cached": 0, "written": 0}

    def ask(self, question: str, max_output_tokens: int = 250) -> str:
        blocks = []
        if self.static_prefix:
            block = {"type": "input_text", "text": self.static_prefix}
            if self.supports_explicit:
                # Only gpt-5.6 understands explicit breakpoints; earlier models
                # cache automatically and would reject the parameter.
                block["prompt_cache_breakpoint"] = {"mode": "explicit"}
            blocks.append(block)
        blocks.append({"type": "input_text", "text": question})

        body = {
            "model": self.model,
            "max_output_tokens": max(16, max_output_tokens),
            "input": [{"role": "user", "content": blocks}],
            "store": False,
        }
        if self.supports_explicit:
            body["prompt_cache_options"] = {"mode": "explicit", "ttl": self.ttl}

        code, data = post(f"{PREFIX}/responses", body, region=self.region)
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        usage = data.get("usage", {})
        details = usage.get("input_tokens_details", {})
        self.totals["input"] += usage.get("input_tokens", 0)
        self.totals["cached"] += details.get("cached_tokens", 0) or 0
        self.totals["written"] += details.get("cache_write_tokens", 0) or 0
        return response_text(data)

    def report(self) -> str:
        t = self.totals
        hit_rate = 100 * t["cached"] / max(t["input"], 1)
        return (
            f"input={t['input']} cached={t['cached']} written={t['written']} "
            f"| cache hit rate {hit_rate:.0f}%"
        )


bot = CachingClient(model=SOL, static_prefix=AGENT_INSTRUCTIONS)
for question in (
    "Which tier for batch evals?",
    "Which tier for live chat?",
    "Do we retry 429?",
):
    answer = bot.ask(question, 200)
    print(f"Q: {question}\nA: {' '.join(answer.split())[:110]}\n")
print(bot.report())

# %% [markdown]
# ## Gotchas — prompt caching on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | 1,024-token minimum | Shorter prefixes are silently not cached — no error |
# | Prefix order | Static first, variable last; exact-prefix match |
# | `explicit` with no breakpoint | Does nothing at all (by design) |
# | Write cost | A small premium on gpt-5.6; free on gpt-5.5 and earlier. Check the pricing page |
# | Read discount | ~90% off uncached input |
# | Quota | Cached tokens **do not** count against input-TPM |
# | Model gating | `prompt_cache_options` is rejected by models that lack support |
# | TTL | 30 min default on gpt-5.6; entries expire if not read |
# | Latency claims | Measure over many calls; single samples are noisy |
#
# ## Next
# - `05-server-side-tools-and-fine-tuning.ipynb` — Lambda MCP tools,
#   notes/tasks, RFT (reinforcement fine-tuning)
# - `02-web-search-and-grounding.ipynb` · `03-tools-and-structured-output.ipynb`
# - Claude's different caching model:
#   `../02-anthropic-claude/02-thinking-tools-and-caching.ipynb`
