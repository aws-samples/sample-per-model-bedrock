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
# # Foundations 3 — Quotas, retries, service tiers, and latency
#
# Making mantle calls survive production. Mantle's throughput model is genuinely
# different from `bedrock-runtime`, and the differences bite if you assume they're
# the same.
#
# **Prerequisite:** `01-endpoints-auth-and-the-three-paths.ipynb`.
#
# ## What this notebook covers
# - The quota model: separate input/output TPM (tokens per minute), **no RPM quota**
# - Why retry with backoff is load-bearing here, not belt-and-braces
# - Service tiers (`default` / `flex` / `priority`) and what `reserved` does
# - Measuring TTFT (time-to-first-token) and throughput per tier, with real numbers
# - Ramp-not-spike guidance

# %%
import json
import statistics
import sys
import time

sys.path.insert(0, "../_shared")
from mantle import err, post, stream_lines, ttft

REGION = "us-east-1"
MODEL = "google.gemma-4-31b"  # in all four mantle Regions
PROMPT = "Explain what a distributed inference engine does, in three sentences."

# %% [markdown]
# ## 1. The quota model
#
# | | `bedrock-runtime` | `bedrock-mantle` |
# |---|---|---|
# | Token quota | one combined TPM | **separate** input TPM and output TPM |
# | Request quota | RPM enforced | **no RPM quota at all** |
# | Increase path | Service Quotas console | AWS Support case |
#
# How a request is admitted:
#
# 1. `input_tokens + max_tokens` is checked against **input TPM**. Over quota → HTTP
#    429.
# 2. Output tokens count against **output TPM** as they generate. Hit it mid-stream
#    and generation stops with a truncating finish reason.
# 3. After completion, the unused part of the `max_tokens` reservation is
#    replenished to your quota.
#
# Two consequences that matter in practice:
#
# - **Setting `max_tokens` far higher than you need reserves quota you won't use**,
#   throttling yourself under load.
# - **Cached tokens are exempt** from the input-TPM quota, so prompt caching buys
#   headroom as well as latency.

# %%
import boto3

sq = boto3.client("service-quotas", region_name=REGION)
paginator = sq.get_paginator("list_service_quotas")
mantle_quotas = []
for page in paginator.paginate(ServiceCode="bedrock"):
    for q in page["Quotas"]:
        if "mantle" in q["QuotaName"].lower():
            mantle_quotas.append((q["QuotaName"], q.get("Value")))

print(f"{len(mantle_quotas)} 'Bedrock Mantle' quotas visible in Service Quotas\n")
for name, value in sorted(mantle_quotas)[:12]:
    print(f"  {value!s:>14}  {name}")
if not mantle_quotas:
    print("  (none surfaced for this account)")

# %% [markdown]
# **Expect this list to be short or empty.** Only a handful of models have
# published per-account TPM quotas today (Claude Opus 4.7 is the documented one at
# 20M input / 4M output TPM). Everything else — including all the open-weight
# families — is governed by *internal service capacity* that is not exposed in
# Service Quotas.
#
# That is why you cannot "just request a limit increase" for most models, and why
# the retry loop below is mandatory rather than optional.

# %% [markdown]
# ## 2. Retry with exponential backoff
#
# Treat **429** (throttle) and **5xx** (capacity shedding) as transient. Other 4xx
# are your bug — fail fast on those.
#
# The `post()` helper in `_shared/mantle.py` already implements this; here is the
# logic laid bare so you can port it.

# %%
import random


import urllib.error
import urllib.request


def open_https(req, timeout: int):
    """urlopen restricted to HTTPS.

    urllib also honours file://, ftp:// and data:// . These URLs are all built
    from literals, but a client that ever takes a URL from data would let those
    schemes read local files, so the guard belongs in the helper (CWE-22).
    """
    if not req.full_url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS URL: {req.full_url[:60]}")
    # nosemgrep: dynamic-urllib-use-detected - scheme verified https above
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310  # noqa: S310


from aws_bedrock_token_generator import provide_token

TRANSIENT = {429, 500, 502, 503, 504}


def call_with_backoff(path: str, body: dict, region=REGION, attempts=5, verbose=True):
    """Retry transient failures; re-mint the token and re-sign on every attempt."""
    url = f"https://bedrock-mantle.{region}.api.aws{path}"
    for attempt in range(attempts):
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {provide_token(region=region)}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with open_https(req, timeout=180) as resp:
                if verbose and attempt:
                    print(f"    succeeded on attempt {attempt + 1}")
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in TRANSIENT and attempt < attempts - 1:
                # Jitter spreads retries; not a security decision.
                delay = min(2**attempt, 16) + random.random()  # nosec B311
                if verbose:
                    print(f"    HTTP {e.code} -> retrying in {delay:.1f}s")
                time.sleep(delay)
                continue
            return e.code, json.loads(e.read() or b"{}")
    return -1, {"error": {"message": "retries exhausted"}}


status, data = call_with_backoff(
    "/openai/v1/responses",
    {"model": MODEL, "input": "Reply with exactly: OK", "max_output_tokens": 16},
)
print("result:", status, json.dumps(data.get("output_text", ""))[:40])

# %% [markdown]
# ### Fail fast on real errors
#
# A permanent 400 must not be retried five times — that just wastes 30 seconds
# before surfacing the same bug.

# %%
start = time.perf_counter()
status, data = call_with_backoff(
    "/openai/v1/responses",
    {
        "model": MODEL,
        "input": "hi",
        "max_output_tokens": 16,
        "top_p": 0.95,
    },  # invalid here
    verbose=False,
)
took = time.perf_counter() - start
print(f"HTTP {status} after {took:.2f}s (no retries — correct)")
print("message:", err(data))

# %% [markdown]
# ## 3. Service tiers
#
# Set `service_tier` to trade latency against cost. Documented tiers are
# `priority`, `default` (Standard), `flex`, and `reserved` — but only some are
# accepted on this endpoint. Probe rather than trust:

# %%
for tier in ("auto", "default", "flex", "priority", "reserved"):
    code, data = post(
        "/openai/v1/responses",
        {
            "model": MODEL,
            "input": "Reply OK",
            "max_output_tokens": 16,
            "service_tier": tier,
        },
        region=REGION,
    )
    resolved = data.get("service_tier") if code == 200 else "-"
    detail = "" if code == 200 else err(data)[:70]
    print(f"  {tier:9} -> {code}  resolved={resolved!s:9} {detail}")

# %% [markdown]
# So on `bedrock-mantle`: `auto`, `default`, `flex`, `priority` all work, and
# **`reserved` is rejected** — reserved capacity is arranged out-of-band with your
# account team, not by passing a parameter.
#
# The response echoes the tier that actually served the request, which is what you
# should log (the resolved tier can differ from what you asked for).
#
# | Tier | Use for | Trade-off |
# |---|---|---|
# | `priority` | customer-facing, latency-critical | premium price, prioritised |
# | `default` | everyday workloads | standard price, standard latency |
# | `flex` | evals, batch-ish, agent side-quests | discounted, processed last |

# %% [markdown]
# ## 4. Measure TTFT and throughput
#
# **TTFT** (time to first token) is dominated by *prefill* — the model reading your
# prompt — plus queue time. Longer prompts mean higher TTFT. Prompt caching cuts
# prefill, which is why it helps latency and not just cost.
#
# We count SSE (server-sent events) frames as a token proxy: exact enough to compare
# tiers fairly.


# %%
def measure(tier: str, model: str = MODEL, prompt: str = PROMPT, max_tokens: int = 200):
    body = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "service_tier": tier,
    }
    return ttft("/openai/v1/responses", body, region=REGION)


print(f"{'tier':10} {'TTFT (s)':>10} {'total (s)':>10} {'frames':>8} {'frames/s':>10}")
print("-" * 52)
tier_results = {}
for tier in ("default", "flex", "priority"):
    m = measure(tier)
    tier_results[tier] = m
    print(
        f"{tier:10} {m['ttft_s']:>10.3f} {m['total_s']:>10.3f} "
        f"{m['frames']:>8} {m['frames_per_s']:>10.1f}"
    )

# %% [markdown]
# Single samples are noisy — one call proves nothing about tier performance. Repeat
# and take medians before drawing conclusions.

# %%
REPEATS = 3
print(f"median of {REPEATS} runs per tier\n")
print(f"{'tier':10} {'TTFT p50':>10} {'total p50':>11} {'frames/s p50':>14}")
print("-" * 48)
for tier in ("default", "flex", "priority"):
    runs = [measure(tier) for _ in range(REPEATS)]
    print(
        f"{tier:10} {statistics.median(r['ttft_s'] for r in runs):>10.3f} "
        f"{statistics.median(r['total_s'] for r in runs):>11.3f} "
        f"{statistics.median(r['frames_per_s'] for r in runs):>14.1f}"
    )

# %% [markdown]
# **Interpreting this honestly.** On an idle account the tiers often look similar,
# because tiering governs *queue priority* and only separates under contention.
# Priority's documented benefit (up to ~25% better output tokens/sec vs Standard)
# shows up when the Region is busy — not in a quiet notebook run.

# %% [markdown]
# ## 5. Prompt length drives TTFT
#
# A direct demonstration of the prefill effect.

# %%
filler = "Distributed systems engineering is a broad discipline. "
print(f"{'approx input tokens':>20} {'TTFT (s)':>10}")
print("-" * 32)
for repeat in (1, 40, 200):
    long_prompt = filler * repeat + "\n\nSummarise the above in one sentence."
    approx_tokens = len(long_prompt) // 4
    m = ttft(
        "/openai/v1/responses",
        {"model": MODEL, "input": long_prompt, "max_output_tokens": 60},
        region=REGION,
    )
    print(f"{approx_tokens:>20} {m['ttft_s']:>10.3f}")

# %% [markdown]
# ## 6. Ramping: don't spike from zero
#
# Because most models have no published quota, throughput is fair-share against
# internal capacity that adapts to your observed usage. Going from 0 to peak in one
# step is the classic way to get `CapacityExceeded` / 503s. Ramp gradually.

# %%
import concurrent.futures as cf


def one_call(i: int):
    code, _ = post(
        "/openai/v1/responses",
        {"model": MODEL, "input": f"Say OK ({i})", "max_output_tokens": 16},
        region=REGION,
    )
    return code


for concurrency in (1, 3, 6):
    started = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        codes = list(pool.map(one_call, range(concurrency)))
    elapsed = time.perf_counter() - started
    ok = sum(1 for c in codes if c == 200)
    print(
        f"concurrency {concurrency:2} -> {ok}/{concurrency} ok in {elapsed:5.2f}s  "
        f"codes={codes}"
    )

# %% [markdown]
# In production, step concurrency up over minutes rather than seconds, and keep
# the backoff loop in place so shed requests are retried instead of dropped.

# %% [markdown]
# ## 7. A production-hardened client pattern
#
# Putting the pieces together: fresh token, tier selection, backoff, timeout, and
# logging the resolved tier.


# %%
class MantleClient:
    """Minimal production shape: retries, tier, resolved-tier logging."""

    def __init__(self, model: str, region: str = REGION, tier: str = "default"):
        self.model, self.region, self.tier = model, region, tier
        self.prefix = (
            "/openai/v1"
            if model.startswith(("google.gemma-4", "openai.gpt-5", "xai."))
            else "/v1"
        )

    def respond(self, prompt: str, max_output_tokens: int = 256) -> dict:
        code, data = post(
            f"{self.prefix}/responses",
            {
                "model": self.model,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
                "service_tier": self.tier,
                "store": False,  # opt out of 30-day retention
            },
            region=self.region,
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        return {
            "text": data.get("output_text", ""),
            "resolved_tier": data.get("service_tier"),
            "usage": data.get("usage", {}),
        }


client = MantleClient(MODEL, tier="flex")
result = client.respond("Name one benefit of queue-based fair sharing.", 80)
print("resolved tier:", result["resolved_tier"])
print("usage:", json.dumps(result["usage"]))
print("text:", result["text"][:160])

# %% [markdown]
# ## Gotchas from this notebook
#
# | Gotcha | Detail |
# |---|---|
# | Separate in/out TPM | Not one combined TPM like `bedrock-runtime` |
# | No RPM quota | Throttling is token-based only |
# | Oversized `max_tokens` | Reserves input quota you don't use → self-throttling |
# | Most models unpublished | No Service Quotas entry; increases via Support case |
# | Cached tokens | Exempt from input-TPM quota |
# | `reserved` tier | Rejected as a parameter — arranged via account team |
# | Resolved tier | Log `service_tier` from the response, not what you asked for |
# | Tier benchmarks | Look flat on an idle account; tiers separate under load |
# | Spiking | Ramp over minutes; 0→peak invites 503s |
#
# ## Next
# Head to your model family:
# `../01-openai-gpt/` · `../02-anthropic-claude/` · `../03-google-gemma/` ·
# `../04-qwen/` …
