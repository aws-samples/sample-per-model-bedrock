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
# # OpenAI GPT on Amazon Bedrock Mantle — Responses API core
#
# OpenAI's models on the `bedrock-mantle` endpoint. This family is the only one
# with **Web Search**, **server-side tools**, and **explicit prompt-cache
# breakpoints**, so it needs more coverage than the others — five notebooks
# rather than one. This one covers the core Responses API; the siblings cover
# the rest.
#
# | Notebook | Covers |
# |---|---|
# | **01 (this one)** | Responses API core, reasoning, streaming, state, async |
# | `02-web-search-and-grounding.ipynb` | Web Search, citations, IAM governance |
# | `03-tools-and-structured-output.ipynb` | Client-side tools, strict JSON |
# | `04-prompt-caching-and-cost.ipynb` | Explicit cache breakpoints, economics |
# | `05-server-side-tools-and-fine-tuning.ipynb` | Lambda MCP (Model Context Protocol), notes/tasks, RFT (reinforcement fine-tuning), batch |
#
# **Models covered**
#
# | Model ID | Notes |
# |---|---|
# | `openai.gpt-5.6-sol` | Frontier; explicit prompt caching |
# | `openai.gpt-5.6-terra` | Frontier sibling |
# | `openai.gpt-5.6-luna` | Frontier sibling |
# | `openai.gpt-5.5` | Previous generation; automatic caching |
# | `openai.gpt-5.4` | Earlier generation |
# | `openai.gpt-oss-120b` / `-20b` | Open-weight; **bare `/v1` path**, server-side tools |
# | `openai.gpt-oss-safeguard-120b` / `-20b` | Safety-classification variants |
#
# ## Two path families inside one provider
# This is the subtlety unique to OpenAI on mantle:
#
# - `openai.gpt-5.*` → **`/openai/v1`**
# - `openai.gpt-oss*` → **bare `/v1`**
#
# Get it wrong and you get a 400 saying the model doesn't exist. We prove it in §2.
#
# ## Self-contained, but see also
# - **Auth, the three URL paths, model discovery** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Projects, cost attribution, data retention / ZDR (zero data retention),
#   CloudWatch** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
# - **Quotas, retries, service tiers, TTFT (time-to-first-token)** →
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
from mantle import err, list_models, parse_json_lenient, post, response_text

# gpt-5.5 and gpt-5.6-sol are NOT in us-west-2 for every account — us-east-1 is
# the safest choice for this family. See section 10.
REGION = "us-east-1"

SOL = "openai.gpt-5.6-sol"
TERRA = "openai.gpt-5.6-terra"
LUNA = "openai.gpt-5.6-luna"
GPT55 = "openai.gpt-5.5"
GPT54 = "openai.gpt-5.4"
OSS120 = "openai.gpt-oss-120b"
OSS20 = "openai.gpt-oss-20b"

GPT5_PREFIX = "/openai/v1"  # gpt-5.x
OSS_PREFIX = "/v1"  # gpt-oss


def prefix_for(model_id: str) -> str:
    """gpt-5.x lives under /openai/v1; gpt-oss under bare /v1."""
    return GPT5_PREFIX if model_id.startswith("openai.gpt-5.") else OSS_PREFIX


print("gpt-5.6-sol  ->", prefix_for(SOL))
print("gpt-oss-120b ->", prefix_for(OSS120))

# %% [markdown]
# ## 1. First call
#
# Auth is a short-term Bedrock API key minted from ambient IAM credentials: valid
# ≤12 h, not refreshable, Region-pinned. (`../00-foundations/01` covers the
# self-refreshing provider and the SigV4 (AWS Signature Version 4) alternative that
# needs no key.)

# %%
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

# One client per path family, because the base URL differs.
gpt5 = OpenAI(
    api_key=provide_token(region=REGION),
    base_url=f"https://bedrock-mantle.{REGION}.api.aws{GPT5_PREFIX}",
)
oss = OpenAI(
    api_key=provide_token(region=REGION),
    base_url=f"https://bedrock-mantle.{REGION}.api.aws{OSS_PREFIX}",
)

resp = gpt5.responses.create(
    model=SOL,
    input="Explain what an inference engine does, in two sentences.",
    max_output_tokens=200,
)
print(resp.output_text)
print("\nusage:", resp.usage.model_dump_json())

# %% [markdown]
# ## 2. The path split, demonstrated
#
# Send each model to both prefixes and watch which combinations work.

# %%
print(f"{'model':24} {'/openai/v1':>12} {'/v1':>8}")
print("-" * 46)
for model in (SOL, GPT55, GPT54, OSS120, OSS20):
    row = {}
    for label, prefix in (("/openai/v1", GPT5_PREFIX), ("/v1", OSS_PREFIX)):
        code, _ = post(
            f"{prefix}/responses",
            {"model": model, "input": "Reply OK", "max_output_tokens": 16},
            region=REGION,
        )
        row[label] = code
    print(f"{model:24} {row['/openai/v1']:>12} {row['/v1']:>8}")

# %% [markdown]
# Two clean groups. Note also that **gpt-5.6 does not serve Chat Completions at
# all** — it is Responses-only, unlike gpt-5.4/5.5 and gpt-oss:

# %%
print(f"{'model':24} {'Responses':>10} {'ChatCompletions':>17}")
print("-" * 54)
for model in (SOL, GPT55, GPT54, OSS120):
    prefix = prefix_for(model)
    code_r, _ = post(
        f"{prefix}/responses",
        {"model": model, "input": "Reply OK", "max_output_tokens": 16},
        region=REGION,
    )
    code_c, _ = post(
        f"{prefix}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_tokens": 16,
        },
        region=REGION,
    )
    print(f"{model:24} {code_r:>10} {code_c:>17}")

# %% [markdown]
# ## 3. Sampling parameters
#
# `temperature` is accepted; **`top_p` is rejected on gpt-5.6** even though
# gpt-oss accepts it. Another reason not to share one sampling config across
# models — see `../03-google-gemma/` where the split runs the other way.

# %%
for model in (SOL, OSS120):
    prefix = prefix_for(model)
    print(f"\n{model}")
    for label, extra in [
        ("temperature=1.0", {"temperature": 1.0}),
        ("top_p=0.95", {"top_p": 0.95}),
    ]:
        code, data = post(
            f"{prefix}/responses",
            {"model": model, "input": "Reply OK", "max_output_tokens": 16, **extra},
            region=REGION,
        )
        print(f"   {label:18} -> HTTP {code} {'' if code == 200 else err(data)[:60]}")

# %%
# max_output_tokens has a MINIMUM of 16 on the Responses API.
for n in (8, 16):
    code, data = post(
        f"{GPT5_PREFIX}/responses",
        {"model": SOL, "input": "Hi", "max_output_tokens": n},
        region=REGION,
    )
    detail = "" if code == 200 else err(data)[:70]
    print(f"max_output_tokens={n:3} -> HTTP {code} {detail}")

# %% [markdown]
# ## 4. Reasoning
#
# Effort ladder is `none` / `low` / `medium` / `high` — `minimal` is rejected.
# The reasoning trace is returned as separate `reasoning` items in `output`, which
# is a Responses-API-only capability.

# %%
for effort in ("none", "minimal", "low", "medium", "high"):
    code, data = post(
        f"{GPT5_PREFIX}/responses",
        {
            "model": SOL,
            "input": "2+2?",
            "max_output_tokens": 32,
            "reasoning": {"effort": effort},
        },
        region=REGION,
    )
    print(f"  effort={effort:8} -> HTTP {code} {'' if code == 200 else err(data)[:60]}")

# %%
reasoned = gpt5.responses.create(
    model=SOL,
    input=(
        "Three switches outside a room control one bulb inside. You may flip "
        "switches freely but enter the room only once. How do you identify the "
        "correct switch?"
    ),
    reasoning={"effort": "high"},
    max_output_tokens=1500,
)
print("output item types:", [i.type for i in reasoned.output])
print("reasoning tokens :", reasoned.usage.output_tokens_details.reasoning_tokens)
print("\n=== ANSWER ===")
print(reasoned.output_text[:400])

# %% [markdown]
# ### Effort changes spend, measurably

# %%
print(f"{'effort':8} {'reasoning tok':>14} {'output tok':>11} {'latency':>9}")
print("-" * 46)
for effort in ("none", "low", "high"):
    started = time.perf_counter()
    r = gpt5.responses.create(
        model=SOL,
        input="What is 17 * 23? Answer with the number only.",
        reasoning={"effort": effort},
        max_output_tokens=800,
    )
    elapsed = time.perf_counter() - started
    print(
        f"{effort:8} {r.usage.output_tokens_details.reasoning_tokens:>14} "
        f"{r.usage.output_tokens:>11} {elapsed:>8.2f}s"
    )

# %% [markdown]
# ## 5. Verbosity and other gpt-5.6 controls
#
# The frontier models accept several extra Responses parameters. Probe them so you
# know what is safe to send.

# %%
extras = [
    ("text.verbosity=low", {"text": {"verbosity": "low"}}),
    ("truncation=auto", {"truncation": "auto"}),
    ("metadata", {"metadata": {"team": "samples"}}),
    ("max_tool_calls=2", {"max_tool_calls": 2}),
    ("parallel_tool_calls", {"parallel_tool_calls": True}),
    (
        "include reasoning.encrypted_content",
        {"include": ["reasoning.encrypted_content"]},
    ),
]
for label, extra in extras:
    code, data = post(
        f"{GPT5_PREFIX}/responses",
        {"model": SOL, "input": "Reply OK", "max_output_tokens": 16, **extra},
        region=REGION,
    )
    print(f"  {label:38} -> HTTP {code} {'' if code == 200 else err(data)[:50]}")

# %%
# Verbosity has a visible effect on answer length.
for verbosity in ("low", "high"):
    r = gpt5.responses.create(
        model=SOL,
        input="What is a load balancer?",
        text={"verbosity": verbosity},
        max_output_tokens=500,
    )
    print(
        f"verbosity={verbosity:5} -> {len(r.output_text):4} chars | "
        f"{r.output_text[:80]!r}"
    )

# %% [markdown]
# ## 6. Streaming
#
# Reasoning and answer text arrive on distinct event types, so you can render a
# "thinking" panel separately from the answer.

# %%
stream = gpt5.responses.create(
    model=SOL,
    input="List three trade-offs of microservice architectures.",
    reasoning={"effort": "low"},
    max_output_tokens=500,
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
# ## 7. Multi-turn: history array vs server-side state

# %%
conversation = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is a circuit breaker in distributed systems?"},
]
first = gpt5.responses.create(model=SOL, input=conversation, max_output_tokens=200)
print("assistant:", first.output_text[:160])

conversation += [
    {"role": "assistant", "content": first.output_text},
    {"role": "user", "content": "What is the usual half-open state for?"},
]
second = gpt5.responses.create(model=SOL, input=conversation, max_output_tokens=200)
print("\nassistant:", second.output_text[:160])

# %% [markdown]
# **Do not replay reasoning items** into the next turn — send only final answers.
# Keep reasoning in your own logs.

# %%
# Server-side state: cheaper on input tokens, but requires store=True which
# retains input and output for 30 days in-Region (see ../00-foundations/02).
turn1 = gpt5.responses.create(
    model=SOL,
    input="My deploy tool is CodeDeploy. Reply: noted.",
    max_output_tokens=32,
    store=True,
)
turn2 = gpt5.responses.create(
    model=SOL,
    input="Which deploy tool did I mention?",
    previous_response_id=turn1.id,
    max_output_tokens=48,
)
print("chained recall:", turn2.output_text[:120])

code, data = post(
    f"{GPT5_PREFIX}/responses",
    {
        "model": SOL,
        "input": "And again?",
        "max_output_tokens": 32,
        "previous_response_id": gpt5.responses.create(
            model=SOL, input="secret. reply ok", max_output_tokens=16, store=False
        ).id,
    },
    region=REGION,
)
print(f"chaining from store=False -> HTTP {code}: {err(data)[:80]}")

# %% [markdown]
# ## 8. Async / background jobs
#
# Long jobs can run detached and be polled — useful for deep-reasoning tasks that
# would otherwise hold an HTTP connection open.

# %%
bg = gpt5.responses.create(
    model=SOL,
    input="Write three paragraphs on the CAP theorem and its practical limits.",
    max_output_tokens=800,
    background=True,
    store=True,
)
print("job:", bg.id, "status:", bg.status)

for _ in range(40):
    time.sleep(3)
    code, polled = post(
        f"{GPT5_PREFIX}/responses/{bg.id}", None, region=REGION, method="GET"
    )
    if polled.get("status") in ("completed", "failed", "cancelled"):
        break
print("final status:", polled.get("status"))
print("text:", response_text(polled)[:220])

# %%
for rid in (turn1.id, bg.id):
    code, _ = post(
        f"{GPT5_PREFIX}/responses/{rid}", None, region=REGION, method="DELETE"
    )
    print(f"DELETE {rid[:30]}… -> {code}")

# %% [markdown]
# ## 9. gpt-oss and the safeguard variants
#
# The open-weight models sit at the bare `/v1` path and support both Responses and
# Chat Completions. The `safeguard` variants are tuned for safety classification.

# %%
r = oss.responses.create(
    model=OSS120,
    input="Explain in one sentence what an open-weight model is.",
    max_output_tokens=150,
)
print("gpt-oss-120b:", r.output_text[:180])

# %%
# The safeguard models classify content against a policy you supply.
code, data = post(
    f"{OSS_PREFIX}/chat/completions",
    {
        "model": "openai.gpt-oss-safeguard-20b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Policy: flag any request seeking personal medical advice. "
                    "Answer with exactly ALLOW or FLAG."
                ),
            },
            {
                "role": "user",
                "content": "What dosage of ibuprofen should I take for my back?",
            },
        ],
        "max_tokens": 300,
    },
    region=REGION,
)
verdict = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
print("safeguard-20b verdict:", repr((verdict or "").strip()[:120]))

# %% [markdown]
# ## 10. Regional footprint — check before you deploy
#
# This family is *not* uniformly available. `gpt-5.5` and `gpt-5.6-sol` were absent
# from `us-west-2` at the time of writing, while `gpt-oss` reaches every Region.

# %%
watch = [SOL, TERRA, LUNA, GPT55, GPT54, OSS120, OSS20]
regions = ("us-east-1", "us-east-2", "us-west-2", "eu-central-1")
inventory = {}
for reg in regions:
    try:
        inventory[reg] = set(list_models(reg))
    except (RuntimeError, OSError) as exc:
        inventory[reg] = set()
        print(f"{reg}: {type(exc).__name__}")

print(f"{'model':32} " + "  ".join(f"{r:>13}" for r in regions))
print("-" * 92)
for model in watch:
    cells = "  ".join(
        f"{('yes' if model in inventory[r] else '-'):>13}" for r in regions
    )
    print(f"{model:32} {cells}")

# %% [markdown]
# ## 11. Compare the generations

# %%
task = "In one sentence, why does batching improve GPU inference throughput?"
print(f"{'model':26} {'latency':>9} {'reason tok':>11} {'out tok':>8}  answer")
print("-" * 108)
for model in (OSS20, OSS120, GPT54, GPT55, SOL):
    prefix = prefix_for(model)
    started = time.perf_counter()
    code, data = post(
        f"{prefix}/responses",
        {
            "model": model,
            "input": task,
            "max_output_tokens": 200,
            "reasoning": {"effort": "low"},
        },
        region=REGION,
    )
    elapsed = time.perf_counter() - started
    if code != 200:
        print(f"{model:26} {'-':>9} {'-':>11} {'-':>8}  HTTP {code}: {err(data)[:34]}")
        continue
    usage = data.get("usage", {})
    details = usage.get("output_tokens_details", {})
    text = " ".join(response_text(data).split())
    print(
        f"{model:26} {elapsed:>8.2f}s {details.get('reasoning_tokens', 0):>11} "
        f"{usage.get('output_tokens', 0):>8}  {text[:40]!r}"
    )

# %% [markdown]
# ## 11b. Service tiers are not uniform in this family
#
# `flex` and `priority` trade cost against queue priority — but the **gpt-5.x
# models accept only `default`**, while gpt-oss accepts all three. Sending `flex`
# to gpt-5.6 is a 400, so tier selection has to be model-aware.

# %%
print(f"{'model':30} " + "  ".join(f"{t:>9}" for t in ("default", "flex", "priority")))
print("-" * 64)
tier_support = {}
for model in (SOL, GPT55, GPT54, OSS120, OSS20):
    prefix = prefix_for(model)
    row, allowed = [], []
    for tier in ("default", "flex", "priority"):
        code, _ = post(
            f"{prefix}/responses",
            {
                "model": model,
                "input": "Reply OK",
                "max_output_tokens": 16,
                "service_tier": tier,
            },
            region=REGION,
        )
        row.append("ok" if code == 200 else str(code))
        if code == 200:
            allowed.append(tier)
    tier_support[model] = allowed
    print(f"{model:30} " + "  ".join(f"{v:>9}" for v in row))

print("\nallowed tiers per model:")
for model, allowed in tier_support.items():
    print(f"   {model:30} {allowed}")

# %% [markdown]
# ## 12. Production hardening

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "openai-gpt-samples",
        "tags": {"Application": "OpenAIGPTDemo", "Environment": "Demo"},
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)


class GPTClient:
    """Production shape: correct path per model, retries, attribution, no retention."""

    # gpt-5.x accepts only the default tier; gpt-oss accepts flex/priority too.
    TIERED_MODELS = ("openai.gpt-oss",)

    def __init__(self, model=SOL, region=REGION, tier="default", project=None):
        self.model, self.region, self.project = model, region, project
        self.prefix = prefix_for(model)
        if tier != "default" and not model.startswith(self.TIERED_MODELS):
            print(
                f"[note] {model} only supports service_tier='default' "
                f"— ignoring requested '{tier}'"
            )
            tier = "default"
        self.tier = tier

    def ask(self, prompt, *, effort="low", max_output_tokens=512, verbosity=None):
        body = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max(16, max_output_tokens),  # API minimum is 16
            "reasoning": {"effort": effort},
            "service_tier": self.tier,
            "store": False,  # opt out of the 30-day retention default
            # top_p deliberately omitted: rejected on gpt-5.6
        }
        if verbosity:
            body["text"] = {"verbosity": verbosity}
        headers = {"OpenAI-Project": self.project} if self.project else None
        code, data = post(
            f"{self.prefix}/responses", body, region=self.region, headers=headers
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        return data


# Asking for "flex" on gpt-5.6 is silently downgraded by the guard above.
bot = GPTClient(model=SOL, tier="flex", project=project_id)
out = bot.ask("Name one benefit of speculative decoding.", verbosity="low")
print("answer:", response_text(out)[:160])
print("tier  :", out.get("service_tier"))

# gpt-oss really does honour flex.
oss_bot = GPTClient(model=OSS120, tier="flex", project=project_id)
oss_out = oss_bot.ask("Name one benefit of batching.", max_output_tokens=120)
print("\ngpt-oss tier:", oss_out.get("service_tier"))

# %%
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — OpenAI GPT on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | **Two path prefixes** | `gpt-5.*` → `/openai/v1`; `gpt-oss*` → bare `/v1` |
# | gpt-5.6 | **Responses-only** — Chat Completions returns 400 |
# | `top_p` | Rejected on gpt-5.6; accepted on gpt-oss |
# | `max_output_tokens` | Minimum **16** |
# | `reasoning.effort` | `none`/`low`/`medium`/`high`; **`minimal` rejected** |
# | Reasoning replay | Send final answers only in multi-turn history |
# | `store=False` | Blocks `previous_response_id` chaining (404) |
# | Region | `gpt-5.5` / `gpt-5.6-sol` absent from us-west-2; none of gpt-5.x in eu-central-1 |
# | `service_tier` | gpt-5.x accepts **`default` only**; `flex`/`priority` → 400. gpt-oss accepts all |
# | Web Search | Only this family — see `02-web-search-and-grounding.ipynb` |
#
# ## Next
# - `02-web-search-and-grounding.ipynb` — the feature unique to this family
# - `03-tools-and-structured-output.ipynb` · `04-prompt-caching-and-cost.ipynb`
# - `05-server-side-tools-and-fine-tuning.ipynb`
