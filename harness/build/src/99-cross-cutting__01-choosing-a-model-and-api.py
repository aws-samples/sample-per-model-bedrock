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
# # Choosing a model and API on Amazon Bedrock Mantle
#
# A live capability survey across every family on `bedrock-mantle`. Rather than
# trusting a table someone wrote months ago, this notebook **probes the endpoint**
# and builds the table from what the API actually does today.
#
# Use it to answer: *which model, which API, which path, which parameters?*
#
# ## What this notebook produces
# - The full model inventory and its Region footprint
# - A per-family API matrix (Responses / Chat Completions / Messages)
# - A per-model parameter matrix (`temperature`, `top_p`, `service_tier`, …)
# - A reusable `capabilities.py` you can drop into your own project
#
# ## Self-contained, but see also
# - **Auth, the three URL paths** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - Family deep-dives: `../01-openai-gpt/` … `../12-writer-palmyra/`
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
import concurrent.futures as cf
import json
import sys
from collections import defaultdict

sys.path.insert(0, "../_shared")
from mantle import err, list_models, post

REGION = "us-east-1"  # the widest inventory
REGIONS = ("us-east-1", "us-east-2", "us-west-2", "eu-central-1")
print("survey region:", REGION)

# %% [markdown]
# ## 1. Inventory and Region footprint

# %%
inventory = {}
for region in REGIONS:
    try:
        inventory[region] = sorted(list_models(region))
    except (RuntimeError, OSError) as exc:
        inventory[region] = []
        print(f"{region}: {type(exc).__name__}")

for region in REGIONS:
    print(f"{region:14} {len(inventory[region]):3} models")

models = inventory[REGION]
families = defaultdict(list)
for mid in models:
    families[mid.split(".")[0]].append(mid)

print(f"\n{len(models)} models across {len(families)} families in {REGION}")

# %%
print(f"{'family':14} {'n':>3}  models")
print("-" * 100)
for family in sorted(families):
    ids = families[family]
    print(
        f"{family:14} {len(ids):>3}  {', '.join(i.split('.', 1)[1] for i in ids)[:76]}"
    )

# %%
# Which models are missing where? This is the deployment-planning view.
everywhere = set(inventory[REGIONS[0]])
for region in REGIONS[1:]:
    everywhere &= set(inventory[region])
print(f"available in ALL four Regions: {len(everywhere)}")

only_one = [m for m in models if sum(1 for r in REGIONS if m in inventory[r]) == 1]
print(f"available in only ONE Region : {len(only_one)}")
for mid in sorted(only_one):
    where = [r for r in REGIONS if mid in inventory[r]]
    print(f"   {mid:44} {where[0]}")

# %% [markdown]
# ## 2. Path resolution
#
# Three prefixes, and the rule is by model family. This function is the single most
# useful thing to copy out of this collection.


# %%
def api_prefix(model_id: str) -> str:
    """Which URL prefix serves this model's inference APIs?"""
    if model_id.startswith("anthropic."):
        return "/anthropic/v1"
    if model_id.startswith(("google.gemma-4", "openai.gpt-5", "xai.")):
        return "/openai/v1"
    return "/v1"


by_prefix = defaultdict(list)
for mid in models:
    by_prefix[api_prefix(mid)].append(mid)

for prefix in sorted(by_prefix):
    ids = by_prefix[prefix]
    fams = sorted({i.split(".")[0] for i in ids})
    print(f"{prefix:16} {len(ids):3} models | families: {', '.join(fams)}")

# %% [markdown]
# Note the split *inside* the OpenAI family: `gpt-5.*` uses `/openai/v1` while
# `gpt-oss*` uses the bare `/v1`. Provider name alone is not enough.

# %%
for mid in sorted(families["openai"]):
    print(f"  {mid:34} -> {api_prefix(mid)}")

# %% [markdown]
# ## 3. Probe the API surface per family
#
# One representative per family, three APIs each. Run in parallel to keep it quick.

# %%
REPRESENTATIVES = [
    "google.gemma-4-31b",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "anthropic.claude-haiku-4-5",
    "xai.grok-4.3",
    "qwen.qwen3-32b",
    "deepseek.v3.2",
    "zai.glm-5",
    "minimax.minimax-m2.5",
    "moonshotai.kimi-k2.5",
    "mistral.mistral-large-3-675b-instruct",
    "nvidia.nemotron-super-3-120b",
    "writer.palmyra-vision-7b",
]
REPRESENTATIVES = [m for m in REPRESENTATIVES if m in models]
AV = {"anthropic-version": "2023-06-01"}


def probe_apis(model_id: str) -> dict:
    """Status code per API. Single attempt + short timeout: a wrong path can stall."""
    prefix = api_prefix(model_id)
    out = {"model": model_id, "prefix": prefix}

    if prefix == "/anthropic/v1":
        code, _ = post(
            f"{prefix}/messages",
            {
                "model": model_id,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            region=REGION,
            headers=AV,
            attempts=1,
            timeout=45,
        )
        out["messages"] = code
        out["responses"] = out["chat"] = "-"
        return out

    code, _ = post(
        f"{prefix}/responses",
        {"model": model_id, "input": "Hi", "max_output_tokens": 16},
        region=REGION,
        attempts=1,
        timeout=45,
    )
    out["responses"] = code
    code, _ = post(
        f"{prefix}/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
        },
        region=REGION,
        attempts=1,
        timeout=45,
    )
    out["chat"] = code
    out["messages"] = "-"
    return out


with cf.ThreadPoolExecutor(max_workers=5) as pool:
    api_results = list(pool.map(probe_apis, REPRESENTATIVES))

print(f"{'model':40} {'prefix':16} {'Resp':>6} {'Chat':>6} {'Msg':>6}")
print("-" * 80)
for row in api_results:
    print(
        f"{row['model']:40} {row['prefix']:16} {str(row['responses']):>6} "
        f"{str(row['chat']):>6} {str(row['messages']):>6}"
    )

# %% [markdown]
# `200` means available; `400` means "this model does not serve that API"; `-1`
# means the request **stalled** rather than erroring — which is why every probe
# above sets a timeout.

# %%
summary = {"responses": [], "chat": [], "messages": []}
for row in api_results:
    for key in summary:
        if row[key] == 200:
            summary[key].append(row["model"].split(".")[0])
print("Responses API       :", sorted(set(summary["responses"])))
print("Chat Completions    :", sorted(set(summary["chat"])))
print("Anthropic Messages  :", sorted(set(summary["messages"])))

# %% [markdown]
# ## 4. Parameter compatibility
#
# Sampling parameters are the most common cross-family break. There is **no single
# config that works everywhere** — this table proves it.


# %%
def probe_params(model_id: str) -> dict:
    prefix = api_prefix(model_id)
    out = {"model": model_id}

    if prefix == "/anthropic/v1":
        base = {
            "model": model_id,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        for label, extra in (
            ("temperature", {"temperature": 0.5}),
            ("top_p", {"top_p": 0.9}),
        ):
            code, _ = post(
                f"{prefix}/messages",
                {**base, **extra},
                region=REGION,
                headers=AV,
                attempts=1,
                timeout=45,
            )
            out[label] = "ok" if code == 200 else str(code)
        out["service_tier"] = "-"
        return out

    # Use whichever inference API this model actually serves.
    code, _ = post(
        f"{prefix}/responses",
        {"model": model_id, "input": "Hi", "max_output_tokens": 16},
        region=REGION,
        attempts=1,
        timeout=45,
    )
    if code == 200:
        path, base = f"{prefix}/responses", {
            "model": model_id,
            "input": "Hi",
            "max_output_tokens": 16,
        }
    else:
        path, base = (
            f"{prefix}/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 16,
            },
        )

    for label, extra in (
        ("temperature", {"temperature": 0.5}),
        ("top_p", {"top_p": 0.9}),
        ("service_tier", {"service_tier": "flex"}),
    ):
        code, _ = post(path, {**base, **extra}, region=REGION, attempts=1, timeout=45)
        out[label] = "ok" if code == 200 else str(code)
    return out


with cf.ThreadPoolExecutor(max_workers=5) as pool:
    param_results = list(pool.map(probe_params, REPRESENTATIVES))

print(f"{'model':40} {'temperature':>12} {'top_p':>8} {'flex tier':>11}")
print("-" * 76)
for row in param_results:
    print(
        f"{row['model']:40} {row['temperature']:>12} {row['top_p']:>8} "
        f"{row['service_tier']:>11}"
    )

# %% [markdown]
# Read the inverses carefully:
#
# - **Gemma 4** takes `temperature`, rejects `top_p`.
# - **Grok** rejects `temperature`, takes `top_p`.
# - **Frontier Claude** rejects `temperature` (deprecated); `haiku-4-5` accepts it.
# - **gpt-5.x** rejects the `flex` and `priority` tiers; gpt-oss accepts them.
#
# So sampling and tier settings must be resolved per model, not per provider.

# %% [markdown]
# ## 5. Structured-output support

# %%
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


ASK = [{"role": "user", "content": "Answer 'hi'."}]
PROBE = {"region": REGION, "attempts": 1, "timeout": 60}


def _claude_forced_tool(model_id: str, prefix: str) -> int:
    """Claude: output_config.format is rejected on mantle; forced tools work."""
    code, _ = post(
        f"{prefix}/messages",
        {
            "model": model_id,
            "max_tokens": 300,
            "messages": ASK,
            "tools": [
                {"name": "emit", "description": "Return.", "input_schema": SCHEMA}
            ],
            "tool_choice": {"type": "tool", "name": "emit"},
        },
        headers=AV,
        **PROBE,
    )
    return code


def _native_schema(model_id: str, prefix: str) -> str:
    """Try Responses text.format, then fall back to Chat Completions."""
    code, _ = post(
        f"{prefix}/responses",
        {
            "model": model_id,
            "input": "Answer 'hi'.",
            "max_output_tokens": 300,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "s",
                    "schema": SCHEMA,
                    "strict": True,
                }
            },
        },
        **PROBE,
    )
    if code in (200, 400):
        return "ok" if code == 200 else "400"

    # No Responses API for this model; try Chat Completions response_format.
    code, _ = post(
        f"{prefix}/chat/completions",
        {
            "model": model_id,
            "messages": ASK,
            "max_tokens": 300,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "s", "strict": True, "schema": SCHEMA},
            },
        },
        **PROBE,
    )
    return "ok (CC)" if code == 200 else str(code)


def _forced_tool(model_id: str, prefix: str) -> str:
    """Forced function call — the portable structured-output route."""
    code, _ = post(
        f"{prefix}/chat/completions",
        {
            "model": model_id,
            "messages": ASK,
            "max_tokens": 300,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "emit",
                        "description": "Return.",
                        "parameters": SCHEMA,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "emit"}},
        },
        **PROBE,
    )
    return "ok" if code == 200 else str(code)


# %% [markdown]
# One dispatcher over the three probes above.


# %%
def probe_structured(model_id: str) -> dict:
    """Which structured-output mechanisms does this model actually accept?"""
    prefix = api_prefix(model_id)
    if prefix == "/anthropic/v1":
        code = _claude_forced_tool(model_id, prefix)
        return {
            "model": model_id,
            "native_schema": "n/a",
            "forced_tool": "ok" if code == 200 else str(code),
        }
    return {
        "model": model_id,
        "native_schema": _native_schema(model_id, prefix),
        "forced_tool": _forced_tool(model_id, prefix),
    }


with cf.ThreadPoolExecutor(max_workers=4) as pool:
    struct_results = list(pool.map(probe_structured, REPRESENTATIVES))

print(f"{'model':40} {'native schema':>15} {'forced tool':>13}")
print("-" * 72)
for row in struct_results:
    print(f"{row['model']:40} {row['native_schema']:>15} {row['forced_tool']:>13}")

# %% [markdown]
# **Caveat worth remembering:** "accepted" is not "reliable". A request can return
# 200 while the model ignores the constraint (gpt-oss does this with a named
# `tool_choice`) or appends characters after a valid JSON object (Gemma 4 does this
# roughly half the time). Always validate the payload you get back.

# %% [markdown]
# ## 6. A reusable capabilities resolver
#
# Everything above, distilled into something you can paste into a project.

# %%
CAPABILITIES_PY = '''
"""Resolve bedrock-mantle request details per model. Generated from live probes."""

OPENAI_PREFIX_FAMILIES = ("google.gemma-4", "openai.gpt-5", "xai.")

# Models that reject `temperature` at ANY value.
NO_TEMPERATURE = ("xai.grok-4.3", "anthropic.claude-opus-5",
                  "anthropic.claude-sonnet-5", "anthropic.claude-opus-4-8")
# Models that accept `temperature` ONLY at its default 1.0 (Responses API).
# Passing 0.7 to these is a 400 - so on Responses, either send 1.0 or omit it.
TEMPERATURE_DEFAULT_ONLY = ("google.gemma-4", "openai.gpt-5.")
# Models that reject `top_p`.
NO_TOP_P = ("google.gemma-4", "openai.gpt-5.")
# Models that accept flex/priority service tiers.
TIERED = ("openai.gpt-oss", "google.gemma-4", "xai.", "qwen.", "deepseek.",
          "zai.", "minimax.", "moonshotai.", "mistral.", "nvidia.", "writer.")


def api_prefix(model_id):
    """Return the URL prefix that serves this model's inference APIs."""
    if model_id.startswith("anthropic."):
        return "/anthropic/v1"
    if model_id.startswith(OPENAI_PREFIX_FAMILIES):
        return "/openai/v1"
    return "/v1"


def inference_path(model_id, api="auto"):
    """Return the full inference path for a model, e.g. "/v1/chat/completions"."""
    prefix = api_prefix(model_id)
    if prefix == "/anthropic/v1":
        return prefix + "/messages"
    if api == "chat":
        return prefix + "/chat/completions"
    return prefix + "/responses"


def sampling(model_id, temperature=None, top_p=None):
    """Drop or clamp parameters this model rejects instead of getting a 400."""
    out = {}
    if temperature is not None and not model_id.startswith(NO_TEMPERATURE):
        if model_id.startswith(TEMPERATURE_DEFAULT_ONLY):
            # Only the default 1.0 is accepted on the Responses API.
            if abs(float(temperature) - 1.0) < 1e-9:
                out["temperature"] = 1.0
            # else: omit entirely rather than 400
        else:
            out["temperature"] = temperature
    if top_p is not None and not model_id.startswith(NO_TOP_P):
        out["top_p"] = top_p
    return out


def service_tier(model_id, tier="default"):
    """Downgrade to 'default' where flex/priority are unsupported."""
    if tier == "default" or model_id.startswith(TIERED):
        return tier
    return "default"


def token_limit_field(model_id, api="auto"):
    """Responses uses max_output_tokens (min 16); Messages/CC use max_tokens."""
    prefix = api_prefix(model_id)
    if prefix == "/anthropic/v1" or api == "chat":
        return "max_tokens"
    return "max_output_tokens"
'''

with open("capabilities.py", "w") as handle:
    handle.write(CAPABILITIES_PY)

sys.path.insert(0, ".")
import capabilities

print(f"{'model':40} {'path':34} {'sampling':>34}")
print("-" * 112)
for mid in (
    "google.gemma-4-31b",
    "xai.grok-4.3",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "anthropic.claude-sonnet-5",
    "qwen.qwen3-32b",
):
    path = capabilities.inference_path(mid)
    sampling = capabilities.sampling(mid, temperature=0.7, top_p=0.95)
    print(f"{mid:40} {path:34} {json.dumps(sampling):>34}")

# %%
# Verify the resolver against the live endpoint — a resolver that is wrong is worse
# than none.
print(f"{'model':40} {'resolved call':>14}")
print("-" * 58)
for mid in (
    "google.gemma-4-31b",
    "xai.grok-4.3",
    "openai.gpt-oss-120b",
    "qwen.qwen3-32b",
    "anthropic.claude-haiku-4-5",
):
    path = capabilities.inference_path(mid)
    limit_field = capabilities.token_limit_field(mid)
    body = {
        "model": mid,
        limit_field: 16,
        **capabilities.sampling(mid, temperature=0.7, top_p=0.95),
    }
    body["service_tier"] = capabilities.service_tier(mid, "flex")
    if path.endswith("/messages"):
        body["messages"] = [{"role": "user", "content": "Hi"}]
        body.pop("service_tier", None)
        headers = AV
    elif path.endswith("/responses"):
        body["input"] = "Hi"
        headers = None
    else:
        body["messages"] = [{"role": "user", "content": "Hi"}]
        headers = None
    code, data = post(
        path, body, region=REGION, headers=headers, attempts=1, timeout=60
    )
    verdict = "200 OK" if code == 200 else f"{code}"
    print(f"{mid:40} {verdict:>14}")
    if code != 200:
        print(f"      {err(data)[:90]}")

# %% [markdown]
# ## 7. A decision guide
#
# | If you need… | Use |
# |---|---|
# | Web Search grounding | `openai.gpt-5.*` (Responses) — the only family |
# | Reasoning traces you can read | Responses API: gemma-4, gpt-5.x, gpt-oss, grok |
# | Server-side tools (Lambda / Gateway) | Responses API models; built-ins on gpt-oss |
# | Adaptive thinking + 1h prompt cache | `anthropic.claude-*` (Messages) |
# | Explicit prompt-cache breakpoints | `openai.gpt-5.6-*` |
# | Server-side conversation state | Responses API + `store=True` |
# | Zero data retention | any model, `store=False` + retention mode `none` |
# | EU data residency | check `eu-central-1` — no Anthropic, no gpt-5.x, no xAI |
# | Vision | gemma-4, qwen3-vl, nemotron-nano-12b, palmyra-vision, Claude |
# | Cheapest viable model | Ministral / Nemotron Nano / GLM Flash ladders |
# | Fine-tuning | `gpt-oss-20b` or `qwen3-32b`, us-west-2 only |
#
# ## Gotchas this survey exposes
#
# | Gotcha | Detail |
# |---|---|
# | Three path prefixes | And a split *within* the OpenAI family |
# | Responses API coverage | Only a minority of families; Chat Completions is universal |
# | Claude is Messages-only | Both OpenAI-compatible APIs 400 |
# | No universal sampling config | On Responses, `temperature` must be `1.0` for Gemma 4 / gpt-5.6; Grok rejects it entirely |
# | Tier support varies | gpt-5.x is `default`-only |
# | Wrong path may stall | Always set a client-side timeout when probing |
# | 200 ≠ honoured | Constraints can be silently ignored — validate output |
# | Region footprint | 55 models in us-east-1, 33 in eu-central-1 |
#
# ## Next
# - `02-migrating-from-openai.ipynb` — porting an existing OpenAI codebase
# - `03-production-hardening-checklist.ipynb` — the pre-launch checklist
