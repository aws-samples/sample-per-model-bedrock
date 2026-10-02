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
# # Migrating to Amazon Bedrock Mantle from OpenAI or Anthropic
#
# The headline promise is real: point your existing SDK at a new base URL and key,
# and it works. This notebook shows that — and then shows the four places where a
# naive port breaks, so you find them here rather than in production.
#
# ## What this notebook covers
# - The two-line change (OpenAI SDK and Anthropic SDK)
# - The four real breakages: path prefixes, sampling params, tiers, absent APIs
# - A compatibility shim that makes one client work across families
# - LLM-gateway configuration
# - A migration checklist you can work through
#
# ## Self-contained, but see also
# - **Auth, the three URL paths** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Live capability survey** → `01-choosing-a-model-and-api.ipynb`
#
# ## Prerequisites
# ```bash
# pip install -r ../requirements.txt
# ```
#
# Needs openai, anthropic, aws-bedrock-token-generator.
#
# `requirements.txt` pins the exact versions this collection was tested
# against. An unpinned install resolves whatever is current, which may be
# untested or compromised (OWASP LLM03, Supply Chain).

# %%
import json
import sys

sys.path.insert(0, "../_shared")
from mantle import err, post, response_text

REGION = "us-east-1"
print("region:", REGION)

# %% [markdown]
# ## 1. The two-line change
#
# For an OpenAI codebase, `base_url` and `api_key` are the entire migration.

# %%
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

# BEFORE (OpenAI):
#   client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#
# AFTER (Bedrock Mantle):
client = OpenAI(
    api_key=provide_token(region=REGION),  # short-term Bedrock key from IAM
    base_url=f"https://bedrock-mantle.{REGION}.api.aws/openai/v1",
)

response = client.responses.create(
    model="openai.gpt-5.6-sol",  # note: Bedrock model IDs, not OpenAI's
    input="Confirm you are reachable in five words.",
    max_output_tokens=60,
)
print(response.output_text)

# %% [markdown]
# Environment-variable form, if you would rather not touch code at all:
#
# ```bash
# export OPENAI_BASE_URL="https://bedrock-mantle.us-east-1.api.aws/openai/v1"
# export OPENAI_API_KEY="$(
#   python -c 'from aws_bedrock_token_generator import provide_token
# print(provide_token(region="us-east-1"))'
# )"
# ```
#
# **Do not** hardcode a long-term key. Short-term keys are presigned SigV4 (AWS
#  Signature Version 4), expire
# within 12 hours, and inherit the permissions of the role that minted them.

# %%
# The Anthropic SDK migrates the same way. Note the base URL omits /v1 —
# the SDK appends it.
import anthropic

claude = anthropic.Anthropic(
    api_key=provide_token(region=REGION),
    base_url=f"https://bedrock-mantle.{REGION}.api.aws/anthropic",
)
message = claude.messages.create(
    model="anthropic.claude-haiku-4-5",
    max_tokens=60,  # required on the Messages API
    messages=[{"role": "user", "content": "Confirm you are reachable in five words."}],
)
print("".join(b.text for b in message.content if b.type == "text"))

# %% [markdown]
# ## 2. Model IDs are not OpenAI's
#
# `gpt-4o` does not exist here. Map your model names first — this is the most common
# first-day error.

# %%
MODEL_MAP = {
    # OpenAI name           -> a reasonable Bedrock Mantle equivalent
    "gpt-4o": "openai.gpt-5.6-sol",
    "gpt-4o-mini": "openai.gpt-5.4",
    "gpt-4-turbo": "openai.gpt-5.5",
    "o1": "openai.gpt-5.6-sol",
    "gpt-3.5-turbo": "openai.gpt-oss-20b",
    # Anthropic
    "claude-3-5-sonnet": "anthropic.claude-sonnet-5",
    "claude-3-haiku": "anthropic.claude-haiku-4-5",
}
for old, new in MODEL_MAP.items():
    print(f"  {old:22} -> {new}")

# %%
# Prove the point: an OpenAI model ID is simply not present.
code, data = post(
    "/openai/v1/responses",
    {"model": "gpt-4o", "input": "Hi", "max_output_tokens": 16},
    region=REGION,
    attempts=1,
    timeout=45,
)
print(f"model='gpt-4o' -> HTTP {code}: {err(data)[:100]}")

# %% [markdown]
# ## 3. Breakage 1 — the path prefix depends on the model
#
# There is no single base URL that serves every model. Three prefixes exist, and the
# OpenAI family is split across two of them.


# %%
def api_prefix(model_id: str) -> str:
    if model_id.startswith("anthropic."):
        return "/anthropic/v1"
    if model_id.startswith(("google.gemma-4", "openai.gpt-5", "xai.")):
        return "/openai/v1"
    return "/v1"


for mid in (
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "google.gemma-4-31b",
    "qwen.qwen3-32b",
    "anthropic.claude-haiku-4-5",
):
    print(f"  {mid:30} -> {api_prefix(mid)}")

# %%
# One client with the wrong prefix fails for half your models.
wrong = OpenAI(
    api_key=provide_token(region=REGION),
    base_url=f"https://bedrock-mantle.{REGION}.api.aws/openai/v1",
)
for mid in ("openai.gpt-5.6-sol", "openai.gpt-oss-120b"):
    try:
        wrong.responses.create(model=mid, input="Hi", max_output_tokens=16)
        print(f"  {mid:24} -> ok on /openai/v1")
    except Exception as exc:
        print(f"  {mid:24} -> {type(exc).__name__}: {str(exc)[:80]}")

# %% [markdown]
# ## 4. Breakage 2 — sampling parameters are per model
#
# A shared `temperature=0.7, top_p=0.95` default, which is completely normal in
# OpenAI code, fails on several models here.

# %%
CANDIDATES = [
    ("openai.gpt-5.6-sol", "/openai/v1"),
    ("google.gemma-4-31b", "/openai/v1"),
    ("xai.grok-4.3", "/openai/v1"),
    ("openai.gpt-oss-120b", "/v1"),
    ("qwen.qwen3-32b", "/v1"),
]
print(f"{'model':26} {'0.7+top_p':>11} {'temp 0.7':>9} {'temp 1.0':>9} {'top_p':>7}")
print("-" * 66)
for mid, prefix in CANDIDATES:
    row = []
    for extra in (
        {"temperature": 0.7, "top_p": 0.95},
        {"temperature": 0.7},
        {"temperature": 1.0},
        {"top_p": 0.95},
    ):
        code, _ = post(
            f"{prefix}/responses",
            {"model": mid, "input": "Hi", "max_output_tokens": 16, **extra},
            region=REGION,
            attempts=1,
            timeout=45,
        )
        row.append("ok" if code == 200 else str(code))
    print(f"{mid:26} {row[0]:>11} {row[1]:>9} {row[2]:>9} {row[3]:>7}")

# %% [markdown]
# Three different behaviours in one table:
#
# - **Gemma 4 / gpt-5.6** reject `top_p`, and accept `temperature` **only at its
#   default `1.0`** — so a ported `temperature=0.7` breaks.
# - **Grok** rejects `temperature` at every value but accepts `top_p`.
# - **gpt-oss / qwen** accept both, at any value.
#
# The safest port is to **send neither parameter** on the Responses API unless you
# have a specific reason, then add each back per model after testing. If you truly
# need sampling control, Chat Completions is more permissive.

# %% [markdown]
# ## 5. Breakage 3 — `service_tier` is not universal

# %%
print(f"{'model':26} {'flex':>8} {'priority':>10}")
print("-" * 48)
for mid, prefix in CANDIDATES:
    row = []
    for tier in ("flex", "priority"):
        code, _ = post(
            f"{prefix}/responses",
            {
                "model": mid,
                "input": "Hi",
                "max_output_tokens": 16,
                "service_tier": tier,
            },
            region=REGION,
            attempts=1,
            timeout=45,
        )
        row.append("ok" if code == 200 else str(code))
    print(f"{mid:26} {row[0]:>8} {row[1]:>10}")

# %% [markdown]
# ## 6. Breakage 4 — the API you use may not exist for your model
#
# If your codebase is built on Chat Completions, note that **gpt-5.6 does not serve
# it**. If it is built on Responses, note that most open-weight families do not.

# %%
print(f"{'model':26} {'Responses':>10} {'ChatCompl':>10}")
print("-" * 50)
for mid, prefix in CANDIDATES:
    code_r, _ = post(
        f"{prefix}/responses",
        {"model": mid, "input": "Hi", "max_output_tokens": 16},
        region=REGION,
        attempts=1,
        timeout=45,
    )
    code_c, _ = post(
        f"{prefix}/chat/completions",
        {
            "model": mid,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
        },
        region=REGION,
        attempts=1,
        timeout=45,
    )
    print(f"{mid:26} {code_r:>10} {code_c:>10}")

# %% [markdown]
# ## 7. A compatibility shim
#
# One class that resolves path, API, parameters and tier per model, so your
# application code stops caring.


# %%
class MantleCompat:
    """Model-agnostic wrapper over the three bedrock-mantle API surfaces."""

    NO_TEMPERATURE = (
        "xai.grok-4.3",
        "anthropic.claude-opus-5",
        "anthropic.claude-sonnet-5",
        "anthropic.claude-opus-4-8",
    )
    NO_TOP_P = ("google.gemma-4", "openai.gpt-5.")
    TIERED = (
        "openai.gpt-oss",
        "google.gemma-4",
        "xai.",
        "qwen.",
        "deepseek.",
        "zai.",
        "minimax.",
        "moonshotai.",
        "mistral.",
        "nvidia.",
        "writer.",
    )
    NO_CHAT_COMPLETIONS = ("openai.gpt-5.6",)

    def __init__(self, region=REGION, project=None):
        self.region, self.project = region, project

    # ---- resolution -----------------------------------------------------
    def prefix(self, model):
        if model.startswith("anthropic."):
            return "/anthropic/v1"
        if model.startswith(("google.gemma-4", "openai.gpt-5", "xai.")):
            return "/openai/v1"
        return "/v1"

    def surface(self, model):
        """Which API to use for this model."""
        if model.startswith("anthropic."):
            return "messages"
        # Families with no Responses API fall back to Chat Completions.
        if model.startswith(
            (
                "qwen.",
                "deepseek.",
                "zai.",
                "minimax.",
                "moonshotai.",
                "mistral.",
                "nvidia.",
                "writer.",
            )
        ):
            return "chat"
        return "responses"

    def _sampling(self, model, temperature, top_p):
        out = {}
        if temperature is not None and not model.startswith(self.NO_TEMPERATURE):
            out["temperature"] = temperature
        if top_p is not None and not model.startswith(self.NO_TOP_P):
            out["top_p"] = top_p
        return out

    def _tier(self, model, tier):
        return (
            tier if (tier == "default" or model.startswith(self.TIERED)) else "default"
        )

    # ---- one call, any model -------------------------------------------
    def _headers(self, surface):
        """Per-surface headers: project attribution, and Anthropic's version pin."""
        headers = {}
        if self.project:
            key = "anthropic-workspace" if surface == "messages" else "OpenAI-Project"
            headers[key] = self.project
        if surface == "messages":
            headers["anthropic-version"] = "2023-06-01"
        return headers

    def _build(
        self,
        model,
        prompt,
        surface,
        prefix,
        system,
        max_tokens,
        temperature,
        top_p,
        tier,
    ):
        """Return (path, body) in whichever wire format this surface expects."""
        sampling = self._sampling(model, temperature, top_p)
        if surface == "messages":
            body = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                **sampling,
            }
            if system:
                body["system"] = system  # Messages takes system as a top-level field
            return f"{prefix}/messages", body

        turns = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        if surface == "chat":
            return f"{prefix}/chat/completions", {
                "model": model,
                "messages": turns,
                "max_tokens": max_tokens,
                "service_tier": self._tier(model, tier),
                **sampling,
            }
        return f"{prefix}/responses", {
            "model": model,
            "input": turns,
            "max_output_tokens": max(16, max_tokens),  # Responses minimum is 16
            "service_tier": self._tier(model, tier),
            "store": False,
            **sampling,
        }

    def complete(
        self,
        model,
        prompt,
        *,
        system=None,
        max_tokens=400,
        temperature=None,
        top_p=None,
        tier="default",
        timeout=120,
        attempts=2,
    ):
        """One call surface over all three APIs.

        `timeout` and `attempts` are bounded on purpose. Reasoning-first models
        (Grok especially) can take many minutes for a single call under load, and
        post()'s defaults (240s x 5) would turn one slow model into a 20-minute
        stall for the whole loop.
        """
        prefix, surface = self.prefix(model), self.surface(model)
        headers = self._headers(surface)
        path, body = self._build(
            model, prompt, surface, prefix, system, max_tokens, temperature, top_p, tier
        )
        code, data = post(
            path,
            body,
            region=self.region,
            headers=headers or None,
            timeout=timeout,
            attempts=attempts,
        )
        if code != 200:
            raise RuntimeError(f"{model}: HTTP {code}: {err(data)}")
        return self._extract(surface, data)

    @staticmethod
    def _extract(surface, data):
        if surface == "messages":
            return "".join(
                b.get("text", "")
                for b in data.get("content", [])
                if b.get("type") == "text"
            )
        if surface == "chat":
            return (data.get("choices") or [{}])[0].get("message", {}).get(
                "content"
            ) or ""
        return response_text(data)


compat = MantleCompat()
PROMPT = "Name one benefit of a managed inference endpoint. One sentence."

# %% [markdown]
# Exercise the shim across every path family — one call site, five models, three
# different wire protocols underneath.

# %%
print(f"{'model':30} {'surface':10}  answer")
print("-" * 96)
for mid in (
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "google.gemma-4-31b",
    "xai.grok-4.3",
    "qwen.qwen3-32b",
    "anthropic.claude-haiku-4-5",
):
    try:
        # Deliberately pass BOTH sampling params and a flex tier — the shim drops
        # whatever this model would reject.
        answer = compat.complete(
            mid, PROMPT, temperature=0.7, top_p=0.95, tier="flex", max_tokens=120
        )
        print(f"{mid:30} {compat.surface(mid):10}  {' '.join(answer.split())[:52]!r}")
    except RuntimeError as exc:
        print(f"{mid:30} {compat.surface(mid):10}  FAILED: {str(exc)[:52]}")

# %% [markdown]
# Same call signature, six models, three different API surfaces — and no 400s,
# because the shim resolved the differences.

# %% [markdown]
# ## 8. Streaming across surfaces
#
# The three APIs stream differently. Normalise it once.

# %%
from mantle import stream_lines


def stream_text(model, prompt, region=REGION, max_tokens=200):
    """Yield text deltas from whichever streaming shape this model uses."""
    helper = MantleCompat(region=region)
    prefix, surface = helper.prefix(model), helper.surface(model)
    headers = {"anthropic-version": "2023-06-01"} if surface == "messages" else None

    if surface == "messages":
        path = f"{prefix}/messages"
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }
    elif surface == "chat":
        path = f"{prefix}/chat/completions"
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        path = f"{prefix}/responses"
        body = {
            "model": model,
            "max_output_tokens": max(16, max_tokens),
            "stream": True,
            "input": prompt,
        }

    for line in stream_lines(path, body, region=region, headers=headers):
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if surface == "messages":
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta.get("text", "")
        elif surface == "chat":
            for choice in event.get("choices", []):
                chunk = (choice.get("delta") or {}).get("content")
                if chunk:
                    yield chunk
        else:
            if event.get("type") == "response.output_text.delta":
                yield event.get("delta", "")


for mid in ("google.gemma-4-31b", "qwen.qwen3-32b", "anthropic.claude-haiku-4-5"):
    print(f"--- {mid} ---")
    chunks = 0
    for piece in stream_text(mid, "Count from one to five.", max_tokens=80):
        chunks += 1
        print(piece, end="", flush=True)
    print(f"\n[{chunks} deltas]\n")

# %% [markdown]
# ## 9. LLM gateways
#
# Because the endpoint is OpenAI-compatible, gateways such as LiteLLM work by
# configuring a custom base URL. The two things to get right are the **per-model
# base URL** (path prefix) and **token refresh**.
#
# ```yaml
# model_list:
#   - model_name: gpt-frontier
#     litellm_params:
#       model: openai/openai.gpt-5.6-sol
#       api_base: https://bedrock-mantle.us-east-1.api.aws/openai/v1
#       api_key: os.environ/BEDROCK_API_KEY
#
#   - model_name: gpt-oss
#     litellm_params:
#       model: openai/openai.gpt-oss-120b
#       api_base: https://bedrock-mantle.us-east-1.api.aws/v1   # bare /v1
#       api_key: os.environ/BEDROCK_API_KEY
#
#   - model_name: qwen
#     litellm_params:
#       model: openai/qwen.qwen3-32b
#       api_base: https://bedrock-mantle.us-east-1.api.aws/v1
#       api_key: os.environ/BEDROCK_API_KEY
# ```
#
# Refresh the key on a schedule — it expires within 12 hours and cannot be renewed:
#
# ```bash
# # cron / sidecar
# export BEDROCK_API_KEY="$(python -c '
# from aws_bedrock_token_generator import provide_token
# print(provide_token(region="us-east-1"))')"
# ```

# %%
# Verify the exact base URLs a gateway config would need.
print("base URLs by model (paste into your gateway config):")
for mid in (
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "google.gemma-4-31b",
    "qwen.qwen3-32b",
    "anthropic.claude-haiku-4-5",
):
    print(
        f"  {mid:30} https://bedrock-mantle.{REGION}.api.aws"
        f"{MantleCompat().prefix(mid)}"
    )

# %% [markdown]
# ## 10. Features you gain, and features you lose
#
# | Moving from | You gain | You lose / must change |
# |---|---|---|
# | OpenAI API | IAM auth, Projects, ZDR (zero data retention), AWS-hosted Web Search, service tiers | OpenAI model IDs; some params; Chat Completions on gpt-5.6 |
# | Anthropic API | IAM auth, Workspaces, `count_tokens` | `output_config.format` (use forced tools) |
# | `bedrock-runtime` | OpenAI/Anthropic-shaped APIs, stateful chat, server-side tools | cross-Region inference, Provisioned Throughput, batch |

# %%
# The gains are real — here is Projects-based attribution, which has no OpenAI
# equivalent.
code, project = post(
    "/v1/organization/projects",
    {
        "name": "migration-samples",
        "tags": {
            "Application": "MigrationDemo",
            "Environment": "Demo",
            "CostCenter": "0000",
        },
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)

attributed = MantleCompat(project=project_id)
print(
    "attributed call:",
    attributed.complete("google.gemma-4-31b", "Reply OK.", max_tokens=20)[:40],
)

code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## 11. Migration checklist
#
# 1. **Map model IDs.** OpenAI/Anthropic names do not exist here.
# 2. **Resolve the path prefix per model.** One base URL will not cover your fleet.
# 3. **Strip shared sampling defaults.** Add `temperature`/`top_p` back per model.
# 4. **Check the API surface exists** for each model (Responses vs Chat vs Messages).
# 5. **Gate `service_tier`** — gpt-5.x is `default`-only.
# 6. **Set `store` explicitly.** It defaults to `true` = 30-day retention.
# 7. **Replace API-key handling** with short-term token minting plus refresh.
# 8. **Add retry/backoff** — mantle has no RPM quota and sheds load.
# 9. **Set client timeouts** — a wrong path can stall rather than 400.
# 10. **Parse output defensively** — 200 does not guarantee the constraint held.
# 11. **Re-point observability** to the `AWS/BedrockMantle` namespace.
# 12. **Create Projects** and tag them for cost attribution.
# 13. **Confirm Region coverage** for every model you depend on.
#
# ## Gotchas — migration
#
# | Gotcha | Detail |
# |---|---|
# | Model IDs | `gpt-4o` etc. do not exist; map them first |
# | One base URL is not enough | Three prefixes, and the OpenAI family spans two |
# | Shared sampling defaults | On Responses, `temperature` must be `1.0` for Gemma 4 / gpt-5.6, and is rejected outright by Grok |
# | `service_tier` | gpt-5.x accepts `default` only |
# | Chat Completions | Absent on gpt-5.6; Responses absent on most open-weight families |
# | `store` default | `true` — 30-day retention unless you opt out |
# | Token lifetime | ≤12 h, not refreshable, Region-pinned |
# | `max_output_tokens` | Minimum 16 on Responses |
# | Long-term keys | Exploration only; they create a static IAM user credential |
#
# ## Next
# `03-production-hardening-checklist.ipynb` — everything to verify before launch.
