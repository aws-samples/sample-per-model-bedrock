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
# # Foundations 1 — Endpoints, auth, and the three URL paths
#
# **Start here.** Every other notebook in this collection assumes what this one
# establishes. The three things worth knowing before you write a line of code:
#
# 1. `bedrock-mantle` is a **second, separate endpoint** next to `bedrock-runtime`.
#    It speaks OpenAI- and Anthropic-compatible APIs.
# 2. There is **no `bedrock-mantle` client in boto3**. The AWS SDK is useful here
#    as a *request signer*, not as a service client.
# 3. Model IDs map to **three different URL paths**. Guessing wrong gives you a
#    404 or a 400, and this is the single most common source of confusion.
#
# ## What this notebook covers
# - Endpoint choice: when `bedrock-mantle`, when `bedrock-runtime`
# - Auth A: SigV4 with botocore · Auth B: short-term API key · Auth C: curl
# - The three path families, demonstrated against live models
# - Model discovery and the regional footprint
# - The IAM permissions you actually need
#
# ## Prerequisites
# ```bash
# pip install -r ../requirements.txt
# ```
#
# Needs openai, boto3, aws-bedrock-token-generator.
#
# `requirements.txt` pins the exact versions this collection was tested
# against. An unpinned install resolves whatever is current, which may be
# untested or compromised (OWASP LLM03, Supply Chain).
#
# You also need AWS credentials in your environment (profile, role, or instance
# metadata) with Bedrock Mantle access.

# %%
import json
import os
import sys
import urllib.error
import urllib.request

import boto3

sys.path.insert(0, "../_shared")
from mantle import safe_print

REGION = "us-east-1"
HOST = f"https://bedrock-mantle.{REGION}.api.aws"

print("boto3", boto3.__version__)
# safe_print redacts the account ID and principal name: this output is
# committed to a public repository (see _shared/mantle.py).
safe_print("caller:", boto3.client("sts").get_caller_identity()["Arn"])
print("host:  ", HOST)

# %% [markdown]
# ## 1. Two endpoints, and why
#
# Both endpoints can serve the *same* model at the *same* per-token price. You
# pick based on the API surface and the features you need, not on cost.

# %%
comparison = [
    ("URL", "bedrock-runtime.{r}.amazonaws.com", "bedrock-mantle.{r}.api.aws"),
    ("APIs", "InvokeModel, Converse", "Responses, Chat Completions, Messages"),
    ("Auth", "SigV4", "SigV4 or Bedrock API key"),
    ("OpenAI SDK drop-in", "no", "yes (base_url + key)"),
    ("Stateful chat", "you manage history", "previous_response_id"),
    ("Cross-region (CRIS)", "geo/global inference profiles", "in-Region only"),
    ("Provisioned Thruput", "yes", "no"),
    ("Batch inference", "yes", "no (use bedrock-runtime)"),
    ("Quotas", "combined TPM + RPM", "separate in/out TPM, no RPM"),
    ("CloudWatch namespace", "AWS/Bedrock", "AWS/BedrockMantle"),
    ("Cost attribution", "inference profiles", "Projects / Workspaces"),
]
w = 21
print(f"{'':{w}} {'bedrock-runtime':36} bedrock-mantle")
print("-" * 100)
for row in comparison:
    print(f"{row[0]:{w}} {row[1]:36} {row[2]}")

# %% [markdown]
# **Rule of thumb.** New applications → `bedrock-mantle`. Reach back to
# `bedrock-runtime` when you specifically need cross-Region inference,
# Provisioned Throughput, batch inference, or a model that isn't on mantle yet.

# %% [markdown]
# ## 2. The three URL path families
#
# This is the part that trips people up. A model's family determines its path:
#
# | Path | Families |
# |---|---|
# | `/openai/v1/…` | `google.gemma-4*`, `openai.gpt-5*`, `xai.*` |
# | `/v1/…` | `openai.gpt-oss*` and every Chat-Completions-only family |
# | `/anthropic/v1/…` | `anthropic.*` only |
#
# Control-plane paths (models, files, projects, fine-tuning, data retention)
# are **always** under `/v1/…`, never `/openai/v1/…`.


# %%
def api_prefix(model_id: str) -> str:
    """Which URL prefix does this model's inference API live under?"""
    if model_id.startswith("anthropic."):
        return "/anthropic/v1"
    if any(model_id.startswith(p) for p in ("google.gemma-4", "openai.gpt-5", "xai.")):
        return "/openai/v1"
    return "/v1"


for m in [
    "google.gemma-4-31b",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "xai.grok-4.3",
    "anthropic.claude-sonnet-5",
    "qwen.qwen3-32b",
]:
    print(f"{m:28} -> {HOST}{api_prefix(m)}")

# %% [markdown]
# ## 3. Auth A — SigV4 with the AWS SDK (no API key)
#
# Best for Lambda / ECS / EC2 where a role is already attached: nothing to store,
# nothing to rotate. Note two things:
#
# - The **signing name is `bedrock`** (`bedrock-mantle` also works).
# - You sign a plain HTTPS request. There is no `boto3.client("bedrock-mantle")`.

# %%
print("boto3 services containing 'bedrock':")
print(" ", [s for s in boto3.Session().get_available_services() if "bedrock" in s])
print("\nNote the absence of 'bedrock-mantle' — hence manual signing below.")

# %%
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.httpsession import URLLib3Session


def sigv4_post(path: str, payload: dict, region: str = REGION, timeout: int = 120):
    """POST to bedrock-mantle signed with ambient AWS credentials."""
    body = json.dumps(payload)
    # get_frozen_credentials() pins a consistent access-key/secret/token triple.
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    req = AWSRequest(
        method="POST",
        url=f"https://bedrock-mantle.{region}.api.aws{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(creds, "bedrock", region).add_auth(req)  # signing name = "bedrock"
    resp = URLLib3Session(timeout=timeout).send(req.prepare())
    parsed = json.loads(resp.text) if resp.text.strip() else {}
    return resp.status_code, parsed


status, data = sigv4_post(
    "/openai/v1/responses",
    {
        "model": "google.gemma-4-31b",
        "input": "Reply with exactly: OK",
        "max_output_tokens": 16,
    },
)
print("SigV4 ->", status, "|", json.dumps(data.get("output_text", ""))[:60])

# %% [markdown]
# ## 4. Auth B — short-term Bedrock API key (what the OpenAI SDK needs)
#
# The OpenAI SDK can only send a bearer token; it cannot SigV4-sign. AWS's
# official `aws-bedrock-token-generator` mints a **short-term** key from your
# normal IAM credentials.
#
# Key facts from the library's own docs:
# - It is a **pre-signed SigV4 request** in a wrapper — so it *is* IAM.
# - Max lifetime **12 hours**; actual = min(requested, credential expiry).
# - **Cannot be refreshed or extended** — you mint a new one.
# - It is **Region-pinned** to the Region it was minted in.
#
# Avoid *long-term* keys outside exploration: they create a real IAM user with a
# static credential.

# %%
from aws_bedrock_token_generator import provide_token

api_key = provide_token(region=REGION)
print("token prefix:", api_key[:22] + "…")
print("length:", len(api_key))
print("\nThe prefix identifies it as a presigned-URL bearer token:")
print(" ", api_key.split("-")[0:3])

# %% [markdown]
# ### A self-refreshing provider
#
# Because tokens cannot be refreshed, long-lived processes should mint on demand
# and track expiry themselves. Keep the TTL short (15 min is plenty) — the token
# *is* your role until it expires.

# %%
import threading
from datetime import datetime, timedelta, timezone


class MantleTokenProvider:
    """Mints short-term Bedrock tokens, refreshing before expiry."""

    def __init__(self, region=REGION, ttl=timedelta(minutes=15), skew_s=120):
        self.region, self.ttl, self.skew_s = region, ttl, skew_s
        self._token = None
        self._expires_at = None
        self._lock = threading.Lock()

    def get(self) -> str:
        now = datetime.now(timezone.utc)
        with self._lock:  # avoid a thundering herd of mints
            if self._token and self._expires_at and now < self._expires_at:
                return self._token
            self._token = provide_token(region=self.region, expiry=self.ttl)
            self._expires_at = now + self.ttl - timedelta(seconds=self.skew_s)
            return self._token


tokens = MantleTokenProvider()
t1 = tokens.get()
t2 = tokens.get()  # served from cache
print("cached on second call:", t1 == t2)
print("expires around:", tokens._expires_at.isoformat(timespec="seconds"))

# %% [markdown]
# ### The OpenAI SDK, pointed at Bedrock
#
# Note: build the client from a *fresh* token. Don't construct one at import time
# and reuse it for hours — the baked-in key expires.

# %%
from openai import OpenAI

gemma = OpenAI(api_key=tokens.get(), base_url=HOST + "/openai/v1")
r = gemma.responses.create(
    model="google.gemma-4-31b",
    input="Reply with exactly: OK",
    max_output_tokens=16,
)
print("OpenAI SDK ->", repr(r.output_text))

# %% [markdown]
# ## 5. Auth C — curl
#
# Both auth styles work from the shell. Useful for smoke tests and CI.
#
# **Pass the key through the environment, never on the command line.** Process
# arguments are world-readable via `ps` on most systems, so a key interpolated
# into an argument leaks to every local user (CWE-214, *Invocation of Process
# Using Visible Sensitive Information*). `curl` reads `$BEDROCK_API_KEY` from the
# environment it inherits, so the literal never appears in the argument list.

# %%
import subprocess  # nosec B404  # noqa: S404

CURL_BEARER = f"""curl -sS -X POST "{HOST}/openai/v1/responses" \
  -H "Authorization: Bearer $BEDROCK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{{"model":"google.gemma-4-31b","input":"Reply OK","max_output_tokens":16}}'"""
print("Bearer-token form:\n", CURL_BEARER, "\n")

# The command string is a fixed literal defined above -- no untrusted input is
# interpolated -- and the secret arrives via `env`, not argv.
# Absolute interpreter path (never resolved via $PATH) and a fixed argv built
# from the literal above -- no untrusted input reaches the command line.
completed = subprocess.run(  # nosec B603 B607  # noqa: S603
    ["/bin/bash", "-c", CURL_BEARER],
    capture_output=True,
    text=True,
    timeout=180,
    check=False,
    env={**os.environ, "BEDROCK_API_KEY": api_key},
)
if completed.returncode != 0:
    print("curl failed:", completed.stderr[:200])
else:
    body = json.loads(completed.stdout)
    print("live result:", body.get("output_text", completed.stdout[:120]))

# %%
# curl can also SigV4-sign natively (curl >= 7.75). Signing name is "bedrock".
curl_sigv4 = f"""curl -sS -X POST "{HOST}/openai/v1/responses" \
  -H "Content-Type: application/json" \
  --aws-sigv4 "aws:amz:{REGION}:bedrock" \
  --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  -H "x-amz-security-token: $AWS_SESSION_TOKEN" \
  -d '{{"model":"google.gemma-4-31b","input":"Reply OK","max_output_tokens":16}}'"""
print("SigV4 form (note: needs x-amz-security-token when using temporary creds):")
print(curl_sigv4)

# %% [markdown]
# ## 6. Model discovery
#
# `GET /v1/models` is the authoritative inventory. **`/openai/v1/models` returns
# 404** — the control plane lives under `/v1` only.


# %%
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


def get_json(path: str, region: str = REGION):
    req = urllib.request.Request(
        f"https://bedrock-mantle.{region}.api.aws{path}",
        headers={"Authorization": f"Bearer {provide_token(region=region)}"},
    )
    try:
        with open_https(req, timeout=90) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {"body": e.read().decode()[:120]}


for path in ("/v1/models", "/openai/v1/models"):
    code, payload = get_json(path)
    n = len(payload.get("data", [])) if code == 200 else "-"
    print(f"GET {path:20} -> {code}  models={n}")

# %%
code, payload = get_json("/v1/models")
model_ids = sorted(m["id"] for m in payload["data"])

families = {}
for mid in model_ids:
    families.setdefault(mid.split(".")[0], []).append(mid)

print(f"{len(model_ids)} models across {len(families)} families in {REGION}\n")
for fam in sorted(families):
    print(f"{fam:12} ({len(families[fam])})")
    for mid in families[fam]:
        print(f"             {mid}")

# %% [markdown]
# ## 7. Which API does each family actually support?
#
# Don't assume — probe. The Responses API is *recommended* by AWS, but it is only
# available on a minority of families. Chat Completions is the universal surface
# for open-weight models, and Claude is Messages-only on mantle.


# %%
def probe_apis(model_id: str) -> dict:
    """Send one minimal request per API and record the status code."""
    prefix = api_prefix(model_id)
    results = {}

    if not model_id.startswith("anthropic."):
        code, _ = sigv4_post(
            f"{prefix}/responses",
            {"model": model_id, "input": "hi", "max_output_tokens": 16},
        )
        results["Responses"] = code
        code, _ = sigv4_post(
            f"{prefix}/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            },
        )
        results["ChatCompletions"] = code
        results["Messages"] = "-"
    else:
        results["Responses"] = results["ChatCompletions"] = "-"
        # Messages needs the anthropic-version header, so use urllib directly.
        req = urllib.request.Request(
            HOST + "/anthropic/v1/messages",
            data=json.dumps(
                {
                    "model": model_id,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with open_https(req, timeout=120) as resp:
                results["Messages"] = resp.status
        except urllib.error.HTTPError as e:
            results["Messages"] = e.code
    return results


representatives = [
    "google.gemma-4-31b",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "xai.grok-4.3",
    "anthropic.claude-sonnet-5",
    "qwen.qwen3-32b",
    "deepseek.v3.2",
]
print(f"{'model':28} {'path':15} {'Responses':>10} {'ChatCompl':>10} {'Messages':>9}")
print("-" * 78)
for m in representatives:
    r = probe_apis(m)
    print(
        f"{m:28} {api_prefix(m):15} {str(r['Responses']):>10} "
        f"{str(r['ChatCompletions']):>10} {str(r['Messages']):>9}"
    )

# %% [markdown]
# Read the 400s as "this API is not available for this model". Concretely:
#
# - **gpt-5.6** is Responses-only (Chat Completions 400s).
# - **qwen / deepseek** and the other open-weight families are Chat-Completions-only.
# - **Claude** is Messages-only on mantle.
#
# Each family notebook in this collection leads with whichever API actually works.

# %% [markdown]
# ## 8. Regional footprint
#
# Model availability differs sharply by Region. Pin your Region per workload.

# %%
REGIONS = ["us-east-1", "us-east-2", "us-west-2", "eu-central-1"]
inventory = {}
for reg in REGIONS:
    code, payload = get_json("/v1/models", region=reg)
    inventory[reg] = (
        sorted(m["id"] for m in payload.get("data", [])) if code == 200 else []
    )
    print(f"{reg:14} {len(inventory[reg]):3} models")

# %%
watch = [
    "anthropic.claude-opus-5",
    "anthropic.claude-haiku-4-5",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "google.gemma-4-31b",
    "xai.grok-4.3",
    "qwen.qwen3-32b",
]
print(f"{'model':30} " + "  ".join(f"{r:>13}" for r in REGIONS))
print("-" * 90)
for m in watch:
    cells = "  ".join(f"{('yes' if m in inventory[r] else '-'):>13}" for r in REGIONS)
    print(f"{m:30} {cells}")

# %% [markdown]
# Practical consequences, all visible above:
#
# - **us-east-1** is the only Region carrying the full Claude set (as of August 2026).
#   Regional availability changes; re-check with `GET /v1/models` per Region.
# - **us-east-2** has no Anthropic models at all.
# - **eu-central-1** is the thinnest: no Anthropic, no gpt-5.x, no xAI.
# - **gemma-4** is the only family present in all four.

# %% [markdown]
# ## 9. IAM — the permissions you need
#
# Attach the managed policy for inference:
#
# ```bash
# aws iam attach-role-policy --role-name YourRole \
#   --policy-arn arn:aws:iam::aws:policy/AmazonBedrockMantleInferenceAccess
# ```
#
# Or scope it yourself. Note that **SigV4 callers do not need
# `CallWithBearerToken`** — only bearer-token (API key) callers do.

# %% [markdown]
# > **The policy below is a DEVELOPMENT example.** It uses the service-scoped
# > wildcards `bedrock-mantle:Get*` and `bedrock-mantle:List*` for brevity. Do not
# > copy it into production: a wildcard grants any future action matching the
# > pattern, so a service release can silently widen your permissions. The
# > production variant in the next cell lists actions explicitly
# > (AWS Well-Architected SEC 3, least privilege).

# %%
inference_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "MantleInference",
            "Effect": "Allow",
            "Action": [
                "bedrock-mantle:CreateInference",
                "bedrock-mantle:Get*",
                "bedrock-mantle:List*",
            ],
            "Resource": "arn:aws:bedrock-mantle:*:*:project/*",
        },
        {
            # Only required when authenticating with a Bedrock API key.
            "Sid": "MantleBearerToken",
            "Effect": "Allow",
            "Action": "bedrock-mantle:CallWithBearerToken",
            "Resource": "*",
        },
        {
            # First call in a fresh account auto-subscribes via Marketplace.
            "Sid": "MarketplaceAutoSubscribe",
            "Effect": "Allow",
            "Action": [
                "aws-marketplace:Subscribe",
                "aws-marketplace:ViewSubscriptions",
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {"aws:CalledViaLast": "bedrock-mantle.amazonaws.com"}
            },
        },
    ],
}
print(json.dumps(inference_policy, indent=2))

# %% [markdown]
# ### The production variant — explicit actions, no wildcards
#
# Copy **this** one. Every action is named, so a future service release cannot
# widen the grant, and a reviewer can see exactly what the workload may do.

# %%
production_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "MantleInference",
            "Effect": "Allow",
            "Action": [
                "bedrock-mantle:CreateInference",
                "bedrock-mantle:GetModel",
                "bedrock-mantle:ListModels",
                "bedrock-mantle:GetProject",
                "bedrock-mantle:ListProjects",
            ],
            "Resource": "arn:aws:bedrock-mantle:*:*:project/*",
        },
        {
            # Only required when authenticating with a Bedrock API key.
            # Omit this statement entirely if you sign with SigV4.
            "Sid": "MantleBearerToken",
            "Effect": "Allow",
            "Action": "bedrock-mantle:CallWithBearerToken",
            "Resource": "*",
        },
    ],
}
print(json.dumps(production_policy, indent=2))
print("\nDrop the Marketplace statement once the account is subscribed.")

# %% [markdown]
# ### Governance: ban long-term API keys org-wide
#
# Short-term keys are fine (they're presigned SigV4 and expire). Long-term keys
# create a static IAM user credential. This SCP allows the former and blocks the
# latter:

# %%
scp = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyLongTermBedrockKeys",
            "Effect": "Deny",
            "Action": [
                "bedrock-mantle:CallWithBearerToken",
                "bedrock:CallWithBearerToken",
            ],
            "Condition": {
                "StringEquals": {"bedrock-mantle:bearerTokenType": "LONG_TERM"}
            },
            "Resource": "*",
        },
        {
            "Sid": "DenyCreatingLongTermKeys",
            "Effect": "Deny",
            "Action": "iam:CreateServiceSpecificCredential",
            "Condition": {
                "StringEquals": {
                    "iam:ServiceSpecificCredentialServiceName": "bedrock.amazonaws.com"
                }
            },
            "Resource": "*",
        },
    ],
}
print(json.dumps(scp, indent=2))
print(
    "\nNote: deny BOTH CallWithBearerToken actions to fully close off API-key access."
)

# %% [markdown]
# ## 10. Shared helper used by the rest of this collection
#
# The other notebooks import `_shared/mantle.py`, which wraps exactly what we
# built above (path resolution, token minting, retrying POST, streaming, TTFT
# (time-to-first-token)).

# %%
sys.path.insert(0, "../_shared")
import mantle

print("helper API surface:")
for fn in [
    "api_prefix",
    "base_url",
    "client",
    "anthropic_client",
    "post",
    "stream_lines",
    "err",
    "list_models",
    "families",
    "response_text",
    "function_calls",
    "ttft",
]:
    print("  mantle." + fn)

code, data = mantle.post(
    "/openai/v1/responses",
    {"model": "google.gemma-4-31b", "input": "Reply OK", "max_output_tokens": 16},
)
print("\nhelper smoke test:", code, repr(mantle.response_text(data)))

# %% [markdown]
# ## Gotchas from this notebook
#
# | Gotcha | Detail |
# |---|---|
# | No boto3 mantle client | Use the SDK to *sign*; there is no `client("bedrock-mantle")` |
# | Signing name | `bedrock` (not `bedrock-mantle`) — both work, `bedrock` is canonical |
# | Wrong path | `/openai/v1` vs `/v1` vs `/anthropic/v1` → 404/400 |
# | `/openai/v1/models` | 404. Inventory is only at `/v1/models` |
# | Token lifetime | Max 12h, **cannot be refreshed**; Region-pinned |
# | Long-term keys | Exploration only; they create a static IAM user credential |
# | Region drift | Claude ≈ us-east-1 only; eu-central-1 carries 33 of 55 models |
# | `CallWithBearerToken` | Needed for API keys, **not** for SigV4 |
#
# ## Next
# - `02-governance-projects-retention-and-observability.ipynb` — Projects, ZDR (zero
#   data retention),
#   CloudWatch
# - `03-scaling-tiers-and-latency.ipynb` — quotas, retries, service tiers, TTFT
# - Then jump to your model family: `../01-openai-gpt/`, `../02-anthropic-claude/`, etc.
