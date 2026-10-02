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
# # Production hardening for Amazon Bedrock Mantle
#
# Everything to get right before a mantle workload carries real traffic. Each
# section is a control you can verify, not advice you have to take on trust.
#
# ## What this notebook covers
# - Credential lifecycle (short-term tokens, refresh, SigV4 (AWS Signature Version 4))
# - Retries, timeouts, and the failure modes that actually occur
# - Quota reality: no RPM, separate in/out TPM (tokens per minute), mostly unpublished
# - Data retention posture and ZDR (zero data retention)
# - Cost attribution with Projects
# - Observability, including the namespace that catches people out
# - Defensive output handling
# - A pre-launch checklist you can run
#
# ## Self-contained, but see also
# - **Auth and the three paths** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Governance and retention** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
# - **Quotas and tiers** → `../00-foundations/03-scaling-tiers-and-latency.ipynb`
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

# %%
import concurrent.futures as cf
import json
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "../_shared")
from mantle import err, list_models, parse_json_lenient, post, response_text

REGION = "us-east-1"
MODEL = "google.gemma-4-31b"  # present in all four mantle Regions
PREFIX = "/openai/v1"
print("region:", REGION, "| model:", MODEL)

# %% [markdown]
# ## 1. Credentials — mint short, refresh often, never store
#
# Short-term Bedrock API keys are presigned SigV4 requests: they *are* IAM. They
# expire within 12 hours, **cannot be refreshed**, and are Region-pinned.
#
# Three rules:
# 1. Mint in-process from the ambient role. Do not put a 15-minute secret in a
#    secrets manager.
# 2. Keep the TTL short. The token is your role until it expires.
# 3. Never ship long-term keys — they create a static IAM user credential.

# %%
import threading
from datetime import datetime, timedelta, timezone

from aws_bedrock_token_generator import provide_token


class TokenProvider:
    """Thread-safe short-term token cache with early refresh."""

    def __init__(self, region=REGION, ttl=timedelta(minutes=15), skew_s=120):
        self.region, self.ttl, self.skew_s = region, ttl, skew_s
        self._token = None
        self._expires_at = None
        self._lock = threading.Lock()
        self.mints = 0

    def get(self) -> str:
        now = datetime.now(timezone.utc)
        with self._lock:  # avoid a thundering herd
            if self._token and self._expires_at and now < self._expires_at:
                return self._token
            self._token = provide_token(region=self.region, expiry=self.ttl)
            self._expires_at = now + self.ttl - timedelta(seconds=self.skew_s)
            self.mints += 1
            return self._token


tokens = TokenProvider()
with cf.ThreadPoolExecutor(max_workers=8) as pool:
    values = list(pool.map(lambda _: tokens.get(), range(8)))
print(
    f"8 concurrent callers -> {tokens.mints} mint(s), "
    f"all identical: {len(set(values)) == 1}"
)
print("expires around:", tokens._expires_at.isoformat(timespec="seconds"))

# %%
# For roles with no need of a bearer token at all, SigV4-sign directly.
# Signing name is "bedrock". SigV4 callers do NOT need CallWithBearerToken.
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.httpsession import URLLib3Session

body = json.dumps({"model": MODEL, "input": "Reply OK", "max_output_tokens": 16})
creds = boto3.Session().get_credentials().get_frozen_credentials()
request = AWSRequest(
    method="POST",
    url=f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}/responses",
    data=body,
    headers={"Content-Type": "application/json"},
)
SigV4Auth(creds, "bedrock", REGION).add_auth(request)
resp = URLLib3Session(timeout=60).send(request.prepare())
print("SigV4 path ->", resp.status_code)

# %% [markdown]
# ## 2. Retries, timeouts, and what actually fails
#
# Mantle has **no RPM quota**; throttling is token-based, and most models have no
# published TPM at all — capacity is internal fair-share. So `429` and `5xx` are
# ordinary operating conditions, not exceptions.
#
# Equally important: a **wrong path can stall** rather than return an error. Without
# a client timeout, a retry loop turns that into a multi-minute hang.

# %%
TRANSIENT = {429, 500, 502, 503, 504}


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


def resilient_call(path, body, *, region=REGION, headers=None, attempts=5, timeout=60):
    """Retry transient failures; fail fast on client errors; always time out."""
    url = f"https://bedrock-mantle.{region}.api.aws{path}"
    payload = json.dumps(body).encode()
    for attempt in range(attempts):
        hdrs = {
            "Authorization": f"Bearer {tokens.get()}",
            "Content-Type": "application/json",
            **(headers or {}),
        }
        req = urllib.request.Request(url, data=payload, headers=hdrs, method="POST")
        try:
            with open_https(req, timeout=timeout) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in TRANSIENT and attempt < attempts - 1:
                time.sleep(
                    # Jitter spreads retries; not a security decision.
                    min(2**attempt, 16)
                    + random.random()  # nosec B311
                )
                continue
            return exc.code, json.loads(exc.read() or b"{}")
        except Exception as exc:  # timeout, connection reset
            if attempt < attempts - 1:
                time.sleep(
                    # Jitter spreads retries; not a security decision.
                    min(2**attempt, 16)
                    + random.random()  # nosec B311
                )
                continue
            return -1, {"error": {"message": f"{type(exc).__name__}"}}
    return -1, {"error": {"message": "retries exhausted"}}


code, data = resilient_call(
    f"{PREFIX}/responses",
    {"model": MODEL, "input": "Reply OK", "max_output_tokens": 16},
)
print("healthy call ->", code, repr(response_text(data)[:30]))

# %%
# A permanent 400 must not be retried — measure that it fails fast.
started = time.perf_counter()
code, data = resilient_call(
    f"{PREFIX}/responses",
    {"model": MODEL, "input": "Hi", "max_output_tokens": 16, "top_p": 0.95},
)
print(
    f"invalid param -> HTTP {code} in {time.perf_counter() - started:.2f}s "
    f"(no retries — correct)"
)
print("message:", err(data)[:90])

# %%
# The stall case, contained by a short timeout. Grok on the bare /v1 responses path
# does not answer at all.
started = time.perf_counter()
code, data = post(
    "/v1/responses",
    {"model": "xai.grok-4.3", "input": "Hi", "max_output_tokens": 16},
    region=REGION,
    attempts=1,
    timeout=20,
)
print(
    f"stalling path -> {code} in {time.perf_counter() - started:.1f}s "
    f"(bounded by the timeout, not by the server)"
)

# %% [markdown]
# ## 3. Concurrency: ramp, don't spike
#
# A cold start from zero to peak concurrency gets shed. Step up gradually.


# %%
def one_call(i):
    code, _ = resilient_call(
        f"{PREFIX}/responses",
        {"model": MODEL, "input": f"Say OK ({i})", "max_output_tokens": 16},
    )
    return code


for concurrency in (1, 4, 8):
    started = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        codes = list(pool.map(one_call, range(concurrency)))
    elapsed = time.perf_counter() - started
    ok = sum(1 for c in codes if c == 200)
    print(f"concurrency {concurrency:2} -> {ok}/{concurrency} ok in {elapsed:5.2f}s")

print("\nIn production, step concurrency up over minutes and keep the backoff loop.")

# %% [markdown]
# ## 4. Data retention posture
#
# Two independent controls. Get both explicit.
#
# | Control | Scope | Default |
# |---|---|---|
# | `store` | per request (Responses) | **`true`** — 30-day retention |
# | `data_retention.mode` | account / project / model | `inherit` |
#
# Modes: `default`, `none` (zero data retention), `provider_data_share`, `inherit`.

# %%
code, retention = post("/v1/data_retention", None, region=REGION, method="GET")
print("account retention:", retention)

# store defaults to true — verify by omitting it.
code, implicit = post(
    f"{PREFIX}/responses",
    {"model": MODEL, "input": "Reply OK", "max_output_tokens": 16},
    region=REGION,
)
code, explicit = post(
    f"{PREFIX}/responses",
    {"model": MODEL, "input": "Reply OK", "max_output_tokens": 16, "store": False},
    region=REGION,
)
print(f"store omitted   -> store={implicit.get('store')}  (retained 30 days)")
print(f"store=False     -> store={explicit.get('store')}")

# %%
# A retrievable response proves retention is real.
retrievable = post(
    f"{PREFIX}/responses",
    {
        "model": MODEL,
        "input": "Remember: alpha.",
        "max_output_tokens": 16,
        "store": True,
    },
    region=REGION,
)[1]
code, fetched = post(
    f"{PREFIX}/responses/{retrievable['id']}", None, region=REGION, method="GET"
)
print(f"GET stored response -> {code} (status={fetched.get('status')})")
code, _ = post(
    f"{PREFIX}/responses/{retrievable['id']}", None, region=REGION, method="DELETE"
)
print(f"DELETE it           -> {code}")

# %%
# Some models are gated by retention mode. Check before you debug your request.
for mid in (MODEL, "anthropic.claude-fable-5"):
    code, info = post(f"/v1/models/{mid}", None, region=REGION, method="GET")
    if code != 200:
        print(f"{mid:34} GET -> {code}")
        continue
    dr = info.get("data_retention", {})
    print(
        f"{mid:34} status={info.get('status'):12} "
        f"allowed_modes={dr.get('allowed_modes')}"
    )
    if info.get("status_reason"):
        print(f"      reason: {info['status_reason'][:100]}")

# %% [markdown]
# ## 5. Cost attribution with Projects
#
# Every production workload should run under its own tagged project. Untagged usage
# is impossible to allocate later.

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "hardening-demo",
        "tags": {
            "Application": "HardeningDemo",
            "Environment": "Demo",
            "Owner": "PlatformTeam",
            "CostCenter": "0000",
        },
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id, "| tags:", project.get("tags"))

# The attribution header differs by API — a classic mistake.
for label, path, body, header in [
    (
        "Responses",
        f"{PREFIX}/responses",
        {"model": MODEL, "input": "Reply OK", "max_output_tokens": 16},
        {"OpenAI-Project": project_id},
    ),
    (
        "ChatCompletions",
        "/v1/chat/completions",
        {
            "model": "qwen.qwen3-32b",
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_tokens": 16,
        },
        {"OpenAI-Project": project_id},
    ),
    (
        "Messages",
        "/anthropic/v1/messages",
        {
            "model": "anthropic.claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply OK"}],
        },
        {"anthropic-workspace": project_id, "anthropic-version": "2023-06-01"},
    ),
]:
    code, _ = post(path, body, region=REGION, headers=header)
    used = [k for k in header if k != "anthropic-version"][0]
    print(f"  {label:16} via {used:22} -> {code}")

# %% [markdown]
# ## 6. Observability — the namespace trap
#
# Mantle publishes to **`AWS/BedrockMantle`**, not `AWS/Bedrock`. A dashboard built
# on the wrong namespace shows **zero errors during an incident**. And the only error
# metric is `InferenceClientErrors` (4xx) — there is **no server-error metric**, so
# 503s must come from your own telemetry.

# %%
cw = boto3.client("cloudwatch", region_name=REGION)
for namespace in ("AWS/BedrockMantle", "AWS/Bedrock"):
    metrics = sorted(
        {
            m["MetricName"]
            for m in cw.list_metrics(Namespace=namespace).get("Metrics", [])
        }
    )
    print(f"{namespace:22} {len(metrics):2} metrics: {', '.join(metrics[:6])}")
print("\nNote the absence of any 5xx/server-error metric in AWS/BedrockMantle.")


# %%
# Because 5xx is invisible server-side, record it client-side.
class CallMetrics:
    """Minimal client-side telemetry — the only place 5xx is visible."""

    def __init__(self):
        self.counts = {
            "ok": 0,
            "throttled": 0,
            "server_error": 0,
            "client_error": 0,
            "timeout": 0,
        }
        self.latencies = []

    def record(self, code, seconds):
        self.latencies.append(seconds)
        if code == 200:
            self.counts["ok"] += 1
        elif code == 429:
            self.counts["throttled"] += 1
        elif code == -1:
            self.counts["timeout"] += 1
        elif 500 <= code < 600:
            self.counts["server_error"] += 1
        else:
            self.counts["client_error"] += 1

    def report(self):
        if not self.latencies:
            return "no calls"
        ordered = sorted(self.latencies)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
        return (
            f"{self.counts} | p50={p50:.2f}s p95={p95:.2f}s " f"n={len(self.latencies)}"
        )


metrics = CallMetrics()
for i in range(6):
    started = time.perf_counter()
    code, _ = resilient_call(
        f"{PREFIX}/responses",
        {"model": MODEL, "input": f"Reply OK ({i})", "max_output_tokens": 16},
        headers={"OpenAI-Project": project_id},
    )
    metrics.record(code, time.perf_counter() - started)
print(metrics.report())

# %% [markdown]
# **CloudTrail:** mantle inference is a **data event** — off by default, and billed
# extra when enabled. Management events (creating projects) appear normally. Note
# that short-term key *generation* is client-side and never logged.

# %% [markdown]
# ## 7. Defensive output handling
#
# HTTP 200 does not mean you got what you asked for. Three real cases:
#
# 1. **Truncation** — `finish_reason="length"` with empty content.
# 2. **Trailing characters** after valid JSON, even in strict mode.
# 3. **Ignored constraints** — a forced tool choice that returns prose.

# %%
SCHEMA = {
    "type": "object",
    "properties": {"language": {"type": "string"}, "typed": {"type": "boolean"}},
    "required": ["language", "typed"],
    "additionalProperties": False,
}


def safe_structured(
    model, prompt, schema, *, prefix=PREFIX, max_output_tokens=600, attempts=3
):
    """Structured output with truncation detection, lenient parsing and retries."""
    budget = max_output_tokens
    for attempt in range(attempts):
        code, data = post(
            f"{prefix}/responses",
            {
                "model": model,
                "input": prompt,
                "max_output_tokens": budget,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "out",
                        "schema": schema,
                        "strict": True,
                    }
                },
                "store": False,
            },
            region=REGION,
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        text = response_text(data)
        if not text.strip():
            budget *= 3  # truncated before emitting anything
            print(f"   attempt {attempt + 1}: empty output, raising budget to {budget}")
            continue
        try:
            parsed = parse_json_lenient(text)  # tolerates trailing characters
        except ValueError as exc:
            print(f"   attempt {attempt + 1}: unparseable ({exc})")
            continue
        missing = set(schema["required"]) - set(parsed)
        if missing:
            print(f"   attempt {attempt + 1}: missing keys {missing}")
            continue
        return parsed
    raise RuntimeError("no valid structured output after retries")


print("structured with guards:")
print("  ", safe_structured(MODEL, "Describe the Go programming language.", SCHEMA))

# %%
# Show the trailing-character problem that motivates lenient parsing.
raw_samples = [
    '{"language":"Go","typed":true}',
    '{"language":"Go","typed":true}\n}',
    '{"language":"Go","typed":true}\nextra text',
]
for sample in raw_samples:
    try:
        json.loads(sample)
        verdict = "json.loads OK"
    except json.JSONDecodeError:
        verdict = "json.loads FAILS"
    print(f"  {verdict:18} | lenient -> {parse_json_lenient(sample)} | {sample!r}")

# %% [markdown]
# ## 8. Region and model availability as a pre-flight check
#
# Verify at startup that every model you depend on exists in your Region. Failing
# here is much better than failing on first traffic.

# %%
REQUIRED = [MODEL, "qwen.qwen3-32b", "anthropic.claude-haiku-4-5", "openai.gpt-5.6-sol"]


def preflight(required, region=REGION):
    """Fail fast at startup if a dependency is missing from this Region."""
    available = set(list_models(region))
    missing = [m for m in required if m not in available]
    return {
        "region": region,
        "ok": not missing,
        "missing": missing,
        "available_count": len(available),
    }


for region in ("us-east-1", "eu-central-1"):
    result = preflight(REQUIRED, region)
    status = "PASS" if result["ok"] else "FAIL"
    print(
        f"{region:14} {status}  ({result['available_count']} models) "
        f"missing={result['missing']}"
    )

# %% [markdown]
# ## 9. A production client, assembled
#
# Every control from above in one place.


# %%
class ProductionClient:
    """Hardened bedrock-mantle client."""

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

    def __init__(self, model, region=REGION, project=None, tier="default"):
        self.model, self.region, self.project = model, region, project
        self.tier = (
            tier if (tier == "default" or model.startswith(self.TIERED)) else "default"
        )
        self.tokens = TokenProvider(region=region)
        self.metrics = CallMetrics()
        self.prefix = (
            "/anthropic/v1"
            if model.startswith("anthropic.")
            else (
                "/openai/v1"
                if model.startswith(("google.gemma-4", "openai.gpt-5", "xai."))
                else "/v1"
            )
        )

    def _sampling(self, temperature, top_p):
        out = {}
        if temperature is not None and not self.model.startswith(self.NO_TEMPERATURE):
            out["temperature"] = temperature
        if top_p is not None and not self.model.startswith(self.NO_TOP_P):
            out["top_p"] = top_p
        return out

    def ask(
        self,
        prompt,
        *,
        max_output_tokens=512,
        temperature=None,
        top_p=None,
        schema=None,
    ):
        body = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max(16, max_output_tokens),  # Responses min is 16
            "service_tier": self.tier,
            "store": False,  # explicit: no 30-day retention
            **self._sampling(temperature, top_p),
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

        started = time.perf_counter()
        code, data = post(
            f"{self.prefix}/responses",
            body,
            region=self.region,
            headers=headers,
            timeout=90,
        )
        self.metrics.record(code, time.perf_counter() - started)
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")

        text = response_text(data)
        if schema:
            if not text.strip():
                raise RuntimeError("empty output — raise max_output_tokens")
            return parse_json_lenient(text)
        return text


bot = ProductionClient(MODEL, project=project_id, tier="flex")
# %% [markdown]
# Exercise it, then read the metrics it recorded.

# %%
print("plain     :", bot.ask("Name one benefit of fair-share scheduling.")[:120])
print(
    "structured:",
    bot.ask("Describe the Rust language.", schema=SCHEMA, max_output_tokens=400),
)
print("metrics   :", bot.metrics.report())

# %% [markdown]
# ## 10. Pre-launch checklist
#
# Run through this before your first real traffic.

# %%
CHECKLIST = [
    ("Short-term tokens minted in-process, TTL <= 15 min", True),
    ("No long-term API keys anywhere in the deployment", True),
    ("Retry with exponential backoff on 429 and 5xx", True),
    ("Fail fast (no retries) on other 4xx", True),
    ("Client-side timeout on every call", True),
    ("Concurrency ramps over minutes, not seconds", True),
    ("store=False unless server-side state is required", True),
    ("data_retention mode chosen deliberately (none for regulated data)", True),
    ("Every workload runs under a tagged Project", True),
    ("Dashboards point at AWS/BedrockMantle, not AWS/Bedrock", True),
    ("5xx tracked client-side (no server metric exists)", True),
    ("Structured output parsed leniently and validated", True),
    ("Sampling params gated per model", True),
    ("service_tier gated per model", True),
    ("Region pre-flight check for every required model", True),
    ("Quota escalation path known (Support case, not Service Quotas)", True),
]
width = max(len(item) for item, _ in CHECKLIST)
for item, done in CHECKLIST:
    print(f"  [{'x' if done else ' '}] {item:{width}}")
print(
    f"\n{sum(1 for _, d in CHECKLIST if d)}/{len(CHECKLIST)} controls implemented "
    f"in this notebook"
)

# %%
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("cleaned up demo project:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — production on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | Token lifetime | ≤12 h, **not refreshable**, Region-pinned — mint and cache |
# | No RPM quota | Token-based throttling only; most models have no published TPM |
# | Quota increases | AWS Support case, **not** the Service Quotas console |
# | Wrong path may stall | Always set a client timeout; bound your retries |
# | `store` defaults to true | 30-day retention unless you opt out per request |
# | Retention gating | A model can be `unavailable` under your retention mode |
# | CloudWatch namespace | `AWS/BedrockMantle` — the wrong one shows zero errors |
# | No 5xx metric | Track server errors from client-side telemetry |
# | CloudTrail | Inference = data events (opt-in, extra cost) |
# | Short-term key minting | Never logged — client-side by design |
# | 200 ≠ correct | Truncation, trailing characters, ignored constraints all give 200 |
# | Per-model params | Sampling and tier support vary — gate them |
# | No CRIS / PT / batch | Cross-Region, Provisioned Throughput and batch are runtime-only |
#
# ## Where next
# - `01-choosing-a-model-and-api.ipynb` — the live capability survey
# - `02-migrating-from-openai.ipynb` — porting an existing codebase
# - `../00-foundations/` — the underlying mechanics in depth
