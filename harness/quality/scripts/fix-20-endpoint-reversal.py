#!/usr/bin/env python3
"""Flip the endpoint recommendation in 00-foundations/01 and fix the wrong rows.

AWS now says, verbatim: "For new applications, we recommend the bedrock-runtime
endpoint." This notebook said the opposite, and every other notebook inherits its
framing. Five rows of its comparison table are also now factually wrong.

Every replacement asserts its hit count, so a target that has moved fails loudly
rather than silently doing nothing.
"""
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbedit import Notebook  # noqa: E402
import os
REPO_ROOT = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

NB = REPO_ROOT + "/00-foundations/01-endpoints-auth-and-the-three-paths.ipynb"
nb = Notebook(NB)

# ---------------------------------------------------------------------------
# 1. The header. It told readers mantle was the thing to learn.
# ---------------------------------------------------------------------------
nb.sub(
    """**Start here.** Every other notebook in this collection assumes what this one
establishes. The three things worth knowing before you write a line of code:

1. `bedrock-mantle` is a **second, separate endpoint** next to `bedrock-runtime`.
   It speaks OpenAI- and Anthropic-compatible APIs.
2. There is **no `bedrock-mantle` client in boto3**. The AWS SDK is useful here
   as a *request signer*, not as a service client.
3. Model IDs map to **three different URL paths**. Guessing wrong gives you a
   404 or a 400, and this is the single most common source of confusion.

## What this notebook covers
- Endpoint choice: when `bedrock-mantle`, when `bedrock-runtime`
- Auth A: SigV4 with botocore · Auth B: short-term API key · Auth C: curl
- The three path families, demonstrated against live models
- Model discovery and the regional footprint
- The IAM permissions you actually need""",
    """**Start here.** Every other notebook in this collection assumes what this one
establishes. The four things worth knowing before you write a line of code:

1. **`bedrock-runtime` is the endpoint AWS recommends for new applications**, and
   since August 2026 it speaks all five APIs: InvokeModel, Converse, OpenAI
   Chat Completions, OpenAI Responses, and Anthropic Messages.
2. `bedrock-mantle` is a **second endpoint**, still fully supported, and it is
   where server-side tool use, `background=true`, Projects and Workspaces live.
3. **Neither endpoint's OpenAI-compatible APIs go through boto3.** You call them
   on URL paths. For those, the AWS SDK is useful as a *request signer* — or you
   use a Bedrock API key, which both endpoints accept.
4. A model's **URL path and even its model ID depend on which endpoint you use**.
   Guessing wrong gives a 400, a 404 — or, on `bedrock-runtime`, a **200 that
   means failure**. §2b is about that last one.

## What this notebook covers
- Endpoint choice: what each one serves, and what AWS recommends
- Auth A: SigV4 with botocore · Auth B: short-term API key · Auth C: curl
- The path families on **both** endpoints, demonstrated against live models
- The `UnknownOperationException` that arrives as HTTP 200
- Model discovery, the ID differences between endpoints, the regional footprint
- The IAM permissions you actually need""",
)

# ---------------------------------------------------------------------------
# 2. Section 1 heading text.
# ---------------------------------------------------------------------------
nb.sub(
    """## 1. Two endpoints, and why

Both endpoints can serve the *same* model at the *same* per-token price. You
pick based on the API surface and the features you need, not on cost.""",
    """## 1. Two endpoints, and why

Both endpoints can serve the *same* model at the *same* per-token price — AWS
states this explicitly: *"Per-token pricing for the same model is identical on
`bedrock-runtime` and `bedrock-mantle`. Choose an endpoint based on the APIs and
capabilities you need, not cost."* So this section is about capability, and the
table is built from what the endpoints answered today.

The one row that used to decide everything — which APIs each endpoint speaks —
changed in August 2026. `bedrock-runtime` picked up Chat Completions, Responses
and Messages, which is why AWS now recommends it as the default starting point.""",
)

# ---------------------------------------------------------------------------
# 3. The comparison table: five rows were wrong, and it was hardcoded.
#    Rebuild it so the API row is probed rather than asserted.
# ---------------------------------------------------------------------------
nb.set_source(
    4,
    '''# Rows marked (probed) are measured below in this cell rather than remembered.
# The API row is exactly the one that went stale, so it gets measured.
import urllib.error
import urllib.request

from aws_bedrock_token_generator import provide_token

RUNTIME_HOST = f"https://bedrock-runtime.{REGION}.amazonaws.com"


def _served(base: str, path: str, payload: dict, extra: dict | None = None) -> bool:
    """True when this endpoint really serves this path.

    Not `status == 200`: bedrock-runtime answers an unrecognised path with 200 and
    a Coral UnknownOperationException in the body. §2b demonstrates it.
    """
    headers = {
        "Authorization": f"Bearer {provide_token(region=REGION)}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310  # noqa: S310
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        # A model-level complaint proves the ROUTE exists; an unknown operation
        # proves it does not. That distinction is the whole point of this probe.
        return "UnknownOperation" not in body and exc.code != 404
    except Exception:
        return False
    return "UnknownOperation" not in body


# One request per API per endpoint, each with a model that endpoint accepts.
CHAT_MT = {"model": "openai.gpt-oss-120b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16}
CHAT_RT = {"model": "openai.gpt-oss-120b-1:0", "messages": [{"role": "user", "content": "Hi"}], "max_completion_tokens": 16}
RESP_MT = {"model": "openai.gpt-oss-120b", "input": "Hi", "max_output_tokens": 16}
RESP_RT = {"model": "us.openai.gpt-5.6-sol", "input": "Hi", "max_output_tokens": 16}
MSG_MT = {"model": "anthropic.claude-opus-5", "max_tokens": 16, "messages": [{"role": "user", "content": "Hi"}]}
MSG_RT = {"model": "us.anthropic.claude-opus-5", "max_tokens": 16, "messages": [{"role": "user", "content": "Hi"}]}
AV = {"anthropic-version": "2023-06-01"}

api_rows = [
    ("Chat Completions", _served(RUNTIME_HOST, "/openai/v1/chat/completions", CHAT_RT),
     _served(HOST, "/v1/chat/completions", CHAT_MT)),
    ("Responses", _served(RUNTIME_HOST, "/openai/v1/responses", RESP_RT),
     _served(HOST, "/v1/responses", RESP_MT)),
    ("Anthropic Messages", _served(RUNTIME_HOST, "/anthropic/v1/messages", MSG_RT, AV),
     _served(HOST, "/anthropic/v1/messages", MSG_MT, AV)),
]
mark = {True: "yes", False: "no"}

comparison = [
    ("URL", "bedrock-runtime.{r}.amazonaws.com", "bedrock-mantle.{r}.api.aws"),
    ("AWS recommends", "yes, for new applications", "fully supported"),
    ("InvokeModel / Converse", "yes", "no"),
]
comparison += [
    (f"{name} (probed)", mark[on_runtime], mark[on_mantle])
    for name, on_runtime, on_mantle in api_rows
]
comparison += [
    ("OpenAI-compatible path", "/openai/v1", "/openai/v1 or /v1, by family"),
    ("Auth", "SigV4 or Bedrock API key", "SigV4 or Bedrock API key"),
    ("OpenAI SDK drop-in", "yes (base_url + key)", "yes (base_url + key)"),
    ("Stateful chat", "store + previous_response_id", "store + previous_response_id"),
    ("Server-side tools", "no", "yes (incl. web search)"),
    ("Async (background=true)", "no (400)", "yes"),
    ("Cross-region (CRIS)", "geo/global/in profiles", "in-Region only"),
    ("App inference profiles", "Converse yes / OpenAI APIs no", "n/a"),
    ("Guardrails", "yes", "no"),
    ("Prompt caching", "yes", "per model"),
    ("Provisioned Throughput", "yes", "no"),
    ("Batch inference", "yes", "no (use bedrock-runtime)"),
    ("Quotas", "fixed RPM + TPM", "queued fair-share, no RPM"),
    ("CloudWatch namespace", "AWS/Bedrock", "AWS/BedrockMantle"),
    ("Cost attribution", "IAM principal, profiles", "Projects / Workspaces"),
]
w = 24
print(f"{'':{w}} {'bedrock-runtime':32} bedrock-mantle")
print("-" * 100)
for row in comparison:
    print(f"{row[0]:{w}} {row[1]:32} {row[2]}")

served_on_runtime = [name for name, rt, _ in api_rows if rt]
print(f"\\n=> bedrock-runtime served {len(served_on_runtime)}/3 of the "
      f"OpenAI- and Anthropic-compatible APIs today: {served_on_runtime}")
print("   Both endpoints are a base-URL change away from an OpenAI SDK codebase.")''',
)

# ---------------------------------------------------------------------------
# 4. The rule of thumb, reversed to match AWS.
# ---------------------------------------------------------------------------
nb.sub(
    """**Rule of thumb.** New applications → `bedrock-mantle`. Reach back to
`bedrock-runtime` when you specifically need cross-Region inference,
Provisioned Throughput, batch inference, or a model that isn't on mantle yet.""",
    """**Rule of thumb, as AWS states it.** New applications → **`bedrock-runtime`**.
From the
[endpoints page](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html):
*"For new applications, we recommend the `bedrock-runtime` endpoint."* It is also
where Guardrails, intelligent prompt routing and cross-Region inference live.

Reach for `bedrock-mantle` when you specifically need one of the things only it
has today:

- **server-side or pre-configured tool use**, including web search
- **asynchronous inference** (`background=true`) for long-running work
- **Projects or Workspaces**, to isolate workloads and attribute cost per
  application
- **a model that is only on mantle** — Gemma 4, GPT-5.4/5.5, Grok 4.3 and
  DeepSeek v3.1 are in that set today; §8 enumerates it live

If you already run `bedrock-mantle`, nothing is being taken away: AWS's wording
is that existing applications *"continue to be fully supported and do not need to
change."* And both endpoints can be used from the same application — pick per use
case, not once per project.

The collection reflects this: notebooks lead with whichever endpoint is right for
the model in front of them, and say which one they chose and why.""",
)

# ---------------------------------------------------------------------------
# 5. Section 2: paths are per-endpoint now.
# ---------------------------------------------------------------------------
nb.sub(
    """## 2. The three URL path families

This is the part that trips people up. A model's family determines its path:

| Path | Families |
|---|---|
| `/openai/v1/…` | `google.gemma-4*`, `openai.gpt-5*`, `xai.*` |
| `/v1/…` | `openai.gpt-oss*` and every Chat-Completions-only family |
| `/anthropic/v1/…` | `anthropic.*` only |

Control-plane paths (models, files, projects, fine-tuning, data retention)
are **always** under `/v1/…`, never `/openai/v1/…`.""",
    """## 2. URL paths — which depend on the endpoint, not just the model

This is the part that trips people up, and it got one level harder in August
2026: the path depends on **both** the model family and the endpoint.

On **`bedrock-mantle`**, three families:

| Path | Families |
|---|---|
| `/openai/v1/…` | `google.gemma-4*`, `openai.gpt-5*`, `xai.*` |
| `/v1/…` | `openai.gpt-oss*` and every Chat-Completions-only family |
| `/anthropic/v1/…` | `anthropic.*` only |

On **`bedrock-runtime`**, two — and the split falls somewhere else:

| Path | Families |
|---|---|
| `/openai/v1/…` | **every** OpenAI-compatible model, `gpt-oss` and `qwen` included |
| `/anthropic/v1/…` | `anthropic.*` only |

There is **no `/v1` inference surface on `bedrock-runtime` at all**. So the same
model moves paths when you move endpoint: `openai.gpt-oss-20b` is `/v1` on mantle,
and its runtime twin `openai.gpt-oss-20b-1:0` is `/openai/v1`.

Control-plane paths (models, files, projects, fine-tuning, data retention) are
mantle's, and always under `/v1/…`, never `/openai/v1/…`.""",
)

# ---------------------------------------------------------------------------
# 6. api_prefix() gains the endpoint argument, and the cell shows both answers.
# ---------------------------------------------------------------------------
nb.set_source(
    7,
    '''def api_prefix(model_id: str, endpoint: str = "mantle") -> str:
    """Which URL prefix serves this model's inference APIs, on this endpoint?

    Copy this with the `endpoint` argument. A version that takes only the model ID
    can only be right about one endpoint, and this collection shipped exactly that
    until bedrock-runtime grew the OpenAI-compatible paths.
    """
    if model_id.startswith("anthropic."):
        return "/anthropic/v1"
    if endpoint == "runtime":
        # Runtime puts every OpenAI-compatible model on /openai/v1.
        return "/openai/v1"
    if any(model_id.startswith(p) for p in ("google.gemma-4", "openai.gpt-5", "xai.")):
        return "/openai/v1"
    return "/v1"


RUNTIME = f"https://bedrock-runtime.{REGION}.amazonaws.com"

print(f"{'model':28} {'mantle':13} runtime")
print("-" * 60)
for m in [
    "google.gemma-4-31b",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "xai.grok-4.3",
    "anthropic.claude-sonnet-5",
    "qwen.qwen3-32b",
]:
    print(f"{m:28} {api_prefix(m):13} {api_prefix(m, 'runtime')}")

moved = [
    m
    for m in ("openai.gpt-oss-120b", "qwen.qwen3-32b", "google.gemma-4-31b")
    if api_prefix(m) != api_prefix(m, "runtime")
]
print(f"\\n=> {len(moved)} of those change path between endpoints: {moved}")
print("   Full URLs for one of them:")
print(f"     mantle : {HOST}{api_prefix('openai.gpt-oss-120b')}/chat/completions")
print(f"     runtime: {RUNTIME}{api_prefix('openai.gpt-oss-120b', 'runtime')}/chat/completions")''',
)

nb.save()
print(f"{NB}: {nb.changes} edits")
