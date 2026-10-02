#!/usr/bin/env python3
"""Add §2b (the HTTP-200 UnknownOperationException) and the per-endpoint model IDs.

Both are new facts, both are the kind that cost real debugging time, and one of
them bites an example in AWS's own user guide.
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
# §2b — inserted after the api_prefix cell (index 7).
# ---------------------------------------------------------------------------
MD_2B = """## 2b. The 200 that means failure

Get the path wrong on `bedrock-mantle` and you get a 404 or a 400 with a message.
Get it wrong on `bedrock-runtime` and you can get **HTTP 200** — with this in the
body:

```json
{"Output":{"__type":"com.amazon.coral.service#UnknownOperationException"},"Version":"1.0"}
```

That is a routing failure wearing a success code. Any client shaped like

```python
if response.status_code == 200:
    answer = response.json()["choices"][0]["message"]["content"]   # KeyError
```

fails on the *parse*, several frames away from the actual mistake — a wrong URL.

This is worth knowing because it is easy to hit. The Chat Completions page in the
Amazon Bedrock User Guide shows a `bedrock-runtime` base URL of
`https://bedrock-runtime.{region}.amazonaws.com/v1` — without `/openai`. The
endpoints page is the one to trust: *"The OpenAI-compatible APIs are called on the
`/openai/v1` paths of this endpoint."* The cell below sends both so you can see
the difference rather than take our word for it.

**The rule:** on `bedrock-runtime`, treat a body containing `UnknownOperation` as
a failure regardless of status. `_shared/bedrock.py` exposes `unknown_op(payload)`
and `ok(status, payload)` for exactly this."""

MD_2B_TAIL = """Three things to take from that table:

1. `/openai/v1/...` is the served path on `bedrock-runtime`. `/v1/...` is not.
2. The wrong path returns **200**, and the right path can return **404** — when
   the model is real but does not serve that API. Status code alone tells you
   almost nothing here; the body tells you everything.
3. A 404 naming the *model* (`The model doesn't exist or doesn't support this
   API`) is a different problem from a 200 naming the *operation*. The first means
   "wrong model for this path", the second means "there is no such path"."""

CODE_2B = '''# Send the same request to the documented-but-unserved path and to the served
# one, and classify by BODY rather than by status code.
PROBE_PATHS = [
    ("/v1/chat/completions", "openai.gpt-oss-120b-1:0"),
    ("/openai/v1/chat/completions", "openai.gpt-oss-120b-1:0"),
    ("/v1/responses", "us.openai.gpt-5.6-sol"),
    ("/openai/v1/responses", "us.openai.gpt-5.6-sol"),
    ("/anthropic/v1/messages", "us.anthropic.claude-opus-5"),
]


def classify(path: str, model_id: str) -> tuple[str, str]:
    """Return (status, verdict) for one runtime path, reading the body."""
    if "responses" in path:
        payload = {"model": model_id, "input": "Hi", "max_output_tokens": 16}
    elif "messages" in path:
        payload = {
            "model": model_id,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hi"}],
        }
    else:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 16,
        }
    headers = {
        "Authorization": f"Bearer {provide_token(region=REGION)}",
        "Content-Type": "application/json",
    }
    if "messages" in path:
        headers["anthropic-version"] = "2023-06-01"
    req = urllib.request.Request(
        RUNTIME + path,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310  # noqa: S310
            status, body = resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:
        return "-", f"{type(exc).__name__}"

    if "UnknownOperation" in body:
        return str(status), "NO SUCH PATH (UnknownOperationException)"
    if status == 200:
        return str(status), "answered"
    try:
        message = json.loads(body).get("error", {}).get("message", "")
    except json.JSONDecodeError:
        message = body[:60]
    return str(status), message[:58]


print(f"{'path':32} {'HTTP':>5}  verdict")
print("-" * 92)
trap_hits = []
for path, model_id in PROBE_PATHS:
    status, verdict = classify(path, model_id)
    if "UnknownOperation" in verdict:
        trap_hits.append(path)
    print(f"{path:32} {status:>5}  {verdict}")

print()
if trap_hits:
    print(f"=> {len(trap_hits)} path(s) returned UnknownOperationException: {trap_hits}")
    codes = {classify(p, m)[0] for p, m in PROBE_PATHS if p in trap_hits}
    print(f"   ...with HTTP status {sorted(codes)}. A status-only check calls that")
    print("   a success. Read the body.")
else:
    print("=> no path returned UnknownOperationException today; the trap may be closed,")
    print("   but keep checking the body — it costs one `in` test.")'''

i = nb.insert_after(7, "markdown", MD_2B)
i = nb.insert_after(i, "code", CODE_2B)
nb.insert_after(i, "markdown", MD_2B_TAIL)

nb.save()
print(f"§2b inserted; {nb.changes} edits")

# ---------------------------------------------------------------------------
# §7 rewritten to cover BOTH endpoints, and to show the ID differences.
# Re-open so the indices reflect the inserts above.
# ---------------------------------------------------------------------------
nb = Notebook(NB)
idx = nb.find("## 7. Which API does each family actually support?")
assert len(idx) == 1, idx
sec7 = idx[0]

nb.set_source(
    sec7,
    """## 7. Which API does each family support — on which endpoint?

Don't assume; probe. And probe **per endpoint**, because the answer differs. On
`bedrock-mantle` the Responses API reaches a minority of families and Claude is
Messages-only. On `bedrock-runtime` the shape is different again: Chat Completions
is broad, Responses is narrow, and Messages serves only the newest Claude models.

One more wrinkle before the code: **the model ID is not the same on both
endpoints.** The next cell resolves it rather than assuming, because a mantle ID
sent to runtime returns *"The provided model identifier is invalid"* — which reads
like a missing model rather than a missing translation.""",
)

nb.set_source(
    sec7 + 1,
    '''from bedrock import runtime_id_for  # maps a mantle model ID to its runtime form

AV = {"anthropic-version": "2023-06-01"}


def _post(base: str, path: str, payload: dict, extra: dict | None = None):
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
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:
        return -1, f"{type(exc).__name__}"


def probe_apis(model_id: str, endpoint: str) -> dict:
    """Status per API for one model on one endpoint. 'ok' means a real 200."""
    base = HOST if endpoint == "mantle" else RUNTIME
    mid = model_id if endpoint == "mantle" else runtime_id_for(model_id, REGION)
    out = {"model": model_id, "id": mid or "-"}
    if mid is None:
        return {**out, "responses": "n/a", "chat": "n/a", "messages": "n/a"}

    prefix = api_prefix(mid, endpoint)

    def verdict(status, body):
        if "UnknownOperation" in body:
            return "no path"
        return "ok" if status == 200 else str(status)

    if prefix == "/anthropic/v1":
        out["responses"] = out["chat"] = "-"
        out["messages"] = verdict(
            *_post(
                base,
                f"{prefix}/messages",
                {
                    "model": mid,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                AV,
            )
        )
        return out

    out["messages"] = "-"
    out["responses"] = verdict(
        *_post(base, f"{prefix}/responses",
               {"model": mid, "input": "Hi", "max_output_tokens": 16})
    )
    # Try both budget field names before concluding an API is missing. gpt-5.6
    # serves Chat Completions but refuses `max_tokens`, and reading that 400 as
    # "no Chat Completions" is how a false claim once reached three notebooks.
    for field in ("max_tokens", "max_completion_tokens"):
        status, body = _post(
            base,
            f"{prefix}/chat/completions",
            {"model": mid, "messages": [{"role": "user", "content": "Hi"}], field: 16},
        )
        if status == 200:
            break
    out["chat"] = verdict(status, body)
    return out


REPRESENTATIVES = [
    "google.gemma-4-31b",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "xai.grok-4.3",
    "xai.grok-4.6",
    "anthropic.claude-opus-5",
    "qwen.qwen3-32b",
    "deepseek.v3.2",
]

for endpoint in ("mantle", "runtime"):
    print(f"=== {endpoint} " + "=" * 74)
    print(f"{'model asked for':26} {'id used there':32} {'Resp':>7} {'Chat':>7} {'Msg':>7}")
    print("-" * 84)
    for m in REPRESENTATIVES:
        row = probe_apis(m, endpoint)
        print(f"{row['model']:26} {row['id']:32} {row['responses']:>7} "
              f"{row['chat']:>7} {row['messages']:>7}")
    print()

renames = [
    (m, runtime_id_for(m, REGION))
    for m in REPRESENTATIVES
    if runtime_id_for(m, REGION) not in (None, m)
]
missing = [m for m in REPRESENTATIVES if runtime_id_for(m, REGION) is None]
print(f"=> {len(renames)} of {len(REPRESENTATIVES)} models are addressed by a "
      f"DIFFERENT id on runtime:")
for old, new in renames:
    print(f"     {old:26} -> {new}")
print(f"=> {len(missing)} are not on runtime at all: {missing}")''',
)

nb.set_source(
    sec7 + 2,
    """Read the error, not just the status code. Four different things arrive here and
they need four different fixes:

| What you see | What it means |
|---|---|
| `200` + `UnknownOperation` in the body | wrong **path** — there is no such operation |
| `404 The model doesn't exist or doesn't support this API` | right path, wrong **model** for it |
| `400 The model 'x' does not support the '/openai/v1/responses' API` | the model is real, the **API** is not available for it |
| `400 Unsupported parameter: 'max_tokens' …` | right path, right model, wrong **parameter** |
| `400 The provided model identifier is invalid` | you sent the **other endpoint's ID** |

The last one is new and it is the easy mistake to make: `openai.gpt-oss-120b` is a
perfectly good model ID *on mantle*, and on runtime it is invalid — the runtime ID
is `openai.gpt-oss-120b-1:0`. `runtime_id_for()` translates.

Concretely, from the tables above:

- **Chat Completions is the broad surface on both endpoints.** On mantle it is the
  universal one for open-weight families; on runtime it reaches most of the
  catalogue too.
- **Responses is narrow on runtime** — the GPT-5.6 profiles and Grok 4.6 today —
  and wider on mantle, where gpt-oss and Gemma 4 also serve it.
- **Claude** is Messages-only on both, and on runtime only the newest models
  answer, addressed by a `us.` or `global.` profile.
- **gpt-5.6** serves both Responses and Chat Completions on both endpoints, and
  rejects `max_tokens` in favour of `max_completion_tokens` on each. That 400 is
  easy to misread as the API being missing; `01-openai-gpt/01` §2b separates them.

Each family notebook in this collection leads with whichever endpoint and API
actually works for its models.""",
)

nb.save()
print(f"§7 rewritten; {nb.changes} edits")
