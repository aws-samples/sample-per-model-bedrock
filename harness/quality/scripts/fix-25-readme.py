#!/usr/bin/env python3
"""Update the README for the August 2026 endpoint changes and Grok 4.6.

Wrong or stale before this change:
  - the endpoint table claimed runtime serves only Converse + InvokeModel
  - it claimed mantle is bearer-token-only (both endpoints take either auth)
  - it gave no recommendation, where AWS now has one
  - the 11-xai-grok row listed only grok-4.3 and said "mantle"
  - the guardrails wording said the header is ignored on "the OpenAI APIs",
    which is true of Responses and false of Chat Completions and Messages
"""
import pathlib
import os
REPO_ROOT = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

README = pathlib.Path(
    REPO_ROOT + "/README.md")
text = README.read_text()
original = text


def swap(old: str, new: str) -> None:
    global text
    if old not in text:
        raise AssertionError(f"README: not found: {old[:80]!r}")
    if text.count(old) != 1:
        raise AssertionError(f"README: {text.count(old)} matches for {old[:60]!r}")
    text = text.replace(old, new)


# ---------------------------------------------------------------------------
swap(
    """## Two endpoints, and which to use

Bedrock serves models through two inference endpoints. **Which one you use is a
per-model fact, not a preference.**

| | `bedrock-runtime` | `bedrock-mantle` |
|---|---|---|
| APIs | Converse, InvokeModel | OpenAI Responses, OpenAI Chat Completions, Anthropic Messages |
| Auth | SigV4 via the AWS SDK | Bearer token (short-term Bedrock API key) |
| Guardrails | `guardrailConfig` on Converse, `guardrailIdentifier` on InvokeModel | **not supported** — screen text yourself with `ApplyGuardrail` |
| Cross-Region inference | geographic and global profiles | in-Region only |
| Reach for it when | you want one AWS-native shape across providers, or need Converse-only features | you have existing OpenAI- or Anthropic-shaped code to move |

Some models are on both, some on only one. Every family table below names the
endpoints for that family, and `_shared/bedrock.py` exposes `endpoints_for(model_id)`
so a notebook can ask the service instead of trusting a table that ages.""",
    """## Two endpoints, and which to use

Bedrock serves models through two inference endpoints, and **AWS recommends
`bedrock-runtime` for new applications**. From the
[endpoints page](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html):
*"For new applications, we recommend the `bedrock-runtime` endpoint."*

Since August 2026 `bedrock-runtime` speaks all five APIs, so the old split — "AWS
shapes here, OpenAI shapes there" — no longer holds:

| | `bedrock-runtime` (recommended) | `bedrock-mantle` |
|---|---|---|
| APIs | Converse, InvokeModel, **OpenAI Chat Completions, OpenAI Responses, Anthropic Messages** | OpenAI Responses, OpenAI Chat Completions, Anthropic Messages |
| OpenAI-compatible path | `/openai/v1` for every model | `/openai/v1` or `/v1`, by family |
| Auth | SigV4 **or** short-term Bedrock API key | SigV4 **or** short-term Bedrock API key |
| Stateful chat | `store` + `previous_response_id` | `store` + `previous_response_id` |
| Server-side tools (incl. web search) | no | **yes** |
| Async inference (`background=true`) | no (400) | **yes** |
| Projects / Workspaces | default project only | **yes** |
| Cross-Region inference | geographic, global and `in.` profiles | in-Region only |
| Guardrails | **yes** — but not on every API; see below | no |
| Provisioned Throughput · batch | yes | no |
| Quotas | fixed RPM + TPM | queued fair-share, no RPM |
| Cost attribution | IAM principal, application inference profiles | Projects / Workspaces |

Per-token pricing for the same model is **identical on both**, so this is a
capability choice, not a cost one. Existing `bedrock-mantle` applications are, in
AWS's words, *"fully supported and do not need to change"* — and both endpoints can
be used from one application, chosen per use case.

Reach for `bedrock-mantle` when you need server-side tool use, asynchronous
inference, Projects or Workspaces, or a model that is only there. Twelve of the 55
models on mantle in `us-east-1` had no `bedrock-runtime` twin when this was written
— Gemma 4, GPT-5.4, GPT-5.5, Grok 4.3, DeepSeek v3.1 and GLM 4.6 among them.

Three things about this that cost real debugging time, all covered in
[`00-foundations/01`](00-foundations/01-endpoints-auth-and-the-three-paths.ipynb):

1. **The same model often has a different ID on each endpoint.** `openai.gpt-oss-20b`
   on mantle is `openai.gpt-oss-20b-1:0` on runtime; `moonshotai.kimi-k2-thinking`
   becomes `moonshot.kimi-k2-thinking`; Claude and the GPT-5.6 and Grok 4.6 profiles
   need a `us.` or `global.` prefix. Sending the wrong one gives *"The provided model
   identifier is invalid"*, which reads like a missing model.
   `runtime_id_for(model_id)` translates.
2. **The URL path depends on the endpoint too.** There is no `/v1` inference surface
   on `bedrock-runtime` at all.
3. **`bedrock-runtime` answers an unserved path with HTTP 200** and a Coral
   `UnknownOperationException` in the body — so `if status == 200` reads a wrong URL
   as a success. Use `ok(status, body)` from `_shared/bedrock.py`.

Every family table below names the endpoints for that family, and
`_shared/bedrock.py` exposes `endpoints_for(model_id)` and `runtime_id_for(model_id)`
so a notebook can ask the service instead of trusting a table that ages.""",
)

# ---------------------------------------------------------------------------
swap(
    "| [`11-xai-grok/`](11-xai-grok/) | grok-4.3 | **mantle** | Responses **and** "
    "Chat Completions | Core inference, always-on reasoning, encrypted reasoning "
    "content |",
    "| [`11-xai-grok/`](11-xai-grok/) | grok-4.6 · grok-4.3 | 4.6 **both** "
    "(runtime: profile-only; mantle: `us-west-2` only) · 4.3 **mantle** | Responses "
    "**and** Chat Completions · Converse | **Grok 4.6**: the effort dial measured, "
    "encrypted reasoning replayed, structured output the model card says is absent · "
    "Grok 4.3: always-on reasoning |",
)

swap(
    "| [`05-deepseek/`](05-deepseek/) | v3.2, v3.1 | both · v3.1 **mantle** |",
    "| [`05-deepseek/`](05-deepseek/) | v3.2, v3.1 | v3.2 both · v3.1 **mantle** |",
)

# ---------------------------------------------------------------------------
swap(
    """Choosing between families, or already have OpenAI code?
[`99-cross-cutting/`](99-cross-cutting/) has a live capability survey, a migration
guide with a self-healing compatibility shim, and a pre-launch checklist that covers
**Guardrails** — including why they are not a `bedrock-mantle` parameter and why the
header that looks like it should work on the OpenAI APIs is accepted and silently
ignored.""",
    """Choosing between families, or already have OpenAI code?
[`99-cross-cutting/`](99-cross-cutting/) has a live capability survey, a migration
guide with a self-healing compatibility shim, and a pre-launch checklist.

That checklist includes a result worth calling out here, because *"Guardrails are
supported on `bedrock-runtime`"* is true and still not enough to build on. The
guardrail header is **enforced** on runtime Chat Completions and runtime Messages,
and **accepted and silently ignored** on runtime Responses and on every
`bedrock-mantle` surface. Nothing in the response distinguishes the two, so
[`99-cross-cutting/03`](99-cross-cutting/03-production-hardening.ipynb) §9b measures
every attachment point two independent ways — a denied topic no model refuses on its
own, and a guardrail ID that does not exist. Where the header is ignored, call
`ApplyGuardrail` explicitly or move the call to Converse.""",
)

# ---------------------------------------------------------------------------
swap(
    "| [`01-endpoints-auth-and-the-three-paths`](00-foundations/01-endpoints-auth-and-the-three-paths.ipynb) | SigV4 · short-term API keys · curl · the three URL paths · model discovery · IAM |",
    "| [`01-endpoints-auth-and-the-three-paths`](00-foundations/01-endpoints-auth-and-the-three-paths.ipynb) | Endpoint choice · SigV4 · short-term API keys · curl · the URL paths on **both** endpoints · the 200 that means failure · per-endpoint model IDs · model discovery · IAM |",
)
swap(
    "| [`04-bedrock-runtime-converse-and-profiles`](00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb) | SigV4 · Converse · content blocks · inference profiles (CRIS) · reading the catalogue · streaming · InvokeModel |",
    "| [`04-bedrock-runtime-converse-and-profiles`](00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb) | SigV4 · Converse · content blocks · inference profiles (CRIS) · reading the catalogue · streaming · InvokeModel · **the OpenAI- and Anthropic-compatible APIs on this endpoint** · server-side conversation state |",
)

# ---------------------------------------------------------------------------
swap(
    """**On `bedrock-mantle`** the OpenAI and Anthropic SDKs expect a bearer token, so mint a
short-term Bedrock API key from the same credentials:

```python
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

client = OpenAI(
    api_key=provide_token(region="us-east-1"),   # expires in <= 12 hours
    base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
)
response = client.responses.create(
    model="google.gemma-4-31b", input="Hello", max_output_tokens=64
)
print(response.output_text)
```

Two things in that first snippet catch people out, and
[`00-foundations/`](00-foundations/) covers both: most Claude models are rejected by
their bare model ID and require the `us.` inference-profile form, and the base URL
prefix on `bedrock-mantle` differs by model family.""",
    """**For the OpenAI-shaped APIs** the SDK expects a bearer token, so mint a short-term
Bedrock API key from the same credentials. This is the recommended endpoint:

```python
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

client = OpenAI(
    api_key=provide_token(region="us-east-1"),   # expires in <= 12 hours
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
)
response = client.responses.create(
    model="us.openai.gpt-5.6-sol", input="Hello", max_output_tokens=2048
)
print(response.output_text)
```

The same code against `bedrock-mantle` needs two changes — the host, and the model
ID, which has no profile prefix there:

```python
client = OpenAI(
    api_key=provide_token(region="us-east-1"),
    base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
)
response = client.responses.create(
    model="openai.gpt-5.6-sol", input="Hello", max_output_tokens=2048
)
```

Four things in those snippets catch people out, and
[`00-foundations/`](00-foundations/) covers all four: most Claude models plus the
GPT-5.6 and Grok 4.6 profiles are rejected by their bare model ID and require the
`us.` form; the base URL prefix on `bedrock-mantle` differs by model family while
`bedrock-runtime` uses `/openai/v1` for everything; reasoning models need a generous
`max_output_tokens` or they return an empty string with HTTP 200; and a wrong path on
`bedrock-runtime` also returns HTTP 200.""",
)

# ---------------------------------------------------------------------------
swap(
    """Least-privilege IAM: `bedrock:InvokeModel` and
`bedrock:InvokeModelWithResponseStream` cover Converse on `bedrock-runtime`; attach
`AmazonBedrockMantleInferenceAccess` for `bedrock-mantle` inference, and
`AmazonBedrockMantleFullAccess` only if you want to create Projects. The model- and
profile-discovery cells also need `bedrock:ListFoundationModels` and
`bedrock:ListInferenceProfiles`.""",
    """Least-privilege IAM: `bedrock:InvokeModel` and
`bedrock:InvokeModelWithResponseStream` cover `bedrock-runtime`, including its
OpenAI- and Anthropic-compatible paths; attach
`AmazonBedrockMantleInferenceAccess` for `bedrock-mantle` inference, and
`AmazonBedrockMantleFullAccess` only if you want to create Projects. The model- and
profile-discovery cells also need `bedrock:ListFoundationModels` and
`bedrock:ListInferenceProfiles`.

One easy-to-miss grant: calling a model through an inference profile on
`bedrock-runtime` needs `bedrock:InvokeModel` on **your account's default project**
(`arn:aws:bedrock:{region}:{account-id}:project/default`) as well as on the profile.
Without it you get an `AccessDeniedException` that names the *model*, which sends you
looking at model access instead. See
[`11-xai-grok/02`](11-xai-grok/02-grok-4-6.ipynb) §14.""",
)

# ---------------------------------------------------------------------------
swap(
    """It also moves. Gemma 4's parameter surface tightened in August 2026 and was later
relaxed again; Grok's did the same. Every notebook therefore **probes** the endpoint in
front of you rather than reprinting a table from the day it was written, and where a
table does appear it is labelled as a snapshot.""",
    """It also moves, in both directions. Gemma 4's parameter surface tightened in August
2026 and was later relaxed again; Grok's did the same. In the same month
`bedrock-runtime` gained three APIs it had never served and AWS changed which
endpoint it recommends. Every notebook therefore **probes** the endpoint in front of
you rather than reprinting a table from the day it was written, and where a table
does appear it is labelled as a snapshot.

Published capability tables are not exempt from this. Two claims on the Grok 4.6
model card did not survive contact with the endpoint — structured outputs are listed
as unsupported on `bedrock-runtime` and work there, and "the Chat Completions API
does not return reasoning tokens" turns out to mean no reasoning *content* while
`usage` still reports the *count*. The notebooks print what they measured.""",
)

assert text != original
README.write_text(text)
print(f"README updated: {len(original)} -> {len(text)} chars")

# ---------------------------------------------------------------------------
# Sanity: no stale claims left.
STALE = [
    "| APIs | Converse, InvokeModel |",
    "Auth | SigV4 via the AWS SDK | Bearer token",
    "not supported** — screen text yourself",
    "the header that looks like it should work on the OpenAI APIs is accepted and silently\nignored",
]
for phrase in STALE:
    assert phrase not in text, f"stale phrase still present: {phrase[:60]!r}"
print("stale-claim check: clean")
