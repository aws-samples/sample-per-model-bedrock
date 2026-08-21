# Per-model samples for Amazon Bedrock

Runnable Jupyter notebooks for calling foundation models on Amazon Bedrock, **one
folder per model family**. Open the folder for the model you are using and it tells
you everything needed for that model — which endpoint serves it, which API to call,
which parameters it accepts, and where it behaves unlike its neighbours.

> This is sample code, for non-production usage. You should work with your security
> and legal teams to meet your organizational security, regulatory and compliance
> requirements before deployment.

The notebooks are teaching material, not production-ready artefacts. They favour
readability over completeness: error handling is shown where it teaches something and
elided where it would obscure the point.

## Why per-model

Bedrock model behaviour is not uniform, and the differences are the part that costs
you time. GPT-5.6 pins `temperature` to its default, rejects `top_p`, and takes
`max_completion_tokens` where the `/v1` families take `max_tokens`. Palmyra Vision has
no usable tool support at all. Most Claude models must be addressed through a
cross-Region inference profile and are rejected by their bare model ID. Newer Claude
models reject `temperature` as deprecated. None of that is discoverable from a generic
example, so each family gets its own notebook rather than a shared one.

It also moves, in both directions. Gemma 4's parameter surface tightened in August
2026 and was later relaxed again; Grok's did the same. In the same month
`bedrock-runtime` gained three APIs it had never served and AWS changed which
endpoint it recommends. Every notebook therefore **probes** the endpoint in front of
you rather than reprinting a table from the day it was written, and where a table
does appear it is labelled as a snapshot.

Published capability tables are not exempt from this. Two claims on the Grok 4.6
model card did not survive contact with the endpoint — structured outputs are listed
as unsupported on `bedrock-runtime` and work there, and "the Chat Completions API
does not return reasoning tokens" turns out to mean no reasoning *content* while
`usage` still reports the *count*. The notebooks print what they measured.

## Two endpoints, and which to use

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
so a notebook can ask the service instead of trusting a table that ages.

## Find your model family

Open one folder. Each notebook is self-contained.

| Folder | Models | Endpoint | API | What the notebooks cover |
|---|---|---|---|---|
| [`01-openai-gpt/`](01-openai-gpt/) | gpt-5.6 sol/terra/luna, gpt-5.5, gpt-5.4 · gpt-oss 20b/120b · gpt-oss-safeguard | gpt-5.6 **both** · earlier gpt-5.x mantle · gpt-oss both | Responses · Chat Completions · Converse | Core inference · **web search** · tools & strict JSON · prompt caching · server-side Lambda tools & fine-tuning · **runtime via inference profile** |
| [`02-anthropic-claude/`](02-anthropic-claude/) | opus-5, sonnet-5, opus-4-8, opus-4-7, haiku-4-5, fable-5 | both | Messages | Core inference · adaptive thinking, tool loops, caching · computer use, memory, compaction |
| [`03-google-gemma/`](03-google-gemma/) | gemma-4 31b · 26b-a4b · e2b · gemma-3 4b · 12b · 27b | gemma 4 **mantle** · gemma 3 both | gemma 4 Responses **and** Chat Completions · gemma 3 Chat Completions | Gemma 4 end to end · Gemma 3 on both endpoints, and why the two generations share almost nothing |
| [`04-qwen/`](04-qwen/) | qwen3 32b/235b/next-80b, coder 30b/480b/next, vl-235b | both | Chat Completions | Core inference & tools · coding models & vision |
| [`05-deepseek/`](05-deepseek/) | v3.2, v3.1 | v3.2 both · v3.1 **mantle** | Chat Completions | Core inference, reasoning effort, tools, structured output |
| [`06-zai-glm/`](06-zai-glm/) | glm-5, glm-4.7, glm-4.7-flash, glm-4.6 | both · 4.6 **mantle** | Chat Completions | Core inference plus cost-aware routing across the size ladder |
| [`07-mistral/`](07-mistral/) | mistral-large-3, ministral 3b/8b/14b, magistral, devstral-2, voxtral | both | Chat Completions | Size ladder & routing · Devstral coding, Voxtral |
| [`08-moonshot-kimi/`](08-moonshot-kimi/) | kimi-k2.5, kimi-k2-thinking | both | Chat Completions | Core inference, long context, agentic patterns |
| [`09-minimax/`](09-minimax/) | minimax-m2.5, m2.1, m2 | both | Chat Completions | Core inference plus a version-migration test across three generations |
| [`10-nvidia-nemotron/`](10-nvidia-nemotron/) | nemotron-super-3-120b, nano 9b/12b/30b | both | Chat Completions | Core inference across the cost/quality curve |
| [`11-xai-grok/`](11-xai-grok/) | grok-4.6 · grok-4.3 | 4.6 **both** (runtime: profile-only; mantle: `us-west-2` only) · 4.3 **mantle** | Responses **and** Chat Completions · Converse | **Grok 4.6**: the effort dial measured, encrypted reasoning replayed, structured output the model card says is absent · Grok 4.3: always-on reasoning |
| [`12-writer-palmyra/`](12-writer-palmyra/) | palmyra-vision-7b · palmyra-x4 · palmyra-x5 | vision both · x4/x5 **runtime** | Chat Completions · Converse | Vision, and working around a model with no tool support · the text models, which need an inference profile |
| [`13-amazon-nova/`](13-amazon-nova/) | nova-micro · nova-lite · nova-pro | **runtime** | Converse | Tier selection graded on a checkable task · vision on lite/pro · what a provider-deprecated model looks like |
| [`14-openai-gpt-oss/`](14-openai-gpt-oss/) | gpt-oss 20b/120b · gpt-oss-safeguard 20b/120b | both | Chat Completions · Converse | Where the reasoning trace lives on each endpoint · policy classification graded on a labelled set |
| [`15-meta-llama/`](15-meta-llama/) | llama4-scout · llama4-maverick | **runtime** | Converse | Mixture-of-experts sizing · inference-profile-only access · vision · validating tool calls |

The Endpoint column says where each family was served when this was written, read
from the live catalogues rather than hand-maintained. **Treat it as a snapshot.**
Models arrive, move between endpoints, and are retired, so check the current answer
for any model with `endpoints_for()` from [`_shared/bedrock.py`](_shared/bedrock.py)
— every family notebook runs that check in its own endpoint section.

For the same reason the notebooks state which models, Regions and features they
*cover*, and probe anything that varies rather than asserting what is unavailable
where. A committed table is evidence of one run, not a specification.

**Where a cell reaches a conclusion, the conclusion is computed from that cell's own
results** rather than written alongside them. That is deliberate, and it is the
single most useful convention here: a paragraph that asserts "this model rejects
`top_p`" is wrong the moment the service changes, and it has been — Gemma 4's
parameter surface tightened in August 2026 and was relaxed again weeks later. A
derived verdict cannot contradict the table above it. When you re-run a notebook and
its summary line reads differently from the committed one, that is the design
working, not a defect.

**Read [`00-foundations/`](00-foundations/) first if you are new to Bedrock** — auth,
endpoints, quotas and governance apply to every family:

| Notebook | Covers |
|---|---|
| [`01-endpoints-auth-and-the-three-paths`](00-foundations/01-endpoints-auth-and-the-three-paths.ipynb) | Endpoint choice · SigV4 · short-term API keys · curl · the URL paths on **both** endpoints · the 200 that means failure · per-endpoint model IDs · model discovery · IAM |
| [`02-governance-projects-and-retention`](00-foundations/02-governance-projects-and-retention.ipynb) | Projects/Workspaces · cost attribution · data retention & ZDR · CloudWatch |
| [`03-scaling-tiers-and-latency`](00-foundations/03-scaling-tiers-and-latency.ipynb) | Quota model · retries & backoff · service tiers · TTFT measurement |
| [`04-bedrock-runtime-converse-and-profiles`](00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb) | SigV4 · Converse · content blocks · inference profiles (CRIS) · reading the catalogue · streaming · InvokeModel · **the OpenAI- and Anthropic-compatible APIs on this endpoint** · server-side conversation state |

Choosing between families, or already have OpenAI code?
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
`ApplyGuardrail` explicitly or move the call to Converse.

## The one shared file

Each folder is meant to answer "how do I use this model" on its own, so the only
thing shared across them is [`_shared/bedrock.py`](_shared/bedrock.py).

`bedrock` is **not** a PyPI package — it is this collection's own helper module.
Notebooks reach it with:

```python
import sys
sys.path.insert(0, "../_shared")
from bedrock import post, converse, safe_print
```

It exists only to remove repetition. **Nothing in it is required to call Bedrock
yourself** — it wraps `boto3` and the OpenAI and Anthropic SDKs, and every
notebook shows the underlying call. Each notebook opens with a table of exactly
the helpers it uses and what they do, so you never have to guess where a name
came from.

Two behaviours are worth knowing before reading any committed output:

- **`post()` and `converse()` never raise on a service error.** They return the
  status and body, so a cell can *show* a 400 instead of stopping the notebook.
  Many cells deliberately provoke an error to demonstrate a limit — a 400 in the
  output is usually the lesson, not a bug.
- **Anything printed from a control-plane response is redacted.** Account IDs, IAM
  principals and opaque service IDs (`proj_…`, `resp_…`) are shortened or replaced,
  because this output is committed to a public repository.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab            # then open any notebook and run all cells
```

You need AWS credentials with Bedrock access. Nothing else — every notebook derives
its own auth from whatever credentials are already in your environment.

**On `bedrock-runtime`** the AWS SDK signs with SigV4, so your existing credentials
work directly:

```python
import boto3

runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
response = runtime.converse(
    modelId="us.anthropic.claude-sonnet-5",      # note the "us." inference profile
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    inferenceConfig={"maxTokens": 64},
)
print(response["output"]["message"]["content"][0]["text"])
```

**For the OpenAI-shaped APIs** the SDK expects a bearer token, so mint a short-term
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
`bedrock-runtime` also returns HTTP 200.

Least-privilege IAM: `bedrock:InvokeModel` and
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
[`11-xai-grok/02`](11-xai-grok/02-grok-4-6.ipynb) §14.

## Cost and cleanup

Each notebook makes tens of small calls with tight token budgets per run. Two cost more than the rest:

- `01-openai-gpt/02-web-search-and-grounding.ipynb` — web search is billed
  separately from tokens.
- `99-cross-cutting/01-choosing-a-model-and-api.ipynb` — a deliberate survey that
  touches every family.

Notebooks that create demo Projects archive them in a final cell.

## Scope

These notebooks demonstrate the documented public APIs of Amazon Bedrock serverless
inference, on both the `bedrock-runtime` and `bedrock-mantle` endpoints.

In scope: models that **return text**, across text and image inputs. Out of scope:
image, video, speech and embedding outputs, rerankers, Bedrock Marketplace, and
models that a newer generation has superseded.

Model behaviour on Bedrock changes without notice — a parameter that is accepted
today can be rejected tomorrow, as happened to Gemma 4 on 12 August 2026. Where a
notebook states a parameter matrix it also shows the probe that produced it, so you
can re-derive today's answer instead of trusting a committed table.

## Disclaimer

The sample code; software libraries; command line tools; proofs of concept;
templates; or other related technology (including any of the foregoing that are
provided by our personnel) is provided to you as AWS Content under the AWS Customer
Agreement, or the relevant written agreement between you and AWS (whichever applies).
You should not use this AWS Content in your production accounts, or on production or
other critical data. You are responsible for testing, securing, and optimizing the
AWS Content, such as sample code, as appropriate for production grade use based on
your specific quality control practices and standards. Deploying AWS Content may
incur AWS charges for creating or using AWS chargeable resources, such as running
Amazon EC2 instances or using Amazon S3 storage.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for how to
report a security issue. Please do not open a public GitHub issue.

## License

MIT-0. See [LICENSE](LICENSE).
