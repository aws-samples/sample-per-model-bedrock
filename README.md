# Per-model samples for Amazon Bedrock

Runnable Jupyter notebooks for calling foundation models on Amazon Bedrock, **one
folder per model family**. Open the folder for the model you are using: it tells you
which endpoint serves it, which API to call, which parameters it accepts, and where it
behaves unlike its neighbours.

> This is sample code, for non-production usage. You should work with your security
> and legal teams to meet your organizational security, regulatory and compliance
> requirements before deployment.

These are teaching material, not production artefacts. Committed output is one run's
evidence; re-running may print different numbers.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab            # then open any notebook and run all cells
```

You need AWS credentials with Bedrock access. Nothing else — every notebook derives
its own auth from your environment.

**Converse**, via the AWS SDK with SigV4:

```python
import boto3

runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
response = runtime.converse(
    modelId="us.anthropic.claude-sonnet-5",      # note the "us." inference profile
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    inferenceConfig={"maxTokens": 256},
)
# Select the text blocks; never index content[0]. A reasoning model puts its trace
# first, and `content[0]["text"]` then raises KeyError.
print("".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b))
```

**The OpenAI-shaped APIs.** Both endpoints accept SigV4 *or* a Bedrock API key on
these paths — but the OpenAI SDK can only send a bearer token, so mint a short-term
key from the same credentials:

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

Four things in those snippets cost people hours, all covered in
[`00-foundations/01`](00-foundations/01-endpoints-auth-and-the-three-paths.ipynb):

- **Model IDs differ per endpoint.** `openai.gpt-oss-20b` on `bedrock-mantle` is
  `openai.gpt-oss-20b-1:0` on `bedrock-runtime`; Claude and the GPT-5.6 and Grok 4.6
  profiles need a `us.` or `global.` prefix. The wrong one gives *"The provided model
  identifier is invalid"*, which reads like a missing model. `runtime_id_for()`
  translates; `endpoints_for()` says which endpoints serve a model at all.
- **So does the URL path.** `bedrock-runtime` serves every OpenAI-compatible model on
  `/openai/v1` and has no `/v1` inference path; on `bedrock-mantle` the prefix depends
  on the model family.
- **A wrong path on `bedrock-runtime` returns HTTP 200**, with a Coral
  `UnknownOperationException` in the body — so `status == 200` reads it as success.
  Use `ok(status, body)` from [`_shared/bedrock.py`](_shared/bedrock.py).
- **Reasoning models spend output tokens before any text.** Too small a
  `max_output_tokens` returns 200 with an empty string. Budget ~2000 as a floor.

IAM: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` cover
`bedrock-runtime` including its OpenAI- and Anthropic-compatible paths; attach
`AmazonBedrockMantleInferenceAccess` for `bedrock-mantle`. Discovery cells also need
`bedrock:ListFoundationModels` and `bedrock:ListInferenceProfiles`. Calling a model
through an inference profile additionally needs `bedrock:InvokeModel` on your
account's default project (`arn:aws:bedrock:{region}:{account-id}:project/default`) —
without it the `AccessDeniedException` names the *model*, which sends you looking in
the wrong place.

## Find your model family

Open one folder. Each notebook is self-contained.

| Folder | Models | Endpoint | API | What the notebooks cover |
|---|---|---|---|---|
| [`01-openai-gpt/`](01-openai-gpt/) | gpt-5.6 sol/terra/luna, gpt-5.5, gpt-5.4 · gpt-oss 20b/120b · gpt-oss-safeguard | gpt-5.6 **both** · earlier gpt-5.x mantle · gpt-oss both | Responses · Chat Completions · Converse | Core inference · web search · tools & strict JSON · prompt caching · server-side Lambda tools & fine-tuning |
| [`02-anthropic-claude/`](02-anthropic-claude/) | opus-5, sonnet-5, opus-4-8, opus-4-7, haiku-4-5, fable-5 | both | Messages | Core inference · adaptive thinking, tool loops, caching · computer use, memory, compaction |
| [`03-google-gemma/`](03-google-gemma/) | gemma-4 31b · 26b-a4b · e2b · gemma-3 4b · 12b · 27b | gemma 4 **mantle** · gemma 3 both | gemma 4 Responses **and** Chat Completions · gemma 3 Chat Completions | Gemma 4 end to end · Gemma 3 on both endpoints, and why the two generations share almost nothing |
| [`04-qwen/`](04-qwen/) | qwen3 32b/235b/next-80b, coder 30b/480b/next, vl-235b | both | Chat Completions | Core inference & tools · coding models & vision |
| [`05-deepseek/`](05-deepseek/) | v3.2, v3.1 | v3.2 both · v3.1 **mantle** | Chat Completions | Core inference, reasoning effort, tools, structured output |
| [`06-zai-glm/`](06-zai-glm/) | glm-5, glm-4.7, glm-4.7-flash, glm-4.6 | both · 4.6 **mantle** | Chat Completions | Core inference plus cost-aware routing across the size ladder |
| [`07-mistral/`](07-mistral/) | mistral-large-3, ministral 3b/8b/14b, magistral, devstral-2, voxtral | both | Chat Completions | Size ladder & routing · Devstral coding, Voxtral |
| [`08-moonshot-kimi/`](08-moonshot-kimi/) | kimi-k2.5, kimi-k2-thinking | both | Chat Completions | Core inference, long context, agentic patterns |
| [`09-minimax/`](09-minimax/) | minimax-m2.5, m2.1, m2 | both | Chat Completions | Core inference plus a version-migration test across three generations |
| [`10-nvidia-nemotron/`](10-nvidia-nemotron/) | nemotron-super-3-120b, nano 9b/12b/30b | both | Chat Completions | Core inference across the cost/quality curve |
| [`11-xai-grok/`](11-xai-grok/) | grok-4.6 · grok-4.3 | 4.6 **both** (runtime: profile-only; mantle: `us-west-2` only) · 4.3 **mantle** | Responses · Chat Completions · Converse | Grok 4.6: the effort dial measured, encrypted reasoning replayed, why `global.` caches worse than `us.` · Grok 4.3: always-on reasoning |
| [`12-writer-palmyra/`](12-writer-palmyra/) | palmyra-vision-7b · palmyra-x4 · palmyra-x5 | vision both · x4/x5 **runtime** | Chat Completions · Converse | Vision, and working around a model with no tool support · the text models, which need an inference profile |
| [`13-amazon-nova/`](13-amazon-nova/) | nova-micro · nova-lite · nova-pro | **runtime** | Converse | Tier selection graded on a checkable task · vision on lite/pro · what a provider-deprecated model looks like |
| [`14-openai-gpt-oss/`](14-openai-gpt-oss/) | gpt-oss 20b/120b · gpt-oss-safeguard 20b/120b | both | Chat Completions · Converse | Where the reasoning trace lives on each endpoint · policy classification graded on a labelled set |
| [`15-meta-llama/`](15-meta-llama/) | llama4-scout · llama4-maverick | **runtime** | Converse | Mixture-of-experts sizing · inference-profile-only access · vision · validating tool calls |

**Treat the Endpoint column as a snapshot.** Models arrive, move between endpoints and
are retired; every family notebook re-checks it with `endpoints_for()` when you run it.

## Start with foundations if you are new to Bedrock

| Notebook | Covers |
|---|---|
| [`01-endpoints-auth-and-the-three-paths`](00-foundations/01-endpoints-auth-and-the-three-paths.ipynb) | Endpoint choice · SigV4 · short-term API keys · curl · URL paths on both endpoints · the 200 that means failure · per-endpoint model IDs · model discovery · IAM |
| [`02-governance-projects-and-retention`](00-foundations/02-governance-projects-and-retention.ipynb) | Projects/Workspaces · cost attribution · data retention & ZDR · CloudWatch |
| [`03-scaling-tiers-and-latency`](00-foundations/03-scaling-tiers-and-latency.ipynb) | Quota model · retries & backoff · service tiers · TTFT measurement |
| [`04-bedrock-runtime-converse-and-profiles`](00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb) | Converse · content blocks · inference profiles (CRIS) · reading the catalogue · streaming · InvokeModel · the OpenAI- and Anthropic-compatible APIs on this endpoint · server-side conversation state |

Choosing between families, or already have OpenAI code?
[`99-cross-cutting/`](99-cross-cutting/) has a live capability survey, a migration
guide with a self-healing compatibility shim, and a pre-launch checklist.

One result from that checklist is worth knowing before you rely on Guardrails: the
`X-Amzn-Bedrock-Guardrail*` header is **enforced** on `bedrock-runtime` Chat
Completions and Messages, and **accepted and silently ignored** on `bedrock-runtime`
Responses and on every `bedrock-mantle` surface. Nothing in the response
distinguishes them.
[`99-cross-cutting/03`](99-cross-cutting/03-production-hardening.ipynb) §9b measures
each attachment point; where the header is ignored, call `ApplyGuardrail` explicitly
or use Converse.

## The one shared file

The only thing shared across folders is [`_shared/bedrock.py`](_shared/bedrock.py) —
this collection's own helper module, not a PyPI package:

```python
import sys
sys.path.insert(0, "../_shared")
from bedrock import post, converse, safe_print
```

It exists only to remove repetition. **Nothing in it is required to call Bedrock
yourself**; it wraps `boto3` and the OpenAI and Anthropic SDKs, and each notebook
opens with a table of exactly the helpers it uses. Two behaviours to know when
reading committed output:

- **`post()` and `converse()` never raise on a service error** — they return the
  status and body so a cell can *show* a 400. Many cells provoke one deliberately; a
  400 in the output is usually the lesson.
- **Control-plane output is redacted.** Account IDs, IAM principals and opaque service
  IDs are replaced, because this output is committed to a public repository.

## Cost, scope and cleanup

Each notebook makes tens of small calls with tight token budgets. Two cost more than
the rest: `01-openai-gpt/02-web-search-and-grounding.ipynb` (web search is billed
separately from tokens) and `99-cross-cutting/01-choosing-a-model-and-api.ipynb` (a
deliberate survey across every family). Notebooks that create demo Projects archive
them in a final cell.

In scope: documented public APIs of Amazon Bedrock serverless inference on both the
`bedrock-runtime` and `bedrock-mantle` endpoints, for models that **return text**,
across text, image and audio inputs. Out of scope: image, video, speech and embedding
outputs, rerankers, Bedrock Marketplace, and models a newer generation has superseded.

Vision and audio cells need an input, so two small files are committed under
[`_shared/assets/`](_shared/assets) (64 KB together): a slide and a seven-second
speech clip, both excerpted from a public AWS talk. Each gives its cell a known
answer, so "read this slide" and "transcribe this" can be scored rather than
admired. Provenance is in [`_shared/bedrock.py`](_shared/bedrock.py).

Model behaviour on Bedrock changes without notice — a parameter accepted today can be
rejected tomorrow. Where a notebook states a parameter matrix it also shows the probe
that produced it, so you can re-derive today's answer.

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
