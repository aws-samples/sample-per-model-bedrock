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

Python 3.11 or later.

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

Four things in those snippets cost people hours. The first three are covered in
[`00-foundations/01`](00-foundations/01-endpoints-auth-and-the-three-paths.ipynb), and
the fourth in
[`00-foundations/04`](00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb)
and each family notebook:

- **Model IDs differ per endpoint.** `openai.gpt-oss-20b` on `bedrock-mantle` is
  `openai.gpt-oss-20b-1:0` on `bedrock-runtime`, and many models on `bedrock-runtime`
  (Claude, the hosted GPT models, Grok 4.6 and 4.7, Kimi K3, Nova 2 Lite, Palmyra X4 and
  X5, Llama 4) are called through an inference profile with a geographic prefix such as
  `us.` or `in.`, or with a `global.` one. The wrong ID gives *"The provided model
  identifier is invalid"* or *"on-demand throughput isn't supported"*, which reads like a
  missing model. `runtime_id_for()` translates; `endpoints_for()` says which endpoints
  serve a model at all.
- **So does the URL path.** `bedrock-runtime` serves every OpenAI-compatible model on
  `/openai/v1` and has no `/v1` inference path; on `bedrock-mantle` the prefix depends
  on the model family, and inside the OpenAI family on the product line rather than the
  version number — hosted GPT on `/openai/v1`, open-weight `gpt-oss` on bare `/v1`.
- **A wrong path on `bedrock-runtime` returns HTTP 200**, with a Coral
  `UnknownOperationException` in the body — so `status == 200` reads it as success.
  Use `ok(status, body)` from [`_shared/bedrock.py`](_shared/bedrock.py).
- **Reasoning models spend output tokens before any text.** Too small a
  `max_output_tokens` returns 200 with an empty string. Budget ~2000 as a floor.

IAM: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` cover
`bedrock-runtime` including its OpenAI- and Anthropic-compatible paths; attach
`AmazonBedrockMantleInferenceAccess` for `bedrock-mantle`. Calling with a Bedrock API
key, as the OpenAI and Anthropic SDKs do, also needs `bedrock:CallWithBearerToken` on
`bedrock-runtime` and `bedrock-mantle:CallWithBearerToken` on `bedrock-mantle`.
Discovery cells also need `bedrock:ListFoundationModels` and
`bedrock:ListInferenceProfiles`. The Responses API on `bedrock-runtime`, used by the
hosted GPT models and Grok 4.6 and 4.7, is authorised against the inference profile, the
foundation model in each Region the profile routes to, and the account's default project
(`arn:aws:bedrock:{region}:{account-id}:project/default`), so allow
`bedrock:InvokeModel` on all three;
[`01-openai-gpt/01`](01-openai-gpt/01-responses-api-core.ipynb) and
[`11-xai-grok/02`](11-xai-grok/02-grok-4-6.ipynb) have the details.

## Find your model family

Open one folder. Each notebook is self-contained.

| Folder | Models | Endpoint | API | What the notebooks cover |
|---|---|---|---|---|
| [`01-openai-gpt/`](01-openai-gpt/) | gpt-6.1 sol · gpt-6 sol/luna/astra · gpt-5.6 sol/terra/luna, gpt-5.5, gpt-5.4 | **both** (runtime: profile-only) · on mantle, gpt-6.1 sol and gpt-6 sol/luna `us-east-1` only and gpt-6-astra `us-west-2` only | Responses · Chat Completions · Converse | Responses API core, reasoning, vision · web search with citations · tools and structured output · prompt caching · MCP tools, batch inference, fine-tuning |
| [`02-anthropic-claude/`](02-anthropic-claude/) | sonnet-5, sonnet-5.5, opus-5.5, opus-5, opus-4-8, opus-4-7, haiku-4-5, haiku-5-5, fable-5, fable-5.1 | fable-5.1, sonnet-5.5 and haiku-5.5 **runtime** only (profile-only) · the rest **both** (runtime: profile-only; mantle: varies by Region) | Messages | Messages API core, vision, token counting · thinking, tool loops, prompt caching · computer use, memory, agent tools |
| [`03-google-gemma/`](03-google-gemma/) | gemma-4 31b · 26b-a4b · e2b · gemma-3 4b · 12b · 27b | gemma 4 **mantle** · gemma 3 both | gemma 4 Responses **and** Chat Completions · gemma 3 Chat Completions | Gemma 4: reasoning, tools, structured output, vision · Gemma 3: structured output, vision, sizes, Converse |
| [`04-qwen/`](04-qwen/) | qwen3 32b/235b/next-80b, coder 30b/480b/next, vl-235b | both · 235b and coder-480b **mantle** | Chat Completions | Reasoning, tools, structured output, Converse · code generation, code review, an agentic coding loop, vision |
| [`05-deepseek/`](05-deepseek/) | v3.2, v3.1 | v3.2 both · v3.1 **mantle** | Chat Completions | Reasoning, tools, structured output, Converse |
| [`06-zai-glm/`](06-zai-glm/) | glm-5, glm-4.7, glm-4.7-flash, glm-4.6 · glm-5.3 | both · 4.6 **mantle** · 5.3 **runtime** (profile-only) | Chat Completions · 5.3 also Responses | Reasoning, tools, structured output, Converse, GLM 5.3 |
| [`07-mistral/`](07-mistral/) | mistral-large-3, ministral 3b/8b/14b, magistral, devstral-2, voxtral | both | Chat Completions | Reasoning with Magistral, tools, choosing a size, vision, Converse · Devstral coding agent, Voxtral transcription |
| [`08-moonshot-kimi/`](08-moonshot-kimi/) | kimi-k2.5, kimi-k2-thinking · kimi-k3 | k2 both · k3 **runtime** (profile-only) | Chat Completions · k3 also Responses | Reasoning, tools, structured output, vision, Converse, Kimi K3 |
| [`09-minimax/`](09-minimax/) | minimax-m2.5, m2.1, m2 | both | Chat Completions | Reasoning, tools, structured output, Converse |
| [`10-nvidia-nemotron/`](10-nvidia-nemotron/) | nemotron-super-3-120b, nano 9b/12b/30b | both | Chat Completions | Reasoning, tools, structured output, vision with Nano 12B v2, Converse |
| [`11-xai-grok/`](11-xai-grok/) | grok-4.7 · grok-4.6 · grok-4.3 | 4.7 **runtime** (profile-only) · 4.6 **both** (runtime: profile-only; mantle: `us-west-2` only) · 4.3 **mantle** | Responses · Chat Completions · Converse | Reasoning, server-side state, tools, structured output, vision · 4.6 and 4.7 also Converse · 4.6 also background mode |
| [`12-writer-palmyra/`](12-writer-palmyra/) | palmyra-vision-7b · palmyra-x4 · palmyra-x5 | vision both · x4/x5 **runtime** | Chat Completions · Converse | Vision and structured output from an image · X4 and X5: tools and structured output through a tool |
| [`13-amazon-nova/`](13-amazon-nova/) | nova-micro · nova-lite · nova-pro · nova-2-lite (profile-only) | **runtime** | Converse | Reasoning, tools, structured output through a tool, vision, choosing a tier |
| [`14-openai-gpt-oss/`](14-openai-gpt-oss/) | gpt-oss 20b/120b · gpt-oss-safeguard 20b/120b | both | Chat Completions · Responses · Converse | Reasoning, tools, structured output, server-side state, Converse · policy classification graded on a labelled set |
| [`15-meta-llama/`](15-meta-llama/) | llama4-scout · llama4-maverick | **runtime** | Converse | Tools, structured output through a tool, vision |

The first notebook in each folder starts with a first call, streaming and multi-turn.

**Treat the Endpoint column as a snapshot.** Models arrive, move between endpoints and
are retired. Most family notebooks re-check it in their setup cell with
`endpoints_for()`; for the rest, call it yourself before you depend on a row.

Where a row says profile-only, the prefix that profile carries belongs to the Region and
not to the model, so resolve it with `runtime_id_for()` rather than writing `us.` or
`global.` into the model ID. As of October 2026 Claude Sonnet 5.5 answered on both `us.`
and `global.` in `us-east-1` and on `global.` alone in `ap-northeast-1`;
[`02-anthropic-claude/01`](02-anthropic-claude/01-messages-api-core.ipynb) measures the
set per Region.

## Start with foundations if you are new to Bedrock

| Notebook | Covers |
|---|---|
| [`01-endpoints-auth-and-the-three-paths`](00-foundations/01-endpoints-auth-and-the-three-paths.ipynb) | The two endpoints · URL paths · SigV4 · short-term API keys · curl · model discovery · model IDs on each endpoint · IAM |
| [`02-governance-projects-and-retention`](00-foundations/02-governance-projects-and-retention.ipynb) | Projects and cost attribution · `store` and conversation state · data retention modes · CloudWatch metrics |
| [`03-scaling-tiers-and-latency`](00-foundations/03-scaling-tiers-and-latency.ipynb) | Quota model · retries with backoff · service tiers · time to first token |
| [`04-bedrock-runtime-converse-and-profiles`](00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb) | Converse · content blocks · inference profiles · streaming · a Converse tool loop · InvokeModel · the OpenAI and Anthropic SDKs on `bedrock-runtime` · choosing an API |

Choosing between families, or already have OpenAI code?
[`99-cross-cutting/`](99-cross-cutting/) has a model and API guide, a migration guide
with a compatibility shim, and a production-hardening checklist.

Check Guardrails on the API you ship. As of September 2026 the
`X-Amzn-Bedrock-Guardrail*` header is applied on every `bedrock-runtime` API (Chat
Completions, Responses and Messages) and accepted but ignored on every `bedrock-mantle`
API, with HTTP 200 either way.
[`99-cross-cutting/03`](99-cross-cutting/03-production-hardening.ipynb) measures each
attachment point; on `bedrock-mantle`, call `ApplyGuardrail` or use Converse.

## The one shared file

The only thing shared across folders is [`_shared/bedrock.py`](_shared/bedrock.py) —
this collection's own helper module, not a PyPI package:

```python
import sys
sys.path.insert(0, "../_shared")
from bedrock import endpoints_for, post, safe_print
```

It exists only to remove repetition. **Nothing in it is required to call Bedrock
yourself**; it wraps `boto3`, the Bedrock token generator and plain HTTPS, and each notebook's
setup cell names the helpers it imports. Two behaviours to know when reading
committed output:

- **`post()` never raises on a service error.** It returns the status and body, so a
  cell can print what the service said.
- **Control-plane output is redacted.** Account IDs, IAM principals and opaque service
  IDs are replaced, because this output is committed to a public repository.

## Cost, scope and cleanup

Each notebook makes tens of small calls with tight token budgets. Two cost more than
the rest: `01-openai-gpt/02-web-search-and-grounding.ipynb` (web search is billed
separately from tokens) and `99-cross-cutting/01-choosing-a-model-and-api.ipynb`
(which calls every family on `bedrock-mantle`). `00-foundations/02` creates a demo Project and archives it
in its final cell; `99-cross-cutting/03` creates a guardrail and deletes it.

In scope: documented public APIs of Amazon Bedrock serverless inference on both the
`bedrock-runtime` and `bedrock-mantle` endpoints, for models that **return text**,
across text, image and audio inputs. Out of scope: image, video, speech and embedding
outputs, rerankers, Bedrock Marketplace, and models the provider has retired.

Vision and audio cells need an input, so three small files are committed under
[`_shared/assets/`](_shared/assets) (75 KB together): a slide and a seven-second
speech clip, both excerpted from a public AWS talk, and a synthetic invoice. Each
gives its cell a known answer, so the cell can check what the model returned.
Provenance is in [`_shared/bedrock.py`](_shared/bedrock.py).

Model behaviour on Bedrock changes without notice: a parameter accepted today can be
rejected tomorrow. Notes about what a model accepts are dated and quote the error
text, so you can tell when one has gone stale.

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
