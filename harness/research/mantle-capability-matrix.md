# bedrock-mantle capability matrix — round 2 (5x deeper)

Probed live 2026-07-30, `us-east-1`/`us-east-2`/`us-west-2`/`eu-central-1`,
short-term Bedrock API key. Every ✅/❌ is an observed HTTP status.

## A. FEATURE CATALOG (what round 1 missed)

| Feature | Status | Notes |
|---|---|---|
| **Web Search** (`tools:[{type:"web_search"}]`) | ✅ GA Aug 2026 | Amazon-built index, tens of billions of docs + knowledge graph. **OpenAI GPT only**: gpt-5.4/5.5/5.6-sol/terra/luna. 400 on gemma-4 & grok. us-east-1/2, us-west-2 only, strictly in-Region. |
| ↳ `external_web_access` | ✅ both values | Defaults **true**; needs `bedrock-websearch:ExternalWebAccess` or → 403 on authz (model still answers from index). Set `false` to stay in AWS boundary. |
| ↳ citations | ✅ verified | `url_citation` annotations w/ title/url/start_index/end_index. Streaming: `response.output_text.annotation.added`. **Must retain+display per acceptable use.** |
| **Server-side tools (Lambda MCP)** | ✅ docs | `tools:[{type:"mcp",connector_id:<lambda-arn>,require_approval:"never"}]`. Lambda implements JSON-RPC `tools/list`+`tools/call`. Lambda needs same IAM policy as caller. |
| **AgentCore Gateway as tool** | ✅ docs (Feb 2026) | `connector_id:<gateway-arn>`; supports **all** Responses-API models; multiple tool calls per turn. |
| **AWS-provided notes/tasks tools** | ✅ verified | gpt-oss 20b/120b built-in, no declaration. Emits `mcp_call`. Session-scoped memory. |
| **Data retention / ZDR** | ✅ verified | `GET /v1/data_retention` (account), project-level `data_retention.mode`. Modes: `default`, `none` (=ZDR), `provider_data_share`, `inherit`. My probe: inherit/none/default ✅; enabled/disabled/zero → 400. |
| **Projects** (create/use/tag/archive) | ✅ full lifecycle verified | `POST /v1/organization/projects`; 1000/account; archived = no new inference, data 30d. |
| ↳ 3 attribution headers | ✅ all verified | `OpenAI-Project` (Responses+CC), `anthropic-workspace` (Messages), `openai-project` also accepted. |
| **Files API** | ✅ `GET /v1/files` 200 | multipart upload; purpose `fine-tune`. NOT at `/openai/v1/files` (404). |
| **Fine-tuning (RFT) API** | ✅ `GET /v1/fine_tuning/jobs` 200 | OpenAI-compatible. **Only gpt-oss-20b + qwen3-32b, us-west-2 only.** Lambda reward fn; checkpoints; FT model → immediate inference. |
| **OpenAI Batch API** | ⚠️ **bedrock-runtime, not mantle** | `https://bedrock-runtime.{r}.amazonaws.com/openai/v1/batches`. Needs `X-Amzn-Bedrock-RoleArn` + `X-Amzn-Bedrock-ModelId`. endpoint must be `/v1/chat/completions`. No tools/structured output in batch. |
| **Computer use (Claude)** | ✅ verified 200 | mantle: `anthropic-beta: computer-use-2025-11-24` **header**; tool `computer_20251124`/`bash_20250124`/`text_editor_20250124`. Beta Service. |
| **Memory tool (Claude)** | ✅ verified 200 | `memory_20250818` + beta `context-management-2025-06-27`. |
| **Context management / compaction** | ✅ verified 200 | `context_management.edits:[{type:"clear_tool_uses_20250919"}]`. |
| **Async/background** | ✅ verified | `background:true` + `store:true` → `in_progress`→`completed`; `GET /openai/v1/responses/{id}`; `DELETE` ✅; `/input_items` → 404. |
| **Response CRUD** | ✅ GET/DELETE | `/input_items` not implemented. |
| **Conversations API** | ❌ 404 | `POST /openai/v1/conversations` not implemented. Use `previous_response_id`. |
| **CloudWatch** | namespace `AWS/BedrockMantle` | Only `InferenceClientErrors` (4xx) — **no 5xx metric**. CloudTrail needs **data events** (extra cost). |
| **CRIS** | ❌ mantle is in-Region only | Geo/global profiles are bedrock-runtime only. |
| **Provisioned Throughput** | ❌ not on mantle | PT/batch/custom-profile quotas are runtime-only. |
| **Guardrails** | ⚠️ headers accepted (200) | `X-Amzn-Bedrock-GuardrailIdentifier/Version`. Effect unverified (fake ID). Model cards list Guardrails under **bedrock-runtime**, not mantle. |
| **CloudFormation** | `AWS::BedrockMantle::Project` | Private resource type (internal spec). |

## B. THREE PATH FAMILIES

| Path | Families |
|---|---|
| `/openai/v1/{responses,chat/completions}` | google gemma-4, openai gpt-5.x, xai |
| `/v1/{responses,chat/completions}` | openai gpt-oss + all CC-only families |
| `/anthropic/v1/messages` (+`/count_tokens`) | anthropic only |
| `/v1/models`, `/v1/files`, `/v1/fine_tuning/*`, `/v1/organization/projects`, `/v1/data_retention` | control-plane (NOT `/openai/v1/*`) |

## C. API availability — 55 models / 12 families

| Family | Responses | Chat Completions | Messages |
|---|---|---|---|
| google gemma-4 | ✅ /openai/v1 | ✅ /openai/v1 | ❌ |
| openai gpt-5.6 | ✅ /openai/v1 | ❌ **400 both** | ❌ |
| openai gpt-5.4/5.5 | ✅ /openai/v1 | ✅ | ❌ |
| openai gpt-oss | ✅ /v1 | ✅ /v1 | ❌ |
| xai grok-4.3 | ✅ /openai/v1 | ✅ /openai/v1 | ❌ |
| anthropic (6) | ❌ | ❌ | ✅ |
| qwen, deepseek, zai, minimax, moonshotai, mistral, nvidia, writer | ❌ 400 | ✅ /v1 | ❌ |

## D. REGIONAL FOOTPRINT (live `/v1/models`)

| Region | Models | Notable |
|---|---|---|
| us-east-1 | **55** | only region w/ full Claude set (6) + all gpt-5.x |
| us-east-2 | 49 | **zero Anthropic** |
| us-west-2 | 47 | Claude haiku-4-5 only; **no gpt-5.5 / 5.6-sol**; only FT region |
| eu-central-1 | 33 | no Anthropic, no gpt-5.x, no xai, no deepseek, no moonshot |

Web Search: us-east-1/2 + us-west-2 (not eu-central-1). Fine-tuning: us-west-2 only.

## E. PER-MODEL PARAMETER QUIRKS

| Model | temperature | top_p | reasoning efforts |
|---|---|---|---|
| gemma-4-31b | ✅ | ❌ 400 | none/low/medium/high ✅, `minimal` ❌ |
| grok-4.3 | ❌ 400 "deprecated" | ✅ | same |
| gpt-5.6-sol | ✅ | ❌ 400 | same |
| gpt-oss-120b | ✅ | ✅ | same |
| claude sonnet-5 / opus-4-8 / opus-5 | ❌ 400 "deprecated" | — | `thinking:{adaptive}` ✅ |
| claude haiku-4-5 | ✅ | — | adaptive ❌ 400 |
| all CC-only families | ✅ | ✅ | `reasoning_effort` ✅ |

gpt-5.6 extras all ✅: `text.verbosity`, `max_tool_calls`, `truncation`, `metadata`,
`parallel_tool_calls`, `include:["reasoning.encrypted_content"]`.

## F. VERIFIED FEATURE BEHAVIOUR

- **Streaming** ✅ all 3 APIs. Events: CC `data:{choices[].delta}`; Responses
  `response.output_text.delta` / `response.reasoning_text.delta` /
  `response.reasoning_part.added|done` / `response.output_text.annotation.added`;
  Messages `event: message_start` … `content_block_delta`.
- **previous_response_id** ✅ gemma-4 recalled "Your name is Alex."
  `store:false` → chain gives **404 not found**.
- **GPT-5.6 explicit cache** ✅ call1 `cache_write=1565, cached=0` → call2
  `cached=1565, cache_write=0`. `prompt_cache_options.mode` explicit/implicit, ttl 30m,
  1024-token min, write billed 1.25×, read −90%.
- **Claude caching** ✅ incl. **1h TTL** (`cache_control.ttl:"1h"` → `cache_creation_input_tokens=2503`).
- **count_tokens** ✅ (`{"input_tokens":9}`) — mantle only, not runtime.
- **Structured output**: Responses `text.format.json_schema` ✅ gemma-4/gpt-5.6/grok;
  CC `response_format` json_object+json_schema ✅ all CC families.
  Claude `output_config.format` ❌ 400 on mantle (use Converse on runtime).
- **Multimodal** ✅ gemma-4 (`input_image`) and qwen3-vl (`image_url`) both read "Red".
  Tiny 1×1 PNG → 400 "Invalid or unsupported image format"; 8×8 works.
- **Tools** ✅ all except `writer.palmyra-vision-7b` (400 — matches its model card).
- **service_tier**: `auto|default|flex|priority` ✅, `reserved` ❌ 400. Echoed in response.

## G. UNDOCUMENTED / CONTRADICTIONS

1. `max_output_tokens` **min 16** on Responses (8 → 400). No CC minimum.
2. `/openai/v1/models` → 404; inventory only at `/v1/models`.
3. Gemma 4 **accepts** JSON-schema constraint keywords (`minLength`,`pattern`) — 200.
   Contradicts internal-code claim of 500s. **Do not sanitize blindly.**
4. Gemma 4 issued **1** tool call when asked for 2 — consistent with "no parallel
   tool calls", but it degrades silently rather than erroring.
5. AWS blog recommends `top_p=0.95` for Gemma 4; Responses **rejects** it.
6. `data_retention` modes: docs/SDK say default|none|provider_data_share|inherit;
   `enabled`/`disabled` → 400.
7. Claude launch dates on model cards say **Jun 10 2025** for Gemma 4 vs Jun 2026 announcement.
8. Quotas: only Claude Opus 4.7 has published TPM (20M in / 4M out). Everything else
   internal capacity. **No RPM quota on mantle.** Increases via Support, not Service Quotas console.

## H. STILL UNVERIFIED

- Guardrail enforcement with a real guardrail ID
- Server-side Lambda MCP + AgentCore Gateway end-to-end (needs deployed Lambda/gateway)
- Reserved tier / capacity reservations (needs account team)
- Files multipart upload + full RFT run (us-west-2, costly)
- Flex vs Priority latency deltas under load
- Batch API end-to-end (needs S3 + service role)

## I. FOUND DURING NOTEBOOK BUILD (round 3)

1. **Gemma 4 "strict" json_schema is NOT reliably strict.** ~4-6 of 8 runs append
   trailing characters after a well-formed object (`\n}`, or unrelated text like
   `\n本项目`). `json.loads()` raises even though the object is correct.
   gpt-5.6-sol produced valid JSON; grok-4.3 returned an EMPTY string for the same
   schema request. Mitigation: `mantle.parse_json_lenient()` walks braces to
   extract the first balanced object. **Never bare-`json.loads()` model output.**
2. `AWS/BedrockMantle` has 8 metrics in practice: BurnDownConsumed,
   EquivalentReservationUnits, InferenceClientErrors, Inferences, InputTokens,
   OutputTokens, TotalInputTokens, TotalOutputTokens. (Still no 5xx metric.)
3. `GET /v1/models/{id}` returns `data_retention` but **not** `allowed_modes` for
   gemma-4 / gpt-5.6 / claude-haiku — treat absence as "inherits account setting".
4. Service Quotas surfaces **zero** "Bedrock Mantle" quotas for this account —
   confirming most models have no published TPM.

## J. FOUND DURING NOTEBOOK EXECUTION (round 4 — all reproduced)

5. **qwen3-coder TRUNCATES tool-call arguments.** `{"path": "calc.py"` with no
   closing brace — **10/10 runs** on a single-argument tool. Two failures follow:
   `json.loads` raises, AND echoing the assistant message back verbatim gets a
   **400** ("Expecting ',' delimiter"). Mitigation: `mantle.parse_json_lenient`
   now closes unbalanced braces; `mantle.repair_tool_arguments` re-serialises
   before echoing.
6. **gpt-oss supports `tool_choice: "auto"` ONLY.** `none`/`required` → 400
   ("Supported options: [\"auto\"]"). The *named-function* form returns **200 but
   silently ignores the constraint** — no tool call, prose instead. 0/4 honoured.
   gpt-5.6 honoured 4/4.
7. **All gpt-5.x reject `service_tier` flex/priority** (400). gpt-oss accepts all
   three. So tier selection must be model-aware.
8. **Forced `tool_choice` on CC families is ~90% reliable** — deepseek returned
   prose with `finish_reason=stop` in 1/10 runs. Always check and retry.
9. **`writer.palmyra-vision-7b` rejects a `system` role** — "Conversation roles
   must alternate user/assistant/…". Also rejects `tools` (matches model card).
10. **MiniMax M2.5 `json_object` returns empty content** at max_tokens=200
    (`finish_reason=length`) — reasoning consumes the budget before the brace.
    Works at 600+. Always check `finish_reason` before parsing.
11. **`claude-fable-5` is gated by data-retention mode**: `status: unavailable`,
    `allowed_modes: ["provider_data_share"]` while the account is `inherit` →
    400 "data retention mode 'default' is not available for this model".
    So `allowed_modes` IS returned for some models (corrects §I.3).
12. **`claude-opus-5` returns a `thinking` block FIRST** even without thinking
    enabled — `content[0].text` is empty/absent. Filter by block type.
13. **Grok `/v1/responses` STALLS** rather than 400ing (45s+ timeout), while
    `/v1/chat/completions` 400s in 0.7s. A retry loop turned this into a 900s
    notebook timeout. Always set client-side timeouts.

## K. FOUND DURING FINAL VERIFICATION (round 5)

14. **`temperature` on the Responses API accepts ONLY its default `1.0`.**
    Swept 0.0 / 0.2 / 0.5 / 0.7 / 0.9 / 1.0 / 1.5 on `google.gemma-4-31b`:
    every value except **1.0 → 400** "Unsupported parameter: 'temperature' is not
    supported with this model." Same on `openai.gpt-5.6-sol` (0.0 and 0.7 → 400,
    1.0 → 200).
    **Chat Completions accepts the full range** (0.0/0.7/1.0 all 200 on gemma-4).
    Corrects §E, which implied gemma-4 accepted `temperature` generally — it does
    not; it accepts only the default. Practical rule: on Responses, omit
    `temperature` and `top_p` entirely unless the model is gpt-oss/qwen-class.
15. Sleep/suspend invalidates a minted token: resuming gives
    `401 invalid_api_key — Signature expired`. Not a code defect; re-mint.

## L. ROUND 6 — corrected by reading the model cards (user prompt: research first)

16. **CORRECTION to §K.14.** The rule is not "temperature must be 1.0" — it is
    **"temperature is accepted only at that model's own documented default"** on the
    Responses API. Grok's model card documents non-standard defaults
    (`temperature=0.7`, `top_p=0.95`, `max_completion_tokens=131072`), and live
    probing matches exactly: Grok accepts 0.7 ✅ / rejects 1.0 ❌; Gemma 4 accepts
    1.0 ✅ / rejects 0.7 ❌. Neither accepts the other's value.
    Practical guidance: omit both params on Responses unless gpt-oss/qwen-class.
17. **Grok reasoning consumes ~400 output tokens before emitting any text.**
    Measured: max_output_tokens=400 → HTTP 200, `status:"incomplete"`,
    reasoning_tokens=397, **0 chars**. 1000 → completed, 260 chars. 2000 → 856 chars.
    `reasoning={"effort":"none"}` → reasoning_tokens=0, full budget for the answer.
    This was the root cause of the notebook "struggling", not a code defect.
18. **Grok reasoning is ENCRYPTED, not readable.** Model card documents
    `include:["reasoning.encrypted_content"]`; verified live — reasoning item keys are
    `['encrypted_content','id','summary','type']`, encrypted_content = 1300 chars.
    Can be replayed on later turns to carry reasoning context. Unlike Gemma 4 /
    gpt-5.x, there is no plaintext trace to display.
19. **mantle emits transient 500 and 503 under load** (observed both on Grok within
    minutes). The OpenAI SDK does **not** retry by default — set
    `OpenAI(..., max_retries=5)`. The `post()` helper already retried, which is why
    helper-based cells survived and bare-SDK cells did not.
