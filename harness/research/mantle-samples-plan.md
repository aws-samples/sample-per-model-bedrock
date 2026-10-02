# Amazon Bedrock Mantle Samples — coverage plan

Repo shape: one folder per model family, numbered notebooks inside, sections
flowing as a natural hardening arc (simple → features → agentic → production).

Grounded in `mantle-capability-matrix.md` (all capability claims probed live
2026-07-30). **55 models / 12 families.**

## Design rules

1. **Three notebook archetypes**, driven by which API a family actually supports:
   - **R** = Responses-primary (`/openai/v1/*` or `/v1/*`) — google, openai, xai
   - **C** = Chat-Completions-only (`/v1/chat/completions`) — 8 families
   - **M** = Messages-only (`/anthropic/v1/messages`) — anthropic
2. **Pin the Region per family.** Footprint differs sharply (us-east-1 = 55 models,
   eu-central-1 = 33, no Anthropic outside us-east-1 except haiku).
3. **Interleave models within a family** — never one model per section.
4. Every notebook ends with the same three closers: quotas/retries, cost &
   attribution (Projects), gotchas table.
5. Cells-only (no saved outputs); `max_output_tokens` small; CI smoke-tests §1 of each.

---

# 00-foundations/  (read first — REQUIRED)

## `01-endpoints-auth-and-the-three-paths.ipynb`
1. What Mantle is; why a second endpoint exists next to `bedrock-runtime`
2. Endpoint/API/auth decision table; when to stay on `bedrock-runtime`
   (CRIS, Provisioned Throughput, Batch, Guardrails, invocation logging)
3. **Auth A — SigV4** with botocore (signing name `bedrock`; note `bedrock-mantle`
   also works). No boto3 `bedrock-mantle` client exists — SDK is a *signer*
4. **Auth B — short-term API key** via `aws-bedrock-token-generator`;
   self-refreshing provider class; why long-term keys are exploration-only
5. **Auth C — curl** `--aws-sigv4` and bearer
6. **The three path families** (the single most important cell):
   `/openai/v1/*` · `/v1/*` · `/anthropic/v1/messages` · control-plane `/v1/*`
7. Model discovery — `GET /v1/models` (note `/openai/v1/models` → **404**),
   group by family, build a live capability table
8. Region footprint sweep across all 4 Regions; print the availability matrix
9. IAM: `AmazonBedrockMantleInferenceAccess` vs `…FullAccess`;
   `CreateInference` + `CallWithBearerToken`; `bearerTokenType` SCP for banning
   long-term keys; SigV4 callers don't need `CallWithBearerToken`

## `02-governance-projects-retention-and-observability.ipynb`
1. Projects = workload boundary; vs Inference Profiles; vs AWS accounts
2. Create a project with tags → `arn`, `id`
3. Attribution three ways: `OpenAI-Project` (Responses/CC), `anthropic-workspace`
   (Messages) — same underlying resource
4. Update tags (add/remove only — no full replace), list, retrieve, archive
   (archived = no new inference, data 30d)
5. **Data retention & ZDR** — `GET /v1/data_retention` (account),
   project `data_retention.mode`, per-model `allowed_modes`.
   Modes: `default` · `none` (=ZDR) · `provider_data_share` · `inherit`.
   (`enabled`/`disabled` → 400.) How `store:true` interacts (30-day, in-Region)
6. Cost allocation via tags → Cost Explorer
7. Observability: `AWS/BedrockMantle` namespace (**not** `AWS/Bedrock`);
   only `InferenceClientErrors` (4xx) — **no 5xx metric**;
   CloudTrail needs **data events** (extra cost); `AWS::BedrockMantle::Project` in CFN
8. `AWS/Bedrock` dashboards show zero during a mantle 503 storm — the trap

## `03-scaling-tiers-and-latency.ipynb`
1. Quota model: separate **input TPM** and **output TPM**, **no RPM quota**;
   `max_tokens` is reserved up-front then replenished
2. Only Claude Opus 4.7 has published TPM; everything else internal capacity →
   increases via Support case, not the Service Quotas console
3. Cached tokens don't count against input TPM
4. Retry/backoff harness (429 + 5xx transient); ramp-not-spike guidance
5. **Service tiers** — `auto|default|flex|priority` (✅), `reserved` → **400**.
   Resolved tier echoed in the response body
6. Measure **TTFT** and **OTPS** per tier, same prompt, tabulate + chart
7. Prefill vs decode: why long prompts raise TTFT; where caching helps
8. `PrivateLink` for egress cost; timeouts & connection reuse

---

# 01-openai-gpt/  (archetype R — richest family, 11 models)

Region: **us-east-1** (gpt-5.5 / 5.6-sol absent from us-west-2).

## `01-responses-api-core.ipynb`
1. First call — `responses.create(input=...)`, read `output_text`
2. `input` as a **message array** (system + user + assistant), multi-turn by hand
3. Streaming — `response.output_text.delta`, `response.reasoning_text.delta`,
   `response.reasoning_part.added|done`
4. Reasoning: `reasoning.effort` `none|low|medium|high` (**`minimal` → 400**);
   reasoning visible only on Responses, never on Chat Completions
5. `text.verbosity`, `truncation`, `metadata`, `max_tool_calls`
6. **Stateful** — `previous_response_id`; `store:false` ⇒ chaining 404s
7. `GET /openai/v1/responses/{id}`, `DELETE` (note `/input_items` → 404)
8. Async — `background:true` + poll to `completed`
9. Interleave: gpt-5.6-sol (frontier) vs gpt-5.4 (cheaper) vs terra/luna
10. `max_output_tokens` **minimum is 16**

## `02-web-search-and-grounding.ipynb`
1. Why grounding; Amazon-built index + knowledge graph; Search vs Fetch
2. Minimal call — `tools:[{"type":"web_search"}]`
3. **Governance first**: `external_web_access` defaults **true**, but
   `AmazonBedrockFullAccess` lacks `bedrock-websearch:ExternalWebAccess` →
   **403 on authz** (model still answers from index). Set `false` to stay in-boundary
4. IAM: `InvokeSearch`, `InvokeFetch`, `ExternalWebAccess`
5. Inspect the multi-round loop — repeated `web_search_call` items in `output`
6. **Citations** — `url_citation` annotations (title/url/start_index/end_index);
   retaining + displaying them is an acceptable-use requirement
7. Streaming citations — `response.output_text.annotation.added`
8. Regional: us-east-1/2 + us-west-2 only, strictly in-Region
9. Negative control: same call on gemma-4 / grok → **400 not supported**
10. Cost note: Web Search is billed separately from tokens

## `03-tools-and-structured-output.ipynb`
1. Client-side tool loop — flat Responses tool shape (`name`/`parameters` at top
   level, unlike CC's nested `function`)
2. `function_call` / `function_call_output` items with `call_id`
3. `tool_choice` auto → required → forced-specific
4. `parallel_tool_calls`
5. **Strict JSON** two ways: `text.format.json_schema` (native) and
   forced-tool-arguments (portable fallback for models lacking it)
6. Schema design that survives: which keywords are safe
7. Pydantic round-trip validation
8. Tool errors and retries

## `04-prompt-caching-and-cost.ipynb`
1. gpt-5.5-and-earlier: **automatic** caching, 1024-token prefix, no write fee
2. gpt-5.6: **explicit breakpoints** —
   `prompt_cache_breakpoint:{mode:"explicit"}` on `input_text`/`input_image`/`input_file`
3. `prompt_cache_options` `mode: implicit|explicit`, `ttl` (default 30m)
4. Read `usage.input_tokens_details.cached_tokens` / `cache_write_tokens`
   — demo: call 1 `write=1565, cached=0` → call 2 `cached=1565, write=0`
5. Economics: write 1.25×, read −90%; cached tokens exempt from input TPM
6. Prompt layout: static prefix first, variable suffix last
7. Agentic loop where system+tools stay cached across turns

## `05-server-side-tools-and-fine-tuning.ipynb`
1. Client-side vs **server-side** vs Anthropic tool modes
2. Built-in **notes** and **tasks** on gpt-oss (no declaration; emits `mcp_call`)
3. Custom **Lambda MCP** tool — JSON-RPC `tools/list` + `tools/call`;
   `tools:[{type:"mcp",connector_id:<lambda-arn>,require_approval:"never"}]`;
   Lambda must carry the caller's IAM policy; `mcp_list_tools` in output
4. **AgentCore Gateway** connector — same shape, gateway ARN; works on all
   Responses models; multiple tool calls per turn
5. gpt-oss-safeguard 20b/120b — classification/guardrail-style usage
6. **Reinforcement fine-tuning** (us-west-2 only, gpt-oss-20b):
   Files API upload (`purpose:"fine-tune"`) → Lambda reward fn → create job →
   events/checkpoints → inference on the FT model id
7. **Batch is on `bedrock-runtime`**, not mantle — `/openai/v1/batches`,
   `X-Amzn-Bedrock-RoleArn` + `ModelId`, S3 in/out, no tools or structured output

---

# 02-anthropic-claude/  (archetype M — 6 models)

Region: **us-east-1** (only Region with the full Claude set; us-west-2 has haiku-4-5 only).
Claude is **Messages-only** on mantle — Responses and Chat Completions both 400.

## `01-messages-api-core.ipynb`
1. Why Messages here (Responses/CC both 400 for Claude)
2. `anthropic-version: 2023-06-01` **header** on mantle vs
   `anthropic_version` in body on runtime
3. Auth: bearer `x-api-key` or `Authorization`; SigV4; Anthropic SDK with `base_url`
4. `system`, multi-turn alternating roles, assistant prefill
5. Streaming SSE — `message_start` → `content_block_delta` → `message_stop`
6. **`temperature` is deprecated → 400** on sonnet-5 / opus-4-8 / opus-5;
   still accepted on haiku-4-5. Interleave to show the split
7. `count_tokens` — **mantle-only**, pre-flight cost estimation
8. Vision; documents; `stop_sequences`

## `02-thinking-tools-and-caching.ipynb`
1. `thinking:{type:"adaptive"}` ✅ sonnet-5/opus-4-8/opus-5, **❌ 400 haiku-4-5**
2. Extended thinking budgets; streaming required above ~21k max_tokens;
   thinking incompatible with temperature/top_p/forced tools
3. Multi-turn: don't replay thinking blocks
4. Tool use — `input_schema`, `tool_choice:{type:"tool"}`
5. **Structured output**: `output_config.format` → **400 on mantle**;
   fall back to forced tool args, or Converse on `bedrock-runtime`
6. **Prompt caching** — `cache_control:{type:"ephemeral"}`; 4-checkpoint max;
   order `tools → system → messages`; **1-hour TTL** verified
   (`cache_creation_input_tokens=2503`); longer TTL must precede shorter
7. Automatic cache management (~20-block lookback)
8. Workspaces via `anthropic-workspace` for per-app cost attribution

## `03-agentic-computer-use-and-memory.ipynb`
1. Beta-service warnings, sandboxing, human-in-the-loop, prompt-injection risk
2. **Computer use** — `anthropic-beta: computer-use-2025-11-24` header (mantle) +
   `computer_20251124`; the screenshot→action→screenshot loop
3. `bash_20250124`, `text_editor_20250124`
4. **Memory tool** — `memory_20250818` + `context-management-2025-06-27`
5. **Compaction** — `context_management.edits:[{type:"clear_tool_uses_20250919"}]`
6. Interleaved thinking between tool calls
7. Long-running agent loop: cache + compaction + memory together
8. Claude Code / LLM-gateway pointed at mantle (env-var config)

---

# 03-google-gemma/  (archetype R — 6 models)

Region: any of the four (gemma-4 is the **only** family in all 4).

## `01-gemma4-end-to-end.ipynb`
1. Family shape: 31b dense / 26b-a4b MoE / e2b compact; pick-a-variant table
2. First call on Responses (`/openai/v1` — **note the `/openai` prefix**, unlike most)
3. **`temperature=1.0` required, `top_p` → 400** — inverse of Grok.
   AWS's own blog recommends `top_p=0.95`; Responses rejects it
4. Streaming; reasoning (`effort` high recommended for **e2b** to stop leakage)
5. Multi-turn by array, then `previous_response_id`; `store:false` trade-off
6. Chat Completions detour — same model, when CC is the better fit,
   and why reasoning content vanishes there
7. Tool use; forced-tool structured output; `text.format.json_schema` also ✅
8. **Only one tool call per turn** — asked for 2, got 1, silently
9. Multimodal image (`input_image`); ≤32 images/request; 3.5 MB body;
   tiny 1×1 PNG rejected as "Invalid or unsupported image format"
10. Interleave 31b vs 26b-a4b vs e2b: cost/latency/quality on one task
11. Gotchas + quotas + project attribution

---

# 04-qwen/  (archetype C — 7 models)

Region: us-east-1 (all 7). Note `qwen3-32b` is one of only two FT-able models (us-west-2).

## `01-qwen3-core-and-tools.ipynb`
1. Chat Completions is the family's only API (**Responses → 400**)
2. First call; `temperature` **and** `top_p` both ✅ (unlike gemma/grok)
3. Streaming SSE deltas
4. `reasoning_effort` on CC; strip stray `<think>` blocks
5. Multi-turn — you own the history (no `previous_response_id` on CC)
6. Tool use — nested `function` shape; forced `tool_choice`
7. Structured output — `response_format` `json_object` **and** `json_schema` ✅
8. Interleave 32b / 235b-a22b / next-80b-a3b on the same task

## `02-qwen3-coder-and-vision.ipynb`
1. Coder family: 30b-a3b / 480b-a35b / coder-next — code gen, diff, repair
2. Long-context code review; fill-in-the-middle style prompting
3. Agentic coding loop with client-side tools (read/write/run)
4. `qwen3-vl-235b-a22b` — image input via CC `image_url` (verified)
5. Chart/document extraction → strict JSON
6. Coder-vs-general comparison table

---

# 05-deepseek/  (archetype C — 2 models)

## `01-deepseek-v3-reasoning.ipynb`
1. CC-only; v3.1 vs v3.2 interleaved
2. Basics, streaming, sampling params
3. `reasoning_effort`; reasoning-trace handling in CC
4. Multi-turn; tools; forced tools
5. `response_format` json_object + json_schema
6. Math/logic benchmark mini-suite comparing efforts
7. Cost per solved problem; gotchas

---

# 06-zai-glm/  (archetype C — 4 models)

## `01-glm-family.ipynb`
1. CC-only; glm-4.6 / 4.7 / 4.7-flash / glm-5
2. Basics, streaming, params
3. `reasoning_effort`; tools; forced tools; structured output
4. **Flash vs full** latency/cost head-to-head (4.7-flash is the only Region-wide
   GLM in eu-central-1 alongside 4.6)
5. Agentic loop; escalation pattern (flash first, escalate to glm-5)
6. Gotchas

---

# 07-mistral/  (archetype C — 8 models, widest size range)

## `01-mistral-text-and-sizes.ipynb`
1. CC-only; the size ladder: ministral 3b → 8b → 14b → mistral-large-3-675b
2. Basics, streaming, params; `reasoning_effort`
3. Tools + structured output across sizes — where small models start failing
4. **Routing pattern**: cheapest model that passes a validation gate
5. `magistral-small-2509` reasoning variant

## `02-devstral-and-voxtral.ipynb`
1. `devstral-2-123b` — agentic coding; multi-file edit loop
2. `voxtral-mini-3b` / `voxtral-small-24b` — audio-capable variants;
   what actually works on mantle CC vs what needs runtime
3. Transcription → structured JSON pipeline
4. Note: mistral-large-3 is **absent from eu-central-1**

---

# 08-moonshot-kimi/  (archetype C — 2 models)

## `01-kimi-k2.ipynb`
1. CC-only; `kimi-k2-thinking` vs `kimi-k2.5`
2. Basics, streaming, params
3. Thinking-model behaviour on CC; `reasoning_effort`
4. Tools, forced tools, structured output
5. Long-context / agentic task
6. Absent from eu-central-1 — Region note

---

# 09-minimax/  (archetype C — 3 models)

## `01-minimax-m2.ipynb`
1. CC-only; m2 / m2.1 / m2.5 generational interleave
2. Basics, streaming, params, `reasoning_effort`
3. Tools; structured output
4. Agentic/tool-heavy workload (M2's stated strength)
5. Version-migration section: same prompt across all three

---

# 10-nvidia-nemotron/  (archetype C — 4 models)

## `01-nemotron-nano-and-super.ipynb`
1. CC-only; nano-9b-v2 / nano-12b-v2 / nano-3-30b / super-3-120b
2. Basics, streaming, params, `reasoning_effort`
3. Tools; structured output
4. Nano-vs-Super cost/quality curve; edge-style small-model framing
5. `nemotron-nano-12b-v2` VL usage

---

# 11-xai-grok/  (archetype R — 1 model, but on `/openai/v1`)

## `01-grok-4-3.ipynb`
1. Responses **and** Chat Completions, both on `/openai/v1`
2. **`temperature` → 400 "deprecated"; `top_p` ✅** — exact inverse of Gemma 4.
   The cell that proves sampling defaults can't be shared across families
3. Always-on reasoning; `effort` none/low/medium/high
4. Streaming; multi-turn; `previous_response_id`
5. Tools; forced tools; `text.format.json_schema` ✅
6. `web_search` → **400 not supported** (contrast with gpt-5.x)
7. Enterprise doc-analysis walkthrough (contract/case-law framing per model card)

---

# 12-writer-palmyra/  (archetype C — 1 model, the "limits" case)

## `01-palmyra-vision.ipynb`
1. CC-only; vision-first model
2. Basics, streaming, params
3. **Tool calling → 400** (matches its model card: "Client-side tool calling: Not
   Supported"). The honest negative case — and the workaround: prompt-only
   extraction + `response_format` json_schema (which **does** work)
4. Image → structured JSON without tools
5. When to pick a specialised vision model over a general multimodal one

---

# 99-cross-cutting/  (advanced; needs deployed prerequisites)

## `01-server-side-tools-deep-dive.ipynb`
Lambda MCP end-to-end (deploy the Lambda in-notebook), AgentCore Gateway,
approval modes, failure semantics, IAM parity requirement.

## `02-migrating-from-openai-and-gateways.ipynb`
Base-URL-and-key swap; LiteLLM / gateway config; per-family shims for the
sampling-param and path differences; a `capabilities.py` helper that resolves
path + params per model id.

## `03-production-hardening-checklist.ipynb`
Token refresh, retries, timeouts, idempotency, cost dashboards, ZDR posture,
tier selection, canary + fallback across families, quota escalation path.

---

# Coverage matrix (folder × capability)

| Folder | API | Tools | StructOut | Cache | Reason | Stateful | Multimodal | WebSearch | Server-side | FT |
|---|---|---|---|---|---|---|---|---|---|---|
| 01-openai-gpt | R+C | ✅ | ✅ | ✅ explicit | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 02-anthropic | M | ✅ | ⚠️ fallback | ✅ 1h TTL | ✅ adaptive | ❌ | ✅ | ❌ | ❌ | ❌ |
| 03-google-gemma | R+C | ✅ 1/turn | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| 04-qwen | C | ✅ | ✅ | ❌ | ✅ effort | ❌ | ✅ VL | ❌ | ❌ | ✅ 32b |
| 05-deepseek | C | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 06-zai-glm | C | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 07-mistral | C | ✅ | ✅ | ❌ | ✅ | ❌ | 🔊 audio | ❌ | ❌ | ❌ |
| 08-moonshot | C | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 09-minimax | C | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 10-nvidia | C | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ 12b | ❌ | ❌ | ❌ |
| 11-xai-grok | R+C | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 12-writer | C | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |

Totals: **13 folders, 27 notebooks.**

# Region pinning per folder

| Folder | Region | Why |
|---|---|---|
| 00-foundations | us-east-1 | all 55 models present |
| 01-openai-gpt | us-east-1 | gpt-5.5 / 5.6-sol absent in us-west-2 |
| ↳ §05 fine-tuning | **us-west-2** | only FT Region |
| 02-anthropic | us-east-1 | only Region with all 6 Claude models |
| 03-google-gemma | us-east-1 (portable to all 4) | only family in every Region |
| 04–10, 12 | us-east-1 | full family present |
| 11-xai-grok | us-east-1 | absent from eu-central-1 |

# Shared assets

- `_shared/mantle.py` — client factory (SigV4 + token), path resolver per model id,
  retry/backoff, streaming helpers, TTFT timer
- `_shared/capabilities.json` — generated from `/v1/models` + probe results
- `requirements.txt`, `README.md` with the three-path diagram
- `scripts/smoke_test.py` — runs §1 of every notebook in CI

# Open questions before build

1. **CC-only folders (05,06,08,09,10)** — keep 1:1 as above, or merge the thin ones
   into a single `open-weight-models/` notebook with interleaved families? Spec says
   1:1; risk is 5 near-duplicate notebooks whose only difference is model character.
2. **Guardrails** — headers return 200 with a *fake* ID and model cards list
   Guardrails under `bedrock-runtime` only. Needs a real guardrail ID to decide
   whether this gets a section or an explicit "not supported on mantle" note.
3. **99-cross-cutting** requires deploying a Lambda + AgentCore Gateway. In scope,
   or defer to a follow-up PR?
