#!/usr/bin/env python3
"""Give GPT-5.6 on bedrock-runtime the full treatment: all three APIs, properly.

4a07af1 named Responses, Chat Completions and Converse but only demonstrated a
one-line call on the first two, with tool use shown on Converse alone. This walks
all three end to end.

Everything below was probed live against us.openai.gpt-5.6-sol in us-east-1 first:

  runtime /responses + tools           -> 200, function_call output item
  runtime /chat/completions + tools    -> 400 unless reasoning_effort="none":
      "Function tools with reasoning_effort are not supported ... in
       /v1/chat/completions. To use function tools, use /v1/responses or set
       reasoning_effort to 'none'."
  runtime /responses + stream          -> 200, typed response.* SSE events
  runtime /responses + store           -> 200, previous_response_id chaining works
  runtime /responses prompt caching    -> cache_write_tokens then cached_tokens
  structured outputs on BOTH OpenAI APIs -> 200 and schema-valid JSON, even though
      the model card lists "Structured outputs" as not supported on bedrock-runtime.
      The notebook now probes it and says the card disagrees, rather than repeating
      a claim I had copied without testing.
"""

import sys

import nbformat

NB = "01-openai-gpt/01-responses-api-core.ipynb"

THREE_APIS_MD = """\
### Three APIs, one model — what actually differs

`bedrock-runtime` serves this model on Responses, Chat Completions and Converse. The
model is identical; what changes is the envelope you write and the reply you parse.

| | Responses | Chat Completions | Converse |
|---|---|---|---|
| Path / call | `POST /openai/v1/responses` | `POST /openai/v1/chat/completions` | `bedrock-runtime` SDK `converse()` |
| Input field | `input` (string or list) | `messages` | `messages` with typed content blocks |
| Output cap | `max_output_tokens` | `max_completion_tokens` | `inferenceConfig.maxTokens` |
| Where the answer is | `output[]` → `content[]` → `output_text` | `choices[0].message.content` | `output.message.content[]` → `text` |
| Reasoning **and** tools together | yes | **no** — see below | tools yes, trace not returned |
| Auth | SigV4 | SigV4 | SigV4 via the SDK |

Reach for Responses if you are writing new OpenAI-shaped code, Chat Completions if you
have an existing corpus of it, and Converse if you want one request shape across every
provider on Bedrock.

The cell below sends the same question four ways — once to `bedrock-mantle` and three
times to `bedrock-runtime` — so the shape differences are visible next to each other.
"""

FOUR_WAYS_CODE = '''\
import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

RUNTIME_BASE = f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1"
PROFILE = f"us.{MODEL}"


def runtime_openai_post(path: str, body: dict, stream: bool = False):
    """POST an OpenAI-shaped payload to bedrock-runtime, signed with SigV4.

    The OpenAI SDK cannot do this - it has no way to SigV4-sign - so the signing is
    spelled out. Note the service name stays "bedrock" even though the host is
    bedrock-runtime.
    """
    payload = json.dumps(body)
    signed = AWSRequest(
        method="POST",
        url=f"{RUNTIME_BASE}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(
        boto3.Session(region_name=REGION).get_credentials().get_frozen_credentials(),
        "bedrock",
        REGION,
    ).add_auth(signed)
    reply = requests.post(
        f"{RUNTIME_BASE}{path}",
        headers=dict(signed.headers),
        data=payload,
        timeout=120,
        stream=stream,
    )
    return reply


QUESTION = "In one sentence: what is idempotency?"

# 1. bedrock-mantle, Responses - the bearer-token path used everywhere above.
code, mantle = post(f"{GPT5_PREFIX}/responses", {"model": MODEL, "input": QUESTION, "max_output_tokens": 60})
print(f"mantle   /openai/v1/responses         {code}  {response_text(mantle)[:78]}")

# 2. bedrock-runtime, Responses - same body, SigV4 instead of a token.
reply = runtime_openai_post("/responses", {"model": PROFILE, "input": QUESTION, "max_output_tokens": 60})
print(f"runtime  /openai/v1/responses         {reply.status_code}  {response_text(reply.json())[:78]}")

# 3. bedrock-runtime, Chat Completions - messages[] and max_completion_tokens.
reply = runtime_openai_post(
    "/chat/completions",
    {"model": PROFILE, "messages": [{"role": "user", "content": QUESTION}], "max_completion_tokens": 60},
)
print(f"runtime  /openai/v1/chat/completions  {reply.status_code}  "
      f"{reply.json()['choices'][0]['message']['content'][:78]}")

# 4. bedrock-runtime, Converse - typed content blocks, via the AWS SDK.
text, _ = converse(MODEL, [{"role": "user", "content": [{"text": QUESTION}]}], max_tokens=60)
print(f"runtime  Converse                     200  {text[:78]}")
'''

TOOLS_MD = """\
### Tool use: the one place the APIs genuinely disagree

Every path supports function tools, but **Chat Completions refuses tools and reasoning
in the same request**. Ask for both and you get a 400 that, unusually, tells you
exactly what to do:

> Function tools with `reasoning_effort` are not supported for
> `us.openai.gpt-5.6-sol` in `/v1/chat/completions`. To use function tools, use
> `/v1/responses` or set `reasoning_effort` to `'none'`.

So on Chat Completions you choose: reasoning, or tools. Responses and Converse give
you both. If you are moving an agent onto this model, that single constraint is the
strongest argument for Responses over Chat Completions.

Note also that the three APIs return a tool call in three different shapes, so the
parsing is never portable even when the request nearly is.
"""

TOOLS_CODE = '''\
STOCK_TOOL_RESPONSES = [
    {
        "type": "function",
        "name": "get_stock_price",
        "description": "Current share price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    }
]
# Chat Completions nests the same thing one level deeper, under "function".
STOCK_TOOL_CHAT = [{"type": "function", "function": STOCK_TOOL_RESPONSES[0].copy()}]
del STOCK_TOOL_CHAT[0]["function"]["type"]

ASK = "What is AMZN trading at? Use the tool."

# Responses: tools and reasoning coexist.
reply = runtime_openai_post(
    "/responses", {"model": PROFILE, "input": ASK, "tools": STOCK_TOOL_RESPONSES, "max_output_tokens": 300}
)
items = [item.get("type") for item in reply.json().get("output", [])]
calls = [(i.get("name"), i.get("arguments")) for i in reply.json().get("output", []) if i.get("type") == "function_call"]
print(f"responses         {reply.status_code}  output items={items}\\n                       calls={calls}")

# Chat Completions, reasoning left at its default: rejected.
reply = runtime_openai_post(
    "/chat/completions",
    {"model": PROFILE, "messages": [{"role": "user", "content": ASK}], "tools": STOCK_TOOL_CHAT,
     "max_completion_tokens": 300},
)
print(f"\\nchat/completions  {reply.status_code}  {reply.json()['error']['message']}")

# Chat Completions with reasoning switched off: accepted.
reply = runtime_openai_post(
    "/chat/completions",
    {"model": PROFILE, "messages": [{"role": "user", "content": ASK}], "tools": STOCK_TOOL_CHAT,
     "reasoning_effort": "none", "max_completion_tokens": 300},
)
choice = reply.json()["choices"][0]
print(f'\\nchat/completions  {reply.status_code}  reasoning_effort="none" -> finish={choice["finish_reason"]}')
for call in choice["message"].get("tool_calls") or []:
    print(f"                       {call['function']['name']}({call['function']['arguments']})")
'''

REST_MD = """\
### What else carries over to `bedrock-runtime`

Four things people assume are `bedrock-mantle` features, probed here rather than
assumed:

- **Streaming.** `"stream": true` on Responses returns the same typed
  `response.*` SSE events.
- **Prompt caching.** Supported on the Responses API. `usage.input_tokens_details`
  reports `cache_write_tokens` on the first call and `cached_tokens` on a repeat.
- **Server-side state.** `store: true` plus `previous_response_id` chains turns
  without you resending history.
- **Structured outputs.** Both OpenAI APIs accept a JSON schema and honour it —
  `text.format` on Responses, `response_format` on Chat Completions.

The last one is worth flagging: the model card lists *Structured outputs* under **not
supported** for `bedrock-runtime`, and the API plainly accepts them. Either the card
means the Bedrock-native feature rather than the OpenAI schema field, or it is behind.
The cell below is the evidence; trust it over this paragraph and over the card, and
re-run it before you rely on the behaviour.
"""

REST_CODE = '''\
# 1. Streaming.
reply = runtime_openai_post(
    "/responses", {"model": PROFILE, "input": "Count to three.", "max_output_tokens": 40, "stream": True},
    stream=True,
)
events = []
for line in reply.iter_lines():
    if line and line.startswith(b"data: "):
        try:
            events.append(json.loads(line[6:]).get("type"))
        except json.JSONDecodeError:
            pass
print(f"streaming         {reply.status_code}  {len(events)} events, e.g. {sorted(set(events))[:3]}")

# 2. Prompt caching. The system block has to be big enough to be worth caching.
PRIMER = [{"role": "system", "content": "You are a terse assistant. " * 400},
          {"role": "user", "content": "Say ok."}]
for attempt in (1, 2):
    reply = runtime_openai_post("/responses", {"model": PROFILE, "input": PRIMER, "max_output_tokens": 20})
    detail = reply.json()["usage"]["input_tokens_details"]
    print(f"caching call {attempt}    {reply.status_code}  "
          f"cache_write={detail.get('cache_write_tokens')} cached={detail.get('cached_tokens')}")

# 3. Server-side state.
first = runtime_openai_post(
    "/responses", {"model": PROFILE, "input": "Remember the number 7.", "max_output_tokens": 30, "store": True}
).json()
follow = runtime_openai_post(
    "/responses",
    {"model": PROFILE, "input": "Which number did I ask you to remember?",
     "previous_response_id": first["id"], "max_output_tokens": 30},
)
print(f"\\nserver-side state 200  chained reply: {response_text(follow.json())[:60]}")

# 4. Structured outputs, both OpenAI APIs.
PLACE_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": False,
}
reply = runtime_openai_post(
    "/responses",
    {"model": PROFILE, "input": "Rome, Italy", "max_output_tokens": 100,
     "text": {"format": {"type": "json_schema", "name": "place", "schema": PLACE_SCHEMA, "strict": True}}},
)
print(f"\\nstructured (resp) {reply.status_code}  {response_text(reply.json())[:60]}")

reply = runtime_openai_post(
    "/chat/completions",
    {"model": PROFILE, "messages": [{"role": "user", "content": "Rome, Italy"}],
     "max_completion_tokens": 100, "reasoning_effort": "none",
     "response_format": {"type": "json_schema",
                         "json_schema": {"name": "place", "schema": PLACE_SCHEMA, "strict": True}}},
)
print(f"structured (chat) {reply.status_code}  {reply.json()['choices'][0]['message']['content'][:60]}")
'''

CONVERSE_MD = """\
### Converse: one shape for every provider

Converse is the reason to be on `bedrock-runtime` if you run more than one model
family. The request and reply shape is identical across providers, so an agent loop
written for Claude or Nova runs against this model unchanged — no new parsing.

Tool use returns an ordinary `toolUse` block. The reasoning trace does **not** come
back: `content` carries a `text` block and nothing else, so the thinking is billed and
unreadable, the same trade the `/v1` families make on Chat Completions.
"""

FEATURES_MD = """\
### Choosing an endpoint for this model

| | `bedrock-runtime` | `bedrock-mantle` |
|---|---|---|
| APIs | Responses · Chat Completions · Converse | Responses · Chat Completions |
| Auth | SigV4 — nothing to mint | bearer token, short-lived |
| Cross-Region inference | **required** (geo or global) | not supported |
| Regional reach | wide | narrow |
| Streaming · prompt caching · server-side state · structured outputs | yes (caching is Responses-only) | yes |
| Server-side tools (web search, code interpreter) | **no** | yes |
| Guardrails | Converse only | — |
| Invocation logs · CloudWatch metrics · Cost Explorer per-model lines | **yes** | limited |
| One request shape across providers | **yes**, via Converse | no |

The trade in one line: `bedrock-mantle` for the server-side tools, `bedrock-runtime`
for AWS-native governance, cross-Region throughput and Converse portability. The
public model card recommends `bedrock-runtime` for new applications.

Feature matrices move faster than notebooks — and as the structured-output row above
shows, the card and the API can disagree. Probe before you commit.
"""


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    path = f"{root.rstrip('/')}/{NB}"
    nb = nbformat.read(path, as_version=4)

    start = next(i for i, c in enumerate(nb.cells)
                 if c.cell_type == "markdown" and c.source.startswith("## Also on"))
    assert len(nb.cells) - start == 10, f"expected 10 trailing cells, found {len(nb.cells) - start}"

    intro = nb.cells[start].source.replace(
        "yes, and on three APIs", "yes, on all three APIs")
    keep = {
        "intro": intro,
        "availability": nb.cells[start + 1].source,
        "bare_md": nb.cells[start + 2].source,
        "bare_code": nb.cells[start + 3].source,
        "geo_md": nb.cells[start + 4].source,
        "converse_code": nb.cells[start + 8].source,
    }

    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    rebuilt = [
        md(keep["intro"]),
        code(keep["availability"]),
        md(keep["bare_md"]),
        code(keep["bare_code"]),
        md(keep["geo_md"]),
        md(THREE_APIS_MD),
        code(FOUR_WAYS_CODE),
        md(TOOLS_MD),
        code(TOOLS_CODE),
        md(REST_MD),
        code(REST_CODE),
        md(CONVERSE_MD),
        code(keep["converse_code"]),
        md(FEATURES_MD),
    ]
    for cell in rebuilt:
        cell.pop("id", None)

    nb.cells[start:] = rebuilt
    nbformat.write(nb, path)
    print(f"section rebuilt: {len(rebuilt)} cells from {start} "
          f"({sum(1 for c in rebuilt if c.cell_type == 'code')} code)")


if __name__ == "__main__":
    main()
