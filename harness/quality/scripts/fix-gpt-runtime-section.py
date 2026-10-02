#!/usr/bin/env python3
"""Correct and complete the GPT-5.6 bedrock-runtime section.

What 04c128e got right: GPT-5.6 is on bedrock-runtime, and the bare model ID is
rejected because in-Region invocation is not offered there. The model card confirms
it — "In-Region endpoint URL: Not supported" for the bedrock-runtime row.

What it got wrong: it framed bedrock-runtime as Converse-only. The launch
(What's New, 17 Aug 2026) adds the **Responses and Chat Completions APIs** on
bedrock-runtime as well, at https://bedrock-runtime.{region}.amazonaws.com/openai/v1,
authenticated with plain SigV4 — no bearer token to mint. Verified live.

Also missing: the geo-vs-global distinction that actually decides which profile a
customer picks (data residency, and Global priced lower per token), and the
per-endpoint feature deltas from the model card.

Sources:
  https://aws.amazon.com/about-aws/whats-new/2026/08/amazon-bedrock-cross-region-openai-v2/
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html
  https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
"""

import sys

import nbformat

NB = "01-openai-gpt/01-responses-api-core.ipynb"

INTRO = """\
## Also on `bedrock-runtime`? GPT-5.6 — yes, and on three APIs

This is the clearest example in the collection of why these notebooks probe instead
of assert. Until August 2026 the GPT-5.x models were `bedrock-mantle` only, and this
section said so. Then [the 17 August 2026
launch](https://aws.amazon.com/about-aws/whats-new/2026/08/amazon-bedrock-cross-region-openai-v2/)
put the GPT-5.6 family (`sol`, `terra`, `luna`) on `bedrock-runtime` — with the
**Responses, Chat Completions *and* Converse** APIs — plus cross-Region inference.
The sentence became false without anything in the notebook changing.

Two things follow, and they are worth separating:

- **Cross-Region inference is now mandatory here.** On `bedrock-runtime` this model
  has no in-Region option at all, so every call names an inference profile.
- **The OpenAI-shaped APIs are not exclusive to `bedrock-mantle` any more.** The same
  Responses payload works on `bedrock-runtime` at
  `https://bedrock-runtime.{region}.amazonaws.com/openai/v1` — signed with SigV4,
  with no bearer token to mint.

The model card carries a tip worth repeating: *"Whenever possible, we recommend using
the `bedrock-runtime` endpoint for new applications."* Earlier GPT-5.x models may
still be mantle-only, so the probe below asks the live catalogues rather than
trusting this paragraph.
"""

BARE_ID_MD = """\
### The bare model ID is rejected, and the error misdirects

Converse will not take `openai.gpt-5.6-sol` as-is. It answers:

> `ValidationException` — Invocation of model ID ... with on-demand throughput
> isn't supported. Retry your request with the ID or ARN of an inference profile
> that contains this model.

That reads like a permissions or entitlement problem. It is not, and it is not a
quirk either — the model card lists the in-Region endpoint URL for the
`bedrock-runtime` row as **"Not supported"**. In-Region invocation simply is not
offered for this model on this endpoint, so the ID must carry a Region scope:
`us.` for the geographic profile, `global.` for the global one.

Most Claude models behave the same way, which is why
[`resolve_runtime_id()`](../_shared/bedrock.py) exists: it looks the model up in the
profile list and returns the form Converse will accept.

The cell below passes `resolve=False` to defeat that helper, so you can see the raw
failure and the fix side by side.
"""

GEO_GLOBAL_MD = """\
### Choosing between `us.` and `global.`

Both work. They are not interchangeable, and the difference is not performance:

| | `us.` — geographic | `global.` — global |
|---|---|---|
| Where inference runs | a commercial Region **inside the US geography** | **any** supported commercial Region worldwide |
| Reach for it when | you have data-residency obligations | you do not, and you want the most capacity |
| Throughput | good | highest — the widest pool during demand spikes |
| Token price | same as in-Region | **lower** for OpenAI models |

Neither changes where your data is *stored*. Cross-Region inference moves the
transient computation only: invocation logs, knowledge bases and configuration stay
in the source Region, and traffic crosses the AWS network encrypted. What moves is
the inference itself, which is exactly what a data-residency reviewer will ask about
— so `global.` is the one that needs a sign-off, not `us.`.

Prices move; the [model
card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html)
carries the current per-token table for In-Region, Geo and Global side by side.
"""

RUNTIME_OPENAI_MD = """\
### The same Responses payload, on `bedrock-runtime`, without a token

Everything above in this notebook talked to `bedrock-mantle` with a bearer token
minted by `provide_token()`. That token is short-lived, so production code needs a
refresh path around it.

On `bedrock-runtime` the same Responses API is served at `/openai/v1/responses` and
authenticated with **SigV4** — the ordinary AWS credential chain, no minting and no
refresh. For a migration this is usually the cheaper half of the change: the request
body does not move at all, only the URL and how the request is signed.

There is no OpenAI SDK shortcut for this, because the SDK cannot SigV4-sign. Signing
by hand is the point of the cell below.
"""

RUNTIME_OPENAI_CODE = '''\
import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

RUNTIME_BASE = f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1"


def runtime_openai_post(path: str, body: dict) -> tuple[int, dict]:
    """POST an OpenAI-shaped payload to bedrock-runtime, signed with SigV4."""
    payload = json.dumps(body)
    signed = AWSRequest(
        method="POST",
        url=f"{RUNTIME_BASE}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    # Service name is "bedrock" even though the host is bedrock-runtime.
    SigV4Auth(
        boto3.Session(region_name=REGION).get_credentials().get_frozen_credentials(),
        "bedrock",
        REGION,
    ).add_auth(signed)
    reply = requests.post(
        f"{RUNTIME_BASE}{path}", headers=dict(signed.headers), data=payload, timeout=90
    )
    return reply.status_code, reply.json()


PROFILE = f"us.{MODEL}"

status, body = runtime_openai_post(
    "/responses",
    {"model": PROFILE, "input": "In one sentence: what is idempotency?", "max_output_tokens": 60},
)
print(f"POST /openai/v1/responses -> {status}")
print("  ", response_text(body) if status == 200 else err(body))

# Chat Completions is served here too, for code that has not moved to Responses.
status, body = runtime_openai_post(
    "/chat/completions",
    {
        "model": PROFILE,
        "messages": [{"role": "user", "content": "Reply with one word: ready?"}],
        "max_completion_tokens": 24,
    },
)
print(f"\\nPOST /openai/v1/chat/completions -> {status}")
print("  ", body["choices"][0]["message"]["content"] if status == 200 else err(body))
'''

FEATURES_MD = """\
### What you give up, and what you gain

Moving to `bedrock-runtime` is not free. Per the model card, on this endpoint:

| | `bedrock-runtime` | `bedrock-mantle` |
|---|---|---|
| Server-side tools (web search, code interpreter) | **no** | yes |
| Structured outputs | **no** | yes |
| Prompt caching | Responses API only | yes |
| Guardrails | Converse API only | — |
| Invocation logs · CloudWatch metrics · Cost Explorer line items | **yes** | limited |
| Cross-Region inference | **required** (geo or global) | not supported |

So the trade is roughly: `bedrock-mantle` for the richest OpenAI feature surface,
`bedrock-runtime` for AWS-native governance — per-model cost attribution, invocation
logging, guardrails on Converse — and for cross-Region throughput. Feature matrices
move faster than notebooks, so treat the model card as authoritative and re-check
before you commit to one.

One thing Converse gives you that neither OpenAI API does: a single request shape
across every provider on Bedrock. An agent loop written for Claude or Nova runs
against this model unchanged.
"""

CONVERSE_MD = """\
### Tool use and the reasoning trace on Converse

Tool use works and comes back as an ordinary `toolUse` block. The reasoning trace
does **not**: `content` carries a `text` block and nothing else, so the thinking is
billed and unreadable — the same trade the `/v1` families make on Chat Completions.
"""


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    path = f"{root.rstrip('/')}/{NB}"
    nb = nbformat.read(path, as_version=4)

    start = next(
        i for i, c in enumerate(nb.cells)
        if c.cell_type == "markdown" and c.source.startswith("## Also on")
    )
    # The section added in 04c128e runs to the end of the notebook: 6 cells.
    assert len(nb.cells) - start == 6, f"expected 6 trailing cells, found {len(nb.cells) - start}"
    availability_code = nb.cells[start + 1].source   # reuse verbatim, it is correct
    bare_id_code = nb.cells[start + 3].source        # reuse verbatim, it is correct
    converse_code = nb.cells[start + 5].source       # reuse verbatim, it is correct

    rebuilt = [
        nbformat.v4.new_markdown_cell(INTRO),
        nbformat.v4.new_code_cell(availability_code),
        nbformat.v4.new_markdown_cell(BARE_ID_MD),
        nbformat.v4.new_code_cell(bare_id_code),
        nbformat.v4.new_markdown_cell(GEO_GLOBAL_MD),
        nbformat.v4.new_markdown_cell(RUNTIME_OPENAI_MD),
        nbformat.v4.new_code_cell(RUNTIME_OPENAI_CODE),
        nbformat.v4.new_markdown_cell(CONVERSE_MD),
        nbformat.v4.new_code_cell(converse_code),
        nbformat.v4.new_markdown_cell(FEATURES_MD),
    ]
    for cell in rebuilt:
        cell.pop("id", None)

    nb.cells[start:] = rebuilt
    nbformat.write(nb, path)
    print(f"rebuilt the runtime section: {len(rebuilt)} cells from cell {start}")


if __name__ == "__main__":
    main()
