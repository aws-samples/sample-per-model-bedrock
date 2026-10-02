#!/usr/bin/env python3
"""Replace the stale "GPT-5.x is mantle-only" section with the runtime + CRIS path.

GPT-5.6 (sol / terra / luna) appeared in the bedrock-runtime catalogue in August
2026 with both geo (us.) and global. inference profiles. The notebook's own probe
would now print runtime=True, but the heading and prose still said "no", and two
other notebooks repeated the claim in passing.

Verified live before writing (us-east-1):
  - all three 5.6 variants are in the runtime catalogue
  - us.* and global.* profiles exist for each
  - the BARE id is rejected: ValidationException "... with on-demand throughput
    isn't supported" -> profile-only, like most Claude models
  - toolConfig works and returns a clean toolUse block
  - no reasoningContent block comes back; content is text only

Run from anywhere: add-gpt-runtime-section.py <repo-root>
"""

import sys

import nbformat

NB = "01-openai-gpt/01-responses-api-core.ipynb"

INTRO = """\
## Also on `bedrock-runtime`? GPT-5.6 — yes, through an inference profile

This is the clearest example in the collection of why these notebooks probe instead
of asserting. Until August 2026 the GPT-5.x models were `bedrock-mantle` only, and
this section said so. The 5.6 family then appeared in the `bedrock-runtime`
catalogue with both geo (`us.`) and `global.` inference profiles, and the sentence
became false without anything in the notebook changing.

So read the cells, not the prose: earlier GPT-5.x models may still be mantle-only,
and the probe below asks the live catalogues.
"""

AVAILABILITY = '''\
from bedrock import endpoints_for, inference_profiles, runtime_models

MODEL = "openai.gpt-5.6-sol"
print(f"{MODEL} -> {endpoints_for(MODEL)}")

print("\\nopenai.* in the bedrock-runtime catalogue today:")
for model in sorted(m for m in runtime_models() if m.startswith("openai.")):
    print("   ", model)

print(f"\\nInference profiles that carry {MODEL}:")
for profile in sorted(p for p in inference_profiles() if p.endswith(MODEL)):
    print("   ", profile)
'''

BARE_ID_MD = """\
### The bare model ID is rejected, and the error misdirects

Converse will not take `openai.gpt-5.6-sol` as-is. It answers:

> `ValidationException` — Invocation of model ID ... with on-demand throughput
> isn't supported. Retry your request with the ID or ARN of an inference profile
> that contains this model.

That reads like a permissions or entitlement problem. It is not. It means *prefix
the ID with a Region scope* — `us.` for the geo profile, `global.` for the global
one. Most Claude models behave the same way, which is why
[`resolve_runtime_id()`](../_shared/bedrock.py) exists: it looks the model up in the
profile list and returns the form Converse will accept.

The cell below passes `resolve=False` to defeat that helper, so you can see the raw
failure and the fix side by side.
"""

BARE_ID_CODE = '''\
from bedrock import converse, resolve_runtime_id

QUESTION = [{"role": "user", "content": [{"text": "In one sentence: what is idempotency?"}]}]

print("resolve_runtime_id() picks:", resolve_runtime_id(MODEL), "\\n")

for model_id in (MODEL, f"us.{MODEL}", f"global.{MODEL}"):
    # resolve=False so the bare ID really is sent bare.
    text, response = converse(model_id, QUESTION, max_tokens=80, resolve=False)
    if response.get("error"):
        print(f"{model_id:28} {response['error']['code']}")
        print(f"{'':28} {response['error']['message'][:96]}...")
    else:
        print(f"{model_id:28} ok, {response['usage']['outputTokens']} output tokens")
        print(f"{'':28} {text}")
'''

TRADEOFF_MD = """\
### What Converse gives you here, and what it does not

Tool use works and comes back as an ordinary `toolUse` block, so an agent loop
written against Converse for Claude or Nova needs no reshaping for this model.

The reasoning trace does **not** come back. `content` carries a `text` block and
nothing else, so the thinking is billed and unreadable — the same trade the `/v1`
families make on Chat Completions. If you need the trace, or the server-side tools,
`background=true` and stored responses from earlier in this notebook, stay on
`bedrock-mantle`.
"""

TRADEOFF_CODE = '''\
from bedrock import converse_reasoning, converse_tool_uses

STOCK_TOOL = [
    {
        "toolSpec": {
            "name": "get_stock_price",
            "description": "Current share price for a ticker symbol.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                }
            },
        }
    }
]

# resolve defaults to True from here on, so the bare ID is fine to pass.
_, tooled = converse(
    MODEL,
    [{"role": "user", "content": [{"text": "What is AMZN trading at? Use the tool."}]}],
    max_tokens=300,
    tools=STOCK_TOOL,
)
print("stopReason:", tooled.get("stopReason"))
for call in converse_tool_uses(tooled):
    print("   tool:", call["name"], "input:", call["input"])

answer, reasoned = converse(
    MODEL,
    [{"role": "user", "content": [{"text": "A bat and ball cost $1.10. The bat costs $1.00 more than the ball. What does the ball cost?"}]}],
    max_tokens=400,
)
blocks = [key for block in reasoned["output"]["message"]["content"] for key in block]
print(f"\\ncontent blocks: {blocks}")
print(f"reasoning returned: {len(converse_reasoning(reasoned))} chars")
print("answer:", answer.strip()[:160])
'''


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    path = f"{root.rstrip('/')}/{NB}"
    nb = nbformat.read(path, as_version=4)

    idx = next(
        i for i, c in enumerate(nb.cells)
        if c.cell_type == "markdown" and c.source.startswith("## Also on")
    )
    # The stale pair is the heading and its single probe cell.
    assert nb.cells[idx + 1].cell_type == "code", "expected a code cell after the heading"
    old = nb.cells[idx : idx + 2]
    assert len(old) == 2, "expected exactly two cells to replace"

    replacement = [
        nbformat.v4.new_markdown_cell(INTRO),
        nbformat.v4.new_code_cell(AVAILABILITY),
        nbformat.v4.new_markdown_cell(BARE_ID_MD),
        nbformat.v4.new_code_cell(BARE_ID_CODE),
        nbformat.v4.new_markdown_cell(TRADEOFF_MD),
        nbformat.v4.new_code_cell(TRADEOFF_CODE),
    ]
    # nbformat stamps new cells with an id; the rest of this notebook has none,
    # so drop it to keep the file's shape consistent.
    for cell in replacement:
        cell.pop("id", None)

    nb.cells[idx : idx + 2] = replacement
    nbformat.write(nb, path)
    print(f"replaced cells {idx}-{idx + 1} with {len(replacement)} cells in {NB}")


if __name__ == "__main__":
    main()
