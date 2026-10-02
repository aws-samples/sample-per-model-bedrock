#!/usr/bin/env python3
"""Fix a false conclusion in 01-openai-gpt/01: "gpt-5.6 is Responses-only".

The probe in section 2 sent `max_tokens` to Chat Completions for every model. The
GPT-5.6 family rejects that parameter outright:

    Unsupported parameter: 'max_tokens' is not supported with this model.

so gpt-5.6 came back 400 and the notebook concluded Chat Completions was
unsupported. It is not. With `max_completion_tokens` the same call returns 200.
Verified for sol, terra and luna; gpt-5.5, gpt-5.4 and gpt-oss accept either name.

The probe conflated two different things — "does this API work" and "does this
parameter work" — and the Gotchas table then hardened the wrong answer into
"Responses-only". Anyone with existing Chat Completions code would have been told
to rewrite it for no reason.

This rewrites the probe to vary the parameter as well as the API, so the output
separates the two, and corrects the table. It is the same failure mode as the
12 August Gemma 4 change, which is worth saying out loud in a collection whose
whole premise is per-model parameter differences.
"""

import sys

import nbformat

NB = "01-openai-gpt/01-responses-api-core.ipynb"

MATRIX_MD = """\
## 2b. Which API, and which output-cap parameter

Two questions that look like one. A 400 from Chat Completions can mean *this model
does not serve Chat Completions* or *this model does not accept the parameter you
sent* — and the fix is completely different.

Every OpenAI model here serves **both** Responses and Chat Completions. What differs
is the name of the output cap:

| | Responses | Chat Completions |
|---|---|---|
| gpt-5.6 (`sol`, `terra`, `luna`) | `max_output_tokens` only | `max_completion_tokens` only |
| gpt-5.5, gpt-5.4, gpt-oss | either | either |

The GPT-5.6 family rejects `max_tokens` with *"Unsupported parameter: 'max_tokens' is
not supported with this model"*. Send the wrong one and you get a 400 that looks like
the API is missing.

This is the same shape of trap as the 12 August 2026 Gemma 4 change, where
`max_tokens` stopped being accepted on Chat Completions. When a request that worked
yesterday starts returning 400 `unsupported_parameter`, suspect the parameter before
you suspect the endpoint.
"""

MATRIX_CODE = '''\
# Vary the API *and* the parameter name, so a 400 tells you which one is at fault.
CAPS = {
    "/responses": ("max_output_tokens", "max_tokens"),
    "/chat/completions": ("max_completion_tokens", "max_tokens"),
}

print(f"{'model':24} {'API':18} {'documented':>11} {'max_tokens':>11}")
print("-" * 68)
for model in (SOL, TERRA, LUNA, GPT55, GPT54, OSS120):
    prefix = prefix_for(model)
    for path, (documented, legacy) in CAPS.items():
        codes = []
        for field in (documented, legacy):
            body = {"model": model, field: 16}
            if path == "/responses":
                body["input"] = "Reply OK"
            else:
                body["messages"] = [{"role": "user", "content": "Reply OK"}]
            code, _ = post(f"{prefix}{path}", body, region=REGION)
            codes.append(code)
        print(f"{model:24} {path:18} {codes[0]:>11} {codes[1]:>11}")

print("\\n200 in the 'documented' column means the API is available.")
print("400 in the 'max_tokens' column is a parameter problem, not a missing API.")
'''

GOTCHAS_OLD = "| gpt-5.6 | **Responses-only** — Chat Completions returns 400 |"
GOTCHAS_NEW = (
    "| gpt-5.6 output cap | `max_output_tokens` on Responses, `max_completion_tokens` "
    "on Chat Completions. **`max_tokens` is rejected outright** — a 400 that looks "
    "like a missing API (§2b) |"
)

REGION_OLD = (
    "| Region | `gpt-5.5` / `gpt-5.6-sol` absent from us-west-2; none of gpt-5.x in "
    "eu-central-1 |"
)
REGION_NEW = (
    "| Region | Availability differs per model and per endpoint, and moves. §10 probes "
    "it; the model card's regional table is authoritative |"
)


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    path = f"{root.rstrip('/')}/{NB}"
    nb = nbformat.read(path, as_version=4)

    # Replace the flawed Responses-vs-ChatCompletions probe (the cell after the
    # "path split" pair) and the markdown that introduces it, if present.
    probe = next(
        i for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and "'ChatCompletions':>17" in c.source
    )
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    replacement = [md(MATRIX_MD), code(MATRIX_CODE)]
    for cell in replacement:
        cell.pop("id", None)
    nb.cells[probe : probe + 1] = replacement
    print(f"replaced the flawed probe at cell {probe} with a markdown+code pair")

    fixed = 0
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        if GOTCHAS_OLD in cell.source:
            cell.source = cell.source.replace(GOTCHAS_OLD, GOTCHAS_NEW, 1)
            fixed += 1
        if REGION_OLD in cell.source:
            cell.source = cell.source.replace(REGION_OLD, REGION_NEW, 1)
            fixed += 1
    assert fixed == 2, f"expected to fix 2 gotcha rows, fixed {fixed}"
    print("corrected the Responses-only row and the volatile Region row")

    nbformat.write(nb, path)
    print(f"{NB}: {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
