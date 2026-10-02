#!/usr/bin/env python3
"""Make the migration guide lead with the recommended endpoint, and link Grok 4.3->4.6."""
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbedit import Notebook  # noqa: E402
import os
REPO_ROOT = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

REPO = REPO_ROOT + "/"

# ===========================================================================
# 99-cross-cutting/02 — the migration guide
# ===========================================================================
NB = REPO + "99-cross-cutting/02-migrating-from-openai.ipynb"
nb = Notebook(NB)

nb.sub(
    """## 1. The two-line change

For an OpenAI codebase, `base_url` and `api_key` are the entire migration.""",
    """## 1. The two-line change

For an OpenAI codebase, `base_url` and `api_key` are the entire migration — and
since August 2026 you have a choice of destination. AWS's guidance is *"For new
applications, we recommend the `bedrock-runtime` endpoint"*, which now serves the
OpenAI Responses and Chat Completions APIs and the Anthropic Messages API alongside
Converse and InvokeModel.

Both endpoints work, and the cell below sends the same request to each so the
difference is visible rather than described. Two things differ:

- **the host**, and on `bedrock-runtime` the path is always `/openai/v1`, where on
  `bedrock-mantle` it is `/openai/v1` or `/v1` depending on the model family;
- **the model ID**. `bedrock-runtime` requires a cross-Region inference profile for
  several families — `us.openai.gpt-5.6-sol`, not `openai.gpt-5.6-sol` — and uses a
  different ID for others (`openai.gpt-oss-20b-1:0` rather than
  `openai.gpt-oss-20b`). Send the wrong one and you get *"The provided model
  identifier is invalid"*, which reads like a missing model.

Pick `bedrock-runtime` unless you need something only `bedrock-mantle` has:
server-side tool use including web search, asynchronous inference with
`background=true`, or Projects and Workspaces. §10 lists both sides.""",
)

nb.set_source(
    nb.find("# BEFORE (OpenAI):")[0],
    '''from aws_bedrock_token_generator import provide_token
from openai import OpenAI

from bedrock import ok, runtime_id_for

# BEFORE (OpenAI):
#   client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#
# AFTER — the same code against each endpoint. Only base_url and the model ID move.
QUESTION = "Confirm you are reachable in five words."
MODEL = "openai.gpt-5.6-sol"

destinations = [
    (
        "bedrock-runtime (recommended)",
        f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
        runtime_id_for(MODEL, REGION),
    ),
    (
        "bedrock-mantle",
        f"https://bedrock-mantle.{REGION}.api.aws/openai/v1",
        MODEL,
    ),
]

for label, base_url, model_id in destinations:
    if model_id is None:
        print(f"{label:30} {MODEL} is not on this endpoint")
        continue
    client = OpenAI(
        api_key=provide_token(region=REGION),  # short-term Bedrock key from IAM
        base_url=base_url,
        max_retries=5,
        timeout=180.0,
    )
    response = client.responses.create(
        model=model_id,
        input=QUESTION,
        max_output_tokens=2000,  # generous: reasoning spends output tokens first
    )
    print(f"{label:30} model={model_id}")
    print(f"{'':30} {response.output_text.strip()[:70]!r}")

# The mistake this section exists to prevent: mantle's ID on runtime.
wrong = OpenAI(
    api_key=provide_token(region=REGION),
    base_url=f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
    max_retries=0,
    timeout=60.0,
)
try:
    wrong.responses.create(model=MODEL, input="Hi", max_output_tokens=2000)
    print(f"\\nmantle id {MODEL!r} on bedrock-runtime: accepted")
except Exception as exc:  # noqa: BLE001 - the error IS the lesson here
    print(f"\\nmantle id {MODEL!r} on bedrock-runtime -> {type(exc).__name__}")
    print(f"  {str(exc)[-130:]}")
    print("  => 'invalid model identifier' means the wrong endpoint's ID, not a")
    print("     missing model. runtime_id_for() translates.")''',
)

nb.sub(
    """Environment-variable form, if you would rather not touch code at all:

```bash
export OPENAI_BASE_URL="https://bedrock-mantle.us-east-1.api.aws/openai/v1"
export OPENAI_API_KEY="$(
  python -c 'from aws_bedrock_token_generator import provide_token
print(provide_token(region="us-east-1"))'
)"
```""",
    """Environment-variable form, if you would rather not touch code at all:

```bash
# bedrock-runtime — the recommended endpoint. Model IDs need the us. profile
# prefix for the gpt-5.6, Grok 4.6 and Claude families.
export OPENAI_BASE_URL="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"

# or bedrock-mantle, if you need server-side tools, background=true, or Projects
# export OPENAI_BASE_URL="https://bedrock-mantle.us-east-1.api.aws/openai/v1"

export OPENAI_API_KEY="$(
  python -c 'from aws_bedrock_token_generator import provide_token
print(provide_token(region="us-east-1"))'
)"
```

The environment-variable route does **not** rewrite model IDs, so this is the form
where the per-endpoint ID difference bites hardest: the same `model=` string that
worked yesterday against mantle returns *"The provided model identifier is invalid"*
after you point `OPENAI_BASE_URL` at runtime.""",
)

nb.sub(
    """# The Anthropic SDK migrates the same way. Note the base URL omits /v1 —
# the SDK appends it.
import anthropic

claude = anthropic.Anthropic(
    api_key=provide_token(region=REGION),
    base_url=f"https://bedrock-mantle.{REGION}.api.aws/anthropic",
)
message = claude.messages.create(
    model="anthropic.claude-haiku-4-5",
    max_tokens=60,  # required on the Messages API
    messages=[{"role": "user", "content": "Confirm you are reachable in five words."}],
)
print("".join(b.text for b in message.content if b.type == "text"))""",
    """# The Anthropic SDK migrates the same way, and to either endpoint. Note the base
# URL omits /v1 -- the SDK appends it.
import anthropic

CLAUDE_MANTLE = "anthropic.claude-opus-5"
CLAUDE_RUNTIME = "us.anthropic.claude-opus-5"   # profile required on runtime

for label, base_url, model_id in (
    ("bedrock-runtime (recommended)",
     f"https://bedrock-runtime.{REGION}.amazonaws.com/anthropic", CLAUDE_RUNTIME),
    ("bedrock-mantle",
     f"https://bedrock-mantle.{REGION}.api.aws/anthropic", CLAUDE_MANTLE),
):
    claude = anthropic.Anthropic(api_key=provide_token(region=REGION),
                                 base_url=base_url, max_retries=5)
    try:
        message = claude.messages.create(
            model=model_id,
            max_tokens=200,  # required on the Messages API
            messages=[{"role": "user", "content": QUESTION}],
        )
        text = "".join(b.text for b in message.content if b.type == "text")
        print(f"{label:30} model={model_id}")
        print(f"{'':30} {text.strip()[:60]!r}  echoed model={message.model!r}")
    except Exception as exc:  # noqa: BLE001 - show it, do not stop the notebook
        print(f"{label:30} {type(exc).__name__}: {str(exc)[-90:]}")

print()
print("=> The Messages surface on bedrock-runtime is narrower than on mantle: it")
print("   serves the newest Claude models, addressed by a us. or global. profile.")
print("   Note it echoes back the SHORT model name, without the profile prefix.")""",
)

nb.save()
print(f"02 §1 updated: {nb.changes} edits")

# ---------------------------------------------------------------------------
nb = Notebook(NB)
if "from bedrock import" in nb.source(2) and "ok" not in nb.source(2):
    print("  (imports cell needs ok/runtime_post — checking)")
print("imports cell 2:")
print(nb.source(2))
