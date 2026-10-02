#!/usr/bin/env python3
"""Correct two false claims of mine, and add the documented behaviour differences.

Reading the dedicated doc page - "Using the Responses API on the bedrock-runtime
endpoint", which I had not read - showed my own section was wrong twice:

  1. I wrote "signed with SigV4 - no bearer token to mint" as though SigV4 were the
     only option. A Bedrock API key works as a bearer token on bedrock-runtime too.
     Verified: 200.
  2. I wrote "There is no OpenAI SDK shortcut for this, because the SDK cannot
     SigV4-sign." False. Point the OpenAI SDK at the runtime base URL with a Bedrock
     API key and it works unchanged. Verified: 200, response id present.

That second claim was the worst kind of error - it told readers to write SigV4
boilerplate they do not need for the simplest migration path.

Also adds the seven documented behaviour differences, all probed:

  background=true            runtime 400 "The background parameter is not supported",
                             mantle 200
  model on follow-ups        required on BOTH endpoints (400 without it on each).
                             The doc says mantle lets you omit it and inherit from
                             previous_response_id; it does not. Reported as measured.
  stored-response lifecycle  GET / DELETE /openai/v1/responses/{id} both 200
  unknown response id        404, identical for never-existed / other-account /
                             not-stored
  server-side tools          absent on runtime, including web search (doc + our
                             notebook 02 evidence)
  app inference profiles     rejected on runtime; system/geo/global work
  guardrails                 do not apply to Responses on runtime - use Converse

Plus the IAM subtlety: creating a response authorises bedrock:InvokeModel on the
inference target AND bedrock:InvokeModel on the account's default project. A policy
scoped only to the model ARN fails.
"""

import sys

import nbformat

NB = "01-openai-gpt/01-responses-api-core.ipynb"

AUTH_MD = """\
### Two ways to authenticate, and the easy migration path

`bedrock-runtime`'s `/openai/v1` accepts **either** credential style, and which you
pick changes how much code you write:

| | Bedrock API key as bearer token | SigV4 |
|---|---|---|
| Works with the OpenAI SDK | **yes**, unchanged | no — the SDK cannot sign |
| Code change from `bedrock-mantle` | **the base URL and the model ID** | swap auth for signing |
| Credential lifecycle | short-term key, needs refreshing | ordinary AWS credential chain, nothing to mint |
| Reach for it when | migrating existing OpenAI-SDK code | you already have AWS credentials in process |

So the shortest migration is genuinely two edits: point `base_url` at
`https://bedrock-runtime.{region}.amazonaws.com/openai/v1` and name the inference
profile instead of the bare model. SigV4 is the better long-term shape — no token to
mint, rotate or leak — but it is not required, and the cell below shows both.

Stored responses have a lifecycle here too, on the same base URL:
`GET /openai/v1/responses/{id}` retrieves one, `POST .../{id}/cancel` cancels an
in-flight one, and `DELETE .../{id}` removes it.

One IAM subtlety worth knowing before you write a policy. Creating a response
authorises **two** resources: `bedrock:InvokeModel` on the inference target, as any
inference call does, **and** `bedrock:InvokeModel` on your account's default project.
A policy scoped to the model ARN alone will fail. Retrieve, cancel and delete
authorise `bedrock:GetInvoke`, `bedrock:CancelInvoke` and `bedrock:DeleteInvoke` on
the project. Response IDs are not IAM resources.
"""

AUTH_CODE = '''\
from openai import OpenAI

from bedrock import token

# 1. The OpenAI SDK, pointed at bedrock-runtime with a Bedrock API key. No signing.
runtime_sdk = OpenAI(api_key=token(REGION), base_url=RUNTIME_BASE)
sdk_reply = runtime_sdk.responses.create(
    model=PROFILE, input="In one word: what does idempotent mean?", max_output_tokens=24
)
print(f"OpenAI SDK + bearer token  -> {response_text(sdk_reply.model_dump())[:60]}")

# 2. The same call signed with SigV4, for code that already holds AWS credentials.
signed = runtime_openai_post(
    "/responses", {"model": PROFILE, "input": "In one word: what does idempotent mean?", "max_output_tokens": 24}
)
print(f"SigV4                      -> {response_text(signed.json())[:60]}")

# 3. Stored-response lifecycle: create, retrieve, delete.
created = runtime_openai_post(
    "/responses", {"model": PROFILE, "input": "Remember the number 9.", "max_output_tokens": 24, "store": True}
).json()
fetched = runtime_sdk.responses.retrieve(created["id"])
deleted = runtime_sdk.responses.delete(created["id"])
print(f"\\nstore -> retrieve -> delete: retrieved status={fetched.status}, deleted={deleted.deleted}")
'''

BEHAVIOUR_MD = """\
### Where `bedrock-runtime` behaves differently, and one place the docs are wrong

Same request format, but not the same behaviour. These are the differences that break
working code, rather than the ones you would notice in a feature table:

- **`background=true` is rejected.** Runtime answers 400 *"The background parameter
  is not supported"*; `bedrock-mantle` accepts it. Asynchronous work stays on mantle.
  `store` is unaffected and still defaults to `true`, so stored multi-turn
  conversations are fine.
- **Server-side tools are absent**, web search included. Client-side function tools
  work on both.
- **Application inference profiles are rejected.** System, geographic and global
  profiles work; an application profile as the inference target returns 400.
- **Guardrails do not apply to the Responses API here.** To guardrail a GPT model on
  `bedrock-runtime`, call Converse instead.
- **A stored response belongs to the Region that served it.** Retrieve, cancel,
  delete and `previous_response_id` are all handled by that Region, and an ID that
  cannot be found returns the same 404 whether it never existed, belongs to another
  account, or was never stored.

**And one correction to the documentation.** The user guide says `model` may be
omitted on a follow-up that carries `previous_response_id` on `bedrock-mantle`, and
that requiring it is a `bedrock-runtime` difference. Both endpoints require it — the
cell below sends the same follow-up to each and both return 400. Treat `model` as
mandatory everywhere and the difference disappears.
"""

BEHAVIOUR_CODE = '''\
# background=true: accepted on mantle, rejected on runtime.
runtime_bg = runtime_openai_post(
    "/responses", {"model": PROFILE, "input": "Write a haiku.", "max_output_tokens": 40, "background": True}
)
mantle_bg, mantle_body = post(
    f"{GPT5_PREFIX}/responses",
    {"model": MODEL, "input": "Write a haiku.", "max_output_tokens": 40, "background": True},
)
print(f'background=true   runtime {runtime_bg.status_code}: '
      f'{runtime_bg.json().get("error", {}).get("message", "")[:56]}')
print(f'                  mantle  {mantle_bg}: {"accepted" if mantle_bg == 200 else err(mantle_body)[:56]}')

# `model` on a follow-up: the docs say mantle lets you omit it. Send it to both.
seed_runtime = runtime_openai_post(
    "/responses", {"model": PROFILE, "input": "Remember the number 9.", "max_output_tokens": 24, "store": True}
).json()
_, seed_mantle = post(
    f"{GPT5_PREFIX}/responses",
    {"model": MODEL, "input": "Remember the number 9.", "max_output_tokens": 24, "store": True},
)

no_model_runtime = runtime_openai_post(
    "/responses", {"input": "Which number?", "previous_response_id": seed_runtime["id"], "max_output_tokens": 24}
)
no_model_mantle, no_model_body = post(
    f"{GPT5_PREFIX}/responses",
    {"input": "Which number?", "previous_response_id": seed_mantle["id"], "max_output_tokens": 24},
)
print(f'\\nfollow-up, no model   runtime {no_model_runtime.status_code} | mantle {no_model_mantle}')
print(f'                      mantle says: {err(no_model_body)[:70]}')
print("Both reject it, so the documented difference does not exist. Always send `model`.")

# An ID that was never stored: 404, and deliberately indistinguishable from an ID
# that belongs to someone else. Retrieve raises, so catch it rather than crash.
try:
    runtime_sdk.responses.retrieve("resp_thisidwillneverexist000")
    print("\\nunknown response id   unexpectedly succeeded")
except Exception as exc:  # openai.NotFoundError
    print(f"\\nunknown response id   {type(exc).__name__}: {str(exc)[:66]}")
'''

FEATURES_MD = """\
### Choosing an endpoint for this model

| | `bedrock-runtime` | `bedrock-mantle` |
|---|---|---|
| APIs | Responses · Chat Completions · Converse | Responses · Chat Completions |
| Auth | SigV4 **or** Bedrock API key | Bedrock API key |
| OpenAI SDK works | **yes**, with the API key | yes |
| Cross-Region inference | **required** (geo or global) | not supported |
| Regional reach | wide | narrow |
| Streaming · prompt caching · server-side state · structured outputs | yes (caching is Responses-only) | yes |
| `background=true` | **no** — 400 | yes |
| Server-side tools (web search, code interpreter) | **no** | yes |
| Application inference profiles | **no** — 400 | n/a |
| Guardrails | Converse only | — |
| Invocation logs · CloudWatch metrics · Cost Explorer per-model lines | **yes** | limited |
| One request shape across providers | **yes**, via Converse | no |

The trade in one line: `bedrock-mantle` for server-side tools and background jobs,
`bedrock-runtime` for AWS-native governance, cross-Region throughput and Converse
portability. The public model card recommends `bedrock-runtime` for new applications.

Feature matrices move faster than notebooks, and this section found the card and the
docs wrong twice — structured outputs above, and the `model`-on-follow-up rule. Probe
before you commit.
"""


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    path = f"{root.rstrip('/')}/{NB}"
    nb = nbformat.read(path, as_version=4)

    def find(pred):
        return next(i for i, c in enumerate(nb.cells) if pred(c))

    # 1. Drop the false "the SDK cannot SigV4-sign, so sign by hand" framing.
    four = find(lambda c: c.cell_type == "code" and "RUNTIME_BASE = " in c.source)
    old_doc = ('    The OpenAI SDK cannot do this - it has no way to SigV4-sign - so the signing is\n'
               '    spelled out. Note the service name stays "bedrock" even though the host is\n'
               '    bedrock-runtime.\n')
    new_doc = ('    Note the service name stays "bedrock" even though the host is bedrock-runtime.\n'
               '    SigV4 is one of two options here; the next section shows the other.\n')
    assert old_doc in nb.cells[four].source, "SigV4 docstring not found as expected"
    nb.cells[four].source = nb.cells[four].source.replace(old_doc, new_doc, 1)

    intro = find(lambda c: c.cell_type == "markdown" and c.source.startswith("## Also on"))
    old_claim = ("  `https://bedrock-runtime.{region}.amazonaws.com/openai/v1` — signed with SigV4,\n"
                 "  with no bearer token to mint.")
    new_claim = ("  `https://bedrock-runtime.{region}.amazonaws.com/openai/v1`, with either SigV4 or\n"
                 "  a Bedrock API key — so the OpenAI SDK works there unchanged.")
    assert old_claim in nb.cells[intro].source, "intro claim not found as expected"
    nb.cells[intro].source = nb.cells[intro].source.replace(old_claim, new_claim, 1)

    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

    # 2. Auth pair goes straight after the four-ways cell.
    auth = [md(AUTH_MD), code(AUTH_CODE)]
    # 3. Behaviour pair goes before the Converse subsection.
    behaviour = [md(BEHAVIOUR_MD), code(BEHAVIOUR_CODE)]
    for cell in auth + behaviour:
        cell.pop("id", None)

    conv = find(lambda c: c.cell_type == "markdown" and c.source.startswith("### Converse: one shape"))
    nb.cells[conv:conv] = behaviour
    nb.cells[four + 1 : four + 1] = auth

    feat = find(lambda c: c.cell_type == "markdown" and c.source.startswith("### Choosing an endpoint"))
    nb.cells[feat].source = FEATURES_MD

    nbformat.write(nb, path)
    print(f"corrected the auth claims; added auth + behaviour sections; {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
