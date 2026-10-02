#!/usr/bin/env python3
"""Teach the three new bedrock-runtime API surfaces in 00-foundations/04.

This notebook was "runtime = Converse + InvokeModel". Runtime now also serves
Chat Completions, Responses and Messages, so the notebook needs those, and one
comment in it ("there is no server-side conversation store on this endpoint") is
now false.
"""
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbedit import Notebook  # noqa: E402
import os
REPO_ROOT = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

NB = (REPO_ROOT + "/"
      "00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb")
nb = Notebook(NB)

# ---------------------------------------------------------------------------
# 1. Header: runtime is no longer "the Converse endpoint".
# ---------------------------------------------------------------------------
nb.sub(
    """Notebook 01 covered `bedrock-mantle`: bearer tokens, the three URL path families,
and the OpenAI- and Anthropic-shaped APIs. This one covers the other endpoint.

`bedrock-runtime` is the AWS-native surface. Three things make it different, and
all three trip people up:

1. **Auth is SigV4 through the AWS SDK.** No token to mint, no expiry to manage.
2. **Many models cannot be called by their model ID.** They require a
   cross-Region *inference profile* and reject the bare ID outright.
3. **The response is a list of typed blocks**, not a string. Indexing `[0]` and
   reading `text` works right up until the model returns a reasoning block first.

Read this before the family notebooks if you have not used Converse before.""",
    """Notebook 01 covered both endpoints at the level of auth, URL paths and model
discovery. This one goes deep on `bedrock-runtime` — the endpoint AWS recommends
for new applications.

It is two things at once, and that is the thing to understand:

- **The AWS-native surface**: `Converse` and `InvokeModel`, through boto3, signed
  with SigV4. Model-agnostic, no token to mint.
- **Since August 2026, an OpenAI- and Anthropic-compatible surface too**: Chat
  Completions, Responses and Messages, on the `/openai/v1` and `/anthropic/v1`
  paths. Called over HTTPS rather than through boto3, with SigV4 *or* a Bedrock
  API key.

Four things trip people up:

1. **Auth differs by surface.** Converse: SigV4 via the SDK, nothing to manage.
   The OpenAI-compatible paths: either, and the OpenAI SDK needs a bearer token.
2. **Many models cannot be called by their model ID.** They require a
   cross-Region *inference profile* and reject the bare ID outright.
3. **The Converse response is a list of typed blocks**, not a string. Indexing
   `[0]` and reading `text` works right up until the model returns a reasoning
   block first.
4. **Runtime's model IDs are not mantle's.** `openai.gpt-oss-20b` is mantle's; on
   runtime the same model is `openai.gpt-oss-20b-1:0`.

Read this before the family notebooks if you have not used Converse before.""",
)

# ---------------------------------------------------------------------------
# 2. The false comment about server-side state.
# ---------------------------------------------------------------------------
nb.sub(
    """        # A prior assistant turn goes here too, same block shape - that is how
        # you carry multi-turn state. There is no server-side conversation store
        # on this endpoint; you resend the history each time.""",
    """        # A prior assistant turn goes here too, same block shape - that is how
        # you carry multi-turn state on Converse: you resend the history each
        # time. Note that this is a property of CONVERSE, not of the endpoint --
        # the Responses API on this same endpoint does keep server-side state
        # (§8 below).""",
)

# ---------------------------------------------------------------------------
# 3. Rewrite §7 (choosing an endpoint) to match AWS's recommendation.
# ---------------------------------------------------------------------------
idx = nb.find("## 7. Choosing an endpoint")
assert len(idx) == 1, idx
nb.set_source(
    idx[0],
    """## 7. Choosing an endpoint

AWS's guidance is now explicit: *"For new applications, we recommend the
`bedrock-runtime` endpoint."* So the question is no longer "which endpoint" but
"do I have a reason to add `bedrock-mantle`". Ask in this order:

1. **Is the model on `bedrock-runtime` at all?** Gemma 4, GPT-5.4, GPT-5.5, Grok
   4.3, DeepSeek v3.1 and GLM 4.6 are mantle-only today. That decides it. Do not
   trust that list — `endpoints_for()` reads both catalogues live, and §7b prints
   the current split.
2. **Do you need something only `bedrock-mantle` has?** Server-side or
   pre-configured tool use including web search; asynchronous inference with
   `background=true`; Projects or Workspaces for per-application cost attribution.
   Those are mantle-only, and they are the honest reasons to use it.
3. **Do you need something only `bedrock-runtime` has?** Guardrails, intelligent
   prompt routing, cross-Region inference, Provisioned Throughput, batch
   inference, `InvokeModel` for non-text modalities.
4. **Neither?** Use `bedrock-runtime`. That is the recommendation, and it is where
   the account-level controls you already use — invocation logging, CloudWatch
   metrics, Cost Explorer attribution — apply to the OpenAI-shaped calls too.

Pricing does not enter into it: per-token pricing for the same model is identical
on both endpoints. And both can be used from one application — choose per use
case.

Then, having chosen the endpoint, choose the API:

| You want | Use |
|---|---|
| one interface across every model | **Converse** |
| raw provider JSON, or a non-text modality | **InvokeModel** |
| to move existing OpenAI code with a base-URL change | **Chat Completions** or **Responses** |
| server-side conversation state, reasoning items, tool loops | **Responses** |
| to move existing Anthropic code | **Messages** |""",
)

nb.save()
print(f"header/§7 updated: {nb.changes} edits")

# ---------------------------------------------------------------------------
# 4. New sections 7b and 8: the endpoint split, and the OpenAI-compatible
#    surface on runtime. Inserted before the Takeaways.
# ---------------------------------------------------------------------------
nb = Notebook(NB)
anchor = nb.find("## 7. Choosing an endpoint")[0]

MD_7B = """### 7b. The split, printed live

Two catalogues, two naming conventions. This cell reconciles them so you can see
which models are actually mantle-only rather than merely *named differently*.

That distinction matters: comparing the two catalogues on exact model IDs makes
`openai.gpt-oss-120b` look mantle-only, when its runtime twin is
`openai.gpt-oss-120b-1:0`. `runtime_id_for()` normalises the four ways the names
differ — version suffix, `-v1:0` suffix, provider prefix (`moonshotai.` vs
`moonshot.`), and a trailing `-instruct`."""

CODE_7B = '''from bedrock import list_models, runtime_id_for

mantle_ids = sorted(list_models(REGION))
mapped = {m: runtime_id_for(m, REGION) for m in mantle_ids}

mantle_only = [m for m, r in mapped.items() if r is None]
renamed = {m: r for m, r in mapped.items() if r and r != m}
same = [m for m, r in mapped.items() if r == m]

print(f"{len(mantle_ids)} models on bedrock-mantle in {REGION}")
print(f"  {len(same):>3} reachable on runtime under the SAME id")
print(f"  {len(renamed):>3} reachable under a DIFFERENT id")
print(f"  {len(mantle_only):>3} not on runtime at all")

print("\\nnot on bedrock-runtime -- these are the real reasons to use mantle:")
for m in mantle_only:
    print(f"    {m}")

print("\\nrenamed (a sample; this is the class of bug that wastes an afternoon):")
for m, r in list(renamed.items())[:8]:
    print(f"    {m:36} -> {r}")
if len(renamed) > 8:
    print(f"    ... and {len(renamed) - 8} more")

# And the reverse direction: runtime carries families mantle never had.
runtime_only_families = sorted(
    {k.split(".")[0] for k in runtime_models(REGION)}
    - {m.split(".")[0] for m in mantle_ids}
)
print(f"\\nfamilies only on runtime: {runtime_only_families}")'''

MD_8 = """## 8. The OpenAI- and Anthropic-compatible APIs, on `bedrock-runtime`

This is what changed in August 2026, and it is the reason the recommendation
moved. The same endpoint that serves Converse also serves:

| Path | API |
|---|---|
| `/openai/v1/chat/completions` | OpenAI Chat Completions |
| `/openai/v1/responses` | OpenAI Responses |
| `/anthropic/v1/messages` | Anthropic Messages |

Three practical notes before the code:

- **These paths are not in boto3.** You call them over HTTPS. `runtime_post()` in
  `_shared/bedrock.py` does it with a bearer token; SigV4 works too.
- **There is no `/v1` inference surface here.** Asking for one returns HTTP **200**
  with a Coral `UnknownOperationException` — `../00-foundations/01` §2b. Use
  `ok(status, body)` rather than `status == 200`.
- **Coverage is uneven, and narrower than Converse.** The cell measures it."""

CODE_8 = '''from bedrock import ok, runtime_post

CANDIDATES = [
    "openai.gpt-oss-120b",
    "openai.gpt-5.6-sol",
    "xai.grok-4.6",
    "anthropic.claude-opus-5",
    "qwen.qwen3-32b",
    "zai.glm-5",
    "amazon.nova-lite-v1",
    "meta.llama4-maverick-17b-instruct-v1",
]
AV = {"anthropic-version": "2023-06-01"}


def surface_check(model_id: str) -> dict:
    """Chat / Responses / Messages / Converse for one model on bedrock-runtime."""
    rid = runtime_id_for(model_id, REGION)
    row = {"asked": model_id, "runtime_id": rid or "-"}
    if rid is None:
        return {**row, "chat": "n/a", "responses": "n/a", "messages": "n/a",
                "converse": "n/a"}

    is_claude = "anthropic." in rid

    if is_claude:
        row["chat"] = row["responses"] = "-"
        code_, data = runtime_post(
            "/anthropic/v1/messages",
            {"model": rid, "max_tokens": 16,
             "messages": [{"role": "user", "content": "Hi"}]},
            region=REGION, headers=AV, attempts=1, timeout=90)
        row["messages"] = "ok" if ok(code_, data) else str(code_)
    else:
        row["messages"] = "-"
        code_, data = runtime_post(
            "/openai/v1/responses",
            {"model": rid, "input": "Hi", "max_output_tokens": 2000},
            region=REGION, attempts=1, timeout=120)
        row["responses"] = "ok" if ok(code_, data) else str(code_)
        # gpt-5.x and Grok want max_completion_tokens; the rest take max_tokens.
        for field in ("max_completion_tokens", "max_tokens"):
            code_, data = runtime_post(
                "/openai/v1/chat/completions",
                {"model": rid, "messages": [{"role": "user", "content": "Hi"}],
                 field: 2000},
                region=REGION, attempts=1, timeout=120)
            if ok(code_, data):
                break
        row["chat"] = "ok" if ok(code_, data) else str(code_)

    _, response = converse(
        rid, [{"role": "user", "content": [{"text": "Hi"}]}],
        max_tokens=512, region=REGION, resolve=False)
    row["converse"] = "err" if response.get("error") else "ok"
    return row

print(f"{'asked for':38} {'runtime id':32} {'Chat':>6} {'Resp':>6} {'Msg':>6} {'Conv':>6}")
print("-" * 100)
rows = [surface_check(m) for m in CANDIDATES]
for r in rows:
    print(f"{r['asked']:38} {r['runtime_id']:32} {r['chat']:>6} "
          f"{r['responses']:>6} {r['messages']:>6} {r['converse']:>6}")

counts = {
    key: sum(1 for r in rows if r[key] == "ok")
    for key in ("chat", "responses", "messages", "converse")
}
print(f"\\n=> of {len(rows)} models: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
print("   Converse is the widest surface on this endpoint by a long way. The")
print("   OpenAI-compatible paths are for moving existing code, not for reach.")'''

MD_8B = """### 8b. Server-side conversation state

The Responses API on `bedrock-runtime` defaults to `store=true` and accepts
`previous_response_id`, so the service holds the history. This is the capability an
earlier version of this notebook said the endpoint did not have.

One caveat from the AWS docs that affects design: *"A stored response belongs to
the AWS Region that served it."* Retrieving, cancelling, deleting, or continuing
with `previous_response_id` all go to that Region. A `global.` profile does not
tell you which Region that was — so use the `us.` geo profile if you need state
pinned to a known geography."""

CODE_8B = '''STATE_MODEL = runtime_id_for("openai.gpt-5.6-sol", REGION)
print("using", STATE_MODEL)

code_, first = runtime_post(
    "/openai/v1/responses",
    {"model": STATE_MODEL, "input": "My badge number is 8812.", "store": True,
     "max_output_tokens": 2000},
    region=REGION, attempts=1, timeout=180)

if not ok(code_, first):
    print(f"turn 1 -> {code_}: {json.dumps(first)[:150]}")
else:
    print("store echoed:", first.get("store"))
    for label, extra in (
        ("with previous_response_id", {"previous_response_id": first["id"]}),
        ("without it (control)", {}),
    ):
        code_, reply = runtime_post(
            "/openai/v1/responses",
            {"model": STATE_MODEL, "input": "What is my badge number? Digits only.",
             "max_output_tokens": 2000, **extra},
            region=REGION, attempts=1, timeout=180)
        if ok(code_, reply):
            answer = "".join(
                p.get("text", "")
                for item in (reply.get("output") or [])
                for p in (item.get("content") or [])
                if isinstance(p, dict)
            ).strip()
            print(f"  {label:26} -> {answer[:70]!r}")
        else:
            print(f"  {label:26} -> {code_}")

    print("\\n=> The control is the point: without the link the model cannot know the")
    print("   number, so the linked answer is evidence of server-side state rather")
    print("   than of a lucky guess. Always pair a state demo with its control.")

# background=true is the one Responses feature this endpoint does not have.
code_, data = runtime_post(
    "/openai/v1/responses",
    {"model": STATE_MODEL, "input": "Hi", "background": True,
     "max_output_tokens": 2000},
    region=REGION, attempts=1, timeout=120)
print(f"\\nbackground=true on runtime -> {code_}: "
      f"{(data.get('error') or {}).get('message', '')[:70]}")
print("=> Asynchronous inference stays on bedrock-mantle.")'''

i = nb.insert_after(anchor, "markdown", MD_7B)
i = nb.insert_after(i, "code", CODE_7B)
i = nb.insert_after(i, "markdown", MD_8)
i = nb.insert_after(i, "code", CODE_8)
i = nb.insert_after(i, "markdown", MD_8B)
nb.insert_after(i, "code", CODE_8B)

nb.save()
print(f"§7b/§8/§8b inserted: {nb.changes} edits")

# ---------------------------------------------------------------------------
# 5. Takeaways.
# ---------------------------------------------------------------------------
nb = Notebook(NB)
nb.sub(
    """- Guardrails attach here, not on `bedrock-mantle`: `guardrailConfig` on Converse,
  `guardrailIdentifier` on `invoke_model`. See `../99-cross-cutting/03` §9b.""",
    """- Guardrails attach here, not on `bedrock-mantle`: `guardrailConfig` on Converse,
  `guardrailIdentifier` on `invoke_model`. **Which of the OpenAI-shaped surfaces
  honour a guardrail header differs between them**, and one accepts it and does
  nothing — `../99-cross-cutting/03` §9b measures each.
- This endpoint serves five APIs, not two. Converse has the widest model coverage;
  the OpenAI-compatible paths exist to let existing code move with a base-URL
  change.
- Runtime model IDs are not mantle model IDs. Resolve, do not reuse.
- On `bedrock-runtime`, a body containing `UnknownOperation` is a failure whatever
  the status says. `ok(status, body)` encodes that.""",
)
nb.save()
print(f"takeaways updated: {nb.changes} edits")
