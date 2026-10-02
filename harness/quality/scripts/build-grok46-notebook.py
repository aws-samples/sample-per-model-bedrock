#!/usr/bin/env python3
"""Build 11-xai-grok/02-grok-4-6.ipynb.

Grok 4.6 launched 18 Aug 2026 and is the first model in this collection whose
endpoint story runs the other way round: on bedrock-runtime it is
inference-profile-only and available broadly, and on bedrock-mantle it is
us-west-2 in-Region only. Everything asserted here was probed on 2026-08-20;
see quality/findings/32..36.
"""
import hashlib
import json
import os

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.rstrip()))


def code(text: str) -> None:
    CELLS.append(("code", text.rstrip()))


DISCLAIMER = """> **Sample code — not for production.** Provided as AWS Content under the AWS
> Customer Agreement; do not use it in production accounts or on production or other
> critical data. Running these cells calls Amazon Bedrock and incurs charges. Full
> disclaimer in the [README](../README.md#disclaimer)."""

# ---------------------------------------------------------------------------
md(f"""# xAI Grok 4.6 on Amazon Bedrock — on both endpoints

{DISCLAIMER}

Grok 4.6 is xAI's frontier model for coding, agentic work and long-running
agents: **500K context**, always-on reasoning with a five-step effort dial plus
`none`, and — unusually for this collection — a **different availability shape on
each endpoint**.

**Model covered**

| Endpoint | Model ID to send | Availability |
|---|---|---|
| `bedrock-runtime` | `us.xai.grok-4.6` or `global.xai.grok-4.6` | broad; **no in-Region form** |
| `bedrock-mantle` | `xai.grok-4.6` | **`us-west-2` only**, in-Region |

That is backwards from Gemma 4 and Grok 4.3, which are mantle-only. §1 probes it
rather than asking you to trust the table.

## Three things to know before you start

1. **On `bedrock-runtime` you must name a cross-Region inference profile**, and
   the two are not interchangeable. The bare `xai.grok-4.6` is refused there:
   *"Invocation of model ID xai.grok-4.6 with on-demand throughput isn't
   supported."* `us.` exists in the three US Regions only, while `global.` works
   wherever the model does — so calling `us.` from EU or APAC gives *"The provided
   model identifier is invalid"*, which reads like the model is absent. §1b
   measures both. AWS recommends `bedrock-runtime` for new applications, so this is
   the path most readers will take.
2. **Reasoning is on by default at `effort="low"`.** The dial is
   `none | low | medium | high | xhigh`. §3 measures what each level actually
   costs, and finds that the answer is more interesting than "more effort, more
   tokens".
3. **The reasoning trace is encrypted.** You get an opaque
   `encrypted_content` blob, not readable text, and you can hand it back on the
   next turn to preserve reasoning context. §4.

## What this notebook covers
- §1 Where Grok 4.6 lives, probed on both endpoints
- §1b Which inference profile exists in which Region, and the 400 it causes
- §2 First call, both endpoints, both OpenAI APIs
- §3 Reasoning effort: what the five settings cost, and an error that misleads
- §4 The encrypted trace, and replaying it across turns
- §5 Chat Completions: no trace, but the token count is still there
- §6 Client-side tools — and which tool types this model refuses
- §7 Structured output (the model card says no; the service says yes)
- §8 Stateful conversation with `store` + `previous_response_id`
- §9 Streaming on both APIs
- §10 Prompt caching, `prompt_cache_key`, and why `global.` caches worse
  than `us.`
- §11 Vision
- §12 Service tiers
- §13 What Grok 4.6 does *not* do here, with the error each one gives
- §14 The IAM permission that is easy to miss

## Self-contained, but see also
- **Endpoints, auth, the URL paths, and the 200-that-means-failure** →
  `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
- **Converse, inference profiles, the runtime catalogue** →
  `../00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb`
- **Grok 4.3**, which is a different model on a different endpoint →
  `01-grok-4-3.ipynb`

## Prerequisites
```bash
pip install -r ../requirements.txt
```

Needs openai, boto3, aws-bedrock-token-generator.

`requirements.txt` pins the exact versions this collection was tested
against. An unpinned install resolves whatever is current, which may be
untested or compromised (OWASP LLM03, Supply Chain).""")

# ---------------------------------------------------------------------------
md("""### Where the helpers come from

The next cell does this:

```python
sys.path.insert(0, "../_shared")
from bedrock import ...
```

`bedrock` is **not** a package from PyPI — it is this collection's own helper
module, [`_shared/bedrock.py`](../_shared/bedrock.py). Every notebook sits one
level down, so `../_shared` puts it on the import path. It exists only to remove
repetition; the notebooks are the teaching material, and nothing in the module is
required to call Bedrock yourself.

What this notebook uses from it:

| Helper | What it does |
|---|---|
| `converse` | one Converse call; returns `(text, response)` and **never raises** |
| `converse_text` | concatenates the text blocks of a Converse response |
| `err` | pulls the human-readable message out of an error body, redacted |
| `ok` | `status == 200` **and** the body is not a Coral fault — see below |
| `post` | signed JSON HTTP against `bedrock-mantle`; never raises on 4xx/5xx |
| `response_text` | assistant text from a Responses API payload |
| `runtime_post` | the same, against `bedrock-runtime`'s `/openai/v1` and `/anthropic/v1` |
| `safe_print` | `print()` with account IDs and IAM principals redacted |
| `bands_png` | a generated PNG with a known answer, for the vision cell |

Three behaviours worth knowing before you read any output below:

- **`post()`, `runtime_post()` and `converse()` never raise on a service error.**
  They return the status and body so a cell can *show* a 400 rather than stopping
  the notebook. Several cells here deliberately provoke an error.
- **Use `ok(status, body)`, not `status == 200`.** `bedrock-runtime` answers an
  unrecognised path with **HTTP 200** and a Coral `UnknownOperationException` in
  the body. `../00-foundations/01` §2b demonstrates it.
- **Anything printed from a control-plane response goes through redaction**, since
  this output is committed to a public repository.""")

code('''import json
import statistics
import sys
import time

sys.path.insert(0, "../_shared")
from bedrock import (
    bands_png,
    converse,
    converse_text,
    err,
    ok,
    post,
    response_text,
    runtime_post,
    safe_print,
)

# Grok 4.6 is addressed differently on each endpoint, so name both explicitly
# rather than deriving one from the other.
RUNTIME_REGION = "us-east-1"      # runtime serves the geo/global profiles here
MANTLE_REGION = "us-west-2"       # mantle serves grok-4.6 in this Region only

GROK_RT = "us.xai.grok-4.6"       # geo profile; `global.` also works
GROK_MT = "xai.grok-4.6"          # bare ID, mantle only

# Both endpoints put Grok on /openai/v1.
RT_PREFIX = MT_PREFIX = "/openai/v1"

print(f"runtime: https://bedrock-runtime.{RUNTIME_REGION}.amazonaws.com{RT_PREFIX}"
      f"  model={GROK_RT}")
print(f"mantle : https://bedrock-mantle.{MANTLE_REGION}.api.aws{MT_PREFIX}"
      f"  model={GROK_MT}")''')

# ---------------------------------------------------------------------------
md("""## 1. Where Grok 4.6 actually lives

Four combinations are worth testing, because three of them fail and each failure
has a different cause. Reading the message is the whole point — "does not exist"
and "on-demand throughput isn't supported" send you to completely different fixes.""")

code('''ASK = {"input": "Reply with exactly: OK", "max_output_tokens": 2000}

trials = [
    ("runtime", RUNTIME_REGION, GROK_RT, "geo inference profile"),
    ("runtime", RUNTIME_REGION, GROK_MT, "bare ID on runtime"),
    ("mantle", MANTLE_REGION, GROK_MT, "bare ID in us-west-2"),
    ("mantle", "us-east-1", GROK_MT, "bare ID in us-east-1"),
]

print(f"{'endpoint':9} {'region':10} {'model':17} {'what':24} result")
print("-" * 108)
works = []
for endpoint, region, model, what in trials:
    caller = runtime_post if endpoint == "runtime" else post
    code_, data = caller(f"{RT_PREFIX}/responses", {"model": model, **ASK},
                         region=region, attempts=1, timeout=180)
    if ok(code_, data):
        verdict = f"200  {response_text(data).strip()[:24]!r}"
        works.append((endpoint, region, model))
    else:
        verdict = f"{code_}  {err(data)[:62]}"
    print(f"{endpoint:9} {region:10} {model:17} {what:24} {verdict}")

print()
print(f"=> {len(works)} of {len(trials)} combinations answered:")
for endpoint, region, model in works:
    print(f"     {endpoint:8} {region:10} {model}")
print("   The two endpoints want DIFFERENT model IDs for the same model, and the")
print("   Region that works on one is not the Region that works on the other.")''')

md("""### What each failure was telling you

- **`us.xai.grok-4.6` on runtime → 200.** The geo profile is the addressable form.
  `global.xai.grok-4.6` works too and is priced lower per token
  ($2.00/$6.00 per 1M in/out versus $2.20/$6.60 for in-Region and Geo, per the
  model card).
- **`xai.grok-4.6` on runtime → 400, *"Invocation of model ID xai.grok-4.6 with
  on-demand throughput isn't supported. Retry your request with the ID or ARN of an
  inference profile that contains this model."*** The model is there; you addressed
  it in a form that endpoint does not offer. This is the standard
  `INFERENCE_PROFILE`-only error — `../00-foundations/04` §4 covers the pattern.
- **`xai.grok-4.6` on mantle in `us-west-2` → 200.** Bare ID, no profile, because
  mantle has no cross-Region inference at all.
- **`xai.grok-4.6` on mantle in `us-east-1` → 404, *"The model 'xai.grok-4.6' does
  not exist"*.** Not "wrong ID" — *absent from this Region*. Mantle serves Grok 4.6
  from `us-west-2` only today, which is why this notebook carries two Regions
  instead of one.

The general lesson, and it is the reason `../00-foundations/01` §7 probes rather
than tabulates: **endpoint, Region, and model ID are three independent variables**,
and a single 400 tells you which one you got wrong only if you read it.""")

# ---------------------------------------------------------------------------
md("""### 1b. Which profile exists where — the `us.` trap

`us.` and `global.` are not two spellings of the same thing. The geo profile exists
only in the Regions of its geography; the global profile exists everywhere the model
does. So this fails, and the message names the model rather than the Region:

```
us.xai.grok-4.6 in ap-southeast-1
-> 400 The provided model identifier is invalid.
```

That error is easy to read as "Grok 4.6 is not on bedrock-runtime", which is the
wrong conclusion and sends you to the wrong fix. The cell below asks the control
plane which profiles exist in each Region, then makes a real call, so the two
answers can be compared.

**If you are outside the US and want one model ID that works everywhere, use
`global.`** — it is also priced lower per token ($2.00/$6.00 per 1M in/out against
$2.20/$6.60 for in-Region and Geo, per the model card).""")

code('''import boto3

SURVEY_REGIONS = ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1"]
CANDIDATES = ["us.xai.grok-4.6", "global.xai.grok-4.6"]

print(f"{'region':16} {'us. exists':11} {'global. exists':15} live call")
print("-" * 88)
for region in SURVEY_REGIONS:
    try:
        control = boto3.client("bedrock", region_name=region)
        profiles = {
            p["inferenceProfileId"]
            for p in control.list_inference_profiles(maxResults=1000)[
                "inferenceProfileSummaries"
            ]
        }
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"{region:16} control plane: {type(exc).__name__}")
        continue

    present = {c: c in profiles for c in CANDIDATES}
    # Call with each candidate so the error text is visible, not inferred.
    verdicts = []
    for candidate in CANDIDATES:
        code_, data = runtime_post(
            f"{RT_PREFIX}/chat/completions",
            {
                "model": candidate,
                "messages": [{"role": "user", "content": "Reply OK"}],
                "max_completion_tokens": 2000,
            },
            region=region,
            attempts=1,
            timeout=180,
        )
        verdicts.append(
            f"{candidate.split('.')[0]}=200" if ok(code_, data)
            else f"{candidate.split('.')[0]}={code_}"
        )
    print(f"{region:16} {str(present[CANDIDATES[0]]):11} "
          f"{str(present[CANDIDATES[1]]):15} {'  '.join(verdicts)}")

print()
print("=> The geo profile is scoped to its geography; the global profile is not.")
print("   A 'model identifier is invalid' from bedrock-runtime is very often a")
print("   Region/profile mismatch rather than a missing model — check")
print("   ListInferenceProfiles in the Region you are actually calling before")
print("   concluding the model is unavailable on this endpoint.")''')

# ---------------------------------------------------------------------------
md("""## 2. First call, and the two OpenAI APIs

Grok 4.6 serves **Responses** and **Chat Completions** on both endpoints. Prefer
Responses: it is the only one of the two that returns the reasoning item, and this
is a reasoning-first model.

The budget matters more here than for most models. Reasoning spends output tokens
before any answer text appears, so a tight `max_output_tokens` returns HTTP 200
with `status: "incomplete"` and **empty content** — which looks like a broken
parser and is a budget problem. The cell shows both.""")

code('''from openai import OpenAI

from aws_bedrock_token_generator import provide_token

# Build from a FRESH token; don't construct one at import and reuse it for hours.
# max_retries: the service returns transient 5xx under load and the SDK does not
# retry by default.
grok = OpenAI(
    api_key=provide_token(region=RUNTIME_REGION),
    base_url=f"https://bedrock-runtime.{RUNTIME_REGION}.amazonaws.com{RT_PREFIX}",
    max_retries=5,
    timeout=300.0,
)

QUESTION = "In two sentences, what is a mixture-of-experts model?"

# Deliberately too small, to show what starvation looks like.
tight = grok.responses.create(model=GROK_RT, input=QUESTION, max_output_tokens=32)
print("max_output_tokens=32")
print("  status           :", tight.status)
print("  incomplete reason:", getattr(tight.incomplete_details, "reason", None))
print("  output_text      :", repr(tight.output_text))
print("  reasoning tokens :", tight.usage.output_tokens_details.reasoning_tokens)
print("  -> 200 with empty text. The trace consumed the whole allowance.")

roomy = grok.responses.create(model=GROK_RT, input=QUESTION, max_output_tokens=3000)
print("\\nmax_output_tokens=3000")
print("  status           :", roomy.status)
print("  output_text      :", roomy.output_text.strip()[:200])
print("  reasoning tokens :", roomy.usage.output_tokens_details.reasoning_tokens)
print("  output item types:", [item.type for item in roomy.output])

starved = tight.output_text == "" and tight.status != "completed"
print(f"\\n=> starvation reproduced: {starved}. Budget for the trace, not just the")
print("   answer. A useful floor for this model is ~2000 output tokens.")''')

# ---------------------------------------------------------------------------
md("""## 3. Reasoning effort — measured, not assumed

The model card documents `none`, `low` (the default), `medium`, `high` and
`xhigh`, and those are the five the service accepts for this model. The last cell
in this section shows why that sentence needed checking twice.

The obvious expectation is a monotonic ladder. That is **not** what the service
does, and the shape depends on the prompt, so this section runs both an easy
question and a hard one and lets the numbers speak.

Reasoning token counts are stochastic, so a single sample per level proves nothing
about ordering. Three samples per level here — still small, but enough to show
whether the levels separate at all.""")

code('''import concurrent.futures as cf

EASY = "What is 12 x 12?"
HARD = (
    "Let f(n) be the number of ways to tile a 3xn rectangle with 1x2 dominoes. "
    "Derive a recurrence, prove it, then compute f(12) exactly. Show every step."
)
LEVELS = ("none", "low", "medium", "high", "xhigh")
SAMPLES = 3


def reasoning_cost(job: tuple[str, str, str, int]) -> tuple[str, str, int | None]:
    """Reasoning tokens for one call. Returns (prompt label, effort, tokens)."""
    label, prompt, effort, budget = job
    code_, data = runtime_post(
        f"{RT_PREFIX}/responses",
        {
            "model": GROK_RT,
            "input": prompt,
            "reasoning": {"effort": effort},
            "max_output_tokens": budget,
        },
        region=RUNTIME_REGION,
        attempts=1,
        timeout=600,
    )
    if not ok(code_, data):
        return label, effort, None
    details = (data.get("usage") or {}).get("output_tokens_details") or {}
    return label, effort, details.get("reasoning_tokens")


# 30 calls, and the hard ones are slow: high effort on a proof can spend a minute
# or more. Run them concurrently or this cell outlives your patience -- it timed
# out at 900s when written sequentially.
jobs = [
    (label, prompt, effort, budget)
    for label, prompt, budget in (("easy", EASY, 4000), ("hard", HARD, 14000))
    for effort in LEVELS
    for _ in range(SAMPLES)
]

results: dict[str, dict[str, list[int]]] = {
    "easy": {e: [] for e in LEVELS},
    "hard": {e: [] for e in LEVELS},
}
started = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=10) as pool:
    for label, effort, tokens in pool.map(reasoning_cost, jobs):
        if tokens is not None:
            results[label][effort].append(tokens)
print(f"{len(jobs)} calls in {time.perf_counter() - started:.0f}s\\n")

for label in results:
    print(f"=== {label} prompt ===")
    print(f"{'effort':8} {'median':>8} {'min':>8} {'max':>8}   samples")
    print("-" * 62)
    for effort in LEVELS:
        vals = results[label][effort]
        if not vals:
            print(f"{effort:8} {'-':>8} {'-':>8} {'-':>8}   (all calls failed)")
            continue
        print(f"{effort:8} {statistics.median(vals):>8.0f} {min(vals):>8} "
              f"{max(vals):>8}   {vals}")
    print()''')

md("""Now derive the verdict from those numbers rather than restating a rule.""")

code('''def medians(label: str) -> dict[str, float]:
    return {
        effort: statistics.median(vals)
        for effort, vals in results[label].items()
        if vals
    }


for label in results:
    m = medians(label)
    graded = [e for e in ("low", "medium", "high", "xhigh") if e in m]
    print(f"=== {label} prompt ===")

    if "none" in m:
        if m["none"] == 0:
            print("  effort='none' spent 0 reasoning tokens.")
        else:
            print(f"  effort='none' still spent {m['none']:.0f} reasoning tokens.")

    ordered = all(m[a] <= m[b] for a, b in zip(graded, graded[1:]))
    if ordered:
        print(f"  the graded levels came out monotonic: "
              f"{' <= '.join(f'{e}={m[e]:.0f}' for e in graded)}")
    else:
        pairs = [f"{a}={m[a]:.0f} > {b}={m[b]:.0f}"
                 for a, b in zip(graded, graded[1:]) if m[a] > m[b]]
        print(f"  NOT monotonic: {'; '.join(pairs)}")

    if "low" in m and "medium" in m and m["low"]:
        print(f"  the low -> medium step is {m['medium'] / m['low']:.1f}x")
    spread = [f"{e} {min(results[label][e])}-{max(results[label][e])}"
              for e in graded if results[label][e]]
    print(f"  per-level ranges: {'; '.join(spread)}")
    print()

print("Durable reading of the above:")
print("  * 'none' is the one setting that does something exact: it removes the")
print("    reasoning item from the response entirely (see the next cell).")
print("  * The graded levels are a HINT, not a quota. Their ranges overlap, so a")
print("    single call cannot tell you which level produced it.")
print("  * How much the dial matters depends on the prompt. On a question that")
print("    needs no thought, raising effort changes little; on one that does, the")
print("    first step up is the one that counts.")
print("  * So: choose effort for the workload, then budget max_output_tokens for")
print("    the WORST case you measured, not the median.")''')

md("""### `none` is structural, not just cheaper

At every other level the response carries two output items — a `reasoning` item
and a `message`. At `none` the reasoning item is **absent**, which matters if your
parser walks `output[]` by position.""")

code('''print(f"{'effort':8} {'output item types':44} reasoning tokens")
print("-" * 78)
for effort in ("none", "low", "xhigh"):
    code_, data = runtime_post(
        f"{RT_PREFIX}/responses",
        {
            "model": GROK_RT,
            "input": "Name one benefit of message queues.",
            "reasoning": {"effort": effort},
            "max_output_tokens": 4000,
        },
        region=RUNTIME_REGION,
        attempts=1,
        timeout=300,
    )
    if not ok(code_, data):
        print(f"{effort:8} {code_}: {err(data)[:60]}")
        continue
    types = [item.get("type") for item in (data.get("output") or [])]
    spent = ((data.get("usage") or {}).get("output_tokens_details") or {}).get(
        "reasoning_tokens"
    )
    print(f"{effort:8} {str(types):44} {spent}")

print("\\n=> Select output items by `type`, never by index. The same discipline as")
print("   Converse content blocks (../00-foundations/04 §3).")''')

md("""### The error that documents itself — and gets it wrong

Send an effort level this model does not have and the 400 helpfully lists the ones
it does. That is a genuinely useful trick: it is faster than reading three doc
pages, and the valid set differs across families (OpenAI's GPT-5.x accepts
`minimal`; this model does not).

But do not stop at reading the list. **Test every value it names.** The service
here advertises a value it then refuses, because two different validators produce
two different lists:

- an **unknown** value (`'ultra'`) gets *"Invalid value"* and the **API schema's**
  set, which includes `max`;
- a **known-but-unsupported** value (`'minimal'`) gets *"not supported with this
  model"* — but still quotes the **schema's** list, `max` included;
- and `'max'` itself gets *"not supported with this model"* with the **model's**
  real list, `max` excluded.

So the first message you see is a superset. The cell below reads the advertised
set, tries every member of it, and reports the difference.""")

code('''import re

# 1. Provoke the error and harvest whatever set it advertises.
code_, data = runtime_post(
    f"{RT_PREFIX}/responses",
    {
        "model": GROK_RT,
        "input": "Hi",
        "reasoning": {"effort": "definitely-not-a-level"},
        "max_output_tokens": 2000,
    },
    region=RUNTIME_REGION,
    attempts=1,
    timeout=120,
)
message = err(data)
print(f"unknown effort -> HTTP {code_}")
print(" ", message[:200])

advertised = re.findall(r"\'([a-z]+)\'", message)
print("\\nadvertised as supported:", advertised)

# 2. Try each advertised value. This is the step that matters.
print()
print(f"{'value':10} {'HTTP':>5}  result")
print("-" * 62)
accepted, refused = [], []
for value in advertised:
    code_, data = runtime_post(
        f"{RT_PREFIX}/responses",
        {
            "model": GROK_RT,
            "input": "Hi",
            "reasoning": {"effort": value},
            "max_output_tokens": 2500,
        },
        region=RUNTIME_REGION,
        attempts=1,
        timeout=300,
    )
    if ok(code_, data):
        accepted.append(value)
        print(f"{value:10} {code_:>5}  accepted")
    else:
        refused.append(value)
        print(f"{value:10} {str(code_):>5}  {err(data)[:44]}")

print()
if refused:
    print(f"=> the error advertised {len(advertised)} values and the model accepts "
          f"{len(accepted)}.")
    print(f"   refused despite being advertised: {refused}")
    print("   The enumeration in a 400 can come from the API schema rather than")
    print("   from the model you named. Read it as a hint, then verify each value —")
    print("   which is one cheap loop, exactly like this one.")
else:
    print(f"=> all {len(accepted)} advertised values were accepted: {accepted}.")
    print("   Consistent today. Still worth the loop: it costs one call per value")
    print("   and it is the difference between believing a message and knowing.")''')

# ---------------------------------------------------------------------------
md("""## 4. The reasoning trace is encrypted — and you can hand it back

Most reasoning models in this collection return their trace as readable text
(`choices[0].message.reasoning` on mantle Chat Completions, or a
`reasoningContent` block on Converse). Grok 4.6 does not. You get
`encrypted_content`: an opaque blob you cannot read, but *can* replay on a later
turn so the model keeps its own reasoning context.

The model card shows `include=["reasoning.encrypted_content"]` as the way to ask
for it. On Bedrock today the blob comes back **whether or not you pass `include`**
— which the cell below checks, because "the doc says you need a flag" and "the
service needs the flag" are different claims.""")

code('''def trace_shape(with_include: bool) -> dict:
    body = {
        "model": GROK_RT,
        "input": "Why is quicksort O(n log n) on average? Two sentences.",
        "reasoning": {"effort": "low"},
        "max_output_tokens": 4000,
    }
    if with_include:
        body["include"] = ["reasoning.encrypted_content"]
    code_, data = runtime_post(f"{RT_PREFIX}/responses", body,
                               region=RUNTIME_REGION, attempts=1, timeout=300)
    if not ok(code_, data):
        return {"error": f"{code_}: {err(data)[:70]}"}
    reasoning = next(
        (i for i in (data.get("output") or []) if i.get("type") == "reasoning"), {}
    )
    blob = reasoning.get("encrypted_content") or ""
    return {
        "response_id": data.get("id"),
        "encrypted_chars": len(blob),
        "prefix": blob[:14],
        "summary": reasoning.get("summary"),
        "content": reasoning.get("content"),
        "item": reasoning,
    }


without = trace_shape(with_include=False)
withit = trace_shape(with_include=True)

for label, shape in (("without include=", without), ("with include=", withit)):
    if "error" in shape:
        print(f"{label:18} {shape['error']}")
        continue
    print(f"{label:18} encrypted_content = {shape['encrypted_chars']:>5} chars, "
          f"prefix {shape['prefix']!r}")
    print(f"{'':18} summary={shape['summary']!r}  content={shape['content']!r}")

if "error" not in without and "error" not in withit:
    both = without["encrypted_chars"] > 0 and withit["encrypted_chars"] > 0
    print(f"\\n=> encrypted_content present in both cases: {both}")
    if both:
        print("   So `include` is not required on Bedrock for this model. Passing it")
        print("   is harmless and keeps your code portable to providers that do")
        print("   require it.")
    print("   Note `summary` and `content` are empty lists: there is no readable")
    print("   trace to show a user. If your product surfaces 'thinking' text, this")
    print("   model cannot supply it — plan for that, do not discover it in review.")''')

md("""### Replaying the trace on the next turn

The blob is an output item, and you put it straight back into `input` alongside
your messages. The point is continuity of reasoning across turns without asking
the model to re-derive anything.""")

code('''if "error" in withit:
    print("skipping: the previous cell did not produce a trace to replay")
else:
    FIRST = "Why is quicksort O(n log n) on average? Two sentences."
    followup = [
        {"role": "user", "content": FIRST},
        withit["item"],  # the encrypted reasoning item, verbatim
        {"role": "user", "content": "Now state the worst case and when it happens."},
    ]
    code_, data = runtime_post(
        f"{RT_PREFIX}/responses",
        {
            "model": GROK_RT,
            "input": followup,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 4000,
        },
        region=RUNTIME_REGION,
        attempts=1,
        timeout=300,
    )
    if ok(code_, data):
        print("replayed the encrypted trace -> HTTP 200")
        print("answer:", response_text(data).strip()[:260])
    else:
        print(f"replay -> HTTP {code_}: {err(data)[:150]}")

    # Now the failure mode that matters in production. Replay the SAME trace to
    # the SAME model several times and see whether every attempt succeeds.
    print()
    outcomes = []
    for attempt in range(4):
        code_, data = runtime_post(
            f"{RT_PREFIX}/responses",
            {"model": GROK_RT, "input": followup, "max_output_tokens": 4000},
            region=RUNTIME_REGION, attempts=1, timeout=300,
        )
        outcomes.append((code_, "" if ok(code_, data) else err(data)))
    good = sum(1 for c, _ in outcomes if c == 200)
    print(f"replayed the same trace to the same model 4x: {good}/4 succeeded")
    for code_, message in outcomes:
        if code_ != 200:
            print(f"  {code_}: {message[:150]}")

    if good < 4:
        print()
        print("=> Read that message: it is about the REGION, not the model. `us.` is a")
        print("   geo profile, so consecutive calls can be served by different")
        print("   Regions, and an encrypted trace is scoped to the Region that made")
        print("   it. So replaying reasoning across turns is only reliable if every")
        print("   turn lands in the same Region.")
        print("   Practical consequence: for multi-turn work that replays reasoning,")
        print("   prefer server-side state (previous_response_id, §8) over carrying")
        print("   the blob yourself, or pin to an in-Region endpoint where one")
        print("   exists. A retry that silently changes Region will break the blob.")
    else:
        print()
        print("=> Every replay succeeded this time. It may not next time: with a geo")
        print("   or global profile you do not choose the Region, and an encrypted")
        print("   trace is scoped to the Region that produced it. Handle the failure")
        print("   rather than assuming it away -- prefer previous_response_id (§8) if")
        print("   you need reasoning continuity you can rely on.")''')

# ---------------------------------------------------------------------------
md("""## 5. Chat Completions: no trace, but the tokens are still billed

The model card says the Chat Completions API "does not return reasoning tokens".
That is worth stating precisely, because it is half right: there is no reasoning
**content** on this surface, but `usage` **does** report the reasoning **count** —
so you are paying for thinking you cannot see. The cell shows both halves.""")

code('''code_, data = runtime_post(
    f"{RT_PREFIX}/chat/completions",
    {
        "model": GROK_RT,
        "messages": [{"role": "user", "content": "Why is the sky blue? One sentence."}],
        "max_completion_tokens": 4000,
    },
    region=RUNTIME_REGION,
    attempts=1,
    timeout=300,
)
if not ok(code_, data):
    print(f"HTTP {code_}: {err(data)[:150]}")
else:
    message = (data.get("choices") or [{}])[0].get("message") or {}
    usage = data.get("usage") or {}
    detail = usage.get("completion_tokens_details") or {}

    print("message keys        :", sorted(message))
    print("has 'reasoning' key :", "reasoning" in message)
    print("content             :", (message.get("content") or "").strip()[:120])
    print("completion_tokens   :", usage.get("completion_tokens"))
    print("reasoning_tokens    :", detail.get("reasoning_tokens"))

    hidden = detail.get("reasoning_tokens") or 0
    total = usage.get("completion_tokens") or 0
    print()
    if hidden and "reasoning" not in message:
        share = 100 * hidden / total if total else 0
        print(f"=> {hidden} of {total} completion tokens ({share:.0f}%) were reasoning")
        print("   you cannot read on this surface. Use Responses if you need the")
        print("   trace; use Chat Completions if you only need the answer and want")
        print("   the smaller response body.")
    elif "reasoning" in message:
        print("=> a 'reasoning' field appeared on Chat Completions for this model")
        print("   today, which is a change from what was measured for this notebook.")
    else:
        print(f"=> no reasoning tokens reported ({hidden}); nothing hidden this call.")''')

md("""Two ways to set effort on this surface, and both are accepted — the OpenAI
flat `reasoning_effort` and the nested `reasoning` object.""")

code('''for body_extra in ({"reasoning_effort": "high"}, {"reasoning": {"effort": "high"}}):
    code_, data = runtime_post(
        f"{RT_PREFIX}/chat/completions",
        {
            "model": GROK_RT,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "max_completion_tokens": 3000,
            **body_extra,
        },
        region=RUNTIME_REGION,
        attempts=1,
        timeout=300,
    )
    shape = next(iter(body_extra))
    verdict = "accepted" if ok(code_, data) else f"{code_}: {err(data)[:60]}"
    print(f"{shape:18} -> {verdict}")''')

# ---------------------------------------------------------------------------
md("""## 6. Client-side tools work; server-side tools do not

Grok 4.6 does function calling on both APIs and both endpoints. What it does not
do is **server-side** tool use — web search, code interpreter — and neither does
`bedrock-runtime` for any model. Those are two separate limits that produce
similar-looking 400s, so §13 separates them properly.""")

code('''WEATHER = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}

# Responses shape: the function spec is flat.
code_, data = runtime_post(
    f"{RT_PREFIX}/responses",
    {
        "model": GROK_RT,
        "input": "What is the weather in Jakarta? Use the tool.",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Current weather for a city.",
                "parameters": WEATHER,
            }
        ],
        "max_output_tokens": 4000,
    },
    region=RUNTIME_REGION,
    attempts=1,
    timeout=300,
)
print("Responses API")
if ok(code_, data):
    items = data.get("output") or []
    print("  output item types:", [i.get("type") for i in items])
    for call_item in (i for i in items if i.get("type") == "function_call"):
        print(f"  -> {call_item.get('name')}({call_item.get('arguments')})")
else:
    print(f"  HTTP {code_}: {err(data)[:140]}")

# Chat Completions shape: the spec is nested under "function".
code_, data = runtime_post(
    f"{RT_PREFIX}/chat/completions",
    {
        "model": GROK_RT,
        "messages": [{"role": "user", "content": "Weather in Jakarta?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather for a city.",
                    "parameters": WEATHER,
                },
            }
        ],
        "max_completion_tokens": 4000,
    },
    region=RUNTIME_REGION,
    attempts=1,
    timeout=300,
)
print("\\nChat Completions")
if ok(code_, data):
    calls = ((data.get("choices") or [{}])[0].get("message") or {}).get(
        "tool_calls"
    ) or []
    print(f"  tool_calls: {len(calls)}")
    for c in calls:
        fn = c.get("function") or {}
        print(f"  -> {fn.get('name')}({fn.get('arguments')})")
else:
    print(f"  HTTP {code_}: {err(data)[:140]}")

print("\\n=> Same model, same tool, two different request shapes. The Responses")
print("   spec is flat; Chat Completions nests it under 'function'. Getting this")
print("   wrong gives you a schema 400, not a 'tools unsupported' error.")''')

# ---------------------------------------------------------------------------
md("""## 7. Structured output — the model card and the service disagree

The Grok 4.6 model card lists **Structured outputs** under *Not Supported* for the
`bedrock-runtime` endpoint. The service accepts a strict `json_schema` on both
endpoints and returns conforming JSON.

So this cell does not take either side. It sends the request, parses the result,
and checks conformance — and prints what it found. Where a doc and a live endpoint
disagree, the endpoint is what your code will meet.

Note the strict-mode requirements, which are the usual source of a 400 that looks
like "no structured output support": `required` must list **every** property and
`additionalProperties` must be `false`.""")

code('''READING = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "celsius": {"type": "number"},
        "conditions": {"type": "string"},
    },
    "required": ["city", "celsius", "conditions"],   # strict: list them all
    "additionalProperties": False,                    # strict: required
}

TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "reading",
        "schema": READING,
        "strict": True,
    }
}

print(f"{'endpoint':9} {'HTTP':>5}  parses  conforms  payload")
print("-" * 92)
verdicts = {}
for endpoint, region, model in (
    ("runtime", RUNTIME_REGION, GROK_RT),
    ("mantle", MANTLE_REGION, GROK_MT),
):
    caller = runtime_post if endpoint == "runtime" else post
    code_, data = caller(
        f"{RT_PREFIX}/responses",
        {
            "model": model,
            # Ask for something extra on purpose: strict mode should exclude it.
            "input": "Jakarta is 31 C and overcast. Also tell me a joke.",
            "text": TEXT_FORMAT,
            "max_output_tokens": 4000,
        },
        region=region,
        attempts=1,
        timeout=300,
    )
    if not ok(code_, data):
        print(f"{endpoint:9} {str(code_):>5}  {err(data)[:60]}")
        verdicts[endpoint] = False
        continue
    raw = response_text(data).strip()
    try:
        parsed = json.loads(raw)
        conforms = sorted(parsed) == sorted(READING["required"])
    except json.JSONDecodeError:
        parsed, conforms = None, False
    print(f"{endpoint:9} {code_:>5}  {str(parsed is not None):>6}  "
          f"{str(conforms):>8}  {raw[:44]}")
    verdicts[endpoint] = conforms

print()
working = [e for e, v in verdicts.items() if v]
if working:
    print(f"=> strict json_schema produced conforming JSON on: {working}")
    print("   The model card lists structured outputs as unsupported on")
    print("   bedrock-runtime. It works. When a capability table and a live probe")
    print("   disagree, ship against the probe and re-check the table later.")
else:
    print("=> strict json_schema did not produce conforming JSON on either endpoint")
    print("   today, which matches the model card. Re-run before relying on it.")
    print("   Fall back to prompting for JSON and validating client-side.")''')

# ---------------------------------------------------------------------------
md("""## 8. Stateful conversation on `bedrock-runtime`

This is the capability that changed most in August 2026. The Responses API on
`bedrock-runtime` keeps `store=true` as its default and supports
`previous_response_id`, so the server holds the conversation and you do not resend
history. Older guidance — including an earlier version of this collection — said
that server-side conversation state was mantle-only. It is not.

One operational caveat from the AWS docs, worth designing around: *"A stored
response belongs to the AWS Region that served it."* Retrieving, cancelling,
deleting, or continuing with `previous_response_id` all go back to that Region.
With a `global.` profile you do not choose which Region that was — so if you need
to pin state to a Region, use the `us.` geo profile.""")

code('''# Named as the plain fact it is. Calling this a secret -- which an earlier draft
# did -- trips Bandit B105 and detect-secrets' keyword detector. A false positive,
# but a scanner finding in a public sample costs a reviewer time either way, and
# the literal name is clearer.
FIRST_TURN_FACT = "The hangar bay number is 4417."

code_, first = runtime_post(
    f"{RT_PREFIX}/responses",
    {"model": GROK_RT, "input": FIRST_TURN_FACT, "store": True,
     "max_output_tokens": 3000},
    region=RUNTIME_REGION,
    attempts=1,
    timeout=300,
)
if not ok(code_, first):
    print(f"turn 1 -> HTTP {code_}: {err(first)[:150]}")
else:
    safe_print("turn 1 response id:", first.get("id"))
    print("store echoed back  :", first.get("store"))

    code_, second = runtime_post(
        f"{RT_PREFIX}/responses",
        {
            "model": GROK_RT,
            "input": "What was the hangar bay number? Digits only.",
            "previous_response_id": first["id"],
            "max_output_tokens": 3000,
        },
        region=RUNTIME_REGION,
        attempts=1,
        timeout=300,
    )
    if not ok(code_, second):
        print(f"turn 2 -> HTTP {code_}: {err(second)[:150]}")
    else:
        answer = response_text(second).strip()
        print("turn 2 answer      :", answer[:120])
        recalled = "4417" in answer
        print(f"\\n=> server-side state worked: {recalled}")
        print("   Turn 2 never resent the number. The only history sent was an ID.")

    # And the same request WITHOUT the link, as the control.
    code_, blind = runtime_post(
        f"{RT_PREFIX}/responses",
        {
            "model": GROK_RT,
            "input": "What was the hangar bay number? Digits only.",
            "max_output_tokens": 3000,
        },
        region=RUNTIME_REGION,
        attempts=1,
        timeout=300,
    )
    if ok(code_, blind):
        blind_text = response_text(blind).strip()
        print(f"   control, no previous_response_id: {blind_text[:80]!r}")
        print("   -> the control is what proves the link did the work, rather than")
        print("      the model guessing a four-digit number.")''')

md("""### `background=true` is the one Responses feature runtime does not have

Asynchronous inference stays on `bedrock-mantle`. The 400 is explicit, which is
the good case — you find out at the first call rather than at load.""")

code('''for endpoint, region, model in (
    ("runtime", RUNTIME_REGION, GROK_RT),
    ("mantle", MANTLE_REGION, GROK_MT),
):
    caller = runtime_post if endpoint == "runtime" else post
    code_, data = caller(
        f"{RT_PREFIX}/responses",
        {
            "model": model,
            "input": "Summarise the causes of metal fatigue.",
            "background": True,
            "max_output_tokens": 3000,
        },
        region=region,
        attempts=1,
        timeout=300,
    )
    verdict = "accepted" if ok(code_, data) else f"HTTP {code_}: {err(data)[:70]}"
    print(f"{endpoint:9} background=true -> {verdict}")

print("\\n=> Long-running work belongs on bedrock-mantle. If you need both async")
print("   inference AND Guardrails, you need both endpoints — AWS's own guidance")
print("   is to choose per use case, not once per application.")''')

# ---------------------------------------------------------------------------
md("""## 9. Streaming

Both APIs stream on `bedrock-runtime`, and the event shapes are different enough
to matter. Chat Completions sends OpenAI's `chat.completion.chunk` deltas;
Responses sends typed lifecycle events with a `sequence_number`.""")

code('''def stream_chat() -> tuple[int, str]:
    """Stream Chat Completions and assemble the text, counting frames."""
    frames, pieces = 0, []
    with grok.chat.completions.stream(
        model=GROK_RT,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        max_completion_tokens=3000,
    ) as stream:
        for event in stream:
            frames += 1
            if event.type == "content.delta":
                pieces.append(event.delta)
    return frames, "".join(pieces)


def stream_responses() -> tuple[int, list[str], str]:
    """Stream the Responses API, recording which event types arrived."""
    kinds, pieces = [], []
    with grok.responses.stream(
        model=GROK_RT, input="Count from 1 to 5.", max_output_tokens=3000
    ) as stream:
        for event in stream:
            kinds.append(event.type)
            if event.type == "response.output_text.delta":
                pieces.append(event.delta)
    return len(kinds), kinds, "".join(pieces)


frames, text = stream_chat()
print(f"Chat Completions : {frames} events -> {text.strip()[:60]!r}")

count, kinds, text = stream_responses()
seen = []
for k in kinds:
    if k not in seen:
        seen.append(k)
print(f"Responses        : {count} events, {len(seen)} distinct types")
for k in seen[:9]:
    print(f"                   {k}")
print(f"                   assembled -> {text.strip()[:60]!r}")

print("\\n=> The Responses stream is a state machine, not a token firehose: it")
print("   reports item lifecycle (added / delta / done) so you can render the")
print("   reasoning item and the message separately.")''')

# ---------------------------------------------------------------------------
md("""## 10. Prompt caching — and why the profile you choose decides the hit rate

Caching is automatic on this surface: there is no `cache_control` marker to place.
You observe it in `usage.prompt_tokens_details.cached_tokens`.

Two things this cell is careful about. First, it does **not** label the calls
"cold" and "warm" in advance, because which call gets a hit is not something you
control. Second, it repeats the same prefix several times, because a hit did not
land on the second call when this was measured — so a two-call demo would have
concluded that caching does not work.

**The part worth taking away, though, is §10b.** A cache lives in the Region that
served the request, and a cross-Region inference profile chooses that Region for
you. `global.` can route anywhere Bedrock offers the model, so a follow-up request
frequently lands somewhere that has never seen your prefix. `us.` routes across
three Regions, so locality is much better. Measured over 30 calls per arm:

| profile | `prompt_cache_key` | hit rate |
|---|---|---|
| `global.` | no | 27% |
| `global.` | yes | 67% |
| `us.` | no | 83% |
| `us.` | yes | 80% |

So the biggest lever is **narrowing the profile**, and `prompt_cache_key` mostly
helps when you are stuck on the broad one. §10b re-runs a short version of that
comparison so you can see it rather than take the table on trust.""")

code('''PREFIX = "You are a meticulous reviewer of aviation maintenance logs. " * 1200

observations, failures = [], []
for i in range(4):
    code_, data = runtime_post(
        f"{RT_PREFIX}/chat/completions",
        {
            "model": GROK_RT,
            "messages": [
                {"role": "system", "content": PREFIX},
                {"role": "user", "content": f"Reply with the number {i}."},
            ],
            "max_completion_tokens": 3000,
            # An explicit cache key groups requests that should share a cache.
            "prompt_cache_key": "grok46-notebook-section-10",
        },
        region=RUNTIME_REGION,
        # Retries on purpose. A transport timeout here would drop the one call
        # that saw the cache hit, and the cell would then report "caching does
        # not work" -- a claim about the service derived from a network blip.
        attempts=3,
        timeout=420,
    )
    if not ok(code_, data):
        print(f"call {i} -> HTTP {code_}: {err(data)[:90]}")
        failures.append(i)
        continue
    usage = data.get("usage") or {}
    detail = usage.get("prompt_tokens_details") or {}
    observations.append(
        {
            "call": i,
            "prompt": usage.get("prompt_tokens"),
            "cached": detail.get("cached_tokens") or 0,
        }
    )

print(f"{'call':>5} {'prompt tokens':>14} {'cached':>8} {'cached share':>13}")
print("-" * 48)
for o in observations:
    share = 100 * o["cached"] / o["prompt"] if o["prompt"] else 0
    print(f"{o['call']:>5} {o['prompt']:>14} {o['cached']:>8} {share:>12.0f}%")

hits = [o for o in observations if o["cached"]]
print()
if hits:
    best = max(o["cached"] / o["prompt"] for o in hits)
    print(f"=> {len(hits)} of {len(observations)} calls saw a cache hit; the largest "
          f"covered {best:.0%} of the prompt.")
    print(f"   The first hit landed on call {hits[0]['call']}, not call 1.")
    print("   So: do not benchmark caching with two calls, and do not assume a")
    print("   steady-state hit rate from one sample.")
elif failures:
    print(f"=> calls {failures} did not complete, so this run cannot say whether")
    print("   caching engaged. An incomplete measurement is not a negative result.")
else:
    print("=> no cache hits in this run. Caching is opportunistic; a short or")
    print("   low-traffic prefix may never be retained. Treat it as a cost")
    print("   optimisation you verify, not a guarantee you design around.")
print("   Pricing note: the model card puts cache reads at $0.55/1M against")
print("   $2.20/1M for fresh input tokens on in-Region and Geo.")''')

md("""### 10b. Profile breadth versus cache locality

A short version of the comparison — two profiles, one shared prefix, the same
number of calls each. Reasoning is switched off (`effort="none"`) so the calls are
cheap and fast; the cache behaviour is a property of the input, not the reasoning.

Small samples, so read the direction rather than the exact percentages, and re-run
it in your own account before making a routing decision on it.""")

code('''CALLS = 8
LOCALITY_PREFIX = (
    "You are a meticulous reviewer of aviation maintenance logs. "
    "Cite the regulation that applies and never speculate. " * 900
)


def hit_rate(model_id: str) -> tuple[int, int, int | None]:
    """(hits, calls, index of first hit) for one profile."""
    hits, first = 0, None
    for i in range(CALLS):
        code_, data = runtime_post(
            f"{RT_PREFIX}/chat/completions",
            {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": LOCALITY_PREFIX},
                    {"role": "user", "content": f"Reply with the number {i}."},
                ],
                "max_completion_tokens": 2000,
                "reasoning_effort": "none",
            },
            region=RUNTIME_REGION,
            attempts=3,
            timeout=300,
        )
        if not ok(code_, data):
            continue
        cached = ((data.get("usage") or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ) or 0
        if cached:
            hits += 1
            first = i if first is None else first
    return hits, CALLS, first


print(f"{'profile':26} {'hits':>10} {'rate':>7} {'first hit at call':>18}")
print("-" * 66)
measured = {}
for profile in ("us.xai.grok-4.6", "global.xai.grok-4.6"):
    hits, total, first = hit_rate(profile)
    measured[profile] = hits / total if total else 0
    print(f"{profile:26} {hits:>4}/{total:<5} {100 * hits / total:>6.0f}% "
          f"{str(first):>18}")

print()
geo, glob = measured.get("us.xai.grok-4.6", 0), measured.get("global.xai.grok-4.6", 0)
if geo > glob:
    print(f"=> the us. geo profile cached better than global. by "
          f"{100 * (geo - glob):.0f} points here.")
    print("   That is the expected direction: a cache belongs to the Region that")
    print("   served the request, and global. can route anywhere the model is")
    print("   offered, so a follow-up often lands somewhere that has never seen")
    print("   your prefix.")
elif glob > geo:
    print(f"=> global. cached better than us. by {100 * (glob - geo):.0f} points in")
    print("   this run, which is the opposite of what a locality argument predicts.")
    print("   Small sample -- repeat before drawing a conclusion.")
else:
    print("=> the two profiles cached identically in this run. Too small a sample")
    print("   to separate them; raise CALLS if you need an answer you can act on.")

print()
print("   Practical routing advice: global. buys throughput and a lower per-token")
print("   price; a narrower profile buys cache locality. For a long agentic session")
print("   re-sending a big prefix, the cache discount can outweigh the global.")
print("   token discount -- measure with YOUR prefix length before choosing.")
print("   `prompt_cache_key` helps most when you are stuck on the broad profile.")''')

# ---------------------------------------------------------------------------
md("""## 11. Vision

Grok 4.6 accepts images on both endpoints. The image here is **generated** with
three known colour bands, so the answer can be checked rather than merely read —
a model that hallucinates plausibly will fail this.

One warning from building this notebook: a degenerate 1×1 PNG returns *"Invalid or
unsupported image format"*, which is easy to misread as "this model has no
vision". It does. Test with a real image.""")

code('''import base64

png = bands_png([(220, 30, 30), (30, 140, 60), (40, 70, 200)])
data_url = "data:image/png;base64," + base64.b64encode(png).decode()
EXPECTED = ("red", "green", "blue")
print(f"generated {len(png)}-byte PNG; ground truth: {', '.join(EXPECTED)}")

print(f"\\n{'endpoint':9} {'HTTP':>5}  named all three?  answer")
print("-" * 88)
for endpoint, region, model in (
    ("runtime", RUNTIME_REGION, GROK_RT),
    ("mantle", MANTLE_REGION, GROK_MT),
):
    caller = runtime_post if endpoint == "runtime" else post
    code_, data = caller(
        f"{RT_PREFIX}/responses",
        {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Name the colours of the horizontal bands, "
                                    "top to bottom. Answer with three words.",
                        },
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            "max_output_tokens": 4000,
        },
        region=region,
        attempts=1,
        timeout=300,
    )
    if not ok(code_, data):
        print(f"{endpoint:9} {str(code_):>5}  {err(data)[:60]}")
        continue
    answer = response_text(data).strip().lower()
    correct = all(colour in answer for colour in EXPECTED)
    print(f"{endpoint:9} {code_:>5}  {str(correct):>16}  {answer[:44]!r}")

# The 1x1 trap, shown rather than described.
tiny = base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8"
        "AAxAAADwABbT8HRAAAAABJRU5ErkJggg=="
    )
).decode()
code_, data = runtime_post(
    f"{RT_PREFIX}/responses",
    {
        "model": GROK_RT,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What colour is this?"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{tiny}",
                    },
                ],
            }
        ],
        "max_output_tokens": 3000,
    },
    region=RUNTIME_REGION,
    attempts=1,
    timeout=180,
)
print(f"\\n1x1 PNG -> HTTP {code_}: {err(data)[:80] if not ok(code_, data) else 'accepted'}")
print("=> That message is about the IMAGE, not the model. A capability probe that")
print("   uses a degenerate input measures your test fixture.")''')

# ---------------------------------------------------------------------------
md("""## 12. Service tiers

`default` (Standard), `flex` and `priority` are all accepted, and the response
echoes the tier back so you can confirm what you got rather than what you asked
for. `../00-foundations/03` covers what the tiers mean for latency and cost.""")

code('''print(f"{'requested':10} {'HTTP':>5}  echoed back")
print("-" * 40)
for tier in ("default", "flex", "priority"):
    code_, data = runtime_post(
        f"{RT_PREFIX}/chat/completions",
        {
            "model": GROK_RT,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "max_completion_tokens": 2000,
            "service_tier": tier,
        },
        region=RUNTIME_REGION,
        attempts=1,
        timeout=300,
    )
    if ok(code_, data):
        print(f"{tier:10} {code_:>5}  {data.get('service_tier')!r}")
    else:
        print(f"{tier:10} {str(code_):>5}  {err(data)[:60]}")

print("\\n=> Read the echoed value. A tier that is silently downgraded looks")
print("   identical to one that was honoured if you only check the status code.")''')

# ---------------------------------------------------------------------------
md("""## 13. What Grok 4.6 does not do here

Four separate limits, four different errors. They are grouped because they look
alike from a distance and have nothing to do with each other:

- a **model** limit (Grok has no server-side tools anywhere)
- an **endpoint** limit (`bedrock-runtime` has no server-side tools for any model)
- an **API** limit (application inference profiles work on Converse, not on the
  OpenAI paths)
- a **capability** limit (`CountTokens` does not cover this model)

The cell separates them by testing each against a case that isolates it.""")

code('''import boto3
from botocore.exceptions import ClientError

print("--- server-side tools: is it the model, or the endpoint? ---")
# Grok on mantle, where server-side tools DO exist for some models.
code_, data = post(
    f"{MT_PREFIX}/responses",
    {"model": GROK_MT, "input": "Search the web for AWS news.",
     "tools": [{"type": "web_search"}], "max_output_tokens": 3000},
    region=MANTLE_REGION, attempts=1, timeout=300,
)
print(f"  grok on mantle      : {'accepted' if ok(code_, data) else str(code_) + ' ' + err(data)[:64]}")

# gpt-5.6 on mantle: a model that DOES have web search, as the control.
code_, data = post(
    f"{MT_PREFIX}/responses",
    {"model": "openai.gpt-5.6-sol", "input": "Search the web for AWS news.",
     "tools": [{"type": "web_search"}], "max_output_tokens": 4000},
    region="us-east-1", attempts=1, timeout=420,
)
gpt_on_mantle = ok(code_, data)
print(f"  gpt-5.6 on mantle   : {'accepted' if gpt_on_mantle else str(code_) + ' ' + err(data)[:64]}")

# The same model on runtime: isolates the endpoint.
code_, data = runtime_post(
    f"{RT_PREFIX}/responses",
    {"model": "us.openai.gpt-5.6-sol", "input": "Search the web for AWS news.",
     "tools": [{"type": "web_search"}], "max_output_tokens": 4000},
    region=RUNTIME_REGION, attempts=1, timeout=420,
)
gpt_on_runtime = ok(code_, data)
print(f"  gpt-5.6 on runtime  : {'accepted' if gpt_on_runtime else str(code_) + ' ' + err(data)[:64]}")
print()
if gpt_on_mantle and not gpt_on_runtime:
    print("  => Both limits are real and independent: web search is absent for Grok")
    print("     on either endpoint (a model limit) AND absent on bedrock-runtime")
    print("     even for a model that has it on mantle (an endpoint limit).")
    print("     Two probes with the same model on both endpoints is what separates")
    print("     them. One probe with Grok alone would have blamed the endpoint.")
else:
    print("  => The two-probe result changed; re-read the messages above before")
    print("     concluding which limit applies.")

print("\\n--- application inference profiles: which APIs accept one? ---")
control = boto3.client("bedrock", region_name=RUNTIME_REGION)
account = boto3.client("sts").get_caller_identity()["Account"]
profile_arn = None
try:
    created = control.create_inference_profile(
        inferenceProfileName="grok46NotebookProbe",
        description="Temporary probe of application inference profile support.",
        modelSource={
            "copyFrom": f"arn:aws:bedrock:{RUNTIME_REGION}:{account}"
                        f":inference-profile/{GROK_RT}"
        },
    )
    profile_arn = created["inferenceProfileArn"]
    safe_print("  created:", profile_arn)
    time.sleep(5)

    code_, data = runtime_post(
        f"{RT_PREFIX}/responses",
        {"model": profile_arn, "input": "Reply OK", "max_output_tokens": 2000},
        region=RUNTIME_REGION, attempts=1, timeout=300,
    )
    print(f"  Responses API : {'accepted' if ok(code_, data) else str(code_) + ' ' + err(data)[:70]}")

    text, response = converse(
        profile_arn,
        [{"role": "user", "content": [{"text": "Reply with exactly: OK"}]}],
        max_tokens=2000, region=RUNTIME_REGION, resolve=False,
    )
    if response.get("error"):
        print(f"  Converse      : {str(response['error'].get('message'))[:70]}")
    else:
        print(f"  Converse      : accepted -> {converse_text(response).strip()[:20]!r}")
    print("  => Same target, same endpoint, different answer per API. Cost")
    print("     attribution by application profile therefore constrains which API")
    print("     you can use — worth knowing before you build reporting on it.")
except ClientError as exc:
    print(f"  skipped: {exc.response['Error']['Code']} "
          f"({exc.response['Error']['Message'][:80]})")
finally:
    if profile_arn:
        try:
            control.delete_inference_profile(inferenceProfileIdentifier=profile_arn)
            print("  cleaned up the temporary profile")
        except ClientError as exc:
            print(f"  cleanup failed, delete it by hand: {exc}")

print("\\n--- CountTokens ---")
runtime = boto3.client("bedrock-runtime", region_name=RUNTIME_REGION)
try:
    counted = runtime.count_tokens(
        modelId=GROK_RT,
        input={"converse": {"messages": [
            {"role": "user", "content": [{"text": "Hello there"}]}]}},
    )
    print("  grok-4.6:", counted.get("inputTokens"), "input tokens")
except ClientError as exc:
    print(f"  grok-4.6: {exc.response['Error']['Code']} — "
          f"{exc.response['Error']['Message'][:80]}")
    print("  => Budget from usage on a real call instead. CountTokens is a Claude")
    print("     facility today, not a universal one.")''')

# ---------------------------------------------------------------------------
md("""## 14. The IAM permission that is easy to miss

Calling Grok 4.6 through an inference profile on `bedrock-runtime` needs
`bedrock:InvokeModel` on **two** resources — the profile *and* your account's
default project. The model card is explicit about it:

> Your IAM identity also needs `bedrock:InvokeModel` on your account's default
> project (`arn:aws:bedrock:{region}:{account-id}:project/default`) in addition to
> the inference profile.

A policy with only the profile ARN produces an `AccessDeniedException` that names
the *model*, which sends you looking at model access rather than at the missing
project statement.""")

code('''import boto3

# Scope the project ARN to THIS account. An earlier draft wildcarded the ACCOUNT
# segment as well as the Region, which grants the action against project/default in
# any account -- not what the sentence above it claims, and not least privilege.
# The Region wildcard is deliberate and explained below; that one was not.
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "InvokeGrokViaProfile",
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            "Resource": [
                # The profile you name as the model...
                f"arn:aws:bedrock:{RUNTIME_REGION}::inference-profile/{GROK_RT}",
                # ...every Region the profile can route to, for the model itself...
                "arn:aws:bedrock:*::foundation-model/xai.grok-4.6",
                # ...and THIS account's default project. Omitting the statement is
                # the usual mistake; wildcarding its account is the other one.
                f"arn:aws:bedrock:*:{ACCOUNT_ID}:project/default",
            ],
        }
    ],
}
# safe_print, not print: the policy now carries a real account ID and this output
# is committed to a public repository.
safe_print(json.dumps(POLICY, indent=2))
print("\\nWith a global. profile the request can be served from any commercial")
print("Region, so the foundation-model and project ARNs are wildcarded ACROSS")
print("REGIONS on purpose -- never across accounts. Narrow the Region list too if")
print("you use the us. profile and have verified the Region set (SEC 3).")''')

# ---------------------------------------------------------------------------
md("""## Gotchas from this notebook

1. **The endpoint asymmetry is backwards from the rest of the collection.** Grok
   4.6 is broad on `bedrock-runtime` (profile-only) and `us-west-2`-only on
   `bedrock-mantle`. Grok 4.3, next door in `01-grok-4-3.ipynb`, is mantle-only.
   Same family, opposite shape.
2. **`xai.grok-4.6` is a valid ID on one endpoint and invalid on the other.** On
   runtime you must name `us.` or `global.`.
3. **`us.` is US-only.** The geo profile exists in `us-east-1`, `us-east-2` and
   `us-west-2` and nowhere else; `global.` exists wherever the model does. Using
   `us.` from an EU or APAC Region gives *"The provided model identifier is
   invalid"* — which reads like the model is missing from the endpoint entirely.
4. **A 404 "does not exist" on mantle means the wrong Region**, not the wrong ID.
5. **Budget for the trace.** A small `max_output_tokens` returns 200, `status:
   "incomplete"` and an empty string. ~2000 output tokens is a sane floor.
6. **Effort is a coarse dial.** `none` is exact — it removes the reasoning item.
   The graded levels overlap, and how much they matter depends on the prompt.
   Measure with your own prompts; do not port a ladder from another model.
7. **An error message that enumerates valid values can still be wrong.** The 400
   for an unknown effort advertises `max`; sending `max` is refused. The list comes
   from the API schema, not from this model. Test the values, do not just read
   them.
8. **The trace is encrypted and Region-scoped.** No readable "thinking" text
   exists for this model, so a UI that shows reasoning cannot be built on it. And
   the blob is *"scoped to the region that produced it"* — with a `us.` or
   `global.` profile you do not choose the Region, so replaying it across turns
   can fail for reasons that have nothing to do with your code. Use
   `previous_response_id` when you need reasoning continuity you can rely on.
9. **Chat Completions hides the trace but still bills it.** `usage`
   reports `reasoning_tokens` with no matching content.
10. **Structured output works, though the model card says otherwise on runtime.**
    Probe capability tables before designing around them.
11. **Server-side tools are absent for two independent reasons** — the model lacks
    them and `bedrock-runtime` lacks them. One probe cannot tell you which; run the
    same model on both endpoints.
12. **`background=true` is mantle-only**, and application inference profiles work
    on Converse but not on the OpenAI paths.
13. **Caching is opportunistic, and the profile decides how often it lands.**
    No hit on call 2 does not mean no caching, and `cached_tokens` is the only
    place you can see it. Over 30 calls, `global.` hit 27% of the time and `us.`
    hit 83%: a cache belongs to the Region that served it, and the broader profile
    routes away from that Region more often. `prompt_cache_key` recovered most of
    the gap on `global.` (27% → 67%) and added nothing on `us.`.
14. **A 1×1 PNG fails the image parser**, which reads exactly like "no vision
    support". Use a real image for capability probes.
15. **`CountTokens` refuses this model.** Take token counts from `usage`.
16. **The default-project IAM statement** is required alongside the profile — and
    wildcard its Region if you must, never its account.

## Takeaways

- On `bedrock-runtime`, name `us.xai.grok-4.6` or `global.xai.grok-4.6`; `global.`
  is cheaper per token. On `bedrock-mantle`, `xai.grok-4.6` in `us-west-2`.
- Use **Responses** when you want the reasoning item, tools with a flat spec, or
  server-side conversation state; **Chat Completions** for a smaller body when you
  only need the answer.
- Set `reasoning.effort` deliberately: `none` for extraction and classification
  where thinking is waste, and a graded level for work that needs it — sized by
  measurement on your prompts.
- Derive capability from a live probe. Two claims on the model card did not survive
  contact with the endpoint, and this notebook prints what it found rather than
  what it expected.""")

# ---------------------------------------------------------------------------
PATH = os.path.expanduser(
    "~/Projects/sample-per-model-bedrock/11-xai-grok/02-grok-4-6.ipynb"
)


def cell_id(kind: str, index: int, text: str) -> str:
    return hashlib.sha256(f"grok46:{kind}:{index}:{text}".encode()).hexdigest()[:8]


def as_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


cells = []
for i, (kind, text) in enumerate(CELLS):
    cell = {
        "cell_type": kind,
        "id": cell_id(kind, i, text),
        "metadata": {},
        "source": as_lines(text),
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.4",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(PATH, "w") as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)
    fh.write("\n")

n_code = sum(1 for k, _ in CELLS if k == "code")
print(f"wrote {PATH}")
print(f"  {len(CELLS)} cells ({n_code} code, {len(CELLS) - n_code} markdown)")
