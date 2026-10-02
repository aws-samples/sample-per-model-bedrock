#!/usr/bin/env python3
"""Audit fix 2 — Gemma 4 and Grok parameter sections.

Both notebooks asserted a parameter surface the service no longer has, and both
printed a hardcoded verdict next to a live probe, so re-running produced prose
contradicting the table directly above it.

The fix is structural, not a wording swap: every verdict is now *computed from the
collected results*. If the surface changes again the output changes with it, and
nothing in the notebook can be left asserting yesterday's answer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


# ---------------------------------------------------------------- Gemma 4 ---

GEMMA_S2_MD = """## 2. Sampling parameters — probe them, every time

Gemma 4's sampling surface is the clearest example in this collection of why these
notebooks probe instead of assert. It has changed twice:

- Until **12 August 2026** it accepted `max_tokens`, the full `temperature` range
  and `top_p`.
- On 12 August it tightened to current OpenAI semantics — `max_tokens` refused in
  favour of `max_completion_tokens`, `temperature` pinned to its default, `top_p`
  rejected. That broke working code overnight, with no release note.
- It has since been **relaxed again**.

So the useful thing here is not a table of accepted values — any such table is a
snapshot that may already be wrong by the time you read it. The useful thing is the
sweep below, which asks the endpoint and derives its own conclusion.

Two facts about this model *are* durable and worth keeping in mind whichever way
the surface is set today:

- `max_output_tokens` has a **minimum of 16** on the Responses API (§2b).
- Greedy decoding at `temperature=0` sends Gemma 4 into repetition loops involving
  reserved vocabulary tokens, so even when `0.0` is accepted it is a poor choice.
  Google's and AWS's guidance is to leave `temperature` at its default of `1.0`.

Sweep the values and read what comes back:"""

GEMMA_S2_CODE = '''print(f"{'temperature':>12} {'HTTP':>6}  detail")
print("-" * 74)
accepted, refused = [], []
for value in (0.0, 0.2, 0.5, 0.7, 0.9, 1.0, 1.5):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": DENSE,
            "input": "Reply with exactly: OK",
            "max_output_tokens": 16,
            "temperature": value,
        },
        region=REGION,
    )
    (accepted if code == 200 else refused).append(value)
    print(f"{value:>12} {code:>6}  {'' if code == 200 else err(data)[:52]}")

code, data = post(
    f"{PREFIX}/responses",
    {"model": DENSE, "input": "Reply OK", "max_output_tokens": 16, "top_p": 0.95},
    region=REGION,
)
top_p_ok = code == 200
print(f"\\n  top_p=0.95        -> HTTP {code} {'' if top_p_ok else err(data)[:70]}")

# The verdict is DERIVED, not written down. A hardcoded conclusion here is how the
# earlier version of this notebook ended up asserting 400s above a table of 200s.
print("\\n--- what this run actually shows ---")
if len(accepted) == 1:
    print(f"  temperature: only {accepted[0]} accepted -> you cannot tune it on this API")
elif refused:
    print(f"  temperature: accepted {accepted}, refused {refused}")
else:
    print(f"  temperature: every value tried was accepted {accepted}")
print(f"  top_p      : {'accepted' if top_p_ok else 'rejected'}")
if not refused and top_p_ok:
    print("  => sampling is fully tunable on Responses for this model today.")
else:
    print("  => sampling is restricted today; omit what is refused rather than")
    print("     hardcoding a value, because this surface has moved before.")'''

GEMMA_S2B_MD = """Read the derived verdict, not the paragraph above it — that is the whole point of
keeping the probe in the notebook rather than only its result.

Whatever the surface allows today, **omitting both parameters** is the portable
choice: it works on every mantle model, and it cannot break when the accepted set
changes."""

GEMMA_CC_CODE = '''# The parameter surface on Chat Completions is NOT necessarily the same as on
# Responses, and it is not stable over time: Gemma 4 tightened on 12 Aug 2026 and
# has since been relaxed. So probe it rather than trusting a table -- including
# the tables in this notebook.
probes = [
    ("max_tokens=16", {"max_tokens": 16}),
    ("max_completion_tokens=16", {"max_completion_tokens": 16}),
    ("+ temperature=0.2", {"max_completion_tokens": 16, "temperature": 0.2}),
    ("+ temperature=1.0", {"max_completion_tokens": 16, "temperature": 1.0}),
    ("+ top_p=0.95", {"max_completion_tokens": 16, "top_p": 0.95}),
]
print(f"{'parameters':<26} {'HTTP':<5} detail")
print("-" * 76)
cc = {}
for label, extra in probes:
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": DENSE,
            "messages": [{"role": "user", "content": "Reply OK"}],
            **extra,
        },
        region=REGION,
    )
    cc[label] = code
    print(f"{label:<26} {code:<5} {'' if code == 200 else err(data)[:42]}")

# Derive the summary from `cc` so it can never contradict the table above.
print()
budget = [k for k in ("max_tokens=16", "max_completion_tokens=16") if cc[k] == 200]
print(f"=> token budget accepted as: {', '.join(b.split('=')[0] for b in budget)}")
sampling_ok = [k for k in probes[2:] if cc[k[0]] == 200]
if len(sampling_ok) == 3:
    print("   temperature and top_p are both accepted here.")
else:
    refused = [k[0].lstrip('+ ') for k in probes[2:] if cc[k[0]] != 200]
    print(f"   refused: {', '.join(refused)}")
print("   Compare with section 2: the two APIs do not have to agree, and neither")
print("   set is stable, so gate sampling per model AND re-probe periodically.")'''


def fix_gemma_01() -> None:
    p = f"{REPO}/03-google-gemma/01-gemma4-end-to-end.ipynb"
    nb = Notebook(p)

    nb.set_source(5, GEMMA_S2_MD)
    nb.set_source(6, GEMMA_S2_CODE)
    nb.set_source(7, GEMMA_S2B_MD)
    nb.set_source(8, GEMMA_CC_CODE)

    # §3 reasoning: the e2b advice was measured with reasoning_tokens, which this
    # model always reports as 0. Measure the trace itself.
    nb.sub(
        '''for model in (COMPACT, DENSE):
    r = client.responses.create(
        model=model,
        input="If 3 shirts dry in 4 hours, how long for 9 shirts on the same line?",
        reasoning={"effort": "high"},
        max_output_tokens=600,
    )
    reasoning_tokens = r.usage.output_tokens_details.reasoning_tokens
    print(
        f"{model:26} reasoning_tokens={reasoning_tokens:5}  "
        f"answer={r.output_text[:90]!r}"
    )''',
        '''# Measure the TRACE, not the token counter. Gemma 4 reports
# output_tokens_details.reasoning_tokens as 0 even when it returns a full
# reasoning item, so counting tokens here shows nothing and reads as "it did not
# think". Count the characters the model actually returned.
for model in (COMPACT, DENSE):
    r = client.responses.create(
        model=model,
        input="If 3 shirts dry in 4 hours, how long for 9 shirts on the same line?",
        reasoning={"effort": "high"},
        max_output_tokens=1200,
    )
    trace = "".join(
        getattr(block, "text", "") or ""
        for item in r.output
        if item.type == "reasoning"
        for block in (item.content or [])
    )
    print(
        f"{model:26} trace={len(trace):5} chars  "
        f"reasoning_tokens={r.usage.output_tokens_details.reasoning_tokens} "
        f"(never itemised)"
    )
    print(f"{'':26} answer={r.output_text[:80]!r}")''')

    # §4 streaming: 400 tokens truncated the answer mid-word, and low effort emits
    # no reasoning deltas, so the "separate event types" claim went undemonstrated.
    nb.sub('''stream = client.responses.create(
    model=DENSE,
    input="List three properties of a good distributed queue.",
    reasoning={"effort": "low"},
    max_output_tokens=400,
    stream=True,
)''',
           '''# effort="high" and a generous budget on purpose: at low effort this model emits
# no reasoning deltas at all, and 400 tokens truncates the answer mid-word -- so
# the earlier version of this cell demonstrated neither of the two event types it
# claims to separate.
stream = client.responses.create(
    model=DENSE,
    input="List three properties of a good distributed queue, one line each.",
    reasoning={"effort": "high"},
    max_output_tokens=2000,
    stream=True,
)''')
    nb.sub('''print("\\n\\n--- event types seen ---")
for name, count in sorted(event_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4}  {name}")''',
           '''print("\\n\\n--- event types seen ---")
for name, count in sorted(event_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4}  {name}")

# Derived, so it cannot claim a separation the run did not show.
reasoning_deltas = event_counts.get("response.reasoning_text.delta", 0)
text_deltas = event_counts.get("response.output_text.delta", 0)
print(f"\\nreasoning deltas={reasoning_deltas}  text deltas={text_deltas}")
if reasoning_deltas and text_deltas:
    print("=> two distinct event types, so a 'thinking' panel can be rendered")
    print("   separately from the answer.")
elif text_deltas:
    print("=> only text deltas this run. The reasoning channel is not guaranteed:")
    print("   it depends on effort and on the prompt, so handle its absence.")
if event_counts.get("response.incomplete"):
    print("=> response.incomplete: the budget ran out. Raise max_output_tokens.")''')

    # §7's "no reasoning content on Chat Completions" is false at high effort.
    nb.sub("""Two differences worth seeing side by side: `messages` instead of `input`,
`max_completion_tokens` instead of `max_output_tokens` — and **no reasoning
content**.

> **This surface changed on 12 Aug 2026.** Gemma 4 previously accepted
> `max_tokens` here, along with a full `temperature` range and `top_p`. It now
> follows current OpenAI semantics: `max_tokens` is rejected in favour of
> `max_completion_tokens`, `temperature` is pinned to its default, and `top_p`
> is unsupported. The probe in section 3 asks the live endpoint, so you see
> today's answer rather than the one that was true when this was written. Treat
> every parameter table in every sample — including these — as a snapshot.""",
           """Two differences worth seeing side by side: `messages` instead of `input`, and
`max_completion_tokens` accepted alongside `max_output_tokens`.

**The reasoning trace is not simply absent here.** At `reasoning_effort="high"`
Gemma 4 returns it in a non-standard `choices[0].message.reasoning` field — a
sibling of `content`, not part of it, and not in the OpenAI specification. At
`none` and `low` the key is absent entirely. `output_tokens_details.reasoning_tokens`
is `0` in every case, so the trace is readable but never itemised: you cannot
separate thinking from answering in the usage figures. The same non-standard field
carries Qwen's and DeepSeek's traces, so treat it as a mantle convention rather
than a Gemma quirk (`../14-openai-gpt-oss/01` covers it for gpt-oss).

> **This surface has changed twice.** Gemma 4 accepted `max_tokens`, a full
> `temperature` range and `top_p` until 12 August 2026; it then tightened to
> current OpenAI semantics with no release note; it has since been relaxed again.
> The probes in sections 2 and 2b ask the live endpoint and derive their own
> verdicts, so you see today's answer. Treat every parameter table in every sample
> — including these — as a snapshot.""")

    # The Chat-Completions reasoning cell asserted the opposite of its own output.
    nb.sub('''message = data.get("choices", [{}])[0].get("message", {}) or {}
details = (data.get("usage", {}) or {}).get("completion_tokens_details") or {}
print(f"HTTP {code} | keys in message: {sorted(message.keys())}")
print("reasoning tokens billed:", details.get("reasoning_tokens"))
print("answer:", (message.get("content") or "")[:80])''',
           '''message = data.get("choices", [{}])[0].get("message", {}) or {}
details = (data.get("usage", {}) or {}).get("completion_tokens_details") or {}
trace = message.get("reasoning") or ""
print(f"HTTP {code} | keys in message: {sorted(message.keys())}")
print(f"message.reasoning     : {len(trace)} chars  <- NOT in the OpenAI schema")
print("reasoning tokens billed:", details.get("reasoning_tokens"), "(never itemised)")
print("answer:", (message.get("content") or "")[:80])
if trace:
    print("\\n=> The trace IS returned here, in a non-standard field. Code written")
    print("   against the published OpenAI schema reads .content only and discards")
    print("   it. Read .reasoning explicitly if you want it.")
else:
    print("\\n=> No trace at this effort level. Try reasoning_effort='high'.")''')

    # ...and the tools verdict, now false, was hardcoded.
    nb.sub('''# Function tools are the one place reasoning must be turned OFF explicitly.
TOOL = [''',
           '''# Whether function tools coexist with reasoning is a per-model, per-date fact:
# Gemma 4 refused the combination in August 2026 and no longer does. Probe both.
TOOL = [''')
    nb.sub('''for label, extra in [
    ("tools alone", {}),
    ('tools + reasoning_effort "none"', {"reasoning_effort": "none"}),
]:''',
           '''tool_results = {}
for label, extra in [
    ("tools alone", {}),
    ('tools + reasoning_effort "none"', {"reasoning_effort": "none"}),
    ('tools + reasoning_effort "high"', {"reasoning_effort": "high"}),
]:''')
    nb.sub('''    choice = data.get("choices", [{}])[0].get("message", {}) or {}
    calls = choice.get("tool_calls") or []
    detail = "" if code == 200 else err(data)[:46]
    print(f"  {label:<32} HTTP {code} tool_calls={len(calls)} {detail}")

print()
print('=> Gemma 4 refuses function tools unless reasoning_effort is "none".')''',
           '''    choice = data.get("choices", [{}])[0].get("message", {}) or {}
    calls = choice.get("tool_calls") or []
    detail = "" if code == 200 else err(data)[:46]
    tool_results[label] = (code, len(calls))
    print(f"  {label:<36} HTTP {code} tool_calls={len(calls)} {detail}")

# Derived verdict. Hardcoding this sentence is exactly how the previous version of
# the notebook came to assert a 400 that the service had stopped returning.
print()
refused = [k for k, (c, _) in tool_results.items() if c != 200]
if not refused:
    print("=> Tools work at every reasoning effort on this model today.")
elif refused == ["tools alone"]:
    print('=> Tools require reasoning_effort to be set explicitly on this model.')
else:
    print(f"=> Refused combinations today: {refused}")
    print('   The message names the fix when it is a reasoning conflict.')''')

    # The client's stale comment.
    nb.sub('''            "temperature": 1.0,  # the ONLY value Responses accepts
            # top_p deliberately omitted: rejected on the Responses API''',
           '''            # temperature and top_p are deliberately omitted rather than set:
            # which values this model accepts has changed twice (see section 2),
            # and omitting them is the one choice that cannot break.''')

    # Gotcha table.
    nb.sub("""| `temperature` | On Responses, **only `1.0` is accepted** — every other value 400s |
| `top_p` | **400 on Responses**; accepted on Chat Completions |
| Tuning sampling | Not possible on Responses — switch to Chat Completions |""",
           """| Sampling surface **moves** | Tightened 12 Aug 2026, relaxed since. §2 probes it and derives the verdict; omitting `temperature`/`top_p` is the choice that cannot break |
| `temperature=0` | Even where accepted, it drives repetition loops over reserved tokens — leave the default |""")
    nb.sub("| Reasoning visibility | Returned on Responses only; billed-but-hidden on Chat Completions |",
           "| Reasoning visibility | Responses returns a `reasoning` item; Chat Completions returns `message.reasoning` at `effort=\"high\"` only — **not** in the OpenAI schema |\n"
           "| `reasoning_tokens` | Always `0` for this model, on both APIs. The trace is readable but never itemised |")
    nb.sub("| Parallel tool calls | Unsupported; model silently issues one |",
           "| Parallel tool calls | Unsupported; model silently issues one |\n"
           "| Tools + reasoning | The combination was refused in Aug 2026 and is not now — §7 probes it |")
    nb.sub("| e2b reasoning | Set `effort=\"high\"` to stop thinking leaking into the answer |",
           "| e2b reasoning | Set `effort=\"high\"` to keep thinking in the reasoning item rather than the answer |")
    nb.save()
    print(f"03-google-gemma/01: {nb.changes} edits")


# ---------------------------------------------------------------- Gemma 3 ---

def fix_gemma_02() -> None:
    nb = Notebook(f"{REPO}/03-google-gemma/02-gemma3-on-both-endpoints.ipynb")

    nb.sub("""| Token budget | `max_tokens` *or* `max_completion_tokens` | `max_completion_tokens` only |
| `temperature` | free | pinned to the default (1) |
| `top_p` | accepted | rejected |""",
           """| Token budget | `max_tokens` *or* `max_completion_tokens` | both today; `max_completion_tokens` only for part of Aug 2026 |
| Sampling | stable and free | **has changed twice** — §1 probes it |""")

    nb.sub('print("Gemma 3 accepts every one of them. Gemma 4 rejects three of the four.")',
           '''# Derived: Gemma 4's answer here changed in Aug 2026 and changed back, so a
# hardcoded sentence would contradict the table above it on some future run.
print("--- what this run shows ---")
for model, row in verdicts.items():
    refused = [label for label, code in row.items() if code != 200]
    if refused:
        print(f"  {model:24} refused: {', '.join(refused)}")
    else:
        print(f"  {model:24} accepted all {len(row)} probes")
print("\\nWhere the two generations differ, that difference is the reason this")
print("collection has a notebook per generation rather than per provider.")''')

    # Collect the codes the derived summary needs.
    nb.sub('''for model in (LARGE, GEMMA4):
    print(f"{model}   (prefix {api_prefix(model)})")
    for label, extra in PROBES:''',
           '''verdicts = {}
for model in (LARGE, GEMMA4):
    print(f"{model}   (prefix {api_prefix(model)})")
    verdicts[model] = {}
    for label, extra in PROBES:''')
    nb.sub('''        detail = "" if status == 200 else err(data)[:56]
        print(f"    {label:<24} HTTP {status} {detail}")
    print()''',
           '''        detail = "" if status == 200 else err(data)[:56]
        verdicts[model][label] = status
        print(f"    {label:<24} HTTP {status} {detail}")
    print()''')

    nb.sub("""- **Check the family, not the provider.** "Google models on Bedrock" is not a
  useful unit. Gemma 3 and Gemma 4 disagree on the endpoint, the path prefix, the
  token-budget parameter name, and whether `temperature` does anything.""",
           """- **Check the family, not the provider.** "Google models on Bedrock" is not a
  useful unit. Gemma 3 and Gemma 4 disagree on the endpoint, the path prefix, and
  which APIs they serve — and Gemma 4's parameter surface has moved twice while
  Gemma 3's has not.""")

    nb.sub("""- **Re-run section 1 before you trust any of this.** Gemma 4's parameter surface
  changed on 12 August 2026 without a release note. Yours may differ from the
  output committed here, and that is the point of keeping the probe in the
  notebook rather than only its result.""",
           """- **Re-run section 1 before you trust any of this.** Gemma 4's parameter surface
  tightened on 12 August 2026 without a release note, and was relaxed again
  afterwards — so the committed output here has been wrong twice. That is the point
  of keeping the probe, and of deriving the summary from its results instead of
  writing the conclusion down.""")
    nb.save()
    print(f"03-google-gemma/02: {nb.changes} edits")


# ------------------------------------------------------------------- Grok ---

GROK_S3_MD = """## 3. Sampling — probe it, because this surface has moved

Grok's model card documents non-standard defaults (`temperature=0.7`,
`top_p=0.95`), and for a period on `bedrock-mantle` the Responses API accepted
`temperature` **only** at that documented default and refused every other value.
That is no longer the case.

Rather than print today's answer as a rule, sweep both parameters on both models
and let the cell say what it found:"""

GROK_S3B_MD = """### Read the derived verdict above, not a remembered rule

There was a real pattern here once — each model accepting `temperature` only at
its own documented default, which is why Grok and Gemma 4 disagreed in opposite
directions. It has not survived, and a notebook that wrote it down as a rule would
now be teaching a 400 that no longer happens.

What is durable:

- **Sampling support is per model and per date.** Gate it, and re-probe.
- **Omitting `temperature` and `top_p` always works.** It is the portable default
  and it cannot break when the accepted set changes.
- The **GPT-5.5 and GPT-5.6 families** are the ones that still refuse both today
  (`temperature` at `1.0` only, `top_p` outright), and newer Claude models reject
  both as deprecated. See `../99-cross-cutting/01` for the live matrix."""


def fix_grok() -> None:
    nb = Notebook(f"{REPO}/11-xai-grok/01-grok-4-3.ipynb")

    nb.sub("""2. Grok **rejects `temperature`** at any value but **accepts `top_p`** — the
   inverse of Gemma 4. Its model card also documents non-standard defaults:
   `temperature=0.7`, `top_p=0.95`, `max_completion_tokens=131072`.""",
           """2. Its model card documents **non-standard defaults** — `temperature=0.7`,
   `top_p=0.95`, `max_completion_tokens=131072`. For part of 2026 the Responses
   API accepted `temperature` only at that default; it no longer restricts it.
   §3 probes both parameters and derives the verdict rather than asserting one.""")

    nb.set_source(10, GROK_S3_MD)
    nb.set_source(12, GROK_S3B_MD)

    # The sweep printed a fixed conclusion under a table of 200s.
    nb.sub('''# The exact error, so it is searchable.
code, data = post(
    f"{PREFIX}/responses",
    {"model": GROK, "input": "Reply OK", "max_output_tokens": 16, "temperature": 0.7},
    region=REGION,
)
# Sweep the values: only Grok's own documented default (0.7) is accepted.
print(f"{'temperature':>12} {'HTTP':>6}  detail")
print("-" * 70)
for value in (0.0, 0.5, 0.7, 1.0, 1.5):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GROK,
            "input": "Reply OK",
            "max_output_tokens": 16,
            "temperature": value,
        },
        region=REGION,
    )
    print(f"{value:>12} {code:>6}  {'' if code == 200 else err(data)[:48]}")
print("\\n=> 0.7 is Grok's documented default. Safest: omit temperature entirely.")''',
           '''# Sweep the whole range and DERIVE the conclusion. The previous version of this
# cell printed "only 0.7 is accepted" beneath a table showing every value at 200.
print(f"{'temperature':>12} {'HTTP':>6}  detail")
print("-" * 70)
accepted, refused = [], []
for value in (0.0, 0.5, 0.7, 1.0, 1.5):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GROK,
            "input": "Reply OK",
            "max_output_tokens": 16,
            "temperature": value,
        },
        region=REGION,
    )
    (accepted if code == 200 else refused).append(value)
    print(f"{value:>12} {code:>6}  {'' if code == 200 else err(data)[:48]}")

print()
if not refused:
    print(f"=> every value accepted {accepted}: temperature is unrestricted here today.")
elif accepted == [0.7]:
    print("=> only 0.7 (Grok's documented default) accepted — you are echoing the")
    print("   default back, not tuning anything.")
else:
    print(f"=> accepted {accepted}, refused {refused}")
print("   Either way, omitting temperature is the portable choice.")''')

    # The API/prefix probe used a 30s timeout, which reported Responses as
    # "stalled" on the very prefix the rest of the notebook uses successfully.
    nb.sub('''# Use a short timeout and a single attempt: one of these combinations does not
# return a clean 400 but simply STALLS (see the note below), so a naive probe with
# retries can hang for many minutes.''',
           '''# One of these combinations does not return a clean 400 but simply STALLS (see the
# note below), so keep attempts=1 and always set a timeout. The timeout has to be
# generous enough for a real answer, though: at 30s this probe reported the
# WORKING prefix as "stalled", because a reasoning-first model can spend that long
# before its first token.''')
    nb.sub('''        code, data = post(
            prefix + suffix, body, region=REGION, attempts=1, timeout=30
        )  # no retries, short timeout''',
           '''        code, data = post(
            prefix + suffix, body, region=REGION, attempts=1, timeout=120
        )  # no retries; bounded, but long enough for a reasoning-first model''')
    nb.sub('''1. Grok is served on **`/openai/v1`** for both APIs. The bare `/v1` does not work.
2. `/v1/chat/completions` returns a clean, fast **400** ("isn't supported on this
   route"), but **`/v1/responses` does not respond at all — it stalls**. A wrong
   path is therefore not always a fast failure.''',
           '''1. Grok is served on **`/openai/v1`** for both APIs. The bare `/v1` does not work.
2. `/v1/chat/completions` returns a clean, fast **400** ("isn't supported on this
   route"), but **`/v1/responses` does not respond at all — it stalls**. A wrong
   path is therefore not always a fast failure.
3. The timeout you choose decides what you *conclude*. Too short and a slow but
   healthy call is indistinguishable from a stall — which is how the earlier
   version of this table reported the working prefix as stalled.''')

    # "At 400 tokens you get nothing back" was contradicted by the sweep.
    nb.sub('''At 400 tokens the whole budget goes to reasoning and you get nothing back. The
practical rules:''',
           '''Read the `chars` column. Where it is `0`, reasoning consumed the entire budget and
the call returned HTTP 200 with an empty string; the exact budget at which that
happens moves run to run, because how long the model thinks is not fixed. §7's
sweep (`max_output_tokens=400`) has returned `0` chars; this one sometimes returns
a short answer at the same budget. That variability *is* the lesson — you cannot
pick a budget once and assume it is safe. The practical rules:''')
    nb.sub('''- **Always check `status`** (or that the text is non-empty) before using a result.
- Budget **≥1000** output tokens for anything you expect an answer from.''',
           '''- **Always check `status`** (or that the text is non-empty) before using a result.
  Never infer success from HTTP 200 on this model.
- Budget **≥1000** output tokens for anything you expect an answer from, and treat
  an empty answer as a retry-with-more-budget path rather than an error.''')

    # Effort table: tokens rise with effort, wall-clock does not.
    nb.sub('''print("\\nHigher effort spends more reasoning tokens and more wall-clock time.")
print("'high' is omitted here: it can exceed 15 minutes for one call under load.")''',
           '''print()
print("Reasoning tokens rise with effort. Wall-clock time does NOT track it")
print("reliably - queue time dominates, so a 'medium' call can finish before a")
print("'low' one. Read the token column for cost and treat latency as noisy.")
print("'high' is omitted here: it can exceed 15 minutes for one call under load.")''')

    # Tier table prose.
    nb.sub('''Tiers govern queue priority and mostly separate under contention, so single
samples on an idle account look flat. See `../00-foundations/03`.''',
           '''Read this as one sample, not a benchmark. `flex` is deliberately deprioritised, so
its TTFT is usually the worst of the three; `default` and `priority` are close on an
idle account and separate under contention. A single `priority` call has also taken
**54 seconds** here while `default` took 4 — queue placement is not a guarantee.
Measure over many calls before quoting a number. See `../00-foundations/03`.''')

    # Gotcha table.
    nb.sub("""| **`temperature`** | Accepted **only at Grok's own default `0.7`** — other values 400 |
| Per-model defaults | Each model accepts `temperature` only at *its* documented default (Grok 0.7, Gemma 4 1.0) |""",
           """| Sampling surface **moves** | `temperature` was once accepted only at Grok's documented `0.7`; §3 probes it and derives the verdict. Omitting it always works |
| Documented defaults ≠ constraints | The model card's `temperature=0.7` is a default, not necessarily the only accepted value |""")
    nb.sub("| Region | Absent from eu-central-1 |",
           "| Region | Absent from eu-central-1 |\n"
           "| Probe timeouts | Too short a timeout turns a slow healthy call into a false \"stall\" (§2) |")
    nb.save()
    print(f"11-xai-grok/01: {nb.changes} edits")


if __name__ == "__main__":
    fix_gemma_01()
    fix_gemma_02()
    fix_grok()
