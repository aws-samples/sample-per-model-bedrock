#!/usr/bin/env python3
"""Audit fix 3 — three defects shared by the Chat-Completions family notebooks.

E1. "the trace is never returned" is false for deepseek, glm, kimi, minimax and
    nemotron-super. Two other notebooks in the same repo (04-qwen/01 §6b and
    14-openai-gpt-oss/01) already document `choices[0].message.reasoning`
    correctly. Replaced with a probe that reports what came back.

E2. The effort table printed completion_tokens against a max_tokens=400 cap, so it
    was never monotonic and minimax's was all-400s. Now measures the trace itself
    at a budget large enough that the cap is not the answer.

E3. The tier prose claimed single samples "look flat"; in eight of nine notebooks
    flex is 2-3x worse than default, which is the documented behaviour. And the
    error branch labelled every failure "tier not supported", including a URLError.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

FAMILY = [
    "04-qwen/01-qwen3-core-and-tools.ipynb",
    "05-deepseek/01-deepseek-v3-reasoning.ipynb",
    "06-zai-glm/01-glm-family.ipynb",
    "08-moonshot-kimi/01-kimi-k2.ipynb",
    "09-minimax/01-minimax-m2.ipynb",
    "10-nvidia-nemotron/01-nemotron-nano-and-super.ipynb",
    "12-writer-palmyra/01-palmyra-vision.ipynb",
]

S6_OLD = """`reasoning_effort` is accepted. The trace is not returned — but the token count
moves, which is how you can tell the model really is thinking harder."""

S6_NEW = """`reasoning_effort` is accepted. Two things about what comes back are worth getting
right, because the obvious reading of both is wrong.

**Where the trace goes.** On `bedrock-mantle` several models in the `/v1` families
return the reasoning in **`choices[0].message.reasoning`** — a sibling of
`content`, not part of it, and **not in the OpenAI specification**. SDK type hints
and any code written against the published schema will not surface it, so you can
pay for reasoning and discard it without noticing. Other models return nothing
there at all. It is a per-model fact, so the cell below asks rather than asserts.
(`../14-openai-gpt-oss/01` covers the same field for gpt-oss.)

**Whether effort changes spend.** Not measurably from one sample per level. Queue
time and the model's own choices dominate, so the token counts below will often
*not* rank in effort order. Measure repeatedly before drawing a cost conclusion."""

EFFORT_HEADER_OLD = """print(f"{'effort':10} {'status':>7} {'completion tokens':>18}")
print("-" * 38)"""

EFFORT_HEADER_NEW = """print(f"{'effort':10} {'HTTP':>5} {'completion tok':>15} {'trace chars':>12}")
print("-" * 46)
efforts = {}"""

EFFORT_BUDGET_OLD = """            "max_tokens": 400,
            "reasoning_effort": effort,"""

EFFORT_BUDGET_NEW = """            # Budget well above the likely answer length. At 400 the cap itself
            # became the measurement: every row read 400 for some models, which
            # tells you nothing about effort.
            "max_tokens": 2000,
            "reasoning_effort": effort,"""

EFFORT_TAIL_OLD = """    tokens = (data.get("usage") or {}).get("completion_tokens", "-")
    print(f"  {effort:8} {code:>7} {tokens!s:>18}")"""

EFFORT_TAIL_NEW = '''    usage = data.get("usage") or {}
    message = (data.get("choices") or [{}])[0].get("message") or {}
    # The non-standard field. `.get()` rather than indexing: plenty of models in
    # this family do not return it at all.
    trace = message.get("reasoning") or ""
    tokens = usage.get("completion_tokens", "-")
    efforts[effort] = (tokens, len(trace))
    print(f"  {effort:8} {code:>5} {tokens!s:>15} {len(trace):>12}")

# Both conclusions are DERIVED. The previous version of this cell asserted that the
# trace is never returned and that the token count rises with effort; for several
# models in this family both statements were false.
print()
if any(chars for _, chars in efforts.values()):
    print("=> The trace IS returned, in choices[0].message.reasoning -- a sibling of")
    print("   `content`, absent from the OpenAI schema. Read it explicitly or you")
    print("   are paying for tokens you never see.")
else:
    print("=> No trace in message.reasoning at any effort for this model: the")
    print("   thinking is billed and unreadable on this API.")
counts = [c for c, _ in efforts.values() if isinstance(c, int)]
if counts and counts != sorted(counts):
    print("=> The completion-token column is NOT in effort order. One sample per")
    print("   level cannot rank them -- do not read a cost curve off this table.")'''

TIER_MD_OLD = """TTFT is dominated by *prefill* (the model reading your prompt) plus queue time.
Service tiers trade cost against queue priority — they mostly separate under
contention, so single samples on an idle account look flat.
(`../00-foundations/03` has the full treatment.)"""

TIER_MD_NEW = """TTFT is dominated by *prefill* (the model reading your prompt) plus queue time.
Service tiers trade cost against queue priority.

Read the table below as **one sample, not a benchmark** — but do not expect the
three rows to look identical either. `flex` is deliberately deprioritised, so its
TTFT is usually the worst of the three by a clear margin; `default` and `priority`
sit close together on an idle account and separate under contention. A single
outlier in either direction is normal, and one `priority` call elsewhere in this
collection took 54 seconds. Take medians over many calls before quoting a number.
(`../00-foundations/03` has the full treatment.)"""

TIER_ERR_OLD = """    if m.get("error"):
        print(f"{tier:10} {m['error']:>32}  (tier not supported by this model)")"""

TIER_ERR_NEW = """    if m.get("error"):
        # Do NOT label every failure "tier not supported". A URLError or a timeout
        # is a transport problem and says nothing about whether the parameter is
        # accepted; reporting it as an unsupported feature invents a limitation
        # the service never claimed. Read the error before attributing a cause.
        detail = str(m["error"])
        if "service_tier" in detail or "unsupported" in detail.lower():
            cause = "tier refused by this model"
        else:
            cause = "transport error - retry; tells you nothing about tier support"
        print(f"{tier:10} {detail[:30]:>32}  ({cause})")"""

GOTCHA_OLD = "| Reasoning trace | `reasoning_effort` works but the trace is never returned |"
GOTCHA_NEW = ("| Reasoning trace | Per model: several models here return it in `choices[0].message.reasoning`, "
              "which is **not** in the OpenAI schema; others return nothing. §6 probes it |")


def main() -> None:
    for rel in FAMILY:
        nb = Notebook(f"{REPO}/{rel}")
        nb.sub_idempotent(S6_OLD, S6_NEW)
        nb.sub_idempotent(EFFORT_HEADER_OLD, EFFORT_HEADER_NEW)
        nb.sub_idempotent(EFFORT_BUDGET_OLD, EFFORT_BUDGET_NEW)
        nb.sub_idempotent(EFFORT_TAIL_OLD, EFFORT_TAIL_NEW)
        nb.sub_idempotent(TIER_MD_OLD, TIER_MD_NEW)
        nb.sub_idempotent(TIER_ERR_OLD, TIER_ERR_NEW)
        nb.sub_idempotent(GOTCHA_OLD, GOTCHA_NEW)
        nb.save()
        print(f"{rel}: {nb.changes} edits")

    # mistral/01 carries the gotcha row but has its own Magistral-specific §6 and a
    # bare "## 10. Latency across the ladder" heading with no prose under it.
    nb = Notebook(f"{REPO}/07-mistral/01-mistral-text-and-sizes.ipynb")
    nb.sub_idempotent(GOTCHA_OLD,
           "| Reasoning trace | Not returned for this family — `message.reasoning` is absent. "
           "Other `/v1` families (qwen, deepseek, glm, kimi, minimax, nemotron-super) **do** return it |")
    nb.sub_idempotent("## 10. Latency across the ladder",
           "## 10. Latency across the ladder\n\n" + TIER_MD_NEW)
    nb.sub_idempotent("""    if m.get("error"):
        print(f"{tier:10} {m['error']:>32}  (not supported by this model)")""",
           TIER_ERR_NEW)
    nb.save()
    print(f"07-mistral/01: {nb.changes} edits")

    # 07-mistral/02 and 04-qwen/02 share the error branch but have no tier table
    # prose; fix the mislabel wherever it appears.
    for rel in ("07-mistral/02-devstral-and-voxtral.ipynb",
                "04-qwen/02-qwen3-coder-and-vision.ipynb",
                "11-xai-grok/01-grok-4-3.ipynb"):
        nb = Notebook(f"{REPO}/{rel}")
        for old in (TIER_ERR_OLD,
                    """    if m.get("error"):
        print(f"{tier:10} {m['error']:>32}  (not supported by this model)")"""):
            try:
                nb.sub_idempotent(old, TIER_ERR_NEW)
            except AssertionError:
                continue
        if nb.changes:
            nb.save()
        print(f"{rel}: {nb.changes} edits")


if __name__ == "__main__":
    main()
