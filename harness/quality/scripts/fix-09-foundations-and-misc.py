#!/usr/bin/env python3
"""Audit fix 9 — foundations 03/04 and the remaining per-notebook defects.

  * f03 §5 claimed longer prompts raise TTFT; the longest prompt was the fastest in
    the committed output, because the repeated filler hit the prompt cache.
  * f03 §6's ramp demo recorded `concurrency 1 -> 1/1 ok in 242.84s`, a cold-start
    outlier that inverted the lesson.
  * f04 §3's "safe read" printed an empty string, because Kimi spent all 600 tokens
    on its reasoning trace and returned no text block to extract.
  * minimax §9 compared three generations and printed `''` for all three.
  * llama §3's prose described a three-call malformed response that a later re-run
    overwrote, so it pointed at output that is no longer there.
  * palmyra §6 claimed the token count tracks effort, and §8 quoted populations that
    are not in its own output.
  * gpt-oss §1 never mentioned server-side state, though gpt-oss supports it
    (audit gap C3).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

PREFILL_MD = """## 5. Prompt length drives TTFT

A direct demonstration of the prefill effect — with one confound removed.

The obvious version of this experiment repeats a filler sentence to grow the prompt,
which makes the long prompts an *exact repeated prefix* and therefore cacheable. A
cached prefix skips prefill, which is the very thing being measured, so the longest
prompt can come back fastest. The cell below gives each prompt unique content and
takes a median of three."""

PREFILL_CODE = '''filler = "Distributed systems engineering is a broad discipline. "
print(f"{'approx input tokens':>20} {'TTFT p50 (s)':>14}")
print("-" * 36)
timings = {}
for repeat in (1, 40, 200):
    samples = []
    for run in range(3):
        # A unique marker per call so no two prompts share a cacheable prefix.
        long_prompt = (
            f"[run {repeat}-{run}] " + filler * repeat
            + "\\n\\nSummarise the above in one sentence."
        )
        approx_tokens = len(long_prompt) // 4
        m = ttft(
            "/openai/v1/responses",
            {"model": MODEL, "input": long_prompt, "max_output_tokens": 60},
            region=REGION,
        )
        samples.append(m["ttft_s"])
    timings[approx_tokens] = statistics.median(samples)
    print(f"{approx_tokens:>20} {timings[approx_tokens]:>14.3f}")

# Derived: state whether prefill actually showed up rather than asserting it.
sizes = sorted(timings)
print()
if timings[sizes[-1]] > timings[sizes[0]]:
    ratio = timings[sizes[-1]] / max(timings[sizes[0]], 1e-9)
    print(f"=> TTFT rose {ratio:.1f}x from {sizes[0]} to {sizes[-1]} input tokens.")
    print("   That is prefill: the model has to read the prompt before it can start.")
else:
    print("=> TTFT did not rise with prompt length on this run. Queue time can")
    print("   dominate prefill on an idle endpoint at these sizes; repeat with a")
    print("   larger spread before concluding anything.")'''

RAMP_CODE = '''import concurrent.futures as cf


def one_call(i: int):
    code, _ = post(
        "/openai/v1/responses",
        {"model": MODEL, "input": f"Say OK ({i})", "max_output_tokens": 16},
        region=REGION,
    )
    return code


# Warm up first, and do not time it. The very first call from a cold process pays
# for TLS setup, credential resolution and token minting; an earlier version of
# this cell recorded 242 s for "concurrency 1" and made ramping look harmful.
one_call(-1)

for concurrency in (1, 3, 6):
    started = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        codes = list(pool.map(one_call, range(concurrency)))
    elapsed = time.perf_counter() - started
    ok = sum(1 for c in codes if c == 200)
    print(
        f"concurrency {concurrency:2} -> {ok}/{concurrency} ok in {elapsed:5.2f}s  "
        f"codes={codes}"
    )
print()
print("What matters here is that none were shed, not the wall-clock numbers: this")
print("is an idle account, so it cannot show the failure mode. The point of ramping")
print("is that a cold 0-to-peak step gets CapacityExceeded, and you cannot")
print("demonstrate that safely from a notebook.")'''

SAFE_READ = '''REASONER = "moonshot.kimi-k2-thinking"

text, response = converse(
    REASONER,
    [{"role": "user", "content": [{"text": "What is 17 * 23? Think it through."}]}],
    # Generous on purpose. At 600 tokens this model spends the entire budget on its
    # reasoning trace and returns NO text block, so converse_text() correctly
    # returns "" -- which made the "safe read" below look broken rather than safe.
    max_tokens=2500,
    region=REGION,
)

blocks = response.get("output", {}).get("message", {}).get("content", [])
print(f"{REASONER} returned {len(blocks)} content block(s):")
for i, block in enumerate(blocks):
    print(f"  content[{i}] -> {next(iter(block))}")

print("\\nthe naive read:")
try:
    print("  content[0]['text'] =", blocks[0]["text"][:40])
except KeyError:
    print("  content[0]['text'] -> KeyError, because block 0 is the reasoning trace")

print("\\nthe safe read:")
answer = converse_text(response).strip()
print("  converse_text()     :", answer[:70] or "(empty - budget exhausted)")
reasoning = converse_reasoning(response)
print("  converse_reasoning():", f"{len(reasoning)} chars" if reasoning else "(none)")
print(f"  stopReason          : {response.get('stopReason')}")
if not answer:
    print("  -> empty text with stopReason=max_tokens means the trace ate the")
    print("     budget. That is a budget problem, not a parsing problem.")

# Same request against Claude. Block ordering here varies between calls, which is
# the reason to never rely on it.
text2, response2 = converse(
    CLAUDE,
    [{"role": "user", "content": [{"text": "What is 17 * 23? Think it through."}]}],
    max_tokens=2500,
    region=REGION,
)
kinds = [
    next(iter(b))
    for b in response2.get("output", {}).get("message", {}).get("content", [])
]
print(f"\\n{CLAUDE} returned blocks: {kinds}")
print("  -> may or may not lead with text; treat the order as undefined.")'''

MINIMAX_COMPARE = '''task = "In one sentence, why is a harmonic mean the right average for speeds?"

print(f"{'model':38} {'latency':>9} {'out tok':>8} {'trace':>7}  answer")
print("-" * 112)
for model in [M2, M21, M25]:
    started = time.perf_counter()
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": task}],
            # These are reasoning models: at 160 tokens the trace consumed the whole
            # budget and all three rows printed '' with finish_reason="length".
            "max_tokens": 1500,
        },
        region=REGION,
    )
    elapsed = time.perf_counter() - started
    if code != 200:
        print(f"{model:38} {'-':>9} {'-':>8} {'-':>7}  HTTP {code}: {err(data)[:34]}")
        continue
    choice = data["choices"][0]
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip().replace("\\n", " ")
    trace = message.get("reasoning") or ""
    print(
        f"{model:38} {elapsed:>8.2f}s "
        f"{data['usage']['completion_tokens']:>8} {len(trace):>7}  "
        f"{(text[:40] or '(empty: finish=' + str(choice.get('finish_reason')) + ')')!r}"
    )
print()
print("The `trace` column is the reasoning this family returns in the non-standard")
print("message.reasoning field (see section 6). Budget for it: it is spent before")
print("any answer text appears.")'''

LLAMA_MD = """## 3. Tool use — and why you must validate every call

Both support tools. Assert the **arguments**, not the fact of a call.

Here is why, recorded from a run while this notebook was being written. Maverick
emitted **three** tool calls for a single question, and the middle one passed back
the tool's own JSON *schema* where the arguments should have been:

```
get_distance_km({'type': 'object', 'properties': {...}, 'required': [...]})
```

A loop that trusted `stopReason: tool_use` and executed every call would have handed
that object to the function. It has not reproduced on every run since — the cell
below usually returns one well-formed call — which is exactly the problem: a
malformed call is rare enough to survive testing and common enough to reach
production.

`stopReason` tells you the model wanted a tool. It says nothing about whether the
arguments are usable. Validate each call against the schema you published before you
execute it, and be ready for more than one call per turn. The cell below checks the
arguments and reports the count."""

PALMYRA_EFFORT_MD = """## 6. Reasoning effort

`reasoning_effort` is accepted — and on this model that appears to be all it is.
The trace is not returned in `message.reasoning` (unlike qwen, deepseek, glm, kimi,
minimax and nemotron-super, which do return it there), and the token counts below do
not track the setting. Read the numbers before you assume the knob does anything for
your workload."""

PALMYRA_EFFORT_TAIL = '''print()
counts = [c for c, _ in efforts.values() if isinstance(c, int)]
if counts and len(set(counts)) == 1:
    print("=> identical token counts at every effort: the parameter is accepted and")
    print("   appears to have no effect on this model.")
elif counts and counts != sorted(counts):
    print("=> the token counts do not rise with effort. Either the parameter has no")
    print("   effect here, or one sample per level is too noisy to tell. Do not")
    print("   budget on the assumption that higher effort costs more on this model.")'''

PALMYRA_JSON_TAIL = '''print()
print("Valid JSON is not correct JSON. Compare the `population` values returned")
print("above for the same question:")
for budget, value in populations.items():
    print(f"   max_tokens={budget:4} -> population={value!r} ({type(value).__name__})")
if len(set(map(str, populations.values()))) > 1:
    print("   Same question, different answers -- and note the types. Schema")
    print("   validation proves shape, never truth: range-check numeric fields")
    print("   yourself, and do not let a float where you expected millions through.")'''

GPT_OSS_STATE_MD = """## 1b. Server-side conversation state

gpt-oss serves the Responses API, so it gets `previous_response_id` — the same
server-side state the GPT-5.x and Gemma 4 families have, and something the
Chat-Completions-only families (qwen, deepseek, mistral, …) cannot do at all. It is
easy to miss, because most of this notebook uses Chat Completions to show where the
reasoning lives.

The trade is the usual one: chaining needs `store=True`, which retains input and
output for 30 days in-Region (see `../00-foundations/02`). `store=False` gives you
privacy and a 404 on the next turn.

Note this is a **Responses-API** capability, not a model one — the same model on
Chat Completions has no equivalent, and the `safeguard` variants have no Responses
API at all, so they have no server-side state either."""

GPT_OSS_STATE = '''# Turn 1 stores; turn 2 refers back with previous_response_id and no history.
first_code, first = post(
    f"{api_prefix(MANTLE_120B)}/responses",
    {
        "model": MANTLE_120B,
        "input": "My deploy tool is CodeDeploy. Reply: noted.",
        "max_output_tokens": 200,
        "store": True,
    },
    region=REGION,
)
second_code, second = post(
    f"{api_prefix(MANTLE_120B)}/responses",
    {
        "model": MANTLE_120B,
        "input": "Which deploy tool did I mention?",
        "previous_response_id": first.get("id"),
        "max_output_tokens": 200,
    },
    region=REGION,
)
from bedrock import response_text

print(f"turn 1 (store=True)  HTTP {first_code}")
print(f"turn 2 (chained)     HTTP {second_code}  ->",
      repr(response_text(second)[:80]))

# And the trade-off, made concrete.
_, private = post(
    f"{api_prefix(MANTLE_120B)}/responses",
    {"model": MANTLE_120B, "input": "Secret: 42. Reply ok.",
     "max_output_tokens": 200, "store": False},
    region=REGION,
)
code, refused = post(
    f"{api_prefix(MANTLE_120B)}/responses",
    {"model": MANTLE_120B, "input": "What was the secret?",
     "previous_response_id": private.get("id"), "max_output_tokens": 200},
    region=REGION,
)
print(f"\\nchaining from store=False -> HTTP {code}: {err(refused)[:60]}")

# The safeguard variants have no Responses API, so no server-side state either.
code, data = post(
    f"{api_prefix('openai.gpt-oss-safeguard-20b')}/responses",
    {"model": "openai.gpt-oss-safeguard-20b", "input": "Hi", "max_output_tokens": 16},
    region=REGION,
    attempts=1,
    timeout=60,
)
print(f"safeguard on /responses   -> HTTP {code}: {err(data)[:70]}")
print("=> no Responses API there, so no previous_response_id either.")'''


def main() -> None:
    # --- foundations 03 ----------------------------------------------------
    nb = Notebook(f"{REPO}/00-foundations/03-scaling-tiers-and-latency.ipynb")
    old = nb.find("## 5. Prompt length drives TTFT")
    assert len(old) == 1, old
    nb.set_source(old[0], PREFILL_MD)
    old = nb.find('filler = "Distributed systems engineering is a broad discipline. "')
    assert len(old) == 1, old
    nb.set_source(old[0], PREFILL_CODE)
    old = nb.find("import concurrent.futures as cf\n\n\ndef one_call(i: int):")
    assert len(old) == 1, old
    nb.set_source(old[0], RAMP_CODE)
    nb.sub("| Tier benchmarks | Look flat on an idle account; tiers separate under load |",
           "| Tier benchmarks | `flex` is deprioritised and usually slowest; `default` vs `priority` separates under load, not on an idle account |\n"
           "| Prompt-length benchmarks | Repeated filler is a *cacheable* prefix, so it measures caching rather than prefill — vary the content (§5) |")
    nb.save()
    print(f"00-foundations/03: {nb.changes} edits")

    # --- foundations 04 ----------------------------------------------------
    nb = Notebook(f"{REPO}/00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb")
    old = nb.find('REASONER = "moonshot.kimi-k2-thinking"')
    assert len(old) == 1, old
    nb.set_source(old[0], SAFE_READ)
    nb.sub("""The cell below proves it with `moonshot.kimi-k2-thinking`, which reliably returns
a `reasoningContent` block ahead of its answer.""",
           """The cell below proves it with `moonshot.kimi-k2-thinking`, which reliably returns
a `reasoningContent` block ahead of its answer — and it budgets 2,500 tokens,
because this model can spend a smaller budget entirely on the trace and return no
text block at all.""")
    nb.sub("""- Prefer Converse over `invoke_model` unless you need something unnormalised.""",
           """- Prefer Converse over `invoke_model` unless you need something unnormalised.
- Budget for the reasoning trace. An empty `text` block with
  `stopReason: max_tokens` is a budget problem, not a parsing problem — and it is
  indistinguishable from a broken parser if you do not print the stop reason.
- Guardrails attach here, not on `bedrock-mantle`: `guardrailConfig` on Converse,
  `guardrailIdentifier` on `invoke_model`. See `../99-cross-cutting/03` §9b.""")
    nb.save()
    print(f"00-foundations/04: {nb.changes} edits")

    # --- foundations 01: endpoint comparison should mention guardrails ------
    nb = Notebook(f"{REPO}/00-foundations/01-endpoints-auth-and-the-three-paths.ipynb")
    nb.sub('    ("Batch inference", "yes", "no (use bedrock-runtime)"),',
           '    ("Batch inference", "yes", "no (use bedrock-runtime)"),\n'
           '    ("Guardrails", "Converse / InvokeModel", "no"),')
    nb.save()
    print(f"00-foundations/01: {nb.changes} edits")

    # --- minimax ----------------------------------------------------------
    nb = Notebook(f"{REPO}/09-minimax/01-minimax-m2.ipynb")
    old = nb.find('task = "In one sentence, why is a harmonic mean the right average for speeds?"')
    assert len(old) == 1, old
    nb.set_source(old[0], MINIMAX_COMPARE)
    nb.sub("| Three live generations | Pin an exact model ID in production |",
           "| Three live generations | Pin an exact model ID in production |\n"
           "| Budget for the trace | These are reasoning models: a small `max_tokens` returns HTTP 200, `finish_reason=\"length\"` and empty `content` |")
    nb.save()
    print(f"09-minimax/01: {nb.changes} edits")

    # --- llama ------------------------------------------------------------
    nb = Notebook(f"{REPO}/15-meta-llama/01-llama4-moe-and-vision.ipynb")
    old = nb.find("## 3. Tool use — and why you must validate every call")
    assert len(old) == 1, old
    nb.set_source(old[0], LLAMA_MD)
    nb.sub("""- **Validate every tool call, do not just count them.** Maverick returned three
  calls for one question and one of them contained the tool's JSON schema instead
  of arguments (section 3). `stopReason: tool_use` is not a correctness signal.""",
           """- **Validate every tool call, do not just count them.** Maverick has returned
  three calls for one question with the tool's JSON schema in place of arguments
  (section 3). It does not happen every run, which is what makes it dangerous.
  `stopReason: tool_use` is not a correctness signal.""")
    nb.save()
    print(f"15-meta-llama/01: {nb.changes} edits")

    # --- palmyra vision ---------------------------------------------------
    nb = Notebook(f"{REPO}/12-writer-palmyra/01-palmyra-vision.ipynb")
    old = nb.find("## 6. Reasoning effort")
    assert len(old) == 1, old
    nb.set_source(old[0], PALMYRA_EFFORT_MD)
    nb.sub("""print("   level cannot rank them -- do not read a cost curve off this table.")""",
           """print("   level cannot rank them -- do not read a cost curve off this table.")
""" + PALMYRA_EFFORT_TAIL)
    nb.sub('''    if finish == "length" and not content.strip():
        print("    -> truncated before any JSON was emitted; raise max_tokens")
    elif content.strip():''',
           '''    if finish == "length" and not content.strip():
        print("    -> truncated before any JSON was emitted; raise max_tokens")
    elif content.strip():''')
    nb.sub('''PROMPT = "Give the capital and population of France as JSON."
for budget in (64, 600):''',
           '''PROMPT = "Give the capital and population of France as JSON."
populations = {}
for budget in (64, 600):''')
    nb.sub('''        try:
            print("    ->", parse_json_lenient(content))
        except ValueError as exc:''',
           '''        try:
            parsed = parse_json_lenient(content)
            populations[budget] = parsed.get("population")
            print("    ->", parsed)
        except ValueError as exc:''')
    nb.sub('''print()
print("Valid JSON is not correct JSON. Compare the population across the two")
print("budgets above: this model has returned 67060681 and 67.4719458 for the")
print("same question. Both parse; one is not a population. Schema validation")
print("proves shape, never truth - range-check numeric fields yourself.")''',
           PALMYRA_JSON_TAIL)
    nb.save()
    print(f"12-writer-palmyra/01: {nb.changes} edits")

    # --- gpt-oss: server-side state --------------------------------------
    nb = Notebook(f"{REPO}/14-openai-gpt-oss/01-gpt-oss-on-both-endpoints.ipynb")
    anchor = nb.find("## 2. `reasoning_effort` changes the spend")
    assert len(anchor) == 1, anchor
    at = nb.insert_after(anchor[0] - 1, "markdown", GPT_OSS_STATE_MD)
    nb.insert_after(at, "code", GPT_OSS_STATE)
    nb.sub("""- **gpt-oss lives on the `/v1` Mantle prefix**, not `/openai/v1` — the prefix follows
  the API family, not the vendor name.""",
           """- **gpt-oss lives on the `/v1` Mantle prefix**, not `/openai/v1` — the prefix follows
  the API family, not the vendor name.
- **Server-side state works here** (§1b). Because gpt-oss serves Responses it gets
  `previous_response_id`, which the Chat-Completions-only families cannot do — and
  which the `safeguard` variants cannot either, since they have no Responses API.""")
    nb.sub("""print()
print("completion_tokens_details:", data.get("usage", {}).get("completion_tokens_details"))
print("=> None. Reasoning tokens are not itemised here, so budget from the")
print("   completion_tokens total rather than expecting a breakdown.")""",
           """print()
details = data.get("usage", {}).get("completion_tokens_details")
print("completion_tokens_details:", details)
if details is None:
    print("=> None. Reasoning tokens are not itemised here, so budget from the")
    print("   completion_tokens total rather than expecting a breakdown.")
else:
    print("=> A breakdown is present now; read it rather than this sentence.")""")
    nb.save()
    print(f"14-openai-gpt-oss/01: {nb.changes} edits")


if __name__ == "__main__":
    main()
