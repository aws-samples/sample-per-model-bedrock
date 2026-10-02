#!/usr/bin/env python3
"""Audit fix 7 — the five 01-openai-gpt notebooks.

Every change here is a cell whose committed output contradicted, or failed to
support, the claim printed beside it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

# --------------------------------------------------------------- 01 core ---

EFFORT_MD = """### Effort changes spend — but only when the task needs thinking

The obvious experiment gives a misleading answer, so run both halves. On an easy
question this model spends **zero** reasoning tokens at every effort level: effort
is a ceiling, not a floor, and the model decides. Only on a task that actually
needs working-out does the setting show up in the bill."""

EFFORT_CODE = '''EASY = "What is 17 * 23? Answer with the number only."
HARD = (
    "Five houses in a row, each a different colour, each owner a different "
    "nationality and pet. The Brit lives in the red house. The Swede keeps dogs. "
    "The Dane drinks tea. The green house is immediately left of the white house. "
    "The green owner drinks coffee. Who owns the fish? Nationality only."
)

print(f"{'task':6} {'effort':8} {'reasoning tok':>14} {'output tok':>11} {'latency':>9}")
print("-" * 54)
spend = {}
for label, prompt in (("easy", EASY), ("hard", HARD)):
    for effort in ("none", "low", "medium", "high"):
        started = time.perf_counter()
        r = gpt5.responses.create(
            model=SOL,
            input=prompt,
            reasoning={"effort": effort},
            # Generous: a tight cap on a reasoning model returns status
            # "incomplete" and an empty answer, which would confound this table.
            max_output_tokens=3000,
        )
        elapsed = time.perf_counter() - started
        tokens = r.usage.output_tokens_details.reasoning_tokens
        spend[(label, effort)] = tokens
        print(
            f"{label:6} {effort:8} {tokens:>14} "
            f"{r.usage.output_tokens:>11} {elapsed:>8.2f}s"
        )

# Derived, because the single claim this cell used to make ("effort changes spend")
# was false on the easy prompt and non-monotonic on the hard one.
easy = [spend[("easy", e)] for e in ("none", "low", "medium", "high")]
hard = [spend[("hard", e)] for e in ("none", "low", "medium", "high")]
print()
if not any(easy):
    print("easy task: 0 reasoning tokens at every effort — the model judged it did")
    print("           not need to think. Raising effort cost nothing.")
if hard[0] == 0 and any(hard[1:]):
    print('hard task: effort="none" spent 0, the rest spent reasoning tokens, so the')
    print("           knob does work — but read the numbers, not the ordering:")
    print(f"           {dict(zip(('none', 'low', 'medium', 'high'), hard))}")
    if hard[1:] != sorted(hard[1:]):
        print("           low/medium/high are NOT in order from one sample each.")
        print("           Take medians before costing a change.")'''

CACHING_RUNTIME = '''# 2. Prompt caching. The system block has to be big enough to be worth caching.
#    Note "first call" is only cold the first time this notebook ever runs against
#    this prefix: the entry survives between runs, so on a re-run both calls read
#    from cache. Report what happened rather than asserting a write.
PRIMER = [{"role": "system", "content": "You are a terse assistant. " * 400},
          {"role": "user", "content": "Say ok."}]
cache_rows = []
for attempt in (1, 2):
    reply = runtime_openai_post("/responses", {"model": PROFILE, "input": PRIMER, "max_output_tokens": 20})
    detail = body_of(reply)["usage"]["input_tokens_details"]
    cache_rows.append((detail.get("cache_write_tokens") or 0, detail.get("cached_tokens") or 0))
    print(f"caching call {attempt}    {reply.status_code}  "
          f"cache_write={detail.get('cache_write_tokens')} cached={detail.get('cached_tokens')}")
if cache_rows[0][0] and cache_rows[1][1]:
    print("                       -> a cold write then a warm read, as designed")
elif cache_rows[1][1]:
    print("                       -> both calls read from cache: the prefix was")
    print("                          already warm from an earlier run. Caching works")
    print("                          across processes, which is the point of it.")
else:
    print("                       -> no cache activity; the prefix may be under the")
    print("                          1,024-token minimum")'''

# ------------------------------------------------------------ 02 websearch ---

CITATIONS_CODE = '''# Ask something that REQUIRES current information. A timeless question (the
# difference between a Region and an AZ, say) is answered from the model's own
# knowledge with no search and therefore no citations -- which made an earlier
# version of this cell print "0 citation(s)" in a section titled "citations are
# mandatory".
response = client.responses.create(
    model=GPT55,
    input=(
        "Name one Amazon Bedrock feature announced in 2026 and say what it does. "
        "Two sentences."
    ),
    tools=[{"type": "web_search", "external_web_access": False}],
    max_output_tokens=600,
)

print("=== ANSWER ===")
print(response.output_text[:600])

print("\\n=== SOURCES ===")
seen = 0
for item in response.output:
    if item.type != "message":
        continue
    for block in item.content:
        for ann in getattr(block, "annotations", None) or []:
            if getattr(ann, "type", None) == "url_citation":
                seen += 1
                print(f"  [{seen}] {ann.title[:70]}")
                print(f"      {ann.url}")
                print(f"      supports characters {ann.start_index}-{ann.end_index}")
rounds = len([i for i in response.output if i.type == "web_search_call"])
print(f"\\n{seen} citation(s) from {rounds} search round(s)")
if not seen:
    print("No citations this run. That is a real outcome, not a bug: the model may")
    print("answer from its own knowledge even with the tool available. Treat an")
    print("uncited answer as ungrounded and do not present it as sourced.")'''

RESEARCHER_CODE = '''researcher = GroundedResearcher(project=project_id)
result = researcher.ask(
    "Name one Amazon Bedrock capability announced in 2026. Two sentences."
)
print("answer :", result["answer"][:260])
print("rounds :", result["search_rounds"])
print("sources:", len(result["sources"]))
for s in result["sources"]:
    print(f"   - {str(s['title'])[:64]}\\n     {s['url']}")
if not result["sources"]:
    # Say so rather than printing an empty heading. A grounded-answer helper that
    # returns no sources has to surface that to its caller, because the caller's
    # obligation to display citations does not disappear when there are none.
    print("   (none — the model answered without searching, so this answer is")
    print("    NOT grounded and must not be presented as cited)")'''

REGION_MD = """Web Search is **strictly regional**: each Region runs its own search and fetch
tier, and queries, index data, and results never cross Region boundaries. It is
available in `us-east-1`, `us-east-2`, and `us-west-2` — not `eu-central-1`.

The cell below shows *why* you cannot even test it in `eu-central-1`, and the
reasoning matters more than the result. Web Search is available on the GPT-5.x
family only (§7). None of that family is in `eu-central-1`. So the 404 you get
there is the **model** being absent, not the tool being refused — and the
conclusion "no Web Search in eu-central-1" follows from the two facts together,
not from this status code on its own. Read what an error actually reports before
you build a claim on it."""

# ----------------------------------------------------------------- 03 tools ---

MAX_TOOL_CALLS = '''# max_tool_calls is DOCUMENTED as a cap on how many calls the model may make. Send
# it and count what comes back, because accepted is not the same as honoured.
print(f"{'max_tool_calls':>15} {'HTTP':>5} {'calls requested':>16}")
print("-" * 40)
observed = {}
for cap in (1, 2, None):
    body = {
        "model": SOL,
        "input": "Check stock for A-100, B-200 and C-300, plus both shipping methods.",
        "tools": TOOLS,
        "max_output_tokens": 600,
        "store": False,
    }
    if cap is not None:
        body["max_tool_calls"] = cap
    code, data = post(f"{GPT5_PREFIX}/responses", body, region=REGION)
    calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
    observed[cap] = len(calls)
    print(f"{cap!s:>15} {code:>5} {len(calls):>16}")

print()
if observed.get(1, 0) > 1 or observed.get(2, 0) > 2:
    print("=> The cap was ACCEPTED (HTTP 200) and IGNORED. This is the '200 does not")
    print("   mean honoured' failure mode, and it is the dangerous kind: there is no")
    print("   error to alert you. If you need a hard limit on tool calls, enforce it")
    print("   in your own loop -- count the calls you execute and stop.")
else:
    print("=> The cap was honoured on this run. Verify it yourself rather than")
    print("   relying on it: enforce the limit in your loop as well.")'''

INVENTED_ARGS = '''# (a) The model invents arguments that don't match your schema. Validate.
#
# FORCE the call. With tool_choice="auto" the model answers "I don't know which
# SKU" in prose and no arguments are invented at all, so the cell demonstrated
# nothing. Compelling the call is what surfaces the invented value.
code, data = post(
    f"{GPT5_PREFIX}/responses",
    {
        "model": SOL,
        "input": "Check stock for the blue widget please.",
        "tools": TOOLS,
        "tool_choice": {"type": "function", "name": "get_stock"},
        "max_output_tokens": 400,
        "store": False,
    },
    region=REGION,
)
calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
print("(a) invented arguments:")
if not calls:
    print("    no tool call even when forced — see §3, forcing is best-effort")
for call in calls:
    args = parse_json_lenient(call["arguments"] or "{}")
    known = str(args.get("sku", "")).upper() in INVENTORY
    print(f"    {call['name']}({args}) -> sku known to us? {known}")
print("    => always validate arguments against your own data before executing.")
print("       'the blue widget' is not a SKU, so whatever arrived above was made up.")'''

# --------------------------------------------------------------- 04 caching ---

AUTO_CACHE = '''def auto_cached_call(model: str, question: str) -> dict:
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": model,
            "max_output_tokens": 150,
            # No prompt_cache_options, no breakpoints — nothing to configure.
            "input": REFERENCE_DOC + "\\n\\n" + question,
            "store": False,
        },
        region=REGION,
    )
    usage = data.get("usage", {})
    details = usage.get("input_tokens_details", {})
    return {
        "code": code,
        "input": usage.get("input_tokens"),
        "cached": details.get("cached_tokens") or 0,
        "written": details.get("cache_write_tokens") or 0,
    }


# Four calls, not two. Automatic caching is best-effort: it decides for itself
# whether a prefix is worth keeping, so the first repeat is often still a miss.
# Two calls made this section print cached=0 twice and read as "it does not work".
for model in (GPT55, GPT54):
    print(f"{model}")
    hits = []
    for n in range(1, 5):
        r = auto_cached_call(model, f"Question {n}: name the default tier. One word.")
        hits.append(r["cached"])
        print(f"   call {n}: input={r['input']} cached={r['cached']} written={r['written']}")
    first_hit = next((i + 1 for i, c in enumerate(hits) if c), None)
    if first_hit:
        print(f"   -> first cache hit on call {first_hit}")
    else:
        print("   -> no cache hit in four calls")

print()
print("=> Automatic caching is not a guarantee and not immediate. Where it matters,")
print("   use a model with EXPLICIT breakpoints (section 2): gpt-5.6 writes on the")
print("   first call and reads on every one after, deterministically.")'''

LATENCY_CODE = '''def timed_call(warm_up: bool) -> float:
    if warm_up:
        cached_call("Warm the cache. One word.")
    started = time.perf_counter()
    cached_call(f"Unique question {time.time():.0f}. One line.")
    return time.perf_counter() - started


# Medians, not single samples. A single pair has shown the WARM call slower than the
# cold one here, which says more about queue noise than about caching.
import statistics

cold = statistics.median(timed_call(warm_up=False) for _ in range(3))
warm = statistics.median(timed_call(warm_up=True) for _ in range(3))
print(f"median latency, cold-ish cache : {cold:.2f}s")
print(f"median latency, warm cache     : {warm:.2f}s")
print()
if warm < cold:
    print(f"=> warm is {100 * (1 - warm / cold):.0f}% faster on this run, which is the")
    print("   prefill saving showing through.")
else:
    print("=> warm was NOT faster on this run. The mechanism is sound — a cached")
    print("   prefix skips prefill — but on short prompts and an idle endpoint the")
    print("   saving is smaller than the run-to-run noise. Measure at your own")
    print("   prompt length and concurrency before quoting a number.")'''

# ------------------------------------------------------- 05 server-side tools ---

TASKS_CODE = '''# The tasks tool: a stack for managing work within a conversation.
code, data = post(
    f"{OSS_PREFIX}/responses",
    {
        "model": OSS120,
        "input": "Use the tasks tool to push a task: review the API documentation.",
        "max_output_tokens": 600,
    },
    region=REGION,
)
print("HTTP", code, "| items:", [i.get("type") for i in data.get("output", [])])
# Print the tool invocation, not just response_text(): when the model ends its turn
# on an mcp_call there is no message item at all, so the answer is legitimately
# empty and printing only that made this cell look broken.
for item in data.get("output", []):
    if item.get("type") == "mcp_call":
        print("  tool     :", item.get("name"))
        print("  arguments:", json.dumps(item.get("arguments"))[:160])
        print("  output   :", json.dumps(item.get("output"))[:160])
answer = response_text(data)
print("answer:", answer[:200] if answer.strip() else "(none — the turn ended on the tool call)")'''

MCP_MD = """### Wiring it up

```python
response = client.responses.create(
    model="openai.gpt-oss-120b",
    tools=[{
        "type": "mcp",
        "server_label": "orders",
        "connector_id": "arn:aws:lambda:us-east-1:123456789012:function:my-mcp-tool",
        "require_approval": "never",     # must be "never"
    }],
    input="What is the status of order 88213?",
)
```

On success the output contains an `mcp_list_tools` item (what Bedrock discovered)
and `mcp_call` items (what it invoked). No credentials are passed — Bedrock uses
the caller's IAM identity.

We do **not** deploy a Lambda in this notebook. Instead the cell below probes the
request *shape* with a well-formed but non-existent ARN, which turns out to teach
something more useful than intended: **Bedrock validates the tool object but not
the ARN**. A missing field is a clean 400 that names it; a connector that cannot
possibly exist is accepted with HTTP 200, because nothing is resolved until the
model actually decides to call the tool.

So a 200 here does not mean your connector works. It means your JSON is valid."""

MCP_CODE = '''import boto3

account_id = boto3.client("sts").get_caller_identity()["Account"]
fake_arn = (
    f"arn:aws:lambda:{REGION}:{account_id}:function:mantle-samples-does-not-exist"
)

print(f"{'tool object':32} {'HTTP':>5}  what the service said")
print("-" * 96)
for label, tool in [
    (
        "well-formed, ARN not real",
        {"type": "mcp", "server_label": "orders", "connector_id": fake_arn,
         "require_approval": "never"},
    ),
    (
        "ARN not even an ARN",
        {"type": "mcp", "server_label": "orders", "connector_id": "not-an-arn",
         "require_approval": "never"},
    ),
    (
        "missing require_approval",
        {"type": "mcp", "server_label": "orders", "connector_id": fake_arn},
    ),
    (
        'require_approval="always"',
        {"type": "mcp", "server_label": "orders", "connector_id": fake_arn,
         "require_approval": "always"},
    ),
    (
        "missing server_label",
        {"type": "mcp", "connector_id": fake_arn, "require_approval": "never"},
    ),
    (
        "missing connector_id",
        {"type": "mcp", "server_label": "orders", "require_approval": "never"},
    ),
]:
    code, data = post(
        f"{OSS_PREFIX}/responses",
        {
            "model": OSS120,
            "input": "Status of order 88213?",
            "tools": [tool],
            "max_output_tokens": 300,
        },
        region=REGION,
        attempts=1,
        timeout=90,
    )
    verdict = code if code != -1 else "stalled"
    detail = "accepted" if code == 200 else err(data)[:64]
    print(f"  {label:30} {verdict!s:>5}  {detail}")'''

MCP_AFTER_MD = """Three things in that table are worth carrying away.

1. **Missing fields are caught, with the field named.** `server_label` produces a
   deserialisation error quoting it; `connector_id` produces
   *"MCP tool 'orders' requires 'connector_id'. Note: `server_url` is not
   supported."* That last clause is a migration gotcha in its own right — the
   OpenAI spec's remote-MCP `server_url` has no equivalent here, so a hosted-MCP
   integration ported from OpenAI must be re-pointed at a Lambda or Gateway ARN.
2. **The ARN is not validated at request time.** A non-existent one, and even a
   string that is not an ARN at all, both return 200. Resolution happens only if
   the model decides to call the tool, so a green request tells you nothing about
   whether your connector is reachable. Test it with a prompt that forces the call.
3. **`require_approval` is not enforced at request time either.** Omitting it, or
   setting `"always"`, is accepted here. The documented requirement is `"never"`
   for the connector to actually execute — which you discover at invocation time,
   not at validation time.

The debugging rule that follows: a **schema** complaint means your request shape is
wrong. A **not-found or access** complaint means the shape was fine and Bedrock
genuinely tried to reach your Lambda. A **200 with no `mcp_call`** means neither —
the model simply chose not to use the tool."""

SERVER_SIDE_CLIENT = '''bot = ServerSideToolClient(project=project_id)  # no connector: built-ins only
result = bot.ask("Use the notes tool to remember: the on-call rota is weekly.")
print("answer     :", result["answer"][:160])
print("tool events:", [i.get("type") for i in result["tool_calls"]])
if not result["tool_calls"]:
    # This happens, and it is worth showing rather than hiding. Instead of an
    # mcp_call item the model sometimes emits the tool syntax inline as text --
    # note the {{note:...}} in the answer above. A client that keys off mcp_call
    # items will record "no tool used" while the model believes it used one.
    print()
    print("No mcp_call item this run. Look at the answer: the model wrote the tool")
    print("syntax into its own text instead of invoking the tool. Built-in tool use")
    print("is not guaranteed per call, so treat the presence of an mcp_call item as")
    print("the only evidence a tool actually ran.")'''


def fix_01() -> None:
    nb = Notebook(f"{REPO}/01-openai-gpt/01-responses-api-core.ipynb")
    nb.sub("### Effort changes spend, measurably", EFFORT_MD)
    old = nb.find('print(f"{\'effort\':8} {\'reasoning tok\':>14} {\'output tok\':>11} {\'latency\':>9}")')
    assert len(old) == 1, old
    nb.set_source(old[0], EFFORT_CODE)

    old = nb.find("# 2. Prompt caching. The system block has to be big enough to be worth caching.")
    assert len(old) == 1, old
    body = nb.source(old[0])
    start = body.index("# 2. Prompt caching.")
    end = body.index("# 3. Server-side state.")
    nb.set_source(old[0], body[:start] + CACHING_RUNTIME + "\n\n" + body[end:])

    nb.sub("| Web Search | Only this family — see `02-web-search-and-grounding.ipynb` |",
           "| Web Search | Only this family — see `02-web-search-and-grounding.ipynb` |\n"
           "| `max_tool_calls` | Accepted and **not enforced** — cap tool calls in your own loop (`03` §4) |\n"
           "| `reasoning.effort` on easy tasks | Spends 0 reasoning tokens at every level; effort is a ceiling, not a floor |")
    nb.save()
    print(f"01-openai-gpt/01: {nb.changes} edits")


def fix_02() -> None:
    nb = Notebook(f"{REPO}/01-openai-gpt/02-web-search-and-grounding.ipynb")
    old = nb.find('''response = client.responses.create(
    model=GPT55,
    input=(
        "What is the difference between an AWS Region and an Availability Zone? "
        "Answer in two sentences."
    ),''')
    assert len(old) == 1, old
    nb.set_source(old[0], CITATIONS_CODE)

    old = nb.find('researcher = GroundedResearcher(project=project_id)')
    assert len(old) == 1, old
    body = nb.source(old[0])
    start = body.index("researcher = GroundedResearcher(project=project_id)")
    nb.set_source(old[0], body[:start] + RESEARCHER_CODE)

    old = nb.find("Web Search is **strictly regional**")
    assert len(old) == 1, old
    nb.set_source(old[0], REGION_MD)

    nb.sub("| Model decides | No search happens on timeless questions — rounds can be 0 |",
           "| Model decides | No search happens on timeless questions — rounds can be 0, and **an uncited answer is not grounded** |\n"
           "| eu-central-1 | No Web Search there because no gpt-5.x is there. The 404 names the *model*, not the tool |")
    nb.save()
    print(f"01-openai-gpt/02: {nb.changes} edits")


def fix_03() -> None:
    nb = Notebook(f"{REPO}/01-openai-gpt/03-tools-and-structured-output.ipynb")
    nb.sub("""- Others accept only a subset. The cell below probes each form, because which
  model supports what changes. In the run committed here, `none` and `required` were
  explicit 400s
  ("Supported options: [\\"auto\\"]"), and the *named-function* form returns
  **HTTP 200 while quietly ignoring the constraint* — the model answers in prose
  with no tool call at all.""",
           """- Others accept only a subset. The cell below probes each form, because which
  model supports what changes. In the run committed here, `none` and `required` were
  explicit 400s ("Supported options: [\\"auto\\"]"), and the *named-function* form
  returned **HTTP 200 while quietly ignoring the constraint** — the model answered
  in prose with no tool call at all.""")

    nb.sub("# max_tool_calls caps how many the model may make in total.", "")
    old = nb.find('''code, data = post(
    f"{GPT5_PREFIX}/responses",
    {
        "model": SOL,
        "input": "Check stock for A-100, B-200 and C-300, plus both shipping methods.",
        "tools": TOOLS,
        "max_tool_calls": 2,
        "max_output_tokens": 600,
        "store": False,
    },
    region=REGION,
)''')
    assert len(old) == 1, old
    nb.set_source(old[0], MAX_TOOL_CALLS)

    old = nb.find("# (a) The model invents arguments that don't match your schema. Validate.")
    assert len(old) == 1, old
    nb.set_source(old[0], INVENTED_ARGS)

    nb.sub("| Parallel calls | Supported on gpt-5.6; Gemma 4 does one per turn |",
           "| Parallel calls | Supported on gpt-5.6; Gemma 4 does one per turn |\n"
           "| `max_tool_calls` | Accepted and **not enforced** (§4). Cap tool calls in your own loop |")
    nb.save()
    print(f"01-openai-gpt/03: {nb.changes} edits")


def fix_04() -> None:
    nb = Notebook(f"{REPO}/01-openai-gpt/04-prompt-caching-and-cost.ipynb")
    old = nb.find("def auto_cached_call(model: str, question: str) -> dict:")
    assert len(old) == 1, old
    nb.set_source(old[0], AUTO_CACHE)

    nb.sub("""No parameters at all: the system caches eligible prefixes of ≥1,024 tokens by
exact match, and there is **no cache-write fee** on these models.""",
           """No parameters at all: the system may cache eligible prefixes of ≥1,024 tokens by
exact match, and there is **no cache-write fee** on these models.

"May" is load-bearing. Automatic caching is best-effort and not immediate — in the
run below one model took several identical calls before the first hit and the other
never hit at all inside four. If caching is part of your cost model, use a model
with explicit breakpoints (§2) where the behaviour is deterministic.""")

    old = nb.find("def timed_call(warm_up: bool) -> float:")
    assert len(old) == 1, old
    nb.set_source(old[0], LATENCY_CODE)

    nb.sub("""## 8. TTL and eviction

The default TTL is **30 minutes** on gpt-5.6, set via `prompt_cache_options.ttl`.
That is long enough to cover the burst of calls a single agent run generates. A
cache entry that is not read within its TTL expires.""",
           """## 8. TTL and eviction

`prompt_cache_options.ttl` takes **`30m`** — and, on this model today, only `30m`.
It is documented as the default, but the probe below shows the other plausible
values are rejected outright, so treat it as a fixed 30 minutes rather than a knob.
That is long enough to cover the burst of calls a single agent run generates. A
cache entry that is not read within its TTL expires.""")

    nb.sub("| TTL | 30 min default on gpt-5.6; entries expire if not read |",
           "| TTL | `30m` is the **only** accepted value on gpt-5.6 (§8), not merely the default |")
    nb.sub("| 1,024-token minimum | Shorter prefixes are silently not cached — no error |",
           "| 1,024-token minimum | Shorter prefixes are silently not cached — no error |\n"
           "| Automatic caching | Best-effort and not immediate on gpt-5.5/5.4. Explicit breakpoints are deterministic |")
    nb.save()
    print(f"01-openai-gpt/04: {nb.changes} edits")


def fix_05() -> None:
    nb = Notebook(f"{REPO}/01-openai-gpt/05-server-side-tools-and-fine-tuning.ipynb")

    old = nb.find("# The tasks tool: a stack for managing work within a conversation.")
    assert len(old) == 1, old
    nb.set_source(old[0], TASKS_CODE)

    old = nb.find("### Wiring it up")
    assert len(old) == 1, old
    nb.set_source(old[0], MCP_MD)

    old = nb.find("account_id = boto3.client(\"sts\").get_caller_identity()[\"Account\"]")
    assert len(old) == 1, old
    nb.set_source(old[0], MCP_CODE)

    old = nb.find("Read these carefully: a **schema** complaint means your request shape is wrong,")
    assert len(old) == 1, old
    nb.set_source(old[0], MCP_AFTER_MD)

    old = nb.find("bot = ServerSideToolClient(project=project_id)  # no connector: built-ins only")
    assert len(old) == 1, old
    nb.set_source(old[0], SERVER_SIDE_CLIENT)

    nb.sub("| `require_approval` | Must be `\"never\"` for the MCP connector |",
           "| `require_approval` | Must be `\"never\"` for the connector to execute — but **not validated at request time** |\n"
           "| ARN validation | Not done at request time. A bogus `connector_id` returns 200; it resolves only when the model calls the tool |\n"
           "| `server_url` | **Not supported.** The OpenAI spec's remote-MCP field has no equivalent — use a Lambda or Gateway ARN |")
    nb.sub("| Built-in tools | `notes` / `tasks` are **gpt-oss only**; do not declare them |",
           "| Built-in tools | `notes` / `tasks` are **gpt-oss only**; do not declare them |\n"
           "| Built-in tool use is per call | The model sometimes writes `{{note:...}}` into its text instead of emitting an `mcp_call`. Only an `mcp_call` item proves a tool ran |")
    nb.save()
    print(f"01-openai-gpt/05: {nb.changes} edits")


if __name__ == "__main__":
    fix_01()
    fix_02()
    fix_03()
    fix_04()
    fix_05()
