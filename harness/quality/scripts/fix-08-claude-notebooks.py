#!/usr/bin/env python3
"""Audit fix 8 — the three 02-anthropic-claude notebooks.

The substantive discovery behind most of these edits: on `bedrock-mantle` Claude
returns thinking as a **signed, opaque block**. The block is there, its `thinking`
string is always empty, and `signature` carries the encrypted trace.
`usage.output_tokens_details.thinking_tokens` tells you what it cost. Both
notebooks were written as though the text were readable, so their thinking demos
printed "(none returned)" and "thinking chars: 0" under headings promising a
visible trace.

Also fixed:
  * "`count_tokens` is available on mantle only — it is not on `bedrock-runtime`"
    is false. `bedrock-runtime` has a `CountTokens` operation; it works for the
    Claude 4.x generation with a bare versioned ID and refuses inference profiles
    and the whole Claude 5 generation. Coverage added (audit gap C2).
  * `text_editor_20250124` 400s; the working type is `text_editor_20250728` with
    name `str_replace_based_edit_tool`.
  * The compaction section showed input_tokens growing and claimed a saving.
  * `thinking.type="enabled"` returns the most useful error in the family, naming
    `adaptive` and `output_config.effort`, and nothing demonstrated it (gap C5).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

# ------------------------------------------------------------------ 01 core ---

BLOCK_READ = '''for model in (SONNET5, OPUS5):
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": model,
            # Budget generously. At 200 tokens a reasoning model spends the lot on
            # thinking and returns no text block at all, so the "safe read" below
            # printed '' and the section demonstrated only half its point.
            "max_tokens": 1500,
            "messages": [
                {
                    "role": "user",
                    "content": "What problem does a write-ahead log solve?",
                }
            ],
        },
        region=REGION,
        headers=AV,
    )
    kinds = [b.get("type") for b in data.get("content", [])]
    print(f"{model:30} blocks={kinds}")
    print(f"    text via helper: {claude_text(data).strip()[:80]!r}")
    first = data["content"][0]
    naive_verdict = "work" if first.get("type") == "text" else "FAIL"
    print(
        f"    content[0]['type'] = {first.get('type')!r} "
        f"-> naive content[0]['text'] would {naive_verdict}"
    )'''

COUNT_TOKENS_MD = """## 8. Count tokens before you spend them

`count_tokens` on the Messages API is a control-plane call that prices a request
without running it — useful for pre-flight cost estimation and for deciding whether
a prompt fits.

**It exists on both endpoints, with different coverage**, which is worth knowing
before you conclude it is missing:

| | `bedrock-mantle` | `bedrock-runtime` |
|---|---|---|
| How | `POST /anthropic/v1/messages/count_tokens` | `bedrock-runtime` `count_tokens` API |
| Claude 5 generation (`opus-5`, `sonnet-5`, `opus-4-8`, `opus-4-7`, `fable-5`) | **works** | **refused** — *"model doesn't support counting tokens"* |
| Claude 4.x (`haiku-4-5`, `opus-4-6`, `opus-4-5`, `opus-4-1`) | works | **works**, with the bare versioned ID |
| Inference profiles (`us.` prefix) | n/a | **refused** |
| Other providers (Nova, Llama, GPT, Gemma…) | n/a | refused |

So on `bedrock-runtime` it is a Claude-4.x-only facility that will not take the
very profile ID Converse requires elsewhere. §8b shows that. On mantle it covers
the current generation, which is usually the one you want."""

COUNT_TOKENS_RUNTIME_MD = """### 8b. The same thing on `bedrock-runtime`, and its two refusals

`bedrock-runtime` exposes `CountTokens` as a first-class operation. Two things about
it catch people out, and both are visible below:

- It **refuses an inference profile**. Every other `bedrock-runtime` call for a
  profile-only model wants the `us.` form; this one wants the bare versioned ID.
- It **refuses the Claude 5 generation** outright, so the models you are most
  likely to be running are the ones it does not price."""

COUNT_TOKENS_RUNTIME = '''from bedrock import resolve_runtime_id, runtime_client

runtime = runtime_client(REGION)
MESSAGE = {"converse": {"messages": [
    {"role": "user", "content": [{"text": "How many tokens is this sentence?"}]}
]}}

print(f"{'model id sent':46} {'result':>34}")
print("-" * 82)
for label in (
    "anthropic.claude-haiku-4-5-20251001-v1:0",   # Claude 4.x, bare + version
    "anthropic.claude-opus-4-6-v1",               # 4.x, no :0 suffix
    resolve_runtime_id(SONNET5, REGION),          # what Converse wants: us. profile
    SONNET5,                                      # Claude 5, bare
):
    try:
        answer = runtime.count_tokens(modelId=label, input=MESSAGE)
        print(f"{label:46} {str(answer['inputTokens']) + ' tokens':>34}")
    except Exception as exc:  # noqa: BLE001 - the refusal IS the lesson
        print(f"{label:46} {str(exc).split(': ')[-1][:34]:>34}")

print()
print("=> Pre-flight pricing on bedrock-runtime is a Claude 4.x facility, and it")
print("   wants the bare versioned ID rather than the inference profile. For the")
print("   Claude 5 generation, use the mantle count_tokens above instead.")'''

# ------------------------------------------------------ 02 thinking/caching ---

THINKING_HELPER = '''def thinking_text(payload: dict) -> str:
    """The thinking blocks' text — which on bedrock-mantle is always empty.

    Kept deliberately, because the empty result is the finding. Claude returns a
    typed `thinking` block whose `thinking` string is "" and whose `signature`
    carries the trace in encrypted form. What it cost is in
    usage.output_tokens_details.thinking_tokens. See section 2.
    """
    return "".join(
        b.get("thinking", "")
        for b in payload.get("content", [])
        if b.get("type") == "thinking"
    )


def thinking_signature(payload: dict) -> str:
    """The opaque signature that stands in for the readable trace."""
    return "".join(
        b.get("signature", "")
        for b in payload.get("content", [])
        if b.get("type") == "thinking"
    )'''

THINKING_MD = """## 2. Reading a thinking response — the block is there, the text is not

With thinking active the `content` array carries a `thinking` block alongside
`text`, which is why `content[0].text` is unsafe. But do not expect to read the
reasoning: on `bedrock-mantle` the block looks like this.

```json
{"type": "thinking", "thinking": "", "signature": "CAISiQIKcAgQEAEYAipApexf3L5..."}
```

`thinking` is **always an empty string**. The trace is carried in `signature`, in
encrypted form, for passing back on a later turn — the same trade Grok makes with
`reasoning.encrypted_content` (`../11-xai-grok/01` §4c). What the thinking cost you
is in `usage.output_tokens_details.thinking_tokens`.

So the three things to read are: whether a `thinking` block is present, how many
`thinking_tokens` it consumed, and — if you plan to continue the conversation —
the `signature`. Never the text.

**Which models emit the block is not what you would guess**, either. It depends on
the model *and* on whether you set `thinking` at all, so the cell probes both."""

THINKING_CODE = '''PUZZLE = (
    "A farmer must cross a river with a wolf, a goat and a cabbage. The boat "
    "holds the farmer plus one item. The wolf eats the goat if left alone "
    "together; the goat eats the cabbage. Give the shortest sequence."
)

print(f"{'model':28} {'request':22} {'blocks':26} {'think tok':>10} {'text len':>9}")
print("-" * 100)
for model in (OPUS5, SONNET5, OPUS48):
    for label, extra in (
        ("no thinking param", {}),
        ("thinking: adaptive", {"thinking": {"type": "adaptive"}}),
    ):
        code, data = post(
            f"{PREFIX}/messages",
            {"model": model, "max_tokens": 4000,
             "messages": [{"role": "user", "content": PUZZLE}], **extra},
            region=REGION,
            headers=AV,
        )
        if code != 200:
            print(f"{model:28} {label:22} HTTP {code} {err(data)[:40]}")
            continue
        blocks = [b.get("type") for b in data.get("content", [])]
        details = (data.get("usage") or {}).get("output_tokens_details") or {}
        print(
            f"{model:28} {label:22} {str(blocks):26} "
            f"{details.get('thinking_tokens', 0):>10} {len(claude_text(data)):>9}"
        )

# Show the block itself, so the empty `thinking` and the signature are visible.
code, data = post(
    f"{PREFIX}/messages",
    {"model": OPUS5, "max_tokens": 3000,
     "messages": [{"role": "user", "content": PUZZLE}]},
    region=REGION,
    headers=AV,
)
print()
for block in data.get("content", []):
    if block.get("type") == "thinking":
        print("thinking block keys :", sorted(block.keys()))
        print(f"  thinking (text)   : {block.get('thinking', '')!r}  <- always empty")
        print(f"  signature         : {len(block.get('signature', ''))} chars, opaque")
details = (data.get("usage") or {}).get("output_tokens_details") or {}
print(f"  thinking_tokens   : {details.get('thinking_tokens')}  <- what it cost")
print(f"\\nhelper thinking_text() -> {thinking_text(data)!r} (empty, as designed)")
print("=== ANSWER ===")
print(claude_text(data)[:400])'''

THINKING_ENABLED_MD = """### 2b. `thinking.type` — and the error that tells you what to send

The Claude 5 generation replaced the older explicit-budget form with adaptive
thinking, and the refusal is unusually helpful: it names both replacement fields.
Worth provoking once, because it is the fastest way to learn the current shape —
and because `thinking: {"type": "enabled", "budget_tokens": N}` is what most
existing Claude code on the internet still sends."""

THINKING_ENABLED = '''for label, thinking in (
    ('type="enabled" + budget_tokens (the older form)',
     {"type": "enabled", "budget_tokens": 1024}),
    ('type="adaptive" (the current form)', {"type": "adaptive"}),
):
    code, data = post(
        f"{PREFIX}/messages",
        {"model": SONNET5, "max_tokens": 2000, "thinking": thinking,
         "messages": [{"role": "user", "content": "Reply OK"}]},
        region=REGION,
        headers=AV,
    )
    print(f"  {label:48} HTTP {code}")
    if code != 200:
        print(f"      {err(data)[:150]}")

print()
print("=> Read that message: it names thinking.type.adaptive AND output_config.effort.")
print("   Errors that tell you the replacement are rare -- this one saves a doc hunt.")'''

STREAM_MD = """## 4. Streaming a thinking response

Thinking and answer text arrive as distinct block types, so a client *can* render
them separately. On `bedrock-mantle`, though, there is nothing to render in the
thinking channel: as §2 showed, the trace is encrypted, so `thinking_delta` events
carry no text even when a `thinking` block is opened. The cell below counts both so
you can see which events actually arrive."""

STREAM_TAIL = '''print(f"\\n\\nthinking chars: {thinking_chars} | answer chars: {text_chars}")
print("--- event types ---")
for name, count in sorted(events.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4}  {name}")

# Derived: the previous version printed "thinking chars: 0" under a heading
# promising a visible thinking panel, with nothing to explain the zero.
print()
if thinking_chars:
    print("=> thinking_delta carried text on this run.")
else:
    print("=> 0 thinking chars, as expected on bedrock-mantle: a thinking block may")
    print("   open and close, but its text is encrypted (see section 2). Budget for")
    print("   the tokens; do not build a UI that expects to display them.")'''

CACHE_LABELS = '''cold = cached_ask("Which tier suits evaluations? One line.")
print("call 1:", {k: v for k, v in cold.items() if k != "text"})
print("   answer:", cold["text"][:90])

warm = cached_ask("What must untagged usage expect? One line.")
print("\\ncall 2:", {k: v for k, v in warm.items() if k != "text"})
print("   answer:", warm["text"][:90])

# Describe what happened rather than labelling the calls cold/warm in advance. The
# cache survives between runs of this notebook, so on any re-run BOTH calls read
# from it and a hardcoded "call 1 (cold)" is simply wrong.
print()
if cold["cache_write"] and warm["cache_read"]:
    print(f"call 1 wrote {cold['cache_write']} tokens, call 2 read {warm['cache_read']}"
          " back — a cold write then a warm read.")
elif warm["cache_read"]:
    print(f"both calls read {warm['cache_read']} tokens from cache: the handbook was")
    print("already warm from an earlier run. Caching persists across processes,")
    print("which is exactly why it pays in production.")
else:
    print("no cache activity — check the prefix clears the per-checkpoint minimum.")'''

TTL_TAIL = '''for ttl in (None, "5m", "1h"):
    label = ttl or "default (5m)"
    try:
        result = cached_ask("Name one retry rule. One line.", ttl=ttl)
        print(
            f"  ttl={label:12} -> write={result['cache_write']} "
            f"read={result['cache_read']}"
        )
    except RuntimeError as exc:
        print(f"  ttl={label:12} -> {exc}")

print()
print("All three are accepted. The rows look identical because they all hit the")
print("same warm entry -- a TTL governs how long an unread entry SURVIVES, which a")
print("sequence of back-to-back calls cannot show. The 1h option matters when the")
print("next turn may be more than five minutes away, not for throughput.")'''

CONVERSE_XREF = """## Converse in earnest — the tool loop, provider parameters, and caching

`01-messages-api-core.ipynb` §13 establishes that Claude answers through Converse on
`bedrock-runtime`. That is the easy part. This section does the three things you
actually need there, because each differs from the `bedrock-mantle` equivalent:

1. **A complete tool round trip** — `toolUse` out, `toolResult` back in. Getting a
   tool *call* is half the job; feeding the result back is where the shapes bite.
2. **`additionalModelRequestFields`** — Converse normalises the common fields, so
   anything provider-specific goes through this escape hatch.
3. **`cachePoint`** — prompt caching is a first-class Converse block, and support
   for it is per model rather than universal."""

# --------------------------------------------------------------- 03 agentic ---

BETA_MD = """## 1. How beta tools are enabled on mantle

Two things have to line up: the **beta header** and a **tool `type` that matches
that beta version**. The failure is one-sided, which is worth seeing — a tool type
without its header is a 400, but the header on its own is harmless."""

BETA_TAIL = '''    code, data = post(f"{PREFIX}/messages", body, region=REGION, headers=headers)
    print(f"  {label:26} -> HTTP {code} {'' if code == 200 else err(data)[:70]}")

print()
print("=> The tool type without its beta header is rejected, and the message names")
print("   the tag it could not match. The header without a matching tool is simply")
print("   ignored, so an unused anthropic-beta value costs nothing.")'''

EDITOR_MD = """## 5. Bash and text-editor tools

The same pattern with different tool types. These pair with the computer-use beta.

**The version suffix is part of the contract**, and a stale one is the most common
way this fails. `text_editor_20250124` is refused by the current models; the tool is
`text_editor_20250728`, and its `name` changed too — `str_replace_based_edit_tool`,
not `str_replace_editor`. The cell below sends both so you can see the refusal and
the fix side by side, and note that **the error lists the types the model does
accept**, which is the quickest way to find the current one."""

EDITOR_CODE = '''bash_tool = {"type": "bash_20250124", "name": "bash"}
editor_stale = {"type": "text_editor_20250124", "name": "str_replace_editor"}
editor_current = {
    "type": "text_editor_20250728",
    "name": "str_replace_based_edit_tool",  # the name is version-specific too
}

for label, tool, prompt in [
    ("bash", bash_tool, "List the files in the current directory."),
    ("editor (stale type)", editor_stale, "Open /tmp/notes.txt and show its contents."),
    ("editor (current)", editor_current, "Open /tmp/notes.txt and show its contents."),
]:
    code, data = post(
        f"{PREFIX}/messages",
        {
            "model": OPUS48,
            "max_tokens": 500,
            "tools": [tool],
            "messages": [{"role": "user", "content": prompt}],
        },
        region=REGION,
        headers={**AV, "anthropic-beta": COMPUTER_USE_BETA},
    )
    print(f"{label:22} -> HTTP {code} | stop={data.get('stop_reason')}")
    for block in tool_uses(data):
        print(f"   {block['name']}: {json.dumps(block['input'])[:130]}")
    if code != 200:
        print(f"   {err(data)[:160]}")

print()
print("=> The refusal for the stale type lists the versions this model accepts.")
print("   Read it rather than guessing: these suffixes move with each beta.")'''

COMPACTION_MD = """## 7. Context management (compaction)

Long tool-using agents accumulate enormous tool-result history. `context_management`
lets Claude clear old tool calls so the context stays affordable.

The saving only shows up in a **comparison**, so the cell below runs the same
conversation twice — once with `clear_tool_uses` and once without — and reports the
input tokens each round. Watching one compacted run on its own tells you nothing:
its `input_tokens` still grows, because the current turn's tool results are new
every time. What compaction removes is the *older* ones."""

COMPACTION_CODE = '''# Build a conversation with several rounds of bulky tool results, twice: with
# compaction and without. The difference between the two is the whole point.
bulky_tool = {
    "name": "fetch_log",
    "description": "Fetch a log extract.",
    "input_schema": {
        "type": "object",
        "properties": {"service": {"type": "string"}},
        "required": ["service"],
    },
}


def fetch_log(service: str) -> dict:
    # Deliberately verbose, like a real log tool.
    return {
        "service": service,
        "lines": [
            f"{service} INFO handled request {i} in {10 + i}ms" for i in range(60)
        ],
    }


def run_agent(compact: bool, rounds: int = 4) -> list:
    """Return the input_tokens seen on each round."""
    conversation = [
        {
            "role": "user",
            "content": "Check the logs for api, worker, scheduler and gateway one "
            "at a time, then tell me which is slowest.",
        }
    ]
    seen = []
    for _ in range(rounds):
        body = {
            "model": OPUS48,
            "max_tokens": 900,
            "tools": [bulky_tool],
            "messages": conversation,
        }
        if compact:
            body["context_management"] = {
                "edits": [{"type": "clear_tool_uses_20250919"}]
            }
        code, data = post(
            f"{PREFIX}/messages", body, region=REGION,
            headers={**AV, "anthropic-beta": CONTEXT_MGMT_BETA},
        )
        if code != 200:
            print("HTTP", code, err(data)[:110])
            break
        seen.append((data.get("usage") or {}).get("input_tokens"))
        blocks = tool_uses(data)
        if not blocks:
            break
        conversation.append({"role": "assistant", "content": data["content"]})
        conversation.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": b["id"],
                        "content": json.dumps(fetch_log(**b["input"])),
                    }
                    for b in blocks
                ],
            }
        )
    return seen


with_compaction = run_agent(compact=True)
without = run_agent(compact=False)

print(f"{'round':>6} {'with compaction':>17} {'without':>10}")
print("-" * 36)
for i in range(max(len(with_compaction), len(without))):
    a = with_compaction[i] if i < len(with_compaction) else None
    b = without[i] if i < len(without) else None
    print(f"{i + 1:>6} {a!s:>17} {b!s:>10}")

print()
if with_compaction and without and len(with_compaction) > 1 and len(without) > 1:
    if max(with_compaction) < max(without):
        print(f"=> peak input_tokens {max(with_compaction)} with compaction vs "
              f"{max(without)} without.")
    else:
        print("=> no clear saving on this run. Compaction only bites once there is")
        print("   stale tool history to drop, which needs enough rounds to")
        print("   accumulate -- and the model decides when to iterate.")
print("   Both runs grow: the CURRENT turn's tool results are always new. What")
print("   clear_tool_uses removes is the older ones.")'''


def fix_01() -> None:
    nb = Notebook(f"{REPO}/02-anthropic-claude/01-messages-api-core.ipynb")
    old = nb.find('''            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": "What problem does a write-ahead log solve?",
                }''')
    assert len(old) == 1, old
    nb.set_source(old[0], BLOCK_READ)

    old = nb.find("## 8. Count tokens before you spend them")
    assert len(old) == 1, old
    nb.set_source(old[0], COUNT_TOKENS_MD)

    # Insert §8b after the second count_tokens cell.
    anchor = nb.find("print(f\"bare prompt          : {bare['input_tokens']} tokens\")")
    assert len(anchor) == 1, anchor
    at = nb.insert_after(anchor[0], "markdown", COUNT_TOKENS_RUNTIME_MD)
    nb.insert_after(at, "code", COUNT_TOKENS_RUNTIME)

    nb.sub("| `temperature` | Acceptance is per model and changes — probe it (§3) before sending it |",
           "| `temperature` | Acceptance is per model and changes — probe it (§4) before sending it |")
    nb.sub("| `count_tokens` | mantle-only; counts system + tool definitions too |",
           "| `count_tokens` | On mantle for the current generation; on `bedrock-runtime` only for Claude **4.x**, and it refuses inference profiles (§8b) |")
    nb.save()
    print(f"02-anthropic-claude/01: {nb.changes} edits")


def fix_02() -> None:
    nb = Notebook(f"{REPO}/02-anthropic-claude/02-thinking-tools-and-caching.ipynb")

    old = nb.find('def thinking_text(payload: dict) -> str:')
    assert len(old) == 1, old
    body = nb.source(old[0])
    start = body.index("def thinking_text")
    end = body.index('print("endpoint:"')
    nb.set_source(old[0], body[:start] + THINKING_HELPER + "\n\n\n" + body[end:])

    old = nb.find("## 2. Reading a thinking response")
    assert len(old) == 1, old
    nb.set_source(old[0], THINKING_MD)

    old = nb.find('''PUZZLE = (
    "A farmer must cross a river with a wolf, a goat and a cabbage. The boat "''')
    assert len(old) == 1, old
    nb.set_source(old[0], THINKING_CODE)

    # §2b: the enabled-vs-adaptive error.
    at = nb.insert_after(old[0], "markdown", THINKING_ENABLED_MD)
    nb.insert_after(at, "code", THINKING_ENABLED)

    old = nb.find("## 4. Streaming a thinking response")
    assert len(old) == 1, old
    nb.set_source(old[0], STREAM_MD)

    nb.sub('''print(f"\\n\\nthinking chars: {thinking_chars} | answer chars: {text_chars}")
print("--- event types ---")
for name, count in sorted(events.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4}  {name}")''', STREAM_TAIL)

    old = nb.find('cold = cached_ask("Which tier suits evaluations? One line.")')
    assert len(old) == 1, old
    body = nb.source(old[0])
    start = body.index('cold = cached_ask(')
    nb.set_source(old[0], body[:start] + CACHE_LABELS)

    old = nb.find('for ttl in (None, "5m", "1h"):')
    assert len(old) == 1, old
    nb.set_source(old[0], TTL_TAIL)

    nb.sub("## 10. Count tokens before you spend them\n\n`count_tokens` is mantle-only and includes system prompts and tool definitions —\nexactly the parts people forget when budgeting.",
           "## 10. Count tokens before you spend them\n\n`count_tokens` includes system prompts and tool definitions — exactly the parts\npeople forget when budgeting. It is on both endpoints with different model\ncoverage; `01-messages-api-core.ipynb` §8 and §8b have the matrix.")
    nb.sub("| `count_tokens` | mantle-only; counts system + tools |",
           "| `count_tokens` | Counts system + tools. On both endpoints, different coverage — see `01` §8b |")
    nb.sub("| Adaptive thinking | **Not on `haiku-4-5`** — 400. Gate per model |",
           "| Adaptive thinking | **Not on `haiku-4-5`** — 400. Gate per model |\n"
           "| Thinking text | Never readable on mantle: the block's `thinking` is `\"\"` and `signature` holds it encrypted. Cost is in `thinking_tokens` |\n"
           "| `thinking.type` | `\"enabled\"` + `budget_tokens` is refused by Claude 5; the error names `adaptive` and `output_config.effort` (§2b) |")

    old = nb.find("## Converse in earnest — the tool loop, provider parameters, and caching")
    assert len(old) == 1, old
    nb.set_source(old[0], CONVERSE_XREF)
    nb.save()
    print(f"02-anthropic-claude/02: {nb.changes} edits")


def fix_03() -> None:
    nb = Notebook(f"{REPO}/02-anthropic-claude/03-agentic-computer-use-and-memory.ipynb")

    old = nb.find("## 1. How beta tools are enabled on mantle")
    assert len(old) == 1, old
    nb.set_source(old[0], BETA_MD)

    nb.sub('''    code, data = post(f"{PREFIX}/messages", body, region=REGION, headers=headers)
    print(f"  {label:26} -> HTTP {code} {'' if code == 200 else err(data)[:70]}")''',
           BETA_TAIL)

    # The opus-5 refusal quotes an internal model alias. Say so, rather than
    # leaving a reader to wonder what "claude-honey" is.
    nb.sub("""## 2. Which models support computer use?

Support varies by model. Probe before you build.""",
           """## 2. Which models support computer use?

Support varies by model. Probe before you build.

One aside on the output: the refusal quotes an internal service-side alias for the
model rather than the ID you sent. That is a reminder that error *strings* are not a
stable contract even when the behaviour they describe is — match on status codes and
documented error types, not on message text.""")

    old = nb.find("## 5. Bash and text-editor tools")
    assert len(old) == 1, old
    nb.set_source(old[0], EDITOR_MD)

    old = nb.find('bash_tool = {"type": "bash_20250124", "name": "bash"}')
    assert len(old) == 1, old
    nb.set_source(old[0], EDITOR_CODE)

    old = nb.find("## 7. Context management (compaction)")
    assert len(old) == 1, old
    nb.set_source(old[0], COMPACTION_MD)

    old = nb.find("# Build a conversation with several rounds of bulky tool results, then compact it.")
    assert len(old) == 1, old
    nb.set_source(old[0], COMPACTION_CODE)

    nb.sub("""Watch `input_tokens` across rounds. Without compaction it grows with every bulky
tool result; with `clear_tool_uses` Claude can drop stale ones. Combine this with
prompt caching (see `02-thinking-tools-and-caching.ipynb`) and a long agent run
stays affordable.""",
           """Both columns grow, because each round adds a fresh tool result. The question is
whether the *older* ones are still being paid for, and that is what the two columns
compare. Combine compaction with prompt caching (see
`02-thinking-tools-and-caching.ipynb`) and a long agent run stays affordable.

If the two columns look the same, the run did not accumulate enough stale history
to matter — which is itself worth knowing before you enable it and assume a saving.""")

    nb.sub("""| Per-user memory isolation | One user's memory must never leak into another's |""",
           """| Per-user memory isolation | One user's memory must never leak into another's |
| Beta tool type versions | The suffix and the `name` both move per beta; a stale pair is a 400 that lists the valid types |""")
    nb.save()
    print(f"02-anthropic-claude/03: {nb.changes} edits")


if __name__ == "__main__":
    fix_01()
    fix_02()
    fix_03()
