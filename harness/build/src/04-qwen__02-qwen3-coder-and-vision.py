# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Qwen3 Coder and Vision on Amazon Bedrock Mantle
#
# The specialist members of the Qwen3 family: three coding models and a
# vision-language model. `01-qwen3-core-and-tools.ipynb` covers the general-purpose
# models and the shared API mechanics.
#
# **Models covered in this notebook**
#
# | Model ID | Notes |
# |---|---|
# | `qwen.qwen3-coder-480b-a35b-instruct` | Largest coder, mixture-of-experts (MoE): 480B total / 35B active |
# | `qwen.qwen3-coder-next` | Newest coder generation |
# | `qwen.qwen3-coder-30b-a3b-instruct` | Small coder, MoE 30B total / 3B active |
# | `qwen.qwen3-vl-235b-a22b-instruct` | Vision-language, MoE 235B total / 22B active |
#
# ## Which API? Chat Completions.
# Bare `/v1` path. The Responses API returns 400 for this family — see
# `01-qwen3-core-and-tools.ipynb` §2 for the proof.
#
# ## Self-contained, but see also
# - **Qwen3 core mechanics, tool use, structured output** →
#   `01-qwen3-core-and-tools.ipynb`
# - **Auth, the three URL paths, model discovery** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Projects, retention/ZDR, CloudWatch** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
# - **Quotas, retries, service tiers** →
#   `../00-foundations/03-scaling-tiers-and-latency.ipynb`
#
# ## Prerequisites
# ```bash
# pip install -r ../requirements.txt
# ```
#
# Needs openai, aws-bedrock-token-generator.
#
# `requirements.txt` pins the exact versions this collection was tested
# against. An unpinned install resolves whatever is current, which may be
# untested or compromised (OWASP LLM03, Supply Chain).

# %%
import base64
import json
import struct
import sys
import time
import zlib

sys.path.insert(0, "../_shared")
from mantle import err, list_models, parse_json_lenient, repair_tool_arguments, post

REGION = "us-east-1"

CODER_480B = "qwen.qwen3-coder-480b-a35b-instruct"
CODER_NEXT = "qwen.qwen3-coder-next"
CODER_30B = "qwen.qwen3-coder-30b-a3b-instruct"
VL_235B = "qwen.qwen3-vl-235b-a22b-instruct"

CODERS = [CODER_30B, CODER_NEXT, CODER_480B]

PREFIX = "/v1"  # bare /v1 — not /openai/v1
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}"
print("base URL:", BASE_URL)
print("coders  :", CODERS)
print("vision  :", VL_235B)

# %%
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

# Fresh token — expires in <=12h and cannot be refreshed.
client = OpenAI(api_key=provide_token(region=REGION), base_url=BASE_URL)

# %% [markdown]
# ## 1. Code generation
#
# Start with the obvious: write a function to a spec. Note `temperature` low —
# code generation benefits from less sampling noise, and this family accepts both
# `temperature` and `top_p`.

# %%
SPEC = """Write a Python function `chunk_by_tokens(text, max_tokens, overlap=0)` that:
- splits text into chunks of at most max_tokens whitespace-separated tokens
- supports an overlap of N tokens between consecutive chunks
- raises ValueError if overlap >= max_tokens
- returns a list of strings
Include a short docstring. No explanation outside the code block."""

completion = client.chat.completions.create(
    model=CODER_480B,
    messages=[{"role": "user", "content": SPEC}],
    max_tokens=900,
    temperature=0.2,
)
generated = completion.choices[0].message.content or ""
print(generated[:1400])
print("\nusage:", completion.usage.model_dump_json())

# %% [markdown]
# ## 1b. Verify what the model wrote — without running it
#
# A notebook about coding models should check the code, not just print it. The
# tempting way is `exec()`. **Do not do that.**
#
# Model output is untrusted input ([OWASP LLM05 — Improper Output
# Handling](https://genai.owasp.org/llmrisk/llm052025-improper-output-handling/)),
# and this kernel holds your live AWS credentials. `exec()` here is arbitrary code
# execution against your own account, triggered by text you did not write. It is
# also unnecessary: a specification like the one above is entirely checkable
# **statically**.
#
# `inspect_code()` and `check_spec()` in `../_shared/mantle.py` parse the source
# with `ast` and report what it declares. Nothing is executed.
#
# **To genuinely run generated code** you need real isolation — a container or
# microVM with no credentials, no network egress, and a CPU/memory cap. AWS Lambda
# in a dedicated account, or Bedrock AgentCore's code-interpreter tool, give you
# that. A notebook kernel does not.

# %%
from mantle import check_spec, extract_code_block, inspect_code

code_text = extract_code_block(generated)
info = inspect_code(code_text)

print("parses            :", info["parses"], info["error"])
print("functions defined :", info["functions"])
print("raises            :", sorted(set(info["raises"])))
print("imports           :", sorted(set(info["imports"])))

verdict = check_spec(
    code_text,
    function="chunk_by_tokens",
    params=["text", "max_tokens", "overlap"],
    raises="ValueError",  # the spec demanded this guard
)
print("\nspec compliance:")
for check in ("parses", "defines", "signature", "guard"):
    print(f"  {check:10} {'PASS' if verdict[check] else 'FAIL'}")
print(f"  {'overall':10} {'PASS' if verdict['ok'] else 'FAIL'} {verdict['reason']}")

# %% [markdown]
# ## 2. Compare the coders on the same spec
#
# The interesting question is not "can it code" but "how small can I go".

# %%
SMALL_SPEC = (
    "Write a Python function `is_palindrome(s)` that ignores case, spaces "
    "and punctuation. Return only the code."
)

# Judged statically: does it parse, and does it define the requested signature?
print(f"{'model':44} {'latency':>9} {'out tok':>8} {'parses':>7} {'signature':>10}")
print("-" * 92)
for model in CODERS:
    started = time.perf_counter()
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": SMALL_SPEC}],
            "max_tokens": 500,
            "temperature": 0.2,
        },
        region=REGION,
    )
    elapsed = time.perf_counter() - started
    if code != 200:
        print(f"{model:44} HTTP {code}: {err(data)[:30]}")
        continue
    body = data["choices"][0]["message"]["content"] or ""
    verdict = check_spec(
        extract_code_block(body), function="is_palindrome", params=["s"]
    )
    print(
        f"{model:44} {elapsed:>8.2f}s "
        f"{data['usage']['completion_tokens']:>8} "
        f"{str(verdict['parses']):>7} {str(verdict['signature']):>10}"
    )

# %% [markdown]
# ## 3. Code review and repair
#
# A more realistic coding workload than generation: find the bug, then fix it.

# %%
BUGGY = '''
def average(values):
    """Return the arithmetic mean of a list of numbers."""
    total = 0
    for v in values:
        total += v
    return total / len(values)


def summarise(rows):
    """Return average score per group."""
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row["score"])
    return {g: average(scores) for g, scores in groups.items()}
'''

review = client.chat.completions.create(
    model=CODER_NEXT,
    messages=[
        {
            "role": "user",
            "content": (
                "Review this code. Identify every bug or robustness "
                f"problem as a short bulleted list.\n```python\n{BUGGY}```"
            ),
        }
    ],
    max_tokens=600,
    temperature=0.2,
)
print(review.choices[0].message.content[:900])

# %%
# Now ask for a structured repair plan — easier to act on programmatically.
REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "bugs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "function": {"type": "string"},
                    "problem": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["function", "problem", "severity"],
                "additionalProperties": False,
            },
        },
        "fixed_code": {"type": "string"},
    },
    "required": ["bugs", "fixed_code"],
    "additionalProperties": False,
}

code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": CODER_480B,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the bugs and return fixed code.\n" f"```python\n{BUGGY}```"
                ),
            }
        ],
        "max_tokens": 1800,
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "repair", "strict": True, "schema": REPAIR_SCHEMA},
        },
    },
    region=REGION,
)
choice = (data.get("choices") or [{}])[0]
content = choice.get("message", {}).get("content")
print("HTTP", code, "| finish_reason:", choice.get("finish_reason"))
if content and content.strip():
    repair = parse_json_lenient(content)
    for bug in repair["bugs"]:
        print(f"  [{bug['severity']:6}] {bug['function']}: {bug['problem'][:80]}")
    print("\n--- fixed code (first 500 chars) ---")
    print(repair["fixed_code"][:500])
else:
    print("empty content — raise max_tokens (reasoning consumed the budget)")

# %% [markdown]
# ## 4. ⚠️ Truncated tool-call arguments — a real, reproducible failure
#
# Before building an agent loop, know about this one. `qwen3-coder` can return
# **tool-call arguments that are cut off mid-object** — e.g. `{"path": "calc.py"`
# with no closing brace. It is not intermittent: in testing it reproduced on 10 of
# 10 calls for a simple single-argument tool.
#
# Two consequences:
# 1. `json.loads()` on the arguments raises.
# 2. Echoing the assistant message back **verbatim** in the next request gets a
#    **400** — the API rejects the malformed JSON you just forwarded.
#
# The fix is to repair the arguments before using *or* echoing them.

# %%
probe_tool = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": CODER_480B,
        "messages": [{"role": "user", "content": "Read calc.py using the tool."}],
        "tools": probe_tool,
        "tool_choice": "auto",
        "max_tokens": 600,
        "temperature": 0.2,
    },
    region=REGION,
)
message = data["choices"][0]["message"]
calls = message.get("tool_calls") or []
if calls:
    raw_args = calls[0]["function"]["arguments"]
    print("raw arguments:", repr(raw_args))
    try:
        json.loads(raw_args)
        print("json.loads     : OK")
    except json.JSONDecodeError as exc:
        print(f"json.loads     : FAILS ({exc.msg})")
    print("lenient parse  :", parse_json_lenient(raw_args))
    print("repaired string:", repair_tool_arguments(raw_args))

# %% [markdown]
# `mantle.parse_json_lenient()` closes unbalanced braces and recovers the fields
# that did arrive; `mantle.repair_tool_arguments()` re-serialises them into a string
# that is safe to echo. Use the repaired string in the message you send back:

# %%
if calls:
    repaired_message = {
        "role": "assistant",
        "tool_calls": [
            {
                **c,
                "function": {
                    **c["function"],
                    "arguments": repair_tool_arguments(c["function"]["arguments"]),
                },
            }
            for c in calls
        ],
    }
    follow_up = [
        {"role": "user", "content": "Read calc.py using the tool."},
        repaired_message,
        {
            "role": "tool",
            "tool_call_id": calls[0]["id"],
            "content": json.dumps({"content": "def divide(a, b):\n    return a / b\n"}),
        },
    ]
    code_ok, _ = post(
        f"{PREFIX}/chat/completions",
        {
            "model": CODER_480B,
            "messages": follow_up,
            "tools": probe_tool,
            "max_tokens": 300,
        },
        region=REGION,
    )
    code_bad, bad = post(
        f"{PREFIX}/chat/completions",
        {
            "model": CODER_480B,
            "messages": [
                follow_up[0],
                {k: v for k, v in message.items() if v is not None},
                follow_up[2],
            ],
            "tools": probe_tool,
            "max_tokens": 300,
        },
        region=REGION,
    )
    print(f"echoing REPAIRED arguments -> HTTP {code_ok}")
    print(f"echoing RAW arguments      -> HTTP {code_bad} {err(bad)[:80]}")

# %% [markdown]
# ## 5. An agentic coding loop
#
# Give the model file tools and let it work. This is the shape of a coding agent:
# read, write, run, iterate — with the argument repair from §4 built in.

# %%
VIRTUAL_FS = {
    "calc.py": (
        "def add(a, b):\n"
        "    return a + b\n\n\n"
        "def divide(a, b):\n"
        "    return a / b\n"
    ),
}


def read_file(path: str) -> dict:
    return {"path": path, "content": VIRTUAL_FS.get(path), "exists": path in VIRTUAL_FS}


def write_file(path: str, content: str) -> dict:
    VIRTUAL_FS[path] = content
    return {"path": path, "bytes": len(content), "ok": True}


def check_file(path: str) -> dict:
    """Statically check that divide() guards against a zero divisor.

    This is the agent's feedback signal, and it is deliberately NOT an exec().
    The file content was written by the model; running it would hand the model
    arbitrary code execution in this kernel (OWASP LLM05). `inspect_code()`
    parses it instead and reports what it declares.

    A real coding agent that must run its own output belongs in a sandbox with
    no credentials and no network - Lambda in an isolated account, or the
    AgentCore code-interpreter tool.
    """
    src = VIRTUAL_FS.get(path)
    if src is None:
        return {"ok": False, "error": "no such file"}
    info = inspect_code(src)
    if not info["parses"]:
        return {"ok": False, "error": f"syntax error: {info['error']}"}
    if "divide" not in info["functions"]:
        return {"ok": False, "error": "divide() not defined"}
    raised = set(info["raises"])
    if not raised:
        return {
            "ok": False,
            "error": "divide() has no raise; a zero divisor would surface a "
            "bare ZeroDivisionError",
        }
    return {"ok": True, "note": f"guards by raising {sorted(raised)}"}


# %% [markdown]
# The tool schemas, the dispatch table, and the loop itself. Note the argument
# repair from §4 applied *before* echoing the assistant message back.

# %%
CODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file to the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_file",
            "description": "Statically check a file against its requirements.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

DISPATCH = {"read_file": read_file, "write_file": write_file, "check_file": check_file}

TASK = (
    "The file calc.py fails its checks. Read it, fix divide() so dividing by "
    "zero raises a ValueError with a clear message, write the file back, then "
    "check the file. Use the tools."
)


# %% [markdown]
# Two helpers first: one to echo the assistant turn back safely, one to run a
# single tool call.


# %%
def echo_assistant_turn(calls: list[dict]) -> dict:
    """Rebuild the assistant message with REPAIRED tool arguments.

    Echoing a truncated argument string verbatim is rejected with a 400 (§4), so
    repair before forwarding.
    """
    return {
        "role": "assistant",
        "tool_calls": [
            {
                **c,
                "function": {
                    **c["function"],
                    "arguments": repair_tool_arguments(c["function"]["arguments"]),
                },
            }
            for c in calls
        ],
    }


def run_tool_call(call: dict) -> tuple[str, dict, dict]:
    """Execute one tool call. Returns (name, args, result); never raises."""
    name = call["function"]["name"]
    try:
        args = parse_json_lenient(call["function"]["arguments"] or "{}")
    except ValueError:
        args = {}
    fn = DISPATCH.get(name)
    try:
        result = fn(**args) if fn else {"error": f"unknown tool {name}"}
    except Exception as exc:  # a tool must never kill the loop
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return name, args, result


# %% [markdown]
# The loop itself. `max_rounds` is a hard bound — an agent without one can spin
# indefinitely on a task it cannot finish (OWASP LLM10, Unbounded Consumption).


# %%
def coding_agent(task: str, model: str = CODER_480B, max_rounds: int = 8) -> dict:
    """Read/write/check loop over the virtual workspace, bounded by max_rounds."""
    convo = [{"role": "user", "content": task}]
    trace = []
    for round_no in range(max_rounds):
        code, data = post(
            f"{PREFIX}/chat/completions",
            {
                "model": model,
                "messages": convo,
                "tools": CODE_TOOLS,
                "tool_choice": "auto",
                "max_tokens": 1500,
                "temperature": 0.2,
            },
            region=REGION,
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        message = data["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            return {
                "answer": message.get("content") or "",
                "trace": trace,
                "rounds": round_no + 1,
            }
        convo.append(echo_assistant_turn(calls))
        for call in calls:
            name, args, result = run_tool_call(call)
            trace.append(
                {
                    "round": round_no + 1,
                    "tool": name,
                    "args": {
                        k: (v[:40] + "…" if isinstance(v, str) and len(v) > 40 else v)
                        for k, v in args.items()
                    },
                    "result": result,
                }
            )
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result),
                }
            )
    return {"answer": "(max rounds reached)", "trace": trace, "rounds": max_rounds}


# %% [markdown]
# Run it, and inspect the trace round by round.

# %%
agent_result = coding_agent(TASK)
print(f"rounds: {agent_result['rounds']}")
for step in agent_result["trace"]:
    print(f"  r{step['round']} {step['tool']:11} -> {json.dumps(step['result'])[:90]}")
print("\nfinal file:")
print(VIRTUAL_FS["calc.py"])
print("checks now:", check_file("calc.py"))

# %% [markdown]
# ## 6. Vision — Qwen3-VL
#
# Images go into Chat Completions as `image_url` content blocks. Base64 data URLs
# and `s3://` URIs are supported; arbitrary `https://` image URLs are not.


# %%
def make_png(width: int, height: int, rgb: tuple) -> bytes:
    """Solid-colour PNG, no third-party dependencies."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


teal = data_url(make_png(96, 96, (0, 128, 128)))

vision = client.chat.completions.create(
    model=VL_235B,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": teal}},
                {"type": "text", "text": "What colour is this image? One word."},
            ],
        }
    ],
    max_tokens=40,
)
print("model sees:", repr((vision.choices[0].message.content or "").strip()))

# %%
# Very small images are rejected — worth knowing so you don't chase a phantom bug.
tiny = data_url(make_png(1, 1, (255, 0, 0)))
code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": VL_235B,
        "max_tokens": 30,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": tiny}},
                    {"type": "text", "text": "Colour?"},
                ],
            }
        ],
    },
    region=REGION,
)
print(f"1x1 PNG -> HTTP {code} {'ok' if code == 200 else err(data)[:80]}")

# %% [markdown]
# ## 7. Structured extraction from an image
#
# The realistic vision workload: turn a picture into typed data. We build a simple
# two-band chart so the extraction has something to actually read.


# %%
def two_band_png(width: int, height: int, top_rgb: tuple, bottom_rgb: tuple) -> bytes:
    rows = []
    for y in range(height):
        rgb = top_rgb if y < height // 2 else bottom_rgb
        rows.append(b"\x00" + bytes(rgb) * width)
    raw = b"".join(rows)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


bands = data_url(two_band_png(128, 128, (200, 30, 30), (30, 30, 200)))

IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "distinct_colours": {"type": "integer"},
        "colours": {"type": "array", "items": {"type": "string"}},
        "layout": {
            "type": "string",
            "enum": ["horizontal_bands", "vertical_bands", "solid", "other"],
        },
    },
    "required": ["distinct_colours", "colours", "layout"],
    "additionalProperties": False,
}

code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": VL_235B,
        "max_tokens": 700,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": bands}},
                    {
                        "type": "text",
                        "text": "Describe the colours and layout of this image.",
                    },
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "image", "strict": True, "schema": IMAGE_SCHEMA},
        },
    },
    region=REGION,
)
choice = (data.get("choices") or [{}])[0]
content = choice.get("message", {}).get("content")
print("HTTP", code, "| finish_reason:", choice.get("finish_reason"))
if content and content.strip():
    print(json.dumps(parse_json_lenient(content), indent=2))
else:
    print("empty content — raise max_tokens")

# %% [markdown]
# ## 8. Which Qwen3 models take images?
#
# Only the VL variant. Sending an image to a text model is not always a clean
# error, so check rather than assume.

# %%
print(f"{'model':44} {'image accepted':>16}")
print("-" * 62)
for model in [VL_235B, CODER_30B, "qwen.qwen3-32b"]:
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": model,
            "max_tokens": 30,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": teal}},
                        {"type": "text", "text": "Colour? One word."},
                    ],
                }
            ],
        },
        region=REGION,
        attempts=1,
        timeout=60,
    )
    if code == 200:
        answer = (data["choices"][0]["message"]["content"] or "").strip()
        verdict = f"yes ({answer[:12]!r})"
    else:
        verdict = f"{code}" if code != -1 else "stalled"
    print(f"{model:44} {verdict:>16}")

# %% [markdown]
# ## 9. Regional availability

# %%
regions = ("us-east-1", "us-east-2", "us-west-2", "eu-central-1")
inventory = {}
for region in regions:
    try:
        inventory[region] = set(list_models(region))
    except (RuntimeError, OSError) as exc:
        inventory[region] = set()
        print(f"{region}: {type(exc).__name__}")

print(f"{'model':44} " + "  ".join(f"{r:>13}" for r in regions))
print("-" * 104)
for model in CODERS + [VL_235B]:
    cells = "  ".join(
        f"{('yes' if model in inventory[r] else '-'):>13}" for r in regions
    )
    print(f"{model:44} {cells}")

# %% [markdown]
# ## 10. Production shape

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "qwen-coder-samples",
        "tags": {"Application": "QwenCoderDemo", "Environment": "Demo"},
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)


class QwenCoder:
    """Coding helper: low temperature, schema-validated output, retries."""

    def __init__(self, model=CODER_480B, region=REGION, tier="default", project=None):
        self.model, self.region, self.tier, self.project = model, region, tier, project

    def _call(self, messages, max_tokens, schema=None, tools=None):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,  # code wants low sampling noise
            "service_tier": self.tier,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True, "schema": schema},
            }
        headers = {"OpenAI-Project": self.project} if self.project else None
        # post() retries 429/5xx with exponential backoff.
        code, data = post(
            f"{PREFIX}/chat/completions", body, region=self.region, headers=headers
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        return data

    def write_function(
        self,
        spec: str,
        *,
        function: str,
        params: list[str] | None = None,
        max_tokens: int = 900,
    ) -> dict:
        """Generate a function and verify it STATICALLY. Never executes output."""
        data = self._call([{"role": "user", "content": spec}], max_tokens)
        body = data["choices"][0]["message"]["content"] or ""
        source = extract_code_block(body)
        verdict = check_spec(source, function=function, params=params)
        return {"ok": verdict["ok"], "verdict": verdict, "source": source}


coder = QwenCoder(project=project_id)
outcome = coder.write_function(
    "Write a Python function `slugify(s)` that lowercases, replaces non-alphanumerics "
    "with hyphens, and collapses repeats. Return only code.",
    function="slugify",
    params=["s"],
)
print("ok:", outcome["ok"], "|", outcome["verdict"])
print(outcome["source"][:320])

# %%
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — Qwen3 Coder and Vision
#
# | Gotcha | Detail |
# |---|---|
# | Path prefix | Bare `/v1` — Responses API returns 400 for this family |
# | Images | Only `qwen3-vl-*`; base64 or `s3://` only, no `https://` URLs |
# | Tiny images | A 1×1 PNG is rejected as an unsupported format |
# | `content` can be `None` | Check `finish_reason` before slicing/parsing |
# | Strict JSON | Parse leniently — models can append text after a valid object |
# | Code temperature | Use ~0.2; the family accepts `temperature` and `top_p` |
# | **Truncated tool arguments** | `qwen3-coder` truncates mid-object; repair before use AND before echoing |
# | Verify generated code | **Statically** (`ast`) — never `exec()` model output in your kernel |
# | Tool errors | Return structured errors so the agent loop can recover |
# | Region | Check per model; not every coder is in every Region |
#
# ## Where next
# - `01-qwen3-core-and-tools.ipynb` — the general-purpose Qwen3 models
# - Other coding model: `../07-mistral/02-devstral-and-voxtral.ipynb`
# - Other vision models: `../10-nvidia-nemotron/`, `../12-writer-palmyra/`
