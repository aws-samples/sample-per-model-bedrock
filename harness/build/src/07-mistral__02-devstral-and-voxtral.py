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
# # Devstral and Voxtral on Amazon Bedrock Mantle — Mistral's specialists
#
# The two specialised branches of the Mistral family: **Devstral 2** for agentic
# coding, and the **Voxtral** models, which are audio-capable elsewhere in the
# Mistral ecosystem. This notebook shows what each actually does on
# `bedrock-mantle` — including where the audio path does *not* reach.
#
# `01-mistral-text-and-sizes.ipynb` covers the general-purpose text models, the size
# ladder, and the shared API mechanics.
#
# **Models covered in this notebook**
#
# | Model ID | Notes |
# |---|---|
# | `mistral.devstral-2-123b` | Agentic coding, 123B |
# | `mistral.voxtral-small-24b-2507` | Voxtral, 24B |
# | `mistral.voxtral-mini-3b-2507` | Voxtral, 3B |
#
# ## Which API? Chat Completions.
# Bare `/v1` path. The Responses API returns 400 for this family — see
# `01-mistral-text-and-sizes.ipynb` §2.
#
# ## Self-contained, but see also
# - **Mistral text models and cost-aware routing** → `01-mistral-text-and-sizes.ipynb`
# - **Auth, the three URL paths, model discovery** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Projects, retention/ZDR, CloudWatch** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
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
import sys
import time

sys.path.insert(0, "../_shared")
from mantle import err, list_models, parse_json_lenient, repair_tool_arguments, post

REGION = "us-east-1"

DEVSTRAL = "mistral.devstral-2-123b"
VOXTRAL_SMALL = "mistral.voxtral-small-24b-2507"
VOXTRAL_MINI = "mistral.voxtral-mini-3b-2507"

PREFIX = "/v1"  # bare /v1 — not /openai/v1
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}"
print("base URL:", BASE_URL)

# %%
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

# Fresh token — expires within 12h and cannot be refreshed.
client = OpenAI(api_key=provide_token(region=REGION), base_url=BASE_URL)

# %% [markdown]
# ## 1. Devstral 2 — code generation
#
# Devstral is tuned for software engineering rather than general chat. Use a low
# `temperature`: this family accepts both `temperature` and `top_p`, and code
# benefits from less sampling noise.

# %%
SPEC = """Write a Python function `retry(fn, attempts=3, base_delay=0.1)` that:
- calls fn() and returns its result
- retries on any exception up to `attempts` times total
- sleeps base_delay * (2 ** attempt) between attempts
- re-raises the last exception if all attempts fail
Return only code with a short docstring."""

completion = client.chat.completions.create(
    model=DEVSTRAL,
    messages=[{"role": "user", "content": SPEC}],
    max_tokens=900,
    temperature=0.2,
)
generated = completion.choices[0].message.content or ""
print(generated[:1200])
print("\nusage:", completion.usage.model_dump_json())

# %% [markdown]
# ## 1b. Verify it statically — never `exec()` model output
#
# Check the generated code without running it. Model output is untrusted input
# ([OWASP LLM05](https://genai.owasp.org/llmrisk/llm052025-improper-output-handling/)),
# and this kernel holds your live AWS credentials — `exec()` here would be
# arbitrary code execution against your own account, triggered by text you did not
# write.
#
# `inspect_code()` in `../_shared/mantle.py` parses the source with `ast` and
# reports what it declares. To *run* generated code you need real isolation:
# Lambda in a dedicated account, or the Bedrock AgentCore code-interpreter tool.

# %%
from mantle import check_spec, extract_code_block, inspect_code

source = extract_code_block(generated)
info = inspect_code(source)

print("parses            :", info["parses"], info["error"])
print("functions defined :", info["functions"])
print("calls made        :", sorted(set(info["calls"]))[:10])

verdict = check_spec(source, function="retry")
print("\nspec compliance:")
print(f"  parses     {'PASS' if verdict['parses'] else 'FAIL'}")
print(f"  defines    {'PASS' if verdict['defines'] else 'FAIL'} retry()")
# The spec asked for exponential backoff: look for a sleep call statically.
sleeps = [c for c in info["calls"] if "sleep" in c.lower()]
print(f"  backs off  {'PASS' if sleeps else 'FAIL'} (sleep calls: {sleeps})")

# %% [markdown]
# ## 2. Multi-file refactoring with tools
#
# The workload Devstral is actually for: give it a workspace and let it work.
#
# **Note the argument repair.** Coding models sometimes emit truncated tool-call
# arguments, and echoing a malformed string back is a 400. `repair_tool_arguments`
# re-serialises whatever parsed. (Reproduced in detail in
# `../04-qwen/02-qwen3-coder-and-vision.ipynb` §4.)

# %%
WORKSPACE = {
    "config.py": "TIMEOUT = 30\nRETRIES = 3\n",
    "client.py": (
        "import config\n\n\n"
        "def fetch(url):\n"
        "    # TODO: honour config.RETRIES\n"
        "    return _get(url, timeout=config.TIMEOUT)\n\n\n"
        "def _get(url, timeout):\n"
        "    return {'url': url, 'timeout': timeout, 'attempts': 1}\n"
    ),
}


def list_files() -> dict:
    return {"files": sorted(WORKSPACE)}


def read_file(path: str) -> dict:
    return {"path": path, "content": WORKSPACE.get(path), "exists": path in WORKSPACE}


def write_file(path: str, content: str) -> dict:
    WORKSPACE[path] = content
    return {"path": path, "bytes": len(content), "ok": True}


def check_retries(path: str = "client.py") -> dict:
    """Statically check that fetch() implements a retry loop.

    The agent's feedback signal, and deliberately NOT an exec(): the file content
    was written by the model, so running it would grant arbitrary code execution
    in this kernel (OWASP LLM05). An agent that must run its own output belongs
    in a sandbox with no credentials and no network egress.
    """
    source = WORKSPACE.get(path)
    if source is None:
        return {"ok": False, "error": "no such file"}
    info = inspect_code(source)
    if not info["parses"]:
        return {"ok": False, "error": f"syntax error: {info['error']}"}
    if "fetch" not in info["functions"]:
        return {"ok": False, "error": "fetch() not defined"}
    # A retry loop needs iteration and a reference to the RETRIES setting.
    loops = "for" in source or "while" in source
    honours_config = "RETRIES" in source
    return {
        "ok": loops and honours_config,
        "has_loop": loops,
        "references_RETRIES": honours_config,
        "note": "wants a loop bounded by config.RETRIES",
    }


# %% [markdown]
# The tool schemas and dispatch table the agent works through.

# %%
CODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
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
            "description": "Write a file.",
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
            "name": "check_retries",
            "description": "Check whether fetch() retries.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
]

DISPATCH = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "check_retries": check_retries,
}


# %% [markdown]
# Two helpers: one to echo the assistant turn back safely, one to run a single
# tool call.


# %%
def echo_assistant_turn(calls: list[dict]) -> dict:
    """Rebuild the assistant message with REPAIRED tool arguments.

    Echoing a truncated argument string verbatim is rejected with a 400. See
    `../04-qwen/02-qwen3-coder-and-vision.ipynb` §4 for the reproduction.
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


def run_tool_call(call: dict) -> tuple[str, dict]:
    """Execute one tool call. Returns (name, result); never raises."""
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
    return name, result


# %% [markdown]
# The loop, bounded by `max_rounds` — an agent without a bound can spin
# indefinitely on a task it cannot finish (OWASP LLM10, Unbounded Consumption).


# %%
def coding_agent(task: str, model: str = DEVSTRAL, max_rounds: int = 8) -> dict:
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
                "max_tokens": 1600,
                "temperature": 0.2,
            },
            region=REGION,
        )
        if code != 200:
            trace.append({"round": round_no + 1, "error": err(data)[:110]})
            break
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
            name, result = run_tool_call(call)
            trace.append(
                {"round": round_no + 1, "tool": name, "result": json.dumps(result)[:80]}
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
# Give it the task and watch the rounds.

# %%
TASK = (
    "client.py has a TODO: fetch() ignores config.RETRIES. List the files, read "
    "client.py, rewrite fetch() so it retries up to config.RETRIES times, write "
    "the file back, then run check_retries. Use the tools."
)

result = coding_agent(TASK)
print(f"rounds: {result['rounds']}")
for step in result["trace"]:
    print(
        f"  r{step['round']} {step.get('tool', 'ERROR'):14} -> "
        f"{step.get('result', step.get('error'))}"
    )
print("\n--- client.py now ---")
print(WORKSPACE["client.py"][:600])
print("check_retries:", check_retries())

# %% [markdown]
# ## 3. Structured code review
#
# Ask for findings as typed data so a pipeline can act on them.

# %%
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "issue": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["location", "issue", "severity"],
                "additionalProperties": False,
            },
        },
        "overall": {"type": "string", "enum": ["approve", "request_changes"]},
    },
    "required": ["findings", "overall"],
    "additionalProperties": False,
}

SUSPECT = """
import os

def load_secret(name):
    return os.environ[name]

def build_query(table, user_input):
    return f"SELECT * FROM {table} WHERE name = '{user_input}'"

def divide_all(values, divisor):
    return [v / divisor for v in values]
"""

code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": DEVSTRAL,
        "messages": [
            {
                "role": "user",
                "content": f"Review this code for bugs and security issues.\n"
                f"```python\n{SUSPECT}```",
            }
        ],
        "max_tokens": 1600,
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "review", "strict": True, "schema": REVIEW_SCHEMA},
        },
    },
    region=REGION,
)
choice = (data.get("choices") or [{}])[0]
content = choice.get("message", {}).get("content")
print("HTTP", code, "| finish_reason:", choice.get("finish_reason"))
if content and content.strip():
    review = parse_json_lenient(content)
    print("verdict:", review["overall"])
    for finding in review["findings"]:
        print(
            f"  [{finding['severity']:6}] {finding['location']}: "
            f"{finding['issue'][:70]}"
        )
else:
    print("empty content — raise max_tokens (reasoning consumed the budget)")

# %% [markdown]
# ## 4. Voxtral — what works on mantle
#
# Voxtral models are **audio-capable** in the wider Mistral ecosystem. On
# `bedrock-mantle` they are served through Chat Completions, so the first question
# is whether audio content blocks are accepted here at all. Test rather than assume.


# %%
# A tiny valid WAV (silence), built without third-party libraries.
def make_wav(seconds: float = 0.2, rate: int = 8000) -> bytes:
    import struct

    frames = int(seconds * rate)
    data = b"\x00\x00" * frames  # 16-bit silence
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


wav_b64 = base64.b64encode(make_wav()).decode()
print("wav bytes:", len(base64.b64decode(wav_b64)))

audio_shapes = [
    (
        "input_audio block",
        [
            {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
            {"type": "text", "text": "Transcribe this audio."},
        ],
    ),
    (
        "audio_url block",
        [
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"},
            },
            {"type": "text", "text": "Transcribe this audio."},
        ],
    ),
]

for label, content in audio_shapes:
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": VOXTRAL_SMALL,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": content}],
        },
        region=REGION,
        attempts=1,
        timeout=90,
    )
    verdict = "accepted" if code == 200 else (str(code) if code != -1 else "stalled")
    print(f"  {label:22} -> {verdict}")
    if code not in (200, -1):
        print(f"      {err(data)[:110]}")

# %% [markdown]
# ### Read the result honestly
#
# If both shapes are rejected, then on `bedrock-mantle` today these models are
# reachable as **text** models through Chat Completions, and audio input is not part
# of that surface. That is a useful negative result: it stops you designing a
# transcription pipeline around an API that will not accept your audio.
#
# For audio on AWS, look at a purpose-built path (for example Amazon Transcribe) and
# use Voxtral here for the text stages.

# %% [markdown]
# ## 5. Voxtral as text models
#
# They work perfectly well as text models, and the 3B is genuinely fast.

# %%
print(f"{'model':40} {'latency':>9} {'out tok':>8}  answer")
print("-" * 100)
for model in (VOXTRAL_MINI, VOXTRAL_SMALL, DEVSTRAL):
    started = time.perf_counter()
    code, data = post(
        f"{PREFIX}/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "In one sentence, what is speaker diarisation?",
                }
            ],
            "max_tokens": 160,
        },
        region=REGION,
    )
    elapsed = time.perf_counter() - started
    if code != 200:
        print(f"{model:40} HTTP {code}: {err(data)[:40]}")
        continue
    text = " ".join((data["choices"][0]["message"]["content"] or "").split())
    print(
        f"{model:40} {elapsed:>8.2f}s "
        f"{data['usage']['completion_tokens']:>8}  {text[:40]!r}"
    )

# %% [markdown]
# ## 6. A text pipeline for transcripts
#
# The realistic division of labour: something else produces the transcript, and a
# Voxtral model turns it into structured data.

# %%
TRANSCRIPT = """
Agent: Thanks for calling, how can I help?
Customer: My order 88213 arrived with a cracked screen. I want a replacement.
Agent: I'm sorry about that. I can raise a replacement today, arriving Thursday.
Customer: Thursday is fine. Do I need to send the broken one back?
Agent: Yes, a return label will be emailed to you. No cost to you.
Customer: Great, thank you.
"""

CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "issue": {"type": "string"},
        "resolution": {"type": "string"},
        "return_required": {"type": "boolean"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
    },
    "required": ["order_id", "issue", "resolution", "return_required", "sentiment"],
    "additionalProperties": False,
}

code, data = post(
    f"{PREFIX}/chat/completions",
    {
        "model": VOXTRAL_SMALL,
        "messages": [
            {"role": "user", "content": f"Extract the call summary.\n{TRANSCRIPT}"}
        ],
        "max_tokens": 700,
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "call", "strict": True, "schema": CALL_SCHEMA},
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
# ## 7. Regional availability

# %%
regions = ("us-east-1", "us-east-2", "us-west-2", "eu-central-1")
inventory = {}
for region in regions:
    try:
        inventory[region] = set(list_models(region))
    except (RuntimeError, OSError) as exc:
        inventory[region] = set()
        print(f"{region}: {type(exc).__name__}")

print(f"{'model':40} " + "  ".join(f"{r:>13}" for r in regions))
print("-" * 100)
for model in (DEVSTRAL, VOXTRAL_SMALL, VOXTRAL_MINI):
    cells = "  ".join(
        f"{('yes' if model in inventory[r] else '-'):>13}" for r in regions
    )
    print(f"{model:40} {cells}")

# %% [markdown]
# ## 8. Production shape

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "devstral-samples",
        "tags": {"Application": "DevstralDemo", "Environment": "Demo"},
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)


class DevstralAgent:
    """Coding agent with argument repair, tool-error containment and round limits."""

    def __init__(
        self,
        model=DEVSTRAL,
        region=REGION,
        tools=None,
        dispatch=None,
        project=None,
        tier="default",
    ):
        self.model, self.region, self.project, self.tier = model, region, project, tier
        self.tools = tools or CODE_TOOLS
        self.dispatch = dispatch or DISPATCH

    def run(self, task: str, max_rounds: int = 8, max_tokens: int = 1600):
        convo = [{"role": "user", "content": task}]
        headers = {"OpenAI-Project": self.project} if self.project else None
        for _ in range(max_rounds):
            # post() retries 429/5xx with exponential backoff.
            code, data = post(
                f"{PREFIX}/chat/completions",
                {
                    "model": self.model,
                    "messages": convo,
                    "tools": self.tools,
                    "tool_choice": "auto",
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "service_tier": self.tier,
                },
                region=self.region,
                headers=headers,
            )
            if code != 200:
                raise RuntimeError(f"HTTP {code}: {err(data)}")
            message = data["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                return message.get("content") or ""
            convo.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            **c,
                            "function": {
                                **c["function"],
                                "arguments": repair_tool_arguments(
                                    c["function"]["arguments"]
                                ),
                            },
                        }
                        for c in calls
                    ],
                }
            )
            for call in calls:
                try:
                    args = parse_json_lenient(call["function"]["arguments"] or "{}")
                except ValueError:
                    args = {}
                fn = self.dispatch.get(call["function"]["name"])
                try:
                    output = fn(**args) if fn else {"error": "unknown tool"}
                except Exception as exc:
                    output = {"error": f"{type(exc).__name__}: {exc}"}
                convo.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(output),
                    }
                )
        return "(max rounds reached)"


agent = DevstralAgent(project=project_id)
print(agent.run("List the workspace files and tell me which one has a TODO.")[:300])

# %%
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — Devstral and Voxtral on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | Path prefix | Bare `/v1`; Responses API returns 400 for this family |
# | **Audio input** | Check the probe in §4 before designing an audio pipeline on mantle |
# | Truncated tool arguments | Repair before use **and** before echoing, or you get a 400 |
# | `content` can be `None` | Check `finish_reason` before slicing/parsing |
# | Strict JSON | Parse leniently — models can append text after a valid object |
# | Code temperature | Use ~0.2; this family accepts `temperature` and `top_p` |
# | Verify generated code | Execute it — specs get partially missed |
# | Tool errors | Return structured errors so the loop can recover |
# | Quotas | No RPM quota; most models have no published TPM (tokens per minute) — retry with backoff |
#
# ## Where next
# - `01-mistral-text-and-sizes.ipynb` — the size ladder and cost-aware routing
# - Other coding models: `../04-qwen/02-qwen3-coder-and-vision.ipynb`
# - Shared mechanics: `../00-foundations/`
