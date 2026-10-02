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
# # Server-side tools, batch, and fine-tuning — OpenAI GPT on Bedrock Mantle
#
# Everything in this family that Bedrock runs **for** you, rather than in your
# client loop:
#
# - **Built-in `notes` and `tasks` tools** on gpt-oss — no declaration needed
# - **Custom Lambda tools** via the MCP (Model Context Protocol) connector — Bedrock
#   invokes your Lambda
# - **AgentCore Gateway** as a tool source
# - **Batch inference** — which lives on `bedrock-runtime`, *not* mantle
# - **Reinforcement fine-tuning** via the OpenAI-compatible Files + jobs APIs
#
# **Prerequisites:** `01-responses-api-core.ipynb` and
# `03-tools-and-structured-output.ipynb` (client-side tools, for contrast).
#
# ## What "server-side" changes
# With client-side tools you receive a `function_call`, execute it, and send the
# result back. With server-side tools **Bedrock calls the tool itself** and returns
# a finished answer. Less orchestration in your app; the tool must live somewhere
# Bedrock can reach.
#
# ## Self-contained, but see also
# - **Auth, the three URL paths** →
#   `../00-foundations/01-endpoints-auth-and-the-three-paths.ipynb`
# - **Projects, retention/ZDR** →
#   `../00-foundations/02-governance-projects-and-retention.ipynb`
#
# ## Prerequisites
# ```bash
# pip install -r ../requirements.txt
# ```
#
# Needs openai, boto3, aws-bedrock-token-generator.
#
# `requirements.txt` pins the exact versions this collection was tested
# against. An unpinned install resolves whatever is current, which may be
# untested or compromised (OWASP LLM03, Supply Chain).

# %%
import json
import sys
import time

sys.path.insert(0, "../_shared")
from mantle import err, list_models, parse_json_lenient, post, response_text

REGION = "us-east-1"

SOL = "openai.gpt-5.6-sol"
OSS120 = "openai.gpt-oss-120b"
OSS20 = "openai.gpt-oss-20b"
SAFEGUARD20 = "openai.gpt-oss-safeguard-20b"
SAFEGUARD120 = "openai.gpt-oss-safeguard-120b"

GPT5_PREFIX = "/openai/v1"
OSS_PREFIX = "/v1"

print("endpoint:", f"https://bedrock-mantle.{REGION}.api.aws")

# %% [markdown]
# ## 1. Built-in `notes` and `tasks` tools
#
# `gpt-oss-20b` and `gpt-oss-120b` ship with two AWS-provided tools. You do **not**
# declare them — just reference them in the prompt. Memory is scoped to the
# conversation.

# %%
code, data = post(
    f"{OSS_PREFIX}/responses",
    {
        "model": OSS120,
        "input": "Use the notes tool to store that my preferred language is Python.",
        "max_output_tokens": 400,
    },
    region=REGION,
)
print("HTTP", code)
print("output item types:", [i.get("type") for i in data.get("output", [])])
for item in data.get("output", []):
    if item.get("type") == "mcp_call":
        print("\nmcp_call:")
        print("  name     :", item.get("name"))
        print("  arguments:", json.dumps(item.get("arguments"))[:200])
        print("  output   :", json.dumps(item.get("output"))[:200])

# %% [markdown]
# The `mcp_call` item is the server-side tool invocation. Notice there is no
# `function_call` for you to service — Bedrock already ran it.

# %%
# The tasks tool: a stack for managing work within a conversation.
code, data = post(
    f"{OSS_PREFIX}/responses",
    {
        "model": OSS120,
        "input": "Use the tasks tool to push a task: review the API documentation.",
        "max_output_tokens": 400,
    },
    region=REGION,
)
print("HTTP", code, "| items:", [i.get("type") for i in data.get("output", [])])
print("answer:", response_text(data)[:200])

# %%
# Both tools are conversation-scoped, so continuity needs previous_response_id.
first = post(
    f"{OSS_PREFIX}/responses",
    {
        "model": OSS120,
        "input": "Use notes to remember: our region is us-east-1.",
        "max_output_tokens": 300,
        "store": True,
    },
    region=REGION,
)[1]
recall = post(
    f"{OSS_PREFIX}/responses",
    {
        "model": OSS120,
        "input": "Check your notes: which region did I mention?",
        "previous_response_id": first.get("id"),
        "max_output_tokens": 300,
    },
    region=REGION,
)[1]
print("recall:", response_text(recall)[:200])

# %%
# Not every model has them. gpt-5.6 has no built-in notes/tasks.
for model, prefix in ((OSS120, OSS_PREFIX), (SOL, GPT5_PREFIX)):
    code, data = post(
        f"{prefix}/responses",
        {
            "model": model,
            "input": "Use the notes tool to store: test.",
            "max_output_tokens": 300,
        },
        region=REGION,
    )
    kinds = [i.get("type") for i in data.get("output", [])]
    print(
        f"{model:24} -> HTTP {code} | items={kinds} "
        f"| used a tool: {'mcp_call' in kinds}"
    )

# %% [markdown]
# ## 2. Custom Lambda tools via the MCP connector
#
# You give Bedrock a **Lambda ARN**; Bedrock discovers the tools it exposes and
# invokes them during inference. The Lambda speaks JSON-RPC (`tools/list`,
# `tools/call`).
#
# **The critical IAM rule:** the Lambda must carry the *same IAM policy* as the
# application calling the model, or invocation fails.

# %%
LAMBDA_HANDLER = '''
import json


def lambda_handler(event, context):
    """Minimal MCP tool server: implements tools/list and tools/call."""
    method = event.get("method")
    params = event.get("params", {})
    request_id = event.get("id")

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": request_id,
            "result": {"tools": [{
                "name": "get_order_status",
                "description": "Look up the status of a customer order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            }]},
        }

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "get_order_status":
            statuses = {"88213": "shipped", "88214": "processing"}
            order_id = args.get("order_id")
            status = statuses.get(order_id, "not_found")
            payload = {"order_id": order_id, "status": status}
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload)}]
                },
            }

    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32601, "message": "Method not found"}}
'''
print(LAMBDA_HANDLER)

# %% [markdown]
# ### Wiring it up
#
# ```python
# response = client.responses.create(
#     model="openai.gpt-oss-120b",
#     tools=[{
#         "type": "mcp",
#         "server_label": "orders",
#         "connector_id": "arn:aws:lambda:us-east-1:123456789012:function:my-mcp-tool",
#         "require_approval": "never",     # must be "never"
#     }],
#     input="What is the status of order 88213?",
# )
# ```
#
# On success the output contains an `mcp_list_tools` item (what Bedrock discovered)
# and `mcp_call` items (what it invoked). No credentials are passed — Bedrock uses
# the caller's IAM identity.
#
# We do **not** deploy a Lambda in this notebook. Instead, verify that the request
# *shape* is accepted, using a well-formed but non-existent ARN, so you can see
# exactly which error means "shape wrong" versus "ARN wrong".

# %%
import boto3

account_id = boto3.client("sts").get_caller_identity()["Account"]
fake_arn = (
    f"arn:aws:lambda:{REGION}:{account_id}:function:mantle-samples-does-not-exist"
)

for label, tool in [
    (
        "well-formed mcp tool",
        {
            "type": "mcp",
            "server_label": "orders",
            "connector_id": fake_arn,
            "require_approval": "never",
        },
    ),
    (
        "missing require_approval",
        {"type": "mcp", "server_label": "orders", "connector_id": fake_arn},
    ),
    (
        "require_approval=always",
        {
            "type": "mcp",
            "server_label": "orders",
            "connector_id": fake_arn,
            "require_approval": "always",
        },
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
    print(f"  {label:26} -> {verdict}")
    if code not in (200, -1):
        print(f"      {err(data)[:120]}")

# %% [markdown]
# Read these carefully: a **schema** complaint means your request shape is wrong,
# while a **not-found / access** complaint means the shape was fine and Bedrock
# genuinely tried to reach your Lambda. That distinction is the fastest way to debug
# a server-side tool.

# %% [markdown]
# ## 3. AgentCore Gateway as a tool source
#
# Same request shape, but `connector_id` is an **AgentCore Gateway ARN** instead of
# a Lambda ARN. Bedrock routes tool calls through the gateway, which gives you
# centralised tool management and discovery.
#
# ```python
# tools=[{
#     "type": "mcp",
#     "server_label": "agentcore_tools",
#     "connector_id": "arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/my-gw",
#     "server_description": "Gateway providing internal tools",
#     "require_approval": "never",
# }]
# ```
#
# Unlike the Lambda path, gateway tools work with **all** Responses-API models and
# support multiple tool calls per turn.

# %%
fake_gateway = f"arn:aws:bedrock-agentcore:{REGION}:{account_id}:gateway/does-not-exist"
code, data = post(
    f"{OSS_PREFIX}/responses",
    {
        "model": OSS120,
        "input": "What tools do you have?",
        "tools": [
            {
                "type": "mcp",
                "server_label": "agentcore_tools",
                "connector_id": fake_gateway,
                "server_description": "Gateway providing internal tools",
                "require_approval": "never",
            }
        ],
        "max_output_tokens": 300,
    },
    region=REGION,
    attempts=1,
    timeout=90,
)
print(f"gateway connector shape -> {code if code != -1 else 'stalled'}")
if code not in (200, -1):
    print("  ", err(data)[:150])

# %% [markdown]
# ## 4. The safeguard models
#
# `gpt-oss-safeguard-20b` and `-120b` are tuned for policy classification. Give a
# policy in the system prompt and constrain the output.

# %%
POLICY = (
    "You are a content classifier. Policy: FLAG any request that seeks "
    "personalised medical, legal, or financial advice. Otherwise ALLOW. "
    "Reply with exactly one word: ALLOW or FLAG."
)

samples = [
    "What dosage of ibuprofen should I take for my back pain?",
    "What is the capital of Portugal?",
    "Should I sue my landlord over my broken boiler?",
    "How does a diesel engine work?",
]

print(f"{'verdict':>8}  request")
print("-" * 74)
for text in samples:
    code, data = post(
        f"{OSS_PREFIX}/chat/completions",
        {
            "model": SAFEGUARD20,
            "messages": [
                {"role": "system", "content": POLICY},
                {"role": "user", "content": text},
            ],
            "max_tokens": 400,
        },
        region=REGION,
    )
    verdict = (
        (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    ).strip()[:12]
    print(f"{verdict:>8}  {text[:60]}")

# %%
# Structured classification is easier to act on than a bare word.
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["allow", "flag"]},
        "category": {
            "type": "string",
            "enum": ["medical", "legal", "financial", "none"],
        },
        "reason": {"type": "string"},
    },
    "required": ["verdict", "category", "reason"],
    "additionalProperties": False,
}

code, data = post(
    f"{OSS_PREFIX}/chat/completions",
    {
        "model": SAFEGUARD120,
        "messages": [
            {"role": "system", "content": POLICY},
            {"role": "user", "content": samples[0]},
        ],
        "max_tokens": 700,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "strict": True,
                "schema": CLASSIFY_SCHEMA,
            },
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
# ## 5. Batch inference lives on `bedrock-runtime`, not mantle
#
# A genuine trap. The OpenAI-compatible **Batch API is not on the mantle endpoint**
# — it is on `bedrock-runtime` under `/openai/v1/batches`. Verify:

# %%
for label, path in [
    ("mantle /v1/batches", "/v1/batches"),
    ("mantle /openai/v1/batches", "/openai/v1/batches"),
]:
    code, data = post(path, None, region=REGION, method="GET", attempts=1, timeout=45)
    print(
        f"  {label:28} -> {code if code != -1 else 'stalled'} "
        f"{err(data)[:60] if code not in (200, -1) else ''}"
    )


# %%
# The real batch endpoint is on bedrock-runtime.
import urllib.error
import urllib.request


def open_https(req, timeout: int):
    """urlopen restricted to HTTPS.

    urllib also honours file://, ftp:// and data:// . These URLs are all built
    from literals, but a client that ever takes a URL from data would let those
    schemes read local files, so the guard belongs in the helper (CWE-22).
    """
    if not req.full_url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS URL: {req.full_url[:60]}")
    # nosemgrep: dynamic-urllib-use-detected - scheme verified https above
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310  # noqa: S310


from aws_bedrock_token_generator import provide_token

runtime_url = f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1/batches"
request = urllib.request.Request(
    runtime_url, headers={"Authorization": f"Bearer {provide_token(region=REGION)}"}
)
try:
    with open_https(request, timeout=60) as resp:
        print(f"GET bedrock-runtime /openai/v1/batches -> {resp.status}")
        print("  body:", resp.read().decode()[:200])
except urllib.error.HTTPError as exc:
    print(f"GET bedrock-runtime /openai/v1/batches -> {exc.code}")
    print("  body:", exc.read().decode()[:200])

# %% [markdown]
# ### Submitting a batch job (shape)
#
# ```python
# client = OpenAI(
#     base_url=f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
#     api_key=provide_token(region=REGION),
#     default_headers={
#         "X-Amzn-Bedrock-RoleArn": "arn:aws:iam::123456789012:role/BatchServiceRole",
#     },
# )
# job = client.batches.create(
#     # Replace with YOUR bucket. Never reference a generic name like
#     # "my-bucket": bucket names are globally unique, so a generic name may
#     # be owned by someone else (S3 bucket squatting).
#     input_file_id="s3://YOUR_BUCKET_NAME/batch-input.jsonl",  # S3 URI, not a file ID
#     endpoint="/v1/chat/completions",                     # must be this
#     completion_window="24h",
#     extra_headers={"X-Amzn-Bedrock-ModelId": "openai.gpt-oss-20b"},
# )
# ```
#
# Bedrock-specific requirements:
# - `X-Amzn-Bedrock-RoleArn` — a batch service role that can read/write your S3
# - `X-Amzn-Bedrock-ModelId` — the model, in a header rather than the body
# - `input_file_id` is an **S3 URI**
# - `endpoint` must be `/v1/chat/completions`
# - Results land beside the input, in a folder named after the batch ID
#
# **Batch does not support tool calling or `response_format`** — each record is
# processed independently, so anything needing a round trip is unavailable.

# %% [markdown]
# ## 6. Fine-tuning with the OpenAI-compatible APIs
#
# Reinforcement fine-tuning (RFT (reinforcement fine-tuning)) on mantle uses the Files
# API plus the fine-tuning
# jobs API. Both are control-plane paths under `/v1`.
#
# **Availability is narrow:** `openai.gpt-oss-20b` and `qwen.qwen3-32b`, in
# **us-west-2 only**.

# %%
FT_REGION = "us-west-2"  # the only fine-tuning Region

for label, path in [
    ("files", "/v1/files"),
    ("fine-tuning jobs", "/v1/fine_tuning/jobs"),
]:
    code, data = post(path, None, region=FT_REGION, method="GET")
    count = len(data.get("data", [])) if code == 200 else "-"
    print(f"  GET {path:26} -> HTTP {code} | items={count}")

# %%
# These control-plane paths are NOT under /openai/v1.
for path in ("/openai/v1/files", "/openai/v1/fine_tuning/jobs"):
    code, data = post(
        path, None, region=FT_REGION, method="GET", attempts=1, timeout=45
    )
    print(f"  GET {path:32} -> {code if code != -1 else 'stalled'}")

# %% [markdown]
# ### The five-step RFT workflow
#
# 1. **Upload the training set** — Files API, `purpose="fine-tune"`, JSONL.
# 2. **Define a reward function** — a Lambda that scores model responses.
# 3. **Create the job** — base model, dataset, reward function, hyperparameters.
# 4. **Monitor** — job status, events, and intermediate checkpoints.
# 5. **Run inference** — use the fine-tuned model ID directly on Responses or Chat
#    Completions. No separate deployment step.
#
# Uploading is a multipart POST, so use the OpenAI SDK rather than hand-rolling it:
#
# ```python
# ft = OpenAI(base_url=f"https://bedrock-mantle.us-west-2.api.aws/v1",
#             api_key=provide_token(region="us-west-2"))
#
# uploaded = ft.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")
#
# job = ft.fine_tuning.jobs.create(
#     model="openai.gpt-oss-20b",
#     training_file=uploaded.id,
#     method={"type": "reinforcement",
#             "reinforcement": {"grader": {"type": "lambda",
#                                          "lambda_arn": "arn:aws:lambda:..."}}},
# )
# ```
#
# We show the JSONL shape and validate it locally rather than starting a real
# training run, which would take hours and cost real money.

# %%
TRAINING_ROWS = [
    {
        "messages": [
            {"role": "user", "content": "What is 12 * 12?"},
        ],
        "reference_answer": "144",
    },
    {
        "messages": [
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
        "reference_answer": "Tokyo",
    },
]

jsonl = "\n".join(json.dumps(row) for row in TRAINING_ROWS)
print("train.jsonl:")
print(jsonl)

# Validate locally before spending money on an upload + job.
ok = True
for i, line in enumerate(jsonl.splitlines(), 1):
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        ok = False
        print(f"  line {i}: INVALID (not JSON: {exc.msg})")
        continue
    # Explicit checks, not asserts: `python -O` strips asserts, and validation
    # that matters must survive it.
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        ok = False
        print(f"  line {i}: INVALID (messages must be a non-empty list)")
    elif messages[0].get("role") != "user":
        ok = False
        print(
            f"  line {i}: INVALID (first role must be user, "
            f"got {messages[0].get('role')!r})"
        )
print("\nlocal validation:", "passed" if ok else "failed")

# %%
# A reward-function Lambda: scores a completion against the reference.
REWARD_LAMBDA = '''
import json


def lambda_handler(event, context):
    """Grade one sample. Bedrock passes the completion and the dataset row."""
    completion = (event.get("completion") or "").strip().lower()
    reference = (event.get("reference_answer") or "").strip().lower()

    if not reference:
        return {"score": 0.0, "reason": "no reference answer"}

    if completion == reference:
        score = 1.0
    elif reference in completion:
        score = 0.7          # correct but verbose
    else:
        score = 0.0

    return {"score": score,
            "reason": f"completion={completion[:40]!r} reference={reference!r}"}
'''
print(REWARD_LAMBDA)

# %%
# Which models can actually be fine-tuned, and where?
print(f"{'model':28} " + "  ".join(f"{r:>13}" for r in ("us-east-1", "us-west-2")))
print("-" * 60)
for model in (OSS20, "qwen.qwen3-32b", SOL):
    cells = []
    for region in ("us-east-1", "us-west-2"):
        try:
            present = model in set(list_models(region))
            cells.append("present" if present else "-")
        except RuntimeError as exc:
            # list_models() raises RuntimeError on a non-200. Report it rather
            # than showing "-", which would misread as "model absent".
            cells.append(f"lookup failed: {exc}"[:13])
    print(f"{model:28} " + "  ".join(f"{c:>13}" for c in cells))
print("\nNote: presence in a Region is necessary but not sufficient — fine-tuning")
print("itself is documented for gpt-oss-20b and qwen3-32b in us-west-2 only.")

# %% [markdown]
# ## 7. Choosing between the tool modes
#
# | | Client-side | Server-side (Lambda / Gateway) | Built-in (notes/tasks) |
# |---|---|---|---|
# | Who executes | your code | Bedrock | Bedrock |
# | Orchestration loop | you write it | none | none |
# | Where the tool lives | your process | Lambda / Gateway | AWS |
# | Latency | round trip per call | in-inference | in-inference |
# | Best for | local logic, fast iteration | shared/governed tools | scratch memory |
# | Models | all with tool support | Responses API models | gpt-oss only |

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "server-side-tools-samples",
        "tags": {"Application": "ServerSideToolsDemo", "Environment": "Demo"},
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)


class ServerSideToolClient:
    """Responses client wired for server-side tools, with attribution and retries."""

    def __init__(self, model=OSS120, region=REGION, connector_arn=None, project=None):
        self.model, self.region, self.project = model, region, project
        self.connector_arn = connector_arn
        self.prefix = GPT5_PREFIX if model.startswith("openai.gpt-5.") else OSS_PREFIX

    def ask(self, prompt, max_output_tokens=600):
        body = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max(16, max_output_tokens),
            "store": False,
        }
        if self.connector_arn:
            body["tools"] = [
                {
                    "type": "mcp",
                    "server_label": "tools",
                    "connector_id": self.connector_arn,
                    "require_approval": "never",
                }
            ]
        headers = {"OpenAI-Project": self.project} if self.project else None
        # post() retries 429/5xx with exponential backoff.
        code, data = post(
            f"{self.prefix}/responses", body, region=self.region, headers=headers
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        return {
            "answer": response_text(data),
            "tool_calls": [
                i
                for i in data.get("output", [])
                if i.get("type") in ("mcp_call", "mcp_list_tools")
            ],
        }


bot = ServerSideToolClient(project=project_id)  # no connector: built-ins only
result = bot.ask("Use the notes tool to remember: the on-call rota is weekly.")
print("answer     :", result["answer"][:160])
print("tool events:", [i.get("type") for i in result["tool_calls"]])

# %%
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — server-side tools, batch and fine-tuning
#
# | Gotcha | Detail |
# |---|---|
# | Built-in tools | `notes` / `tasks` are **gpt-oss only**; do not declare them |
# | `mcp_call` vs `function_call` | Server-side calls need no action from you |
# | `require_approval` | Must be `"never"` for the MCP connector |
# | Lambda IAM | Must carry the same policy as the calling application |
# | **Batch** | On **`bedrock-runtime`**, not mantle; `/openai/v1/batches` |
# | Batch limits | No tool calling and no `response_format` |
# | Batch headers | `X-Amzn-Bedrock-RoleArn` + `X-Amzn-Bedrock-ModelId` |
# | Fine-tuning scope | `gpt-oss-20b` + `qwen3-32b`, **us-west-2 only** |
# | Control-plane paths | `/v1/files`, `/v1/fine_tuning/*` — **not** under `/openai/v1` |
# | Conversation-scoped memory | notes/tasks need `previous_response_id` to persist |
#
# ## Where next
# - `01-responses-api-core.ipynb` · `03-tools-and-structured-output.ipynb`
# - `02-web-search-and-grounding.ipynb` — a server-side tool AWS fully operates
# - Claude's agentic tools:
#   `../02-anthropic-claude/03-agentic-computer-use-and-memory.ipynb`
