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
# # Web Search on Amazon Bedrock Mantle — grounding OpenAI GPT models
#
# **Web Search** is a built-in, server-side tool on `bedrock-mantle`. Add one
# parameter and the model can retrieve current information mid-request and cite its
# sources. AWS builds and hosts the index — there is no third-party search provider
# to onboard, no separate API key, and no orchestration loop for you to write.
#
# ## What makes it different from "just add a search tool"
# - The index is **built and operated by Amazon** — tens of billions of documents
#   plus a knowledge graph for high-confidence facts.
# - Retrieval is served **from inside the AWS boundary** by default; your request
#   data does not leave AWS for search.
# - The model runs the whole search loop server-side, possibly **several rounds**,
#   and returns one grounded answer with citations.
#
# ## Availability
# **OpenAI GPT models only**: `gpt-5.4`, `gpt-5.5`, `gpt-5.6-sol/terra/luna`, via the
# **Responses API**. Not available on Gemma 4, Grok, or the open-weight families —
# we prove that in §7. Regions: `us-east-1`, `us-east-2`, `us-west-2` (strictly
# in-Region; queries never cross Region boundaries).
#
# ## Read this before you ship a grounded answer
#
# A grounded response is **model-authored text that your application publishes**.
# The committed outputs in this notebook are real answers captured at run time, and
# like any model output they may paraphrase a source loosely or characterise a
# business relationship in words AWS would not choose. That is exactly why the
# citations are mandatory: they let a human verify the claim.
#
# Two habits follow, and they apply to your code as much as to this notebook:
#
# 1. **Prefer factual technical questions** over prompts about partnerships,
#    competitive positioning, or roadmap. You cannot verify those at run time.
# 2. **Review grounded output before surfacing it**, or constrain it with a system
#    prompt and validate against the citations you were given.
#
# ## Self-contained, but see also
# - **Core Responses API for this family** → `01-responses-api-core.ipynb`
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
import json
import sys

sys.path.insert(0, "../_shared")
from mantle import err, post, response_text, stream_lines

REGION = "us-east-1"  # Web Search: us-east-1 / us-east-2 / us-west-2 only
PREFIX = "/openai/v1"  # gpt-5.x path

GPT56 = "openai.gpt-5.6-sol"
GPT55 = "openai.gpt-5.5"
GPT54 = "openai.gpt-5.4"

print("endpoint:", f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}")

# %% [markdown]
# ## 1. The smallest possible grounded call
#
# One extra entry in `tools` is the whole integration. The model decides whether
# the question actually needs current information.

# %%
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

client = OpenAI(
    api_key=provide_token(region=REGION),
    base_url=f"https://bedrock-mantle.{REGION}.api.aws{PREFIX}",
)

response = client.responses.create(
    model=GPT55,
    input="Name one AWS service launch announced recently. One sentence.",
    tools=[{"type": "web_search", "external_web_access": False}],
    max_output_tokens=400,
)
print(response.output_text)

# %% [markdown]
# ## 2. What actually happened server-side
#
# The `output` array records the search loop. You will typically see alternating
# `reasoning` and `web_search_call` items before the final `message` — the model
# reformulating its query and searching again until it can answer.

# %%
print("output item types, in order:")
for i, item in enumerate(response.output):
    print(f"  {i:2}. {item.type}")

search_calls = [i for i in response.output if i.type == "web_search_call"]
print(f"\nsearch rounds executed server-side: {len(search_calls)}")
print("You wrote no tool-calling loop — Bedrock ran all of it.")

# %% [markdown]
# ## 3. Governance: the `external_web_access` trap
#
# This is the single most important thing in this notebook.
#
# - `external_web_access` **defaults to `true`** (matching the OpenAI spec, so
#   existing code needs no change).
# - But `AmazonBedrockFullAccess` **does not grant**
#   `bedrock-websearch:ExternalWebAccess`.
# - So the default configuration, from a typical identity, **fails the
#   authorization check with a 403** for external access. The request does not
#   fail: the model grounds its answer in the Bedrock index and cached fetches, and
#   tells you it could not reach the external web.
#
# Two clean configurations:
#
# | Goal | Setting | Permission needed |
# |---|---|---|
# | Stay inside the AWS boundary | `external_web_access: false` | none extra |
# | Allow external web reach | leave default `true` | `bedrock-websearch:ExternalWebAccess` |

# %%
for external in (False, True):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GPT55,
            "input": "Name one AWS launch from this year. One sentence.",
            "max_output_tokens": 400,
            "tools": [{"type": "web_search", "external_web_access": external}],
        },
        region=REGION,
    )
    rounds = len(
        [i for i in data.get("output", []) if i.get("type") == "web_search_call"]
    )
    text = response_text(data)
    print(
        f"external_web_access={str(external):5} -> HTTP {code} | "
        f"search rounds={rounds} | chars={len(text)}"
    )
    print(f"   {text[:150]!r}\n")

# %% [markdown]
# Both work here because retrieval is currently served from the Bedrock index
# either way. The setting governs *whether search and fetch may reach the live
# external web* — which matters for your data-boundary posture, and may matter more
# in future releases. **Set it explicitly rather than relying on the default.**

# %%
# The IAM policy for the "stay inside AWS" posture.
websearch_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "WebSearchInBoundary",
            "Effect": "Allow",
            "Action": [
                "bedrock-websearch:InvokeSearch",
                "bedrock-websearch:InvokeFetch",
            ],
            # Resource "*" is required here: these are non-resource-based actions
            # (there is no ARN for "a web search"), the same pattern as
            # ecr:GetAuthorizationToken. Check the IAM service-authorization
            # reference for bedrock-websearch before narrowing it -- if
            # resource-level scoping is ever added, scope it.
            "Resource": "*",
        },
        # Deliberately NOT granting bedrock-websearch:ExternalWebAccess.
        # Pair this with "external_web_access": false on every request.
    ],
}
print(json.dumps(websearch_policy, indent=2))

# %% [markdown]
# ## 4. Citations — and why they are mandatory
#
# Grounded answers carry `url_citation` annotations: title, URL, and the character
# span of the answer they support.
#
# **AWS's acceptable-use terms require you to retain and display these citations**
# in anything you surface to end users. This is not optional polish.

# %%
response = client.responses.create(
    model=GPT55,
    input=(
        "What is the difference between an AWS Region and an Availability Zone? "
        "Answer in two sentences."
    ),
    tools=[{"type": "web_search", "external_web_access": False}],
    max_output_tokens=600,
)

print("=== ANSWER ===")
print(response.output_text[:600])

print("\n=== SOURCES ===")
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
print(f"\n{seen} citation(s)")

# %%
# Same data from a raw HTTP response, for non-SDK callers.
code, data = post(
    f"{PREFIX}/responses",
    {
        "model": GPT55,
        "input": "One recent AWS launch, one sentence.",
        "max_output_tokens": 400,
        "tools": [{"type": "web_search", "external_web_access": False}],
    },
    region=REGION,
)
citations = [
    ann
    for item in data.get("output", [])
    if item.get("type") == "message"
    for block in item.get("content", [])
    for ann in (block.get("annotations") or [])
    if ann.get("type") == "url_citation"
]
print(
    f"HTTP {code} | citations found at output[].content[].annotations[]: "
    f"{len(citations)}"
)
if citations:
    print(json.dumps(citations[0], indent=2)[:400])

# %% [markdown]
# ### A citation renderer
#
# A small helper that turns the response into text plus a numbered source list —
# the shape most applications actually need.


# %%
def render_grounded(payload: dict) -> str:
    """Format a grounded answer with a numbered source list."""
    text = response_text(payload)
    sources, seen_urls = [], set()
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            for ann in block.get("annotations") or []:
                if (
                    ann.get("type") == "url_citation"
                    and ann.get("url") not in seen_urls
                ):
                    seen_urls.add(ann["url"])
                    sources.append((ann.get("title", "(untitled)"), ann["url"]))
    lines = [text.strip(), ""]
    if sources:
        lines.append("Sources:")
        lines += [f"  [{i}] {t[:70]}\n      {u}" for i, (t, u) in enumerate(sources, 1)]
    else:
        lines.append("(no sources cited — the model answered from its own knowledge)")
    return "\n".join(lines)


code, data = post(
    f"{PREFIX}/responses",
    {
        "model": GPT56,
        "input": "What is the newest Amazon Bedrock feature? Two sentences.",
        "max_output_tokens": 600,
        "tools": [{"type": "web_search", "external_web_access": False}],
    },
    region=REGION,
)
print(render_grounded(data))

# %% [markdown]
# ## 5. Streaming grounded answers
#
# Text arrives as `response.output_text.delta`. Citations arrive as their own
# event — `response.output_text.annotation.added` — so you can attach footnotes as
# the answer is being written.

# %%
stream_body = {
    "model": GPT55,
    "input": "Name one recent AWS AI announcement. One sentence.",
    "max_output_tokens": 500,
    "tools": [{"type": "web_search", "external_web_access": False}],
    "stream": True,
}

annotations_seen, deltas, event_types = [], 0, {}
print("--- live ---")
for line in stream_lines(f"{PREFIX}/responses", stream_body, region=REGION):
    if not line.startswith("data: "):
        continue
    payload = line[6:].strip()
    if payload == "[DONE]":
        break
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        continue
    etype = event.get("type", "")
    event_types[etype] = event_types.get(etype, 0) + 1
    if etype == "response.output_text.delta":
        deltas += 1
        print(event.get("delta", ""), end="", flush=True)
    elif etype == "response.output_text.annotation.added":
        annotations_seen.append(event.get("annotation", {}))

print(f"\n\ntext deltas: {deltas} | citation events: {len(annotations_seen)}")
for ann in annotations_seen[:3]:
    print(f"  [source] {str(ann.get('title'))[:60]} -> {str(ann.get('url'))[:70]}")

print("\n--- streaming event types ---")
for name, count in sorted(event_types.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {count:4}  {name}")

# %% [markdown]
# ## 6. When the model chooses *not* to search
#
# Web Search is not a forced retrieval step. The model only invokes it when the
# question needs current information — which keeps latency and cost down.

# %%
probes = [
    ("needs current info", "What is the latest Amazon Bedrock model announcement?"),
    ("timeless knowledge", "What does the acronym HTTP stand for?"),
    ("pure arithmetic", "What is 144 divided by 12?"),
]
print(f"{'question type':22} {'search rounds':>14} {'citations':>10}")
print("-" * 50)
for label, question in probes:
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GPT55,
            "input": question + " Be brief.",
            "max_output_tokens": 400,
            "tools": [{"type": "web_search", "external_web_access": False}],
        },
        region=REGION,
    )
    rounds = len(
        [i for i in data.get("output", []) if i.get("type") == "web_search_call"]
    )
    cites = sum(
        1
        for item in data.get("output", [])
        if item.get("type") == "message"
        for block in item.get("content", [])
        for ann in (block.get("annotations") or [])
        if ann.get("type") == "url_citation"
    )
    print(f"{label:22} {rounds:>14} {cites:>10}")

# %% [markdown]
# ## 7. Model and Region limits
#
# Web Search is OpenAI-GPT-only. Anything else is an explicit 400 — useful to see
# so you don't design around a capability a model doesn't have.

# %%
print(f"{'model':30} {'web_search':>12}")
print("-" * 46)
candidates = [
    (GPT56, PREFIX),
    (GPT55, PREFIX),
    (GPT54, PREFIX),
    ("google.gemma-4-31b", "/openai/v1"),
    ("xai.grok-4.3", "/openai/v1"),
    ("openai.gpt-oss-120b", "/v1"),
]
for model, prefix in candidates:
    code, data = post(
        f"{prefix}/responses",
        {
            "model": model,
            "input": "Any recent news? One sentence.",
            "max_output_tokens": 200,
            "tools": [{"type": "web_search", "external_web_access": False}],
        },
        region=REGION,
    )
    verdict = "supported" if code == 200 else f"{code}"
    print(f"{model:30} {verdict:>12}")
    if code != 200:
        print(f"      {err(data)[:90]}")

# %% [markdown]
# Web Search is **strictly regional**: each Region runs its own search and fetch
# tier, and queries, index data, and results never cross Region boundaries. It is
# available in `us-east-1`, `us-east-2`, and `us-west-2` — not `eu-central-1`.

# %%
for region in ("us-east-1", "us-east-2", "eu-central-1"):
    code, data = post(
        f"{PREFIX}/responses",
        {
            "model": GPT55,
            "input": "One recent AWS launch. One sentence.",
            "max_output_tokens": 300,
            "tools": [{"type": "web_search", "external_web_access": False}],
        },
        region=region,
    )
    print(f"{region:14} -> HTTP {code} {'' if code == 200 else err(data)[:70]}")

# %% [markdown]
# ## 8. A grounded research helper
#
# Production shape: explicit boundary setting, citation extraction, retries, and
# project attribution for cost tracking.

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "websearch-samples",
        "tags": {"Application": "WebSearchDemo", "Environment": "Demo"},
    },
    region=REGION,
)
project_id = project.get("id")
print("project:", code, project_id)


class GroundedResearcher:
    """Web-Search-grounded answers with citations, staying inside the AWS boundary."""

    def __init__(
        self, model=GPT55, region=REGION, project=None, external_web_access=False
    ):
        self.model, self.region, self.project = model, region, project
        self.external = external_web_access

    def ask(self, question: str, max_output_tokens: int = 700) -> dict:
        headers = {"OpenAI-Project": self.project} if self.project else None
        # post() retries 429/5xx with exponential backoff.
        code, data = post(
            f"{PREFIX}/responses",
            {
                "model": self.model,
                "input": question,
                "max_output_tokens": max(16, max_output_tokens),
                # Be explicit: the default is True and needs an extra IAM action.
                "tools": [{"type": "web_search", "external_web_access": self.external}],
                "store": False,  # opt out of the 30-day retention default
            },
            region=self.region,
            headers=headers,
        )
        if code != 200:
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        sources, seen = [], set()
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for block in item.get("content", []):
                for ann in block.get("annotations") or []:
                    if ann.get("type") == "url_citation" and ann["url"] not in seen:
                        seen.add(ann["url"])
                        sources.append({"title": ann.get("title"), "url": ann["url"]})
        return {
            "answer": response_text(data),
            "sources": sources,
            "search_rounds": len(
                [
                    i
                    for i in data.get("output", [])
                    if i.get("type") == "web_search_call"
                ]
            ),
            "usage": data.get("usage", {}),
        }


researcher = GroundedResearcher(project=project_id)
result = researcher.ask("What is Amazon Bedrock Mantle? Answer in two sentences.")
print("answer :", result["answer"][:260])
print("rounds :", result["search_rounds"])
print("sources:")
for s in result["sources"]:
    print(f"   - {str(s['title'])[:64]}\n     {s['url']}")

# %%
code, archived = post(
    f"/v1/organization/projects/{project_id}/archive", {}, region=REGION
)
print("archived:", code, archived.get("status"))

# %% [markdown]
# ## Gotchas — Web Search on bedrock-mantle
#
# | Gotcha | Detail |
# |---|---|
# | **`external_web_access` defaults to `true`** | And `AmazonBedrockFullAccess` lacks `ExternalWebAccess` → 403 on the authz check |
# | Failure is soft | The model still answers from the Bedrock index and says it couldn't reach the web |
# | Citations are mandatory | Acceptable-use requires retaining and displaying them |
# | OpenAI GPT only | Gemma 4, Grok, gpt-oss all return 400 |
# | Responses API only | Not available via Chat Completions |
# | Strictly regional | us-east-1 / us-east-2 / us-west-2; no cross-Region routing |
# | Model decides | No search happens on timeless questions — rounds can be 0 |
# | Billing | Web Search is charged separately from tokens |
# | Bulk extraction banned | You may not build a competing index from results |
#
# ## Next
# - `03-tools-and-structured-output.ipynb` — your own client-side tools
# - `04-prompt-caching-and-cost.ipynb` — explicit cache breakpoints
# - `05-server-side-tools-and-fine-tuning.ipynb` — Lambda MCP tools,
#   notes/tasks, RFT (reinforcement fine-tuning)
