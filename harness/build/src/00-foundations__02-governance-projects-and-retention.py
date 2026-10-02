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
# # Foundations 2 — Projects, data retention, and observability
#
# Governance for `bedrock-mantle`: how to isolate workloads, attribute cost,
# control whether AWS retains your prompts, and find your metrics.
#
# **Prerequisite:** `01-endpoints-auth-and-the-three-paths.ipynb` (auth + paths).
#
# ## What this notebook covers
# - **Projects** — the workload boundary; create, tag, use, archive
# - Attribution from all three APIs (each uses a *different* header)
# - **Data retention / ZDR (zero data retention)** — account, project, and per-model
#   scopes
# - Observability — the CloudWatch namespace trap and CloudTrail data events
#
# ## IAM
# Project management needs more than inference. `AmazonBedrockMantleFullAccess`
# covers it; `…InferenceAccess` alone can read but not create projects.

# %%
import json
import sys

sys.path.insert(0, "../_shared")
from mantle import err, post, redact_ids, safe_print

REGION = "us-east-1"
print("region:", REGION)

# %% [markdown]
# ## 1. What a Project is
#
# A Project is a logical boundary for a workload — an app, an environment, an
# experiment. It gives you IAM-enforceable isolation, tag-based cost tracking,
# and per-project observability, without spinning up separate AWS accounts.
#
# Every account already has a `default` project. Up to **1000** per account.
#
# | | Projects (mantle) | Inference profiles (runtime) |
# |---|---|---|
# | APIs | Responses, Chat Completions, Messages | Invoke, Converse |
# | Endpoint | `bedrock-mantle` | `bedrock-runtime` |
# | Access control | project ARN in IAM | profile ARN in IAM |
# | Cost tracking | tags on the project | tags on the profile |
#
# Note the naming: with the OpenAI APIs these are called **Projects**; with the
# Anthropic Messages API the same resource is called a **Workspace**. Same thing,
# same management API, different request header.

# %%
code, payload = post("/v1/organization/projects", None, region=REGION, method="GET")
print("GET /v1/organization/projects ->", code)
for p in payload.get("data", []):
    print(
        f"  {p['id']:32} {p.get('name',''):24} status={p.get('status')} "
        f"retention={(p.get('data_retention') or {}).get('mode')}"
    )

# %% [markdown]
# ## 2. Create a project
#
# Tags are the hook for AWS Cost Explorer, so set them at creation.

# %%
code, project = post(
    "/v1/organization/projects",
    {
        "name": "mantle-samples-foundations",
        "tags": {
            "Application": "MantleSamples",
            "Environment": "Demo",
            "Owner": "SampleReader",
            "CostCenter": "0000",
        },
    },
    region=REGION,
)
print("create ->", code)
safe_print(json.dumps(project, indent=2)[:600])

PROJECT_ID = project.get("id")
print("\nPROJECT_ID =", PROJECT_ID)

# %% [markdown]
# The `arn` is Bedrock-specific (not in the OpenAI spec) and is what you put in
# IAM policies to scope access to this project. `data_retention.mode` defaults to
# `inherit`, meaning "whatever the account is set to" — more on that below.

# %% [markdown]
# ## 3. Attribute inference to the project — three different headers
#
# This is the part worth memorising: **the header depends on the API**.
#
# | API | Header |
# |---|---|
# | Responses | `OpenAI-Project: <project-id>` |
# | Chat Completions | `OpenAI-Project: <project-id>` |
# | Anthropic Messages | `anthropic-workspace: <project-id>` |

# %%
attribution_tests = [
    (
        "Responses",
        "/openai/v1/responses",
        {"model": "google.gemma-4-31b", "input": "Reply OK", "max_output_tokens": 16},
        {"OpenAI-Project": PROJECT_ID},
    ),
    (
        "ChatCompletions",
        "/v1/chat/completions",
        {
            "model": "qwen.qwen3-32b",
            "messages": [{"role": "user", "content": "Reply OK"}],
            "max_tokens": 16,
        },
        {"OpenAI-Project": PROJECT_ID},
    ),
    (
        "Messages",
        "/anthropic/v1/messages",
        {
            "model": "anthropic.claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply OK"}],
        },
        {"anthropic-workspace": PROJECT_ID, "anthropic-version": "2023-06-01"},
    ),
]

for label, path, body, headers in attribution_tests:
    code, data = post(path, body, region=REGION, headers=headers)
    hdr = [k for k in headers if k != "anthropic-version"][0]
    print(f"{label:16} via {hdr:22} -> {code} {'' if code == 200 else err(data)[:70]}")

# %% [markdown]
# All three return 200. Their usage now rolls up under this project, so the tags
# you set earlier become cost-allocation dimensions in Cost Explorer.
#
# With the OpenAI SDK you can bind the project once at construction:
#
# ```python
# client = OpenAI(api_key=..., base_url=..., project=PROJECT_ID)
# ```

# %% [markdown]
# ## 4. Manage the project lifecycle
#
# Tags are add/remove only — there is **no** operation that replaces the whole tag
# set at once.

# %%
code, updated = post(
    f"/v1/organization/projects/{PROJECT_ID}",
    {"add_tags": {"Version": "1.0", "Team": "Samples"}},
    region=REGION,
)
print("add_tags ->", code, "|", (updated.get("tags") or {}))

code, updated = post(
    f"/v1/organization/projects/{PROJECT_ID}",
    {"remove_tag_keys": ["Version"]},
    region=REGION,
)
print("remove_tag_keys ->", code, "|", (updated.get("tags") or {}))

code, updated = post(
    f"/v1/organization/projects/{PROJECT_ID}",
    {"name": "mantle-samples-foundations-v2"},
    region=REGION,
)
print("rename ->", code, "|", updated.get("name"))

# %% [markdown]
# ## 5. Data retention and Zero Data Retention (ZDR)
#
# Two independent controls decide whether your prompts persist:
#
# **(a) `store` on the Responses API** — per request. Defaults to `true`, which
# retains input *and* output for 30 days, in-Region, encrypted, scoped to the
# project. That is what makes `previous_response_id` work. Set `store=False` to
# opt out per call — but then you cannot chain.
#
# **(b) `data_retention.mode`** — account / project / model scope. This is the
# ZDR control, and it governs whether classifier-flagged content is retained for
# offline review.
#
# | Mode | Meaning |
# |---|---|
# | `default` | Standard data handling for the model |
# | `none` | **Zero data retention** |
# | `provider_data_share` | Data may be shared with the model provider |
# | `inherit` | No mode set at this scope; defer to the parent |

# %%
code, retention = post("/v1/data_retention", None, region=REGION, method="GET")
print("GET /v1/data_retention ->", code)
print(json.dumps(retention, indent=2)[:400])

# %%
# Which modes does the API actually accept? Probe rather than guess.
for mode in (
    "inherit",
    "default",
    "none",
    "provider_data_share",
    "enabled",
    "disabled",
):
    code, data = post(
        "/v1/organization/projects",
        {"name": f"retention-probe-{mode}", "data_retention": {"mode": mode}},
        region=REGION,
    )
    if code == 200:
        print(f"  {mode:22} accepted -> {data.get('data_retention')}")
        post(f"/v1/organization/projects/{data['id']}/archive", {}, region=REGION)
    else:
        print(f"  {mode:22} rejected {code}: {err(data)[:78]}")

# %% [markdown]
# So the valid set is `inherit`, `default`, `none`, `provider_data_share` —
# `enabled`/`disabled` are not modes, despite being intuitive guesses.
#
# You can also check what a specific model permits, via `allowed_modes`:

# %%
# Retention is also visible per model. Depending on the model and your account
# posture, the payload may carry an `allowed_modes` list; if it is absent, the
# model inherits the account/project setting.
for mid in ("openai.gpt-5.6-sol", "google.gemma-4-31b", "anthropic.claude-haiku-4-5"):
    code, model_info = post(f"/v1/models/{mid}", None, region=REGION, method="GET")
    print(f"GET /v1/models/{mid} -> {code}")
    if code == 200:
        print("   data_retention:", model_info.get("data_retention"))
        print("   allowed_modes: ", model_info.get("allowed_modes", "(not returned)"))

# %% [markdown]
# **Why this matters:** not every model supports every mode. If you set an
# account-wide mode that a model does not allow, calls to that model fail. Check
# the per-model payload before rolling out a retention posture — and note that
# `allowed_modes` is only returned for some models, so treat its absence as
# "inherits the account setting" rather than "no restrictions".

# %%
# store=True vs store=False, and the consequence for chaining.
code, first = post(
    "/openai/v1/responses",
    {
        "model": "google.gemma-4-31b",
        "input": "My name is Ada. Reply OK.",
        "max_output_tokens": 16,
        "store": True,
    },
    region=REGION,
)
stored_id = first.get("id")
print("store=True  ->", code, "id:", stored_id)

code, second = post(
    "/openai/v1/responses",
    {
        "model": "google.gemma-4-31b",
        "input": "Remember nothing. Reply OK.",
        "max_output_tokens": 16,
        "store": False,
    },
    region=REGION,
)
unstored_id = second.get("id")
print("store=False ->", code, "id:", redact_ids(unstored_id))

code, chained = post(
    "/openai/v1/responses",
    {
        "model": "google.gemma-4-31b",
        "input": "What is my name?",
        "previous_response_id": stored_id,
        "max_output_tokens": 32,
    },
    region=REGION,
)
from mantle import response_text

print("\nchain from stored   ->", code, repr(response_text(chained)[:60]))

code, chained_bad = post(
    "/openai/v1/responses",
    {
        "model": "google.gemma-4-31b",
        "input": "What did I say?",
        "previous_response_id": unstored_id,
        "max_output_tokens": 32,
    },
    region=REGION,
)
print("chain from unstored ->", code, err(chained_bad)[:80])

# %% [markdown]
# Exactly as designed: `store=False` gives you privacy, and costs you the ability
# to chain. If you need both privacy and multi-turn, send the full history
# yourself each turn.

# %% [markdown]
# ## 6. Observability — the namespace trap
#
# This catches people out. Mantle publishes to **`AWS/BedrockMantle`**, not
# `AWS/Bedrock`. A dashboard built on `AWS/Bedrock` shows **zero errors** during
# an active mantle incident, because the traffic is in a different namespace.
#
# And in `AWS/BedrockMantle` the only error metric is `InferenceClientErrors`
# (**4xx only**) — there is no 5xx/503 metric. A 503 storm is therefore invisible
# in CloudWatch metrics; you have to look at your own client-side telemetry.

# %%
import boto3

cw = boto3.client("cloudwatch", region_name=REGION)
for ns in ("AWS/BedrockMantle", "AWS/Bedrock"):
    resp = cw.list_metrics(Namespace=ns)
    names = sorted({m["MetricName"] for m in resp.get("Metrics", [])})
    print(f"{ns:22} {len(names):3} distinct metrics")
    for n in names[:10]:
        print("      ", n)
    if not names:
        print("       (none yet in this account/region)")

# %% [markdown]
# **CloudTrail.** Mantle inference is recorded as **data events**, which are off
# by default and cost extra. Management events (creating projects, etc.) appear
# normally. Note also that *short-term API key generation is not logged at all* —
# it happens client-side.
#
# For CloudFormation users, projects are modelled as `AWS::BedrockMantle::Project`.

# %% [markdown]
# ## 7. Clean up
#
# Archiving stops new inference but keeps historical data queryable for ~30 days.

# %%
code, archived = post(
    f"/v1/organization/projects/{PROJECT_ID}/archive", {}, region=REGION
)
print("archive ->", code, "status:", archived.get("status"))

code, listing = post("/v1/organization/projects", None, region=REGION, method="GET")
active = [p["id"] for p in listing.get("data", []) if p.get("status") == "active"]
print(f"\nactive projects remaining: {len(active)}")

# %% [markdown]
# ## Gotchas from this notebook
#
# | Gotcha | Detail |
# |---|---|
# | Header differs per API | `OpenAI-Project` vs `anthropic-workspace` |
# | Project vs Workspace | Same resource, different name and header |
# | Tag updates | Add/remove only — no whole-set replace |
# | `store` defaults to **true** | 30-day retention unless you opt out per request |
# | `store=False` breaks chaining | `previous_response_id` → 404 |
# | Retention modes | `inherit`/`default`/`none`/`provider_data_share` only |
# | Per-model `allowed_modes` | A model may reject your account-wide mode |
# | CloudWatch namespace | `AWS/BedrockMantle`, **not** `AWS/Bedrock` |
# | No 5xx metric | Only `InferenceClientErrors` (4xx) exists |
# | CloudTrail | Inference = data events (opt-in, extra cost) |
# | Archived projects | No new inference; data readable ~30 days |
#
# ## Next
# `03-scaling-tiers-and-latency.ipynb` — quotas, retries, service tiers, TTFT
# (time-to-first-token).
