#!/usr/bin/env python3
"""Make the two cross-cutting notebooks endpoint-aware.

99-cross-cutting/01 offers `api_prefix()` as "the single most useful thing to copy
out of this collection". It is mantle-only, and silently wrong for
bedrock-runtime. Its pasteable `capabilities.py` has the same gap.

99-cross-cutting/02 is the migration guide, so it should lead with the endpoint AWS
recommends rather than presenting mantle as the destination.
"""
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbedit import Notebook  # noqa: E402
import os
REPO_ROOT = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

REPO = REPO_ROOT + "/"

# ===========================================================================
# 99-cross-cutting/01
# ===========================================================================
NB1 = REPO + "99-cross-cutting/01-choosing-a-model-and-api.ipynb"
nb = Notebook(NB1)

nb.sub(
    """## 2. Path resolution

Three prefixes, and the rule is by model family. This function is the single most
useful thing to copy out of this collection.""",
    """## 2. Path resolution — per endpoint, not just per model

Prefixes depend on the model family **and** on the endpoint, and the split falls in
a different place on each:

| | `bedrock-mantle` | `bedrock-runtime` |
|---|---|---|
| `/openai/v1` | `google.gemma-4*`, `openai.gpt-5*`, `xai.*` | **every** OpenAI-compatible model |
| `/v1` | everything else non-Anthropic | **does not exist** |
| `/anthropic/v1` | `anthropic.*` | `anthropic.*` |

So `openai.gpt-oss-120b` is `/v1` on mantle and its runtime twin
`openai.gpt-oss-120b-1:0` is `/openai/v1`. A resolver that takes only a model ID can
be right about one endpoint at a time — this collection shipped exactly that until
`bedrock-runtime` grew the OpenAI-compatible paths in August 2026.

Copy the version below, with the `endpoint` argument.""",
)

nb.set_source(
    nb.find("def api_prefix(model_id: str) -> str:")[0],
    '''import re


def api_prefix(model_id: str, endpoint: str = "mantle") -> str:
    """Which URL prefix serves this model's inference APIs, on this endpoint?"""
    # Strip a geo/global inference-profile prefix first: "us.anthropic.claude-opus-5"
    # does not start with "anthropic.", and a naive check routes it to /openai/v1.
    bare = re.sub(r"^(us|eu|apac|global|in)\\.", "", model_id)
    if bare.startswith("anthropic."):
        return "/anthropic/v1"
    if endpoint == "runtime":
        return "/openai/v1"
    if bare.startswith(("google.gemma-4", "openai.gpt-5", "xai.")):
        return "/openai/v1"
    return "/v1"


by_prefix = defaultdict(list)
for mid in models:
    by_prefix[api_prefix(mid)].append(mid)

print("on bedrock-mantle:")
for prefix in sorted(by_prefix):
    ids = by_prefix[prefix]
    fams = sorted({i.split(".")[0] for i in ids})
    print(f"  {prefix:16} {len(ids):3} models | families: {', '.join(fams)}")

# The same models, addressed on bedrock-runtime. Note that the /v1 bucket empties.
runtime_ids = [r for m in models if (r := runtime_id_for(m, REGION)) is not None]
runtime_by_prefix = defaultdict(list)
for rid in runtime_ids:
    runtime_by_prefix[api_prefix(rid, "runtime")].append(rid)

print(f"\\non bedrock-runtime ({len(runtime_ids)} of {len(models)} are there):")
for prefix in sorted(runtime_by_prefix):
    ids = runtime_by_prefix[prefix]
    fams = sorted({i.split(".")[-1].split(".")[0] if i.startswith(("us.", "global."))
                   else i.split(".")[0] for i in ids})
    print(f"  {prefix:16} {len(ids):3} models | families: {', '.join(fams)}")

moved = [
    m for m in models
    if (r := runtime_id_for(m, REGION)) is not None
    and api_prefix(m) != api_prefix(r, "runtime")
]
print(f"\\n=> {len(moved)} model(s) change PATH between endpoints, e.g. "
      f"{moved[:3]}")
print(f"=> bedrock-runtime serves no /v1 inference path at all: "
      f"{'/v1' not in runtime_by_prefix}")''',
)

nb.save()
print(f"01 §2 updated: {nb.changes} edits")

# ---------------------------------------------------------------------------
# The pasteable resolver.
# ---------------------------------------------------------------------------
nb = Notebook(NB1)
nb.sub(
    '''OPENAI_PREFIX_FAMILIES = ("google.gemma-4", "openai.gpt-5", "xai.")''',
    '''OPENAI_PREFIX_FAMILIES = ("google.gemma-4", "openai.gpt-5", "xai.")

# A geo/global inference-profile prefix is not part of the family name, and
# bedrock-runtime requires one for several families. Strip it before matching.
PROFILE_PREFIX = re.compile(r"^(us|eu|apac|global|in)\\.")''',
)

nb.sub(
    '''def api_prefix(model_id):
    """Return the URL prefix that serves this model's inference APIs."""
    if model_id.startswith("anthropic."):
        return "/anthropic/v1"
    if model_id.startswith(OPENAI_PREFIX_FAMILIES):
        return "/openai/v1"
    return "/v1"


def surface(model_id):
    """Which API this model actually serves: messages / chat / responses."""
    if model_id.startswith("anthropic."):
        return "messages"
    if model_id.startswith(CHAT_ONLY):
        return "chat"
    return "responses"''',
    '''def api_prefix(model_id, endpoint="mantle"):
    """Return the URL prefix that serves this model's inference APIs.

    `endpoint` is "mantle" or "runtime", and it changes the answer: runtime serves
    every OpenAI-compatible model on /openai/v1 and has no /v1 inference path.
    """
    bare = PROFILE_PREFIX.sub("", model_id)
    if bare.startswith("anthropic."):
        return "/anthropic/v1"
    if endpoint == "runtime":
        return "/openai/v1"
    if bare.startswith(OPENAI_PREFIX_FAMILIES):
        return "/openai/v1"
    return "/v1"


def surface(model_id, endpoint="mantle"):
    """Which API this model actually serves: messages / chat / responses.

    Narrower on bedrock-runtime, where only the GPT-5.6 and Grok 4.6 profiles
    serve Responses and only the newest Claude models serve Messages.
    """
    bare = PROFILE_PREFIX.sub("", model_id)
    if bare.startswith("anthropic."):
        return "messages"
    if endpoint == "runtime":
        return "responses" if bare.startswith(RUNTIME_RESPONSES) else "chat"
    if bare.startswith(CHAT_ONLY):
        return "chat"
    return "responses"''',
)

nb.sub(
    '''# Models that accept flex/priority service tiers. Not a Messages parameter at all.
TIERED = ("openai.gpt-oss", "google.gemma-", "xai.", "qwen.", "deepseek.",
          "zai.", "minimax.", "moonshotai.", "mistral.", "nvidia.", "writer.")''',
    '''# Models that accept flex/priority service tiers. Not a Messages parameter at all.
TIERED = ("openai.gpt-oss", "google.gemma-", "xai.", "qwen.", "deepseek.",
          "zai.", "minimax.", "moonshotai.", "mistral.", "nvidia.", "writer.")

# On bedrock-runtime the Responses API reaches only these families; everything
# else OpenAI-compatible there is Chat Completions. Much narrower than on mantle.
RUNTIME_RESPONSES = ("openai.gpt-5.6", "xai.grok-4.6")

# Features that exist on exactly one endpoint. Check before you pick.
MANTLE_ONLY_FEATURES = ("server_side_tools", "web_search", "background",
                        "projects", "workspaces")
RUNTIME_ONLY_FEATURES = ("guardrails", "prompt_routing", "cross_region",
                         "provisioned_throughput", "batch")''',
)

nb.sub(
    '''def inference_path(model_id, api="auto"):
    """Full inference path, e.g. "/v1/chat/completions".

    `api="auto"` picks the surface the model actually serves. Defaulting to
    /responses for everything non-Anthropic -- which an earlier version of this
    module did -- 400s for every Chat-Completions-only family.
    """
    prefix = api_prefix(model_id)
    chosen = surface(model_id) if api == "auto" else api''',
    '''def inference_path(model_id, api="auto", endpoint="mantle"):
    """Full inference path, e.g. "/v1/chat/completions".

    `api="auto"` picks the surface the model actually serves. Defaulting to
    /responses for everything non-Anthropic -- which an earlier version of this
    module did -- 400s for every Chat-Completions-only family.
    """
    prefix = api_prefix(model_id, endpoint)
    chosen = surface(model_id, endpoint) if api == "auto" else api''',
)

nb.sub(
    '''def token_limit_field(model_id, api="auto"):
    """The budget parameter this model+API expects.

    Getting this wrong yields a 400 that reads exactly like "this API does not
    exist here", which is a trap worth avoiding by construction.
    """
    chosen = surface(model_id) if api == "auto" else api''',
    '''def token_limit_field(model_id, api="auto", endpoint="mantle"):
    """The budget parameter this model+API expects.

    Getting this wrong yields a 400 that reads exactly like "this API does not
    exist here", which is a trap worth avoiding by construction.
    """
    chosen = surface(model_id, endpoint) if api == "auto" else api''',
)

nb.sub(
    '''"""Resolve bedrock-mantle request details per model. Generated from live probes.''',
    '''"""Resolve Bedrock request details per model and endpoint. From live probes.''',
)

nb.sub(
    '''The durable design is the one in `02-migrating-from-openai.ipynb` section 7: send
the request, and when the service returns a 400 that *names* a parameter, drop that
parameter and retry. Mantle is consistent about naming it. Use this module for
routing (which path, which API, which budget field) and let error handling deal
with sampling.
"""''',
    '''The durable design is the one in `02-migrating-from-openai.ipynb` section 7: send
the request, and when the service returns a 400 that *names* a parameter, drop that
parameter and retry. Bedrock is consistent about naming it. Use this module for
routing (which path, which API, which budget field) and let error handling deal
with sampling.

Every routing function takes `endpoint="mantle"` or `endpoint="runtime"`, because
the path, the API surface and the model ID all differ between the two. It does NOT
translate model IDs -- ask the service, via `runtime_id_for()` in
`_shared/bedrock.py`, rather than encoding a mapping that will age.
"""
import re''',
)

nb.save()
print(f"01 §6 resolver updated: {nb.changes} edits")

# ---------------------------------------------------------------------------
# The resolver verification cell must exercise both endpoints.
# ---------------------------------------------------------------------------
nb = Notebook(NB1)
verify = nb.find("# Verify the resolver against the live endpoint. A resolver that is wrong is")
assert len(verify) == 1, verify
nb.set_source(
    verify[0],
    '''# Verify the resolver against the live endpoints. A resolver that is wrong is
# worse than none: it turns "read the docs" into "debug a 400 in production".
# So exercise it on BOTH endpoints and count the failures.
namespace: dict = {}
exec(CAPABILITIES_PY, namespace)  # nosec B102  # noqa: S102 - our own literal above

resolved_path = namespace["inference_path"]
resolved_field = namespace["token_limit_field"]
resolved_sampling = namespace["sampling"]

CHECK = [
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "anthropic.claude-opus-5",
    "qwen.qwen3-32b",
    "deepseek.v3.2",
    "zai.glm-5",
    "google.gemma-4-31b",
    "xai.grok-4.6",
]
AV = {"anthropic-version": "2023-06-01"}
print(f"{'model':26} {'endpoint':8} {'resolved path':30} {'field':22} result")
print("-" * 104)

failures = []
for model_id in CHECK:
    for endpoint in ("mantle", "runtime"):
        if endpoint == "mantle":
            send_id = model_id if model_id in models else None
        else:
            send_id = runtime_id_for(model_id, REGION)
        if send_id is None:
            print(f"{model_id:26} {endpoint:8} {'-- not on this endpoint --':30} "
                  f"{'-':22} skipped")
            continue

        path = resolved_path(send_id, endpoint=endpoint)
        field = resolved_field(send_id, endpoint=endpoint)
        body = {"model": send_id, field: 2000, **resolved_sampling(send_id, 0.5)}
        if path.endswith("/responses"):
            body["input"] = "Reply OK"
        else:
            body["messages"] = [{"role": "user", "content": "Reply OK"}]

        caller = post if endpoint == "mantle" else runtime_post
        code, data = caller(
            path, body, region=REGION,
            headers=AV if path.endswith("/messages") else None,
            attempts=1, timeout=120,
        )
        good = ok(code, data)
        if not good:
            failures.append((model_id, endpoint, code, err(data)[:60]))
        print(f"{model_id:26} {endpoint:8} {path:30} {field:22} "
              f"{'ok' if good else str(code) + ' ' + err(data)[:34]}")

print()
if failures:
    print(f"=> {len(failures)} combination(s) failed. The resolver is wrong for these,")
    print("   which is exactly the situation this cell exists to catch:")
    for model_id, endpoint, code, message in failures:
        print(f"     {model_id} on {endpoint}: {code} {message}")
else:
    print(f"=> every resolved request succeeded, on both endpoints, for the "
          f"{len(CHECK)} models above.")
    print("   That is the bar for pasting this into a project. Re-run it when you")
    print("   add a model, and treat a failure here as a resolver bug rather than")
    print("   a service problem.")''',
)
nb.save()
print(f"01 verification updated: {nb.changes} edits")
