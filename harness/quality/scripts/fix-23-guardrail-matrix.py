#!/usr/bin/env python3
"""Rewrite 99-cross-cutting/03 §9b with the measured per-surface guardrail matrix.

The previous version said the guardrail header is "accepted and silently ignored"
on bedrock-runtime's OpenAI APIs. Its probe used the Responses surface, where that
is true. It is NOT true of Chat Completions or Messages on the same endpoint, both
of which enforce the header. Verified two independent ways -- see
quality/findings/37-guardrail-matrix.json.
"""
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbedit import Notebook  # noqa: E402
import os
REPO_ROOT = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

NB = (REPO_ROOT + "/"
      "99-cross-cutting/03-production-hardening.ipynb")
nb = Notebook(NB)

# ---------------------------------------------------------------------------
# The section intro and its table.
# ---------------------------------------------------------------------------
head = nb.find("## 9b. Guardrails — not a `bedrock-mantle` feature, and the trap that hides it")
assert len(head) == 1, head

# The heading itself is being renamed, so the guard sees a "dropped" heading.
# Verified: cell holds only the 9b section, and the replacement covers all of it.
nb.set_source(
    head[0],
    drops_ok=True,
    text="""## 9b. Guardrails — and the surfaces that accept the header without using it

Amazon Bedrock Guardrails is the service's content-safety control: denied topics,
content filters, word filters, PII redaction, contextual grounding. AWS documents
it as a `bedrock-runtime` feature, and it is — but *"available on bedrock-runtime"*
is not granular enough to build on, because **the five APIs on that endpoint do not
all honour it**.

The cells below measure every attachment point. The important column is the last
one: what happens when the guardrail does *not* apply.

| How you attach it | Result |
|---|---|
| `guardrailConfig` on **Converse** | **enforced** — `stopReason=guardrail_intervened` |
| `guardrailIdentifier` on **InvokeModel** | **enforced** |
| **`ApplyGuardrail`** called directly | **enforced** — returns `GUARDRAIL_INTERVENED` |
| `X-Amzn-Bedrock-Guardrail*` headers on runtime **Chat Completions** | **enforced** |
| `X-Amzn-Bedrock-Guardrail*` headers on runtime **Messages** | **enforced** |
| `X-Amzn-Bedrock-Guardrail*` headers on runtime **Responses** | **200 — silently ignored** |
| `X-Amzn-Bedrock-Guardrail*` headers on any **`bedrock-mantle`** surface | **200 — silently ignored** |
| `guardrailConfig` in the **body** of Chat Completions, either endpoint | **200 — silently ignored** |
| `guardrailConfig` in the **body** of Responses | 400 `Unknown parameter` — safe |
| `guardrailConfig` in the **body** of Messages | 400 `Extra inputs are not permitted` — safe |

**The silently-ignored rows are the dangerous ones.** The request succeeds, nothing
in the response says the guardrail was skipped, and a team that sets the header and
sees HTTP 200 believes it has protection it does not have. The rows that return a
400 are *safer*, because they fail at the first call in development.

### How to test this yourself, and why one test is not enough

A guardrail block and a model declining on its own look identical from outside. Ask
a model for investment advice and it may refuse for its own reasons — so a probe
built on a topic models dislike cannot distinguish "the guardrail worked" from "the
model was cautious". This section therefore uses two signals:

1. **A DENY topic no model has any reason to refuse** — tulip cultivation — with a
   no-header control proving the model answers it happily.
2. **A guardrail identifier that does not exist.** A surface that enforces
   guardrails must reject it; a surface that ignores the header returns 200. This
   signal does not depend on model behaviour at all.

When the two agree, the row is trustworthy. That matters here specifically: an
earlier version of this notebook probed only the Responses surface and generalised
its result to *"bedrock-runtime's OpenAI APIs"*. The result was right; the
generalisation was not.""",
)

# ---------------------------------------------------------------------------
# Rewrite the guardrail-creation cell to use the unambiguous topic.
# ---------------------------------------------------------------------------
create = nb.find('name="mantle-samples-hardening-demo"')
assert len(create) == 1, create
nb.set_source(
    create[0],
    '''import boto3

control = boto3.client("bedrock", region_name=REGION)
runtime = boto3.client("bedrock-runtime", region_name=REGION)

# The DENY topic is deliberately banal. A guardrail that blocks investment advice
# cannot be told apart from a model that declines to give investment advice, and
# that ambiguity is how a wrong conclusion gets published. No model refuses to
# discuss tulips, so a refusal here can only be the guardrail.
guardrail = control.create_guardrail(
    name="bedrock-samples-hardening-demo",
    description="Throwaway guardrail for the per-model Bedrock samples. Safe to delete.",
    topicPolicyConfig={
        "topicsConfig": [
            {
                "name": "Tulips",
                "definition": "Any discussion of tulips, tulip bulbs, or tulip "
                              "cultivation.",
                "examples": ["Tell me about tulips."],
                "type": "DENY",
            }
        ]
    },
    contentPolicyConfig={
        "filtersConfig": [
            {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"}
        ]
    },
    blockedInputMessaging="BLOCKED_BY_GUARDRAIL (input).",
    blockedOutputsMessaging="BLOCKED_BY_GUARDRAIL (output).",
)
GUARDRAIL_ID, GUARDRAIL_VERSION = guardrail["guardrailId"], guardrail["version"]
safe_print("created guardrail:", GUARDRAIL_ID, "version", GUARDRAIL_VERSION)

# A new guardrail takes a moment to become READY. Polling beats sleeping: the
# wait is short on a good day and this does not fail on a slow one.
for _ in range(25):
    status = control.get_guardrail(
        guardrailIdentifier=GUARDRAIL_ID, guardrailVersion=GUARDRAIL_VERSION
    )["status"]
    if status == "READY":
        break
    time.sleep(2)
print("status:", status)

BLOCK_MARK = "BLOCKED_BY_GUARDRAIL"
DENIED = "In two sentences, describe how tulips are cultivated."
ALLOWED = "In two sentences, describe how potatoes are cultivated."''',
)

# ---------------------------------------------------------------------------
# ApplyGuardrail cell: keep the shape, fix the prompts.
# ---------------------------------------------------------------------------
apply_cell = nb.find("ApplyGuardrail — a pre-filter you can run before any endpoint")
assert len(apply_cell) == 1, apply_cell
nb.set_source(
    apply_cell[0],
    '''# ApplyGuardrail evaluates text with no model and no inference cost, so it works
# as a pre-filter in front of ANY endpoint or API -- including the surfaces further
# down that ignore the header. If you need guardrails on one of those, this is the
# answer.
print("ApplyGuardrail — a pre-filter you can run before any endpoint or API")
print("-" * 72)
for text in (DENIED, ALLOWED):
    verdict = runtime.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source="INPUT",
        content=[{"text": {"text": text}}],
    )
    topics = [
        t["name"]
        for assessment in verdict.get("assessments", [])
        for t in assessment.get("topicPolicy", {}).get("topics", [])
    ]
    print(f"  {verdict['action']:22} topics={str(topics):24} {text[:38]}")

print()
print("=> action=GUARDRAIL_INTERVENED means do not send it. Run the same call with")
print("   source='OUTPUT' on the model's reply to screen what you return.")
print("   Two calls per turn is the cost of guardrailing a surface that will not")
print("   guardrail itself.")''',
)

nb.save()
print(f"intro + creation + ApplyGuardrail updated: {nb.changes} edits")

# ---------------------------------------------------------------------------
# Replace the Converse + trap cell with the full six-surface matrix.
# ---------------------------------------------------------------------------
nb = Notebook(NB)
trap = nb.find("The silent-header trap on bedrock-runtime's OpenAI APIs")
assert len(trap) == 1, trap

nb.set_source(
    trap[0],
    '''# Shape 1: guardrailConfig on Converse. Works across providers, and the stop
# reason names what happened.
print("Converse with guardrailConfig")
print("-" * 72)
for model in ("amazon.nova-micro-v1", "anthropic.claude-sonnet-5"):
    try:
        reply = runtime.converse(
            modelId=resolve_runtime_id(model, REGION),
            messages=[{"role": "user", "content": [{"text": DENIED}]}],
            inferenceConfig={"maxTokens": 200},
            guardrailConfig={
                "guardrailIdentifier": GUARDRAIL_ID,
                "guardrailVersion": GUARDRAIL_VERSION,
            },
        )
        text = "".join(
            b.get("text", "") for b in reply["output"]["message"]["content"]
        )
        print(f"  {model:28} stop={reply['stopReason']:22} "
              f"blocked={BLOCK_MARK in text}")
    except Exception as exc:  # noqa: BLE001 - report, do not stop the notebook
        print(f"  {model:28} {type(exc).__name__}: {str(exc)[-60:]}")

# Converse also VALIDATES the identifier, which is the behaviour to expect from a
# surface that really applies it.
try:
    runtime.converse(
        modelId=resolve_runtime_id("amazon.nova-micro-v1", REGION),
        messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        inferenceConfig={"maxTokens": 16},
        guardrailConfig={"guardrailIdentifier": "gr-doesnotexist000",
                         "guardrailVersion": "1"},
    )
    print("  bogus identifier             accepted (200) <- would be a bad sign")
except Exception as exc:  # noqa: BLE001
    print(f"  bogus identifier             rejected: {type(exc).__name__}")''',
)

MD_MATRIX = """### The six OpenAI- and Anthropic-shaped surfaces, measured

Same guardrail, same question, six surfaces. Each row reports both signals:
whether the banal denied topic was actually blocked, and whether a nonexistent
guardrail ID was rejected. The final column flags any row where the two disagree —
if that happens, do not trust the row, re-run it."""

CODE_MATRIX = '''RUNTIME_HOST = f"https://bedrock-runtime.{REGION}.amazonaws.com"
MANTLE_HOST = f"https://bedrock-mantle.{REGION}.api.aws"

SURFACES = [
    # (label, host, path, response shape, body, extra headers)
    ("runtime Chat Completions", RUNTIME_HOST, "/openai/v1/chat/completions", "chat",
     {"model": "openai.gpt-oss-20b-1:0", "max_completion_tokens": 300}, {}),
    ("runtime Responses", RUNTIME_HOST, "/openai/v1/responses", "responses",
     {"model": "us.openai.gpt-5.6-sol", "max_output_tokens": 2000}, {}),
    ("runtime Messages", RUNTIME_HOST, "/anthropic/v1/messages", "messages",
     {"model": "us.anthropic.claude-opus-5", "max_tokens": 300},
     {"anthropic-version": "2023-06-01"}),
    ("mantle Chat Completions", MANTLE_HOST, "/v1/chat/completions", "chat",
     {"model": "openai.gpt-oss-20b", "max_tokens": 300}, {}),
    ("mantle Responses", MANTLE_HOST, "/openai/v1/responses", "responses",
     {"model": "openai.gpt-5.6-sol", "max_output_tokens": 2000}, {}),
    ("mantle Messages", MANTLE_HOST, "/anthropic/v1/messages", "messages",
     {"model": "anthropic.claude-opus-5", "max_tokens": 300},
     {"anthropic-version": "2023-06-01"}),
]

GOOD_HEADERS = {
    "X-Amzn-Bedrock-GuardrailIdentifier": GUARDRAIL_ID,
    "X-Amzn-Bedrock-GuardrailVersion": GUARDRAIL_VERSION,
}
BOGUS_HEADERS = {
    "X-Amzn-Bedrock-GuardrailIdentifier": "gr-doesnotexist000",
    "X-Amzn-Bedrock-GuardrailVersion": "1",
}


def send(base, path, shape, body, headers):
    """POST with a bearer token; return (status, assistant text)."""
    request_body = dict(body)
    if shape == "responses":
        request_body["input"] = DENIED
    else:
        request_body["messages"] = [{"role": "user", "content": DENIED}]
    hdrs = {
        "Authorization": f"Bearer {provide_token(region=REGION)}",
        "Content-Type": "application/json",
        **headers,
    }
    request = urllib.request.Request(
        base + path, data=json.dumps(request_body).encode(), headers=hdrs,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as reply:  # nosec B310  # noqa: S310
            status, raw = reply.status, reply.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:120]
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}"

    payload = json.loads(raw) if raw.strip() else {}
    if shape == "responses":
        return status, response_text(payload)
    if shape == "messages":
        return status, "".join(
            b.get("text", "") for b in (payload.get("content") or [])
            if isinstance(b, dict)
        )
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    return status, message.get("content") or ""


print(f"{'surface':26} {'control':>8} {'header':>8} {'bogus id':>10} {'verdict':>22} agree")
print("-" * 92)
rows = []
for label, base, path, shape, body, extra in SURFACES:
    # Control: no guardrail. Proves the model answers, so a later refusal is
    # attributable to the guardrail rather than to the model's own caution.
    _, control_text = send(base, path, shape, body, extra)
    answered = len(control_text.strip()) > 40

    _, guarded_text = send(base, path, shape, body, {**extra, **GOOD_HEADERS})
    blocked = BLOCK_MARK in guarded_text

    bogus_status, _ = send(base, path, shape, body, {**extra, **BOGUS_HEADERS})
    rejects_bogus = bogus_status != 200

    agree = blocked == rejects_bogus
    verdict = "ENFORCED" if blocked else "IGNORED (accepts, no-ops)"
    rows.append({"surface": label, "enforced": blocked, "agree": agree,
                 "answered": answered})
    print(f"{label:26} {str(answered):>8} {str(blocked):>8} "
          f"{('reject' if rejects_bogus else 'accept'):>10} {verdict:>22} {agree}")

print()
enforced = [r["surface"] for r in rows if r["enforced"]]
ignored = [r["surface"] for r in rows if not r["enforced"]]
disagreed = [r["surface"] for r in rows if not r["agree"]]
unanswered = [r["surface"] for r in rows if not r["answered"]]

if unanswered:
    print(f"!! control failed on {unanswered}: the model did not answer the benign")
    print("   question, so those rows prove nothing. Fix the control first.")
print(f"=> guardrail header ENFORCED on: {enforced or 'none'}")
print(f"=> guardrail header IGNORED on : {ignored or 'none'}")
if disagreed:
    print(f"!! the two signals disagree on {disagreed} — re-run before trusting it.")
else:
    print("   Both signals agreed on every row: the surfaces that blocked the topic")
    print("   are exactly the surfaces that rejected a nonexistent guardrail ID.")
print()
print("   The operational rule: attach a guardrail, then VERIFY it on the exact")
print("   surface you ship. 'Guardrails are supported on this endpoint' is true and")
print("   still leaves you unprotected on some of its APIs. Where the header is")
print("   ignored, use ApplyGuardrail explicitly or move the call to Converse.")'''

i = nb.insert_after(trap[0], "markdown", MD_MATRIX)
nb.insert_after(i, "code", CODE_MATRIX)
nb.save()
print(f"matrix cells inserted: {nb.changes} edits")
