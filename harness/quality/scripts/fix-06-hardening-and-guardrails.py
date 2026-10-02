#!/usr/bin/env python3
"""Audit fix 6 — 99-cross-cutting/03, production hardening.

Defects fixed:
  * §2's "a permanent 400 must not be retried" used `top_p` on Gemma 4, which now
    returns 200 — so the cell's whole lesson breaks on re-run.
  * §9's `ProductionClient` computed an `/anthropic/v1` or `/v1` prefix and then
    always POSTed `{prefix}/responses`, which 400s for every Claude and
    open-weight model. Its NO_TEMPERATURE / NO_TOP_P lists were also stale.

Coverage added (audit gap C1):
  * Guardrails. Three mentions across 33 notebooks and no demonstration, in a
    collection whose production-hardening checklist otherwise covers every control.
    The section below is the honest answer for a mantle workload: guardrails are
    not a mantle parameter at all, and the header that looks like it should work on
    the OpenAI APIs is accepted and silently ignored. `ApplyGuardrail` called
    directly is the pattern that does work from any endpoint.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PATH = f"{REPO}/99-cross-cutting/03-production-hardening.ipynb"

FAIL_FAST = '''# A permanent 400 must not be retried — measure that it fails fast.
#
# The invalid input here is `max_output_tokens` below the documented minimum of 16.
# That choice matters: an earlier version of this cell sent `top_p` to Gemma 4,
# which was a 400 when written and is a 200 now, so the cell stopped demonstrating
# anything. Pick an invariant to violate, not a per-model restriction.
started = time.perf_counter()
code, data = resilient_call(
    f"{PREFIX}/responses",
    {"model": MODEL, "input": "Hi", "max_output_tokens": 8},
)
print(
    f"invalid param -> HTTP {code} in {time.perf_counter() - started:.2f}s "
    f"(no retries — correct)"
)
print("message:", err(data)[:90])'''

PRODUCTION_CLIENT = '''class ProductionClient:
    """Hardened bedrock-mantle client.

    Routing is the part that used to be wrong here. An earlier version computed the
    right path *prefix* per model and then always POSTed `{prefix}/responses`, so it
    400ed for every Claude model and every Chat-Completions-only family — i.e. for
    most of the catalogue. Resolve the **surface** as well as the prefix.

    Sampling is handled by dropping what the service names in a 400 rather than by
    carrying a per-model table. Those tables have gone stale twice in this
    collection's lifetime; the error message has not moved.
    """

    CHAT_ONLY = ("qwen.", "deepseek.", "zai.", "minimax.", "moonshotai.",
                 "mistral.", "nvidia.", "writer.", "openai.gpt-oss-safeguard",
                 "google.gemma-3")
    COMPLETION_TOKENS = ("openai.gpt-5.6",)
    TIERED = ("openai.gpt-oss", "google.gemma-", "xai.", "qwen.", "deepseek.",
              "zai.", "minimax.", "moonshotai.", "mistral.", "nvidia.", "writer.")
    TUNABLE = ("temperature", "top_p", "service_tier")

    def __init__(self, model, region=REGION, project=None, tier="default"):
        self.model, self.region, self.project = model, region, project
        self.tokens = TokenProvider(region=region)
        self.metrics = CallMetrics()
        if model.startswith("anthropic."):
            self.prefix, self.surface = "/anthropic/v1", "messages"
        elif model.startswith(self.CHAT_ONLY):
            self.prefix = "/openai/v1" if model.startswith(
                ("google.gemma-4", "openai.gpt-5", "xai.")) else "/v1"
            self.surface = "chat"
        else:
            self.prefix = "/openai/v1" if model.startswith(
                ("google.gemma-4", "openai.gpt-5", "xai.")) else "/v1"
            self.surface = "responses"
        self.tier = (tier if (tier == "default" or model.startswith(self.TIERED))
                     else "default")

    def _budget_field(self):
        if self.surface == "responses":
            return "max_output_tokens"
        if self.surface == "chat" and self.model.startswith(self.COMPLETION_TOKENS):
            return "max_completion_tokens"
        return "max_tokens"

    def _build(self, prompt, budget, schema, sampling):
        body = {"model": self.model, self._budget_field(): max(16, budget),
                **sampling}
        if self.surface == "messages":
            body["messages"] = [{"role": "user", "content": prompt}]
            return f"{self.prefix}/messages", body
        body["service_tier"] = self.tier
        if self.surface == "chat":
            body["messages"] = [{"role": "user", "content": prompt}]
            if schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "out", "strict": True, "schema": schema},
                }
            return f"{self.prefix}/chat/completions", body
        body["input"] = prompt
        body["store"] = False  # explicit: no 30-day retention
        if schema:
            body["text"] = {"format": {"type": "json_schema", "name": "out",
                                       "schema": schema, "strict": True}}
        return f"{self.prefix}/responses", body

    @staticmethod
    def _refused_param(message):
        """Which tunable the service just named, if any. See 02-migrating §7."""
        import re

        quoted = re.findall(r"[\\'`\\"]([a-z_]+)[\\'`\\"]", message or "")
        for name in quoted:
            if name in ProductionClient.TUNABLE:
                return name
        if "API" not in (message or ""):
            for name in ProductionClient.TUNABLE:
                if name in (message or ""):
                    return name
        return None

    def _extract(self, data):
        if self.surface == "messages":
            return "".join(b.get("text", "") for b in data.get("content", [])
                           if b.get("type") == "text")
        if self.surface == "chat":
            return (data.get("choices") or [{}])[0].get("message", {}).get(
                "content") or ""
        return response_text(data)

    def ask(self, prompt, *, max_output_tokens=512, temperature=None, top_p=None,
            schema=None):
        sampling = {}
        if temperature is not None:
            sampling["temperature"] = temperature
        if top_p is not None:
            sampling["top_p"] = top_p

        headers = {}
        if self.project:
            key = ("anthropic-workspace" if self.surface == "messages"
                   else "OpenAI-Project")
            headers[key] = self.project
        if self.surface == "messages":
            headers["anthropic-version"] = "2023-06-01"

        for _ in range(len(self.TUNABLE) + 1):
            path, body = self._build(prompt, max_output_tokens, schema, sampling)
            started = time.perf_counter()
            code, data = post(path, body, region=self.region,
                              headers=headers or None, timeout=90)
            self.metrics.record(code, time.perf_counter() - started)
            if code == 200:
                text = self._extract(data)
                if schema:
                    if not text.strip():
                        raise RuntimeError("empty output — raise max_output_tokens")
                    return parse_json_lenient(text)
                return text
            refused = self._refused_param(err(data))
            if refused == "service_tier" and self.tier != "default":
                self.tier = "default"
                continue
            if refused in sampling:
                sampling.pop(refused)
                continue
            raise RuntimeError(f"HTTP {code}: {err(data)}")
        raise RuntimeError("request still refused after dropping every tunable")


bot = ProductionClient(MODEL, project=project_id, tier="flex")'''

EXERCISE = '''print("plain     :", bot.ask("Name one benefit of fair-share scheduling.")[:120])
print(
    "structured:",
    bot.ask("Describe the Rust language.", schema=SCHEMA, max_output_tokens=400),
)
print("metrics   :", bot.metrics.report())

# The routing fix, exercised: the same class against a Chat-Completions-only model
# and a Messages-only model. Both 400ed in the previous version of this notebook.
for other in ("qwen.qwen3-32b", "anthropic.claude-haiku-4-5"):
    probe = ProductionClient(other, project=project_id, tier="flex")
    answer = probe.ask("Name one benefit of queues. One sentence.",
                       max_output_tokens=120, temperature=0.7, top_p=0.95)
    print(f"{other:28} [{probe.surface}] {' '.join(answer.split())[:60]}")'''

GUARDRAILS_MD = '''## 9b. Guardrails — not a `bedrock-mantle` feature, and the trap that hides it

Amazon Bedrock Guardrails is the service's content-safety control: denied topics,
content filters, word filters, PII redaction, contextual grounding. Every other
control in this notebook is reachable from `bedrock-mantle`. This one is not, and
the way it fails is worth more than the fact itself.

| How you might attach a guardrail | What happens |
|---|---|
| `guardrailConfig` in a `bedrock-mantle` request body | **400** `Unknown parameter: 'guardrailConfig'` |
| `guardrailConfig` in a `bedrock-runtime` `/openai/v1` body | **400** `Unknown parameter` |
| `X-Amzn-Bedrock-GuardrailIdentifier` header on `bedrock-runtime` `/openai/v1` | **200 — and silently ignored** |
| `guardrailConfig` on Converse | works — `stopReason=guardrail_intervened` |
| `guardrailIdentifier` on `InvokeModel` | works |
| `ApplyGuardrail` called directly | works, from any endpoint |

**The third row is the dangerous one.** The header is accepted, the request
succeeds, and the guardrail does nothing. A team that sets it and sees HTTP 200
has no protection and no error to tell them so. We verified this with a guardrail
that denies investment advice: with the header attached, the model returned a
direct buy recommendation.

So there are exactly two supported shapes:

1. **Use Converse on `bedrock-runtime`** and pass `guardrailConfig`. This is the
   canonical path and the one AWS documents.
2. **Call `ApplyGuardrail` yourself** — a standalone `bedrock-runtime` API that
   evaluates text against a guardrail and returns a verdict. It takes no model and
   costs no inference, so it works as a pre-filter on input and a post-filter on
   output *even for a `bedrock-mantle` workload*. That is the answer if you need
   both mantle and guardrails.

The cells below create a throwaway guardrail, demonstrate both shapes plus the
silent-header trap, and delete it again.'''

GUARDRAILS_SETUP = '''import boto3

control = boto3.client("bedrock", region_name=REGION)
runtime = boto3.client("bedrock-runtime", region_name=REGION)

# A deliberately narrow guardrail so the verdict is unambiguous. Created here and
# deleted in the last cell of this section -- like the demo project above, this
# notebook cleans up what it makes.
guardrail = control.create_guardrail(
    name="mantle-samples-hardening-demo",
    description="Throwaway guardrail for the per-model Bedrock samples. Safe to delete.",
    topicPolicyConfig={
        "topicsConfig": [
            {
                "name": "InvestmentAdvice",
                "definition": "Specific recommendations to buy or sell securities.",
                "examples": ["Should I buy AMZN stock?"],
                "type": "DENY",
            }
        ]
    },
    contentPolicyConfig={
        "filtersConfig": [
            {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"}
        ]
    },
    blockedInputMessaging="Blocked by guardrail (input).",
    blockedOutputsMessaging="Blocked by guardrail (output).",
)
GUARDRAIL_ID, GUARDRAIL_VERSION = guardrail["guardrailId"], guardrail["version"]
safe_print("created guardrail:", GUARDRAIL_ID, "version", GUARDRAIL_VERSION)

# A new guardrail takes a moment to become READY.
for _ in range(20):
    if control.get_guardrail(
        guardrailIdentifier=GUARDRAIL_ID, guardrailVersion=GUARDRAIL_VERSION
    )["status"] == "READY":
        break
    time.sleep(2)
print("status:", control.get_guardrail(
    guardrailIdentifier=GUARDRAIL_ID, guardrailVersion=GUARDRAIL_VERSION)["status"])'''

GUARDRAILS_APPLY = '''DENIED = "Should I buy AMZN stock right now? Give me a direct recommendation."
ALLOWED = "What is the difference between an index fund and an ETF?"

# Shape 2 first, because it is the one that works for a mantle workload: evaluate
# the text yourself, then decide whether to call the model at all.
print("ApplyGuardrail — a pre-filter you can run before any endpoint")
print("-" * 66)
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
    print(f"  {verdict['action']:22} topics={topics}  {text[:40]}")

print()
print("=> action=GUARDRAIL_INTERVENED means do not send it. Run the same call with")
print("   source='OUTPUT' on the model's reply to screen what you return.")'''

GUARDRAILS_CONVERSE = '''# Shape 1: guardrailConfig on Converse. Works across providers -- Nova, Claude and
# GPT-5.6 all honour it, and the stop reason names what happened.
print("Converse with guardrailConfig")
print("-" * 66)
for model in ("amazon.nova-micro-v1", "anthropic.claude-haiku-4-5"):
    try:
        reply = runtime.converse(
            modelId=resolve_runtime_id(model, REGION),
            messages=[{"role": "user", "content": [{"text": DENIED}]}],
            inferenceConfig={"maxTokens": 120},
            guardrailConfig={
                "guardrailIdentifier": GUARDRAIL_ID,
                "guardrailVersion": GUARDRAIL_VERSION,
            },
        )
        text = "".join(
            b.get("text", "") for b in reply["output"]["message"]["content"]
        )
        print(f"  {model:28} stop={reply['stopReason']:22} {text[:34]!r}")
    except Exception as exc:  # noqa: BLE001 - report, do not stop the notebook
        print(f"  {model:28} {type(exc).__name__}: {str(exc)[-60:]}")

# And the trap. The header is accepted and does nothing.
print()
print("The silent-header trap on bedrock-runtime's OpenAI APIs")
print("-" * 66)
signed_base = f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1"
for label, extra_headers in (
    ("no guardrail header  ", {}),
    ("WITH guardrail header", {
        "X-Amzn-Bedrock-GuardrailIdentifier": GUARDRAIL_ID,
        "X-Amzn-Bedrock-GuardrailVersion": GUARDRAIL_VERSION,
    }),
):
    payload = json.dumps({
        "model": f"us.{'openai.gpt-5.6-sol'}",
        "input": DENIED,
        "max_output_tokens": 200,
    })
    request = AWSRequest(
        method="POST", url=f"{signed_base}/responses", data=payload,
        headers={"Content-Type": "application/json", **extra_headers},
    )
    SigV4Auth(
        boto3.Session(region_name=REGION).get_credentials().get_frozen_credentials(),
        "bedrock", REGION,
    ).add_auth(request)
    reply = URLLib3Session(timeout=120).send(request.prepare())
    body = json.loads(reply.text) if reply.text.strip() else {}
    blocked = "Blocked by guardrail" in reply.text
    answer = " ".join(response_text(body).split())[:52]
    print(f"  {label} HTTP {reply.status_code} blocked={blocked!s:5} {answer!r}")

print()
print("=> Both calls answered. The header is not an error and not a control:")
print("   it is ignored. Guardrail a GPT model through Converse, or screen the")
print("   text yourself with ApplyGuardrail.")'''

GUARDRAILS_CLEANUP = '''# Delete the throwaway guardrail. Only the one this notebook created.
control.delete_guardrail(guardrailIdentifier=GUARDRAIL_ID)
remaining = [g["name"] for g in control.list_guardrails().get("guardrails", [])]
safe_print("deleted mantle-samples-hardening-demo | guardrails left in account:",
           len(remaining))'''


def main() -> None:
    nb = Notebook(PATH)

    # §2 fail-fast demo.
    old = nb.find("# A permanent 400 must not be retried — measure that it fails fast.")
    assert len(old) == 1, old
    nb.set_source(old[0], FAIL_FAST)

    # §9 production client.
    old = nb.find('class ProductionClient:')
    assert len(old) == 1, old
    nb.set_source(old[0], PRODUCTION_CLIENT)

    old = nb.find('print("plain     :", bot.ask("Name one benefit of fair-share scheduling.")[:120])')
    assert len(old) == 1, old
    nb.set_source(old[0], EXERCISE)

    # The client now needs response_text, parse_json_lenient and resolve_runtime_id.
    nb.sub("from bedrock import err, list_models, parse_json_lenient, post, response_text, safe_print",
           "from bedrock import (\n"
           "    err,\n"
           "    list_models,\n"
           "    parse_json_lenient,\n"
           "    post,\n"
           "    resolve_runtime_id,\n"
           "    response_text,\n"
           "    safe_print,\n"
           ")")

    # Guardrails section, inserted after the production-client exercise and before
    # the pre-launch checklist.
    anchor = nb.find("## 10. Pre-launch checklist")
    assert len(anchor) == 1, anchor
    at = anchor[0] - 1
    at = nb.insert_after(at, "markdown", GUARDRAILS_MD)
    at = nb.insert_after(at, "code", GUARDRAILS_SETUP)
    at = nb.insert_after(at, "code", GUARDRAILS_APPLY)
    at = nb.insert_after(at, "code", GUARDRAILS_CONVERSE)
    nb.insert_after(at, "code", GUARDRAILS_CLEANUP)

    # The checklist was silent on guardrails.
    nb.sub('    ("Quota escalation path known (Support case, not Service Quotas)", True),',
           '    ("Quota escalation path known (Support case, not Service Quotas)", True),\n'
           '    ("Guardrails applied via Converse or ApplyGuardrail, not a header", True),')

    # Helper table + gotchas.
    nb.sub("| `safe_print` | `print()` with account IDs, IAM principals and opaque service IDs redacted |",
           "| `resolve_runtime_id` | turns a model ID into the form Converse will accept, adding the `us.` profile prefix when one is required |\n"
           "| `safe_print` | `print()` with account IDs, IAM principals and opaque service IDs redacted |")
    nb.sub("| No CRIS / PT / batch | Cross-Region, Provisioned Throughput and batch are runtime-only |",
           "| No CRIS / PT / batch | Cross-Region, Provisioned Throughput and batch are runtime-only |\n"
           "| **No guardrails on mantle** | Not a parameter here. The `bedrock-runtime` OpenAI-API *header* is accepted and **silently ignored** — use Converse or `ApplyGuardrail` (§9b) |\n"
           "| Routing, not just prefixes | A client must resolve the API surface as well as the path, or it 400s on every Claude and Chat-Completions-only model |")
    nb.sub("- Defensive output handling\n- A pre-launch checklist you can run",
           "- Defensive output handling\n- Guardrails, and why they are not a mantle parameter\n"
           "- A pre-launch checklist you can run")
    nb.save()
    print(f"99-cross-cutting/03: {nb.changes} edits")


if __name__ == "__main__":
    main()
