#!/usr/bin/env python3
"""Round 4: two results from round 3 that their own probes could not support.

1. "Guardrails are not enforced on the runtime Responses surface" — round 3 asked
   for investment advice, which the model refused on its own. A self-refusal and a
   guardrail block look identical from outside. Redone with a DENY topic the model
   has no reason to refuse, plus the bogus-identifier test, which does not depend
   on model behaviour at all.
2. The mantle-only model list — round 3 compared exact IDs, so `openai.gpt-oss-120b`
   came out "mantle only" when its runtime twin is `openai.gpt-oss-120b-1:0`.

Writes quality/findings/36-claims-round4.json.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
from bedrock import endpoints_for, list_models, runtime_id_for, token  # noqa: E402

E = "us-east-1"
RT = f"https://bedrock-runtime.{E}.amazonaws.com"
MT = f"https://bedrock-mantle.{E}.api.aws"
TOK = token(E)
GROK = "us.xai.grok-4.6"


def call(base, path, body, extra=None, timeout=240):
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {TOK}"}
    if extra:
        h.update(extra)
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            return e.code, json.loads(b or b"{}")
        except Exception:
            return e.code, {"raw": b[:250].decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        return None, {"exception": f"{type(e).__name__}: {e}"[:180]}


def why(c, d, n=150):
    if c == 200:
        return "200"
    e = d.get("error") or d
    m = e.get("message") if isinstance(e, dict) else None
    return f"{c}: {str(m or json.dumps(d))[:n]}"


def rtext(d):
    return "".join(p.get("text", "") for it in (d.get("output") or [])
                   for p in (it.get("content") or []) if isinstance(p, dict))


F = {}

# --- 1. Guardrails, with a topic no model refuses on its own ---------------
try:
    import boto3
    bc = boto3.client("bedrock", region_name=E)
    gr = bc.create_guardrail(
        name=f"probe-r4-{int(time.time())}",
        description="Temporary probe: DENY a benign topic so a block is unambiguous.",
        topicPolicyConfig={"topicsConfig": [{
            "name": "Tulips",
            "definition": "Any discussion of tulips, tulip bulbs, or tulip cultivation.",
            "examples": ["Tell me about tulips."], "type": "DENY"}]},
        blockedInputMessaging="BLOCKED_BY_GUARDRAIL",
        blockedOutputsMessaging="BLOCKED_BY_GUARDRAIL")
    gid, gver = gr["guardrailId"], gr["version"]
    time.sleep(18)
    ASK = "In two sentences, describe how tulips are cultivated."
    HDR = {"X-Amzn-Bedrock-GuardrailIdentifier": gid,
           "X-Amzn-Bedrock-GuardrailVersion": gver}
    BAD = {"X-Amzn-Bedrock-GuardrailIdentifier": "gr-doesnotexist000",
           "X-Amzn-Bedrock-GuardrailVersion": "1"}
    G = {}

    # Control: no header at all. Proves the model WILL answer, so a later
    # refusal can only be the guardrail.
    c, d = call(RT, "/openai/v1/chat/completions",
                {"model": "openai.gpt-oss-20b-1:0",
                 "messages": [{"role": "user", "content": ASK}],
                 "max_completion_tokens": 300})
    t = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    G["control runtime chat, no header"] = {
        "status": why(c, d, 90), "answered": len(t) > 40, "head": t[:80]}

    c, d = call(RT, "/openai/v1/responses",
                {"model": GROK, "input": ASK, "max_output_tokens": 3000})
    G["control runtime responses, no header"] = {
        "status": why(c, d, 90), "answered": len(rtext(d)) > 40,
        "head": rtext(d)[:80]}

    # Now with the guardrail.
    for label, base, path, body, extra in (
        ("runtime chat", RT, "/openai/v1/chat/completions",
         {"model": "openai.gpt-oss-20b-1:0",
          "messages": [{"role": "user", "content": ASK}],
          "max_completion_tokens": 300}, HDR),
        ("runtime responses", RT, "/openai/v1/responses",
         {"model": GROK, "input": ASK, "max_output_tokens": 3000}, HDR),
        ("runtime messages", RT, "/anthropic/v1/messages",
         {"model": "us.anthropic.claude-opus-5", "max_tokens": 300,
          "messages": [{"role": "user", "content": ASK}]},
         {**HDR, "anthropic-version": "2023-06-01"}),
        ("mantle chat", MT, "/v1/chat/completions",
         {"model": "openai.gpt-oss-20b",
          "messages": [{"role": "user", "content": ASK}],
          "max_tokens": 300}, HDR),
        ("mantle responses", MT, "/openai/v1/responses",
         {"model": "openai.gpt-5.6-sol", "input": ASK,
          "max_output_tokens": 3000}, HDR),
    ):
        c, d = call(base, path, body, extra)
        if "responses" in path:
            text = rtext(d)
        elif "messages" in path:
            text = "".join(b.get("text", "") for b in (d.get("content") or [])
                           if isinstance(b, dict))
        else:
            text = ((d.get("choices") or [{}])[0].get("message") or {}).get(
                "content") or ""
        G[label + " + guardrail header"] = {
            "status": why(c, d, 90),
            "blocked": "BLOCKED_BY_GUARDRAIL" in text,
            "head": text[:80]}

    # The discriminator that does not depend on the model: a guardrail ID that
    # does not exist. A surface that enforces guardrails must reject it.
    for label, base, path, body in (
        ("runtime chat", RT, "/openai/v1/chat/completions",
         {"model": "openai.gpt-oss-20b-1:0",
          "messages": [{"role": "user", "content": "Say OK."}],
          "max_completion_tokens": 64}),
        ("runtime responses", RT, "/openai/v1/responses",
         {"model": GROK, "input": "Say OK.", "max_output_tokens": 2048}),
        ("runtime messages", RT, "/anthropic/v1/messages",
         {"model": "us.anthropic.claude-opus-5", "max_tokens": 64,
          "messages": [{"role": "user", "content": "Say OK."}]}),
        ("mantle chat", MT, "/v1/chat/completions",
         {"model": "openai.gpt-oss-20b",
          "messages": [{"role": "user", "content": "Say OK."}],
          "max_tokens": 64}),
        ("mantle responses", MT, "/openai/v1/responses",
         {"model": "openai.gpt-5.6-sol", "input": "Say OK.",
          "max_output_tokens": 2048}),
    ):
        extra = dict(BAD)
        if "messages" in path:
            extra["anthropic-version"] = "2023-06-01"
        c, d = call(base, path, body, extra)
        G[label + " + BOGUS id"] = why(c, d, 110)

    bc.delete_guardrail(guardrailIdentifier=gid)
    G["deleted"] = True
    F["guardrails"] = G
except Exception as e:  # noqa: BLE001
    F["guardrails"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

# --- 2. The true endpoint split, on normalised IDs ------------------------
mantle = sorted(list_models(E))
split = {"mantle_only": [], "both": []}
for m in mantle:
    rid = runtime_id_for(m, E)
    (split["both"] if rid else split["mantle_only"]).append(
        m if rid is None else f"{m} -> {rid}")
F["endpoint_split_normalised"] = split
F["counts"] = {"mantle": len(mantle),
               "mantle_only": len(split["mantle_only"]),
               "on_both": len(split["both"])}
# Spot-check endpoints_for agrees with runtime_id_for.
F["endpoints_for_spotcheck"] = {
    m: endpoints_for(m, E)
    for m in ("google.gemma-4-31b", "openai.gpt-oss-120b", "xai.grok-4.3",
              "xai.grok-4.6", "openai.gpt-5.5", "openai.gpt-5.6-sol",
              "zai.glm-4.6", "zai.glm-5", "moonshotai.kimi-k2-thinking")}

# --- 3. code_interpreter / web_search are per MODEL, not per endpoint -----
# Round 3's "Supported tool types are: function, mcp, custom, namespace,
# tool_search" came from a GROK request. Check whether gpt-5.6 gets a different
# list, which would make that enumeration model-specific rather than universal.
F["tool_support_is_per_model"] = {}
for mid, base, region_label in (("openai.gpt-5.6-sol", MT, "mantle"),
                                ("xai.grok-4.3", MT, "mantle")):
    for tool in ({"type": "web_search"},
                 {"type": "code_interpreter", "container": {"type": "auto"}}):
        c, d = call(base, "/openai/v1/responses",
                    {"model": mid, "input": "Hi", "tools": [tool],
                     "max_output_tokens": 2500})
        F["tool_support_is_per_model"][f"{region_label} {mid} {tool['type']}"] = why(
            c, d, 140)

out = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "36-claims-round4.json")
with open(out, "w") as f:
    json.dump(F, f, indent=2)
print(json.dumps(F, indent=2))
print(f"\nwrote {out}")
