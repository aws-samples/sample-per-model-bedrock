#!/usr/bin/env python3
"""Pin every row of the guardrail-attachment matrix.

This feeds a security table in 99-cross-cutting/03, so each row is measured two
independent ways:

  A. a DENY topic the model has no reason to refuse on its own (tulips), with a
     no-header control proving the model answers it;
  B. a guardrail identifier that does not exist -- a surface that enforces
     guardrails must reject it, and this signal does not depend on the model.

When A and B agree, the row is trustworthy. Writes
quality/findings/37-guardrail-matrix.json.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
from bedrock import token  # noqa: E402

E = "us-east-1"
RT = f"https://bedrock-runtime.{E}.amazonaws.com"
MT = f"https://bedrock-mantle.{E}.api.aws"
TOK = token(E)
ASK = "In two sentences, describe how tulips are cultivated."
BLOCK_TEXT = "BLOCKED_BY_GUARDRAIL"


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
        return None, {"exception": f"{type(e).__name__}: {e}"[:160]}


def brief(c, d, n=110):
    if c == 200:
        return "200"
    e = d.get("error") or d
    m = e.get("message") if isinstance(e, dict) else None
    return f"{c}: {str(m or json.dumps(d))[:n]}"


def text_of(kind, d):
    if kind == "responses":
        return "".join(p.get("text", "") for it in (d.get("output") or [])
                       for p in (it.get("content") or []) if isinstance(p, dict))
    if kind == "messages":
        return "".join(b.get("text", "") for b in (d.get("content") or [])
                       if isinstance(b, dict))
    return ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


# (label, base, path, kind, body-without-guardrail, extra headers)
SURFACES = [
    ("runtime Chat Completions", RT, "/openai/v1/chat/completions", "chat",
     {"model": "openai.gpt-oss-20b-1:0",
      "messages": [{"role": "user", "content": ASK}],
      "max_completion_tokens": 400}, {}),
    ("runtime Responses", RT, "/openai/v1/responses", "responses",
     {"model": "us.openai.gpt-5.6-sol", "input": ASK,
      "max_output_tokens": 3000}, {}),
    ("runtime Messages", RT, "/anthropic/v1/messages", "messages",
     {"model": "us.anthropic.claude-opus-5", "max_tokens": 400,
      "messages": [{"role": "user", "content": ASK}]},
     {"anthropic-version": "2023-06-01"}),
    ("mantle Chat Completions", MT, "/v1/chat/completions", "chat",
     {"model": "openai.gpt-oss-20b",
      "messages": [{"role": "user", "content": ASK}], "max_tokens": 400}, {}),
    ("mantle Responses", MT, "/openai/v1/responses", "responses",
     {"model": "openai.gpt-5.6-sol", "input": ASK,
      "max_output_tokens": 3000}, {}),
    ("mantle Messages", MT, "/anthropic/v1/messages", "messages",
     {"model": "anthropic.claude-opus-5", "max_tokens": 400,
      "messages": [{"role": "user", "content": ASK}]},
     {"anthropic-version": "2023-06-01"}),
]

import boto3  # noqa: E402
bc = boto3.client("bedrock", region_name=E)
gr = bc.create_guardrail(
    name=f"probe-matrix-{int(time.time())}",
    description="Temporary probe: DENY a benign topic so a block is unambiguous.",
    topicPolicyConfig={"topicsConfig": [{
        "name": "Tulips",
        "definition": "Any discussion of tulips, tulip bulbs, or tulip cultivation.",
        "examples": ["Tell me about tulips."], "type": "DENY"}]},
    blockedInputMessaging=BLOCK_TEXT, blockedOutputsMessaging=BLOCK_TEXT)
GID, GVER = gr["guardrailId"], gr["version"]
for _ in range(25):
    if bc.get_guardrail(guardrailIdentifier=GID,
                        guardrailVersion=GVER)["status"] == "READY":
        break
    time.sleep(2)

HDR = {"X-Amzn-Bedrock-GuardrailIdentifier": GID,
       "X-Amzn-Bedrock-GuardrailVersion": GVER}
BOGUS = {"X-Amzn-Bedrock-GuardrailIdentifier": "gr-doesnotexist000",
         "X-Amzn-Bedrock-GuardrailVersion": "1"}
CFG = {"guardrailIdentifier": GID, "guardrailVersion": GVER}

rows = []
try:
    for label, base, path, kind, body, extra in SURFACES:
        row = {"surface": label}

        c, d = call(base, path, body, extra)
        row["control_no_guardrail"] = brief(c, d, 70)
        row["control_answered"] = len(text_of(kind, d)) > 40

        c, d = call(base, path, body, {**extra, **HDR})
        row["header_status"] = brief(c, d, 70)
        row["header_blocked"] = BLOCK_TEXT in text_of(kind, d)

        c, d = call(base, path, {**body, "guardrailConfig": CFG}, extra)
        row["body_guardrailConfig"] = brief(c, d, 70)
        row["body_guardrailConfig_blocked"] = BLOCK_TEXT in text_of(kind, d)

        c, d = call(base, path,
                    {**body, "amazon-bedrock-guardrailConfig": CFG}, extra)
        row["body_amazon_prefixed"] = brief(c, d, 70)
        row["body_amazon_prefixed_blocked"] = BLOCK_TEXT in text_of(kind, d)

        short = {"model": body["model"], **({} if kind != "chat" else {}),
                 "max_tokens": 16}
        probe = dict(body)
        c, d = call(base, path, probe, {**extra, **BOGUS})
        row["bogus_id"] = brief(c, d, 80)
        row["validates_id"] = c is not None and c != 200

        # Two independent signals; note whether they agree.
        row["enforces"] = row["header_blocked"]
        row["signals_agree"] = row["header_blocked"] == row["validates_id"]
        rows.append(row)
        print(f"{label:26} header={'BLOCK' if row['header_blocked'] else 'pass ':6} "
              f"bogus={'reject' if row['validates_id'] else 'accept':6} "
              f"agree={row['signals_agree']}")
finally:
    bc.delete_guardrail(guardrailIdentifier=GID)
    print("guardrail deleted")

# Converse and ApplyGuardrail, for completeness.
extra_shapes = {}
rt = boto3.client("bedrock-runtime", region_name=E)
gr = bc.create_guardrail(
    name=f"probe-matrix2-{int(time.time())}",
    description="Temporary probe: DENY a benign topic so a block is unambiguous.",
    topicPolicyConfig={"topicsConfig": [{
        "name": "Tulips",
        "definition": "Any discussion of tulips, tulip bulbs, or tulip cultivation.",
        "examples": ["Tell me about tulips."], "type": "DENY"}]},
    blockedInputMessaging=BLOCK_TEXT, blockedOutputsMessaging=BLOCK_TEXT)
GID2, GVER2 = gr["guardrailId"], gr["version"]
for _ in range(25):
    if bc.get_guardrail(guardrailIdentifier=GID2,
                        guardrailVersion=GVER2)["status"] == "READY":
        break
    time.sleep(2)
try:
    r = rt.converse(modelId="openai.gpt-oss-20b-1:0",
                    messages=[{"role": "user", "content": [{"text": ASK}]}],
                    inferenceConfig={"maxTokens": 400},
                    guardrailConfig={"guardrailIdentifier": GID2,
                                     "guardrailVersion": GVER2})
    t = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
    extra_shapes["Converse guardrailConfig"] = {
        "stopReason": r.get("stopReason"), "blocked": BLOCK_TEXT in t}
    try:
        rt.converse(modelId="openai.gpt-oss-20b-1:0",
                    messages=[{"role": "user", "content": [{"text": "Hi"}]}],
                    inferenceConfig={"maxTokens": 16},
                    guardrailConfig={"guardrailIdentifier": "gr-doesnotexist000",
                                     "guardrailVersion": "1"})
        extra_shapes["Converse bogus id"] = "accepted (200)"
    except Exception as e:  # noqa: BLE001
        extra_shapes["Converse bogus id"] = f"{type(e).__name__}: {str(e)[:100]}"

    a = rt.apply_guardrail(guardrailIdentifier=GID2, guardrailVersion=GVER2,
                           source="INPUT", content=[{"text": {"text": ASK}}])
    extra_shapes["ApplyGuardrail"] = {"action": a.get("action")}
finally:
    bc.delete_guardrail(guardrailIdentifier=GID2)

out = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "37-guardrail-matrix.json")
with open(out, "w") as f:
    json.dump({"rows": rows, "other_shapes": extra_shapes}, f, indent=2)
print(json.dumps({"rows": rows, "other_shapes": extra_shapes}, indent=2))
print(f"\nwrote {out}")
