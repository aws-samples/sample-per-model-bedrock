#!/usr/bin/env python3
"""Probe Grok 4.6 and the bedrock-runtime feature surface in detail.

Everything the notebooks are about to assert gets measured here first. Writes
quality/findings/32-grok46-features.json.

    python3 grok46-and-runtime-features.py
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

REGION = "us-east-1"
WEST = "us-west-2"
RT = f"https://bedrock-runtime.{REGION}.amazonaws.com"
RT_W = f"https://bedrock-runtime.{WEST}.amazonaws.com"
MT_W = f"https://bedrock-mantle.{WEST}.api.aws"
MT = f"https://bedrock-mantle.{REGION}.api.aws"
TOK = {REGION: token(REGION), WEST: token(WEST)}
GROK_RT = "us.xai.grok-4.6"
GROK_MT = "xai.grok-4.6"
ANTH = {"anthropic-version": "2023-06-01"}


def call(base, path, body, region=REGION, extra=None, timeout=180, stream=False):
    raw = json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {TOK[region]}"}
    if extra:
        h.update(extra)
    req = urllib.request.Request(base + path, data=raw, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if stream:
                lines = [ln.decode("utf-8", "replace")
                         for ln in r.read().splitlines() if ln.strip()]
                return r.status, {"_stream_lines": len(lines),
                                  "_first": lines[0][:120] if lines else "",
                                  "_sample": [x[:90] for x in lines[:4]]}
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            return e.code, json.loads(b or b"{}")
        except Exception:
            return e.code, {"raw": b[:300].decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        return None, {"exception": f"{type(e).__name__}: {e}"[:200]}


def unknown_op(d):
    return "UnknownOperation" in json.dumps(d)[:400]


def brief(c, d, n=150):
    if unknown_op(d):
        return f"{c}: UnknownOperationException (path not served)"
    if c == 200:
        return "200"
    e = d.get("error") or d
    m = e.get("message") if isinstance(e, dict) else None
    return f"{c}: {str(m or json.dumps(d))[:n]}"


F = {}

# --- 1. The doc's own /v1/chat/completions example -------------------------
# docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html
# shows base_url ".../v1" on bedrock-runtime. Check it with the doc's own model.
F["doc_path_v1_chat"] = {}
for mid in ("us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5",
            "openai.gpt-oss-20b-1:0"):
    c, d = call(RT, "/v1/chat/completions",
                {"model": mid, "messages": [{"role": "user", "content": "Hi"}],
                 "max_completion_tokens": 16})
    F["doc_path_v1_chat"][mid] = brief(c, d)
c, d = call(RT, "/openai/v1/chat/completions",
            {"model": "us.anthropic.claude-sonnet-5",
             "messages": [{"role": "user", "content": "Hi"}],
             "max_completion_tokens": 16})
F["doc_path_v1_chat"]["control:/openai/v1 + sonnet-5"] = brief(c, d)

# --- 2. Claude on runtime /anthropic/v1/messages: which ID form? -----------
F["claude_id_forms"] = {}
for mid in ("us.anthropic.claude-haiku-4-5",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic.claude-haiku-4-5",
            "us.anthropic.claude-sonnet-4-6",
            "global.anthropic.claude-opus-5",
            "us.anthropic.claude-opus-5"):
    c, d = call(RT, "/anthropic/v1/messages",
                {"model": mid, "max_tokens": 16,
                 "messages": [{"role": "user", "content": "Hi"}]}, extra=ANTH)
    F["claude_id_forms"][mid] = brief(c, d, 110)

# --- 3. Grok 4.6 on mantle: us-west-2 only, per the model card -------------
F["grok_mantle"] = {}
for label, base, region in (("mantle us-west-2", MT_W, WEST),
                            ("mantle us-east-1", MT, REGION)):
    c, d = call(base, "/openai/v1/chat/completions",
                {"model": GROK_MT, "messages": [{"role": "user", "content": "Hi"}],
                 "max_completion_tokens": 2048}, region=region)
    F["grok_mantle"][label] = brief(c, d, 110)
c, d = call(RT_W, "/openai/v1/chat/completions",
            {"model": GROK_MT, "messages": [{"role": "user", "content": "Hi"}],
             "max_completion_tokens": 2048}, region=WEST)
F["grok_mantle"]["runtime us-west-2 bare id"] = brief(c, d, 110)

# --- 4. Reasoning effort: low / medium / high / xhigh ----------------------
Q = ("A bag has 4 red and 6 blue marbles. Two are drawn without replacement. "
     "What is the probability both are the same colour? Give the fraction.")
F["grok_effort"] = {}
for effort in ("low", "medium", "high", "xhigh", "minimal", "none"):
    t0 = time.time()
    c, d = call(RT, "/openai/v1/responses",
                {"model": GROK_RT, "input": Q, "reasoning": {"effort": effort},
                 "max_output_tokens": 3000})
    row = {"status": brief(c, d, 130), "secs": round(time.time() - t0, 1)}
    if c == 200:
        u = d.get("usage") or {}
        row["reasoning_tokens"] = (u.get("output_tokens_details") or {}).get(
            "reasoning_tokens")
        row["output_tokens"] = u.get("output_tokens")
        row["text_len"] = len("".join(
            part.get("text", "")
            for it in (d.get("output") or [])
            for part in (it.get("content") or []) if isinstance(part, dict)))
        row["output_types"] = [i.get("type") for i in (d.get("output") or [])]
    F["grok_effort"][effort] = row

# --- 5. include: reasoning.encrypted_content, and the chat surface ---------
c, d = call(RT, "/openai/v1/responses",
            {"model": GROK_RT, "input": Q, "reasoning": {"effort": "low"},
             "include": ["reasoning.encrypted_content"], "max_output_tokens": 3000})
F["grok_include_encrypted"] = {"status": brief(c, d)}
enc = None
if c == 200:
    items = d.get("output") or []
    F["grok_include_encrypted"]["output_types"] = [i.get("type") for i in items]
    for it in items:
        if it.get("encrypted_content"):
            enc = it
            F["grok_include_encrypted"]["encrypted_len"] = len(it["encrypted_content"])
            F["grok_include_encrypted"]["summary_field"] = it.get("summary")
            F["grok_include_encrypted"]["content_field"] = it.get("content")
    F["grok_include_encrypted"]["prev_id"] = d.get("id", "")[:14] + "..."
    F["grok_include_encrypted"]["store_echo"] = d.get("store")

# Without include= -- is encrypted_content present anyway?
c, d = call(RT, "/openai/v1/responses",
            {"model": GROK_RT, "input": Q, "max_output_tokens": 3000})
F["grok_no_include"] = {"status": brief(c, d)}
if c == 200:
    items = d.get("output") or []
    F["grok_no_include"]["output_types"] = [i.get("type") for i in items]
    F["grok_no_include"]["any_encrypted"] = any(i.get("encrypted_content")
                                                for i in items)

# Chat Completions: model card says it "does not return reasoning tokens"
c, d = call(RT, "/openai/v1/chat/completions",
            {"model": GROK_RT, "messages": [{"role": "user", "content": Q}],
             "max_completion_tokens": 3000})
F["grok_chat"] = {"status": brief(c, d)}
if c == 200:
    m = (d.get("choices") or [{}])[0].get("message") or {}
    u = d.get("usage") or {}
    F["grok_chat"]["message_keys"] = sorted(m)
    F["grok_chat"]["has_reasoning_field"] = "reasoning" in m
    F["grok_chat"]["content_len"] = len(m.get("content") or "")
    F["grok_chat"]["usage_reasoning_tokens"] = (
        u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    F["grok_chat"]["finish_reason"] = (d.get("choices") or [{}])[0].get(
        "finish_reason")
# reasoning_effort on the chat surface
for key, body in (
    ("chat reasoning_effort=high",
     {"model": GROK_RT, "messages": [{"role": "user", "content": "Hi"}],
      "reasoning_effort": "high", "max_completion_tokens": 2048}),
    ("chat reasoning={effort}",
     {"model": GROK_RT, "messages": [{"role": "user", "content": "Hi"}],
      "reasoning": {"effort": "high"}, "max_completion_tokens": 2048}),
):
    c, d = call(RT, "/openai/v1/chat/completions", body)
    F["grok_chat"][key] = brief(c, d, 110)

# --- 6. Multi-turn: feed encrypted reasoning back -------------------------
F["grok_multiturn"] = {}
if enc:
    followup = [
        {"role": "user", "content": Q},
        enc,
        {"role": "user", "content": "Now give the decimal to 4 places."},
    ]
    c, d = call(RT, "/openai/v1/responses",
                {"model": GROK_RT, "input": followup,
                 "include": ["reasoning.encrypted_content"],
                 "max_output_tokens": 3000})
    F["grok_multiturn"]["replay_encrypted"] = brief(c, d, 200)
    if c == 200:
        F["grok_multiturn"]["text"] = "".join(
            p.get("text", "") for it in (d.get("output") or [])
            for p in (it.get("content") or []) if isinstance(p, dict))[:200]

# previous_response_id / store on the runtime Responses surface
c, d = call(RT, "/openai/v1/responses",
            {"model": GROK_RT, "input": "Remember the number 41.",
             "store": True, "max_output_tokens": 2048})
F["grok_multiturn"]["store_true"] = brief(c, d, 160)
if c == 200:
    rid = d.get("id")
    F["grok_multiturn"]["store_echo"] = d.get("store")
    c2, d2 = call(RT, "/openai/v1/responses",
                  {"model": GROK_RT, "input": "What number did I say?",
                   "previous_response_id": rid, "max_output_tokens": 2048})
    F["grok_multiturn"]["previous_response_id"] = brief(c2, d2, 160)

# --- 7. Streaming on each runtime surface --------------------------------
F["streaming"] = {}
c, d = call(RT, "/openai/v1/chat/completions",
            {"model": GROK_RT, "messages": [{"role": "user", "content": "Count to 5."}],
             "max_completion_tokens": 2048, "stream": True}, stream=True)
F["streaming"]["chat"] = brief(c, d) + f" lines={d.get('_stream_lines')}"
c, d = call(RT, "/openai/v1/responses",
            {"model": GROK_RT, "input": "Count to 5.", "max_output_tokens": 2048,
             "stream": True}, stream=True)
F["streaming"]["responses"] = brief(c, d) + f" lines={d.get('_stream_lines')}"
c, d = call(RT, "/anthropic/v1/messages",
            {"model": "us.anthropic.claude-haiku-4-5", "max_tokens": 64,
             "messages": [{"role": "user", "content": "Count to 5."}],
             "stream": True}, extra=ANTH, stream=True)
F["streaming"]["messages"] = brief(c, d) + f" lines={d.get('_stream_lines')}"

# --- 8. Tools and structured output on grok 4.6 --------------------------
TOOL = {"type": "function", "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"], "additionalProperties": False}}
c, d = call(RT, "/openai/v1/responses",
            {"model": GROK_RT, "input": "Weather in Jakarta? Use the tool.",
             "tools": [TOOL], "max_output_tokens": 3000})
F["grok_tools_responses"] = brief(c, d, 160)
if c == 200:
    F["grok_tools_responses"] += " types=" + str(
        [i.get("type") for i in (d.get("output") or [])])

c, d = call(RT, "/openai/v1/chat/completions",
            {"model": GROK_RT, "messages": [{"role": "user",
                                             "content": "Weather in Jakarta?"}],
             "tools": [{"type": "function", "function": {
                 "name": "get_weather",
                 "parameters": TOOL["parameters"]}}],
             "max_completion_tokens": 3000})
F["grok_tools_chat"] = brief(c, d, 160)
if c == 200:
    m = (d.get("choices") or [{}])[0].get("message") or {}
    F["grok_tools_chat"] += f" tool_calls={len(m.get('tool_calls') or [])}"

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"},
                                           "celsius": {"type": "number"}},
          "required": ["city", "celsius"], "additionalProperties": False}
for label, base, mid, region in (("runtime", RT, GROK_RT, REGION),
                                 ("mantle us-west-2", MT_W, GROK_MT, WEST)):
    c, d = call(base, "/openai/v1/responses",
                {"model": mid, "input": "Jakarta is 31 C. Return JSON.",
                 "text": {"format": {"type": "json_schema", "name": "w",
                                     "schema": SCHEMA, "strict": True}},
                 "max_output_tokens": 3000}, region=region)
    F[f"grok_structured_{label.replace(' ', '_')}"] = brief(c, d, 160)

# --- 9. Guardrails on the runtime OpenAI surface -------------------------
# Previously the mantle header was accepted and silently ignored. Recheck on
# runtime, which the docs now name as the endpoint where Guardrails live.
F["guardrails"] = {}
try:
    import boto3
    bc = boto3.client("bedrock", region_name=REGION)
    gr = bc.create_guardrail(
        name=f"probe-runtime-openai-{int(time.time())}",
        description="Temporary probe of guardrail enforcement on /openai/v1.",
        topicPolicyConfig={"topicsConfig": [{
            "name": "InvestmentAdvice",
            "definition": "Recommendations about buying or selling securities.",
            "examples": ["Should I buy Amazon stock?"], "type": "DENY"}]},
        blockedInputMessaging="Blocked input.",
        blockedOutputsMessaging="Blocked output.")
    gid, gver = gr["guardrailId"], gr["version"]
    F["guardrails"]["created"] = True
    time.sleep(12)
    ask = "Should I buy Amazon stock right now? Give a direct recommendation."

    c, d = call(RT, "/openai/v1/chat/completions",
                {"model": "openai.gpt-oss-20b-1:0",
                 "messages": [{"role": "user", "content": ask}],
                 "max_completion_tokens": 300},
                extra={"X-Amzn-Bedrock-GuardrailIdentifier": gid,
                       "X-Amzn-Bedrock-GuardrailVersion": gver})
    txt = ""
    if c == 200:
        txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    F["guardrails"]["openai_v1_header"] = {
        "status": brief(c, d, 120), "blocked": "Blocked input" in txt,
        "text_head": txt[:110]}

    body_g = {"model": "openai.gpt-oss-20b-1:0",
              "messages": [{"role": "user", "content": ask}],
              "max_completion_tokens": 300,
              "amazon-bedrock-guardrailConfig": {
                  "guardrailIdentifier": gid, "guardrailVersion": gver}}
    c, d = call(RT, "/openai/v1/chat/completions", body_g)
    txt = ""
    if c == 200:
        txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    F["guardrails"]["openai_v1_body"] = {
        "status": brief(c, d, 120), "blocked": "Blocked input" in txt,
        "text_head": txt[:110]}

    rt = boto3.client("bedrock-runtime", region_name=REGION)
    r = rt.converse(modelId="openai.gpt-oss-20b-1:0",
                    messages=[{"role": "user", "content": [{"text": ask}]}],
                    inferenceConfig={"maxTokens": 300},
                    guardrailConfig={"guardrailIdentifier": gid,
                                     "guardrailVersion": gver})
    ctext = "".join(b.get("text", "")
                    for b in r["output"]["message"]["content"])
    F["guardrails"]["converse_guardrailConfig"] = {
        "stopReason": r.get("stopReason"), "blocked": "Blocked" in ctext,
        "text_head": ctext[:110]}

    a = rt.apply_guardrail(guardrailIdentifier=gid, guardrailVersion=gver,
                           source="INPUT", content=[{"text": {"text": ask}}])
    F["guardrails"]["apply_guardrail"] = {"action": a.get("action")}

    bc.delete_guardrail(guardrailIdentifier=gid)
    F["guardrails"]["deleted"] = True
except Exception as e:  # noqa: BLE001
    F["guardrails"]["error"] = f"{type(e).__name__}: {str(e)[:200]}"

# --- 10. Prompt caching on grok 4.6 -------------------------------------
big = ("You are a meticulous reviewer of aviation maintenance logs. " * 260)
F["grok_caching"] = []
for i in range(3):
    c, d = call(RT, "/openai/v1/chat/completions",
                {"model": GROK_RT,
                 "messages": [{"role": "system", "content": big},
                              {"role": "user", "content": f"Reply with the number {i}."}],
                 "max_completion_tokens": 2048})
    if c == 200:
        u = d.get("usage") or {}
        F["grok_caching"].append({
            "call": i, "prompt_tokens": u.get("prompt_tokens"),
            "details": u.get("prompt_tokens_details")})
    else:
        F["grok_caching"].append({"call": i, "status": brief(c, d, 110)})

# --- 11. Count tokens: model card says unsupported on runtime for grok ---
F["count_tokens"] = {}
for mid in (GROK_RT, "us.anthropic.claude-haiku-4-5"):
    try:
        import boto3
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        r = rt.count_tokens(modelId=mid, input={"converse": {
            "messages": [{"role": "user", "content": [{"text": "Hello there"}]}]}})
        F["count_tokens"][mid] = r.get("inputTokens")
    except Exception as e:  # noqa: BLE001
        F["count_tokens"][mid] = f"{type(e).__name__}: {str(e)[:130]}"

out = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "32-grok46-features.json")
with open(out, "w") as f:
    json.dump(F, f, indent=2)
print(json.dumps(F, indent=2))
print(f"\nwrote {out}")
