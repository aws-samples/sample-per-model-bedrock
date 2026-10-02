#!/usr/bin/env python3
"""Third pass: the questions rounds 1-2 could not answer.

Round 2 tried to test "server-side tools are unavailable on bedrock-runtime"
using Grok, which does not support them on EITHER endpoint -- so that probe
measured the model, not the endpoint. Redone here with a model that does.
Also: real-image vision, prompt_cache_key, effort separation on a hard prompt,
and a repeat of the security-relevant guardrail result.

Writes quality/findings/34-claims-round3.json.
"""
import base64
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
from bedrock import bands_png, token  # noqa: E402

E, W = "us-east-1", "us-west-2"
RT = f"https://bedrock-runtime.{E}.amazonaws.com"
MT_E = f"https://bedrock-mantle.{E}.api.aws"
MT_W = f"https://bedrock-mantle.{W}.api.aws"
TOK = {E: token(E), W: token(W)}
GROK = "us.xai.grok-4.6"
GROK_MT = "xai.grok-4.6"


def call(base, path, body, region=E, extra=None, timeout=240):
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {TOK[region]}"}
    if extra:
        h.update(extra)
    req = urllib.request.Request(base + path, data=data, headers=h, method="POST")
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


def why(c, d, n=170):
    if "UnknownOperation" in json.dumps(d)[:400]:
        return f"{c}: UnknownOperationException (path not served)"
    if c == 200:
        return "200"
    e = d.get("error") or d
    m = e.get("message") if isinstance(e, dict) else None
    return f"{c}: {str(m or json.dumps(d))[:n]}"


def rtext(d):
    return "".join(p.get("text", "") for it in (d.get("output") or [])
                   for p in (it.get("content") or []) if isinstance(p, dict))


F = {}

# --- 1. Server-side tools: same model, both endpoints ----------------------
# gpt-5.x is the family that HAS web search on bedrock-mantle. On runtime only
# the gpt-5.6 profiles exist, so pair each endpoint with an ID it accepts.
WS = [{"type": "web_search"}]
ASK = "Using web search, what is the headline feature of the newest AWS Bedrock release?"
F["server_side_tools"] = {}
for label, base, mid, region in (
    ("mantle gpt-5.6-sol", MT_E, "openai.gpt-5.6-sol", E),
    ("runtime us.gpt-5.6-sol", RT, "us.openai.gpt-5.6-sol", E),
    ("mantle gpt-5.5", MT_E, "openai.gpt-5.5", E),
):
    c, d = call(base, "/openai/v1/responses",
                {"model": mid, "input": ASK, "tools": WS,
                 "max_output_tokens": 4000}, region=region)
    row = {"status": why(c, d, 200)}
    if c == 200:
        row["output_types"] = [i.get("type") for i in (d.get("output") or [])]
        row["text_head"] = rtext(d)[:130]
    F["server_side_tools"][label] = row

# The tool types the service enumerated in its own error message.
F["tool_types"] = {}
for t in ("function", "mcp", "custom", "namespace", "tool_search",
          "web_search", "code_interpreter", "image_generation"):
    if t == "function":
        spec = {"type": "function", "name": "f", "description": "d",
                "parameters": {"type": "object", "properties": {},
                               "additionalProperties": False}}
    elif t == "mcp":
        spec = {"type": "mcp", "server_label": "x",
                "server_url": "https://example.invalid/mcp"}
    elif t == "custom":
        spec = {"type": "custom", "name": "c", "description": "d"}
    else:
        spec = {"type": t}
    c, d = call(MT_E, "/openai/v1/responses",
                {"model": "openai.gpt-5.6-sol", "input": "Hi",
                 "tools": [spec], "max_output_tokens": 2000})
    F["tool_types"][f"mantle {t}"] = why(c, d, 130)
    c, d = call(RT, "/openai/v1/responses",
                {"model": "us.openai.gpt-5.6-sol", "input": "Hi",
                 "tools": [spec], "max_output_tokens": 2000})
    F["tool_types"][f"runtime {t}"] = why(c, d, 130)

# --- 2. Vision on grok 4.6, with a real image ----------------------------
# Round 2 used a 1x1 PNG and got "Invalid or unsupported image format", which
# does not distinguish "no vision" from "degenerate image".
png = bands_png([(220, 30, 30), (30, 140, 60), (40, 70, 200)])  # red, green, blue
b64 = base64.b64encode(png).decode()
F["vision"] = {"image_bytes": len(png)}
for label, base, mid, region in (("runtime", RT, GROK, E),
                                 ("mantle_us_west_2", MT_W, GROK_MT, W)):
    c, d = call(base, "/openai/v1/responses",
                {"model": mid, "max_output_tokens": 3000,
                 "input": [{"role": "user", "content": [
                     {"type": "input_text",
                      "text": "List the colours of the horizontal bands, top to bottom."},
                     {"type": "input_image",
                      "image_url": f"data:image/png;base64,{b64}"}]}]},
                region=region)
    F["vision"][f"{label}_responses"] = why(c, d, 160)
    if c == 200:
        F["vision"][f"{label}_text"] = rtext(d)[:160]
# A model that is documented as multimodal, as a control on the image itself.
c, d = call(MT_E, "/openai/v1/responses",
            {"model": "google.gemma-4-31b", "max_output_tokens": 2000,
             "input": [{"role": "user", "content": [
                 {"type": "input_text", "text": "List the band colours."},
                 {"type": "input_image",
                  "image_url": f"data:image/png;base64,{b64}"}]}]})
F["vision"]["control_gemma4_mantle"] = why(c, d, 160)
if c == 200:
    F["vision"]["control_gemma4_text"] = rtext(d)[:160]

# --- 3. prompt_cache_key ------------------------------------------------
big = "You are a meticulous reviewer of aviation maintenance logs. " * 1200
F["prompt_cache_key"] = []
for i in range(4):
    c, d = call(RT, "/openai/v1/chat/completions",
                {"model": GROK,
                 "messages": [{"role": "system", "content": big},
                              {"role": "user", "content": f"Say {i}."}],
                 "max_completion_tokens": 2048,
                 "prompt_cache_key": "samples-round3-fixed-key"})
    if c == 200:
        u = d.get("usage") or {}
        F["prompt_cache_key"].append(
            {"call": i, "prompt": u.get("prompt_tokens"),
             "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")})
    else:
        F["prompt_cache_key"].append({"call": i, "status": why(c, d, 110)})

# --- 4. Effort separation on a genuinely hard prompt --------------------
HARD = ("Let f(n) be the number of ways to tile a 3xn rectangle with 1x2 dominoes. "
        "Derive a closed-form recurrence, prove it, then compute f(12) exactly. "
        "Show every step of the proof.")


def one(args):
    effort, _ = args
    c, d = call(RT, "/openai/v1/responses",
                {"model": GROK, "input": HARD, "reasoning": {"effort": effort},
                 "max_output_tokens": 16000}, timeout=420)
    if c != 200:
        return effort, None
    u = d.get("usage") or {}
    return effort, (u.get("output_tokens_details") or {}).get("reasoning_tokens")


jobs = [(e, i) for e in ("none", "low", "medium", "high", "xhigh")
        for i in range(3)]
hard = {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for effort, rt in ex.map(one, jobs):
        hard.setdefault(effort, []).append(rt)
F["effort_hard_prompt"] = {
    e: {"samples": v,
        "median": statistics.median([x for x in v if x is not None])
        if any(x is not None for x in v) else None}
    for e, v in hard.items()}

# --- 5. Guardrails, repeated, on both OpenAI surfaces ------------------
F["guardrails"] = {}
try:
    import boto3
    bc = boto3.client("bedrock", region_name=E)
    gr = bc.create_guardrail(
        name=f"probe-r3-{int(time.time())}",
        description="Temporary probe of guardrail enforcement on OpenAI surfaces.",
        topicPolicyConfig={"topicsConfig": [{
            "name": "InvestmentAdvice",
            "definition": "Recommendations about buying or selling securities.",
            "examples": ["Should I buy Amazon stock?"], "type": "DENY"}]},
        blockedInputMessaging="BLOCKED_IN",
        blockedOutputsMessaging="BLOCKED_OUT")
    gid, gver = gr["guardrailId"], gr["version"]
    time.sleep(15)
    ask = "Should I buy Amazon stock right now? Give a direct recommendation."
    HDR = {"X-Amzn-Bedrock-GuardrailIdentifier": gid,
           "X-Amzn-Bedrock-GuardrailVersion": gver}

    for rep in range(2):
        c, d = call(RT, "/openai/v1/chat/completions",
                    {"model": "openai.gpt-oss-20b-1:0",
                     "messages": [{"role": "user", "content": ask}],
                     "max_completion_tokens": 400}, extra=HDR)
        t = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        F["guardrails"][f"runtime chat header rep{rep}"] = {
            "status": why(c, d, 90), "blocked": "BLOCKED_IN" in t,
            "head": t[:70]}

    c, d = call(RT, "/openai/v1/responses",
                {"model": GROK, "input": ask, "max_output_tokens": 3000},
                extra=HDR)
    F["guardrails"]["runtime responses header"] = {
        "status": why(c, d, 90), "blocked": "BLOCKED_IN" in rtext(d),
        "head": rtext(d)[:70]}

    c, d = call(RT, "/anthropic/v1/messages",
                {"model": "us.anthropic.claude-opus-5", "max_tokens": 400,
                 "messages": [{"role": "user", "content": ask}]},
                extra={**HDR, "anthropic-version": "2023-06-01"})
    at = "".join(b.get("text", "") for b in (d.get("content") or [])
                 if isinstance(b, dict))
    F["guardrails"]["runtime messages header"] = {
        "status": why(c, d, 90), "blocked": "BLOCKED_IN" in at, "head": at[:70]}

    # Same header on bedrock-mantle: previously accepted and ignored.
    c, d = call(MT_E, "/v1/chat/completions",
                {"model": "openai.gpt-oss-20b",
                 "messages": [{"role": "user", "content": ask}],
                 "max_tokens": 400}, extra=HDR)
    t = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    F["guardrails"]["mantle chat header"] = {
        "status": why(c, d, 90), "blocked": "BLOCKED_IN" in t, "head": t[:70]}

    # Does an unknown guardrail id error, or get ignored? Tells us whether the
    # header is validated at all on each surface.
    BAD = {"X-Amzn-Bedrock-GuardrailIdentifier": "gr-doesnotexist000",
           "X-Amzn-Bedrock-GuardrailVersion": "1"}
    c, d = call(RT, "/openai/v1/chat/completions",
                {"model": "openai.gpt-oss-20b-1:0",
                 "messages": [{"role": "user", "content": "Say OK."}],
                 "max_completion_tokens": 64}, extra=BAD)
    F["guardrails"]["runtime chat bogus id"] = why(c, d, 130)
    c, d = call(MT_E, "/v1/chat/completions",
                {"model": "openai.gpt-oss-20b",
                 "messages": [{"role": "user", "content": "Say OK."}],
                 "max_tokens": 64}, extra=BAD)
    F["guardrails"]["mantle chat bogus id"] = why(c, d, 130)

    bc.delete_guardrail(guardrailIdentifier=gid)
    F["guardrails"]["deleted"] = True
except Exception as e:  # noqa: BLE001
    F["guardrails"]["error"] = f"{type(e).__name__}: {str(e)[:200]}"

# --- 6. Endpoint availability per model, both catalogues ---------------
F["endpoint_split"] = {}
try:
    import boto3
    bc = boto3.client("bedrock", region_name=E)
    rt_ids = {m["modelId"] for m in bc.list_foundation_models()["modelSummaries"]}
    c, d = call(MT_E, "/v1/models", {}, timeout=60)
    # /v1/models is a GET; do it properly
    req = urllib.request.Request(
        MT_E + "/v1/models",
        headers={"Authorization": f"Bearer {TOK[E]}"}, method="GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        mt_ids = {m["id"] for m in json.loads(r.read())["data"]}
    F["endpoint_split"]["mantle_only"] = sorted(mt_ids - rt_ids)
    F["endpoint_split"]["both_same_id"] = sorted(mt_ids & rt_ids)
    F["endpoint_split"]["mantle_count"] = len(mt_ids)
    F["endpoint_split"]["runtime_count"] = len(rt_ids)
except Exception as e:  # noqa: BLE001
    F["endpoint_split"]["error"] = f"{type(e).__name__}: {str(e)[:160]}"

out = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "34-claims-round3.json")
with open(out, "w") as f:
    json.dump(F, f, indent=2)
print(json.dumps(F, indent=2))
print(f"\nwrote {out}")
