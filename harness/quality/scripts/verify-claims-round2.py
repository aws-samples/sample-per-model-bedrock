#!/usr/bin/env python3
"""Second pass: settle every claim the notebooks are about to make.

Round 1 produced single samples and a few results that contradict AWS's own
documentation. Anything that will become a sentence in a notebook is measured
here with enough repetition to be worth asserting. Writes
quality/findings/33-claims-round2.json.
"""
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
from bedrock import token  # noqa: E402

E, W = "us-east-1", "us-west-2"
RT = f"https://bedrock-runtime.{E}.amazonaws.com"
MT_W = f"https://bedrock-mantle.{W}.api.aws"
MT_E = f"https://bedrock-mantle.{E}.api.aws"
TOK = {E: token(E), W: token(W)}
GROK = "us.xai.grok-4.6"
GROK_MT = "xai.grok-4.6"
ANTH = {"anthropic-version": "2023-06-01"}
CLAUDE_RT = "us.anthropic.claude-opus-5"


def call(base, path, body, region=E, extra=None, timeout=240, raw_lines=False):
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {TOK[region]}"}
    if extra:
        h.update(extra)
    req = urllib.request.Request(base + path, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if raw_lines:
                ls = [x.decode("utf-8", "replace")
                      for x in r.read().splitlines() if x.strip()]
                return r.status, {"_n": len(ls), "_head": ls[:3]}
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


def resp_text(d):
    return "".join(p.get("text", "") for it in (d.get("output") or [])
                   for p in (it.get("content") or []) if isinstance(p, dict))


F = {}
Q = ("A bag has 4 red and 6 blue marbles. Two are drawn without replacement. "
     "What is the probability both are the same colour? Give the fraction.")

# --- A. Reasoning effort with repetition ----------------------------------
# Round 1 saw low=250 medium=418 high=259 xhigh=250 -- non-monotonic from one
# sample each. Take several samples before saying anything about ordering.
def one_effort(args):
    effort, i = args
    c, d = call(RT, "/openai/v1/responses",
                {"model": GROK, "input": Q, "reasoning": {"effort": effort},
                 "max_output_tokens": 6000})
    if c != 200:
        return effort, None
    u = d.get("usage") or {}
    return effort, (u.get("output_tokens_details") or {}).get("reasoning_tokens")


jobs = [(e, i) for e in ("none", "low", "medium", "high", "xhigh")
        for i in range(4)]
F["effort_samples"] = {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for effort, rt in ex.map(one_effort, jobs):
        F["effort_samples"].setdefault(effort, []).append(rt)
F["effort_summary"] = {
    e: {"samples": v,
        "median": statistics.median([x for x in v if x is not None])
        if any(x is not None for x in v) else None}
    for e, v in F["effort_samples"].items()}

# --- B. Structured output on grok: is it actually enforced? ---------------
SCHEMA = {"type": "object",
          "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
          "required": ["city", "celsius"], "additionalProperties": False}
F["structured"] = {}
for label, base, mid, region in (("runtime", RT, GROK, E),
                                 ("mantle_us_west_2", MT_W, GROK_MT, W)):
    c, d = call(base, "/openai/v1/responses",
                {"model": mid,
                 "input": "Jakarta is 31 degrees C. Also tell me a joke.",
                 "text": {"format": {"type": "json_schema", "name": "w",
                                     "schema": SCHEMA, "strict": True}},
                 "max_output_tokens": 6000}, region=region)
    row = {"status": why(c, d)}
    if c == 200:
        t = resp_text(d)
        row["text"] = t[:200]
        try:
            parsed = json.loads(t)
            row["parses"] = True
            row["keys"] = sorted(parsed)
            row["conforms"] = sorted(parsed) == ["celsius", "city"]
        except Exception as exc:  # noqa: BLE001
            row["parses"] = False
            row["parse_error"] = str(exc)[:90]
    F["structured"][label] = row

# --- C. encrypted_content with and without include= -----------------------
F["encrypted"] = {}
for label, extra_body in (("with_include",
                           {"include": ["reasoning.encrypted_content"]}),
                          ("without_include", {})):
    lens = []
    for _ in range(2):
        c, d = call(RT, "/openai/v1/responses",
                    {"model": GROK, "input": Q, "reasoning": {"effort": "low"},
                     "max_output_tokens": 6000, **extra_body})
        if c == 200:
            lens.append([len(i.get("encrypted_content") or "")
                         for i in (d.get("output") or [])])
        else:
            lens.append(why(c, d, 90))
    F["encrypted"][label] = lens

# --- D. Prompt caching: bigger prefix ------------------------------------
F["caching"] = {}
for size, reps in (("~2.6k", 260), ("~12k", 1200), ("~24k", 2400)):
    big = "You are a meticulous reviewer of aviation maintenance logs. " * reps
    seq = []
    for i in range(3):
        c, d = call(RT, "/openai/v1/chat/completions",
                    {"model": GROK,
                     "messages": [{"role": "system", "content": big},
                                  {"role": "user", "content": f"Say {i}."}],
                     "max_completion_tokens": 2048})
        if c == 200:
            u = d.get("usage") or {}
            seq.append({"prompt": u.get("prompt_tokens"),
                        "details": u.get("prompt_tokens_details")})
        else:
            seq.append(why(c, d, 90))
    F["caching"][size] = seq

# --- E. Streaming on all three runtime surfaces, valid models ------------
F["streaming"] = {}
c, d = call(RT, "/openai/v1/chat/completions",
            {"model": GROK, "messages": [{"role": "user", "content": "Count to 5."}],
             "max_completion_tokens": 3000, "stream": True}, raw_lines=True)
F["streaming"]["chat"] = {"status": why(c, d), **d}
c, d = call(RT, "/openai/v1/responses",
            {"model": GROK, "input": "Count to 5.", "max_output_tokens": 3000,
             "stream": True}, raw_lines=True)
F["streaming"]["responses"] = {"status": why(c, d), **d}
c, d = call(RT, "/anthropic/v1/messages",
            {"model": CLAUDE_RT, "max_tokens": 128,
             "messages": [{"role": "user", "content": "Count to 5."}],
             "stream": True}, extra=ANTH, raw_lines=True)
F["streaming"]["messages"] = {"status": why(c, d), **d}

# --- F. Which Anthropic inference profiles exist -------------------------
import boto3  # noqa: E402
bc = boto3.client("bedrock", region_name=E)
profs = sorted(p["inferenceProfileId"]
               for p in bc.list_inference_profiles(maxResults=1000)[
                   "inferenceProfileSummaries"])
F["anthropic_profiles"] = [p for p in profs if "anthropic" in p]
F["openai_xai_profiles"] = [p for p in profs
                            if "openai" in p or "xai" in p]

# --- G. Doc claims about the runtime Responses surface -------------------
F["doc_claims"] = {}
c, d = call(RT, "/openai/v1/responses",
            {"model": GROK, "input": "Hi", "max_output_tokens": 2048,
             "background": True})
F["doc_claims"]["runtime background=true"] = why(c, d)
c, d = call(MT_W, "/openai/v1/responses",
            {"model": GROK_MT, "input": "Hi", "max_output_tokens": 2048,
             "background": True}, region=W)
F["doc_claims"]["mantle background=true"] = why(c, d)

# Application inference profile as the target -> doc says 400 on runtime
try:
    ap = bc.create_inference_profile(
        inferenceProfileName=f"probe{int(time.time())}",
        description="Temporary probe of application inference profile support.",
        modelSource={"copyFrom":
                     f"arn:aws:bedrock:{E}:{boto3.client('sts').get_caller_identity()['Account']}"
                     f":inference-profile/{GROK}"})
    arn = ap["inferenceProfileArn"]
    F["doc_claims"]["app_profile_created"] = True
    time.sleep(5)
    c, d = call(RT, "/openai/v1/responses",
                {"model": arn, "input": "Hi", "max_output_tokens": 2048})
    F["doc_claims"]["runtime responses + app profile"] = why(c, d)
    c, d = call(RT, "/openai/v1/chat/completions",
                {"model": arn, "messages": [{"role": "user", "content": "Hi"}],
                 "max_completion_tokens": 2048})
    F["doc_claims"]["runtime chat + app profile"] = why(c, d)
    try:
        rt = boto3.client("bedrock-runtime", region_name=E)
        r = rt.converse(modelId=arn,
                        messages=[{"role": "user",
                                   "content": [{"text": "Hi"}]}],
                        inferenceConfig={"maxTokens": 2048})
        F["doc_claims"]["converse + app profile"] = "ok " + str(r.get("stopReason"))
    except Exception as e:  # noqa: BLE001
        F["doc_claims"]["converse + app profile"] = f"{type(e).__name__}: {str(e)[:130]}"
    bc.delete_inference_profile(inferenceProfileIdentifier=arn)
    F["doc_claims"]["app_profile_deleted"] = True
except Exception as e:  # noqa: BLE001
    F["doc_claims"]["app_profile_error"] = f"{type(e).__name__}: {str(e)[:180]}"

# output_config.format on Messages: doc says mantle rejects, runtime is via Converse
OC = {"model": CLAUDE_RT, "max_tokens": 256,
      "messages": [{"role": "user", "content": "Jakarta, 31C. JSON only."}],
      "output_config": {"format": {"type": "json_schema", "name": "w",
                                   "schema": SCHEMA}}}
c, d = call(RT, "/anthropic/v1/messages", OC, extra=ANTH)
F["doc_claims"]["runtime messages output_config.format"] = why(c, d)
c, d = call(MT_E, "/anthropic/v1/messages",
            {**OC, "model": "anthropic.claude-opus-5"}, extra=ANTH)
F["doc_claims"]["mantle messages output_config.format"] = why(c, d)

# Server-side tools on runtime Responses: doc says unavailable
for label, tools in (("web_search", [{"type": "web_search"}]),
                     ("code_interpreter", [{"type": "code_interpreter",
                                            "container": {"type": "auto"}}])):
    c, d = call(RT, "/openai/v1/responses",
                {"model": GROK, "input": "What is the AWS Bedrock pricing page URL?",
                 "tools": tools, "max_output_tokens": 3000})
    F["doc_claims"][f"runtime responses {label}"] = why(c, d, 140)
    c, d = call(MT_W, "/openai/v1/responses",
                {"model": GROK_MT, "input": "What is the AWS Bedrock pricing URL?",
                 "tools": tools, "max_output_tokens": 3000}, region=W)
    F["doc_claims"][f"mantle responses {label}"] = why(c, d, 140)

# --- H. Service tiers on grok 4.6 ---------------------------------------
F["service_tier"] = {}
for tier in ("default", "flex", "priority"):
    c, d = call(RT, "/openai/v1/chat/completions",
                {"model": GROK, "messages": [{"role": "user", "content": "Hi"}],
                 "max_completion_tokens": 2048, "service_tier": tier})
    F["service_tier"][tier] = why(c, d, 130)
    if c == 200:
        F["service_tier"][tier + "_echo"] = d.get("service_tier")

# --- I. Vision on grok 4.6 (the model card's modality table looks generic) -
PNG_1PX = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8"
           "AAxAAADwABbT8HRAAAAABJRU5ErkJggg==")
c, d = call(RT, "/openai/v1/responses",
            {"model": GROK, "max_output_tokens": 3000,
             "input": [{"role": "user", "content": [
                 {"type": "input_text", "text": "What colour is this image?"},
                 {"type": "input_image",
                  "image_url": f"data:image/png;base64,{PNG_1PX}"}]}]})
F["grok_vision_responses"] = why(c, d, 200)
c, d = call(RT, "/openai/v1/chat/completions",
            {"model": GROK, "max_completion_tokens": 3000,
             "messages": [{"role": "user", "content": [
                 {"type": "text", "text": "What colour is this image?"},
                 {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{PNG_1PX}"}}]}]})
F["grok_vision_chat"] = why(c, d, 200)

# --- J. Context window: what does over-length look like? -----------------
c, d = call(RT, "/openai/v1/chat/completions",
            {"model": GROK, "max_completion_tokens": 2048,
             "messages": [{"role": "user", "content": "word " * 200_000}]},
            timeout=300)
F["grok_over_context"] = why(c, d, 220)

# --- K. gpt-5.6 on runtime: does it behave like grok? --------------------
F["gpt56_runtime"] = {}
c, d = call(RT, "/openai/v1/responses",
            {"model": "us.openai.gpt-5.6-sol", "input": Q,
             "max_output_tokens": 4000, "reasoning": {"effort": "high"}})
F["gpt56_runtime"]["responses effort=high"] = why(c, d, 140)
if c == 200:
    F["gpt56_runtime"]["output_types"] = [i.get("type") for i in (d.get("output") or [])]
    F["gpt56_runtime"]["any_encrypted"] = any(i.get("encrypted_content")
                                              for i in (d.get("output") or []))
    F["gpt56_runtime"]["store_echo"] = d.get("store")
c, d = call(RT, "/openai/v1/chat/completions",
            {"model": "us.openai.gpt-5.6-sol",
             "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64})
F["gpt56_runtime"]["chat max_tokens"] = why(c, d, 140)
c, d = call(RT, "/openai/v1/chat/completions",
            {"model": "us.openai.gpt-5.6-sol",
             "messages": [{"role": "user", "content": "Hi"}],
             "max_completion_tokens": 2048})
F["gpt56_runtime"]["chat max_completion_tokens"] = why(c, d, 140)
c, d = call(RT, "/openai/v1/responses",
            {"model": "global.openai.gpt-5.6-sol", "input": "Hi",
             "max_output_tokens": 2048})
F["gpt56_runtime"]["global profile"] = why(c, d, 140)
c, d = call(RT, "/openai/v1/responses",
            {"model": "openai.gpt-5.6-sol", "input": "Hi",
             "max_output_tokens": 2048})
F["gpt56_runtime"]["bare id"] = why(c, d, 140)

out = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "33-claims-round2.json")
with open(out, "w") as f:
    json.dump(F, f, indent=2)
print(json.dumps(F, indent=2))
print(f"\nwrote {out}")
