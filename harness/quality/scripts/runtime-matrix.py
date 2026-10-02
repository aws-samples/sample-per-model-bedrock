#!/usr/bin/env python3
"""The definitive bedrock-runtime capability matrix: model x API surface.

Answers the questions this round turns on:
  - which of Chat Completions / Responses / Messages does runtime serve, per model
  - what model ID does runtime want (they differ from bedrock-mantle's)
  - which need an inference profile
  - does Converse still work for all of them

Writes quality/findings/31-runtime-matrix.json.

    python3 runtime-matrix.py [--region us-east-1]

Note on the 200 trap: bedrock-runtime answers an *unrecognised path* with
HTTP 200 and a Coral UnknownOperationException in the body. `ok()` below treats
that as a failure, because it is one.
"""
import concurrent.futures as cf
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
from bedrock import token  # noqa: E402

REGION = "us-east-1"
if "--region" in sys.argv:
    REGION = sys.argv[sys.argv.index("--region") + 1]

RUNTIME = f"https://bedrock-runtime.{REGION}.amazonaws.com"
TOKEN = token(REGION)
ANTH_HDR = {"anthropic-version": "2023-06-01"}


def call(path, body, extra=None, timeout=120):
    raw = json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
    if extra:
        h.update(extra)
    req = urllib.request.Request(RUNTIME + path, data=raw, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body_raw = e.read()
        try:
            return e.code, json.loads(body_raw or b"{}")
        except Exception:
            return e.code, {"raw": body_raw[:300].decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        return None, {"exception": f"{type(e).__name__}: {e}"[:160]}


def unknown_op(data):
    """True when the body is a Coral UnknownOperationException, whatever the code."""
    return "UnknownOperation" in json.dumps(data)[:400]


def ok(code, data):
    return code == 200 and not unknown_op(data)


def why(code, data, limit=110):
    if unknown_op(data):
        return f"{code}: UnknownOperationException (path not served)"
    if ok(code, data):
        return "ok"
    e = data.get("error") or data
    msg = e.get("message") if isinstance(e, dict) else None
    msg = msg or data.get("message") or data.get("Message") or json.dumps(data)
    return f"{code}: {str(msg)[:limit]}"


PROMPT = "Reply with exactly: OK"


def probe(mid):
    """Probe one runtime model ID across the three HTTP surfaces plus Converse."""
    out = {"model": mid}

    # Generous budget: reasoning models spend the whole allowance on reasoning
    # and return content=null, which looks like a failure but is a budget bug.
    c, d = call("/openai/v1/chat/completions",
                {"model": mid, "messages": [{"role": "user", "content": PROMPT}],
                 "max_completion_tokens": 2048})
    out["chat"] = why(c, d)
    if ok(c, d):
        m = (d.get("choices") or [{}])[0].get("message") or {}
        out["chat_text_len"] = len(m.get("content") or "")
        out["chat_reasoning_field"] = "reasoning" in m
        out["chat_reasoning_tokens"] = ((d.get("usage") or {})
                                        .get("completion_tokens_details") or {}
                                        ).get("reasoning_tokens")
        out["chat_echo_model"] = d.get("model")

    c, d = call("/openai/v1/responses",
                {"model": mid, "input": PROMPT, "max_output_tokens": 2048})
    out["responses"] = why(c, d)
    if ok(c, d):
        types = [i.get("type") for i in (d.get("output") or [])]
        out["responses_output_types"] = types
        out["responses_encrypted"] = any(
            "encrypted_content" in i for i in (d.get("output") or []))
        out["responses_store_field"] = d.get("store")

    c, d = call("/anthropic/v1/messages",
                {"model": mid, "max_tokens": 512,
                 "messages": [{"role": "user", "content": PROMPT}]}, ANTH_HDR)
    out["messages"] = why(c, d)
    if ok(c, d):
        out["messages_echo_model"] = d.get("model")

    # Converse, through boto3 so the SigV4/SDK path is covered too.
    try:
        import boto3
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        r = rt.converse(modelId=mid,
                        messages=[{"role": "user",
                                   "content": [{"text": PROMPT}]}],
                        inferenceConfig={"maxTokens": 512})
        out["converse"] = "ok"
        out["converse_stop"] = r.get("stopReason")
    except Exception as e:  # noqa: BLE001
        out["converse"] = f"{type(e).__name__}: {str(e)[:110]}"

    return out


def main():
    import boto3
    b = boto3.client("bedrock", region_name=REGION)
    fms = b.list_foundation_models()["modelSummaries"]
    profiles = {p["inferenceProfileId"]
                for p in b.list_inference_profiles(maxResults=1000)[
                    "inferenceProfileSummaries"]}

    # Text-out models only; skip image/video/embedding/speech and the
    # PROVISIONED-only context-length variants (":24k", ":300k", ...).
    targets = []
    for m in fms:
        if "TEXT" not in m.get("outputModalities", []):
            continue
        types = m.get("inferenceTypesSupported", [])
        if types == ["PROVISIONED"] or not types:
            continue
        mid = m["modelId"]
        if "ON_DEMAND" in types:
            targets.append(mid)
        else:  # INFERENCE_PROFILE only -- probe the us. form, which is what works
            targets.append(f"us.{mid}" if f"us.{mid}" in profiles else mid)

    targets = sorted(set(targets))
    print(f"probing {len(targets)} runtime text models x 4 surfaces\n")

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(probe, targets))

    out_path = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "31-runtime-matrix.json")
    with open(out_path, "w") as f:
        json.dump({"region": REGION, "rows": rows}, f, indent=2)

    hdr = f"{'model':44} {'chat':10} {'responses':10} {'messages':10} converse"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def s(k):
            return "OK" if r[k] == "ok" else r[k].split(":")[0]
        print(f"{r['model']:44} {s('chat'):10} {s('responses'):10} "
              f"{s('messages'):10} {'OK' if r['converse']=='ok' else r['converse'].split(':')[0]}")

    print("\n=== distinct failure reasons ===")
    seen = {}
    for r in rows:
        for k in ("chat", "responses", "messages", "converse"):
            if r[k] != "ok":
                seen.setdefault(r[k][:95], []).append(f"{r['model']}.{k}")
    for msg, who in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        print(f"[{len(who):3}] {msg}")
        print(f"      e.g. {', '.join(who[:3])}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
