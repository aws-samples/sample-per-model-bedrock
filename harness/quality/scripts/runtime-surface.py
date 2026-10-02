#!/usr/bin/env python3
"""Map which API surfaces bedrock-runtime serves, with which auth, for which models.

The user's premise for this round is that AWS now prefers bedrock-runtime and that
Responses / Chat Completions / Messages all work there. That is a claim about the
service, so it gets probed rather than believed. Writes
quality/findings/30-runtime-surface.json.

    python3 runtime-surface.py [--region us-east-1]
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
MANTLE = f"https://bedrock-mantle.{REGION}.api.aws"
TOKEN = token(REGION)


def sigv4_headers(method, url, body, service="bedrock"):
    """Sign a raw HTTP request with SigV4 using botocore, so we can compare auth."""
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    sess = boto3.Session()
    creds = sess.get_credentials().get_frozen_credentials()
    req = AWSRequest(method=method, url=url, data=body,
                     headers={"Content-Type": "application/json"})
    SigV4Auth(creds, service, REGION).add_auth(req)
    return dict(req.headers)


def call(host, path, body, auth="bearer", method="POST", extra=None):
    url = host + path
    raw = json.dumps(body).encode() if body is not None else None
    if auth == "bearer":
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {TOKEN}"}
    else:
        headers = sigv4_headers(method, url, raw or b"")
    if extra:
        headers.update(extra)
    req = urllib.request.Request(url, data=raw, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw_body = e.read()
        try:
            return e.code, json.loads(raw_body or b"{}")
        except Exception:
            return e.code, {"raw": raw_body[:300].decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        return None, {"exception": f"{type(e).__name__}: {e}"[:200]}


def brief(code, data, limit=170):
    if code == 200:
        return "200"
    msg = ""
    for k in ("message", "Message", "error", "detail", "raw", "exception"):
        v = data.get(k)
        if isinstance(v, dict):
            v = v.get("message") or json.dumps(v)
        if v:
            msg = str(v)
            break
    if not msg:
        msg = json.dumps(data)
    return f"{code}: {msg[:limit]}"


# ---------------------------------------------------------------------------
# 1. Which paths exist on runtime at all, and with which auth.
# ---------------------------------------------------------------------------
CHAT = {"model": "openai.gpt-oss-20b",
        "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 16}
RESP = {"model": "openai.gpt-5.5", "input": "Say OK.", "max_output_tokens": 16}
MSG = {"model": "us.anthropic.claude-haiku-4-5", "max_tokens": 16,
       "messages": [{"role": "user", "content": "Say OK."}],
       "anthropic_version": "bedrock-2023-05-31"}
ANTH_HDR = {"anthropic-version": "2023-06-01"}

PATHS = [
    ("runtime", "/openai/v1/chat/completions", CHAT, None),
    ("runtime", "/openai/v1/responses", RESP, None),
    ("runtime", "/anthropic/v1/messages", MSG, ANTH_HDR),
    ("runtime", "/v1/chat/completions", CHAT, None),
    ("runtime", "/v1/messages", MSG, ANTH_HDR),
    ("runtime", "/openai/v1/completions",
     {"model": "openai.gpt-oss-20b", "prompt": "Say OK.", "max_tokens": 16}, None),
    ("runtime", "/openai/v1/embeddings",
     {"model": "amazon.titan-embed-text-v2:0", "input": "hi"}, None),
    ("mantle", "/openai/v1/chat/completions", CHAT, None),
    ("mantle", "/openai/v1/responses", RESP, None),
    ("mantle", "/anthropic/v1/messages",
     {**MSG, "model": "anthropic.claude-haiku-4-5"}, ANTH_HDR),
]

GETS = [
    ("runtime", "/openai/v1/models"),
    ("runtime", "/v1/models"),
    ("mantle", "/v1/models"),
    ("mantle", "/openai/v1/models"),
]

HOSTS = {"runtime": RUNTIME, "mantle": MANTLE}


def probe_path(item):
    hostname, path, body, extra = item
    out = {"host": hostname, "path": path, "model": body.get("model")}
    for auth in ("bearer", "sigv4"):
        c, d = call(HOSTS[hostname], path, body, auth=auth, extra=extra)
        out[auth] = brief(c, d)
        if c == 200:
            out[auth + "_keys"] = sorted(d)[:12]
    return out


def probe_get(item):
    hostname, path = item
    out = {"host": hostname, "path": path, "method": "GET"}
    for auth in ("bearer", "sigv4"):
        c, d = call(HOSTS[hostname], path, None, auth=auth, method="GET")
        if c == 200:
            n = len(d.get("data", d) if isinstance(d, dict) else d)
            out[auth] = f"200 ({n} entries)"
        else:
            out[auth] = brief(c, d)
    return out


# ---------------------------------------------------------------------------
# 2. Model addressing on the runtime OpenAI surface: bare vs us. vs global.
# ---------------------------------------------------------------------------
ADDRESSING = [
    "xai.grok-4.6", "us.xai.grok-4.6", "global.xai.grok-4.6",
    "xai.grok-4.3",
    "openai.gpt-oss-20b", "us.openai.gpt-oss-20b",
    "openai.gpt-5.5", "us.openai.gpt-5.5",
    "anthropic.claude-sonnet-5", "us.anthropic.claude-sonnet-5",
    "google.gemma-4-31b", "us.google.gemma-4-31b",
    "amazon.nova-2-lite-v1:0", "us.amazon.nova-2-lite-v1:0",
    "qwen.qwen3-32b", "deepseek.v3.2", "zai.glm-5", "moonshotai.kimi-k2.5",
    "meta.llama4-maverick-17b-instruct-v1:0",
    "us.meta.llama4-maverick-17b-instruct-v1:0",
]


def probe_addressing(mid):
    out = {"model": mid}
    for surface in ("chat", "responses"):
        if surface == "chat":
            path, body = "/openai/v1/chat/completions", {
                "model": mid, "messages": [{"role": "user", "content": "Say OK."}],
                "max_completion_tokens": 32}
        else:
            path, body = "/openai/v1/responses", {
                "model": mid, "input": "Say OK.", "max_output_tokens": 32}
        c, d = call(RUNTIME, path, body)
        out["runtime_" + surface] = brief(c, d, 120)
    return out


def main():
    findings = {"region": REGION}
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        findings["paths"] = list(ex.map(probe_path, PATHS))
        findings["catalogues"] = list(ex.map(probe_get, GETS))
        findings["addressing"] = list(ex.map(probe_addressing, ADDRESSING))

    out = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "30-runtime-surface.json")
    with open(out, "w") as f:
        json.dump(findings, f, indent=2)

    print("=== PATHS (POST) ===")
    for r in findings["paths"]:
        print(f"{r['host']:8} {r['path']:32} bearer={r['bearer'][:70]}")
        print(f"{'':8} {'':32} sigv4 ={r['sigv4'][:70]}")
    print("\n=== CATALOGUES (GET) ===")
    for r in findings["catalogues"]:
        print(f"{r['host']:8} {r['path']:22} bearer={r['bearer'][:50]:52} "
              f"sigv4={r['sigv4'][:50]}")
    print("\n=== ADDRESSING on runtime ===")
    for r in findings["addressing"]:
        print(f"{r['model']:46} chat={r['runtime_chat'][:44]:46} "
              f"resp={r['runtime_responses'][:44]}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
