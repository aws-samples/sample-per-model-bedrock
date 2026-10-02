#!/usr/bin/env python3
"""Probe every bedrock-mantle model for the capabilities the notebooks assert.

Writes JSON to quality/findings/15-capability-matrix.json. The point is to have a
ground truth to check every table row against, rather than re-reading prose and
trusting it. Claims about parameter support are the highest-churn class in this
collection -- Gemma 4's surface has now changed twice -- so they get probed, not
remembered.

    python3 capability-matrix.py [--region us-east-1] [--models a,b,c]
"""
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
from bedrock import err, list_models, post  # noqa: E402

REGION = "us-east-1"
if "--region" in sys.argv:
    REGION = sys.argv[sys.argv.index("--region") + 1]

ANTHROPIC_HDR = {"anthropic-version": "2023-06-01"}


def prefix_of(mid: str) -> str:
    if mid.startswith("anthropic."):
        return "/anthropic/v1"
    if mid.startswith(("google.gemma-4", "openai.gpt-5", "xai.")):
        return "/openai/v1"
    return "/v1"


# A schema that is valid under strict mode: `required` must list every property
# and `additionalProperties` must be false. Getting this wrong produces a 400
# that looks like "the model has no tool support".
STRICT_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def probe(mid: str) -> dict:
    pre = prefix_of(mid)
    out = {"model": mid, "prefix": pre, "region": REGION}

    def call(path, body, headers=None, budget_note=None):
        code, data = post(path, body, region=REGION, attempts=1, timeout=90,
                          headers=headers)
        return code, data

    # --- which API surfaces answer at all -------------------------------------
    surfaces = {}
    if not mid.startswith("anthropic."):
        c, d = call(f"{pre}/responses",
                    {"model": mid, "input": "Hi", "max_output_tokens": 16})
        surfaces["responses"] = c
        if c != 200:
            out["responses_err"] = err(d)[:150]
        # Chat Completions: try both budget parameter names before concluding
        # the API is absent. Blaming the path for a parameter rejection is the
        # defect this whole matrix exists to prevent.
        c1, d1 = call(f"{pre}/chat/completions",
                      {"model": mid, "messages": [{"role": "user", "content": "Hi"}],
                       "max_tokens": 16})
        if c1 == 200:
            surfaces["chat"] = 200
            out["chat_budget_param"] = "max_tokens"
        else:
            c2, d2 = call(f"{pre}/chat/completions",
                          {"model": mid,
                           "messages": [{"role": "user", "content": "Hi"}],
                           "max_completion_tokens": 16})
            surfaces["chat"] = c2
            out["chat_budget_param"] = "max_completion_tokens" if c2 == 200 else None
            out["chat_err"] = err(d2 if c2 != 200 else d1)[:150]
    else:
        c, d = call(f"{pre}/messages",
                    {"model": mid, "max_tokens": 16,
                     "messages": [{"role": "user", "content": "Hi"}]},
                    headers=ANTHROPIC_HDR)
        surfaces["messages"] = c
        if c != 200:
            out["messages_err"] = err(d)[:150]
    out["surfaces"] = surfaces

    # Pick the surface we will use for the remaining probes.
    if surfaces.get("responses") == 200:
        use, path = "responses", f"{pre}/responses"
    elif surfaces.get("chat") == 200:
        use, path = "chat", f"{pre}/chat/completions"
    elif surfaces.get("messages") == 200:
        use, path = "messages", f"{pre}/messages"
    else:
        out["usable"] = False
        return out
    out["usable"] = True
    out["probed_on"] = use

    def base(budget=32):
        if use == "responses":
            return {"model": mid, "input": "Hi", "max_output_tokens": budget}
        if use == "chat":
            key = out.get("chat_budget_param") or "max_tokens"
            return {"model": mid, "messages": [{"role": "user", "content": "Hi"}],
                    key: budget}
        return {"model": mid, "max_tokens": budget,
                "messages": [{"role": "user", "content": "Hi"}]}

    hdr = ANTHROPIC_HDR if use == "messages" else None

    # --- sampling parameters --------------------------------------------------
    params = {}
    for name, extra in (
        ("temperature_0.7", {"temperature": 0.7}),
        ("temperature_1.0", {"temperature": 1.0}),
        ("top_p_0.95", {"top_p": 0.95}),
        ("both", {"temperature": 0.7, "top_p": 0.95}),
        ("tier_flex", {"service_tier": "flex"}),
        ("tier_priority", {"service_tier": "priority"}),
    ):
        c, d = call(path, {**base(), **extra}, headers=hdr)
        params[name] = c if c == 200 else f"{c}: {err(d)[:90]}"
    out["params"] = params

    # --- stateful conversation (Responses only) ------------------------------
    if use == "responses":
        c, first = call(path, {"model": mid, "input": "Remember the number 42.",
                               "max_output_tokens": 64, "store": True})
        if c == 200 and first.get("id"):
            c2, second = call(path, {"model": mid,
                                     "input": "What number did I say?",
                                     "previous_response_id": first["id"],
                                     "max_output_tokens": 64})
            out["previous_response_id"] = c2 if c2 == 200 else f"{c2}: {err(second)[:90]}"
        else:
            out["previous_response_id"] = f"store failed: {c}"

    # --- tools ---------------------------------------------------------------
    if use == "responses":
        tools = [{"type": "function", "name": "get_weather",
                  "description": "Weather for a city", "parameters": STRICT_SCHEMA,
                  "strict": True}]
        body = {"model": mid, "input": "Weather in Paris?",
                "max_output_tokens": 256, "tools": tools}
    elif use == "chat":
        key = out.get("chat_budget_param") or "max_tokens"
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Weather for a city",
            "parameters": STRICT_SCHEMA}}]
        body = {"model": mid, "messages": [{"role": "user",
                "content": "Weather in Paris?"}], key: 256, "tools": tools}
    else:
        tools = [{"name": "get_weather", "description": "Weather for a city",
                  "input_schema": STRICT_SCHEMA}]
        body = {"model": mid, "max_tokens": 256, "tools": tools,
                "messages": [{"role": "user", "content": "Weather in Paris?"}]}
    c, d = call(path, body, headers=hdr)
    out["tools"] = c if c == 200 else f"{c}: {err(d)[:110]}"

    # --- structured output ---------------------------------------------------
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    if use == "responses":
        body = {**base(128), "text": {"format": {"type": "json_schema", "name": "a",
                                                 "schema": schema, "strict": True}}}
    elif use == "chat":
        body = {**base(128), "response_format": {"type": "json_schema", "json_schema": {
            "name": "a", "schema": schema, "strict": True}}}
    else:
        body = None
    if body:
        c, d = call(path, body, headers=hdr)
        out["structured_output"] = c if c == 200 else f"{c}: {err(d)[:110]}"

    # --- reasoning knob ------------------------------------------------------
    if use == "responses":
        c, d = call(path, {**base(128), "reasoning": {"effort": "low"}})
    elif use == "chat":
        c, d = call(path, {**base(128), "reasoning_effort": "low"})
    else:
        c, d = call(path, {**base(2048),
                           "thinking": {"type": "enabled", "budget_tokens": 1024}},
                    headers=hdr)
    out["reasoning_knob"] = c if c == 200 else f"{c}: {err(d)[:110]}"
    return out


def main() -> None:
    if "--models" in sys.argv:
        ids = sys.argv[sys.argv.index("--models") + 1].split(",")
    else:
        raw = list_models(REGION)
        ids = sorted(m["id"] if isinstance(m, dict) else m for m in raw)
    print(f"probing {len(ids)} models in {REGION}", file=sys.stderr)
    results = []
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(probe, ids):
            results.append(r)
            print(f"  done {r['model']}", file=sys.stderr)
    dest = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "15-capability-matrix.json")
    with open(dest, "w") as fh:
        json.dump({"region": REGION, "models": results}, fh, indent=1)
    print(f"wrote {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
