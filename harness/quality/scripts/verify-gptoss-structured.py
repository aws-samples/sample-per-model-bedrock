#!/usr/bin/env python3
"""Is strict json_schema usable for gpt-oss on any Bedrock surface?

Two colleagues report trouble: strict json_schema on mantle Responses hangs, and
reasoning tags leak into gpt-oss output on runtime Chat Completions. The
recommended workaround in one thread is Converse `outputConfig`. First pass here
suggested Converse ALSO returns unparseable output for gpt-oss -- a stray "{" and
newline before the real object.

Four surfaces x 3 repeats, with gpt-5.6 and grok-4.6 as controls to separate
"gpt-oss problem" from "Bedrock structured-output problem". Writes
quality/findings/40-gptoss-structured.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
from bedrock import err, ok, post, response_text, runtime_post  # noqa: E402

REGION = "us-east-1"
REPEATS = 3
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
    "required": ["city", "celsius"],
    "additionalProperties": False,
}
ASK = "Jakarta is 31 degrees C. Return JSON."


def verdict(text):
    text = (text or "").strip()
    if not text:
        return "empty", "(no text -- check the token budget, not the schema)"
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        if text.startswith("```"):
            return "fenced", text[:90]
        if text.startswith("{\n{") or text.startswith("{{"):
            return "doubled-brace", text[:90]
        if "<reasoning>" in text:
            return "reasoning-leak", text[:90]
        return "unparseable", text[:90]
    if sorted(parsed) != ["celsius", "city"]:
        return "wrong-keys", text[:90]
    return "conforms", text[:90]


# Budget generously: a reasoning model spends output tokens before any text, and a
# starved call returns an EMPTY string, which a schema check reports as
# "unparseable". That is a budget bug masquerading as a structured-output bug, and
# the first version of this probe fell for it.
BUDGET = 8000


def mantle_prefix(model):
    """gpt-5.x and xai are /openai/v1 on mantle; gpt-oss and the rest are /v1.

    The first version of this probe hardcoded /v1, so gpt-5.6 returned "isn't
    supported on this route" and looked like a structured-output failure.
    """
    return "/openai/v1" if model.startswith(("openai.gpt-5", "xai.", "google.gemma-4")) else "/v1"


def budget_field(model):
    """gpt-5.x refuses max_tokens in favour of max_completion_tokens."""
    bare = model.split(".", 1)[-1] if model.startswith(("us.", "global.")) else model
    return "max_completion_tokens" if "gpt-5" in bare else "max_tokens"


def mantle_responses(model):
    return post(f"{mantle_prefix(model)}/responses", {
        "model": model, "input": ASK, "max_output_tokens": BUDGET,
        "text": {"format": {"type": "json_schema", "name": "w",
                            "schema": SCHEMA, "strict": True}}},
        region=REGION, attempts=1, timeout=240)


def mantle_chat(model):
    return post(f"{mantle_prefix(model)}/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": ASK}],
        budget_field(model): BUDGET,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "w", "schema": SCHEMA, "strict": True}}},
        region=REGION, attempts=1, timeout=240)


def runtime_chat(model):
    return runtime_post("/openai/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": ASK}],
        budget_field(model): BUDGET,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "w", "schema": SCHEMA, "strict": True}}},
        region=REGION, attempts=1, timeout=240)


def runtime_responses(model):
    return runtime_post("/openai/v1/responses", {
        "model": model, "input": ASK, "max_output_tokens": BUDGET,
        "text": {"format": {"type": "json_schema", "name": "w",
                            "schema": SCHEMA, "strict": True}}},
        region=REGION, attempts=1, timeout=240)


def text_of(kind, data):
    """Pull the assistant text for whichever surface produced it.

    Note the .lower(): the surface labels are "runtime Responses" with a capital R,
    and a case-sensitive `"responses" in kind` sent every Responses payload down the
    Chat Completions branch, which found no `choices` and reported "empty". Three
    models looked like they returned nothing. They had all conformed.
    """
    if "responses" in kind.lower():
        return response_text(data)
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")


HTTP_SURFACES = [
    ("mantle Responses", mantle_responses),
    ("mantle ChatCompletions", mantle_chat),
    ("runtime Responses", runtime_responses),
    ("runtime ChatCompletions", runtime_chat),
]

MODELS = {
    "openai.gpt-oss-120b": {"mantle": "openai.gpt-oss-120b",
                            "runtime": "openai.gpt-oss-120b-1:0"},
    "openai.gpt-5.6-sol": {"mantle": "openai.gpt-5.6-sol",
                           "runtime": "us.openai.gpt-5.6-sol"},
    "xai.grok-4.6": {"mantle": None, "runtime": "us.xai.grok-4.6"},
}

F = {"region": REGION, "repeats": REPEATS, "http": {}, "converse": {}}

for model_label, ids in MODELS.items():
    for surface_label, fn in HTTP_SURFACES:
        which = "mantle" if surface_label.startswith("mantle") else "runtime"
        mid = ids[which]
        key = f"{model_label} :: {surface_label}"
        if mid is None:
            F["http"][key] = ["n/a on this endpoint"]
            continue
        outcomes = []
        for _ in range(REPEATS):
            code, data = fn(mid)
            if not ok(code, data):
                outcomes.append(f"{code}: {err(data)[:60]}")
                continue
            state, sample = verdict(text_of(surface_label, data))
            outcomes.append(f"{state}: {sample}")
        F["http"][key] = outcomes
        print(f"{key:52} {outcomes[0][:64]}", flush=True)

# Converse outputConfig. Note the shape: `schema` is a JSON *string*, and the
# jsonSchema sits under a `structure` tagged union. Passing a dict raises
# ParamValidationError, which is easy to mistake for "not supported".
import boto3  # noqa: E402
rt = boto3.client("bedrock-runtime", region_name=REGION)
OUTPUT_CONFIG = {
    "textFormat": {
        "type": "json_schema",
        "structure": {"jsonSchema": {"name": "w", "schema": json.dumps(SCHEMA)}},
    }
}
for model_label, ids in MODELS.items():
    mid = ids["runtime"]
    outcomes = []
    for _ in range(REPEATS):
        try:
            r = rt.converse(
                modelId=mid,
                messages=[{"role": "user", "content": [{"text": ASK}]}],
                inferenceConfig={"maxTokens": 3000},
                outputConfig=OUTPUT_CONFIG,
            )
            text = "".join(b.get("text", "")
                           for b in r["output"]["message"]["content"])
            state, sample = verdict(text)
            outcomes.append(f"{state}: {sample}")
        except Exception as exc:  # noqa: BLE001
            outcomes.append(f"{type(exc).__name__}: {str(exc)[:70]}")
    F["converse"][f"{model_label} :: runtime Converse"] = outcomes
    print(f"{model_label + ' :: runtime Converse':52} {outcomes[0][:64]}", flush=True)

out = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "40-gptoss-structured.json")
with open(out, "w") as f:
    json.dump(F, f, indent=2)

print("\n=== summary ===")
for section in ("http", "converse"):
    for key, outcomes in F[section].items():
        states = [o.split(":")[0] for o in outcomes]
        print(f"{key:52} {states}")
print(f"\nwrote {out}")
