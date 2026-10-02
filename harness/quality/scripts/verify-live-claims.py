#!/usr/bin/env python3
"""Check the notebooks' hardcoded capability rules against the live service.

`99-cross-cutting/01` and `02` carry a resolver whose rules are literal tuples of
model prefixes: which families need `/openai/v1`, which serve Chat Completions only,
which reject sampling parameters, which accept service tiers. Those literals are the
repository's most perishable content. Every one of them was true when written, and
Gemma 4 and Grok have each moved twice since.

The literals are READ OUT OF THE NOTEBOOK rather than restated here, so this checker
cannot drift from the thing it checks. If someone edits the tuple, this checks the new
tuple.

    python3 verify-live-claims.py [--quick]

Exit 1 if any rule disagrees with the service.
"""
from __future__ import annotations

import json
import os
import re
import sys

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO + "/_shared")

from bedrock import api_prefix, err, list_models, post  # noqa: E402

REGION = "us-east-1"
NB = f"{REPO}/99-cross-cutting/01-choosing-a-model-and-api.ipynb"
QUICK = "--quick" in sys.argv

failures: list[str] = []
checks = 0


def record(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  ok    {label}")
    else:
        failures.append(f"{label} — {detail}")
        print(f"  FAIL  {label}  {detail}")


def literal(name: str) -> list[str]:
    """The tuple/list assigned to `name` in the notebook, as a list of strings."""
    text = "\n".join("".join(c["source"]) for c in json.load(open(NB))["cells"])
    m = re.search(rf"^{name}\s*=\s*[\(\[](.*?)[\)\]]", text, re.M | re.S)
    if not m:
        raise SystemExit(f"literal {name} not found in {NB}")
    return [s for s in re.findall(r'["\']([^"\']+)["\']', m.group(1))]


def sample(models: list[str], limit: int) -> list[str]:
    return models[:limit] if QUICK else models


def main() -> None:
    catalogue = sorted(list_models(REGION))
    print(f"mantle catalogue: {len(catalogue)} models in {REGION}\n")

    # ---- 1. path prefix per family -------------------------------------------
    print("=== OPENAI_PREFIX_FAMILIES: which families sit on /openai/v1 ===")
    families = literal("OPENAI_PREFIX_FAMILIES")
    for model in catalogue:
        expected = "/openai/v1" if any(model.startswith(f) for f in families) else None
        actual = api_prefix(model)
        if model.startswith("anthropic."):
            continue  # /anthropic/v1 is a separate family, not part of this rule
        if expected == "/openai/v1":
            record(actual == "/openai/v1",
                   f"{model} -> /openai/v1", f"api_prefix says {actual}")
        else:
            record(actual != "/openai/v1",
                   f"{model} -> not /openai/v1", f"api_prefix says {actual}")

    # ---- 2. Chat-Completions-only families -----------------------------------
    print("\n=== CHAT_ONLY: Responses must 400 for these ===")
    chat_only = literal("CHAT_ONLY")
    targets = [m for m in catalogue if any(m.startswith(p) for p in chat_only)]
    for model in sample(targets, 4):
        prefix = api_prefix(model)
        code, data = post(f"{prefix}/responses",
                          {"model": model, "input": "hi", "max_output_tokens": 16},
                          region=REGION, attempts=1, timeout=60)
        record(code != 200, f"{model}: Responses refused",
               f"got {code}; CHAT_ONLY says it should not serve Responses")

    # ---- 3. sampling-parameter rules ----------------------------------------
    print("\n=== NO_SAMPLING: temperature and top_p must both be refused ===")
    for model in sample(literal("NO_SAMPLING"), 2):
        if model not in catalogue:
            record(True, f"{model}: not in catalogue, rule moot")
            continue
        prefix = api_prefix(model)
        body = {"model": model, "max_tokens": 16, "temperature": 0.5,
                "messages": [{"role": "user", "content": "hi"}]}
        path = f"{prefix}/messages" if prefix == "/anthropic/v1" else \
            f"{prefix}/chat/completions"
        if prefix == "/anthropic/v1":
            body["anthropic_version"] = "bedrock-2023-05-31"
        code, data = post(path, body, region=REGION, attempts=1, timeout=60)
        record(code != 200, f"{model}: temperature refused",
               f"got {code}, so NO_SAMPLING is stale")

    print("\n=== TEMPERATURE_DEFAULT_ONLY: temperature=1.0 ok, 0.2 refused ===")
    # This restriction is per (model, API), not per model. gpt-5.5 refuses a
    # non-default temperature on Responses and accepts one on Chat Completions.
    # Probing Chat Completions alone therefore reports the rule as stale, which is
    # exactly the wrong conclusion: the resolver routes gpt-5.5 to Responses. So test
    # on the surface the resolver actually uses for each model.
    chat_only = literal("CHAT_ONLY")
    for family in sample(literal("TEMPERATURE_DEFAULT_ONLY"), 3):
        model = next((m for m in catalogue if m.startswith(family)), None)
        if not model:
            record(True, f"{family}: no model in catalogue, rule moot")
            continue
        prefix = api_prefix(model)
        routed_to_chat = any(model.startswith(p) for p in chat_only)
        if routed_to_chat:
            budget = ("max_completion_tokens"
                      if any(model.startswith(f)
                             for f in literal("COMPLETION_TOKENS_FAMILIES"))
                      else "max_tokens")
            path = f"{prefix}/chat/completions"
            base = {"model": model, budget: 16,
                    "messages": [{"role": "user", "content": "hi"}]}
        else:
            path = f"{prefix}/responses"
            base = {"model": model, "input": "hi", "max_output_tokens": 16}
        surface = "chat" if routed_to_chat else "responses"
        ok_code, _ = post(path, {**base, "temperature": 1.0},
                          region=REGION, attempts=1, timeout=60)
        bad_code, _ = post(path, {**base, "temperature": 0.2},
                           region=REGION, attempts=1, timeout=60)
        record(ok_code == 200, f"{model} on {surface}: temperature=1.0 accepted",
               f"got {ok_code}")
        record(bad_code != 200, f"{model} on {surface}: temperature=0.2 refused",
               f"got {bad_code} on the routed surface, so the rule is stale")

    print("\n=== COMPLETION_TOKENS_FAMILIES: max_tokens must be refused ===")
    for family in sample(literal("COMPLETION_TOKENS_FAMILIES"), 2):
        model = next((m for m in catalogue if m.startswith(family)), None)
        if not model:
            record(True, f"{family}: no model in catalogue, rule moot")
            continue
        prefix = api_prefix(model)
        code, data = post(f"{prefix}/chat/completions",
                          {"model": model, "max_tokens": 16,
                           "messages": [{"role": "user", "content": "hi"}]},
                          region=REGION, attempts=1, timeout=60)
        record(code != 200, f"{model}: max_tokens refused",
               f"got {code}, so it does not need max_completion_tokens")

    # ---- 4. runtime Responses surface ---------------------------------------
    print("\n=== RUNTIME_RESPONSES against the recorded runtime matrix ===")
    matrix = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "31-runtime-matrix.json")
    if os.path.exists(matrix):
        rows = json.load(open(matrix))["rows"]
        served = {re.sub(r"^(us|global)\.", "", r["model"]).split(":")[0]
                  for r in rows if r.get("responses") == "ok"}
        claimed = literal("RUNTIME_RESPONSES")
        unclaimed = [m for m in served if not any(m.startswith(c) for c in claimed)]
        unused = [c for c in claimed if not any(m.startswith(c) for m in served)]
        record(not unclaimed, "every runtime-Responses model matches a claimed prefix",
               f"unclaimed: {unclaimed}")
        record(not unused, "every claimed prefix matches a served model",
               f"claimed but unserved: {unused}")
    else:
        record(False, "runtime matrix present", f"missing {matrix}")

    # RUNTIME_MESSAGES_CLAUDE is a LIST, not a rule, because no ID shape predicts
    # which Claude models serve /anthropic/v1/messages on bedrock-runtime. A list goes
    # stale, so it is checked in BOTH directions: every claimed model must answer, and
    # every Claude that answers must be claimed. The previous rule -- inferred from a
    # single counter-example -- was false in both directions within weeks.
    print("\nRUNTIME_MESSAGES_CLAUDE, both directions")
    try:
        from bedrock import inference_profiles, runtime_models, runtime_post

        claimed = literal("RUNTIME_MESSAGES_CLAUDE")
        av = {"anthropic-version": "2023-06-01"}
        ids = set()
        for entry in runtime_models(REGION).values():
            if "anthropic." in entry.get("id", ""):
                ids.add(entry["id"])
        try:
            ids |= {p for p in (inference_profiles(REGION) or []) if "anthropic." in p}
        except Exception:  # noqa: BLE001
            pass
        geo = sorted(i for i in ids if i.startswith("us."))
        if QUICK:
            geo = geo[:6]

        answers, refuses = [], []
        for mid in geo:
            code, _ = runtime_post(
                "/anthropic/v1/messages",
                {"model": mid, "max_tokens": 16,
                 "messages": [{"role": "user", "content": "Hi"}]},
                region=REGION, headers=av, attempts=1, timeout=60)
            (answers if code == 200 else refuses).append(mid)

        def is_claimed(mid: str) -> bool:
            bare = re.sub(r"^(us|eu|apac|in|global)\.", "", mid)
            return any(bare.startswith(c) for c in claimed)

        missing = [m for m in answers if not is_claimed(m)]
        stale = [m for m in refuses if is_claimed(m)]
        record(not missing,
               "every Claude that answers runtime Messages is in the tuple",
               f"answers but unclaimed: {missing}")
        record(not stale,
               "every model the tuple claims really answers",
               f"claimed but refused: {stale}")
        print(f"        {len(answers)} answer, {len(refuses)} refuse, "
              f"of {len(geo)} us.* Claude ID(s)")
    except Exception as exc:  # noqa: BLE001
        record(False, "RUNTIME_MESSAGES_CLAUDE probed",
               f"{type(exc).__name__}: {exc}")

    print(f"\n{checks} checks, {len(failures)} failed")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
