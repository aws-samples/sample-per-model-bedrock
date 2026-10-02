#!/usr/bin/env python3
"""Prove the pre-launch checklist in `99-cross-cutting/03` can fail.

The checklist was wrong twice. Version one summed hardcoded `True` literals and
printed "17/17 controls implemented". Version two replaced them with expressions that
looked derived but were structurally true for any code: `529 in TRANSIENT` over a
literal set, `bool(SERVER_FAULT_TEXT)` over a literal tuple, `hasattr(tokens, "get")`.
Both versions printed a full score beside a client whose retry policy nothing had
checked.

This runs the cell's real evidence expressions against stubbed run state, once
healthy and once per sabotage, and requires the score to drop each time. A checklist
that cannot fail is not a checklist.

    python3 test-checklist-derivation.py
"""
import json
import os
import sys
from datetime import timedelta

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "_shared"))
from bedrock import parse_json_lenient  # noqa: E402

NB = os.path.join(REPO, "99-cross-cutting", "03-production-hardening.ipynb")
TRANSIENT = {429, 500, 502, 503, 504, 529}
SERVER_FAULT_TEXT = ("internal server error", "internal failure", "internal error")


class TokenStub:
    ttl = timedelta(minutes=15)
    mints = 1


class ExpiredTokenStub:
    ttl = timedelta(hours=12)
    mints = 1


def sources() -> tuple[str, str, str]:
    nb = json.load(open(NB))
    cell41 = cell22 = cell25 = None
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "EVIDENCE = {" in src:
            cell41 = src
        elif "class CallMetrics" in src:
            cell22 = src[src.index("class CallMetrics"):src.index("metrics = CallMetrics()")]
        elif src.startswith("SCHEMA = {"):
            cell25 = src[:src.index("def safe_structured")]
    missing = [n for n, v in (("EVIDENCE cell", cell41), ("CallMetrics", cell22),
                              ("SCHEMA", cell25)) if v is None]
    if missing:
        raise SystemExit(f"could not locate in the notebook: {', '.join(missing)}")
    return cell41, cell22, cell25


def env_for(sabotage: str | None, evidence_src: str, metrics_src: str,
            schema_src: str) -> dict:
    # `transient` is a per-sabotage copy so "drops-529" can remove a status from the
    # set WITHOUT the checklist's expectations moving with it. That is the whole
    # point: round two derived the expected answers from TRANSIENT, so deleting 529
    # deleted the check too and the score stayed 6/6.
    transient = set(TRANSIENT)
    if sabotage == "drops-529":
        transient.discard(529)

    def should_retry(code: int, parsed: dict) -> bool:
        if sabotage == "retries-everything":
            return True
        if sabotage == "retries-nothing":
            return False
        if code in transient:
            return True
        if 400 <= code < 500:
            return any(m in json.dumps(parsed).lower() for m in SERVER_FAULT_TEXT)
        return False

    log = [{"path": "/x", "timeout": 60, "statuses": [200], "sleeps": 0},
           {"path": "/x", "timeout": 60, "statuses": [400], "sleeps": 0}]
    if sabotage == "retries-everything":
        log[1] = {"path": "/x", "timeout": 60, "statuses": [400, 400, 400],
                  "sleeps": 2}
    if sabotage == "no-timeout":
        log[0]["timeout"] = None
    if sabotage == "no-calls":
        log = []

    # The timeout tick now proves the deadline reaches the socket by asking for one
    # that cannot be met, so the stub has to answer that call. "no-socket-timeout"
    # models a client that records a timeout and never applies it.
    def resilient_call(path, body, *, attempts=5, timeout=60, **kw):
        if timeout < 1 and sabotage != "no-socket-timeout":
            return -1, {"error": {"message": "timeout"}}
        return 200, {"output": []}

    clock = iter([0.0, 0.05] + [0.0] * 50)

    class _Time:
        @staticmethod
        def perf_counter():
            return next(clock, 0.0)

    env = {
        "json": json, "timedelta": timedelta, "should_retry": should_retry,
        "TRANSIENT": transient, "SERVER_FAULT_TEXT": SERVER_FAULT_TEXT,
        "CALL_LOG": log,
        "tokens": ExpiredTokenStub() if sabotage == "long-ttl" else TokenStub(),
        "parse_json_lenient": parse_json_lenient,
        "resilient_call": resilient_call, "time": _Time,
        "PREFIX": "/openai/v1", "MODEL": "google.gemma-4-31b",
        "print": lambda *a, **k: None,
    }
    exec(metrics_src, env)  # noqa: S102 - the notebook's own class, read from disk
    env["metrics"] = env["CallMetrics"]()
    exec(schema_src, env)  # noqa: S102
    exec(evidence_src, env)  # noqa: S102
    return env


def main() -> None:
    evidence_src, metrics_src, schema_src = sources()
    healthy = env_for(None, evidence_src, metrics_src, schema_src)["EVIDENCE"]
    total = len(healthy)
    score = sum(bool(v) for v in healthy.values())
    print(f"  healthy run: {score}/{total}")
    if score != total:
        failing = [k for k, v in healthy.items() if not v]
        print(f"  FAIL a healthy run should tick everything; failing: {failing}")
        sys.exit(1)

    bad = []
    for sabotage in ("retries-everything", "retries-nothing", "no-timeout",
                     "no-calls", "long-ttl", "drops-529", "no-socket-timeout"):
        ev = env_for(sabotage, evidence_src, metrics_src, schema_src)["EVIDENCE"]
        n = sum(bool(v) for v in ev.values())
        dropped = [k.split(" ")[0] for k, v in ev.items() if not v]
        print(f"  {sabotage:<20} -> {n}/{total}  dropped: {dropped}")
        if n >= total:
            bad.append(sabotage)

    print()
    if bad:
        print("FAIL these sabotages did not move the score, so the tick is inert:")
        for s in bad:
            print(f"  - {s}")
        sys.exit(1)
    print("checklist derivation has teeth: every sabotage drops the score")


if __name__ == "__main__":
    main()
