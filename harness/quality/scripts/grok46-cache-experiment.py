#!/usr/bin/env python3
"""Does prompt_cache_key or the us. profile improve Grok 4.6's cache hit rate?

A colleague reported 77/100 implicit-cache hits on `global.xai.grok-4.6` via the
Responses API, with the first seven calls all missing, against ~100% on xAI direct.
Their repro does not set `prompt_cache_key`, and uses the `global.` profile, which
can be served from any commercial Region.

Two hypotheses worth testing before suggesting either to anyone:
  H1  an explicit prompt_cache_key improves affinity
  H2  the `us.` geo profile beats `global.` because it routes across 3 Regions
      instead of all of them

Four arms, same shared prefix, same call count. Writes
quality/findings/39-grok46-cache-experiment.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
from bedrock import err, ok, runtime_post  # noqa: E402

REGION = "us-east-1"
N = int(os.environ.get("N", "30"))
PREFIX = (
    "You are a meticulous reviewer of aviation maintenance logs. Cite the "
    "regulation that applies and never speculate beyond the record. " * 260
)

ARMS = [
    ("global, no cache key", "global.xai.grok-4.6", None),
    ("global, cache key", "global.xai.grok-4.6", "grok46-cache-experiment"),
    ("us, no cache key", "us.xai.grok-4.6", None),
    ("us, cache key", "us.xai.grok-4.6", "grok46-cache-experiment"),
]


def run_arm(label, model, cache_key):
    rows = []
    for i in range(N):
        body = {
            "model": model,
            "instructions": PREFIX,
            "input": f"Reply with the number {i}.",
            "max_output_tokens": 2000,
        }
        if cache_key:
            body["prompt_cache_key"] = cache_key
        started = time.perf_counter()
        code, data = runtime_post("/openai/v1/responses", body, region=REGION,
                                  attempts=3, timeout=300)
        if not ok(code, data):
            rows.append({"i": i, "error": f"{code}: {err(data)[:70]}"})
            continue
        usage = data.get("usage") or {}
        detail = usage.get("input_tokens_details") or {}
        rows.append({
            "i": i,
            "input": usage.get("input_tokens"),
            "cached": detail.get("cached_tokens", 0),
            "secs": round(time.perf_counter() - started, 2),
        })
    hits = [r for r in rows if r.get("cached")]
    first_hit = hits[0]["i"] if hits else None
    ok_rows = [r for r in rows if "error" in r or r.get("input")]
    return {
        "label": label,
        "model": model,
        "cache_key": cache_key,
        "calls": len(rows),
        "errors": sum(1 for r in rows if "error" in r),
        "hits": len(hits),
        "hit_rate": round(100 * len(hits) / max(1, len(ok_rows)), 1),
        "first_hit_at_call": first_hit,
        "misses_after_first_hit": (
            sum(1 for r in rows if first_hit is not None
                and r["i"] > first_hit and not r.get("cached") and "error" not in r)
            if first_hit is not None else None
        ),
        "rows": rows,
    }


def main():
    results = []
    for label, model, key in ARMS:
        print(f"running: {label} ({N} calls)...", flush=True)
        results.append(run_arm(label, model, key))
        r = results[-1]
        print(f"  hits {r['hits']}/{r['calls']}  ({r['hit_rate']}%)  "
              f"first hit at call {r['first_hit_at_call']}  errors {r['errors']}",
              flush=True)

    out = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "39-grok46-cache-experiment.json")
    with open(out, "w") as f:
        json.dump({"region": REGION, "calls_per_arm": N, "arms": results}, f, indent=2)

    print()
    print(f"{'arm':24} {'hits':>10} {'rate':>7} {'1st hit':>8} {'misses after':>13}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:24} {r['hits']:>4}/{r['calls']:<5} {r['hit_rate']:>6}% "
              f"{str(r['first_hit_at_call']):>8} {str(r['misses_after_first_hit']):>13}")

    by_key = {r["label"]: r["hit_rate"] for r in results}
    print()
    gk = by_key.get("global, cache key", 0) - by_key.get("global, no cache key", 0)
    uk = by_key.get("us, cache key", 0) - by_key.get("us, no cache key", 0)
    gu = (by_key.get("us, no cache key", 0) - by_key.get("global, no cache key", 0))
    print(f"H1 prompt_cache_key effect: global {gk:+.1f} pts, us {uk:+.1f} pts")
    print(f"H2 us. vs global. effect (no key): {gu:+.1f} pts")
    print()
    if max(gk, uk) >= 15:
        print("=> prompt_cache_key materially improves the hit rate. Worth reporting.")
    elif gu >= 15:
        print("=> the us. geo profile materially beats global. Worth reporting.")
    else:
        print("=> neither lever moved the hit rate much in this sample. The behaviour")
        print("   looks intrinsic to the implicit cache rather than a client-side")
        print("   setting, which means the open service ticket is the right channel")
        print("   and there is no config-level workaround to offer.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
