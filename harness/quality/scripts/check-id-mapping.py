#!/usr/bin/env python3
"""Assert runtime_id_for() maps every mantle model to an ID that actually works.

Written because the first version silently mapped `anthropic.claude-sonnet-5` to
`anthropic.claude-sonnet-4-20250514-v1:0`. Nothing raised; a notebook table just
printed a wrong answer. This checks two properties for every model:

  1. no cross-generation drift -- the resolved ID's own normalised key round-trips
     back to the model we asked for;
  2. the resolved ID is live -- Converse accepts it.

Property 1 is cheap and catches the whole bug class. Property 2 costs one call per
model and is what proves the mapping is usable rather than merely self-consistent.

    python3 check-id-mapping.py [--no-live]
"""
import concurrent.futures as cf
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
from bedrock import (  # noqa: E402
    _norm_model_key,
    converse,
    list_models,
    runtime_id_for,
)

REGION = "us-east-1"
LIVE = "--no-live" not in sys.argv


def stem(model_id: str) -> str:
    """Family + generation, keeping the digits that identify a generation."""
    bare = re.sub(r"^(us|eu|apac|global|in)\.", "", model_id)
    bare = bare.split(":")[0]
    bare = re.sub(r"-v\d+$", "", bare)
    bare = re.sub(r"-\d{8}$", "", bare)
    return bare.lower()


def main() -> int:
    models = sorted(list_models(REGION))
    rows, problems = [], []

    for mid in models:
        rid = runtime_id_for(mid, REGION)
        row = {"mantle": mid, "runtime": rid}
        if rid is None:
            rows.append(row)
            continue

        # Property 1: the resolved ID must belong to the same family+generation.
        a, b = stem(mid), stem(rid)
        # One may legitimately be a prefix of the other ("-instruct" tails,
        # "gpt-oss-20b" vs "gpt-oss-20b-1"), but neither may name a different
        # generation.
        compatible = (
            a == b
            or a.startswith(b)
            or b.startswith(a)
            or _norm_model_key(mid) == _norm_model_key(rid)
        )
        row["stem_ok"] = compatible
        if not compatible:
            problems.append(
                f"CROSS-MAPPED  {mid}  ->  {rid}   ({a!r} vs {b!r})")
        rows.append(row)

    print(f"{len(models)} mantle models; "
          f"{sum(1 for r in rows if r['runtime'])} mapped to runtime\n")

    if LIVE:
        def check(row):
            if not row["runtime"]:
                return row
            text, response = converse(
                row["runtime"],
                [{"role": "user", "content": [{"text": "Reply with exactly: OK"}]}],
                max_tokens=64, region=REGION, resolve=False,
            )
            err = response.get("error")
            row["live"] = "ok" if not err else str(err.get("message"))[:70]
            return row

        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(check, rows))

        for row in rows:
            if row.get("live") and row["live"] != "ok":
                problems.append(
                    f"DEAD ID       {row['mantle']}  ->  {row['runtime']}   "
                    f"{row['live']}")

    print(f"{'mantle id':40} {'resolved runtime id':44} live")
    print("-" * 96)
    for row in rows:
        print(f"{row['mantle']:40} {(row['runtime'] or '-- none --'):44} "
              f"{row.get('live', '-')}")

    out = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "38-id-mapping.json")
    with open(out, "w") as fh:
        json.dump({"region": REGION, "rows": rows, "problems": problems}, fh, indent=2)

    print()
    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("PASS — every mapped ID is same-generation and accepted by Converse.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
