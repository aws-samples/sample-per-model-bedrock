#!/usr/bin/env python3
"""Split the claim inventory into DERIVED and ASSERTED.

The defect class this repository keeps producing is a verdict written into prose,
sitting beside a live probe that was never consulted. `extract-claims.py` finds the
verdicts. This decides which ones the notebook itself establishes.

A claim is DERIVED if the distinctive tokens it depends on -- the model ID, the API
path, the Region, the field name, the status -- appear in committed OUTPUT somewhere in
the same notebook. That is deliberately generous about position: notebooks legitimately
summarise in a heading before probing, and a claim proved in section 3 may be restated
in section 9.

Everything else is ASSERTED. An asserted claim is not automatically wrong -- some are
correct facts that simply have no probe -- but it is exactly the population where a
stale fact survives, so each one needs evidence from outside the repo.

    python3 evidence-audit.py                    # summary + the asserted list
    python3 evidence-audit.py --json out.json
    python3 evidence-audit.py --self-test
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location(
    "extract_claims", os.path.join(HERE, "extract-claims.py"))
extract_claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_claims)

# Tokens too common to carry evidence: finding "200" in an output proves nothing about
# a claim that a specific call returns 200.
WEAK = {"200", "400", "v1", "/v1", "1", "2", "3", "one", "two", "three"}


def outputs_text(nb: dict) -> str:
    """Every character of committed output in a notebook, concatenated."""
    parts = []
    for cell in nb["cells"]:
        for out in cell.get("outputs") or []:
            for key in ("text", "ename", "evalue"):
                val = out.get(key)
                if isinstance(val, list):
                    parts.append("".join(val))
                elif isinstance(val, str):
                    parts.append(val)
            data = out.get("data") or {}
            for val in data.values():
                if isinstance(val, list):
                    parts.append("".join(val))
                elif isinstance(val, str):
                    parts.append(val)
    return "\n".join(parts)


def code_text(nb: dict) -> str:
    """Every character of code in a notebook. A field name present only in the source
    is weaker evidence than one echoed by the service, but it does show the notebook
    exercises it rather than merely describing it."""
    return "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")


def tokens(claim: dict) -> list[str]:
    """The distinctive strings a claim stands or falls on."""
    out: list[str] = list(claim["ids"])
    s = claim["sentence"]
    out += extract_claims.PATH_RE.findall(s)
    out += extract_claims.REGION_RE.findall(s)
    out += extract_claims.fields(s)
    return [t for t in {t for t in out if t and t.lower() not in WEAK} if len(t) > 3]


def audit(path: str) -> list[dict]:
    nb = json.load(open(path))
    outs = outputs_text(nb)
    code = code_text(nb)
    rows = []
    for claim in extract_claims.harvest(path):
        toks = tokens(claim)
        in_output = [t for t in toks if t in outs]
        in_code = [t for t in toks if t in code and t not in in_output]
        if not toks:
            # A SUPPORT- or NUMBER-only sentence with no distinctive token. Its
            # subject is whatever the section is about, which this cannot resolve, so
            # it is reported separately rather than guessed at.
            status = "UNTOKENED"
        elif in_output:
            status = "DERIVED"
        elif in_code:
            status = "EXERCISED"
        else:
            status = "ASSERTED"
        rows.append({**claim, "tokens": toks, "status": status,
                     "evidence": in_output or in_code})
    return rows


SELF_TEST_NB = {
    "cells": [
        {"cell_type": "markdown", "id": "a",
         "source": ["The model `us.xai.grok-4.6` returns 200 on this path.\n"]},
        {"cell_type": "code", "id": "b", "source": ["print(1)\n"],
         "outputs": [{"output_type": "stream", "name": "stdout",
                      "text": ["us.xai.grok-4.6 -> 200\n"]}]},
        {"cell_type": "markdown", "id": "c",
         "source": ["The model `google.gemma-4-31b` is only in us-west-2.\n"]},
        {"cell_type": "markdown", "id": "d",
         "source": ["We call `input_audio` here, which the endpoint accepts.\n"]},
        {"cell_type": "code", "id": "e",
         "source": ["body = {'input_audio': clip}\n"], "outputs": []},
    ],
    "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
}


def self_test() -> int:
    tmp = os.path.join(HERE, ".evidence-selftest.ipynb")
    json.dump(SELF_TEST_NB, open(tmp, "w"))
    try:
        rows = audit(tmp)
        got = {r["cell"]: r["status"] for r in rows}
        want = {0: "DERIVED", 2: "ASSERTED", 3: "EXERCISED"}
        bad = 0
        for cell, expect in want.items():
            if got.get(cell) != expect:
                print(f"  cell {cell}: want {expect}, got {got.get(cell)!r}")
                bad += 1
        # And the categories must have survived the round trip.
        if not any("REGION" in r["categories"] for r in rows):
            print("  REGION claim vanished between extract and audit")
            bad += 1
        print(f"  {len(want)} cases, {bad} problem(s)")
        return bad
    finally:
        os.remove(tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--status", default="ASSERTED")
    ap.add_argument("--show", type=int, default=60)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)

    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "*", "*.ipynb"))):
        rows.extend(audit(path))

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"{len(rows)} claim(s)\n")
    for status in ("DERIVED", "EXERCISED", "ASSERTED", "UNTOKENED"):
        print(f"  {status:<10} {counts.get(status, 0):>5}")

    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
        print(f"\nfull inventory -> {args.json}")

    picked = [r for r in rows if r["status"] == args.status and not r["hedged"]]
    print(f"\n{len(picked)} unhedged {args.status} claim(s); "
          f"showing {min(args.show, len(picked))}\n")
    for r in picked[:args.show]:
        print(f"  {r['file']} cell {r['cell']}  [{','.join(r['categories'])}]")
        print(f"      {r['sentence'][:150]}")
        if r["tokens"]:
            print(f"      tokens: {', '.join(r['tokens'][:6])}")


if __name__ == "__main__":
    main()
