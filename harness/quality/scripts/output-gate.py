#!/usr/bin/env python3
"""Flag code cells whose *output* is empty, truncated or self-contradicting.

Criterion 1 of the audit is not "does the cell run" but "does it yield the output
a reader would expect". A cell that prints `answer: ''` runs fine and teaches the
opposite of what it claims. This finds those mechanically.

    python3 output-gate.py [root]
"""
import glob
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

# A printed label followed by nothing, or by an empty string literal.
EMPTY_LABEL = re.compile(r"^\s*[\w §()/.\-']{1,40}\s*:\s*$")
# An empty literal at end of line, however it was labelled: `x -> 200 | ""`
# is as empty as `x: ""`, and the first form is how the output_text bug surfaced.
EMPTY_REPR = re.compile(r"""(''|""|\[\]|\{\}|:\s*None)\s*$""")
ZERO_CHARS = re.compile(r"\b0 chars\b|\(none\)|\b0 blocks\b")
TRACEBACK = re.compile(r"Traceback \(most recent call last\)")


def out_text(out) -> str:
    t = out.get("output_type")
    if t == "stream":
        return "".join(out.get("text", []))
    if t in ("execute_result", "display_data"):
        return "".join(out.get("data", {}).get("text/plain", []))
    if t == "error":
        return "ERROROUT " + str(out.get("ename")) + ": " + str(out.get("evalue"))
    return ""


def main() -> None:
    findings = []
    stats = {"cells": 0, "code": 0, "unrun": 0, "silent": 0}
    for path in sorted(glob.glob(os.path.join(ROOT, "*/*.ipynb"))):
        nb = json.load(open(path))
        prev = 0
        for i, c in enumerate(nb["cells"]):
            stats["cells"] += 1
            if c["cell_type"] != "code":
                continue
            stats["code"] += 1
            src = "".join(c["source"])
            exe = c.get("execution_count")
            outs = c.get("outputs", [])
            body = "\n".join(out_text(o) for o in outs)

            if exe is None:
                stats["unrun"] += 1
                findings.append((path, i, "NEVER-RUN", "execution_count is null"))
            elif exe != prev + 1:
                findings.append((path, i, "OUT-OF-ORDER", f"exec={exe} after {prev}"))
            if exe is not None:
                prev = exe

            # A print() inside a def/class body does not run when the cell runs, so a
            # definition-only cell producing no output is correct, not silent. Strip
            # indented lines before deciding.
            top_level = "\n".join(
                ln for ln in src.splitlines() if ln and not ln[0].isspace()
            )
            wants_output = re.search(r"\b(print|display)\s*\(", top_level)
            if wants_output and not body.strip():
                stats["silent"] += 1
                findings.append((path, i, "SILENT", "calls print() but produced nothing"))

            for o in outs:
                if o.get("output_type") == "error":
                    findings.append(
                        (path, i, "ERROR-OUT", f"{o.get('ename')}: {o.get('evalue')}"[:90])
                    )
            if TRACEBACK.search(body):
                findings.append((path, i, "TRACEBACK", "traceback in stream output"))

            lines = body.splitlines()
            for n, ln in enumerate(lines):
                # A trailing label is only a defect when nothing follows it. A
                # label that heads a block of real output is doing its job, and
                # so is `try:` inside a model-generated code listing.
                if EMPTY_LABEL.match(ln) and ln.strip() not in ("", ":"):
                    rest = [x for x in lines[n + 1:] if x.strip()]
                    if not rest:
                        findings.append((path, i, "EMPTY-LABEL", ln.strip()[:80]))
                elif EMPTY_REPR.search(ln):
                    findings.append((path, i, "EMPTY-VALUE", ln.strip()[:80]))
                elif ZERO_CHARS.search(ln) and "reasoning=0 chars |" not in ln:
                    findings.append((path, i, "ZERO-CONTENT", ln.strip()[:80]))
            # Budget truncation: the answer got cut off mid-thought.
            if re.search(r"'?status'?[:=] ?'?incomplete|stopReason'?[:=] ?'?max_tokens"
                         r"|finish_reason'?[:=] ?'?length", body):
                findings.append((path, i, "TRUNCATED", "output cut off by token budget"))

    by_kind = {}
    for f in findings:
        by_kind.setdefault(f[2], []).append(f)
    print(f"cells={stats['cells']} code={stats['code']} "
          f"never-run={stats['unrun']} silent={stats['silent']}")
    print(f"{len(findings)} findings\n")
    for kind in sorted(by_kind, key=lambda k: -len(by_kind[k])):
        print(f"## {kind}  ({len(by_kind[kind])})")
        for path, i, _, detail in by_kind[kind]:
            print(f"   {path.replace(ROOT + '/', ''):58} [{i:2}] {detail}")
        print()


if __name__ == "__main__":
    main()
