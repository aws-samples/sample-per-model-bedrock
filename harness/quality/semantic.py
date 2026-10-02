"""Semantic review aid: find cells that RAN but whose output looks wrong.

`quality/audit.py` checks that cells executed. This asks a harder question: does the
output actually demonstrate what the cell claims? A cell can exit 0 and still be
useless — an empty answer, a table of error codes, a "comparison" with one row.

Everything here is a *lead to read*, not a verdict. Each hit is reviewed by hand;
the point is to make sure nothing is skipped.

Usage:
    python quality/semantic.py                # all notebooks, grouped by signal
    python quality/semantic.py <notebook>     # one notebook, verbose
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("REPO") or Path(__file__).resolve().parents[2])

# Output text that suggests the cell did not achieve its purpose.
SUSPECT = [
    # An HTTP error where the cell was not deliberately demonstrating one.
    (r"\bHTTP (?:4\d\d|5\d\d)\b", "http-error", "HTTP error in output"),
    (r"\bHTTP -1\b", "stalled", "request stalled (-1)"),
    # Empty or missing model answers.
    (r"answer\s*[:=]\s*['\"]{2}", "empty-answer", "empty answer string"),
    (r"^\s*\(empty", "empty-answer", "explicitly empty output"),
    (r"\bstatus\s*[:=]\s*['\"]?incomplete", "incomplete", "status=incomplete"),
    (r"finish_reason\s*[:=]\s*['\"]?length", "truncated", "finish_reason=length"),
    # Failures that were caught and printed rather than raised.
    (r"\bFAILED\b|\bFAIL\b(?!ED)", "reported-failure", "cell printed FAIL"),
    (r"\berror\b\s*[:=]", "reported-error", "cell printed an error field"),
    (r"\bNone\b\s*$", "none-value", "output ends in None"),
    (r"timed out", "timeout", "timeout reported in output"),
    (r"\bnot found\b", "not-found", "'not found' in output"),
    (r"lookup failed", "lookup-failed", "lookup failure"),
    (r"\b0 (?:models|findings|citations|rounds|chunks|deltas)\b", "zero-count",
     "a count that should be non-zero is 0"),
    (r"\(none\)", "none-listed", "'(none)' in a listing"),
    (r"\bunavailable\b", "unavailable", "model reported unavailable"),
]

# Cells that legitimately show an error are the ones teaching a failure mode.
# Recognise them from the code or the markdown immediately above.
DELIBERATE = re.compile(
    r"deliberately|on purpose|must 400|should 400|expect(?:ed)? (?:a )?(?:400|404|failure|error)"
    r"|prove|proof|demonstrat|wrong (?:path|route|prefix)|invalid|rejects?|rejected"
    r"|does not support|not supported|fails fast|fail fast|trap|gotcha|breakage"
    r"|no such|non-existent|does-not-exist|fake|stall",
    re.I,
)


def cells_of(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("cells", [])


def output_text(cell: dict) -> str:
    parts: list[str] = []
    for out in cell.get("outputs") or []:
        kind = out.get("output_type")
        if kind == "stream":
            parts.append("".join(out.get("text") or []))
        elif kind in ("execute_result", "display_data"):
            plain = (out.get("data") or {}).get("text/plain")
            if isinstance(plain, list):
                parts.append("".join(plain))
            elif plain:
                parts.append(str(plain))
        elif kind == "error":
            parts.append(f"ERROR {out.get('ename')}: {out.get('evalue')}")
    return "".join(parts)


def review(path: Path) -> list[tuple]:
    """Return [(cell_no, signal, note, evidence)] for one notebook."""
    hits: list[tuple] = []
    cells = cells_of(path)
    preceding_md = ""
    code_no = 0
    for cell in cells:
        source = "".join(cell.get("source") or [])
        if cell.get("cell_type") == "markdown":
            preceding_md = source
            continue

        code_no += 1
        out = output_text(cell)
        context = source + "\n" + preceding_md
        excused = bool(DELIBERATE.search(context))

        if not out.strip():
            # A setup cell with no print is fine; one that calls the API is not.
            if re.search(r"\b(?:post|create|client\.|print)\s*\(", source):
                hits.append((code_no, "silent", "API/print call produced no output",
                             source.strip().split("\n")[0][:70]))
            preceding_md = ""
            continue

        for pattern, signal, note in SUSPECT:
            match = re.search(pattern, out, re.M)
            if not match:
                continue
            if excused and signal in {
                "http-error", "stalled", "reported-failure", "reported-error",
                "not-found", "unavailable", "timeout", "truncated", "incomplete",
                "empty-answer",
            }:
                continue
            line = out[max(0, match.start() - 60):match.end() + 60]
            hits.append((code_no, signal, note, " ".join(line.split())[:120]))
        preceding_md = ""
    return hits


def main() -> int:
    targets = (
        [Path(sys.argv[1])] if len(sys.argv) > 1
        else sorted(ROOT.glob("*/*.ipynb"))
    )
    total = 0
    by_signal: dict[str, int] = {}
    for nb in targets:
        hits = review(nb)
        if not hits:
            continue
        print(f"\n=== {nb.relative_to(ROOT) if nb.is_relative_to(ROOT) else nb}")
        for code_no, signal, note, evidence in hits:
            print(f"  cell {code_no:3}  [{signal:16}] {note}")
            print(f"            {evidence}")
            by_signal[signal] = by_signal.get(signal, 0) + 1
            total += 1
    print(f"\n{'=' * 70}\nleads to review: {total}")
    for signal, count in sorted(by_signal.items(), key=lambda kv: -kv[1]):
        print(f"  {signal:18} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
