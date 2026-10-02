#!/usr/bin/env python3
"""Check that committed notebook output says what it should, not just that it exists.

Every defect this repository has shipped was visible in committed output and nobody
looked: a row printing a runtime model ID beside "mantle" only, an audio probe whose
"accepted" came from sending silence, a keyword score of 0 next to a printed line that
plainly contained the keyword.

`run-notebooks.py` proves cells do not raise, which is a much weaker property. This
checks the output itself.

Precision is the whole point. A first version of this file reported 17 findings, all
17 false: deliberate demonstrations (`content[0]['text'] -> KeyError` is the lesson),
model-generated prose that mentions exception names, a code model emitting `# TODO`
inside generated code, cross-notebook `§13` references, and `reasoning=0 chars` as a
real measured result. A checker that cries wolf gets ignored, so every rule below is
narrowed to a shape that cannot occur in a correct run, and `--self-test` proves each
one still fires on a planted defect.

    python3 verify-outputs.py [notebook ...]
    python3 verify-outputs.py --self-test
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

# The placeholder the repo's redactor substitutes. Anything else 12-digit is suspect.
REDACTED_ACCOUNT = "123456789012"

LEAKS = re.compile(
    r"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}"
    r"|Bearer\s+ey[A-Za-z0-9._-]{20,}"
    r"|aws_secret_access_key\s*[=:]"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|<YOUR_[A-Z_]+>",
    re.I)

# A real traceback, not prose that happens to name an exception class.
TRACEBACK = re.compile(r"^Traceback \(most recent call last\)", re.M)

# Words that mark a line as reporting a failure. If one of these is present, a 200 or
# an UnknownOperation on the same line is the cell doing its job.
NEGATIVE = re.compile(
    r"no such|not served|not available|not supported|unsupported|fail|trap|refus"
    r"|reject|denied|invalid|missing|\bnot\b|✗|absent|empty|error\b", re.I)


def cell_output(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs") or []:
        parts.append("".join(out.get("text") or []))
        data = out.get("data") or {}
        plain = data.get("text/plain")
        parts.append("".join(plain) if isinstance(plain, list) else str(plain or ""))
    return "\n".join(p for p in parts if p)


def has_error_output(cell: dict) -> bool:
    return any(o.get("output_type") == "error" for o in cell.get("outputs") or [])


def prints_at_module_level(src: str) -> bool:
    """True only if a print() runs when the cell runs, not one inside a def/class."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                break
        else:
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                        and sub.func.id == "print"):
                    return True
    return False


def _line_at(text: str, index: int) -> str:
    """The WHOLE line containing `index`.

    Taking `text[:m.end()].splitlines()[-1]` truncates the line at the match, which
    silently drops any suppressing word that follows the number. That let
    "usage: 420740000000 total tokens" be reported as an account ID, because the
    "tokens" that should have excluded it was cut off.
    """
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    return text[start:] if end == -1 else text[start:end]


def check_scores(text: str) -> list[str]:
    """n/m with n > m cannot happen. 0/m on a scored line means the probe found nothing."""
    found = []
    for m in re.finditer(r"(?<![\d./\w])(\d{1,3})\s*/\s*(\d{1,3})(?![\d./%\w])", text):
        got, total = int(m.group(1)), int(m.group(2))
        line = _line_at(text, m.start())
        if total == 0 or total > 60 or "/v1" in line:
            continue
        # ARITHMETIC, not a score. A model working through a maths problem writes
        # "Time = Gap / Relative Speed = 60 / 30 = 2", and `60/30` was reported as an
        # impossible score. A computed division has an `=` and a result after it; a
        # score does not. LaTeX markers are the same signal from the other direction.
        after = text[m.end():m.end() + 12]
        if re.match(r"\s*=", after):
            continue
        if re.search(r"\\text|\\frac|\\times|\\div|\$\\|\$[A-Za-z\\]", line):
            continue
        if got > total:
            found.append(f"impossible score {got}/{total}: {line.strip()[:90]}")
        elif got == 0 and re.search(r"recall|callout|correct|score|hits", line, re.I):
            found.append(f"zero score {got}/{total}: {line.strip()[:90]}")
    return found


def check_contradictions(text: str) -> list[str]:
    """Result rows that contradict themselves, unless the row reports a failure."""
    found = []
    for line in text.splitlines():
        if NEGATIVE.search(line):
            continue
        low = line.lower()
        runtime_id = re.search(r"\b(?:us|global)\.[a-z0-9.-]+|\b[a-z0-9.-]+-\d:0\b", line)
        if runtime_id and "mantle" in low and "runtime" not in low:
            found.append(f"runtime id on a mantle-only row: {line.strip()[:95]}")
        if "unknownoperation" in low and re.search(r"\b200\b", line):
            found.append(f"UnknownOperation read as success: {line.strip()[:95]}")
    return found


def check_leaks(text: str) -> list[str]:
    found = []
    if LEAKS.search(text):
        found.append(f"credential-shaped string in output: {LEAKS.search(text).group(0)[:40]}")
    for m in re.finditer(r"(?<![\d.])(\d{12})(?![\d.])", text):
        if m.group(1) == REDACTED_ACCOUNT:
            continue
        line = _line_at(text, m.start())
        if re.search(r"token|byte|char|\bms\b|latency|population|count|tps|rpm", line, re.I):
            continue
        found.append(f"unredacted 12-digit id: {line.strip()[:90]}")
    return found


def check_flow(nb: dict) -> list[str]:
    """Local section numbering must be monotonic and locally-referenced sections exist."""
    found, numbers = [], []
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for n, suffix, title in re.findall(r"^##\s+(\d+)([a-z]?)\.\s*(.*)$",
                                           "".join(cell["source"]), re.M):
            numbers.append((int(n) + (0.5 if suffix else 0.0), title.strip()))
    seq = [n for n, _ in numbers]
    for i in range(1, len(seq)):
        if seq[i] < seq[i - 1]:
            found.append(f"section numbering goes backwards: {seq[i - 1]} then {seq[i]}")
    present = {int(n) for n, _ in numbers}
    for cell in nb["cells"]:
        for line in "".join(cell["source"]).splitlines():
            # Skip cross-notebook references: those carry a path or a .ipynb name.
            if re.search(r"\.ipynb|\.\./|/0\d-|/9\d-", line):
                continue
            # A section reference QUALIFIED by a notebook path points at another
            # notebook -- "`00-foundations/04` §8c" -- and has no business being
            # checked against local headings. check-crossrefs.py validates those
            # against the target notebook's own heading count, which is the only
            # place that can. Without this guard the rule reported a correct
            # cross-notebook reference as a dangling local one.
            for m in re.finditer(r"(?:[Ss]ection|§)\s?(\d+)", line):
                before = line[max(0, m.start() - 46):m.start()]
                if re.search(r"[0-9]{2}-[a-z0-9-]+/[0-9]{2}[^\s]*[`'\"]?\s*$", before):
                    continue
                ref = m.group(1)
                if numbers and int(ref) not in present:
                    found.append(f"references section {ref}, which has no local heading: "
                                 f"{line.strip()[:70]}")
    return found


def check_notebook(path: str) -> list[str]:
    nb = json.load(open(path))
    findings = [f"flow: {f}" for f in check_flow(nb)]
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        text = cell_output(cell)
        if has_error_output(cell):
            findings.append(f"cell {i}: error output present")
        if TRACEBACK.search(text):
            findings.append(f"cell {i}: traceback in stream output")
        for label, results in (("", check_scores(text)),
                               ("", check_contradictions(text)),
                               ("", check_leaks(text))):
            findings += [f"cell {i}: {r}" for r in results]
        if prints_at_module_level(src) and not text.strip():
            findings.append(f"cell {i}: prints at module level but produced no output")
    return findings


SELF_TESTS = [
    ("impossible score", "callouts 5/3  title=True", "impossible score"),
    ("zero score", "keyword recall 0/3", "zero score"),
    ("runtime id on mantle row",
     "openai.gpt-oss-120b   openai.gpt-oss-120b-1:0   mantle", "runtime id"),
    ("unknown op as success", "/v1/responses   200   UnknownOperationException",
     "read as success"),
    ("leaked key", "key=AKIAIOSFODNN7EXAMPLE", "credential-shaped"),  # pragma: allowlist secret
    ("real account id", '"arn": "arn:aws:bedrock:us-east-1:987654321098:project/x"',
     "unredacted"),
]


def self_test() -> int:
    """Prove each rule still fires. A checker nobody has falsified is not a checker."""
    bad = 0
    for label, text, expect in SELF_TESTS:
        hits = check_scores(text) + check_contradictions(text) + check_leaks(text)
        ok = any(expect in h for h in hits)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:26} -> {hits or 'nothing'}")
        bad += not ok
    # And prove the known false-positive shapes stay silent.
    quiet = [
        ("deliberate KeyError demo", "content[0]['text'] -> KeyError, block 0 is reasoning"),
        ("real zero measurement", "provider fields  reasoning=0 chars | Let's solve"),
        ("labelled trap", "/v1/responses   200  NO SUCH PATH (UnknownOperationException)"),
        ("redaction placeholder", 'arn:aws:bedrock:us-east-1:123456789012:project/x'),
        ("token count", "usage: 420740000000 total tokens"),
        # A model's arithmetic, not a score: `60/30` with 60 > 30 looked impossible.
        ("model arithmetic",
         "*   Time = $\\text{Gap} / \\text{Relative Speed} = 60 / 30 = 2"),
        ("plain arithmetic with an equals", "so 60 / 30 = 2 hours in total"),
    ]
    for label, text in quiet:
        hits = check_scores(text) + check_contradictions(text) + check_leaks(text)
        ok = not hits
        print(f"  {'ok  ' if ok else 'FAIL'} silent on {label:22} -> {hits or 'nothing'}")
        bad += not ok
    # check_flow takes a notebook rather than a line of output, so it needs its own
    # fixtures. This case is a false positive worth pinning: a section reference
    # QUALIFIED by a notebook path is a cross-notebook pointer, not a dangling local
    # one, and check-crossrefs.py is the checker that can validate it.
    flow_cases = [
        ("qualified cross-notebook reference stays silent",
         "measured over the 15 ids in `00-foundations/04` \u00a78c: both hold", 0),
        ("bare dangling local reference is caught", "see section 8 for the rest", 1),
        ("valid local reference stays silent", "see section 2 for the rest", 0),
    ]
    for label, line, want in flow_cases:
        nb = {"cells": [
            {"cell_type": "markdown", "id": "a", "source": ["## 1. One\n"]},
            {"cell_type": "markdown", "id": "b", "source": ["## 2. Two\n"]},
            {"cell_type": "markdown", "id": "c", "source": [line + "\n"]},
        ]}
        got = [f for f in check_flow(nb) if "no local heading" in f]
        ok = len(got) == want
        print(f"  {'ok  ' if ok else 'FAIL'} {label:50} -> {got or 'nothing'}")
        bad += not ok

    print(f"\nself-test: {'all rules behave' if not bad else str(bad) + ' rule(s) wrong'}")
    return 1 if bad else 0


def main() -> None:
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    targets = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(
        glob.glob(os.path.join(REPO, "*", "*.ipynb")))
    targets = [t if os.path.isabs(t) else os.path.join(REPO, t) for t in targets]
    total = 0
    for path in targets:
        found = check_notebook(path)
        if found:
            total += len(found)
            print(f"\n{os.path.relpath(path, REPO)}")
            for line in found:
                print(f"  {line}")
    print(f"\n{total} output finding(s) across {len(targets)} notebook(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
