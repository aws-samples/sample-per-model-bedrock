#!/usr/bin/env python3
"""Find cells that report a truncated answer as if it were a measurement.

Two of these have shipped:

  10-nvidia-nemotron/01  vision scored 1/3 because the answer was cut off mid-phrase
                         at "Lower compute cost," -- a 260-token cap read as a
                         capability limit. Raised to 500: 3/3.
  07-mistral/01     the reasoning-effort table reported 700 completion tokens for
                         magistral at BOTH effort levels, because 700 was max_tokens.
                         The column measured the budget, not the model. At a real
                         budget the two levels differ in opposite directions per
                         model, which is the whole point of the cell.

The tell is a number in the output that EQUALS a budget in the source, in a cell that
never prints `finish_reason`. A cell that prints finish_reason is not caught, and
should not be: several cells demonstrate truncation on purpose and say so.

    python3 check-truncation.py
    python3 check-truncation.py --self-test
"""
import argparse
import glob
import json
import os
import re
import sys

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

BUDGET_KEYS = ("max_tokens", "max_output_tokens", "max_completion_tokens",
               "maxTokens", "budget")
# A cell that shows the reader the stop reason has disclosed the truncation.
DISCLOSES = re.compile(
    r"finish_reason|finish\b|stopReason|stop_reason|truncat|\bceiling\b|"
    r"hit the .*budget|raise .*budget|incomplete", re.I)


def budgets(source: str) -> set[int]:
    """Numeric token budgets the cell requests."""
    found: set[int] = set()
    for key in BUDGET_KEYS:
        for m in re.finditer(rf"{key}\W{{1,4}}(\d{{2,6}})", source):
            found.add(int(m.group(1)))
    # `BUDGET = 4000` style constants defined in the cell
    for m in re.finditer(r"\b(?:BUDGET|CAP|LIMIT)\s*=\s*(\d{2,6})", source):
        found.add(int(m.group(1)))
    return found


def output_text(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs") or []:
        parts.append("".join(out.get("text") or []))
        for val in (out.get("data") or {}).values():
            parts.append("".join(val) if isinstance(val, list) else str(val))
    return "\n".join(parts)


def suspicious(source: str, out: str) -> list[str]:
    """Budget values that appear in the output of a cell that hides the stop reason."""
    if not out.strip():
        return []
    if DISCLOSES.search(source):
        return []
    hits = []
    for budget in budgets(source):
        # Guard against `[\w.-]` on both sides, not just digits: `30` matched inside
        # `qwen3-coder-30b-a3b-instruct`, a model name, and reported a truncation
        # that did not exist.
        pattern = rf"(?<![\w.-]){budget}(?![\w.-])"
        # And the number has to be sitting where a COUNT sits. `200` is a plausible
        # token budget and also the most common string in any of these outputs,
        # because it is an HTTP status: `-> HTTP 200` was reported as a truncated
        # answer in four notebooks. So require a count-ish label within 40 characters
        # before the number, and reject an HTTP-status context outright.
        # A COLUMN table carries its label in the header row, not before each value,
        # so "label within 40 characters" cannot see it. The self-test caught that on
        # the very shape this checker was written for. So a token-count header
        # anywhere in the output licenses count-context for the whole output.
        COUNT_LABEL = (r"tok(?:en)?s?\b|chars?\b|\blen\b|\bsize\b|completion|"
                       r"output|\bout\b|budget|spent")
        header = bool(re.search(COUNT_LABEL, out, re.I))
        counted = 0
        for m in re.finditer(pattern, out):
            before = out[max(0, m.start() - 40):m.start()]
            if re.search(r"HTTP\s*$|status\s*[:=]?\s*$|code\s*[:=]?\s*$|->\s*$",
                         before, re.I):
                continue
            # An ECHO of the request is not a measurement. Three notebooks print the
            # budget back at the reader -- inside a curl command, or as the row label
            # `max_tokens=16` in a parameter-acceptance table -- and every one was
            # reported as a truncated answer.
            if re.search(r"(?:max_tokens|max_output_tokens|max_completion_tokens|"
                         r"maxTokens)\W{0,3}$", before):
                continue
            if header or re.search(COUNT_LABEL, before, re.I):
                counted += 1
        if counted:
            hits.append(f"{budget} appears {counted}x in output where a token count "
                        f"sits, and is a token budget in the source; the cell never "
                        f"prints a stop reason, so a truncated answer is "
                        f"indistinguishable from a complete one")
    return hits


def check(path: str) -> list[str]:
    nb = json.load(open(path))
    problems = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for message in suspicious("".join(cell["source"]), output_text(cell)):
            problems.append(f"cell {i}: {message}")
    return problems


SELF_TEST = [
    # (source, output, expected finding count)
    # The Mistral defect, as it shipped.
    # The real cell prints a `completion tokens` header above the column; the first
    # version of this fixture omitted it and so tested a shape that never occurs.
    ('print(f"{\'model\':40} {\'effort\':8} {\'completion tokens\':>18}")\n'
     'for effort in ("low","high"):\n'
     '    body = {"max_tokens": 700, "reasoning_effort": effort}\n'
     '    print(f"{model} {effort} {tokens}")',
     "model      effort    completion tokens\n"
     "magistral  low                     700\n"
     "magistral  high                    700\n", 1),
    # The same cell once it prints finish_reason: not a finding.
    ('body = {"max_tokens": 700}\nprint(finish_reason, tokens)',
     "magistral low 700 length\n", 0),
    # A cell that deliberately demonstrates truncation.
    ('# 250 truncates on purpose, to show the trailing-character problem\n'
     'body = {"max_tokens": 250}',
     "max_tokens=  250 HTTP 200 chars=295\n", 0),
    # A budget that does not appear in the output at all.
    ('body = {"max_tokens": 2000}\nprint(answer)',
     "The capital of France is Paris.\n", 0),
    # A budget appearing once because the answer used it all, with no disclosure --
    # still worth flagging, which is the point. Needs a count label to be found.
    ('body = {"max_output_tokens": 512}\nprint("output tokens:", n)',
     "output tokens: 512\n", 1),
    # `200` as an HTTP status, not a token count. This shape produced four false
    # positives on the first run.
    ('body = {"max_tokens": 200}\nprint(f"  correct -> HTTP {code}")',
     "  correct: header + tool     -> HTTP 200\n  header only -> HTTP 200\n", 0),
    # A budget digit-sequence occurring inside a MODEL NAME.
    ('body = {"max_tokens": 30}\nprint(f"{model:40} {verdict}")',
     "qwen.qwen3-coder-30b-a3b-instruct     400\n", 0),
    # The budget echoed back as a row LABEL in a parameter-acceptance table.
    ('print(f"{label:26} {code:>4}")\nbody = {"max_tokens": 16}',
     "parameters    HTTP  detail\nmax_tokens=16   200\n"
     "max_completion_tokens=16   200\n", 0),
    # The budget echoed inside a printed curl command.
    ('body = {"max_output_tokens": 16}\nprint(curl_command)',
     'curl -d \'{"model":"x","input":"Reply OK","max_output_tokens":16}\'\n'
     "live result: 'OK'\n", 0),
    # The deepseek shape: a labelled count that equals the cap, answer empty.
    ('for effort in ("low","high"):\n    body = {"max_tokens": 600}\n'
     '    print(f"--- effort={effort} completion_tokens={n} ---")',
     "--- effort=low completion_tokens=390 ---\nsome answer\n"
     "--- effort=high completion_tokens=600 ---\n\n", 1),
]


def self_test() -> int:
    bad = 0
    for source, out, want in SELF_TEST:
        got = suspicious(source, out)
        if len(got) != want:
            print(f"  want {want}, got {len(got)}: {got}")
            print(f"    source: {source[:70]!r}")
            bad += 1
    print(f"  {len(SELF_TEST)} cases, {bad} problem(s)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(1 if self_test() else 0)

    total = 0
    for path in sorted(glob.glob(os.path.join(REPO, "*", "*.ipynb"))):
        found = check(path)
        if found:
            total += len(found)
            print(f"\n{os.path.relpath(path, REPO)}")
            for line in found:
                print(f"  {line}")
    print(f"\n{total} possible truncation-as-measurement site(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
