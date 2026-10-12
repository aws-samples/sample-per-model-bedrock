"""Checks for parse_json_lenient, the helper the structured-output and
tool-call cells use to recover a JSON object from untidy model output.

Run it with `python3 _shared/test_parse_json_lenient.py` from the repository
root. It calls no AWS service, so it does not depend on a model emitting the
awkward output on the run you happen to make.

The function's two halves pull against each other: trailing characters after a
complete object have to be ignored, while an object truncated mid-flight has to
be closed and kept. A change that helps one can quietly break the other, so
both are pinned here, next to the stray-brace output that used to raise.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bedrock import parse_json_lenient  # noqa: E402  (after the path insert)

INVOICE = {
    "invoice_id": "INV-1042",
    "total": 1280.5,
    "currency": "AUD",
    "vendor": "Acme Pty Ltd",
}

# (what a model emitted, what parse_json_lenient should return)
PARSES = [
    ('{"a": 1}', {"a": 1}),
    # Gemma 4 appends characters after a complete object in strict mode.
    ('{"a": 1} and then some chatter', {"a": 1}),
    ("```json\n{\"a\": 1}\n```", {"a": 1}),
    # A stray brace ahead of the real object. Observed on gpt-oss, two runs in
    # four of 14-openai-gpt-oss/01, and the object after it is well formed.
    ("{\n" + json.dumps(INVOICE), INVOICE),
    ('Here you go: { {"a": 1}', {"a": 1}),
    # qwen3-coder truncates tool-call arguments mid-object, and mid-string.
    ('{"sku": "A-100", "quantity": 4', {"sku": "A-100", "quantity": 4}),
    ('{"path": "x.py', {"path": "x.py"}),
    # A truncated outer object is worth more than the inner one it holds,
    # including when the inner one is complete and would parse on its own. The
    # first case alone cannot pin this: its inner object is truncated too, so
    # there is no balanced inner object to be returned by mistake.
    ('{"a": {"b": 1', {"a": {"b": 1}}),
    ('{"a": {"b": 1}, "c": 2', {"a": {"b": 1}, "c": 2}),
    (
        '{"tool": "search", "args": {"q": "bedrock"}, "id": "call_1',
        {"tool": "search", "args": {"q": "bedrock"}, "id": "call_1"},
    ),
    # Both faults at once: a stray brace, and no closing brace either.
    ('Here you go: { {"a": 1', {"a": 1}),
    # Braces inside a string are text, not structure.
    ('{"text": "a } b {"} junk', {"text": "a } b {"}),
    ('{"nested": {"b": 1}} junk', {"nested": {"b": 1}}),
]

# Output with no recoverable object has to raise rather than return a guess,
# and the error has to be the helper's own ValueError: a json.JSONDecodeError
# escaping would subclass ValueError and so satisfy a bare `except ValueError`
# while telling the caller nothing about which text failed.
#
# Only the last of these is new behaviour. It is a model stuck repeating a
# token: the walk gives up after _MAX_OBJECT_STARTS starts, so it raises
# promptly instead of running quadratically. The first three raise on any
# version of the helper, and are here so a future change cannot start
# returning a guess for them.
RAISES = ["", "no object here at all", "{not json at all}", "{" * 5000]


def main() -> int:
    failures = []
    for text, expected in PARSES:
        try:
            got = parse_json_lenient(text)
        except Exception as exc:  # noqa: BLE001  (any failure is a failure)
            failures.append(f"raised {type(exc).__name__}: {exc}  for {text!r}")
            continue
        if got != expected:
            failures.append(f"got {got!r}, expected {expected!r}  for {text!r}")
    for text in RAISES:
        try:
            got = parse_json_lenient(text)
        except json.JSONDecodeError as exc:
            failures.append(f"leaked JSONDecodeError: {exc}  for {text[:40]!r}")
            continue
        except ValueError:
            continue
        failures.append(
            f"returned {got!r} instead of raising ValueError for {text[:40]!r}"
        )

    checks = len(PARSES) + len(RAISES)
    print(f"parse_json_lenient: {checks - len(failures)}/{checks} checks passed")
    for line in failures:
        print("  FAIL", line)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
