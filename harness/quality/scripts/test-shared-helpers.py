#!/usr/bin/env python3
"""Regression tests for `_shared/bedrock.py`, the module all 34 notebooks import.

Every case here is a defect that was actually shipped, not a hypothetical. Two review
rounds found the same shape twice: a fix that was correct for the example in front of
it and wrong one input over. `extract_code_block` was fixed three times before it
handled an indented fence.

    python3 test-shared-helpers.py
"""
import contextlib
import io
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
import bedrock  # noqa: E402

FAILURES: list[str] = []


def chk(label: str, got, want) -> None:
    if got == want:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")
        FAILURES.append(label)


def test_extract_code_block() -> None:
    """Wrong in three consecutive rounds, each fix breaking the previous one:

      1. matching a bare language tag left "python title=x" in the source;
      2. anchoring both fences to column 0 rejected an indented fence, which is what
         a model emits under a numbered list;
      3. allowing any text after the marker let a prose line that merely STARTS with
         an inline span swallow the real block -- and restricting the character set to
         fix that rejected the legitimate "python title=x" from (1).

    Every case below is one of those, or a shape that broke while fixing one.
    """
    print("\nextract_code_block")
    f = bedrock.extract_code_block
    chk("indented under a list item",
        f("1. Do this:\n   ```python\n   x = 1\n   ```\n2. Done."), "x = 1")
    chk("indent stripped so Python parses",
        f("  ```python\n  def g():\n      return 1\n  ```"),
        "def g():\n    return 1")
    chk("unindented", f("```python\nx = 1\n```"), "x = 1")
    chk("info string with a space in it", f("```python title=x\ny = 2\n```"), "y = 2")
    chk("bare fence, no language", f("```\nn = 7\n```"), "n = 7")
    chk("four backticks", f("````python\nz = 3\n````"), "z = 3")
    chk("tilde fence", f("~~~python\nw = 4\n~~~"), "w = 4")
    chk("inline span mid-sentence does not open a match",
        f("Use ``` fences ``` like so:\n```python\nq = 5\n```"), "q = 5")
    chk("prose STARTING with an inline span does not either",
        f("``` is the fence marker. Here is the code:\n```python\n"
          "def f():\n    return 1\n```"), "def f():\n    return 1")
    chk("truncated generation with no closing fence",
        f("```python\ndef f(x):\n    return x\n"), "def f(x):\n    return x")
    chk("prose before the block", f("Here you go:\n```python\nm = 8\n```\nDone."),
        "m = 8")
    chk("blank line inside the block survives",
        f("  ```py\n  a = 1\n\n  b = 2\n  ```"), "a = 1\n\nb = 2")
    chk("no fence returns the prose", f("just prose"), "just prose")
    chk("empty input", f(""), "")
    chk("None input", f(None), "")
    # Whatever comes back must be parseable Python, or inspect_code() blames the
    # model for the extractor's mistake -- which is how (2) and (3) shipped.
    import ast
    for label, text in [
        ("truncated", "```python\ndef f(x):\n    return x\n"),
        ("indented", "1. Do:\n   ```python\n   def h():\n       return 2\n   ```"),
        ("info string", "```python title=x\ndef k():\n    return 3\n```"),
    ]:
        try:
            ast.parse(f(text))
            print(f"  ok    {label} extraction parses as Python")
        except SyntaxError as exc:
            print(f"  FAIL  {label} extraction does not parse: {exc}")
            FAILURES.append(f"{label} parse")


def test_inspect_code() -> None:
    """`classes` was documented as top-level but collected with ast.walk, and the
    parameter list dropped positional-only, *args and **kwargs."""
    print("\ninspect_code")
    nested = bedrock.inspect_code(
        "def outer():\n    class Inner:\n        def m(self): pass\n")
    chk("class nested in a function is not top-level", nested["classes"], [])
    top = bedrock.inspect_code("class A:\n    def m(self): pass\n")
    chk("real top-level class", top["classes"], ["A"])
    chk("its method is reported separately", sorted(top["methods"]), ["A.m"])
    chk("and not as a function", sorted(top["functions"]), [])
    full = bedrock.inspect_code(
        "def parse_config(path, /, strict=False, *rest, **kw):\n"
        "    raise ValueError('x')\n")
    chk("every parameter kind, in order", full["functions"]["parse_config"],
        ["path", "strict", "*rest", "**kw"])
    chk("raise recorded", full["raises"], ["ValueError"])
    broken = bedrock.inspect_code("def f(:\n")
    chk("syntax error reported, not raised", broken["parses"], False)
    chk("with a line number", broken["error"].startswith("line "), True)


def test_check_spec() -> None:
    """`reason` was empty whenever the function existed, so a signature or guard
    mismatch printed a bare False with nothing for the reader to act on."""
    print("\ncheck_spec")
    good = bedrock.check_spec(
        "def parse_config(path, /, strict=False):\n    raise ValueError('x')\n",
        function="parse_config", params=["path", "strict"], raises="ValueError")
    chk("passes on correct code", good["ok"], True)
    chk("and says nothing", good["reason"], "")
    wrong_sig = bedrock.check_spec(
        "def parse_config(p):\n    raise ValueError('x')\n",
        function="parse_config", params=["path"], raises="ValueError")
    chk("signature mismatch fails", wrong_sig["ok"], False)
    chk("and names both lists",
        wrong_sig["reason"], "parse_config takes ['p'], expected ['path']")
    as_method = bedrock.check_spec(
        "class C:\n    def parse_config(self, path): pass\n",
        function="parse_config", params=["path"])
    chk("a method is not a top-level def", as_method["ok"], False)
    chk("and the reason says where it went",
        "method C.parse_config" in as_method["reason"], True)
    no_guard = bedrock.check_spec(
        "def parse_config(path):\n    return 1\n",
        function="parse_config", params=["path"], raises="ValueError")
    chk("missing guard fails", no_guard["ok"], False)
    chk("and names what was expected",
        no_guard["reason"], "parse_config raises nothing, expected ValueError")
    # Adding *args/**kwargs to the reported signature (round two, correctly) broke the
    # `==` comparison for code that MEETS the spec and also accepts a catch-all.
    kwargs_ok = bedrock.check_spec(
        "def chunk_by_tokens(text, max_tokens, overlap=50, **kwargs):\n"
        "    raise ValueError('x')\n",
        function="chunk_by_tokens", params=["text", "max_tokens", "overlap"],
        raises="ValueError")
    chk("a spec-compliant signature with **kwargs passes", kwargs_ok["ok"], True)
    short = bedrock.check_spec(
        "def chunk_by_tokens(text, max_tokens):\n    raise ValueError('x')\n",
        function="chunk_by_tokens", params=["text", "max_tokens", "overlap"],
        raises="ValueError")
    chk("a genuinely missing parameter still fails", short["ok"], False)
    # `raise json.JSONDecodeError(...)` recorded nothing, because only `.id` was read.
    attr_raise = bedrock.check_spec(
        "import json\ndef f():\n    raise json.JSONDecodeError('m', 'd', 0)\n",
        function="f", raises="JSONDecodeError")
    chk("an exception raised through an attribute path is seen", attr_raise["ok"], True)


def test_err() -> None:
    """`err()` assumed `error` is a dict; mantle sometimes sends a string."""
    print("\nerr")
    chk("dict error", bedrock.err({"error": {"message": "boom"}}), "boom")
    chk("string error", bedrock.err({"error": "Internal server error"}),
        "Internal server error")
    # Not "": when the body is not the expected shape, err() shows the RAW body so
    # the reader sees what arrived. Returning "" would hide it.
    chk("None payload shows the raw body", bedrock.err(None), "null")
    chk("empty dict shows the raw body", bedrock.err({}), "{}")
    chk("string payload shows the raw body", bedrock.err("plain text"),
        '"plain text"')
    chk("limit is honoured", len(bedrock.err({"error": {"message": "x" * 900}},
                                             limit=50)) <= 53, True)


def test_degradation_is_announced() -> None:
    """A control-plane failure must not become a confident negative in silence.
    Round 1 wired the notice into the mantle branch only."""
    print("\ncontrol-plane degradation")
    bedrock._WARNED.clear()
    with mock.patch.object(bedrock, "runtime_models",
                           side_effect=RuntimeError("AccessDenied")):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            answer = bedrock.endpoints_for("openai.gpt-oss-120b")
            resolved = bedrock.runtime_id_for("openai.gpt-oss-120b")
        out = buf.getvalue()
    chk("endpoints_for still answers", answer["runtime"], False)
    chk("runtime_id_for still answers", resolved, None)
    chk("and the degradation is printed",
        "could not list bedrock-runtime models" in out, True)
    bedrock._WARNED.clear()


def test_keyword_recall() -> None:
    """Scoring and display must collapse whitespace the same way, or an NBSP scores
    0 while the printed line looks correct."""
    print("\nkeyword_recall")
    chk("plain hit", bedrock.keyword_recall("we save cost on S3", ("save cost",)),
        (1, 1))
    chk("non-breaking space still matches",
        bedrock.keyword_recall("we save\u00a0cost on S3", ("save cost",)), (1, 1))
    chk("newline inside the phrase",
        bedrock.keyword_recall("we save\ncost", ("save cost",)), (1, 1))
    chk("case-insensitive", bedrock.keyword_recall("SAVE COST", ("save cost",)), (1, 1))
    chk("miss", bedrock.keyword_recall("nothing here", ("save cost",)), (0, 1))
    chk("empty answer", bedrock.keyword_recall("", ("a",)), (0, 1))
    chk("None answer", bedrock.keyword_recall(None, ("a",)), (0, 1))


def test_norm_model_key() -> None:
    """A `-vN` tail is the version marker, so a digit before it is a generation.
    Eating it collapsed opus-4-7 and opus-4-8 onto one key."""
    print("\n_norm_model_key round trips")
    pairs = [
        ("openai.gpt-oss-20b", "openai.gpt-oss-20b-1:0"),
        ("qwen.qwen3-32b", "qwen.qwen3-32b-v1:0"),
        ("zai.glm-5", "zai.glm-5-v1:0"),
        ("anthropic.claude-opus-4-7", "anthropic.claude-opus-4-7-v1:0"),
        ("anthropic.claude-opus-4-8", "anthropic.claude-opus-4-8-v1:0"),
        ("moonshotai.kimi-k2-thinking", "moonshot.kimi-k2-thinking"),
    ]
    for mantle, runtime in pairs:
        chk(f"{mantle} == {runtime}",
            bedrock._norm_model_key(mantle) == bedrock._norm_model_key(runtime), True)
    chk("opus-4-7 and opus-4-8 stay distinct",
        bedrock._norm_model_key("anthropic.claude-opus-4-7")
        != bedrock._norm_model_key("anthropic.claude-opus-4-8"), True)


def main() -> None:
    for fn in (test_extract_code_block, test_inspect_code, test_check_spec, test_err,
               test_degradation_is_announced, test_keyword_recall,
               test_norm_model_key):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all shared-helper regression tests pass")


if __name__ == "__main__":
    main()
