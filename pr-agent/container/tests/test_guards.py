#!/usr/bin/env python3
"""Tests for the guards that make an autonomous agent safe to leave running.

These are not coverage tests. Each one plants the specific mistake the guard exists to
catch, and fails if the guard lets it through. That shape is deliberate: this project's
review history is full of checks that passed because the evidence and the claim came
from the same place, so a test that merely calls a function and asserts it returned
something is worse than no test -- it manufactures confidence.

Read each test as: "if someone deleted this guard tomorrow, would this test go red?"
If the answer is no, the test is not pulling its weight and should be rewritten.

Run directly (`python tests/test_guards.py`) or under pytest. No network, no AWS.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.tools.github_tools import (  # noqa: E402
    BRANCH_PREFIX,
    UnsafeOperation,
    _assert_safe_branch,
)
from agent.tools.scan_tools import compute_gate, extract_notebooks  # noqa: E402
from agent.tools.source_tools import RefusedSource, _check_url, feed_recent  # noqa: E402

FAILURES: list[str] = []


def expect(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL  {message}")


def expect_raises(exc_type: type[Exception], fn, message: str) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{message} (raised {type(exc).__name__} instead)")
        print(f"  FAIL  {message} (raised {type(exc).__name__} instead)")
        return
    FAILURES.append(f"{message} (did not raise)")
    print(f"  FAIL  {message} (did not raise)")


# --------------------------------------------------------------- branch containment
def test_branch_containment() -> None:
    """The agent must not be able to push over the trunk of a public AWS repository.

    Everything here is a branch name a confused or prompt-injected agent could plausibly
    produce. The guard has to refuse all of them.
    """
    for bad in (
        "main",
        "master",
        "origin/main",
        "HEAD",
        "release/2026-09",
        "bot/main",                        # right owner, wrong prefix
        "bot/bedrock-update",              # prefix without the trailing dash
        "notbot/bedrock-update-x",         # prefix present but not at the start
        "../main",
        "bot/bedrock-update-x/..",         # traversal out of the prefix
        "",
        "  ",
    ):
        expect_raises(
            UnsafeOperation,
            lambda b=bad: _assert_safe_branch(b),
            f"branch guard accepted {bad!r}",
        )

    # ...and it must still allow the branch the agent actually needs, or the guard is
    # useless in the other direction and every cycle fails.
    for good in (f"{BRANCH_PREFIX}claude-fable-5-2", f"{BRANCH_PREFIX}nova-3-regions"):
        try:
            _assert_safe_branch(good)
        except UnsafeOperation as exc:
            expect(False, f"branch guard refused a legitimate branch {good!r}: {exc}")


# -------------------------------------------------------------------- the scan gate
def test_gate_requires_scanners_to_have_run() -> None:
    """A zero from a scanner that never started is the defect this gate exists for.

    This exact failure shipped once in this project: `detect-secrets` was not on PATH,
    exited 127, its empty output parsed as zero findings, and a scan report said clean.
    """
    clean = {"bandit": {"ran": True, "blocking": 0},
             "semgrep": {"ran": True, "blocking": 0},
             "checkov": {"ran": True, "blocking": 0},
             "ash": {"ran": True, "blocking": 0}}
    expect(compute_gate(clean)["gate_passed"] is True,
           "gate failed a report where every scanner ran clean")

    # One scanner absent, and nothing else changed. This must NOT pass.
    silent = dict(clean, ash={"ran": False, "error": "ash not installed", "blocking": 0})
    verdict = compute_gate(silent)
    expect(verdict["gate_passed"] is False,
           "gate passed while a scanner did not run -- a false zero would ship")
    expect(verdict["did_not_run"] == ["ash"],
           f"gate did not name the scanner that failed: {verdict.get('did_not_run')}")
    expect("not a pass" in verdict.get("gate_note", "").lower(),
           "gate note does not say plainly that this is not a pass")

    # A blocking finding must fail even when every scanner ran.
    found = dict(clean, semgrep={"ran": True, "blocking": 1})
    expect(compute_gate(found)["gate_passed"] is False,
           "gate passed with a blocking finding present")

    # Both problems at once must still fail, and report both.
    both = dict(clean, ash={"ran": False, "blocking": 0}, bandit={"ran": True, "blocking": 3})
    v = compute_gate(both)
    expect(v["gate_passed"] is False, "gate passed with both a dead scanner and a finding")
    expect(v["blocking_totals"] == {"bandit": 3}, f"blocking totals wrong: {v['blocking_totals']}")

    # An empty report is not a pass. "No scanners ran" is the strongest possible
    # false zero, and the obvious `not did_not_run and not blocking` reading of an
    # empty dict is True -- so this is the case a naive implementation gets wrong.
    expect(compute_gate({})["gate_passed"] is False,
           "gate passed an empty report: no scanner ran at all")


# ------------------------------------------------------- source reading containment
def test_internal_sources_are_refused() -> None:
    """This agent writes to a public repository, so it must not read internal hosts."""
    os.environ["INTERNAL_HOST_SUFFIXES"] = "corp.example.com,wiki.example.net,.example.internal"
    for bad in (
        "https://wiki.example.net/bin/view/Bedrock",
        "https://some-team.corp.example.com/roadmap",
        "https://tools.example.internal/x",
        "http://aws.amazon.com/whats-new",       # plaintext
        "ftp://aws.amazon.com/x",
    ):
        expect_raises(RefusedSource, lambda u=bad: _check_url(u),
                      f"source guard accepted {bad!r}")

    for good in (
        "https://aws.amazon.com/about-aws/whats-new/recent/feed/",
        "https://docs.aws.amazon.com/bedrock/latest/userguide/doc-history.html",
        "https://repost.aws/tags/TA4IVCeWI1RJmzHrl0Nu0bYQ/",
        "https://www.aboutamazon.com/news/aws",
    ):
        try:
            _check_url(good)
        except RefusedSource as exc:
            expect(False, f"source guard refused a legitimate public source {good!r}: {exc}")


def _feed(dates: list[str]) -> str:
    items = "".join(
        f"<item><title>Item {d}</title><link>https://example.invalid/{d}</link>"
        f"<pubDate>{datetime.fromisoformat(d).strftime('%a, %d %b %Y 00:00:00 +0000')}"
        f"</pubDate></item>"
        for d in dates
    )
    return f"<?xml version='1.0'?><rss version='2.0'><channel>{items}</channel></rss>"


def test_feed_coverage_is_not_assumed(monkeypatch_target=None) -> None:
    """An empty feed result is only evidence if the feed reaches back far enough.

    A feed holds a fixed number of entries. Ask What's New for 30 days and it answers
    with the 100 it has, which may span 10. Reporting "0 new" from that is a false
    negative, and on a discovery step a false negative is invisible.
    """
    import agent.tools.source_tools as st

    today = datetime.now(timezone.utc).date()
    def days_ago(n: int) -> str:
        return (today - timedelta(days=n)).isoformat()

    # Feed reaching back 20 days, asked for 10 -> the window IS covered.
    served = _feed([days_ago(n) for n in (2, 8, 14, 20)])
    original = st._get
    st._get = lambda url, timeout=60, limit=0: (served, False, len(served))
    try:
        r = feed_recent("https://aws.amazon.com/feed", days_ago(10))
        expect(r.get("window_covered") is True,
               f"claimed the window was not covered when the feed predates it: {r.get('oldest_entry_in_feed')}")
        expect(r["matched"] == 2, f"matched {r['matched']} entries within 10 days, expected 2")
        expect(r.get("coverage_note") == "",
               "emitted a coverage warning for a window that is covered")

        # Feed reaching back only 3 days, asked for 30 -> NOT covered, and it must say so.
        served = _feed([days_ago(n) for n in (1, 2, 3)])
        r = feed_recent("https://aws.amazon.com/feed", days_ago(30))
        expect(r.get("window_covered") is False,
               "claimed a 30-day window was covered by a feed holding 3 days")
        expect("does NOT mean nothing was announced" in r.get("coverage_note", ""),
               "did not warn that an empty result here is not evidence")

        # Boundary: an entry dated exactly at the cutoff is inside the window. Off by one
        # here silently drops the oldest day every single run.
        served = _feed([days_ago(5)])
        r = feed_recent("https://aws.amazon.com/feed", days_ago(5))
        expect(r["matched"] == 1,
               "dropped an entry dated exactly on the cutoff (off-by-one in the window)")

        # A truncated feed must be an error, never a partial parse -- a partial parse
        # silently loses whatever was cut off.
        #
        # The body served here is deliberately COMPLETE and valid XML while the
        # truncation flag is set. Serving a half-cut body instead would let the XML
        # parser produce the error, and the test would pass with the truncation check
        # deleted -- which is exactly what it did on the first attempt. Only the
        # truncation guard can catch this one, and the assertion names the byte limit
        # so a ParseError cannot satisfy it either.
        st._get = lambda url, timeout=60, limit=0: (served, True, 99_999_999)
        r = feed_recent("https://aws.amazon.com/feed", days_ago(5))
        expect("error" in r and "cut off" in r.get("error", ""),
               f"a truncated feed was parsed as if complete: {str(r)[:120]}")

        # Something that is not XML at all -- a captive portal or an error page.
        st._get = lambda url, timeout=60, limit=0: ("<html>Access Denied</html>", False, 26)
        r = feed_recent("https://aws.amazon.com/feed", days_ago(5))
        expect("error" in r and r.get("matched") is None,
               "reported entries from a response that is not a feed")
    finally:
        st._get = original


# ------------------------------------------------------------- notebook extraction
def test_notebook_extraction_keeps_provenance(tmp: Path | None = None) -> None:
    """A finding that names a path nobody can open is a finding nobody acts on."""
    import json
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="test-nb-"))
    (root / "01-section").mkdir()
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# heading\n"]},
            {"cell_type": "code", "source": ["import os\n", "!pip install boto3\n",
                                             "%matplotlib inline\n", "x = 1\n"]},
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    (root / "01-section" / "demo.ipynb").write_text(json.dumps(nb))

    out = extract_notebooks(root)
    files = sorted(p.name for p in out.glob("*.py"))
    expect(files == ["01-section__demo.py"], f"unexpected extraction output: {files}")
    text = (out / "01-section__demo.py").read_text()
    expect("01-section/demo.ipynb" in text,
           "extracted file does not name the notebook it came from")
    expect("cell [1]" in text, "extracted file does not name the cell it came from")
    expect("# !pip install" in text and "# %matplotlib" in text,
           "shell escapes and magics were left as-is, so the file is not valid Python")
    expect("# heading" not in text, "markdown was extracted as if it were code")

    # The extracted file must actually compile, or the scanners see a parse error and
    # report nothing -- another route to a false zero.
    try:
        compile(text, "extracted", "exec")
    except SyntaxError as exc:
        expect(False, f"extracted notebook is not valid Python: {exc}")


def test_scanner_actually_finds_a_planted_finding() -> None:
    """The scanner must find a known vulnerability in every file shape the repo uses.

    This is the test whose absence let a real defect ship. `test_gate_...` above proves
    the gate fails when a scanner does not run — but a scanner that runs and reports zero
    is the same false zero wearing different clothes, and nothing checked for it.

    The first version of `extract_scannable` globbed `*/*.ipynb` and `_shared/*.py`. It
    silently skipped root-level notebooks, notebooks more than one level deep, and every
    Python file outside `_shared/` — including `99-cross-cutting/capabilities.py`, which
    is a real file in the repository. Bandit reported zero for it because it never saw it.

    So: plant a HIGH-severity finding in each shape and require it to be found in each.
    The expectation is written out longhand rather than computed from the extractor, so
    the two cannot agree by construction.

    The planted snippet passes a **variable** to `shell=True`, which matters. Measured
    with bandit 1.8.6 on 2026-09-08:

        subprocess.call("ls -l", shell=True)   B602  LOW    (literal argument)
        subprocess.call(cmd, shell=True)       B602  HIGH   (variable argument)
        os.system(cmd)                         B605  HIGH

    The first version of this test used the literal form and asserted it was HIGH, which
    it is not, so the test failed for a reason that had nothing to do with the defect it
    was written to catch.
    """
    import json
    import shutil as sh
    import tempfile

    from agent.tools.scan_tools import _bandit, extract_scannable

    if not sh.which("bandit"):
        expect(False, "bandit is not on PATH, so this test could not run -- reported as a "
                      "failure rather than a skip, because a silently skipped security "
                      "test is the problem this file exists to prevent")
        return

    VULN = (
        "import os\n"
        "import subprocess\n"
        'cmd = os.environ["PLANTED_BY_TEST"]\n'
        "subprocess.call(cmd, shell=True)\n"
    )
    nb = {"cells": [{"cell_type": "code", "source": VULN.splitlines(keepends=True)}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

    shapes = {
        "notebook one level deep":   ("01-family/demo.ipynb", json.dumps(nb)),
        "notebook at the root":      ("root.ipynb", json.dumps(nb)),
        "notebook three levels deep": ("a/b/c/deep.ipynb", json.dumps(nb)),
        "py in _shared":             ("_shared/helper.py", VULN),
        "py outside _shared":        ("99-cross-cutting/capabilities.py", VULN),
        "py at the root":            ("toplevel.py", VULN),
    }

    for label, (rel, content) in shapes.items():
        repo = Path(tempfile.mkdtemp(prefix="test-scan-"))
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        extracted = extract_scannable(repo)
        try:
            result = _bandit(extracted)
            if not result.get("ran"):
                expect(False, f"{label}: bandit did not run ({result.get('error')})")
                continue
            expect(
                result.get("blocking", 0) >= 1,
                f"{label} ({rel}): bandit found {result.get('total')} finding(s), "
                f"{result.get('blocking')} blocking, for a file containing a HIGH-severity "
                "shell injection. The file was almost certainly never scanned.",
            )
            expect(
                any(rel.split("/")[-1].replace(".ipynb", "") in d for d in result.get("detail", [])),
                f"{label}: the finding does not name the file it came from, so nobody can "
                f"act on it. detail={result.get('detail')}",
            )
        finally:
            sh.rmtree(repo, ignore_errors=True)
            sh.rmtree(extracted, ignore_errors=True)

    # A clean tree must still come back clean, or the check above passes for the wrong
    # reason -- a scanner that flags everything is not a working scanner.
    repo = Path(tempfile.mkdtemp(prefix="test-scan-clean-"))
    (repo / "01-family").mkdir()
    (repo / "01-family" / "ok.py").write_text("import json\nprint(json.dumps({'a': 1}))\n")
    extracted = extract_scannable(repo)
    try:
        clean = _bandit(extracted)
        expect(clean.get("ran") is True, f"bandit did not run on the clean tree: {clean}")
        expect(clean.get("blocking", 0) == 0,
               f"bandit reported {clean.get('blocking')} blocking finding(s) on harmless "
               f"code, so a positive result proves nothing: {clean.get('detail')}")
    finally:
        sh.rmtree(repo, ignore_errors=True)
        sh.rmtree(extracted, ignore_errors=True)


def test_flattened_names_do_not_collide() -> None:
    """Two different files must not flatten to the same scan filename.

    `extract_scannable` encodes a path into a filename by replacing `/` with `__`, so
    `a/b.py` and `a__b.py` both become `a__b.py`. Whichever is written second wins and
    the other is never scanned — a file silently missing from the sweep, which is the
    same false zero as a scanner that never ran, just harder to notice.

    This test existed only after a sabotage run showed the uniqueness logic was asserted
    in a docstring and checked by nothing.
    """
    import shutil as sh
    import tempfile

    from agent.tools.scan_tools import extract_scannable

    repo = Path(tempfile.mkdtemp(prefix="test-collide-"))
    try:
        (repo / "a").mkdir()
        (repo / "a" / "b.py").write_text("X = 'from a/b.py'\n")
        (repo / "a__b.py").write_text("X = 'from a__b.py'\n")
        # A notebook that flattens onto a .py name, which is the other way in.
        (repo / "c").mkdir()
        (repo / "c" / "d.ipynb").write_text(
            '{"cells":[{"cell_type":"code","source":["Y = 1\\n"]}],'
            '"metadata":{},"nbformat":4,"nbformat_minor":5}'
        )
        (repo / "c__d.py").write_text("Y = 2\n")

        out = extract_scannable(repo)
        try:
            produced = sorted(p.name for p in out.iterdir())
            expect(len(produced) == 4,
                   f"4 source files flattened to {len(produced)} output file(s), so at "
                   f"least one was silently dropped from the scan: {produced}")
            bodies = [p.read_text() for p in out.iterdir()]
            for marker in ("from a/b.py", "from a__b.py"):
                expect(any(marker in b for b in bodies),
                       f"the file containing {marker!r} never reached the scanner")
        finally:
            sh.rmtree(out, ignore_errors=True)
    finally:
        sh.rmtree(repo, ignore_errors=True)


def test_empty_sweep_is_not_a_clean_result() -> None:
    """0 findings across 0 files must not read the same as 0 findings across 35.

    Without a file count in the result those two are the same JSON, and this project has
    already shipped one false pass of exactly that shape: a report reading "0 findings
    across 0 notebooks" against a path that did not exist. The agent's own selftest
    flagged the ambiguity again on the real repository, which is why the count is now in
    every scanner result and an empty sweep is `ran: false`.
    """
    import shutil as sh
    import tempfile

    from agent.tools.scan_tools import _with_scope, compute_gate

    # A scanner that ran clean over real files: a pass, and it says how many.
    real = Path(tempfile.mkdtemp(prefix="test-scope-"))
    (real / "a.py").write_text("import json\n")
    (real / "b.py").write_text("import os\n")
    try:
        ok = _with_scope({"ran": True, "total": 0, "blocking": 0}, real)
        expect(ok["ran"] is True, f"a clean sweep over 2 files was marked as not run: {ok}")
        expect(ok["files_scanned"] == 2,
               f"files_scanned reported {ok.get('files_scanned')}, expected 2")
    finally:
        sh.rmtree(real, ignore_errors=True)

    # The same clean-looking result over nothing at all: not a pass.
    empty = Path(tempfile.mkdtemp(prefix="test-scope-empty-"))
    try:
        none = _with_scope({"ran": True, "total": 0, "blocking": 0}, empty)
        expect(none["ran"] is False,
               "a scanner that looked at zero files was reported as having run clean")
        expect(none["files_scanned"] == 0, f"files_scanned wrong: {none}")
        expect("zero files" in none.get("error", ""),
               f"the error does not say the scanner saw no files: {none.get('error')}")
        # And the gate must reject it, which is the consequence that actually matters.
        verdict = compute_gate({
            "bandit": none, "semgrep": none, "checkov": none, "ash": none,
        })
        expect(verdict["gate_passed"] is False,
               "the gate passed a report in which every scanner looked at zero files")
    finally:
        sh.rmtree(empty, ignore_errors=True)


# ------------------------------------------------------------------ orphan branches
def test_merged_branches_are_not_orphans() -> None:
    """Only a bot branch that never had a pull request is unfinished work.

    The repository keeps merged branches, so a check that compared bot branches with
    OPEN pull requests alone listed every merged one as unfinished, and told the agent
    to resume sixteen finished changes. Stub the GitHub API with one branch in each
    state and check that only the one with no pull request at all is reported.
    """
    import asyncio

    import agent.tools.github_tools as gt

    def pr(ref: str) -> dict:
        return {"number": 1, "head": {"ref": ref}, "title": ref, "user": {"login": "bot"},
                "updated_at": "2026-09-28T00:00:00Z"}

    def fake_api(method: str, path: str, *args, **kwargs):
        if "/pulls?state=open" in path:
            return [pr(f"{BRANCH_PREFIX}open-one")]
        if "/pulls/" in path and "/files" in path:
            return []
        if "/pulls?state=closed" in path:
            if "page=1" in path:
                return [pr(f"{BRANCH_PREFIX}merged-one"), pr(f"{BRANCH_PREFIX}closed-one")]
            return []
        if "/branches" in path:
            return [{"name": name, "commit": {"sha": "0123456789abcdef"}}
                    for name in ("main", f"{BRANCH_PREFIX}open-one", f"{BRANCH_PREFIX}merged-one",
                                 f"{BRANCH_PREFIX}closed-one", f"{BRANCH_PREFIX}orphan-one")]
        if "/commits/" in path:
            return {"commit": {"message": "unfinished change", "committer": {"date": "2026-09-28"}}}
        raise AssertionError(f"unexpected API call {method} {path}")

    original = gt._api
    gt._api = fake_api
    try:
        handler = getattr(gt.github_list_open_prs, "handler", gt.github_list_open_prs)
        text = json.dumps(asyncio.run(handler({})))
    finally:
        gt._api = original
    section = text.split("NO pull request", 1)[-1] if "NO pull request" in text else ""
    expect(f"{BRANCH_PREFIX}orphan-one" in section, "a branch with no pull request was not reported")
    for finished in ("merged-one", "closed-one", "open-one"):
        expect(f"{BRANCH_PREFIX}{finished}" not in section,
               f"the {finished} branch was reported as unfinished work")


def main() -> int:
    tests = [
        test_branch_containment,
        test_gate_requires_scanners_to_have_run,
        test_internal_sources_are_refused,
        test_feed_coverage_is_not_assumed,
        test_notebook_extraction_keeps_provenance,
        test_scanner_actually_finds_a_planted_finding,
        test_flattened_names_do_not_collide,
        test_empty_sweep_is_not_a_clean_result,
        test_merged_branches_are_not_orphans,
    ]
    for fn in tests:
        before = len(FAILURES)
        print(f"{fn.__name__}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            # A test that raises is a failed test, not a crashed run. Letting the
            # exception escape printed a traceback with no "failure(s)" line, so an
            # automated sabotage check reading for that line scored the sabotage as
            # undetected -- the harness hid a defect its own test had caught.
            import traceback as tb

            FAILURES.append(f"{fn.__name__} raised {type(exc).__name__}: {exc}")
            print(f"  FAIL  raised {type(exc).__name__}: {exc}")
            print("        " + "        ".join(tb.format_exc().splitlines(True)[-3:]))
        if len(FAILURES) == before:
            print("  ok")
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"\n{len(tests)} test groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
