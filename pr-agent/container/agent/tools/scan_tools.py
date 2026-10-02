"""The security gates a change must clear before a pull request exists.

Bandit, Semgrep, Checkov and ASH. Notebooks are not Python files, so code cells are
extracted to a scratch tree first, with a comment mapping each file back to the
notebook and cell it came from -- otherwise a finding names a path that does not exist
and nobody can act on it.

One rule is enforced here rather than trusted to the prompt: `scan_all` reports
`gate_passed` as a computed boolean over the parsed findings. The agent cannot conclude
"scans are green" from a summary it wrote itself.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Any

from claude_agent_sdk import tool

log = logging.getLogger("pr-agent.scan")
BLOCKING = ("HIGH", "CRITICAL")


def _text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(payload)}]}


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # noqa: S603


SKIP_DIRS = {".git", ".ipynb_checkpoints", "__pycache__", ".venv", "venv",
             "node_modules", ".pytest_cache", "build", "dist"}


def _flatten(rel: Path, taken: set[str]) -> str:
    """A unique filename encoding the path it came from.

    Uniqueness is enforced rather than assumed: `a/b.py` and `a__b.py` both flatten to
    `a__b.py`, and silently overwriting one with the other would drop a file from the
    scan -- a finding that never appears rather than a finding that is wrong.
    """
    base = str(rel).replace("/", "__")
    if base.endswith(".ipynb"):
        base = base[: -len(".ipynb")] + ".py"
    name, suffix = base.rsplit(".", 1)
    candidate, n = base, 1
    while candidate in taken:
        candidate = f"{name}~{n}.{suffix}"
        n += 1
    taken.add(candidate)
    return candidate


def _relevant(path: Path, repo: Path) -> bool:
    rel = path.relative_to(repo)
    return not any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1])


def extract_scannable(repo: Path) -> Path:
    """Every notebook code cell and every .py file, flattened for the scanners.

    Walks the whole tree rather than globbing `*/*.ipynb` and `_shared/*.py`, which would
    skip root-level notebooks, notebooks more than one level deep, and every Python file
    outside `_shared/`. A scanner reports zero findings for a file it never saw, and a
    zero from a file that was never read is indistinguishable from a clean one.

    Each output file names the notebook and cell it came from, so a finding points at a
    location a human can open.
    """
    dest = Path(tempfile.mkdtemp(prefix="scan-extract-"))
    taken: set[str] = set()

    for nb_path in sorted(repo.rglob("*.ipynb")):
        if not _relevant(nb_path, repo):
            continue
        try:
            nb = json.loads(nb_path.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            log.warning("could not parse %s; it will not be scanned", nb_path)
            continue
        rel = nb_path.relative_to(repo)
        lines = [f"# extracted from {rel}\n"]
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            lines.append(f"\n# --- {rel} cell [{i}] ---\n")
            for line in "".join(cell.get("source", [])).split("\n"):
                stripped = line.lstrip()
                # IPython magics and shell escapes are not Python; comment them so the
                # scanners see valid source rather than a parse error. A parse error
                # means the file is skipped, which is another route to a false zero.
                if stripped.startswith(("!", "%")):
                    line = line.replace(stripped[0], f"# {stripped[0]}", 1)
                lines.append(line + "\n")
        (dest / _flatten(rel, taken)).write_text("".join(lines))

    for py in sorted(repo.rglob("*.py")):
        if not _relevant(py, repo):
            continue
        rel = py.relative_to(repo)
        try:
            body = py.read_text()
        except (OSError, UnicodeDecodeError):
            log.warning("could not read %s; it will not be scanned", py)
            continue
        (dest / _flatten(rel, taken)).write_text(f"# extracted from {rel}\n{body}")

    log.info("extracted %d file(s) from %s for scanning", len(taken), repo)
    return dest


# Kept as an alias: the name is referenced in the security-gate skill's explanation of
# how a finding maps back to a notebook.
extract_notebooks = extract_scannable


def _bandit(target: Path) -> dict[str, Any]:
    proc = _run(["bandit", "-r", str(target), "-f", "json", "-q"])
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"ran": False, "error": (proc.stderr or proc.stdout)[-400:]}
    sev: dict[str, int] = {}
    for issue in data.get("results", []):
        sev[issue["issue_severity"].upper()] = sev.get(issue["issue_severity"].upper(), 0) + 1
    return {
        "ran": True,
        "total": len(data.get("results", [])),
        "by_severity": sev,
        "blocking": sum(sev.get(s, 0) for s in BLOCKING),
        "detail": [
            f"{i['issue_severity']} {i['test_id']} {Path(i['filename']).name}:"
            f"{i['line_number']} {i['issue_text'][:110]}"
            for i in data.get("results", []) if i["issue_severity"].upper() in BLOCKING
        ][:25],
    }


def _semgrep(target: Path) -> dict[str, Any]:
    proc = _run([
        "semgrep", "--config", "p/security-audit", "--config", "p/python",
        "--config", "p/secrets", "--json", "--quiet", "--metrics=off", str(target),
    ])
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"ran": False, "error": (proc.stderr or proc.stdout)[-400:]}
    results = data.get("results", [])
    sev: dict[str, int] = {}
    for r in results:
        s = (r.get("extra", {}).get("severity") or "INFO").upper()
        sev[s] = sev.get(s, 0) + 1
    # Semgrep grades ERROR/WARNING/INFO; ERROR is the blocking tier.
    return {
        "ran": True,
        "total": len(results),
        "by_severity": sev,
        "blocking": sev.get("ERROR", 0),
        "detail": [
            f"{r['extra'].get('severity')} {r['check_id'].split('.')[-1][:50]} "
            f"{Path(r['path']).name}:{r['start']['line']}"
            for r in results if (r.get("extra", {}).get("severity") or "").upper() == "ERROR"
        ][:25],
    }


def _checkov(target: Path) -> dict[str, Any]:
    proc = _run([
        "checkov", "-d", str(target), "--framework", "secrets",
        "--compact", "--quiet", "-o", "json",
    ])
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"ran": False, "error": (proc.stderr or proc.stdout)[-400:]}
    if isinstance(data, list):
        data = data[0] if data else {}
    failed = data.get("results", {}).get("failed_checks", [])
    return {
        "ran": True,
        "total": len(failed),
        "blocking": len(failed),  # a leaked secret is always blocking
        "detail": [
            f"{c.get('check_id')} {c.get('file_path')}:{c.get('file_line_range')}"
            for c in failed
        ][:25],
    }


def _ash(repo: Path) -> dict[str, Any]:
    if not shutil.which("ash"):
        return {"ran": False, "error": "ash not installed in this image"}
    out = Path(tempfile.mkdtemp(prefix="ash-"))
    proc = _run([
        "ash", "--source-dir", str(repo), "--output-dir", str(out),
        "--mode", "local", "--strategy", "sequential",
    ], timeout=3600)
    agg = out / "ash_aggregated_results.json"
    if not agg.exists():
        return {"ran": False, "error": (proc.stdout or proc.stderr)[-500:]}
    data = json.loads(agg.read_text())
    results = data.get("scanner_results", {})
    actionable = sum(v.get("actionable_finding_count", 0) for v in results.values())
    ran = [k for k, v in results.items() if v.get("status") != "MISSING"]
    return {
        "ran": True,
        "scanners": sorted(ran),
        "total": sum(v.get("finding_count", 0) for v in results.values()),
        "blocking": actionable,
        "detail": [
            f"{k}: {v.get('actionable_finding_count')} actionable ({v.get('status')})"
            for k, v in results.items() if v.get("actionable_finding_count", 0)
        ],
    }


SCANNERS = {"bandit": _bandit, "semgrep": _semgrep, "checkov": _checkov}

# The executables this module shells out to. Declared here, next to the code that calls
# them, and read by verify_image.py at build time -- so adding a scanner below without
# adding it to the image fails the build instead of failing silently at 3am with a zero
# finding count from a command that does not exist.
REQUIRED_COMMANDS: tuple[str, ...] = (*SCANNERS, "ash")


def compute_gate(report: dict[str, Any]) -> dict[str, Any]:
    """The verdict, computed from the parsed findings. Pure, so it can be tested.

    Separate from `scan_all` on purpose. This is the single decision that stands between
    the agent and a pull request, and the failure that matters is not a wrong finding
    count -- it is a scanner that never started returning an empty result which then
    reads as zero. So "did not run" is disqualifying here, at the same level as a
    high-severity finding, and `tests/test_guards.py` plants exactly that case.
    """
    scanners = {k: v for k, v in report.items() if isinstance(v, dict)}
    # The expectation comes from REQUIRED_COMMANDS, not from the report. Reading the
    # expected set out of the report itself would make an empty report pass -- there
    # would be nothing in it to disagree with. `tests/test_guards.py` plants that case.
    absent = [c for c in REQUIRED_COMMANDS if c not in scanners]
    did_not_run = sorted(
        [k for k, v in scanners.items() if not v.get("ran")] + absent
    )
    blocking = {k: v["blocking"] for k, v in scanners.items() if v.get("blocking")}
    out = dict(report)
    out["scanners_expected"] = list(REQUIRED_COMMANDS)
    out["did_not_run"] = did_not_run
    out["blocking_totals"] = blocking
    out["gate_passed"] = not did_not_run and not blocking
    if did_not_run:
        out["gate_note"] = (
            f"{did_not_run} did not run, so this is not a pass. A zero from a scanner "
            "that never started is not evidence. Fix the scanner, or say plainly in the "
            "pull request which scanner did not run and why."
            + (f" Never reported at all: {absent}." if absent else "")
        )
    return out


def _count(target: Path) -> int:
    return len([p for p in target.iterdir() if p.is_file()])


def _with_scope(result: dict[str, Any], target: Path) -> dict[str, Any]:
    """Attach how many files the scanner actually looked at, and refuse an empty sweep.

    "0 findings" and "0 files scanned" are the same JSON unless the count is in it. That
    ambiguity has already produced one false pass in this project -- a report reading
    "0 findings across 0 notebooks" against a path that did not exist. So the count is
    always reported, and an empty tree is `ran: False`: scanning nothing is not scanning.
    """
    n = _count(target)
    result = dict(result, files_scanned=n)
    if n == 0:
        return {
            "ran": False,
            "files_scanned": 0,
            "error": "nothing was extracted from the repository, so this scanner looked "
                     "at zero files. That is not a clean result -- check that repo_path "
                     "points at the clone and that it contains .ipynb or .py files.",
        }
    return result


@tool(
    "scan_run",
    "Run ONE scanner and return parsed findings. scanner is bandit, semgrep, checkov or "
    "ash. repo_path is the clone directory. Every notebook code cell and every .py file "
    "in the tree is extracted first, so findings point at the notebook and cell they came "
    "from. `total` counts all findings; `blocking` counts only high and critical; "
    "`detail` lists ONLY the blocking ones, so an empty `detail` beside a non-zero "
    "`total` means every finding was low or medium and is expected, not a bug. "
    "`files_scanned` says how many files the scanner looked at -- read it, because "
    "0 findings across 0 files is not a clean result and is reported as ran: false. "
    "Use this to check a specific fix quickly; use scan_all for the gate.",
    {
        "scanner": Annotated[str, "bandit | semgrep | checkov | ash"],
        "repo_path": Annotated[str, "Path to the cloned repository."],
    },
)
async def scan_run(args: dict[str, Any]) -> dict[str, Any]:
    name = (args.get("scanner") or "").strip().lower()
    repo = Path(args.get("repo_path") or os.environ.get("AGENT_WORKDIR", "/app/work"))
    if not repo.is_dir():
        return _text(f"ERROR: {repo} is not a directory")
    try:
        if name == "ash":
            return _text(json.dumps(_ash(repo), indent=1))
        if name not in SCANNERS:
            return _text(f"ERROR: unknown scanner {name!r}; use {sorted(SCANNERS)} or ash")
        extracted = extract_scannable(repo)
        try:
            return _text(json.dumps(_with_scope(SCANNERS[name](extracted), extracted), indent=1))
        finally:
            shutil.rmtree(extracted, ignore_errors=True)
    except subprocess.TimeoutExpired:
        return _text(f"ERROR: {name} timed out")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "scan_all",
    "Run every scanner and return a computed gate verdict. `gate_passed` is true only "
    "when every scanner RAN and none reported a high or critical finding -- a scanner "
    "that failed to run is NOT a pass, because a zero from a tool that never started is "
    "not evidence. This is the gate: do not open a pull request until it passes.",
    {"repo_path": Annotated[str, "Path to the cloned repository."]},
)
async def scan_all(args: dict[str, Any]) -> dict[str, Any]:
    repo = Path(args.get("repo_path") or os.environ.get("AGENT_WORKDIR", "/app/work"))
    if not repo.is_dir():
        return _text(f"ERROR: {repo} is not a directory")
    report: dict[str, Any] = {}
    extracted = extract_scannable(repo)
    try:
        for name, fn in SCANNERS.items():
            try:
                report[name] = _with_scope(fn(extracted), extracted)
            except Exception as exc:  # noqa: BLE001
                report[name] = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            report["ash"] = _ash(repo)
        except Exception as exc:  # noqa: BLE001
            report["ash"] = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(extracted, ignore_errors=True)

    return _text(json.dumps(compute_gate(report), indent=1))
