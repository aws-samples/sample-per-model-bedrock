#!/usr/bin/env python3
"""Run every scanner over the repository and print one comparable summary.

    python3 scan-all.py [--skip ash]

Notebooks are not Python files, so the code cells are extracted to a scratch tree
first. The extraction keeps a line-for-line mapping comment at the top of each file
so a finding can be traced back to a cell.

Scanners: Bandit, Semgrep, Checkov (secrets), detect-secrets, pip-audit, and AWS ASH.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
WORK = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "scan")
OUT = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings")


VENV_BIN = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), ".venv", "bin")
# Every scanner that did not run, and why. A scanner invoked by bare name that is not
# on PATH exits 127, its stdout is empty, and `json.loads(r.stdout or "{}")` then
# reports ZERO FINDINGS -- a clean bill of health from a tool that never started.
# detect-secrets did exactly that: it lives only in the venv, so "0 potential secrets
# in 0 file(s)" was printed for months without the binary ever being invoked.
UNAVAILABLE: dict[str, str] = {}


def tool(name: str) -> str | None:
    """Absolute path to a scanner, venv first, or None."""
    candidate = os.path.join(VENV_BIN, name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(name)
    return found


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def run_json(name: str, argv: str, cwd: str | None = None) -> dict | list | None:
    """Run a scanner and return its parsed JSON, or None if it could not run.

    None is deliberately distinct from an empty result. The caller must report
    "did not run" rather than "no findings".
    """
    binary = tool(name)
    if binary is None:
        UNAVAILABLE[name] = "not installed (checked the venv and PATH)"
        return None
    r = sh(f"{'cd ' + cwd + ' && ' if cwd else ''}{binary} {argv}")
    if r.returncode == 127:
        UNAVAILABLE[name] = "exit 127 -- not executable"
        return None
    try:
        return json.loads(r.stdout or "")
    except json.JSONDecodeError:
        # Findings-present exit codes are normal; unparseable output is not.
        UNAVAILABLE[name] = (f"exit {r.returncode}, output did not parse as JSON: "
                            f"{(r.stderr or r.stdout or '')[-300:].strip()}")
        return None


def extract() -> int:
    """Write each notebook's code cells to WORK/extracted/<name>.py."""
    dest = f"{WORK}/extracted"
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    count = 0
    for path in sorted(glob.glob(f"{REPO}/*/*.ipynb")):
        nb = json.load(open(path))
        rel = os.path.relpath(path, REPO)
        lines = [f"# extracted from {rel}\n"]
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            lines.append(f"\n# --- cell [{i}] ---\n")
            body = "".join(cell["source"])
            # IPython magics and shell escapes are not Python; comment them out so
            # the scanners see valid source rather than a parse error.
            for line in body.split("\n"):
                stripped = line.lstrip()
                if stripped.startswith(("!", "%")):
                    line = line.replace(stripped[0], f"# {stripped[0]}", 1)
                lines.append(line + "\n")
        open(f"{dest}/{rel.replace('/', '__')[:-6]}.py", "w").writelines(lines)
        count += 1
    shutil.copy(f"{REPO}/_shared/bedrock.py", f"{dest}/_shared__bedrock.py")
    return count


def main() -> None:
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    n = extract()
    src = f"{WORK}/extracted"
    print(f"extracted code from {n} notebooks + _shared/bedrock.py\n")
    summary = {}

    # --- Bandit -----------------------------------------------------------
    data = run_json("bandit", f"-r {src} -f json -q")
    data = {} if data is None else data
    sev = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for issue in data.get("results", []):
        sev[issue["issue_severity"]] = sev.get(issue["issue_severity"], 0) + 1
    loc = data.get("metrics", {}).get("_totals", {}).get("loc", 0)
    summary["bandit"] = {"total": len(data.get("results", [])), **sev, "loc": loc}
    print(f"bandit          {len(data.get('results', []))} findings "
          f"(high {sev['HIGH']}, medium {sev['MEDIUM']}, low {sev['LOW']}) "
          f"over {loc} LOC")
    for issue in data.get("results", [])[:20]:
        print(f"   {issue['issue_severity']:6} {issue['test_id']} "
              f"{os.path.basename(issue['filename'])}:{issue['line_number']} "
              f"{issue['issue_text'][:80]}")
    json.dump(data, open(f"{OUT}/18-bandit.json", "w"), indent=1)

    # --- Semgrep ----------------------------------------------------------
    data = run_json("semgrep", "--config p/security-audit --config p/python "
                    f"--config p/secrets --json --quiet --metrics=off {src}")
    data = {} if data is None else data
    results = data.get("results", [])
    summary["semgrep"] = {"total": len(results)}
    print(f"\nsemgrep         {len(results)} findings")
    for f in results[:20]:
        print(f"   {f['extra'].get('severity', '?'):8} {f['check_id'].split('.')[-1][:44]} "
              f"{os.path.basename(f['path'])}:{f['start']['line']}")
    json.dump(data, open(f"{OUT}/19-semgrep.json", "w"), indent=1)

    # --- Checkov (secrets framework over the real repo) --------------------
    data = run_json("checkov",
                    f"-d {REPO} --framework secrets --compact --quiet -o json")
    data = {} if data is None else data
    if isinstance(data, list):
        data = data[0] if data else {}
    failed = data.get("results", {}).get("failed_checks", [])
    summary["checkov"] = {"failed": len(failed)}
    print(f"\ncheckov         {len(failed)} failed secret checks")
    for c in failed[:20]:
        print(f"   {c.get('check_id')} {c.get('file_path')}:{c.get('file_line_range')}")
    json.dump(data, open(f"{OUT}/20-checkov.json", "w"), indent=1)

    # --- detect-secrets ---------------------------------------------------
    data = run_json("detect-secrets", "scan --all-files", cwd=REPO)
    data = {} if data is None else data
    secrets = data.get("results", {})
    total = sum(len(v) for v in secrets.values())
    summary["detect_secrets"] = {"total": total, "files": len(secrets)}
    print(f"\ndetect-secrets  {total} potential secrets in {len(secrets)} file(s)")
    for path, hits in list(secrets.items())[:20]:
        for h in hits:
            print(f"   {path}:{h.get('line_number')} {h.get('type')}")
    json.dump(data, open(f"{OUT}/21-detect-secrets.json", "w"), indent=1)

    # --- pip-audit --------------------------------------------------------
    data = run_json("pip-audit",
                    f"-r {REPO}/requirements.txt --format json "
                    "--progress-spinner off")
    data = {} if data is None else data
    vulns = [d for d in data.get("dependencies", []) if d.get("vulns")]
    summary["pip_audit"] = {"vulnerable": len(vulns)}
    print(f"\npip-audit       {len(vulns)} vulnerable dependencies")
    for d in vulns:
        print(f"   {d['name']} {d.get('version')}: "
              f"{[v.get('id') for v in d.get('vulns', [])]}")
    json.dump(data, open(f"{OUT}/22-pip-audit.json", "w"), indent=1)

    # --- AWS ASH ----------------------------------------------------------
    if "ash" not in sys.argv:
        print("\nash (AWS Automated Security Helper)")
        ash_bin = tool("ash")
        if ash_bin is None:
            UNAVAILABLE["ash"] = "not installed (checked the venv and PATH)"
            print("   DID NOT RUN -- binary not found. This is NOT a clean result.")
        else:
            ash_out = f"{WORK}/ash"
            shutil.rmtree(ash_out, ignore_errors=True)
            r = sh(f"{ash_bin} --source-dir {REPO} --output-dir {ash_out} "
                   f"--mode local --strategy sequential 2>&1 | tail -40")
            print("   " + "\n   ".join((r.stdout or "").strip().splitlines()[-18:]))
            agg = f"{ash_out}/reports/ash.summary.txt"
            if os.path.exists(agg):
                shutil.copy(agg, f"{OUT}/23-ash-summary.txt")
            else:
                UNAVAILABLE["ash"] = "ran but produced no summary report"

    summary["did_not_run"] = UNAVAILABLE
    json.dump(summary, open(f"{OUT}/24-scan-summary.json", "w"), indent=1)
    print("\n--- summary ---")
    print(json.dumps(summary, indent=1))
    if UNAVAILABLE:
        print(f"\n!! {len(UNAVAILABLE)} scanner(s) DID NOT RUN. A zero from a scanner")
        print("   that never started is not evidence of anything:")
        for name, why in UNAVAILABLE.items():
            print(f"     {name}: {why}")
        sys.exit(1)
    print("\nevery scanner ran, and each result above is a real measurement")


if __name__ == "__main__":
    main()
