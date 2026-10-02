#!/usr/bin/env bash
# Run every check this repository has, in the order that fails cheapest first.
#
# The point is that one command answers "is the repo sound?". Before this existed the
# checks were run ad hoc and by memory, which is how a stale capability rule survived
# in 13 places and a helper that returned an ID Converse rejects survived in a
# committed table.
#
#   ./verify-all.sh          static checks only (seconds, no service calls)
#   ./verify-all.sh --live   also probe the live service (minutes, costs money)
#
# Exits non-zero if any check fails, and says which.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$(cd "$HERE/../.." && pwd)/.venv/bin/python3}"
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
# export, not just assign: the checks below run in child processes that read
# os.environ["REPO"], and a bare shell assignment is not visible to them.
export REPO
LIVE=0
[ "${1:-}" = "--live" ] && LIVE=1

# Refuse to run against a repo that is not there. Without this, every glob-based
# check reports "0 findings across 0 notebooks" and the suite prints ALL CHECKS PASSED
# for a path that does not exist -- 13 of the 17 checks did exactly that when REPO was
# pointed at /tmp/nonexistent. A gate that passes vacuously is worse than no gate, and
# it is the same false-zero that had detect-secrets reporting a clean bill of health
# without the binary ever being invoked.
EXPECTED_NOTEBOOKS=34
if [ ! -d "$REPO" ]; then
  printf 'REFUSING TO RUN: REPO does not exist: %s\n' "$REPO"
  exit 2
fi
found=$(find "$REPO" -mindepth 2 -maxdepth 2 -name '*.ipynb' | wc -l | tr -d ' ')
if [ "$found" -lt "$EXPECTED_NOTEBOOKS" ]; then
  printf 'REFUSING TO RUN: found %s notebook(s) under %s, expected at least %s.\n' \
    "$found" "$REPO" "$EXPECTED_NOTEBOOKS"
  printf 'A suite that scans nothing reports success. Fix REPO, or lower\n'
  printf 'EXPECTED_NOTEBOOKS deliberately if the collection really shrank.\n'
  exit 2
fi
printf 'repo: %s (%s notebooks)\n' "$REPO" "$found"

failed=()
run() {
  local label="$1"; shift
  printf '\n=== %s\n' "$label"
  if "$@"; then
    printf '    PASS  %s\n' "$label"
  else
    printf '    FAIL  %s\n' "$label"
    failed+=("$label")
  fi
}

# ---- static: no network, no cost -------------------------------------------
run "notebook JSON + nbformat + cell ids" "$PY" - <<'PY'
import glob, json, os, sys, warnings
warnings.simplefilter("ignore")
import nbformat
repo = os.environ["REPO"]  # exported by verify-all.sh; no stale fallback
bad = 0
for f in sorted(glob.glob(os.path.join(repo, "*", "*.ipynb"))):
    nb = json.load(open(f))
    nbformat.validate(nbformat.read(f, as_version=4))
    missing = [i for i, c in enumerate(nb["cells"]) if "id" not in c]
    if missing:
        print(f"  {os.path.basename(f)}: {len(missing)} cell(s) without an id")
        bad += 1
print(f"  {len(glob.glob(os.path.join(repo, '*', '*.ipynb')))} notebooks checked")
sys.exit(1 if bad else 0)
PY

run "every code cell parses, no invalid escapes" "$PY" - <<'PY'
import glob, json, os, sys, warnings
repo = os.environ["REPO"]  # exported by verify-all.sh; no stale fallback
bad = 0
for f in sorted(glob.glob(os.path.join(repo, "*", "*.ipynb"))):
    for i, c in enumerate(json.load(open(f))["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith(("!", "%")):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                compile(src, "<cell>", "exec")
            except (SyntaxError, SyntaxWarning) as exc:
                print(f"  {os.path.basename(f)} cell {i}: {exc}")
                bad += 1
sys.exit(1 if bad else 0)
PY

run "shared-helper regressions"         "$PY" "$HERE/test-shared-helpers.py"
run "names used but never bound"        "$PY" "$HERE/check-dangling-names.py"
run "claim-extractor self-test"         "$PY" "$HERE/extract-claims.py" --self-test
run "evidence-audit self-test"          "$PY" "$HERE/evidence-audit.py" --self-test
run "endpoint/ID matcher self-test"     "$PY" "$HERE/check-endpoint-id-match.py" --self-test
run "model ID matches its endpoint"     "$PY" "$HERE/check-endpoint-id-match.py"
run "cross-reference checker self-test" "$PY" "$HERE/check-crossrefs.py" --self-test
run "notebook cross-references"        "$PY" "$HERE/check-crossrefs.py"
run "truncation checker self-test"     "$PY" "$HERE/check-truncation.py" --self-test
run "truncation reported as a result"  "$PY" "$HERE/check-truncation.py"
run "checklist can actually fail"      "$PY" "$HERE/test-checklist-derivation.py"
run "output-verifier self-test"         "$PY" "$HERE/verify-outputs.py" --self-test
run "committed output invariants"       "$PY" "$HERE/verify-outputs.py"

run "no error outputs anywhere" "$PY" - <<'PY'
import glob, json, os, sys
repo = os.environ["REPO"]  # exported by verify-all.sh; no stale fallback
n = 0
for f in sorted(glob.glob(os.path.join(repo, "*", "*.ipynb"))):
    for i, c in enumerate(json.load(open(f))["cells"]):
        for o in c.get("outputs") or []:
            if o.get("output_type") == "error":
                print(f"  {os.path.basename(f)} cell {i}: {o.get('ename')}")
                n += 1
print(f"  {n} error output(s)")
sys.exit(1 if n else 0)
PY

run "no synthetic media generators left" bash -c '
  ! grep -rl "def make_png\|def make_wav\|def two_band_png\|bands_png" \
      "$REPO" --include="*.ipynb" --include="*.py" --exclude-dir=harness --exclude-dir=pr-agent 2>/dev/null'

# ---- live: real service calls, real money ----------------------------------
if [ "$LIVE" -eq 1 ]; then
  run "README claims vs live catalogues"  "$PY" "$HERE/verify-readme.py"
  run "runtime model-ID mapping"          "$PY" "$HERE/check-id-mapping.py"
  run "hardcoded capability rules"        "$PY" "$HERE/verify-live-claims.py"
  run "availability claims"               "$PY" "$HERE/verify-region-claims.py"
  run "endpoint catalogue refresh"        "$PY" "$HERE/check-endpoint-id-match.py" --live
else
  printf '\n=== live checks skipped (pass --live to include them)\n'
fi

printf '\n%s\n' "----------------------------------------------------------------"
if [ ${#failed[@]} -eq 0 ]; then
  printf 'ALL CHECKS PASSED\n'
  exit 0
fi
printf 'FAILED: %d\n' "${#failed[@]}"
for f in "${failed[@]}"; do printf '  - %s\n' "$f"; done
exit 1
