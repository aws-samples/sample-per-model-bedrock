#!/bin/bash
# Execute EVERY cell of EVERY notebook against the live endpoint.
# Fails loudly on the first cell error so nothing ships untested.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY="${PY:-python3}"
ROOT="${REPO:-$(cd "$HERE/../.." && pwd)}"   # the sample repository
SRC="${SRC:-$HERE/src}"
# AWS credentials come from the default chain; export AWS_PROFILE to pick a profile.
export PYDEVD_DISABLE_FILE_VALIDATION=1
LOG="${LOG:-${TMPDIR:-/tmp}/notebook-run.log}"
: > "$LOG"

target="${1:-}"
rc_all=0; passed=0; failed=0
for src in $(ls "$SRC"/*.py | sort); do
  stem=$(basename "$src" .py)
  folder="${stem%%__*}"; name="${stem#*__}"
  [ -n "$target" ] && [[ "$stem" != *"$target"* ]] && continue
  out="$ROOT/$folder/$name.ipynb"
  mkdir -p "$ROOT/$folder"
  echo "=== $folder/$name.ipynb" | tee -a "$LOG"
  $PY -m jupytext --to ipynb "$src" -o "$out" --quiet 2>&1 | tail -1
  start=$(date +%s)
  $PY -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=900 \
      --ExecutePreprocessor.kernel_name=python3 \
      --ExecutePreprocessor.allow_errors=False \
      "$out" > "/tmp/nbout_$stem.log" 2>&1
  rc=$?; dur=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then
    n=$($PY -c "import nbformat;nb=nbformat.read('$out',as_version=4);print(sum(1 for c in nb.cells if c.cell_type=='code'))")
    echo "    PASS  ${dur}s  ${n} cells" | tee -a "$LOG"; passed=$((passed+1))
  else
    echo "    FAIL  ${dur}s" | tee -a "$LOG"
    grep -A14 "Error\|Traceback\|CellExecutionError" "/tmp/nbout_$stem.log" | head -26 | tee -a "$LOG"
    rc_all=1; failed=$((failed+1))
  fi
done
echo "=== passed=$passed failed=$failed" | tee -a "$LOG"
exit $rc_all
