#!/bin/bash
# Convert jupytext .py sources -> .ipynb, then EXECUTE every cell in place.
# Fails loudly on the first cell error so nothing ships untested.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY="${PY:-python3}"
ROOT="${REPO:-$(cd "$HERE/../.." && pwd)}"   # the sample repository
SRC="${SRC:-$HERE/src}"
# AWS credentials come from the default chain; export AWS_PROFILE to pick a profile.
export PYDEVD_DISABLE_FILE_VALIDATION=1

target="${1:-}"
rc_all=0
for src in $(ls "$SRC"/*.py | sort); do
  stem=$(basename "$src" .py)          # e.g. 00-foundations__01-endpoints
  folder="${stem%%__*}"
  name="${stem#*__}"
  [ -n "$target" ] && [[ "$stem" != *"$target"* ]] && continue
  out="$ROOT/$folder/$name.ipynb"
  mkdir -p "$ROOT/$folder"
  echo "=============================================================="
  echo ">>> $folder/$name.ipynb"
  $PY -m jupytext --to ipynb "$src" -o "$out" --quiet 2>&1 | tail -2
  start=$(date +%s)
  $PY -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=900 \
      --ExecutePreprocessor.kernel_name=python3 \
      --ExecutePreprocessor.allow_errors=False \
      "$out" > "/tmp/nbout_$stem.log" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then
    ncells=$($PY -c "import nbformat,sys;nb=nbformat.read('$out',as_version=4);print(sum(1 for c in nb.cells if c.cell_type=='code'))")
    echo "    PASS  ${dur}s  ${ncells} code cells executed"
  else
    echo "    FAIL  ${dur}s  --- error tail ---"
    grep -A 18 "Error\|Traceback\|CellExecutionError" "/tmp/nbout_$stem.log" | head -32
    rc_all=1
  fi
done
echo "=============================================================="
[ $rc_all -eq 0 ] && echo "ALL EXECUTED CLEAN" || echo "SOME FAILURES — see above"
exit $rc_all
