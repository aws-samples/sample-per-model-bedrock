#!/usr/bin/env python3
"""Execute notebooks in place, from their own directory, and report what failed.

    python3 run-notebooks.py [pattern ...] [--strict] [--timeout N]

Each notebook runs with its own directory as cwd, because every one of them does
`sys.path.insert(0, "../_shared")`. Outputs are written back in place.

By default errors are captured rather than raised, so one run surfaces every broken
cell instead of stopping at the first. `--strict` fails on the first error, which is
what the final verification pass wants.
"""
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def error_cells(nb) -> list[tuple[int, str]]:
    found = []
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "error":
                found.append((i, f"{out.get('ename')}: {str(out.get('evalue'))[:150]}"))
    return found


def run(path: str, timeout: int, strict: bool) -> dict:
    rel = os.path.relpath(path, REPO)
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": os.path.dirname(path)}},
        allow_errors=not strict,
    )
    started = time.perf_counter()
    failure = None
    try:
        client.execute()
    except CellExecutionError as exc:
        failure = str(exc)[:300]
    except CellTimeoutError as exc:
        # A timeout is NOT a CellExecutionError, so without this clause it
        # propagates and aborts the whole sweep - which it did, leaving 11
        # notebooks untested while the summary line never printed.
        failure = f"TIMEOUT after {timeout}s: {str(exc)[:200]}"
    finally:
        nbformat.write(nb, path)
    elapsed = time.perf_counter() - started
    errors = error_cells(nb)
    status = "FAIL" if (errors or failure) else "ok"
    print(f"{status:4} {elapsed:7.1f}s  {rel}")
    for i, message in errors:
        print(f"         cell [{i}] {message}")
    if failure and not errors:
        print(f"         {failure}")
    sys.stdout.flush()
    return {"path": rel, "seconds": round(elapsed, 1),
            "errors": errors, "failure": failure}


def main() -> None:
    strict = "--strict" in sys.argv
    timeout = 900
    argv = sys.argv[1:]
    if "--timeout" in argv:
        at = argv.index("--timeout")
        timeout = int(argv[at + 1])
        # Drop the flag AND its value: leaving the value behind makes it look like
        # a glob pattern, which matches nothing and silently runs zero notebooks.
        del argv[at:at + 2]
    argv = [a for a in argv if not a.startswith("--")]

    import glob
    if argv:
        paths = []
        for pattern in argv:
            paths.extend(sorted(glob.glob(os.path.join(REPO, pattern))))
    else:
        paths = sorted(glob.glob(os.path.join(REPO, "*/*.ipynb")))

    print(f"executing {len(paths)} notebook(s), cell timeout {timeout}s, "
          f"strict={strict}\n")
    results = [run(p, timeout, strict) for p in paths]

    failed = [r for r in results if r["errors"] or r["failure"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} clean, "
          f"{sum(r['seconds'] for r in results) / 60:.1f} min total")
    dest = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")), "quality", "findings", "17-run-report.json")
    with open(dest, "w") as fh:
        json.dump(results, fh, indent=1)
    if failed:
        print("\nfailed:")
        for r in failed:
            print(f"  {r['path']}: {len(r['errors'])} error cell(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
