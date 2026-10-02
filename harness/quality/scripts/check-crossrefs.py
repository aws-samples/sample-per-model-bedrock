#!/usr/bin/env python3
"""Every notebook path a notebook mentions must exist.

Three notebooks pointed at files that were never there:

    03-production-hardening-checklist.ipynb            -> 03-production-hardening.ipynb
    02-governance-projects-retention-and-observability -> 02-governance-projects-and-retention

They read as plausible names, which is exactly why nobody noticed. In a collection
whose navigation is "see `../04-qwen/01` §6", a dead cross-reference is a dead end for
the reader and there is nothing in a notebook run that would catch it.

Also checks the section anchors used in prose (`§6`, `section 7`) against the number of
`## N.` headings in the target notebook, because a renumbered section is the same class
of rot.

    python3 check-crossrefs.py
    python3 check-crossrefs.py --self-test
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

# `../04-qwen/01-qwen3-core-and-tools.ipynb`, `13-amazon-nova/01`, bare filenames.
NB_REF = re.compile(r"(?:\.\./)?([0-9]{2}-[a-z0-9-]+/[0-9]{2}-[a-z0-9.-]+\.ipynb)")
BARE_NB = re.compile(r"(?<![\w/-])([0-9]{2}-[a-z0-9.-]+\.ipynb)")
# `§6`, `§13`, `section 7`
SECTION = re.compile(r"(?:§|section\s+)([0-9]{1,2})\b", re.I)
# `../04-qwen/01` §6  -- a directory-and-index reference with a section
DIR_SECTION = re.compile(
    r"(?:\.\./)?([0-9]{2}-[a-z0-9-]+)/([0-9]{2})[^\n§]{0,60}?§([0-9]{1,2})")


def notebooks() -> set[str]:
    return {os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "*/*.ipynb"))}


def section_count(rel: str) -> int:
    """Highest `## N.` heading number in a notebook."""
    path = os.path.join(REPO, rel)
    if not os.path.exists(path):
        return 0
    highest = 0
    for cell in json.load(open(path))["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for m in re.finditer(r"^##\s+([0-9]{1,2})[.)]", "".join(cell["source"]), re.M):
            highest = max(highest, int(m.group(1)))
    return highest


def check(path: str, known: set[str]) -> list[str]:
    rel = os.path.relpath(path, REPO)
    own_dir = os.path.dirname(rel)
    basenames = {os.path.basename(k): k for k in known}
    problems = []
    nb = json.load(open(path))
    for i, cell in enumerate(nb["cells"]):
        text = "".join(cell["source"])
        for ref in set(NB_REF.findall(text)):
            if ref not in known:
                problems.append(f"cell {i}: points at {ref!r}, which does not exist")
        for ref in set(BARE_NB.findall(text)):
            # A bare filename means "in my own directory", falling back to anywhere.
            if os.path.join(own_dir, ref) in known or ref in basenames:
                continue
            problems.append(f"cell {i}: points at {ref!r}, which does not exist")
        # A `dir/NN ... §M` reference: does the target have a section M?
        for directory, index, section in set(DIR_SECTION.findall(text)):
            target = next((k for k in known
                           if k.startswith(f"{directory}/{index}-")), None)
            if target is None:
                problems.append(
                    f"cell {i}: points at {directory}/{index} §{section}, and no "
                    f"notebook {index}- exists in {directory}")
                continue
            highest = section_count(target)
            if highest and int(section) > highest:
                problems.append(
                    f"cell {i}: points at {target} §{section}, but that notebook's "
                    f"last section is {highest}")
    return sorted(set(problems))


SELF_TEST = [
    # (text, known set, expected finding count)
    ("See `03-production-hardening-checklist.ipynb` for the rest.",
     {"99-cross-cutting/03-production-hardening.ipynb"}, 1),
    ("See `03-production-hardening.ipynb` for the rest.",
     {"99-cross-cutting/03-production-hardening.ipynb"}, 0),
    ("Covered in `../04-qwen/01-qwen3-core-and-tools.ipynb`.",
     {"04-qwen/01-qwen3-core-and-tools.ipynb"}, 0),
    ("Covered in `../04-qwen/09-nonexistent.ipynb`.",
     {"04-qwen/01-qwen3-core-and-tools.ipynb"}, 1),
    ("No notebook reference at all, just prose about §6.",
     {"04-qwen/01-qwen3-core-and-tools.ipynb"}, 0),
]


def self_test() -> int:
    bad = 0
    import tempfile
    for text, known, want in SELF_TEST:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False,
                                          dir=REPO)
        json.dump({"cells": [{"cell_type": "markdown", "id": "a",
                              "source": [text]}],
                   "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, tmp)
        tmp.close()
        try:
            got = check(tmp.name, known)
        finally:
            os.remove(tmp.name)
        if len(got) != want:
            print(f"  want {want}, got {len(got)}: {got}")
            print(f"    in: {text[:70]!r}")
            bad += 1
    print(f"  {len(SELF_TEST)} cases, {bad} problem(s)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(1 if self_test() else 0)

    known = notebooks()
    total = 0
    for path in sorted(glob.glob(os.path.join(REPO, "*/*.ipynb"))):
        found = check(path, known)
        if found:
            total += len(found)
            print(f"\n{os.path.relpath(path, REPO)}")
            for line in found:
                print(f"  {line}")
    print(f"\n{total} dead cross-reference(s) across {len(known)} notebook(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
