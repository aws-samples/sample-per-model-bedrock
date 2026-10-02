#!/usr/bin/env python3
"""Add the standard AWS Content disclaimer, as requested in the open-source review.

Two placements:

  README.md          the full disclaimer verbatim, in its own `## Disclaimer`
                     section, immediately before `## Security`.
  every notebook     a short blockquote under the H1 that carries the substance
                     and links to the canonical text above.

Markdown only. No code cell is touched, which is what lets this skip the
33-notebook re-execution cycle — see the hash proof in the sibling verify script.

Idempotent: running twice changes nothing.

Usage: add-aws-content-disclaimer.py <repo-or-sample-folder>
"""

import glob
import os
import sys

import nbformat

# Verbatim as supplied in the open-source review, Aug 2026. Rewrapped for the
# README's column width; wording is untouched and must stay that way.
README_SECTION = """## Disclaimer

The sample code; software libraries; command line tools; proofs of concept;
templates; or other related technology (including any of the foregoing that are
provided by our personnel) is provided to you as AWS Content under the AWS Customer
Agreement, or the relevant written agreement between you and AWS (whichever applies).
You should not use this AWS Content in your production accounts, or on production or
other critical data. You are responsible for testing, securing, and optimizing the
AWS Content, such as sample code, as appropriate for production grade use based on
your specific quality control practices and standards. Deploying AWS Content may
incur AWS charges for creating or using AWS chargeable resources, such as running
Amazon EC2 instances or using Amazon S3 storage.

"""

NOTEBOOK_BLOCKQUOTE = """> **Sample code — not for production.** Provided as AWS Content under the AWS
> Customer Agreement; do not use it in production accounts or on production or other
> critical data. Running these cells calls Amazon Bedrock and incurs charges. Full
> disclaimer in the [README](../README.md#disclaimer).
"""

ANCHOR = "## Security"
MARKER = "provided to you as AWS Content under the AWS Customer"
NB_MARKER = "**Sample code — not for production.**"


def patch_readme(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if MARKER in text:
        return "already present"
    if ANCHOR not in text:
        raise SystemExit(f"{path}: no {ANCHOR!r} anchor to insert before")
    text = text.replace(ANCHOR, README_SECTION + ANCHOR, 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return "added"


def patch_notebook(path: str) -> str:
    nb = nbformat.read(path, as_version=4)
    cell = nb.cells[0]
    if cell.cell_type != "markdown":
        raise SystemExit(f"{path}: first cell is {cell.cell_type}, not markdown")
    if NB_MARKER in cell.source:
        return "already present"

    lines = cell.source.split("\n")
    heading = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), None)
    if heading is None:
        raise SystemExit(f"{path}: no H1 in the first cell")

    # Sit directly under the title, before the intro prose.
    tail = lines[heading + 1 :]
    while tail and not tail[0].strip():
        tail.pop(0)
    cell.source = "\n".join(
        lines[: heading + 1] + ["", NOTEBOOK_BLOCKQUOTE.rstrip("\n"), ""] + tail
    )
    nbformat.write(nb, path)
    return "added"


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    os.chdir(root)

    print(f"== {root}")
    print(f"   README.md: {patch_readme('README.md')}")

    for path in sorted(glob.glob("*/*.ipynb")):
        print(f"   {path}: {patch_notebook(path)}")


if __name__ == "__main__":
    main()
