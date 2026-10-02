"""Surgical notebook edits that fail loudly rather than silently missing.

Every fix in this audit is a targeted replacement, and the failure mode that
matters is a replacement that quietly matches nothing — that is how the stale
claims survived three earlier rounds. So `sub` asserts the expected hit count.

    from nbedit import Notebook
    nb = Notebook("path.ipynb")
    nb.sub("old text", "new text", count=1)          # anywhere
    nb.sub_cell(8, "old", "new")                     # one cell
    nb.set_source(8, "...")                          # replace a cell wholesale
    nb.save()
"""
from __future__ import annotations

import hashlib
import json


class Notebook:
    def __init__(self, path: str):
        self.path = path
        with open(path) as fh:
            self.nb = json.load(fh)
        self.changes = 0

    # -- reading ------------------------------------------------------------
    def source(self, index: int) -> str:
        return "".join(self.nb["cells"][index]["source"])

    def find(self, needle: str) -> list[int]:
        return [i for i, c in enumerate(self.nb["cells"])
                if needle in "".join(c["source"])]

    # -- writing ------------------------------------------------------------
    def set_source(self, index: int, text: str, *, drops_ok: bool = False) -> None:
        """Replace a cell wholesale.

        Guard: if the cell being replaced contains markdown headings that the
        replacement does not, this silently deletes a section. That has now
        happened twice in this audit — once to a class definition sharing a cell
        with a demo, once to a "## Takeaways" section sharing a cell with "## 7".
        Both times the notebook still ran, so nothing caught it. Pass
        `drops_ok=True` only when you have checked and mean it.
        """
        # Markdown cells only: a `#` at the start of a line in a CODE cell is a
        # comment, not a heading, and treating it as one makes the guard cry wolf.
        if self.nb["cells"][index]["cell_type"] == "markdown":
            old = self.source(index)
            lost = [h for h in _headings(old) if h not in _headings(text)]
        else:
            lost = []
        if lost and not drops_ok:
            raise AssertionError(
                f"{self.path} cell {index}: replacement drops heading(s) {lost}. "
                f"Split the cell, extend the replacement, or pass drops_ok=True.")
        self.nb["cells"][index]["source"] = _splitlines(text)
        self.changes += 1

    def sub(self, old: str, new: str, count: int = 1) -> None:
        """Replace `old` with `new` across all cells; assert it hit `count` cells."""
        hits = 0
        for cell in self.nb["cells"]:
            text = "".join(cell["source"])
            if old in text:
                hits += text.count(old)
                cell["source"] = _splitlines(text.replace(old, new))
        if hits != count:
            raise AssertionError(
                f"{self.path}: expected {count} occurrence(s) of {old[:70]!r}, found {hits}")
        self.changes += hits

    def sub_idempotent(self, old: str, new: str, count: int = 1) -> bool:
        """Like `sub`, but a no-op if `new` is already in place.

        These fix scripts get re-run after a partial failure, so an edit that has
        already landed must not look like a missing target.
        """
        if any(new in "".join(c["source"]) for c in self.nb["cells"]):
            return False
        self.sub(old, new, count)
        return True

    def sub_cell(self, index: int, old: str, new: str) -> None:
        """Targeted replacement inside one cell.

        The set_source heading guard does not apply: this replaces a specific
        substring, so it cannot silently drop the rest of the cell. Renaming a
        heading through here is normal and must not trip the guard.
        """
        text = self.source(index)
        if old not in text:
            raise AssertionError(
                f"{self.path} cell {index}: {old[:70]!r} not found")
        self.set_source(index, text.replace(old, new), drops_ok=True)

    def insert_after(self, index: int, cell_type: str, text: str) -> int:
        """Insert a new cell after `index`. Returns the new cell's index.

        nbformat >= 4.5 requires a unique `id` per cell, and every existing cell
        here has one, so derive a stable 8-hex id from the content rather than
        leaving it to Jupyter to backfill on first save.
        """
        cell_id = hashlib.sha256(
            f"{self.path}:{index}:{text}".encode()).hexdigest()[:8]
        cell: dict = {"cell_type": cell_type, "id": cell_id, "metadata": {},
                      "source": _splitlines(text)}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        self.nb["cells"].insert(index + 1, cell)
        self.changes += 1
        return index + 1

    def delete(self, index: int) -> None:
        del self.nb["cells"][index]
        self.changes += 1

    def save(self) -> None:
        # nbformat writes a trailing newline; match it so git sees a minimal diff.
        with open(self.path, "w") as fh:
            json.dump(self.nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")


def _headings(text: str) -> list[str]:
    """Markdown headings in a cell, used by the set_source guard."""
    return [ln.strip() for ln in text.split("\n") if ln.startswith("#")]


def _splitlines(text: str) -> list[str]:
    """Notebook `source` is a list of lines, each keeping its newline but the last."""
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
