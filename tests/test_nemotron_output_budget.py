"""Regression checks for the notebook's documented limits; no inference calls."""

import json
from pathlib import Path
import unittest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "10-nvidia-nemotron"
    / "01-nemotron-nano-and-super.ipynb"
)


class NemotronOutputBudgetTest(unittest.TestCase):
    def setUp(self):
        self.cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        self.intro = "".join(self.cells[0]["source"])

    def test_table_keeps_context_windows_and_card_output_maxima_distinct(self):
        expected = {
            "Nemotron 3 Super 120B": ("262,144", "32K"),
            "Nemotron 3 Nano 30B": ("262,144", "8K"),
            "Nemotron Nano 12B v2": ("131,072", "8K"),
            "Nemotron Nano 9B v2": ("131,072", "8K"),
        }
        for model, limits in expected.items():
            with self.subTest(model=model):
                row = next(
                    line
                    for line in self.intro.splitlines()
                    if line.startswith(f"| [{model}]")
                )
                self.assertEqual(
                    tuple(value.strip() for value in row.split("|")[4:6]), limits
                )


if __name__ == "__main__":
    unittest.main()
