"""Regression checks for the notebook's documented limits; no inference calls."""

import json
from pathlib import Path
import unittest

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "10-nvidia-nemotron" / "01-nemotron-nano-and-super.ipynb"
)


class NemotronOutputBudgetTest(unittest.TestCase):
    def setUp(self):
        self.cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        self.intro = "".join(self.cells[0]["source"])
        self.markdown = "\n".join(
            "".join(cell["source"]) for cell in self.cells if cell["cell_type"] == "markdown"
        )

    def test_table_keeps_the_four_models_windows_distinct(self):
        expected = {
            "Nemotron 3 Super 120B": "262,144",
            "Nemotron 3 Nano 30B": "262,144",
            "Nemotron Nano 12B v2": "131,072",
            "Nemotron Nano 9B v2": "131,072",
        }
        for model, window in expected.items():
            with self.subTest(model=model):
                row = next(
                    line for line in self.intro.splitlines() if line.startswith(f"| [{model}]")
                )
                self.assertEqual(row.split("|")[4].strip(), window)

    def test_correction_attributes_reported_measurements_and_separates_output_limit(self):
        self.assertIn("issues/424", self.intro)
        self.assertIn("22 October 2026", self.intro)
        self.assertIn("us-east-1", self.intro)
        self.assertIn("Max output tokens: 32K", self.intro)
        self.assertIn("200,000", self.intro)
        self.assertIn("not a guarantee", self.intro)
        self.assertNotIn("window measured here is half", self.intro)
        self.assertNotIn("window is twice Super", self.intro)

    def test_output_budget_row_does_not_diagnose_a_500_as_an_oversized_budget(self):
        row = next(
            line for line in self.markdown.splitlines() if line.startswith("| Output budget |")
        )
        self.assertIn("262,144", row)
        self.assertIn("131,072", row)
        self.assertIn("your prompt contains 148 characters", row)
        self.assertIn("500", row)
        self.assertNotIn("Read a 500 on a large", row)
        self.assertNotIn("Two limits, and on Super they disagree", row)
        self.assertNotIn("maximum context length is 131072", row)

    def test_reasoning_budget_respects_both_the_card_and_the_context_window(self):
        reasoning = next(
            "".join(cell["source"])
            for cell in self.cells
            if cell["cell_type"] == "markdown"
            and "".join(cell["source"]).startswith("## 4. Reasoning")
        )
        self.assertIn("output maximum", reasoning)
        self.assertIn("context window minus", reasoning)


if __name__ == "__main__":
    unittest.main()
