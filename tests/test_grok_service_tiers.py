"""Execute the service-tier cell with the pinned SDK and mocked HTTP only."""

import contextlib
import io
import json
from pathlib import Path
import unittest

import httpx
from openai import APIStatusError, OpenAI

NOTEBOOK = Path(__file__).resolve().parents[1] / "11-xai-grok" / "01-grok-4-3.ipynb"


class GrokServiceTiersTest(unittest.TestCase):
    def setUp(self):
        self.cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]

    def tier_cell(self):
        cells = [cell for cell in self.cells if cell.get("id") == "grok-service-tiers-code"]
        self.assertEqual(len(cells), 1, "The service-tier example must exist exactly once")
        return cells[0]

    def run_cell(
        self,
        *,
        answer="Canberra",
        status="completed",
        served=None,
        omit_tier=False,
        null_tier=False,
        http_status=200,
    ):
        cell = self.tier_cell()
        requests, output = [], io.StringIO()

        def handle(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/openai/v1/responses")
            body = json.loads(request.content)
            requests.append(body)
            if http_status != 200:
                return httpx.Response(http_status, json={"error": {"message": "mock API failure"}})
            response = {
                "id": "resp-test",
                "object": "response",
                "created_at": 0,
                "model": "xai.grok-4.3",
                "status": status,
                "output": [
                    {
                        "type": "message",
                        "id": "msg-test",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": answer, "annotations": []}],
                    }
                ],
            }
            if null_tier:
                response["service_tier"] = None
            elif not omit_tier:
                response["service_tier"] = served or (
                    "default" if body["service_tier"] == "auto" else body["service_tier"]
                )
            return httpx.Response(200, json=response)

        with httpx.Client(transport=httpx.MockTransport(handle)) as http_client, OpenAI(
            api_key="test-token",
            base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
            http_client=http_client,
            max_retries=0,
        ) as client, contextlib.redirect_stdout(output):
            # Execute only this checked-in cell, never caller-provided code.
            exec(  # nosec B102  # noqa: S102
                compile("".join(cell["source"]), f"{NOTEBOOK.name}:service-tiers", "exec"),
                {"client": client, "GROK": "xai.grok-4.3"},
            )
        return requests, output.getvalue()

    def test_four_bounded_requests_use_the_model_endpoint_and_disable_storage(self):
        requests, output = self.run_cell()
        self.assertEqual(
            [request["service_tier"] for request in requests],
            ["default", "flex", "priority", "auto"],
        )
        for request in requests:
            self.assertEqual(request["model"], "xai.grok-4.3")
            self.assertEqual(request["reasoning"], {"effort": "none"})
            self.assertEqual(request["max_output_tokens"], 200)
            self.assertFalse(request["store"])
            self.assertIn("capital of Australia", request["input"])
        self.assertEqual(output.count("says Canberra: True"), 4)
        auto = next(line for line in output.splitlines() if line.startswith("asked auto"))
        self.assertIn("served default", auto)

    def test_prints_the_served_tier_not_the_requested_one(self):
        _, output = self.run_cell(served="priority")
        self.assertEqual(output.count("served priority"), 4)

    def test_missing_served_tier_is_not_invented(self):
        for options in ({"omit_tier": True}, {"null_tier": True}):
            with self.subTest(options=options):
                _, output = self.run_cell(**options)
                self.assertEqual(output.count("served not reported"), 4)

    def test_answer_check_accepts_case_and_a_final_period(self):
        for answer in ("CANBERRA", " Canberra. "):
            with self.subTest(answer=answer):
                _, output = self.run_cell(answer=answer)
                self.assertEqual(output.count("says Canberra: True"), 4)

    def test_incomplete_and_failed_outputs_are_not_treated_as_answers(self):
        for status in ("incomplete", "failed"):
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.run_cell(status=status)

    def test_wrong_or_empty_answer_fails_the_known_answer_check(self):
        for answer in ("Sydney", "", "Not Canberra"):
            with self.subTest(answer=answer), self.assertRaisesRegex(RuntimeError, "Canberra"):
                self.run_cell(answer=answer)

    def test_api_errors_are_not_hidden(self):
        for status in (400, 401, 429, 500):
            with self.subTest(status=status):
                with self.assertRaises(APIStatusError) as caught:
                    self.run_cell(http_status=status)
                self.assertEqual(caught.exception.status_code, status)

    def test_guidance_distinguishes_documentation_from_dated_observations(self):
        markdown = "\n".join(
            "".join(cell["source"]) for cell in self.cells if cell["cell_type"] == "markdown"
        )
        for text in (
            "## 10. Service tiers",
            "1.75",
            "0.5",
            "22 October 2026",
            "issues/427",
            "ultrafast",
        ):
            self.assertIn(text, markdown)
        self.assertIn("| Service tiers |", markdown)

    def test_new_cell_has_no_fabricated_execution_evidence(self):
        cell = self.tier_cell()
        self.assertIsNone(cell["execution_count"])
        self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
