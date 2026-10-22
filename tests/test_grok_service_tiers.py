"""Execute the service-tier cell with the pinned SDK and mocked HTTP only."""

import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx
from openai import APIStatusError, OpenAI

NOTEBOOK = Path(__file__).resolve().parents[1] / "11-xai-grok" / "01-grok-4-3.ipynb"
TIERS = ("default", "flex", "priority", "auto", "reserved", "ultrafast")
REFUSALS = {
    "reserved": ("invalid_value", "Invalid value: 'reserved'."),
    "ultrafast": (
        "validation_error",
        "'ultrafast' is not supported for 'service_tier' on this model",
    ),
}


class GrokServiceTiersTest(unittest.TestCase):
    def setUp(self):
        self.cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]

    def tier_cell(self):
        cells = [
            cell for cell in self.cells if cell.get("id") == "grok-service-tiers-code"
        ]
        self.assertEqual(
            len(cells), 1, "The service-tier example must exist exactly once"
        )
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
        error_tier="default",
        refusals=None,
    ):
        cell = self.tier_cell()
        setup = next(
            cell
            for cell in self.cells
            if cell["cell_type"] == "code"
            and "client = OpenAI(" in "".join(cell["source"])
        )
        requests, output = [], io.StringIO()
        namespace = {}
        refusals = REFUSALS if refusals is None else refusals

        def handle(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/openai/v1/responses")
            body = json.loads(request.content)
            requests.append(body)
            tier = body["service_tier"]
            if http_status != 200 and tier == error_tier:
                return httpx.Response(
                    http_status, json={"error": {"message": "mock API failure"}}
                )
            if tier in refusals:
                code, message = refusals[tier]
                return httpx.Response(
                    400, json={"error": {"code": code, "message": message}}
                )
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
                        "content": [
                            {"type": "output_text", "text": answer, "annotations": []}
                        ],
                    }
                ],
            }
            if null_tier:
                response["service_tier"] = None
            elif not omit_tier:
                response["service_tier"] = served or (
                    "default"
                    if body["service_tier"] == "auto"
                    else body["service_tier"]
                )
            return httpx.Response(200, json=response)

        with contextlib.ExitStack() as stack:

            def client(**kwargs):
                kwargs["http_client"] = stack.enter_context(
                    httpx.Client(transport=httpx.MockTransport(handle))
                )
                kwargs["max_retries"] = 0
                return stack.enter_context(OpenAI(**kwargs))

            token_provider = stack.enter_context(
                patch(
                    "aws_bedrock_token_generator.provide_token",
                    return_value="test-token",
                )
            )
            stack.enter_context(patch("openai.OpenAI", side_effect=client))
            stack.enter_context(
                patch("sys.path", [str(NOTEBOOK.parents[1] / "_shared"), *sys.path])
            )
            stack.enter_context(contextlib.redirect_stdout(output))
            for checked_in_cell in (setup, cell):
                # Execute only these fixed, checked-in cells, never caller-provided code.
                exec(  # nosec B102  # noqa: S102
                    compile(
                        "".join(checked_in_cell["source"]),
                        f"{NOTEBOOK.name}:service-tiers",
                        "exec",
                    ),
                    namespace,
                )
            token_provider.assert_called_once_with(region="us-east-1")
        return requests, output.getvalue(), namespace["refusals"]

    def test_six_bounded_requests_use_the_model_endpoint_and_disable_storage(self):
        requests, output, _ = self.run_cell()
        self.assertEqual(
            [request["service_tier"] for request in requests],
            list(TIERS),
        )
        for request in requests:
            self.assertEqual(request["model"], "xai.grok-4.3")
            self.assertEqual(request["reasoning"], {"effort": "none"})
            self.assertEqual(request["max_output_tokens"], 200)
            self.assertFalse(request["store"])
            self.assertIn("capital of Australia", request["input"])
        self.assertEqual(output.count("says Canberra: True"), 4)
        auto = next(
            line for line in output.splitlines() if line.startswith("asked auto")
        )
        self.assertIn("served default", auto)

    def test_prints_the_served_tier_not_the_requested_one(self):
        _, output, _ = self.run_cell(served="priority")
        self.assertEqual(output.count("served priority"), 4)

    def test_missing_served_tier_is_not_invented(self):
        for options in ({"omit_tier": True}, {"null_tier": True}):
            with self.subTest(options=options):
                _, output, _ = self.run_cell(**options)
                self.assertEqual(output.count("served not reported"), 4)

    def test_answer_check_accepts_case_and_a_final_period(self):
        for answer in ("CANBERRA", " Canberra. "):
            with self.subTest(answer=answer):
                _, output, _ = self.run_cell(answer=answer)
                self.assertEqual(output.count("says Canberra: True"), 4)

    def test_incomplete_and_failed_outputs_are_not_treated_as_answers(self):
        for status in ("incomplete", "failed"):
            with self.subTest(status=status), self.assertRaisesRegex(
                RuntimeError, "incomplete"
            ):
                self.run_cell(status=status)

    def test_wrong_or_empty_answer_fails_the_known_answer_check(self):
        for answer in ("Sydney", "", "Not Canberra"):
            with self.subTest(answer=answer), self.assertRaisesRegex(
                RuntimeError, "Canberra"
            ):
                self.run_cell(answer=answer)

    def test_api_errors_are_not_hidden(self):
        for status in (401, 429, 500):
            for tier in TIERS:
                with self.subTest(status=status, tier=tier):
                    with self.assertRaises(APIStatusError) as caught:
                        self.run_cell(http_status=status, error_tier=tier)
                    self.assertEqual(caught.exception.status_code, status)

    def test_a_400_for_any_tier_is_recorded_and_the_sweep_continues(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                code, message = "invalid_value", f"Tier {tier} refused for this account"
                requests, output, recorded = self.run_cell(
                    refusals={tier: (code, message)}
                )
                self.assertEqual(
                    [request["service_tier"] for request in requests], list(TIERS)
                )
                self.assertEqual(recorded, [(tier, message)])
                self.assertIn(f"asked {tier:8} refused: {code}", output)
                self.assertIn(f"{tier}: {message}", output)
                self.assertEqual(output.count("says Canberra: True"), len(TIERS) - 1)

    def test_reserved_and_ultrafast_refusals_keep_the_service_details(self):
        requests, output, recorded = self.run_cell()
        self.assertEqual([request["service_tier"] for request in requests], list(TIERS))
        self.assertEqual(
            recorded, [(tier, message) for tier, (_, message) in REFUSALS.items()]
        )
        for tier, (code, message) in REFUSALS.items():
            self.assertIn(f"asked {tier:8} refused: {code}", output)
            self.assertIn(f"{tier}: {message}", output)

    def test_newly_available_tiers_are_checked_as_successful_responses(self):
        requests, output, recorded = self.run_cell(refusals={})
        self.assertEqual(len(requests), len(TIERS))
        self.assertEqual(recorded, [])
        self.assertEqual(output.count("says Canberra: True"), len(TIERS))


if __name__ == "__main__":
    unittest.main()
