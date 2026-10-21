"""Endpoint discovery regression tests; no AWS credentials or model calls needed."""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_shared"))
import bedrock


class EndpointAvailabilityTest(unittest.TestCase):
    def setUp(self):
        bedrock._MANTLE_CATALOGUE_CACHE.clear()
        bedrock._MANTLE_CATALOGUE_FAILED.clear()
        bedrock._WARNED.clear()
        self.addCleanup(bedrock._MANTLE_CATALOGUE_CACHE.clear)
        self.addCleanup(bedrock._MANTLE_CATALOGUE_FAILED.clear)
        self.addCleanup(bedrock._WARNED.clear)
        self.runtime = self.enterContext(patch.object(bedrock, "runtime_models", return_value={}))

    def discover(self, entries, model="anthropic.claude-example", region="us-east-1"):
        output = io.StringIO()
        with patch.object(bedrock, "post", return_value=(200, {"data": entries})), contextlib.redirect_stdout(output):
            result = bedrock.endpoints_for(model, region)
        return result, output.getvalue()

    def test_available_model_is_reported_without_warning(self):
        result, output = self.discover([{"id": "anthropic.claude-example", "status": "available"}])
        self.assertEqual(result, {"mantle": True, "runtime": False})
        self.assertEqual(output, "")

    def test_unavailable_model_explains_account_retention_mode_without_hiding_runtime(self):
        self.runtime.return_value = {"example": {"id": "anthropic.claude-example"}}
        result, output = self.discover([{
            "id": "anthropic.claude-example", "status": "unavailable",
            "status_reason": "not available under the current retention mode",
            "data_retention": {"mode": "default", "allowed_modes": ["aws_review"]},
        }])
        self.assertEqual(result, {"mantle": False, "runtime": True})
        for text in ("unavailable", "default", "aws_review", "us-east-1", "not available under", "NOT because the model is absent"):
            self.assertIn(text, output)

    def test_missing_unknown_or_null_status_does_not_claim_availability(self):
        for status in (None, "loading", "UNAVAILABLE"):
            with self.subTest(status=status):
                bedrock._MANTLE_CATALOGUE_CACHE.clear()
                bedrock._WARNED.clear()
                result, output = self.discover([{"id": "anthropic.claude-example", "status": status}])
                self.assertFalse(result["mantle"])
                self.assertIn("catalogue", output)
        bedrock._MANTLE_CATALOGUE_CACHE.clear()
        result, _ = self.discover([{"id": "anthropic.claude-example"}])
        self.assertFalse(result["mantle"])

    def test_absent_model_does_not_emit_unavailable_warning(self):
        result, output = self.discover([])
        self.assertFalse(result["mantle"])
        self.assertEqual(output, "")

    def test_normalized_alias_can_match_an_available_variant(self):
        result, output = self.discover([
            {"id": "openai.gpt-oss-20b", "status": "unavailable"},
            {"id": "openai.gpt-oss-20b-1:0", "status": "available"},
        ], model="us.openai.gpt-oss-20b-1:0")
        self.assertTrue(result["mantle"])
        self.assertEqual(output, "")

    def test_warning_is_once_per_normalized_model_and_region(self):
        entries = [{"id": "openai.gpt-oss-20b", "status": "unavailable"}]
        _, first = self.discover(entries, model="openai.gpt-oss-20b")
        _, repeated = self.discover(entries, model="us.openai.gpt-oss-20b-1:0")
        _, other_region = self.discover(entries, model="openai.gpt-oss-20b", region="us-west-2")
        self.assertTrue(first)
        self.assertEqual(repeated, "")
        self.assertTrue(other_region)

    def test_catalogue_failure_preserves_independent_runtime_answer(self):
        self.runtime.return_value = {"example": {"id": "anthropic.claude-example"}}
        output = io.StringIO()
        with patch.object(bedrock, "post", side_effect=RuntimeError("private error detail")), contextlib.redirect_stdout(output):
            result = bedrock.endpoints_for("anthropic.claude-example")
        self.assertEqual(result, {"mantle": False, "runtime": True})
        self.assertIn("catalogue is unavailable", output.getvalue())
        self.assertNotIn("private error detail", output.getvalue())

    def test_runtime_failure_does_not_hide_available_mantle_model(self):
        self.runtime.side_effect = RuntimeError("private runtime detail")
        result, output = self.discover([{"id": "anthropic.claude-example", "status": "available"}])
        self.assertEqual(result, {"mantle": True, "runtime": False})
        self.assertIn("runtime", output)
        self.assertNotIn("private runtime detail", output)

    def test_available_status_with_another_account_mode_is_not_overridden(self):
        result, output = self.discover([{
            "id": "anthropic.claude-example", "status": "available",
            "data_retention": {"mode": "aws_review", "allowed_modes": ["aws_review"]},
        }])
        self.assertTrue(result["mantle"])
        self.assertEqual(output, "")

    def test_retention_metadata_can_be_absent_or_null(self):
        for metadata in (None, {}):
            with self.subTest(metadata=metadata):
                bedrock._MANTLE_CATALOGUE_CACHE.clear()
                bedrock._WARNED.clear()
                result, output = self.discover([{
                    "id": "anthropic.claude-example", "status": "unavailable", "data_retention": metadata,
                }])
                self.assertFalse(result["mantle"])
                self.assertIn("status", output)

    def test_inventory_ids_stay_sorted_and_include_unavailable_models(self):
        entries = [{"id": "z", "status": "unavailable"}, {"id": "a", "status": "available"}]
        with patch.object(bedrock, "post", return_value=(200, {"data": entries})) as post:
            self.assertEqual(bedrock.list_models(), ["a", "z"])
            result = bedrock.list_model_entries()
            result[0]["status"] = "changed"
            result.clear()
            self.assertEqual(bedrock.list_model_entries()[0]["status"], "available")
            post.assert_called_once_with("/v1/models", None, region="us-east-1", method="GET")

    def test_nested_retention_edits_do_not_disturb_cached_entries(self):
        entries = [{
            "id": "a", "status": "available",
            "data_retention": {"mode": "default", "allowed_modes": ["default"]},
        }]
        with patch.object(bedrock, "post", return_value=(200, {"data": entries})) as post:
            result = bedrock.list_model_entries()
            result[0]["data_retention"]["allowed_modes"].append("poison")
            result[0]["data_retention"]["mode"] = "changed"
            fresh = bedrock.list_model_entries()
            self.assertEqual(fresh[0]["data_retention"], {
                "mode": "default", "allowed_modes": ["default"],
            })
            fresh[0]["data_retention"]["allowed_modes"] = ["replaced"]
            self.assertEqual(bedrock.list_model_entries()[0]["data_retention"]["allowed_modes"], ["default"])
            post.assert_called_once_with("/v1/models", None, region="us-east-1", method="GET")

    def test_permanent_failures_are_cached_but_transient_failures_retry(self):
        with patch.object(bedrock, "post", return_value=(403, {"message": "denied"})) as post:
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    bedrock.list_model_entries()
            self.assertEqual(post.call_count, 1)
        bedrock._MANTLE_CATALOGUE_FAILED.clear()
        with patch.object(bedrock, "post", side_effect=[(503, {}), (200, {"data": []})]) as post:
            with self.assertRaises(RuntimeError):
                bedrock.list_model_entries()
            self.assertEqual(bedrock.list_models(), [])
            self.assertEqual(post.call_count, 2)

    def test_notebook_discovery_counts_available_models_and_runs_without_inference(self):
        notebook = json.loads((ROOT / "00-foundations/01-endpoints-auth-and-the-three-paths.ipynb").read_text())
        source = "".join(notebook["cells"][13]["source"])
        entries = [{"id": "a-unavailable", "status": "unavailable"}, {"id": "b-available", "status": "available"}]
        client = SimpleNamespace(
            list_foundation_models=lambda **_: {"modelSummaries": []},
            get_paginator=lambda _: SimpleNamespace(paginate=lambda: [{"inferenceProfileSummaries": []}]),
        )
        namespace = {"REGION": "us-east-1", "boto3": SimpleNamespace(client=lambda *_, **__: client),
                     "endpoints_for": bedrock.endpoints_for, "runtime_id_for": lambda *_: None}
        output = io.StringIO()
        with patch.object(bedrock, "post", return_value=(200, {"data": entries})) as post, contextlib.redirect_stdout(output):
            # Execute only the fixed, checked-in discovery cell, never user-supplied code.
            exec(compile(source, "foundations:discovery", "exec"), namespace)  # nosec B102  # noqa: S102
        self.assertIn("1 of 2 listed models are available here", output.getvalue())
        self.assertIn("for example ['b-available']", output.getvalue())
        self.assertNotIn("for example ['a-unavailable'", output.getvalue())
        post.assert_called_once_with("/v1/models", None, region="us-east-1", method="GET")


if __name__ == "__main__":
    unittest.main()
