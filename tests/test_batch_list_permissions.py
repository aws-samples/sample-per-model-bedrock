"""Run the notebook's actual code cells with the pinned SDK and mocked HTTP.

Run: .venv/bin/python -m unittest discover -s tests -v
No AWS credentials, token generation, model calls, or cloud resources are used.
"""

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
from openai import APIStatusError, OpenAI


NOTEBOOK = Path(__file__).resolve().parents[1] / "01-openai-gpt" / "05-server-side-tools-and-fine-tuning.ipynb"


class BatchListPermissionsTest(unittest.TestCase):
    def run_notebook(self, status=200, jobs=None):
        requests = []
        namespace = {}
        output = io.StringIO()
        cells = json.loads(NOTEBOOK.read_text())["cells"]

        def handle(request):
            requests.append(request.url.path)
            self.assertEqual(request.method, "GET")
            if request.url.path == "/openai/v1/batches":
                self.assertEqual(request.url.params["limit"], "20")
                if status != 200:
                    return httpx.Response(status, json={"error": {"message": "private caller details"}})
                data = jobs or []
            elif request.url.path == "/v1/models":
                data = [{"id": "openai.gpt-oss-20b"}, {"id": "qwen.qwen3-32b"}]
            elif request.url.path in ("/v1/files", "/v1/fine_tuning/jobs"):
                data = []
            else:
                self.fail(f"Unexpected request: {request.url.path}")
            return httpx.Response(200, json={"object": "list", "data": data})

        with contextlib.ExitStack() as stack:
            def client(**kwargs):
                kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handle))
                kwargs["max_retries"] = 0
                return stack.enter_context(OpenAI(**kwargs))

            stack.enter_context(patch("aws_bedrock_token_generator.provide_token", return_value="test-token"))
            stack.enter_context(patch("openai.OpenAI", side_effect=client))
            stack.enter_context(contextlib.redirect_stdout(output))
            for index, cell in enumerate(cells):
                if cell["cell_type"] == "code":
                    # Execute only this fixed, checked-in notebook, never user-supplied code.
                    exec(  # nosec B102  # noqa: S102
                        compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell-{index}", "exec"),
                        namespace,
                    )

        return namespace, output.getvalue(), requests

    def assert_following_cells_ran(self, namespace, output, requests):
        self.assertEqual(requests, ["/v1/models", "/openai/v1/batches", "/v1/files", "/v1/fine_tuning/jobs"])
        self.assertEqual(namespace["bad"], {})
        self.assertIn("format problems: none", output)
        self.assertIn("hedged    score=0.0", output)

    def test_denied_batch_listing_continues_through_training_and_reward_cells(self):
        namespace, output, requests = self.run_notebook(status=403)
        self.assertIn("cannot list batch jobs with these credentials", output)
        self.assertIn("bedrock:ListModelInvocationJobs", output)
        self.assertIn("bedrock:CreateModelInvocationJob", output)
        self.assertNotIn("private caller details", output)
        self.assert_following_cells_ran(namespace, output, requests)

    def test_allowed_empty_listing_preserves_output_and_continues(self):
        namespace, output, requests = self.run_notebook()
        self.assertIn("0 batch job(s) in us-east-1", output)
        self.assertNotIn("cannot list batch jobs", output)
        self.assert_following_cells_ran(namespace, output, requests)

    def test_allowed_nonempty_listing_preserves_job_output(self):
        namespace, output, requests = self.run_notebook(jobs=[{
            "id": "batch-test", "object": "batch", "status": "completed", "created_at": 123,
        }])
        self.assertIn("1 batch job(s) in us-east-1", output)
        self.assertIn("completed    created 123", output)
        self.assert_following_cells_ran(namespace, output, requests)

    def test_non_permission_errors_are_not_hidden(self):
        for status in (401, 429, 500):
            with self.subTest(status=status):
                with self.assertRaises(APIStatusError) as caught:
                    self.run_notebook(status=status)
                self.assertEqual(caught.exception.status_code, status)


if __name__ == "__main__":
    unittest.main()
