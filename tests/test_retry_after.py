"""Retry-After handling in the shared post() loop and in the hardening notebook's copy.

Run: .venv/bin/python -m unittest discover -s tests -v
No AWS credentials, token generation, model calls, or cloud resources are used.
"""

import email.message
import email.utils
import json
from pathlib import Path
import sys
import time
import unittest
import unittest.mock
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

import bedrock  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parents[1] / "99-cross-cutting" / "03-production-hardening.ipynb"


def headers(value=None, name="Retry-After"):
    """An HTTPError.headers stand-in: the same type, so lookups are case-insensitive."""
    held = email.message.Message()
    if value is not None:
        held[name] = value
    return held


class RetryAfterSecondsTest(unittest.TestCase):
    def test_delta_seconds(self):
        self.assertEqual(bedrock._retry_after_seconds(headers("7")), 7.0)
        self.assertEqual(bedrock._retry_after_seconds(headers("0.5")), 0.5)

    def test_http_date(self):
        when = email.utils.formatdate(time.time() + 12, usegmt=True)
        self.assertAlmostEqual(bedrock._retry_after_seconds(headers(when)), 12, delta=2)

    def test_case_insensitive(self):
        self.assertEqual(bedrock._retry_after_seconds(headers("9", name="retry-after")), 9.0)

    def test_nothing_usable(self):
        # The last one parses as a date and cannot be represented as a timestamp, so it
        # reaches mktime_tz rather than failing in parsedate_tz like the others.
        for value in (None, "soon", "", "Tue, 20 Oct 99999 00:00:00 GMT"):
            self.assertIsNone(bedrock._retry_after_seconds(headers(value)), value)
        self.assertIsNone(bedrock._retry_after_seconds(None))


class PostHonoursRetryAfterTest(unittest.TestCase):
    """post() waits what the header names, and falls back to its backoff otherwise."""

    def waited(self, first_status, header_value):
        slept, attempts = [], []

        def fake_open(request, timeout):
            attempts.append(request.full_url)
            if len(attempts) == 1:
                raise urllib.error.HTTPError(
                    request.full_url, first_status, "", headers(header_value), None)
            # A parameter 400 is not retryable, so the loop returns on attempt two.
            raise urllib.error.HTTPError(
                request.full_url, 400, "", headers(), None)

        with unittest.mock.patch.object(bedrock, "_open_https", fake_open), \
                unittest.mock.patch.object(bedrock, "token", lambda region=None: "stub"), \
                unittest.mock.patch.object(bedrock.time, "sleep", slept.append), \
                unittest.mock.patch.object(bedrock.random, "random", lambda: 0.0):
            status, _ = bedrock.post("/openai/v1/responses", {"input": "x"}, attempts=3)
        self.assertEqual((status, len(attempts)), (400, 2))
        self.assertEqual(len(slept), 1)
        return slept[0]

    def test_header_sets_the_wait(self):
        self.assertEqual(self.waited(429, "7"), 7.0)

    def test_no_header_uses_the_backoff(self):
        self.assertEqual(self.waited(429, None), 1.0)  # min(2**0, 16)

    def test_past_the_ceiling_uses_the_backoff(self):
        self.assertEqual(self.waited(503, str(bedrock._RETRY_AFTER_MAX + 1)), 1.0)

    def test_zero_and_negative_use_the_backoff(self):
        self.assertEqual(self.waited(429, "0"), 1.0)
        self.assertEqual(self.waited(429, "-5"), 1.0)


class NotebookCopyAgreesTest(unittest.TestCase):
    """The hardening notebook teaches the same policy, so run its cell, not a paraphrase."""

    def retry_delay(self):
        cells = json.loads(NOTEBOOK.read_text())["cells"]
        sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        defining = [s for s in sources if "def retry_delay(" in s]
        self.assertEqual(len(defining), 1, "retry_delay is defined in exactly one cell")
        namespace = {"email": email, "time": time, "random": __import__("random")}
        # The point of the test is to run the committed cell rather than a copy of it,
        # so that the notebook cannot drift from the shared loop unnoticed. The source
        # is a file in this repository, not input.
        exec(  # nosec B102  # noqa: S102
            compile(defining[0], f"{NOTEBOOK.name}:retry_delay", "exec"), namespace
        )
        return namespace["retry_delay"]

    def test_same_decisions_as_the_shared_loop(self):
        retry_delay = self.retry_delay()
        when = email.utils.formatdate(time.time() + 12, usegmt=True)
        backoff = min(2 ** 3, 16)
        for value, expected in (("7", 7.0), (when, 12.0), (None, backoff),
                                ("soon", backoff), ("600", backoff),
                                ("0", backoff), ("-5", backoff),
                                ("Tue, 20 Oct 99999 00:00:00 GMT", backoff)):
            got = retry_delay(3, headers(value)) if value is not None else retry_delay(3)
            # retry_delay adds its jitter inside, so compare against [expected, expected+1).
            self.assertGreaterEqual(got, expected - 2, value)
            self.assertLess(got, expected + 1, value)


if __name__ == "__main__":
    unittest.main()
