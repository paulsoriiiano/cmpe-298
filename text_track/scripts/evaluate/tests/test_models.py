"""Unit tests for models.complete_with_retry()'s explicit retry/backoff logic.

Real network calls are never made — FakeModelClient is substituted into _CLIENT_CACHE, and
time.sleep is mocked so backoff delays don't cost real wall-clock time in the test suite.
"""
import unittest
from unittest import mock

from .. import models
from .fakes import FakeModelClient, canned_response

MODEL_KEY = "claude_sonnet_4_6"  # max_retries=3 (the ModelConfig default) applies here


class _StatefulResponder:
    """Raises `exception` for the first `fail_count` calls, then returns `response`."""

    def __init__(self, exception, fail_count, response=None):
        self.exception = exception
        self.fail_count = fail_count
        self.response = response or canned_response("ok")
        self.call_count = 0

    def __call__(self, system, user):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            return self.exception
        return self.response


class CompleteWithRetryTests(unittest.TestCase):
    def setUp(self):
        sleep_patcher = mock.patch.object(models.time, "sleep")
        self.mock_sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def _install_fake(self, responder):
        fake_client = FakeModelClient(responder)
        patcher = mock.patch.dict(models._CLIENT_CACHE, {MODEL_KEY: fake_client}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake_client

    def test_success_on_first_attempt_has_zero_retries(self):
        responder = _StatefulResponder(exception=None, fail_count=0)
        fake_client = self._install_fake(responder)

        response, exception, retry_count, latency_ms = models.complete_with_retry(
            MODEL_KEY, system="sys", user="hi",
        )

        self.assertIsNotNone(response)
        self.assertIsNone(exception)
        self.assertEqual(retry_count, 0)
        self.assertEqual(len(fake_client.calls), 1)
        self.mock_sleep.assert_not_called()

    def test_timeout_then_success_retries_once(self):
        responder = _StatefulResponder(exception=TimeoutError("timed out"), fail_count=1)
        fake_client = self._install_fake(responder)

        response, exception, retry_count, latency_ms = models.complete_with_retry(
            MODEL_KEY, system="sys", user="hi",
        )

        self.assertIsNotNone(response)
        self.assertIsNone(exception)
        self.assertEqual(retry_count, 1)
        self.assertEqual(len(fake_client.calls), 2)
        self.mock_sleep.assert_called_once_with(2.0)  # first backoff step

    def test_exhausted_retries_returns_last_exception(self):
        exc = ConnectionError("connection refused")
        responder = _StatefulResponder(exception=exc, fail_count=999)  # always fails
        fake_client = self._install_fake(responder)

        response, exception, retry_count, latency_ms = models.complete_with_retry(
            MODEL_KEY, system="sys", user="hi",
        )

        self.assertIsNone(response)
        self.assertIs(exception, exc)
        config = models.MODEL_REGISTRY[MODEL_KEY]
        self.assertEqual(retry_count, config.max_retries)
        self.assertEqual(len(fake_client.calls), config.max_retries + 1)
        self.assertEqual(self.mock_sleep.call_count, config.max_retries)
        self.mock_sleep.assert_has_calls([mock.call(2.0), mock.call(4.0), mock.call(8.0)])

    def test_non_retryable_exception_fails_immediately(self):
        class BadRequestError(Exception):
            status_code = 400

        exc = BadRequestError("bad request")
        responder = _StatefulResponder(exception=exc, fail_count=999)
        fake_client = self._install_fake(responder)

        response, exception, retry_count, latency_ms = models.complete_with_retry(
            MODEL_KEY, system="sys", user="hi",
        )

        self.assertIsNone(response)
        self.assertIs(exception, exc)
        self.assertEqual(retry_count, 0)
        self.assertEqual(len(fake_client.calls), 1)
        self.mock_sleep.assert_not_called()

    def test_rate_limit_429_is_retryable(self):
        class RateLimitError(Exception):
            status_code = 429

        responder = _StatefulResponder(exception=RateLimitError("rate limited"), fail_count=1)
        self._install_fake(responder)

        response, exception, retry_count, _ = models.complete_with_retry(
            MODEL_KEY, system="sys", user="hi",
        )
        self.assertIsNotNone(response)
        self.assertEqual(retry_count, 1)

    def test_server_5xx_is_retryable(self):
        class InternalServerError(Exception):
            status_code = 503

        responder = _StatefulResponder(exception=InternalServerError("bad gateway"), fail_count=1)
        self._install_fake(responder)

        response, exception, retry_count, _ = models.complete_with_retry(
            MODEL_KEY, system="sys", user="hi",
        )
        self.assertIsNotNone(response)
        self.assertEqual(retry_count, 1)


if __name__ == "__main__":
    unittest.main()
