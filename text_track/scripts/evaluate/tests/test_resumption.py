"""Resumption tests: simulate a crash mid-pivot (translate succeeds, reason stage fails
with an infrastructure error), restart, and assert the translate stage is NOT re-run and
only the missing reason stage is retried. Zero real API calls (FakeModelClient throughout).
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from .. import models
from ..run import run_evaluation
from .fakes import FakeModelClient, canned_response


class ControllableResponder:
    """Translate calls always succeed; reason calls fail with a simulated infrastructure
    error until fail_reason_stage is flipped off (simulating "the outage is now over")."""

    def __init__(self):
        self.fail_reason_stage = True

    def __call__(self, system, user):
        if "translator" in system.lower():
            return canned_response(f"[translated] {user}")
        if self.fail_reason_stage:
            return ConnectionError("simulated infrastructure failure")
        return canned_response("Reasoning...\n<answer>42</answer>")


class ResumptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

        self.responder = ControllableResponder()
        self.fake_client = FakeModelClient(self.responder)
        patcher = mock.patch.dict(
            models._CLIENT_CACHE, {"claude_sonnet_4_6": self.fake_client}, clear=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _all_records(self):
        records = []
        for name in sorted(os.listdir(self.tmp_dir)):
            if name.endswith(".jsonl"):
                with open(os.path.join(self.tmp_dir, name)) as f:
                    records.extend(json.loads(line) for line in f if line.strip())
        return records

    def test_translate_stage_not_rerun_after_resuming_a_failed_reason_stage(self):
        # First run: translate succeeds, reason stage hits a simulated infra failure.
        run_evaluation(
            condition_keys=["A_IE"], model_keys=["claude_sonnet_4_6"], limit=1,
            runs_dir=self.tmp_dir,
        )
        calls_after_first_run = list(self.fake_client.calls)
        self.assertEqual(len(calls_after_first_run), 2)  # 1 translate + 1 failed reason attempt

        records = self._all_records()
        translate_records = [r for r in records if r["stage"] == "translate"]
        reason_records = [r for r in records if r["stage"] == "reason"]
        self.assertEqual(len(translate_records), 1)
        self.assertEqual(translate_records[0]["failure_type"], "correct")
        self.assertEqual(len(reason_records), 1)
        self.assertEqual(reason_records[0]["failure_type"], "infrastructure_api_failure")

        # "Outage" is over; restart under a fresh run_id (a real restart would also get a
        # fresh run_id — resumption must not depend on reusing the old one).
        self.responder.fail_reason_stage = False
        run_evaluation(
            condition_keys=["A_IE"], model_keys=["claude_sonnet_4_6"], limit=1,
            runs_dir=self.tmp_dir,
        )

        calls_after_second_run = self.fake_client.calls
        new_calls = calls_after_second_run[len(calls_after_first_run):]
        # Exactly one new call (the retried reason stage) — the translate stage must NOT
        # have been re-issued.
        self.assertEqual(len(new_calls), 1)
        new_system, _ = new_calls[0]
        self.assertNotIn("translator", new_system.lower())

        records = self._all_records()
        translate_records = [r for r in records if r["stage"] == "translate"]
        reason_records = [r for r in records if r["stage"] == "reason"]
        # Still exactly one translate record overall (not duplicated).
        self.assertEqual(len(translate_records), 1)
        # Now two reason records exist across the two run files (the failed attempt plus
        # the successful retry) — this is expected: each is its own immutable log entry;
        # ResumeIndex treats the row with a non-retryable outcome as authoritative.
        self.assertEqual(len(reason_records), 2)
        failure_types = {r["failure_type"] for r in reason_records}
        self.assertIn("infrastructure_api_failure", failure_types)
        self.assertTrue(failure_types - {"infrastructure_api_failure"})  # the successful retry


if __name__ == "__main__":
    unittest.main()
