"""Tests for inspect_repetition_scores.py's backfill of legacy (evaluator_v3 and earlier)
records, which predate the repetition_ratio/is_truncated fields — needed so real pilot data
can be used for threshold calibration without rerunning.
"""
import os
import sys
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SCRIPTS_DIR)
import inspect_repetition_scores as irs  # noqa: E402


class BackfillLegacyFieldsTests(unittest.TestCase):
    def test_legacy_record_missing_both_fields_gets_backfilled(self):
        record = {
            "raw_response": "hello world foo bar hello world foo bar",
            "finish_reason": "stop",
        }
        backfilled = irs.backfill_legacy_fields(record)
        self.assertIsNotNone(backfilled["repetition_ratio"])
        self.assertFalse(backfilled["is_truncated"])

    def test_legacy_record_truncated_via_length_is_backfilled_true(self):
        record = {"raw_response": "cut off", "finish_reason": "length"}
        backfilled = irs.backfill_legacy_fields(record)
        self.assertTrue(backfilled["is_truncated"])

    def test_legacy_record_truncated_via_max_tokens_is_backfilled_true(self):
        record = {"raw_response": "cut off", "finish_reason": "max_tokens"}
        backfilled = irs.backfill_legacy_fields(record)
        self.assertTrue(backfilled["is_truncated"])

    def test_modern_record_with_fields_already_set_is_unchanged(self):
        record = {
            "raw_response": "anything", "finish_reason": "stop",
            "repetition_ratio": 0.42, "is_truncated": False,
        }
        backfilled = irs.backfill_legacy_fields(record)
        self.assertEqual(backfilled["repetition_ratio"], 0.42)
        self.assertFalse(backfilled["is_truncated"])

    def test_missing_raw_response_does_not_crash_and_yields_zero_ratio(self):
        record = {"finish_reason": "stop"}
        backfilled = irs.backfill_legacy_fields(record)
        self.assertEqual(backfilled["repetition_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
