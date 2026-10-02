"""Unit tests for the current flat-JSONL text-track analyzer."""

from __future__ import annotations

import os
import sys
import unittest


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SCRIPTS_DIR)
import analyze  # noqa: E402


def row(*, item_id="item-1", condition_key="A_EE", stage="reason", is_correct=True,
        failure_type="correct", **extra):
    value = {
        "item_id": item_id,
        "condition_key": condition_key,
        "stage": stage,
        "is_correct": is_correct,
        "failure_type": failure_type,
    }
    value.update(extra)
    return value


class AccuracyTests(unittest.TestCase):
    def test_accuracy_counts_boolean_results_and_wilson_interval(self):
        result = analyze.accuracy([
            row(is_correct=True),
            row(item_id="item-2", is_correct=False, failure_type="substantively_incorrect"),
            row(item_id="item-3", is_correct=None),
        ])
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["total"], 2)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertIsNotNone(result["ci_wilson_95"]["low"])

    def test_pivot_translation_failure_is_end_to_end_incorrect(self):
        outcomes = analyze.condition_outcomes([
            row(condition_key="A_IE", stage="translate", is_correct=None,
                failure_type="truncation"),
        ], "A_IE")
        self.assertEqual(outcomes, {"item-1": False})

    def test_completed_pivot_uses_reasoning_outcome(self):
        outcomes = analyze.condition_outcomes([
            row(condition_key="A_IE", stage="translate", is_correct=None,
                failure_type="translation_completed"),
            row(condition_key="A_IE", stage="reason", is_correct=True),
        ], "A_IE")
        self.assertEqual(outcomes, {"item-1": True})

    def test_missing_pivot_reasoning_is_excluded_when_translation_is_usable(self):
        outcomes = analyze.condition_outcomes([
            row(condition_key="A_EI", stage="translate", is_correct=None,
                failure_type="translation_completed"),
        ], "A_EI")
        self.assertEqual(outcomes, {})


class PairedComparisonTests(unittest.TestCase):
    def test_paired_test_reports_directional_discordance(self):
        index = {
            ("model", "A_IE", "reason"): {
                "i1": row(item_id="i1", condition_key="A_IE", is_correct=True),
                "i2": row(item_id="i2", condition_key="A_IE", is_correct=False),
                "i3": row(item_id="i3", condition_key="A_IE", is_correct=True),
            },
            ("model", "A_II", "reason"): {
                "i1": row(item_id="i1", condition_key="A_II", is_correct=False),
                "i2": row(item_id="i2", condition_key="A_II", is_correct=True),
                "i3": row(item_id="i3", condition_key="A_II", is_correct=True),
            },
        }
        result = analyze.paired_test(index, "model", "A_IE", "A_II", seed=42)
        self.assertEqual(result["n_paired"], 3)
        self.assertEqual(result["first_only_correct"], 1)
        self.assertEqual(result["second_only_correct"], 1)
        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["both_incorrect"], 0)
        self.assertAlmostEqual(result["delta_first_minus_second"], 0.0)
        self.assertEqual(result["mcnemar_exact_p"], 1.0)


class FailureRateTests(unittest.TestCase):
    def test_failure_rates_use_independent_record_flags(self):
        result = analyze.failure_rates([
            row(is_correct=False, failure_type="truncation", is_truncated=True,
                degeneration_candidate=True, format_compliant=False),
            row(item_id="item-2", is_correct=False,
                failure_type="substantively_incorrect", is_truncated=False,
                degeneration_candidate=False, format_compliant=True),
        ])
        self.assertEqual(result["total_reasoning_or_direct_records"], 2)
        self.assertEqual(result["truncated_count"], 1)
        self.assertEqual(result["degeneration_candidate_count"], 1)
        self.assertEqual(result["format_noncompliant_count"], 1)
        self.assertEqual(result["failure_type_counts"]["truncation"], 1)


if __name__ == "__main__":
    unittest.main()
