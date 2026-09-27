"""Tests for analyze.py's core functions (not the CLI/file-writing path — tests never call
main() or touch the real text_track/data/analysis.md / evaluation_results.jsonl, which are
preserved as preliminary artifacts from the old 3-pass pipeline per the conference plan's
file priorities: "Do not combine old and new experimental results.")
"""
import os
import sys
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SCRIPTS_DIR)
import analyze  # noqa: E402


def _row(**overrides):
    defaults = dict(
        run_id="run-1", item_id="gsm8k_0", model_key="claude_sonnet_4_6",
        condition_key="A_EE", stage="reason", source="gsm8k", is_correct=True,
        is_truncated=False, repetition_ratio=0.0, degeneration_candidate=False,
        failure_type="correct", protocol_version="protocol_v2", evaluator_version="evaluator_v4",
        raw_response="<answer>109</answer>",
    )
    defaults.update(overrides)
    return defaults


class AccuracyDenominatorTests(unittest.TestCase):
    def test_reason_stage_with_correctness_is_in_denominator(self):
        self.assertTrue(analyze.in_accuracy_denominator(_row(stage="reason", is_correct=False)))

    def test_direct_stage_with_correctness_is_in_denominator(self):
        self.assertTrue(analyze.in_accuracy_denominator(_row(stage="direct", is_correct=True)))

    def test_translate_stage_never_in_denominator(self):
        row = _row(stage="translate", is_correct=None)
        self.assertFalse(analyze.in_accuracy_denominator(row))

    def test_infrastructure_failure_not_in_denominator(self):
        row = _row(stage="reason", is_correct=None, failure_type="infrastructure_api_failure")
        self.assertFalse(analyze.in_accuracy_denominator(row))

    def test_parser_failure_not_in_denominator(self):
        row = _row(stage="reason", is_correct=None, failure_type="parser_failure")
        self.assertFalse(analyze.in_accuracy_denominator(row))

    def test_truncation_refusal_missing_answer_all_in_denominator_as_incorrect(self):
        for failure_type in ("truncation", "refusal", "missing_answer", "invalid_answer_format",
                              "repetition_degeneration", "substantively_incorrect"):
            row = _row(stage="reason", is_correct=False, failure_type=failure_type)
            self.assertTrue(analyze.in_accuracy_denominator(row), failure_type)
            self.assertFalse(analyze.is_success(row), failure_type)


class AccuracyTableTests(unittest.TestCase):
    def test_denominator_includes_all_model_output_failures(self):
        rows = [
            _row(item_id="i1", is_correct=True, failure_type="correct"),
            _row(item_id="i2", is_correct=False, failure_type="truncation"),
            _row(item_id="i3", is_correct=False, failure_type="repetition_degeneration"),
            _row(item_id="i4", is_correct=False, failure_type="refusal"),
            _row(item_id="i5", is_correct=False, failure_type="missing_answer"),
            _row(item_id="i6", is_correct=False, failure_type="invalid_answer_format"),
            # excluded: infra failure and a translate-stage row
            _row(item_id="i7", is_correct=None, failure_type="infrastructure_api_failure"),
            _row(item_id="i8", is_correct=None, failure_type="translation_completed", stage="translate"),
        ]
        table = analyze.accuracy_table(rows, ("model_key", "condition_key"))
        cell = table[("claude_sonnet_4_6", "A_EE")]
        self.assertEqual(cell["total"], 6)  # i1..i6 only
        self.assertEqual(cell["correct"], 1)  # only i1

    def test_grouped_by_source(self):
        rows = [
            _row(item_id="i1", source="gsm8k", is_correct=True),
            _row(item_id="i2", source="bbh_causal_judgement", is_correct=False),
        ]
        table = analyze.accuracy_table(rows, ("model_key", "condition_key", "source"))
        self.assertEqual(table[("claude_sonnet_4_6", "A_EE", "gsm8k")]["total"], 1)
        self.assertEqual(table[("claude_sonnet_4_6", "A_EE", "bbh_causal_judgement")]["total"], 1)


class ValidateTests(unittest.TestCase):
    def test_duplicate_keys_produce_a_warning(self):
        rows = [_row(item_id="i1"), _row(item_id="i1")]
        warnings = analyze.validate(rows)
        self.assertTrue(any("duplicate" in w.lower() for w in warnings))

    def test_infra_failures_produce_a_warning(self):
        rows = [_row(item_id="i1", is_correct=None, failure_type="infrastructure_api_failure")]
        warnings = analyze.validate(rows)
        self.assertTrue(any("infrastructure" in w.lower() for w in warnings))

    def test_clean_data_has_no_warnings(self):
        rows = [_row(item_id="i1"), _row(item_id="i2")]
        self.assertEqual(analyze.validate(rows), [])


class PairedComparisonTests(unittest.TestCase):
    def test_paired_by_item_id_across_conditions(self):
        rows = [
            _row(item_id="i1", condition_key="A_EE", is_correct=True),
            _row(item_id="i2", condition_key="A_EE", is_correct=True),
            _row(item_id="i3", condition_key="A_EE", is_correct=False),
            _row(item_id="i1", condition_key="A_II", is_correct=True),
            _row(item_id="i2", condition_key="A_II", is_correct=False),  # discordant: A only
            _row(item_id="i3", condition_key="A_II", is_correct=True),   # discordant: B only
        ]
        result = analyze.paired_comparison(rows, "claude_sonnet_4_6", "A_EE", "A_II")
        self.assertEqual(result["n_paired"], 3)
        self.assertEqual(result["both"], 1)
        self.assertEqual(result["a_only"], 1)
        self.assertEqual(result["b_only"], 1)
        self.assertEqual(result["neither"], 0)

    def test_unpaired_items_are_excluded(self):
        rows = [
            _row(item_id="i1", condition_key="A_EE", is_correct=True),
            _row(item_id="i2", condition_key="A_II", is_correct=True),  # no matching A_EE row
        ]
        result = analyze.paired_comparison(rows, "claude_sonnet_4_6", "A_EE", "A_II")
        self.assertIsNone(result)  # no shared items at all

    def test_infra_failure_excluded_from_pairing(self):
        rows = [
            _row(item_id="i1", condition_key="A_EE", is_correct=True),
            _row(item_id="i1", condition_key="A_II", is_correct=None, failure_type="infrastructure_api_failure"),
        ]
        result = analyze.paired_comparison(rows, "claude_sonnet_4_6", "A_EE", "A_II")
        self.assertIsNone(result)  # i1 excluded from A_II's denominator-eligible set


class ResolveConditionOutcomesTests(unittest.TestCase):
    def test_non_pivot_condition_uses_reason_stage_directly(self):
        rows = [_row(item_id="i1", condition_key="A_EE", stage="reason", is_correct=True)]
        outcomes = analyze.resolve_condition_outcomes(rows, "claude_sonnet_4_6", "A_EE")
        self.assertEqual(outcomes, {"i1": True})

    def test_pivot_with_completed_translation_and_reason_uses_reason_outcome(self):
        rows = [
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="translation_completed"),
            _row(item_id="i1", condition_key="A_IE", stage="reason", is_correct=False,
                 failure_type="substantively_incorrect"),
        ]
        outcomes = analyze.resolve_condition_outcomes(rows, "claude_sonnet_4_6", "A_IE")
        self.assertEqual(outcomes, {"i1": False})

    def test_pivot_model_caused_translation_failure_with_no_reason_row_is_incorrect(self):
        rows = [
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="translation_format_failure"),
        ]
        outcomes = analyze.resolve_condition_outcomes(rows, "claude_sonnet_4_6", "A_IE")
        self.assertEqual(outcomes, {"i1": False})

    def test_pivot_infra_failure_translation_with_no_reason_row_is_unresolved(self):
        rows = [
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="infrastructure_api_failure"),
        ]
        outcomes = analyze.resolve_condition_outcomes(rows, "claude_sonnet_4_6", "A_IE")
        self.assertIsNone(outcomes["i1"])


class AccuracyTableStagedPivotTests(unittest.TestCase):
    def test_model_caused_translation_failure_counts_as_incorrect_not_dropped(self):
        rows = [
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="translation_format_failure"),
            _row(item_id="i2", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="translation_completed"),
            _row(item_id="i2", condition_key="A_IE", stage="reason", is_correct=True,
                 failure_type="correct"),
        ]
        table = analyze.accuracy_table(rows, ("model_key", "condition_key"))
        cell = table[("claude_sonnet_4_6", "A_IE")]
        self.assertEqual(cell["total"], 2)  # i1 (incorrect) + i2 (correct) — i1 not dropped
        self.assertEqual(cell["correct"], 1)

    def test_infra_failure_translation_excluded_from_denominator(self):
        rows = [
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="infrastructure_api_failure"),
        ]
        table = analyze.accuracy_table(rows, ("model_key", "condition_key"))
        self.assertNotIn(("claude_sonnet_4_6", "A_IE"), table)


class PairedComparisonStagedPivotTests(unittest.TestCase):
    def test_model_caused_translation_failure_pairs_as_incorrect(self):
        rows = [
            _row(item_id="i1", condition_key="A_EE", stage="reason", is_correct=True),
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="translation_format_failure"),
        ]
        result = analyze.paired_comparison(rows, "claude_sonnet_4_6", "A_EE", "A_IE")
        self.assertIsNotNone(result)
        self.assertEqual(result["n_paired"], 1)
        self.assertEqual(result["a_only"], 1)  # correct under A_EE, incorrect under A_IE

    def test_infra_failure_translation_excluded_from_pairing(self):
        rows = [
            _row(item_id="i1", condition_key="A_EE", stage="reason", is_correct=True),
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="infrastructure_api_failure"),
        ]
        result = analyze.paired_comparison(rows, "claude_sonnet_4_6", "A_EE", "A_IE")
        self.assertIsNone(result)


class ManifestCompletenessTests(unittest.TestCase):
    def _manifest(self, **overrides):
        defaults = dict(
            models=["claude_sonnet_4_6"],
            planned_item_ids_by_condition={"A_EE": ["i1", "i2"]},
        )
        defaults.update(overrides)
        return defaults

    def test_all_planned_units_present_has_no_warnings(self):
        rows = [
            _row(item_id="i1", condition_key="A_EE", stage="reason"),
            _row(item_id="i2", condition_key="A_EE", stage="reason"),
        ]
        manifest = self._manifest()
        warnings = analyze.check_manifest_completeness(rows, manifest)
        self.assertEqual(warnings, [])

    def test_missing_row_entirely_is_flagged(self):
        rows = [_row(item_id="i1", condition_key="A_EE", stage="reason")]
        manifest = self._manifest()
        warnings = analyze.check_manifest_completeness(rows, manifest)
        self.assertTrue(any("i2" in w or "1 planned" in w for w in warnings))

    def test_pivot_missing_translate_row_is_flagged(self):
        manifest = self._manifest(planned_item_ids_by_condition={"A_IE": ["i1"]})
        warnings = analyze.check_manifest_completeness([], manifest)
        self.assertTrue(any("translate" in w for w in warnings))

    def test_pivot_missing_reason_after_model_caused_translation_failure_not_flagged(self):
        rows = [
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="translation_format_failure"),
        ]
        manifest = self._manifest(planned_item_ids_by_condition={"A_IE": ["i1"]})
        warnings = analyze.check_manifest_completeness(rows, manifest)
        self.assertEqual(warnings, [])

    def test_pivot_missing_reason_after_infra_failure_not_flagged_here(self):
        # This is surfaced by validate()'s NOT_EVALUATED_FAILURE_TYPES warning instead —
        # check_manifest_completeness only flags rows that are missing outright.
        rows = [
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="infrastructure_api_failure"),
        ]
        manifest = self._manifest(planned_item_ids_by_condition={"A_IE": ["i1"]})
        warnings = analyze.check_manifest_completeness(rows, manifest)
        self.assertEqual(warnings, [])

    def test_pivot_completed_translation_missing_reason_is_flagged(self):
        # A genuinely unexpected gap: translation succeeded but reason never ran/wrote.
        rows = [
            _row(item_id="i1", condition_key="A_IE", stage="translate", is_correct=None,
                 failure_type="translation_completed"),
        ]
        manifest = self._manifest(planned_item_ids_by_condition={"A_IE": ["i1"]})
        warnings = analyze.check_manifest_completeness(rows, manifest)
        self.assertTrue(any("reason" in w for w in warnings))

    def test_validate_includes_manifest_warnings_when_manifest_given(self):
        rows = [_row(item_id="i1", condition_key="A_EE", stage="reason")]
        manifest = self._manifest()
        warnings = analyze.validate(rows, manifest)
        self.assertTrue(any("planned" in w.lower() for w in warnings))

    def test_validate_skips_manifest_check_when_no_manifest_given(self):
        rows = [_row(item_id="i1", condition_key="A_EE", stage="reason")]
        self.assertEqual(analyze.validate(rows), [])


class HolmAdjustTests(unittest.TestCase):
    def test_preserves_input_order(self):
        p_values = [0.04, 0.01, 0.03]
        adjusted = analyze.holm_adjust(p_values)
        self.assertEqual(len(adjusted), 3)
        # Holm-adjusted p-values are monotonically non-decreasing with the sorted rank, and
        # each adjustment is >= the raw p-value.
        for raw, adj in zip(p_values, adjusted):
            self.assertGreaterEqual(adj, raw)


class WilsonCiTests(unittest.TestCase):
    def test_zero_total_returns_zero_interval(self):
        self.assertEqual(analyze.wilson_ci(0, 0), (0.0, 0.0))

    def test_interval_contains_point_estimate(self):
        lo, hi = analyze.wilson_ci(8, 10)
        self.assertLessEqual(lo, 0.8)
        self.assertGreaterEqual(hi, 0.8)


if __name__ == "__main__":
    unittest.main()
