"""Tests for generate_annotation_template.py's row-building logic (not the CLI/file I/O).

Regression coverage: build_annotation_rows() used to derive rationale_present via a naive
bool(generated_rationale) check, which treated "<answer>109</answer>" as containing a
rationale. rationale_present is now a human-verified blank column; has_text_beyond_answer
is the machine-derived diagnostic instead.
"""
import os
import sys
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SCRIPTS_DIR)
import generate_annotation_template as gat  # noqa: E402


def _record(**overrides):
    defaults = dict(
        run_id="run-1", item_id="gsm8k_0", model_key="claude_sonnet_4_6",
        condition_key="A_EE", stage="reason", source="gsm8k", original_input="q",
        generated_translation=None, generated_rationale="<answer>109</answer>",
        raw_response="<answer>109</answer>", extracted_answer="109", canonical_answer="109",
        is_correct=True, format_compliant=True, failure_type="correct",
    )
    defaults.update(overrides)
    return defaults


class BuildAnnotationRowsTests(unittest.TestCase):
    def test_rationale_present_is_blank_not_inferred(self):
        rows = gat.build_annotation_rows([_record()])
        self.assertEqual(rows[0]["rationale_present"], "")

    def test_has_text_beyond_answer_false_for_answer_only(self):
        rows = gat.build_annotation_rows([_record(raw_response="<answer>109</answer>")])
        self.assertFalse(rows[0]["has_text_beyond_answer"])

    def test_has_text_beyond_answer_true_with_explanation(self):
        rows = gat.build_annotation_rows([_record(
            raw_response="Allen will be 109 in ten years.\n<answer>109</answer>",
        )])
        self.assertTrue(rows[0]["has_text_beyond_answer"])

    def test_requested_language_none_for_direct_controls(self):
        rows = gat.build_annotation_rows([
            _record(condition_key="A_E0", stage="direct", raw_response="<answer>109</answer>"),
            _record(condition_key="A_I0", stage="direct", raw_response="<answer>109</answer>"),
        ])
        self.assertIsNone(rows[0]["requested_language"])
        self.assertIsNone(rows[1]["requested_language"])

    def test_requested_language_set_for_reasoning_conditions(self):
        rows = gat.build_annotation_rows([_record(condition_key="A_II", stage="reason")])
        self.assertEqual(rows[0]["requested_language"], "ilo")


if __name__ == "__main__":
    unittest.main()
