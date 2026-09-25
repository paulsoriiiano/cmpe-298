"""Unit tests for grading.classify_result() covering every FailureType branch.
No network calls: uses canned ModelResponse objects only (see fakes.CANNED).
"""
import unittest

from ..grading import FailureType, classify_result, format_compliant_for
from .fakes import CANNED


class ClassifyResultTests(unittest.TestCase):
    def test_infrastructure_api_failure_on_exception(self):
        failure_type, extracted, is_correct = classify_result(
            response=None, exception=ConnectionError("boom"), canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.INFRASTRUCTURE_API_FAILURE)
        self.assertIsNone(extracted)
        self.assertIsNone(is_correct)
        self.assertIsNone(format_compliant_for(failure_type))

    def test_truncation(self):
        failure_type, _, _ = classify_result(
            response=CANNED["truncated"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.TRUNCATION)
        self.assertFalse(format_compliant_for(failure_type))

    def test_refusal(self):
        failure_type, _, _ = classify_result(
            response=CANNED["refusal"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.REFUSAL)

    def test_repetition_degeneration(self):
        failure_type, _, _ = classify_result(
            response=CANNED["repetitive"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.REPETITION_DEGENERATION)

    def test_missing_answer(self):
        failure_type, extracted, _ = classify_result(
            response=CANNED["missing_answer"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.MISSING_ANSWER)
        self.assertIsNone(extracted)

    def test_translation_format_failure_when_solved_instead_of_translated(self):
        failure_type, _, _ = classify_result(
            response=CANNED["translation_solved_instead"], exception=None,
            canonical_answer=None, source="gsm8k", stage="translate",
        )
        self.assertEqual(failure_type, FailureType.TRANSLATION_FORMAT_FAILURE)

    def test_translation_format_failure_when_empty(self):
        failure_type, _, _ = classify_result(
            response=CANNED["translation_empty"], exception=None,
            canonical_answer=None, source="gsm8k", stage="translate",
        )
        self.assertEqual(failure_type, FailureType.TRANSLATION_FORMAT_FAILURE)

    def test_translation_ok(self):
        failure_type, _, _ = classify_result(
            response=CANNED["translation_ok"], exception=None,
            canonical_answer=None, source="gsm8k", stage="translate",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)

    def test_invalid_answer_format(self):
        failure_type, extracted, is_correct = classify_result(
            response=CANNED["invalid_format"], exception=None, canonical_answer="A",
            source="mmlu_formal_logic", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.INVALID_ANSWER_FORMAT)
        self.assertEqual(extracted, "maybe purple")
        self.assertFalse(is_correct)
        self.assertTrue(format_compliant_for(failure_type))

    def test_correct_number(self):
        failure_type, extracted, is_correct = classify_result(
            response=CANNED["correct_number"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertEqual(extracted, "109")
        self.assertTrue(is_correct)

    def test_correct_letter(self):
        failure_type, extracted, is_correct = classify_result(
            response=CANNED["correct_letter"], exception=None, canonical_answer="A",
            source="mmlu_formal_logic", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertTrue(is_correct)

    def test_substantively_incorrect(self):
        failure_type, extracted, is_correct = classify_result(
            response=CANNED["wrong_number"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.SUBSTANTIVELY_INCORRECT)
        self.assertEqual(extracted, "42")
        self.assertFalse(is_correct)
        self.assertTrue(format_compliant_for(failure_type))


if __name__ == "__main__":
    unittest.main()
