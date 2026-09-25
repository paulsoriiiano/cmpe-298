"""Unit tests for grading.classify_result() covering every FailureType branch, plus the
fallback-extraction regression tests for responses that skip the required <answer> tag.
No network calls: uses canned ModelResponse objects only (see fakes.CANNED).
"""
import unittest

from ..grading import FailureType, classify_result
from .fakes import CANNED, canned_response


class ClassifyResultTests(unittest.TestCase):
    def test_infrastructure_api_failure_on_exception(self):
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=None, exception=ConnectionError("boom"), canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.INFRASTRUCTURE_API_FAILURE)
        self.assertIsNone(extracted)
        self.assertIsNone(is_correct)
        self.assertIsNone(format_compliant)

    def test_truncation(self):
        failure_type, _, _, format_compliant = classify_result(
            response=CANNED["truncated"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.TRUNCATION)
        self.assertFalse(format_compliant)

    def test_refusal(self):
        failure_type, _, _, format_compliant = classify_result(
            response=CANNED["refusal"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.REFUSAL)
        self.assertFalse(format_compliant)

    def test_repetition_degeneration(self):
        failure_type, _, _, format_compliant = classify_result(
            response=CANNED["repetitive"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.REPETITION_DEGENERATION)
        self.assertFalse(format_compliant)

    def test_missing_answer(self):
        failure_type, extracted, _, format_compliant = classify_result(
            response=CANNED["missing_answer"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.MISSING_ANSWER)
        self.assertIsNone(extracted)
        self.assertFalse(format_compliant)

    def test_translation_format_failure_when_solved_instead_of_translated(self):
        failure_type, _, _, format_compliant = classify_result(
            response=CANNED["translation_solved_instead"], exception=None,
            canonical_answer=None, source="gsm8k", stage="translate",
        )
        self.assertEqual(failure_type, FailureType.TRANSLATION_FORMAT_FAILURE)
        self.assertIsNone(format_compliant)

    def test_translation_format_failure_when_empty(self):
        failure_type, _, _, format_compliant = classify_result(
            response=CANNED["translation_empty"], exception=None,
            canonical_answer=None, source="gsm8k", stage="translate",
        )
        self.assertEqual(failure_type, FailureType.TRANSLATION_FORMAT_FAILURE)
        self.assertIsNone(format_compliant)

    def test_translation_ok(self):
        failure_type, _, _, format_compliant = classify_result(
            response=CANNED["translation_ok"], exception=None,
            canonical_answer=None, source="gsm8k", stage="translate",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertIsNone(format_compliant)

    def test_invalid_answer_format_is_noncompliant(self):
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=CANNED["invalid_format"], exception=None, canonical_answer="A",
            source="mmlu_formal_logic", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.INVALID_ANSWER_FORMAT)
        self.assertEqual(extracted, "maybe purple")
        self.assertFalse(is_correct)
        self.assertFalse(format_compliant)  # tag present but unusable content is NOT compliant

    def test_correct_number_with_tag_is_compliant(self):
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=CANNED["correct_number"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertEqual(extracted, "109")
        self.assertTrue(is_correct)
        self.assertTrue(format_compliant)

    def test_correct_letter(self):
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=CANNED["correct_letter"], exception=None, canonical_answer="A",
            source="mmlu_formal_logic", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertTrue(is_correct)
        self.assertTrue(format_compliant)

    def test_substantively_incorrect_with_tag_is_compliant(self):
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=CANNED["wrong_number"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.SUBSTANTIVELY_INCORRECT)
        self.assertEqual(extracted, "42")
        self.assertFalse(is_correct)
        self.assertTrue(format_compliant)


class FallbackExtractionTests(unittest.TestCase):
    """Regression tests for the real pilot failure: qwen_3_6_27b produced a fully correct
    GSM8K derivation ending in \\boxed{109} with no <answer> tag at all — classify_result()
    must recover the answer via fallback, grade it correctly, but mark it noncompliant."""

    def test_gsm8k_boxed_fallback_is_correct_but_noncompliant(self):
        response = canned_response(
            "Step by step... 99 + 10 = 109\n\n$$\n\\boxed{109}\n$$"
        )
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertEqual(extracted, "109")
        self.assertTrue(is_correct)
        self.assertFalse(format_compliant)

    def test_gsm8k_final_answer_sentence_fallback(self):
        response = canned_response("...therefore the final answer is 109.")
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertEqual(extracted, "109")
        self.assertTrue(is_correct)
        self.assertFalse(format_compliant)

    def test_multiple_choice_boxed_letter_fallback(self):
        response = canned_response("Reasoning... the answer is \\boxed{A}")
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=response, exception=None, canonical_answer="A",
            source="mmlu_formal_logic", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertTrue(is_correct)
        self.assertFalse(format_compliant)

    def test_multiple_choice_parenthesized_letter_fallback(self):
        response = canned_response("Working through the options... I conclude (E) is correct.")
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=response, exception=None, canonical_answer="E",
            source="bbh_logical_deduction", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertTrue(is_correct)
        self.assertFalse(format_compliant)

    def test_causal_judgement_explicit_yes_fallback(self):
        response = canned_response("Considering the causal chain, the answer is Yes.")
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=response, exception=None, canonical_answer="YES",
            source="bbh_causal_judgement", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertTrue(is_correct)
        self.assertFalse(format_compliant)

    def test_causal_judgement_ilokano_wen_maps_to_yes(self):
        response = canned_response("Iti panagbaligbig, ti sungbat ket Wen.")
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=response, exception=None, canonical_answer="YES",
            source="bbh_causal_judgement", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertEqual(extracted, "Wen")
        self.assertTrue(is_correct)
        self.assertFalse(format_compliant)

    def test_causal_judgement_ilokano_saan_maps_to_no(self):
        response = canned_response("Ti sungbat ket Saan, gapu ta awan ti nadumaduma.")
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=response, exception=None, canonical_answer="NO",
            source="bbh_causal_judgement", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.CORRECT)
        self.assertTrue(is_correct)

    def test_no_fallback_pattern_matches_is_missing_answer(self):
        response = canned_response("I thought about it extensively but reached no conclusion.")
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(failure_type, FailureType.MISSING_ANSWER)
        self.assertIsNone(extracted)
        self.assertFalse(format_compliant)

    def test_tagged_answer_takes_priority_over_fallback(self):
        response = canned_response(
            "Some scratch work mentions \\boxed{999} but <answer>109</answer> is final."
        )
        failure_type, extracted, is_correct, format_compliant = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(extracted, "109")
        self.assertTrue(is_correct)
        self.assertTrue(format_compliant)


if __name__ == "__main__":
    unittest.main()
