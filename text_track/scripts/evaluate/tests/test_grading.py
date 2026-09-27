"""Unit tests for grading.classify_result() covering every FailureType branch, the
truncation/degeneration separation, the is_correct semantics table, and the fallback-
extraction regression tests for responses that skip the required <answer> tag.
No network calls: uses canned ModelResponse objects only (see fakes.CANNED).
"""
import unittest

from ..grading import (
    REPETITION_DEGENERATION_THRESHOLD, TRUNCATION_FINISH_REASONS, FailureType, classify_result,
    compute_repetition_ratio, has_text_beyond_answer, strip_answer_content,
)
from .fakes import CANNED, canned_response


class ClassifyResultTests(unittest.TestCase):
    def test_infrastructure_api_failure_on_exception(self):
        result = classify_result(
            response=None, exception=ConnectionError("boom"), canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.INFRASTRUCTURE_API_FAILURE)
        self.assertIsNone(result.extracted_answer)
        self.assertIsNone(result.is_correct)  # correctness genuinely unknown — no response at all
        self.assertIsNone(result.format_compliant)
        self.assertIsNone(result.is_truncated)
        self.assertIsNone(result.repetition_ratio)
        self.assertIsNone(result.degeneration_candidate)

    def test_non_repetitive_truncation_is_incorrect_not_degenerate(self):
        result = classify_result(
            response=CANNED["truncated"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.TRUNCATION)
        self.assertFalse(result.format_compliant)
        self.assertIs(result.is_correct, False)  # model failed to complete the task
        self.assertTrue(result.is_truncated)
        self.assertFalse(result.degeneration_candidate)

    def test_openai_style_length_finish_reason_is_truncated(self):
        response = canned_response("cut off mid-sen", finish_reason="length")
        result = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertTrue(result.is_truncated)
        self.assertEqual(result.failure_type, FailureType.TRUNCATION)

    def test_anthropic_style_max_tokens_finish_reason_is_truncated(self):
        # Regression: Anthropic's Messages API uses "max_tokens", not "length", for
        # truncation — checking only "length" silently missed every Anthropic truncation.
        response = canned_response("cut off mid-sen", finish_reason="max_tokens")
        result = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertTrue(result.is_truncated)
        self.assertEqual(result.failure_type, FailureType.TRUNCATION)
        self.assertIn("max_tokens", TRUNCATION_FINISH_REASONS)
        self.assertIn("length", TRUNCATION_FINISH_REASONS)

    def test_refusal_is_incorrect(self):
        result = classify_result(
            response=CANNED["refusal"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.REFUSAL)
        self.assertFalse(result.format_compliant)
        self.assertIs(result.is_correct, False)

    def test_repetitive_with_stop_finish_is_degenerate_and_incorrect(self):
        result = classify_result(
            response=CANNED["repetitive"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.REPETITION_DEGENERATION)
        self.assertFalse(result.format_compliant)
        self.assertIs(result.is_correct, False)
        self.assertFalse(result.is_truncated)  # CANNED["repetitive"] finishes with "stop"
        self.assertTrue(result.degeneration_candidate)
        self.assertGreaterEqual(result.repetition_ratio, REPETITION_DEGENERATION_THRESHOLD)

    def test_repetitive_with_length_finish_preserves_both_facts(self):
        """The motivating real-pilot case: a response can be BOTH truncated AND degenerate.
        failure_type picks repetition_degeneration (checked before truncation in the
        precedence order), but is_truncated stays True — neither fact is discarded."""
        response = canned_response(" ".join(["loop token repeat"] * 40), finish_reason="length")
        result = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.REPETITION_DEGENERATION)
        self.assertTrue(result.is_truncated)
        self.assertTrue(result.degeneration_candidate)
        self.assertIs(result.is_correct, False)

    def test_missing_answer_is_incorrect(self):
        result = classify_result(
            response=CANNED["missing_answer"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.MISSING_ANSWER)
        self.assertIsNone(result.extracted_answer)
        self.assertFalse(result.format_compliant)
        self.assertIs(result.is_correct, False)

    def test_translation_format_failure_when_solved_instead_of_translated(self):
        result = classify_result(
            response=CANNED["translation_solved_instead"], exception=None,
            canonical_answer=None, source="gsm8k", stage="translate",
        )
        self.assertEqual(result.failure_type, FailureType.TRANSLATION_FORMAT_FAILURE)
        self.assertIsNone(result.format_compliant)
        self.assertIsNone(result.is_correct)  # correctness not applicable to translation rows

    def test_translation_format_failure_when_empty(self):
        result = classify_result(
            response=CANNED["translation_empty"], exception=None,
            canonical_answer=None, source="gsm8k", stage="translate",
        )
        self.assertEqual(result.failure_type, FailureType.TRANSLATION_FORMAT_FAILURE)
        self.assertIsNone(result.format_compliant)

    def test_translation_ok(self):
        result = classify_result(
            response=CANNED["translation_ok"], exception=None,
            canonical_answer=None, source="gsm8k", stage="translate",
        )
        self.assertEqual(result.failure_type, FailureType.TRANSLATION_COMPLETED)
        self.assertIsNone(result.format_compliant)
        self.assertIsNone(result.is_correct)

    def test_translate_stage_never_auto_classifies_degeneration(self):
        """Caution from the precedence design: applying automatic degeneration detection to
        the translate stage risks a false positive blocking the reasoning stage from
        running. The score is still recorded, but degeneration_candidate is always False and
        failure_type is never REPETITION_DEGENERATION for stage="translate"."""
        repetitive_translation = canned_response(" ".join(["loop token repeat"] * 40))
        result = classify_result(
            response=repetitive_translation, exception=None, canonical_answer=None,
            source="gsm8k", stage="translate",
        )
        self.assertNotEqual(result.failure_type, FailureType.REPETITION_DEGENERATION)
        self.assertFalse(result.degeneration_candidate)
        self.assertGreater(result.repetition_ratio, REPETITION_DEGENERATION_THRESHOLD)

    def test_invalid_answer_format_is_noncompliant_and_incorrect(self):
        result = classify_result(
            response=CANNED["invalid_format"], exception=None, canonical_answer="A",
            source="mmlu_formal_logic", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.INVALID_ANSWER_FORMAT)
        self.assertEqual(result.extracted_answer, "maybe purple")
        self.assertIs(result.is_correct, False)
        self.assertFalse(result.format_compliant)

    def test_correct_number_with_tag_is_compliant(self):
        result = classify_result(
            response=CANNED["correct_number"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertEqual(result.extracted_answer, "109")
        self.assertTrue(result.is_correct)
        self.assertTrue(result.format_compliant)

    def test_tagged_wen_is_correct_but_format_noncompliant(self):
        """Real-pilot-driven case: <answer>Wen</answer> is semantically YES (correct), but
        the format contract specifically requires the canonical YES/NO vocabulary — using a
        same-meaning Ilokano token is not the same as following the contract, even though a
        tag was present."""
        response = canned_response("Rason...\n<answer>Wen</answer>")
        result = classify_result(
            response=response, exception=None, canonical_answer="YES",
            source="bbh_causal_judgement", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertEqual(result.extracted_answer, "Wen")
        self.assertTrue(result.is_correct)
        self.assertFalse(result.format_compliant)

    def test_tagged_yes_uppercase_is_correct_and_compliant(self):
        response = canned_response("Reasoning...\n<answer>YES</answer>")
        result = classify_result(
            response=response, exception=None, canonical_answer="YES",
            source="bbh_causal_judgement", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertTrue(result.is_correct)
        self.assertTrue(result.format_compliant)

    def test_tagged_yes_lowercase_is_correct_and_compliant_after_normalization(self):
        response = canned_response("Reasoning...\n<answer>Yes</answer>")
        result = classify_result(
            response=response, exception=None, canonical_answer="YES",
            source="bbh_causal_judgement", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertTrue(result.is_correct)
        self.assertTrue(result.format_compliant)

    def test_correct_letter(self):
        result = classify_result(
            response=CANNED["correct_letter"], exception=None, canonical_answer="A",
            source="mmlu_formal_logic", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertTrue(result.is_correct)
        self.assertTrue(result.format_compliant)

    def test_substantively_incorrect_with_tag_is_compliant(self):
        result = classify_result(
            response=CANNED["wrong_number"], exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.SUBSTANTIVELY_INCORRECT)
        self.assertEqual(result.extracted_answer, "42")
        self.assertFalse(result.is_correct)
        self.assertTrue(result.format_compliant)


class DegenerationThresholdTests(unittest.TestCase):
    """REPETITION_DEGENERATION_THRESHOLD is NOT YET CALIBRATED from real pilot data (see
    grading.py's module-level comment and inspect_repetition_scores.py) — these tests only
    verify the boundary/comparison logic itself (>=), not the specific threshold value."""

    def test_ratio_exactly_at_threshold_is_a_candidate(self):
        self.assertGreaterEqual(REPETITION_DEGENERATION_THRESHOLD, 0.0)
        # Build a response whose repetition_ratio is engineered to sit at/above threshold.
        # 4 distinct tokens repeated so unique/total ngrams gives ratio >= threshold.
        text = " ".join(["a", "b", "c", "d"] * 20)  # period-4 cycle -> exactly 4 distinct 4-grams
        ratio = compute_repetition_ratio(text)
        self.assertGreaterEqual(ratio, REPETITION_DEGENERATION_THRESHOLD)
        result = classify_result(
            response=canned_response(text), exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.REPETITION_DEGENERATION)

    def test_normal_long_reasoning_is_not_falsely_flagged_degenerate(self):
        # A long but varied, legitimate multi-step derivation — no repeated 4-grams.
        text = (
            "Let the ages of Darrell and Allen be in the ratio of seven to eleven. "
            "Let their current ages be seven x and eleven x respectively, where x is a "
            "positive integer common factor. The total combined age today equals one "
            "hundred sixty two years, so seven x plus eleven x equals one hundred sixty "
            "two, which simplifies to eighteen x equals one hundred sixty two, giving x "
            "equal to nine. Allen's current age is therefore eleven times nine, which "
            "equals ninety nine years old today. Ten years from now, Allen will be "
            "ninety nine plus ten, which is one hundred nine years old.\n<answer>109</answer>"
        )
        ratio = compute_repetition_ratio(text)
        self.assertLess(ratio, REPETITION_DEGENERATION_THRESHOLD)
        result = classify_result(
            response=canned_response(text), exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertNotEqual(result.failure_type, FailureType.REPETITION_DEGENERATION)
        self.assertFalse(result.degeneration_candidate)


class RepetitionRatioFormulaTests(unittest.TestCase):
    def test_empty_text_is_zero(self):
        self.assertEqual(compute_repetition_ratio(""), 0.0)
        self.assertEqual(compute_repetition_ratio(None), 0.0)

    def test_shorter_than_ngram_size_is_zero(self):
        self.assertEqual(compute_repetition_ratio("one two three"), 0.0)  # 3 tokens, n=4

    def test_all_unique_ngrams_is_zero(self):
        text = "the quick brown fox jumps over the lazy dog today"
        self.assertEqual(compute_repetition_ratio(text), 0.0)

    def test_fully_repeated_text_approaches_one(self):
        text = " ".join(["repeat"] * 50)
        ratio = compute_repetition_ratio(text)
        self.assertGreater(ratio, 0.9)

    def test_formula_is_one_minus_unique_over_total(self):
        # 8 tokens -> 5 four-grams; "a b c d" repeated twice gives 4 unique out of 5 total...
        # construct something with a known, hand-computed ratio instead for clarity:
        text = "a b c a b c a b"  # tokens: a b c a b c a b (8 tokens, 5 4-grams)
        tokens = text.split()
        n = 4
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        expected = 1 - (len(set(ngrams)) / len(ngrams))
        self.assertAlmostEqual(compute_repetition_ratio(text), expected)


class FallbackExtractionTests(unittest.TestCase):
    """Regression tests for the real pilot failure: qwen_3_6_27b produced a fully correct
    GSM8K derivation ending in \\boxed{109} with no <answer> tag at all — classify_result()
    must recover the answer via fallback, grade it correctly, but mark it noncompliant."""

    def test_gsm8k_boxed_fallback_is_correct_but_noncompliant(self):
        response = canned_response(
            "Step by step... 99 + 10 = 109\n\n$$\n\\boxed{109}\n$$"
        )
        result = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertEqual(result.extracted_answer, "109")
        self.assertTrue(result.is_correct)
        self.assertFalse(result.format_compliant)

    def test_gsm8k_final_answer_sentence_fallback(self):
        response = canned_response("...therefore the final answer is 109.")
        result = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertEqual(result.extracted_answer, "109")
        self.assertTrue(result.is_correct)
        self.assertFalse(result.format_compliant)

    def test_multiple_choice_boxed_letter_fallback(self):
        response = canned_response("Reasoning... the answer is \\boxed{A}")
        result = classify_result(
            response=response, exception=None, canonical_answer="A",
            source="mmlu_formal_logic", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertTrue(result.is_correct)
        self.assertFalse(result.format_compliant)

    def test_multiple_choice_parenthesized_letter_fallback(self):
        response = canned_response("Working through the options... I conclude (E) is correct.")
        result = classify_result(
            response=response, exception=None, canonical_answer="E",
            source="bbh_logical_deduction", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertTrue(result.is_correct)
        self.assertFalse(result.format_compliant)

    def test_causal_judgement_explicit_yes_fallback(self):
        response = canned_response("Considering the causal chain, the answer is Yes.")
        result = classify_result(
            response=response, exception=None, canonical_answer="YES",
            source="bbh_causal_judgement", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertTrue(result.is_correct)
        self.assertFalse(result.format_compliant)

    def test_causal_judgement_ilokano_wen_maps_to_yes(self):
        response = canned_response("Iti panagbaligbig, ti sungbat ket Wen.")
        result = classify_result(
            response=response, exception=None, canonical_answer="YES",
            source="bbh_causal_judgement", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertEqual(result.extracted_answer, "Wen")
        self.assertTrue(result.is_correct)
        self.assertFalse(result.format_compliant)

    def test_causal_judgement_ilokano_saan_maps_to_no(self):
        response = canned_response("Ti sungbat ket Saan, gapu ta awan ti nadumaduma.")
        result = classify_result(
            response=response, exception=None, canonical_answer="NO",
            source="bbh_causal_judgement", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.CORRECT)
        self.assertTrue(result.is_correct)

    def test_no_fallback_pattern_matches_is_missing_answer(self):
        response = canned_response("I thought about it extensively but reached no conclusion.")
        result = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.failure_type, FailureType.MISSING_ANSWER)
        self.assertIsNone(result.extracted_answer)
        self.assertFalse(result.format_compliant)
        self.assertIs(result.is_correct, False)

    def test_tagged_answer_takes_priority_over_fallback(self):
        response = canned_response(
            "Some scratch work mentions \\boxed{999} but <answer>109</answer> is final."
        )
        result = classify_result(
            response=response, exception=None, canonical_answer="109",
            source="gsm8k", stage="reason",
        )
        self.assertEqual(result.extracted_answer, "109")
        self.assertTrue(result.is_correct)
        self.assertTrue(result.format_compliant)


class RationalePresenceHeuristicTests(unittest.TestCase):
    """A naive bool(generated_rationale) check treats <answer>109</answer> as containing a
    rationale, which is exactly the answer-only behavior a rationale-presence check needs to
    catch. has_text_beyond_answer() strips the tag/fallback forms first."""

    def test_answer_tag_only_is_false(self):
        self.assertFalse(has_text_beyond_answer("<answer>109</answer>"))

    def test_boxed_only_is_false(self):
        self.assertFalse(has_text_beyond_answer("$$\\boxed{109}$$"))

    def test_one_sentence_explanation_plus_answer_is_true(self):
        text = "Allen will be 109 in ten years.\n<answer>109</answer>"
        self.assertTrue(has_text_beyond_answer(text))

    def test_multi_step_explanation_plus_answer_is_true(self):
        text = (
            "Let x be the common ratio unit. 7x + 11x = 162, so x = 9. "
            "Allen's current age is 11*9 = 99. In 10 years: 99 + 10 = 109.\n"
            "<answer>109</answer>"
        )
        self.assertTrue(has_text_beyond_answer(text))

    def test_empty_response_is_false(self):
        self.assertFalse(has_text_beyond_answer(""))
        self.assertFalse(has_text_beyond_answer(None))

    def test_strip_answer_content_removes_tag_and_boxed(self):
        text = "Reasoning here.\n$$\\boxed{109}$$\n<answer>109</answer>"
        self.assertEqual(strip_answer_content(text), "Reasoning here.")

    def test_final_answer_sentence_with_leading_article_is_answer_only(self):
        # Regression: "The final answer is 109." used to leave "The" behind as a false
        # positive "explanation" because only the "final answer is X" phrase itself (not
        # the leading article) was being stripped.
        self.assertFalse(has_text_beyond_answer("The final answer is 109."))
        self.assertEqual(strip_answer_content("The final answer is 109."), "")

    def test_final_answer_sentence_after_real_work_still_has_text(self):
        text = "Let x=9. The final answer is 109."
        self.assertTrue(has_text_beyond_answer(text))
        self.assertEqual(strip_answer_content(text), "Let x=9.")

    def test_standalone_option_letter_is_answer_only(self):
        self.assertFalse(has_text_beyond_answer("(A)"))
        self.assertFalse(has_text_beyond_answer("A"))

    def test_standalone_yes_no_is_answer_only(self):
        self.assertFalse(has_text_beyond_answer("YES"))
        self.assertFalse(has_text_beyond_answer("NO."))

    def test_standalone_wen_saan_is_answer_only(self):
        self.assertFalse(has_text_beyond_answer("Wen"))
        self.assertFalse(has_text_beyond_answer("Saan."))

    def test_yes_within_a_real_explanation_still_has_text(self):
        # "Yes" is only stripped when it's the ENTIRE remainder, not whenever the word
        # appears — a real explanation that happens to start with "Yes," must not be erased.
        text = "Yes, because the causal chain holds.\n<answer>Yes</answer>"
        self.assertTrue(has_text_beyond_answer(text))


if __name__ == "__main__":
    unittest.main()
