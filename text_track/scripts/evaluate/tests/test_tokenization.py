"""Unit tests for tokenization.py. Native (HPC) tokenizer loading needs network access to
the HF Hub, so these tests force the tiktoken fallback path (via a non-HPC model_key) rather
than depending on real network calls — the native-vs-fallback selection logic itself
(_load_native_tokenizer returning None for non-"hpc" providers) is exercised directly.
"""
import unittest

from ..tokenization import (
    TIKTOKEN_ENCODING, count_stage_tokens, count_tokens_for_model, count_tokens_tiktoken,
    descriptive_counts, tokenizer_identity,
)

NON_HPC_MODEL_KEY = "claude_sonnet_4_6"  # never attempts native tokenizer loading


class TiktokenFallbackTests(unittest.TestCase):
    def test_count_tokens_tiktoken_nonempty(self):
        self.assertGreater(count_tokens_tiktoken("What is 17 + 25?"), 0)

    def test_count_tokens_tiktoken_empty(self):
        self.assertEqual(count_tokens_tiktoken(""), 0)
        self.assertEqual(count_tokens_tiktoken(None), 0)

    def test_non_hpc_model_falls_back_to_tiktoken(self):
        tokenizer_id, _ = tokenizer_identity(NON_HPC_MODEL_KEY)
        self.assertEqual(tokenizer_id, f"tiktoken/{TIKTOKEN_ENCODING}")

    def test_count_tokens_for_model_matches_tiktoken_for_non_hpc(self):
        text = "What is 17 + 25?"
        self.assertEqual(
            count_tokens_for_model(text, NON_HPC_MODEL_KEY), count_tokens_tiktoken(text),
        )


class StageTokenCountsTests(unittest.TestCase):
    def test_counts_both_questions_regardless_of_condition(self):
        result = count_stage_tokens(
            model_key=NON_HPC_MODEL_KEY, question_en="What is 2+2?",
            question_ilo="Mano ti 2+2?", translation=None, rationale=None,
        )
        self.assertGreater(result["question_en_tokens"], 0)
        self.assertGreater(result["question_ilo_tokens"], 0)
        self.assertIsNone(result["translation_tokens"])
        self.assertIsNone(result["rationale_tokens"])

    def test_tokenization_tax_ratio_computed(self):
        result = count_stage_tokens(
            model_key=NON_HPC_MODEL_KEY, question_en="Hi", question_ilo="Kablaaw",
            translation=None, rationale=None,
        )
        self.assertIsInstance(result["tokenization_tax_ratio"], float)
        self.assertAlmostEqual(
            result["tokenization_tax_ratio"],
            result["question_ilo_tokens"] / result["question_en_tokens"],
        )

    def test_translation_and_rationale_counted_when_present(self):
        result = count_stage_tokens(
            model_key=NON_HPC_MODEL_KEY, question_en="Hi", question_ilo="Kablaaw",
            translation="Translated text here.", rationale="Reasoning text here.",
        )
        self.assertGreater(result["translation_tokens"], 0)
        self.assertGreater(result["rationale_tokens"], 0)

    def test_tokenizer_identity_fields_present(self):
        result = count_stage_tokens(
            model_key=NON_HPC_MODEL_KEY, question_en="Hi", question_ilo="Kablaaw",
            translation=None, rationale=None,
        )
        self.assertEqual(result["tokenizer_model_id"], f"tiktoken/{TIKTOKEN_ENCODING}")
        self.assertIsInstance(result["tokenizer_revision"], str)


class DescriptiveCountsTests(unittest.TestCase):
    def test_word_and_char_counts(self):
        result = descriptive_counts("one two three")
        self.assertEqual(result["word_count"], 3)
        self.assertEqual(result["char_count"], len("one two three"))

    def test_empty_text(self):
        result = descriptive_counts("")
        self.assertEqual(result["word_count"], 0)
        self.assertEqual(result["char_count"], 0)


if __name__ == "__main__":
    unittest.main()
