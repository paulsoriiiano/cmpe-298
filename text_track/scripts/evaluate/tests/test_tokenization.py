"""Unit tests for tokenization.py. Loading a REAL native tokenizer needs network access to
the HF Hub, so most tests here either force the tiktoken fallback path (via a non-HPC
model_key) or mock tokenizers.Tokenizer.from_pretrained for the HPC-provider code path —
no test in this file makes a real network call.
"""
import os
import unittest
from unittest import mock

from .. import tokenization
from ..tokenization import (
    TIKTOKEN_ENCODING, count_stage_tokens, count_tokens_for_model, count_tokens_tiktoken,
    descriptive_counts, tokenizer_identity,
)

NON_HPC_MODEL_KEY = "claude_sonnet_4_6"  # never attempts native tokenizer loading
HPC_MODEL_KEY = "qwen_3_6_27b"


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


class NativeHpcTokenizerTests(unittest.TestCase):
    """Mocks tokenizers.Tokenizer.from_pretrained — no real network calls. Covers the fatal-
    on-failure behavior and correct source/revision/add_special_tokens plumbing for HPC
    models, per PROTOCOL.md section 8: a silent tiktoken fallback for an HPC model would
    corrupt the tokenization-tax measurement without anyone noticing."""

    def setUp(self):
        tokenization._native_tokenizer_cache.clear()
        self.addCleanup(tokenization._native_tokenizer_cache.clear)

    def _mock_tokenizer(self, num_tokens=3):
        mock_tokenizer = mock.Mock()
        mock_tokenizer.encode.return_value = mock.Mock(ids=list(range(num_tokens)))
        return mock_tokenizer

    def test_from_pretrained_called_with_repo_and_revision_env_overrides(self):
        env = {"HPC_TOKENIZER_REPO": "my-org/my-tokenizer", "HPC_TOKENIZER_REVISION": "deadbeef"}
        with mock.patch.dict(os.environ, env), \
             mock.patch("tokenizers.Tokenizer.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = self._mock_tokenizer()
            count_tokens_for_model("hello", HPC_MODEL_KEY)
            mock_from_pretrained.assert_called_once_with("my-org/my-tokenizer", revision="deadbeef")

    def test_falls_back_to_hpc_hf_repo_when_tokenizer_repo_absent(self):
        os.environ.pop("HPC_TOKENIZER_REPO", None)
        with mock.patch.dict(os.environ, {"HPC_HF_REPO": "my-org/hf-repo"}), \
             mock.patch("tokenizers.Tokenizer.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = self._mock_tokenizer()
            count_tokens_for_model("hello", HPC_MODEL_KEY)
            called_source = mock_from_pretrained.call_args.args[0]
            self.assertEqual(called_source, "my-org/hf-repo")

    def test_encode_called_with_add_special_tokens_false(self):
        with mock.patch("tokenizers.Tokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = self._mock_tokenizer(num_tokens=1)
            mock_from_pretrained.return_value = mock_tokenizer
            count_tokens_for_model("hi", HPC_MODEL_KEY)
            mock_tokenizer.encode.assert_called_once_with("hi", add_special_tokens=False)

    def test_hpc_tokenizer_load_failure_is_fatal(self):
        with mock.patch(
            "tokenizers.Tokenizer.from_pretrained", side_effect=OSError("network unreachable"),
        ):
            with self.assertRaises(RuntimeError):
                count_tokens_for_model("hi", HPC_MODEL_KEY)

    def test_hpc_tokenizer_load_failure_stops_tokenizer_identity_too(self):
        with mock.patch(
            "tokenizers.Tokenizer.from_pretrained", side_effect=OSError("network unreachable"),
        ):
            with self.assertRaises(RuntimeError):
                tokenizer_identity(HPC_MODEL_KEY)


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
