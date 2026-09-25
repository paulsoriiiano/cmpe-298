"""Unit tests for tokenization.py's approximate system/user/rendered token counts."""
import unittest

from ..tokenization import TOKENIZER_ID, count_prompt_tokens, count_tokens


class TokenizationTests(unittest.TestCase):
    def test_count_tokens_nonempty(self):
        self.assertGreater(count_tokens("What is 17 + 25?"), 0)

    def test_count_tokens_empty(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_tokens(None), 0)

    def test_count_prompt_tokens_breakdown(self):
        result = count_prompt_tokens("You are a helpful assistant.", "What is 2+2?")
        self.assertEqual(result["tokenizer_id"], TOKENIZER_ID)
        self.assertIsInstance(result["tokenizer_revision"], str)
        self.assertGreater(result["system_prompt_tokens"], 0)
        self.assertGreater(result["user_input_tokens"], 0)
        # Rendered should be roughly the sum of the two parts (allowing for the joining
        # newline token) — a rough sanity bound, not an exact equality.
        self.assertGreaterEqual(
            result["rendered_prompt_tokens"],
            result["system_prompt_tokens"] + result["user_input_tokens"],
        )

    def test_longer_system_prompt_yields_more_system_tokens(self):
        short = count_prompt_tokens("Solve it.", "2+2?")
        long = count_prompt_tokens(
            "You are an expert problem solver. " * 10, "2+2?",
        )
        self.assertGreater(long["system_prompt_tokens"], short["system_prompt_tokens"])


if __name__ == "__main__":
    unittest.main()
