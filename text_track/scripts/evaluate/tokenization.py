"""Local token counting, used to break a call's prompt into system/user/rendered pieces.

Provider `usage` objects (already captured in ModelResponse.input_tokens/output_tokens)
only report a single combined prompt-token total, so they can't show how much of the
context budget is the (fixed, per-condition) system prompt vs. the (per-item) user
question. That split matters for comparing conditions whose system prompts differ in
length (e.g. the two-part reasoning rule vs. the direct-answer rule).

This uses tiktoken's cl100k_base encoding as a single consistent approximation across all
providers/models. It is NOT the exact tokenizer for Claude, Llama, Qwen, or GPT-5.2 —
tokenizer_id/tokenizer_revision are recorded on every ResultRecord precisely so this
approximation is traceable and swappable later, rather than silently treated as exact.
The exact, authoritative input/output counts remain each provider's own `usage` object.
"""
import tiktoken

TOKENIZER_ID = "tiktoken/cl100k_base"
TOKENIZER_REVISION = tiktoken.__version__

_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def count_prompt_tokens(system: str, user: str) -> dict:
    """Returns system/user/rendered counts for one (system, user) prompt pair.
    rendered_prompt_tokens is an approximate concatenation, not a real chat-template
    rendering (role markers/special tokens aren't modeled) — useful as a rough sanity
    check against the provider's own reported input_tokens, not as an exact figure."""
    system_tokens = count_tokens(system)
    user_tokens = count_tokens(user)
    rendered_tokens = count_tokens(f"{system}\n{user}")
    return {
        "system_prompt_tokens": system_tokens,
        "user_input_tokens": user_tokens,
        "rendered_prompt_tokens": rendered_tokens,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
