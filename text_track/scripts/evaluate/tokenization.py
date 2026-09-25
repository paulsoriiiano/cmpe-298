"""Token counting: model-native tokenizer where one is loadable (currently the two
HPC/vLLM-served Qwen models, via the lightweight `tokenizers` library and their public HF
tokenizer.json), tiktoken cl100k_base as a fallback/reference approximation everywhere else
(Anthropic and OpenAI don't expose a locally loadable tokenizer; Llama-3-8B-Instruct's repo
is gated and may not be loadable without auth).

Native counts are what actually determines real context usage and cost for the HPC models,
so they're the primary figures recorded (question_en_tokens, question_ilo_tokens,
translation_tokens, rationale_tokens, tokenization_tax_ratio). tiktoken is kept only as an
optional, clearly-labeled reference measurement for providers with no native tokenizer
available — see tokenizer_identity(), which reports which one was actually used.

Provider-reported input_tokens/output_tokens (already on ModelResponse) remain the exact,
authoritative figures for cost accounting; nothing here overrides those.
"""
import os

import tiktoken

from .models import MODEL_REGISTRY

TIKTOKEN_ENCODING = "cl100k_base"

_tiktoken_encoding = None
_native_tokenizer_cache: dict[str, object] = {}


def _get_tiktoken_encoding():
    global _tiktoken_encoding
    if _tiktoken_encoding is None:
        _tiktoken_encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return _tiktoken_encoding


def count_tokens_tiktoken(text: str | None) -> int:
    """Reference/fallback approximation — NOT the exact tokenizer for any specific model."""
    if not text:
        return 0
    return len(_get_tiktoken_encoding().encode(text))


def _load_native_tokenizer(model_key: str):
    """Best-effort: load the model's own tokenizer via the `tokenizers` library. Returns
    None (falls back to tiktoken) if unavailable — no network access, gated repo, or a
    provider with no public tokenizer file at all (Anthropic, OpenAI)."""
    if model_key in _native_tokenizer_cache:
        return _native_tokenizer_cache[model_key]

    config = MODEL_REGISTRY[model_key]
    tokenizer = None
    if config.provider == "hpc":
        source = (
            os.environ.get("HPC_TOKENIZER_REPO")
            or config.tokenizer_repo
            or config.hf_repo
            or config.model_id
        )
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_pretrained(source)
        except Exception:  # noqa: BLE001 - network/auth/missing-file — fall back to tiktoken
            tokenizer = None

    _native_tokenizer_cache[model_key] = tokenizer
    return tokenizer


def tokenizer_identity(model_key: str) -> tuple[str, str]:
    """Returns (tokenizer_model_id, tokenizer_revision) for whichever tokenizer is actually
    used to count tokens for this model_key."""
    tokenizer = _load_native_tokenizer(model_key)
    if tokenizer is not None:
        config = MODEL_REGISTRY[model_key]
        revision = os.environ.get("HPC_TOKENIZER_REVISION") or config.revision or "unknown"
        tokenizer_id = os.environ.get("HPC_TOKENIZER_REPO") or config.tokenizer_repo or config.hf_repo or config.model_id
        return tokenizer_id, revision
    return f"tiktoken/{TIKTOKEN_ENCODING}", tiktoken.__version__


def count_tokens_for_model(text: str | None, model_key: str) -> int:
    """Model-native count where a native tokenizer loaded successfully, else the tiktoken
    approximation. Check tokenizer_identity(model_key) to know which was actually used."""
    if not text:
        return 0
    tokenizer = _load_native_tokenizer(model_key)
    if tokenizer is not None:
        return len(tokenizer.encode(text).ids)
    return count_tokens_tiktoken(text)


def count_stage_tokens(
    *, model_key: str, question_en: str, question_ilo: str,
    translation: str | None, rationale: str | None,
) -> dict:
    """Per-(item, model) token breakdown. question_en/question_ilo are always counted (both
    are always available on every dataset item, regardless of which condition is running) so
    the "tokenization tax" of Ilokano vs. English can be compared across every record, not
    just pivot conditions. translation/rationale are counted only when actually produced for
    this record (None otherwise)."""
    tokenizer_model_id, tokenizer_revision = tokenizer_identity(model_key)

    question_en_tokens = count_tokens_for_model(question_en, model_key)
    question_ilo_tokens = count_tokens_for_model(question_ilo, model_key)
    translation_tokens = count_tokens_for_model(translation, model_key) if translation else None
    rationale_tokens = count_tokens_for_model(rationale, model_key) if rationale else None

    tokenization_tax_ratio = (
        question_ilo_tokens / question_en_tokens if question_en_tokens else None
    )

    return {
        "question_en_tokens": question_en_tokens,
        "question_ilo_tokens": question_ilo_tokens,
        "translation_tokens": translation_tokens,
        "rationale_tokens": rationale_tokens,
        "tokenizer_model_id": tokenizer_model_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenization_tax_ratio": tokenization_tax_ratio,
    }


def descriptive_counts(text: str | None) -> dict:
    """Word/character counts — supplementary descriptive stats only, not a substitute for
    either provider token counts (exact, for cost) or native tokenizer counts (for the
    tokenization-tax claim)."""
    if not text:
        return {"word_count": 0, "char_count": 0}
    return {"word_count": len(text.split()), "char_count": len(text)}
