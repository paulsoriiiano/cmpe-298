"""Token counting: model-native tokenizer where one is loadable (currently the two
HPC/vLLM-served Qwen models, via the lightweight `tokenizers` library and their public HF
tokenizer.json), tiktoken cl100k_base as a fallback/reference approximation everywhere else
(Anthropic and OpenAI don't expose a locally loadable tokenizer; Llama-3-8B-Instruct's repo
is gated and may not be loadable without auth).

Native counts are what actually determines real context usage and cost for the HPC models,
so they're the primary figures recorded (question_en_tokens, question_ilo_tokens,
translation_tokens, rationale_tokens, tokenization_tax_ratio). For HPC models specifically,
a failed native-tokenizer load is FATAL (raises immediately) rather than silently falling
back to tiktoken — a silent fallback would produce a tokenization-tax number that looks
valid but isn't measuring the actual model's tokenizer, which is the entire point of this
module for those models. tiktoken is only ever a fallback for providers with no native
tokenizer available at all (Anthropic, OpenAI) — see tokenizer_identity(), which reports
which one was actually used.

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


class UnresolvedTokenizerRevisionError(RuntimeError):
    """Raised when an HPC model's tokenizer revision would resolve to the mutable "main"
    branch instead of an immutable, reproducible reference."""


def _resolve_hpc_tokenizer_source(config, *, require_pinned_revision: bool = True) -> tuple[str, str]:
    """Single source of truth for which tokenizer repo/revision an HPC model's native
    tokenizer is loaded from — used by both _load_native_tokenizer() and
    tokenizer_identity(), so the reported identity always matches what was actually loaded
    (the same class of bug as the resume-key drift: never compute the same fact twice in
    two places that can drift apart).

    Precedence:
      source:   HPC_TOKENIZER_REPO env var > HPC_HF_REPO env var > config.tokenizer_repo >
                config.hf_repo > config.model_id.
      revision: HPC_TOKENIZER_REVISION env var > HPC_CHECKPOINT_COMMIT_SHA env var >
                HPC_MODEL_REVISION env var > config.revision > "main".

    HPC_CHECKPOINT_COMMIT_SHA/HPC_MODEL_REVISION are checked here (not just
    HPC_TOKENIZER_REVISION) because in practice the tokenizer ships alongside the model
    checkpoint in the same repo/revision — if an operator has already pinned the model's
    checkpoint commit or revision, that pin should apply to the tokenizer too unless a
    tokenizer-specific override is given.

    require_pinned_revision (default True) rejects a resolution that falls all the way
    through to the "main" default with UnresolvedTokenizerRevisionError — "main" is a
    mutable branch, not a reproducible reference, and silently tokenizing against whatever
    "main" happens to contain today would make the tokenization-tax measurement
    unreproducible without anyone noticing.
    """
    source = (
        os.environ.get("HPC_TOKENIZER_REPO")
        or os.environ.get("HPC_HF_REPO")
        or config.tokenizer_repo
        or config.hf_repo
        or config.model_id
    )
    revision = (
        os.environ.get("HPC_TOKENIZER_REVISION")
        or os.environ.get("HPC_CHECKPOINT_COMMIT_SHA")
        or os.environ.get("HPC_MODEL_REVISION")
        or config.revision
        or "main"
    )
    if require_pinned_revision and revision == "main":
        raise UnresolvedTokenizerRevisionError(
            f"No pinned tokenizer revision resolved for source {source!r} — refusing to "
            f"fall back to the mutable 'main' branch. Set one of HPC_TOKENIZER_REVISION, "
            f"HPC_CHECKPOINT_COMMIT_SHA, or HPC_MODEL_REVISION to an immutable commit SHA "
            f"so the tokenizer (and the tokenization-tax numbers it produces) can be "
            f"reproduced exactly."
        )
    return source, revision


def _load_native_tokenizer(model_key: str):
    """Loads the model's own tokenizer via the `tokenizers` library. Returns None (falls
    back to tiktoken) for non-HPC providers, which have no public tokenizer file to load in
    the first place (Anthropic, OpenAI). For HPC providers, a load failure is FATAL — raises
    RuntimeError immediately rather than silently returning None, since a silent tiktoken
    fallback for an HPC model would corrupt the tokenization-tax measurement without anyone
    noticing."""
    if model_key in _native_tokenizer_cache:
        return _native_tokenizer_cache[model_key]

    config = MODEL_REGISTRY[model_key]
    if config.provider != "hpc":
        _native_tokenizer_cache[model_key] = None
        return None

    source, revision = _resolve_hpc_tokenizer_source(config)
    from tokenizers import Tokenizer
    try:
        tokenizer = Tokenizer.from_pretrained(source, revision=revision)
    except Exception as e:  # noqa: BLE001 - re-raised as a clear, actionable RuntimeError
        raise RuntimeError(
            f"Failed to load the native tokenizer for {model_key!r} "
            f"(source={source!r}, revision={revision!r}): {type(e).__name__}: {e}. "
            f"This is fatal for HPC models — set HPC_TOKENIZER_REPO/HPC_HF_REPO/"
            f"HPC_TOKENIZER_REVISION in .env, or fix network/auth access, rather than "
            f"silently falling back to an approximate tokenizer for a local model."
        ) from e

    _native_tokenizer_cache[model_key] = tokenizer
    return tokenizer


def tokenizer_identity(model_key: str) -> tuple[str, str]:
    """Returns (tokenizer_model_id, tokenizer_revision) for whichever tokenizer is actually
    used to count tokens for this model_key. Raises for an HPC model whose native tokenizer
    can't load, or whose revision can't be pinned to something other than "main" — see
    _load_native_tokenizer() / _resolve_hpc_tokenizer_source()."""
    config = MODEL_REGISTRY[model_key]
    if config.provider == "hpc":
        _load_native_tokenizer(model_key)  # raises if unavailable; populates the cache
        return _resolve_hpc_tokenizer_source(config)
    return f"tiktoken/{TIKTOKEN_ENCODING}", tiktoken.__version__


def count_tokens_for_model(text: str | None, model_key: str) -> int:
    """Model-native count where a native tokenizer is required (HPC providers — raises if
    unavailable), else the tiktoken approximation. Special tokens are excluded
    (add_special_tokens=False) since we're counting the content itself, not a
    ready-to-run-through-the-model sequence with BOS/EOS/role markers added."""
    if not text:
        return 0
    tokenizer = _load_native_tokenizer(model_key)
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)
    return count_tokens_tiktoken(text)


def count_stage_tokens(
    *, model_key: str, question_en: str, question_ilo: str,
    translation: str | None, rationale: str | None,
) -> dict:
    """Per-(item, model) token breakdown. question_en/question_ilo are always counted (both
    are always available on every dataset item, regardless of which condition is running) so
    the "tokenization tax" of Ilokano vs. English can be compared across every record, not
    just pivot conditions. translation/rationale are counted only when actually produced for
    this record (None otherwise) — callers are responsible for passing rationale=None for
    direct-answer conditions (A_E0/A_I0), which have no rationale by design."""
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
