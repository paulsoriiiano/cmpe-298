"""Answer extraction, normalization, grading, and failure-type classification.

extract_answer/normalize_answer/_first_number/_matches/grade are carried over from the
original text_track/scripts/evaluate.py, with grade() simplified to compare against a
single canonical_answer (dataset.jsonl now precomputes this) instead of two per-language
golds.
"""
import re
from enum import Enum

from .models import ModelResponse


class FailureType(str, Enum):
    CORRECT = "correct"
    SUBSTANTIVELY_INCORRECT = "substantively_incorrect"
    INVALID_ANSWER_FORMAT = "invalid_answer_format"
    MISSING_ANSWER = "missing_answer"
    TRANSLATION_FORMAT_FAILURE = "translation_format_failure"
    REFUSAL = "refusal"
    TRUNCATION = "truncation"
    REPETITION_DEGENERATION = "repetition_degeneration"
    PARSER_FAILURE = "parser_failure"
    INFRASTRUCTURE_API_FAILURE = "infrastructure_api_failure"


# format_compliant is None for the two categories below (the failure isn't about output shape)
_FORMAT_NOT_APPLICABLE = {FailureType.PARSER_FAILURE, FailureType.INFRASTRUCTURE_API_FAILURE}
_FORMAT_NON_COMPLIANT = {
    FailureType.MISSING_ANSWER,
    FailureType.TRANSLATION_FORMAT_FAILURE,
    FailureType.REFUSAL,
    FailureType.TRUNCATION,
    FailureType.REPETITION_DEGENERATION,
}

_ANSWER_SHAPE_BY_SOURCE = {
    "gsm8k": re.compile(r"^-?\d+(\.\d+)?$"),
    "bbh_logical_deduction": re.compile(r"^[A-E]$"),
    "bbh_causal_judgement": re.compile(r"^(YES|NO)$"),
    "mmlu_conceptual_physics": re.compile(r"^[A-D]$"),
    "mmlu_formal_logic": re.compile(r"^[A-D]$"),
}

_REFUSAL_PATTERNS = re.compile(
    r"\b(i cannot|i can't|i'm unable to|i am unable to|i won't|i will not|"
    r"as an ai( language model)?,? i)\b",
    re.IGNORECASE,
)


def extract_answer(text: str) -> str | None:
    """Extract the text inside the LAST well-formed <answer>...</answer> tag pair."""
    matches = re.findall(r'<answer>((?:(?!<answer>).)*?)</answer>', text, re.IGNORECASE | re.DOTALL)
    for m in reversed(matches):
        if m.strip():
            return m.strip()
    return None


def normalize_answer(text: str | None) -> str | None:
    """Normalize for comparison: lowercase, drop commas, strip surrounding
    brackets/parens/punctuation, and remove common answer prefixes."""
    if text is None:
        return None
    t = text.strip().lower().replace(",", "")
    t = t.strip("().[]:;")
    for prefix in ("the answer is ", "final answer ", "answer ", "option "):
        if t.startswith(prefix):
            t = t[len(prefix):]
    return t.strip("().[]:; ")


def _first_number(text: str | None) -> str | None:
    if text is None:
        return None
    m = re.search(r'-?\d+(?:\.\d+)?', text.replace(",", ""))
    return m.group(0) if m else None


def _matches(extracted: str | None, gold: str | None) -> bool:
    if extracted is None or gold is None:
        return False
    if normalize_answer(extracted) == normalize_answer(gold):
        return True
    ne, ng = _first_number(extracted), _first_number(gold)
    if ne is not None and ng is not None:
        try:
            return float(ne) == float(ng)
        except ValueError:
            return False
    return False


def grade(extracted: str | None, canonical_answer: str) -> bool:
    """Correct if the extracted answer matches the item's canonical_answer."""
    return _matches(extracted, canonical_answer)


def _repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of n-grams (by whitespace token) that recur more than once. High values
    indicate the model looped/degenerated rather than produced varied text."""
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    seen = {}
    for g in ngrams:
        seen[g] = seen.get(g, 0) + 1
    repeated = sum(1 for count in seen.values() if count > 1)
    return repeated / len(seen) if seen else 0.0


def looks_like_translation_format_failure(translated_text: str) -> bool:
    """Heuristic: a translation-stage response that is empty, wraps itself in <answer>
    tags (i.e. tried to solve instead of translate), or is implausibly short/long."""
    if not translated_text or not translated_text.strip():
        return True
    if extract_answer(translated_text) is not None:
        return True
    return False


def classify_result(
    *,
    response: ModelResponse | None,
    exception: Exception | None,
    canonical_answer: str | None,
    source: str | None,
    stage: str,
) -> tuple[FailureType, str | None, bool | None]:
    """Classify a completed API call into a FailureType.

    Returns (failure_type, extracted_answer, is_correct). extracted_answer/is_correct are
    None for stages/failures where grading doesn't apply (translate stage, infra failure,
    parser failure).
    """
    if exception is not None:
        return FailureType.INFRASTRUCTURE_API_FAILURE, None, None

    assert response is not None

    if response.finish_reason == "length":
        return FailureType.TRUNCATION, None, None

    if _REFUSAL_PATTERNS.search(response.text or ""):
        return FailureType.REFUSAL, None, None

    if _repetition_ratio(response.text or "") > 0.3:
        return FailureType.REPETITION_DEGENERATION, None, None

    if stage == "translate":
        if looks_like_translation_format_failure(response.text):
            return FailureType.TRANSLATION_FORMAT_FAILURE, None, None
        return FailureType.CORRECT, None, None  # "correct" here just means "usable translation"

    try:
        extracted = extract_answer(response.text or "")
        if extracted is None:
            return FailureType.MISSING_ANSWER, None, None

        normalized = normalize_answer(extracted)
        shape = _ANSWER_SHAPE_BY_SOURCE.get(source) if source else None
        if shape is not None and normalized is not None and not shape.match(normalized.upper()):
            return FailureType.INVALID_ANSWER_FORMAT, extracted, False

        is_correct = grade(extracted, canonical_answer) if canonical_answer is not None else False
        if is_correct:
            return FailureType.CORRECT, extracted, True
        return FailureType.SUBSTANTIVELY_INCORRECT, extracted, False
    except Exception:  # noqa: BLE001 - our own bug, not the model's/API's fault
        return FailureType.PARSER_FAILURE, None, None


def format_compliant_for(failure_type: FailureType) -> bool | None:
    if failure_type in _FORMAT_NOT_APPLICABLE:
        return None
    if failure_type in _FORMAT_NON_COMPLIANT:
        return False
    return True
