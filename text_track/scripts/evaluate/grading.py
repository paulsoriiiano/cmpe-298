"""Answer extraction, normalization, grading, and failure-type classification.

extract_answer/normalize_answer/_first_number/_matches/grade are carried over from the
original text_track/scripts/evaluate.py, with grade() simplified to compare against a
single canonical_answer (dataset.jsonl now precomputes this) instead of two per-language
golds.

Real pilot output surfaced a gap: some models (e.g. qwen_3_6_27b) skip the required
<answer> tag entirely and instead end with LaTeX \\boxed{...} or a "final answer is X"
sentence. classify_result() now falls back to source-aware extraction when no tag is
found, but tracks tag presence separately as `format_compliant` — a fallback-derived
correct answer is still semantically CORRECT, just format-noncompliant.
"""
import re
from enum import Enum

from .models import ModelResponse


class FailureType(str, Enum):
    CORRECT = "correct"
    SUBSTANTIVELY_INCORRECT = "substantively_incorrect"
    INVALID_ANSWER_FORMAT = "invalid_answer_format"
    MISSING_ANSWER = "missing_answer"
    TRANSLATION_COMPLETED = "translation_completed"
    TRANSLATION_FORMAT_FAILURE = "translation_format_failure"
    REFUSAL = "refusal"
    TRUNCATION = "truncation"
    REPETITION_DEGENERATION = "repetition_degeneration"
    PARSER_FAILURE = "parser_failure"
    INFRASTRUCTURE_API_FAILURE = "infrastructure_api_failure"


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

# Ilokano <-> English tokens that are semantically equivalent but not string-equal, so
# normalize_answer() alone can't relate them (mirrors add_canonical_answer.py's mapping).
_ILOKANO_YES_NO = {"wen": "yes", "saan": "no"}

_BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')
_FINAL_ANSWER_RE = re.compile(r'final answer(?:\s+is)?\s*[:\-]?\s*([^\n.]+)', re.IGNORECASE)
_YES_NO_WORD_RE = re.compile(r'\b(yes|no|wen|saan)\b', re.IGNORECASE)
_OPTION_LETTER_RE = re.compile(r'\(([A-E])\)')

_MULTIPLE_CHOICE_SOURCES = {"bbh_logical_deduction", "mmlu_conceptual_physics", "mmlu_formal_logic"}


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


def _semantic_class(text: str | None) -> str | None:
    """normalize_answer() output, mapped through the Ilokano yes/no lookup so 'Wen' and
    'Yes' (and 'Saan'/'No') compare equal for grading purposes."""
    n = normalize_answer(text)
    if n is None:
        return None
    return _ILOKANO_YES_NO.get(n, n)


def _matches(extracted: str | None, gold: str | None) -> bool:
    if extracted is None or gold is None:
        return False
    if _semantic_class(extracted) == _semantic_class(gold):
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


def extract_fallback_answer(text: str | None, source: str | None) -> str | None:
    """Source-aware fallback extraction for responses that skip the required <answer> tag.
    Tried in order: LaTeX \\boxed{...} (most common for GSM8K-style math), an explicit
    "final answer is X" sentence, then source-specific standalone tokens (YES/NO/Wen/Saan
    for causal judgement, a parenthesized option letter for multiple choice). Returns the
    LAST match of whichever pattern hits, mirroring extract_answer()'s "last mention wins"
    behavior for multi-step reasoning that restates earlier candidate answers."""
    if not text:
        return None

    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    final = _FINAL_ANSWER_RE.findall(text)
    if final:
        return final[-1].strip()

    if source == "bbh_causal_judgement":
        words = _YES_NO_WORD_RE.findall(text)
        if words:
            return words[-1]

    if source in _MULTIPLE_CHOICE_SOURCES:
        letters = _OPTION_LETTER_RE.findall(text)
        if letters:
            return letters[-1]

    return None


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
) -> tuple[FailureType, str | None, bool | None, bool | None]:
    """Classify a completed API call into a FailureType.

    Returns (failure_type, extracted_answer, is_correct, format_compliant).
    extracted_answer/is_correct are None for stages/failures where grading doesn't apply
    (translate stage, infra failure, parser failure). format_compliant tracks whether the
    required <answer> tag was actually present — independent of correctness, since a
    fallback-extracted answer (e.g. from \\boxed{...}) can be semantically CORRECT while
    still being format-noncompliant. format_compliant is None where the tag contract isn't
    applicable (translate stage, infra/parser failure).
    """
    if exception is not None:
        return FailureType.INFRASTRUCTURE_API_FAILURE, None, None, None

    assert response is not None

    if response.finish_reason == "length":
        return FailureType.TRUNCATION, None, None, False

    if _REFUSAL_PATTERNS.search(response.text or ""):
        return FailureType.REFUSAL, None, None, False

    if _repetition_ratio(response.text or "") > 0.3:
        return FailureType.REPETITION_DEGENERATION, None, None, False

    if stage == "translate":
        if looks_like_translation_format_failure(response.text):
            return FailureType.TRANSLATION_FORMAT_FAILURE, None, None, None
        # TRANSLATION_COMPLETED means only: the API call succeeded, a nonempty translation
        # candidate came back, and it didn't contain an answer tag. It does NOT mean the
        # translation is linguistically correct or faithful — that's a separate manual
        # annotation pass (see generate_annotation_template.py / PROTOCOL.md section 9),
        # not something this heuristic can judge.
        return FailureType.TRANSLATION_COMPLETED, None, None, None

    try:
        text = response.text or ""
        tagged = extract_answer(text)
        if tagged is not None:
            extracted = tagged
            format_compliant = True
        else:
            extracted = extract_fallback_answer(text, source)
            format_compliant = False

        if extracted is None:
            return FailureType.MISSING_ANSWER, None, None, False

        shape = _ANSWER_SHAPE_BY_SOURCE.get(source) if source else None

        # Lenient check (via _semantic_class): is this even a plausible/gradable answer
        # for this source? "Wen"/"Saan" pass here — they're valid Ilokano tokens for the
        # causal-judgement contract's underlying semantics, just not the literal YES/NO
        # vocabulary the contract specifies. A response that isn't even in the right
        # ballpark (e.g. a sentence where a bare letter was expected) fails here.
        lenient_class = _semantic_class(extracted)
        if shape is not None and lenient_class is not None and not shape.match(lenient_class.upper()):
            return FailureType.INVALID_ANSWER_FORMAT, extracted, False, False

        # Strict check (via normalize_answer, no Ilokano mapping): does the LITERAL content
        # match the canonical vocabulary the format contract requires? "Wen"/"Saan" fail
        # here even though they're semantically gradable — the contract specifically wants
        # YES/NO, not a same-meaning Ilokano token. This only diverges from the lenient
        # check for bbh_causal_judgement; every other source's shape check is identical
        # either way, since there's no per-language token divergence there.
        strict_class = normalize_answer(extracted)
        strict_ok = shape is None or strict_class is None or bool(shape.match(strict_class.upper()))
        format_compliant = format_compliant and strict_ok

        is_correct = grade(extracted, canonical_answer) if canonical_answer is not None else False
        if is_correct:
            return FailureType.CORRECT, extracted, True, format_compliant
        return FailureType.SUBSTANTIVELY_INCORRECT, extracted, False, format_compliant
    except Exception:  # noqa: BLE001 - our own bug, not the model's/API's fault
        return FailureType.PARSER_FAILURE, None, None, None
