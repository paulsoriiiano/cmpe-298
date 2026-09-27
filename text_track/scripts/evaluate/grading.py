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
from dataclasses import dataclass
from enum import Enum

from .models import ModelResponse

# NOT YET CALIBRATED FROM REAL DATA. This is a placeholder pending manual inspection of a
# real pilot run's repetition_ratio distribution (known-degenerate responses vs. normal
# Ilokano/English/long-legitimate-reasoning responses) via
# text_track/scripts/inspect_repetition_scores.py — see PROTOCOL.md section 3. Do not treat
# this value as preregistered; confirm/adjust it against real pilot data before the full run.
REPETITION_DEGENERATION_THRESHOLD = 0.5

# Anthropic's Messages API uses "max_tokens" for its truncation finish_reason; OpenAI-style
# APIs (OpenAI direct, HF router, vLLM) use "length". Checking only "length" silently missed
# every Anthropic truncation.
TRUNCATION_FINISH_REASONS = {"length", "max_tokens"}


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


def compute_repetition_ratio(text: str | None, n: int = 4) -> float:
    """1 - (unique n-grams / total n-grams), by whitespace token, using 4-grams by default.
    0.0 means every n-gram is distinct (no repetition); values approaching 1.0 mean the
    output is mostly the same n-gram repeated. Computed for every response (not just ones
    already suspected of degenerating) so a threshold can be chosen from the real
    distribution — see REPETITION_DEGENERATION_THRESHOLD and
    text_track/scripts/inspect_repetition_scores.py."""
    if not text:
        return 0.0
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1 - (unique / total) if total else 0.0


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


_ANSWER_TAG_RE = re.compile(r'<answer>.*?</answer>', re.IGNORECASE | re.DOTALL)

# Strips the WHOLE "final answer is X" clause, including any leading words back to the start
# of the sentence/string (e.g. "The final answer is 109." -> ""), unlike _FINAL_ANSWER_RE
# above (which only captures from "final answer" onward, for extraction purposes — leaving
# "The" behind was exactly the bug this separate stripping-only pattern fixes: a naive
# per-match removal left leading filler words like "The" as a false-positive "explanation").
_FINAL_ANSWER_SENTENCE_RE = re.compile(
    r'[^.\n]*\bfinal answer\b[^.\n]*[.!]?', re.IGNORECASE
)

# A remainder consisting of ONLY a standalone option letter or yes/no/wen/saan token (with
# optional surrounding parens/punctuation) is still answer-only, even with no <answer> tag,
# \boxed{}, or "final answer" phrase at all.
_STANDALONE_ANSWER_RE = re.compile(r'^\(?(?:[A-E]|YES|NO|Wen|Saan)\)?[.!]?$', re.IGNORECASE)


def strip_answer_content(text: str | None) -> str:
    """Remove the answer tag and known fallback-answer forms (\\boxed{...}, a "final answer
    is X" sentence, or a standalone option letter / yes-no-wen-saan token) from a response,
    leaving whatever text remains. Used to check whether a response actually contains an
    explanation, as opposed to only an answer — a naive `bool(generated_rationale)` check is
    wrong here: <answer>109</answer> is nonempty but is exactly the answer-only behavior a
    rationale-presence check needs to catch."""
    if not text:
        return ""
    remainder = _ANSWER_TAG_RE.sub("", text)
    remainder = _BOXED_RE.sub("", remainder)
    remainder = _FINAL_ANSWER_SENTENCE_RE.sub("", remainder)
    # Leftover LaTeX display-math delimiters ($$...$$) around a now-removed \boxed{...}
    # aren't meaningful text on their own.
    remainder = remainder.replace("$$", "").replace("$", "")
    remainder = remainder.strip()
    if _STANDALONE_ANSWER_RE.match(remainder):
        return ""
    return remainder


def has_text_beyond_answer(text: str | None) -> bool:
    """True if strip_answer_content(text) leaves any word characters — not just leftover
    punctuation/whitespace. This is a heuristic diagnostic, not a substitute for human
    judgment of whether that remainder is an actual explanation (see
    generate_annotation_template.py's rationale_present, which is intentionally left for a
    human annotator rather than inferred automatically)."""
    return re.search(r'\w', strip_answer_content(text)) is not None


def looks_like_translation_format_failure(translated_text: str) -> bool:
    """Heuristic: a translation-stage response that is empty, wraps itself in <answer>
    tags (i.e. tried to solve instead of translate), or is implausibly short/long."""
    if not translated_text or not translated_text.strip():
        return True
    if extract_answer(translated_text) is not None:
        return True
    return False


@dataclass(frozen=True)
class ClassificationResult:
    failure_type: FailureType
    extracted_answer: str | None
    is_correct: bool | None
    format_compliant: bool | None
    is_truncated: bool | None       # finish_reason in TRUNCATION_FINISH_REASONS — independent of failure_type
    repetition_ratio: float | None   # compute_repetition_ratio(response.text) — always recorded
    degeneration_candidate: bool | None  # repetition_ratio >= threshold; only ever True for reason/direct


def classify_result(
    *,
    response: ModelResponse | None,
    exception: Exception | None,
    canonical_answer: str | None,
    source: str | None,
    stage: str,
) -> ClassificationResult:
    """Classify a completed API call. Precedence (first match wins):
    1. Infrastructure/API failure (exception)
    2. Parser/internal evaluator failure (our own code broke, not the model's fault)
    3. Translation-stage validation (stage == "translate" is fully self-contained; also
       checks refusal/truncation before falling back to the format-failure heuristic — a
       nonempty but truncated/refused translation is NOT translation_completed)
    4. Refusal
    5. Repetition degeneration (reason/direct stages only — see below)
    6. Non-degenerate truncation
    7. Missing answer
    8. Answer format and substantive correctness

    is_correct semantics: True/False for anything the model actually produced (a wrong
    answer, a truncated response, a degenerate loop, a refusal, a missing/malformed answer
    are all False — the model failed to complete the task, which IS an incorrect outcome).
    None means correctness genuinely couldn't be evaluated: translation-stage rows (no
    canonical answer applies), infrastructure failures (no response at all), and parser
    failures (our own bug, not a judgment about the model's output).

    is_truncated/repetition_ratio are recorded independently of failure_type whenever a
    response exists — e.g. a record can be failure_type=repetition_degeneration with
    is_truncated=True, preserving both facts instead of collapsing them into one field.
    degeneration_candidate (repetition_ratio >= REPETITION_DEGENERATION_THRESHOLD) is only
    ever True for reason/direct stages: applying it to the translate stage risks a false
    positive blocking the subsequent reasoning stage from running at all (see run.py's
    staged-pivot check against TRANSLATION_COMPLETED), so translation rows only ever get the
    score recorded, never an automatic degeneration classification.
    """
    if exception is not None:
        return ClassificationResult(
            failure_type=FailureType.INFRASTRUCTURE_API_FAILURE, extracted_answer=None,
            is_correct=None, format_compliant=None, is_truncated=None,
            repetition_ratio=None, degeneration_candidate=None,
        )

    assert response is not None
    text = response.text or ""
    is_truncated = response.finish_reason in TRUNCATION_FINISH_REASONS
    repetition_ratio = compute_repetition_ratio(text)
    degeneration_candidate = (
        stage in ("reason", "direct") and repetition_ratio >= REPETITION_DEGENERATION_THRESHOLD
    )

    try:
        if stage == "translate":
            # A nonempty translation can still be unusable: refused or truncated
            # mid-sentence. Checking only emptiness/answer-tag-misuse (the original
            # looks_like_translation_format_failure() check) let a truncated or refused
            # translation through as TRANSLATION_COMPLETED, and the reasoning stage would
            # then proceed from an incomplete translation.
            if _REFUSAL_PATTERNS.search(text):
                failure_type = FailureType.REFUSAL
            elif is_truncated:
                failure_type = FailureType.TRUNCATION
            elif looks_like_translation_format_failure(response.text):
                failure_type = FailureType.TRANSLATION_FORMAT_FAILURE
            else:
                # TRANSLATION_COMPLETED means only: the API call succeeded, a nonempty
                # translation candidate came back, it wasn't refused or truncated, and it
                # didn't contain an answer tag. It does NOT mean the translation is
                # linguistically correct or faithful — that's a separate manual annotation
                # pass (generate_annotation_template.py / PROTOCOL.md section 9), not
                # something this heuristic can judge.
                failure_type = FailureType.TRANSLATION_COMPLETED
            return ClassificationResult(
                failure_type=failure_type, extracted_answer=None, is_correct=None,
                format_compliant=None, is_truncated=is_truncated,
                repetition_ratio=repetition_ratio, degeneration_candidate=False,
            )

        # reason / direct stages from here on.
        if _REFUSAL_PATTERNS.search(text):
            return ClassificationResult(
                failure_type=FailureType.REFUSAL, extracted_answer=None, is_correct=False,
                format_compliant=False, is_truncated=is_truncated,
                repetition_ratio=repetition_ratio, degeneration_candidate=degeneration_candidate,
            )

        if degeneration_candidate:
            return ClassificationResult(
                failure_type=FailureType.REPETITION_DEGENERATION, extracted_answer=None,
                is_correct=False, format_compliant=False, is_truncated=is_truncated,
                repetition_ratio=repetition_ratio, degeneration_candidate=True,
            )

        if is_truncated:
            return ClassificationResult(
                failure_type=FailureType.TRUNCATION, extracted_answer=None, is_correct=False,
                format_compliant=False, is_truncated=True, repetition_ratio=repetition_ratio,
                degeneration_candidate=degeneration_candidate,
            )

        tagged = extract_answer(text)
        if tagged is not None:
            extracted = tagged
            format_compliant = True
        else:
            extracted = extract_fallback_answer(text, source)
            format_compliant = False

        if extracted is None:
            return ClassificationResult(
                failure_type=FailureType.MISSING_ANSWER, extracted_answer=None,
                is_correct=False, format_compliant=False, is_truncated=is_truncated,
                repetition_ratio=repetition_ratio, degeneration_candidate=degeneration_candidate,
            )

        shape = _ANSWER_SHAPE_BY_SOURCE.get(source) if source else None

        # Lenient check (via _semantic_class): is this even a plausible/gradable answer
        # for this source? "Wen"/"Saan" pass here — they're valid Ilokano tokens for the
        # causal-judgement contract's underlying semantics, just not the literal YES/NO
        # vocabulary the contract specifies. A response that isn't even in the right
        # ballpark (e.g. a sentence where a bare letter was expected) fails here.
        lenient_class = _semantic_class(extracted)
        if shape is not None and lenient_class is not None and not shape.match(lenient_class.upper()):
            return ClassificationResult(
                failure_type=FailureType.INVALID_ANSWER_FORMAT, extracted_answer=extracted,
                is_correct=False, format_compliant=False, is_truncated=is_truncated,
                repetition_ratio=repetition_ratio, degeneration_candidate=degeneration_candidate,
            )

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
        failure_type = FailureType.CORRECT if is_correct else FailureType.SUBSTANTIVELY_INCORRECT
        return ClassificationResult(
            failure_type=failure_type, extracted_answer=extracted, is_correct=is_correct,
            format_compliant=format_compliant, is_truncated=is_truncated,
            repetition_ratio=repetition_ratio, degeneration_candidate=degeneration_candidate,
        )
    except Exception:  # noqa: BLE001 - our own bug, not the model's/API's fault
        return ClassificationResult(
            failure_type=FailureType.PARSER_FAILURE, extracted_answer=None, is_correct=None,
            format_compliant=None, is_truncated=is_truncated, repetition_ratio=repetition_ratio,
            degeneration_candidate=degeneration_candidate,
        )
