"""Experimental condition definitions.

See text_track/PROTOCOL.md for the full specification these mirror.
"""
from dataclasses import dataclass

PROTOCOL_VERSION = "protocol_v1"

# ---------------- Shared prompt fragments ---------------- #

ANSWER_RULE_V1 = (
    "After your reasoning, end your response with the final answer enclosed in "
    "<answer></answer> tags. The tags must contain ONLY the final value — a single "
    "number, letter, or short word (such as a multiple-choice option) — with no units, "
    "no explanation, and no extra text.\n"
    "Examples: <answer>109</answer>, <answer>A</answer>, <answer>Yes</answer>."
)

DIRECT_ANSWER_RULE_V1 = (
    "Respond with ONLY the final answer enclosed in <answer></answer> tags. Do not show "
    "any reasoning or explanation. The tags must contain ONLY the final value — a single "
    "number, letter, or short word — with no units, no explanation, and no extra text.\n"
    "Examples: <answer>109</answer>, <answer>A</answer>, <answer>Yes</answer>."
)

# ---------------- Single-call reasoning prompts ---------------- #

PROMPT_A_EE = (
    "You are an expert problem solver. Carefully read the problem the user provides and "
    "solve it. Show your complete step-by-step reasoning in English.\n" + ANSWER_RULE_V1
)

PROMPT_A_II = (
    "You are an expert problem solver who is fluent in Ilokano. The user will give you a "
    "problem written in Ilokano. Solve it and show your complete step-by-step reasoning "
    "ENTIRELY IN ILOKANO. Do not write your reasoning in English.\n" + ANSWER_RULE_V1
)

PROMPT_A_E0 = "You are an expert problem solver.\n" + DIRECT_ANSWER_RULE_V1

PROMPT_A_I0 = (
    "You are an expert problem solver who is fluent in Ilokano. The user will give you a "
    "problem written in Ilokano.\n" + DIRECT_ANSWER_RULE_V1
)

# ---------------- Staged pivot prompts (translate stage) ---------------- #
# Translation-only prompts: must NOT solve the problem, only translate it.

TRANSLATE_ILO_TO_EN_SYSTEM = (
    "You are a professional Ilokano-to-English translator. Translate the user's message "
    "from Ilokano into English as faithfully as possible, preserving all numbers, names, "
    "and multiple-choice option labels exactly. Do NOT solve the problem, do NOT explain "
    "it, and do NOT add any commentary. Output ONLY the English translation, with no "
    "preamble, no answer, and no <answer> tags."
)

TRANSLATE_EN_TO_ILO_SYSTEM = (
    "You are a professional English-to-Ilokano translator. Translate the user's message "
    "from English into Ilokano as faithfully as possible, preserving all numbers, names, "
    "and multiple-choice option labels exactly. Do NOT solve the problem, do NOT explain "
    "it, and do NOT add any commentary. Output ONLY the Ilokano translation, with no "
    "preamble, no answer, and no <answer> tags."
)

# ---------------- Staged pivot prompts (reasoning stage) ---------------- #
# These run in a FRESH context: the model sees ONLY the translated text produced above,
# never the original-language question and never the canonical answer.

REASON_EN_FROM_TRANSLATION_SYSTEM = (
    "You are an expert problem solver. The user will give you a problem statement "
    "(already translated into English by someone else). Solve it and show your complete "
    "step-by-step reasoning in English.\n" + ANSWER_RULE_V1
)

REASON_ILO_FROM_TRANSLATION_SYSTEM = (
    "You are an expert problem solver who is fluent in Ilokano. The user will give you a "
    "problem statement (already translated into Ilokano by someone else). Solve it and "
    "show your complete step-by-step reasoning ENTIRELY IN ILOKANO.\n" + ANSWER_RULE_V1
)


@dataclass(frozen=True)
class ConditionConfig:
    key: str                            # "A_EE" | "A_II" | "A_IE" | "A_EI" | "A_E0" | "A_I0"
    input_field: str                     # "question_en" | "question_ilo"
    requires_translation: bool
    translation_source_lang: str | None   # "en" | "ilo" | None
    translation_target_lang: str | None
    reasoning_lang: str | None             # "en" | "ilo" | None (None = direct-answer control)
    new_context_for_reasoning: bool
    prompt_version: str
    implemented: bool = True


CONDITIONS: dict[str, ConditionConfig] = {
    "A_EE": ConditionConfig(
        key="A_EE",
        input_field="question_en",
        requires_translation=False,
        translation_source_lang=None,
        translation_target_lang=None,
        reasoning_lang="en",
        new_context_for_reasoning=False,
        prompt_version="v1",
    ),
    "A_II": ConditionConfig(
        key="A_II",
        input_field="question_ilo",
        requires_translation=False,
        translation_source_lang=None,
        translation_target_lang=None,
        reasoning_lang="ilo",
        new_context_for_reasoning=False,
        prompt_version="v1",
    ),
    "A_IE": ConditionConfig(
        key="A_IE",
        input_field="question_ilo",
        requires_translation=True,
        translation_source_lang="ilo",
        translation_target_lang="en",
        reasoning_lang="en",
        new_context_for_reasoning=True,
        prompt_version="v1",
    ),
    "A_EI": ConditionConfig(
        key="A_EI",
        input_field="question_en",
        requires_translation=True,
        translation_source_lang="en",
        translation_target_lang="ilo",
        reasoning_lang="ilo",
        new_context_for_reasoning=True,
        prompt_version="v1",
    ),
    "A_E0": ConditionConfig(
        key="A_E0",
        input_field="question_en",
        requires_translation=False,
        translation_source_lang=None,
        translation_target_lang=None,
        reasoning_lang=None,
        new_context_for_reasoning=False,
        prompt_version="v1",
    ),
    "A_I0": ConditionConfig(
        key="A_I0",
        input_field="question_ilo",
        requires_translation=False,
        translation_source_lang=None,
        translation_target_lang=None,
        reasoning_lang=None,
        new_context_for_reasoning=False,
        prompt_version="v1",
    ),
}

# A_E0 / A_I0 run only over this fixed, deterministically-sampled subset (see
# text_track/scripts/generate_stratified_subset.py and text_track/data/stratified_subset_300.json)
# rather than the full 1000 items, per PROTOCOL.md section 1.
STRATIFIED_SUBSET_CONDITIONS = {"A_E0", "A_I0"}

IMPLEMENTED_CONDITIONS = [c.key for c in CONDITIONS.values() if c.implemented]


def get_condition(key: str) -> ConditionConfig:
    try:
        return CONDITIONS[key]
    except KeyError:
        raise ValueError(f"Unknown condition: {key!r}. Known: {list(CONDITIONS)}") from None


def require_implemented(key: str) -> ConditionConfig:
    cond = get_condition(key)
    if not cond.implemented:
        raise NotImplementedError(
            f"Condition {key!r} is defined in conditions.py but not yet implemented. "
            f"See text_track/PROTOCOL.md section 1."
        )
    return cond
