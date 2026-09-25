# Experimental Protocol

- Protocol version: `protocol_v1`
- Date: 2026-09-23
- Pinned dataset version: `dataset_conference_v1.1` (see `data/dataset_manifest.json`)
- **STATUS: not yet piloted. No model calls have been made under this protocol.**

## 1. Conditions

| Condition | Input | Steps | Calls | Maps to |
|---|---|---|---|---|
| `A_EE` (primary) | `question_en` | question → English rationale → canonical answer | 1 | `evaluate.py`'s `pass1_english_baseline` |
| `A_II` (primary) | `question_ilo` | question → Ilokano rationale → canonical answer | 1 | `evaluate.py`'s `pass2_native_ilokano` |
| `A_IE` (primary, pivot) | `question_ilo` | Ilokano→English translation (separate call, saved) → **new/fresh context** → English rationale using *only* the saved translation → canonical answer | 2 | **needs refactor** — see gap below |
| `A_EI` (secondary) | `question_en` | English→Ilokano translation (separate call, saved) → **new/fresh context** → Ilokano rationale using only the saved translation → canonical answer | 2 | needs new pass |
| `A_E0` (optional control) | `question_en` | question → canonical answer, no rationale | 1 | needs new pass |
| `A_I0` (optional control) | `question_ilo` | question → canonical answer, no rationale | 1 | needs new pass |

`A_EE`/`A_II`/`A_IE` are the minimum publishable experiment. `A_EI` is
secondary and, along with `A_E0`/`A_I0`, is the first thing cut if the
schedule slips.

**Design gap (must be fixed in the Phase 3 evaluator refactor, not assumed
away):** the current `pass3_english_pivot` (`PASS3_SYSTEM` in `evaluate.py`)
does translate-then-solve as **one continuous prompt/response**. The
protocol requires `A_IE` (and `A_EI`) to be **two separate model calls**,
with the reasoning call started in a fresh context that only sees the saved
translation — not the original-language question. This is not yet
implemented anywhere in the repo.

### `A_E0`/`A_I0` sampling

Run on a stratified 300-item subset, proportional to each source's share of
the full 1,000 (30% of each, rounded):

| Source | Full count | Subset (30%) |
|---|---|---|
| gsm8k | 400 | 120 |
| bbh_logical_deduction | 250 | 75 |
| bbh_causal_judgement | 50 | 15 |
| mmlu_conceptual_physics | 174 | 52 |
| mmlu_formal_logic | 126 | 38 |
| **Total** | **1000** | **300** |

Every model must receive exactly the same selected IDs.

## 2. Canonical answer format per source

`answer_en`/`answer_ilo` retain their original per-source format (full CoT
for GSM8K, raw letters/words). `canonical_answer` (added in dataset version
`v1.1`) is the single, language-agnostic, format-normalized final value used
for grading, identical across every condition:

| Source | Canonical format | Derivation |
|---|---|---|
| `gsm8k` | bare integer/decimal, e.g. `"109"` | strip through `"#### "` in `answer_en` (same as `evaluate.py`'s `parse_expected_answer()`) |
| `bbh_logical_deduction` | bare letter `A`–`E`, e.g. `"E"` | strip parens from `"(E)"`, uppercase |
| `bbh_causal_judgement` | `"YES"` / `"NO"` | `answer_en` `"Yes"→"YES"`, `"No"→"NO"` (Ilokano `"Wen"`/`"Saan"` map to the same class — see `add_canonical_answer.py`) |
| `mmlu_conceptual_physics`, `mmlu_formal_logic` | bare letter `A`–`D` | `answer_en.strip().upper()` (no-op today) |

Note BBH logical deduction's range is `A`–`E` (5-way), while MMLU's is
`A`–`D` (4-way) — same letter convention, different option counts.

Every `canonical_answer` value in the frozen dataset was validated against
`evaluate.py`'s own `parse_expected_answer()`/`normalize_answer()` before
being written — see `text_track/scripts/add_canonical_answer.py`. Validation
passed for all 1,000 items with zero failures.

## 3. Output schema

Every condition, including the direct-answer controls, uses the same
tagged-answer contract already defined by `_ANSWER_RULE` in `evaluate.py`:
the model must end its response with `<answer>VALUE</answer>`, containing
only the final value (no units, no explanation).

Grading must record two separate axes, per the conference plan:
- **Semantic correctness**: does the extracted, normalized value match
  `canonical_answer` regardless of surface form (e.g. `"(A)"`, `"a"`, `"Wen"`
  are all valid surface forms of the same semantic answer).
- **Format compliance**: did the model actually emit a well-formed
  `<answer>` tag containing (after normalization) exactly the canonical
  token, with no extra content.

`evaluate.py` currently only returns a single `is_correct` boolean and does
not separate these two axes — adding a `format_ok` field alongside
`is_correct` is a scoped requirement for **Phase 3 (evaluator refactor)**,
not implemented here.

## 4. Prompts

Current prompts in `evaluate.py` (`PASS1_SYSTEM`, `PASS2_SYSTEM`,
`PASS3_SYSTEM`, `_ANSWER_RULE`) are labeled `prompt_v0`. New prompts needed
for `A_IE`'s two-call redesign, `A_EI`, `A_E0`, and `A_I0` will be drafted in
Phase 3/4 and labeled `prompt_v1`+. Prompt text is not finalized in this
document; once drafted, prompts should live alongside the other pass
prompts in `evaluate.py` (or a dedicated prompts module if the refactor
warrants it), with each semantic change bumping the version number.

## 5. Model settings (placeholder)

| Model | Temperature | Max tokens | Thinking mode | Provider |
|---|---|---|---|---|
| `claude-sonnet-4-6` | 0.0 | 2048 | off | Anthropic |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 0.0 | 2048 | n/a | HF router |

These are the values currently hard-coded in `evaluate.py` and are inherited
defaults, not yet ratified as the paper's final experimental settings. Final
model roster and settings are decided in **Phase 4 (pilot)**.

## 6. Statistical comparisons

Primary comparison family (paired, since every condition runs over the same
1,000 item IDs — McNemar-style paired tests, not independent-samples):

- `A_EE` vs `A_II` — overall language gap
- `A_II` vs `A_IE` — English-pivot benefit
- `A_EE` vs `A_IE` — remaining pivot gap

Report both pooled (all 1,000 items) and per-source breakdowns, since chance
baselines and difficulty differ substantially between GSM8K (open-ended
numeric) and the multiple-choice/binary sources.

## 7. Known provenance gaps (documented, not resolved)

These are carried over from `data/audit_summary.md` and are **not**
addressed by this protocol or by `add_canonical_answer.py`:

1. **MMLU answer encoding.** HuggingFace's raw MMLU `answer` field is a
   0-indexed integer (0–3). At some point before the dataset was first
   committed, this was converted to a letter (`A`–`D`). No script currently
   in the repo (`collect_dataset.py`, `translate.py`, `convert.py`) performs
   this conversion — it is not reproducible from tracked history. Values
   were spot-checked as correct during the linguistic audit.
2. **BBH causal judgement Ilokano tokens.** `answer_ilo` values `"Wen"`/`"Saan"`
   are not producible by any code path currently in `translate.py` (which
   would copy `"Yes"`/`"No"` verbatim for non-GSM8K sources). This was very
   likely a manual correction, consistent with the ~35 per-item fix commits
   noted in `audit_summary.md`.

Neither gap is fabricated a resolution here; both are stated as
unreproducible-from-current-scripts for the record.

## 8. Non-goals for this phase

- Does **not** refactor `evaluate.py`'s grading/parsing/prompt logic
  (Phase 3).
- Does **not** run any pilot or full experiment under any condition
  (Phase 4).
- Does **not** modify `analyze.py`.
- Does **not** attempt to resolve the provenance gaps in section 7.
