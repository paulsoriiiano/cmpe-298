# Experimental Protocol

- Protocol version: `protocol_v1`
- Date: 2026-09-23 (implementation status updated 2026-09-24)
- Pinned dataset version: `dataset_conference_v1.1` (see `data/dataset_manifest.json`)
- **STATUS: implemented in `text_track/scripts/evaluate/`, not yet piloted. No real
  model API calls have been made under this protocol (only fake-client tests).**

## 1. Conditions

| Condition | Input | Steps | Calls | Status |
|---|---|---|---|---|
| `A_EE` (primary) | `question_en` | question → English rationale → canonical answer | 1 | implemented |
| `A_II` (primary) | `question_ilo` | question → Ilokano rationale → canonical answer | 1 | implemented |
| `A_IE` (primary, pivot) | `question_ilo` | Ilokano→English translation (separate call, saved) → **new/fresh context** → English rationale using *only* the saved translation → canonical answer | 2 | implemented |
| `A_EI` (secondary) | `question_en` | English→Ilokano translation (separate call, saved) → **new/fresh context** → Ilokano rationale using only the saved translation → canonical answer | 2 | implemented |
| `A_E0` (control) | `question_en` | question → canonical answer, no rationale | 1 | implemented (stratified 300-item subset only) |
| `A_I0` (control) | `question_ilo` | question → canonical answer, no rationale | 1 | implemented (stratified 300-item subset only) |

`A_EE`/`A_II`/`A_IE` are the minimum publishable experiment. `A_EI` and
`A_E0`/`A_I0` are lower priority and, per the schedule, are the first things
cut if time runs out during full execution — but all six are implemented in
the evaluator, so nothing is blocked on further coding to run any of them.

`A_IE`/`A_EI` are implemented as **two separate model calls** in
`run.py`'s `_run_staged_pivot_condition()`: a translation call whose output
is saved as its own `ResultRecord` before the reasoning call starts, and a
reasoning call in a fresh context (no shared message history) that receives
only the saved translation — never the original-language question or the
canonical answer.

### `A_E0`/`A_I0` sampling

Run on a fixed, deterministically-sampled 300-item subset, proportional to
each source's share of the full 1,000 (30% of each, rounded). Generated once
by `text_track/scripts/generate_stratified_subset.py` (seed 42) into the
static artifact `text_track/data/stratified_subset_300.json` — not
resampled at run time, so every model call against these conditions uses
exactly the same 300 IDs regardless of machine or Python version:

| Source | Full count | Subset (30%) |
|---|---|---|
| gsm8k | 400 | 120 |
| bbh_logical_deduction | 250 | 75 |
| bbh_causal_judgement | 50 | 15 |
| mmlu_conceptual_physics | 174 | 52 |
| mmlu_formal_logic | 126 | 38 |
| **Total** | **1000** | **300** |

Verified: the generated subset matches this table exactly (see script
output). `run.py` restricts `A_E0`/`A_I0` to these 300 IDs automatically;
every other condition still runs over the full 1,000.

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
`normalize_answer()` (now in `text_track/scripts/evaluate/grading.py`)
before being written — see `text_track/scripts/add_canonical_answer.py`.
Validation passed for all 1,000 items with zero failures.

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

This is implemented via `text_track/scripts/evaluate/storage.py`'s
`ResultRecord`, which records `is_correct` and `format_compliant` as
separate fields (`grading.classify_result()` derives both from the same
failure-type classification — see `grading.FailureType`).

## 4. Prompts

Prompts live in `text_track/scripts/evaluate/conditions.py`, versioned
`prompt_v1` (the `ConditionConfig.prompt_version` field, recorded on every
`ResultRecord`). This covers the single-call reasoning prompts (`A_EE`/`A_II`),
the direct-answer prompts (`A_E0`/`A_I0`), and the staged-pivot
translation/reasoning prompts (`A_IE`/`A_EI`) — including the
translation-only prompts, which explicitly instruct the model not to solve
the problem. Any future wording change should bump this to `prompt_v2` and
note what changed.

## 5. Model settings

| Model key | Model ID | Temperature | Max tokens | Provider | Status |
|---|---|---|---|---|---|
| `claude_sonnet_4_6` | `claude-sonnet-4-6` | 0.0 | 2048 | Anthropic | ready |
| `llama_3_8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | 0.0 | 2048 | HF router | ready |
| `gpt_5_2_thinking` | `gpt-5.2-thinking` | n/a (reasoning model) | 2048 | OpenAI direct API | ready |
| `qwen_3_6_27b` | `qwen3.6-27b` | 0.0 | 2048 | HPC via vLLM | **blocked** — model weights still being staged on HPC; endpoint not yet configured (`HPC_VLLM_BASE_URL`) |
| `qwen_sealion_v4_5_27b_it` | `qwen-sealion-v4.5-27b-it` | 0.0 | 2048 | HPC via vLLM | **blocked**, same reason |

The three non-HPC models are implemented and callable today (`text_track/scripts/evaluate/models.py`).
The two Qwen models reuse the same OpenAI-compatible client code path (vLLM's
server is OpenAI-compatible) but need `HPC_VLLM_BASE_URL` set once the vLLM
server is actually running with the model weights loaded — until then,
selecting them raises a clear `RuntimeError` rather than silently failing.
Final settings (including whether GPT-5.2 Thinking needs a `reasoning_effort`
parameter once its actual API surface is confirmed) are still subject to
revision during **Phase 4 (pilot)**.

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

## 8. Non-goals (still deferred)

- Does **not** run any pilot or full experiment under any condition against
  real APIs — that's **Phase 4**. Everything above is implemented and
  tested against a fake model client only (zero real API calls made so far).
- Does **not** modify `analyze.py` — its rewrite for the new per-item×model×
  condition×stage result schema is a separate, later phase.
- Does **not** attempt to resolve the provenance gaps in section 7.
- Does **not** implement the actual HPC/vLLM deployment for the Qwen
  models — only the client code path, gated on `HPC_VLLM_BASE_URL` being
  set once the server is running.
