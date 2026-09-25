# Experimental Protocol

- Protocol version: `protocol_v2`
- Date: 2026-09-23 (implementation status updated 2026-09-24; v2 revision 2026-09-25;
  resumption/tokenization/annotation hardening 2026-09-25)
- Pinned dataset version: `dataset_conference_v1.1` (see `data/dataset_manifest.json`)
- **STATUS: implemented in `text_track/scripts/evaluate/`, not yet piloted. No real
  model API calls have been made under `protocol_v2` (only fake-client tests so far).**

**Why v2 exists**: a real pilot call to `qwen_3_6_27b` under `protocol_v1` surfaced two
problems this revision fixes: (1) the reasoning prompts were satisfiable with only a tag
and no explanation, which isn't what the reasoning conditions are meant to measure, and
(2) the grader only recognized the literal `<answer>` tag, so a fully correct response
ending in `\boxed{109}` was misclassified as a missing answer. Since prompts changed,
`PROTOCOL_VERSION` and every condition's `prompt_version` both bumped to `v2` — the
resume key includes both, so no `protocol_v1`/`prompt_v1` result can be silently reused
under `v2` (see section 8).

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

Reasoning conditions (`A_EE`/`A_II`/`A_IE`/`A_EI`) require a **two-part
response**: a written solution explanation in the specified language,
followed by the final answer enclosed in `<answer>VALUE</answer>` tags. An
answer-only response with no explanation does not satisfy these conditions
— that's what the direct-answer controls (`A_E0`/`A_I0`) are for. See
`TWO_PART_RULE_V2` in `conditions.py`.

Grading records two separate axes, per the conference plan:
- **Semantic correctness** (`is_correct`): does the extracted answer match
  `canonical_answer` regardless of surface form (e.g. `"(A)"`, `"a"`, and
  Ilokano `"Wen"` for a `YES` canonical answer are all semantically
  equivalent).
- **Format compliance** (`format_compliant`): did the model follow the
  required contract exactly — a well-formed `<answer>` tag containing
  literally the canonical vocabulary?

These two axes are independent, not derived from one another. Two real
cases from piloting motivate this:
- A response with no `<answer>` tag at all, ending in LaTeX `\boxed{109}`,
  is recovered via fallback extraction (see below), graded `is_correct=True`,
  but `format_compliant=False` — the tag contract wasn't followed, even
  though the answer is right.
- `<answer>Wen</answer>` for a `bbh_causal_judgement` item whose canonical
  answer is `YES`: `Wen` is semantically `YES` (`is_correct=True`), but the
  format contract specifically requires the `YES`/`NO` vocabulary, not a
  same-meaning Ilokano token — so `format_compliant=False` even though a
  tag was present:

  | Response | Semantic correctness | Format compliance |
  |---|---|---|
  | `<answer>YES</answer>` | correct | compliant |
  | `<answer>Yes</answer>` | correct | compliant (after case normalization) |
  | `<answer>Wen</answer>` | correct | **noncompliant** |
  | `<answer>Saan</answer>` (gold `NO`) | correct | **noncompliant** |

**Fallback extraction**: `grading.extract_fallback_answer()` recovers an
answer when no `<answer>` tag is present, trying in order: LaTeX
`\boxed{...}`, an explicit "final answer is X" sentence, then source-specific
patterns (a parenthesized option letter for multiple choice, an explicit
Yes/No/Wen/Saan token for causal judgement). A fallback-recovered answer is
graded normally but is always `format_compliant=False`.

This is implemented via `text_track/scripts/evaluate/storage.py`'s
`ResultRecord`, which records `is_correct` and `format_compliant` as
separate fields; both are derived in `grading.classify_result()` (see
`grading.FailureType`).

**Translation stage status**: the translate stage's success status is
`translation_completed`, not `correct` — it means only that the API call
succeeded, a nonempty translation candidate came back, and it didn't
contain an answer tag (i.e. the model didn't try to solve instead of
translate). It says nothing about whether the translation is linguistically
faithful — that's a separate manual check (section 9).

## 4. Prompts

Prompts live in `text_track/scripts/evaluate/conditions.py`, versioned
`prompt_v2` (the `ConditionConfig.prompt_version` field, recorded on every
`ResultRecord`). `v2` changes from `v1`:

- **Two-part enforcement** (`TWO_PART_RULE_V2`): `A_EE`, `A_II`, and both
  reasoning-stage prompts of `A_IE`/`A_EI` now explicitly require a written
  explanation followed by the answer tag, and explicitly forbid an
  answer-only response. (`A_E0`/`A_I0` are unchanged — answer-only is
  correct for those.)
- **Ilokano-only enforcement**: `A_II` and `A_EI`'s reasoning-stage prompt
  add: *"Do not return only the final answer. Provide the solution
  explanation entirely in Ilokano. Do not use English except for proper
  names, mathematical notation, and the canonical final-answer token."*
- **EN→ILO translation completeness**: `TRANSLATE_EN_TO_ILO_SYSTEM` (used by
  `A_EI`'s translate stage) adds: *"Translate the entire problem into
  Ilokano. Do not leave sentences in English except proper names and option
  labels."*

Any future wording change should bump this to `prompt_v3` and note what
changed — `ResultRecord`'s resume key includes `prompt_version` (section 8),
so a version bump automatically prevents old-wording results from being
mistaken for new-wording ones on resume.

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

## 8. Resumption and reproducibility metadata

**Resume key**: `ResultRecord`'s resume key (`storage.make_resume_key()`) is
`(dataset_version, protocol_version, prompt_version, evaluator_version,
config_fingerprint, item_id, model_key, condition_key, stage)`.
`config_fingerprint` (`models.config_fingerprint()`) is a short hash of
everything that determines a model's actual behavior — checkpoint,
precision, sampling settings, serving config, tokenizer revision — so a
changed model configuration (a different vLLM precision, an updated
checkpoint) can never be silently treated as equivalent to an older run's
results, the same way `prompt_version`/`protocol_version` protect against a
prompt-wording change being silently reused. `evaluator_version`
(`storage.EVALUATOR_VERSION`, currently `"evaluator_v1"`) exists as a
**separate** axis from `protocol_version`/`prompt_version`: a grading or
tokenization fix (e.g. the answer-only detection fix, or the
rationale-token stripping change) doesn't touch prompts at all and wouldn't
otherwise bump those — without `evaluator_version` in the key, a fresh
`run_id` could silently reuse another run's completed work via
`ResumeIndex` even though the two runs' evaluator code disagreed on what a
record's fields mean. Bump `EVALUATOR_VERSION` whenever grading/tokenization/
extraction logic changes in a way that would change what a record means,
even if dataset/protocol/prompt/model config all stay the same.

**Older result files can't crash a run**: `ResumeIndex.load_from_runs_dir()`
tolerates JSONL rows from an older schema (e.g. a `protocol_v1` file missing
fields this revision added) by filling missing fields with `None` rather
than raising — such rows simply never match a current lookup key (different
`protocol_version`/`evaluator_version`/`config_fingerprint`), so old data is
cleanly ignored, not fatal. A corrupt/truncated line is skipped the same way.

**Stable run IDs**: `run_evaluation(..., run_id=...)` / the CLI's `--run-id`
accept a predetermined ID (e.g. one assigned per SLURM job). Restarting a
failed job under the *same* `--run-id` appends to the same result JSONL
rather than starting a new file. `write_run_manifest()` refuses to overwrite
an existing manifest for that `run_id` if the new invocation's
`dataset_version`/`protocol_version`/`evaluator_version`/`conditions`/
`models`/`model_settings`/`git_commit`/`planned_item_ids_by_condition` don't
all match exactly —
raising `ValueError` rather than silently applying different settings (or a
different evaluator code version, or a different item selection) to a
run_id that already has results under the old ones. On a compatible
restart, the manifest's original `created_at` is preserved and
`last_resumed_at` is updated to the restart time — so it's always clear
when a run_id's results span more than one invocation. When a run reuses
another run's completed work (any `run_id` found in `ResumeIndex`, not just
its own), those source `run_id`s are recorded in `resuming_from_run_ids`.

**Run manifest** (`storage.write_run_manifest()`, written before any API
calls): also records `git_commit` (the evaluator's own commit hash at run
time) and `model_settings` — one dict per model in the run
(`models.manifest_settings_for()`, plus `config_fingerprint`), including
model ID, provider, temperature, max output tokens, seed, and (where
applicable) `hf_repo`, `checkpoint_commit_sha`, `tokenizer_repo`,
`revision`, `context_length`, `precision`, `thinking_mode`, `vllm_version`,
and `flashinfer_sampler`. Hosted APIs (Anthropic/OpenAI/HF router) don't
expose most of the serving-level fields to clients, so those stay `null`
for those providers rather than guessed. For HPC/vLLM models, values not
known when the registry entry was written can be supplied via env vars at
manifest-build time (`HPC_HF_REPO`, `HPC_CHECKPOINT_COMMIT_SHA`,
`HPC_MODEL_REVISION`, `HPC_TOKENIZER_REPO`, `HPC_TOKENIZER_REVISION`,
`HPC_CONTEXT_LENGTH`, `HPC_THINKING_MODE`, `HPC_PRECISION`,
`HPC_VLLM_VERSION`, `HPC_FLASHINFER_SAMPLER`) instead of hardcoding them
into `models.py` ahead of the HPC serving setup being finalized —
numeric/boolean values (`HPC_CONTEXT_LENGTH`, `HPC_FLASHINFER_SAMPLER`) are
parsed into their proper types, not left as raw strings.

**Token counts** (`tokenization.py`): each `ResultRecord` records
`question_en_tokens`/`question_ilo_tokens` (always, since both are
available on every dataset item regardless of condition), plus
`translation_tokens`/`rationale_tokens` when this record actually produced
one, and `tokenization_tax_ratio` (`question_ilo_tokens /
question_en_tokens`). `rationale_tokens` counts the explanation **with the
answer tag/fallback stripped out** (`grading.strip_answer_content()`) — not
the complete response — so a mostly-answer-tag response with one filler
word isn't counted as a full rationale's worth of tokens;
`input_tokens`/`output_tokens` (below) remain the exact counts for the
*complete* response, unaffected by this stripping. `rationale_tokens` is
always `None` for `A_E0`/`A_I0` (direct-answer controls have no rationale
by design, so nothing is tokenized as one even if a model adds unsolicited
explanatory text anyway).

For HPC-served models, token counts are **model-native** (loaded via the
`tokenizers` library from the model's own `tokenizer.json` on the HF Hub,
with `add_special_tokens=False` since we're counting the content itself,
not a ready-to-run sequence with BOS/EOS/role markers) — the primary
figures for the tokenization-tax claim, since that's what actually
determines real context usage and cost for those models. **A failed
native-tokenizer load for an HPC model is fatal** (raises immediately,
stopping the run) rather than silently falling back to tiktoken — a silent
fallback would produce a tokenization-tax number that looks valid but isn't
measuring the actual model's tokenizer. Repo resolution precedence:
`HPC_TOKENIZER_REPO` env var → `HPC_HF_REPO` env var →
`config.tokenizer_repo` → `config.hf_repo` → `config.model_id`. Revision
resolution precedence: `HPC_TOKENIZER_REVISION` env var →
`HPC_CHECKPOINT_COMMIT_SHA` env var → `HPC_MODEL_REVISION` env var →
`config.revision` → `"main"` — the checkpoint/model-revision env vars are
checked here (not just the tokenizer-specific one) because in practice the
tokenizer ships alongside the model checkpoint in the same repo/revision.
**A resolution that falls all the way through to `"main"` is rejected**
(`tokenization.UnresolvedTokenizerRevisionError`, an HPC-only check) —
`"main"` is a mutable branch, not a reproducible reference, so an HPC run
must have at least one of the three revision env vars set. The resolved
repo/revision are recorded both per-record (`tokenizer_model_id`/
`tokenizer_revision`) and directly in the manifest's `model_settings`
(`resolved_tokenizer_model_id`/`resolved_tokenizer_revision`) — not only
indirectly via `config_fingerprint`'s opaque hash. For non-HPC providers
(Anthropic, OpenAI, gated repos), `tiktoken` `cl100k_base` is the only
option and is used as a clearly-labeled fallback approximation (no pinned
revision requirement applies there). `word_count`/`char_count` are
supplementary descriptive stats only — not a substitute for either
token-count figure. `input_tokens`/`output_tokens` (each provider's own
exact `usage` figures, for the *complete* response) remain the
authoritative totals for cost accounting.

## 9. Manipulation / post-run annotation checks

Automatic language identification for Ilokano is not trusted for judging
whether a model actually reasoned in Ilokano when asked to, or whether a
translation is faithful — these require human annotation, done as a
**separate pass that never modifies the immutable inference JSONL**.

`text_track/scripts/generate_annotation_template.py <run_id>` reads a
completed run's result file and writes
`text_track/data/annotations/{run_id}_annotation_template.csv`, keyed by
`(run_id, item_id, model_key, condition_key, stage)`, with:

- Derived automatically (not a judgment call): `has_text_beyond_answer`
  (`grading.has_text_beyond_answer()` — strips the answer tag, known
  fallback-answer forms (`\boxed{...}`, a full "the final answer is X"
  sentence including its leading words), and a standalone option letter or
  YES/NO/Wen/Saan token, before checking for any remaining word characters,
  so answer-only responses like `<answer>109</answer>`, `"The final answer
  is 109."`, or a bare `"(A)"` all correctly read `False` — a real
  explanation that happens to start with "Yes," is not affected, since the
  standalone-token strip only applies when that token is the *entire*
  remainder) and `requested_language` (`None` for `A_E0`/`A_I0` —
  direct-answer controls have a language-neutral canonical answer and no
  rationale, so no rationale-language judgment
  applies to them).
- Left blank for a human annotator: `rationale_present` (deliberately
  **not** auto-derived — `has_text_beyond_answer` is a hint, not a
  determination; a response can have leftover text that still isn't a real
  explanation), `language_compliance` (`compliant` / `mixed` /
  `noncompliant` / `uncertain`), `translation_faithfulness` (`accurate` /
  `minor_error` / `major_error` / `unusable`), `translation_error_type`
  (`none` / `lexical` / `morphological` / `semantic` / `omission` /
  `addition`), `annotation_notes`, `annotator`.

Annotations are joined back against the inference data during analysis by
the same key tuple — `analyze.py`'s eventual rewrite (already a known
deferred item, section 11) will need to perform that join, not read
annotations as if they were part of the inference schema.

## 10. Pilot requirements before full-scale runs

Before downloading/running the full 27B models, run a **source-diverse**
pilot: at least one item from each of the five sources (GSM8K, BBH logical
deduction, BBH causal judgement, MMLU conceptual physics, MMLU formal
logic), across all six conditions. A single-item (`gsm8k_0`-only) pilot is
what surfaced this `v2` revision's fixes — a source-diverse pilot is needed
to also exercise: four- and five-option letters, YES/NO vs. Wen/Saan,
Ilokano rationale compliance, English↔Ilokano translation behavior,
fallback extraction, and direct-answer behavior, per source.

Use `run_evaluation(..., item_ids=[...])` or the CLI's `--item-ids` flag to
restrict a run to a hand-picked list of IDs across all sources — this
overrides both the full-dataset default and the `A_E0`/`A_I0`
stratified-subset restriction, so a pilot can exercise every condition on
exactly the chosen items:

```bash
python3 -m text_track.scripts.evaluate \
    --item-ids gsm8k_0 bbh_logic_0 bbh_causal_250 mmlu_logic_0 mmlu_physics_126 \
    --conditions A_EE A_II A_IE A_EI A_E0 A_I0 \
    --models <model_key>
```

Only after this pilot's outputs have been inspected (raw responses, not
just aggregate pass/fail) should the full 1,000-item runs proceed.

## 11. Non-goals (still deferred)

- Does **not** run the full 1,000-item experiment against real APIs yet —
  that requires the source-diverse pilot (section 10) to pass first.
  Everything above is implemented and tested against a fake model client
  only, aside from the single-item exploratory pilot that motivated `v2`.
- Does **not** modify `analyze.py` — its rewrite for the new per-item×model×
  condition×stage result schema, including the annotation join (section 9),
  is a separate, later phase.
- Does **not** attempt to resolve the provenance gaps in section 7.
- Does **not** implement the actual HPC/vLLM deployment for the Qwen
  models — only the client code path, gated on `HPC_VLLM_BASE_URL` being
  set once the server is running (the user is handling the SLURM/HPC side
  directly, outside this repo).
- `tokenization.py` uses the model-native tokenizer for HPC/vLLM models
  (Qwen), falling back to `tiktoken` `cl100k_base` only where no native
  tokenizer is loadable (Anthropic, OpenAI, gated repos). Exact native
  tokenization for Claude/GPT-5.2/Llama is not implemented — those always
  use the tiktoken approximation.
