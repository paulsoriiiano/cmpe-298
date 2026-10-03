# Regrade changelog (evaluator_v5 -> evaluator_v6)

Offline regrade of stored raw responses; **no model was re-queried** and the original result files are untouched. Corrected copies are in `results_regraded/`. Only `reason`/`direct` stage rows can change; translate-stage rows are copied as-is.

- Records whose grading changed: **107**
- Correctness verdict flipped (wrong -> right): **83**
- Label-only changes (still incorrect, failure type relabelled): **24**
- Validation: the ORIGINAL grader applied to the stored `raw_response` reproduces the stored grade for every reason/direct row: gpt_5_2: 0 mismatches, qwen_3_6_27b: 0 mismatches, qwen_sealion_v4_5_27b_it: 0 mismatches.

## Rules applied

**R1.** Repetition flag overrode a normally-finished, answered response. The 4-gram repetition heuristic fired on a response with finish_reason=stop that committed to an answer (e.g. truth-table solutions). It is now applied only to responses that were truncated or produced no answer; the response is graded on its answer instead.

**R2.** Refusal regex fired on ordinary reasoning ("I cannot assume...") in a response that committed to an answer. Refusal is now only assigned when no answer was produced.

**R3.** Right value with decoration ("60%", "$18") was marked invalid_answer_format and wrong. It is now graded on its value and flagged format_compliant=False, as the protocol's two-axis rule (semantic correctness vs format compliance) specifies.

## Changes by model and rule (all / verdict flipped)

| Model | Rule | Records changed | Verdict flipped |
|---|---|---|---|
| gpt_5_2 | R1 | 39 | 39 |
| gpt_5_2 | R3 | 21 | 20 |
| qwen_3_6_27b | R1 | 25 | 17 |
| qwen_3_6_27b | R2 | 3 | 1 |
| qwen_3_6_27b | R3 | 2 | 1 |
| qwen_sealion_v4_5_27b_it | R1 | 13 | 4 |
| qwen_sealion_v4_5_27b_it | R2 | 4 | 1 |

## Changes by condition (verdict flips only)

| Model | A_EE | A_II | A_IE | A_EI | A_E0 | A_I0 |
|---|---|---|---|---|---|---|
| gpt_5_2 | 13 | 15 | 14 | 15 | 1 | 1 |
| qwen_3_6_27b | 1 | 6 | 1 | 11 | 0 | 0 |
| qwen_sealion_v4_5_27b_it | 0 | 2 | 1 | 2 | 0 | 0 |

## Item-level edits

Columns: gold = canonical answer; answer/type/correct show old -> new. `rep` = repetition ratio recorded at run time; `fin` = finish_reason.

### gpt_5_2 (60 records)

| Item | Cond | Rule | Gold | Answer | Failure type | Correct | Format ok | fin | rep |
|---|---|---|---|---|---|---|---|---|---|
| mmlu_logic_11 | A_EE | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.4 |
| mmlu_logic_114 | A_EE | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.363 |
| mmlu_logic_121 | A_EE | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.323 |
| mmlu_logic_14 | A_EE | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.399 |
| mmlu_logic_29 | A_EE | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.432 |
| mmlu_logic_30 | A_EE | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.42 |
| mmlu_logic_80 | A_EE | R1 | C | None -> C | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.338 |
| mmlu_logic_82 | A_EE | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.356 |
| mmlu_logic_88 | A_EE | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.36 |
| mmlu_logic_11 | A_EI | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.418 |
| mmlu_logic_114 | A_EI | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.356 |
| mmlu_logic_12 | A_EI | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.397 |
| mmlu_logic_121 | A_EI | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.32 |
| mmlu_logic_14 | A_EI | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.36 |
| mmlu_logic_29 | A_EI | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.405 |
| mmlu_logic_30 | A_EI | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.34 |
| mmlu_logic_80 | A_EI | R1 | C | None -> C | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.318 |
| mmlu_logic_82 | A_EI | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.322 |
| mmlu_logic_88 | A_EI | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.334 |
| mmlu_logic_11 | A_IE | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.451 |
| mmlu_logic_114 | A_IE | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.338 |
| mmlu_logic_12 | A_IE | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.413 |
| mmlu_logic_121 | A_IE | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.351 |
| mmlu_logic_14 | A_IE | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.4 |
| mmlu_logic_29 | A_IE | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.454 |
| mmlu_logic_30 | A_IE | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.366 |
| mmlu_logic_80 | A_IE | R1 | C | None -> C | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.302 |
| mmlu_logic_82 | A_IE | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.37 |
| mmlu_logic_88 | A_IE | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.422 |
| mmlu_logic_105 | A_II | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.33 |
| mmlu_logic_11 | A_II | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.406 |
| mmlu_logic_114 | A_II | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.304 |
| mmlu_logic_12 | A_II | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.386 |
| mmlu_logic_121 | A_II | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.332 |
| mmlu_logic_14 | A_II | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.354 |
| mmlu_logic_29 | A_II | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.463 |
| mmlu_logic_80 | A_II | R1 | C | None -> C | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.368 |
| mmlu_logic_82 | A_II | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.338 |
| mmlu_logic_88 | A_II | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.41 |
| gsm8k_90 | A_E0 | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_134 | A_EE | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_27 | A_EE | R3 | 25 | 25% -> 25% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_60 | A_EE | R3 | 10 | 10% -> 10% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_90 | A_EE | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.031 |
| gsm8k_124 | A_EI | R3 | 20 | 20% -> 20% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_134 | A_EI | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_264 | A_EI | R3 | 40 | 40% -> 40% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_60 | A_EI | R3 | 10 | 10% -> 10% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_90 | A_EI | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.047 |
| gsm8k_90 | A_I0 | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_134 | A_IE | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_264 | A_IE | R3 | 40 | 40% -> 40% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_60 | A_IE | R3 | 10 | 10% -> 10% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_90 | A_IE | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.015 |
| gsm8k_124 | A_II | R3 | 20 | 20% -> 20% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_134 | A_II | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_264 | A_II | R3 | 40 | 40% -> 40% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_27 | A_II | R3 | 25 | 16.67% -> 16.67% | invalid_answer_format -> substantively_incorrect | False -> False | False -> False | stop | 0.026 |
| gsm8k_60 | A_II | R3 | 10 | 10% -> 10% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.0 |
| gsm8k_90 | A_II | R3 | 60 | 60% -> 60% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.067 |

### qwen_3_6_27b (30 records)

| Item | Cond | Rule | Gold | Answer | Failure type | Correct | Format ok | fin | rep |
|---|---|---|---|---|---|---|---|---|---|
| mmlu_logic_105 | A_EE | R1 | B | None -> C | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.36 |
| bbh_causal_270 | A_EI | R1 | YES | None -> Wen | repetition_degeneration -> correct | False -> True | False -> False | stop | 0.372 |
| bbh_logic_102 | A_EI | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.353 |
| bbh_logic_105 | A_EI | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.308 |
| bbh_logic_136 | A_EI | R1 | B | None -> C | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.313 |
| bbh_logic_170 | A_EI | R1 | A | None -> D | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.337 |
| bbh_logic_33 | A_EI | R1 | E | None -> E | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.378 |
| gsm8k_16 | A_EI | R1 | 17 | None -> 17 | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.319 |
| gsm8k_167 | A_EI | R1 | 3000 | None -> 1000 | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.542 |
| gsm8k_358 | A_EI | R1 | 10 | None -> 10 | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.303 |
| gsm8k_385 | A_EI | R1 | 348 | None -> 207 | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.306 |
| mmlu_logic_11 | A_EI | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.386 |
| mmlu_logic_12 | A_EI | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.316 |
| mmlu_logic_30 | A_EI | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.4 |
| mmlu_logic_33 | A_EI | R1 | A | None -> A | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.329 |
| mmlu_logic_5 | A_EI | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.365 |
| mmlu_physics_211 | A_EI | R1 | B | None -> D | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.404 |
| bbh_causal_291 | A_II | R1 | YES | None -> Wen | repetition_degeneration -> correct | False -> True | False -> False | stop | 0.415 |
| bbh_causal_298 | A_II | R1 | NO | None -> Wen | repetition_degeneration -> substantively_incorrect | False -> False | False -> False | stop | 0.313 |
| bbh_logic_18 | A_II | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.313 |
| bbh_logic_80 | A_II | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.302 |
| mmlu_logic_105 | A_II | R1 | B | None -> C | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.348 |
| mmlu_logic_12 | A_II | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.308 |
| mmlu_physics_209 | A_II | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.335 |
| mmlu_physics_222 | A_II | R1 | C | None -> C | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.429 |
| gsm8k_83 | A_IE | R2 | 312 | None -> 156 | refusal -> substantively_incorrect | False -> False | False -> True | stop | 0.042 |
| mmlu_physics_250 | A_IE | R2 | D | None -> D | refusal -> correct | False -> True | False -> True | stop | 0.033 |
| mmlu_physics_285 | A_IE | R2 | B | None -> None | refusal -> truncation | False -> False | False -> False | length | 0.048 |
| gsm8k_27 | A_EE | R3 | 25 | 25% -> 25% | invalid_answer_format -> correct | False -> True | False -> False | stop | 0.051 |
| gsm8k_27 | A_EI | R3 | 25 | 38.89% -> 38.89% | invalid_answer_format -> substantively_incorrect | False -> False | False -> False | stop | 0.199 |

### qwen_sealion_v4_5_27b_it (17 records)

| Item | Cond | Rule | Gold | Answer | Failure type | Correct | Format ok | fin | rep |
|---|---|---|---|---|---|---|---|---|---|
| mmlu_logic_105 | A_EE | R1 | B | None -> C | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.316 |
| bbh_causal_269 | A_EI | R1 | YES | None -> Saan | repetition_degeneration -> substantively_incorrect | False -> False | False -> False | stop | 0.463 |
| bbh_logic_5 | A_EI | R1 | B | None -> B | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.302 |
| gsm8k_114 | A_EI | R1 | 10 | None -> 3 | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.354 |
| gsm8k_270 | A_EI | R1 | 22 | None -> 22 | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.455 |
| gsm8k_97 | A_EI | R1 | 291 | None -> 7 | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.591 |
| mmlu_physics_201 | A_EI | R1 | A | None -> C | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.365 |
| bbh_causal_298 | A_II | R1 | NO | None -> Wen | repetition_degeneration -> substantively_incorrect | False -> False | False -> False | stop | 0.433 |
| gsm8k_158 | A_II | R1 | 40 | None -> 0 | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.317 |
| gsm8k_336 | A_II | R1 | 18 | None -> 0 | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.313 |
| gsm8k_72 | A_II | R1 | 410 | None -> 410 | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.3 |
| mmlu_logic_12 | A_II | R1 | D | None -> D | repetition_degeneration -> correct | False -> True | False -> True | stop | 0.319 |
| mmlu_logic_2 | A_II | R1 | A | None -> D | repetition_degeneration -> substantively_incorrect | False -> False | False -> True | stop | 0.445 |
| gsm8k_215 | A_IE | R2 | 27 | None -> 19.64 | refusal -> substantively_incorrect | False -> False | False -> True | stop | 0.063 |
| gsm8k_278 | A_IE | R2 | 75 | None -> None | refusal -> truncation | False -> False | False -> False | length | 0.11 |
| gsm8k_87 | A_IE | R2 | 43,500 | None -> 43500 | refusal -> correct | False -> True | False -> True | stop | 0.062 |
| bbh_logic_42 | A_II | R2 | D | None -> None | refusal -> truncation | False -> False | False -> False | length | 0.076 |

## Not changed (deliberately)

- Truncated outputs with no answer, including repetition loops: still incorrect, still `repetition_degeneration` / `truncation`.
- `<answer>None</answer>` on multiple-choice items (the model declared the problem inconsistent): still `invalid_answer_format`, incorrect.
- Non-numeric GSM8K answers ("Cannot be determined"), fractions that are not the gold value, and wrong values: unchanged.
- All translate-stage rows and all infrastructure/parser outcomes (there are none).
