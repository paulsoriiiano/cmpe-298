# Numerical Analysis — 3-Pass Cross-Lingual Reasoning (Ilokano vs. English)

Historical three-pass analysis retained for reference. Claude Sonnet 4.6 and Llama 3 8B are no longer part of the current model comparison; the active analysis uses GPT-5.2, Qwen3.6-27B, and Qwen-SEA-LION-v4.5-27B-IT.

**Historical passes.** P1 = English question → English reasoning (baseline). P2 = Ilokano question → reasoning entirely in Ilokano (native). P3 = Ilokano question → translate to English, then reason in English (pivot). A pass without a usable answer counted as incorrect.

| Historical model | P1 English | P2 Native Ilokano | P3 English Pivot | Δ_total | Δ_comp | Δ_reason | D_rel |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 96.3% | 88.4% | 90.5% | +7.9 pp | +5.8 pp | +2.1 pp | 2.3% |
| Llama 3 8B | 62.6% | 19.2% | 26.3% | +43.4 pp | +36.3 pp | +7.1 pp | 27.0% |

Historical passes with no extractable answer: Claude P1/P2/P3 = 0/6/2; Llama P1/P2/P3 = 25/146/23.

Historical per-source accuracy:

| Model | Source | P1 | P2 | P3 | Δ_reason (P3−P2) |
|---|---|---:|---:|---:|---:|
| Claude | BBH causal judgement (n=50) | 64.0% | 58.0% | 50.0% | -8.0 pp |
| Claude | BBH logical deduction (n=250) | 100.0% | 90.4% | 92.4% | +2.0 pp |
| Claude | GSM8K (n=400) | 97.8% | 91.2% | 94.0% | +2.8 pp |
| Claude | MMLU conceptual physics (n=174) | 97.1% | 88.5% | 91.4% | +2.9 pp |
| Claude | MMLU formal logic (n=126) | 96.0% | 87.3% | 90.5% | +3.2 pp |
| Llama | BBH causal judgement (n=50) | 48.0% | 54.0% | 38.0% | -16.0 pp |
| Llama | BBH logical deduction (n=250) | 52.8% | 29.2% | 31.6% | +2.4 pp |
| Llama | GSM8K (n=400) | 80.8% | 8.2% | 20.0% | +11.8 pp |
| Llama | MMLU conceptual physics (n=174) | 56.9% | 20.1% | 31.0% | +10.9 pp |
| Llama | MMLU formal logic (n=126) | 38.1% | 19.0% | 24.6% | +5.6 pp |

The historical P2-vs-P3 McNemar exact two-sided p-values were 3.142e-02 for Claude and 1.212e-05 for Llama. 

---

# TRT Metrics — GPT-5.2, Qwen3.6-27B, and Qwen-SEA-LION

This report analyzes the six-condition evaluator outputs.
Model-caused pivot translation failures count as end-to-end failures. Missing reasoning records caused by those failures are excluded from reasoning-only paired comparisons.

## Current evaluation conditions

- `A_EE` / P1: English input → English reasoning baseline.
- `A_II` / P2: Ilokano input → Ilokano reasoning.
- `A_IE` / P3: Ilokano input → separate English translation → fresh-context English reasoning.
- `A_EI`: English input → separate Ilokano translation → fresh-context Ilokano reasoning (reverse pivot).
- `A_E0` and `A_I0`: direct-answer controls on the fixed 300-item subset.

## Result inventory

| Model | Records | Condition completeness |
|---|---:|---|
| GPT-5.2 | 6600 | A_EE=1000, A_II=1000, A_IE=1000, A_EI=1000 reasoning/direct records |
| Qwen3.6-27B | 6538 | A_EE=1000, A_II=1000, A_IE=1000, A_EI=938 reasoning/direct records |
| Qwen-SEA-LION-v4.5-27B-IT | 6578 | A_EE=1000, A_II=1000, A_IE=1000, A_EI=978 reasoning/direct records |

## Accuracy by model and condition

| Model | A_EE | A_II | A_IE | A_EI | A_E0 | A_I0 |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.2 | 94.4% | 83.6% | 86.3% | 86.4% | 74.7% | 66.0% |
| Qwen3.6-27B | 94.9% | 68.9% | 81.9% | 49.0% | 64.7% | 54.7% |
| Qwen-SEA-LION-v4.5-27B-IT | 95.3% | 73.8% | 82.3% | 55.0% | 69.0% | 57.0% |

## Paired gaps and McNemar tests

| Model | Comparison | n | Accuracy difference | Bootstrap 95% CI | Exact p | Holm p |
|---|---|---:|---:|---:|---:|---:|
| GPT-5.2 | overall_language_gap | 1000 | +10.8 pp | [+8.6 pp, +13.1 pp] | 8.427e-21 | 3.371e-20 |
| GPT-5.2 | pivot_recovery | 1000 | +2.7 pp | [+0.6 pp, +4.8 pp] | 1.409e-02 | 1.409e-02 |
| GPT-5.2 | pivot_penalty | 1000 | -8.1 pp | [-10.2 pp, -6.0 pp] | 3.292e-14 | 9.876e-14 |
| GPT-5.2 | reverse_pivot_degradation | 1000 | +8.0 pp | [+5.9 pp, +10.2 pp] | 9.678e-14 | 1.936e-13 |
| Qwen3.6-27B | overall_language_gap | 1000 | +26.0 pp | [+23.1 pp, +28.9 pp] | 4.946e-66 | 1.484e-65 |
| Qwen3.6-27B | pivot_recovery | 1000 | +13.0 pp | [+10.1 pp, +16.0 pp] | 2.448e-17 | 2.448e-17 |
| Qwen3.6-27B | pivot_penalty | 1000 | -13.0 pp | [-15.4 pp, -10.7 pp] | 1.918e-27 | 3.837e-27 |
| Qwen3.6-27B | reverse_pivot_degradation | 938 | +43.4 pp | [+40.0 pp, +46.7 pp] | 2.729e-107 | 1.091e-106 |
| Qwen-SEA-LION-v4.5-27B-IT | overall_language_gap | 1000 | +21.5 pp | [+18.7 pp, +24.3 pp] | 5.302e-50 | 1.591e-49 |
| Qwen-SEA-LION-v4.5-27B-IT | pivot_recovery | 1000 | +8.5 pp | [+5.8 pp, +11.2 pp] | 7.930e-10 | 7.930e-10 |
| Qwen-SEA-LION-v4.5-27B-IT | pivot_penalty | 1000 | -13.0 pp | [-15.4 pp, -10.6 pp] | 1.918e-27 | 3.837e-27 |
| Qwen-SEA-LION-v4.5-27B-IT | reverse_pivot_degradation | 978 | +39.1 pp | [+35.8 pp, +42.3 pp] | 5.088e-101 | 2.035e-100 |

## TRT deltas

| Model | Total language gap AEE-AII | Reasoning penalty AIE-AII | Comprehension penalty AEE-AIE | Relative reasoning degradation |
|---|---:|---:|---:|---:|
| GPT-5.2 | +10.8 pp | +2.7 pp | +8.1 pp | 3.1% |
| Qwen3.6-27B | +26.0 pp | +13.0 pp | +13.0 pp | 15.9% |
| Qwen-SEA-LION-v4.5-27B-IT | +21.5 pp | +8.5 pp | +13.0 pp | 10.3% |

## Item-level pivot recovery and regression

The `pivot_recovery` comparison is A_IE versus A_II. `first_only_correct` means pivot correct/native incorrect; `second_only_correct` means native correct/pivot incorrect.

| Model | Paired items | Recovered | Regressed | Both correct | Both incorrect |
|---|---:|---:|---:|---:|---:|
| GPT-5.2 | 1000 | 70 | 43 | 793 | 94 |
| Qwen3.6-27B | 1000 | 187 | 57 | 632 | 124 |
| Qwen-SEA-LION-v4.5-27B-IT | 1000 | 139 | 54 | 684 | 123 |

## Domain-level accuracy

### GPT-5.2

| Source | A_EE | A_II | A_IE | A_EI |
|---|---:|---:|---:|---:|
| bbh_causal_judgement | 64.0% | 64.0% | 68.0% | 64.0% |
| bbh_logical_deduction | 99.6% | 84.4% | 85.2% | 88.8% |
| gsm8k | 96.5% | 86.8% | 88.5% | 90.2% |
| mmlu_conceptual_physics | 91.4% | 78.7% | 83.9% | 77.6% |
| mmlu_formal_logic | 93.7% | 86.5% | 92.1% | 90.5% |

### Qwen3.6-27B

| Source | A_EE | A_II | A_IE | A_EI |
|---|---:|---:|---:|---:|
| bbh_causal_judgement | 56.0% | 48.0% | 60.0% | 24.0% |
| bbh_logical_deduction | 98.8% | 52.8% | 83.2% | 29.6% |
| gsm8k | 97.2% | 78.5% | 80.5% | 47.2% |
| mmlu_conceptual_physics | 96.6% | 73.0% | 86.2% | 71.3% |
| mmlu_formal_logic | 92.9% | 73.0% | 86.5% | 72.2% |

### Qwen-SEA-LION-v4.5-27B-IT

| Source | A_EE | A_II | A_IE | A_EI |
|---|---:|---:|---:|---:|
| bbh_causal_judgement | 62.0% | 58.0% | 54.0% | 22.0% |
| bbh_logical_deduction | 98.4% | 60.4% | 82.8% | 53.6% |
| gsm8k | 97.5% | 80.5% | 83.0% | 44.5% |
| mmlu_conceptual_physics | 96.0% | 75.3% | 85.1% | 75.3% |
| mmlu_formal_logic | 94.4% | 83.3% | 86.5% | 76.2% |

## Failure rates

Failure rates are calculated over available reasoning/direct records for each condition.

| Model | Condition | n | Wrong/substantive | Missing answer | Truncated | Degeneration candidate | Format noncompliant |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-5.2 | A_EE | 1000 | 4.5% | 0.0% | 0.0% | 0.9% | 1.5% |
| GPT-5.2 | A_II | 1000 | 12.9% | 0.0% | 0.0% | 1.0% | 9.1% |
| GPT-5.2 | A_IE | 1000 | 11.4% | 0.0% | 0.0% | 1.0% | 2.7% |
| GPT-5.2 | A_EI | 1000 | 10.4% | 0.0% | 0.0% | 1.0% | 3.9% |
| Qwen3.6-27B | A_EE | 1000 | 3.5% | 0.0% | 1.6% | 0.1% | 1.7% |
| Qwen3.6-27B | A_II | 1000 | 11.5% | 0.0% | 19.5% | 13.6% | 23.4% |
| Qwen3.6-27B | A_IE | 1000 | 11.8% | 0.0% | 6.3% | 0.0% | 6.3% |
| Qwen3.6-27B | A_EI | 938 | 16.2% | 0.0% | 28.6% | 17.7% | 33.4% |
| Qwen-SEA-LION-v4.5-27B-IT | A_EE | 1000 | 3.5% | 0.0% | 1.2% | 0.1% | 1.2% |
| Qwen-SEA-LION-v4.5-27B-IT | A_II | 1000 | 13.1% | 0.0% | 12.8% | 11.0% | 17.8% |
| Qwen-SEA-LION-v4.5-27B-IT | A_IE | 1000 | 11.7% | 0.0% | 6.0% | 0.0% | 6.0% |
| Qwen-SEA-LION-v4.5-27B-IT | A_EI | 978 | 16.4% | 0.0% | 22.5% | 19.9% | 29.0% |

## Pivot translation-stage completion and missing reasoning records

A translation-stage truncation/refusal/format failure is counted as an end-to-end failure for the pivot condition. Because no reasoning call can validly consume an unusable translation, the corresponding reasoning record is absent and is excluded from reasoning-only paired tests. In the current files, every missing `A_EI` reasoning record is explained by a translation-stage truncation (`failure_type=truncation`, `finish_reason=length`).

| Model | A_IE completed/total | A_EI completed/total | Missing A_EI reasoning | Cause |
|---|---:|---:|---:|---|
| GPT-5.2 | 1000/1000 (100.0%) | 1000/1000 (100.0%) | 0 | none |
| Qwen3.6-27B | 1000/1000 (100.0%) | 938/1000 (93.8%) | 62 | translation truncation |
| Qwen-SEA-LION-v4.5-27B-IT | 1000/1000 (100.0%) | 978/1000 (97.8%) | 22 | translation truncation |

## Tokenization and output statistics

The JSON report contains mean, median, and sample count for input/output tokens, question token counts, translation/rationale tokens, tokenization-tax ratio, latency, retries, repetition ratio, word count, and character count for every model and condition.

| Model | Condition | Question EN tokens (mean) | Question ILO tokens (mean) | Tax ratio (mean) | Rationale tokens (mean) | Output tokens (mean) |
|---|---|---:|---:|---:|---:|---:|
| GPT-5.2 | A_EE | 89.75 | 165.52 | 1.84 | 170.06 | 181.89 |
| GPT-5.2 | A_II | 89.75 | 165.52 | 1.84 | 338.19 | 324.86 |
| GPT-5.2 | A_IE | 89.75 | 165.52 | 1.84 | 190.22 | 201.82 |
| GPT-5.2 | A_EI | 89.75 | 165.52 | 1.84 | 333.26 | 319.93 |
| Qwen3.6-27B | A_EE | 92.08 | 154.60 | 1.68 | 567.32 | 590.55 |
| Qwen3.6-27B | A_II | 92.08 | 154.60 | 1.68 | 963.89 | 982.43 |
| Qwen3.6-27B | A_IE | 92.08 | 154.60 | 1.68 | 676.12 | 700.87 |
| Qwen3.6-27B | A_EI | 87.07 | 145.65 | 1.67 | 1127.98 | 1148.19 |
| Qwen-SEA-LION-v4.5-27B-IT | A_EE | 92.08 | 154.60 | 1.68 | 536.06 | 557.53 |
| Qwen-SEA-LION-v4.5-27B-IT | A_II | 92.08 | 154.60 | 1.68 | 836.10 | 852.60 |
| Qwen-SEA-LION-v4.5-27B-IT | A_IE | 92.08 | 154.60 | 1.68 | 622.06 | 644.50 |
| Qwen-SEA-LION-v4.5-27B-IT | A_EI | 89.30 | 149.92 | 1.68 | 1006.94 | 1023.76 |

## Tokenization

| Model | Condition | English question tokens (mean) | Ilokano question tokens (mean) | Tokenization ratio (mean) | Translation tokens (mean) | Rationale tokens (mean) | Output tokens (mean) |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-5.2 | A_EE | 89.75 | 165.52 | 1.84 | n/a | 171.06 | 177.83 |
| GPT-5.2 | A_II | 89.75 | 165.52 | 1.84 | n/a | 338.30 | 345.00 |
| GPT-5.2 | A_IE | 89.75 | 165.52 | 1.84 | 90.19 | 191.00 | 143.99 |
| GPT-5.2 | A_EI | 89.75 | 165.52 | 1.84 | 176.12 | 333.37 | 258.07 |
| Qwen3.6-27B | A_EE | 89.75 | 165.52 | 1.84 | n/a | 554.83 | 561.30 |
| Qwen3.6-27B | A_II | 89.75 | 165.52 | 1.84 | n/a | 1012.36 | 1017.73 |
| Qwen3.6-27B | A_IE | 89.75 | 165.52 | 1.84 | 89.28 | 659.59 | 377.53 |
| Qwen3.6-27B | A_EI | 89.75 | 165.52 | 1.84 | 290.25 | 1159.65 | 713.65 |
| Qwen-SEA-LION-v4.5-27B-IT | A_EE | 89.75 | 165.52 | 1.84 | n/a | 523.18 | 529.64 |
| Qwen-SEA-LION-v4.5-27B-IT | A_II | 89.75 | 165.52 | 1.84 | n/a | 865.43 | 871.22 |
| Qwen-SEA-LION-v4.5-27B-IT | A_IE | 89.75 | 165.52 | 1.84 | 89.06 | 605.91 | 350.58 |
| Qwen-SEA-LION-v4.5-27B-IT | A_EI | 89.75 | 165.52 | 1.84 | 198.21 | 1028.92 | 611.44 |

## Scope limitation

Translation faithfulness and detailed linguistic error taxonomy are not judged. The separate translation and reasoning outputs are preserved for a later human annotation pass.