# Numerical Analysis — 3-Pass Cross-Lingual Reasoning (Ilokano vs. English)

Older three-pass analysis retained for reference. Claude Sonnet 4.6 and Llama 3 8B are no longer part of the current model comparison; the current analysis uses GPT-5.2, Qwen3.6-27B, and Qwen-SEA-LION-v4.5-27B-IT.

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
| GPT-5.2 | 93.1% | 82.1% | 84.9% | 84.9% | 74.3% | 65.7% |
| Qwen3.6-27B | 94.8% | 68.3% | 81.8% | 47.9% | 64.7% | 54.7% |
| Qwen-SEA-LION-v4.5-27B-IT | 95.3% | 73.6% | 82.2% | 54.8% | 69.0% | 57.0% |

## Paired gaps and McNemar tests

| Model | Comparison | n | Accuracy difference | 95% CI | Exact p | Holm p |
|---|---|---:|---:|---:|---:|---:|
| GPT-5.2 | overall_language_gap | 1000 | +11.0 pp | [+8.7 pp, +13.3 pp] | 2.768e-21 | 1.107e-20 |
| GPT-5.2 | pivot_recovery | 1000 | +2.8 pp | [+0.7 pp, +4.9 pp] | 1.112e-02 | 1.112e-02 |
| GPT-5.2 | pivot_penalty | 1000 | -8.2 pp | [-10.3 pp, -6.1 pp] | 1.964e-14 | 5.893e-14 |
| GPT-5.2 | reverse_pivot_degradation | 1000 | +8.2 pp | [+6.1 pp, +10.4 pp] | 3.496e-14 | 6.991e-14 |
| Qwen3.6-27B | overall_language_gap | 1000 | +26.5 pp | [+23.6 pp, +29.4 pp] | 2.772e-68 | 8.316e-68 |
| Qwen3.6-27B | pivot_recovery | 1000 | +13.5 pp | [+10.5 pp, +16.5 pp] | 3.957e-18 | 3.957e-18 |
| Qwen3.6-27B | pivot_penalty | 1000 | -13.0 pp | [-15.4 pp, -10.7 pp] | 1.918e-27 | 3.837e-27 |
| Qwen3.6-27B | reverse_pivot_degradation | 938 | +44.5 pp | [+41.0 pp, +47.8 pp] | 3.726e-111 | 1.490e-110 |
| Qwen-SEA-LION-v4.5-27B-IT | overall_language_gap | 1000 | +21.7 pp | [+18.9 pp, +24.5 pp] | 1.513e-50 | 4.540e-50 |
| Qwen-SEA-LION-v4.5-27B-IT | pivot_recovery | 1000 | +8.6 pp | [+5.9 pp, +11.3 pp] | 6.787e-10 | 6.787e-10 |
| Qwen-SEA-LION-v4.5-27B-IT | pivot_penalty | 1000 | -13.1 pp | [-15.5 pp, -10.7 pp] | 1.063e-27 | 2.125e-27 |
| Qwen-SEA-LION-v4.5-27B-IT | reverse_pivot_degradation | 978 | +39.3 pp | [+36.0 pp, +42.5 pp] | 1.344e-101 | 5.375e-101 |

## TRT deltas

| Model | Total language gap AEE-AII | Reasoning penalty AIE-AII | Comprehension penalty AEE-AIE | Relative reasoning degradation (AIE-AII / AIE x 100) |
|---|---:|---:|---:|---:|
| GPT-5.2 | +11.0 pp | +2.8 pp | +8.2 pp | 3.3% |
| Qwen3.6-27B | +26.5 pp | +13.5 pp | +13.0 pp | 16.5% |
| Qwen-SEA-LION-v4.5-27B-IT | +21.7 pp | +8.6 pp | +13.1 pp | 10.5% |

## Item-level pivot recovery and regression

The `pivot_recovery` comparison is A_IE versus A_II. `first_only_correct` means pivot correct/native incorrect; `second_only_correct` means native correct/pivot incorrect.

| Model | Paired items | Recovered | Regressed | Both correct | Both incorrect |
|---|---:|---:|---:|---:|---:|
| GPT-5.2 | 1000 | 71 | 43 | 778 | 108 |
| Qwen3.6-27B | 1000 | 193 | 58 | 625 | 124 |
| Qwen-SEA-LION-v4.5-27B-IT | 1000 | 141 | 55 | 681 | 123 |

## Domain-level accuracy

### GPT-5.2

| Source | A_EE | A_II | A_IE | A_EI |
|---|---:|---:|---:|---:|
| bbh_causal_judgement | 64.0% | 64.0% | 68.0% | 64.0% |
| bbh_logical_deduction | 99.6% | 84.4% | 85.2% | 88.8% |
| gsm8k | 95.5% | 85.5% | 87.5% | 89.0% |
| mmlu_conceptual_physics | 91.4% | 78.7% | 83.9% | 77.6% |
| mmlu_formal_logic | 86.5% | 78.6% | 84.1% | 82.5% |

### Qwen3.6-27B

| Source | A_EE | A_II | A_IE | A_EI |
|---|---:|---:|---:|---:|
| bbh_causal_judgement | 56.0% | 46.0% | 60.0% | 22.0% |
| bbh_logical_deduction | 98.8% | 52.0% | 83.2% | 28.4% |
| gsm8k | 97.0% | 78.5% | 80.5% | 46.8% |
| mmlu_conceptual_physics | 96.6% | 71.8% | 85.6% | 71.3% |
| mmlu_formal_logic | 92.9% | 72.2% | 86.5% | 68.3% |

### Qwen-SEA-LION-v4.5-27B-IT

| Source | A_EE | A_II | A_IE | A_EI |
|---|---:|---:|---:|---:|
| bbh_causal_judgement | 62.0% | 58.0% | 54.0% | 22.0% |
| bbh_logical_deduction | 98.4% | 60.4% | 82.8% | 53.2% |
| gsm8k | 97.5% | 80.2% | 82.8% | 44.2% |
| mmlu_conceptual_physics | 96.0% | 75.3% | 85.1% | 75.3% |
| mmlu_formal_logic | 94.4% | 82.5% | 86.5% | 76.2% |

## Failure rates

Failure rates are calculated over available reasoning/direct records for each condition.

| Model | Condition | n | Wrong/substantive | Missing answer | Truncated | Degeneration candidate | Format noncompliant |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-5.2 | A_EE | 1000 | 4.5% | 0.0% | 0.0% | 0.9% | 2.4% |
| GPT-5.2 | A_II | 1000 | 12.8% | 0.0% | 0.0% | 1.0% | 10.1% |
| GPT-5.2 | A_IE | 1000 | 11.4% | 0.0% | 0.0% | 1.0% | 3.7% |
| GPT-5.2 | A_EI | 1000 | 10.4% | 0.0% | 0.0% | 1.0% | 4.9% |
| Qwen3.6-27B | A_EE | 1000 | 3.4% | 0.0% | 1.6% | 0.1% | 1.8% |
| Qwen3.6-27B | A_II | 1000 | 11.3% | 0.0% | 19.5% | 13.6% | 24.0% |
| Qwen3.6-27B | A_IE | 1000 | 11.7% | 0.0% | 6.3% | 0.0% | 6.5% |
| Qwen3.6-27B | A_EI | 938 | 15.6% | 0.0% | 28.6% | 17.7% | 35.0% |
| Qwen-SEA-LION-v4.5-27B-IT | A_EE | 1000 | 3.4% | 0.0% | 1.2% | 0.1% | 1.3% |
| Qwen-SEA-LION-v4.5-27B-IT | A_II | 1000 | 12.7% | 0.0% | 12.8% | 11.0% | 18.3% |
| Qwen-SEA-LION-v4.5-27B-IT | A_IE | 1000 | 11.6% | 0.0% | 6.0% | 0.0% | 6.2% |
| Qwen-SEA-LION-v4.5-27B-IT | A_EI | 978 | 16.0% | 0.0% | 22.5% | 19.9% | 29.6% |

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