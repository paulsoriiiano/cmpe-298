# Numerical Analysis — 3-Pass Cross-Lingual Reasoning (Ilokano vs. English)

Dataset: `../data/evaluation_results_3pass.jsonl` — **1000 items**, sources: bbh_causal_judgement, bbh_logical_deduction, gsm8k, mmlu_conceptual_physics, mmlu_formal_logic.

**Passes.** P1 = English question → English chain-of-thought (baseline). P2 = Ilokano question → reasoning entirely in Ilokano (native). P3 = Ilokano question → translate to English, then reason in English (pivot).

**Scoring.** A pass is correct only if it emitted a parseable `<answer>` matching the gold (English or Ilokano reference). A pass with no extractable answer (refusal, degeneration, or no answer tag) counts as incorrect — the model failed to produce a usable answer under that condition.

## 1. Accuracy by model and pass

| Model | P1 English | P2 Native Ilokano | P3 English Pivot |
|---|---|---|---|
| Claude Sonnet 4.6 | 963/1000 (96.3%) | 884/1000 (88.4%) | 905/1000 (90.5%) |
| Llama 3 8B | 626/1000 (62.6%) | 192/1000 (19.2%) | 263/1000 (26.3%) |

Passes with no extractable answer (counted incorrect above):

| Model | P1 | P2 | P3 |
|---|---|---|---|
| Claude Sonnet 4.6 | 0 | 6 | 2 |
| Llama 3 8B | 25 | 146 | 23 |

## 2. Performance deltas (the gaps)

- **Total Language Gap** Δ_total = P1 − P2 (raw English-vs-Ilokano gap)
- **Comprehension Penalty** Δ_comp = P1 − P3 (translation/understanding loss; both reason in English)
- **Reasoning Penalty** Δ_reason = P3 − P2 (the thesis — collapse when forced to reason in Ilokano tokens)
- **Relative Reasoning Degradation** D_rel = (P3 − P2) / P3 × 100% (of the items the model understood, the share it failed purely from reasoning in Ilokano)

| Model | P1 | P2 | P3 | Δ_total | Δ_comp | Δ_reason | D_rel |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 96.3% | 88.4% | 90.5% | +7.9 | +5.8 | +2.1 | 2.3% |
| Llama 3 8B | 62.6% | 19.2% | 26.3% | +43.4 | +36.3 | +7.1 | 27.0% |

## 3. Statistical significance — McNemar's test (P2 vs. P3)

Same model, same items, two paired conditions (native vs. pivot). Discordant pairs drive the test: **b** = items the model got right under the pivot but wrong natively; **c** = right natively but wrong under the pivot. Two-sided exact binomial p-value; chi-square reported with Yates continuity correction (df=1).

### Claude Sonnet 4.6

| | Passed P3 (pivot) | Failed P3 (pivot) |
|---|---|---|
| **Passed P2 (native)** | 851 (both correct) | 33 (native only) |
| **Failed P2 (native)** | 54 (pivot only) | 62 (both wrong) |

- Discordant pairs: b (pivot-only) = 54, c (native-only) = 33
- McNemar χ² (continuity-corrected) = 4.60, p = 3.201e-02
- Exact binomial two-sided p = 3.142e-02 — **significant** (p < 0.05)

### Llama 3 8B

| | Passed P3 (pivot) | Failed P3 (pivot) |
|---|---|---|
| **Passed P2 (native)** | 98 (both correct) | 94 (native only) |
| **Failed P2 (native)** | 165 (pivot only) | 643 (both wrong) |

- Discordant pairs: b (pivot-only) = 165, c (native-only) = 94
- McNemar χ² (continuity-corrected) = 18.92, p = 1.364e-05
- Exact binomial two-sided p = 1.212e-05 — **significant** (p < 0.05)

## 4. Accuracy by source benchmark

### Claude Sonnet 4.6

| Source | P1 English | P2 Native | P3 Pivot | Δ_reason (P3−P2) |
|---|---|---|---|---|
| bbh_causal_judgement (n=50) | 64.0% | 58.0% | 50.0% | -8.0 |
| bbh_logical_deduction (n=250) | 100.0% | 90.4% | 92.4% | +2.0 |
| gsm8k (n=400) | 97.8% | 91.2% | 94.0% | +2.8 |
| mmlu_conceptual_physics (n=174) | 97.1% | 88.5% | 91.4% | +2.9 |
| mmlu_formal_logic (n=126) | 96.0% | 87.3% | 90.5% | +3.2 |

### Llama 3 8B

| Source | P1 English | P2 Native | P3 Pivot | Δ_reason (P3−P2) |
|---|---|---|---|---|
| bbh_causal_judgement (n=50) | 48.0% | 54.0% | 38.0% | -16.0 |
| bbh_logical_deduction (n=250) | 52.8% | 29.2% | 31.6% | +2.4 |
| gsm8k (n=400) | 80.8% | 8.2% | 20.0% | +11.8 |
| mmlu_conceptual_physics (n=174) | 56.9% | 20.1% | 31.0% | +10.9 |
| mmlu_formal_logic (n=126) | 38.1% | 19.0% | 24.6% | +5.6 |

