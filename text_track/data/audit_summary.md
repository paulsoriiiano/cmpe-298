# Dataset Audit Summary

Frozen dataset version: `dataset_conference_v1` (see `dataset_manifest.json`).

## Translation provenance

English questions were translated to Ilokano using **Claude Sonnet 4.5**
(`scripts/translate.py`). `scripts/collect_dataset.py` is a legacy/superseded
script (GPT-4o translation path) that was not used to produce the final
dataset. The pipeline was:

1. Translate English source questions/answers to Ilokano with Claude
   (`translate.py`).
2. Linguistically audit a random sample of the translated items
   (`audit_sheet.csv`).
3. Manually review and correct items across the full dataset based on audit
   findings (~35 per-item fix commits in git history).

## Audit population and sample

- Population size: 1,000 items
- Audit sample: 280 items (28%)
- Sampling procedure: simple random sample drawn from the full 1,000-item
  population (not stratified by source)
- Observed distribution of the sample across sources:

  | Source | Sample n |
  |---|---|
  | GSM8K | 116 |
  | BBH Logical Deduction | 66 |
  | MMLU Conceptual Physics | 49 |
  | MMLU Formal Logic | 34 |
  | BBH Causal Judgement | 15 |

## Reviewers

Single reviewer. No second-coder inter-annotator agreement was measured;
adjudication procedure is not applicable.

## Audit findings (n = 280)

| Category | Count | % of sample |
|---|---|---|
| Tagalog interference | 20 | 7.1% |
| Lexical hallucination | 16 | 5.7% |
| Linguistic drift | 5 | 1.8% |
| Focus/system error | 0 | 0.0% |

Naturalness score distribution (0–5 scale, 3 items unscored):

| Score | Count |
|---|---|
| 5 | 95 |
| 4 | 120 |
| 3 | 52 |
| 2 | 4 |
| 0 | 6 |

34 of 280 rows (12.1%) include free-text researcher notes.

## Corrections

Approximately 35 individual items were manually corrected in git history
following the audit (one commit per item, e.g. `fix: edit gsm8k_60`,
`fix: retranslated item 429`), spanning items both inside and outside the
280-item audited sample as part of a broader whole-dataset review pass.
