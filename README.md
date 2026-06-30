# Benchmarking Text and Speech AI Capabilities in Ilokano

This project evaluates the text reasoning and speech recognition capabilities of AI models in Ilokano, a low-resource Philippine language. It is organized into two independent tracks.

---

## Text Track — Cross-Lingual Reasoning Benchmark

**Research question:** Does LLM accuracy drop when a model must reason in Ilokano tokens vs. first translating to English?

### Dataset

1,000 bilingual (English + Ilokano) reasoning problems drawn from:

| Source | n | Type |
|---|---|---|
| GSM8K | 400 | Grade-school math |
| BBH Logical Deduction | 250 | Logical ordering |
| BBH Causal Judgement | 50 | Causal inference |
| MMLU Conceptual Physics | 174 | Multiple-choice physics |
| MMLU Formal Logic | 126 | Multiple-choice logic |

English questions were machine-translated to Ilokano using Claude Sonnet 4.5 (`collect_dataset.py`).

### 3-Pass Evaluation Protocol

Each question is evaluated three ways per model:

| Pass | Input | Reasoning language |
|---|---|---|
| P1 — English baseline | English question | English |
| P2 — Native Ilokano | Ilokano question | Ilokano |
| P3 — English pivot | Ilokano question | Translate to English first, then reason |

P2 vs. P3 isolates the **reasoning penalty**: both passes see the same Ilokano question, but P3 lets the model reason in English. A significant P3 > P2 gap means the model understands Ilokano but struggles to reason in it.

### Results

| Model | P1 English | P2 Native Ilokano | P3 English Pivot | Δ_reason (P3−P2) |
|---|---|---|---|---|
| Claude Sonnet 4.6 | 96.3% | 88.4% | 90.5% | +2.1 pp |
| Llama 3 8B | 62.6% | 19.2% | 26.3% | +7.1 pp |

Both differences are statistically significant (McNemar's exact test, p < 0.05). Llama 3 8B shows a 27% relative reasoning degradation when forced to reason in Ilokano vs. pivoting to English.

### Scripts

```
text_track/scripts/
├── collect_dataset.py   # Pull from HuggingFace + translate with GPT-4o
├── translate.py         # Translation utilities
├── evaluate.py          # Run 3-pass evaluation (Claude + Llama via HF router)
├── analyze.py           # Compute deltas, McNemar's test, write analysis.md
└── convert.py           # Format conversion utilities
```

Run the pipeline in order:

```bash
python text_track/scripts/collect_dataset.py   # builds data/dataset.jsonl
python text_track/scripts/evaluate.py          # writes data/evaluation_results_3pass.jsonl
python text_track/scripts/analyze.py           # writes data/analysis.md
```

---

## Speech Track — ASR Evaluation

Evaluates Whisper Large-v3 on Bible audio in English and Ilokano, measuring Word Error Rate (WER) and Character Error Rate (CER) at the chapter level.

### Dataset

Bible chapters from Matthew, Mark, and Luke in two languages:

```
speech_track/data/
├── audio_dataset/
│   ├── en/    # English MP3s (MAT, MRK, LUK chapters)
│   └── ilo/   # Ilokano MP3s
├── asr_evaluation_master.csv   # Verse-level reference texts + normalized forms
└── references/chapter_level/  # Chapter-level reference transcripts
```

### Scripts

```
speech_track/scripts/
├── bible_verses_scraper_and_cleaner.py   # Scrape and clean verse references
├── build_chapter_references.py           # Aggregate verses into chapter-level .txt files
├── run_whisper_large_v3.py               # Transcribe audio with Whisper
├── normalize_predictions.py              # Normalize hypothesis text for fair WER comparison
├── compute_wer.py                        # Compute WER/CER, write results CSVs
└── validate_asr_master.py                # Sanity-check the master CSV
```

Run the pipeline in order:

```bash
python speech_track/scripts/bible_verses_scraper_and_cleaner.py
python speech_track/scripts/build_chapter_references.py
python speech_track/scripts/run_whisper_large_v3.py
python speech_track/scripts/normalize_predictions.py
python speech_track/scripts/compute_wer.py   # writes results/whisper_large_v3_*.csv
```

---

## Setup

Requires Python ≥ 3.13. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=...
HF_TOKEN=...
```
