"""
Compute chapter-level WER and CER for Whisper Large-v3.

References:
    references/chapter_level/<language>/<chapter>.txt

Hypotheses:
    transcriptions/whisper_large_v3_normalized/<language>/<chapter>.txt

Outputs:
    results/whisper_large_v3_chapter_results.csv
    results/whisper_large_v3_summary.csv
"""

from pathlib import Path

import pandas as pd
from jiwer import wer, cer


REF_ROOT = Path("references/chapter_level")
HYP_ROOT = Path("transcriptions/whisper_large_v3_normalized")
RESULTS_DIR = Path("results")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for language in ["en", "ilo"]:
        ref_dir = REF_ROOT / language
        hyp_dir = HYP_ROOT / language

        if not ref_dir.exists():
            raise FileNotFoundError(f"Missing reference directory: {ref_dir}")

        if not hyp_dir.exists():
            raise FileNotFoundError(f"Missing hypothesis directory: {hyp_dir}")

        for ref_path in sorted(ref_dir.glob("*.txt")):
            chapter_id = ref_path.stem
            hyp_path = hyp_dir / ref_path.name

            if not hyp_path.exists():
                print(f"Missing hypothesis for {language} {chapter_id}: {hyp_path}")
                continue

            reference = ref_path.read_text(encoding="utf-8").strip()
            hypothesis = hyp_path.read_text(encoding="utf-8").strip()

            rows.append({
                "model": "whisper_large_v3",
                "language": language,
                "chapter_id": chapter_id,
                "wer": wer(reference, hypothesis),
                "cer": cer(reference, hypothesis),
                "reference_word_count": len(reference.split()),
                "hypothesis_word_count": len(hypothesis.split()),
            })

    results = pd.DataFrame(rows)

    chapter_out = RESULTS_DIR / "whisper_large_v3_chapter_results.csv"
    results.to_csv(chapter_out, index=False)

    summary = (
        results
        .groupby(["model", "language"])
        .agg(
            mean_wer=("wer", "mean"),
            median_wer=("wer", "median"),
            mean_cer=("cer", "mean"),
            median_cer=("cer", "median"),
            chapters=("chapter_id", "count"),
            reference_words=("reference_word_count", "sum"),
            hypothesis_words=("hypothesis_word_count", "sum"),
        )
        .reset_index()
    )

    summary_out = RESULTS_DIR / "whisper_large_v3_summary.csv"
    summary.to_csv(summary_out, index=False)

    print(f"Wrote {chapter_out}")
    print(f"Wrote {summary_out}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()