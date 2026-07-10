"""Compute chapter-level WER/CER and language-level summary statistics."""

import argparse
from pathlib import Path

import pandas as pd
from jiwer import wer, cer


def compute_scores(
    ref_root: Path,
    hyp_root: Path,
    results_dir: Path,
    model_name: str,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for ref_dir in sorted([path for path in ref_root.iterdir() if path.is_dir()]):
        language = ref_dir.name
        hyp_dir = hyp_root / language

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
                "model": model_name,
                "language": language,
                "chapter_id": chapter_id,
                "wer": wer(reference, hypothesis),
                "cer": cer(reference, hypothesis),
                "reference_word_count": len(reference.split()),
                "hypothesis_word_count": len(hypothesis.split()),
            })

    results = pd.DataFrame(rows)

    chapter_out = results_dir / f"{model_name}_chapter_results.csv"
    results.to_csv(chapter_out, index=False)

    summary = (
        results.groupby(["model", "language"])
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

    summary_out = results_dir / f"{model_name}_summary.csv"
    summary.to_csv(summary_out, index=False)

    print(f"Wrote {chapter_out}")
    print(f"Wrote {summary_out}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-root", default="references/chapter_level")
    parser.add_argument("--hyp-root", default="transcriptions/whisper_large_v3_normalized")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-name", default="whisper_large_v3")
    args = parser.parse_args()

    compute_scores(
        Path(args.ref_root),
        Path(args.hyp_root),
        Path(args.results_dir),
        args.model_name,
    )


if __name__ == "__main__":
    main()