"""Inspect a completed run's repetition_ratio distribution, to support choosing (and
preregistering) REPETITION_DEGENERATION_THRESHOLD in text_track/scripts/evaluate/grading.py
from real data — NOT by guessing.

Per PROTOCOL.md section 3, inspect at least:
  - known/suspected degenerate responses (is_truncated or high repetition_ratio)
  - normal Ilokano responses
  - normal English responses
  - long but legitimate reasoning responses (high word_count, low repetition_ratio)
before committing to a threshold for the full run. This script does not choose the
threshold for you — it prints the distribution and flags candidate examples so a human can
read the actual text and confirm.

Usage:
    python3 text_track/scripts/inspect_repetition_scores.py <run_id> [--top N]
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "eval_runs")

sys.path.insert(0, SCRIPT_DIR)
from evaluate.grading import TRUNCATION_FINISH_REASONS, compute_repetition_ratio  # noqa: E402


def backfill_legacy_fields(record: dict) -> dict:
    """evaluator_v3 (and earlier) records predate repetition_ratio/is_truncated —
    compute them from raw_response/finish_reason so old pilot data can be used for
    threshold calibration without rerunning. Records that already have these fields
    (evaluator_v4+) are returned unchanged.

    Infrastructure failures have no raw_response at all (the API call never returned a
    response) — leave both fields as None rather than backfilling repetition_ratio=0.0 /
    is_truncated=False, which would fabricate "no repetition, not truncated" facts about a
    response that never existed and bias the calibration distribution."""
    if record.get("raw_response") is None:
        return record
    if record.get("repetition_ratio") is None:
        record["repetition_ratio"] = compute_repetition_ratio(record["raw_response"])
    if record.get("is_truncated") is None:
        record["is_truncated"] = record.get("finish_reason") in TRUNCATION_FINISH_REASONS
    return record


def load_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(backfill_legacy_fields(json.loads(line)))
    return records


def load_run(run_id: str) -> list[dict]:
    path = os.path.join(RUNS_DIR, f"{run_id}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Is {run_id!r} a real run_id?")
    return load_rows(path)


def _percentile(sorted_values, pct):
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def print_distribution(label: str, records: list[dict]) -> None:
    ratios = sorted(r["repetition_ratio"] for r in records if r.get("repetition_ratio") is not None)
    print(f"\n{label} (n={len(ratios)})")
    if not ratios:
        print("  (no scored records)")
        return
    print(f"  min={ratios[0]:.3f}  p50={_percentile(ratios, 0.50):.3f}  "
          f"p90={_percentile(ratios, 0.90):.3f}  p99={_percentile(ratios, 0.99):.3f}  "
          f"max={ratios[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect repetition_ratio distribution for threshold calibration."
    )
    parser.add_argument("run_id", nargs="?", help="run_id under text_track/data/eval_runs/")
    parser.add_argument(
        "--path", help="Explicit path to a result JSONL, instead of run_id — for results "
        "stored outside the repo's default data/eval_runs directory (e.g. copied over "
        "from HPC).",
    )
    parser.add_argument("--top", type=int, default=10, help="How many top-scoring examples to print.")
    args = parser.parse_args()

    if not args.run_id and not args.path:
        parser.error("Provide a run_id or --path.")
    records = load_rows(args.path) if args.path else load_run(args.run_id)
    scored = [r for r in records if r.get("repetition_ratio") is not None]

    print_distribution("All scored records", scored)
    print_distribution("Truncated (finish_reason == 'length')", [r for r in scored if r.get("is_truncated")])
    print_distribution("Not truncated", [r for r in scored if not r.get("is_truncated")])
    print_distribution("Ilokano-language stages (A_II reason, A_EI reason/translate, A_I0)", [
        r for r in scored
        if (r["condition_key"], r["stage"]) in {
            ("A_II", "reason"), ("A_EI", "reason"), ("A_EI", "translate"), ("A_I0", "direct"),
        }
    ])
    print_distribution("English-language stages (A_EE reason, A_IE reason/translate, A_E0)", [
        r for r in scored
        if (r["condition_key"], r["stage"]) in {
            ("A_EE", "reason"), ("A_IE", "reason"), ("A_IE", "translate"), ("A_E0", "direct"),
        }
    ])
    print_distribution(
        "Long legitimate-looking responses (word_count >= 100, not truncated)",
        [r for r in scored if not r.get("is_truncated") and (r.get("word_count") or 0) >= 100],
    )

    print(f"\nTop {args.top} highest repetition_ratio records (inspect the raw text before "
          f"deciding these are actually degenerate — a high score is a CANDIDATE, not a "
          f"confirmed classification):")
    for r in sorted(scored, key=lambda r: r["repetition_ratio"], reverse=True)[:args.top]:
        text = r.get("raw_response") or ""
        preview = text[:160].replace("\n", " ")
        print(
            f"  ratio={r['repetition_ratio']:.3f} truncated={r.get('is_truncated')} "
            f"model={r['model_key']} cond={r['condition_key']} stage={r['stage']} "
            f"item={r['item_id']}\n    {preview!r}"
        )

    print(
        "\nOnce you've read enough of these to be confident, set "
        "grading.REPETITION_DEGENERATION_THRESHOLD to the chosen value and record the "
        "rationale (which examples justified it) in PROTOCOL.md before the full run."
    )


if __name__ == "__main__":
    main()
