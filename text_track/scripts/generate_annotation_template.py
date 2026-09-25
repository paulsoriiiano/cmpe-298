"""Bootstrap a manual annotation template from a completed evaluator run.

Manipulation/language checks (does the model actually reason in Ilokano when asked to?
is a translation faithful?) can't be trusted to automatic language ID — see PROTOCOL.md
section 9. This script reads one run's result JSONL (immutable, never modified) and writes
a CSV with the judgment columns blank, ready for a human annotator to fill in and join back
against the inference data during analysis by (run_id, item_id, model_key, condition_key,
stage).

Usage:
    python3 text_track/scripts/generate_annotation_template.py <run_id>
"""
import argparse
import json
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "eval_runs")
ANNOTATIONS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "annotations")

KEY_COLUMNS = ["run_id", "item_id", "model_key", "condition_key", "stage"]

CONTEXT_COLUMNS = [
    "source", "original_input", "generated_translation", "generated_rationale",
    "extracted_answer", "canonical_answer", "is_correct", "format_compliant", "failure_type",
]

# Filled in programmatically — mechanically derivable from the run data, not a judgment call.
DERIVED_ANNOTATION_COLUMNS = ["rationale_present", "requested_language"]

# Left blank — require a human annotator's judgment.
BLANK_ANNOTATION_COLUMNS = [
    "language_compliance",      # compliant | mixed | noncompliant | uncertain
    "translation_faithfulness",  # accurate | minor_error | major_error | unusable
    "translation_error_type",     # none | lexical | morphological | semantic | omission | addition
    "annotation_notes",
    "annotator",
]

# requested_language per (condition_key, stage) — mirrors conditions.py's reasoning_lang /
# translation_target_lang, without importing the evaluate package (this script only reads
# already-written result JSONL, it doesn't need the condition definitions at runtime).
_REQUESTED_LANGUAGE = {
    ("A_EE", "reason"): "en",
    ("A_II", "reason"): "ilo",
    ("A_IE", "translate"): "en",
    ("A_IE", "reason"): "en",
    ("A_EI", "translate"): "ilo",
    ("A_EI", "reason"): "ilo",
    ("A_E0", "direct"): "en",
    ("A_I0", "direct"): "ilo",
}


def load_run(run_id: str) -> list[dict]:
    path = os.path.join(RUNS_DIR, f"{run_id}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Is {run_id!r} a real run_id?")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_annotation_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        row = {col: r.get(col) for col in KEY_COLUMNS + CONTEXT_COLUMNS}
        row["rationale_present"] = bool((r.get("generated_rationale") or "").strip())
        row["requested_language"] = _REQUESTED_LANGUAGE.get((r["condition_key"], r["stage"]))
        for col in BLANK_ANNOTATION_COLUMNS:
            row[col] = ""
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap a manual annotation template CSV from a completed run."
    )
    parser.add_argument("run_id")
    args = parser.parse_args()

    records = load_run(args.run_id)
    rows = build_annotation_rows(records)

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    output_path = os.path.join(ANNOTATIONS_DIR, f"{args.run_id}_annotation_template.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
