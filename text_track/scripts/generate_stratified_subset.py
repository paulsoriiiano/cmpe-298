"""One-off generation of the fixed 300-item stratified subset used by the A_E0/A_I0
direct-answer control conditions (see text_track/PROTOCOL.md section 1).

Selection is deterministic (fixed seed) and computed once into a static JSON artifact
rather than resampled at run time, so "every model must receive exactly the same selected
IDs" holds regardless of Python version, dataset row order, or which machine runs the
evaluator.
"""
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "data", "dataset.jsonl")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "stratified_subset_300.json")

SEED = 42
SUBSET_FRACTION = 0.30

# Expected counts per PROTOCOL.md section 1 (30% of each source's full count, rounded).
EXPECTED_COUNTS = {
    "gsm8k": 120,
    "bbh_logical_deduction": 75,
    "bbh_causal_judgement": 15,
    "mmlu_conceptual_physics": 52,
    "mmlu_formal_logic": 38,
}


def main():
    items_by_source = {}
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            items_by_source.setdefault(item["source"], []).append(item["id"])

    rng = random.Random(SEED)
    selected_ids = []
    actual_counts = {}
    for source, ids in sorted(items_by_source.items()):
        ids_sorted = sorted(ids)  # deterministic input order before shuffling
        n = round(len(ids_sorted) * SUBSET_FRACTION)
        rng.shuffle(ids_sorted)
        chosen = sorted(ids_sorted[:n])
        selected_ids.extend(chosen)
        actual_counts[source] = len(chosen)

    print("Selected counts per source:")
    mismatches = []
    for source, expected in EXPECTED_COUNTS.items():
        actual = actual_counts.get(source, 0)
        status = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            mismatches.append((source, expected, actual))
        print(f"  {source:28s}: {actual} (expected {expected}) [{status}]")

    unexpected_sources = set(actual_counts) - set(EXPECTED_COUNTS)
    if unexpected_sources:
        mismatches.append(("unexpected_sources", 0, list(unexpected_sources)))

    if mismatches:
        print(f"\nMISMATCH against PROTOCOL.md's expected per-source counts: {mismatches}")
        print("Not writing output — update EXPECTED_COUNTS or investigate the dataset first.")
        raise SystemExit(1)

    selected_ids.sort()
    output = {
        "total": len(selected_ids),
        "seed": SEED,
        "subset_fraction": SUBSET_FRACTION,
        "source_counts": actual_counts,
        "ids": selected_ids,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(selected_ids)} IDs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
