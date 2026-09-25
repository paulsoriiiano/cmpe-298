"""One-off migration: add a `canonical_answer` field to dataset.jsonl.

canonical_answer is a single, language-agnostic, format-normalized final
answer per item (see text_track/PROTOCOL.md section 2 for the derivation
rules and rationale). answer_en / answer_ilo are left untouched.

Validates every derived value against evaluate.py's own parse_expected_answer
and normalize_answer before writing anything, so canonical_answer is provably
consistent with the current grading logic.
"""
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.jsonl")

sys.path.insert(0, SCRIPT_DIR)
from evaluate import parse_expected_answer, normalize_answer  # noqa: E402

# Ilokano <-> English tokens that are semantically equivalent but not
# string-equal, so normalize_answer() alone can't relate them.
_ILOKANO_YES_NO = {"wen": "yes", "saan": "no"}


def derive_canonical_answer(item):
    source = item["source"]
    answer_en = item["answer_en"]

    if source == "gsm8k":
        return parse_expected_answer(answer_en)
    if source == "bbh_logical_deduction":
        return answer_en.strip().strip("()").upper()
    if source == "bbh_causal_judgement":
        mapping = {"yes": "YES", "no": "NO"}
        return mapping[answer_en.strip().lower()]
    if source in ("mmlu_conceptual_physics", "mmlu_formal_logic"):
        return answer_en.strip().upper()
    raise ValueError(f"Unknown source: {source!r} (id={item.get('id')})")


def normalized_class(text):
    """normalize_answer() output, mapped through the Ilokano yes/no lookup
    so 'Wen' and 'Yes' land in the same class for validation purposes."""
    n = normalize_answer(text)
    return _ILOKANO_YES_NO.get(n, n)


def validate(item, canonical_answer):
    expected = normalized_class(parse_expected_answer(item["answer_en"]))
    got = normalized_class(canonical_answer)
    if got != expected:
        return f"mismatch vs answer_en: canonical={canonical_answer!r} -> {got!r}, expected {expected!r}"

    if item["source"] == "bbh_causal_judgement":
        ilo_class = normalized_class(parse_expected_answer(item["answer_ilo"]))
        if ilo_class != got:
            return f"mismatch vs answer_ilo: canonical={canonical_answer!r} -> {got!r}, answer_ilo class {ilo_class!r}"

    return None


def main():
    items = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    failures = []
    per_source_counts = {}
    for item in items:
        canonical_answer = derive_canonical_answer(item)
        error = validate(item, canonical_answer)
        per_source_counts.setdefault(item["source"], 0)
        per_source_counts[item["source"]] += 1
        if error:
            failures.append((item["id"], item["source"], error))
        else:
            item["canonical_answer"] = canonical_answer

    print("Per-source item counts:")
    for source, count in sorted(per_source_counts.items()):
        print(f"  {source:28s}: {count}")

    if failures:
        print(f"\nVALIDATION FAILED — {len(failures)} item(s) did not validate:")
        for item_id, source, error in failures:
            print(f"  [{source}] {item_id}: {error}")
        print("\nNo changes written.")
        sys.exit(1)

    print(f"\nValidation passed for all {len(items)} items.")

    backup_path = DATASET_PATH + ".bak"
    shutil.copyfile(DATASET_PATH, backup_path)
    print(f"Backed up original to {backup_path}")

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote canonical_answer into {DATASET_PATH}")


if __name__ == "__main__":
    main()
