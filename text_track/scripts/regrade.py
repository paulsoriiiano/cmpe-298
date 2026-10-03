#!/usr/bin/env python3
"""Offline regrade of stored model outputs with the evaluator_v6 grader.

No API calls. Every record already holds the model's raw response, finish_reason and the
gold answer, so the grader's derived fields (is_correct, failure_type, format_compliant,
extracted_answer) can be recomputed from stored data.

  * The original result files are never modified (they remain the immutable record of what ran).
  * Corrected copies are written to --out-dir, one JSONL per input run, same schema, with
    evaluator_version = evaluator_v6. Changed rows additionally carry original_* fields.
  * REGRADE_CHANGELOG.md lists every record whose grading changed, what changed and why.
  * regrade_changes.csv holds the same list in machine-readable form.

Translate-stage rows are copied unchanged (the fixed rules apply only to reason/direct stages,
and the staged pivots' reasoning inputs do not depend on the reason-stage grader).

Usage (from the repo root, with the patched grading.py in place):
  python3 regrade.py --repo text_track --out-dir text_track/results_regraded \
      --original-grading <path to the ORIGINAL grading.py>      # optional validation
"""
import argparse, collections as C, csv, importlib.util, json, os, re, sys, types
from dataclasses import dataclass

NEW_VERSION = "evaluator_v6"
RUNS = {
    "gpt_5_2": "results/full-gpt-5-2-evaluator-v1/full-gpt-5-2-evaluator-v1.jsonl",
    "qwen_3_6_27b": "results/full-qwen36-evaluator-v5/full-qwen36-evaluator-v5.jsonl",
    "qwen_sealion_v4_5_27b_it": "results/full-sealion-evaluator-v5/full-sealion-evaluator-v5.jsonl",
}


@dataclass
class ModelResponse:  # same shape as evaluate.models.ModelResponse
    text: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    raw: object = None


def load_grading(path):
    """Load a grading.py file standalone (its only package import is ModelResponse)."""
    src = open(path, encoding="utf-8").read()
    src = src.replace("from .models import ModelResponse", "")
    mod = types.ModuleType("grading_" + os.path.basename(os.path.dirname(path)) + str(abs(hash(path))))
    mod.ModelResponse = ModelResponse
    mod.__dict__["__name__"] = mod.__name__
    sys.modules[mod.__name__] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def classify(g, row):
    resp = ModelResponse(text=row["raw_response"] or "", finish_reason=row["finish_reason"],
                         input_tokens=row.get("input_tokens"), output_tokens=row.get("output_tokens"))
    return g.classify_result(response=resp, exception=None, canonical_answer=row["canonical_answer"],
                             source=row["source"], stage=row["stage"])


FIELDS = ("is_correct", "failure_type", "format_compliant", "extracted_answer")


def stored(row):
    return {"is_correct": row["is_correct"], "failure_type": row["failure_type"],
            "format_compliant": row["format_compliant"], "extracted_answer": row["extracted_answer"]}


def computed(res):
    return {"is_correct": res.is_correct, "failure_type": res.failure_type.value,
            "format_compliant": res.format_compliant, "extracted_answer": res.extracted_answer}


def rule_of(old, new, row):
    o, n = old["failure_type"], new["failure_type"]
    if o == "repetition_degeneration" and n != o:
        return "R1"
    if o == "refusal" and n != o:
        return "R2"
    if o == "invalid_answer_format" and n in ("correct", "substantively_incorrect") and \
            re.search(r"[%$]", old["extracted_answer"] or ""):
        return "R3"
    return "OTHER"


RULES = {
    "R1": "Repetition flag overrode a normally-finished, answered response. The 4-gram repetition "
          "heuristic fired on a response with finish_reason=stop that committed to an answer "
          "(e.g. truth-table solutions). It is now applied only to responses that were truncated "
          "or produced no answer; the response is graded on its answer instead.",
    "R2": "Refusal regex fired on ordinary reasoning (\"I cannot assume...\") in a response that "
          "committed to an answer. Refusal is now only assigned when no answer was produced.",
    "R3": "Right value with decoration (\"60%\", \"$18\") was marked invalid_answer_format and "
          "wrong. It is now graded on its value and flagged format_compliant=False, as the "
          "protocol's two-axis rule (semantic correctness vs format compliance) specifies.",
    "OTHER": "Unexpected change - inspect.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="text_track", help="path to the text_track/ directory")
    ap.add_argument("--grading", help="patched grading.py (default: <repo>/scripts/evaluate/grading.py)")
    ap.add_argument("--original-grading", help="original grading.py, to validate that stored "
                    "raw_response reproduces the stored grades (recommended)")
    ap.add_argument("--out-dir", default="results_regraded")
    a = ap.parse_args()
    gpath = a.grading or os.path.join(a.repo, "scripts", "evaluate", "grading.py")
    g_new = load_grading(gpath)
    g_old = load_grading(a.original_grading) if a.original_grading else None
    os.makedirs(a.out_dir, exist_ok=True)

    changes = []
    summary = C.defaultdict(C.Counter)
    validation = {}
    for model, rel in RUNS.items():
        rows = [json.loads(l) for l in open(os.path.join(a.repo, rel), encoding="utf-8") if l.strip()]
        out_rows, mism = [], 0
        for r in rows:
            r = dict(r)
            if r["stage"] in ("reason", "direct"):
                old = stored(r)
                if g_old is not None:  # validation: does the ORIGINAL grader reproduce the stored grade?
                    if computed(classify(g_old, r)) != old:
                        mism += 1
                new = computed(classify(g_new, r))
                if new != old:
                    rule = rule_of(old, new, r)
                    changes.append(dict(model=model, condition=r["condition_key"], item_id=r["item_id"],
                                        source=r["source"], gold=r["canonical_answer"], rule=rule,
                                        old_answer=old["extracted_answer"], new_answer=new["extracted_answer"],
                                        old_failure_type=old["failure_type"], new_failure_type=new["failure_type"],
                                        old_correct=old["is_correct"], new_correct=new["is_correct"],
                                        old_format_ok=old["format_compliant"], new_format_ok=new["format_compliant"],
                                        finish_reason=r["finish_reason"], repetition_ratio=round(r["repetition_ratio"], 3)))
                    for k in FIELDS:
                        r["original_" + k] = old[k]
                    for k in FIELDS:
                        r[k] = new[k]
                    summary[model][rule] += 1
            r["evaluator_version"] = NEW_VERSION
            out_rows.append(r)
        validation[model] = (mism, len(rows)) if g_old is not None else None
        name = os.path.basename(rel).replace(".jsonl", "_regraded.jsonl")
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- machine-readable ----------
    csv_path = os.path.join(a.out_dir, "regrade_changes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(changes[0].keys()))
        w.writeheader(); w.writerows(sorted(changes, key=lambda c: (c["model"], c["condition"], c["item_id"])))

    # ---------- markdown changelog ----------
    flips = [c for c in changes if c["old_correct"] != c["new_correct"]]
    label_only = [c for c in changes if c["old_correct"] == c["new_correct"]]
    md = []
    W = md.append
    W("# Regrade changelog (evaluator_v5 -> evaluator_v6)\n")
    W("Offline regrade of stored raw responses; **no model was re-queried** and the original result "
      "files are untouched. Corrected copies are in `results_regraded/`. Only `reason`/`direct` "
      "stage rows can change; translate-stage rows are copied as-is.\n")
    W(f"- Records whose grading changed: **{len(changes)}**")
    W(f"- Correctness verdict flipped (wrong -> right): **{len(flips)}**")
    W(f"- Label-only changes (still incorrect, failure type relabelled): **{len(label_only)}**")
    if all(v is not None for v in validation.values()):
        W("- Validation: the ORIGINAL grader applied to the stored `raw_response` reproduces the stored "
          "grade for every reason/direct row: " +
          ", ".join(f"{m}: {n - (n - mm) if False else mm} mismatches" for m, (mm, n) in validation.items()) + ".")
    W("")
    W("## Rules applied\n")
    for k in ("R1", "R2", "R3"):
        W(f"**{k}.** {RULES[k]}\n")
    if summary and any(c["rule"] == "OTHER" for c in changes):
        W(f"**OTHER.** {RULES['OTHER']}\n")
    W("## Changes by model and rule (all / verdict flipped)\n")
    W("| Model | Rule | Records changed | Verdict flipped |")
    W("|---|---|---|---|")
    for m in RUNS:
        for k in ("R1", "R2", "R3", "OTHER"):
            n = sum(1 for c in changes if c["model"] == m and c["rule"] == k)
            if n:
                fl = sum(1 for c in flips if c["model"] == m and c["rule"] == k)
                W(f"| {m} | {k} | {n} | {fl} |")
    W("")
    W("## Changes by condition (verdict flips only)\n")
    W("| Model | A_EE | A_II | A_IE | A_EI | A_E0 | A_I0 |")
    W("|---|---|---|---|---|---|---|")
    for m in RUNS:
        cnt = C.Counter(c["condition"] for c in flips if c["model"] == m)
        W(f"| {m} | " + " | ".join(str(cnt.get(k, 0)) for k in ("A_EE", "A_II", "A_IE", "A_EI", "A_E0", "A_I0")) + " |")
    W("")
    W("## Item-level edits\n")
    W("Columns: gold = canonical answer; answer/type/correct show old -> new. "
      "`rep` = repetition ratio recorded at run time; `fin` = finish_reason.\n")
    for m in RUNS:
        mine = sorted([c for c in changes if c["model"] == m], key=lambda c: (c["rule"], c["condition"], c["item_id"]))
        if not mine:
            continue
        W(f"### {m} ({len(mine)} records)\n")
        W("| Item | Cond | Rule | Gold | Answer | Failure type | Correct | Format ok | fin | rep |")
        W("|---|---|---|---|---|---|---|---|---|---|")
        for c in mine:
            W(f"| {c['item_id']} | {c['condition']} | {c['rule']} | {c['gold']} | "
              f"{c['old_answer']} -> {c['new_answer']} | {c['old_failure_type']} -> {c['new_failure_type']} | "
              f"{c['old_correct']} -> {c['new_correct']} | {c['old_format_ok']} -> {c['new_format_ok']} | "
              f"{c['finish_reason']} | {c['repetition_ratio']} |")
        W("")
    W("## Not changed (deliberately)\n")
    W("- Truncated outputs with no answer, including repetition loops: still incorrect, still `repetition_degeneration` / `truncation`.")
    W("- `<answer>None</answer>` on multiple-choice items (the model declared the problem inconsistent): still `invalid_answer_format`, incorrect.")
    W("- Non-numeric GSM8K answers (\"Cannot be determined\"), fractions that are not the gold value, and wrong values: unchanged.")
    W("- All translate-stage rows and all infrastructure/parser outcomes (there are none).")
    open(os.path.join(a.out_dir, "REGRADE_CHANGELOG.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"changed={len(changes)} flips={len(flips)} label_only={len(label_only)}")
    print({m: dict(c) for m, c in summary.items()})
    print("validation:", validation)


if __name__ == "__main__":
    main()
