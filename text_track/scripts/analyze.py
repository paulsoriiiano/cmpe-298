"""Numerical analysis of the staged, six-condition cross-lingual evaluation.

Rewritten for the flat per-(item, model, condition, stage) JSONL schema written by
text_track/scripts/evaluate/ (see storage.ResultRecord) — the previous version of this
script assumed a nested {model: {pass: {...}}} row per item, hard-coded to Claude/Llama and
the old 3-pass names, and cannot read the current schema at all.

Conditions: A_EE, A_II, A_IE, A_EI (reasoning; staged pivots A_IE/A_EI have a "translate"
stage that is NOT graded) and A_E0/A_I0 (direct-answer controls, stubbed as of this writing).

Accuracy denominator: one resolved condition-level outcome per planned item (see
resolve_condition_outcomes()). For single-call conditions (A_EE/A_II/A_E0/A_I0) this is
just that item's reason/direct-stage is_correct. For staged pivot conditions (A_IE/A_EI),
a model-caused translation failure (translation_format_failure, truncation, or refusal)
counts as an end-to-end incorrect outcome for that item, even though no reason-stage row
exists — the model failed to produce a usable pivot translation, which IS an incorrect
outcome, not a missing one. Infrastructure/parser failures (translation- or reason-stage)
remain unresolved (excluded from both numerator and denominator) and are reported as a
validation warning (or a hard failure under --strict) rather than silently folded into
either — they represent incomplete work, not a graded model outcome.

Usage:
    python3 text_track/scripts/analyze.py <run_id> [--strict]
    python3 text_track/scripts/analyze.py --path /path/to/some.jsonl
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "eval_runs")
ANALYSES_DIR = os.path.join(SCRIPT_DIR, "..", "data", "analyses")
# NOTE: this must never default to data/analysis.md — that file is a preserved artifact
# from the old 3-pass pipeline (per the conference plan: "Do not combine old and new
# experimental results"). Each new-schema run gets its own dated file under ANALYSES_DIR.

REASON_CONDITIONS_PRIMARY = ["A_EE", "A_II", "A_IE"]
ALL_REASONING_CONDITIONS = ["A_EE", "A_II", "A_IE", "A_EI", "A_E0", "A_I0"]
GRADED_STAGES = {"reason", "direct"}
NOT_EVALUATED_FAILURE_TYPES = {"infrastructure_api_failure", "parser_failure"}

# The primary comparison family per PROTOCOL.md section 6 — paired (same item, same model),
# compared via McNemar's test since every condition runs over the same item IDs.
PRIMARY_COMPARISONS = [
    ("A_EE", "A_II", "overall language gap"),
    ("A_II", "A_IE", "English-pivot benefit"),
    ("A_EE", "A_IE", "remaining pivot gap"),
]


def load_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _planned_keys(manifest: dict) -> set:
    """The full set of (item_id, model_key, condition_key, stage) units the CURRENT run's
    manifest actually plans to produce — used to bound what merge_resumed_rows() is allowed
    to pull in from an ancestor run's file (which may contain thousands of unrelated rows
    from a much larger prior experiment)."""
    planned = set()
    planned_by_condition = manifest.get("planned_item_ids_by_condition", {})
    model_keys = manifest.get("models", [])
    for condition_key, item_ids in planned_by_condition.items():
        for stage in _expected_stages(condition_key):
            for model_key in model_keys:
                for item_id in item_ids:
                    planned.add((item_id, model_key, condition_key, stage))
    return planned


def merge_resumed_rows(rows: list[dict], ancestor_run_ids: list[str], manifest: dict) -> list[dict]:
    """A run that resumed from prior runs only WRITES the units it actually executed
    itself — units it skipped (because a prior run already completed them) are never
    copied into its own JSONL. Reading only {run_id}.jsonl therefore silently under-counts
    completeness for any resumed run. The manifest's resuming_from_run_ids already gives
    the exact set of other run files that hold those skipped units (resume_index scans
    every file in RUNS_DIR, so a unit is attributed to whichever run's file actually
    contains it — one level is enough; there is no deeper chain to walk).

    An ancestor run's file is NOT filtered to the current run's item selection by default —
    it may be a much larger prior experiment (e.g. 1,000 items) that this run only resumed
    5 of. Pulling in every row from that file would silently inflate this run's analysis
    with unrelated data. So a row is only merged in if its key is BOTH (a) not already
    present in `rows`, (b) one of THIS manifest's planned_item_ids_by_condition units, and
    (c) compatible with this run's dataset_version/protocol_version/evaluator_version and
    (per model) config_fingerprint — an ancestor row that used a different model
    configuration is not a valid substitute for this run's own planned unit.
    """
    present = {(r["item_id"], r["model_key"], r["condition_key"], r["stage"]) for r in rows}
    planned_keys = _planned_keys(manifest)
    dataset_version = manifest.get("dataset_version")
    protocol_version = manifest.get("protocol_version")
    evaluator_version = manifest.get("evaluator_version")
    model_settings = manifest.get("model_settings", {})

    merged = list(rows)
    for ancestor_run_id in ancestor_run_ids:
        ancestor_path = os.path.join(RUNS_DIR, f"{ancestor_run_id}.jsonl")
        if not os.path.exists(ancestor_path):
            continue
        for r in load_rows(ancestor_path):
            key = (r["item_id"], r["model_key"], r["condition_key"], r["stage"])
            if key in present or key not in planned_keys:
                continue
            if r.get("dataset_version") != dataset_version:
                continue
            if r.get("protocol_version") != protocol_version:
                continue
            if r.get("evaluator_version") != evaluator_version:
                continue
            expected_fp = model_settings.get(r["model_key"], {}).get("config_fingerprint")
            if expected_fp is None or r.get("config_fingerprint") != expected_fp:
                continue
            merged.append(r)
            present.add(key)
    return merged


def load_run(run_id: str) -> list[dict]:
    path = os.path.join(RUNS_DIR, f"{run_id}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Is {run_id!r} a real run_id?")
    rows = load_rows(path)
    try:
        manifest = load_manifest(run_id)
    except FileNotFoundError:
        return rows
    ancestor_run_ids = manifest.get("resuming_from_run_ids", [])
    if not ancestor_run_ids:
        return rows
    return merge_resumed_rows(rows, ancestor_run_ids, manifest)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial) p-value on discordant counts b, c.
    Under H0 each discordant pair is a fair coin: b ~ Binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_chi2_cc(b, c):
    """McNemar chi-square statistic with continuity correction (df=1)."""
    if b + c == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / (b + c)


def chi2_sf_df1(x):
    """Survival function (upper-tail p-value) of chi-square with 1 df."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment. Returns adjusted p-values in the ORIGINAL input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    adjusted = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion — better-behaved than the normal
    approximation near 0/1, which matters for small per-source samples (e.g. n=15)."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ---------------- validation ---------------- #

def validate(rows: list[dict], manifest: dict | None = None) -> list[str]:
    """Returns a list of warning strings. Does not raise — callers decide (via --strict)
    whether any warnings should be treated as fatal.

    If `manifest` is given (the run's manifest.json, loaded via load_manifest()), also
    checks experiment completeness against it: every planned item/model/condition/stage
    unit must have a row. Without the manifest this can't be checked at all — rows-only
    validation has no way to know about work that never ran in the first place."""
    warnings = []

    seen = Counter((r["item_id"], r["model_key"], r["condition_key"], r["stage"]) for r in rows)
    duplicates = {k: v for k, v in seen.items() if v > 1}
    if duplicates:
        warnings.append(f"{len(duplicates)} duplicate (item, model, condition, stage) keys found.")

    not_evaluated = [r for r in rows if r.get("failure_type") in NOT_EVALUATED_FAILURE_TYPES]
    if not_evaluated:
        by_cell = Counter((r["model_key"], r["condition_key"], r["failure_type"]) for r in not_evaluated)
        detail = "; ".join(f"{mk}/{ck}/{ft}={n}" for (mk, ck, ft), n in sorted(by_cell.items()))
        warnings.append(
            f"{len(not_evaluated)} infrastructure/parser failure(s) excluded from accuracy "
            f"denominators (rerun these before treating numbers as final): {detail}"
        )

    protocol_versions = {r.get("protocol_version") for r in rows}
    evaluator_versions = {r.get("evaluator_version") for r in rows}
    if len(protocol_versions) > 1:
        warnings.append(f"Mixed protocol_version values in this file: {protocol_versions}")
    if len(evaluator_versions) > 1:
        warnings.append(f"Mixed evaluator_version values in this file: {evaluator_versions}")

    if manifest is not None:
        warnings.extend(check_manifest_completeness(rows, manifest))

    return warnings


# ---------------- accuracy ---------------- #

def in_accuracy_denominator(r: dict) -> bool:
    return r["stage"] in GRADED_STAGES and r["is_correct"] is not None


def is_success(r: dict) -> bool:
    return r["is_correct"] is True


PIVOT_CONDITIONS = {"A_IE", "A_EI"}
DIRECT_CONDITIONS = {"A_E0", "A_I0"}

# Translate-stage failure_types that are the MODEL's fault, not infrastructure's — a
# translation that was refused, truncated, or malformed all mean the model failed to
# produce a usable pivot translation, so the item counts as end-to-end incorrect rather
# than being excluded as unresolved. Infra/parser failures are the only case that stays
# unresolved (None) — see resolve_condition_outcomes().
MODEL_CAUSED_TRANSLATION_FAILURES = {
    "translation_format_failure", "truncation", "refusal",
}


def _expected_stages(condition_key: str) -> list[str]:
    if condition_key in PIVOT_CONDITIONS:
        return ["translate", "reason"]
    if condition_key in DIRECT_CONDITIONS:
        return ["direct"]
    return ["reason"]


def load_manifest(run_id: str, runs_dir: str | None = None) -> dict:
    path = os.path.join(runs_dir if runs_dir is not None else RUNS_DIR, f"{run_id}.manifest.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_manifest_completeness(rows: list[dict], manifest: dict) -> list[str]:
    """Verifies every planned (item, model, condition, stage) unit from the manifest has a
    corresponding row. This catches work that never ran at all (e.g. a crashed run that was
    never resumed) — validate()'s other checks only look at rows that DO exist, so they
    can't detect units that are simply missing outright.

    Missing-reason-after-model-caused-translation-failure and
    missing-reason-after-infra-failure are NOT reported as separate warnings here — those
    are expected, understood states already surfaced by resolve_condition_outcomes() (the
    former resolves to an end-to-end-incorrect outcome, the latter to unresolved and is
    already flagged by validate()'s NOT_EVALUATED_FAILURE_TYPES warning). This check only
    flags units with NO row at all for a stage that should have at least attempted a call.
    """
    warnings = []
    present = {
        (r["item_id"], r["model_key"], r["condition_key"], r["stage"]) for r in rows
    }
    translate_outcome = {
        (r["item_id"], r["model_key"], r["condition_key"]): r.get("failure_type")
        for r in rows if r["stage"] == "translate"
    }

    planned_by_condition = manifest.get("planned_item_ids_by_condition", {})
    model_keys = manifest.get("models", [])
    missing = []
    for condition_key, item_ids in planned_by_condition.items():
        expected_stages = _expected_stages(condition_key)
        for model_key in model_keys:
            for item_id in item_ids:
                for stage in expected_stages:
                    if (item_id, model_key, condition_key, stage) in present:
                        continue
                    if stage == "reason" and condition_key in PIVOT_CONDITIONS:
                        translate_failure_type = translate_outcome.get(
                            (item_id, model_key, condition_key)
                        )
                        if translate_failure_type in MODEL_CAUSED_TRANSLATION_FAILURES or (
                            translate_failure_type
                            in NOT_EVALUATED_FAILURE_TYPES
                        ):
                            continue
                    missing.append((item_id, model_key, condition_key, stage))

    if missing:
        by_cell = Counter((mk, ck, stage) for _, mk, ck, stage in missing)
        detail = "; ".join(f"{mk}/{ck}/{stage}={n}" for (mk, ck, stage), n in sorted(by_cell.items()))
        warnings.append(
            f"{len(missing)} planned (item, model, condition, stage) unit(s) from the "
            f"manifest have NO row at all — never ran, or ran but was never written: {detail}"
        )
    return warnings


def resolve_condition_outcomes(rows: list[dict], model_key: str, condition_key: str) -> dict:
    """Returns {item_id: is_correct_or_None} — one condition-level outcome per item, for
    single-stage conditions (direct/reason) as well as staged pivots.

    For A_IE/A_EI, a failed translation means no "reason" row is ever produced for that
    item — simply reading reason-stage rows (as in_accuracy_denominator does) would make
    that item silently vanish from the denominator rather than count against the pivot
    condition's accuracy. Reconstruct the outcome per item instead of fabricating a fake
    reasoning record in the raw JSONL (which must stay an immutable record of what actually
    ran):
      - a completed reason-stage row exists -> use its is_correct
      - the translate-stage row is a model-caused failure (MODEL_CAUSED_TRANSLATION_FAILURES:
        translation_format_failure, truncation, or refusal) with no corresponding reason
        row -> end-to-end incorrect (False), since the model failed to produce a usable
        pivot translation
      - the translate-stage row is an infrastructure/parser failure with no corresponding
        reason row -> unresolved (None); this is incomplete work, not a graded outcome
      - no rows at all for this item/model/condition -> not present in the returned dict
    """
    condition_rows = [
        r for r in rows if r["model_key"] == model_key and r["condition_key"] == condition_key
    ]
    if condition_key not in PIVOT_CONDITIONS:
        return {
            r["item_id"]: r["is_correct"] for r in condition_rows if in_accuracy_denominator(r)
        }

    reason_by_item = {r["item_id"]: r for r in condition_rows if r["stage"] == "reason"}
    translate_by_item = {r["item_id"]: r for r in condition_rows if r["stage"] == "translate"}

    outcomes = {}
    for item_id, translate_row in translate_by_item.items():
        reason_row = reason_by_item.get(item_id)
        if reason_row is not None:
            outcomes[item_id] = reason_row["is_correct"]
        elif translate_row.get("failure_type") in MODEL_CAUSED_TRANSLATION_FAILURES:
            outcomes[item_id] = False
        else:
            outcomes[item_id] = None
    return outcomes


def accuracy_table(rows: list[dict], group_keys: tuple) -> dict:
    """Groups denominator-eligible rows by group_keys (e.g. ("model_key", "condition_key")
    or ("model_key", "condition_key", "source")) and returns
    {group: {"correct": int, "total": int}}.

    For A_IE/A_EI, outcomes are reconstructed per item via resolve_condition_outcomes() so
    model-caused translation failures count as incorrect instead of silently disappearing
    from the denominator; unresolved (infra/parser-failure) items are excluded here, same
    as elsewhere, and reported separately by validate().
    """
    item_source = {r["item_id"]: r["source"] for r in rows}
    table = defaultdict(lambda: {"correct": 0, "total": 0})
    pairs = {(r["model_key"], r["condition_key"]) for r in rows}

    for model_key, condition_key in pairs:
        outcomes = resolve_condition_outcomes(rows, model_key, condition_key)
        for item_id, is_correct in outcomes.items():
            if is_correct is None:
                continue
            group_values = {"model_key": model_key, "condition_key": condition_key}
            if "source" in group_keys:
                group_values["source"] = item_source.get(item_id)
            key = tuple(group_values[k] for k in group_keys)
            table[key]["total"] += 1
            if is_correct:
                table[key]["correct"] += 1
    return dict(table)


def rate_table(rows: list[dict], flag_field: str, group_keys: tuple, stage_filter=GRADED_STAGES) -> dict:
    """Fraction of rows (within `stage_filter`) where rows[flag_field] is truthy, grouped by
    group_keys. Used for truncation/degeneration rates — these are computed over ALL
    stage-matching rows (not just the accuracy denominator), since a translate-stage
    degeneration score is still worth reporting even though it's never auto-classified."""
    table = defaultdict(lambda: {"flagged": 0, "total": 0})
    for r in rows:
        if r["stage"] not in stage_filter:
            continue
        if r.get(flag_field) is None:
            continue
        key = tuple(r[k] for k in group_keys)
        table[key]["total"] += 1
        if r.get(flag_field):
            table[key]["flagged"] += 1
    return dict(table)


# ---------------- paired comparisons ---------------- #

def paired_comparison(rows: list[dict], model_key: str, cond_a: str, cond_b: str) -> dict | None:
    """McNemar comparison of two conditions' correctness for one model, paired by item_id.
    Only items present (and resolved, i.e. not None) in BOTH conditions count. Pivot
    conditions (A_IE/A_EI) use resolve_condition_outcomes() so a model-caused translation
    failure counts as incorrect rather than making the item vanish from the pairing."""
    outcomes_a = resolve_condition_outcomes(rows, model_key, cond_a)
    outcomes_b = resolve_condition_outcomes(rows, model_key, cond_b)
    by_item_a = {item_id: v for item_id, v in outcomes_a.items() if v is not None}
    by_item_b = {item_id: v for item_id, v in outcomes_b.items() if v is not None}
    shared_items = sorted(set(by_item_a) & set(by_item_b))
    if not shared_items:
        return None

    both = a_only = b_only = neither = 0
    for item_id in shared_items:
        ca, cb = by_item_a[item_id], by_item_b[item_id]
        if ca and cb:
            both += 1
        elif ca and not cb:
            a_only += 1
        elif cb and not ca:
            b_only += 1
        else:
            neither += 1

    # b/c convention: b = correct under cond_b but not cond_a; c = correct under cond_a but not cond_b
    b, c = b_only, a_only
    chi2 = mcnemar_chi2_cc(b, c)
    # Accuracies must be computed over shared_items — the same paired sample McNemar uses —
    # not over each condition's full independently-available set. If one condition has an
    # extra item the other lacks, using each condition's own denominator would report an
    # accuracy difference inconsistent with the paired test's actual sample.
    acc_a = sum(1 for i in shared_items if by_item_a[i]) / len(shared_items)
    acc_b = sum(1 for i in shared_items if by_item_b[i]) / len(shared_items)
    return {
        "n_paired": len(shared_items), "both": both, "a_only": a_only, "b_only": b_only,
        "neither": neither, "discordant_b": b, "discordant_c": c,
        "p_exact": mcnemar_exact(b, c), "chi2_cc": chi2, "p_chi2": chi2_sf_df1(chi2),
        "accuracy_a": acc_a * 100, "accuracy_b": acc_b * 100,
        "paired_diff_pp": (acc_b - acc_a) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze a staged, six-condition evaluation run.")
    parser.add_argument("run_id", nargs="?", help="run_id under text_track/data/eval_runs/")
    parser.add_argument("--path", help="Explicit path to a result JSONL, instead of run_id.")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with an error instead of a warning if infrastructure/parser failures "
             "or duplicate keys are found (i.e. refuse to report numbers until rerun).",
    )
    parser.add_argument(
        "--manifest", help="Explicit path to a run manifest.json. Required for experiment "
        "completeness checking when using --path with a JSONL that has no sibling "
        "<name>.manifest.json next to it (e.g. HPC results copied without their manifest) "
        "— under --strict, a missing manifest is fatal.",
    )
    parser.add_argument(
        "--output", help="Output path for the analysis report. Defaults to "
        "text_track/data/analyses/<run_id>_analysis.md — never overwrites the preserved "
        "legacy text_track/data/analysis.md artifact.",
    )
    args = parser.parse_args()

    if not args.run_id and not args.path:
        parser.error("Provide a run_id or --path.")
    path = args.path or os.path.join(RUNS_DIR, f"{args.run_id}.jsonl")
    rows = load_rows(path) if args.path else load_run(args.run_id)

    if args.output:
        out_path = args.output
    else:
        stem = args.run_id or os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(ANALYSES_DIR, f"{stem}_analysis.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    manifest = None
    manifest_warnings = []
    if args.path:
        manifest_path = args.manifest or (os.path.splitext(args.path)[0] + ".manifest.json")
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except FileNotFoundError:
            manifest_warnings.append(
                f"no manifest found at {manifest_path!r} — experiment completeness cannot "
                f"be checked. Pass --manifest explicitly, or place a sibling "
                f"<name>.manifest.json next to the JSONL."
            )
    else:
        try:
            manifest = load_manifest(args.run_id)
        except FileNotFoundError:
            manifest_warnings.append(
                f"no manifest found for run_id {args.run_id!r} — experiment completeness "
                f"cannot be checked."
            )

    warnings = manifest_warnings + validate(rows, manifest)
    for w in warnings:
        print(f"WARNING: {w}")
    if warnings and args.strict:
        print("\n--strict was given: refusing to report numbers until the above are resolved.")
        sys.exit(1)

    models = sorted({r["model_key"] for r in rows})
    conditions_present = sorted({r["condition_key"] for r in rows}, key=lambda c: ALL_REASONING_CONDITIONS.index(c) if c in ALL_REASONING_CONDITIONS else 99)
    sources = sorted({r["source"] for r in rows})

    acc_by_model_cond = accuracy_table(rows, ("model_key", "condition_key"))
    acc_by_model_cond_source = accuracy_table(rows, ("model_key", "condition_key", "source"))
    trunc_by_model_cond = rate_table(rows, "is_truncated", ("model_key", "condition_key"))
    degen_by_model_cond = rate_table(rows, "degeneration_candidate", ("model_key", "condition_key"))
    degen_by_model_cond_source = rate_table(rows, "degeneration_candidate", ("model_key", "condition_key", "source"))

    def pct(cell):
        return (cell["correct"] / cell["total"] * 100) if cell["total"] else 0.0

    def rate_pct(cell):
        return (cell["flagged"] / cell["total"] * 100) if cell["total"] else 0.0

    L = []
    W = L.append
    W("# Numerical Analysis — Staged Six-Condition Cross-Lingual Reasoning\n")
    W(f"Source file: `{path}` — **{len(rows)} records**, models: {', '.join(models)}, "
      f"conditions present: {', '.join(conditions_present)}, sources: {', '.join(sources)}.\n")
    if warnings:
        W("**Validation warnings** (see console output above) — numbers below exclude "
          "infrastructure/parser failures from the denominator entirely rather than "
          "silently counting them as model errors:\n")
        for w in warnings:
            W(f"- {w}")
        W("")

    # 1. Accuracy by model x condition
    W("## 1. Accuracy by model and condition\n")
    W("Denominator: one resolved condition-level outcome per planned item. For pivot "
      "conditions (`A_IE`/`A_EI`), model-caused translation failures (format failure, "
      "truncation, refusal) count as incorrect; infrastructure and parser failures remain "
      "unresolved and are excluded. For all conditions, truncation, repetition "
      "degeneration, refusal, missing-answer, and invalid-format all count as incorrect, "
      "per the is_correct semantics in grading.py.\n")
    header = "| Model | " + " | ".join(conditions_present) + " |"
    W(header)
    W("|---|" + "---|" * len(conditions_present))
    for mk in models:
        cells = []
        for ck in conditions_present:
            c = acc_by_model_cond.get((mk, ck), {"correct": 0, "total": 0})
            lo, hi = wilson_ci(c["correct"], c["total"])
            cells.append(f"{c['correct']}/{c['total']} ({pct(c):.1f}%, 95% CI {lo*100:.1f}-{hi*100:.1f})")
        W(f"| {mk} | " + " | ".join(cells) + " |")
    W("")

    # 2. Truncation / degeneration rates
    W("## 2. Truncation and repetition-degeneration rates\n")
    W("Computed over reason/direct-stage records (denominator-eligible), independent of "
      "final failure_type — a record can be both truncated and flagged as a degeneration "
      "candidate; both facts are counted here separately.\n")
    W("| Model | Condition | Truncation rate | Degeneration-candidate rate |")
    W("|---|---|---|---|")
    for mk in models:
        for ck in conditions_present:
            t = trunc_by_model_cond.get((mk, ck), {"flagged": 0, "total": 0})
            d = degen_by_model_cond.get((mk, ck), {"flagged": 0, "total": 0})
            if t["total"] == 0 and d["total"] == 0:
                continue
            W(f"| {mk} | {ck} | {t['flagged']}/{t['total']} ({rate_pct(t):.1f}%) "
              f"| {d['flagged']}/{d['total']} ({rate_pct(d):.1f}%) |")
    W("")
    W("### By source\n")
    W("| Model | Condition | Source | Degeneration-candidate rate |")
    W("|---|---|---|---|")
    for (mk, ck, src), d in sorted(degen_by_model_cond_source.items()):
        if d["total"] == 0:
            continue
        W(f"| {mk} | {ck} | {src} | {d['flagged']}/{d['total']} ({rate_pct(d):.1f}%) |")
    W("")
    W("**Reminder**: `degeneration_candidate` is an automatic flag against "
      "`grading.REPETITION_DEGENERATION_THRESHOLD`, which is NOT YET calibrated from real "
      "pilot data as of this writing (see `inspect_repetition_scores.py`). Treat the rates "
      "above as candidates pending human confirmation, not a final degeneration rate.\n")

    # 3. Primary paired comparisons
    W("## 3. Primary accuracy comparisons (paired, McNemar)\n")
    W("Per PROTOCOL.md section 6: `A_EE` vs `A_II` (overall language gap), `A_II` vs `A_IE` "
      "(English-pivot benefit), `A_EE` vs `A_IE` (remaining pivot gap). Paired by item_id "
      "within each model, since every condition runs over the same item IDs.\n")
    W("**Holm correction scope**: applied *per model*, across that model's 3 primary "
      "comparisons — not pooled across models. Each model is evaluated as its own family "
      "of hypotheses; a p-value from `qwen_3_6_27b` never affects the adjusted threshold "
      "for `claude_sonnet_4_6`. This is a deliberate, pre-registered choice, stated here "
      "explicitly per PROTOCOL.md.\n")
    for mk in models:
        comparisons = []
        for cond_a, cond_b, label in PRIMARY_COMPARISONS:
            if cond_a not in conditions_present or cond_b not in conditions_present:
                continue
            result = paired_comparison(rows, mk, cond_a, cond_b)
            if result is None:
                continue
            comparisons.append((cond_a, cond_b, label, result))
        if not comparisons:
            continue
        W(f"### {mk}\n")
        p_values = [r["p_exact"] for _, _, _, r in comparisons]
        holm = holm_adjust(p_values)
        W("| Comparison | n paired | Acc A | Acc B | Diff (pp) | Discordant (b,c) | "
          "Exact p | Holm-adjusted p |")
        W("|---|---|---|---|---|---|---|---|")
        for (cond_a, cond_b, label, r), p_holm in zip(comparisons, holm):
            W(f"| {cond_a} vs {cond_b} ({label}) | {r['n_paired']} | {r['accuracy_a']:.1f}% "
              f"| {r['accuracy_b']:.1f}% | {r['paired_diff_pp']:+.1f} "
              f"| ({r['discordant_b']}, {r['discordant_c']}) | {r['p_exact']:.3e} "
              f"| {p_holm:.3e} |")
        W("")

    # 4. Per-source accuracy
    W("## 4. Accuracy by source\n")
    W("**Descriptive only** — no per-source paired McNemar test is computed. Several "
      "sources have small cells (e.g. `bbh_causal_judgement` n=15/50), where an exact "
      "binomial test would have very low power and a non-significant result would be "
      "uninformative rather than a genuine null finding. Per-source results below are "
      "reported as accuracy breakdowns for descriptive/exploratory purposes only; the "
      "pre-registered primary inferential comparisons are the pooled ones in section 3.\n")
    for mk in models:
        W(f"### {mk}\n")
        W("| Source | " + " | ".join(conditions_present) + " |")
        W("|---|" + "---|" * len(conditions_present))
        for src in sources:
            cells = []
            for ck in conditions_present:
                c = acc_by_model_cond_source.get((mk, ck, src), {"correct": 0, "total": 0})
                cells.append(f"{c['correct']}/{c['total']} ({pct(c):.1f}%)" if c["total"] else "—")
            W(f"| {src} | " + " | ".join(cells) + " |")
        W("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
