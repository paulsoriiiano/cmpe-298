"""Analyze the TRT evaluation for GPT-5.2 and the two Qwen models.

This consumes the flat ResultRecord JSONL files emitted by evaluate/run.py and writes:
* analysis.md: human-readable tables and results
* analysis.json: machine-readable summaries

It reports exact-match accuracy, the four paired condition gaps, paired McNemar tests, bootstrap confidence intervals, recovery and
regression counts, domain breakdowns, failure rates, translation-stage
completion, and token statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
import re


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUTS = {
    "gpt_5_2": os.path.join(ROOT, "results", "full-gpt-5-2-evaluator-v1", "full-gpt-5-2-evaluator-v1.jsonl"),
    "qwen_3_6_27b": os.path.join(ROOT, "results", "full-qwen36-evaluator-v5", "full-qwen36-evaluator-v5.jsonl"),
    "qwen_sealion_v4_5_27b_it": os.path.join(ROOT, "results", "full-sealion-evaluator-v5", "full-sealion-evaluator-v5.jsonl"),
}
DEFAULT_OUTPUT_MD = os.path.join(ROOT, "data", "analysis.md")
DEFAULT_OUTPUT_JSON = os.path.join(ROOT, "data", "analysis.json")
DEFAULT_DATASET = os.path.join(ROOT, "data", "dataset.jsonl")
DEFAULT_REGRADED_DIR = os.path.join(ROOT, "results_regraded")

REGRADED_FILENAMES = {
    "gpt_5_2": "full-gpt-5-2-evaluator-v1_regraded.jsonl",
    "qwen_3_6_27b": "full-qwen36-evaluator-v5_regraded.jsonl",
    "qwen_sealion_v4_5_27b_it": "full-sealion-evaluator-v5_regraded.jsonl",
}

MODEL_NAMES = {
    "gpt_5_2": "GPT-5.2",
    "qwen_3_6_27b": "Qwen3.6-27B",
    "qwen_sealion_v4_5_27b_it": "Qwen-SEA-LION-v4.5-27B-IT",
}
PRIMARY_CONDITIONS = ["A_EE", "A_II", "A_IE", "A_EI"]
ALL_CONDITIONS = PRIMARY_CONDITIONS + ["A_E0", "A_I0"]
CONDITION_LABELS = {
    "A_EE": "English input / English reasoning",
    "A_II": "Ilokano input / Ilokano reasoning",
    "A_IE": "Ilokano input / English pivot",
    "A_EI": "English input / Ilokano pivot",
    "A_E0": "English direct-answer control",
    "A_I0": "Ilokano direct-answer control",
}
COMPARISONS = [
    ("A_EE", "A_II", "overall_language_gap"),
    ("A_IE", "A_II", "pivot_recovery"),
    ("A_IE", "A_EE", "pivot_penalty"),
    ("A_EE", "A_EI", "reverse_pivot_degradation"),
]


def load_records(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def input_paths(results_dir=None):
    if results_dir is None:
        return DEFAULT_INPUTS
    return {
        model: os.path.join(results_dir, filename)
        for model, filename in REGRADED_FILENAMES.items()
    }


def mean_or_none(values):
    return statistics.mean(values) if values else None


def median_or_none(values):
    return statistics.median(values) if values else None


def proportion_ci(successes, total, z=1.959963984540054):
    """Wilson 95% confidence interval for one proportion."""
    if not total:
        return {"low": None, "high": None}
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return {"low": max(0.0, centre - half), "high": min(1.0, centre + half)}


def exact_mcnemar_p(first_only, second_only):
    n = first_only + second_only
    if n == 0:
        return 1.0
    k = min(first_only, second_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def chi_square_mcnemar(first_only, second_only):
    n = first_only + second_only
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    statistic = (abs(first_only - second_only) - 1) ** 2 / n
    return {"statistic": statistic, "p_value": math.erfc(math.sqrt(statistic / 2))}


def bootstrap_difference(first, second, iterations=10000, seed=42):
    """Paired bootstrap CI for accuracy(first)-accuracy(second)."""
    n = len(first)
    if not n:
        return {"low": None, "high": None}
    rng = random.Random(seed)
    differences = []
    for _ in range(iterations):
        total = 0
        for _ in range(n):
            index = rng.randrange(n)
            total += first[index] - second[index]
        differences.append(total / n)
    differences.sort()
    return {
        "low": differences[int(0.025 * (iterations - 1))],
        "high": differences[int(0.975 * (iterations - 1))],
    }


def holm_adjust(p_values):
    """Holm step-down correction, preserving the input keys."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        corrected = min(1.0, (count - rank) * value)
        running = max(running, corrected)
        adjusted[key] = running
    return adjusted


def paired_records(index, model, first_condition, second_condition):
    first = index.get((model, first_condition, "reason"))
    second = index.get((model, second_condition, "reason"))
    ids = sorted(set(first or {}) & set(second or {}))
    pairs = []
    for item_id in ids:
        a = first[item_id].get("is_correct")
        b = second[item_id].get("is_correct")
        if isinstance(a, bool) and isinstance(b, bool):
            pairs.append((item_id, a, b))
    return pairs


def paired_test(index, model, first_condition, second_condition, seed):
    pairs = paired_records(index, model, first_condition, second_condition)
    first = [int(a) for _, a, _ in pairs]
    second = [int(b) for _, _, b in pairs]
    first_only = sum(a and not b for a, b in zip(first, second))
    second_only = sum(b and not a for a, b in zip(first, second))
    first_correct = sum(first)
    second_correct = sum(second)
    total = len(pairs)
    delta = (first_correct - second_correct) / total if total else None
    chi = chi_square_mcnemar(first_only, second_only)
    return {
        "n_paired": total,
        "first_condition": first_condition,
        "second_condition": second_condition,
        "first_accuracy": first_correct / total if total else None,
        "second_accuracy": second_correct / total if total else None,
        "delta_first_minus_second": delta,
        "delta_ci_bootstrap_95": bootstrap_difference(first, second, seed=seed),
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "both_correct": sum(a and b for a, b in zip(first, second)),
        "both_incorrect": sum(not a and not b for a, b in zip(first, second)),
        "mcnemar_exact_p": exact_mcnemar_p(first_only, second_only),
        "mcnemar_chi2_cc": chi["statistic"],
        "mcnemar_chi2_p": chi["p_value"],
        "item_ids": [item_id for item_id, _, _ in pairs],
    }


def accuracy(rows):
    usable = [r for r in rows if isinstance(r.get("is_correct"), bool)]
    correct = sum(r["is_correct"] for r in usable)
    result = {"correct": correct, "total": len(usable), "accuracy": correct / len(usable) if usable else None}
    result["ci_wilson_95"] = proportion_ci(correct, len(usable))
    return result


MODEL_CAUSED_TRANSLATION_FAILURES = {"translation_format_failure", "truncation", "refusal"}


def condition_outcomes(rows, condition):
    """Resolve one end-to-end outcome per item.

    A pivot translation that was truncated/refused/malformed is an actual model
    failure and counts as False for end-to-end accuracy. Reasoning-only paired tests can still exclude it.
    """
    condition_rows = [r for r in rows if r.get("condition_key") == condition]
    if condition not in ("A_IE", "A_EI"):
        return {
            r["item_id"]: r.get("is_correct")
            for r in condition_rows
            if r.get("stage") in ("reason", "direct") and isinstance(r.get("is_correct"), bool)
        }
    reason = {r["item_id"]: r for r in condition_rows if r.get("stage") == "reason"}
    outcomes = {}
    for r in condition_rows:
        if r.get("stage") != "translate":
            continue
        item_id = r["item_id"]
        if item_id in reason and isinstance(reason[item_id].get("is_correct"), bool):
            outcomes[item_id] = reason[item_id]["is_correct"]
        elif r.get("failure_type") in MODEL_CAUSED_TRANSLATION_FAILURES:
            outcomes[item_id] = False
    return outcomes


def resolved_accuracy(rows, condition):
    outcomes = condition_outcomes(rows, condition)
    correct = sum(value is True for value in outcomes.values())
    result = {"correct": correct, "total": len(outcomes), "accuracy": correct / len(outcomes) if outcomes else None}
    result["ci_wilson_95"] = proportion_ci(correct, len(outcomes))
    return result


def failure_rates(rows):
    reasoning = [r for r in rows if r.get("stage") in ("reason", "direct")]
    counts = Counter(r.get("failure_type") or "unknown" for r in reasoning)
    total = len(reasoning)
    trunc = sum(bool(r.get("is_truncated")) for r in reasoning)
    degeneration = sum(bool(r.get("degeneration_candidate")) for r in reasoning)
    no_answer = sum(r.get("failure_type") == "missing_answer" for r in reasoning)
    format_fail = sum(r.get("format_compliant") is False for r in reasoning)
    def rate(value):
        return value / total if total else None
    return {
        "total_reasoning_or_direct_records": total,
        "failure_type_counts": dict(sorted(counts.items())),
        "failure_type_rates": {key: rate(value) for key, value in sorted(counts.items())},
        "truncated_count": trunc,
        "truncated_rate": rate(trunc),
        "degeneration_candidate_count": degeneration,
        "degeneration_candidate_rate": rate(degeneration),
        "missing_answer_count": no_answer,
        "missing_answer_rate": rate(no_answer),
        "format_noncompliant_count": format_fail,
        "format_noncompliant_rate": rate(format_fail),
    }


def token_stats(rows):
    fields = [
        "input_tokens", "output_tokens", "question_en_tokens", "question_ilo_tokens",
        "translation_tokens", "rationale_tokens", "tokenization_tax_ratio", "word_count",
        "char_count", "latency_ms", "retry_count", "repetition_ratio",
    ]
    output = {}
    for field in fields:
        values = [r[field] for r in rows if isinstance(r.get(field), (int, float)) and not isinstance(r.get(field), bool)]
        output[field] = {"n": len(values), "mean": mean_or_none(values), "median": median_or_none(values)}
    return output


def translation_stats(rows):
    translations = [r for r in rows if r.get("stage") == "translate"]
    counts = Counter(r.get("failure_type") or "unknown" for r in translations)
    completed = sum(r.get("failure_type") == "translation_completed" for r in translations)
    return {
        "total_translation_records": len(translations),
        "completed": completed,
        "completion_rate": completed / len(translations) if translations else None,
        "status_counts": dict(sorted(counts.items())),
        "token_stats": token_stats(translations),
        "faithfulness_annotation_status": "not performed; requires human review",
    }


def tiktoken_analysis(records_by_model, dataset_path=DEFAULT_DATASET):
    """Recompute comparable token statistics for every model with the repo tokenizer module.

    The evaluation's native counts remain in the report for model-specific accounting. This
    second pass uses the existing evaluate/tokenization.py implementation and cl100k_base for
    all three models, so cross-model tokenization comparisons do not mix tokenizers.
    """
    try:
        import sys
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from evaluate import tokenization
    except (ImportError, ModuleNotFoundError) as exc:
        return {"available": False, "error": str(exc)}

    dataset = {}
    with open(dataset_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                dataset[row["id"]] = row

    def clean_rationale(text):
        if not text:
            return ""
        return re.sub(r"<answer>.*?</answer>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    def summary(values):
        return {"n": len(values), "mean": mean_or_none(values), "median": median_or_none(values)}

    fields = [
        "input_tokens", "output_tokens", "question_en_tokens", "question_ilo_tokens",
        "translation_tokens", "rationale_tokens", "tokenization_tax_ratio", "word_count", "char_count",
    ]
    output = {}
    for model, rows in records_by_model.items():
        by_condition = defaultdict(list)
        for row in rows:
            item = dataset[row["item_id"]]
            question_en = item["question_en"]
            question_ilo = item["question_ilo"]
            translation = row.get("generated_translation") or ""
            raw_response = row.get("raw_response") or ""
            # Translation-stage raw_response is the translation itself, not a reasoning
            # rationale. Count rationale tokens only for reasoning/direct records; using
            # raw_response as a fallback here would double-count translation text as rationale.
            rationale_source = row.get("generated_rationale") if row.get("stage") in ("reason", "direct") else ""
            rationale = clean_rationale(rationale_source)
            q_en = tokenization.count_tokens_tiktoken(question_en)
            q_ilo = tokenization.count_tokens_tiktoken(question_ilo)
            record = {
                "item_id": row["item_id"],
                "input_tokens": tokenization.count_tokens_tiktoken(row.get("original_input") or ""),
                "output_tokens": tokenization.count_tokens_tiktoken(raw_response),
                "question_en_tokens": q_en,
                "question_ilo_tokens": q_ilo,
                "translation_tokens": tokenization.count_tokens_tiktoken(translation) if translation else None,
                "rationale_tokens": tokenization.count_tokens_tiktoken(rationale) if rationale else None,
                "tokenization_tax_ratio": q_ilo / q_en if q_en else None,
                "word_count": len(raw_response.split()),
                "char_count": len(raw_response),
            }
            by_condition[row["condition_key"]].append(record)
        output[model] = {}
        for condition in ALL_CONDITIONS:
            condition_rows = by_condition.get(condition, [])
            question_by_item = {}
            for row in condition_rows:
                question_by_item.setdefault(row["item_id"], row)
            stats = {field: summary([r[field] for r in condition_rows if r[field] is not None]) for field in fields}
            for field in ("question_en_tokens", "question_ilo_tokens", "tokenization_tax_ratio"):
                stats[field] = summary([r[field] for r in question_by_item.values() if r[field] is not None])
            output[model][condition] = stats
    return {
        "available": True,
        "encoding": tokenization.TIKTOKEN_ENCODING,
        "tiktoken_version": tokenization.tiktoken.__version__,
        "note": "Recomputed from saved dataset and model outputs for all three current models using one shared tiktoken encoding. Native counts remain for model-specific context accounting.",
        "models": output,
    }


def build_analysis(records_by_model):
    index = {}
    inventory = {}
    for model, rows in records_by_model.items():
        local = defaultdict(dict)
        duplicates = 0
        for row in rows:
            key = (row.get("condition_key"), row.get("stage"), row.get("item_id"))
            if key in local[row.get("condition_key"), row.get("stage")]:
                duplicates += 1
            local[row.get("condition_key"), row.get("stage")][row.get("item_id")] = row
        for (condition, stage), by_item in local.items():
            index[(model, condition, stage)] = by_item
        inventory[model] = {
            "display_name": MODEL_NAMES.get(model, model),
            "records": len(rows),
            "duplicate_records_replaced": duplicates,
            "conditions": {},
        }
        for condition in ALL_CONDITIONS:
            condition_rows = [r for r in rows if r.get("condition_key") == condition and r.get("stage") in ("reason", "direct")]
            translation_rows = [r for r in rows if r.get("condition_key") == condition and r.get("stage") == "translate"]
            inventory[model]["conditions"][condition] = {
                "reason_or_direct_records": len(condition_rows),
                "translation_records": len(translation_rows),
                "missing_reason_or_direct_from_1000": max(0, 1000 - len(condition_rows)) if condition in PRIMARY_CONDITIONS else None,
                "accuracy": resolved_accuracy(rows, condition),
                "failure_rates": failure_rates(condition_rows),
                "token_stats": token_stats(condition_rows),
                "translation_stats": translation_stats(translation_rows),
            }

    results = {"inventory": inventory, "models": {}}
    for model in records_by_model:
        model_rows = records_by_model[model]
        conditions = inventory[model]["conditions"]
        model_result = {
            "display_name": MODEL_NAMES.get(model, model),
            "accuracy": {c: conditions[c]["accuracy"] for c in ALL_CONDITIONS},
            "failure_rates": {c: conditions[c]["failure_rates"] for c in ALL_CONDITIONS},
            "token_stats": {c: conditions[c]["token_stats"] for c in ALL_CONDITIONS},
            "translation_stats": {c: conditions[c]["translation_stats"] for c in ["A_IE", "A_EI"]},
            "domain_accuracy": {},
            "paired_comparisons": {},
            "legacy_trt_deltas": {},
        }
        sources = sorted({r.get("source", "unknown") for r in model_rows})
        for source in sources:
            model_result["domain_accuracy"][source] = {}
            for condition in ALL_CONDITIONS:
                source_ids = {r.get("item_id") for r in model_rows if r.get("source") == source}
                outcomes = {item_id: value for item_id, value in condition_outcomes(model_rows, condition).items() if item_id in source_ids}
                correct = sum(value is True for value in outcomes.values())
                model_result["domain_accuracy"][source][condition] = {
                    "correct": correct,
                    "total": len(outcomes),
                    "accuracy": correct / len(outcomes) if outcomes else None,
                    "ci_wilson_95": proportion_ci(correct, len(outcomes)),
                }
        p = {c: conditions[c]["accuracy"]["accuracy"] for c in PRIMARY_CONDITIONS}
        if p["A_IE"] is not None and p["A_II"] is not None:
            model_result["legacy_trt_deltas"]["total_language_gap_AEE_minus_AII"] = p["A_EE"] - p["A_II"]
            model_result["legacy_trt_deltas"]["reasoning_penalty_AIE_minus_AII"] = p["A_IE"] - p["A_II"]
            model_result["legacy_trt_deltas"]["comprehension_penalty_AEE_minus_AIE"] = p["A_EE"] - p["A_IE"]
            model_result["legacy_trt_deltas"]["relative_reasoning_degradation"] = ((p["A_IE"] - p["A_II"]) / p["A_IE"]) if p["A_IE"] else None
        pvals = {}
        for first, second, label in COMPARISONS:
            comparison = paired_test(index, model, first, second, seed=42 + len(model_result["paired_comparisons"]))
            model_result["paired_comparisons"][label] = comparison
            pvals[label] = comparison["mcnemar_exact_p"]
        adjusted = holm_adjust(pvals)
        for label, value in adjusted.items():
            model_result["paired_comparisons"][label]["mcnemar_exact_p_holm"] = value
        results["models"][model] = model_result
    return results


def pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def signed_pp(value):
    return "n/a" if value is None else f"{100 * value:+.1f} pp"


def render_markdown(results):
    lines = [
        "# Numerical Analysis — 3-Pass Cross-Lingual Reasoning (Ilokano vs. English)",
        "",
        "Historical three-pass analysis retained for reference. Claude Sonnet 4.6 and Llama 3 8B are no longer part of the current model comparison; the active analysis uses GPT-5.2, Qwen3.6-27B, and Qwen-SEA-LION-v4.5-27B-IT.",
        "",
        "**Historical passes.** P1 = English question → English reasoning (baseline). P2 = Ilokano question → reasoning entirely in Ilokano (native). P3 = Ilokano question → translate to English, then reason in English (pivot). A pass without a usable answer counted as incorrect.",
        "",
        "| Historical model | P1 English | P2 Native Ilokano | P3 English Pivot | Δ_total | Δ_comp | Δ_reason | D_rel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| Claude Sonnet 4.6 | 96.3% | 88.4% | 90.5% | +7.9 pp | +5.8 pp | +2.1 pp | 2.3% |",
        "| Llama 3 8B | 62.6% | 19.2% | 26.3% | +43.4 pp | +36.3 pp | +7.1 pp | 27.0% |",
        "",
        "Historical passes with no extractable answer: Claude P1/P2/P3 = 0/6/2; Llama P1/P2/P3 = 25/146/23.",
        "",
        "Historical per-source accuracy:",
        "",
        "| Model | Source | P1 | P2 | P3 | Δ_reason (P3−P2) |",
        "|---|---|---:|---:|---:|---:|",
        "| Claude | BBH causal judgement (n=50) | 64.0% | 58.0% | 50.0% | -8.0 pp |",
        "| Claude | BBH logical deduction (n=250) | 100.0% | 90.4% | 92.4% | +2.0 pp |",
        "| Claude | GSM8K (n=400) | 97.8% | 91.2% | 94.0% | +2.8 pp |",
        "| Claude | MMLU conceptual physics (n=174) | 97.1% | 88.5% | 91.4% | +2.9 pp |",
        "| Claude | MMLU formal logic (n=126) | 96.0% | 87.3% | 90.5% | +3.2 pp |",
        "| Llama | BBH causal judgement (n=50) | 48.0% | 54.0% | 38.0% | -16.0 pp |",
        "| Llama | BBH logical deduction (n=250) | 52.8% | 29.2% | 31.6% | +2.4 pp |",
        "| Llama | GSM8K (n=400) | 80.8% | 8.2% | 20.0% | +11.8 pp |",
        "| Llama | MMLU conceptual physics (n=174) | 56.9% | 20.1% | 31.0% | +10.9 pp |",
        "| Llama | MMLU formal logic (n=126) | 38.1% | 19.0% | 24.6% | +5.6 pp |",
        "",
        "The historical P2-vs-P3 McNemar exact two-sided p-values were 3.142e-02 for Claude and 1.212e-05 for Llama. Those models are retained here only as historical context, not pooled with the active results.",
        "",
        "---",
        "",
        "# TRT Metrics — GPT-5.2, Qwen3.6-27B, and Qwen-SEA-LION",
        "",
        "This report analyzes the six-condition evaluator outputs. Human-reviewed translation faithfulness and error taxonomy are excluded.",
        "Model-caused pivot translation failures count as end-to-end failures. Missing reasoning records caused by those failures are excluded from reasoning-only paired comparisons.",
        "",
        "## Current evaluation conditions",
        "",
        "The original three-pass design remains identifiable in the current protocol:",
        "",
        "- `A_EE` / P1: English input → English reasoning baseline.",
        "- `A_II` / P2: Ilokano input → Ilokano reasoning.",
        "- `A_IE` / P3: Ilokano input → separate English translation → fresh-context English reasoning.",
        "- `A_EI`: English input → separate Ilokano translation → fresh-context Ilokano reasoning (reverse pivot).",
        "- `A_E0` and `A_I0`: direct-answer controls on the fixed 300-item subset.",
        "",
        "For `A_IE` and `A_EI`, translation and reasoning are separate result records. Translation faithfulness is not automatically inferred; only translation-stage completion and model-caused translation failures are reported here.",
        "",
        "## Result inventory",
        "",
        "| Model | Records | Condition completeness |",
        "|---|---:|---|",
    ]
    for model, inv in results["inventory"].items():
        completeness = ", ".join(f"{c}={v['reason_or_direct_records']}" for c, v in inv["conditions"].items() if c in PRIMARY_CONDITIONS)
        lines.append(f"| {inv['display_name']} | {inv['records']} | {completeness} reasoning/direct records |")

    lines += ["", "## Accuracy by model and condition", "", "| Model | A_EE | A_II | A_IE | A_EI | A_E0 | A_I0 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model, result in results["models"].items():
        cells = [pct(result["accuracy"][c]["accuracy"]) for c in ALL_CONDITIONS]
        lines.append(f"| {result['display_name']} | " + " | ".join(cells) + " |")

    lines += ["", "## Paired gaps and McNemar tests", "", "| Model | Comparison | n | Accuracy difference | Bootstrap 95% CI | Exact p | Holm p |", "|---|---|---:|---:|---:|---:|---:|"]
    for model, result in results["models"].items():
        for _, _, label in COMPARISONS:
            c = result["paired_comparisons"][label]
            ci = c["delta_ci_bootstrap_95"]
            ci_text = "n/a" if ci["low"] is None else f"[{signed_pp(ci['low'])}, {signed_pp(ci['high'])}]"
            lines.append(f"| {result['display_name']} | {label} | {c['n_paired']} | {signed_pp(c['delta_first_minus_second'])} | {ci_text} | {c['mcnemar_exact_p']:.3e} | {c['mcnemar_exact_p_holm']:.3e} |")

    lines += ["", "## Legacy TRT deltas", "", "| Model | Total language gap AEE-AII | Reasoning penalty AIE-AII | Comprehension penalty AEE-AIE | Relative reasoning degradation |", "|---|---:|---:|---:|---:|"]
    for model, result in results["models"].items():
        d = result["legacy_trt_deltas"]
        lines.append(f"| {result['display_name']} | {signed_pp(d.get('total_language_gap_AEE_minus_AII'))} | {signed_pp(d.get('reasoning_penalty_AIE_minus_AII'))} | {signed_pp(d.get('comprehension_penalty_AEE_minus_AIE'))} | {pct(d.get('relative_reasoning_degradation'))} |")

    lines += ["", "## Item-level pivot recovery and regression", "", "The `pivot_recovery` comparison is A_IE versus A_II. `first_only_correct` means pivot correct/native incorrect; `second_only_correct` means native correct/pivot incorrect.", "", "| Model | Paired items | Recovered | Regressed | Both correct | Both incorrect |", "|---|---:|---:|---:|---:|---:|"]
    for model, result in results["models"].items():
        c = result["paired_comparisons"]["pivot_recovery"]
        lines.append(f"| {result['display_name']} | {c['n_paired']} | {c['first_only_correct']} | {c['second_only_correct']} | {c['both_correct']} | {c['both_incorrect']} |")

    lines += ["", "## Domain-level accuracy", ""]
    for model, result in results["models"].items():
        lines += [f"### {result['display_name']}", "", "| Source | A_EE | A_II | A_IE | A_EI |", "|---|---:|---:|---:|---:|"]
        for source, conditions in result["domain_accuracy"].items():
            lines.append(f"| {source} | " + " | ".join(pct(conditions[c]["accuracy"]) for c in PRIMARY_CONDITIONS) + " |")
        lines.append("")

    lines += ["## Failure rates", "", "Failure rates are calculated over available reasoning/direct records for each condition.", "", "| Model | Condition | n | Wrong/substantive | Missing answer | Truncated | Degeneration candidate | Format noncompliant |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model, result in results["models"].items():
        for condition in PRIMARY_CONDITIONS:
            f = result["failure_rates"][condition]
            wrong = f["failure_type_rates"].get("substantively_incorrect", 0)
            lines.append(f"| {result['display_name']} | {condition} | {f['total_reasoning_or_direct_records']} | {pct(wrong)} | {pct(f['missing_answer_rate'])} | {pct(f['truncated_rate'])} | {pct(f['degeneration_candidate_rate'])} | {pct(f['format_noncompliant_rate'])} |")

    lines += ["", "## Pivot translation-stage completion and missing reasoning records", "", "A translation-stage truncation/refusal/format failure is counted as an end-to-end failure for the pivot condition. Because no reasoning call can validly consume an unusable translation, the corresponding reasoning record is absent and is excluded from reasoning-only paired tests. In the current files, every missing `A_EI` reasoning record is explained by a translation-stage truncation (`failure_type=truncation`, `finish_reason=length`); no missing record is attributed to an API or parser failure.", "", "| Model | A_IE completed/total | A_EI completed/total | Missing A_EI reasoning | Cause |", "|---|---:|---:|---:|---|"]
    for model, result in results["models"].items():
        values = []
        for c in ("A_IE", "A_EI"):
            t = result["translation_stats"][c]
            values.append(f"{t['completed']}/{t['total_translation_records']} ({pct(t['completion_rate'])})")
        missing_reason = results["inventory"][model]["conditions"]["A_EI"]["translation_records"] - results["inventory"][model]["conditions"]["A_EI"]["reason_or_direct_records"]
        cause = "none" if missing_reason == 0 else "translation truncation"
        lines.append(f"| {result['display_name']} | {values[0]} | {values[1]} | {missing_reason} | {cause} |")

    lines += ["", "## Tokenization and output statistics", "", "The JSON report contains mean, median, and sample count for input/output tokens, question token counts, translation/rationale tokens, tokenization-tax ratio, latency, retries, repetition ratio, word count, and character count for every model and condition.", "", "| Model | Condition | Question EN tokens (mean) | Question ILO tokens (mean) | Tax ratio (mean) | Rationale tokens (mean) | Output tokens (mean) |", "|---|---|---:|---:|---:|---:|---:|"]
    for model, result in results["models"].items():
        for c in PRIMARY_CONDITIONS:
            t = result["token_stats"][c]
            vals = [t[field]["mean"] for field in ("question_en_tokens", "question_ilo_tokens", "tokenization_tax_ratio", "rationale_tokens", "output_tokens")]
            rendered = ["n/a" if v is None else f"{v:.2f}" for v in vals]
            lines.append(f"| {result['display_name']} | {c} | " + " | ".join(rendered) + " |")

    tiktoken = results.get("tiktoken_analysis", {})
    if tiktoken.get("available"):
        lines += [
            "", "## Tiktoken-based tokenization rerun", "",
            f"The original evaluation used native tokenizers for the Qwen runs and `{tiktoken['encoding']}` for GPT. For comparable cross-model statistics, the saved dataset and outputs were recomputed with `{tiktoken['encoding']}` (tiktoken {tiktoken['tiktoken_version']}) through `scripts/evaluate/tokenization.py`. These replace mixed-tokenizer values for comparative tokenization claims; native counts remain in the JSON records for model-specific context accounting.",
            "",
            "| Model | Condition | English question tokens (mean) | Ilokano question tokens (mean) | Tokenization ratio (mean) | Translation tokens (mean) | Rationale tokens (mean) | Output tokens (mean) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for model, conditions in tiktoken["models"].items():
            for condition in PRIMARY_CONDITIONS:
                cell = conditions[condition]
                def fmt(field):
                    value = cell[field]["mean"]
                    return "n/a" if value is None else f"{value:.2f}"
                lines.append("| " + " | ".join([
                    results["models"][model]["display_name"], condition,
                    fmt("question_en_tokens"), fmt("question_ilo_tokens"),
                    fmt("tokenization_tax_ratio"), fmt("translation_tokens"),
                    fmt("rationale_tokens"), fmt("output_tokens"),
                ]) + " |")

    lines += ["", "## Scope limitation", "", "Translation faithfulness, semantic adequacy, and detailed linguistic error taxonomy are not automatically judged. The separate translation and reasoning outputs are preserved for a later human annotation pass.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Analyze the three re-graded JSONL files in this directory.",
    )
    args = parser.parse_args()

    input_files = input_paths(args.results_dir)
    records_by_model = {}
    for model, path in input_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        records_by_model[model] = load_records(path)

    result = build_analysis(records_by_model)
    result["tiktoken_analysis"] = tiktoken_analysis(records_by_model)
    result["metadata"] = {
        "models": list(records_by_model),
        "input_files": input_files,
        "primary_conditions": PRIMARY_CONDITIONS,
        "all_conditions": ALL_CONDITIONS,
        "comparisons": [{"first": a, "second": b, "label": label} for a, b, label in COMPARISONS],
        "bootstrap_iterations": 10000,
        "bootstrap_seed": 42,
        "human_error_taxonomy": "excluded",
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(args.output_md, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(result))
        handle.write("\n")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    for model, inv in result["inventory"].items():
        print(model, inv["records"], {c: inv["conditions"][c]["reason_or_direct_records"] for c in PRIMARY_CONDITIONS})


if __name__ == "__main__":
    main()
