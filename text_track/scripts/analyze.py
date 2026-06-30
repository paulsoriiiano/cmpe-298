"""Numerical analysis of the 3-pass cross-lingual evaluation.

Produces the metrics that establish the research gap for LLM reasoning in Ilokano:

  P1 = pass1_english_baseline accuracy  (English question -> English CoT)
  P2 = pass2_native_ilokano  accuracy   (Ilokano question -> Ilokano CoT)
  P3 = pass3_english_pivot   accuracy   (Ilokano question -> translate-then-English CoT)

Deltas (per model):
  Total Language Gap   d_total  = P1 - P2
  Comprehension Penalty d_comp  = P1 - P3
  Reasoning Penalty     d_reason = P3 - P2   <- the thesis: >0 means reasoning collapses in Ilokano
  Relative Reasoning Degradation D_rel = (P3 - P2) / P3

Significance: McNemar's exact test on the paired P2-vs-P3 outcomes (same model, same
items, two conditions). Exact binomial two-sided p-value on the discordant pairs — no
scipy dependency. Also reports the chi-square statistic with continuity correction.

A null/refused/error pass (extracted_answer is None) counts as is_correct=False: the
model failed to produce a usable answer under that condition. This is the standard and
conservative choice; the markdown report breaks out how many such failures each cell has
so the reader can see what drives the gap.

Writes text_track/data/analysis.md.
"""
import json
import math
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "..", "data", "evaluation_results_3pass.jsonl")
OUT = os.path.join(SCRIPT_DIR, "..", "data", "analysis.md")

MODELS = [("claude_sonnet_4_6", "Claude Sonnet 4.6"), ("llama_3_8b", "Llama 3 8B")]
PASSES = [
    ("pass1_english_baseline", "P1 English baseline"),
    ("pass2_native_ilokano", "P2 Native Ilokano"),
    ("pass3_english_pivot", "P3 English pivot"),
]


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    """Survival function (upper-tail p-value) of chi-square with 1 df.
    For df=1, P(X > x) = erfc(sqrt(x/2))."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def cell(row, model_key, pass_key):
    return row.get("evaluations", {}).get(model_key, {}).get(pass_key, {})


def main():
    rows = load_rows(DATA)
    n = len(rows)
    sources = sorted({r.get("source", "?") for r in rows})

    # --- accuracy per model/pass (overall + per source) ---
    # correct counts, and "no answer" counts (extracted is None)
    acc = {m: {p: {"correct": 0, "total": 0, "noans": 0} for p, _ in PASSES} for m, _ in MODELS}
    acc_src = {m: {s: {p: {"correct": 0, "total": 0} for p, _ in PASSES} for s in sources}
               for m, _ in MODELS}

    for r in rows:
        s = r.get("source", "?")
        for mk, _ in MODELS:
            for pk, _ in PASSES:
                c = cell(r, mk, pk)
                acc[mk][pk]["total"] += 1
                acc_src[mk][s][pk]["total"] += 1
                if c.get("is_correct"):
                    acc[mk][pk]["correct"] += 1
                    acc_src[mk][s][pk]["correct"] += 1
                if c.get("extracted_answer") is None:
                    acc[mk][pk]["noans"] += 1

    def pct(corr, tot):
        return (corr / tot * 100) if tot else 0.0

    # --- McNemar P2 vs P3 per model ---
    mcnemar = {}
    for mk, _ in MODELS:
        both = p2only = p3only = neither = 0
        for r in rows:
            c2 = bool(cell(r, mk, "pass2_native_ilokano").get("is_correct"))
            c3 = bool(cell(r, mk, "pass3_english_pivot").get("is_correct"))
            if c2 and c3:
                both += 1
            elif c2 and not c3:
                p2only += 1   # passed native, failed pivot
            elif c3 and not c2:
                p3only += 1   # failed native, passed pivot
            else:
                neither += 1
        # discordant: b = passed P3 / failed P2, c = passed P2 / failed P3
        b, c = p3only, p2only
        mcnemar[mk] = {
            "both": both, "p2only": p2only, "p3only": p3only, "neither": neither,
            "b_p3not2": b, "c_p2not3": c,
            "p_exact": mcnemar_exact(b, c),
            "chi2_cc": mcnemar_chi2_cc(b, c),
            "p_chi2": chi2_sf_df1(mcnemar_chi2_cc(b, c)),
        }

    # ---------------- build markdown ----------------
    L = []
    W = L.append
    W("# Numerical Analysis — 3-Pass Cross-Lingual Reasoning (Ilokano vs. English)\n")
    W(f"Dataset: `{os.path.relpath(DATA, SCRIPT_DIR)}` — **{n} items**, "
      f"sources: {', '.join(sources)}.\n")
    W("**Passes.** P1 = English question → English chain-of-thought (baseline). "
      "P2 = Ilokano question → reasoning entirely in Ilokano (native). "
      "P3 = Ilokano question → translate to English, then reason in English (pivot).\n")
    W("**Scoring.** A pass is correct only if it emitted a parseable `<answer>` matching "
      "the gold (English or Ilokano reference). A pass with no extractable answer "
      "(refusal, degeneration, or no answer tag) counts as incorrect — the model failed "
      "to produce a usable answer under that condition.\n")

    # 1. Headline accuracy
    W("## 1. Accuracy by model and pass\n")
    W("| Model | P1 English | P2 Native Ilokano | P3 English Pivot |")
    W("|---|---|---|---|")
    P = {}
    for mk, mname in MODELS:
        a = acc[mk]
        p1 = pct(a["pass1_english_baseline"]["correct"], a["pass1_english_baseline"]["total"])
        p2 = pct(a["pass2_native_ilokano"]["correct"], a["pass2_native_ilokano"]["total"])
        p3 = pct(a["pass3_english_pivot"]["correct"], a["pass3_english_pivot"]["total"])
        P[mk] = (p1, p2, p3)
        W(f"| {mname} "
          f"| {a['pass1_english_baseline']['correct']}/{a['pass1_english_baseline']['total']} ({p1:.1f}%) "
          f"| {a['pass2_native_ilokano']['correct']}/{a['pass2_native_ilokano']['total']} ({p2:.1f}%) "
          f"| {a['pass3_english_pivot']['correct']}/{a['pass3_english_pivot']['total']} ({p3:.1f}%) |")
    W("")
    W("Passes with no extractable answer (counted incorrect above):\n")
    W("| Model | P1 | P2 | P3 |")
    W("|---|---|---|---|")
    for mk, mname in MODELS:
        a = acc[mk]
        W(f"| {mname} | {a['pass1_english_baseline']['noans']} "
          f"| {a['pass2_native_ilokano']['noans']} | {a['pass3_english_pivot']['noans']} |")
    W("")

    # 2. Deltas
    W("## 2. Performance deltas (the gaps)\n")
    W("- **Total Language Gap** Δ_total = P1 − P2 (raw English-vs-Ilokano gap)\n"
      "- **Comprehension Penalty** Δ_comp = P1 − P3 (translation/understanding loss; "
      "both reason in English)\n"
      "- **Reasoning Penalty** Δ_reason = P3 − P2 (the thesis — collapse when forced to "
      "reason in Ilokano tokens)\n"
      "- **Relative Reasoning Degradation** D_rel = (P3 − P2) / P3 × 100% (of the items "
      "the model understood, the share it failed purely from reasoning in Ilokano)\n")
    W("| Model | P1 | P2 | P3 | Δ_total | Δ_comp | Δ_reason | D_rel |")
    W("|---|---|---|---|---|---|---|---|")
    for mk, mname in MODELS:
        p1, p2, p3 = P[mk]
        d_total, d_comp, d_reason = p1 - p2, p1 - p3, p3 - p2
        d_rel = ((p3 - p2) / p3 * 100) if p3 else float("nan")
        W(f"| {mname} | {p1:.1f}% | {p2:.1f}% | {p3:.1f}% "
          f"| {d_total:+.1f} | {d_comp:+.1f} | {d_reason:+.1f} | {d_rel:.1f}% |")
    W("")

    # 3. McNemar
    W("## 3. Statistical significance — McNemar's test (P2 vs. P3)\n")
    W("Same model, same items, two paired conditions (native vs. pivot). Discordant pairs "
      "drive the test: **b** = items the model got right under the pivot but wrong "
      "natively; **c** = right natively but wrong under the pivot. Two-sided exact "
      "binomial p-value; chi-square reported with Yates continuity correction (df=1).\n")
    for mk, mname in MODELS:
        m = mcnemar[mk]
        W(f"### {mname}\n")
        W("| | Passed P3 (pivot) | Failed P3 (pivot) |")
        W("|---|---|---|")
        W(f"| **Passed P2 (native)** | {m['both']} (both correct) | {m['c_p2not3']} (native only) |")
        W(f"| **Failed P2 (native)** | {m['b_p3not2']} (pivot only) | {m['neither']} (both wrong) |")
        W("")
        sig = "**significant** (p < 0.05)" if m["p_exact"] < 0.05 else "not significant (p ≥ 0.05)"
        W(f"- Discordant pairs: b (pivot-only) = {m['b_p3not2']}, c (native-only) = {m['c_p2not3']}")
        W(f"- McNemar χ² (continuity-corrected) = {m['chi2_cc']:.2f}, "
          f"p = {m['p_chi2']:.3e}")
        W(f"- Exact binomial two-sided p = {m['p_exact']:.3e} — {sig}")
        W("")

    # 4. Per-source accuracy
    W("## 4. Accuracy by source benchmark\n")
    for mk, mname in MODELS:
        W(f"### {mname}\n")
        W("| Source | P1 English | P2 Native | P3 Pivot | Δ_reason (P3−P2) |")
        W("|---|---|---|---|---|")
        for s in sources:
            a = acc_src[mk][s]
            p1 = pct(a["pass1_english_baseline"]["correct"], a["pass1_english_baseline"]["total"])
            p2 = pct(a["pass2_native_ilokano"]["correct"], a["pass2_native_ilokano"]["total"])
            p3 = pct(a["pass3_english_pivot"]["correct"], a["pass3_english_pivot"]["total"])
            tot = a["pass1_english_baseline"]["total"]
            W(f"| {s} (n={tot}) | {p1:.1f}% | {p2:.1f}% | {p3:.1f}% | {p3 - p2:+.1f} |")
        W("")

    # console summary
    print(f"Loaded {n} rows. Sources: {sources}")
    for mk, mname in MODELS:
        p1, p2, p3 = P[mk]
        m = mcnemar[mk]
        print(f"\n{mname}: P1={p1:.1f}% P2={p2:.1f}% P3={p3:.1f}%  "
              f"d_total={p1-p2:+.1f} d_comp={p1-p3:+.1f} d_reason={p3-p2:+.1f}")
        print(f"  McNemar P2vP3: b(pivot-only)={m['b_p3not2']} c(native-only)={m['c_p2not3']} "
              f"exact p={m['p_exact']:.3e}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
