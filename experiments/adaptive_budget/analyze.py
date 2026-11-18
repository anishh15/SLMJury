#!/usr/bin/env python3
"""Adaptive budget: how cheap can a judge be without losing accuracy?"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from slmjury.analysis import paths
from slmjury.analysis.accuracy import CORRUPT_FILES, Counts, file_counts, verdict_pairs
from slmjury.parsers.normalizer import set_strict_labels

NOTES = """\
Uses the token-budget sweep to compare four ways of choosing a judge's budget:

  always_t10              10 tokens for every item
  always_t8192            8,192 tokens for every item
  domain_heuristic        one budget per answer type (numeric, latex,
                          multiple_choice): the median knee budget of that
                          type, rounded up to a swept budget
  confidence_escalation   10 tokens, and a retry at 256 tokens for each item
                          whose 10-token verdict is Undefined

The knee of a judge x dataset curve is the smallest budget whose accuracy is
within 0.5 points of the best budget for that cell.

Escalation is simulated item by item: the 256-token verdict replaces the
10-token verdict only on the items that were Undefined, and the result is
scored like any other run. Every item pays the 10-token attempt, and each
retried item also pays the 256-token retry.
Cost is relative to always_t8192 (budget / 8,192 per item), averaged over
items. Accuracy is micro-averaged over judges, students and datasets.

Judges that always think (phi4r-14b, phi4rp-14b, phi4mr-3.8b) have no
10-token budget, so no policy that starts at 10 tokens applies to them, and
they are left out.

Reads (under results/, or $SLMJURY_RESULTS), with no new runs needed:
  token_budget/budget_sweep/<judge>/<student>_<dataset>_b<budget>.json
  (written by experiments/token_budget/run_budget_sweep.py)

Writes results/analysis/adaptive_budget/:
  knee_budgets.csv                 judge x dataset: knee budget and accuracy at each budget
  domain_heuristic.json            the per-domain budgets the heuristic policy uses
  policy_comparison.csv            the four policies over all judges
  policy_comparison_by_judge.csv   the four policies per judge
and the plots, to results/figures/analysis/adaptive_budget/ (needs matplotlib):
  policy_comparison.png            accuracy and cost of each policy
  budget_curves.png                accuracy against budget per dataset

Run it from the repository root.
"""

OUT_NAME = "adaptive_budget"

STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]
BUDGETS = [10, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
ALL_DATASETS = [
    "gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
    "hellaswag", "winogrande", "truthfulqa",
]

DOMAIN_MAP = {
    "gsm8k": "numeric", "gsm_plus": "numeric", "math": "latex",
    "arc_challenge": "multiple_choice", "arc_easy": "multiple_choice",
    "hellaswag": "multiple_choice", "winogrande": "multiple_choice",
    "truthfulqa": "multiple_choice",
}

KNEE_THRESHOLD_PP = 0.5
ESCALATION_BASE, ESCALATION_TO = 10, 256

# Judges with no 10-token budget.
ALWAYS_THINKING = {"phi4r-14b", "phi4rp-14b", "phi4mr-3.8b"}


def sweep_dir() -> Path:
    return paths.experiment("token_budget") / "budget_sweep"


def sweep_path(judge, student, dataset, budget) -> Path:
    return sweep_dir() / judge / f"{student}_{dataset}_b{budget}.json"


def discover_judges() -> list[str]:
    return sorted(d.name for d in sweep_dir().iterdir() if d.is_dir())


def build_counts(judges) -> dict:
    """{judge: {dataset: {budget: Counts}}}, summed over students."""
    table = {}
    for judge in judges:
        table[judge] = {}
        for dataset in ALL_DATASETS:
            for budget in BUDGETS:
                per_student = [
                    file_counts(judge, student, dataset, budget,
                                sweep_path(judge, student, dataset, budget))
                    for student in STUDENTS
                ]
                present = [c for c in per_student if c is not None]
                if present:
                    agg = Counts()
                    for c in present:
                        agg = agg + c
                    table[judge].setdefault(dataset, {})[budget] = agg
    return table


def simulate_escalation(judge, dataset) -> tuple[Counts, int] | None:
    """Counts for the escalation policy on one judge x dataset, and how many items were retried."""
    agg = Counts()
    escalated = 0
    for student in STUDENTS:
        base_path = sweep_path(judge, student, dataset, ESCALATION_BASE)
        esc_path = sweep_path(judge, student, dataset, ESCALATION_TO)
        if not (base_path.exists() and esc_path.exists()):
            continue
        try:
            with open(base_path) as f:
                base = json.load(f)
            with open(esc_path) as f:
                esc = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"unreadable result file, {base_path} or {esc_path}: {e}") from e
        if len(base) != len(esc):
            raise ValueError(f"{base_path} has {len(base)} items, "
                             f"{esc_path} has {len(esc)}")

        base_pairs = verdict_pairs(dataset, base)
        esc_pairs = verdict_pairs(dataset, esc)
        for (bv, oracle), (ev, _) in zip(base_pairs, esc_pairs):
            verdict = bv
            if verdict not in ("correct", "incorrect"):
                verdict = ev          # retry this item at the larger budget
                escalated += 1
            agg.total += 1
            if verdict in ("correct", "incorrect"):
                agg.valid += 1
                if verdict == oracle:
                    agg.correct += 1
    if not agg.total:
        return None
    return agg, escalated


def find_knee(acc_by_budget: dict[int, float]) -> tuple[int, float, float]:
    """Smallest budget within KNEE_THRESHOLD_PP of this cell's best accuracy."""
    max_acc = max(acc_by_budget.values())
    threshold = max_acc - KNEE_THRESHOLD_PP
    for budget in BUDGETS:
        if budget in acc_by_budget and acc_by_budget[budget] >= threshold:
            return budget, max_acc, acc_by_budget[budget]
    best = max(acc_by_budget, key=acc_by_budget.get)
    return best, max_acc, acc_by_budget[best]


def plot_policies(results: list[dict], judges: list[str], counts: dict, out_dir: Path):
    """Two diagnostic charts: the policies side by side, and the per-dataset budget curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the PNG charts.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Adaptive budget policy comparison", fontsize=14, fontweight="bold")
    names = [r["policy"].replace("_", "\n") for r in results]
    accs = [r["avg_accuracy"] for r in results]
    costs = [r["relative_cost"] for r in results]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    bars = ax1.bar(names, accs, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy by policy")
    ax1.set_ylim(min(accs) - 3, max(accs) + 2)
    for bar, a in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{a:.2f}%", ha="center", va="bottom", fontweight="bold")

    bars = ax2.bar(names, costs, color=colors, edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("Relative cost (B=8192 = 1.0)")
    ax2.set_yscale("log")
    ax2.set_title("Inference cost by policy (log scale)")
    for bar, c, r in zip(bars, costs, results):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{c:.4f}\n({r['cost_savings_pct']:.1f}% saved)",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "policy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'policy_comparison.png'}")

    datasets = sorted({d for j in judges for d in counts[j]})
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), squeeze=False)
    fig.suptitle("Accuracy vs budget, per dataset", fontsize=14, fontweight="bold")
    cmap = plt.cm.tab20.colors
    judge_colors = {j: cmap[i % len(cmap)] for i, j in enumerate(judges)}
    for ax, dataset in zip(axes.flat, datasets):
        for judge in judges:
            pts = sorted(counts[judge].get(dataset, {}))
            if not pts:
                continue
            ax.plot(pts,
                    [counts[judge][dataset][b].correct / counts[judge][dataset][b].total * 100
                     for b in pts],
                    "o-", label=judge, color=judge_colors[judge],
                    markersize=3, linewidth=1.2)
        ax.set_title(f"{dataset} ({DOMAIN_MAP.get(dataset, '?')})", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(b) for b in BUDGETS], rotation=45, fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Budget (max tokens)")
        ax.set_ylabel("Accuracy (%)")
    axes.flat[0].legend(fontsize=6, loc="lower right", ncol=2)
    plt.tight_layout()
    fig.savefig(out_dir / "budget_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'budget_curves.png'}")


def main():
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table
    out_dir = paths.analysis_dir(OUT_NAME)

    judges = [j for j in discover_judges() if j not in ALWAYS_THINKING]
    if not judges:
        raise SystemExit(f"No budget-sweep data under {sweep_dir()}")
    print(f"Judges ({len(judges)}): {', '.join(judges)}")
    print(f"Excluded (no B=10 configuration): "
          f"{', '.join(sorted(ALWAYS_THINKING & set(discover_judges())))}\n")

    print("Step 1: reading sweep counts...")
    counts = build_counts(judges)

    # ── Knee budgets ───────────────────────────────────────────────────────
    print(f"Step 2: knee budgets (within {KNEE_THRESHOLD_PP}pp of best)...")
    knee_rows = []
    for judge in judges:
        for dataset in sorted(counts[judge]):
            acc_by_b = {b: c.correct / c.total * 100
                        for b, c in counts[judge][dataset].items() if c.total}
            if not acc_by_b:
                continue
            knee_b, max_acc, knee_acc = find_knee(acc_by_b)
            knee_rows.append({
                "judge": judge, "dataset": dataset,
                "domain": DOMAIN_MAP.get(dataset, "unknown"),
                "knee_budget": knee_b,
                "max_accuracy": round(max_acc, 2),
                "knee_accuracy": round(knee_acc, 2),
                "accuracy_gap": round(max_acc - knee_acc, 2),
                **{f"b{b}": round(acc_by_b[b], 2) if b in acc_by_b else None
                   for b in BUDGETS},
            })

    knee_csv = out_dir / "knee_budgets.csv"
    with open(knee_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["judge", "dataset", "domain", "knee_budget",
                           "max_accuracy", "knee_accuracy", "accuracy_gap"]
            + [f"b{b}" for b in BUDGETS], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(knee_rows)
    print(f"  Saved: {knee_csv}")

    # ── Domain heuristic ───────────────────────────────────────────────────
    print("\nStep 3: domain heuristic...")
    domain_knees = defaultdict(list)
    for row in knee_rows:
        domain_knees[row["domain"]].append(row["knee_budget"])
    heuristic = {}
    for domain, knees in sorted(domain_knees.items()):
        median = float(np.median(knees))
        recommended = next((b for b in BUDGETS if b >= median), BUDGETS[-1])
        heuristic[domain] = {
            "recommended_budget": recommended, "median_knee": median,
            "min_knee": min(knees), "max_knee": max(knees),
            "n_observations": len(knees),
        }
        print(f"  {domain:<18} B={recommended:<6} median knee={median:<7.0f} "
              f"range [{min(knees)}, {max(knees)}]  n={len(knees)}")
    with open(out_dir / "domain_heuristic.json", "w") as f:
        json.dump(heuristic, f, indent=2)

    # ── Policies, micro-averaged ───────────────────────────────────────────
    print("\nStep 4: evaluating policies (micro-averaged over all items)...")

    def fixed_policy(judge, budget) -> tuple[Counts, float]:
        agg = Counts()
        for dataset in counts[judge]:
            c = counts[judge][dataset].get(budget)
            if c:
                agg = agg + c
        return agg, budget / 8192

    def heuristic_policy(judge) -> tuple[Counts, float]:
        agg, weighted_cost, n = Counts(), 0.0, 0
        for dataset in counts[judge]:
            b = heuristic[DOMAIN_MAP[dataset]]["recommended_budget"]
            c = counts[judge][dataset].get(b)
            if c:
                agg = agg + c
                weighted_cost += (b / 8192) * c.total
                n += c.total
        return agg, (weighted_cost / n if n else 0.0)

    def escalation_policy(judge) -> tuple[Counts, float]:
        agg, escalated_total = Counts(), 0
        for dataset in counts[judge]:
            res = simulate_escalation(judge, dataset)
            if res is None:
                continue
            c, escalated = res
            agg = agg + c
            escalated_total += escalated
        if not agg.total:
            return agg, 0.0
        frac = escalated_total / agg.total
        # Every item pays the first attempt; a retried item also pays the retry.
        cost = ESCALATION_BASE / 8192 + frac * (ESCALATION_TO / 8192)
        return agg, cost

    policies = {
        "always_t10": ("Always B=10 (cheapest)", "B=10 for all",
                       lambda j: fixed_policy(j, 10)),
        "always_t8192": ("Always B=8192 (most expensive)", "B=8192 for all",
                         lambda j: fixed_policy(j, 8192)),
        "domain_heuristic": (
            "Per-domain budget from the knee analysis",
            ", ".join(f"{d}->B={h['recommended_budget']}"
                      for d, h in sorted(heuristic.items())),
            heuristic_policy),
        "confidence_escalation": (
            f"Start at B={ESCALATION_BASE}, retry the item at B={ESCALATION_TO} "
            f"when the verdict is Undefined",
            f"B={ESCALATION_BASE} -> B={ESCALATION_TO} on Undefined",
            escalation_policy),
    }

    by_judge_rows, totals = [], {}
    escalation_fracs = []
    for name, (desc, rule, fn) in policies.items():
        agg_all, cost_weighted, n_all = Counts(), 0.0, 0
        for judge in judges:
            c, cost = fn(judge)
            if not c.total:
                continue
            by_judge_rows.append({
                "judge": judge, "policy": name,
                "accuracy": round(c.correct / c.total * 100, 2),
                "ifr": round(c.valid / c.total * 100, 2),
                "relative_cost": round(cost, 6), "n_items": c.total,
            })
            agg_all = agg_all + c
            cost_weighted += cost * c.total
            n_all += c.total
            if name == "confidence_escalation":
                escalation_fracs.append(
                    (cost - ESCALATION_BASE / 8192)
                    / (ESCALATION_TO / 8192)
                )
        totals[name] = {
            "policy": name, "description": desc, "budget_rule": rule,
            # Six decimals, so a gap between two policies is rounded once, where it is printed.
            "avg_accuracy": round(agg_all.correct / agg_all.total * 100, 6),
            "ifr": round(agg_all.valid / agg_all.total * 100, 2),
            "relative_cost": cost_weighted / n_all,
            "n_items": agg_all.total,
        }

    base_cost = totals["always_t8192"]["relative_cost"]
    base_acc = totals["always_t8192"]["avg_accuracy"]
    for t in totals.values():
        t["cost_savings_pct"] = (1 - t["relative_cost"] / base_cost) * 100
        t["accuracy_vs_t8192"] = round(t["avg_accuracy"] - base_acc, 6)

    print(f"\n  {'Policy':<24}{'Accuracy':>10}{'IFR':>8}{'Rel cost':>11}"
          f"{'Savings':>10}{'vs t8192':>11}")
    print("  " + "-" * 74)
    for name in policies:
        t = totals[name]
        print(f"  {name:<24}{t['avg_accuracy']:>9.2f}%{t['ifr']:>7.2f}%"
              f"{t['relative_cost']:>11.5f}{t['cost_savings_pct']:>9.2f}%"
              f"{t['accuracy_vs_t8192']:>+10.2f}pp")
    if escalation_fracs:
        print(f"\n  Measured escalation rate: "
              f"{np.mean(escalation_fracs) * 100:.2f}% of items retried "
              f"(range {min(escalation_fracs) * 100:.2f}%–"
              f"{max(escalation_fracs) * 100:.2f}% across judges)")

    policy_csv = out_dir / "policy_comparison.csv"
    with open(policy_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["policy", "description", "budget_rule", "avg_accuracy",
                           "ifr", "relative_cost", "cost_savings_pct",
                           "accuracy_vs_t8192", "n_items"])
        writer.writeheader()
        writer.writerows(totals[n] for n in policies)
    print(f"\nSaved: {policy_csv}")

    judge_csv = out_dir / "policy_comparison_by_judge.csv"
    with open(judge_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_judge_rows[0].keys()))
        writer.writeheader()
        writer.writerows(by_judge_rows)
    print(f"Saved: {judge_csv}")

    plot_policies([totals[n] for n in policies], judges, counts,
                  paths.analysis_figures_dir(OUT_NAME))

    if CORRUPT_FILES:
        print(f"\n{len(CORRUPT_FILES)} unreadable file(s) skipped:")
        for p in CORRUPT_FILES:
            print(f"   {p}")


if __name__ == "__main__":
    main()
