#!/usr/bin/env python3
"""Token budget: the accuracy curve from 10 to 8,192 tokens, and the single-factor split."""

import argparse
import csv
from pathlib import Path

from slmjury.analysis import paths
from slmjury.analysis.accuracy import CORRUPT_FILES, file_counts, micro
from slmjury.parsers.normalizer import set_strict_labels

NOTES = """\
Two analyses of the closed-ended verdicts:

* Budget sweep. Accuracy per judge, dataset and budget for nine budgets, and
  per judge and budget over all of its datasets.
* Single factor. Accuracy under each condition of run_single_factor.py, and
  its change from the 10-token baseline. Each condition changes one factor
  (budget, prompt, thinking or temperature), so the change shows what that
  factor alone does. A judge is scored on the datasets that every one of its
  conditions has, so a condition with more datasets cannot shift its own delta.

Datasets are found on disk. Accuracy is counted from the per-item judgement
files with slmjury.analysis.accuracy and micro-averaged over students and
datasets. Rates in the CSVs are computed from the integer counts and rounded
once, to six decimals.

Reads (under results/, or $SLMJURY_RESULTS):
  token_budget/budget_sweep/<judge>/<student>_<dataset>_b<budget>.json
  token_budget/single_factor/<judge>/<condition>/<student>_<dataset>.json

Writes results/analysis/token_budget/:
  budget_sweep.csv            judge x dataset accuracy at each budget
  budget_sweep_overall.csv    judge accuracy at each budget, micro-averaged over datasets
  single_factor_table.csv     judge x condition accuracy, IFR and change from baseline_t10
  single_factor_coverage.csv  the conditions and common datasets of each judge

experiments/adaptive_budget/analyze.py reads the same budget sweep.

The two runners in this folder judge the two main students (qwen2.5-32b,
llama3.1-8b) on all eight closed-ended datasets; docs/REPRODUCE.md has the
commands. The results cover fourteen of the sixteen small judges, all but
phi4r-14b and phi4rp-14b. phi4mr-3.8b always thinks, so it has no 10-token
budget and runs three of the six single-factor conditions.

Run it from the repository root.
"""

OUT_NAME = "token_budget"

STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]
ALL_DATASETS = [
    "gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
    "hellaswag", "winogrande", "truthfulqa",
]
BUDGETS = [10, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Single-factor conditions and the budget each one is decoded at. Each condition
# changes one factor relative to baseline_t10.
CONDITIONS = {
    "baseline_t10": 10,
    "baseline_t8192": 8192,
    "factor_budget": 8192,       # quick prompt, large budget
    "factor_prompt": 10,         # reasoned prompt, small budget
    "factor_thinking": 8192,
    "factor_temperature": 10,
}
CONDITION_LABEL = {
    "baseline_t10": "Baseline B=10",
    "baseline_t8192": "Baseline B=8192",
    "factor_budget": "+ Budget only",
    "factor_prompt": "+ Prompt-style only",
    "factor_thinking": "+ Thinking only",
    "factor_temperature": "+ Temperature only",
}


def sweep_dir() -> Path:
    return paths.experiment("token_budget") / "budget_sweep"


def single_factor_dir() -> Path:
    return paths.experiment("token_budget") / "single_factor"


def _subdirs(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(d.name for d in path.iterdir() if d.is_dir())


def sweep_datasets(judge: str) -> list[str]:
    """Datasets this judge has at least one sweep file for, in ALL_DATASETS order."""
    judge_dir = sweep_dir() / judge
    if not judge_dir.exists():
        return []
    found = {
        dataset for dataset in ALL_DATASETS
        for student in STUDENTS for budget in BUDGETS
        if (judge_dir / f"{student}_{dataset}_b{budget}.json").exists()
    }
    return [d for d in ALL_DATASETS if d in found]


def single_factor_datasets(judge: str, condition: str) -> list[str]:
    """Datasets this judge has at least one file for under ``condition``."""
    cond_dir = single_factor_dir() / judge / condition
    if not cond_dir.exists():
        return []
    found = {
        dataset for dataset in ALL_DATASETS for student in STUDENTS
        if (cond_dir / f"{student}_{dataset}.json").exists()
    }
    return [d for d in ALL_DATASETS if d in found]


def build_budget_sweep(judges: list[str]) -> dict:
    """{judge: {"by_dataset": {ds: {b: metrics}}, "overall": {b: metrics}, "datasets": [...]}}."""
    table = {}
    for judge in judges:
        datasets = sweep_datasets(judge)
        by_dataset, overall = {}, {}
        for budget in BUDGETS:
            budget_counts = []
            for dataset in datasets:
                ds_counts = [
                    file_counts(
                        judge, student, dataset, budget,
                        sweep_dir() / judge / f"{student}_{dataset}_b{budget}.json",
                    )
                    for student in STUDENTS
                ]
                m = micro(ds_counts)
                if m:
                    by_dataset.setdefault(dataset, {})[budget] = m
                    budget_counts.extend(c for c in ds_counts if c is not None)
            m_all = micro(budget_counts)
            if m_all:
                overall[budget] = m_all
        table[judge] = {"by_dataset": by_dataset, "overall": overall,
                        "datasets": datasets}
    return table


def pct(m: dict, key: str = "correct") -> float:
    """A rate from its integer counts, to six decimals.

    Starting from micro()'s rounded value instead would round twice when the
    tables and figures round again (-67.5537 -> -67.55 -> -67.5).
    """
    return round(m[key] / m["total"] * 100, 6)


def build_single_factor(judges: list[str]) -> dict:
    """{judge: {"conditions": {condition: metrics}, ...}}, micro-averaged.

    Each judge is scored on the datasets common to all of its conditions.
    """
    table = {}
    for judge in judges:
        per_cond_ds = {c: set(single_factor_datasets(judge, c)) for c in CONDITIONS}
        available = [c for c in CONDITIONS if per_cond_ds[c]]
        if not available:
            continue
        common = set.intersection(*(per_cond_ds[c] for c in available))
        results = {}
        for cond in available:
            max_tokens = CONDITIONS[cond]
            counts = [
                file_counts(
                    judge, student, dataset, max_tokens,
                    single_factor_dir() / judge / cond / f"{student}_{dataset}.json",
                )
                for dataset in sorted(common) for student in STUDENTS
            ]
            results[cond] = micro(counts)
        table[judge] = {
            "conditions": results,
            "common_datasets": sorted(common),
            "conditions_present": available,
        }
    return table


def main():
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table
    out_dir = paths.analysis_dir(OUT_NAME)
    sweep_judges = _subdirs(sweep_dir())
    single_factor_judges = _subdirs(single_factor_dir())

    # ── Budget sweep ───────────────────────────────────────────────────────
    print(f"Budget-sweep judges ({len(sweep_judges)}): {', '.join(sweep_judges)}\n")
    sweep = build_budget_sweep(sweep_judges)

    print("── Overall accuracy by budget (micro-averaged over all available "
          "datasets x 2 students) ──")
    print(f"  {'Judge':<15}{'ds':>4}" + "".join(f"{'b' + str(b):>8}" for b in BUDGETS))
    print("  " + "-" * (19 + 8 * len(BUDGETS)))
    ordered = sorted(sweep, key=lambda j: -(
        sweep[j]["overall"].get(8192) or {"accuracy": 0})["accuracy"])
    for judge in ordered:
        row = f"  {judge:<15}{len(sweep[judge]['datasets']):>4}"
        for b in BUDGETS:
            m = sweep[judge]["overall"].get(b)
            row += f"{m['accuracy']:>8.2f}" if m else f"{'—':>8}"
        print(row)

    sweep_csv = out_dir / "budget_sweep.csv"
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judge", "dataset"] + [f"b{b}" for b in BUDGETS])
        for judge in sweep_judges:
            for dataset in sweep[judge]["datasets"]:
                writer.writerow(
                    [judge, dataset]
                    + [pct(sweep[judge]["by_dataset"][dataset][b])
                       if b in sweep[judge]["by_dataset"].get(dataset, {}) else ""
                       for b in BUDGETS]
                )
    print(f"\nSaved: {sweep_csv}")

    overall_csv = out_dir / "budget_sweep_overall.csv"
    with open(overall_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judge", "n_datasets", "n_items_per_budget"]
                        + [f"b{b}" for b in BUDGETS])
        for judge in sweep_judges:
            ov = sweep[judge]["overall"]
            n_items = ov[max(ov)]["total"] if ov else 0
            writer.writerow(
                [judge, len(sweep[judge]["datasets"]), n_items]
                + [pct(ov[b]) if b in ov else "" for b in BUDGETS]
            )
    print(f"Saved: {overall_csv}")

    # ── Single factor ──────────────────────────────────────────────────────
    print(f"\nSingle-factor judges ({len(single_factor_judges)}): "
          f"{', '.join(single_factor_judges)}\n")
    fac = build_single_factor(single_factor_judges)

    print(f"  {'Judge':<15}{'Condition':<24}{'Acc':>9}{'delta vs B=10':>15}{'items':>10}")
    print("  " + "-" * 73)
    fac_rows, cov_rows = [], []
    for judge in sorted(fac):
        entry = fac[judge]
        base = entry["conditions"].get("baseline_t10")
        for cond in CONDITIONS:
            m = entry["conditions"].get(cond)
            if m is None:
                continue
            delta = (round((m["correct"] / m["total"] - base["correct"] / base["total"]) * 100, 6)
                     if base and cond != "baseline_t10" else None)
            print(f"  {judge:<15}{CONDITION_LABEL[cond]:<24}"
                  f"{m['accuracy']:>9.2f}"
                  f"{(f'{delta:+.2f}' if delta is not None else '—'):>15}"
                  f"{m['total']:>10,}")
            fac_rows.append({
                "judge": judge, "condition": cond, "label": CONDITION_LABEL[cond],
                "accuracy": pct(m),
                "ifr": pct(m, "valid"),
                "delta_vs_t10": delta,
                "n_items": m["total"], "n_files": m["n_files"],
                "n_datasets": len(entry["common_datasets"]),
            })
        cov_rows.append({
            "judge": judge,
            "n_conditions": len(entry["conditions_present"]),
            "conditions": ";".join(entry["conditions_present"]),
            "n_common_datasets": len(entry["common_datasets"]),
            "common_datasets": ";".join(entry["common_datasets"]),
        })
        print()

    fac_csv = out_dir / "single_factor_table.csv"
    with open(fac_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["judge", "condition", "label", "accuracy", "ifr",
                           "delta_vs_t10", "n_items", "n_files", "n_datasets"],
        )
        writer.writeheader()
        writer.writerows(fac_rows)
    print(f"Saved: {fac_csv}")

    cov_csv = out_dir / "single_factor_coverage.csv"
    with open(cov_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cov_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cov_rows)
    print(f"Saved: {cov_csv}")

    if CORRUPT_FILES:
        print(f"\n{len(CORRUPT_FILES)} unreadable result file(s) skipped:")
        for p in CORRUPT_FILES:
            print(f"   {p}")

    # The budget alone: factor_budget against baseline_t10, to four decimals.
    print("\n=== Budget-only effect (factor_budget - baseline_t10) ===")
    for judge in sorted(fac):
        conds = fac[judge]["conditions"]
        if conds.get("baseline_t10") and conds.get("factor_budget"):
            d = conds["factor_budget"]["accuracy"] - conds["baseline_t10"]["accuracy"]
            print(f"  {judge:<15}{d:+.4f} points")


if __name__ == "__main__":
    main()
