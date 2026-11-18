#!/usr/bin/env python3
"""Main results: individual judges, personas, juries, debate and open-ended scoring."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from slmjury.analysis import paths

NOTES = """\
Reads the evaluation files written by scripts/run_evaluation.py and the
scoring evaluations written by scripts/run_scoring_judge.py, and
micro-averages their integer counts (correct / total summed over files), as
generate_judge_summary does.

Reads (under results/core/, or $SLMJURY_RESULTS/core/):
  evaluations/<judge>/<student>_<dataset>_t<budget>.json
  personas/evaluations/<judge>/<persona>_<student>_<dataset>_t<budget>.json
  juries/evaluations/<panel>/<student>_<dataset>.json
  debate/evaluations/<configuration>/<student>_<dataset>.json
  scoring_evaluations/<judge>/{summeval,mtbench_<reference>_<student>}_eval.json

Writes results/analysis/main_results/:
  individual.csv     accuracy and IFR per judge and budget
  per_dataset.csv    accuracy per judge, budget and dataset
  personas.csv       accuracy under each persona, with the no-persona base
  juries.csv         every three-judge jury (majority vote)
  debate.csv         every debate configuration; variant is cross_architecture
                     (three different models) or same_model (one model at
                     three temperatures)
  open_ended.csv     SummEval and MT-Bench correlations per judge

Run it from the repository root.
"""

OUT_NAME = "main_results"

DATASETS = ["gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
            "hellaswag", "winogrande", "truthfulqa"]
STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]
BUDGETS = [10, 8192]
PERSONAS = ["Strict", "Lenient", "Industry", "Logic", "Safety", "Helpfulness"]

# These three cross-architecture debate configurations ran at 4,096 tokens,
# because their transcripts did not fit the context window at 8,192. The
# budget is not stored in the result files, so it is listed here and checked
# against the configurations found.
DEBATE_BUDGET_4096 = {
    "llama3.2-3b+qwen3-4b+phi4mi-3.8b",
    "qwen2.5-1.5b+qwen3-1.7b+phi4mi-3.8b",
    "qwen3-4b+phi4mi-3.8b+qwen2.5-3b",
}


def summary(path: Path) -> dict | None:
    """Total, correct and valid counts from one evaluation file, or None if absent."""
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    c = d["counts"]
    return {"total": d["total_problems"],
            "correct": c["correct_verdicts"],
            "valid": c["valid_verdicts"]}


def micro(parts) -> dict | None:
    """Micro-average: sum the counts over files, then divide."""
    total = correct = valid = files = 0
    for p in parts:
        if p is None:
            continue
        total += p["total"]
        correct += p["correct"]
        valid += p["valid"]
        files += 1
    if not total:
        return None
    return {"accuracy": round(correct / total * 100, 6),
            "ifr": round(valid / total * 100, 6),
            "total": total, "n_files": files}


def write(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    path = paths.analysis_dir(OUT_NAME) / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}  ({len(rows)} rows)")


def judges() -> list[str]:
    d = paths.core("evaluations")
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and p.name != "summaries")


# ── Individual judges ───────────────────────────────────────────────────────
def individual() -> None:
    evaluations = paths.core("evaluations")
    rows, per_ds = [], []
    for judge in judges():
        for budget in BUDGETS:
            cells = {
                (s, d): summary(evaluations / judge / f"{s}_{d}_t{budget}.json")
                for s in STUDENTS for d in DATASETS
            }
            agg = micro(cells.values())
            if agg is None:
                continue          # always-thinking judges have no B=10 run
            rows.append({"judge": judge, "max_tokens": budget,
                         "accuracy": agg["accuracy"], "ifr": agg["ifr"],
                         "n_items": agg["total"], "n_files": agg["n_files"]})
            row = {"judge": judge, "max_tokens": budget}
            for d in DATASETS:
                one = micro(cells[(s, d)] for s in STUDENTS)
                row[d] = one["accuracy"] if one else None
            row["avg"] = agg["accuracy"]
            per_ds.append(row)

    # The delta is only defined for judges that ran at both budgets.
    by_judge: dict[str, dict[int, float]] = {}
    for r in rows:
        by_judge.setdefault(r["judge"], {})[r["max_tokens"]] = r["accuracy"]
    for r in rows:
        both = by_judge[r["judge"]]
        r["delta_t10_minus_t8192"] = (round(both[10] - both[8192], 6)
                                      if len(both) == 2 else None)

    write("individual.csv",
          ["judge", "max_tokens", "accuracy", "ifr",
           "delta_t10_minus_t8192", "n_items", "n_files"], rows)
    write("per_dataset.csv",
          ["judge", "max_tokens"] + DATASETS + ["avg"], per_ds)


# ── Personas ────────────────────────────────────────────────────────────────
def personas() -> None:
    evaluations = paths.core("evaluations")
    base = paths.core("personas") / "evaluations"
    rows = []
    for judge in sorted(p.name for p in base.iterdir() if p.is_dir()):
        for budget in BUDGETS:
            row = {"judge": judge, "max_tokens": budget}
            # The no-persona baseline is the judge's individual run.
            row["base"] = (micro(
                summary(evaluations / judge / f"{s}_{d}_t{budget}.json")
                for s in STUDENTS for d in DATASETS) or {}).get("accuracy")
            for p_ in PERSONAS:
                agg = micro(
                    summary(base / judge / f"{p_}_{s}_{d}_t{budget}.json")
                    for s in STUDENTS for d in DATASETS)
                row[p_] = agg["accuracy"] if agg else None
            vals = [row[p_] for p_ in PERSONAS if row[p_] is not None]
            row["range"] = round(max(vals) - min(vals), 6) if vals else None
            row["worst_persona"] = (min(PERSONAS, key=lambda p_: row[p_])
                                    if len(vals) == len(PERSONAS) else None)
            row["best_persona"] = (max(PERSONAS, key=lambda p_: row[p_])
                                   if len(vals) == len(PERSONAS) else None)
            rows.append(row)
    write("personas.csv",
          ["judge", "max_tokens", "base"] + PERSONAS
          + ["range", "best_persona", "worst_persona"], rows)


# ── Juries and debate ───────────────────────────────────────────────────────
def _panels(root: Path, label: str) -> list[dict]:
    base = root / "evaluations"
    rows = []
    for panel in sorted(p.name for p in base.iterdir() if p.is_dir()):
        agg = micro(summary(base / panel / f"{s}_{d}.json")
                    for s in STUDENTS for d in DATASETS)
        if agg is None:
            continue
        rows.append({label: panel, "accuracy": agg["accuracy"],
                     "ifr": agg["ifr"], "n_items": agg["total"],
                     "n_files": agg["n_files"]})
    return sorted(rows, key=lambda r: -r["accuracy"])


def juries() -> None:
    write("juries.csv",
          ["panel", "accuracy", "ifr", "n_items", "n_files"],
          _panels(paths.core("juries"), "panel"))


def debate() -> None:
    rows = _panels(paths.core("debate"), "configuration")
    for r in rows:
        combo = r["configuration"]
        r["max_tokens"] = 4096 if combo in DEBATE_BUDGET_4096 else 8192
        # Same-model configurations run one model at three temperatures
        # (their names carry "-t0+"); the others run three different models.
        r["variant"] = "same_model" if "-t0+" in combo else "cross_architecture"
    missing = DEBATE_BUDGET_4096 - {r["configuration"] for r in rows}
    if missing:
        sys.exit(f"debate: reduced-budget configurations not found: {missing}")
    write("debate.csv",
          ["configuration", "variant", "max_tokens", "accuracy", "ifr",
           "n_items", "n_files"], rows)


# ── Open-ended scoring ──────────────────────────────────────────────────────
def open_ended() -> None:
    """SummEval and MT-Bench correlations, as the scoring pipeline wrote them.

    MT-Bench is scored against two reference models on two students, so a
    judge's MT-Bench value is the mean over the available pairs.
    """
    base = paths.core("scoring_evaluations")
    METRICS = ["pearson", "spearman", "cohens_kappa", "accuracy", "mse"]
    DIMS = ["coherence", "consistency", "fluency", "relevance"]
    rows = []
    for judge in sorted(p.name for p in base.iterdir() if p.is_dir()):
        row: dict = {"judge": judge}

        se_path = base / judge / "summeval_eval.json"
        if se_path.exists():
            se = json.load(open(se_path))
            per_dim = se.get("per_dimension") or se.get("per_category") or {}
            # Use the evaluator's own `overall` block (mean of the four
            # dimensions) rather than averaging again here.
            for m in METRICS:
                row[f"summeval_{m}"] = se["overall"].get(m)
            row["summeval_n_valid"] = se["overall"].get("n_valid")
            row["summeval_n_total"] = se["overall"].get("n_total")
            for d in DIMS:
                for m in ("spearman", "cohens_kappa"):
                    row[f"summeval_{d}_{m}"] = (per_dim.get(d) or {}).get(m)

        pairs = sorted(base.joinpath(judge).glob("mtbench_*_eval.json"))
        for m in METRICS:
            vals = [json.load(open(p))["overall"][m] for p in pairs]
            vals = [v for v in vals if v is not None]
            row[f"mtbench_{m}"] = round(sum(vals) / len(vals), 6) if vals else None
        row["mtbench_n_pairs"] = len(pairs)
        rows.append(row)

    fields = ["judge"] + [f"summeval_{m}" for m in METRICS] \
        + ["summeval_n_valid", "summeval_n_total"] \
        + [f"summeval_{d}_{m}" for d in DIMS for m in ("spearman", "cohens_kappa")] \
        + [f"mtbench_{m}" for m in METRICS] + ["mtbench_n_pairs"]
    write("open_ended.csv", fields, rows)


def main() -> None:
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    individual()
    personas()
    juries()
    debate()
    open_ended()


if __name__ == "__main__":
    main()
