#!/usr/bin/env python3
"""Small vs. large judges: accuracy, open-ended correlation and the scaling ladder."""

import argparse
import csv
import json
import math
from pathlib import Path

from slmjury.analysis import paths
from slmjury.analysis.accuracy import CORRUPT_FILES, Counts, file_counts, micro
from slmjury.parsers.normalizer import set_strict_labels

NOTES = """\
Compares the sixteen small judges with six large judges (70B-122B) and the
Qwen3-32B ladder judge on the same students and datasets. Large-judge accuracy
is counted from the per-item judgement files with slmjury.analysis.accuracy
and micro-averaged; small-judge accuracy comes from the per-judge summaries
written by scripts/run_evaluation.py.

Each row records how many datasets it covers. complete_8ds marks the rows that
cover all eight datasets and so can be compared directly.

Reads (under results/, or $SLMJURY_RESULTS):
  large_judges/<judge>/<student>_<dataset>_t<budget>.json
  large_judges/scoring_evaluations/<judge>/*_eval.json
  core/summaries/individual/<judge>_t<budget>.json
  core/scoring_evaluations/<judge>/*_eval.json

Writes results/analysis/large_judges/:
  scaling_ladder.csv        Qwen3 and Qwen2.5 families ordered by size, with the gain per step
  small_vs_large.csv        one row per judge: accuracy per budget, best budget, correlations
  per_dataset.csv           judge x budget x dataset accuracy, two decimals
  per_dataset_precise.csv   the same grid to six decimals, from the integer counts
  domain_gap.csv            math vs. general-knowledge accuracy per judge and budget

The large judges are configured under large_judge_models in
slmjury/configs/models.yaml and run through the same scripts as the small
judges, writing to results/large_judges; docs/REPRODUCE.md has the commands.

Run it from the repository root.
"""

OUT_NAME = "large_judges"

STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]
TOKEN_SETTINGS = [10, 8192]
ALL_DATASETS = [
    "gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
    "hellaswag", "winogrande", "truthfulqa",
]
MATH_DATASETS = ["gsm8k", "gsm_plus", "math"]
GENERAL_DATASETS = ["hellaswag", "winogrande", "truthfulqa"]

# Large judges: display name and parameter count (billions, total).
LARGE_JUDGES = {
    "qwen2.5-72b": {"name": "Qwen2.5-72B", "params": 72.0},
    "llama3.3-70b": {"name": "Llama-3.3-70B", "params": 70.0},
    "gpt-oss-120b": {"name": "GPT-OSS-120B", "params": 120.0},
    "deepseek-r1-distill-llama-70b": {"name": "DeepSeek-R1-Distill-Llama-70B", "params": 70.0},
    "qwen3.5-122b": {"name": "Qwen3.5-122B-A10B", "params": 122.0},
    "glm-4.5-air": {"name": "GLM-4.5-Air", "params": 106.0},
}

# Judges that reason before answering and cannot give a verdict within 10
# tokens, so they are only scored at 8,192.
ALWAYS_THINKING = {"gpt-oss-120b", "deepseek-r1-distill-llama-70b", "glm-4.5-air",
                   "phi4r-14b", "phi4rp-14b", "phi4mr-3.8b"}

# A (judge, budget) cell with a lower instruction-following rate is treated as
# a failed run when choosing a judge's best budget.
MIN_USABLE_IFR = 50.0

# All sixteen small judges, with parameter counts.
SLM_JUDGES = {
    "phi4-14b": 14.0, "phi4r-14b": 14.0, "phi4rp-14b": 14.0,
    "phi4mi-3.8b": 3.8, "phi4mr-3.8b": 3.8,
    "qwen3-14b": 14.0, "qwen3-8b": 8.0, "qwen3-4b": 4.0,
    "qwen3-1.7b": 1.7, "qwen3-0.6b": 0.6,
    "qwen2.5-7b": 7.0, "qwen2.5-3b": 3.0, "qwen2.5-1.5b": 1.5,
    "llama3.1-8b": 8.0, "llama3.2-3b": 3.0, "llama3.2-1b": 1.0,
}

# The mid-size judge that fills in the Qwen3 scaling ladder. It is typed
# "Ladder", not "Large".
LADDER_JUDGES = {
    "qwen3-32b":    {"name": "Qwen3-32B",    "params": 32.0},
}

# The two model families of the ladder, smallest first. Membership is listed
# explicitly because "qwen3.5-122b" also starts with "qwen3".
LADDER_FAMILIES = {
    "Qwen3": ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b",
              "qwen3-32b", "qwen3.5-122b"],
    "Qwen2.5": ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-72b"],
}

# MT-Bench reference models averaged over for the open-ended correlation.
MTBENCH_REFERENCES = ["gpt-oss-120b", "qwen3.5-397b"]


def large_results() -> Path:
    return paths.experiment("large_judges")


def load_large_accuracy(judge: str) -> dict:
    """Per-dataset and overall accuracy for a large or ladder judge, micro-averaged."""
    jdir = large_results() / judge
    out = {}
    for max_tokens in TOKEN_SETTINGS:
        if max_tokens == 10 and judge in ALWAYS_THINKING:
            continue
        per_dataset, all_counts = {}, []
        for dataset in ALL_DATASETS:
            ds_counts = [
                file_counts(
                    judge, student, dataset, max_tokens,
                    jdir / f"{student}_{dataset}_t{max_tokens}.json",
                )
                for student in STUDENTS
            ]
            ds = micro(ds_counts)
            if ds:
                per_dataset[dataset] = ds
                all_counts.extend(c for c in ds_counts if c is not None)
        overall = micro(all_counts)
        if overall:
            out[max_tokens] = {"overall": overall, "by_dataset": per_dataset}
    return out


def load_slm_accuracy(judge: str) -> dict:
    """Overall and per-dataset accuracy for a small judge, from its summary files.

    The summaries store integer counts, so the rates are recomputed from them
    the same way as for the large judges.
    """
    summaries = paths.core("summaries") / "individual"
    out = {}
    for max_tokens in TOKEN_SETTINGS:
        path = summaries / f"{judge}_t{max_tokens}.json"
        if not path.exists():
            continue
        with open(path) as f:
            s = json.load(f)
        total = s.get("total_problems", 0)
        if not total:
            continue
        correct = s["total_correct"]
        out[max_tokens] = {
            "overall": {
                "accuracy": round(correct / total * 100, 4),
                "ifr": round(s["total_valid"] / total * 100, 4),
                "correct": correct,
                "valid": s["total_valid"],
                "total": total,
                "n_files": s.get("num_files_aggregated", 0),
            },
            "by_dataset": {
                ds: {
                    "accuracy": round(v["correct"] / v["total"] * 100, 4),
                    "ifr": round(v["valid"] / v["total"] * 100, 4),
                    "correct": v["correct"],
                    "valid": v["valid"], "total": v["total"],
                }
                for ds, v in s.get("by_dataset", {}).items() if v["total"]
            },
        }
    return out


def load_scoring(sdir: Path, judge: str) -> dict:
    """SummEval Spearman rho and mean MT-Bench Spearman rho from *_eval.json files.

    A judge that is also an MT-Bench reference model is not correlated against itself, so ``mtbench_n_pairs``
    (the reference x student pairs in the mean) differs between judges.
    """
    out = {"summeval_rho": None, "mtbench_rho": None, "mtbench_n_pairs": 0}
    if not sdir.exists():
        return out

    se_path = sdir / "summeval_eval.json"
    if se_path.exists():
        with open(se_path) as f:
            out["summeval_rho"] = json.load(f).get("overall", {}).get("spearman")

    rhos = []
    for reference in MTBENCH_REFERENCES:
        if reference == judge:
            continue
        for student in STUDENTS:
            mb_path = sdir / f"mtbench_{reference}_{student}_eval.json"
            if not mb_path.exists():
                continue
            with open(mb_path) as f:
                rho = json.load(f).get("overall", {}).get("spearman")
            if rho is not None:
                rhos.append(rho)
    if rhos:
        out["mtbench_rho"] = round(sum(rhos) / len(rhos), 4)
        out["mtbench_n_pairs"] = len(rhos)
    return out


def domain_accuracy(by_dataset: dict, datasets: list[str]) -> float | None:
    """Micro-averaged accuracy over a subset of datasets."""
    counts = [
        Counts(total=v["total"], correct=v["correct"], valid=v["valid"])
        for ds, v in by_dataset.items() if ds in datasets
    ]
    m = micro(counts)
    return m["accuracy"] if m else None


def build_rows() -> list[dict]:
    large_scoring = large_results() / "scoring_evaluations"
    slm_scoring = paths.core("scoring_evaluations")
    rows = []
    sources = [
        (LARGE_JUDGES, "Large", load_large_accuracy, large_scoring),
        (LADDER_JUDGES, "Ladder", load_large_accuracy, large_scoring),
        (SLM_JUDGES, "SLM", load_slm_accuracy, slm_scoring),
    ]
    for roster, jtype, loader, scoring_dir in sources:
        for judge, info in roster.items():
            params = info["params"] if isinstance(info, dict) else info
            display = info["name"] if isinstance(info, dict) else judge
            acc = loader(judge)
            if not acc:
                continue
            scoring = load_scoring(scoring_dir / judge, judge)
            # Only budgets whose verdicts parsed can decide the best budget.
            usable = {t: v for t, v in acc.items()
                      if v["overall"]["ifr"] >= MIN_USABLE_IFR}
            best_tok = max(usable or acc,
                           key=lambda t: (usable or acc)[t]["overall"]["accuracy"])
            best = acc[best_tok]
            rows.append({
                "judge": judge,
                "display": display,
                "type": jtype,
                "params_b": params,
                "acc_t10": acc.get(10, {}).get("overall", {}).get("accuracy"),
                "acc_t8192": acc.get(8192, {}).get("overall", {}).get("accuracy"),
                "ifr_t10": acc.get(10, {}).get("overall", {}).get("ifr"),
                "ifr_t8192": acc.get(8192, {}).get("overall", {}).get("ifr"),
                "best_acc": best["overall"]["accuracy"],
                "best_budget": best_tok,
                "summeval_rho": scoring["summeval_rho"],
                "mtbench_rho": scoring["mtbench_rho"],
                "mtbench_n_pairs": scoring["mtbench_n_pairs"],
                "n_datasets": len(best["by_dataset"]),
                "n_items": best["overall"]["total"],
                "complete_8ds": len(best["by_dataset"]) == len(ALL_DATASETS),
                "usable_ifr": bool(usable),
                "_acc": acc,
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> Path:
    """Write rows to CSV with the first row's keys as header; no rows -> empty file."""
    if not rows:
        path.write_text("")
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def scaling_ladder(rows: list[dict]) -> list[dict]:
    """One row per (family, judge), ordered by size, with the step from the rung below.

    Each step has the parameter multiple, the accuracy gain in points and the gain per doubling of parameters.
    A rung with no results is kept (best_acc empty, present = False), so it reads as a gap in the ladder.
    """
    by_key = {r["judge"]: r for r in rows}
    out = []
    for family, members in LADDER_FAMILIES.items():
        prev_acc = prev_params = None
        for judge in members:
            r = by_key.get(judge)
            params = (r["params_b"] if r else
                      (LADDER_JUDGES.get(judge, {}).get("params")
                       or LARGE_JUDGES.get(judge, {}).get("params")
                       or SLM_JUDGES.get(judge)))
            acc = r["best_acc"] if r else None
            row = {
                "family": family,
                "judge": judge,
                "display": r["display"] if r else judge,
                "params_b": params,
                "best_acc": acc,
                "best_budget": r["best_budget"] if r else None,
                "summeval_rho": r["summeval_rho"] if r else None,
                "mtbench_rho": r["mtbench_rho"] if r else None,
                "present": r is not None,
                "param_multiple_from_prev": None,
                "acc_gain_pp": None,
                "pp_per_doubling": None,
            }
            if acc is not None and prev_acc is not None and params and prev_params:
                mult = params / prev_params
                gain = acc - prev_acc
                row["param_multiple_from_prev"] = round(mult, 3)
                row["acc_gain_pp"] = round(gain, 4)
                # Points per doubling puts steps of different size (2x, 8x)
                # on one scale.
                if mult > 1:
                    row["pp_per_doubling"] = round(gain / math.log2(mult), 4)
            if acc is not None:
                prev_acc, prev_params = acc, params
            out.append(row)
    return out


def ladder_summary(ladder: list[dict]) -> list[str]:
    """A few lines describing each family's curve, for the console."""
    lines = []
    for family in LADDER_FAMILIES:
        pts = [r for r in ladder if r["family"] == family and r["best_acc"] is not None]
        if len(pts) < 3:
            lines.append(f"{family}: only {len(pts)} rungs measured, curve incomplete")
            continue
        first, last = pts[0], pts[-1]
        span = last["params_b"] / first["params_b"]
        total = last["best_acc"] - first["best_acc"]
        top = pts[-1]
        lines.append(
            f"{family}: {first['display']} {first['best_acc']:.2f} -> "
            f"{last['display']} {last['best_acc']:.2f} "
            f"({span:.0f}x parameters, {total:+.2f} pp total)"
        )
        if top["param_multiple_from_prev"]:
            lines.append(
                f"    the last {top['param_multiple_from_prev']:.1f}x gives "
                f"{top['acc_gain_pp']:+.2f} pp"
            )
        early = pts[1]
        if early["param_multiple_from_prev"]:
            lines.append(
                f"    the first {early['param_multiple_from_prev']:.1f}x gives "
                f"{early['acc_gain_pp']:+.2f} pp"
            )
    return lines


def main():
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table
    out_dir = paths.analysis_dir(OUT_NAME)
    rows = build_rows()

    # ── Scaling ladder ─────────────────────────────────────────────────────
    ladder = scaling_ladder(rows)
    ladder_csv = write_csv(out_dir / "scaling_ladder.csv", ladder)
    print("=== Scaling ladder ===")
    for line in ladder_summary(ladder):
        print("  " + line)
    missing = [r["judge"] for r in ladder if not r["present"]]
    if missing:
        print(f"  no results: {', '.join(missing)}")
    print(f"  -> {ladder_csv}\n")

    rows.sort(key=lambda r: r["best_acc"] or 0, reverse=True)

    print(f"{'Judge':<32}{'Type':<7}{'Params':>8}{'Acc@t10':>10}{'Acc@t8192':>11}"
          f"{'Best':>8}{'SE rho':>9}{'MB rho':>8}{'#ds':>5}{'items':>9}")
    print("-" * 107)
    for r in rows:
        def fmt(v, dp=2):
            return f"{v:.{dp}f}" if v is not None else "N/A"
        flag = ""
        if not r["usable_ifr"]:
            flag = f"  <- UNUSABLE (IFR {r['ifr_t10'] or r['ifr_t8192']:.1f}%)"
        elif not r["complete_8ds"]:
            flag = "  <- partial coverage"
        print(f"{r['judge']:<32}{r['type']:<7}"
              f"{(str(r['params_b']) + 'B') if r['params_b'] else 'n/a':>8}"
              f"{fmt(r['acc_t10']):>10}{fmt(r['acc_t8192']):>11}"
              f"{fmt(r['best_acc']):>8}{fmt(r['summeval_rho'], 3):>9}"
              f"{fmt(r['mtbench_rho'], 3):>8}{r['n_datasets']:>5}"
              f"{r['n_items']:>9,}{flag}")

    # ── Overall comparison ─────────────────────────────────────────────────
    out_csv = out_dir / "small_vs_large.csv"
    fieldnames = ["judge", "display", "type", "params_b", "acc_t10", "acc_t8192",
                  "ifr_t10", "ifr_t8192", "best_acc", "best_budget",
                  "summeval_rho", "mtbench_rho", "mtbench_n_pairs",
                  "n_datasets", "n_items", "complete_8ds", "usable_ifr"]
    # The MT-Bench value is a mean over both reference models, so the column is
    # named mtbench_rho_tworef.
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[c if c != "mtbench_rho" else "mtbench_rho_tworef"
                           for c in fieldnames], extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{("mtbench_rho_tworef" if k == "mtbench_rho" else k): v
                           for k, v in r.items()} for r in rows])
    print(f"\nSaved: {out_csv}")

    # ── Per-dataset accuracy, both budgets ─────────────────────────────────
    per_ds_csv = out_dir / "per_dataset.csv"
    with open(per_ds_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judge", "type", "budget"] + ALL_DATASETS)
        for r in rows:
            for tok in TOKEN_SETTINGS:
                by_ds = r["_acc"].get(tok, {}).get("by_dataset", {})
                if not by_ds:
                    continue
                writer.writerow(
                    [r["judge"], r["type"], f"t{tok}"]
                    + [round(by_ds[ds]["accuracy"], 2) if ds in by_ds else ""
                       for ds in ALL_DATASETS]
                )
    print(f"Saved: {per_ds_csv}")

    # The same grid to six decimals from the integer counts, for anything that
    # rounds again later.
    precise_csv = out_dir / "per_dataset_precise.csv"
    with open(precise_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judge", "type", "budget"] + ALL_DATASETS)
        for r in rows:
            for tok in TOKEN_SETTINGS:
                by_ds = r["_acc"].get(tok, {}).get("by_dataset", {})
                if not by_ds:
                    continue
                writer.writerow(
                    [r["judge"], r["type"], f"t{tok}"]
                    + [round(by_ds[ds]["correct"] / by_ds[ds]["total"] * 100, 6)
                       if ds in by_ds else "" for ds in ALL_DATASETS]
                )
    print(f"Saved: {precise_csv}")

    # ── Math vs. general domain gap ────────────────────────────────────────
    # math_acc, general_acc and gap are unweighted means of the three dataset
    # accuracies in each domain. The *_itemweighted columns micro-average over
    # items instead, which lets the larger datasets dominate.
    gap_csv = out_dir / "domain_gap.csv"
    gap_rows = []
    for r in rows:
        for tok in TOKEN_SETTINGS:
            by_ds = r["_acc"].get(tok, {}).get("by_dataset", {})
            if not by_ds:
                continue
            math_w = domain_accuracy(by_ds, MATH_DATASETS)
            gen_w = domain_accuracy(by_ds, GENERAL_DATASETS)
            if math_w is None or gen_w is None:
                continue
            if not all(d in by_ds for d in MATH_DATASETS + GENERAL_DATASETS):
                continue

            def macro(ds):
                return sum(by_ds[d]["correct"] / by_ds[d]["total"] * 100 for d in ds) / len(ds)
            math_acc, gen_acc = macro(MATH_DATASETS), macro(GENERAL_DATASETS)
            gap_rows.append({
                "judge": r["judge"], "type": r["type"], "params_b": r["params_b"],
                "budget": f"t{tok}",
                "math_acc": round(math_acc, 6), "general_acc": round(gen_acc, 6),
                "gap": round(math_acc - gen_acc, 6),
                "math_acc_itemweighted": round(math_w, 2),
                "general_acc_itemweighted": round(gen_w, 2),
                "gap_itemweighted": round(math_w - gen_w, 2),
                "n_math_datasets": sum(1 for d in MATH_DATASETS if d in by_ds),
                "n_general_datasets": sum(1 for d in GENERAL_DATASETS if d in by_ds),
            })
    with open(gap_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gap_rows)
    print(f"Saved: {gap_csv}")

    # ── Best small vs. best large, complete rows only ──────────────────────
    comparable = [r for r in rows if r["complete_8ds"] and r["usable_ifr"]]
    slm_best = max((r for r in comparable if r["type"] == "SLM"),
                   key=lambda r: r["best_acc"], default=None)
    large_best = max((r for r in comparable if r["type"] == "Large"),
                     key=lambda r: r["best_acc"], default=None)
    print("\n=== Best judges (rows covering all 8 datasets) ===")
    if slm_best and large_best:
        print(f"Best small judge: {slm_best['judge']} {slm_best['best_acc']:.2f}% "
              f"(t{slm_best['best_budget']})")
        print(f"Best large judge: {large_best['judge']} {large_best['best_acc']:.2f}% "
              f"(t{large_best['best_budget']})")
        print(f"Gap (large - small): {large_best['best_acc'] - slm_best['best_acc']:+.2f} pp")
    partial = [r["judge"] for r in rows if r["usable_ifr"] and not r["complete_8ds"]]
    if partial:
        print(f"\nExcluded (partial dataset coverage): {', '.join(partial)}")
    unusable = [r["judge"] for r in rows if not r["usable_ifr"]]
    if unusable:
        print(f"Excluded (verdicts did not parse, IFR < {MIN_USABLE_IFR}%): "
              f"{', '.join(unusable)}")
    if CORRUPT_FILES:
        print(f"\n{len(CORRUPT_FILES)} unreadable result file(s) skipped:")
        for p in CORRUPT_FILES:
            print(f"   {p}")


if __name__ == "__main__":
    main()
