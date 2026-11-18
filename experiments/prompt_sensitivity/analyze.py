#!/usr/bin/env python3
"""Prompt sensitivity: how much does a judge's accuracy move when only the prompt wording changes?"""

import argparse
import csv
from pathlib import Path

from slmjury.analysis import paths
from slmjury.analysis.accuracy import CORRUPT_FILES, file_counts, micro
from slmjury.analysis.metrics import rank_stability
from slmjury.parsers.normalizer import set_strict_labels

NOTES = """\
Each judge graded the two main students under seven prompt variants (see
slmjury/configs/prompt_variants.yaml). Five of them (original, minimal,
gpt_style, rubric_based, cot_style) form the main comparison; adversarial and
few_shot are stress tests reported against original.

Two rules keep the comparison fair:

1. Accuracy is micro-averaged, sum(correct) / sum(items) over files, as
   generate_judge_summary does. Averaging per-file accuracies would give
   small datasets (TruthfulQA, 684 items) the same weight as large ones
   (ARC-Easy, 2,376).

2. Variants are compared on common cells. For each judge, only the (student,
   dataset) cells present under every variant are scored. Otherwise a judge
   with 32 files under one variant and 7 under another would show a
   difference that comes from coverage, not from the prompt.

Reads (under results/, or $SLMJURY_RESULTS):
  prompt_sensitivity/<variant>/<judge>/<student>_<dataset>_t<budget>.json

Writes results/analysis/prompt_sensitivity/:
  accuracy_table.csv          judge x budget accuracy per variant and the range,
                              each judge on its own common cells; stress
                              variants appear as extra rows (variant_group)
  shared_accuracy_table.csv   the judges with all five main variants, scored on
                              the cells every such judge shares, so all rows
                              are on the same items
  rank_stability.csv          mean and minimum pairwise Kendall tau between the
                              judge rankings the five main variants give
  stress_variants.csv         adversarial and few_shot against original, with
                              the unparseable share beside each accuracy, so a
                              formatting failure can be told from a wrong verdict
  coverage.csv                which cells each judge has under each variant

The runs come from run_prompt_variants.py in this folder, once per small judge
and variant (docs/REPRODUCE.md has the commands). Its verdicts are read here;
its SummEval and MT-Bench scores are read by analyze_open_ended.py.

Run it from the repository root.
"""

OUT_NAME = "prompt_sensitivity"

CORE_VARIANTS = ["original", "minimal", "gpt_style", "rubric_based", "cot_style"]
EXTRA_VARIANTS = ["adversarial", "few_shot"]
ALL_VARIANTS = CORE_VARIANTS + EXTRA_VARIANTS

STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]
ALL_DATASETS = [
    "gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
    "hellaswag", "winogrande", "truthfulqa",
]
TOKEN_SETTINGS = [10, 8192]


def variant_dir(variant: str) -> Path:
    return paths.experiment(OUT_NAME) / variant


def discover_judges() -> list[str]:
    judges = set()
    for variant in ALL_VARIANTS:
        d = variant_dir(variant)
        if d.exists():
            # scoring_evaluations/ holds the open-ended metrics, not a judge.
            judges.update(p.name for p in d.iterdir()
                          if p.is_dir() and p.name != "scoring_evaluations")
    return sorted(judges)


def cells_present(judge: str, variant: str, max_tokens: int) -> set[tuple[str, str]]:
    """(student, dataset) cells this (judge, variant, budget) has a file for."""
    judge_dir = variant_dir(variant) / judge
    if not judge_dir.exists():
        return set()
    return {
        (student, dataset)
        for student in STUDENTS for dataset in ALL_DATASETS
        if (judge_dir / f"{student}_{dataset}_t{max_tokens}.json").exists()
    }


def accuracy_on_cells(judge, variant, max_tokens, cells) -> dict | None:
    """Micro-averaged accuracy restricted to an explicit set of cells."""
    judge_dir = variant_dir(variant) / judge
    counts = [
        file_counts(judge, student, dataset, max_tokens,
                    judge_dir / f"{student}_{dataset}_t{max_tokens}.json")
        for student, dataset in sorted(cells)
    ]
    return micro(counts)


def build_table(judges, variants, max_tokens) -> tuple[dict, dict]:
    """Accuracy per (judge, variant) on each judge's common cells.

    Returns (accuracy_table, coverage_table).
    """
    acc_table, coverage = {}, {}
    for judge in judges:
        per_variant_cells = {v: cells_present(judge, v, max_tokens) for v in variants}
        available = [v for v in variants if per_variant_cells[v]]
        if len(available) < 2:
            continue
        common = set.intersection(*(per_variant_cells[v] for v in available))
        if not common:
            continue
        coverage[judge] = {
            "variants_available": available,
            "n_common_cells": len(common),
            "datasets": sorted({d for _, d in common}),
            "students": sorted({s for s, _ in common}),
            "per_variant_cells": {v: len(per_variant_cells[v]) for v in variants},
        }
        acc_table[judge] = {
            v: accuracy_on_cells(judge, v, max_tokens, common) for v in available
        }
    return acc_table, coverage


def main():
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table
    out_dir = paths.analysis_dir(OUT_NAME)
    print("=== Prompt sensitivity ===\n")
    judges = discover_judges()
    print(f"Judges with any results ({len(judges)}): {', '.join(judges)}\n")

    acc_rows, stability_rows, coverage_rows, shared_rows = [], [], [], []
    stress_rows: list[dict] = []

    for max_tokens in TOKEN_SETTINGS:
        label = f"t{max_tokens}"
        print(f"\n{'=' * 78}\n  Budget {label}\n{'=' * 78}")

        # ── The five main variants ─────────────────────────────────────────
        acc_table, coverage = build_table(judges, CORE_VARIANTS, max_tokens)

        header = f"  {'Judge':<16}" + "".join(f"{v[:10]:>12}" for v in CORE_VARIANTS)
        print(header + f"{'Range':>9}{'cells':>7}{'ds':>4}")
        print("  " + "-" * (16 + 12 * len(CORE_VARIANTS) + 20))

        for judge in sorted(acc_table, key=lambda j: -(
                acc_table[j].get("original") or {"accuracy": 0})["accuracy"]):
            vals = acc_table[judge]
            got = [vals[v]["accuracy"] for v in CORE_VARIANTS if vals.get(v)]
            rng = round(max(got) - min(got), 6) if len(got) > 1 else None
            cov = coverage[judge]
            row = f"  {judge:<16}"
            for v in CORE_VARIANTS:
                row += (f"{vals[v]['accuracy']:>12.2f}" if vals.get(v) else f"{'—':>12}")
            row += (f"{rng:>9.2f}" if rng is not None else f"{'—':>9}")
            row += f"{cov['n_common_cells']:>7}{len(cov['datasets']):>4}"
            print(row)

            acc_rows.append({
                "judge": judge, "max_tokens": max_tokens,
                **{f"acc_{v}": (vals[v]["accuracy"] if vals.get(v) else None)
                   for v in CORE_VARIANTS},
                "acc_range": rng,
                "n_variants": len(got),
                "n_common_cells": cov["n_common_cells"],
                "n_datasets": len(cov["datasets"]),
                "n_items": (vals.get("original") or list(vals.values())[0])["total"],
                "complete_5_variants": len(got) == len(CORE_VARIANTS),
            })

        # ── Rank stability over the judges with all five variants ──────────
        # Ranking judges against each other also needs every judge scored on
        # the same items, so the cells are intersected across judges as well
        # as across variants.
        full = [j for j in acc_table
                if all(acc_table[j].get(v) for v in CORE_VARIANTS)]
        if len(full) >= 2:
            shared_cells = set.intersection(*(
                set.intersection(*(cells_present(j, v, max_tokens)
                                   for v in CORE_VARIANTS))
                for j in full
            ))
            by_variant = {
                v: [accuracy_on_cells(j, v, max_tokens, shared_cells)["accuracy"]
                    for j in full]
                for v in CORE_VARIANTS
            }
            stab = rank_stability(by_variant)
            print(f"\n  Rank stability over {len(full)} judges with all 5 variants "
                  f"({len(shared_cells)} cells shared by every judge, "
                  f"{len({d for _, d in shared_cells})} datasets):")

            # Every judge and variant on the identical shared cells.
            for judge in full:
                per_variant = {
                    v: accuracy_on_cells(judge, v, max_tokens, shared_cells)
                    for v in CORE_VARIANTS
                }
                accs = [per_variant[v]["accuracy"] for v in CORE_VARIANTS]
                shared_rows.append({
                    "judge": judge, "max_tokens": max_tokens,
                    **{f"acc_{v}": per_variant[v]["accuracy"]
                       for v in CORE_VARIANTS},
                    "acc_range": round(max(accs) - min(accs), 6),
                    "n_cells": len(shared_cells),
                    "n_datasets": len({d for _, d in shared_cells}),
                    "n_items": per_variant["original"]["total"],
                })
            print(f"    mean tau = {stab['mean_tau']:.4f}   min tau = {stab['min_tau']:.4f}")
            for (v1, v2), tau in sorted(stab["pairwise"].items()):
                print(f"      {v1:<14} vs {v2:<14} tau = {tau:+.3f}")
            stability_rows.append({
                "max_tokens": max_tokens, "label": label,
                # Six decimals: tau over thirteen judges is a multiple of 1/78,
                # so the mean of ten of them repeats, and four decimals would
                # be rounded a second time when printed to three.
                "mean_tau": round(stab["mean_tau"], 6),
                "min_tau": round(stab["min_tau"], 6),
                "n_variants": len(CORE_VARIANTS),
                "n_judges": len(full),
                "n_shared_cells": len(shared_cells),
                "n_shared_datasets": len({d for _, d in shared_cells}),
                "judges": ";".join(sorted(full)),
            })

        # ── Stress variants against original, on common cells ──────────────
        extra_table, extra_cov = build_table(
            judges, ["original"] + EXTRA_VARIANTS, max_tokens,
        )
        extra_present = {
            j: v for j, v in extra_table.items()
            if any(v.get(e) for e in EXTRA_VARIANTS)
        }
        if extra_present:
            print("\n  Stress variants (vs original, common cells only):")
            print(f"    {'Judge':<16}{'original':>10}" +
                  "".join(f"{e:>13}" for e in EXTRA_VARIANTS))
            for judge in sorted(extra_present):
                vals = extra_present[judge]
                row = f"    {judge:<16}"
                row += (f"{vals['original']['accuracy']:>10.2f}"
                        if vals.get("original") else f"{'—':>10}")
                for e in EXTRA_VARIANTS:
                    row += (f"{vals[e]['accuracy']:>13.2f}" if vals.get(e) else f"{'—':>13}")
                print(row)
                for e in EXTRA_VARIANTS:
                    if vals.get(e):
                        acc_rows.append({
                            "judge": judge, "max_tokens": max_tokens,
                            f"acc_{e}": vals[e]["accuracy"],
                            "acc_original": (vals["original"]["accuracy"]
                                             if vals.get("original") else None),
                            "n_common_cells": extra_cov[judge]["n_common_cells"],
                            "n_datasets": len(extra_cov[judge]["datasets"]),
                            "n_items": vals[e]["total"],
                            "variant_group": f"extra:{e}",
                        })

            # A judge that emits no parseable verdict and a judge that emits a
            # confident wrong one both lose accuracy. Reporting the unparseable
            # share (100 - IFR, on the same cells) beside the accuracy tells
            # the two apart.
            for judge in sorted(extra_present):
                vals = extra_present[judge]
                if not vals.get("original"):
                    continue
                base = vals["original"]
                row = {
                    "judge": judge, "max_tokens": max_tokens,
                    "n_common_cells": extra_cov[judge]["n_common_cells"],
                    "n_datasets": len(extra_cov[judge]["datasets"]),
                    "n_items": base["total"],
                    "acc_original": base["accuracy"],
                    "unparseable_original": round(100 - base["ifr"], 6),
                }
                for e in EXTRA_VARIANTS:
                    v = vals.get(e)
                    row[f"acc_{e}"] = v["accuracy"] if v else None
                    row[f"delta_{e}"] = (round(v["accuracy"] - base["accuracy"], 6)
                                         if v else None)
                    row[f"unparseable_{e}"] = round(100 - v["ifr"], 6) if v else None
                stress_rows.append(row)

        for judge, cov in coverage.items():
            coverage_rows.append({
                "judge": judge, "max_tokens": max_tokens,
                "n_common_cells": cov["n_common_cells"],
                "n_datasets": len(cov["datasets"]),
                "datasets": ";".join(cov["datasets"]),
                "variants_available": ";".join(cov["variants_available"]),
                **{f"cells_{v}": cov["per_variant_cells"].get(v, 0)
                   for v in CORE_VARIANTS},
            })

    # ── Save ───────────────────────────────────────────────────────────────
    acc_csv = out_dir / "accuracy_table.csv"
    fieldnames = (["judge", "max_tokens"]
                  + [f"acc_{v}" for v in ALL_VARIANTS]
                  + ["acc_range", "n_variants", "n_common_cells", "n_datasets",
                     "n_items", "complete_5_variants", "variant_group"])
    with open(acc_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(acc_rows)
    print(f"\nSaved: {acc_csv}")

    shared_csv = out_dir / "shared_accuracy_table.csv"
    with open(shared_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=(["judge", "max_tokens"]
                           + [f"acc_{v}" for v in CORE_VARIANTS]
                           + ["acc_range", "n_cells", "n_datasets", "n_items"]))
        writer.writeheader()
        writer.writerows(shared_rows)
    print(f"Saved: {shared_csv}")

    stab_csv = out_dir / "rank_stability.csv"
    with open(stab_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["max_tokens", "label", "mean_tau", "min_tau",
                           "n_variants", "n_judges", "n_shared_cells",
                           "n_shared_datasets", "judges"],
        )
        writer.writeheader()
        writer.writerows(stability_rows)
    print(f"Saved: {stab_csv}")

    stress_csv = out_dir / "stress_variants.csv"
    with open(stress_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=(["judge", "max_tokens", "acc_original"]
                           + [c for e in EXTRA_VARIANTS
                              for c in (f"acc_{e}", f"delta_{e}")]
                           + ["unparseable_original"]
                           + [f"unparseable_{e}" for e in EXTRA_VARIANTS]
                           + ["n_common_cells", "n_datasets", "n_items"]))
        writer.writeheader()
        writer.writerows(stress_rows)
    print(f"Saved: {stress_csv}")

    cov_csv = out_dir / "coverage.csv"
    with open(cov_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coverage_rows)
    print(f"Saved: {cov_csv}")

    if CORRUPT_FILES:
        print(f"\n!! {len(CORRUPT_FILES)} unreadable result file(s) skipped:")
        for p in CORRUPT_FILES:
            print(f"   {p}")


if __name__ == "__main__":
    main()
