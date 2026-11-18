#!/usr/bin/env python3
"""Do judges agree on how to rank the students?"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from slmjury.analysis import paths
from slmjury.analysis.metrics import kendall_tau

NOTES = """\
If leniency were a family effect, a Qwen judge would place a Qwen-lineage
student higher than a Llama judge does. This script measures the orderings
directly, for the eight_datasets and five_datasets panels at both budgets.

Two orderings are read from the family-bias matrices:

  accuracy            each judge's students from easiest to hardest to grade;
                      this is the ordering that should follow the students'
                      own solver accuracy
  false_accept_rate   each judge's students from the one it is strictest with
                      to the one it is most lenient with

Agreement is Kendall's tau between the orderings of two judges, averaged over
every pair of judges. The last column compares each judge's ordering with the
students' solver accuracy and averages over judges. Every ordering is a
ranking without ties, so tau-a (slmjury.analysis.metrics.kendall_tau) is the
usual tau-b.

Reads results/analysis/family_bias/ (written by analyze.py in this folder):
  {eight_datasets,five_datasets}_student_solver_accuracy.csv
  {eight_datasets,five_datasets}_{accuracy,false_accept_rate}_matrix_t{10,8192}.csv

Writes results/analysis/family_bias/student_ordering.csv

Run it from the repository root, after analyze.py.
"""

OUT_NAME = "family_bias"

# The two weakest judges disagree with everything, including with themselves
# between runs. Both pools are reported so the effect of leaving them out shows.
WEAK = {"llama3.2-1b", "qwen3-0.6b"}


def matrix(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    students = [c for c in rows[0] if c not in ("judge", "judge_family")]
    return rows, students


def orderings(rows, students, descending: bool):
    """Each judge's students, best first (accuracy) or strictest first (FAR)."""
    out = {}
    for r in rows:
        vals = [(s, float(r[s])) for s in students if r[s] not in ("", "None")]
        vals.sort(key=lambda x: -x[1] if descending else x[1])
        out[r["judge"]] = [s for s, _ in vals]
    return out


def agreement(orders: dict[str, list[str]]):
    """Kendall tau for every pair of judges over the students both ranked."""
    taus = []
    for a, b in itertools.combinations(orders, 2):
        ra = {s: i for i, s in enumerate(orders[a])}
        rb = {s: i for i, s in enumerate(orders[b])}
        common = [s for s in ra if s in rb]
        if len(common) < 3:
            continue
        taus.append(kendall_tau([ra[s] for s in common],
                                [rb[s] for s in common]))
    return taus


def main() -> None:
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    src = paths.results_dir() / "analysis" / OUT_NAME
    out = []
    for panel in ("eight_datasets", "five_datasets"):
        spath = paths.require(src / f"{panel}_student_solver_accuracy.csv",
                              f"{panel}_student_solver_accuracy.csv "
                              "(run experiments/family_bias/analyze.py first)")
        with open(spath) as f:
            solver = {r["student"]: float(r["solver_accuracy"])
                      for r in csv.DictReader(f)}

        for metric, descending in (("accuracy", True), ("false_accept_rate", False)):
            for budget in ("t10", "t8192"):
                path = src / f"{panel}_{metric}_matrix_{budget}.csv"
                if not path.exists():
                    continue
                rows, students = matrix(path)
                for pool in ("all", "capable"):
                    sel = [r for r in rows
                           if pool == "all" or r["judge"] not in WEAK]
                    orders = orderings(sel, students, descending)
                    taus = agreement(orders)
                    if not taus:
                        continue

                    # Each judge's ordering against the students' solver accuracy.
                    ranked = sorted((s for s in students if s in solver),
                                    key=lambda s: -solver[s])
                    sr = {s: i for i, s in enumerate(ranked)}
                    vs = []
                    for order in orders.values():
                        seq = [s for s in order if s in sr]
                        if len(seq) < 3:
                            continue
                        vs.append(kendall_tau([sr[s] for s in seq],
                                              list(range(len(seq)))))

                    out.append({
                        "panel": panel,
                        "metric": metric,
                        "budget": budget,
                        "judge_pool": pool,
                        "n_judges": len(orders),
                        "n_students": len(students),
                        "n_distinct_orderings": len({tuple(v) for v in orders.values()}),
                        "mean_pairwise_tau": round(sum(taus) / len(taus), 4),
                        "min_pairwise_tau": round(min(taus), 4),
                        "mean_tau_vs_solver_accuracy": round(sum(vs) / len(vs), 4),
                    })

    out_csv = paths.analysis_dir(OUT_NAME) / "student_ordering.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"Saved: {out_csv}  ({len(out)} rows)")
    for r in out:
        if r["metric"] == "accuracy" and r["judge_pool"] == "all":
            print(f"  {r['panel']:14} {r['budget']:6} accuracy: "
                  f"{r['n_distinct_orderings']} orderings over {r['n_judges']} judges, "
                  f"mean tau {r['mean_pairwise_tau']:.3f}, "
                  f"vs solver accuracy {r['mean_tau_vs_solver_accuracy']:.3f}")


if __name__ == "__main__":
    main()
