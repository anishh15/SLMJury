#!/usr/bin/env python3
"""Lineage difference-in-differences and paired tests along the Qwen3 ladder."""

from __future__ import annotations

import argparse
import csv
import sys

from slmjury.analysis import paths
from slmjury.analysis.stats import cluster_bootstrap_diff, mcnemar, tost_equivalence
from slmjury.parsers.normalizer import set_strict_labels

import data
from analyze import out_dir, write_csv
from data import DISPLAY, best_budget, build_item_index, load_vector, paired_correctness

NOTES = """\
lineage_difference_in_differences.csv
  The effect of shared lineage on the false-accept rate (FAR), on the
  eight_datasets_non_phi4_judges panel of the family-bias analysis
  (11 judges x 6 students). For a judge family F and the student from F's
  lineage:

      [FAR of F's judges on that student - FAR of F's judges on the non-F students]
    - [the same two FARs for the judges outside F]

  Each FAR is a simple mean over judge x student cells. A positive value
  means F's judges are more lenient with their own lineage's student than
  the other judges are.

ladder_paired.csv
  The Qwen3 scaling ladder (0.6B to 122B), each rung against the one below
  it, plus Qwen3-32B against Phi-4. Each judge is used at its better budget
  and both are compared on the same items: exact McNemar test, cluster
  bootstrap 95% CI on the accuracy difference (10,000 resamples within
  datasets) and TOST for equivalence within +-0.5 points.

Reads:
  results/analysis/family_bias/eight_datasets_non_phi4_judges_false_accept_rate_matrix_t{10,8192}.csv
      (written by experiments/family_bias/analyze.py)
  the aligned correctness vectors of data.py (built from the judgement files
  on first use and cached)

Writes results/analysis/significance/lineage_difference_in_differences.csv
and ladder_paired.csv.

Run it from the repository root, after experiments/family_bias/analyze.py.
"""

FAMILY_JUDGES = {
    "Llama": ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"],
    "Qwen2.5": ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"],
    "Qwen3": ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"],
}
STUDENT_FAMILY = {
    "qwen2.5-32b": "Qwen2.5", "llama3.1-8b": "Llama",
    "deepseek-r1-0528-qwen3-8b": "Qwen3", "mistral-7b-v0.3": "Mistral",
    "gemma4-31b-student": "Gemma", "qwen3-14b-student": "Qwen3",
}
LINEAGE_STUDENT = {"Llama": "llama3.1-8b", "Qwen2.5": "qwen2.5-32b",
                   "Qwen3": "qwen3-14b-student"}


def lineage_difference_in_differences() -> list[dict]:
    fb = paths.results_dir() / "analysis" / "family_bias"
    out = []
    for budget in ("t10", "t8192"):
        fp = paths.require(fb / f"eight_datasets_non_phi4_judges_false_accept_rate_matrix_{budget}.csv",
                           "family-bias FAR matrix (run experiments/family_bias/analyze.py)")
        with open(fp) as fh:
            mat = {r["judge"]: r for r in csv.DictReader(fh)}
        studs = [c for c in next(iter(mat.values())) if c not in ("judge", "judge_family")]

        def mean(js, ss):
            v = [float(mat[j][s]) for j in js for s in ss]
            return sum(v) / len(v)

        for fam, fj in FAMILY_JUDGES.items():
            s = LINEAGE_STUDENT[fam]
            others = [x for x in studs if STUDENT_FAMILY[x] != fam]
            nonf = [j for j in mat if j not in fj]
            a_on, a_off = mean(fj, [s]), mean(fj, others)
            b_on, b_off = mean(nonf, [s]), mean(nonf, others)
            out.append({
                "budget": budget.replace("t", ""), "family": fam, "lineage_student": s,
                "far_family_on_student": a_on, "far_family_on_others": a_off,
                "far_other_judges_on_student": b_on,
                "far_other_judges_on_others": b_off,
                "did": (a_on - a_off) - (b_on - b_off),
                "n_family_judges": len(fj), "n_other_judges": len(nonf),
            })
    return out


def ladder() -> list[dict]:
    index = build_item_index()
    rungs = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b",
             "qwen3-32b", "qwen3.5-122b"]
    extra = [("qwen3-32b", "phi4-14b")]
    judges = set(rungs) | {j for p in extra for j in p}
    vecs = {}
    for j in judges:
        for b in data.budgets_for(j):
            vecs[(j, b)] = load_vector(j, b, index)
    best = {j: best_budget(vecs, j) for j in judges}

    def row(a, b):
        va, vb = vecs[(a, best[a])], vecs[(b, best[b])]
        x, y, cl = paired_correctness(va, vb, index)
        mc = mcnemar(x, y)
        bt = cluster_bootstrap_diff(x, y, cl, n_boot=10_000, seed=20260916)
        tt = tost_equivalence(x, y, cl, margin_points=0.5, n_boot=10_000, seed=20260916)
        return {
            "judge_a": DISPLAY.get(a, "Qwen3-32B" if a == "qwen3-32b" else a),
            "judge_b": DISPLAY.get(b, "Qwen3-32B" if b == "qwen3-32b" else b),
            "key_a": a, "key_b": b, "budget_a": best[a], "budget_b": best[b],
            "acc_a": va.accuracy(), "acc_b": vb.accuracy(),
            "diff_pp": bt.point, "discordant_a_only": mc.as_row()["discordant_a_only"],
            "discordant_b_only": mc.as_row()["discordant_b_only"],
            "mcnemar_p_exact": mc.p_exact, "boot_ci_lo_pp": bt.lo,
            "boot_ci_hi_pp": bt.hi, "tost_equivalent": tt["tost_equivalent"],
        }

    rows = [row(rungs[i + 1], rungs[i]) for i in range(len(rungs) - 1)]
    rows += [row(a, b) for a, b in extra]
    return rows


def main() -> int:
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table
    out = out_dir()
    did = lineage_difference_in_differences()
    write_csv(out / "lineage_difference_in_differences.csv", did)
    for r in did:
        print(f"  B={r['budget']:>5} {r['family']:<8} DiD {r['did']:+.4f}")
    lad = ladder()
    write_csv(out / "ladder_paired.csv", lad)
    for r in lad:
        print(f"  {r['judge_a']:<20} - {r['judge_b']:<16} {r['diff_pp']:+.3f} "
              f"[{r['boot_ci_lo_pp']:+.3f}, {r['boot_ci_hi_pp']:+.3f}] p={r['mcnemar_p_exact']:.3g} "
              f"equiv={r['tost_equivalent']}")
    print(f"\nWrote {out}/lineage_difference_in_differences.csv and ladder_paired.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
