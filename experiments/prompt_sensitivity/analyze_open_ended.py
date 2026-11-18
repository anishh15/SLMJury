#!/usr/bin/env python3
"""Prompt sensitivity, open-ended: does a judge's agreement with reference scores move with the prompt wording?"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from slmjury.analysis import paths, scoring
from slmjury.analysis.metrics import rank_stability

NOTES = """\
Scores the SummEval and MT-Bench runs of each judge under the seven prompt
variants the same way experiments/large_judges/analyze.py scores open-ended
judging, so the numbers are comparable:

  summeval_rho        overall Spearman rho against the human ratings
  mtbench_rho         mean overall Spearman rho over the (reference model,
                      student) pairs; a judge is never compared with itself
                      as reference
  mtbench_rho_common  the same against qwen3.5-397b only, the one reference
                      model that is not also a judge, so every judge is
                      scored on the same pairs

The metrics are recomputed from the raw scoring records with
slmjury.analysis.scoring so they are rounded once. When a raw file is
missing, the stored *_eval.json value is used instead.

Reads (under results/, or $SLMJURY_RESULTS):
  prompt_sensitivity/<variant>/<judge>/{summeval,mtbench_<reference>_<student>}.json
  prompt_sensitivity/<variant>/scoring_evaluations/<judge>/*_eval.json

Writes results/analysis/prompt_sensitivity/:
  open_ended_table.csv           judge x variant: SummEval and MT-Bench rho
  open_ended_summary.csv         one row per judge and metric: rho under each
                                 variant, the range over the five main ones,
                                 and the change under each stress variant
  open_ended_rank_stability.csv  Kendall tau between the judge orderings the
                                 five main variants give

The runs come from run_prompt_variants.py with --prompt-type scoring
(docs/REPRODUCE.md has the commands). Run it from the repository root.
"""

OUT_NAME = "prompt_sensitivity"

CORE_VARIANTS = ["original", "minimal", "gpt_style", "rubric_based", "cot_style"]
EXTRA_VARIANTS = ["adversarial", "few_shot"]
ALL_VARIANTS = CORE_VARIANTS + EXTRA_VARIANTS

STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]
MTBENCH_REFERENCES = ["gpt-oss-120b", "qwen3.5-397b"]
COMMON_REFERENCE = "qwen3.5-397b"


def results_root() -> Path:
    return paths.experiment(OUT_NAME)


def scoring_dir(variant: str) -> Path:
    return results_root() / variant / "scoring_evaluations"


def discover_judges() -> list[str]:
    judges = set()
    for v in ALL_VARIANTS:
        d = scoring_dir(v)
        if d.exists():
            judges.update(p.name for p in d.iterdir() if p.is_dir())
    return sorted(judges)


def load_scoring(variant: str, judge: str) -> dict:
    """SummEval and averaged MT-Bench Spearman rho for one (variant, judge)."""
    d = scoring_dir(variant) / judge
    out = {"summeval_rho": None, "mtbench_rho": None, "mtbench_n_pairs": 0,
           "mtbench_rho_common": None, "summeval_parse_fail": None}
    if not d.exists():
        return out

    raw_dir = results_root() / variant / judge
    se = d / "summeval_eval.json"
    if (raw_dir / "summeval.json").exists():
        with open(raw_dir / "summeval.json") as f:
            blob = scoring.summeval(json.load(f))
        rho = blob["overall"].get("spearman")
        out["summeval_rho"] = None if rho is None else round(rho, 10)
        out["summeval_parse_fail"] = blob["overall"].get("parse_failure_rate")
    elif se.exists():
        with open(se) as f:
            blob = json.load(f)
        out["summeval_rho"] = blob.get("overall", {}).get("spearman")
        out["summeval_parse_fail"] = blob.get("overall", {}).get("parse_failure_rate")

    rhos, fails, common = [], [], []
    for reference in MTBENCH_REFERENCES:
        if reference == judge:
            continue
        for student in STUDENTS:
            p = d / f"mtbench_{reference}_{student}_eval.json"
            raw = raw_dir / f"mtbench_{reference}_{student}.json"
            if raw.exists():
                with open(raw) as f:
                    blob = scoring.mtbench(json.load(f))
            elif p.exists():
                with open(p) as f:
                    blob = json.load(f)
            else:
                continue
            rho = blob.get("overall", {}).get("spearman")
            if rho is not None:
                rhos.append(rho)
                if reference == COMMON_REFERENCE:
                    common.append(rho)
            pf = blob.get("overall", {}).get("parse_failure_rate")
            if pf is not None:
                fails.append(pf)
    if rhos:
        out["mtbench_rho"] = round(sum(rhos) / len(rhos), 10)
        out["mtbench_n_pairs"] = len(rhos)
    if len(common) == len(STUDENTS):
        out["mtbench_rho_common"] = round(sum(common) / len(common), 10)
    out["mtbench_parse_fail"] = round(sum(fails) / len(fails), 6) if fails else None
    return out


def rng(vals):
    vals = [v for v in vals if v is not None]
    return round(max(vals) - min(vals), 10) if len(vals) > 1 else None


def main():
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    out_dir = paths.analysis_dir(OUT_NAME)
    judges = discover_judges()
    table = {j: {v: load_scoring(v, j) for v in ALL_VARIANTS} for j in judges}

    # ── per (judge, variant) ────────────────────────────────────────────────
    with open(out_dir / "open_ended_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "variant", "summeval_rho", "mtbench_rho",
                    "mtbench_n_pairs", "summeval_parse_failure_rate",
                    "mtbench_parse_failure_rate", "mtbench_rho_common"])
        for j in judges:
            for v in ALL_VARIANTS:
                r = table[j][v]
                w.writerow([j, v, r["summeval_rho"], r["mtbench_rho"],
                            r["mtbench_n_pairs"], r["summeval_parse_fail"],
                            r.get("mtbench_parse_fail"), r["mtbench_rho_common"]])

    # ── per judge, one row per metric ───────────────────────────────────────
    summary_rows = []
    for metric in ("summeval_rho", "mtbench_rho", "mtbench_rho_common"):
        for j in judges:
            core = [table[j][v][metric] for v in CORE_VARIANTS]
            row = {"judge": j, "metric": metric}
            for v in ALL_VARIANTS:
                row[v] = table[j][v][metric]
            row["range_core"] = rng(core)
            base = table[j]["original"][metric]
            for v in EXTRA_VARIANTS:
                got = table[j][v][metric]
                row[f"delta_{v}"] = (round(got - base, 10)
                                     if (got is not None and base is not None)
                                     else None)
            row["complete_5_variants"] = all(c is not None for c in core)
            summary_rows.append(row)

    cols = (["judge", "metric"] + ALL_VARIANTS + ["range_core"]
            + [f"delta_{v}" for v in EXTRA_VARIANTS] + ["complete_5_variants"])
    with open(out_dir / "open_ended_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)

    # ── rank stability across the five main variants ────────────────────────
    stab_rows = []
    for metric in ("summeval_rho", "mtbench_rho", "mtbench_rho_common"):
        pool = [j for j in judges
                if all(table[j][v][metric] is not None for v in CORE_VARIANTS)]
        if len(pool) < 3:
            continue
        by_variant = {v: [table[j][v][metric] for j in pool] for v in CORE_VARIANTS}
        stab = rank_stability(by_variant)
        stab_rows.append({
            "metric": metric, "n_judges": len(pool),
            "n_variants": len(CORE_VARIANTS),
            "mean_tau": round(stab["mean_tau"], 6) if stab["mean_tau"] is not None else None,
            "min_tau": round(stab["min_tau"], 6) if stab["min_tau"] is not None else None,
            **{f"tau_{a}__{b}": round(t, 6) for (a, b), t in stab["pairwise"].items()
               if t is not None},
        })
    if stab_rows:
        keys = sorted({k for r in stab_rows for k in r},
                      key=lambda k: (k not in ("metric", "n_judges", "n_variants",
                                               "mean_tau", "min_tau"), k))
        with open(out_dir / "open_ended_rank_stability.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(stab_rows)

    # ── console ─────────────────────────────────────────────────────────────
    print("=== Prompt sensitivity of SummEval / MT-Bench rho ===\n")
    for metric, label in (("summeval_rho", "SummEval Spearman rho"),
                          ("mtbench_rho", "MT-Bench Spearman rho (mean over pairs)"),
                          ("mtbench_rho_common", "MT-Bench Spearman rho, common reference")):
        print(f"  {label}")
        print(f"  {'Judge':<15}" + "".join(f"{v[:10]:>12}" for v in ALL_VARIANTS)
              + f"{'range':>9}")
        print("  " + "-" * (15 + 12 * len(ALL_VARIANTS) + 9))
        rows = [r for r in summary_rows if r["metric"] == metric]
        rows.sort(key=lambda r: -(r["original"] if r["original"] is not None else -9))
        for r in rows:
            cells = "".join(
                (f"{r[v]:>12.3f}" if r[v] is not None else f"{'—':>12}")
                for v in ALL_VARIANTS)
            rr = f"{r['range_core']:>9.3f}" if r["range_core"] is not None else f"{'—':>9}"
            print(f"  {r['judge']:<15}{cells}{rr}")
        print()
    for r in stab_rows:
        print(f"  rank stability [{r['metric']}]: {r['n_judges']} judges, "
              f"mean tau {r['mean_tau']:.3f}, min tau {r['min_tau']:.3f}")
    print(f"\nSaved: {out_dir}/open_ended_table.csv")
    print(f"Saved: {out_dir}/open_ended_summary.csv")
    print(f"Saved: {out_dir}/open_ended_rank_stability.csv")


if __name__ == "__main__":
    main()
