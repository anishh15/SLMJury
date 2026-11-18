#!/usr/bin/env python3
"""Confidence intervals on the reported open-ended correlations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations

from slmjury.analysis import scoring
from slmjury.analysis.metrics import spearman

import analyze as A
from data import ALL_JUDGES, DISPLAY, LADDER_JUDGES, LARGE_JUDGES, STUDENTS

NOTES = """\
analyze.py bootstraps pooled correlations (SummEval with the four dimensions
pooled, MT-Bench with both students pooled). The reported statistics are
different: SummEval rho is the mean of the four per-dimension Spearman
correlations, and MT-Bench rho on the common reference model (qwen3.5-397b)
is the mean of the two per-student correlations. This script bootstraps
those statistics, after checking that each point estimate matches the stored
evaluation files.

  SummEval: the 100 articles (16 summaries each) are resampled, 10,000 times.
  MT-Bench: the 80 questions are resampled, with the same draw for both students.

Every judge is resampled with the same draws, so the difference between two
judges gets a paired interval. Point estimates are computed without
intermediate rounding (slmjury.analysis.scoring), and values are written to
six decimals so that rounding them to three for a table rounds only once.

The 22 judges of analyze.py are joined by the Qwen3-32B ladder judge.

Reads (under results/, or $SLMJURY_RESULTS):
  core/scoring/<judge>/, large_judges/<judge>/
                                               summeval.json, mtbench_qwen3.5-397b_<student>.json
  core/scoring_evaluations/<judge>/, large_judges/scoring_evaluations/<judge>/
                                               mtbench_qwen3.5-397b_<student>_eval.json

Writes results/analysis/significance/:
  open_ended_reported_ci.csv  per judge: SummEval rho and MT-Bench rho with 95% CIs
  open_ended_paired.csv       paired differences between every two judges
  mtbench_common_metrics.csv  Pearson, Spearman, kappa, accuracy and MSE on the
                              common reference, mean over the two students
  summeval_metrics.csv        the same five metrics for SummEval (mean over the
                              four dimensions), with per-dimension Spearman and kappa

Run it from the repository root.
"""

N_BOOT = 10_000
SEED = 20260916
COMMON = "qwen3.5-397b"
METRICS = ["pearson", "spearman", "cohens_kappa", "accuracy", "mse"]


def judges() -> list[str]:
    return list(ALL_JUDGES) + list(LADDER_JUDGES)


def type_of(j: str) -> str:
    if j in LADDER_JUDGES:
        return "Ladder"
    return "Large" if j in LARGE_JUDGES else "SLM"


def name(j: str) -> str:
    return "Qwen3-32B" if j == "qwen3-32b" else DISPLAY.get(j, j)


def pct(sorted_vals: list[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


# ── SummEval ────────────────────────────────────────────────────────────────

def summeval(js, rho):
    from slmjury.core.scoring_evaluator import evaluate_summeval
    ref = json.loads((A.scoring_dir("phi4-14b") / "summeval.json").read_text())
    order = [(r["article_id"], r["summary_id"]) for r in ref]
    articles = sorted({a for a, _ in order})
    by_article = {a: [i for i, (aa, _) in enumerate(order) if aa == a] for a in articles}

    mats, point = {}, {}
    for j in js:
        mats[j] = A.summeval_matrix(j, order)
        point[j] = A.summeval_rho(mats[j], range(len(order)), rho)
        raw = json.loads((A.scoring_dir(j) / "summeval.json").read_text())
        stored = evaluate_summeval(raw)
        if abs(point[j] - stored["overall"]["spearman"]) > 1e-3:
            raise SystemExit(f"{j}: SummEval {point[j]:.4f} != stored {stored['overall']['spearman']}")
        # The evaluator rounds to four decimals, so the reported point estimate
        # is taken from the unrounded metric.
        point[j] = scoring.summeval(raw)["overall"]["spearman"]

    rng = random.Random(SEED)
    draws = []
    for _ in range(N_BOOT):
        pick = [articles[rng.randrange(len(articles))] for _ in articles]
        draws.append([i for a in pick for i in by_article[a]])
    boot = {j: [A.summeval_rho(mats[j], idx, rho) for idx in draws] for j in js}
    return point, boot


# ── MT-Bench on the common reference ────────────────────────────────────────

def _num(x):
    return None if x is None else float(x)


def mtbench(js, rho):
    data, point = {}, {}
    qids = None
    for j in js:
        per = {}
        for s in STUDENTS:
            recs = json.loads((A.scoring_dir(j) / f"mtbench_{COMMON}_{s}.json").read_text())
            recs = sorted(recs, key=lambda r: int(r["question_id"]))
            ids = [r["question_id"] for r in recs]
            if qids is None:
                qids = ids
            if ids != qids:
                raise SystemExit(f"{j}/{s}: question order differs")
            per[s] = [(_num(r["judge_score"]), _num(r["oracle_score"])) for r in recs]
        data[j] = per

    def stat(per, idx):
        vals = []
        for s in STUDENTS:
            pairs = [per[s][i] for i in idx if per[s][i][0] is not None and per[s][i][1] is not None]
            xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
            if len(pairs) > 2 and len(set(xs)) > 1 and len(set(ys)) > 1:
                vals.append(rho(xs, ys))
        return sum(vals) / len(vals) if len(vals) == len(STUDENTS) else None

    n = len(qids)
    for j in js:
        point[j] = stat(data[j], range(n))
        stored, precise = [], []
        for s in STUDENTS:
            ev = A.eval_dir(j) / f"mtbench_{COMMON}_{s}_eval.json"
            stored.append(json.loads(ev.read_text())["overall"]["spearman"])
            raw = json.loads((A.scoring_dir(j) / f"mtbench_{COMMON}_{s}.json").read_text())
            precise.append(scoring.mtbench(raw)["overall"]["spearman"])
        if abs(point[j] - sum(stored) / len(stored)) > 1e-3:
            raise SystemExit(f"{j}: MT-Bench {point[j]:.4f} != stored mean {sum(stored)/len(stored):.4f}")
        point[j] = sum(precise) / len(precise)

    rng = random.Random(SEED + 1)
    draws = [[rng.randrange(n) for _ in range(n)] for _ in range(N_BOOT)]
    boot = {j: [stat(data[j], idx) for idx in draws] for j in js}
    return point, boot


# ── Five-metric tables ──────────────────────────────────────────────────────

def five_metric_rows(js):
    mt, se = [], []
    for j in js:
        d = A.scoring_dir(j)
        evs = [scoring.mtbench(json.loads((d / f"mtbench_{COMMON}_{s}.json").read_text()))["overall"]
               for s in STUDENTS]
        row = {"judge": name(j), "key": j, "type": type_of(j)}
        for m in METRICS:
            vals = [e[m] for e in evs if e.get(m) is not None]
            row[m] = sum(vals) / len(vals) if vals else None
        row["parse_failure_rate"] = sum(e["parse_failure_rate"] for e in evs) / len(evs)
        mt.append(row)

        s_ev = scoring.summeval(json.loads((d / "summeval.json").read_text()))
        o = s_ev["overall"]
        srow = {"judge": name(j), "key": j, "type": type_of(j)}
        for m in METRICS:
            srow[m] = o.get(m)
        dims = s_ev.get("per_dimension", {})
        for dim in A.SUMMEVAL_DIMS:
            for key in ("spearman", "cohens_kappa"):
                srow[f"{dim}_{key}"] = dims.get(dim, {}).get(key)
        srow["n_valid"] = o.get("n_valid")
        srow["n_total"] = o.get("n_total")
        se.append(srow)
    return mt, se


def main() -> int:
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    rho = spearman
    js = judges()
    out = A.out_dir()
    print(f"{len(js)} judges, {N_BOOT:,} resamples")

    se_point, se_boot = summeval(js, rho)
    print("SummEval point estimates match the stored evaluations")
    mb_point, mb_boot = mtbench(js, rho)
    print("MT-Bench point estimates match the stored evaluations")

    rows = []
    for j in js:
        s = sorted(v for v in se_boot[j] if v is not None)
        m = sorted(v for v in mb_boot[j] if v is not None)
        rows.append({
            "judge": name(j), "key": j, "type": type_of(j),
            "params_b": ALL_JUDGES.get(j, LADDER_JUDGES.get(j)),
            "summeval_rho": se_point[j],
            "summeval_ci_lo": pct(s, 0.025), "summeval_ci_hi": pct(s, 0.975),
            "mtbench_rho_common": mb_point[j],
            "mtbench_ci_lo": pct(m, 0.025), "mtbench_ci_hi": pct(m, 0.975),
            "n_boot": N_BOOT,
        })
    rows.sort(key=lambda r: -r["summeval_rho"])
    A.write_csv(out / "open_ended_reported_ci.csv", rows)

    pairs = []
    for a, b in combinations(js, 2):
        for bench, point, boot in (("summeval", se_point, se_boot), ("mtbench", mb_point, mb_boot)):
            diffs = sorted(x - y for x, y in zip(boot[a], boot[b]) if x is not None and y is not None)
            lo, hi = pct(diffs, 0.025), pct(diffs, 0.975)
            pairs.append({
                "benchmark": bench, "judge_a": name(a), "judge_b": name(b), "key_a": a, "key_b": b,
                "type_a": type_of(a), "type_b": type_of(b),
                "rho_a": point[a], "rho_b": point[b],
                "diff": point[a] - point[b], "ci_lo": lo, "ci_hi": hi,
                "verdict": "a higher" if lo > 0 else "b higher" if hi < 0 else "not distinguishable",
            })
    A.write_csv(out / "open_ended_paired.csv", pairs)

    mt, se = five_metric_rows(js)
    A.write_csv(out / "mtbench_common_metrics.csv", mt)
    A.write_csv(out / "summeval_metrics.csv", se)

    for r in rows:
        print(f"  {r['judge']:<26}{r['type']:<7} SummEval {r['summeval_rho']:.3f} "
              f"[{r['summeval_ci_lo']:.3f}, {r['summeval_ci_hi']:.3f}]   MT-Bench "
              f"{r['mtbench_rho_common']:.3f} [{r['mtbench_ci_lo']:.3f}, {r['mtbench_ci_hi']:.3f}]")
    for p in pairs:
        if {p["key_a"], p["key_b"]} == {"phi4-14b", "phi4r-14b"}:
            print(f"  {p['benchmark']}: {p['judge_a']} - {p['judge_b']} = {p['diff']:+.3f} "
                  f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] {p['verdict']}")
    print(f"\nWrote {out}/open_ended_reported_ci.csv, open_ended_paired.csv, "
          f"mtbench_common_metrics.csv, summeval_metrics.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
