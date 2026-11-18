#!/usr/bin/env python3
"""Significance tests and confidence intervals for the judge comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from slmjury.analysis import paths
from slmjury.analysis.accuracy import CORRUPT_FILES, expected_verdict, file_counts, micro
from slmjury.analysis.metrics import spearman
from slmjury.analysis.stats import (
    benjamini_hochberg, cluster_bootstrap_diff, macro_micro, mcnemar,
    spearman_bootstrap_ci, tost_equivalence, wilson_interval,
)
from slmjury.parsers.answer import get_dataset_type
from slmjury.parsers.normalizer import set_strict_labels

from data import (
    ALL_JUDGES, DATASETS, DISPLAY, FAMILY, LARGE_JUDGES, OUT_NAME, SLM_JUDGES, STUDENTS,
    best_budget, budgets_for, build_item_index, is_large, judgement_path, load_all,
    paired_correctness,
)

NOTES = """\
Every closed-ended comparison is paired: two judges are compared on the same
64,824 items (two students, eight datasets), using the aligned correctness
vectors from data.py. The tests are:

  McNemar            exact two-sided test on the discordant items
  cluster bootstrap  95% CI on the accuracy difference, items resampled
                     within each dataset (10,000 resamples)
  TOST               two one-sided tests for equivalence within +-0.5 points
  Benjamini-Hochberg false-discovery-rate q-values over all judge pairs
  Wilson interval    for a single proportion (accuracy, extraction failures)
  Spearman bootstrap 95% CI on a correlation, pairs resampled together

Each judge is used at its better budget (10 or 8,192 tokens). The script has
three parts, each writing complete tables on its own; by default it runs all
three in order.

small-vs-large
  paired_tests.csv              each large judge against the five best small
                                judges: McNemar, bootstrap CI, TOST, micro and
                                macro accuracy
  mtbench_common_reference.csv  MT-Bench correlation per judge, averaged three
                                ways: over every reference model except the
                                judge itself (rho_all_refs), over the one
                                reference that is not also a judge
                                (rho_common_ref, qwen3.5-397b), and with both
                                students pooled into one correlation with a
                                bootstrap CI
  extraction_accounting.csv/.json
                                items whose student answer cannot be
                                extracted, per student and dataset; they are
                                scored Incorrect and stay in the denominator
all-pairs
  paired_tests_all_pairs.csv    every pair of the 22 judges (231 pairs), with
                                BH q-values
  tost_equivalence.csv          the TOST columns of the same pairs
  macro_vs_micro.csv            micro and macro accuracy, Wilson interval and
                                IFR for every judge and budget
  open_ended_ci.csv             bootstrap CIs on every SummEval and MT-Bench
                                correlation (dimensions and students pooled)
open-ended
  summeval_paired.csv           the five best small judges (by closed-ended
                                accuracy) against each large judge on SummEval,
                                paired, with articles resampled
  mtbench_reference_agreement.csv/.json
                                Spearman correlation between the two MT-Bench
                                reference models, per student, with a CI
  reference_free_penalty_by_size.csv/.json
                                accuracy lost without the reference answer,
                                per judge, and its rank correlation with size

Reads (under results/, or $SLMJURY_RESULTS):
  core/judgements/, large_judges/         per-item verdicts (through data.py)
  core/scoring/<judge>/, large_judges/<judge>/
                                          per-item SummEval and MT-Bench scores
  core/scoring_evaluations/<judge>/, large_judges/scoring_evaluations/<judge>/
                                          stored correlation files
  core/mtbench_references/<reference>/<student>.json
                                          reference-model MT-Bench scores
  judging_protocols/referencefree/<judge>/ reference-free verdicts
  core/student_solutions/<student>/<dataset>.json

Writes the tables above to results/analysis/significance/ and its cache to
results/cache/significance/ (see data.py).

examples, from the repository root:
  python experiments/significance/analyze.py                      # all parts
  python experiments/significance/analyze.py --part all-pairs     # one part
"""

EQUIV_MARGIN_POINTS = 0.5
N_BOOT = 10_000
SEED = 20260916

# The small-vs-large part compares each large judge with this many of the best
# small judges.
TOP_SLM_N = 5

# MT-Bench reference models. GPT-OSS-120B is also a judge and is never
# correlated against itself, so only qwen3.5-397b is common to every judge.
REFERENCES = ["qwen3.5-397b", "gpt-oss-120b"]
COMMON_REFERENCE = "qwen3.5-397b"

SUMMEVAL_DIMS = ["coherence", "consistency", "fluency", "relevance"]


def out_dir() -> Path:
    return paths.analysis_dir(OUT_NAME)


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n {text}\n{'=' * 78}")


# ── Paired tests ────────────────────────────────────────────────────────────

def best_rows(vectors) -> dict[str, tuple[int, float]]:
    """judge -> (its best budget, the accuracy there)."""
    out = {}
    for judge in ALL_JUDGES:
        try:
            b = best_budget(vectors, judge)
        except KeyError:
            continue
        out[judge] = (b, vectors[(judge, b)].accuracy())
    return out


def _one_pair(judge_a, judge_b, vectors, best, index, n_boot, margin) -> dict:
    """Every paired statistic for one ordered pair, a minus b."""
    ba, bb = best[judge_a][0], best[judge_b][0]
    va, vb = vectors[(judge_a, ba)], vectors[(judge_b, bb)]
    a, b, clusters = paired_correctness(va, vb, index)

    mc = mcnemar(a, b)
    boot = cluster_bootstrap_diff(a, b, clusters, n_boot=n_boot)

    # Macro accuracy weights every dataset equally; micro lets the two largest
    # datasets (gsm_plus and hellaswag, 41,188 of the 64,824 items) dominate.
    per_ds_a, per_ds_b = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for ai, bi, ds in zip(a, b, clusters):
        per_ds_a[ds][0] += ai
        per_ds_a[ds][1] += 1
        per_ds_b[ds][0] += bi
        per_ds_b[ds][1] += 1
    mm_a = macro_micro(per_ds_a.values())
    mm_b = macro_micro(per_ds_b.values())

    tost = tost_equivalence(a, b, clusters, margin_points=margin, n_boot=n_boot)

    # Equivalence and a significant difference are separate findings, and a
    # pair can show both. TOST rejecting means the difference is smaller than
    # the margin. McNemar rejecting (with a bootstrap CI that excludes zero)
    # means the difference is reliably not zero; on 64,824 paired items even a
    # 0.3-point gap does that. The verdict names which of the two holds.
    significant = mc.p_exact < 0.05 and not boot.contains_zero
    if tost["tost_equivalent"] and not significant:
        verdict = f"equivalent within {margin} pp and not distinguishable"
    elif tost["tost_equivalent"]:
        verdict = f"equivalent within {margin} pp (gap reliably nonzero but smaller)"
    elif not significant:
        verdict = f"not distinguishable (too wide to claim {margin} pp equivalence)"
    elif boot.point > 0:
        verdict = "a better"
    else:
        verdict = "b better"

    row = {
        "judge_a": DISPLAY.get(judge_a, judge_a),
        "judge_b": DISPLAY.get(judge_b, judge_b),
        "key_a": judge_a, "key_b": judge_b,
        "type_a": "Large" if is_large(judge_a) else "SLM",
        "type_b": "Large" if is_large(judge_b) else "SLM",
        "params_a": ALL_JUDGES[judge_a], "params_b": ALL_JUDGES[judge_b],
        "budget_a": ba, "budget_b": bb,
        "acc_a_micro": mm_a["micro"], "acc_b_micro": mm_b["micro"],
        "acc_a_macro": mm_a["macro"], "acc_b_macro": mm_b["macro"],
        "micro_diff_pp": mm_a["micro"] - mm_b["micro"],
        "macro_diff_pp": mm_a["macro"] - mm_b["macro"],
    }
    row.update(mc.as_row())
    row.update(boot.as_row("boot_"))
    row["tost_margin_pp"] = tost["tost_margin_pp"]
    row["tost_equivalent"] = tost["tost_equivalent"]
    row["tost_min_margin_pp"] = tost["tost_min_margin_pp"]
    row["mcnemar_significant"] = significant
    row["verdict"] = verdict
    return row


def paired_tests(vectors, index, full_matrix: bool, n_boot: int, margin: float) -> list[dict]:
    """Each large judge against the best small judges, or every pair of judges."""
    best = best_rows(vectors)
    slm_ranked = sorted(
        (j for j in SLM_JUDGES if j in best), key=lambda j: -best[j][1]
    )
    large_ranked = sorted(
        (j for j in LARGE_JUDGES if j in best), key=lambda j: -best[j][1]
    )

    if full_matrix:
        judges = large_ranked + slm_ranked
        pairs = [(judges[i], judges[j])
                 for i in range(len(judges)) for j in range(i + 1, len(judges))]
    else:
        top = slm_ranked[:TOP_SLM_N]
        pairs = [(lg, slm) for lg in large_ranked for slm in top]

    print(f"  {len(pairs)} pairs x {n_boot:,} bootstrap resamples")
    rows = []
    for k, (ja, jb) in enumerate(pairs, 1):
        rows.append(_one_pair(ja, jb, vectors, best, index, n_boot, margin))
        if k % 25 == 0 or k == len(pairs):
            print(f"    {k}/{len(pairs)}")

    # Benjamini-Hochberg over all the pairs in the table.
    bh = benjamini_hochberg([r["mcnemar_p_exact"] for r in rows])
    for r, q, rej in zip(rows, bh["qvalues"], bh["reject"]):
        r["mcnemar_q_bh"] = q
        r["significant_after_bh"] = rej
    return rows


# ── Open-ended scores ───────────────────────────────────────────────────────

def scoring_dir(judge: str) -> Path:
    """Per-item SummEval and MT-Bench scores of one judge."""
    if judge in SLM_JUDGES:
        return paths.core("scoring") / judge
    return paths.experiment("large_judges") / judge


def eval_dir(judge: str) -> Path:
    """Stored correlation files (``*_eval.json``) of one judge."""
    if judge in SLM_JUDGES:
        return paths.core("scoring_evaluations") / judge
    return paths.experiment("large_judges") / "scoring_evaluations" / judge


def _load_mtbench(judge: str, reference: str) -> list[tuple[float, float]]:
    """(judge score, reference score) pooled over both students, or [].

    The reference model's score is stored under ``oracle_score``.
    """
    d = scoring_dir(judge)
    pairs = []
    for student in STUDENTS:
        fp = d / f"mtbench_{reference}_{student}.json"
        if not fp.exists():
            continue
        for r in json.loads(fp.read_text()):
            js, os_ = r.get("judge_score"), r.get("oracle_score")
            if js is None or os_ is None:
                continue
            pairs.append((float(js), float(os_)))
    return pairs


def _load_summeval(judge: str) -> dict[str, list[tuple[float, float]]]:
    """Per dimension, (judge_score, human_score)."""
    fp = scoring_dir(judge) / "summeval.json"
    if not fp.exists():
        return {}
    out = {dim: [] for dim in SUMMEVAL_DIMS}
    for r in json.loads(fp.read_text()):
        for dim in SUMMEVAL_DIMS:
            js, hs = r.get(f"judge_{dim}"), r.get(f"human_{dim}")
            if js is None or hs is None:
                continue
            out[dim].append((float(js), float(hs)))
    return out


def _mtbench_eval_rhos(judge: str) -> dict[str, dict[str, float]]:
    """{reference: {student: rho}} read from the stored MT-Bench correlation files."""
    d = eval_dir(judge)
    out: dict[str, dict[str, float]] = {}
    if not d.exists():
        return out
    for fp in sorted(d.glob("mtbench_*_eval.json")):
        stem = fp.stem[len("mtbench_"):-len("_eval")]
        for student in STUDENTS:
            if stem.endswith("_" + student):
                reference = stem[: -len("_" + student)]
                rho = json.loads(fp.read_text())["overall"].get("spearman")
                if rho is not None:
                    out.setdefault(reference, {})[student] = float(rho)
                break
    return out


def mtbench_common_reference(n_boot: int) -> list[dict]:
    """MT-Bench correlation per judge, on every reference and on the common one (qwen3.5-397b).

    A judge is never correlated with itself, and GPT-OSS-120B is also a reference model, so it has two eval files
    and every other judge four. The common reference puts all judges on the same data.
    """
    rows = []
    for judge in ALL_JUDGES:
        by_reference = _mtbench_eval_rhos(judge)
        row = {
            "judge": DISPLAY.get(judge, judge), "key": judge,
            "type": "Large" if is_large(judge) else "SLM",
            "params_b": ALL_JUDGES[judge], "family": FAMILY.get(judge, "?"),
            "is_also_a_reference": judge in REFERENCES,
        }

        non_self = {o: v for o, v in by_reference.items() if o != judge}
        flat = [r for v in non_self.values() for r in v.values()]
        row["rho_all_refs"] = sum(flat) / len(flat) if flat else None
        row["n_eval_files_all_refs"] = len(flat)

        common = by_reference.get(COMMON_REFERENCE, {})
        row["rho_common_ref"] = sum(common.values()) / len(common) if common else None
        row["n_eval_files_common"] = len(common)
        for reference in REFERENCES:
            got = by_reference.get(reference, {})
            row[f"rho_{reference}"] = ("self" if reference == judge
                                    else (sum(got.values()) / len(got) if got else None))

        if row["rho_all_refs"] is not None and row["rho_common_ref"] is not None:
            row["rho_common_minus_all"] = row["rho_common_ref"] - row["rho_all_refs"]
        else:
            row["rho_common_minus_all"] = None

        # Pooled estimate and CI, from the per-item scores.
        pairs = _load_mtbench(judge, COMMON_REFERENCE)
        if pairs:
            ci = spearman_bootstrap_ci([p[0] for p in pairs], [p[1] for p in pairs],
                                       n_boot=n_boot)
            row["rho_pooled_common"] = ci["spearman"]
            row["pooled_ci_lo"] = ci["ci_lo"]
            row["pooled_ci_hi"] = ci["ci_hi"]
            row["n_pooled_items"] = ci["n_pairs"]
        else:
            row["rho_pooled_common"] = row["pooled_ci_lo"] = row["pooled_ci_hi"] = None
            row["n_pooled_items"] = 0

        # SummEval is scored against human labels and needs no reference model.
        sf = eval_dir(judge) / "summeval_eval.json"
        row["summeval_rho"] = (float(json.loads(sf.read_text())["overall"]["spearman"])
                               if sf.exists() else None)
        rows.append(row)

    rows.sort(key=lambda r: (r["rho_common_ref"] is None, -(r["rho_common_ref"] or 0)))

    # A judge with no eval files would otherwise appear as a blank row.
    blank = [r["key"] for r in rows
             if r["rho_common_ref"] is None and not r["is_also_a_reference"]]
    if blank:
        raise paths.MissingResults(
            "MT-Bench eval files not found for: " + ", ".join(blank)
            + ". Looked in results/core/scoring_evaluations and "
              "results/large_judges/scoring_evaluations."
        )
    return rows


def open_ended_cis(n_boot: int) -> list[dict]:
    """Bootstrap CIs on every SummEval and MT-Bench correlation."""
    rows = []
    for judge in ALL_JUDGES:
        summ = _load_summeval(judge)
        for dim in SUMMEVAL_DIMS:
            pairs = summ.get(dim, [])
            if not pairs:
                continue
            ci = spearman_bootstrap_ci([p[0] for p in pairs], [p[1] for p in pairs],
                                       n_boot=n_boot)
            rows.append({
                "judge": DISPLAY.get(judge, judge), "key": judge,
                "type": "Large" if is_large(judge) else "SLM",
                "benchmark": "summeval", "reference": "human", "dimension": dim,
                **ci,
            })
        # SummEval overall: the four dimensions pooled into one correlation.
        pooled = [p for dim in SUMMEVAL_DIMS for p in summ.get(dim, [])]
        if pooled:
            ci = spearman_bootstrap_ci([p[0] for p in pooled], [p[1] for p in pooled],
                                       n_boot=n_boot)
            rows.append({
                "judge": DISPLAY.get(judge, judge), "key": judge,
                "type": "Large" if is_large(judge) else "SLM",
                "benchmark": "summeval", "reference": "human", "dimension": "overall",
                **ci,
            })
        for reference in REFERENCES:
            if judge == reference:
                continue
            pairs = _load_mtbench(judge, reference)
            if not pairs:
                continue
            ci = spearman_bootstrap_ci([p[0] for p in pairs], [p[1] for p in pairs],
                                       n_boot=n_boot)
            rows.append({
                "judge": DISPLAY.get(judge, judge), "key": judge,
                "type": "Large" if is_large(judge) else "SLM",
                "benchmark": "mtbench", "reference": reference, "dimension": "overall",
                **ci,
            })
    return rows


# ── Extraction accounting ───────────────────────────────────────────────────

def extraction_accounting() -> tuple[list[dict], dict]:
    """Count the items whose student answer cannot be extracted.

    Such an item's expected verdict is "incorrect" and it is never dropped, so the denominator does not shrink.
    The count depends only on the student solutions, so it comes from one complete judge's files.
    """
    ref = "llama3.1-8b"
    handling = "scored Incorrect; never excluded"
    rows = []
    total = failures = 0
    for student in STUDENTS:
        for dataset in DATASETS:
            fp = judgement_path(ref, student, dataset, 8192)
            if not fp.exists():
                continue
            records = json.loads(fp.read_text())
            dtype = get_dataset_type(dataset)
            n_fail = 0
            for item in records:
                _oracle, resolved = expected_verdict(item, dtype)
                if (not resolved) or resolved == "UNABLE_TO_EXTRACT":
                    n_fail += 1
            rows.append({
                "student": student, "dataset": dataset, "dataset_type": dtype,
                "n_items": len(records), "n_extraction_failures": n_fail,
                "pct_failures": n_fail / len(records) * 100 if records else 0.0,
                "handling": handling,
            })
            total += len(records)
            failures += n_fail

    lo, hi = wilson_interval(failures, total)
    summary = {
        "n_items_total": total,
        "n_extraction_failures": failures,
        "pct_failures": failures / total * 100 if total else 0.0,
        "wilson_ci_lo_pct": lo, "wilson_ci_hi_pct": hi,
        "handling": handling,
        "reference_judge": ref,
    }
    rows.append({
        "student": "ALL", "dataset": "ALL", "dataset_type": "",
        "n_items": total, "n_extraction_failures": failures,
        "pct_failures": summary["pct_failures"],
        "handling": handling,
    })
    return rows, summary


# ── Macro vs micro ──────────────────────────────────────────────────────────

def macro_vs_micro(vectors, index) -> list[dict]:
    rows = []
    offsets = index.offsets
    for (judge, budget), vec in sorted(vectors.items()):
        per_ds = defaultdict(lambda: [0, 0])
        for (student, dataset), (start, end) in offsets.items():
            for i in range(start, end):
                if vec.present[i]:
                    per_ds[dataset][0] += vec.correct[i]
                    per_ds[dataset][1] += 1
        mm = macro_micro(per_ds.values())
        if mm["micro"] is None:
            continue
        lo, hi = wilson_interval(mm["correct"], mm["total"])
        rows.append({
            "judge": DISPLAY.get(judge, judge), "key": judge,
            "type": "Large" if is_large(judge) else "SLM",
            "params_b": ALL_JUDGES[judge], "budget": budget,
            "micro_accuracy": mm["micro"], "macro_accuracy": mm["macro"],
            "macro_minus_micro_pp": mm["macro_minus_micro"],
            "wilson_lo_pct": lo, "wilson_hi_pct": hi,
            "ifr": vec.ifr(),
            "n_items": mm["total"], "n_datasets": mm["n_groups"],
            "complete": vec.complete,
        })
    rows.sort(key=lambda r: -r["micro_accuracy"])
    return rows


# ── Judging without a reference ─────────────────────────────────────────────

def summeval_matrix(judge: str, order: list[tuple]) -> list[list]:
    """[[judge_dim..., human_dim...]] per summary, aligned to ``order``."""
    fp = paths.require(scoring_dir(judge) / "summeval.json", f"SummEval scores of {judge}")
    by_key = {(r["article_id"], r["summary_id"]): r for r in json.loads(fp.read_text())}
    rows = []
    for key in order:
        r = by_key[key]
        rows.append([r.get(f"judge_{d}") for d in SUMMEVAL_DIMS]
                    + [r.get(f"human_{d}") for d in SUMMEVAL_DIMS])
    return rows


def summeval_rho(rows: list[list], idx, rho) -> float | None:
    """SummEval rho as reported: the mean over dimensions of the per-dimension rho."""
    vals = []
    for k in range(len(SUMMEVAL_DIMS)):
        pairs = [(rows[i][k], rows[i][4 + k]) for i in idx
                 if rows[i][k] is not None and rows[i][4 + k] is not None]
        xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
        if len(pairs) > 2 and len(set(xs)) > 1 and len(set(ys)) > 1:
            vals.append(rho(xs, ys))
    return sum(vals) / len(vals) if vals else None


def summeval_paired(vectors, n_boot: int) -> list[dict]:
    """The five best small judges against every large judge on SummEval, paired.

    The five are the best by closed-ended accuracy, not by the metric compared. Resampling is by article: each
    of the 100 articles has 16 system summaries, and summaries of one article are not independent.
    """
    from slmjury.core.scoring_evaluator import evaluate_summeval

    rho = spearman
    best = best_rows(vectors)
    slms = sorted((j for j in SLM_JUDGES if j in best), key=lambda j: -best[j][1])[:TOP_SLM_N]
    larges = [j for j in LARGE_JUDGES if j in best]

    ref = json.loads((scoring_dir(slms[0]) / "summeval.json").read_text())
    order = [(r["article_id"], r["summary_id"]) for r in ref]
    articles = sorted({a for a, _ in order})
    by_article = {a: [i for i, (aa, _) in enumerate(order) if aa == a] for a in articles}

    mats, full = {}, {}
    for j in slms + larges:
        mats[j] = summeval_matrix(j, order)
        full[j] = summeval_rho(mats[j], range(len(order)), rho)
        # The point estimate must match the scoring evaluator before it is resampled.
        stored = evaluate_summeval(json.loads((scoring_dir(j) / "summeval.json").read_text()))
        if abs(full[j] - stored["overall"]["spearman"]) > 1e-3:
            raise ValueError(f"{j}: SummEval rho {full[j]:.4f} does not match "
                             f"evaluate_summeval's {stored['overall']['spearman']:.4f}")

    rng = random.Random(SEED)
    samples = []
    for _ in range(n_boot):
        pick = [articles[rng.randrange(len(articles))] for _ in articles]
        samples.append([i for a in pick for i in by_article[a]])
    boot = {j: [summeval_rho(mats[j], idx, rho) for idx in samples] for j in slms + larges}

    rows = []
    for s_ in slms:
        for l_ in larges:
            diffs = sorted(a - b for a, b in zip(boot[s_], boot[l_])
                           if a is not None and b is not None)
            lo = diffs[int(0.025 * len(diffs))]
            hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
            rows.append({
                "slm": DISPLAY[s_], "slm_key": s_, "slm_params_b": ALL_JUDGES[s_],
                "large": DISPLAY[l_], "large_key": l_, "large_params_b": ALL_JUDGES[l_],
                "rho_slm": full[s_], "rho_large": full[l_],
                "diff": full[s_] - full[l_],
                "ci_lo": lo, "ci_hi": hi,
                "verdict": ("SLM better" if lo > 0 else "large better" if hi < 0
                            else "not distinguishable"),
                "n_articles": len(articles), "n_summaries": len(order), "n_boot": n_boot,
            })
    return rows


def mtbench_reference_agreement(n_boot: int) -> tuple[list[dict], dict]:
    """Spearman correlation between the two MT-Bench reference models, one per student, averaged.

    A judge's correlation with a reference carries that reference's noise, so this agreement shows how
    reliable the reference is. It is computed the way judge rho is.
    """

    root = paths.core("mtbench_references")
    per_student, rows = {}, []
    for student in STUDENTS:
        a = json.loads((root / "gpt-oss-120b" / f"{student}.json").read_text())
        b = json.loads((root / COMMON_REFERENCE / f"{student}.json").read_text())
        if len(a) != len(b) or any(x["turn1_response"] != y["turn1_response"]
                                   for x, y in zip(a, b)):
            raise ValueError(f"{student}: the two reference files are not aligned")
        # Each reference model's score is stored under ``oracle_score``.
        pairs = [(x["oracle_score"], y["oracle_score"]) for x, y in zip(a, b)
                 if x["oracle_score"] is not None and y["oracle_score"] is not None]
        per_student[student] = pairs
        rows.append({"student": student, "n_conversations": len(pairs),
                     "rho_reference_vs_reference": spearman([p for p, _ in pairs],
                                                            [q for _, q in pairs])})

    rng = random.Random(SEED)
    means = []
    for _ in range(n_boot):
        vals = []
        for pairs in per_student.values():
            smp = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            xs, ys = [p for p, _ in smp], [q for _, q in smp]
            if len(set(xs)) > 1 and len(set(ys)) > 1:
                vals.append(spearman(xs, ys))
        if len(vals) == len(per_student):
            means.append(sum(vals) / len(vals))
    means.sort()
    summary = {
        "rho_reference_vs_reference_mean": sum(r["rho_reference_vs_reference"] for r in rows) / len(rows),
        "ci_lo": means[int(0.025 * len(means))],
        "ci_hi": means[min(len(means) - 1, int(0.975 * len(means)))],
        "n_boot": len(means),
        "references": ["gpt-oss-120b", COMMON_REFERENCE],
    }
    return rows, summary


def reference_free_penalty_by_size(vectors) -> tuple[list[dict], dict]:
    """Accuracy lost without the reference answer, per judge, against judge size.

    Each protocol is at its best budget. Reference-based accuracy comes from the aligned vectors, reference-free
    accuracy from results/judging_protocols/referencefree, for judges with all sixteen files at a budget.
    """
    rho = spearman
    rf_root = paths.experiment("judging_protocols") / "referencefree"
    best = best_rows(vectors)
    rows = []
    for judge in ALL_JUDGES:
        if judge not in best:
            continue
        rf = {}
        for b in budgets_for(judge):
            counts = []
            for student in STUDENTS:
                for dataset in DATASETS:
                    path = rf_root / judge / f"{student}_{dataset}_t{b}.json"
                    counts.append(file_counts(judge, student, dataset, b, path))
            if all(c is not None for c in counts):
                rf[b] = micro(counts)["accuracy"]
        if not rf:
            continue
        rb_budget, rb_acc = best[judge]
        rf_budget = max(rf, key=rf.get)
        rows.append({
            "judge": DISPLAY[judge], "key": judge,
            "type": "Large" if is_large(judge) else "SLM",
            "family": FAMILY.get(judge, "?"), "params_b": ALL_JUDGES[judge],
            "ref_based_best": rb_acc, "ref_based_budget": rb_budget,
            "ref_free_best": rf[rf_budget], "ref_free_budget": rf_budget,
            "penalty_pp": rb_acc - rf[rf_budget],
        })

    def corr(sub):
        if len(sub) < 4:
            return None
        return rho([r["params_b"] for r in sub], [r["penalty_pp"] for r in sub])

    # The share of items whose expected verdict is Correct: the score of a
    # judge that answers "Correct" to everything. Reference-free accuracy close
    # to it means the judge has mostly stopped telling answers apart.
    solutions = paths.core("student_solutions")
    n = c = 0
    for student in STUDENTS:
        for dataset in DATASETS:
            recs = json.loads((solutions / student / f"{dataset}.json").read_text())
            dtype = get_dataset_type(dataset)
            c += sum(1 for r in recs if expected_verdict(r, dtype)[0] == "correct")
            n += len(recs)
    always_correct = c / n * 100

    slm = [r for r in rows if r["type"] == "SLM"]
    summary = {
        "n_judges": len(rows),
        "spearman_params_vs_penalty_slm": corr(slm),
        "spearman_params_vs_penalty_slm_excl_llama3.2-1b":
            corr([r for r in slm if r["key"] != "llama3.2-1b"]),
        "spearman_params_vs_penalty_all": corr(rows),
        "always_correct_baseline_pct": always_correct,
        "note": ("llama3.2-1b is shown separately because its reference-based score "
                 "reflects failing to follow the instructions (IFR 94% at B=10), "
                 "not judging ability"),
    }
    rows.sort(key=lambda r: (r["family"], r["params_b"]))
    return rows, summary


# ── Parts ───────────────────────────────────────────────────────────────────

def _print_large_vs_best_slm(rows: list[dict]) -> None:
    if not rows:
        return
    _banner("Large judges against the best small judge, same items")
    hdr = (f"{'large judge':<26}{'vs SLM':<22}{'micro':>9}{'macro':>9}"
           f"{'McNemar p':>11}{'95% CI (points)':>22}  verdict")
    print(hdr)
    print("-" * len(hdr))
    seen = set()
    for r in rows:
        if r["type_a"] != "Large" or r["key_a"] in seen:
            continue
        seen.add(r["key_a"])
        ci = f"[{r['boot_ci_lo_pp']:+.3f}, {r['boot_ci_hi_pp']:+.3f}]"
        print(f"{r['judge_a']:<26}{r['judge_b']:<22}"
              f"{r['micro_diff_pp']:>+9.3f}{r['macro_diff_pp']:>+9.3f}"
              f"{r['mcnemar_p_exact']:>11.4f}{ci:>22}  {r['verdict']}")
    print("\nDifferences are large minus small, in accuracy points. A CI that "
          "contains zero\nmeans the two judges cannot be told apart on these items.")


def part_small_vs_large(vectors, index, args) -> list[Path]:
    out, written = out_dir(), []

    _banner("Paired tests, large judges x the best small judges")
    rows = paired_tests(vectors, index, full_matrix=False,
                        n_boot=args.n_boot, margin=args.margin_points)
    written.append(write_csv(out / "paired_tests.csv", rows))
    _print_large_vs_best_slm(rows)

    _banner("MT-Bench on the reference every judge shares")
    mt = mtbench_common_reference(args.n_boot)
    written.append(write_csv(out / "mtbench_common_reference.csv", mt))
    print(f"{'judge':<26}{'type':<7}{'all refs':>9}{'common ref':>12}"
          f"{'shift':>9}{'pooled rho [95% CI]':>28}{'SummEval':>10}")
    print("-" * 96)
    for r in mt[:10]:
        if r["rho_common_ref"] is None:
            continue
        ci = (f"{r['rho_pooled_common']:.3f} [{r['pooled_ci_lo']:.3f}, "
              f"{r['pooled_ci_hi']:.3f}]" if r["pooled_ci_lo"] is not None else "-")
        se = f"{r['summeval_rho']:.4f}" if r["summeval_rho"] is not None else "-"
        print(f"{r['judge']:<26}{r['type']:<7}{r['rho_all_refs']:>9.4f}"
              f"{r['rho_common_ref']:>12.4f}{r['rho_common_minus_all']:>+9.4f}{ci:>28}{se:>10}")
    best_slm = next(r for r in mt if r["type"] == "SLM" and r["rho_common_ref"] is not None)
    best_large = next(r for r in mt if r["type"] == "Large" and r["rho_common_ref"] is not None)
    print(f"\n  best small judge on the common reference: {best_slm['judge']} "
          f"{best_slm['rho_common_ref']:.4f}")
    print(f"  best large judge on the common reference: {best_large['judge']} "
          f"{best_large['rho_common_ref']:.4f}")

    _banner("Extraction accounting")
    ex_rows, summary = extraction_accounting()
    written.append(write_csv(out / "extraction_accounting.csv", ex_rows))
    (out / "extraction_accounting.json").write_text(json.dumps(summary, indent=2))
    written.append(out / "extraction_accounting.json")
    print(f"  {summary['n_extraction_failures']} of {summary['n_items_total']} "
          f"items have no extractable student answer ({summary['pct_failures']:.4f}%), "
          f"95% CI [{summary['wilson_ci_lo_pct']:.4f}%, {summary['wilson_ci_hi_pct']:.4f}%]")
    print(f"  handling: {summary['handling']}")
    return written


def part_all_pairs(vectors, index, args) -> list[Path]:
    out, written = out_dir(), []

    _banner("Every pair of judges, with BH correction")
    rows = paired_tests(vectors, index, full_matrix=True,
                        n_boot=args.n_boot, margin=args.margin_points)
    written.append(write_csv(out / "paired_tests_all_pairs.csv", rows))
    n_sig = sum(1 for r in rows if r["significant_after_bh"])
    n_equiv = sum(1 for r in rows if r["tost_equivalent"])
    print(f"  {len(rows)} pairs: {n_sig} significant after BH, "
          f"{n_equiv} equivalent at +-{args.margin_points} points")

    _banner(f"TOST at +-{args.margin_points} points")
    tost_rows = [{
        "judge_a": r["judge_a"], "judge_b": r["judge_b"],
        "key_a": r["key_a"], "key_b": r["key_b"],
        "type_a": r["type_a"], "type_b": r["type_b"],
        "diff_pp": r["micro_diff_pp"],
        "tost_margin_pp": r["tost_margin_pp"],
        "tost_equivalent": r["tost_equivalent"],
        "tost_min_margin_pp": r["tost_min_margin_pp"],
        "mcnemar_p_exact": r["mcnemar_p_exact"],
        "mcnemar_q_bh": r["mcnemar_q_bh"],
        "verdict": r["verdict"],
    } for r in rows]
    written.append(write_csv(out / "tost_equivalence.csv", tost_rows))

    _banner("Macro vs micro, every judge")
    mm = macro_vs_micro(vectors, index)
    written.append(write_csv(out / "macro_vs_micro.csv", mm))
    biggest = max(mm, key=lambda r: abs(r["macro_minus_micro_pp"]))
    print(f"  largest macro-micro gap: {biggest['judge']} t{biggest['budget']} "
          f"{biggest['macro_minus_micro_pp']:+.3f} points")

    _banner("Bootstrap CIs on every open-ended correlation")
    oe = open_ended_cis(args.n_boot)
    written.append(write_csv(out / "open_ended_ci.csv", oe))
    print(f"  {len(oe)} correlations with CIs")
    return written


def part_open_ended(vectors, index, args) -> list[Path]:
    out, written = out_dir(), []
    n3 = min(args.n_boot, 2000)   # each article resample recomputes four correlations per judge

    _banner("SummEval, small vs large judges, articles resampled")
    sp = summeval_paired(vectors, n3)
    written.append(write_csv(out / "summeval_paired.csv", sp))
    print(f"{'SLM (by closed-ended rank)':<24}{'vs large':<26}{'rho SLM':>9}{'rho large':>11}"
          f"{'diff':>9}{'95% CI':>20}  verdict")
    for r in sp:
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(f"{r['slm']:<24}{r['large']:<26}{r['rho_slm']:>9.3f}{r['rho_large']:>11.3f}"
              f"{r['diff']:>+9.3f}{ci:>20}  {r['verdict']}")

    _banner("Agreement between the two MT-Bench reference models")
    mt_rows, mt_sum = mtbench_reference_agreement(n3)
    written.append(write_csv(out / "mtbench_reference_agreement.csv", mt_rows))
    (out / "mtbench_reference_agreement.json").write_text(json.dumps(mt_sum, indent=2))
    written.append(out / "mtbench_reference_agreement.json")
    for r in mt_rows:
        print(f"  {r['student']:<14} gpt-oss-120b vs {COMMON_REFERENCE}: "
              f"rho = {r['rho_reference_vs_reference']:.4f}")
    print(f"  mean {mt_sum['rho_reference_vs_reference_mean']:.4f}  "
          f"95% CI [{mt_sum['ci_lo']:.3f}, {mt_sum['ci_hi']:.3f}]")

    _banner("Reference-free accuracy loss against judge size")
    pen_rows, pen_sum = reference_free_penalty_by_size(vectors)
    written.append(write_csv(out / "reference_free_penalty_by_size.csv", pen_rows))
    (out / "reference_free_penalty_by_size.json").write_text(json.dumps(pen_sum, indent=2))
    written.append(out / "reference_free_penalty_by_size.json")
    fam = None
    for r in pen_rows:
        if r["family"] != fam:
            fam = r["family"]
            print(f"  {fam}")
        print(f"    {r['judge']:<26}{r['params_b']:>6}B  {r['ref_based_best']:>6.2f} -> "
              f"{r['ref_free_best']:>6.2f}   drop {r['penalty_pp']:>6.2f} points")
    def three(v):
        return "n/a" if v is None else f"{v:.3f}"
    print(f"\n  Spearman(size, drop), small judges: {three(pen_sum['spearman_params_vs_penalty_slm'])}"
          f"   excl. Llama-3.2-1B: {three(pen_sum['spearman_params_vs_penalty_slm_excl_llama3.2-1b'])}")
    print(f"  a judge answering 'Correct' to everything scores "
          f"{pen_sum['always_correct_baseline_pct']:.2f}%")
    return written


PARTS = {
    "small-vs-large": part_small_vs_large,
    "all-pairs": part_all_pairs,
    "open-ended": part_open_ended,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Significance tests for the judge comparisons.", epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--part", choices=list(PARTS), default=None,
                    help="run one part (default: all three, in order)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help=f"bootstrap resamples (default {N_BOOT})")
    ap.add_argument("--margin-points", type=float, default=EQUIV_MARGIN_POINTS,
                    help="TOST equivalence margin, in accuracy points (default 0.5)")
    # Older spelling, kept working but not shown in --help.
    ap.add_argument("--margin-pp", dest="margin_points", type=float,
                    default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore the cached item index and vectors")
    args = ap.parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table

    index = build_item_index(force=args.rebuild)

    _banner("Loading aligned per-item correctness vectors")
    vectors = load_all(index=index, rebuild=args.rebuild)
    incomplete = [f"{j} t{b}" for (j, b), v in vectors.items() if not v.complete]
    if incomplete:
        print(f"\n  incomplete vectors (excluded from best-budget choice): "
              f"{', '.join(incomplete)}")

    written = []
    for part in ([args.part] if args.part else list(PARTS)):
        written += PARTS[part](vectors, index, args)

    _banner("Wrote")
    for p in written:
        print(f"  {p}")
    if CORRUPT_FILES:
        print(f"\n!! {len(CORRUPT_FILES)} unreadable result file(s) skipped:")
        for p in CORRUPT_FILES:
            print(f"   {p}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except paths.MissingResults as e:
        print(f"\nMissing input:\n  {e}", file=sys.stderr)
        sys.exit(2)
