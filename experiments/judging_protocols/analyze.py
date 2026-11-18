#!/usr/bin/env python3
"""Judging protocols: do the verdicts hold up when the protocol changes?"""

import argparse
import csv
import json
import re
from pathlib import Path

from slmjury.analysis import paths
from slmjury.analysis.accuracy import CORRUPT_FILES, file_counts, micro
from slmjury.analysis.metrics import position_bias_flip_rate
from slmjury.analysis.pairwise import read_choice
from slmjury.parsers.normalizer import set_strict_labels

NOTES = """\
Compares four ways of asking a judge about the same student answers on the
eight closed-ended datasets:

  reference_based   the package's own prompts (results/core/judgements, and
                    results/large_judges for the large judges)
  reference_free    the ground truth is left out of the prompt
  geval             a 1-10 score thresholded at 7; 8,192 tokens only
  pairwise          two students' answers side by side, in both orders

Only reference_free removes the ground truth. G-Eval and pairwise still show
it and change the form of the answer.

For the first three, accuracy and instruction-following rate are counted from
the per-item judgement files with slmjury.analysis.accuracy and micro-averaged
over students and datasets. The reference-based row is counted here too, from
the same files as the main results, so every protocol is scored by the same
code.

For pairwise, every stored response is read again with
slmjury.analysis.pairwise.read_choice (the stored verdict field is not used),
and the flip rate is the share of items whose pick changes with the answer
order, among items with an A or B in both orders. 0.5 is what random picks
give. budget_invariant marks cells whose forward choices are the same at 10
and 8,192 tokens: the prompt asks for one letter, so a judge that does not
think gives the same answer at both budgets, and the two cells are one
measurement.

Reads (under results/, or $SLMJURY_RESULTS):
  judging_protocols/referencefree/<judge>/<student>_<dataset>_t<budget>.json
  judging_protocols/geval/<judge>/<student>_<dataset>_t<budget>.json
  judging_protocols/pairwise/<judge>/<a>_vs_<b>_<dataset>_t<budget>_{fwd,rev}.json
  core/judgements/<judge>/<student>_<dataset>_t<budget>.json
  large_judges/<judge>/<student>_<dataset>_t<budget>.json    (if present)

Writes results/analysis/judging_protocols/:
  protocol_comparison.csv    judge x protocol: accuracy and IFR at each budget
  protocol_per_dataset.csv   judge x protocol x budget x dataset
  pairwise_flip_rate.csv     judge x pair x dataset x budget: flip rate and parse rate
  pairwise_summary.csv       judge x budget: mean, min and max flip rate

The three protocol runners in this folder judged the two main students
(qwen2.5-32b, llama3.1-8b) on all eight datasets, for the sixteen small judges
and for qwen2.5-72b and gpt-oss-120b; docs/REPRODUCE.md has the commands. The
reference-based rows need the original judgements (scripts/run_judge.py) for
the small judges and experiments/large_judges for the large ones.

Run it from the repository root.
"""

OUT_NAME = "judging_protocols"

STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]
ALL_DATASETS = [
    "gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
    "hellaswag", "winogrande", "truthfulqa",
]
TOKEN_SETTINGS = [10, 8192]
DS_PATTERN = "|".join(ALL_DATASETS)
PAIRWISE_RE = re.compile(rf"^(?P<a>.+?)_vs_(?P<b>.+?)_(?P<dataset>{DS_PATTERN})"
                         rf"_t(?P<tok>\d+)_fwd\.json$")


def protocols_dir() -> Path:
    return paths.experiment("judging_protocols")


def _protocol_file(sub: str, judge: str, name: str) -> Path:
    return protocols_dir() / sub / judge / name


def _reference_file(judge: str, name: str) -> Path:
    """The judge's reference-based file: the original runs first, then large_judges."""
    slm = paths.core("judgements") / judge / name
    if slm.exists():
        return slm
    return paths.results_dir() / "large_judges" / judge / name


# G-Eval scores on a rubric and so needs room to reason: 8,192 tokens only.
PROTOCOLS = {
    "reference_based": (lambda j, s, d, t: _reference_file(j, f"{s}_{d}_t{t}.json"),
                        TOKEN_SETTINGS),
    "reference_free": (lambda j, s, d, t: _protocol_file("referencefree", j,
                                                         f"{s}_{d}_t{t}.json"),
                       TOKEN_SETTINGS),
    "geval": (lambda j, s, d, t: _protocol_file("geval", j, f"{s}_{d}_t{t}.json"),
              [8192]),
}


def _judges(sub: str) -> list[str]:
    """Judges with a folder under judging_protocols/<sub>/."""
    d = protocols_dir() / sub
    if not d.is_dir():
        return []
    return sorted(x.name for x in d.iterdir() if x.is_dir())


def protocol_accuracy(judge: str, path_fn, max_tokens: int) -> dict:
    """Overall and per-dataset accuracy for one judge, protocol and budget."""
    per_dataset, all_counts = {}, []
    for dataset in ALL_DATASETS:
        ds_counts = [
            file_counts(judge, student, dataset, max_tokens,
                        path_fn(judge, student, dataset, max_tokens))
            for student in STUDENTS
        ]
        m = micro(ds_counts)
        if m:
            per_dataset[dataset] = m
            all_counts.extend(c for c in ds_counts if c is not None)
    return {"overall": micro(all_counts), "by_dataset": per_dataset}


def analyze_pairwise(judge: str) -> list[dict]:
    """Flip rate per (pair, dataset, budget) from matching fwd/rev files."""
    fwd_files = sorted((protocols_dir() / "pairwise" / judge).glob("*_fwd.json"))
    rows = []
    for fwd_path in fwd_files:
        m = PAIRWISE_RE.match(fwd_path.name)
        if not m:
            continue
        rev_path = fwd_path.with_name(fwd_path.name.replace("_fwd.json", "_rev.json"))
        if not rev_path.exists():
            continue
        with open(fwd_path) as f:
            fwd = json.load(f)
        with open(rev_path) as f:
            rev = json.load(f)
        if len(fwd) != len(rev):
            # The order swap is compared item by item.
            raise ValueError(f"{fwd_path} has {len(fwd)} items, {rev_path} has {len(rev)}")
        fwd = [{"choice": read_choice(x.get("judge_response", ""))} for x in fwd]
        rev = [{"choice": read_choice(x.get("judge_response", ""))} for x in rev]
        flip = position_bias_flip_rate(fwd, rev, verdict_key="choice")
        parsed = sum(x["choice"] != "Undefined" for x in fwd + rev) / (2 * len(fwd))

        # Are the forward choices the same at the other budget?
        other_tok = 8192 if int(m["tok"]) == 10 else 10
        twin = fwd_path.with_name(
            fwd_path.name.replace(f"_t{m['tok']}_", f"_t{other_tok}_")
        )
        budget_invariant = None
        if twin.exists():
            with open(twin) as f:
                twin_fwd = json.load(f)
            budget_invariant = (
                [x["choice"] for x in fwd]
                == [read_choice(x.get("judge_response", "")) for x in twin_fwd]
            )

        rows.append({
            "judge": judge,
            "pair": f"{m['a']}_vs_{m['b']}",
            "dataset": m["dataset"],
            "max_tokens": int(m["tok"]),
            "flip_rate": round(flip, 6) if flip == flip else None,
            "n": len(fwd),
            "parse_rate": round(parsed, 6),
            "budget_invariant": budget_invariant,
        })
    return rows


def main():
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table
    out_dir = paths.analysis_dir(OUT_NAME)

    pointwise_judges = sorted(set(_judges("referencefree")) | set(_judges("geval")))
    pairwise_judges = sorted(_judges("pairwise"))
    print(f"Pointwise judges: {', '.join(pointwise_judges)}")
    print(f"Pairwise judges:  {', '.join(pairwise_judges)}\n")

    # ── 1. Accuracy per protocol ───────────────────────────────────────────
    comp_rows, per_ds_rows = [], []
    print("── Cross-protocol accuracy (8 datasets x 2 students) ──")
    print(f"  {'Judge':<14}{'Protocol':<18}{'Acc@t10':>10}{'IFR@t10':>10}"
          f"{'Acc@t8192':>11}{'IFR@t8192':>11}{'ds':>4}")
    print("  " + "-" * 78)
    for judge in pointwise_judges:
        for proto, (path_fn, budgets) in PROTOCOLS.items():
            res = {t: protocol_accuracy(judge, path_fn, t) for t in budgets}
            got = {t: r for t, r in res.items() if r["overall"]}
            if not got:
                continue
            any_res = next(iter(got.values()))

            def cell(t, key):
                return got[t]["overall"][key] if t in got else None

            def show(t, key, width):
                v = cell(t, key)
                text = f"{v:.2f}" if v is not None else "—"
                return f"{text:>{width}}"

            print(f"  {judge:<14}{proto:<18}"
                  + show(10, "accuracy", 10) + show(10, "ifr", 10)
                  + show(8192, "accuracy", 11) + show(8192, "ifr", 11)
                  + f"{len(any_res['by_dataset']):>4}")

            comp_rows.append({
                "judge": judge, "protocol": proto,
                "acc_t10": cell(10, "accuracy"), "ifr_t10": cell(10, "ifr"),
                "acc_t8192": cell(8192, "accuracy"), "ifr_t8192": cell(8192, "ifr"),
                "n_datasets": len(any_res["by_dataset"]),
                "n_items": any_res["overall"]["total"],
            })
            for t, r in got.items():
                for dataset, m in r["by_dataset"].items():
                    per_ds_rows.append({
                        "judge": judge, "protocol": proto, "max_tokens": t,
                        "dataset": dataset,
                        "accuracy": round(m["accuracy"], 4),
                        "ifr": round(m["ifr"], 4), "n_items": m["total"],
                    })
        print()

    comp_csv = out_dir / "protocol_comparison.csv"
    with open(comp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comp_rows)
    print(f"Saved: {comp_csv}")

    per_ds_csv = out_dir / "protocol_per_dataset.csv"
    with open(per_ds_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_ds_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_ds_rows)
    print(f"Saved: {per_ds_csv}\n")

    # ── 2. Pairwise position bias ──────────────────────────────────────────
    pw_rows = []
    for judge in pairwise_judges:
        pw_rows.extend(analyze_pairwise(judge))

    print("── Pairwise position-bias flip rate (0.5 = random, 1.0 = pure "
          "position bias) ──")
    print(f"  {'Judge':<14}{'tok':>6}{'cells':>7}{'mean':>8}{'min':>8}{'max':>8}")
    print("  " + "-" * 51)
    summary_rows = []
    for judge in pairwise_judges:
        for tok in TOKEN_SETTINGS:
            vals = [r["flip_rate"] for r in pw_rows
                    if r["judge"] == judge and r["max_tokens"] == tok
                    and r["flip_rate"] is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            print(f"  {judge:<14}{tok:>6}{len(vals):>7}{mean:>8.3f}"
                  f"{min(vals):>8.3f}{max(vals):>8.3f}")
            summary_rows.append({
                "judge": judge, "max_tokens": tok, "n_cells": len(vals),
                "mean_flip_rate": round(mean, 6),
                "min_flip_rate": round(min(vals), 6),
                "max_flip_rate": round(max(vals), 6),
                "mean_parse_rate": round(sum(
                    r["parse_rate"] * r["n"] for r in pw_rows
                    if r["judge"] == judge and r["max_tokens"] == tok) / sum(
                    r["n"] for r in pw_rows
                    if r["judge"] == judge and r["max_tokens"] == tok), 6),
                "min_dataset": min(
                    (r for r in pw_rows if r["judge"] == judge
                     and r["max_tokens"] == tok and r["flip_rate"] is not None),
                    key=lambda r: r["flip_rate"])["dataset"],
                "max_dataset": max(
                    (r for r in pw_rows if r["judge"] == judge
                     and r["max_tokens"] == tok and r["flip_rate"] is not None),
                    key=lambda r: r["flip_rate"])["dataset"],
            })
        # Both budgets together
        vals = [r["flip_rate"] for r in pw_rows
                if r["judge"] == judge and r["flip_rate"] is not None]
        if vals:
            summary_rows.append({
                "judge": judge, "max_tokens": "both", "n_cells": len(vals),
                "mean_flip_rate": round(sum(vals) / len(vals), 6),
                "min_flip_rate": round(min(vals), 6),
                "max_flip_rate": round(max(vals), 6),
                "mean_parse_rate": round(sum(
                    r["parse_rate"] * r["n"] for r in pw_rows if r["judge"] == judge) / sum(
                    r["n"] for r in pw_rows if r["judge"] == judge), 6),
                "min_dataset": "", "max_dataset": "",
            })

    invariant = sorted({r["judge"] for r in pw_rows if r["budget_invariant"]})
    if invariant:
        print(f"\n  Choices are identical at t10 and t8192 for "
              f"{', '.join(invariant)}:")
        print("  the pairwise prompt asks for one letter, so a deterministic judge")
        print("  gives the same answer at both budgets. Report one flip rate for these.")

    pw_csv = out_dir / "pairwise_flip_rate.csv"
    with open(pw_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["judge", "pair", "dataset", "max_tokens", "flip_rate",
                           "n", "parse_rate", "budget_invariant"],
        )
        writer.writeheader()
        writer.writerows(pw_rows)
    print(f"\nSaved: {pw_csv}")

    sum_csv = out_dir / "pairwise_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {sum_csv}")

    if CORRUPT_FILES:
        print(f"\n{len(CORRUPT_FILES)} unreadable result file(s) skipped:")
        for p in CORRUPT_FILES:
            print(f"   {p}")


if __name__ == "__main__":
    main()
