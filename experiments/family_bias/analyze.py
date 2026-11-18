#!/usr/bin/env python3
"""Family bias: does a judge grade students from its own model family more leniently?"""

import argparse
import csv
import json
from pathlib import Path

from slmjury.analysis import paths
from slmjury.analysis.accuracy import expected_verdict, verdict_pairs
from slmjury.parsers.answer import get_dataset_type
from slmjury.parsers.normalizer import set_strict_labels

NOTES = """\
Every judge grades a panel of students from several model families. For each
judge and student the script computes

  FAR  = P(judge says Correct   | expected verdict is Incorrect)   leniency
  FRR  = P(judge says Incorrect | expected verdict is Correct)     strictness

and accuracy. A judge family that favours its own lineage would show a higher
FAR on same-family students than on the others. The expected verdict for each
item comes from that item's own record in the judgement file, so nothing is
joined by position across files.

Panels. Coverage is uneven, so the results are reported as five panels. Each
one is a complete rectangle: every judge in it has a judgement for every
student, dataset and budget the panel reports, and no cell is imputed.

  panel                            judges x students x datasets x budgets
  eight_datasets_non_phi4_judges     11   x  5 (+1) x    8     x   2
  eight_datasets                     13   x  5      x    8     x   2
  eight_datasets_all_judges          16   x  5      x    8     x   1
  five_datasets                      13   x  7      x    5     x   2
  five_datasets_all_judges           16   x  7      x    5     x   1

eight_datasets is the largest complete rectangle at both budgets, and
five_datasets holds every student. The two *_all_judges panels add the three
Phi-4 reasoning judges, at 8,192 tokens only. eight_datasets_non_phi4_judges
keeps the eleven judges outside the Phi-4 family, the only judges that graded
qwen3-14b-student (the +1).

Two limits set these shapes. Three Phi-4 reasoning judges always think before
answering and have no 10-token configuration, so a panel with both budgets
holds 13 of the 16 judges. Two students (glm-4.7-flash, qwen3.5-35b-student)
were generated on 5 datasets rather than 8, so an 8-dataset panel holds 5 of
the 7 students.

qwen3-14b-student is a strong Qwen3 student, added because the other Qwen3
students are weak and lineage could not be separated from answer quality. It
is marked with origin "probe" and joins a panel only when every cell of that
panel exists for it; with the current results that is
eight_datasets_non_phi4_judges. A student that has been judged only in part
is listed on the console instead.

Reads (under results/, or $SLMJURY_RESULTS):
  family_bias/judgements/<judge>/<student>_<dataset>_t<budget>.json
  core/judgements/<judge>/<student>_<dataset>_t<budget>.json         (qwen2.5-32b, llama3.1-8b)
  family_bias/student_solutions/<student>/<dataset>.json
  core/student_solutions/<student>/<dataset>.json                    (qwen2.5-32b, llama3.1-8b)

Writes results/analysis/family_bias/, for each panel <p>:
  <p>_false_accept_rate_matrix_t<budget>.csv   FAR per judge x student
  <p>_false_reject_rate_matrix_t<budget>.csv   FRR per judge x student
  <p>_accuracy_matrix_t<budget>.csv            accuracy per judge x student
  <p>_family_summary.csv                       mean FAR on same-family vs. other students
  <p>_student_solver_accuracy.csv              how often each student solved the item itself
  <p>_variance_decomposition.csv               spread across students vs. across judges

experiments/family_bias/student_ordering.py reads the eight_datasets and
five_datasets tables. docs/REPRODUCE.md lists the runs behind them (students
and judgements). Run it from the repository root.
"""

OUT_NAME = "family_bias"

EIGHT_DATASETS = [
    "gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
    "hellaswag", "winogrande", "truthfulqa",
]
FIVE_DATASETS = ["arc_easy", "arc_challenge", "gsm8k", "winogrande", "truthfulqa"]
TOKEN_SETTINGS = [10, 8192]

# origin: "main" for the two main students, "new" for the wider panel, and
# "probe" for strong same-family students that join a panel only once complete.
STUDENT_INFO = {
    "qwen2.5-32b": {"family": "Qwen2.5", "origin": "main"},
    "llama3.1-8b": {"family": "Llama", "origin": "main"},
    "deepseek-r1-0528-qwen3-8b": {"family": "Qwen3", "origin": "new"},
    "mistral-7b-v0.3": {"family": "Mistral", "origin": "new"},
    "gemma4-31b-student": {"family": "Gemma", "origin": "new"},
    "glm-4.7-flash": {"family": "GLM", "origin": "new"},
    "qwen3.5-35b-student": {"family": "Qwen3", "origin": "new"},
    "qwen3-14b-student": {"family": "Qwen3", "origin": "probe"},
}

PROBE_STUDENTS = [k for k, v in STUDENT_INFO.items() if v["origin"] == "probe"]

JUDGE_FAMILY = {
    "qwen2.5-1.5b": "Qwen2.5", "qwen2.5-3b": "Qwen2.5", "qwen2.5-7b": "Qwen2.5",
    "qwen3-0.6b": "Qwen3", "qwen3-1.7b": "Qwen3", "qwen3-4b": "Qwen3",
    "qwen3-8b": "Qwen3", "qwen3-14b": "Qwen3",
    "llama3.2-1b": "Llama", "llama3.2-3b": "Llama", "llama3.1-8b": "Llama",
    "phi4-14b": "Phi-4", "phi4mi-3.8b": "Phi-4",
}
# Judges with no 10-token configuration (they always reason first).
REASONING_ONLY_FAMILY = {"phi4r-14b": "Phi-4", "phi4rp-14b": "Phi-4",
                         "phi4mr-3.8b": "Phi-4"}
ALL_JUDGE_FAMILY = {**JUDGE_FAMILY, **REASONING_ONLY_FAMILY}

DUAL_BUDGET_JUDGES = list(JUDGE_FAMILY)            # 13: both budgets
ALL_JUDGES = list(ALL_JUDGE_FAMILY)                # 16: every judge

# Students whose solutions cover all 8 datasets.
EIGHT_DATASET_STUDENTS = ["qwen2.5-32b", "llama3.1-8b", "deepseek-r1-0528-qwen3-8b",
                  "mistral-7b-v0.3", "gemma4-31b-student"]

# Every student except the probes; probes are added per panel by with_probes().
ALL_STUDENTS = [s for s in STUDENT_INFO if s not in PROBE_STUDENTS]

# The two weakest judges, left out of the "capable judges" pool.
WEAK = {"llama3.2-1b", "qwen3-0.6b"}

# The 11 judges outside the Phi-4 family.
NON_PHI_JUDGES = [j for j in DUAL_BUDGET_JUDGES if JUDGE_FAMILY[j] != "Phi-4"]

PANELS = {
    "eight_datasets_non_phi4_judges": {
        "judges": NON_PHI_JUDGES, "students": EIGHT_DATASET_STUDENTS,
        "datasets": EIGHT_DATASETS, "weak_judges": WEAK,
        "note": "11 judges x 5 students x 8 datasets x 2 budgets "
                "-- judges outside the Phi-4 family",
    },
    "eight_datasets": {
        "judges": DUAL_BUDGET_JUDGES, "students": EIGHT_DATASET_STUDENTS,
        "datasets": EIGHT_DATASETS, "weak_judges": WEAK,
        "note": "13 judges x 5 students x 8 datasets x 2 budgets "
                "-- the largest complete rectangle",
    },
    "eight_datasets_all_judges": {
        "judges": ALL_JUDGES, "students": EIGHT_DATASET_STUDENTS,
        "datasets": EIGHT_DATASETS, "budgets": [8192], "weak_judges": WEAK,
        "note": "16 judges x 5 students x 8 datasets, B=8192 only "
                "-- every judge, full dataset grid",
    },
    "five_datasets": {
        "judges": DUAL_BUDGET_JUDGES, "students": ALL_STUDENTS,
        "datasets": FIVE_DATASETS, "weak_judges": WEAK,
        "note": "13 judges x 7 students x 5 shared datasets x 2 budgets "
                "-- every student, both budgets",
    },
    "five_datasets_all_judges": {
        "judges": ALL_JUDGES, "students": ALL_STUDENTS,
        "datasets": FIVE_DATASETS, "budgets": [8192], "weak_judges": WEAK,
        "note": "16 judges x 7 students x 5 shared datasets, B=8192 only "
                "-- every judge and every student",
    },
}

# File name for each metric: <panel>_<name>_matrix_t<budget>.csv
METRIC_FILE_NAMES = {"far": "false_accept_rate", "frr": "false_reject_rate",
                     "accuracy": "accuracy"}


def judgement_path(judge: str, student: str, dataset: str, tok: int) -> Path | None:
    """The judgement file for one cell: family_bias/ first, then the original runs."""
    name = f"{student}_{dataset}_t{tok}.json"
    new = paths.experiment(OUT_NAME) / "judgements" / judge / name
    if new.exists():
        return new
    base = paths.core("judgements") / judge / name
    return base if base.exists() else None


def cell_metrics(judge: str, student: str, tok: int, datasets) -> dict | None:
    """FAR, FRR and accuracy for one judge-student cell over the given datasets.

    FAR and FRR count only parsed Correct/Incorrect verdicts; accuracy keeps
    unparsed verdicts in the denominator, as JudgeEvaluator does.
    """
    false_accept = true_negative = 0
    false_reject = true_positive = 0
    agree = valid = total = 0
    missing = []

    for dataset in datasets:
        path = judgement_path(judge, student, dataset, tok)
        if path is None:
            missing.append(dataset)
            continue
        with open(path) as f:
            judgements = json.load(f)
        for verdict, oracle in verdict_pairs(dataset, judgements):
            total += 1
            if verdict not in ("correct", "incorrect"):
                continue
            valid += 1
            if verdict == oracle:
                agree += 1
            if oracle == "incorrect":
                true_negative += 1
                if verdict == "correct":
                    false_accept += 1
            else:
                true_positive += 1
                if verdict == "incorrect":
                    false_reject += 1

    if not valid:
        return None
    return {
        "far": round(false_accept / true_negative, 6) if true_negative else None,
        "frr": round(false_reject / true_positive, 6) if true_positive else None,
        "accuracy": round(agree / total * 100, 4),
        "ifr": round(valid / total * 100, 4),
        "n_items": total,
        "missing_datasets": missing,
    }


def student_solver_accuracy(student: str, datasets) -> dict:
    """Share of items the student itself answered correctly, overall and per dataset."""
    correct = total = 0
    per_dataset = {}
    for dataset in datasets:
        path = paths.experiment(OUT_NAME) / "student_solutions" / student / f"{dataset}.json"
        if not path.exists():
            path = paths.core("student_solutions") / student / f"{dataset}.json"
        if not path.exists():
            continue
        with open(path) as f:
            items = json.load(f)
        dtype = get_dataset_type(dataset)
        ds_correct = sum(
            1 for item in items if expected_verdict(item, dtype)[0] == "correct"
        )
        per_dataset[dataset] = round(ds_correct / len(items) * 100, 4)
        correct += ds_correct
        total += len(items)
    return {
        "solver_accuracy": round(correct / total * 100, 4) if total else None,
        "n_items": total,
        "per_dataset": per_dataset,
    }


def student_complete(student: str, cfg: dict) -> bool:
    """Every judge x dataset x budget of the panel has a judgement for ``student``."""
    return all(
        judgement_path(j, student, d, t) is not None
        for j in cfg["judges"]
        for d in cfg["datasets"]
        for t in cfg.get("budgets", TOKEN_SETTINGS)
    )


def with_probes(cfg: dict) -> tuple[dict, list[str], list[str]]:
    """The panel plus every probe student that completes it.

    A student with only some cells judged would leave holes in the panel, so
    it is left out and reported by name instead.
    """
    added, pending = [], []
    for s_ in PROBE_STUDENTS:
        (added if student_complete(s_, cfg) else pending).append(s_)
    return {**cfg, "students": list(cfg["students"]) + added}, added, pending


def run_panel(panel: str, cfg: dict, out_dir: Path):
    cfg, added, pending = with_probes(cfg)
    judges, students, datasets = cfg["judges"], cfg["students"], cfg["datasets"]
    print(f"\n{'=' * 78}\n  Panel {panel}: {cfg['note']}\n{'=' * 78}")
    if added:
        print(f"  + probe students complete for this panel: {', '.join(added)}")
    if pending:
        print(f"  . probe students not complete here:       {', '.join(pending)}")

    solver = {s: student_solver_accuracy(s, datasets) for s in students}
    print(f"\n  {'Student':<28}{'Family':<10}{'Origin':<8}{'Solver acc':>11}{'items':>9}")
    for s in sorted(students, key=lambda s: -(solver[s]["solver_accuracy"] or 0)):
        info = STUDENT_INFO[s]
        print(f"  {s:<28}{info['family']:<10}{info['origin']:<8}"
              f"{solver[s]['solver_accuracy']:>10.2f}%{solver[s]['n_items']:>9,}")

    with open(out_dir / f"{panel}_student_solver_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["student", "family", "origin", "solver_accuracy",
                         "n_items"] + datasets)
        for s in students:
            sv = solver[s]
            writer.writerow([s, STUDENT_INFO[s]["family"], STUDENT_INFO[s]["origin"],
                             sv["solver_accuracy"], sv["n_items"]]
                            + [sv["per_dataset"].get(d, "") for d in datasets])

    family_rows, var_rows = [], []

    for tok in cfg.get("budgets", TOKEN_SETTINGS):
        label = f"t{tok}"
        matrix = {
            j: {s: cell_metrics(j, s, tok, datasets) for s in students}
            for j in judges
        }

        # A cell that misses any of the panel's datasets is dropped, so every
        # remaining cell covers the same grid.
        for j in judges:
            for s in students:
                m = matrix[j][s]
                if m and m["missing_datasets"]:
                    print(f"  !! {j} x {s} ({label}) missing "
                          f"{m['missing_datasets']} -- cell excluded")
                    matrix[j][s] = None

        for metric in ("far", "frr", "accuracy"):
            out = out_dir / f"{panel}_{METRIC_FILE_NAMES[metric]}_matrix_{label}.csv"
            with open(out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["judge", "judge_family"] + students)
                for j in judges:
                    writer.writerow(
                        [j, ALL_JUDGE_FAMILY[j]]
                        + [(matrix[j][s][metric] if matrix[j][s] else "")
                           for s in students]
                    )

        print(f"\n  -- Same-family leniency by judge family ({label}) --")
        for jfam in sorted(set(ALL_JUDGE_FAMILY[j] for j in judges)):
            fam_judges = [j for j in judges if ALL_JUDGE_FAMILY[j] == jfam]
            own = [s for s in students if STUDENT_INFO[s]["family"] == jfam]
            if not own:
                family_rows.append({
                    "budget": label, "judge_family": jfam, "far_same": None,
                    "far_other": None, "delta": None, "measurable": False,
                    "same_students": "", "same_student_origin": "",
                    "mean_solver_acc_same": None, "mean_solver_acc_other": None,
                })
                print(f"  {jfam:<10}: no same-family student in the panel")
                continue

            def mean_far(ss):
                vals = [matrix[j][s]["far"] for j in fam_judges for s in ss
                        if matrix[j][s] and matrix[j][s]["far"] is not None]
                return sum(vals) / len(vals) if vals else None

            other = [s for s in students if s not in own]
            far_same, far_other = mean_far(own), mean_far(other)
            delta = (far_same - far_other
                     if far_same is not None and far_other is not None else None)

            def mean_solver(ss):
                vals = [solver[s]["solver_accuracy"] for s in ss
                        if solver[s]["solver_accuracy"] is not None]
                return round(sum(vals) / len(vals), 4) if vals else None

            origins = sorted({STUDENT_INFO[s]["origin"] for s in own})
            print(f"  {jfam:<10}: FAR_same={far_same:.4f}  FAR_other={far_other:.4f}"
                  f"  delta={delta:+.4f}  (same: {', '.join(own)}"
                  f" [{'/'.join(origins)}], solver {mean_solver(own):.1f}%"
                  f" vs {mean_solver(other):.1f}%)")
            family_rows.append({
                "budget": label, "judge_family": jfam,
                "far_same": round(far_same, 6), "far_other": round(far_other, 6),
                "delta": round(delta, 6), "measurable": True,
                "same_students": ";".join(own),
                "same_student_origin": ";".join(origins),
                "mean_solver_acc_same": mean_solver(own),
                "mean_solver_acc_other": mean_solver(other),
            })

        # Which axis moves accuracy more: the range of per-student means
        # (averaged over judges) against the range of per-judge means.
        weak = cfg["weak_judges"]
        for name, pool in ((f"all {len(judges)} judges", judges),
                           ("capable judges only",
                            [j for j in judges if j not in weak])):
            s_means = {
                s: sum(matrix[j][s]["accuracy"] for j in pool if matrix[j][s]) /
                   max(1, sum(1 for j in pool if matrix[j][s]))
                for s in students
            }
            j_means = {
                j: sum(matrix[j][s]["accuracy"] for s in students if matrix[j][s]) /
                   max(1, sum(1 for s in students if matrix[j][s]))
                for j in pool
            }
            s_rng = max(s_means.values()) - min(s_means.values())
            j_rng = max(j_means.values()) - min(j_means.values())
            var_rows.append({
                "budget": label, "pool": name, "n_judges": len(pool),
                "student_range_pp": round(s_rng, 4),
                "judge_range_pp": round(j_rng, 4),
                "dominant_axis": "student" if s_rng > j_rng else "judge",
                "excluded": ";".join(sorted(weak)) if "capable" in name else "",
            })

    with open(out_dir / f"{panel}_family_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
        writer.writeheader()
        writer.writerows(family_rows)

    with open(out_dir / f"{panel}_variance_decomposition.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(var_rows[0].keys()))
        writer.writeheader()
        writer.writerows(var_rows)

    print("\n  Variance decomposition:")
    for r in var_rows:
        print(f"    {r['budget']} {r['pool']:<24} student {r['student_range_pp']:>6} points"
              f"  judge {r['judge_range_pp']:>6} points  -> {r['dominant_axis']}")

    print(f"\n  Saved: {out_dir}/{panel}_*.csv")


def main():
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table
    out_dir = paths.analysis_dir(OUT_NAME)
    print("=== Family bias: same-family leniency ===")
    print("FAR = P(Correct | expected Incorrect) -- higher means more lenient")
    for panel, cfg in PANELS.items():
        run_panel(panel, cfg, out_dir)


if __name__ == "__main__":
    main()
