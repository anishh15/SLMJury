"""CLI entry-point: evaluate persona, jury (majority vote) and debate runs, and student accuracy."""

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from slmjury.core.evaluator import JudgeEvaluator
from slmjury.parsers.answer import get_dataset_type
from slmjury.parsers.normalizer import answers_are_equivalent

NOTES = """\
scripts/run_evaluation.py covers the individual judges. This script does the
same for the strategy runs. Every judgement file written by run_persona.py, by
generate_all_ensembles() (juries) and by run_debate.py goes through
JudgeEvaluator, which compares each verdict with the expected verdict from the
ground truth. The per-file results are then summed into one summary per judge
and budget (personas) or per panel or configuration (juries, debate). Accuracy
is how often the verdict matches the ground truth, not how often the judge
said Correct.

Reads, under <results>/core/ (<results> is --results-dir, which defaults to
$SLMJURY_RESULTS if set, else results/):
  personas/<judge>/<persona>_<student>_<dataset>_t<budget>.json
  juries/<panel>/<student>_<dataset>.json
  debate/<configuration>/<student>_<dataset>.json
  student_solutions/<student>/<dataset>.json

Writes, under <results>/core/:
  personas/evaluations/<judge>/<persona>_<student>_<dataset>_t<budget>.json
  juries/evaluations/<panel>/<student>_<dataset>.json
  debate/evaluations/<configuration>/<student>_<dataset>.json
  summaries/personas/<judge>_t<budget>.json
  summaries/juries/<panel>.json
  summaries/debate/<configuration>.json
  summaries/student_accuracy.json

experiments/main_results/analyze.py reads the evaluation files.

examples:
  python scripts/run_strategy_evaluation.py                 # all four steps
  python scripts/run_strategy_evaluation.py --personas --debate
  python scripts/run_strategy_evaluation.py --results-dir path/to/results
"""

DATASET_PATTERN = r"(gsm8k|gsm_plus|math|arc_easy|arc_challenge|hellaswag|winogrande|truthfulqa)"


def parse_persona_filename(filename):
    """Split ``<Persona>_<student>_<dataset>_t<N>.json`` into its parts, or None."""
    match = re.match(
        rf"^(.+?)_(.+?)_{DATASET_PATTERN}_t(\d+)\.json$",
        Path(filename).name,
    )
    if not match:
        return None
    return {
        "persona": match.group(1),
        "student": match.group(2),
        "dataset": match.group(3),
        "tokens": int(match.group(4)),
    }


def parse_panel_filename(filename):
    """Split ``<student>_<dataset>.json`` into its parts, or None."""
    match = re.match(
        rf"^(.+?)_{DATASET_PATTERN}\.json$",
        Path(filename).name,
    )
    if not match:
        return None
    return {
        "student": match.group(1),
        "dataset": match.group(2),
    }


def aggregate_eval_files(eval_files):
    """Sum counts over evaluation files, overall, per dataset and per persona."""
    total_problems = 0
    total_correct = 0
    total_valid = 0
    by_dataset = defaultdict(lambda: {"total": 0, "correct": 0, "valid": 0})
    by_persona = defaultdict(lambda: {"total": 0, "correct": 0, "valid": 0})

    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)

        dataset = data.get("dataset", "unknown")
        persona = data.get("persona")
        n = data["total_problems"]
        correct = data["counts"]["correct_verdicts"]
        valid = data["counts"]["valid_verdicts"]

        total_problems += n
        total_correct += correct
        total_valid += valid

        by_dataset[dataset]["total"] += n
        by_dataset[dataset]["correct"] += correct
        by_dataset[dataset]["valid"] += valid

        if persona:
            by_persona[persona]["total"] += n
            by_persona[persona]["correct"] += correct
            by_persona[persona]["valid"] += valid

    for d in list(by_dataset.values()) + list(by_persona.values()):
        d["accuracy"] = round(d["correct"] / d["total"], 4) if d["total"] else 0
        d["ifr"] = round(d["valid"] / d["total"], 4) if d["total"] else 0

    return {
        "total_problems": total_problems,
        "total_correct": total_correct,
        "total_valid": total_valid,
        "overall_accuracy": round(total_correct / total_problems, 4) if total_problems else 0,
        "overall_ifr": round(total_valid / total_problems, 4) if total_problems else 0,
        "by_dataset": dict(by_dataset),
        "by_persona": dict(by_persona) if by_persona else None,
        "timestamp": datetime.now().isoformat(),
    }


def evaluate_personas(results):
    """Evaluate every persona judgement file, then summarise per judge and budget."""
    persona_dir = results / "personas"
    eval_base = persona_dir / "evaluations"
    summary_dir = results / "summaries" / "personas"

    if not persona_dir.exists():
        print(f"  No persona runs found at {persona_dir}")
        return
    summary_dir.mkdir(parents=True, exist_ok=True)

    for judge_dir in sorted(persona_dir.iterdir()):
        if not judge_dir.is_dir() or judge_dir.name == "evaluations":
            continue
        judge_model = judge_dir.name
        by_tokens = defaultdict(list)

        for f in sorted(judge_dir.glob("*.json")):
            parsed = parse_persona_filename(f.name)
            if not parsed:
                continue

            with open(f) as fp:
                judgements = json.load(fp)

            eval_dir = eval_base / judge_model
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_file = eval_dir / (f"{parsed['persona']}_{parsed['student']}_"
                                    f"{parsed['dataset']}_t{parsed['tokens']}.json")

            evaluator = JudgeEvaluator(
                judge_model=judge_model,
                student_model=parsed["student"],
                dataset=parsed["dataset"],
                max_tokens=parsed["tokens"],
                judgements=judgements,
            )
            evaluator.evaluate()

            eval_result = {
                **evaluator.summary,
                "persona": parsed["persona"],
                "student_model": parsed["student"],
                "dataset": parsed["dataset"],
                "dataset_type": evaluator.dataset_type,
            }
            with open(eval_file, "w") as fp:
                json.dump(eval_result, fp, indent=2)

            by_tokens[parsed["tokens"]].append(eval_file)

        for max_tokens, eval_files in sorted(by_tokens.items()):
            summary = {
                "judge_model": judge_model,
                "max_tokens": max_tokens,
                **aggregate_eval_files(eval_files),
            }
            out_file = summary_dir / f"{judge_model}_t{max_tokens}.json"
            with open(out_file, "w") as fp:
                json.dump(summary, fp, indent=2)
            print(f"  Persona {judge_model}_t{max_tokens}: "
                  f"{summary['overall_accuracy'] * 100:.2f}% ({len(eval_files)} files)")


def evaluate_panels(results, folder, label):
    """Evaluate every jury or debate file under ``<results>/core/<folder>``, then summarise.

    Each subfolder is one jury panel or one debate configuration; its name is stored as
    ``judge_model`` and ``ensemble_name``, and ``max_tokens`` is 0 (a panel has no single budget).
    """
    runs_dir = results / folder
    eval_base = runs_dir / "evaluations"
    summary_dir = results / "summaries" / folder

    if not runs_dir.exists():
        print(f"  No {label} runs found at {runs_dir}")
        return
    summary_dir.mkdir(parents=True, exist_ok=True)

    for panel_dir in sorted(runs_dir.iterdir()):
        if not panel_dir.is_dir() or panel_dir.name == "evaluations":
            continue
        panel = panel_dir.name
        eval_dir = eval_base / panel
        eval_files = []

        for f in sorted(panel_dir.glob("*.json")):
            parsed = parse_panel_filename(f.name)
            if not parsed:
                continue

            with open(f) as fp:
                judgements = json.load(fp)

            evaluator = JudgeEvaluator(
                judge_model=panel,
                student_model=parsed["student"],
                dataset=parsed["dataset"],
                max_tokens=0,
                judgements=judgements,
            )
            evaluator.evaluate()

            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_file = eval_dir / f"{parsed['student']}_{parsed['dataset']}.json"
            eval_result = {
                **evaluator.summary,
                "ensemble_name": panel,
                "student_model": parsed["student"],
                "dataset": parsed["dataset"],
                "dataset_type": evaluator.dataset_type,
            }
            with open(eval_file, "w") as fp:
                json.dump(eval_result, fp, indent=2)
            eval_files.append(eval_file)

        if eval_files:
            summary = {"ensemble_name": panel, **aggregate_eval_files(eval_files)}
            out_file = summary_dir / f"{panel}.json"
            with open(out_file, "w") as fp:
                json.dump(summary, fp, indent=2)
            print(f"  {label} {panel}: "
                  f"{summary['overall_accuracy'] * 100:.2f}% ({len(eval_files)} files)")


def evaluate_students(results):
    """Count correct and valid student answers per model and dataset.

    A valid answer is any non-empty answer other than UNABLE_TO_EXTRACT; a
    correct one is equivalent to the ground truth under ``answers_are_equivalent``.
    """
    students_dir = results / "student_solutions"
    if not students_dir.exists():
        print(f"  No student solutions found at {students_dir}")
        return

    accuracy = {}
    for model_dir in sorted(students_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        accuracy[model] = {}

        for f in sorted(model_dir.glob("*.json")):
            dataset = f.stem
            with open(f) as fp:
                data = json.load(fp)

            dataset_type = get_dataset_type(dataset)
            correct = 0
            valid = 0
            for item in data:
                student_ans = item.get("student_answer", "")
                gt_ans = item.get("ground_truth_answer", "")
                if student_ans and student_ans != "UNABLE_TO_EXTRACT":
                    valid += 1
                if answers_are_equivalent(student_ans, gt_ans, dataset_type):
                    correct += 1

            total = len(data)
            accuracy[model][dataset] = {
                "correct": correct,
                "valid": valid,
                "total": total,
                "accuracy": round(correct / total, 4) if total else 0,
                "accuracy_pct": round(correct / total * 100, 2) if total else 0,
            }
            print(f"  Student {model}/{dataset}: {correct}/{total} "
                  f"({accuracy[model][dataset]['accuracy_pct']}%)")

        accuracy[model]["last_updated"] = datetime.now().isoformat()

    out_file = results / "summaries" / "student_accuracy.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(accuracy, fp, indent=4)
    print(f"  Saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate persona, jury (majority vote) and debate runs, and student\n"
                    "accuracy. With no step flags, all four steps run.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--student-accuracy", action="store_true",
                        help="Student accuracy -> core/summaries/student_accuracy.json")
    parser.add_argument("--personas", action="store_true",
                        help="Persona runs -> core/personas/evaluations/, "
                             "core/summaries/personas/")
    parser.add_argument("--juries", action="store_true",
                        help="Majority-vote juries -> core/juries/evaluations/, "
                             "core/summaries/juries/")
    parser.add_argument("--debate", action="store_true",
                        help="Debate runs -> core/debate/evaluations/, "
                             "core/summaries/debate/")
    # Older spellings, kept working but not shown in --help.
    parser.add_argument("--students", dest="student_accuracy", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--voting", dest="juries", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--results-dir", default=os.environ.get("SLMJURY_RESULTS", "results"),
                        help="Results folder; runs are read from and written to its "
                             "core/ subfolder (default: $SLMJURY_RESULTS if set, else results)")
    args = parser.parse_args()

    run_all = not (args.student_accuracy or args.personas or args.juries or args.debate)
    results = Path(args.results_dir) / "core"

    if run_all or args.student_accuracy:
        print("=== Student accuracy ===")
        evaluate_students(results)
    if run_all or args.personas:
        print("\n=== Persona evaluations and summaries ===")
        evaluate_personas(results)
    if run_all or args.juries:
        print("\n=== Jury (majority-vote) evaluations and summaries ===")
        evaluate_panels(results, "juries", "Jury")
    if run_all or args.debate:
        print("\n=== Debate evaluations and summaries ===")
        evaluate_panels(results, "debate", "Debate")
    print("\nDone.")


if __name__ == "__main__":
    main()
