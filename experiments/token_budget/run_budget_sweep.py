#!/usr/bin/env python3
"""Judge the main students at nine token budgets, from 10 to 8,192."""

import argparse
import json
import logging
from pathlib import Path

from slmjury.analysis import paths
from slmjury.configs import get_judge_config
from slmjury.core.evaluator import JudgeEvaluator
from slmjury.core.judge import JudgeModel

NOTES = """\
Shows whether accuracy changes gradually between the quick (10-token) and the
reasoned (8,192-token) verdict or jumps at some budget. Each judge is loaded
once and JudgeModel.evaluate_batch is called with every budget in turn,
exactly as scripts/run_judge.py calls it for 10 and 8,192. The package picks
the prompt and the thinking mode from the budget: PROMPT_QUICK at 10 tokens and
PROMPT_REASONED above it, with thinking on only at 8,192. Judges with
always_thinks skip the 10-token budget.

After each file the script logs its accuracy and instruction-following rate,
computed by JudgeEvaluator, and at the end it prints a judge x dataset x
budget table averaged over the students. Existing files are read instead of
re-judged unless --force is given; a file that does not parse is judged again.

Reads results/core/student_solutions/<student>/<dataset>.json
Writes results/token_budget/budget_sweep/<judge>/<student>_<dataset>_b<budget>.json

examples, from the repository root:
  python experiments/token_budget/run_budget_sweep.py --judges qwen3-8b
  python experiments/token_budget/run_budget_sweep.py --judges phi4-14b --tp 2 --gpu-mem 0.85

experiments/token_budget/analyze.py and experiments/adaptive_budget/analyze.py
read the sweep.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]

DEFAULT_BUDGETS = [10, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# The sixteen small judges.
DEFAULT_JUDGES = [
    "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b",
    "llama3.2-1b", "llama3.2-3b", "llama3.1-8b",
    "phi4-14b", "phi4r-14b", "phi4rp-14b", "phi4mi-3.8b", "phi4mr-3.8b",
]

# The eight closed-ended datasets.
DEFAULT_DATASETS = [
    "arc_challenge", "arc_easy", "gsm8k", "gsm_plus",
    "hellaswag", "math", "truthfulqa", "winogrande",
]


def run_judge_sweep(
    judge_key: str,
    datasets: list[str],
    students: list[str],
    budgets: list[int],
    solutions_dir: Path,
    output_dir: Path,
    force: bool = False,
    model_config: dict | None = None,
) -> dict:
    """Judge every student x dataset at every budget with one loaded judge.

    Returns:
        {dataset: {student: {budget: {"accuracy": float, "ifr": float, "n": int}}}}
    """
    judge_cfg = get_judge_config(judge_key)

    effective_budgets = budgets
    if judge_cfg.get("always_thinks"):
        effective_budgets = [b for b in budgets if b != 10]
        logger.info("Model %s always thinks — skipping B=10", judge_key)

    judge = JudgeModel(
        judge_key,
        output_dir=output_dir / judge_key,
        model_config=model_config,
    )
    results_map = {}

    try:
        for student in students:
            for dataset in datasets:
                sol_file = solutions_dir / student / f"{dataset}.json"
                if not sol_file.exists():
                    logger.warning("Solutions not found: %s", sol_file)
                    continue

                with open(sol_file) as f:
                    student_results = json.load(f)

                for budget in sorted(effective_budgets):
                    out_file = output_dir / judge_key / f"{student}_{dataset}_b{budget}.json"

                    judgements = None
                    if out_file.exists() and not force:
                        try:
                            with open(out_file) as f:
                                judgements = json.load(f)
                            logger.info(
                                "SKIP %s × %s × B=%d — already judged",
                                judge_key, dataset, budget,
                            )
                        except (json.JSONDecodeError, ValueError):
                            logger.warning("Unreadable JSON file (%s), judging again", out_file)

                    if judgements is None:
                        logger.info("Judging %s × %s × B=%d", judge_key, dataset, budget)
                        judgements = judge.evaluate_batch(student_results, budget)

                        # Same layout as JudgeModel.save_results, with a b<budget> suffix.
                        out_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_file, "w") as f:
                            json.dump(judgements, f, indent=4)

                    evaluator = JudgeEvaluator(
                        judge_model=judge_key,
                        student_model=student,
                        dataset=dataset,
                        max_tokens=budget,
                        judgements=judgements,
                    )
                    summary = evaluator.evaluate()
                    acc = summary["metrics"]["accuracy"] * 100
                    ifr = summary["metrics"]["instruction_following_rate"] * 100
                    logger.info("  B=%d: acc=%.2f%%, IFR=%.2f%%", budget, acc, ifr)

                    results_map.setdefault(dataset, {}).setdefault(student, {})[budget] = {
                        "accuracy": round(acc, 4),
                        "ifr": round(ifr, 4),
                        "n": len(judgements),
                    }
    finally:
        judge.cleanup()

    return results_map


def main():
    parser = argparse.ArgumentParser(
        description="Judge the main students at several token budgets.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES,
                        help="Judge keys from models.yaml (default: the 16 small judges)")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="Closed-ended datasets (default: all 8)")
    parser.add_argument("--max-tokens", nargs="+", type=int, default=DEFAULT_BUDGETS,
                        help="Token budgets (default: 10 64 128 256 512 1024 2048 4096 8192)")
    # Older spelling, kept working but not shown in --help.
    parser.add_argument("--budgets", dest="max_tokens", nargs="+", type=int,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--students", nargs="+", default=STUDENTS)
    parser.add_argument(
        "--solutions-dir", default=paths.default_dir("core", "student_solutions"),
        help="Directory containing student solution files",
    )
    parser.add_argument(
        "--output-dir", default=paths.default_dir("token_budget", "budget_sweep"),
        help="Output directory; a sub-folder per judge is created",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-judge even if result files already exist")

    # Hardware overrides, as in scripts/run_judge.py
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=None,
        help="Override tensor_parallel_size from models.yaml",
    )
    parser.add_argument(
        "--gpu-memory-utilization", "--gpu-mem", type=float, default=None,
        help="Override gpu_memory_utilization from models.yaml",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="Override max_num_seqs from models.yaml",
    )
    args = parser.parse_args()

    model_config = {}
    if args.tensor_parallel_size is not None:
        model_config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        model_config["max_num_seqs"] = args.max_num_seqs

    solutions_dir = Path(args.solutions_dir)
    output_dir = Path(args.output_dir)

    all_results = {}
    for judge_key in args.judges:
        logger.info("=== Budget sweep: %s ===", judge_key)
        all_results[judge_key] = run_judge_sweep(
            judge_key, args.datasets, args.students, args.max_tokens,
            solutions_dir, output_dir, args.force,
            model_config=model_config or None,
        )

    # Accuracy per dataset and budget, averaged over the students.
    print("\n=== Budget sweep summary ===")
    budget_header = "\t".join(f"B={b}" for b in sorted(args.max_tokens))
    for judge_key in args.judges:
        print(f"\n{judge_key}:")
        print(f"  {'Dataset':<15}\t{budget_header}")
        for dataset in args.datasets:
            row = [f"{dataset:<15}"]
            for budget in sorted(args.max_tokens):
                accs = [
                    all_results.get(judge_key, {}).get(dataset, {}).get(s, {})
                    .get(budget, {}).get("accuracy")
                    for s in args.students
                ]
                accs = [a for a in accs if a is not None]
                avg = sum(accs) / len(accs) if accs else None
                row.append(f"{avg:.2f}%" if avg is not None else "N/A")
            print("  " + "\t".join(row))


if __name__ == "__main__":
    main()
