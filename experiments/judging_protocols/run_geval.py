#!/usr/bin/env python3
"""G-Eval-style judging: a score from 1 to 10, where 7 or more counts as Correct."""

import argparse
import json
import logging
from pathlib import Path

from datasets_order import ordered_solution_files

from slmjury.analysis import paths
from slmjury.configs import get_judge_config
from slmjury.core.judge import JudgeModel
from slmjury.parsers.score import parse_score

NOTES = """\
Same as scripts/run_judge.py except for the prompt and the verdict. PROMPT_GEVAL
still shows the ground truth but asks for a rubric score instead of a verdict.
It is passed to JudgeModel.evaluate_batch through prompt_template at every
budget. Each response is then read with slmjury.parsers.score.parse_score on a
1-10 scale and thresholded; a response with no score is Undefined. Model
loading and sampling are the package's. Judges with always_thinks skip 10
tokens. The default budget is 8,192 only, since a rubric score needs room to
reason.

The judge is looked up in judge_models and then large_judge_models in
slmjury/configs/models.yaml, so the same command runs small and large judges.
Datasets are visited in the order of datasets_order.py. Existing files are
skipped unless --force is given.

Reads --solutions-dir/<student>/<dataset>.json (default results/core/student_solutions)
Writes --output-dir/<judge>/<student>_<dataset>_t<budget>.json
  (default results/judging_protocols/geval). Each record has the usual
  judgement fields plus numeric_score and threshold.

example, from the repository root:
  python experiments/judging_protocols/run_geval.py --judge qwen3-4b --max-tokens 8192

experiments/judging_protocols/analyze.py reads the results.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 7   # a score >= 7 (on 1-10) is Correct

# Score from 1 to 10, then threshold.
PROMPT_GEVAL = (
    "Your role is to evaluate the quality of the student's answer.\n\n"
    "Assign a score from 1 to 10, where:\n"
    "  1-3 = Clearly incorrect or missing the key concepts\n"
    "  4-6 = Partially correct or missing important details\n"
    "  7-10 = Correct and complete\n\n"
    "Give your score in the format: Score: X\n\n"
    "Question: {question}\n\n"
    "Ground truth: {ground_truth}\n\n"
    "Student answer: {student_answer}"
)


def threshold_score(score: int | None, threshold: int) -> str:
    """Turn a numeric score into Correct, Incorrect or (no score) Undefined."""
    if score is None:
        return "Undefined"
    return "Correct" if score >= threshold else "Incorrect"


def main():
    parser = argparse.ArgumentParser(
        description="G-Eval-style judging: score from 1 to 10, then threshold.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--judge", required=True, help="Judge model key from models.yaml")
    parser.add_argument(
        "--max-tokens", type=int, nargs="+", default=[8192],
        help="Token settings (default: 8192, scoring needs room to reason)",
    )
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"Lowest score counted as Correct (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--solutions-dir", default=paths.default_dir("core", "student_solutions"),
        help="Directory containing student solution files",
    )
    parser.add_argument(
        "--output-dir", default=paths.default_dir("judging_protocols", "geval"),
        help="Output directory; a sub-folder per judge is created",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-judge even if result files already exist")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Judge only these datasets (default: all eight; overrides --exclude-datasets)",
    )
    # Older spelling, kept working but not shown in --help.
    parser.add_argument("--only-datasets", dest="datasets", nargs="*",
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--exclude-datasets", nargs="*", default=[],
        help="Dataset names to skip (default: none). Ignored with --datasets",
    )

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

    only_datasets = set(args.datasets) if args.datasets else None
    exclude_datasets = set(args.exclude_datasets)
    output_dir = Path(args.output_dir)

    judge_cfg = get_judge_config(args.judge)

    token_settings = args.max_tokens
    if judge_cfg.get("always_thinks"):
        token_settings = [t for t in token_settings if t != 10]
        logger.info("Model %s always thinks — skipping max_tokens=10", args.judge)

    model_config = {}
    if args.tensor_parallel_size is not None:
        model_config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        model_config["max_num_seqs"] = args.max_num_seqs

    # save_results adds the /<judge> folder itself.
    judge = JudgeModel(
        args.judge,
        output_dir=output_dir,
        model_config=model_config or None,
    )

    solutions_dir = Path(args.solutions_dir)
    skipped = 0

    for student_dir in sorted(solutions_dir.iterdir()):
        if not student_dir.is_dir():
            continue
        student_model = student_dir.name

        for solution_file in ordered_solution_files(student_dir):
            dataset = solution_file.stem
            if only_datasets is not None:
                if dataset not in only_datasets:
                    continue
            elif dataset in exclude_datasets:
                continue

            for max_tokens in token_settings:
                out_file = (
                    output_dir / args.judge
                    / f"{student_model}_{dataset}_t{max_tokens}.json"
                )
                if out_file.exists() and not args.force:
                    logger.info(
                        "SKIP %s × %s (tokens=%d) — already judged",
                        student_model, dataset, max_tokens,
                    )
                    skipped += 1
                    continue

                with open(solution_file) as f:
                    student_results = json.load(f)

                logger.info(
                    "Judging %s × %s (tokens=%d, G-Eval, threshold=%d)",
                    student_model, dataset, max_tokens, args.threshold,
                )
                # evaluate_batch also parses a Correct/Incorrect verdict, which a
                # score response does not contain; only judge_response is used.
                raw_results = judge.evaluate_batch(
                    student_results, max_tokens, prompt_template=PROMPT_GEVAL,
                )

                results = []
                score_counts = {"Correct": 0, "Incorrect": 0, "Undefined": 0}
                for r in raw_results:
                    # max_score=10: the parser's default of 5 would clamp every
                    # score to 5 or less, and nothing would reach the threshold.
                    numeric_score = parse_score(r["judge_response"], max_score=10)
                    verdict = threshold_score(numeric_score, args.threshold)
                    score_counts[verdict] += 1
                    results.append({
                        "problem_id": r["problem_id"],
                        "question": r["question"],
                        "ground_truth_reasoning": r["ground_truth_reasoning"],
                        "ground_truth_answer": r["ground_truth_answer"],
                        "student_reasoning": r["student_reasoning"],
                        "student_answer": r["student_answer"],
                        "judge_response": r["judge_response"],
                        "numeric_score": numeric_score,
                        "threshold": args.threshold,
                        "judgement": verdict,   # Correct/Incorrect/Undefined
                    })

                logger.info("G-Eval score distribution: %s", score_counts)
                judge.save_results(results, student_model, dataset, max_tokens)

    judge.cleanup()
    if skipped:
        logger.info("Skipped %d already-judged combinations.", skipped)
    logger.info("G-Eval judging complete.")


if __name__ == "__main__":
    main()
