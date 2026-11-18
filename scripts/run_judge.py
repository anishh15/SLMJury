"""CLI entry-point: run a single judge model across student solutions."""

import argparse
import json
import logging
from pathlib import Path

from slmjury.analysis import paths
from slmjury.configs import get_judge_config
from slmjury.core.judge import JudgeModel

NOTES = """\
example:
  python scripts/run_judge.py --judge qwen3-4b --max-tokens 10 8192

One process per judge: run it once for each key under judge_models in
slmjury/configs/models.yaml (and large_judge_models for the large judges).
Existing judgement files are skipped unless --force is given.
"""


def token_settings(judge: str, requested: list[int]) -> list[int]:
    """The budgets to run for ``judge``: a judge that always reasons skips the 10-token quick verdict."""
    if get_judge_config(judge).get("always_thinks"):
        return [t for t in requested if t != 10]
    return list(requested)


def main():
    parser = argparse.ArgumentParser(
        description="Run judge model evaluation.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--judge", required=True, help="Judge model key from models.yaml")
    parser.add_argument(
        "--max-tokens", type=int, nargs="+", default=[10, 8192],
        help="Token settings to evaluate (default: 10 8192)",
    )
    parser.add_argument(
        "--solutions-dir", default=paths.default_dir("core", "student_solutions"),
        help="Directory containing student solution files",
    )
    parser.add_argument("--output-dir", default=paths.default_dir("core", "judgements"))
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Only judge these datasets (default: every dataset found in --solutions-dir)",
    )
    parser.add_argument(
        "--students", nargs="+", default=None,
        help="Only judge these students (default: every student in --solutions-dir)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-judge even if judgement files already exist",
    )

    # Hardware overrides
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
        help="Override max_num_seqs from models.yaml (lower = less KV cache pressure)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # The lookup covers the large judges too, so gpt-oss-120b and the like run at 8,192 only.
    budgets = token_settings(args.judge, args.max_tokens)
    if budgets != list(args.max_tokens):
        logger.info("Model %s always thinks — skipping max_tokens=10", args.judge)

    # Build model_config override from CLI args
    model_config = {}
    if args.tensor_parallel_size is not None:
        model_config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        model_config["max_num_seqs"] = args.max_num_seqs

    judge = JudgeModel(
        args.judge,
        output_dir=Path(args.output_dir),
        model_config=model_config or None,
    )

    output_dir = Path(args.output_dir)
    solutions_dir = Path(args.solutions_dir)
    skipped = 0

    for student_dir in sorted(solutions_dir.iterdir()):
        if not student_dir.is_dir():
            continue
        student_model = student_dir.name
        if args.students and student_model not in args.students:
            continue

        for solution_file in sorted(student_dir.glob("*.json")):
            dataset = solution_file.stem
            if args.datasets and dataset not in args.datasets:
                continue

            for max_tokens in budgets:
                # Skip if judgement file already exists (unless --force)
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
                    "Judging %s × %s (tokens=%d)", student_model, dataset, max_tokens,
                )
                results = judge.evaluate_batch(student_results, max_tokens)
                judge.save_results(results, student_model, dataset, max_tokens)

    judge.cleanup()
    if skipped:
        logger.info("Skipped %d already-judged combinations (use --force to re-judge).", skipped)
    logger.info("All evaluations complete.")


if __name__ == "__main__":
    main()
