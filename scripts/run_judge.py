"""CLI entry-point: run a single judge model across student solutions.

Usage:
    python scripts/run_judge.py --judge qwen3-4b --max-tokens 10 8192

    # Run all 16 judge models — use the bash wrapper:
    #   bash bash/run_judges.sh
"""

import argparse
import json
import logging
from pathlib import Path

from slmjury.configs import load_models_config
from slmjury.core.judge import JudgeModel


def main():
    parser = argparse.ArgumentParser(description="Run judge model evaluation.")
    parser.add_argument("--judge", required=True, help="Judge model key from models.yaml")
    parser.add_argument(
        "--max-tokens", type=int, nargs="+", default=[10, 8192],
        help="Token settings to evaluate (default: 10 8192)",
    )
    parser.add_argument(
        "--solutions-dir", default="results/student_solutions",
        help="Directory containing student solution files",
    )
    parser.add_argument("--output-dir", default="results/judgements")
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

    config = load_models_config()
    judge_cfg = config["judge_models"].get(args.judge, {})

    # Skip max_tokens=10 for always-thinks models
    token_settings = args.max_tokens
    if judge_cfg.get("always_thinks"):
        token_settings = [t for t in token_settings if t != 10]
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

        for solution_file in sorted(student_dir.glob("*.json")):
            dataset = solution_file.stem

            for max_tokens in token_settings:
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
