"""CLI entry-point: run judge model evaluation across student solutions."""

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

    judge = JudgeModel(args.judge, output_dir=Path(args.output_dir))

    solutions_dir = Path(args.solutions_dir)
    for student_dir in sorted(solutions_dir.iterdir()):
        if not student_dir.is_dir():
            continue
        student_model = student_dir.name

        for solution_file in sorted(student_dir.glob("*.json")):
            dataset = solution_file.stem

            with open(solution_file) as f:
                student_results = json.load(f)

            for max_tokens in token_settings:
                logger.info(
                    "Judging %s × %s (tokens=%d)", student_model, dataset, max_tokens,
                )
                results = judge.evaluate_batch(student_results, max_tokens)
                judge.save_results(results, student_model, dataset, max_tokens)

    judge.cleanup()
    logger.info("All evaluations complete.")


if __name__ == "__main__":
    main()
