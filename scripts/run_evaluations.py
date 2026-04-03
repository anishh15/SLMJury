"""CLI entry-point: evaluate all judgement files and generate summary reports."""

import argparse
import json
import logging
from pathlib import Path

from slmjury.core.evaluator import (
    JudgeEvaluator,
    parse_judgement_filename,
    generate_judge_summary,
    list_judge_models,
    evaluate_judge_with_tokens,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate judge accuracy and generate summaries.",
    )
    parser.add_argument(
        "--judge-model", type=str, default="all",
        help="Judge model key to evaluate, or 'all' (default: all)",
    )
    parser.add_argument(
        "--max-tokens", type=int, choices=[10, 8192], default=None,
        help="Filter by max_tokens setting (default: both 10 and 8192)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Generate summary files after evaluation",
    )
    parser.add_argument(
        "--judgements-dir", default="results/judgements",
        help="Directory containing judgement files",
    )
    parser.add_argument(
        "--output-dir", default="results/evaluations",
        help="Directory for saving evaluation results",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    judgements_dir = Path(args.judgements_dir)
    output_dir = Path(args.output_dir)

    # Determine which models and token settings to evaluate
    if args.judge_model.lower() == "all":
        judge_models = list_judge_models(judgements_dir)
    else:
        judge_models = [args.judge_model]

    token_settings = [args.max_tokens] if args.max_tokens else [10, 8192]

    logger.info(
        "Evaluating %d judge model(s) × %d token setting(s)",
        len(judge_models), len(token_settings),
    )

    for judge_model in judge_models:
        for max_tokens in token_settings:
            evaluate_judge_with_tokens(
                judge_model, max_tokens,
                judgements_dir=judgements_dir,
                output_dir=output_dir,
            )
            if args.summary:
                generate_judge_summary(
                    judge_model, max_tokens,
                    eval_dir=output_dir,
                    output_dir=output_dir / "summaries",
                )

    logger.info("All evaluations complete.")


if __name__ == "__main__":
    main()
