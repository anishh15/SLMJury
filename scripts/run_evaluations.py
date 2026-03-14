"""CLI entry-point: evaluate all judgement files and generate summary reports."""

import argparse
import json
import logging
from pathlib import Path

from slmjury.core.evaluator import JudgeEvaluator, parse_judgement_filename, generate_judge_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate judge accuracy and generate summaries.")
    parser.add_argument(
        "--judgements-dir", default="results/judgements",
        help="Directory containing judgement files",
    )
    parser.add_argument("--output-dir", default="results/summaries")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    judgements_dir = Path(args.judgements_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for judge_dir in sorted(judgements_dir.iterdir()):
        if not judge_dir.is_dir():
            continue

        for f in sorted(judge_dir.glob("*.json")):
            try:
                judge, student, dataset, tokens = parse_judgement_filename(f.name)
            except ValueError:
                logger.warning("Skipping unparseable file: %s", f.name)
                continue

            with open(f) as fp:
                judgements = json.load(fp)

            evaluator = JudgeEvaluator(judge, student, dataset, tokens, judgements)
            summary = evaluator.evaluate()
            all_summaries.append(summary)

            logger.info(
                "%s → %s × %s (t=%d): acc=%.2f%% ifr=%.2f%%",
                judge, student, dataset, tokens,
                evaluator.accuracy * 100, evaluator.ifr * 100,
            )

    # Generate aggregate summary
    aggregate = generate_judge_summary(all_summaries)

    summary_file = output_dir / "judge_evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(aggregate, f, indent=4)

    logger.info("Summary saved to %s (%d evaluations)", summary_file, len(all_summaries))


if __name__ == "__main__":
    main()
