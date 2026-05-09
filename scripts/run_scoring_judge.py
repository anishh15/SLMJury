"""CLI entry-point: run scoring evaluation for open-ended datasets.

Scores SummEval or MT-Bench using a single judge model (one model per process).
For MT-Bench, the model loads once and scores all oracle files automatically.

Usage:
    python scripts/run_scoring_judge.py --judge qwen3-4b --dataset summeval
    python scripts/run_scoring_judge.py --judge qwen3-4b --dataset mtbench \
        --oracle-scores results/mtbench_oracle/

    # Run all scoring judges — use the bash wrappers:
    #   bash bash/run_scoring_judges.sh      # SummEval
    #   bash bash/run_mtbench_judges.sh      # MT-Bench
"""

import argparse
import json
import logging
from pathlib import Path

from slmjury.core.scoring_judge import ScoringJudge
from slmjury.core.scoring_evaluator import (
    evaluate_summeval,
    evaluate_mtbench,
    save_evaluation,
)
from slmjury.data import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Run scoring evaluation for open-ended datasets.",
    )
    parser.add_argument(
        "--judge", required=True,
        help="Judge model key from models.yaml",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["summeval", "mtbench"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Token setting for scoring (default: 8192)",
    )
    parser.add_argument(
        "--output-dir", default="results/scoring",
        help="Directory for saving scoring results",
    )
    parser.add_argument(
        "--eval-dir", default="results/scoring_evaluations",
        help="Directory for saving evaluation metrics",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score even if result files already exist",
    )

    # For MT-Bench: path to pre-computed oracle scores (file or directory)
    parser.add_argument(
        "--oracle-scores", default=None,
        help=(
            "Path to oracle scores for MT-Bench (file or directory). "
            "e.g., results/mtbench_oracle/"
        ),
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
        help="Override max_num_seqs from models.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    eval_dir = Path(args.eval_dir)

    # Skip logic (summeval only — mtbench handles per-file skipping below)
    if args.dataset == "summeval":
        skip_file = output_dir / args.judge / "summeval.json"
        if skip_file.exists() and not args.force:
            logger.info(
                "SKIP %s × summeval — already scored (use --force)",
                args.judge,
            )
            return

    # Build model_config override
    model_config = {}
    if args.tensor_parallel_size is not None:
        model_config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        model_config["max_num_seqs"] = args.max_num_seqs

    judge = ScoringJudge(
        args.judge,
        output_dir=output_dir,
        model_config=model_config or None,
    )

    if args.dataset == "summeval":
        data = load_dataset("summeval")
        logger.info("Loaded %d SummEval pairs", len(data))

        results = judge.score_summeval(data, max_tokens=args.max_tokens)
        judge.save_results(results, "summeval")

        # Compute evaluation metrics
        eval_result = evaluate_summeval(results)
        save_evaluation(eval_result, args.judge, "summeval", eval_dir)

        # Print summary
        overall = eval_result["overall"]
        logger.info(
            "SummEval overall — Pearson: %s, Spearman: %s, κ: %s",
            overall.get("pearson"), overall.get("spearman"),
            overall.get("cohens_kappa"),
        )
        for dim, metrics in eval_result["per_dimension"].items():
            logger.info(
                "  %s — r=%.3f, ρ=%.3f, κ=%.3f",
                dim,
                metrics["pearson"] or 0,
                metrics["spearman"] or 0,
                metrics["cohens_kappa"] or 0,
            )

    elif args.dataset == "mtbench":
        # MT-Bench requires student responses + oracle scores
        if not args.oracle_scores:
            logger.error(
                "MT-Bench requires --oracle-scores (file or directory)."
            )
            judge.cleanup()
            return

        # Collect oracle files (file or directory)
        oracle_path = Path(args.oracle_scores)
        if oracle_path.is_file():
            oracle_files = [oracle_path]
        elif oracle_path.is_dir():
            oracle_files = sorted(oracle_path.rglob("*.json"))
        else:
            logger.error("Not found: %s", oracle_path)
            judge.cleanup()
            return

        if not oracle_files:
            logger.error("No oracle JSON files found in %s", oracle_path)
            judge.cleanup()
            return

        logger.info(
            "Scoring %d oracle file(s) with judge %s",
            len(oracle_files), args.judge,
        )

        # Load MT-Bench prompts once
        base_data = load_dataset("mtbench")

        # Score each oracle file with the same loaded model
        for ofile in oracle_files:
            # Derive suffix: gpt-oss-120b/llama3.1-8b.json → gpt-oss-120b_llama3.1-8b
            suffix = f"{ofile.parent.name}_{ofile.stem}"

            # Skip if already scored
            out_file = output_dir / args.judge / f"mtbench_{suffix}.json"
            if out_file.exists() and not args.force:
                logger.info("SKIP %s — already scored", out_file.name)
                continue

            with open(ofile) as f:
                oracle_data = json.load(f)

            # Merge oracle data into a fresh copy of MT-Bench prompts
            data = [dict(item) for item in base_data]
            for item, oracle in zip(data, oracle_data):
                item["turn1_question"] = item["turn1"]
                item["turn1_response"] = oracle["turn1_response"]
                item["turn2_question"] = item["turn2"]
                item["turn2_response"] = oracle["turn2_response"]
                item["oracle_score"] = oracle["oracle_score"]
                item["question_id"] = item.get("question_id", item["problem_id"])

            results = judge.score_mtbench(data, max_tokens=args.max_tokens)
            judge.save_results(results, "mtbench", suffix=suffix)

            eval_result = evaluate_mtbench(results)
            save_evaluation(eval_result, args.judge, "mtbench", eval_dir, suffix=suffix)

            overall = eval_result["overall"]
            logger.info(
                "MT-Bench [%s] — Pearson: %s, Spearman: %s, κ: %s",
                suffix,
                overall.get("pearson"), overall.get("spearman"),
                overall.get("cohens_kappa"),
            )

    judge.cleanup()
    logger.info("Scoring complete for %s × %s", args.judge, args.dataset)


if __name__ == "__main__":
    main()

