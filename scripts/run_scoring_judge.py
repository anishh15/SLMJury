"""CLI entry-point: run scoring evaluation for open-ended datasets."""

import argparse
import json
import logging
from pathlib import Path

from slmjury.analysis import paths
from slmjury.core.scoring_judge import ScoringJudge
from slmjury.core.scoring_evaluator import (
    evaluate_summeval,
    evaluate_mtbench,
    save_evaluation,
)
from slmjury.data import load_dataset

NOTES = """\
Scores SummEval, MT-Bench or both with one judge model per process. The model
loads once; for MT-Bench it then scores every reference file it finds.

examples:
  python scripts/run_scoring_judge.py --judge qwen3-4b --datasets summeval
  python scripts/run_scoring_judge.py --judge qwen3-4b --datasets mtbench \\
      --reference-scores results/core/mtbench_references/
  python scripts/run_scoring_judge.py --judge qwen3-4b --datasets summeval mtbench

Run it once per judge. MT-Bench needs the reference scores from
scripts/run_mtbench_reference.py first.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run scoring evaluation for open-ended datasets.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--judge", required=True,
        help="Judge model key from models.yaml",
    )
    datasets = parser.add_mutually_exclusive_group(required=True)
    datasets.add_argument(
        "--datasets", nargs="+", choices=["summeval", "mtbench"],
        help="Datasets to score: summeval, mtbench or both",
    )
    # Older spelling, kept working but not shown in --help.
    datasets.add_argument("--dataset", dest="datasets", nargs="+",
                          choices=["summeval", "mtbench"], help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Token setting for scoring (default: 8192)",
    )
    parser.add_argument(
        "--output-dir", default=paths.default_dir("core", "scoring"),
        help="Directory for saving scoring results",
    )
    parser.add_argument(
        "--eval-dir", default=paths.default_dir("core", "scoring_evaluations"),
        help="Directory for saving evaluation metrics",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score even if result files already exist",
    )

    # For MT-Bench: the reference models' scores (file or directory)
    parser.add_argument(
        "--reference-scores", default=paths.default_dir("core", "mtbench_references"),
        help=(
            "MT-Bench reference scores, one file or a directory of "
            "<reference>/<student>.json files "
            "(default: core/mtbench_references in the results folder)"
        ),
    )
    parser.add_argument("--oracle-scores", dest="reference_scores",
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)

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
    datasets = list(dict.fromkeys(args.datasets))

    # SummEval is one file per judge, skipped when it exists; MT-Bench skips
    # per reference file below.
    if "summeval" in datasets:
        skip_file = output_dir / args.judge / "summeval.json"
        if skip_file.exists() and not args.force:
            logger.info(
                "SKIP %s × summeval — already scored (use --force)",
                args.judge,
            )
            datasets.remove("summeval")

    # MT-Bench needs the reference scores (file or directory)
    reference_files = []
    if "mtbench" in datasets:
        reference_path = Path(args.reference_scores)
        if reference_path.is_file():
            reference_files = [reference_path]
        elif reference_path.is_dir():
            reference_files = sorted(reference_path.rglob("*.json"))
        else:
            logger.error("Reference scores not found: %s", reference_path)
            return
        if not reference_files:
            logger.error("No reference JSON files found in %s", reference_path)
            return

    if not datasets:
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

    if "summeval" in datasets:
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

    if "mtbench" in datasets:
        logger.info(
            "Scoring %d reference file(s) with judge %s",
            len(reference_files), args.judge,
        )

        # Load MT-Bench prompts once
        base_data = load_dataset("mtbench")

        # Score each reference file with the same loaded model
        for ref_file in reference_files:
            # Derive suffix: gpt-oss-120b/llama3.1-8b.json → gpt-oss-120b_llama3.1-8b
            suffix = f"{ref_file.parent.name}_{ref_file.stem}"

            # Skip if already scored
            out_file = output_dir / args.judge / f"mtbench_{suffix}.json"
            if out_file.exists() and not args.force:
                logger.info("SKIP %s — already scored", out_file.name)
                continue

            with open(ref_file) as f:
                reference_data = json.load(f)

            # Merge the student responses and reference scores into a fresh
            # copy of the MT-Bench prompts. ``oracle_score`` is the field name
            # the reference files and the evaluator use.
            data = [dict(item) for item in base_data]
            for item, ref in zip(data, reference_data):
                item["turn1_question"] = item["turn1"]
                item["turn1_response"] = ref["turn1_response"]
                item["turn2_question"] = item["turn2"]
                item["turn2_response"] = ref["turn2_response"]
                item["oracle_score"] = ref["oracle_score"]
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
    logger.info("Scoring complete for %s × %s", args.judge, ", ".join(datasets))


if __name__ == "__main__":
    main()
