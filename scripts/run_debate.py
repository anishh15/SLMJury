"""CLI entry-point: run debate configurations over every student and dataset."""

import argparse
import json
import logging
from pathlib import Path

from slmjury.analysis import paths
from slmjury.strategies.debate import (
    DEFAULT_DEBATE_MAX_TOKENS,
    MAD_COMBINATIONS,
    make_combo_name,
    run_debate,
    save_debate_results,
)

NOTES = """\
A configuration is three agents: three different judges, or one judge at three
temperatures (MAD_COMBINATIONS in slmjury/strategies/debate.py, numbered 0 to 9).
The agents debate each item for up to 5 rounds until they agree. Items still
split after the last round take the majority verdict.

examples:
  python scripts/run_debate.py --configs 0
  python scripts/run_debate.py --configs 0,3,5 --tp 1 --gpu-mem 0.85
  python scripts/run_debate.py --configs all

Writes results/core/debate/<configuration>/<student>_<dataset>.json. To score
the verdicts against the ground truth (writes results/core/debate/evaluations/
and results/core/summaries/debate/), run:
  python scripts/run_strategy_evaluation.py --debate
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run debate configurations over every student and dataset.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    configs = parser.add_mutually_exclusive_group(required=True)
    configs.add_argument(
        "--configs",
        help="Debate configuration index (0-9), a comma-separated list, or 'all'",
    )
    # Older spelling, kept working but not shown in --help.
    configs.add_argument("--combo-index", dest="configs", help=argparse.SUPPRESS)
    parser.add_argument(
        "--solutions-dir", default=paths.default_dir("core", "student_solutions"),
        help="Directory containing student solution files",
    )
    parser.add_argument("--output-dir", default=paths.default_dir("core", "debate"))
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_DEBATE_MAX_TOKENS,
        help=f"Generation budget per agent turn (default: {DEFAULT_DEBATE_MAX_TOKENS}; "
             "configurations 1, 2 and 4 mostly used 4096; see docs/REPRODUCE.md)",
    )
    parser.add_argument(
        "--students", nargs="+", default=None,
        help="Only debate these students (default: every student in --solutions-dir)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Only debate these datasets (default: every dataset found in --solutions-dir)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if debate result files already exist",
    )

    # Hardware overrides
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=None,
        help="Override tensor_parallel_size for all models",
    )
    parser.add_argument(
        "--gpu-memory-utilization", "--gpu-mem", type=float, default=None,
        help="Override gpu_memory_utilization for all models",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="Override max_num_seqs for all models (lower = less KV cache pressure)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    solutions_dir = Path(args.solutions_dir)
    output_dir = Path(args.output_dir)

    # Apply hardware overrides via config patching
    if args.gpu_memory_utilization is not None or args.max_num_seqs is not None:
        import slmjury.strategies.debate as dmod
        _orig = dmod.load_models_config
        mem_util = args.gpu_memory_utilization
        max_num_seqs = args.max_num_seqs

        def _patched():
            c = _orig()
            for v in c.get("judge_models", {}).values():
                if mem_util is not None:
                    v["gpu_memory_utilization"] = mem_util
                if max_num_seqs is not None:
                    v["max_num_seqs"] = max_num_seqs
            return c

        dmod.load_models_config = _patched
        if mem_util is not None:
            logger.info("GPU memory utilization override: %.2f", mem_util)
        if max_num_seqs is not None:
            # Also patch _DEBATE_MAX_NUM_SEQS — debate.py hardcodes per-model
            # values that would silently overwrite the config-level override.
            for k in dmod._DEBATE_MAX_NUM_SEQS:
                dmod._DEBATE_MAX_NUM_SEQS[k] = max_num_seqs
            logger.info("max_num_seqs override: %d", max_num_seqs)

    # Determine which configurations to run
    if args.configs.lower() == "all":
        combo_indices = list(range(len(MAD_COMBINATIONS)))
    else:
        combo_indices = [int(i) for i in args.configs.split(",")]

    skipped = 0
    for idx in combo_indices:
        combo_models, combo_temps = MAD_COMBINATIONS[idx]
        combo_name = make_combo_name(combo_models, combo_temps)

        logger.info(
            "[Configuration %d/%d] %s (max_tokens=%d)",
            idx + 1, len(MAD_COMBINATIONS), combo_name, args.max_tokens,
        )

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

                # Skip if result file already exists (unless --force)
                out_file = output_dir / combo_name / f"{student_model}_{dataset}.json"
                if out_file.exists() and not args.force:
                    logger.info(
                        "  SKIP %s × %s — already debated",
                        student_model, dataset,
                    )
                    skipped += 1
                    continue

                with open(solution_file) as f:
                    student_results = json.load(f)

                logger.info(
                    "  %s × %s (%d problems)",
                    student_model, dataset, len(student_results),
                )

                debate_state = run_debate(
                    combo_models=combo_models,
                    combo_temps=combo_temps,
                    student_results=student_results,
                    dataset_name=dataset,
                    tp_size_override=args.tensor_parallel_size,
                    max_tokens=args.max_tokens,
                )

                save_debate_results(
                    combo_name=combo_name,
                    debate_state=debate_state,
                    student_model=student_model,
                    dataset=dataset,
                    output_dir=output_dir,
                )

                # Print summary
                resolved_counts = {}
                for state in debate_state.values():
                    r = state["resolved_at_round"]
                    resolved_counts[r] = resolved_counts.get(r, 0) + 1
                logger.info("    Resolution: %s", dict(sorted(
                    resolved_counts.items(),
                    key=lambda x: (isinstance(x[0], str), x[0]),
                )))

        logger.info("Configuration %d complete: %s", idx, combo_name)

    if skipped:
        logger.info("Skipped %d already-completed debates (use --force to re-run).", skipped)
    logger.info("All debates complete.")


if __name__ == "__main__":
    main()
