"""CLI entry-point: run multi-agent debate (MAD) experiments.

Runs debate combinations across all student-dataset combos. Each combo
involves 3 agents debating for up to 5 rounds to reach consensus, with
majority vote fallback.

Usage:
    # Run a specific combo by index (0-9)
    CUDA_VISIBLE_DEVICES=2,3 python scripts/run_mad.py --combo-index 0

    # Run all 10 combos sequentially
    CUDA_VISIBLE_DEVICES=2,3 python scripts/run_mad.py --combo-index all

    # Run with hardware overrides
    python scripts/run_mad.py --combo-index 0 --tp 1 --gpu-mem 0.8
"""

import argparse
import json
import logging
from pathlib import Path

from slmjury.strategies.debate import (
    MAD_COMBINATIONS,
    make_combo_name,
    run_debate,
    save_debate_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent debate experiments.",
    )
    parser.add_argument(
        "--combo-index", required=True,
        help="Combo index (0-9), comma-separated list, or 'all'",
    )
    parser.add_argument(
        "--solutions-dir", default="results/student_solutions",
        help="Directory containing student solution files",
    )
    parser.add_argument("--output-dir", default="results/multi_agent_debate")

    # Hardware overrides
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=None,
        help="Override tensor_parallel_size for all models",
    )
    parser.add_argument(
        "--gpu-memory-utilization", "--gpu-mem", type=float, default=None,
        help="Override gpu_memory_utilization for all models",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    solutions_dir = Path(args.solutions_dir)
    output_dir = Path(args.output_dir)

    # Apply gpu_memory_utilization override via config patching
    if args.gpu_memory_utilization is not None:
        import slmjury.strategies.debate as dmod
        _orig = dmod.load_models_config
        mem_util = args.gpu_memory_utilization

        def _patched():
            c = _orig()
            for v in c.get("judge_models", {}).values():
                v["gpu_memory_utilization"] = mem_util
            return c

        dmod.load_models_config = _patched
        logger.info("GPU memory utilization override: %.2f", mem_util)

    # Determine which combos to run
    if args.combo_index.lower() == "all":
        combo_indices = list(range(len(MAD_COMBINATIONS)))
    else:
        combo_indices = [int(i) for i in args.combo_index.split(",")]

    for idx in combo_indices:
        combo_models, combo_temps = MAD_COMBINATIONS[idx]
        combo_name = make_combo_name(combo_models, combo_temps)

        logger.info(
            "[Combo %d/%d] %s", idx + 1, len(MAD_COMBINATIONS), combo_name,
        )

        for student_dir in sorted(solutions_dir.iterdir()):
            if not student_dir.is_dir():
                continue
            student_model = student_dir.name

            for solution_file in sorted(student_dir.glob("*.json")):
                dataset = solution_file.stem

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

        logger.info("Combo %d complete: %s", idx, combo_name)

    logger.info("All debate experiments complete.")


if __name__ == "__main__":
    main()
