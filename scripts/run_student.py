"""CLI entry-point: run a single student model across datasets.

Usage:
    python scripts/run_student.py --model qwen2.5-32b --datasets gsm8k math

    # Run all student models — use the bash wrapper:
    #   bash bash/run_students.sh
"""

import argparse
import logging
from pathlib import Path

from slmjury.configs import load_models_config
from slmjury.core.solver import StudentSolver
from slmjury.data import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Run student model inference.")
    parser.add_argument("--model", required=True, help="Student model key from models.yaml")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
                 "hellaswag", "winogrande", "truthfulqa"],
        help="Datasets to solve (default: all)",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Limit problems per dataset")
    parser.add_argument("--output-dir", default="results/student_solutions")

    # Hardware overrides
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=None,
        help="Override tensor_parallel_size from models.yaml",
    )
    parser.add_argument(
        "--gpu-memory-utilization", "--gpu-mem", type=float, default=None,
        help="Override gpu_memory_utilization from models.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # Build model_config override from CLI args
    model_config = {}
    if args.tensor_parallel_size is not None:
        model_config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization

    solver = StudentSolver(
        args.model,
        output_dir=Path(args.output_dir),
        model_config=model_config or None,
    )

    for dataset_name in args.datasets:
        logger.info("Loading dataset: %s", dataset_name)
        problems = load_dataset(dataset_name)
        results = solver.solve_batch(problems, dataset_name, num_samples=args.num_samples)
        solver.save_results(results, dataset_name)

    solver.cleanup()
    logger.info("All datasets complete.")


if __name__ == "__main__":
    main()
