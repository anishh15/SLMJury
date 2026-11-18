"""CLI entry-point: run a single student model across datasets."""

import argparse
import logging
from pathlib import Path

from slmjury.analysis import paths
from slmjury.configs import load_models_config
from slmjury.core.solver import StudentSolver
from slmjury.data import load_dataset

NOTES = """\
example:
  python scripts/run_student.py --student qwen2.5-32b --datasets gsm8k math

The study uses two students, qwen2.5-32b and llama3.1-8b. Run the script once
for each (all eight closed-ended datasets by default).
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run student model inference.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    student = parser.add_mutually_exclusive_group(required=True)
    student.add_argument("--student", help="Student model key from student_models in models.yaml")
    # Older spelling, kept working but not shown in --help.
    student.add_argument("--model", dest="student", help=argparse.SUPPRESS)
    parser.add_argument(
        "--datasets", nargs="+",
        default=["gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
                 "hellaswag", "winogrande", "truthfulqa"],
        help="Closed-ended datasets to solve (default: all 8). "
             "Open-ended datasets (SummEval, MT-Bench) use run_scoring_judge.py.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Limit problems per dataset")
    parser.add_argument("--output-dir", default=paths.default_dir("core", "student_solutions"))

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

    # Every dataset loads before the model, so a bad download stops the run first.
    datasets = {}
    for dataset_name in args.datasets:
        logger.info("Loading dataset: %s", dataset_name)
        datasets[dataset_name] = load_dataset(dataset_name)

    solver = StudentSolver(
        args.student,
        output_dir=Path(args.output_dir),
        model_config=model_config or None,
    )

    for dataset_name, problems in datasets.items():
        results = solver.solve_batch(problems, dataset_name, num_samples=args.num_samples)
        solver.save_results(results, dataset_name)

    solver.cleanup()
    logger.info("All datasets complete.")


if __name__ == "__main__":
    main()
