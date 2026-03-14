"""CLI entry-point: run student model inference across datasets."""

import argparse
import logging

from slmjury.configs import load_models_config
from slmjury.core.solver import StudentSolver
from slmjury.data import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Run student model inference.")
    parser.add_argument("--model", required=True, help="Student model key from models.yaml")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge"],
        help="Datasets to solve (default: all)",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Limit problems per dataset")
    parser.add_argument("--output-dir", default="results/student_solutions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    solver = StudentSolver(args.model, output_dir=args.output_dir)

    for dataset_name in args.datasets:
        logger.info("Loading dataset: %s", dataset_name)
        problems = load_dataset(dataset_name)
        results = solver.solve_batch(problems, dataset_name, num_samples=args.num_samples)
        solver.save_results(results, dataset_name)

    solver.cleanup()
    logger.info("All datasets complete.")


if __name__ == "__main__":
    main()
