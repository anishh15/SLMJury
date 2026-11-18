#!/usr/bin/env python3
"""Generate solutions for the family-bias students with vLLM."""

import argparse
import logging
from pathlib import Path

from slmjury.analysis import paths
from slmjury.core.solver import StudentSolver
from slmjury.data import load_dataset

NOTES = """\
Same as scripts/run_student.py, but writes to results/family_bias/student_solutions
by default. The student is looked up under student_models in
slmjury/configs/models.yaml. StudentSolver samples every student the same way
(temperature 0.7, top_p 0.9, 1,024 new tokens, no seed), so solutions from
different students are comparable. Because sampling is unseeded, regenerating
a dataset gives different answers from the ones already judged.

The students vLLM serves are deepseek-r1-0528-qwen3-8b, mistral-7b-v0.3 and
qwen3-14b-student. gemma4-31b-student, glm-4.7-flash and qwen3.5-35b-student
use architectures that need a newer transformers than vLLM supported; generate
them with run_students_hf.py in this folder.

Writes results/family_bias/student_solutions/<student>/<dataset>.json

examples, from the repository root:
  python experiments/family_bias/run_students.py --student mistral-7b-v0.3
  python experiments/family_bias/run_students.py --student qwen3-14b-student --tp 2 --gpu-mem 0.85
"""

ALL_DATASETS = ["gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
                "hellaswag", "winogrande", "truthfulqa"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate family-bias student solutions with vLLM.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    student = parser.add_mutually_exclusive_group(required=True)
    student.add_argument("--student", help="Student model key from student_models in models.yaml")
    # Older spelling, kept working but not shown in --help.
    student.add_argument("--model", dest="student", help=argparse.SUPPRESS)
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Closed-ended datasets to solve (default: all 8)",
    )
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit problems per dataset")
    parser.add_argument("--output-dir", default=paths.default_dir("family_bias", "student_solutions"))

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
