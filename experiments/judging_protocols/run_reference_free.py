#!/usr/bin/env python3
"""Judge the students without showing the judge the ground truth."""

import argparse
import json
import logging
from pathlib import Path

from datasets_order import ordered_solution_files

from slmjury.analysis import paths
from slmjury.configs import get_judge_config
from slmjury.core.judge import JudgeModel

NOTES = """\
Same as scripts/run_judge.py except for the prompt. PROMPT_REFERENCEFREE_QUICK
and PROMPT_REFERENCEFREE_REASONED are PROMPT_QUICK and PROMPT_REASONED with the
ground-truth line removed and the instructions reworded to match. They are
passed to JudgeModel.evaluate_batch through prompt_template: the quick text at
10 tokens and the reasoned text otherwise, the same way the package chooses its
own prompts. Model loading, sampling, verdict parsing and the output format are
the package's. Judges with always_thinks skip 10 tokens.

The judge is looked up in judge_models and then large_judge_models in
slmjury/configs/models.yaml, so the same command runs small and large judges.
Datasets are visited in the order of datasets_order.py. Existing files are
skipped unless --force is given.

Reads --solutions-dir/<student>/<dataset>.json (default results/core/student_solutions)
Writes --output-dir/<judge>/<student>_<dataset>_t<budget>.json
  (default results/judging_protocols/referencefree)

example, from the repository root:
  python experiments/judging_protocols/run_reference_free.py --judge qwen3-4b --max-tokens 10 8192

experiments/judging_protocols/analyze.py reads the results.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# PROMPT_QUICK and PROMPT_REASONED without the ground truth.
PROMPT_REFERENCEFREE_QUICK = (
    "Your role is to evaluate the student's answer to the question.\n\n"
    "Respond with exactly one word:\n"
    "- 'Correct': If the student's final answer is correct.\n"
    "- 'Incorrect': If the student's final answer is incorrect.\n\n"
    "Focus only on the final answer, ignore reasoning steps.\n\n"
    "Question: {question}\n\n"
    "Student answer: {student_answer}"
)

PROMPT_REFERENCEFREE_REASONED = (
    "Your role is to evaluate the student's answer to the question.\n\n"
    "First, explain your reasoning for the judgment.\n"
    "Then, on a new line, give the final verdict in the exact format:\n"
    "\\boxed{{CORRECT}} or \\boxed{{INCORRECT}}\n\n"
    "Focus only on the final answer when judging correctness. "
    "Ignore reasoning steps in the student's solution.\n\n"
    "Question: {question}\n\n"
    "Student answer: {student_answer}\n"
)


def main():
    parser = argparse.ArgumentParser(
        description="Reference-free judging: no ground truth in the prompt.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--judge", required=True, help="Judge model key from models.yaml")
    parser.add_argument(
        "--max-tokens", type=int, nargs="+", default=[10, 8192],
        help="Token settings to evaluate (default: 10 8192)",
    )
    parser.add_argument(
        "--solutions-dir", default=paths.default_dir("core", "student_solutions"),
        help="Directory containing student solution files",
    )
    parser.add_argument(
        "--output-dir", default=paths.default_dir("judging_protocols", "referencefree"),
        help="Output directory; a sub-folder per judge is created",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-judge even if result files already exist")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Judge only these datasets (default: all eight; overrides --exclude-datasets)",
    )
    # Older spelling, kept working but not shown in --help.
    parser.add_argument("--only-datasets", dest="datasets", nargs="*",
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--exclude-datasets", nargs="*", default=[],
        help="Dataset names to skip (default: none). Ignored with --datasets",
    )

    # Hardware overrides, as in scripts/run_judge.py
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

    only_datasets = set(args.datasets) if args.datasets else None
    exclude_datasets = set(args.exclude_datasets)
    output_dir = Path(args.output_dir)

    judge_cfg = get_judge_config(args.judge)

    token_settings = args.max_tokens
    if judge_cfg.get("always_thinks"):
        token_settings = [t for t in token_settings if t != 10]
        logger.info("Model %s always thinks — skipping max_tokens=10", args.judge)

    model_config = {}
    if args.tensor_parallel_size is not None:
        model_config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        model_config["max_num_seqs"] = args.max_num_seqs

    # save_results adds the /<judge> folder itself.
    judge = JudgeModel(
        args.judge,
        output_dir=output_dir,
        model_config=model_config or None,
    )

    solutions_dir = Path(args.solutions_dir)
    skipped = 0

    for student_dir in sorted(solutions_dir.iterdir()):
        if not student_dir.is_dir():
            continue
        student_model = student_dir.name

        for solution_file in ordered_solution_files(student_dir):
            dataset = solution_file.stem
            if only_datasets is not None:
                if dataset not in only_datasets:
                    continue
            elif dataset in exclude_datasets:
                continue

            for max_tokens in token_settings:
                out_file = (
                    output_dir / args.judge
                    / f"{student_model}_{dataset}_t{max_tokens}.json"
                )
                if out_file.exists() and not args.force:
                    logger.info(
                        "SKIP %s × %s (tokens=%d) — already judged",
                        student_model, dataset, max_tokens,
                    )
                    skipped += 1
                    continue

                with open(solution_file) as f:
                    student_results = json.load(f)

                template = (PROMPT_REFERENCEFREE_QUICK if max_tokens == 10
                            else PROMPT_REFERENCEFREE_REASONED)
                logger.info(
                    "Judging %s × %s (tokens=%d, ref-free)",
                    student_model, dataset, max_tokens,
                )
                results = judge.evaluate_batch(
                    student_results, max_tokens, prompt_template=template,
                )
                judge.save_results(results, student_model, dataset, max_tokens)

    judge.cleanup()
    if skipped:
        logger.info("Skipped %d already-judged combinations.", skipped)
    logger.info("Reference-free judging complete.")


if __name__ == "__main__":
    main()
