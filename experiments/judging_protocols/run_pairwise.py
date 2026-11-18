#!/usr/bin/env python3
"""Pairwise judging in both answer orders, to measure position bias."""

import argparse
import json
import logging
from pathlib import Path

from datasets_order import ordered_solution_files

from slmjury.analysis import paths
from slmjury.analysis.metrics import position_bias_flip_rate
from slmjury.analysis.pairwise import read_choice
from slmjury.configs import get_judge_config
from slmjury.core.judge import JudgeModel

NOTES = """\
The judge sees one question, its ground truth and two students' answers, and
answers "A" or "B" for the more correct one. Every pair is judged twice: in
the forward run student a's answer is A, in the reverse run it is B. A judge
without position bias picks the same student both times; the share of items
where its pick follows the position instead is the flip rate.

PROMPT_PAIRWISE is filled in here and passed through JudgeModel.evaluate_batch
with the template "{student_answer}", which forwards the filled prompt
unchanged, so model loading and sampling are the package's. The choice is read
from each response with slmjury.analysis.pairwise.read_choice. Judges with
always_thinks skip 10 tokens.

By default every ordered pair of students (a, b) with a != b is run, which
gives both "a_vs_b" and "b_vs_a" files. --unordered-pairs runs each pair once.
Because (a, b) reversed shows the answers in the same order as (b, a) forward,
this halves the cost and loses no prompt.

The judge is looked up in judge_models and then large_judge_models in
slmjury/configs/models.yaml, so the same command runs small and large judges.
Datasets are visited in the order of datasets_order.py. A pair is skipped when
both of its files exist, unless --force is given.

Reads --solutions-dir/<student>/<dataset>.json (default results/core/student_solutions)
Writes --output-dir/<judge>/<a>_vs_<b>_<dataset>_t<budget>_{fwd,rev}.json
  (default results/judging_protocols/pairwise)

example, from the repository root:
  python experiments/judging_protocols/run_pairwise.py --judge qwen3-4b --max-tokens 10 8192

experiments/judging_protocols/analyze.py reads the results.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Kept short so the wording itself adds as little as possible.
PROMPT_PAIRWISE = (
    "You are evaluating two student answers to a question.\n\n"
    "Decide which answer is more correct.\n"
    "Respond with exactly 'A' or 'B'.\n\n"
    "Question: {question}\n\n"
    "Ground truth: {ground_truth}\n\n"
    "Answer A: {answer_a}\n\n"
    "Answer B: {answer_b}"
)

# Template for evaluate_batch that forwards the already filled pairwise prompt.
PASSTHROUGH = "{student_answer}"


def build_pairwise_items(
    sol_a: list[dict],
    sol_b: list[dict],
    student_a: str,
    student_b: str,
    reversed_order: bool = False,
) -> list[dict]:
    """One item per problem, with the two answers in forward or reversed order."""
    items = []
    for a, b in zip(sol_a, sol_b):
        assert a["problem_id"] == b["problem_id"], \
            f"Mismatched problem IDs: {a['problem_id']} vs {b['problem_id']}"
        if reversed_order:
            first_ans = b["student_reasoning"]
            second_ans = a["student_reasoning"]
            first_student = student_b
            second_student = student_a
        else:
            first_ans = a["student_reasoning"]
            second_ans = b["student_reasoning"]
            first_student = student_a
            second_student = student_b

        items.append({
            "problem_id": a["problem_id"],
            "question": a["question"],
            "ground_truth_reasoning": a["ground_truth_reasoning"],
            "ground_truth_answer": a["ground_truth_answer"],
            "student_a": first_student,
            "student_b": second_student,
            "answer_a": first_ans,
            "answer_b": second_ans,
            "reversed": reversed_order,
        })
    return items


def run_pairwise_batch(
    judge: JudgeModel,
    items: list[dict],
    max_tokens: int,
) -> list[dict]:
    """Judge a list of pairwise items and read the A/B choice from each response."""
    # Each record carries its filled prompt as student_reasoning, which the
    # passthrough template sends to the judge unchanged.
    pseudo = []
    for item in items:
        prompt = PROMPT_PAIRWISE.format(
            question=item["question"],
            ground_truth=item["ground_truth_answer"],
            answer_a=item["answer_a"],
            answer_b=item["answer_b"],
        )
        pseudo.append({
            "problem_id": item["problem_id"],
            "question": item["question"],
            "ground_truth_reasoning": item["ground_truth_reasoning"],
            "ground_truth_answer": item["ground_truth_answer"],
            "student_reasoning": prompt,
            "student_answer": None,
        })

    raw_results = judge.evaluate_batch(pseudo, max_tokens, prompt_template=PASSTHROUGH)

    results = []
    for item, raw in zip(items, raw_results):
        results.append({
            "problem_id": item["problem_id"],
            "question": item["question"],
            "ground_truth_answer": item["ground_truth_answer"],
            "student_a": item["student_a"],
            "student_b": item["student_b"],
            "reversed": item["reversed"],
            "judge_response": raw["judge_response"],
            "verdict": read_choice(raw["judge_response"]),   # "A", "B" or "Undefined"
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise judging in both answer orders (position bias).",
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
        "--output-dir", default=paths.default_dir("judging_protocols", "pairwise"),
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
    parser.add_argument(
        "--unordered-pairs", action="store_true",
        help="Run each unordered student pair once instead of both (a,b) and "
             "(b,a). (a,b) reversed and (b,a) forward show the answers in the "
             "same order, so this halves the cost and loses no prompt. "
             "Default off: both orders, as in the reported results.",
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

    judge = JudgeModel(
        args.judge,
        output_dir=output_dir,
        model_config=model_config or None,
    )

    solutions_dir = Path(args.solutions_dir)
    student_dirs = sorted(d for d in solutions_dir.iterdir() if d.is_dir())

    if args.unordered_pairs:
        pairs = [
            (student_dirs[i], student_dirs[j])
            for i in range(len(student_dirs))
            for j in range(i + 1, len(student_dirs))
        ]
    else:
        pairs = [
            (student_dirs[i], student_dirs[j])
            for i in range(len(student_dirs))
            for j in range(len(student_dirs))
            if i != j
        ]
    logger.info(
        "Pairwise: %d %s pairs from %d students",
        len(pairs), "unordered" if args.unordered_pairs else "ordered",
        len(student_dirs),
    )

    for dir_a, dir_b in pairs:
        student_a = dir_a.name
        student_b = dir_b.name

        for sol_file_a in ordered_solution_files(dir_a):
            dataset = sol_file_a.stem
            if only_datasets is not None:
                if dataset not in only_datasets:
                    continue
            elif dataset in exclude_datasets:
                continue
            sol_file_b = dir_b / sol_file_a.name
            if not sol_file_b.exists():
                continue

            for max_tokens in token_settings:
                stem = f"{student_a}_vs_{student_b}_{dataset}_t{max_tokens}"
                fwd_out = output_dir / args.judge / f"{stem}_fwd.json"
                rev_out = output_dir / args.judge / f"{stem}_rev.json"

                if fwd_out.exists() and rev_out.exists() and not args.force:
                    logger.info(
                        "SKIP %s vs %s × %s (t=%d)",
                        student_a, student_b, dataset, max_tokens,
                    )
                    continue

                with open(sol_file_a) as f:
                    sol_a = json.load(f)
                with open(sol_file_b) as f:
                    sol_b = json.load(f)

                logger.info(
                    "Pairwise: %s vs %s × %s (t=%d)", student_a, student_b, dataset, max_tokens,
                )
                fwd_items = build_pairwise_items(sol_a, sol_b, student_a, student_b,
                                                 reversed_order=False)
                rev_items = build_pairwise_items(sol_a, sol_b, student_a, student_b,
                                                 reversed_order=True)

                fwd_results = run_pairwise_batch(judge, fwd_items, max_tokens)
                rev_results = run_pairwise_batch(judge, rev_items, max_tokens)

                fwd_out.parent.mkdir(parents=True, exist_ok=True)
                with open(fwd_out, "w") as f:
                    json.dump(fwd_results, f, indent=4)
                with open(rev_out, "w") as f:
                    json.dump(rev_results, f, indent=4)

                flip = position_bias_flip_rate(fwd_results, rev_results, verdict_key="verdict")
                logger.info(
                    "  Flip rate: %.3f  (%.1f%% position-biased)",
                    flip, flip * 100,
                )

    judge.cleanup()
    logger.info("Pairwise judging complete.")


if __name__ == "__main__":
    main()
