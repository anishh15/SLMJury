#!/usr/bin/env python3
"""Single-factor study: split the 10 -> 8,192-token accuracy change into its causes."""

import argparse
import json
import logging
from pathlib import Path

from slmjury.analysis import paths
from slmjury.configs import get_judge_config
from slmjury.core.evaluator import JudgeEvaluator
from slmjury.core.judge import JudgeModel

NOTES = """\
The quick and the reasoned verdict differ in more than the budget. At 10 tokens
the package uses PROMPT_QUICK, greedy decoding and no thinking; at 8,192 it
uses PROMPT_REASONED, and judges with enable_thinking also think and sample at
temperature 0.6. Each factor condition below starts from the 10-token baseline
and changes one of these, through the prompt_style, thinking and temperature
arguments of JudgeModel.evaluate_batch:

  condition            budget  prompt    thinking  temperature
  baseline_t10            10   (auto)    (auto)    (auto)       as scripts/run_judge.py
  baseline_t8192        8192   (auto)    (auto)    (auto)       as scripts/run_judge.py
  factor_budget         8192   quick     off       0
  factor_prompt           10   reasoned  off       0
  factor_thinking       8192   quick     on        0.6
  factor_temperature      10   quick     off       0.6

The two baselines call evaluate_batch with the budget only, so they match the
package's own runs. thinking=True turns thinking on only for judges whose
config enables it. Judges with always_thinks have no 10-token configuration,
so they skip baseline_t10, factor_prompt and factor_temperature.

Existing files are read instead of re-judged unless --force is given; a file
that does not parse is judged again. At the end the script prints, per
dataset, each condition's accuracy and its change from baseline_t10.

Reads results/core/student_solutions/<student>/<dataset>.json
Writes results/token_budget/single_factor/<judge>/<condition>/<student>_<dataset>.json

examples, from the repository root:
  python experiments/token_budget/run_single_factor.py --judges qwen3-8b --factor all
  python experiments/token_budget/run_single_factor.py --judges qwen3-8b --factor thinking

experiments/token_budget/analyze.py reads the results.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]

# The sixteen small judges.
DEFAULT_JUDGES = [
    "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b",
    "llama3.2-1b", "llama3.2-3b", "llama3.1-8b",
    "phi4-14b", "phi4r-14b", "phi4rp-14b", "phi4mi-3.8b", "phi4mr-3.8b",
]

# None leaves the choice to evaluate_batch, which decides from the budget.
CONDITIONS = {
    "baseline_t10": {
        "max_tokens": 10,
        "prompt_style": None,   # PROMPT_QUICK
        "thinking": None,       # off
        "temperature": None,    # 0
        "label": "Baseline B=10",
    },
    "baseline_t8192": {
        "max_tokens": 8192,
        "prompt_style": None,   # PROMPT_REASONED
        "thinking": None,       # on if the judge has enable_thinking
        "temperature": None,    # 0.6 when thinking, else 0
        "label": "Baseline B=8192",
    },
    "factor_budget": {
        "max_tokens": 8192,
        "prompt_style": "quick",
        "thinking": False,
        "temperature": 0,
        "label": "Factor: Budget only",
        "note": "The larger budget alone, with the quick prompt and no thinking",
    },
    "factor_prompt": {
        "max_tokens": 10,
        "prompt_style": "reasoned",
        "thinking": False,
        "temperature": 0,
        "label": "Factor: Prompt-style only",
        "note": "The reasoned prompt alone; the output is still cut at 10 tokens",
    },
    "factor_thinking": {
        "max_tokens": 8192,
        "prompt_style": "quick",
        "thinking": True,
        "temperature": 0.6,     # the package's sampling temperature when thinking
        "label": "Factor: Thinking only",
        "note": "Thinking alone, with the quick prompt and the full budget",
    },
    "factor_temperature": {
        "max_tokens": 10,
        "prompt_style": "quick",
        "thinking": False,
        "temperature": 0.6,
        "label": "Factor: Temperature only",
        "note": "Sampling instead of greedy decoding, at 10 tokens",
    },
}


def run_one_condition(
    judge: JudgeModel,
    judge_key: str,
    condition_name: str,
    condition: dict,
    student: str,
    dataset: str,
    solutions_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> list[dict]:
    """Judge one student x dataset under one condition and save the result."""
    out_dir = output_dir / judge_key / condition_name
    out_file = out_dir / f"{student}_{dataset}.json"

    if out_file.exists() and not force:
        try:
            with open(out_file) as f:
                data = json.load(f)
            logger.info(
                "SKIP %s × %s × %s — already judged",
                judge_key, condition_name, dataset,
            )
            return data
        except (json.JSONDecodeError, ValueError):
            logger.warning("Unreadable JSON file (%s), judging again", out_file)

    sol_file = solutions_dir / student / f"{dataset}.json"
    if not sol_file.exists():
        logger.warning("Solutions not found: %s", sol_file)
        return []

    with open(sol_file) as f:
        student_results = json.load(f)

    logger.info(
        "Judging %s × %s × %s × %s",
        judge_key, student, dataset, condition["label"],
    )

    max_tokens = condition["max_tokens"]
    if condition_name in ("baseline_t10", "baseline_t8192"):
        # The budget only, as scripts/run_judge.py calls it.
        results = judge.evaluate_batch(student_results, max_tokens)
    else:
        results = judge.evaluate_batch(
            student_results,
            max_tokens,
            prompt_style=condition.get("prompt_style"),
            thinking=condition.get("thinking"),
            temperature=condition.get("temperature"),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)   # same format as JudgeModel.save_results
    logger.info("Saved %d results → %s", len(results), out_file)
    return results


def compute_accuracy(
    judge_key: str, condition_name: str, student: str, dataset: str, output_dir: Path,
) -> float | None:
    """Accuracy (%) of one saved condition file, from ``JudgeEvaluator``."""
    out_file = output_dir / judge_key / condition_name / f"{student}_{dataset}.json"
    if not out_file.exists():
        return None

    with open(out_file) as f:
        judgements = json.load(f)

    evaluator = JudgeEvaluator(
        judge_model=judge_key,
        student_model=student,
        dataset=dataset,
        max_tokens=CONDITIONS[condition_name]["max_tokens"],
        judgements=judgements,
    )
    summary = evaluator.evaluate()
    return summary["metrics"]["accuracy"] * 100


def run_factor(
    judge_key: str,
    factor: str,
    datasets: list[str],
    students: list[str],
    solutions_dir: Path,
    output_dir: Path,
    force: bool,
    model_config: dict | None = None,
):
    """Run the conditions selected by ``factor`` for one judge, loaded once."""
    judge_cfg = get_judge_config(judge_key)

    if judge_cfg.get("always_thinks"):
        logger.info("Model %s always thinks — 10-token conditions will be skipped", judge_key)

    if factor == "all":
        conditions_to_run = list(CONDITIONS.items())
    elif factor == "baselines":
        conditions_to_run = [
            ("baseline_t10", CONDITIONS["baseline_t10"]),
            ("baseline_t8192", CONDITIONS["baseline_t8192"]),
        ]
    else:
        conditions_to_run = [
            ("baseline_t10", CONDITIONS["baseline_t10"]),
            (f"factor_{factor}", CONDITIONS[f"factor_{factor}"]),
        ]

    judge = JudgeModel(
        judge_key,
        output_dir=output_dir / judge_key,
        model_config=model_config or None,
    )

    try:
        for condition_name, condition in conditions_to_run:
            if condition["max_tokens"] == 10 and judge_cfg.get("always_thinks"):
                logger.info("Skipping %s (always_thinks model)", condition_name)
                continue

            for student in students:
                for dataset in datasets:
                    run_one_condition(
                        judge, judge_key, condition_name, condition,
                        student, dataset, solutions_dir, output_dir, force=force,
                    )
    finally:
        judge.cleanup()

    # Accuracy per condition and its change from baseline_t10, mean over students.
    print(f"\n=== Single-factor attribution: {judge_key} ===")
    for dataset in datasets:
        b10 = None
        b8192 = None
        for s in students:
            a = compute_accuracy(judge_key, "baseline_t10", s, dataset, output_dir)
            b = compute_accuracy(judge_key, "baseline_t8192", s, dataset, output_dir)
            if a is not None:
                b10 = (b10 or 0) + a
            if b is not None:
                b8192 = (b8192 or 0) + b
        if b10 is not None:
            b10 /= len(students)
        if b8192 is not None:
            b8192 /= len(students)
        total_delta = (b8192 - b10) if (b10 and b8192) else None

        print(f"\n  Dataset: {dataset}")
        if total_delta is not None:
            print(f"  Total Δ (t8192 - t10): {total_delta:+.2f}%")
        for cond_name, _condition in conditions_to_run:
            accs = [compute_accuracy(judge_key, cond_name, s, dataset, output_dir)
                    for s in students]
            accs = [a for a in accs if a is not None]
            avg = sum(accs) / len(accs) if accs else None
            delta = (avg - b10) if (avg is not None and b10 and cond_name != "baseline_t10") else None
            delta_str = f"{delta:+.2f}%" if delta is not None else "—"
            print(f"    {cond_name:<28} acc={avg:.2f}%  Δ={delta_str}" if avg else
                  f"    {cond_name:<28} acc=N/A      Δ=—")


def main():
    parser = argparse.ArgumentParser(
        description="Change one factor of the quick verdict at a time.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES,
                        help="Judge keys from models.yaml (default: the 16 small judges)")
    parser.add_argument(
        "--factor", default="baselines",
        choices=["all", "baselines", "budget", "prompt", "thinking", "temperature"],
        help="Conditions to run: 'all', the two 'baselines' (default), or "
             "baseline_t10 plus one factor",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=[
            "arc_challenge", "arc_easy", "gsm8k", "gsm_plus",
            "hellaswag", "math", "truthfulqa", "winogrande",
        ],
        help="Closed-ended datasets (default: all 8)",
    )
    parser.add_argument("--students", nargs="+", default=STUDENTS)
    parser.add_argument(
        "--solutions-dir", default=paths.default_dir("core", "student_solutions"),
        help="Directory containing student solution files",
    )
    parser.add_argument(
        "--output-dir", default=paths.default_dir("token_budget", "single_factor"),
        help="Output directory; sub-folders per judge and condition are created",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-judge even if result files already exist")

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

    model_config = {}
    if args.tensor_parallel_size is not None:
        model_config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        model_config["max_num_seqs"] = args.max_num_seqs

    for judge_key in args.judges:
        run_factor(
            judge_key, args.factor, args.datasets, args.students,
            Path(args.solutions_dir), Path(args.output_dir), args.force,
            model_config=model_config or None,
        )


if __name__ == "__main__":
    main()
