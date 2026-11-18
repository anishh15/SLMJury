#!/usr/bin/env python3
"""Run a judge with one reworded prompt from slmjury/configs/prompt_variants.yaml."""

import argparse
import json
import logging
from pathlib import Path

from slmjury.analysis import paths
from slmjury.configs import get_judge_config, load_prompt_variants
from slmjury.core.judge import JudgeModel
from slmjury.core.scoring_evaluator import (
    evaluate_mtbench,
    evaluate_summeval,
    save_evaluation,
)
from slmjury.core.scoring_judge import ScoringJudge
from slmjury.data import load_dataset

NOTES = """\
The judge, students, items, sampling and verdict parsing are the same as in
scripts/run_judge.py and scripts/run_scoring_judge.py. Only the prompt text
changes. The variant is passed to the package through the optional
prompt_template argument of JudgeModel.evaluate_batch,
ScoringJudge.score_summeval and ScoringJudge.score_mtbench. The original
variant passes nothing, so it runs the package's default prompts.

Two prompt types:
  verdict   closed-ended; the "quick" text is used at 10 tokens and the
            "reasoned" text at 8,192, as the package chooses between
            PROMPT_QUICK and PROMPT_REASONED. Judges with always_thinks
            skip 10 tokens.
  scoring   SummEval and MT-Bench, scored against every reference-model file
            under --reference-scores (layout <reference>/<student>.json)

Writes, under --output-dir/<variant>/ (default results/prompt_sensitivity):
  <judge>/<student>_<dataset>_t<budget>.json         verdicts
  <judge>/summeval.json, <judge>/mtbench_<reference>_<student>.json
  scoring_evaluations/<judge>/{summeval,mtbench_<reference>_<student>}_eval.json

Existing files are skipped unless --force is given.

examples, from the repository root:
  python experiments/prompt_sensitivity/run_prompt_variants.py \\
      --judge qwen3-4b --variant minimal --max-tokens 10 8192
  python experiments/prompt_sensitivity/run_prompt_variants.py \\
      --judge qwen3-4b --variant minimal --prompt-type scoring --max-tokens 8192
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VARIANTS = ["original", "minimal", "gpt_style", "rubric_based", "cot_style",
            "adversarial", "few_shot"]


def variant_prompt(variant: str, style: str, prompt_type: str) -> str | None:
    """The prompt text for one variant, or None for ``original``.

    ``style`` is "quick" or "reasoned" for verdicts and "summeval" or "mtbench"
    for scoring; ``prompt_type`` is "verdict" or "scoring".
    """
    if variant == "original":
        return None
    variants = load_prompt_variants()
    section = variants[f"{prompt_type}_prompts"]
    if variant not in section:
        raise ValueError(
            f"Variant '{variant}' not found in {prompt_type}_prompts. "
            f"Available: {list(section)}"
        )
    entry = section[variant]
    if style not in entry:
        raise ValueError(f"Style '{style}' not found in variant '{variant}'")
    return entry[style]


# ── Verdicts (closed-ended) ─────────────────────────────────────────────────

def run_verdict(args, judge_cfg, model_config):
    """Judge every student solution file; same loop as scripts/run_judge.py."""
    token_settings = args.max_tokens
    if judge_cfg.get("always_thinks"):
        token_settings = [t for t in token_settings if t != 10]
        logger.info("Model %s always thinks — skipping max_tokens=10", args.judge)

    output_dir = Path(args.output_dir) / args.variant
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

        for solution_file in sorted(student_dir.glob("*.json")):
            dataset = solution_file.stem

            if args.datasets and dataset not in args.datasets:
                continue

            for max_tokens in token_settings:
                out_file = (
                    output_dir / args.judge
                    / f"{student_model}_{dataset}_t{max_tokens}.json"
                )
                if out_file.exists() and not args.force:
                    logger.info(
                        "SKIP %s × %s (tokens=%d, variant=%s) — already judged",
                        student_model, dataset, max_tokens, args.variant,
                    )
                    skipped += 1
                    continue

                with open(solution_file) as f:
                    student_results = json.load(f)

                style = "quick" if max_tokens == 10 else "reasoned"
                logger.info(
                    "Judging %s × %s (tokens=%d, variant=%s)",
                    student_model, dataset, max_tokens, args.variant,
                )
                results = judge.evaluate_batch(
                    student_results, max_tokens,
                    prompt_template=variant_prompt(args.variant, style, "verdict"),
                )
                judge.save_results(results, student_model, dataset, max_tokens)

    # Keep the weights in the HF cache: several variants of one judge often
    # run at the same time and share it.
    judge.cleanup(clear_hf_cache=False)
    if skipped:
        logger.info(
            "Skipped %d already-judged combinations (use --force to re-judge).",
            skipped,
        )
    logger.info("Verdict run complete.")


# ── Scoring (open-ended) ────────────────────────────────────────────────────

def run_scoring(args, judge_cfg, model_config):
    """Score SummEval and MT-Bench; same steps as scripts/run_scoring_judge.py."""
    output_dir = Path(args.output_dir) / args.variant
    eval_dir = output_dir / "scoring_evaluations"
    max_tokens = args.max_tokens[0]

    summeval_out = output_dir / args.judge / "summeval.json"
    summeval_done = summeval_out.exists() and not args.force
    reference_dir = Path(args.reference_scores)

    judge = ScoringJudge(
        args.judge,
        output_dir=output_dir,
        model_config=model_config or None,
    )

    # ── SummEval ───────────────────────────────────────────────────────────
    if summeval_done:
        logger.info(
            "SKIP %s × summeval (variant=%s) — already scored",
            args.judge, args.variant,
        )
    else:
        data = load_dataset("summeval")
        logger.info("Loaded %d SummEval pairs", len(data))
        logger.info("Scoring summeval (variant=%s)", args.variant)
        results = judge.score_summeval(
            data, max_tokens=max_tokens,
            prompt_template=variant_prompt(args.variant, "summeval", "scoring"),
        )
        judge.save_results(results, "summeval")

        eval_result = evaluate_summeval(results)
        save_evaluation(eval_result, args.judge, "summeval", eval_dir)

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

    # ── MT-Bench ───────────────────────────────────────────────────────────
    if not reference_dir.is_dir():
        logger.warning(
            "MT-Bench reference scores not found (%s) — skipping MT-Bench scoring",
            reference_dir,
        )
    else:
        reference_files = sorted(reference_dir.rglob("*.json"))
        if not reference_files:
            logger.warning("No reference-score JSON files found in %s", reference_dir)
        else:
            logger.info(
                "Scoring %d MT-Bench reference file(s) (variant=%s)",
                len(reference_files), args.variant,
            )
            base_data = load_dataset("mtbench")
            template = variant_prompt(args.variant, "mtbench", "scoring")

            for ref_file in reference_files:
                # <reference>/<student>.json -> suffix <reference>_<student>
                suffix = f"{ref_file.parent.name}_{ref_file.stem}"

                out_file = output_dir / args.judge / f"mtbench_{suffix}.json"
                if out_file.exists() and not args.force:
                    logger.info("SKIP %s — already scored", out_file.name)
                    continue

                with open(ref_file) as f:
                    reference_data = json.load(f)

                # Attach the student responses and reference scores to the
                # MT-Bench questions, as scripts/run_scoring_judge.py does.
                data = [dict(item) for item in base_data]
                for item, ref in zip(data, reference_data):
                    item["turn1_question"] = item["turn1"]
                    item["turn1_response"] = ref["turn1_response"]
                    item["turn2_question"] = item["turn2"]
                    item["turn2_response"] = ref["turn2_response"]
                    item["oracle_score"] = ref["oracle_score"]
                    item["question_id"] = item.get("question_id", item["problem_id"])

                logger.info("Scoring mtbench %s (variant=%s)", suffix, args.variant)
                results = judge.score_mtbench(
                    data, max_tokens=max_tokens, prompt_template=template,
                )
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

    judge.cleanup(clear_hf_cache=False)   # shared HF cache, see run_verdict()
    logger.info("Scoring run complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Run a judge with a reworded prompt from prompt_variants.yaml.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--judge", required=True, help="Judge model key from models.yaml")
    parser.add_argument(
        "--variant", required=True, choices=VARIANTS,
        help="Prompt variant from slmjury/configs/prompt_variants.yaml",
    )
    parser.add_argument(
        "--prompt-type", default="verdict", choices=["verdict", "scoring"],
        help="'verdict' for closed-ended, 'scoring' for SummEval/MT-Bench (default: verdict)",
    )
    parser.add_argument(
        "--max-tokens", type=int, nargs="+", default=None,
        help="Token settings to evaluate (default: 10 8192 for verdict, "
             "8192 for scoring). Scoring uses the first value.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Only judge these datasets (default: every dataset in --solutions-dir)",
    )
    parser.add_argument(
        "--solutions-dir", default=paths.default_dir("core", "student_solutions"),
        help="Directory containing student solution files",
    )
    parser.add_argument(
        "--reference-scores", default=paths.default_dir("core", "mtbench_references"),
        help="Directory of MT-Bench reference scores, <reference>/<student>.json",
    )
    # Older spelling, kept working but not shown in --help.
    parser.add_argument("--oracle-scores", dest="reference_scores",
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir", default=paths.default_dir("prompt_sensitivity"),
        help="Base output directory; a sub-folder per variant is created",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if result files already exist",
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
    if args.max_tokens is None:
        args.max_tokens = [10, 8192] if args.prompt_type == "verdict" else [8192]

    judge_cfg = get_judge_config(args.judge)

    model_config = {}
    if args.tensor_parallel_size is not None:
        model_config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        model_config["max_num_seqs"] = args.max_num_seqs

    if args.prompt_type == "verdict":
        run_verdict(args, judge_cfg, model_config)
    else:
        run_scoring(args, judge_cfg, model_config)


if __name__ == "__main__":
    main()
