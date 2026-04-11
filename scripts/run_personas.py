"""CLI entry-point: run persona effect evaluation for a judge model.

Evaluates a judge model with all 6 persona system prompts across all
student-dataset combinations and both token settings.

Usage:
    # Run all personas for a single judge model
    CUDA_VISIBLE_DEVICES=2,3 python scripts/run_personas.py --judge qwen3-4b

    # Specific personas and token settings
    python scripts/run_personas.py --judge qwen3-4b --personas Strict Lenient --max-tokens 10

    # All 5 persona models at once (sequentially)
    python scripts/run_personas.py --judge all
"""

import argparse
import json
import logging
from pathlib import Path

from slmjury.configs import load_models_config
from slmjury.core.judge import JudgeModel
from slmjury.strategies.persona import get_personas, run_persona_evaluation

# The 5 models used for persona experiments
PERSONA_MODELS = ["qwen3-4b", "phi4mi-3.8b", "llama3.1-8b", "qwen3-14b", "phi4-14b"]


def main():
    parser = argparse.ArgumentParser(
        description="Run persona effect evaluation for judge models.",
    )
    parser.add_argument(
        "--judge", required=True,
        help="Judge model key, or 'all' to run all 5 persona models",
    )
    parser.add_argument(
        "--personas", nargs="+", default=None,
        help="Persona names to run (default: all 6)",
    )
    parser.add_argument(
        "--max-tokens", type=int, nargs="+", default=[10, 8192],
        help="Token settings (default: 10 8192)",
    )
    parser.add_argument(
        "--solutions-dir", default="results/student_solutions",
        help="Directory containing student solution files",
    )
    parser.add_argument("--output-dir", default="results/persona_effect")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if persona result files already exist",
    )

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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = load_models_config()
    all_personas = get_personas()
    solutions_dir = Path(args.solutions_dir)
    output_dir = Path(args.output_dir)

    # Determine which personas to run
    if args.personas:
        personas = {k: all_personas[k] for k in args.personas}
    else:
        personas = all_personas

    # Determine which models to run
    if args.judge.lower() == "all":
        judge_models = PERSONA_MODELS
    else:
        judge_models = [args.judge]

    # Build model_config override from CLI args
    model_config = {}
    if args.tensor_parallel_size is not None:
        model_config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization

    for judge_key in judge_models:
        judge_cfg = config["judge_models"].get(judge_key, {})

        # Skip max_tokens=10 for always-thinks models
        token_settings = list(args.max_tokens)
        if judge_cfg.get("always_thinks"):
            token_settings = [t for t in token_settings if t != 10]
            logger.info(
                "Model %s always thinks — skipping max_tokens=10", judge_key,
            )

        logger.info(
            "Loading judge model: %s (%d personas × %d token settings)",
            judge_key, len(personas), len(token_settings),
        )
        judge = JudgeModel(judge_key, model_config=model_config or None)

        skipped = 0
        for student_dir in sorted(solutions_dir.iterdir()):
            if not student_dir.is_dir():
                continue
            student_model = student_dir.name

            for solution_file in sorted(student_dir.glob("*.json")):
                dataset = solution_file.stem

                for max_tokens in token_settings:
                    for persona_name, persona_prompt in personas.items():
                        # Skip if result file already exists (unless --force)
                        out_file = (
                            output_dir / judge_key
                            / f"{persona_name}_{student_model}_{dataset}_t{max_tokens}.json"
                        )
                        if out_file.exists() and not args.force:
                            logger.info(
                                "SKIP %s + %s: %s × %s (tokens=%d) — already exists",
                                judge_key, persona_name,
                                student_model, dataset, max_tokens,
                            )
                            skipped += 1
                            continue

                        with open(solution_file) as f:
                            student_results = json.load(f)

                        logger.info(
                            "%s + %s: %s × %s (tokens=%d, n=%d)",
                            judge_key, persona_name,
                            student_model, dataset,
                            max_tokens, len(student_results),
                        )
                        run_persona_evaluation(
                            judge_model_key=judge_key,
                            student_results=student_results,
                            student_model=student_model,
                            dataset=dataset,
                            max_tokens=max_tokens,
                            persona_name=persona_name,
                            persona_prompt=persona_prompt,
                            judge=judge,
                            output_dir=output_dir,
                        )

        judge.cleanup()
        if skipped:
            logger.info("Skipped %d already-completed persona runs (use --force to re-run).", skipped)
        logger.info("Completed all personas for %s", judge_key)

    logger.info("All persona evaluations complete.")


if __name__ == "__main__":
    main()
