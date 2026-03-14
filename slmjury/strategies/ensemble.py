"""Majority voting ensemble — combines judgements from multiple SLM judges.

Generates all C(n,3) three-model combinations from top individual judges and
majority-votes each problem's verdict. Output format matches individual runs for
evaluation reuse.
"""

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

from slmjury.configs import load_models_config
from slmjury.core.evaluator import parse_judgement_filename

logger = logging.getLogger(__name__)


def majority_vote(verdicts: list[str]) -> str:
    """Determine the majority verdict from a list of judgements.

    Args:
        verdicts: List of verdict strings ("Correct", "Incorrect", or "Undefined").

    Returns:
        "Correct" if ≥2 agree correct, "Incorrect" if ≥2 agree incorrect,
        otherwise "Undefined".
    """
    correct_count = sum(1 for v in verdicts if v == "Correct")
    incorrect_count = sum(1 for v in verdicts if v == "Incorrect")

    if correct_count >= 2:
        return "Correct"
    elif incorrect_count >= 2:
        return "Incorrect"
    return "Undefined"


def load_judgements_by_key(
    model: str,
    max_tokens: int,
    judgements_dir: Path = Path("results/judgements"),
) -> dict[tuple[str, str], list[dict]]:
    """Load all judgement files for a model/token combination.

    Args:
        model: Judge model key.
        max_tokens: Token setting.
        judgements_dir: Base directory for individual judgements.

    Returns:
        Dict keyed by (student_model, dataset) → list of judgement dicts.
    """
    model_dir = judgements_dir / model
    pattern = f"{model}_*_tokens-{max_tokens}.json"

    result: dict[tuple[str, str], list[dict]] = {}
    for f in model_dir.glob(pattern):
        _, student, dataset, _ = parse_judgement_filename(f.name)
        with open(f) as fp:
            result[(student, dataset)] = json.load(fp)

    return result


def combine_judgements(
    judgements_list: list[dict[tuple[str, str], list[dict]]],
    model_names: list[str],
) -> dict[tuple[str, str], list[dict]]:
    """Combine judgements from 3 models via majority voting.

    Args:
        judgements_list: List of 3 dicts (one per model), each keyed by
            (student, dataset).
        model_names: List of 3 model key strings.

    Returns:
        Combined dict with majority-voted verdicts.
    """
    combined: dict[tuple[str, str], list[dict]] = {}

    for key in judgements_list[0]:
        if any(key not in j for j in judgements_list[1:]):
            logger.warning("Skipping %s — not present in all models", key)
            continue

        problems = judgements_list[0][key]
        combined_problems = []

        for i, problem in enumerate(problems):
            verdicts = [j[key][i]["judgement"] for j in judgements_list]
            result = {**problem}
            result["judgement"] = majority_vote(verdicts)
            result["individual_verdicts"] = {
                name: j[key][i]["judgement"]
                for name, j in zip(model_names, judgements_list)
            }
            combined_problems.append(result)

        combined[key] = combined_problems

    return combined


def make_ensemble_name(
    models: list[str],
    token_settings: Optional[dict[str, int]] = None,
) -> str:
    """Create a readable ensemble name.

    Args:
        models: List of model keys.
        token_settings: Optional dict mapping model key → token setting.

    Returns:
        Name like 'qwen3-4b-t10+phi4mi-3.8b-t10+qwen2.5-3b-t10'.
    """
    if token_settings:
        parts = [f"{m}-t{token_settings[m]}" for m in models]
    else:
        parts = list(models)
    return "+".join(parts)


def save_ensemble_results(
    combined: dict[tuple[str, str], list[dict]],
    ensemble_name: str,
    output_dir: Path = Path("results/majority_voting"),
) -> list[Path]:
    """Save combined judgements per (student, dataset) pair.

    Args:
        combined: Combined judgement dict from combine_judgements.
        ensemble_name: Name for the ensemble subdirectory.
        output_dir: Base output directory.

    Returns:
        List of saved file paths.
    """
    saved = []
    out_dir = output_dir / ensemble_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for (student, dataset), problems in combined.items():
        filename = f"{ensemble_name}_{student}_{dataset}.json"
        out_file = out_dir / filename
        with open(out_file, "w") as f:
            json.dump(problems, f, indent=4)
        saved.append(out_file)

    logger.info("Saved %d ensemble files to %s", len(saved), out_dir)
    return saved


def get_ensemble_judges() -> dict[str, int]:
    """Load ensemble judge config from models.yaml.

    Returns:
        Dict mapping model key → best token setting.
    """
    config = load_models_config()
    judges = config.get("ensemble_judges", [])
    return {j["key"]: j["tokens"] for j in judges}


def generate_all_ensembles(
    judgements_dir: Path = Path("results/judgements"),
    output_dir: Path = Path("results/majority_voting"),
):
    """Generate majority-vote ensembles for all C(5,3)=10 three-model combinations.

    Args:
        judgements_dir: Directory containing individual judgement files.
        output_dir: Directory for saving ensemble results.
    """
    token_settings = get_ensemble_judges()
    models = list(token_settings.keys())
    all_combos = list(combinations(models, 3))

    logger.info(
        "Generating %d ensembles from models: %s",
        len(all_combos), ", ".join(models),
    )

    # Pre-load all model judgements
    all_judgements = {}
    for model in models:
        tokens = token_settings[model]
        all_judgements[model] = load_judgements_by_key(model, tokens, judgements_dir)
        logger.info(
            "Loaded %s (tokens=%d): %d combinations",
            model, tokens, len(all_judgements[model]),
        )

    for i, combo in enumerate(all_combos, 1):
        ensemble_name = make_ensemble_name(list(combo), token_settings)
        logger.info("[%d/%d] %s", i, len(all_combos), ensemble_name)

        judgements_list = [all_judgements[m] for m in combo]
        combined = combine_judgements(judgements_list, list(combo))
        save_ensemble_results(combined, ensemble_name, output_dir)

    logger.info("All %d ensembles generated.", len(all_combos))
