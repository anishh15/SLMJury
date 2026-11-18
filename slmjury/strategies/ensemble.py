"""Juries: three SLM judges decide each verdict by majority vote.

Builds every three-judge jury (all C(n,3) combinations) of the top individual judges. The output matches the
individual runs, so scripts/run_strategy_evaluation.py --juries scores it the same way.
"""

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

from slmjury.analysis import paths
from slmjury.configs import load_models_config
from slmjury.core.evaluator import parse_judgement_filename

logger = logging.getLogger(__name__)


def majority_vote(verdicts: list[str]) -> str:
    """"Correct" or "Incorrect" if at least 2 verdicts agree on it, else "Undefined"."""
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
    judgements_dir: Path = Path("results/core/judgements"),
) -> dict[tuple[str, str], list[dict]]:
    """Load one judge's judgement files at one budget, keyed by (student, dataset)."""
    model_dir = judgements_dir / model
    pattern = f"*_t{max_tokens}.json"

    result: dict[tuple[str, str], list[dict]] = {}
    for f in model_dir.glob(pattern):
        student, dataset, _ = parse_judgement_filename(f.name)
        with open(f) as fp:
            result[(student, dataset)] = json.load(fp)

    return result


def combine_judgements(
    judgements_list: list[dict[tuple[str, str], list[dict]]],
    model_names: list[str],
) -> dict[tuple[str, str], list[dict]]:
    """Majority-vote the verdicts of 3 judges for every (student, dataset) key all of them have.

    Each item keeps the first judge's fields and adds ``individual_verdicts``.
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
    """A readable jury name such as 'qwen3-4b-t10+phi4mi-3.8b-t10+qwen2.5-3b-t10'.

    The budgets are added only when ``token_settings`` is given.
    """
    if token_settings:
        parts = [f"{m}-t{token_settings[m]}" for m in models]
    else:
        parts = list(models)
    return "+".join(parts)


def save_ensemble_results(
    combined: dict[tuple[str, str], list[dict]],
    ensemble_name: str,
    output_dir: Path = Path("results/core/juries"),
) -> list[Path]:
    """Save each (student, dataset) entry to ``<output_dir>/<ensemble_name>/<student>_<dataset>.json``; returns the paths."""
    saved = []
    out_dir = output_dir / ensemble_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for (student, dataset), problems in combined.items():
        filename = f"{student}_{dataset}.json"
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
    judgements_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
    """Generate majority-vote juries for all C(5,3) = 10 three-model combinations.

    Reads ``judgements_dir`` and writes ``output_dir`` (default core/judgements and core/juries in the results
    folder). ``python scripts/run_strategy_evaluation.py --juries`` evaluates the results.
    """
    root = paths.results_dir() / "core"
    judgements_dir = Path(judgements_dir) if judgements_dir else root / "judgements"
    output_dir = Path(output_dir) if output_dir else root / "juries"
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
