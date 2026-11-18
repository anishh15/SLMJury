"""Configuration management for SLMJury."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent
MODELS_YAML = CONFIG_DIR / "models.yaml"
PROMPT_VARIANTS_YAML = CONFIG_DIR / "prompt_variants.yaml"


def load_models_config() -> dict:
    """Load the centralized model configuration."""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)


def load_prompt_variants() -> dict:
    """Load the judge prompt variants (``verdict_prompts`` and ``scoring_prompts``)."""
    with open(PROMPT_VARIANTS_YAML) as f:
        return yaml.safe_load(f)


JUDGE_SECTIONS = ("judge_models", "large_judge_models")


def get_judge_config(model_key: str, config: dict | None = None) -> dict:
    """Return a copy of a judge's config, looked up in every judge section.

    Small judges live in ``judge_models``; the large and ladder judges live in
    ``large_judge_models``. Raises ValueError for an unknown key.
    """
    config = config or load_models_config()
    for section in JUDGE_SECTIONS:
        judges = config.get(section) or {}
        if model_key in judges:
            return dict(judges[model_key])
    known = [k for s in JUDGE_SECTIONS for k in (config.get(s) or {})]
    raise ValueError(f"Unknown judge model: {model_key}. Available: {known}")
