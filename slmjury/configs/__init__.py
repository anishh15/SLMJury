"""Configuration management for SLMJury."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent
MODELS_YAML = CONFIG_DIR / "models.yaml"


def load_models_config() -> dict:
    """Load the centralized model configuration."""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)
