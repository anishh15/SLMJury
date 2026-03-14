"""Dataset loading utilities for SLMJury."""

from slmjury.data.gsm8k import load_gsm8k
from slmjury.data.gsm_plus import load_gsm_plus
from slmjury.data.math import load_math
from slmjury.data.arc_easy import load_arc_easy
from slmjury.data.arc_challenge import load_arc_challenge

DATASET_LOADERS = {
    "gsm8k": load_gsm8k,
    "gsm_plus": load_gsm_plus,
    "math": load_math,
    "arc_easy": load_arc_easy,
    "arc_challenge": load_arc_challenge,
}


def load_dataset(name: str, **kwargs) -> list[dict]:
    """Load a dataset by name. Supported: gsm8k, gsm_plus, math, arc_easy, arc_challenge."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name](**kwargs)


__all__ = [
    "load_dataset",
    "load_gsm8k",
    "load_gsm_plus",
    "load_math",
    "load_arc_easy",
    "load_arc_challenge",
    "DATASET_LOADERS",
]
