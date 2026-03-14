"""GSM-Plus dataset loader.

Downloads and caches the GSM-Plus math reasoning dataset from HuggingFace.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "gsm_plus"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "gsm_plus.json"


def load_gsm_plus(cache_path: Optional[Path] = None) -> list[dict]:
    """Load GSM-Plus dataset, downloading from HuggingFace if not cached.

    Args:
        cache_path: Custom path to the cache JSON file.

    Returns:
        List of problem dicts with keys: problem_id, question,
        ground_truth_reasoning, ground_truth_answer.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached GSM-Plus from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading GSM-Plus from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("qintongli/GSM-Plus", split="test")
    except Exception as e:
        logger.error("Failed to load GSM-Plus: %s", e)
        return []

    problems = []
    for i, item in enumerate(dataset):
        problems.append({
            "problem_id": i,
            "question": item["question"],
            "ground_truth_reasoning": item["solution"],
            "ground_truth_answer": item["answer"],
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info("Cached %d GSM-Plus problems to %s", len(problems), cache_file)
    return problems
