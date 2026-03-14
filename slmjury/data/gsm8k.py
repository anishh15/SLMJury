"""GSM8K dataset loader.

Downloads and caches the GSM8K math reasoning dataset from HuggingFace.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "gsm8k"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "gsm8k.json"


def load_gsm8k(cache_path: Optional[Path] = None) -> list[dict]:
    """Load GSM8K dataset, downloading from HuggingFace if not cached.

    Args:
        cache_path: Custom path to the cache JSON file.

    Returns:
        List of problem dicts with keys: problem_id, question,
        ground_truth_reasoning, ground_truth_answer.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached GSM8K from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading GSM8K from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        logger.error("Failed to load GSM8K: %s", e)
        return []

    problems = []
    for i, item in enumerate(dataset):
        full_answer = item["answer"]
        final_answer = (
            full_answer.split("####")[-1].strip()
            if "####" in full_answer
            else full_answer
        )
        problems.append({
            "problem_id": i,
            "question": item["question"],
            "ground_truth_reasoning": full_answer,
            "ground_truth_answer": final_answer,
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info("Cached %d GSM8K problems to %s", len(problems), cache_file)
    return problems
