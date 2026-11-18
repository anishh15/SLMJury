"""ARC-Challenge dataset loader.

Downloads and caches the ARC-Challenge multiple choice dataset from HuggingFace.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "arc_challenge"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "arc_challenge.json"


def load_arc_challenge(cache_path: Optional[Path] = None) -> list[dict]:
    """Load ARC-Challenge, downloading it from Hugging Face if not cached.

    Returns problem dicts with problem_id, question, ground_truth_reasoning and
    ground_truth_answer. ``cache_path`` overrides the cache JSON file.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached ARC-Challenge from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading ARC-Challenge from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    except Exception as e:
        logger.error("Failed to load ARC-Challenge: %s", e)
        return []

    problems = []
    for i, item in enumerate(dataset):
        choices = item["choices"]
        choices_text = "  ".join(
            f"{label}) {text}"
            for label, text in zip(choices["label"], choices["text"])
        )
        problems.append({
            "problem_id": i,
            "question": f"{item['question']}\n{choices_text}",
            "ground_truth_reasoning": "",
            "ground_truth_answer": item["answerKey"],
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info("Cached %d ARC-Challenge problems to %s", len(problems), cache_file)
    return problems
