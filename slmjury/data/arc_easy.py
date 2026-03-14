"""ARC-Easy dataset loader.

Downloads and caches the ARC-Easy multiple choice dataset from HuggingFace.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "arc_easy"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "arc_easy.json"


def load_arc_easy(cache_path: Optional[Path] = None) -> list[dict]:
    """Load ARC-Easy dataset, downloading from HuggingFace if not cached.

    Args:
        cache_path: Custom path to the cache JSON file.

    Returns:
        List of problem dicts with keys: problem_id, question,
        ground_truth_reasoning, ground_truth_answer.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached ARC-Easy from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading ARC-Easy from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    except Exception as e:
        logger.error("Failed to load ARC-Easy: %s", e)
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

    logger.info("Cached %d ARC-Easy problems to %s", len(problems), cache_file)
    return problems
