"""MT-Bench dataset loader.

Downloads and caches MT-Bench from HuggingFace (philschmid/mt-bench).
Uses only Turn 1 prompts (80 questions) across 8 categories.

Requires a student model to generate responses and an oracle model to
score them. The SLM judge then also scores them for comparison.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "mtbench"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "mtbench.json"


def load_mtbench(cache_path: Optional[Path] = None) -> list[dict]:
    """Load MT-Bench dataset, downloading from HuggingFace if not cached.

    Extracts Turn 1 prompts only (single-turn evaluation).

    Args:
        cache_path: Custom path to the cache JSON file.

    Returns:
        List of dicts with keys: problem_id, question_id, category, question.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached MT-Bench from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading MT-Bench from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("philschmid/mt-bench", split="train")
    except Exception as e:
        logger.error("Failed to load MT-Bench: %s", e)
        return []

    problems = []
    for i, item in enumerate(dataset):
        # MT-Bench has 2 turns; we only use Turn 1
        turn1_question = item["turns"][0] if item["turns"] else ""

        problems.append({
            "problem_id": i,
            "question_id": item.get("question_id", i),
            "category": item.get("category", "general"),
            "question": turn1_question,
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info(
        "Cached %d MT-Bench Turn-1 prompts to %s", len(problems), cache_file,
    )
    return problems
