"""MT-Bench loader (philschmid/mt-bench on Hugging Face), downloaded once and cached.

Each of the 80 questions has 2 turns, a main prompt and a follow-up that depends on the student's
Turn 1 response. A student answers both turns; the reference models and the SLM judges score them.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "mtbench"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "mtbench.json"


def load_mtbench(cache_path: Optional[Path] = None) -> list[dict]:
    """Load MT-Bench with both turn prompts, downloading it from Hugging Face if not cached.

    Returns dicts with problem_id, question_id, category, turn1 and turn2. ``cache_path`` overrides
    the cache JSON file.
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
        turns = item.get("turns", [])
        turn1 = turns[0] if len(turns) > 0 else ""
        turn2 = turns[1] if len(turns) > 1 else ""

        problems.append({
            "problem_id": i,
            "question_id": item.get("question_id", i),
            "category": item.get("category", "general"),
            "turn1": turn1,
            "turn2": turn2,
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info(
        "Cached %d MT-Bench questions (2 turns each) to %s",
        len(problems), cache_file,
    )
    return problems
