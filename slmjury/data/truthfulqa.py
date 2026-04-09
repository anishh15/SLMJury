"""TruthfulQA (Multiple Choice) dataset loader.

Downloads and caches the TruthfulQA MC dataset from HuggingFace.
Uses the multiple_choice config, validation split (only split available).

Schema: question (str), choices (list of 4 strings), label (int 0-3 ClassLabel).
Tests whether models select truthful answers over common misconceptions.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "truthfulqa"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "truthfulqa.json"

# Label index to letter mapping
_INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_truthfulqa(cache_path: Optional[Path] = None) -> list[dict]:
    """Load TruthfulQA MC dataset, downloading from HuggingFace if not cached.

    Args:
        cache_path: Custom path to the cache JSON file.

    Returns:
        List of problem dicts with keys: problem_id, question,
        ground_truth_reasoning, ground_truth_answer.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached TruthfulQA from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading TruthfulQA MC from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "EleutherAI/truthful_qa_mc", "multiple_choice", split="validation",
        )
    except Exception as e:
        logger.error("Failed to load TruthfulQA: %s", e)
        return []

    problems = []
    for i, item in enumerate(dataset):
        question_text = item["question"].strip()
        choices = item["choices"]

        choices_text = "  ".join(
            f"{letter}) {choice}"
            for letter, choice in zip("ABCD", choices)
        )
        question = f"{question_text}\n{choices_text}"

        # label is an int (0-3) via ClassLabel
        label = item["label"]
        answer = _INDEX_TO_LETTER.get(label, str(label))

        problems.append({
            "problem_id": i,
            "question": question,
            "ground_truth_reasoning": "",
            "ground_truth_answer": answer,
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info("Cached %d TruthfulQA problems to %s", len(problems), cache_file)
    return problems
