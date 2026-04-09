"""HellaSwag dataset loader.

Downloads and caches the HellaSwag commonsense completion dataset from HuggingFace.
Uses the validation split (test labels are hidden).

Schema: ctx (context), endings (list of 4 completions), label (str "0"-"3"),
        activity_label (activity description).
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "hellaswag"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "hellaswag.json"

# Label index to letter mapping
_INDEX_TO_LETTER = {"0": "A", "1": "B", "2": "C", "3": "D"}


def load_hellaswag(cache_path: Optional[Path] = None) -> list[dict]:
    """Load HellaSwag dataset, downloading from HuggingFace if not cached.

    Args:
        cache_path: Custom path to the cache JSON file.

    Returns:
        List of problem dicts with keys: problem_id, question,
        ground_truth_reasoning, ground_truth_answer.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached HellaSwag from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading HellaSwag from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("Rowan/hellaswag", split="validation")
    except Exception as e:
        logger.error("Failed to load HellaSwag: %s", e)
        return []

    problems = []
    for i, item in enumerate(dataset):
        # Format question with activity context and inline choices
        ctx = item["ctx"].strip()
        activity = item.get("activity_label", "").strip()
        endings = item["endings"]

        choices_text = "  ".join(
            f"{letter}) {ending}"
            for letter, ending in zip("ABCD", endings)
        )

        question = f"[Activity: {activity}]\n{ctx}\n{choices_text}" if activity else f"{ctx}\n{choices_text}"

        label_str = str(item["label"])
        answer = _INDEX_TO_LETTER.get(label_str, label_str)

        problems.append({
            "problem_id": i,
            "question": question,
            "ground_truth_reasoning": "",
            "ground_truth_answer": answer,
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info("Cached %d HellaSwag problems to %s", len(problems), cache_file)
    return problems
