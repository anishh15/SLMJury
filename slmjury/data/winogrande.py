"""WinoGrande dataset loader.

Downloads and caches the WinoGrande pronoun resolution dataset from HuggingFace.
Uses the winogrande_xl config, validation split (test labels are hidden).

Schema: sentence (str with _ blank), option1, option2, answer (str "1" or "2").
Only 2 choices (A/B), not 4.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "winogrande"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "winogrande.json"

# Answer index to letter mapping (only 2 options)
_INDEX_TO_LETTER = {"1": "A", "2": "B"}


def load_winogrande(cache_path: Optional[Path] = None) -> list[dict]:
    """Load WinoGrande dataset, downloading from HuggingFace if not cached.

    Args:
        cache_path: Custom path to the cache JSON file.

    Returns:
        List of problem dicts with keys: problem_id, question,
        ground_truth_reasoning, ground_truth_answer.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached WinoGrande from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading WinoGrande from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    except Exception as e:
        logger.error("Failed to load WinoGrande: %s", e)
        return []

    problems = []
    for i, item in enumerate(dataset):
        sentence = item["sentence"].strip()
        option1 = item["option1"].strip()
        option2 = item["option2"].strip()

        choices_text = f"A) {option1}  B) {option2}"
        question = f"{sentence}\n{choices_text}"

        answer = _INDEX_TO_LETTER.get(item["answer"], item["answer"])

        problems.append({
            "problem_id": i,
            "question": question,
            "ground_truth_reasoning": "",
            "ground_truth_answer": answer,
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info("Cached %d WinoGrande problems to %s", len(problems), cache_file)
    return problems
