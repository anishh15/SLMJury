"""MATH dataset loader.

Downloads and caches the MATH reasoning dataset (7 subjects) from HuggingFace.
Extracts ground truth answers from \\boxed{} LaTeX expressions.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "math"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "math.json"

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def load_math(cache_path: Optional[Path] = None) -> list[dict]:
    """Load MATH dataset, downloading from HuggingFace if not cached.

    Downloads all 7 subject configs and merges them into a single list.

    Args:
        cache_path: Custom path to the cache JSON file.

    Returns:
        List of problem dicts with keys: problem_id, question,
        ground_truth_reasoning, ground_truth_answer.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached MATH from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading MATH from HuggingFace...")
    try:
        from datasets import load_dataset

        all_items = []
        for subject in MATH_SUBJECTS:
            logger.info("  Loading %s...", subject)
            subset = load_dataset(
                "EleutherAI/hendrycks_math",
                subject,
                split="test",
                trust_remote_code=True,
                token=True,
            )
            all_items.extend(subset)
    except Exception as e:
        logger.error("Failed to load MATH: %s", e)
        return []

    problems = []
    for i, item in enumerate(all_items):
        solution = item["solution"]
        answer = _extract_boxed_answer(solution)
        problems.append({
            "problem_id": i,
            "question": item["problem"],
            "ground_truth_reasoning": solution,
            "ground_truth_answer": answer,
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info("Cached %d MATH problems to %s", len(problems), cache_file)
    return problems


def _extract_boxed_answer(solution: str) -> str:
    """Extract content from the last \\boxed{} in a solution, handling nested braces."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""

    start = idx + len("\\boxed{")
    brace_count = 1
    end = start
    while end < len(solution) and brace_count > 0:
        if solution[end] == "{":
            brace_count += 1
        elif solution[end] == "}":
            brace_count -= 1
        end += 1

    return solution[start:end - 1].strip()
