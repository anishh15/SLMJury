"""MATH loader: the 5,000 test problems of the seven subjects, from pinned Parquet files on the Hub.

The ground-truth answer is the content of the last \\boxed{} in each reference solution.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from slmjury.data.hub import check_size, read_parquet_rows

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "math"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "math.json"

# The dataset's own Parquet files, pinned to one commit of its repository.
MATH_REPO = "EleutherAI/hendrycks_math"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_SIZE = 5000

# Subjects in study order; problem_id counts through them in this order.
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
    """The MATH test set as problem dicts (problem_id, question, ground_truth_reasoning, ground_truth_answer).

    Downloads once and caches the result as JSON. Raises DatasetLoadError when the
    download fails or the item count is not 5,000.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached MATH from %s", cache_file)
        with open(cache_file) as f:
            problems = json.load(f)
        return check_size("MATH", problems, MATH_SIZE,
                          source=f"{cache_file} (delete it to download again)")

    logger.info("Downloading MATH from %s at %s...", MATH_REPO, MATH_REVISION[:7])
    rows = []
    for subject in MATH_SUBJECTS:
        rows.extend(read_parquet_rows(
            MATH_REPO, f"{subject}/test-00000-of-00001.parquet", MATH_REVISION))

    problems = []
    for i, item in enumerate(rows):
        solution = item["solution"]
        problems.append({
            "problem_id": i,
            "question": item["problem"],
            "ground_truth_reasoning": solution,
            "ground_truth_answer": _extract_boxed_answer(solution),
        })
    check_size("MATH", problems, MATH_SIZE, source=f"{MATH_REPO}@{MATH_REVISION[:7]}")

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
