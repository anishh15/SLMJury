"""TruthfulQA loader: the 684 four-choice questions (multiple_choice config, validation split).

Read from the Hub's pinned Parquet export of EleutherAI/truthful_qa_mc. Each question lists
its choices as A) to D), and the ground-truth answer is the letter of the truthful choice.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from slmjury.data.hub import check_size, read_parquet_rows

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "truthfulqa"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "truthfulqa.json"

# The Hub's Parquet export (refs/convert/parquet), pinned to one commit.
TRUTHFULQA_REPO = "EleutherAI/truthful_qa_mc"
TRUTHFULQA_REVISION = "b6468d07f67cdd1ba48292eb55a515b8a97d02ea"
TRUTHFULQA_FILE = "multiple_choice/validation/0000.parquet"
TRUTHFULQA_SIZE = 684

# Label index to letter mapping
_INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_truthfulqa(cache_path: Optional[Path] = None) -> list[dict]:
    """TruthfulQA as problem dicts (problem_id, question, ground_truth_reasoning, ground_truth_answer).

    Downloads once and caches the result as JSON. Raises DatasetLoadError when the
    download fails or the item count is not 684.
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached TruthfulQA from %s", cache_file)
        with open(cache_file) as f:
            problems = json.load(f)
        return check_size("TruthfulQA", problems, TRUTHFULQA_SIZE,
                          source=f"{cache_file} (delete it to download again)")

    logger.info("Downloading TruthfulQA from %s at %s...", TRUTHFULQA_REPO, TRUTHFULQA_REVISION[:7])
    rows = read_parquet_rows(TRUTHFULQA_REPO, TRUTHFULQA_FILE, TRUTHFULQA_REVISION)

    problems = []
    for i, item in enumerate(rows):
        choices_text = "  ".join(
            f"{letter}) {choice}" for letter, choice in zip("ABCD", item["choices"])
        )
        label = item["label"]
        problems.append({
            "problem_id": i,
            "question": f"{item['question'].strip()}\n{choices_text}",
            "ground_truth_reasoning": "",
            "ground_truth_answer": _INDEX_TO_LETTER.get(label, str(label)),
        })
    check_size("TruthfulQA", problems, TRUTHFULQA_SIZE,
               source=f"{TRUTHFULQA_REPO}@{TRUTHFULQA_REVISION[:7]}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info("Cached %d TruthfulQA problems to %s", len(problems), cache_file)
    return problems
