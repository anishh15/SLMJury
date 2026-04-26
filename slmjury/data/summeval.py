"""SummEval dataset loader.

Downloads and caches the SummEval summarization evaluation dataset from
HuggingFace (mteb/summeval). Flattens 100 articles × 16 machine summaries
into 1,600 scored pairs, each with human expert scores on 4 dimensions:
coherence, consistency, fluency, and relevance.

No student model needed — human scores serve as ground truth.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "summeval"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "summeval.json"

DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]


def load_summeval(cache_path: Optional[Path] = None) -> list[dict]:
    """Load SummEval dataset, downloading from HuggingFace if not cached.

    Each entry represents one (article, machine_summary) pair with averaged
    human scores across annotators for each quality dimension.

    Args:
        cache_path: Custom path to the cache JSON file.

    Returns:
        List of dicts with keys: problem_id, source_text, summary,
        coherence, consistency, fluency, relevance (each a float 1-5).
    """
    cache_file = cache_path or DEFAULT_CACHE_FILE

    if cache_file.exists():
        logger.info("Loading cached SummEval from %s", cache_file)
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Downloading SummEval from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("mteb/summeval", split="test")
    except Exception as e:
        logger.error("Failed to load SummEval: %s", e)
        return []

    problems = []
    pid = 0

    for article_idx, item in enumerate(dataset):
        source_text = item["text"]
        machine_summaries = item["machine_summaries"]

        # Each dimension has a list of scores (one per machine summary),
        # and each score is itself a list of annotator ratings.
        # Average across annotators to get one score per summary per dimension.
        for summary_idx, summary in enumerate(machine_summaries):
            scores = {}
            for dim in DIMENSIONS:
                annotator_scores = item[dim][summary_idx]
                if isinstance(annotator_scores, list):
                    scores[dim] = round(
                        sum(annotator_scores) / len(annotator_scores), 2,
                    )
                else:
                    scores[dim] = float(annotator_scores)

            problems.append({
                "problem_id": pid,
                "article_id": article_idx,
                "summary_id": summary_idx,
                "source_text": source_text,
                "summary": summary,
                **scores,
            })
            pid += 1

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=4)

    logger.info(
        "Cached %d SummEval pairs (%d articles × %d summaries) to %s",
        len(problems), len(dataset), len(machine_summaries), cache_file,
    )
    return problems
