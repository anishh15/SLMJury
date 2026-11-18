"""SummEval loader (mteb/summeval on Hugging Face): 1,600 summaries with human scores.

100 articles x 16 machine summaries, each scored by experts for coherence, consistency, fluency and
relevance. The human scores are the ground truth, so no student model is needed.
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
    """Load SummEval, downloading it from Hugging Face if not cached.

    One dict per (article, machine summary) pair: problem_id, source_text, summary, and the annotators'
    mean coherence, consistency, fluency and relevance (floats, 1-5). ``cache_path`` overrides the cache file.
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
