"""Accuracy counts for judgement files, computed the way JudgeEvaluator and the summaries compute them.

Per file, Undefined verdicts stay in the denominator; across files, counts are summed before dividing.
Importing this module changes no global state. Analyses turn on strict MATH labels themselves.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from slmjury.core.evaluator import JudgeEvaluator
from slmjury.parsers.answer import extract_answer, get_dataset_type
from slmjury.parsers.normalizer import answers_are_equivalent

__all__ = ["Counts", "file_counts", "micro", "verdict_pairs", "expected_verdict",
           "CORRUPT_FILES"]

# Paths of result files that failed to parse, filled in by `file_counts`.
CORRUPT_FILES: list[str] = []


@dataclass
class Counts:
    """Raw counts for one judgement file, the unit of micro-averaging."""

    total: int = 0        # all judgement records (JudgeEvaluator's denominator)
    correct: int = 0      # verdict matches the expected verdict
    valid: int = 0        # verdict parsed as Correct/Incorrect (IFR numerator)

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            total=self.total + other.total,
            correct=self.correct + other.correct,
            valid=self.valid + other.valid,
        )


@contextlib.contextmanager
def _quiet():
    """Silence the evaluator's printout and log records for one call, then restore logging."""
    before = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        logging.disable(before)


def file_counts(
    judge: str,
    student: str,
    dataset: str,
    max_tokens: int,
    path: Path,
) -> Counts | None:
    """Counts for one judgement file, computed by ``JudgeEvaluator``.

    Returns None when the file is absent, empty or unreadable, so callers can
    tell "not run" from "run and scored 0".
    """
    if not path.exists():
        return None
    try:
        with open(path) as f:
            judgements = json.load(f)
    except json.JSONDecodeError:
        # A run killed mid-write leaves a truncated file. It is treated as
        # absent and recorded so the caller can list it.
        CORRUPT_FILES.append(str(path))
        return None
    if not isinstance(judgements, list) or not judgements:
        return None

    with _quiet():
        evaluator = JudgeEvaluator(
            judge_model=judge, student_model=student, dataset=dataset,
            max_tokens=max_tokens, judgements=judgements,
        )
        evaluator.evaluate()

    # The integer counters, not the rounded rates, so micro averages stay exact.
    return Counts(
        total=len(judgements),
        correct=evaluator.correct_count,
        valid=evaluator.valid_count,
    )


def micro(counts_iterable) -> dict | None:
    """Micro-average a collection of ``Counts``; None entries are skipped."""
    agg = Counts()
    n_files = 0
    for c in counts_iterable:
        if c is None:
            continue
        agg = agg + c
        n_files += 1
    if not agg.total:
        return None
    return {
        "accuracy": round(agg.correct / agg.total * 100, 6),
        "ifr": round(agg.valid / agg.total * 100, 6),
        "correct": agg.correct,
        "valid": agg.valid,
        "total": agg.total,
        "n_files": n_files,
    }


def expected_verdict(item: dict, dataset_type: str) -> tuple[str, str]:
    """Expected verdict ("correct"/"incorrect") and student answer for one record.

    Same logic as ``JudgeEvaluator._get_expected_verdict``, for analyses that
    need a label per item rather than an aggregate.
    """
    student_answer = item.get("student_answer", "")
    if not student_answer or student_answer == "UNABLE_TO_EXTRACT":
        student_answer = extract_answer(item.get("student_reasoning", ""), dataset_type)
    ground_truth = item.get("ground_truth_answer", "")
    if answers_are_equivalent(student_answer, ground_truth, dataset_type):
        return "correct", student_answer
    return "incorrect", student_answer


def verdict_pairs(dataset: str, judgements: list[dict]) -> list[tuple[str, str]]:
    """(judge verdict, expected verdict) for every record, both lower-cased.

    Both values come from the same record, so the pairs stay aligned even
    when files differ in length.
    """
    dtype = get_dataset_type(dataset)
    pairs = []
    for item in judgements:
        oracle, _ = expected_verdict(item, dtype)
        pairs.append((item.get("judgement", "Undefined").lower(), oracle))
    return pairs
