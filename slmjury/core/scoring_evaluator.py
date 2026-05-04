"""Scoring evaluator — correlation-based metrics for open-ended evaluation.

Computes agreement between SLM judge scores and ground truth (human for
SummEval, oracle for MT-Bench) using:
- Pearson correlation (linear agreement)
- Spearman rank correlation (robust to non-linearity)
- Cohen's kappa (binary agreement after thresholding)
- Accuracy (binary threshold agreement)
- MSE (absolute score calibration)
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default threshold for binarizing scores: ≥ threshold = GOOD, < threshold = BAD
DEFAULT_BINARY_THRESHOLD = 4


def _pearson_r(x: list[float], y: list[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient.

    Returns None if either variable has zero variance.
    """
    n = len(x)
    if n < 2:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None

    return cov / denom


def _spearman_rho(x: list[float], y: list[float]) -> Optional[float]:
    """Compute Spearman rank correlation coefficient.

    Converts values to ranks, then computes Pearson on ranks.
    """
    if len(x) < 2:
        return None

    def _ranks(values):
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) - 1 and indexed[j][1] == indexed[j + 1][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1  # 1-indexed average rank for ties
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rank_x = _ranks(x)
    rank_y = _ranks(y)
    return _pearson_r(rank_x, rank_y)


def _cohens_kappa(
    x: list[float],
    y: list[float],
    threshold: int = DEFAULT_BINARY_THRESHOLD,
) -> Optional[float]:
    """Compute Cohen's kappa for binary agreement.

    Binarizes scores: ≥ threshold = 1 (GOOD), < threshold = 0 (BAD).

    Returns None if there is no variance in agreement.
    """
    n = len(x)
    if n == 0:
        return None

    x_bin = [1 if v >= threshold else 0 for v in x]
    y_bin = [1 if v >= threshold else 0 for v in y]

    # Build confusion matrix
    a = sum(1 for xi, yi in zip(x_bin, y_bin) if xi == 1 and yi == 1)
    b = sum(1 for xi, yi in zip(x_bin, y_bin) if xi == 1 and yi == 0)
    c = sum(1 for xi, yi in zip(x_bin, y_bin) if xi == 0 and yi == 1)
    d = sum(1 for xi, yi in zip(x_bin, y_bin) if xi == 0 and yi == 0)

    po = (a + d) / n  # observed agreement
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)  # expected

    if pe == 1.0:
        return 1.0 if po == 1.0 else None

    return (po - pe) / (1 - pe)


def _binary_accuracy(
    x: list[float],
    y: list[float],
    threshold: int = DEFAULT_BINARY_THRESHOLD,
) -> Optional[float]:
    """Binary classification accuracy after thresholding."""
    if not x:
        return None

    x_bin = [1 if v >= threshold else 0 for v in x]
    y_bin = [1 if v >= threshold else 0 for v in y]

    correct = sum(1 for xi, yi in zip(x_bin, y_bin) if xi == yi)
    return correct / len(x)


def _mse(x: list[float], y: list[float]) -> Optional[float]:
    """Mean squared error between two score lists."""
    if not x:
        return None
    return sum((xi - yi) ** 2 for xi, yi in zip(x, y)) / len(x)


def evaluate_scores(
    judge_scores: list[Optional[float]],
    ground_truth: list[float],
    threshold: int = DEFAULT_BINARY_THRESHOLD,
) -> dict:
    """Compute all correlation metrics between judge and ground truth scores.

    Filters out pairs where judge_score is None (parse failures).

    Args:
        judge_scores: List of judge-predicted scores (may contain None).
        ground_truth: List of ground truth scores.
        threshold: Binary classification threshold for kappa/accuracy.

    Returns:
        Dict with metrics: pearson, spearman, cohens_kappa, accuracy, mse,
        n_valid, n_total, parse_failure_rate.
    """
    n_total = len(judge_scores)

    # Filter valid pairs
    valid_pairs = [
        (j, g)
        for j, g in zip(judge_scores, ground_truth)
        if j is not None
    ]
    n_valid = len(valid_pairs)

    if n_valid < 2:
        return {
            "pearson": None,
            "spearman": None,
            "cohens_kappa": None,
            "accuracy": None,
            "mse": None,
            "n_valid": n_valid,
            "n_total": n_total,
            "parse_failure_rate": 1.0 - (n_valid / n_total) if n_total else 0,
        }

    j_scores = [p[0] for p in valid_pairs]
    g_scores = [p[1] for p in valid_pairs]

    return {
        "pearson": _safe_round(_pearson_r(j_scores, g_scores)),
        "spearman": _safe_round(_spearman_rho(j_scores, g_scores)),
        "cohens_kappa": _safe_round(_cohens_kappa(j_scores, g_scores, threshold)),
        "accuracy": _safe_round(_binary_accuracy(j_scores, g_scores, threshold)),
        "mse": _safe_round(_mse(j_scores, g_scores)),
        "n_valid": n_valid,
        "n_total": n_total,
        "parse_failure_rate": round(1.0 - (n_valid / n_total), 4) if n_total else 0,
    }


def evaluate_summeval(results: list[dict], threshold: int = DEFAULT_BINARY_THRESHOLD) -> dict:
    """Evaluate SummEval scoring results across all 4 dimensions.

    Args:
        results: List of scored dicts from ScoringJudge.score_summeval().
        threshold: Binary classification threshold.

    Returns:
        Dict mapping dimension name → metrics dict, plus an 'overall' average.
    """
    from slmjury.parsers.score import SUMMEVAL_DIMENSIONS

    per_dim = {}
    for dim in SUMMEVAL_DIMENSIONS:
        judge_key = f"judge_{dim}"
        human_key = f"human_{dim}"

        judge_scores = [r[judge_key] for r in results]
        ground_truth = [r[human_key] for r in results]

        per_dim[dim] = evaluate_scores(judge_scores, ground_truth, threshold)

    # Compute overall averages across dimensions
    metrics_to_avg = ["pearson", "spearman", "cohens_kappa", "accuracy", "mse"]
    overall = {}
    for metric in metrics_to_avg:
        values = [
            per_dim[dim][metric]
            for dim in SUMMEVAL_DIMENSIONS
            if per_dim[dim][metric] is not None
        ]
        overall[metric] = _safe_round(sum(values) / len(values)) if values else None

    overall["n_valid"] = min(per_dim[dim]["n_valid"] for dim in SUMMEVAL_DIMENSIONS)
    overall["n_total"] = results[0]["problem_id"] + 1 if results else 0
    overall["n_total"] = len(results)

    return {"per_dimension": per_dim, "overall": overall}


def evaluate_mtbench(results: list[dict], threshold: int = DEFAULT_BINARY_THRESHOLD) -> dict:
    """Evaluate MT-Bench scoring results.

    Each question has a single holistic score covering both turns.

    Args:
        results: List of scored dicts from ScoringJudge.score_mtbench().
        threshold: Binary classification threshold.

    Returns:
        Dict with overall metrics and per-category breakdown.
    """
    judge_scores = [r["judge_score"] for r in results]
    ground_truth = [r["oracle_score"] for r in results]

    overall = evaluate_scores(judge_scores, ground_truth, threshold)

    # Per-category breakdown
    categories: dict[str, list] = {}
    for r in results:
        cat = r.get("category", "general")
        categories.setdefault(cat, []).append(r)

    per_category = {}
    for cat, cat_results in sorted(categories.items()):
        cat_judge = [r["judge_score"] for r in cat_results]
        cat_oracle = [r["oracle_score"] for r in cat_results]
        per_category[cat] = evaluate_scores(cat_judge, cat_oracle, threshold)

    return {"overall": overall, "per_category": per_category}


def save_evaluation(
    eval_result: dict,
    judge_model: str,
    dataset: str,
    output_dir: Path = Path("results/scoring_evaluations"),
    suffix: str = "",
) -> Path:
    """Save evaluation metrics to JSON.

    Args:
        eval_result: Evaluation result dict.
        judge_model: Judge model key.
        dataset: Dataset name ('summeval' or 'mtbench').
        output_dir: Output directory.
        suffix: Optional filename suffix.

    Returns:
        Path to the saved file.
    """
    out_dir = output_dir / judge_model
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = [dataset]
    if suffix:
        parts.append(suffix)
    filename = "_".join(parts) + "_eval.json"
    out_file = out_dir / filename

    with open(out_file, "w") as f:
        json.dump(eval_result, f, indent=4)

    logger.info("Saved evaluation to %s", out_file)
    return out_file


def _safe_round(value: Optional[float], decimals: int = 4) -> Optional[float]:
    """Round a value, returning None if input is None."""
    if value is None:
        return None
    return round(value, decimals)
