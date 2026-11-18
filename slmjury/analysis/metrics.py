"""Correlation, agreement and ranking metrics used by the analyses.

pearson, spearman, cohens_kappa, the evaluate_* functions and save_evaluation are re-exported from
slmjury.core.scoring_evaluator. Self-test: python -m slmjury.analysis.metrics
"""

from __future__ import annotations

from slmjury.core.scoring_evaluator import (  # noqa: F401  (re-export)
    DEFAULT_BINARY_THRESHOLD,
    _binary_accuracy,
    _cohens_kappa,
    _mse,
    _pearson_r,
    _spearman_rho,
    evaluate_mtbench,
    evaluate_scores,
    evaluate_summeval,
    save_evaluation,
)


def pearson(x: list, y: list):
    return _pearson_r(x, y)


def spearman(x: list, y: list):
    return _spearman_rho(x, y)


def cohens_kappa(x: list, y: list, threshold: int = DEFAULT_BINARY_THRESHOLD):
    return _cohens_kappa(x, y, threshold)


# ── Rank stability ──────────────────────────────────────────────────────────

def kendall_tau(rank_a: list[float], rank_b: list[float]):
    """Kendall tau (tau-a) between two rankings of the same items."""
    assert len(rank_a) == len(rank_b)
    n = len(rank_a)
    if n < 2:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = rank_a[i] - rank_a[j]
            diff_b = rank_b[i] - rank_b[j]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total if total else None


def _rank_desc(values: list[float]) -> list[float]:
    """Rank values descending (rank 1 = highest value). Ties get the average rank."""
    n = len(values)
    order = sorted(range(n), key=lambda i: -values[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def rank_stability(accuracy_by_variant: dict[str, list[float]]) -> dict:
    """Pairwise Kendall tau between the judge rankings of each prompt variant.

    ``accuracy_by_variant`` maps each variant to per-judge accuracies, all in the same judge order.
    Returns {"mean_tau", "min_tau", "pairwise": {(v1, v2): tau}}.
    """
    variants = list(accuracy_by_variant.keys())
    pairwise = {}
    taus = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            v1, v2 = variants[i], variants[j]
            rank1 = _rank_desc(accuracy_by_variant[v1])
            rank2 = _rank_desc(accuracy_by_variant[v2])
            tau = kendall_tau(rank1, rank2)
            pairwise[(v1, v2)] = tau
            if tau is not None:
                taus.append(tau)
    return {
        "mean_tau": sum(taus) / len(taus) if taus else None,
        "min_tau": min(taus) if taus else None,
        "pairwise": pairwise,
    }


# ── False accept / false reject ─────────────────────────────────────────────

def false_accept_rate(
    verdicts: list[str],
    expected: list[str],
    accept_label: str = "correct",
) -> float:
    """Share of truly incorrect items the judge labelled correct (lower-cased labels)."""
    truly_negative = [v for v, e in zip(verdicts, expected) if e != accept_label]
    if not truly_negative:
        return float("nan")
    return sum(1 for v in truly_negative if v == accept_label) / len(truly_negative)


def false_reject_rate(
    verdicts: list[str],
    expected: list[str],
    accept_label: str = "correct",
) -> float:
    """Share of truly correct items the judge did not label correct."""
    truly_positive = [v for v, e in zip(verdicts, expected) if e == accept_label]
    if not truly_positive:
        return float("nan")
    return sum(1 for v in truly_positive if v != accept_label) / len(truly_positive)


def far_frr_matrix(
    judge_results: dict[str, dict[str, list[dict]]],
    oracle_results: dict[str, list[str]],
) -> dict:
    """FAR, FRR and accuracy for every judge and student, over items with a Correct/Incorrect verdict.

    Takes {judge: {student: [judgement dicts]}} and the expected verdicts {student: ["correct" | "incorrect"]};
    returns {judge: {student: {"far", "frr", "accuracy"}}}.
    """
    matrix = {}
    for judge_key, by_student in judge_results.items():
        matrix[judge_key] = {}
        for student_key, judgements in by_student.items():
            expected = oracle_results.get(student_key, [])
            verdicts = [j.get("judgement", "Undefined").lower() for j in judgements]
            n = min(len(verdicts), len(expected))
            v, e = verdicts[:n], expected[:n]
            valid = [(vv, ee) for vv, ee in zip(v, e) if vv in ("correct", "incorrect")]
            if not valid:
                matrix[judge_key][student_key] = {
                    "far": float("nan"), "frr": float("nan"), "accuracy": float("nan"),
                }
                continue
            vv, ee = zip(*valid)
            far = false_accept_rate(list(vv), list(ee))
            frr = false_reject_rate(list(vv), list(ee))
            acc = sum(1 for a, b in zip(vv, ee) if a == b) / len(vv)
            matrix[judge_key][student_key] = {"far": far, "frr": frr, "accuracy": acc}
    return matrix


# ── Position bias ───────────────────────────────────────────────────────────

def position_bias_flip_rate(
    fwd_results: list[dict],
    rev_results: list[dict],
    verdict_key: str = "judgement",
) -> float:
    """Share of items where the judge's preference changes when A and B are swapped.

    A consistent judge answers "A" in one run and "B" in the other. Only items with an A/B answer in both
    runs count; 0 is fully consistent, 0.5 is what random answers give.
    """
    assert len(fwd_results) == len(rev_results)
    flips = valid = 0
    for fwd, rev in zip(fwd_results, rev_results):
        f = fwd.get(verdict_key, "").lower()
        r = rev.get(verdict_key, "").lower()
        if f in ("a", "b") and r in ("a", "b"):
            valid += 1
            fwd_prefers_first = (f == "a")
            rev_prefers_first = (r == "b")
            if fwd_prefers_first != rev_prefers_first:
                flips += 1
    return flips / valid if valid > 0 else float("nan")


# ── Pareto front ────────────────────────────────────────────────────────────

def pareto_front(
    points: list[tuple],
    maximize_quality: bool = True,
    minimize_cost: bool = True,
) -> list[tuple]:
    """Pareto-optimal subset of (cost, quality, label) points, sorted by cost."""
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if p is q:
                continue
            cost_ok = (q[0] <= p[0]) if minimize_cost else (q[0] >= p[0])
            qual_ok = (q[1] >= p[1]) if maximize_quality else (q[1] <= p[1])
            cost_better = (q[0] < p[0]) if minimize_cost else (q[0] > p[0])
            qual_better = (q[1] > p[1]) if maximize_quality else (q[1] < p[1])
            if cost_ok and qual_ok and (cost_better or qual_better):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return sorted(pareto, key=lambda x: x[0])


if __name__ == "__main__":
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert abs(pearson(x, y) - (-1.0)) < 1e-9
    assert abs(spearman(x, y) - (-1.0)) < 1e-9
    print(f"pearson / spearman of reversed ranks = {pearson(x, y):.3f} / {spearman(x, y):.3f}")

    k = cohens_kappa([5.0, 4.0, 3.0, 2.0, 1.0, 4.0, 5.0, 3.0],
                     [5.0, 4.0, 2.0, 2.0, 1.0, 3.0, 5.0, 4.0], threshold=4)
    print(f"cohens_kappa(threshold=4) = {k:.3f}")

    tau = kendall_tau([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0])
    assert abs(tau - 1 / 3) < 1e-9
    print(f"kendall_tau([1,2,3,4], [2,1,4,3]) = {tau:.3f}")

    stab = rank_stability({
        "original": [0.90, 0.88, 0.86, 0.80],
        "minimal": [0.89, 0.88, 0.85, 0.81],
        "gpt_style": [0.91, 0.87, 0.86, 0.79],
    })
    print(f"rank_stability mean_tau = {stab['mean_tau']:.3f}")

    pf = pareto_front([(10, 0.90, "A"), (20, 0.92, "B"), (5, 0.85, "C"), (15, 0.91, "D")])
    assert [p[2] for p in pf] == ["C", "A", "D", "B"]
    print(f"pareto_front = {[p[2] for p in pf]}")

    far = false_accept_rate(["correct", "correct", "incorrect", "correct"],
                            ["incorrect", "correct", "incorrect", "incorrect"])
    assert abs(far - 2 / 3) < 1e-9
    print(f"false_accept_rate = {far:.3f}")

    print("All checks passed.")
