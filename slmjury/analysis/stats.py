"""Significance tests and intervals for comparing judges: McNemar, cluster bootstrap, TOST, BH, Wilson.

Needs only the standard library; numpy, when present, speeds up the bootstrap. Self-test (also checks
against scipy when it is installed): python -m slmjury.analysis.stats
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

try:                                   # optional, speed only
    import numpy as _np
except Exception:                      # pragma: no cover
    _np = None

__all__ = [
    "McNemarResult", "BootstrapResult", "mcnemar", "discordant_counts",
    "cluster_bootstrap_diff", "tost_equivalence", "bootstrap_ci",
    "spearman_bootstrap_ci", "benjamini_hochberg", "macro_micro",
    "wilson_interval",
]

DEFAULT_BOOT = 10_000
DEFAULT_SEED = 20260916          # fixed so every bootstrap result is reproducible


# ── McNemar ─────────────────────────────────────────────────────────────────

@dataclass
class McNemarResult:
    b: int                  # a correct, b wrong
    c: int                  # a wrong, b correct
    n_discordant: int
    n_pairs: int
    p_exact: float          # two-sided exact binomial
    p_midp: float           # mid-p correction (less conservative)
    p_chi2_cc: float        # chi-square with continuity correction (for reference)
    odds_ratio: float       # b / c, inf when c == 0

    def as_row(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "discordant_a_only": self.b,
            "discordant_b_only": self.c,
            "n_discordant": self.n_discordant,
            "mcnemar_p_exact": self.p_exact,
            "mcnemar_p_midp": self.p_midp,
            "mcnemar_p_chi2_cc": self.p_chi2_cc,
            "odds_ratio": (None if math.isinf(self.odds_ratio)
                           else self.odds_ratio),
        }


def discordant_counts(a_correct, b_correct) -> tuple[int, int, int]:
    """(b, c, n): b = a right and b wrong, c = a wrong and b right, n = items.

    Both arguments are per-item 0/1 sequences over the same items in the same
    order; aligning them is the caller's job.
    """
    a = list(a_correct)
    b_ = list(b_correct)
    if len(a) != len(b_):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b_)} items")
    b_only = sum(1 for x, y in zip(a, b_) if x and not y)
    c_only = sum(1 for x, y in zip(a, b_) if y and not x)
    return b_only, c_only, len(a)


def _binom_cdf_half(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 0.5), exact via integer binomials."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = sum(math.comb(n, i) for i in range(k + 1))
    return total / (2 ** n)


def _binom_pmf_half(k: int, n: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) / (2 ** n)


def mcnemar(a_correct, b_correct) -> McNemarResult:
    """Exact two-sided McNemar test on two paired per-item correctness vectors.

    Under the null, the discordant pairs are Binomial(b + c, 1/2). Concordant
    pairs say nothing about which judge is better and drop out.
    """
    b, c, n_pairs = discordant_counts(a_correct, b_correct)
    n = b + c

    if n == 0:                                   # the judges never disagree
        return McNemarResult(b, c, 0, n_pairs, 1.0, 1.0, 1.0, float("nan"))

    m = min(b, c)
    # The exact binomial tail, not the chi-square approximation, stays valid with few discordant pairs.
    p_exact = min(1.0, 2.0 * _binom_cdf_half(m, n))
    # mid-p: half the probability of the observed point, both tails.
    p_midp = min(1.0, 2.0 * (_binom_cdf_half(m - 1, n) + 0.5 * _binom_pmf_half(m, n)))

    chi2 = (abs(b - c) - 1) ** 2 / n if abs(b - c) >= 1 else 0.0
    p_chi2 = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0

    odds = float("inf") if c == 0 else b / c
    return McNemarResult(b, c, n, n_pairs, p_exact, p_midp, p_chi2, odds)


# ── Paired cluster bootstrap ────────────────────────────────────────────────

@dataclass
class BootstrapResult:
    point: float                    # observed difference, in percentage points
    lo: float
    hi: float
    se: float
    alpha: float
    n_boot: int
    n_items: int
    n_clusters: int
    clusters: list = field(default_factory=list, repr=False)

    @property
    def contains_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    def as_row(self, prefix: str = "") -> dict:
        return {
            f"{prefix}diff_pp": self.point,
            f"{prefix}ci_lo_pp": self.lo,
            f"{prefix}ci_hi_pp": self.hi,
            f"{prefix}se_pp": self.se,
            f"{prefix}ci_contains_zero": self.contains_zero,
            f"{prefix}n_items": self.n_items,
            f"{prefix}n_clusters": self.n_clusters,
        }


def _ternary_counts(diffs) -> tuple[int, int, int]:
    """(n_plus, n_minus, n_zero) for a sequence of -1/0/+1 differences."""
    plus = minus = 0
    total = 0
    for d in diffs:
        total += 1
        if d > 0:
            plus += 1
        elif d < 0:
            minus += 1
    return plus, minus, total - plus - minus


def cluster_bootstrap_diff(
    a_correct,
    b_correct,
    clusters,
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Percentile CI on (accuracy_a - accuracy_b), in accuracy points.

    Items are resampled with replacement within each cluster (normally the dataset), and cluster sizes stay
    fixed: items within a dataset are exchangeable, while the dataset sizes are set by the study design.
    """
    a = list(a_correct)
    b = list(b_correct)
    cl = list(clusters)
    if not (len(a) == len(b) == len(cl)):
        raise ValueError(f"unpaired inputs: {len(a)}, {len(b)}, {len(cl)}")
    n = len(a)
    if n == 0:
        raise ValueError("no items")

    by_cluster: dict[object, list[int]] = {}
    for ai, bi, ci in zip(a, b, cl):
        by_cluster.setdefault(ci, []).append((1 if ai else 0) - (1 if bi else 0))

    point = sum(sum(v) for v in by_cluster.values()) / n * 100.0

    # Per cluster the resampled sum depends only on (n_plus, n_minus, n_zero),
    # so a trinomial draw replaces resampling the items.
    specs = []
    for key, diffs in by_cluster.items():
        plus, minus, zero = _ternary_counts(diffs)
        specs.append((key, plus, minus, zero, plus + minus + zero))

    if _np is not None:
        rng = _np.random.default_rng(seed)
        totals = _np.zeros(n_boot, dtype=_np.int64)
        for _key, plus, minus, zero, n_s in specs:
            draws = rng.multinomial(n_s, [plus / n_s, minus / n_s, zero / n_s], size=n_boot)
            totals += draws[:, 0] - draws[:, 1]
        stats = totals / n * 100.0
        lo = float(_np.percentile(stats, 100 * alpha / 2))
        hi = float(_np.percentile(stats, 100 * (1 - alpha / 2)))
        se = float(_np.std(stats, ddof=1))
    else:                                            # pragma: no cover
        rng = random.Random(seed)
        stats = []
        for _ in range(n_boot):
            total = 0
            for _key, plus, minus, zero, n_s in specs:
                # The trinomial as two binomial draws in sequence.
                p_plus = plus / n_s
                np_ = _binomial_rv(rng, n_s, p_plus)
                rest = n_s - np_
                p_minus = 0.0 if (minus + zero) == 0 else minus / (minus + zero)
                nm = _binomial_rv(rng, rest, p_minus)
                total += np_ - nm
            stats.append(total / n * 100.0)
        stats.sort()
        lo = stats[int(alpha / 2 * n_boot)]
        hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
        mean = sum(stats) / n_boot
        se = math.sqrt(sum((s - mean) ** 2 for s in stats) / (n_boot - 1))

    return BootstrapResult(
        point=point, lo=lo, hi=hi, se=se, alpha=alpha, n_boot=n_boot,
        n_items=n, n_clusters=len(specs), clusters=[s[0] for s in specs],
    )


def _binomial_rv(rng: random.Random, n: int, p: float) -> int:   # pragma: no cover
    """Binomial draw without numpy, used only when numpy is missing.

    Large n uses a normal approximation, so results differ slightly from the
    numpy path.
    """
    if n == 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    if n < 40:
        return sum(1 for _ in range(n) if rng.random() < p)
    mu, sigma = n * p, math.sqrt(n * p * (1 - p))
    return max(0, min(n, int(round(rng.gauss(mu, sigma)))))


# ── TOST equivalence ────────────────────────────────────────────────────────

def tost_equivalence(
    a_correct,
    b_correct,
    clusters,
    margin_points: float = 0.5,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Two one-sided tests: is |accuracy_a - accuracy_b| smaller than ``margin_points``?

    Same as the (1 - 2*alpha) bootstrap CI lying inside +-margin (the 90% CI at alpha = 0.05). The margin
    and the differences are in accuracy points, as are the ``_pp`` keys of the result.
    """
    res = cluster_bootstrap_diff(
        a_correct, b_correct, clusters,
        n_boot=n_boot, alpha=2 * alpha, seed=seed,
    )
    equivalent = (res.lo > -margin_points) and (res.hi < margin_points)
    return {
        "diff_pp": res.point,
        "tost_margin_pp": margin_points,
        "tost_ci_lo_pp": res.lo,
        "tost_ci_hi_pp": res.hi,
        "tost_ci_level": 1 - 2 * alpha,
        "tost_equivalent": equivalent,
        # The smallest margin the data would have supported.
        "tost_min_margin_pp": max(abs(res.lo), abs(res.hi)),
        "n_items": res.n_items,
        "n_boot": n_boot,
    }


# ── Generic bootstrap and correlation CIs ───────────────────────────────────

def bootstrap_ci(
    items,
    statistic,
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Percentile CI for any statistic of a list of items, resampled i.i.d."""
    data = list(items)
    n = len(data)
    if n < 2:
        return {"point": None, "lo": None, "hi": None, "n": n, "n_boot": 0}

    point = statistic(data)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        s = statistic(sample)
        if s is not None and not (isinstance(s, float) and math.isnan(s)):
            stats.append(s)
    if not stats:
        return {"point": point, "lo": None, "hi": None, "n": n, "n_boot": 0}

    stats.sort()
    lo = stats[int(alpha / 2 * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return {
        "point": point, "lo": lo, "hi": hi,
        "n": n, "n_boot": len(stats), "alpha": alpha,
    }


def spearman_bootstrap_ci(
    x, y,
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict:
    """CI on Spearman rho, with (x, y) pairs resampled together.

    rho comes from slmjury.core.scoring_evaluator, as in the scoring pipeline, so the interval is around the
    reported value. The import is lazy, to keep this module free of package imports at load time.
    """
    from slmjury.analysis.metrics import spearman

    pairs = list(zip(x, y))

    def stat(sample):
        if len(sample) < 3:
            return None
        xs = [p[0] for p in sample]
        ys = [p[1] for p in sample]
        if len(set(xs)) < 2 or len(set(ys)) < 2:      # rho is undefined on a constant
            return None
        return spearman(xs, ys)

    out = bootstrap_ci(pairs, stat, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "spearman": out["point"],
        "ci_lo": out["lo"],
        "ci_hi": out["hi"],
        "n_pairs": out["n"],
        "n_boot": out["n_boot"],
    }


# ── Multiple comparisons ────────────────────────────────────────────────────

def benjamini_hochberg(pvalues, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg step-up procedure. Returns q-values and which tests survive."""
    ps = list(pvalues)
    m = len(ps)
    if m == 0:
        return {"qvalues": [], "reject": [], "alpha": alpha, "n": 0}

    order = sorted(range(m), key=lambda i: ps[i])
    q = [0.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank + 1                      # 1-based rank of idx in ascending order
        val = min(prev, ps[idx] * m / i)
        q[idx] = val
        prev = val
    return {
        "qvalues": q,
        "reject": [qq <= alpha for qq in q],
        "alpha": alpha,
        "n": m,
        "n_reject": sum(1 for qq in q if qq <= alpha),
    }


# ── Averaging and a proportion CI ───────────────────────────────────────────

def macro_micro(groups) -> dict:
    """Micro and macro averages from (correct, total) pairs, one pair per group.

    Micro weights every item equally, so large datasets dominate it; macro
    weights every group (dataset) equally.
    """
    pairs = [(int(c), int(t)) for c, t in groups if t]
    if not pairs:
        return {"micro": None, "macro": None, "n_groups": 0, "total": 0}
    tot_c = sum(c for c, _ in pairs)
    tot_n = sum(t for _, t in pairs)
    micro = tot_c / tot_n * 100.0
    macro = sum(c / t for c, t in pairs) / len(pairs) * 100.0
    return {
        "micro": micro,
        "macro": macro,
        "macro_minus_micro": macro - micro,
        "n_groups": len(pairs),
        "total": tot_n,
        "correct": tot_c,
    }


def wilson_interval(correct: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for one proportion, in percentage points.

    Meant for a single judge's accuracy. A difference between two judges on the
    same items is paired and needs the bootstrap above.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _z_for(alpha)
    p = correct / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return ((centre - half) * 100.0, (centre + half) * 100.0)


def _z_for(alpha: float) -> float:
    """Two-sided normal quantile by bisection on erfc."""
    target = alpha
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== stats self-test ===\n")

    # 1. McNemar with 550 and 450 discordant pairs.
    a = [1] * 550 + [0] * 450 + [1] * 1000 + [0] * 1000
    b = [0] * 550 + [1] * 450 + [1] * 1000 + [0] * 1000
    r = mcnemar(a, b)
    assert (r.b, r.c, r.n_discordant) == (550, 450, 1000)
    print(f"mcnemar(550, 450)  exact p = {r.p_exact:.4f}  mid-p = {r.p_midp:.4f}  "
          f"chi2cc p = {r.p_chi2_cc:.4f}")
    assert 0.0017 < r.p_exact < 0.0018, r.p_exact

    try:
        from scipy.stats import binomtest
        ref = binomtest(450, 1000, 0.5).pvalue
        print(f"  scipy binomtest    = {ref:.6f}   ours = {r.p_exact:.6f}  "
              f"delta = {abs(ref - r.p_exact):.2e}")
        assert abs(ref - r.p_exact) < 1e-9
    except ImportError:
        print("  (scipy not installed, cross-check skipped)")

    # Degenerate cases.
    assert mcnemar([1, 1], [1, 1]).p_exact == 1.0
    assert mcnemar([1] * 10, [0] * 10).p_exact < 0.005
    print("  degenerate cases ok")

    # 2. Cluster bootstrap: two judges 0.5 points apart on 20,000 items.
    a2 = [1] * 9000 + [0] * 1000 + [1] * 9100 + [0] * 900
    b2 = [1] * 9000 + [0] * 1000 + [1] * 9000 + [0] * 1000
    cl2 = ["ds1"] * 10000 + ["ds2"] * 10000
    boot = cluster_bootstrap_diff(a2, b2, cl2, n_boot=2000, seed=1)
    print(f"\ncluster_bootstrap  point = {boot.point:+.3f} points   "
          f"95% CI [{boot.lo:+.3f}, {boot.hi:+.3f}]   clusters = {boot.n_clusters}")
    assert abs(boot.point - 0.5) < 1e-9, boot.point     # 100/20000 = 0.5 points
    assert boot.lo < boot.point < boot.hi

    # An identical pair gives a CI containing zero.
    same = cluster_bootstrap_diff(a2, a2, cl2, n_boot=500, seed=2)
    assert same.point == 0.0 and same.contains_zero
    print("  identical judges -> point 0.0, CI contains zero  ok")

    # 3. TOST: a 0.5-point difference is not equivalent at a 0.2-point margin
    #    and is equivalent at a 2-point margin.
    t_tight = tost_equivalence(a2, b2, cl2, margin_points=0.2, n_boot=2000, seed=3)
    t_loose = tost_equivalence(a2, b2, cl2, margin_points=2.0, n_boot=2000, seed=3)
    print(f"\ntost margin 0.2 points -> equivalent = {t_tight['tost_equivalent']}   "
          f"margin 2.0 points -> equivalent = {t_loose['tost_equivalent']}")
    assert t_tight["tost_equivalent"] is False
    assert t_loose["tost_equivalent"] is True

    # 4. Spearman CI: a perfectly monotone pair gives rho = 1.
    xs = list(range(30))
    ys = [v * 2 + 1 for v in xs]
    sc = spearman_bootstrap_ci(xs, ys, n_boot=500, seed=4)
    print(f"\nspearman CI (perfect monotone) = {sc['spearman']} "
          f"[{sc['ci_lo']}, {sc['ci_hi']}]  n = {sc['n_pairs']}")
    assert sc["spearman"] == 1.0

    # 5. BH: q-values are monotone in p, and a known small case.
    _ps = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    bh = benjamini_hochberg(_ps)
    qs = bh["qvalues"]
    assert all(qs[i] <= qs[i + 1] + 1e-12 for i in range(len(qs) - 1)), qs
    print(f"\nBH q-values = {[round(q, 4) for q in qs]}  rejected = {bh['n_reject']}")
    assert bh["n_reject"] == 2, bh["n_reject"]
    try:
        from scipy.stats import false_discovery_control
        ref = [float(q) for q in false_discovery_control(_ps, method="bh")]
        assert max(abs(r - o) for r, o in zip(ref, qs)) < 1e-12
        print("  matches scipy.stats.false_discovery_control")
    except ImportError:
        print("  (scipy not installed, cross-check skipped)")

    # 6. Macro vs micro on eight groups of unequal size.
    mm = macro_micro([(900, 1000), (450, 500), (80, 100), (300, 400),
                      (1500, 2000), (90, 100), (180, 200), (70, 100)])
    print(f"\nmacro_micro  micro = {mm['micro']:.3f}  macro = {mm['macro']:.3f}  "
          f"delta = {mm['macro_minus_micro']:+.3f}")

    # 7. Wilson.
    lo, hi = wilson_interval(8000, 10000)
    print(f"wilson(8000/10000) = [{lo:.3f}, {hi:.3f}] %")
    assert lo < 80.0 < hi

    print("\nAll tests passed.")
