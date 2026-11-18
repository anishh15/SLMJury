"""Open-ended metrics (SummEval, MT-Bench) without intermediate rounding.

The scoring evaluator rounds each metric, and so the SummEval ``overall`` means, to four decimals. These helpers
run its own functions on the raw records with rounding turned off, so each value is rounded once, when printed.
"""

from __future__ import annotations

from slmjury.core import scoring_evaluator as _SE

__all__ = ["summeval", "mtbench"]


def _unrounded(fn, *args, **kwargs):
    original = _SE._safe_round
    _SE._safe_round = lambda value, decimals=4: value
    try:
        return fn(*args, **kwargs)
    finally:
        _SE._safe_round = original


def summeval(records: list[dict]) -> dict:
    """``evaluate_summeval`` on raw SummEval records, without rounding."""
    return _unrounded(_SE.evaluate_summeval, records)


def mtbench(records: list[dict]) -> dict:
    """``evaluate_mtbench`` on raw MT-Bench records, without rounding."""
    return _unrounded(_SE.evaluate_mtbench, records)
