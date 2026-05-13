"""Robust answer normalization and comparison across dataset types.

Compares predicted answers against ground truth for numeric (GSM8K, GSM-Plus),
LaTeX (MATH), and multiple choice (ARC-Easy, ARC-Challenge, HellaSwag,
WinoGrande, TruthfulQA) datasets. Uses SymPy for symbolic math comparison
and handles edge cases like fractions, units, and commas.
"""

import re
from fractions import Fraction
from typing import Optional, Tuple


def answers_are_equivalent(predicted: str, ground_truth: str, dataset_type: str = "numeric") -> bool:
    """Check if predicted answer matches ground truth based on dataset type.

    Args:
        predicted: The model's predicted answer.
        ground_truth: The correct answer.
        dataset_type: One of "numeric", "latex", "multiple_choice".

    Returns:
        True if answers are equivalent.
    """
    if not predicted or not ground_truth:
        return False

    pred = str(predicted).strip()
    gold = str(ground_truth).strip()

    if pred.upper() == "UNABLE_TO_EXTRACT":
        return False

    comparators = {
        "multiple_choice": _compare_multiple_choice,
        "latex": _compare_latex,
    }
    return comparators.get(dataset_type, _compare_numeric)(pred, gold)


def numbers_are_equivalent(a: str, b: str) -> bool:
    """Legacy alias — use answers_are_equivalent() instead."""
    return answers_are_equivalent(a, b, "numeric")


def normalize_numeric(value: str) -> str:
    """Normalize a numeric string for comparison.

    Handles commas, currency symbols, units, fractions, and word multipliers.

    Args:
        value: The numeric string to normalize.

    Returns:
        Normalized numeric string.
    """
    if not value or not isinstance(value, str):
        return str(value).strip() if value else ""

    text = value.strip()
    if text.upper() in ("UNABLE_TO_EXTRACT", "UNDEFINED", ""):
        return text.upper()

    # Remove LaTeX formatting
    text = _strip_latex(text)

    # Handle fractions (e.g., "1/2")
    fraction_val = _try_parse_fraction(text)
    if fraction_val is not None:
        return _format_number(fraction_val)

    # Handle word multipliers (million, billion, etc.)
    text, multiplier = _extract_multiplier(text)

    # Remove symbols and units
    text = _strip_symbols_and_units(text)
    text = text.replace(',', '')

    # Extract numeric portion
    match = re.search(r'-?\d+\.?\d*', text)
    if not match:
        return value.strip()

    try:
        num_value = float(match.group()) * multiplier
        return _format_number(num_value)
    except ValueError:
        return value.strip()



# --- Multiple choice comparison ---

def _compare_multiple_choice(predicted: str, ground_truth: str) -> bool:
    """Compare multiple choice answers with number-to-letter conversion."""
    pred = predicted.strip().upper()
    gold = ground_truth.strip().upper()

    if pred == gold:
        return True

    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
    return num_to_letter.get(pred, pred) == num_to_letter.get(gold, gold)


# --- LaTeX comparison ---

def _compare_latex(predicted: str, ground_truth: str) -> bool:
    """Compare LaTeX expressions using string normalization, fraction eval, and SymPy."""
    pred = predicted.strip()
    gold = ground_truth.strip()

    # Strategy 1: Normalized string comparison
    if _normalize_latex_string(pred) == _normalize_latex_string(gold):
        return True

    # Strategy 2: Fraction evaluation (fast path)
    pred_frac = _evaluate_latex_fraction(pred)
    gold_frac = _evaluate_latex_fraction(gold)
    if pred_frac is not None and gold_frac is not None:
        if abs(pred_frac - gold_frac) < 1e-9:
            return True

    # Strategy 3: SymPy symbolic comparison
    try:
        if _compare_symbolic(pred, gold):
            return True
    except Exception:
        pass

    # Strategy 4: Numeric evaluation via SymPy
    try:
        pred_val = _evaluate_latex_numerically(pred)
        gold_val = _evaluate_latex_numerically(gold)
        if pred_val is not None and gold_val is not None:
            if abs(pred_val - gold_val) < 1e-9:
                return True
            if gold_val != 0 and abs((pred_val - gold_val) / gold_val) < 1e-9:
                return True
    except Exception:
        pass

    # Strategy 5: Plain number extraction
    pred_num = _extract_number_from_latex(pred)
    gold_num = _extract_number_from_latex(gold)
    if pred_num is not None and gold_num is not None:
        try:
            if abs(float(pred_num) - float(gold_num)) < 1e-9:
                return True
        except (ValueError, TypeError):
            pass

    return False


def _evaluate_latex_fraction(text: str) -> Optional[float]:
    """Evaluate LaTeX fractions like \\frac{1}{2} to numeric values (fast path)."""
    if not text:
        return None

    s = text.strip().replace("$", "")

    try:
        return float(s)
    except ValueError:
        pass

    frac_match = re.match(r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}', s)
    if frac_match:
        try:
            num, den = float(frac_match.group(1)), float(frac_match.group(2))
            return num / den if den != 0 else None
        except ValueError:
            pass

    slash_match = re.match(r'^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$', s)
    if slash_match:
        try:
            num, den = float(slash_match.group(1)), float(slash_match.group(2))
            return num / den if den != 0 else None
        except ValueError:
            pass

    return None


def _normalize_latex_string(text: str) -> str:
    """Normalize LaTeX string: remove $, \text{}, single-char braces, and operators."""
    if not text:
        return ""

    s = text.strip().replace("$", "")
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\{(\w)\}', r'\1', s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = re.sub(r'\s*([+\-*/=])\s*', r'\1', s)
    return s.strip()


def _compare_symbolic(pred: str, gold: str) -> bool:
    """Compare two LaTeX expressions symbolically using SymPy."""
    try:
        from sympy import simplify, N
        from sympy.parsing.latex import parse_latex

        pred_clean = _prepare_for_sympy(pred)
        gold_clean = _prepare_for_sympy(gold)

        pred_expr = parse_latex(pred_clean)
        gold_expr = parse_latex(gold_clean)

        if simplify(pred_expr - gold_expr) == 0:
            return True

        try:
            if abs(complex(N(pred_expr)) - complex(N(gold_expr))) < 1e-9:
                return True
        except (TypeError, ValueError):
            pass

        return False
    except (ImportError, Exception):
        return False


def _prepare_for_sympy(text: str) -> str:
    """Clean LaTeX string for SymPy parsing."""
    s = text.strip().replace("$", "")
    s = re.sub(r'\\text\s*\{[^}]*\}', '', s)
    return s


def _evaluate_latex_numerically(text: str) -> Optional[float]:
    """Evaluate a LaTeX expression to a numeric value via SymPy."""
    try:
        from sympy import N
        from sympy.parsing.latex import parse_latex

        expr = parse_latex(_prepare_for_sympy(text))
        result = complex(N(expr))
        return result.real if abs(result.imag) < 1e-9 else None
    except Exception:
        num = _extract_number_from_latex(text)
        if num is not None:
            try:
                return float(num)
            except ValueError:
                pass
        return None


def _extract_number_from_latex(text: str) -> Optional[str]:
    """Extract a plain number from LaTeX, stripping formatting."""
    if not text:
        return None

    s = text.strip().replace("$", "").replace(",", "")
    s = re.sub(r'\\boxed\s*\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\text\s*\{[^}]*\}', '', s)

    match = re.search(r'-?\d+\.?\d*', s)
    return match.group() if match else None


# --- Numeric comparison ---

def _compare_numeric(predicted: str, ground_truth: str) -> bool:
    """Compare two numeric strings with normalization and tolerance."""
    norm_pred = normalize_numeric(predicted)
    norm_gold = normalize_numeric(ground_truth)

    if norm_pred == norm_gold:
        return True

    try:
        val_pred = float(norm_pred)
        val_gold = float(norm_gold)

        if val_pred == 0 and val_gold == 0:
            return True
        if val_pred == 0 or val_gold == 0:
            return abs(val_pred - val_gold) < 1e-9

        return abs(val_pred - val_gold) / max(abs(val_pred), abs(val_gold)) < 1e-9
    except (ValueError, TypeError):
        return False


# --- Internal helpers ---

def _strip_latex(text: str) -> str:
    """Remove LaTeX formatting artifacts."""
    text = re.sub(r'\\text\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '').replace('$', '')
    return text


def _try_parse_fraction(text: str) -> Optional[float]:
    """Try to parse text as a simple fraction (e.g., '1/2')."""
    if re.match(r'^-?\d+/\d+$', text.strip()):
        try:
            return float(Fraction(text.strip()))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def _extract_multiplier(text: str) -> Tuple[str, float]:
    """Extract word multipliers (million, billion, etc.) from text."""
    multiplier = 1.0
    for word, mult in [
        ('trillion', 1_000_000_000_000),
        ('billion', 1_000_000_000),
        ('million', 1_000_000),
        ('thousand', 1_000),
    ]:
        if word in text.lower():
            multiplier = mult
            text = re.sub(rf'\s*{word}s?\s*', ' ', text, flags=re.IGNORECASE)
            break
    return text.strip(), multiplier


def _strip_symbols_and_units(text: str) -> str:
    """Remove currency symbols, percentage signs, and common unit words."""
    text = re.sub(r'^[$£€¥₹]+|[$£€¥₹]+$', '', text)
    text = text.replace('%', '')

    units = [
        r'\s*dollars?', r'\s*cents?', r'\s*liters?', r'\s*litres?',
        r'\s*grams?', r'\s*kilograms?', r'\s*kg', r'\s*meters?',
        r'\s*metres?', r'\s*feet', r'\s*foot', r'\s*inches?',
        r'\s*miles?', r'\s*hours?', r'\s*minutes?', r'\s*seconds?',
        r'\s*days?', r'\s*weeks?', r'\s*months?', r'\s*years?',
        r'\s*people', r'\s*persons?', r'\s*items?', r'\s*pieces?',
        r'\s*blocks?', r'\s*cookies?', r'\s*oranges?', r'\s*apples?',
        r'\s*teachers?', r'\s*students?', r'\s*calories?', r'\s*points?',
        r'\s*gallons?',
    ]
    for unit in units:
        text = re.sub(rf'{unit}\s*\.?\s*$', '', text, flags=re.IGNORECASE)
    return text.strip()


def _format_number(value: float) -> str:
    """Format a number, removing unnecessary decimal places."""
    if value == int(value):
        return str(int(value))
    return f"{value:.10f}".rstrip('0').rstrip('.')
