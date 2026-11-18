"""Answer normalization and comparison for numeric, LaTeX (MATH) and multiple-choice datasets.

MATH pairs are looked up in slmjury/parsers/math_labels.json, which scripts/build_math_labels.py builds from
the rules in ``_strip_format`` and ``_compare_latex_raw``; a pair the table lacks is compared live under a time limit.
"""

import hashlib
import json
import logging
import math
import re
import signal
import threading
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def answers_are_equivalent(predicted: str, ground_truth: str, dataset_type: str = "numeric") -> bool:
    """Whether ``predicted`` matches ``ground_truth`` for a dataset type ("numeric", "latex" or "multiple_choice").

    An empty answer or UNABLE_TO_EXTRACT never matches.
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
    """Normalize a numeric string for comparison: commas, currency symbols, units, fractions and word multipliers."""
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

LABEL_FILE = Path(__file__).resolve().parent / "math_labels.json"
TIME_LIMIT_S = 8     # per pair; a pair that hits it is not equivalent

_labels: Optional[dict] = None
_label_sha: Optional[str] = None
_strict = False
timed_out_pairs: list = []


class MissingLabel(LookupError):
    """Strict mode met a pair the label file does not hold."""


class ComparisonTimeout(BaseException):
    """The time limit fired. A BaseException, so no `except Exception` can swallow it."""


def set_strict_labels(on: bool = True) -> None:
    """In strict mode a pair missing from the label file raises instead of being computed."""
    global _strict
    _strict = on


def label_key(predicted: str, ground_truth: str) -> str:
    """The label file's key for one stripped (answer, gold) pair."""
    return predicted + "\x00" + ground_truth


def labels_digest(labels: dict) -> str:
    """SHA-256 of the labels alone, so a rebuild with the same labels keeps its sha."""
    text = json.dumps(labels, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _load_labels() -> dict:
    """The tracked labels, verified against their recorded sha, or {} without the file."""
    global _labels, _label_sha
    if _labels is None:
        if LABEL_FILE.exists():
            blob = json.loads(LABEL_FILE.read_text())
            if labels_digest(blob["labels"]) != blob["labels_sha256"]:
                raise ValueError(f"{LABEL_FILE} does not match its recorded sha")
            _labels, _label_sha = blob["labels"], blob["labels_sha256"]
        else:
            _labels = {}
    return _labels


def label_sha() -> str:
    """The loaded label file's sha. Raises when there is no label file."""
    _load_labels()
    if _label_sha is None:
        raise FileNotFoundError(f"{LABEL_FILE} not found: run python scripts/build_math_labels.py")
    return _label_sha


def _compare_latex(predicted: str, ground_truth: str) -> bool:
    """Label of one pair: the tracked file's, or the live rules under the time limit."""
    pred = predicted.strip()
    gold = ground_truth.strip()

    hit = _load_labels().get(label_key(pred, gold))
    if hit is not None:
        return hit
    if _strict:
        raise MissingLabel(
            f"no label for answer {pred!r} with gold {gold!r}: run python scripts/build_math_labels.py"
        )
    equivalent, timed_out = compare_with_limit(pred, gold)
    if timed_out:
        timed_out_pairs.append((pred, gold))
        logger.warning("LaTeX comparison hit %d s, scored not equivalent: %r, %r",
                       TIME_LIMIT_S, pred, gold)
    return equivalent


def compare_with_limit(pred: str, gold: str, seconds: float = TIME_LIMIT_S) -> Tuple[bool, bool]:
    """(equivalent, timed_out) from the rules, under a time limit where the platform has one."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        return _compare_latex_raw(pred, gold), False

    def expire(*_):
        raise ComparisonTimeout()

    previous = signal.signal(signal.SIGALRM, expire)
    try:
        try:
            # The interval re-fires the limit if a bare `except:` in a library eats it.
            signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
            return _compare_latex_raw(pred, gold), False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except ComparisonTimeout:
        return False, True
    finally:
        signal.signal(signal.SIGALRM, previous)


_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_UNICODE = [("√", "\\sqrt"), ("π", "\\pi"), ("∞", "\\infty"), ("∪", "\\cup"), ("≤", "\\le "),
            ("≥", "\\ge "), ("≠", "\\ne "), ("×", "\\times "), ("·", "\\cdot "), ("−", "-"),
            ("–", "-"), ("°", "")]


def _spellings(s: str) -> str:
    """Unicode and plain-text spellings of LaTeX, trailing punctuation, units, number formats."""
    s = s.strip()
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: "^{" + m.group(0).translate(_SUPERSCRIPTS) + "}", s)
    for u, l in _UNICODE:
        s = s.replace(u, l)
    s = re.sub(r"\\sqrt\s*(\d+|[a-zA-Z](?![a-zA-Z]))", r"\\sqrt{\1}", s)
    s = re.sub(r"(?<![\\a-zA-Z])sqrt\s*\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"(?<![\\a-zA-Z])sqrt\s*\{", r"\\sqrt{", s)
    s = re.sub(r"(?<![\\a-zA-Z])pi(?![a-zA-Z])", r"\\pi ", s)
    s = re.sub(r"(?<![\\a-zA-Z])(sin|cos|tan|cot|sec|csc|log|ln)(?![a-zA-Z])", r"\\\1", s)
    s = re.sub(r"\.\s*\}$", "}", s)
    s = re.sub(r"(\\\)|\\\]|\$|[.,;:?\s])+$", "", s)
    s = re.sub(r"^(\\\(|\\\[|\s)+", "", s).replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\\(?:mbox|textbf|textit|mathrm|text)\s*\{", r"\\text{", s)
    s = re.sub(r"(\\text\{[^{}]*\})\s*\^\s*\{?\d\}?", r"\1", s)
    s = s.replace("{,}", ",")
    s = re.sub(r"^\$?\\(?=\d)", "", s)
    s = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", s)
    return re.sub(r"(?<=\d)\^\{\\?(?:text|mathrm)?\{?(?:st|nd|rd|th)\}?\}", "", s)


def _mixed(m) -> str:
    """An integer beside a proper numeric fraction is a mixed number."""
    return f"({m[1]}+{m[2]}/{m[3]})" if int(m[2]) < int(m[3]) else m[0]


def _strip_format(s: str) -> str:
    """Formatting pass run before every strategy. Nothing here evaluates mathematics."""
    s = _spellings(s)
    if not s:
        return s
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r'(?<!\\)\\[,;:! ]', '', s)     # spacing commands, not a \\ row break
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r'\^\s*\{?\s*\\circ\s*\}?', '', s)
    # A leading name: x =, f(x) =, x_1 =, \theta =, x \in
    s = re.sub(r"^\$?\s*(?:[a-zA-Z](?:_\{?\w+\}?)?(?:\([^()=]*\))?"
               r"|\\(?:[a-zA-Z]+|mathbf\{[a-z]\})(?:_\{?\w+\}?)?)"
               r"\s*(?:=|\\in)\s*", "", s)
    s = re.sub(r"\+\s*\\infty", r"\\infty", s)
    s = re.sub(r"(?<![\d./])(\d+)\s+(\d+)\s*/\s*(\d+)(?![\d/])", _mixed, s)
    s = re.sub(r'\s+', '', s)
    # A numeric fraction stays one number: written a/b alone, kept as \frac beside a symbol
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"(?<![\d.}\w])(\d+)\\frac\{(\d+)\}\{(\d+)\}", _mixed, s)
    s = re.sub(r"(?<![\w)])(?:(?<=matrix\})|(?<!\}))\\frac\{(-?\d+)\}\{(\d+)\}"
               r"(?![\w(]|\\(?!end)[a-zA-Z])", r"\1/\2", s)
    return re.sub(r"(?<![\w}\)/])(-?\d+)/(\d+)(?=\\pi|[a-zA-Z])", r"\\frac{\1}{\2}", s)


def _split_top(s: str, sep: str = ",") -> list:
    """Split at a top-level separator, ignoring any inside brackets or braces."""
    out, depth, cur, i = [], 0, "", 0
    while i < len(s):
        if s.startswith("\\{", i) or s.startswith("\\}", i):
            depth += 1 if s[i + 1] == "{" else -1
            cur += s[i:i + 2]
            i += 2
            continue
        c = s[i]
        depth += (c in "([{") - (c in ")]}")
        if depth == 0 and s.startswith(sep, i):
            out.append(cur)
            cur = ""
            i += len(sep)
            continue
        cur += c
        i += 1
    return out + [cur]


_MATRIX = re.compile(r"^\\begin\{([pbB]?matrix)\}(.*)\\end\{\1\}$", re.S)


def _matrix(s: str) -> Optional[list]:
    """Rows of a matrix string, each a list of cells, or None."""
    m = _MATRIX.match(s)
    if not m:
        return None
    return [r.split("&") for r in m.group(2).strip().strip("\\").split("\\\\") if r.strip()]


def _list_items(s: str) -> list:
    """Items of a bare answer list. "and", "or" separate items; a \\pm item stands for two."""
    s = re.sub(r"\\text\s*\{\s*(?:,\s*)?(?:and|or)\s*\}", ",", s)
    s = re.sub(r"(?<=[\s,\d}])(?:and|or)(?=[\s\-\d\\])", ",", s)
    if s.startswith("\\{") and s.endswith("\\}"):
        s = s[2:-2]
    out = []
    for p in (x for x in _split_top(s) if x.strip()):
        if "\\pm" in p or "\\mp" in p:
            out += [p.replace("\\pm", "+").replace("\\mp", "-"),
                    p.replace("\\pm", "-").replace("\\mp", "+")]
        else:
            out.append(p)
    return out


def _same_items(a: list, b: list) -> bool:
    """Same items in any order."""
    if len(a) != len(b):
        return False
    left = list(b)
    for x in a:
        hit = next((i for i, y in enumerate(left) if _compare_latex_raw(x, y)), None)
        if hit is None:
            return False
        left.pop(hit)
    return True


def _compare_structured(p: str, g: str) -> bool:
    """Same shape and equal parts, for matrices, tuples, intervals, unions and lists."""
    def clean(s):
        s = s.strip().replace("$", "")
        for x in ("\\left", "\\right", "\\,", "\\!"):
            s = s.replace(x, "")
        return re.sub(r"\s+", " ", re.sub(r"(?<!\\)\\ ", "", s)).strip()

    p, g = clean(p), clean(g)
    pm, gm = _matrix(p), _matrix(g)
    if pm is not None and gm is not None:
        return ([len(r) for r in pm] == [len(r) for r in gm]
                and all(_compare_latex_raw(x, y) for r, q in zip(pm, gm) for x, y in zip(r, q)))
    if (pm is None) != (gm is None):  # a vector written as a matrix or as a tuple
        m, t = (pm, g) if pm is not None else (gm, p)
        v = [r[0] for r in m] if all(len(r) == 1 for r in m) else (m[0] if len(m) == 1 else None)
        if v is not None and t[:1] == "(" and t[-1:] == ")":
            e = _split_top(t[1:-1])
            return len(e) == len(v) and all(_compare_latex_raw(x, y) for x, y in zip(e, v))
    pu, gu = _split_top(p, "\\cup"), _split_top(g, "\\cup")
    if len(pu) > 1 and len(gu) > 1:
        return "\\cap" not in p + g and _same_items(pu, gu)
    if p[:1] in ("(", "[") and p[-1:] in (")", "]") and p[:1] == g[:1] and p[-1:] == g[-1:]:
        a, b = _split_top(p[1:-1]), _split_top(g[1:-1])
        if len(a) > 1 and len(a) == len(b):
            return all(_compare_latex_raw(x, y) for x, y in zip(a, b))
    if p[:1] not in ("(", "[") and g[:1] not in ("(", "["):
        a, b = _list_items(p), _list_items(g)
        if (len(a) > 1 or len(b) > 1) and len(a) == len(b):
            return _same_items(a, b)
    return False


_BASE = re.compile(r"^(\d+(?:\.\d+)?)_\{?(\d+|[a-zA-Z])\}?$")


def _strip_base(p: str, g: str) -> Tuple[str, str]:
    """A base subscript on one side only is notation; on both sides it stays."""
    a, b = _BASE.match(p), _BASE.match(g)
    if a and not b:
        return a.group(1), g
    if b and not a:
        return p, b.group(1)
    return p, g


def _same_word(p: str, g: str) -> bool:
    """Word answers ignore case; a choice letter ignores its parentheses."""
    if re.fullmatch(r"[A-Za-z(), ]*", p) and p.lower() == g.lower():
        return True
    one = r"\(?[A-Za-z]\)?"
    return (bool(re.fullmatch(one, p) and re.fullmatch(one, g))
            and p.strip("()").lower() == g.strip("()").lower())


def _close(a: float, b: float) -> bool:
    """Equal within 1e-9, and the difference is also small against the numbers themselves."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) < 1e-9 and abs(a - b) <= 1e-6 * max(abs(a), abs(b))


def _compare_latex_raw(predicted: str, ground_truth: str) -> bool:
    """Compare LaTeX expressions using string normalization, fraction eval, and SymPy."""
    raw_pred = predicted.strip()
    raw_gold = ground_truth.strip()
    pred, gold = _strip_base(_strip_format(raw_pred), _strip_format(raw_gold))

    # Strategy 1: Normalized string comparison
    pred_norm, gold_norm = _normalize_latex_string(pred), _normalize_latex_string(gold)
    if pred_norm == gold_norm or _same_word(pred_norm, gold_norm):
        return True

    # Strategy 2: Fraction evaluation (fast path)
    pred_frac = _evaluate_latex_fraction(pred)
    gold_frac = _evaluate_latex_fraction(gold)
    if pred_frac is not None and gold_frac is not None:
        if _close(pred_frac, gold_frac):
            return True

    # Strategy 3: SymPy symbolic comparison
    if _compare_symbolic(pred, gold):
        return True

    # Strategy 4: Numeric evaluation via SymPy
    pred_val = _evaluate_latex_numerically(pred)
    gold_val = _evaluate_latex_numerically(gold)
    if pred_val is not None and gold_val is not None:
        if _close(pred_val, gold_val):
            return True

    # Strategy 5: Simple numeric answers wrapped in unnecessary LaTeX
    pred_num = _plain_number(pred)
    gold_num = _plain_number(gold)
    if pred_num is not None and gold_num is not None:
        try:
            if abs(float(pred_num) - float(gold_num)) < 1e-9:
                return True
        except (ValueError, TypeError):
            pass

    # Strategy 6: Structured answers, compared part by part
    if _compare_structured(raw_pred, raw_gold):
        return True

    # Strategy 7: A closed numeric expression equals its last side
    if "=" in raw_pred:
        sides = raw_pred.replace("$", "").split("=")
        left = re.sub(r"\\(?:d?frac|sqrt|cdot|times|div|left|right)", "", "".join(sides[:-1]))
        if not re.search(r"[a-zA-Z]", left):
            return _compare_latex_raw(sides[-1], raw_gold)

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

    frac_match = re.fullmatch(r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}', s)
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
    s = re.sub(r'\s*,\s*', ',', s)
    return s.strip()


def _latex_parser():
    """SymPy's LaTeX parser and grammar. A missing antlr4 runtime raises ImportError here.

    Callers import this outside their try blocks: without the runtime, strategies
    3 and 4 cannot run, and that must stop the comparison rather than score it.
    """
    import antlr4
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.latex._antlr.latexlexer import LaTeXLexer
    from sympy.parsing.latex._antlr.latexparser import LaTeXParser

    return antlr4, parse_latex, LaTeXLexer, LaTeXParser


def _fully_parsed(text: str) -> bool:
    """True when SymPy's LaTeX grammar consumes the whole string, not just a prefix."""
    antlr4, _, LaTeXLexer, LaTeXParser = _latex_parser()

    lexer = LaTeXLexer(antlr4.InputStream(text))
    lexer.removeErrorListeners()
    stream = antlr4.CommonTokenStream(lexer)
    parser = LaTeXParser(stream)
    parser.removeErrorListeners()
    parser.math()
    return stream.LA(1) == antlr4.Token.EOF and parser.getNumberOfSyntaxErrors() == 0


def _compare_symbolic(pred: str, gold: str) -> bool:
    """Compare two LaTeX expressions symbolically using SymPy."""
    _, parse_latex, _, _ = _latex_parser()
    from sympy import simplify, N

    try:
        pred_clean = _prepare_for_sympy(pred)
        gold_clean = _prepare_for_sympy(gold)

        if not (_fully_parsed(pred_clean) and _fully_parsed(gold_clean)):
            return False
        pred_expr = parse_latex(pred_clean)
        gold_expr = parse_latex(gold_clean)

        if simplify(pred_expr - gold_expr) == 0:
            return True

        try:
            pv, gv = complex(N(pred_expr)), complex(N(gold_expr))
            return (pv != 0 and gv != 0 and _close(pv.real, gv.real)
                    and abs(pv.imag - gv.imag) < 1e-9)
        except (TypeError, ValueError):
            pass

        return False
    except Exception:
        return False


def _prepare_for_sympy(text: str) -> str:
    """Clean LaTeX string for SymPy parsing."""
    s = text.strip().replace("$", "")
    s = re.sub(r'\\text\s*\{[^}]*\}', '', s)
    # x(x+1) is a product; f, g, h and p stay functions
    return re.sub(r"(?<![\\a-zA-Z])([a-eijk-oqr-z])\(", r"\1\\cdot(", s).replace("\\pi(", "\\pi\\cdot(")


def _evaluate_latex_numerically(text: str) -> Optional[float]:
    """Evaluate a LaTeX expression to a numeric value via SymPy."""
    _, parse_latex, _, _ = _latex_parser()
    from sympy import N

    try:
        clean = _prepare_for_sympy(text)
        if not _fully_parsed(clean):
            return None
        expr = parse_latex(clean)
        value = N(expr)
        result = complex(value)
        if value != 0 and result == 0:      # underflow is not zero
            return None
        return result.real if abs(result.imag) < 1e-9 else None
    except Exception:
        return None


def _plain_number(text: str) -> Optional[str]:
    """A single plain number once LaTeX wrappers are stripped, otherwise None."""
    if not text:
        return None
    s = text.strip().replace("$", "")
    s = re.sub(r'\\boxed\s*\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\text\s*\{[^}]*\}', '', s)
    s = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s).strip()
    return s if re.fullmatch(r'-?(\d+\.?\d*|\.\d+)', s) else None


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
