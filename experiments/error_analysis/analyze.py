#!/usr/bin/env python3
"""Error analysis: what kind of mistake does each judge make?"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from slmjury.analysis import paths
from slmjury.analysis.accuracy import expected_verdict
from slmjury.parsers.answer import get_dataset_type
from slmjury.parsers.normalizer import _strip_format
from slmjury.parsers.normalizer import set_strict_labels

NOTES = """\
Every item where a judge's verdict disagrees with the expected verdict, or
where the verdict did not parse, is a disagreement. Each one is sorted into a
category by rules, checked in this order:

  INSTRUCTION_FAILURE        the verdict did not parse (Correct/Incorrect missing)
  CRITICAL_THINKING_FAILURE  the ground truth is empty or "None" (GSM-Plus items
                             with no valid answer)
  FORMAT_MISMATCH            no student answer could be extracted, or a
                             multiple-choice answer uses letters where the ground
                             truth uses numbers (or the reverse)
  SEMANTIC_EQUIVALENCE       the answers match under a lenient comparison, or are
                             numerically close and the judge accepted them
  HALLUCINATED_COMPARISON    the judge's verdict contradicts a plain comparison of
                             the two answers
  REASONING_DISTRACTION      the judge's response talks about the student's
                             reasoning or method
  UNCLASSIFIED               none of the above

PARTIAL_CREDIT has a column in the output, but no rule assigns it. The lenient
comparisons only sort disagreements into categories. The expected verdict
itself comes from slmjury.analysis.accuracy.expected_verdict, the same rule
JudgeEvaluator uses.

The error distribution covers the sixteen small judges and, when their results
are present, the large judges and the Qwen3-32B ladder judge, all through the
same code. Phi-4 (phi4-14b) also gets a full list of its disagreements and a
chart by answer type.

Reads (under results/, or $SLMJURY_RESULTS), with no new runs needed:
  core/judgements/<judge>/<student>_<dataset>_t<budget>.json    (scripts/run_judge.py)
  large_judges/<judge>/<student>_<dataset>_t<budget>.json       (if present)

Writes results/analysis/error_analysis/:
  error_distribution.csv          per judge: disagreement counts, direction and category
  disagreements_phi4-14b.csv      every disagreement of phi4-14b, with truncated texts
and the plots, to results/figures/analysis/error_analysis/ (needs matplotlib):
  error_distribution.png          stacked bar chart of categories
  fp_fn_by_domain_phi4-14b.png    phi4-14b errors by answer type

Run it from the repository root.
"""

OUT_NAME = "error_analysis"

STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]
DATASETS = [
    "gsm8k", "gsm_plus", "math",
    "arc_easy", "arc_challenge",
    "hellaswag", "winogrande", "truthfulqa",
]
TOKEN_SETTINGS = [10, 8192]

# The judge whose disagreements are listed in full.
FOCUS_MODEL = "phi4-14b"

ALL_JUDGES = [
    "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b",
    "llama3.2-1b", "llama3.2-3b", "llama3.1-8b",
    "phi4-14b", "phi4r-14b", "phi4rp-14b", "phi4mi-3.8b", "phi4mr-3.8b",
]

# Large judges, read from results/large_judges. Their files use the same
# <student>_<dataset>_t<budget>.json layout, so the same code scores them.
LARGE_JUDGES = ["llama3.3-70b", "qwen2.5-72b", "gpt-oss-120b",
                "qwen3.5-122b", "glm-4.5-air",
                "deepseek-r1-distill-llama-70b"]

# Mid-size judge from the scaling ladder, also in results/large_judges. It is
# labelled "Ladder" rather than "SLM" or "Large".
LADDER_JUDGES = ["qwen3-32b"]

CATEGORIES = [
    "INSTRUCTION_FAILURE",
    "CRITICAL_THINKING_FAILURE",
    "FORMAT_MISMATCH",
    "SEMANTIC_EQUIVALENCE",
    "PARTIAL_CREDIT",
    "REASONING_DISTRACTION",
    "HALLUCINATED_COMPARISON",
    "UNCLASSIFIED",
]


def large_judge_dirs() -> dict[str, Path]:
    """judge -> the folder holding its judge folder, for every folder in results/large_judges."""
    root = paths.results_dir() / "large_judges"
    if not root.is_dir():
        return {}
    return {d.name: root for d in root.iterdir() if d.is_dir()}


# ── Lenient comparisons, used only to categorise ────────────────────────────

def _lenient_numeric_match(a: str, b: str) -> bool:
    """Whether two strings are the same number, after stripping units and symbols."""
    def _parse(s):
        s = str(s).strip().lower()
        for token in ("$", ",", "%", "\\", "dollars", "approximately", "about",
                      "\\$", "!", " "):
            s = s.replace(token, "")
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass
        try:
            return float(s)
        except ValueError:
            return None
    va, vb = _parse(a), _parse(b)
    if va is not None and vb is not None:
        return abs(va - vb) < 1e-6
    return False


def _lenient_numeric_close(a: str, b: str, rtol: float = 0.005) -> bool:
    """Whether two non-integers differ only by rounding (58.625 vs 58.63).

    An integer has no digits to round, so 2003 vs 2004 is a different answer
    however close the two are.
    """
    def _parse(s):
        s = str(s).strip().lower()
        for token in ("$", ",", "%", "\\", " ", "!"):
            s = s.replace(token, "")
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass
        try:
            return float(s)
        except ValueError:
            return None
    va, vb = _parse(a), _parse(b)
    if va is not None and vb is not None:
        if va.is_integer() or vb.is_integer():
            return False
        diff = abs(va - vb)
        # Close: within the relative tolerance or 0.05, and not equal.
        return diff <= max(abs(va) * rtol, 0.05) and diff > 1e-6
    return False


def _lenient_mc_match(a: str, b: str) -> bool:
    """Whether two multiple-choice answers start with the same option label."""
    def _clean(s):
        return re.sub(r"[^a-dA-D1-4]", "", s).upper()
    ca, cb = _clean(a), _clean(b)
    return bool(ca) and bool(cb) and ca[0] == cb[0]


def _mc_incomparable(a: str, b: str) -> bool:
    """Whether one answer is an option number and the other an option letter.

    For example ground truth "2" and student answer "A": the two labelling
    schemes cannot be compared.
    """
    a_clean = re.sub(r"[^a-zA-Z0-9]", "", a).strip()
    b_clean = re.sub(r"[^a-zA-Z0-9]", "", b).strip()
    if not a_clean or not b_clean:
        return False
    a_is_num = a_clean.isdigit()
    b_is_num = b_clean.isdigit()
    a_is_alpha = a_clean.isalpha() and len(a_clean) == 1
    b_is_alpha = b_clean.isalpha() and len(b_clean) == 1
    return (a_is_num and b_is_alpha) or (a_is_alpha and b_is_num)


def _parse_latex_fraction(s: str):
    """A LaTeX fraction (\\frac{19}{30}), plain fraction or number as a float, or None."""
    s = s.strip()
    m = re.match(r"\\?[dt]?frac\{([^}]+)\}\{([^}]+)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0].strip()) / float(parts[1].strip())
            except (ValueError, ZeroDivisionError):
                pass
    try:
        return float(s)
    except ValueError:
        return None


def _lenient_latex_match(a: str, b: str) -> bool:
    """Whether two LaTeX answers match after formatting is stripped.

    Formatting goes through the comparator's own ``_strip_format``, and a comma
    counts only as a thousands separator, so (2,5) never matches (25).
    """
    na, nb = _strip_format(a).lower(), _strip_format(b).lower()
    if na == nb:
        return True

    fa, fb = _parse_latex_fraction(a), _parse_latex_fraction(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-6

    ca = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", na)
    cb = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", nb)
    return "," not in ca + cb and _lenient_numeric_match(ca, cb)


def auto_classify_error(item: dict, dataset: str) -> str:
    """The error category of one disagreement, or "UNCLASSIFIED".

    ``item`` is a judgement record plus expected_verdict and judge_verdict; ``dataset`` decides the answer type.
    """
    judge_verdict = item.get("judge_verdict", "").lower()
    expected_verdict = item.get("expected_verdict", "").lower()
    judge_response = item.get("judge_response", "")
    student_answer = str(item.get("student_answer", "") or "")
    ground_truth = str(item.get("ground_truth_answer", "") or "")

    dataset_type = get_dataset_type(dataset)

    # 1. The judge gave no valid verdict.
    if judge_verdict not in ("correct", "incorrect"):
        return "INSTRUCTION_FAILURE"

    # 2. The problem has no valid answer (GSM-Plus critical-thinking items);
    #    the student should have said so.
    if ground_truth in ("", "None", "none", "N/A", "null"):
        return "CRITICAL_THINKING_FAILURE"

    # 3. No student answer could be extracted, but the judge gave a verdict.
    if student_answer in ("", "UNABLE_TO_EXTRACT"):
        return "FORMAT_MISMATCH"

    # 4. The answers match under a lenient comparison although the expected
    #    verdict is Incorrect.
    if expected_verdict == "incorrect":
        is_equivalent = False
        if dataset_type == "numeric":
            is_equivalent = _lenient_numeric_match(student_answer, ground_truth)
        elif dataset_type == "multiple_choice":
            is_equivalent = _lenient_mc_match(student_answer, ground_truth)
        elif dataset_type == "latex":
            is_equivalent = _lenient_latex_match(student_answer, ground_truth)

        if is_equivalent:
            return "SEMANTIC_EQUIVALENCE"

    # 4b. A rounding difference that the judge accepted.
    if dataset_type in ("numeric", "latex") and expected_verdict == "incorrect":
        if _lenient_numeric_close(student_answer, ground_truth):
            if judge_verdict == "correct":
                return "SEMANTIC_EQUIVALENCE"

    # 5. Option number against option letter.
    if dataset_type == "multiple_choice" and _mc_incomparable(student_answer, ground_truth):
        return "FORMAT_MISMATCH"

    # 6. The verdict contradicts a plain comparison of the answers.
    if judge_verdict == "incorrect" and expected_verdict == "correct":
        if student_answer.strip().lower() == ground_truth.strip().lower():
            return "HALLUCINATED_COMPARISON"
        match_fn = {
            "numeric": _lenient_numeric_match,
            "multiple_choice": _lenient_mc_match,
            "latex": _lenient_latex_match,
        }.get(dataset_type)
        if match_fn and match_fn(student_answer, ground_truth):
            return "HALLUCINATED_COMPARISON"

    if judge_verdict == "correct" and expected_verdict == "incorrect":
        if student_answer.strip() and ground_truth.strip():
            if dataset_type == "multiple_choice":
                s_clean = re.sub(r"[^a-dA-D]", "", student_answer).upper()
                g_clean = re.sub(r"[^a-dA-D]", "", ground_truth).upper()
                if s_clean and g_clean and s_clean[0] != g_clean[0]:
                    return "HALLUCINATED_COMPARISON"
            if dataset_type == "numeric":
                try:
                    s_val = float(student_answer.replace(",", "").replace("$", "").strip())
                    g_val = float(ground_truth.replace(",", "").replace("$", "").strip())
                    if abs(s_val - g_val) > 0.01:
                        return "HALLUCINATED_COMPARISON"
                except (ValueError, AttributeError):
                    pass
            if dataset_type == "latex":
                sf = _parse_latex_fraction(student_answer)
                gf = _parse_latex_fraction(ground_truth)
                if sf is not None and gf is not None and abs(sf - gf) > 1e-6:
                    return "HALLUCINATED_COMPARISON"

    # 7. The judge discusses the student's reasoning and reaches the wrong verdict.
    if judge_verdict != expected_verdict:
        response_lower = judge_response.lower()
        distraction_phrases = [
            "the reasoning is", "the student's reasoning",
            "the approach is", "the method is",
            "follows the correct steps", "correct approach",
            "shows good understanding", "demonstrates understanding",
            "the work shown", "the calculation process",
            "correctly solved", "correctly identified",
            "correct steps", "correct methodology",
        ]
        if any(phrase in response_lower for phrase in distraction_phrases):
            return "REASONING_DISTRACTION"

    return "UNCLASSIFIED"


# ── Disagreements ───────────────────────────────────────────────────────────

def collect_disagreements(judge: str, judgements_dir: Path) -> list[dict]:
    """Every item where the judge's verdict is wrong or did not parse, with its context."""
    disagreements = []

    for student in STUDENTS:
        for dataset in DATASETS:
            for max_tokens in TOKEN_SETTINGS:
                path = judgements_dir / judge / f"{student}_{dataset}_t{max_tokens}.json"
                if not path.exists():
                    continue

                with open(path) as f:
                    judgements = json.load(f)

                dataset_type = get_dataset_type(dataset)

                for item in judgements:
                    expected, student_answer = expected_verdict(item, dataset_type)
                    ground_truth = item.get("ground_truth_answer", "")
                    judge_verdict = item.get("judgement", "Undefined").lower()

                    if judge_verdict in ("correct", "incorrect") and judge_verdict != expected:
                        error_direction = (
                            "false_positive" if judge_verdict == "correct" else "false_negative"
                        )
                        category = auto_classify_error(
                            {**item, "expected_verdict": expected,
                             "judge_verdict": judge_verdict,
                             "student_answer": student_answer},
                            dataset,
                        )
                        disagreements.append({
                            "judge": judge,
                            "student": student,
                            "dataset": dataset,
                            "domain": get_dataset_type(dataset),
                            "max_tokens": max_tokens,
                            "problem_id": item.get("problem_id"),
                            "question": item.get("question", "")[:200],
                            "ground_truth_answer": ground_truth,
                            "student_answer": student_answer,
                            "student_reasoning": item.get("student_reasoning", "")[:300],
                            "judge_response": item.get("judge_response", "")[:500],
                            "expected_verdict": expected,
                            "judge_verdict": judge_verdict,
                            "error_direction": error_direction,
                            "auto_category": category,
                        })
                    elif judge_verdict not in ("correct", "incorrect"):
                        disagreements.append({
                            "judge": judge,
                            "student": student,
                            "dataset": dataset,
                            "domain": get_dataset_type(dataset),
                            "max_tokens": max_tokens,
                            "problem_id": item.get("problem_id"),
                            "question": item.get("question", "")[:200],
                            "ground_truth_answer": ground_truth,
                            "student_answer": student_answer,
                            "student_reasoning": item.get("student_reasoning", "")[:300],
                            "judge_response": item.get("judge_response", "")[:500],
                            "expected_verdict": expected,
                            "judge_verdict": "undefined",
                            "error_direction": "ifr_failure",
                            "auto_category": "INSTRUCTION_FAILURE",
                        })

    return disagreements


def compute_error_distribution(judges: list[str], judgements_dir: Path) -> list[dict]:
    """Disagreement counts by direction and category, one row per judge."""
    results = []
    for judge in judges:
        judge_dir = judgements_dir / judge
        if not judge_dir.exists():
            continue

        print(f"  Processing {judge}...")
        disagreements = collect_disagreements(judge, judgements_dir)
        counts = Counter(d["auto_category"] for d in disagreements)

        # The denominator is the same student x dataset x budget grid the
        # disagreements come from, not every JSON file in the folder.
        total_items = 0
        for student in STUDENTS:
            for dataset in DATASETS:
                for max_tokens in TOKEN_SETTINGS:
                    path = judge_dir / f"{student}_{dataset}_t{max_tokens}.json"
                    if path.exists():
                        with open(path) as fh:
                            total_items += len(json.load(fh))

        directions = Counter(d["error_direction"] for d in disagreements)
        row = {
            "judge": judge,
            "total_items": total_items,
            "total_disagreements": len(disagreements),
            "disagreement_rate": round(len(disagreements) / total_items * 100, 2) if total_items else 0,
            "false_positives": directions.get("false_positive", 0),
            "false_negatives": directions.get("false_negative", 0),
            "ifr_failures": directions.get("ifr_failure", 0),
            "fp_fn_ratio": (round(directions.get("false_positive", 0)
                                  / directions["false_negative"], 2)
                            if directions.get("false_negative") else None),
        }
        for cat in CATEGORIES:
            row[cat] = counts.get(cat, 0)
            row[f"{cat}_pct"] = (round(counts.get(cat, 0) / len(disagreements) * 100, 1)
                                 if disagreements else 0)

        results.append(row)

    return results


# ── Charts ──────────────────────────────────────────────────────────────────

def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the PNG charts.")
        return None
    return plt


def plot_error_distribution(dist_rows: list[dict], out_path: Path):
    """Stacked bar chart of error categories per judge."""
    plt = _pyplot()
    if plt is None:
        return
    judges = [r["judge"] for r in dist_rows]
    categories_to_plot = [c for c in CATEGORIES if any(r.get(c, 0) > 0 for r in dist_rows)]

    if not categories_to_plot:
        print("  No errors to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(judges) * 0.8), 7))

    colors = {
        "INSTRUCTION_FAILURE": "#e74c3c",
        "FORMAT_MISMATCH": "#f39c12",
        "SEMANTIC_EQUIVALENCE": "#3498db",
        "PARTIAL_CREDIT": "#9b59b6",
        "REASONING_DISTRACTION": "#2ecc71",
        "HALLUCINATED_COMPARISON": "#e67e22",
        "UNCLASSIFIED": "#95a5a6",
    }

    x = np.arange(len(judges))
    width = 0.6
    bottom = np.zeros(len(judges))

    for cat in categories_to_plot:
        values = [r.get(cat, 0) for r in dist_rows]
        ax.bar(x, values, width, bottom=bottom, label=cat.replace("_", " ").title(),
               color=colors.get(cat, "#95a5a6"), edgecolor="white", linewidth=0.5)
        bottom += np.array(values)

    ax.set_xlabel("Judge Model", fontsize=12)
    ax.set_ylabel("Number of Disagreements", fontsize=12)
    ax.set_title("Error Category Distribution by Judge", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(judges, rotation=45, ha="right", fontsize=9)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fp_fn_by_domain(disagreements: list[dict], judge: str, out_path: Path):
    """False positives, false negatives and unparsed verdicts by answer type, per budget."""
    plt = _pyplot()
    if plt is None:
        return
    domains = sorted(set(d["domain"] for d in disagreements))
    token_settings = sorted(set(d["max_tokens"] for d in disagreements))

    fig, axes = plt.subplots(1, len(token_settings), figsize=(7 * len(token_settings), 5))
    if len(token_settings) == 1:
        axes = [axes]

    fig.suptitle(f"Error Direction by Domain -- {judge}", fontsize=14, fontweight="bold")

    for ax, ts in zip(axes, token_settings):
        subset = [d for d in disagreements if d["max_tokens"] == ts]
        fp_counts = Counter(d["domain"] for d in subset if d["error_direction"] == "false_positive")
        fn_counts = Counter(d["domain"] for d in subset if d["error_direction"] == "false_negative")
        ifr_counts = Counter(d["domain"] for d in subset if d["error_direction"] == "ifr_failure")

        x = np.arange(len(domains))
        width = 0.25
        ax.bar(x - width, [fp_counts.get(d, 0) for d in domains], width,
               label="False Positive", color="#e74c3c", edgecolor="white")
        ax.bar(x, [fn_counts.get(d, 0) for d in domains], width,
               label="False Negative", color="#3498db", edgecolor="white")
        ax.bar(x + width, [ifr_counts.get(d, 0) for d in domains], width,
               label="IFR Failure", color="#95a5a6", edgecolor="white")

        ax.set_title(f"t{ts}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, fontsize=10)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table
    out_dir = paths.analysis_dir(OUT_NAME)
    judgements_dir = paths.core("judgements")

    available_judges = sorted(
        d.name for d in judgements_dir.iterdir()
        if d.is_dir() and d.name in ALL_JUDGES
    )
    print(f"Available judges: {', '.join(available_judges)}")
    print(f"Focus model: {FOCUS_MODEL}\n")

    # ── Disagreements of the focus model ───────────────────────────────────
    print(f"Collecting disagreements for {FOCUS_MODEL}...")
    focus_disagreements = collect_disagreements(FOCUS_MODEL, judgements_dir)
    print(f"  Found {len(focus_disagreements)} disagreements")

    cat_counts = Counter(d["auto_category"] for d in focus_disagreements)
    print("\n  By category:")
    for cat in CATEGORIES:
        if cat_counts.get(cat, 0) > 0:
            print(f"    {cat:<30} {cat_counts[cat]:>5} "
                  f"({cat_counts[cat] / len(focus_disagreements) * 100:.1f}%)")

    dir_counts = Counter(d["error_direction"] for d in focus_disagreements)
    print("\n  By direction:")
    for direction, count in sorted(dir_counts.items()):
        print(f"    {direction:<20} {count:>5}")

    ts_counts = Counter(d["max_tokens"] for d in focus_disagreements)
    print("\n  By token setting:")
    for ts, count in sorted(ts_counts.items()):
        print(f"    t{ts:<20} {count:>5}")

    focus_csv = out_dir / f"disagreements_{FOCUS_MODEL}.csv"
    if focus_disagreements:
        fields = list(focus_disagreements[0].keys())
        with open(focus_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(focus_disagreements)
        print(f"\nSaved: {focus_csv}")

    # ── Error distribution across judges ───────────────────────────────────
    print("\nComputing the error distribution across all judges...")
    dist_rows = compute_error_distribution(available_judges, judgements_dir)
    for r in dist_rows:
        r["judge_type"] = "SLM"

    large_dirs = large_judge_dirs()
    for label, roster in (("Large", LARGE_JUDGES), ("Ladder", LADDER_JUDGES)):
        present = sorted(j for j in large_dirs if j in roster)
        if not present:
            continue
        print(f"\n  {label} judges: {', '.join(present)}")
        by_root: dict[Path, list[str]] = {}
        for j in present:
            by_root.setdefault(large_dirs[j], []).append(j)
        for root, judges in by_root.items():
            rows = compute_error_distribution(judges, root)
            for r in rows:
                r["judge_type"] = label
            dist_rows += rows

    print(f"\n  {'Judge':<16} {'Total':>7} {'Disagree':>9} {'Rate':>7} {'IFR Fail':>9} "
          f"{'Fmt':>5} {'Sem':>5} {'Uncl':>6}")
    print("  " + "-" * 72)
    for r in dist_rows:
        print(f"  {r['judge']:<16} {r['total_items']:>7} {r['total_disagreements']:>9} "
              f"{r['disagreement_rate']:>6.1f}% {r.get('INSTRUCTION_FAILURE', 0):>9} "
              f"{r.get('FORMAT_MISMATCH', 0):>5} {r.get('SEMANTIC_EQUIVALENCE', 0):>5} "
              f"{r.get('UNCLASSIFIED', 0):>6}")

    dist_csv = out_dir / "error_distribution.csv"
    if dist_rows:
        fields = list(dist_rows[0].keys())
        with open(dist_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dist_rows)
        print(f"\nSaved: {dist_csv}")

    # ── Charts ─────────────────────────────────────────────────────────────
    plot_dir = paths.analysis_figures_dir(OUT_NAME)
    plot_error_distribution(dist_rows, plot_dir / "error_distribution.png")
    plot_fp_fn_by_domain(focus_disagreements, FOCUS_MODEL,
                         plot_dir / f"fp_fn_by_domain_{FOCUS_MODEL}.png")


if __name__ == "__main__":
    main()
