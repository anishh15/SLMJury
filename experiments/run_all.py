#!/usr/bin/env python3
"""Run every analysis in dependency order, then the figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

NOTES = """\
Each step runs one script from experiments/ in a separate process with the
current Python interpreter and the current working directory. The run stops
at the first step that fails, and each step's wall-clock time is printed as
it finishes, with a summary at the end.

Three steps read the output of an earlier one: scaling_cost reads
large_judges/small_vs_large.csv, and family_bias/student_ordering and
significance/supplement read the family-bias matrices. The order runs those
inputs first.

The significance steps build the cache of aligned correctness vectors in
results/cache/significance/ the first time they run; later runs reuse it.

examples, from the repository root:
  python experiments/run_all.py                  # every step
  python experiments/run_all.py --list           # print the steps and stop
  python experiments/run_all.py --from family_bias
  python experiments/run_all.py --only significance/all_pairs

Results are read from results/ (or $SLMJURY_RESULTS, which the child
processes inherit) and written to results/analysis/<experiment>/ (tables) and
results/figures/ (plots and figures).
"""

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"

# (step name, script relative to experiments/, extra arguments)
STEPS: list[tuple[str, str, list[str]]] = [
    ("main_results",                     "main_results/analyze.py",                 []),
    ("large_judges",                     "large_judges/analyze.py",                 []),
    ("scaling_cost",                     "scaling_cost/analyze.py",                 []),
    ("family_bias",                      "family_bias/analyze.py",                  []),
    ("family_bias/student_ordering",     "family_bias/student_ordering.py",         []),
    ("prompt_sensitivity",               "prompt_sensitivity/analyze.py",           []),
    ("prompt_sensitivity/open_ended",    "prompt_sensitivity/analyze_open_ended.py", []),
    ("token_budget",                     "token_budget/analyze.py",                 []),
    ("adaptive_budget",                  "adaptive_budget/analyze.py",              []),
    ("judging_protocols",                "judging_protocols/analyze.py",            []),
    ("error_analysis",                   "error_analysis/analyze.py",               []),
    ("significance/small_vs_large",      "significance/analyze.py",                 ["--part", "small-vs-large"]),
    ("significance/all_pairs",           "significance/analyze.py",                 ["--part", "all-pairs"]),
    ("significance/open_ended_tests",    "significance/analyze.py",                 ["--part", "open-ended"]),
    ("significance/open_ended",          "significance/open_ended.py",              []),
    ("significance/supplement",          "significance/supplement.py",              []),
    ("main_results/debate_persuasion",   "main_results/debate_persuasion.py",       []),
    ("main_results/variance",            "main_results/analyze_variance.py",        []),
]

# ── Figures hook ────────────────────────────────────────────────────────────
# The last step draws the figures from results/analysis. It runs only when
# figures/make_figures.py exists; until then it is listed and skipped.
FIGURES_STEP = "figures"
FIGURES_SCRIPT = REPO / "figures" / "make_figures.py"


def all_steps() -> list[tuple[str, Path, list[str]]]:
    steps = [(name, EXPERIMENTS / script, args) for name, script, args in STEPS]
    steps.append((FIGURES_STEP, FIGURES_SCRIPT, []))
    return steps


def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:04.1f}s" if m else f"{s:.1f}s"


def main() -> int:
    steps = all_steps()
    names = [name for name, _, _ in steps]

    ap = argparse.ArgumentParser(
        description="Run every analysis in order, then the figures.", epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="print the steps in order and stop")
    group.add_argument("--from", dest="start", metavar="STEP", choices=names,
                       help="start at this step and run the rest")
    group.add_argument("--only", metavar="STEP", choices=names, help="run this step alone")
    args = ap.parse_args()

    if args.list:
        for i, (name, script, extra) in enumerate(steps, 1):
            cmd = " ".join([str(script.relative_to(REPO)), *extra])
            note = "" if script.exists() else "   (skipped: script not present)"
            print(f"{i:>2}. {name:<38} {cmd}{note}")
        return 0

    if args.only:
        selected = [s for s in steps if s[0] == args.only]
    elif args.start:
        selected = steps[names.index(args.start):]
    else:
        selected = steps

    timings: list[tuple[str, float, str]] = []
    t_all = time.perf_counter()
    for name, script, extra in selected:
        if not script.exists():
            if name == FIGURES_STEP:
                print(f"\n=== {name}: {script.relative_to(REPO)} not present, skipped", flush=True)
                timings.append((name, 0.0, "skipped"))
                continue
            print(f"\n=== {name}: {script} not found", file=sys.stderr)
            return 1
        cmd = [sys.executable, str(script), *extra]
        print(f"\n=== {name}: {' '.join([str(script.relative_to(REPO)), *extra])}", flush=True)
        t0 = time.perf_counter()
        code = subprocess.run(cmd).returncode
        dt = time.perf_counter() - t0
        if code != 0:
            timings.append((name, dt, f"failed ({code})"))
            print(f"=== {name} failed with exit code {code} after {_fmt(dt)}", flush=True)
            break
        timings.append((name, dt, "ok"))
        print(f"=== {name} done in {_fmt(dt)}", flush=True)

    total = time.perf_counter() - t_all
    print("\nStep timings")
    for name, dt, status in timings:
        print(f"  {name:<38} {_fmt(dt):>10}  {status}")
    print(f"  {'total':<38} {_fmt(total):>10}")
    failed = any(status.startswith("failed") for _, _, status in timings)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
