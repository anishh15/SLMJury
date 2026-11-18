#!/usr/bin/env python3
"""Run-to-run variance: how often two runs of the same judge give different verdicts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from slmjury.analysis import paths
from slmjury.configs import load_models_config

NOTES = """\
The token-budget single-factor study runs its own baseline conditions
(baseline_t10 and baseline_t8192), with the same prompts, items and code as
the original individual-judge runs. Comparing the two runs item by item shows
how reproducible each judge's verdicts are.

Greedy decoding with vLLM is not fully deterministic. Continuous batching
changes the order of floating-point reductions, so a token whose top two
logits are nearly tied can go either way, and one changed token changes the
rest of the generation. Confident judges are affected little, unsure judges
more.

Judges with enable_thinking in slmjury/configs/models.yaml sample at
temperature 0.6 at 8,192 tokens, so their rows at that budget are labelled
"sampled". At 10 tokens thinking is off for every judge and decoding is
greedy.

Reads:
  results/core/judgements/<judge>/<student>_<dataset>_t<budget>.json
  results/token_budget/single_factor/<judge>/baseline_t<budget>/<student>_<dataset>.json
Writes: results/analysis/main_results/run_variance.csv

Run it from the repository root.
"""

DATASETS = ["gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
            "hellaswag", "winogrande", "truthfulqa"]
STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]


def thinking_judges() -> set[str]:
    """Judges that sample rather than decode greedily at 8,192 tokens."""
    cfg = load_models_config()
    return {name for name, spec in (cfg.get("judge_models") or {}).items()
            if isinstance(spec, dict) and spec.get("enable_thinking")}


def load(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


def main() -> None:
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    core = paths.core("judgements")
    rerun = paths.require(paths.experiment("token_budget") / "single_factor",
                          "results/token_budget/single_factor")
    thinking = thinking_judges()

    out = []
    for budget, condition in ((10, "baseline_t10"), (8192, "baseline_t8192")):
        for judge in sorted(p.name for p in rerun.iterdir() if p.is_dir()):
            items = verdict_diff = response_diff = 0
            skipped = 0
            for student in STUDENTS:
                for dataset in DATASETS:
                    a = load(core / judge / f"{student}_{dataset}_t{budget}.json")
                    b = load(rerun / judge / condition / f"{student}_{dataset}.json")
                    if a is None or b is None:
                        skipped += 1
                        continue
                    if len(a) != len(b):
                        # Different item counts cannot be compared position
                        # by position, so the cell is skipped.
                        skipped += 1
                        continue
                    for x, y in zip(a, b):
                        items += 1
                        if x.get("judgement") != y.get("judgement"):
                            verdict_diff += 1
                        if (x.get("judge_response") or "") != (y.get("judge_response") or ""):
                            response_diff += 1
            if not items:
                continue
            greedy = not (judge in thinking and budget == 8192)
            out.append({
                "judge": judge,
                "max_tokens": budget,
                "decoding": "greedy" if greedy else "sampled",
                "n_items": items,
                "n_cells_skipped": skipped,
                "verdict_disagreement_pct": round(verdict_diff / items * 100, 4),
                "response_disagreement_pct": round(response_diff / items * 100, 4),
            })

    path = paths.analysis_dir("main_results") / "run_variance.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"Saved: {path}  ({len(out)} rows)")

    greedy = [r for r in out if r["decoding"] == "greedy"]
    lo = min(greedy, key=lambda r: r["verdict_disagreement_pct"])
    hi = max(greedy, key=lambda r: r["verdict_disagreement_pct"])
    print(f"  greedy verdict disagreement: "
          f"{lo['verdict_disagreement_pct']:.2f}% ({lo['judge']} B={lo['max_tokens']}) "
          f"to {hi['verdict_disagreement_pct']:.2f}% ({hi['judge']} B={hi['max_tokens']})")


if __name__ == "__main__":
    main()
