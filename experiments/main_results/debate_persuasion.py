#!/usr/bin/env python3
"""Debate persuasion: when the agents disagree at the start, how often does a right answer lose?"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from slmjury.analysis import paths
from slmjury.analysis.accuracy import expected_verdict
from slmjury.parsers.answer import get_dataset_type
from slmjury.parsers.normalizer import set_strict_labels

NOTES = """\
Debate can only change items where the three agents open with different
verdicts; when all three agree, the protocol stops. Among those items, this
counts the ones where at least one agent was right at the outset, and reports
the share that still ended with the wrong final verdict.

Only same-model configurations are counted (one model at three temperatures,
names containing "-t0+"), so differences in agent capability do not affect
the result.

Reads:  results/core/debate/<configuration>/<student>_<dataset>.json
Writes: results/analysis/main_results/debate_persuasion.csv

Run it from the repository root.
"""


def analyse(config_dir: Path) -> dict | None:
    items = disagreed = swayed = held = 0
    for path in sorted(config_dir.glob("*.json")):
        # File names are <student>_<dataset>.json and dataset names contain
        # underscores, so split once from the left.
        dataset = path.stem.split("_", 1)[1]
        dtype = get_dataset_type(dataset)
        for record in json.load(open(path)):
            rounds = record.get("debate_rounds") or []
            if not rounds:
                continue
            opening = [v["verdict"].lower()
                       for _, v in sorted(rounds[0].items())
                       if isinstance(v, dict) and v.get("verdict")]
            if len(opening) < 2:
                continue
            items += 1
            if len(set(opening)) == 1:
                continue                     # consensus, debate stops
            disagreed += 1
            oracle, _ = expected_verdict(record, dtype)
            if not any(v == oracle for v in opening):
                continue                     # no agent was right at the start
            if (record.get("judgement") or "").lower() == oracle:
                held += 1
            else:
                swayed += 1

    if not (swayed + held):
        return None
    return {
        "configuration": config_dir.name,
        "n_items": items,
        "n_initial_disagreement": disagreed,
        "pct_initial_disagreement": round(disagreed / items * 100, 2),
        "n_recoverable": swayed + held,
        "n_swayed_to_wrong": swayed,
        "pct_swayed_to_wrong": round(swayed / (swayed + held) * 100, 2),
    }


def main() -> None:
    # --help prints the summary above and exits; the script takes no options.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table
    debate = paths.core("debate")
    out = []
    for d in sorted(p for p in debate.iterdir()
                    if p.is_dir() and p.name != "evaluations"):
        if "-t0+" not in d.name:             # cross-architecture; skipped
            continue
        row = analyse(d)
        if row:
            out.append(row)

    out.sort(key=lambda r: -r["pct_swayed_to_wrong"])
    path = paths.analysis_dir("main_results") / "debate_persuasion.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"Saved: {path}  ({len(out)} configurations)")
    for r in out:
        print(f"  {r['configuration'][:44]:46} "
              f"disagree {r['pct_initial_disagreement']:>5.1f}%   "
              f"swayed to wrong {r['pct_swayed_to_wrong']:>5.1f}% "
              f"({r['n_swayed_to_wrong']}/{r['n_recoverable']})")


if __name__ == "__main__":
    main()
