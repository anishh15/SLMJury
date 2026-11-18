#!/usr/bin/env python3
"""Build or check slmjury/parsers/math_labels.json, the label of every MATH answer pair in results/.

Run from the repository root: python scripts/build_math_labels.py [--check] [--out FILE].
Building needs sympy==1.13.1 and antlr4-python3-runtime 4.11; --check reports pairs the table lacks.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

from slmjury.analysis import paths
from slmjury.parsers.answer import extract_answer
from slmjury.parsers.normalizer import (
    LABEL_FILE, TIME_LIMIT_S, compare_with_limit, label_key, labels_digest,
)

SYMPY = "1.13.1"
ANTLR = "4.11."                  # the runtime that SymPy 1.13's generated LaTeX grammar targets

# Folders under results/ that hold no per-item records with a MATH answer.
NOT_JUDGEMENTS = {"analysis", "cache", "figures", "evaluations", "summaries", "scoring",
                  "scoring_evaluations", "mtbench_references", "mtbench_responses"}


def check_versions() -> dict:
    """The pinned versions, or stop: labels depend on the parser's version."""
    sympy = importlib.metadata.version("sympy")
    antlr = importlib.metadata.version("antlr4-python3-runtime")
    if sympy != SYMPY or not antlr.startswith(ANTLR):
        sys.exit(f"need sympy=={SYMPY} and antlr4-python3-runtime=={ANTLR}*, "
                 f"found sympy {sympy}, antlr4 {antlr}")
    return {"sympy": sympy, "antlr4": antlr}


def math_files() -> list[Path]:
    """Every MATH result file under results/, student solutions included."""
    root = paths.require(paths.results_dir(), "results folder")
    found, packed = [], []
    for folder, subdirs, names in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in NOT_JUDGEMENTS]
        for name in names:
            if name.endswith(".json") and ("_math" in name or name == "math.json"):
                found.append(Path(folder) / name)
            elif name.endswith(".json.gz") and "_math" in name \
                    and not (Path(folder) / name[:-3]).exists():
                packed.append(Path(folder) / name)
    if packed:
        sys.exit(f"{len(packed)} MATH files are still compressed; decompress them first: "
                 f"{packed[:3]}")
    return sorted(found)


def pair_of(item: dict) -> tuple[str, str] | None:
    """The pair ``answers_are_equivalent`` looks up for a record, or None if it stops earlier."""
    answer = item.get("student_answer", "")
    if not answer or answer == "UNABLE_TO_EXTRACT":
        answer = extract_answer(item.get("student_reasoning", ""), "latex")
    gold = item.get("ground_truth_answer", "")
    if not answer or not gold:
        return None
    answer, gold = str(answer).strip(), str(gold).strip()
    if answer.upper() == "UNABLE_TO_EXTRACT":
        return None
    return answer, gold


def scan_file(path: Path) -> tuple[Path, int, set | None]:
    """(path, items, unique pairs). Pairs are None when the file has no answer field (pairwise)."""
    with open(path) as f:
        items = json.load(f)
    if not isinstance(items, list) or not items or "ground_truth_answer" not in items[0] \
            or "student_reasoning" not in items[0]:
        return path, 0, None
    return path, len(items), {p for p in map(pair_of, items) if p is not None}


def scan_all(pool) -> set:
    """Unique pairs over every MATH file."""
    files = math_files()
    pairs: set = set()
    n_items = n_read = 0
    skipped = []
    for path, n, found in pool.imap_unordered(scan_file, files, 8):
        if found is None:
            skipped.append(path)
            continue
        pairs |= found
        n_items += n
        n_read += 1
    other = [p for p in skipped if "pairwise" not in p.parts]
    if other:
        sys.exit(f"{len(other)} MATH files have no answer field: {other[:3]}")
    print(f"read {n_read} MATH files, {n_items:,} items, {len(pairs):,} unique pairs "
          f"({len(skipped)} pairwise files have no answer field)", flush=True)
    return pairs


def _init_worker() -> None:
    warnings.filterwarnings("ignore")


def label_pair(pair: tuple[str, str]) -> tuple[str, str, bool, bool]:
    """(answer, gold, equivalent, timed_out). Any fault in the rules propagates."""
    equivalent, timed_out = compare_with_limit(*pair)
    return (*pair, bool(equivalent), timed_out)


def build_labels(pool, pairs: set, versions: dict, out: Path) -> None:
    labels, timeouts = {}, []
    for i, (answer, gold, equivalent, timed_out) in enumerate(
            pool.imap_unordered(label_pair, sorted(pairs), 40), 1):
        labels[label_key(answer, gold)] = equivalent
        if timed_out:
            timeouts.append([answer, gold])
        if i % 4000 == 0:
            print(f"  {i:,} of {len(pairs):,} labelled, {len(timeouts)} timeouts", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and rename, so an interrupted run leaves no half-written table
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "labels": labels, "labels_sha256": labels_digest(labels),
        "time_limit_s": TIME_LIMIT_S, "timeouts": sorted(timeouts), "versions": versions,
    }, sort_keys=True, indent=0) + "\n")
    tmp.replace(out)
    print(f"wrote {out}: {len(labels):,} labels, {sum(labels.values()):,} true, "
          f"{len(timeouts)} timeouts", flush=True)


def read_label_file(path: Path) -> dict:
    """The table as written, with its sha checked."""
    blob = json.loads(path.read_text())
    if labels_digest(blob["labels"]) != blob["labels_sha256"]:
        sys.exit(f"{path} does not match its recorded sha")
    return blob


def coverage(pool, labels: dict) -> int:
    """Read every MATH file again and count the pairs the table does not hold."""
    missing = [p for p in scan_all(pool) if label_key(*p) not in labels]
    for answer, gold in missing[:20]:
        print(f"  missing: {answer!r} with {gold!r}")
    print(f"coverage check: {len(missing)} missing pairs")
    return len(missing)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build or check the MATH equivalence table.")
    ap.add_argument("--check", action="store_true",
                    help="confirm the table holds every pair found under results/; write nothing")
    ap.add_argument("--out", type=Path, default=LABEL_FILE,
                    help="table to write, or to check with --check "
                         "(default: the shipped slmjury/parsers/math_labels.json)")
    args = ap.parse_args()

    with mp.Pool(mp.cpu_count(), _init_worker) as pool:
        if not args.check:
            versions = check_versions()
            build_labels(pool, scan_all(pool), versions, args.out)
        blob = read_label_file(args.out)
        missing = coverage(pool, blob["labels"])
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
