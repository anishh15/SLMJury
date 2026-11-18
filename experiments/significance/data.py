#!/usr/bin/env python3
"""Aligned per-item correctness vectors for the significance tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from slmjury.analysis import paths
from slmjury.analysis.accuracy import expected_verdict
from slmjury.parsers import normalizer
from slmjury.parsers.answer import get_dataset_type
from slmjury.parsers.normalizer import set_strict_labels

NOTES = """\
Every test in this folder compares two judges on the same items, so both
judges' results have to describe the same items in the same order. This module
does that alignment once, and the analysis scripts import it.

* A judge is correct on an item when its verdict matches the expected verdict
  from slmjury.analysis.accuracy.expected_verdict, the same rule
  JudgeEvaluator uses (MATH answers are compared through the shipped label
  table). An Undefined verdict counts as wrong and the item stays in the
  denominator, so accuracies here equal the per-judge accuracies elsewhere.
* Items are keyed by (student, dataset, problem_id). Every judgement file must
  carry the same ordered id list as the item index; a mismatch raises an error
  instead of comparing different items.
* Small judges are read from results/core/judgements, large and ladder judges
  from results/large_judges.

The panel is two students (qwen2.5-32b, llama3.1-8b) on eight datasets, 64,824
items, the set every judge was run on.

Cache. Reading the judgement files takes a while, so each (judge, budget) is
reduced once to three 0/1 strings and stored in results/cache/significance/:

  item_index.json                the ordered item keys and their hash
  correct_<judge>_t<budget>.json one judge at one budget

A cached vector is used only when its item index, MATH label table and verdict
files are unchanged since it was built. The folder can be deleted at any time;
the next run rebuilds it from the judgement files. Pass --rebuild to the
analysis scripts to ignore it.

Reads (under results/, or $SLMJURY_RESULTS):
  core/judgements/<judge>/<student>_<dataset>_t<budget>.json
  large_judges/<judge>/<student>_<dataset>_t<budget>.json

Run it from the repository root with --build to fill the cache ahead of the
analyses.
"""

OUT_NAME = "significance"

# ── The panel ───────────────────────────────────────────────────────────────
# Every judge was run on these two students and eight datasets. Adding more
# would leave some judges without results and break the pairing.
STUDENTS = ["qwen2.5-32b", "llama3.1-8b"]
DATASETS = ["gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
            "hellaswag", "winogrande", "truthfulqa"]
BUDGETS = [10, 8192]

# Parameter counts in billions, from slmjury/configs/models.yaml.
SLM_JUDGES = {
    "qwen2.5-1.5b": 1.5, "qwen2.5-3b": 3.0,  "qwen2.5-7b": 7.0,
    "qwen3-0.6b": 0.6,   "qwen3-1.7b": 1.7,  "qwen3-4b": 4.0,
    "qwen3-8b": 8.0,     "qwen3-14b": 14.0,
    "llama3.2-1b": 1.0,  "llama3.2-3b": 3.0, "llama3.1-8b": 8.0,
    "phi4-14b": 14.0,    "phi4mi-3.8b": 3.8,
    "phi4r-14b": 14.0,   "phi4rp-14b": 14.0, "phi4mr-3.8b": 3.8,
}
LARGE_JUDGES = {
    "qwen3.5-122b": 122.0, "glm-4.5-air": 106.0, "gpt-oss-120b": 120.0,
    "qwen2.5-72b": 72.0,   "llama3.3-70b": 70.0,
    "deepseek-r1-distill-llama-70b": 70.0,
}
# The mid-size judge of the Qwen3 scaling ladder. Its results sit with the
# large judges; only open_ended.py and supplement.py include it.
LADDER_JUDGES = {"qwen3-32b": 32.0}

# Judges that reason before answering have no 10-token row: the budget cuts
# them off mid-thought and nearly every verdict comes back Undefined.
THINKS_ONLY = {"phi4r-14b", "phi4rp-14b", "phi4mr-3.8b",
               "gpt-oss-120b", "deepseek-r1-distill-llama-70b", "glm-4.5-air"}

ALL_JUDGES = {**SLM_JUDGES, **LARGE_JUDGES}

DISPLAY = {
    "qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-3b": "Qwen2.5-3B",
    "qwen2.5-7b": "Qwen2.5-7B", "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
    "qwen3-14b": "Qwen3-14B", "llama3.2-1b": "Llama-3.2-1B",
    "llama3.2-3b": "Llama-3.2-3B", "llama3.1-8b": "Llama-3.1-8B",
    "phi4-14b": "Phi-4", "phi4mi-3.8b": "Phi-4-mini",
    "phi4r-14b": "Phi-4-Reasoning", "phi4rp-14b": "Phi-4-Reasoning-Plus",
    "phi4mr-3.8b": "Phi-4-mini-Reasoning",
    "qwen3.5-122b": "Qwen3.5-122B-A10B", "glm-4.5-air": "GLM-4.5-Air",
    "gpt-oss-120b": "GPT-OSS-120B", "qwen2.5-72b": "Qwen2.5-72B",
    "llama3.3-70b": "Llama-3.3-70B",
    "deepseek-r1-distill-llama-70b": "DeepSeek-R1-Distill-70B",
}

FAMILY = {
    **{k: "Qwen2.5" for k in ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-72b")},
    **{k: "Qwen3" for k in ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b",
                            "qwen3-14b", "qwen3.5-122b")},
    **{k: "Llama" for k in ("llama3.2-1b", "llama3.2-3b", "llama3.1-8b", "llama3.3-70b")},
    **{k: "Phi-4" for k in ("phi4-14b", "phi4mi-3.8b", "phi4r-14b", "phi4rp-14b",
                            "phi4mr-3.8b")},
    "gpt-oss-120b": "GPT-OSS", "glm-4.5-air": "GLM",
    "deepseek-r1-distill-llama-70b": "DeepSeek",
}

MATH_DATASETS = {"gsm8k", "gsm_plus", "math"}


def is_large(judge: str) -> bool:
    return judge in LARGE_JUDGES


def budgets_for(judge: str) -> list[int]:
    return [8192] if judge in THINKS_ONLY else list(BUDGETS)


def judge_dir(judge: str) -> Path:
    """results/core/judgements/<judge> for small judges, results/large_judges/<judge> otherwise."""
    if judge in SLM_JUDGES:
        return paths.core("judgements") / judge
    return paths.experiment("large_judges") / judge


def judgement_path(judge: str, student: str, dataset: str, budget: int) -> Path:
    """Where one judgement file lives. The caller reports a missing file as a missing cell."""
    return judge_dir(judge) / f"{student}_{dataset}_t{budget}.json"


def cache_dir() -> Path:
    return paths.cache_dir(OUT_NAME)


# ── Item index ──────────────────────────────────────────────────────────────

@dataclass
class ItemIndex:
    """The ordered item list every correctness vector is aligned to."""
    keys: list[tuple[str, str, int]]
    dataset_of: list[str]
    student_of: list[str]
    sha: str

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def offsets(self) -> dict[tuple[str, str], tuple[int, int]]:
        """(student, dataset) -> [start, end) span in the vector."""
        out, start, cur = {}, 0, None
        for i, (s, d, _) in enumerate(self.keys):
            if (s, d) != cur:
                if cur is not None:
                    out[cur] = (start, i)
                cur, start = (s, d), i
        if cur is not None:
            out[cur] = (start, len(self.keys))
        return out


def _index_path() -> Path:
    return cache_dir() / "item_index.json"


def build_item_index(force: bool = False) -> ItemIndex:
    """Read the problem ids once, from a reference judge, and fix their order.

    llama3.1-8b at 8,192 tokens is the reference because it has every cell. Any complete judge gives the same
    index: every judgement file is written from the same student solution file, so the ids come from the student.
    """
    p = _index_path()
    if p.exists() and not force:
        raw = json.loads(p.read_text())
        keys = [(s, d, int(i)) for s, d, i in raw["keys"]]
        return ItemIndex(keys, [k[1] for k in keys], [k[0] for k in keys], raw["sha"])

    ref = "llama3.1-8b"
    keys: list[tuple[str, str, int]] = []
    for student in STUDENTS:
        for dataset in DATASETS:
            fp = paths.require(judgement_path(ref, student, dataset, 8192),
                               f"reference judgement file ({ref}, {student}, {dataset})")
            with open(fp) as f:
                recs = json.load(f)
            keys.extend((student, dataset, int(r["problem_id"])) for r in recs)

    sha = hashlib.sha256(
        "\n".join(f"{s}|{d}|{i}" for s, d, i in keys).encode()
    ).hexdigest()[:16]

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"sha": sha, "n": len(keys), "keys": keys}))
    return ItemIndex(keys, [k[1] for k in keys], [k[0] for k in keys], sha)


# ── Correctness vectors ─────────────────────────────────────────────────────

_CACHE_VERSION = 2   # bump when the meaning of a cached vector changes


def _cache_path(judge: str, budget: int) -> Path:
    return cache_dir() / f"correct_{judge}_t{budget}.json"


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vector_key(judge: str, budget: int, index: ItemIndex) -> str:
    """Cache key of one vector: the item index, the MATH label table and the bytes of every verdict file."""
    try:
        labels = normalizer.label_sha()
    except FileNotFoundError:
        labels = "no label table"
    parts = [f"version {_CACHE_VERSION}", f"items {index.sha}", f"labels {labels}"]
    for student, dataset in index.offsets:
        fp = judgement_path(judge, student, dataset, budget)
        parts.append(f"{student}/{dataset} {_file_digest(fp) if fp.exists() else 'missing'}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


@dataclass
class JudgeVector:
    """One judge at one budget, aligned to the item index."""
    judge: str
    budget: int
    correct: list[int]       # 1 if the verdict matched the expected verdict, else 0
    valid: list[int]         # 1 if the verdict parsed as Correct/Incorrect (the IFR numerator)
    present: list[int]       # 1 if a judgement file covered this item
    n_files: int
    missing_cells: list[str]

    @property
    def n_present(self) -> int:
        return sum(self.present)

    def accuracy(self) -> float:
        n = self.n_present
        return sum(self.correct) / n * 100.0 if n else float("nan")

    def ifr(self) -> float:
        n = self.n_present
        return sum(self.valid) / n * 100.0 if n else float("nan")

    @property
    def complete(self) -> bool:
        return not self.missing_cells


def _verdict_pairs(dataset: str, records: list[dict]) -> list[tuple[str, str]]:
    """(judge verdict, expected verdict) per record, both lower-cased."""
    dtype = get_dataset_type(dataset)
    out = []
    for item in records:
        oracle, _ = expected_verdict(item, dtype)
        out.append((item.get("judgement", "Undefined").lower(), oracle))
    return out


def load_vector(
    judge: str,
    budget: int,
    index: ItemIndex | None = None,
    use_cache: bool = True,
    rebuild: bool = False,
) -> JudgeVector:
    """Aligned correctness vector for one (judge, budget), from the cache or the files."""
    index = index or build_item_index()
    cp = _cache_path(judge, budget)
    key = vector_key(judge, budget, index) if use_cache else None

    if use_cache and not rebuild and cp.exists():
        blob = json.loads(cp.read_text())
        if blob.get("key") == key:
            return JudgeVector(
                judge, budget,
                [int(c) for c in blob["correct"]],
                [int(c) for c in blob["valid"]],
                [int(c) for c in blob["present"]],
                blob["n_files"], blob["missing_cells"],
            )

    n = len(index)
    correct = [0] * n
    valid = [0] * n
    present = [0] * n
    n_files = 0
    missing: list[str] = []

    for (student, dataset), (start, end) in index.offsets.items():
        fp = judgement_path(judge, student, dataset, budget)
        if not fp.exists():
            missing.append(f"{student}/{dataset}")
            continue
        try:
            with open(fp) as f:
                records = json.load(f)
        except json.JSONDecodeError:
            missing.append(f"{student}/{dataset} (corrupt)")
            continue

        span = end - start
        if len(records) != span:
            raise ValueError(
                f"{judge} t{budget} {student}/{dataset}: {len(records)} records "
                f"but the item index expects {span}."
            )
        ids_here = [int(r["problem_id"]) for r in records]
        ids_index = [k[2] for k in index.keys[start:end]]
        if ids_here != ids_index:
            raise ValueError(
                f"{judge} t{budget} {student}/{dataset}: problem_id order differs "
                f"from the item index, so items would be paired wrongly."
            )

        n_files += 1
        for offset, (verdict, oracle) in enumerate(_verdict_pairs(dataset, records)):
            i = start + offset
            present[i] = 1
            if verdict in ("correct", "incorrect"):
                valid[i] = 1
                if verdict == oracle:
                    correct[i] = 1

    vec = JudgeVector(judge, budget, correct, valid, present, n_files, missing)
    if use_cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({
            "version": _CACHE_VERSION, "sha": index.sha, "key": key,
            "judge": judge, "budget": budget, "n_files": n_files,
            "missing_cells": missing,
            "correct": "".join(map(str, correct)),
            "valid": "".join(map(str, valid)),
            "present": "".join(map(str, present)),
        }))
    return vec


def load_all(
    judges=None, index: ItemIndex | None = None, rebuild: bool = False, quiet: bool = False,
) -> dict[tuple[str, int], JudgeVector]:
    """Every (judge, budget) vector of the panel, or of ``judges`` when given."""
    index = index or build_item_index()
    judges = judges or list(ALL_JUDGES)
    out = {}
    for judge in judges:
        for budget in budgets_for(judge):
            vec = load_vector(judge, budget, index, rebuild=rebuild)
            out[(judge, budget)] = vec
            if not quiet:
                flag = "" if vec.complete else f"  MISSING {len(vec.missing_cells)} cells"
                print(f"  {judge:<32} t{budget:<5} acc={vec.accuracy():7.4f}%  "
                      f"ifr={vec.ifr():7.4f}%  n={vec.n_present}{flag}")
    return out


def best_budget(vectors: dict[tuple[str, int], JudgeVector], judge: str) -> int:
    """The budget at which this judge scores higher.

    Only complete vectors are eligible, so a budget with missing cells cannot
    win by having been scored on a subset of the items.
    """
    options = [(v.accuracy(), b) for (j, b), v in vectors.items()
               if j == judge and v.complete and v.n_present]
    if not options:
        raise KeyError(f"no complete vector for {judge}")
    return max(options)[1]


def paired_correctness(
    vec_a: JudgeVector, vec_b: JudgeVector, index: ItemIndex,
) -> tuple[list[int], list[int], list[str]]:
    """Both vectors restricted to the items both judges covered, with each item's dataset."""
    a, b, clusters = [], [], []
    for i in range(len(index)):
        if vec_a.present[i] and vec_b.present[i]:
            a.append(vec_a.correct[i])
            b.append(vec_b.correct[i])
            clusters.append(index.dataset_of[i])
    return a, b, clusters


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the cache of aligned per-item correctness vectors.",
        epilog=NOTES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="build (or load) the vector of every judge and budget")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore cached vectors and the cached item index")
    ap.add_argument("--judges", nargs="+", default=None,
                    help="judge keys to build (default: the small and large judges)")
    args = ap.parse_args()
    set_strict_labels(True)  # every MATH pair must come from the shipped label table

    if not args.build:
        ap.error("pass --build")

    index = build_item_index(force=args.rebuild)
    print(f"Building vectors against item index {index.sha} ({len(index)} items)\n")
    vectors = load_all(args.judges, index, rebuild=args.rebuild)
    incomplete = [f"{j} t{b}" for (j, b), v in vectors.items() if not v.complete]
    print(f"\n{len(vectors)} vectors cached in {cache_dir()}")
    if incomplete:
        print(f"incomplete: {', '.join(incomplete)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
