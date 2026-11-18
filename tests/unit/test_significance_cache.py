"""The significance cache is rebuilt when a verdict file or the MATH label table differs, even with the same item ids."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from slmjury.analysis import paths
from slmjury.parsers import normalizer

DATA_PY = Path(__file__).resolve().parents[2] / "experiments" / "significance" / "data.py"


@pytest.fixture
def data(tmp_path, monkeypatch):
    """experiments/significance/data.py over a two-item panel written to a temporary results folder."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "results"))
    (tmp_path / "results" / "core" / "judgements").mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("significance_data", DATA_PY)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    sys.path.insert(0, str(DATA_PY.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(DATA_PY.parent))
    for student in module.STUDENTS:
        for dataset in module.DATASETS:
            write_verdicts(module, student, dataset, ["Correct", "Incorrect"])
    return module


def write_verdicts(module, student, dataset, verdicts):
    path = module.judgement_path("llama3.1-8b", student, dataset, 8192)
    path.parent.mkdir(parents=True, exist_ok=True)
    gold = "B" if dataset not in module.MATH_DATASETS else "4"
    path.write_text(json.dumps([
        {"problem_id": i, "question": "q", "ground_truth_answer": gold, "student_answer": gold,
         "student_reasoning": "", "judgement": v} for i, v in enumerate(verdicts)]))


def test_same_ids_new_verdicts_rebuild_the_vector(data):
    index = data.build_item_index()
    first = data.load_vector("llama3.1-8b", 8192, index)
    student, dataset = data.STUDENTS[0], "arc_easy"
    write_verdicts(data, student, dataset, ["Incorrect", "Incorrect"])
    second = data.load_vector("llama3.1-8b", 8192, index)
    start, _ = index.offsets[(student, dataset)]
    assert first.correct[start] == 1
    assert second.correct[start] == 0
    assert data.build_item_index().sha == index.sha


def test_unchanged_files_use_the_cache(data, monkeypatch):
    index = data.build_item_index()
    data.load_vector("llama3.1-8b", 8192, index)
    monkeypatch.setattr(data, "_verdict_pairs", lambda *a: pytest.fail("cache was not used"))
    data.load_vector("llama3.1-8b", 8192, index)


def test_new_label_table_rebuilds_the_vector(data, monkeypatch):
    index = data.build_item_index()
    data.load_vector("llama3.1-8b", 8192, index)
    old_key = data.vector_key("llama3.1-8b", 8192, index)
    monkeypatch.setattr(normalizer, "label_sha", lambda: "a different table")
    assert data.vector_key("llama3.1-8b", 8192, index) != old_key
    rebuilt = []
    real = data._verdict_pairs
    monkeypatch.setattr(data, "_verdict_pairs", lambda *a: rebuilt.append(1) or real(*a))
    data.load_vector("llama3.1-8b", 8192, index)
    assert rebuilt
