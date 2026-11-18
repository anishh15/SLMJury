"""Tests for the MATH and TruthfulQA loaders: pinned Parquet sources, study order and size checks."""

import json
import re

import pytest

from slmjury import data
from slmjury.data import hub, math as math_data, truthfulqa


def _math_rows(n_per_subject):
    """Fake Parquet rows per subject; each problem names its subject and position."""
    def rows(repo, filename, revision):
        subject = filename.split("/")[0]
        return [{"problem": f"{subject} {i}", "solution": f"so \\boxed{{{subject}-{i}}}",
                 "level": "Level 1", "type": subject} for i in range(n_per_subject[subject])]
    return rows


def _even_split(total):
    subjects = math_data.MATH_SUBJECTS
    counts = {s: total // len(subjects) for s in subjects}
    counts[subjects[0]] += total - sum(counts.values())
    return counts


class TestPinnedSources:
    def test_revisions_are_commit_hashes(self):
        for rev in (math_data.MATH_REVISION, truthfulqa.TRUTHFULQA_REVISION):
            assert re.fullmatch(r"[0-9a-f]{40}", rev)

    def test_study_sizes(self):
        assert math_data.MATH_SIZE == 5000
        assert truthfulqa.TRUTHFULQA_SIZE == 684


class TestMath:
    def test_loads_subjects_in_study_order_and_caches(self, tmp_path, monkeypatch):
        calls = []
        rows = _math_rows(_even_split(5000))

        def fake(repo, filename, revision):
            calls.append((repo, filename, revision))
            return rows(repo, filename, revision)

        monkeypatch.setattr(math_data, "read_parquet_rows", fake)
        cache = tmp_path / "math.json"
        problems = math_data.load_math(cache_path=cache)

        assert len(problems) == 5000
        assert [c[1].split("/")[0] for c in calls] == math_data.MATH_SUBJECTS
        assert {c[0] for c in calls} == {math_data.MATH_REPO}
        assert {c[2] for c in calls} == {math_data.MATH_REVISION}
        assert problems[0] == {"problem_id": 0, "question": "algebra 0",
                               "ground_truth_reasoning": "so \\boxed{algebra-0}",
                               "ground_truth_answer": "algebra-0"}
        assert problems[-1]["question"].startswith("precalculus")
        assert [p["problem_id"] for p in problems] == list(range(5000))

        monkeypatch.setattr(math_data, "read_parquet_rows", lambda *a: pytest.fail("cache not used"))
        assert math_data.load_math(cache_path=cache) == problems

    def test_count_mismatch_stops_before_caching(self, tmp_path, monkeypatch):
        monkeypatch.setattr(math_data, "read_parquet_rows", _math_rows(_even_split(4999)))
        cache = tmp_path / "math.json"
        with pytest.raises(hub.DatasetLoadError, match="4999 items .* the study uses 5000"):
            math_data.load_math(cache_path=cache)
        assert not cache.exists()

    def test_empty_download_stops(self, tmp_path, monkeypatch):
        monkeypatch.setattr(math_data, "read_parquet_rows", lambda *a: [])
        with pytest.raises(hub.DatasetLoadError, match="no items loaded"):
            math_data.load_math(cache_path=tmp_path / "math.json")

    def test_wrong_size_cache_stops(self, tmp_path):
        cache = tmp_path / "math.json"
        cache.write_text(json.dumps([{"problem_id": 0}]))
        with pytest.raises(hub.DatasetLoadError, match="delete it to download again"):
            math_data.load_math(cache_path=cache)

    def test_failed_download_stops(self, tmp_path, monkeypatch):
        def broken(*args, **kwargs):
            raise OSError("network down")

        pytest.importorskip("huggingface_hub")
        pytest.importorskip("pyarrow")
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", broken)
        with pytest.raises(hub.DatasetLoadError, match="could not download"):
            math_data.load_math(cache_path=tmp_path / "math.json")


class TestTruthfulQA:
    @staticmethod
    def _rows(n):
        return lambda repo, filename, revision: [
            {"question": f" Q{i}? ", "choices": ["w", "x", "y", "z"], "label": i % 4} for i in range(n)]

    def test_formats_choices_and_letters(self, tmp_path, monkeypatch):
        monkeypatch.setattr(truthfulqa, "read_parquet_rows", self._rows(684))
        problems = truthfulqa.load_truthfulqa(cache_path=tmp_path / "tqa.json")
        assert len(problems) == 684
        assert problems[1] == {"problem_id": 1, "question": "Q1?\nA) w  B) x  C) y  D) z",
                               "ground_truth_reasoning": "", "ground_truth_answer": "B"}

    def test_count_mismatch_stops(self, tmp_path, monkeypatch):
        monkeypatch.setattr(truthfulqa, "read_parquet_rows", self._rows(683))
        with pytest.raises(hub.DatasetLoadError, match="the study uses 684"):
            truthfulqa.load_truthfulqa(cache_path=tmp_path / "tqa.json")


class TestDispatcher:
    def test_empty_dataset_stops(self, monkeypatch):
        monkeypatch.setitem(data.DATASET_LOADERS, "gsm8k", lambda **kw: [])
        with pytest.raises(data.DatasetLoadError, match="gsm8k: no items loaded"):
            data.load_dataset("gsm8k")
