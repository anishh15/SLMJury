"""Tests for slmjury.analysis (paths, accuracy, pairwise, scoring) on tiny fixtures."""

import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import pytest

from slmjury.analysis import pairwise, paths, scoring
from slmjury.core import scoring_evaluator
from slmjury.parsers import normalizer
from slmjury.parsers.score import parse_score


@pytest.fixture
def results(tmp_path, monkeypatch):
    """An empty results root, used through $SLMJURY_RESULTS."""
    root = tmp_path / "results"
    root.mkdir()
    monkeypatch.setenv(paths.ENV_VAR, str(root))
    return root


@pytest.fixture
def accuracy():
    """The accuracy module."""
    from slmjury.analysis import accuracy as module
    return module


def _item(student, gold, verdict):
    return {"problem_id": 0, "question": "q", "ground_truth_reasoning": "",
            "ground_truth_answer": gold, "student_reasoning": "", "student_answer": student,
            "judge_response": verdict, "judgement": verdict}


def _write(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))
    return path


# --- paths ---


class TestPaths:
    def test_results_dir_follows_the_environment(self, results):
        assert paths.results_dir() == results

    def test_results_dir_defaults_to_results(self, monkeypatch):
        monkeypatch.delenv(paths.ENV_VAR, raising=False)
        assert paths.results_dir() == Path("results")

    def test_missing_folder_raises(self, results):
        with pytest.raises(paths.MissingResults, match=paths.ENV_VAR):
            paths.core("judgements")
        with pytest.raises(paths.MissingResults, match="results/core/judgements"):
            paths.core("judgements")
        with pytest.raises(paths.MissingResults, match="results/large_judges"):
            paths.experiment("large_judges")

    def test_missing_results_is_a_file_not_found_error(self):
        assert issubclass(paths.MissingResults, FileNotFoundError)

    def test_unknown_core_name_raises(self, results):
        with pytest.raises(ValueError):
            paths.core("not_a_folder")

    def test_core_is_under_results_core(self, results):
        (results / "core" / "judgements").mkdir(parents=True)
        assert paths.core("judgements") == results / "core" / "judgements"

    def test_analysis_dir_is_created(self, results):
        out = paths.analysis_dir("main_results")
        assert out == results / "analysis" / "main_results" and out.is_dir()

    def test_analysis_figures_dir_is_created(self, results):
        out = paths.analysis_figures_dir("error_analysis")
        assert out == results / "figures" / "analysis" / "error_analysis" and out.is_dir()

    def test_cache_dir_is_created(self, results):
        out = paths.cache_dir("significance")
        assert out == results / "cache" / "significance" and out.is_dir()


# --- accuracy ---


class TestAccuracy:
    @pytest.fixture
    def judgements(self, results):
        jdir = results / "core" / "judgements" / "judge-x"
        # gsm8k: one right verdict, one wrong, one unparsed (Undefined).
        _write(jdir / "student-y_gsm8k_t10.json", [
            _item("42", "42", "Correct"),
            _item("41", "42", "Correct"),
            _item("7", "7", "Undefined"),
        ])
        # arc_easy: both verdicts right.
        _write(jdir / "student-y_arc_easy_t10.json", [
            _item("B", "B", "Correct"),
            _item("A", "C", "Incorrect"),
        ])
        return paths.core("judgements") / "judge-x"

    def test_file_counts(self, accuracy, judgements):
        c = accuracy.file_counts("judge-x", "student-y", "gsm8k", 10,
                                 judgements / "student-y_gsm8k_t10.json")
        assert (c.total, c.correct, c.valid) == (3, 1, 2)
        c = accuracy.file_counts("judge-x", "student-y", "arc_easy", 10,
                                 judgements / "student-y_arc_easy_t10.json")
        assert (c.total, c.correct, c.valid) == (2, 2, 2)

    def test_micro_sums_counts_before_dividing(self, accuracy, judgements):
        counts = [accuracy.file_counts("judge-x", "student-y", ds, 10,
                                       judgements / f"student-y_{ds}_t10.json")
                  for ds in ("gsm8k", "arc_easy")]
        m = accuracy.micro(counts + [None])
        assert m == {"accuracy": 60.0, "ifr": 80.0, "correct": 3, "valid": 4,
                     "total": 5, "n_files": 2}

    def test_micro_of_nothing_is_none(self, accuracy):
        assert accuracy.micro([]) is None
        assert accuracy.micro([None]) is None

    def test_absent_file_is_none(self, accuracy, judgements):
        assert accuracy.file_counts("judge-x", "student-y", "gsm8k", 8192,
                                    judgements / "student-y_gsm8k_t8192.json") is None

    def test_truncated_file_is_none_and_listed(self, accuracy, judgements, monkeypatch):
        bad = judgements / "student-y_gsm8k_t8192.json"
        bad.write_text('[{"judgement": "Corr')
        monkeypatch.setattr(accuracy, "CORRUPT_FILES", [])
        assert accuracy.file_counts("judge-x", "student-y", "gsm8k", 8192, bad) is None
        assert accuracy.CORRUPT_FILES == [str(bad)]

    def test_import_changes_no_global_state(self):
        code = ("import logging; from slmjury.parsers import normalizer; "
                "import slmjury.analysis.accuracy; "
                "print(normalizer._strict, logging.root.manager.disable)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[2], check=True).stdout.split()
        assert out == ["False", "0"]

    def test_file_counts_silences_only_the_evaluator(self, accuracy, judgements, capsys, caplog):
        logging.disable(logging.NOTSET)
        accuracy.file_counts("judge-x", "student-y", "gsm8k", 10,
                             judgements / "student-y_gsm8k_t10.json")
        assert capsys.readouterr().out == ""
        assert logging.root.manager.disable == logging.NOTSET
        with caplog.at_level(logging.WARNING):
            logging.getLogger("slmjury.test").warning("still logging")
        assert "still logging" in caplog.text

    def test_verdict_pairs(self, accuracy):
        pairs = accuracy.verdict_pairs("gsm8k", [_item("41", "42", "Correct"),
                                                 _item("42", "42", "Incorrect")])
        assert pairs == [("correct", "incorrect"), ("incorrect", "correct")]


# --- pairwise ---


class TestReadChoice:
    @pytest.mark.parametrize("text, choice", [
        ("A", "A"),
        ("**B**.", "B"),
        ("The second one is better.</think>\nB", "B"),
        ("Both look fine.assistantfinalA", "A"),
        (r"\boxed{A}", "A"),
        ("After comparing them, the final answer is B.", "B"),
        ("Both answers A and B are correct.", "Undefined"),
        ("This is a good answer.", "Undefined"),
        ("", "Undefined"),
        (None, "Undefined"),
    ])
    def test_read_choice(self, text, choice):
        assert pairwise.read_choice(text) == choice


# --- scoring ---


class TestScoring:
    RECORDS = [{"judge_score": j, "oracle_score": o, "category": "writing"}
               for j, o in [(1, 1), (2, 1), (3, 2), (4, 5), (5, 4), (None, 3)]]

    def test_mtbench_is_not_rounded(self):
        r = scoring.mtbench(self.RECORDS)["overall"]
        expected = 10 / math.sqrt(132)        # Pearson r of the five scored pairs
        assert r["pearson"] == pytest.approx(expected, abs=1e-12)
        assert r["pearson"] != round(expected, 4)
        assert (r["n_valid"], r["n_total"]) == (5, 6)

    def test_matches_the_rounded_evaluator(self):
        raw = scoring.mtbench(self.RECORDS)["overall"]
        stored = scoring_evaluator.evaluate_mtbench(self.RECORDS)["overall"]
        assert stored["pearson"] == round(raw["pearson"], 4)
        assert stored["spearman"] == round(raw["spearman"], 4)

    def test_rounding_is_restored(self):
        scoring.mtbench(self.RECORDS)
        assert scoring_evaluator._safe_round(0.123456) == 0.1235


# --- score parser ---


class TestLongDigitRuns:
    @pytest.mark.parametrize("text", [
        "9" * 5000,
        r"\boxed{" + "9" * 5000 + "}",
        "**" + "1" * 5000 + "**",
    ])
    def test_long_digit_run_gives_no_score(self, text):
        assert parse_score(text) is None
