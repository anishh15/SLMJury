"""CLI tests for the runners: judge budgets, debate budgets, results paths and loading order."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNERS = sorted(REPO.glob("scripts/run_*.py")) + sorted(REPO.glob("experiments/*/run_*.py"))


def load_script(path: Path):
    """Import a script from scripts/ or experiments/ as a module, without running main()."""
    spec = importlib.util.spec_from_file_location(f"runner_{path.stem}_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _solutions(root: Path, student="qwen2.5-32b", datasets=("gsm8k",)) -> Path:
    for ds in datasets:
        f = root / student / f"{ds}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps([{"problem_id": 0}]))
    return root


# --- run_judge.py ---


class FakeJudge:
    budgets: list = []

    def __init__(self, *args, **kwargs):
        pass

    def evaluate_batch(self, results, max_tokens):
        FakeJudge.budgets.append(max_tokens)
        return []

    def save_results(self, *args):
        pass

    def cleanup(self):
        pass


class TestRunJudge:
    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        module = load_script(REPO / "scripts" / "run_judge.py")
        monkeypatch.setattr(module, "JudgeModel", FakeJudge)
        solutions = _solutions(tmp_path / "solutions")

        def go(*argv):
            FakeJudge.budgets = []
            monkeypatch.setattr(sys, "argv", ["run_judge.py", *argv, "--solutions-dir", str(solutions),
                                              "--output-dir", str(tmp_path / "out")])
            module.main()
            return FakeJudge.budgets
        return go

    def test_large_judge_that_always_reasons_gets_8192_only(self, run):
        assert run("--judge", "gpt-oss-120b") == [8192]

    def test_small_judge_gets_both_default_budgets(self, run):
        assert run("--judge", "qwen3-4b") == [10, 8192]

    def test_small_judge_that_always_reasons_gets_8192_only(self, run):
        assert run("--judge", "phi4r-14b") == [8192]

    def test_unknown_judge_stops(self, run):
        with pytest.raises(ValueError, match="Unknown judge model"):
            run("--judge", "no-such-judge")


# --- run_debate.py ---


class TestRunDebate:
    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        module = load_script(REPO / "scripts" / "run_debate.py")
        calls = []

        def fake_debate(**kwargs):
            calls.append((kwargs["student_results"][0]["student"], kwargs["dataset_name"], kwargs["max_tokens"]))
            return {}

        monkeypatch.setattr(module, "run_debate", fake_debate)
        monkeypatch.setattr(module, "save_debate_results", lambda **kwargs: None)
        root = tmp_path / "solutions"
        for student in ("qwen2.5-32b", "llama3.1-8b"):
            for ds in ("gsm8k", "gsm_plus"):
                f = root / student / f"{ds}.json"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps([{"problem_id": 0, "student": student}]))

        def go(*argv):
            calls.clear()
            monkeypatch.setattr(sys, "argv", ["run_debate.py", "--configs", "2", *argv,
                                              "--solutions-dir", str(root), "--output-dir", str(tmp_path / "out")])
            module.main()
            return list(calls)
        return go

    def test_default_budget_is_8192(self, run):
        assert {c[2] for c in run()} == {8192}

    def test_max_tokens_flag(self, run):
        assert {c[2] for c in run("--max-tokens", "4096")} == {4096}

    def test_student_and_dataset_filters(self, run):
        assert run("--students", "qwen2.5-32b", "--datasets", "gsm_plus") == [("qwen2.5-32b", "gsm_plus", 8192)]


# --- every runner honours $SLMJURY_RESULTS ---


class Captured(Exception):
    pass


def _parser_of(path: Path, monkeypatch) -> argparse.ArgumentParser:
    def grab(self, *args, **kwargs):
        raise Captured(self)

    try:
        module = load_script(path)
    except ImportError as e:
        pytest.skip(f"{path.name} needs {e.name}")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    monkeypatch.setattr(sys, "argv", [path.name])
    with pytest.raises(Captured) as info:
        module.main()
    return info.value.args[0]


PATH_DESTS = {"solutions_dir", "output_dir", "eval_dir", "reference_scores", "judgements_dir",
              "summaries_dir", "results_dir"}


@pytest.mark.parametrize("path", RUNNERS, ids=lambda p: str(p.relative_to(REPO)))
def test_runner_defaults_follow_slmjury_results(path, tmp_path, monkeypatch):
    root = tmp_path / "elsewhere"
    monkeypatch.setenv("SLMJURY_RESULTS", str(root))
    parser = _parser_of(path, monkeypatch)
    defaults = {a.dest: a.default for a in parser._actions
                if a.dest in PATH_DESTS and isinstance(a.default, str) and a.default != argparse.SUPPRESS}
    assert defaults, f"{path.name} has no results path option"
    for dest, value in defaults.items():
        assert Path(value).is_relative_to(root), f"{path.name} --{dest} defaults to {value}"


# --- datasets load before the model ---


@pytest.mark.parametrize("script", ["scripts/run_student.py", "experiments/family_bias/run_students.py"])
def test_student_runner_stops_before_loading_a_model(script, monkeypatch):
    from slmjury.data import DatasetLoadError

    module = load_script(REPO / script)

    def bad_load(name):
        raise DatasetLoadError(f"{name}: no items loaded")

    monkeypatch.setattr(module, "load_dataset", bad_load)
    monkeypatch.setattr(module, "StudentSolver", lambda *a, **k: pytest.fail("model loaded before the data"))
    monkeypatch.setattr(sys, "argv", [script, "--student", "qwen2.5-32b", "--datasets", "math"])
    with pytest.raises(DatasetLoadError):
        module.main()
