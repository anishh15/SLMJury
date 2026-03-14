"""Unit tests for slmjury.core.evaluator — metrics computation and filename parsing."""

import pytest

from slmjury.core.evaluator import JudgeEvaluator, parse_judgement_filename


# --- parse_judgement_filename ---


class TestParseJudgementFilename:
    def test_standard_filename(self):
        judge, student, dataset, tokens = parse_judgement_filename(
            "qwen2.5-7b_llama3.1-8b_gsm8k_tokens-8192.json"
        )
        assert judge == "qwen2.5-7b"
        assert student == "llama3.1-8b"
        assert dataset == "gsm8k"
        assert tokens == 8192

    def test_underscore_dataset(self):
        judge, student, dataset, tokens = parse_judgement_filename(
            "qwen2.5-7b_llama3.1-8b_arc_challenge_tokens-10.json"
        )
        assert dataset == "arc_challenge"
        assert tokens == 10

    def test_invalid_filename(self):
        with pytest.raises(ValueError, match="Cannot parse tokens"):
            parse_judgement_filename("bad_filename.json")


# --- JudgeEvaluator ---


class TestJudgeEvaluator:
    @pytest.fixture
    def sample_judgements(self):
        return [
            {
                "problem_id": "p1",
                "question": "2+2?",
                "ground_truth_reasoning": "",
                "ground_truth_answer": "4",
                "student_reasoning": "2+2=4",
                "student_answer": "4",
                "judge_response": "Correct",
                "judgement": "Correct",
            },
            {
                "problem_id": "p2",
                "question": "3+3?",
                "ground_truth_reasoning": "",
                "ground_truth_answer": "6",
                "student_reasoning": "3+3=5",
                "student_answer": "5",
                "judge_response": "Incorrect",
                "judgement": "Incorrect",
            },
            {
                "problem_id": "p3",
                "question": "1+1?",
                "ground_truth_reasoning": "",
                "ground_truth_answer": "2",
                "student_reasoning": "1+1=2",
                "student_answer": "2",
                "judge_response": "hmm",
                "judgement": "Undefined",
            },
        ]

    def test_evaluate_returns_summary(self, sample_judgements):
        evaluator = JudgeEvaluator(
            judge_model="test-judge",
            student_model="test-student",
            dataset="gsm8k",
            max_tokens=10,
            judgements=sample_judgements,
        )
        summary = evaluator.evaluate()
        assert isinstance(summary, dict)
        assert "metrics" in summary
        assert "counts" in summary

    def test_accuracy_calculation(self, sample_judgements):
        evaluator = JudgeEvaluator(
            judge_model="test-judge",
            student_model="test-student",
            dataset="gsm8k",
            max_tokens=10,
            judgements=sample_judgements,
        )
        evaluator.evaluate()
        # p1: student=4, gt=4, correct → judge says Correct → match
        # p2: student=5, gt=6, incorrect → judge says Incorrect → match
        # p3: Undefined → not counted as valid
        assert evaluator.accuracy == pytest.approx(2 / 3, abs=0.01)

    def test_ifr_calculation(self, sample_judgements):
        evaluator = JudgeEvaluator(
            judge_model="test-judge",
            student_model="test-student",
            dataset="gsm8k",
            max_tokens=10,
            judgements=sample_judgements,
        )
        evaluator.evaluate()
        # 2 out of 3 are valid (Correct or Incorrect)
        assert evaluator.ifr == pytest.approx(2 / 3, abs=0.01)

    def test_empty_judgements(self):
        evaluator = JudgeEvaluator(
            judge_model="test-judge",
            student_model="test-student",
            dataset="gsm8k",
            max_tokens=10,
            judgements=[],
        )
        summary = evaluator.evaluate()
        assert evaluator.accuracy == 0.0
        assert evaluator.ifr == 0.0
