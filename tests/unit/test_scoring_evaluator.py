"""Tests for the scoring evaluator module."""

import unittest

from slmjury.core.scoring_evaluator import (
    _pearson_r,
    _spearman_rho,
    _cohens_kappa,
    _binary_accuracy,
    _mse,
    evaluate_scores,
    evaluate_summeval,
)


class TestPearsonR(unittest.TestCase):
    """Tests for Pearson correlation."""

    def test_perfect_positive(self):
        r = _pearson_r([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(r, 1.0)

    def test_perfect_negative(self):
        r = _pearson_r([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        self.assertAlmostEqual(r, -1.0)

    def test_no_correlation(self):
        r = _pearson_r([1, 2, 3, 4, 5], [3, 3, 3, 3, 3])
        self.assertIsNone(r)  # zero variance in y

    def test_insufficient_data(self):
        r = _pearson_r([1], [2])
        self.assertIsNone(r)


class TestSpearmanRho(unittest.TestCase):
    """Tests for Spearman rank correlation."""

    def test_perfect_rank(self):
        rho = _spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        self.assertAlmostEqual(rho, 1.0)

    def test_inverse_rank(self):
        rho = _spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        self.assertAlmostEqual(rho, -1.0)

    def test_tied_ranks(self):
        rho = _spearman_rho([1, 2, 2, 4], [1, 3, 3, 5])
        self.assertIsNotNone(rho)
        self.assertGreater(rho, 0)


class TestCohensKappa(unittest.TestCase):
    """Tests for Cohen's kappa."""

    def test_perfect_agreement(self):
        k = _cohens_kappa([5, 5, 1, 1], [5, 5, 1, 1], threshold=4)
        self.assertAlmostEqual(k, 1.0)

    def test_no_agreement(self):
        k = _cohens_kappa([5, 5, 1, 1], [1, 1, 5, 5], threshold=4)
        self.assertAlmostEqual(k, -1.0)

    def test_empty(self):
        k = _cohens_kappa([], [], threshold=4)
        self.assertIsNone(k)


class TestBinaryAccuracy(unittest.TestCase):
    """Tests for binary classification accuracy."""

    def test_perfect(self):
        acc = _binary_accuracy([5, 5, 1, 1], [5, 5, 1, 1], threshold=4)
        self.assertAlmostEqual(acc, 1.0)

    def test_half(self):
        acc = _binary_accuracy([5, 1, 5, 1], [1, 5, 1, 5], threshold=4)
        self.assertAlmostEqual(acc, 0.0)


class TestMSE(unittest.TestCase):
    """Tests for mean squared error."""

    def test_zero_error(self):
        m = _mse([3, 4, 5], [3, 4, 5])
        self.assertAlmostEqual(m, 0.0)

    def test_nonzero(self):
        m = _mse([1, 2, 3], [2, 3, 4])
        self.assertAlmostEqual(m, 1.0)


class TestEvaluateScores(unittest.TestCase):
    """Tests for the main evaluate_scores function."""

    def test_with_none_judge_scores(self):
        result = evaluate_scores(
            [3, None, 5, 4, None], [3, 4, 5, 4, 2],
        )
        self.assertEqual(result["n_total"], 5)
        self.assertEqual(result["n_valid"], 3)
        self.assertAlmostEqual(result["parse_failure_rate"], 0.4)

    def test_all_none(self):
        result = evaluate_scores([None, None], [3, 4])
        self.assertIsNone(result["pearson"])
        self.assertEqual(result["n_valid"], 0)

    def test_valid_computation(self):
        result = evaluate_scores(
            [1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
        )
        self.assertAlmostEqual(result["pearson"], 1.0)
        self.assertAlmostEqual(result["spearman"], 1.0)
        self.assertEqual(result["n_valid"], 5)


class TestEvaluateSummeval(unittest.TestCase):
    """Tests for SummEval evaluation."""

    def test_basic(self):
        results = [
            {
                "problem_id": i,
                "judge_coherence": 4, "judge_consistency": 3,
                "judge_fluency": 5, "judge_relevance": 4,
                "human_coherence": 4.0, "human_consistency": 3.0,
                "human_fluency": 5.0, "human_relevance": 4.0,
            }
            for i in range(10)
        ]
        eval_result = evaluate_summeval(results)
        self.assertIn("per_dimension", eval_result)
        self.assertIn("overall", eval_result)
        self.assertIn("coherence", eval_result["per_dimension"])


if __name__ == "__main__":
    unittest.main()
