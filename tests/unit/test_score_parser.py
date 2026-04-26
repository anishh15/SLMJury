"""Tests for the score parser module."""

import unittest

from slmjury.parsers.score import parse_score, parse_multi_score


class TestParseScore(unittest.TestCase):
    """Tests for single-score extraction."""

    def test_boxed_format(self):
        self.assertEqual(parse_score(r"The score is \boxed{4}"), 4)

    def test_boxed_with_denominator(self):
        self.assertEqual(parse_score(r"\boxed{3/5}"), 3)

    def test_double_backslash_boxed(self):
        self.assertEqual(parse_score(r"\\boxed{5}"), 5)

    def test_markdown_bold(self):
        self.assertEqual(parse_score("I rate this **4**"), 4)

    def test_markdown_bold_with_denominator(self):
        self.assertEqual(parse_score("Rating: **3/5**"), 3)

    def test_bracket_format(self):
        self.assertEqual(parse_score("My score: [[4]]"), 4)

    def test_labeled_score(self):
        self.assertEqual(parse_score("Score: 3"), 3)

    def test_labeled_rating(self):
        self.assertEqual(parse_score("Rating: 5/5"), 5)

    def test_out_of_pattern(self):
        self.assertEqual(parse_score("I give this 4 out of 5"), 4)

    def test_overall_score(self):
        self.assertEqual(parse_score("Overall score: 2"), 2)

    def test_standalone_integer_fallback(self):
        self.assertEqual(parse_score("This is a 3"), 3)

    def test_last_valid_integer(self):
        self.assertEqual(parse_score("Scores considered: 2, 3, then 4"), 4)

    def test_out_of_range_ignored(self):
        # 7 is out of range (max=5), should fall through to find valid int
        self.assertEqual(parse_score("Score: 7. Actually 3."), 3)

    def test_empty_string(self):
        self.assertIsNone(parse_score(""))

    def test_none_input(self):
        self.assertIsNone(parse_score(None))

    def test_no_numbers(self):
        self.assertIsNone(parse_score("This response has no numbers"))

    def test_custom_max_score(self):
        self.assertEqual(parse_score(r"\boxed{8}", max_score=10), 8)

    def test_boxed_last_match_wins(self):
        self.assertEqual(
            parse_score(r"First \boxed{2}, then \boxed{4}"), 4,
        )

    def test_zero_score_invalid(self):
        """Scores below 1 should not be extracted."""
        self.assertIsNone(parse_score(r"\boxed{0}"))


class TestParseMultiScore(unittest.TestCase):
    """Tests for multi-dimension score extraction."""

    def test_boxed_structured_format(self):
        response = r"\boxed{coherence=4, consistency=3, fluency=5, relevance=4}"
        scores = parse_multi_score(response)
        self.assertEqual(scores["coherence"], 4)
        self.assertEqual(scores["consistency"], 3)
        self.assertEqual(scores["fluency"], 5)
        self.assertEqual(scores["relevance"], 4)

    def test_boxed_with_equals(self):
        response = r"\\boxed{coherence=5, consistency=5, fluency=4, relevance=3}"
        scores = parse_multi_score(response)
        self.assertEqual(scores["coherence"], 5)
        self.assertEqual(scores["fluency"], 4)

    def test_scattered_labels(self):
        response = (
            "Coherence: 4\n"
            "Consistency: 3\n"
            "Fluency: 5\n"
            "Relevance: 4"
        )
        scores = parse_multi_score(response)
        self.assertEqual(scores["coherence"], 4)
        self.assertEqual(scores["consistency"], 3)
        self.assertEqual(scores["fluency"], 5)
        self.assertEqual(scores["relevance"], 4)

    def test_mixed_format_labels(self):
        response = (
            "**Coherence**: 3\n"
            "Consistency = 4\n"
            "fluency - 5\n"
            "Relevance: 2"
        )
        scores = parse_multi_score(response)
        self.assertEqual(scores["coherence"], 3)
        self.assertEqual(scores["consistency"], 4)
        self.assertEqual(scores["relevance"], 2)

    def test_partial_extraction(self):
        response = "Coherence: 4, Relevance: 3"
        scores = parse_multi_score(response)
        self.assertEqual(scores["coherence"], 4)
        self.assertIsNone(scores["consistency"])
        self.assertIsNone(scores["fluency"])
        self.assertEqual(scores["relevance"], 3)

    def test_empty_response(self):
        scores = parse_multi_score("")
        self.assertTrue(all(v is None for v in scores.values()))

    def test_out_of_range_ignored(self):
        response = r"\boxed{coherence=7, consistency=3, fluency=5, relevance=4}"
        scores = parse_multi_score(response)
        self.assertIsNone(scores["coherence"])  # 7 > max_score=5
        self.assertEqual(scores["consistency"], 3)

    def test_custom_dimensions(self):
        scores = parse_multi_score(
            "accuracy: 4, depth: 3",
            dimensions=("accuracy", "depth"),
        )
        self.assertEqual(scores["accuracy"], 4)
        self.assertEqual(scores["depth"], 3)


if __name__ == "__main__":
    unittest.main()
