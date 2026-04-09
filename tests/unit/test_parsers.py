"""Unit tests for slmjury.parsers — answer extraction, normalization, and judgement parsing."""

import pytest

from slmjury.parsers.answer import extract_answer, get_dataset_type
from slmjury.parsers.normalizer import (
    answers_are_equivalent,
    normalize_numeric,
    numbers_are_equivalent,
)
from slmjury.parsers.judgement import parse_judgement, parse_judgement_detailed


# --- get_dataset_type ---


class TestGetDatasetType:
    def test_numeric_datasets(self):
        assert get_dataset_type("gsm8k") == "numeric"
        assert get_dataset_type("gsm_plus") == "numeric"

    def test_latex_dataset(self):
        assert get_dataset_type("math") == "latex"

    def test_multiple_choice_datasets(self):
        assert get_dataset_type("arc_easy") == "multiple_choice"
        assert get_dataset_type("arc_challenge") == "multiple_choice"

    def test_general_multiple_choice_datasets(self):
        assert get_dataset_type("hellaswag") == "multiple_choice"
        assert get_dataset_type("winogrande") == "multiple_choice"
        assert get_dataset_type("truthfulqa") == "multiple_choice"
        assert get_dataset_type("truthful_qa") == "multiple_choice"

    def test_unknown_defaults_to_numeric(self):
        assert get_dataset_type("unknown_dataset") == "numeric"


# --- extract_answer ---


class TestExtractAnswer:
    def test_numeric_hash_extraction(self):
        assert extract_answer("The answer is #### 42") == "42"

    def test_numeric_hash_with_comma(self):
        result = extract_answer("#### 1,200")
        assert result in ("1,200", "1200")  # parser may normalize commas

    def test_latex_boxed_extraction(self):
        result = extract_answer(r"\boxed{x^2 + 1}", "latex")
        assert result == "x^2 + 1"

    def test_multiple_choice_letter(self):
        assert extract_answer("The answer is B", "multiple_choice") == "B"

    def test_multiple_choice_uppercase(self):
        result = extract_answer("the answer is d", "multiple_choice")
        assert result.upper() == "D"

    def test_empty_input(self):
        result = extract_answer("", "numeric")
        assert result in ("", "UNABLE_TO_EXTRACT")


# --- normalize_numeric ---


class TestNormalizeNumeric:
    def test_removes_commas(self):
        assert normalize_numeric("1,200") == "1200"

    def test_removes_dollar_sign(self):
        assert normalize_numeric("$42") == "42"

    def test_removes_percent(self):
        assert normalize_numeric("85%") == "85"

    def test_preserves_negative(self):
        result = normalize_numeric("-3.14")
        assert float(result) == pytest.approx(-3.14)

    def test_handles_spaces(self):
        assert normalize_numeric(" 42 ") == "42"


# --- answers_are_equivalent ---


class TestAnswersAreEquivalent:
    def test_exact_numeric_match(self):
        assert answers_are_equivalent("42", "42") is True

    def test_numeric_with_comma(self):
        assert answers_are_equivalent("1,000", "1000") is True

    def test_numeric_mismatch(self):
        assert answers_are_equivalent("42", "43") is False

    def test_multiple_choice_match(self):
        assert answers_are_equivalent("B", "B", "multiple_choice") is True

    def test_multiple_choice_by_index(self):
        assert answers_are_equivalent("2", "B", "multiple_choice") is True

    def test_multiple_choice_mismatch(self):
        assert answers_are_equivalent("A", "B", "multiple_choice") is False

    def test_empty_strings(self):
        assert answers_are_equivalent("", "") is False

    def test_one_empty(self):
        assert answers_are_equivalent("42", "") is False


# --- numbers_are_equivalent ---


class TestNumbersAreEquivalent:
    def test_integer_match(self):
        assert numbers_are_equivalent("42", "42") is True

    def test_float_match(self):
        assert numbers_are_equivalent("3.14", "3.14") is True

    def test_close_floats(self):
        assert numbers_are_equivalent("3.14159", "3.14159") is True


# --- parse_judgement ---


class TestParseJudgement:
    def test_simple_correct(self):
        assert parse_judgement("Correct") == "Correct"

    def test_simple_incorrect(self):
        assert parse_judgement("Incorrect") == "Incorrect"

    def test_boxed_correct(self):
        assert parse_judgement(r"\boxed{CORRECT}") == "Correct"

    def test_boxed_incorrect(self):
        assert parse_judgement(r"\boxed{INCORRECT}") == "Incorrect"

    def test_empty_response(self):
        assert parse_judgement("") == "Undefined"

    def test_garbage_response(self):
        assert parse_judgement("lorem ipsum dolor sit amet") == "Undefined"

    def test_case_insensitive(self):
        assert parse_judgement("correct") == "Correct"

    def test_sentence_with_verdict(self):
        result = parse_judgement("The answer is correct.")
        assert result == "Correct"


# --- parse_judgement_detailed ---


class TestParseJudgementDetailed:
    def test_boxed_high_confidence(self):
        result = parse_judgement_detailed(r"\boxed{CORRECT}")
        assert result.verdict == "Correct"
        assert result.confidence == "high"
        assert result.method == "boxed"

    def test_quick_verdict(self):
        result = parse_judgement_detailed("Incorrect")
        assert result.verdict == "Incorrect"
        assert result.confidence == "medium"

    def test_undefined_low_confidence(self):
        result = parse_judgement_detailed("I'm not sure about this")
        assert result.verdict == "Undefined"
        assert result.confidence in ("low", "none")

    def test_empty_none_confidence(self):
        result = parse_judgement_detailed("")
        assert result.verdict == "Undefined"
        assert result.confidence == "none"
