"""Unit tests for slmjury.parsers — answer extraction, normalization, and judgement parsing."""

import json
import sys

import pytest

from slmjury.parsers import normalizer
from slmjury.parsers.answer import extract_answer, get_dataset_type
from slmjury.parsers.normalizer import (
    _compare_latex_raw,
    answers_are_equivalent,
    compare_with_limit,
    label_sha,
    labels_digest,
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


# --- LaTeX comparison rules ---


class TestLatexRules:
    @pytest.mark.parametrize("answer, gold, expected", [
        (r"\frac{1}{2}", r"\frac{1}{4}", False),
        (r"x^2+2x+1", r"(x+1)^2", True),
        (r"\frac{\pi^2}{4}", r"\frac{\pi^2}{6}", False),
        ("x^2 - 1", "x^2", False),
        (r"-\frac{1}{4}", "-0.25", True),
        (r"x^{1/2}", r"\sqrt{x}", True),
        ("4, 6, 14, 15", "4,6,14,15", True),
        ("4, 6, 14, 15", "4,6,14,16", False),
        ("4, 6", "4", False),
        ("4", "4, -2 + 3i, -2 - 3i", False),
        ("(2, 5)", "(2,5)", True),
        ("(2, 5)", "(2,6)", False),
        ("[-2,7]", "[-2,8]", False),
        (r"10\sqrt{2}", r"10\sqrt3", False),
        ("35/64", r"\frac{35}{128}", False),
        (r"\$12", "12", True),
        ("x=5", "5", True),
        (r"12\text{ cm}", "12", True),
        (r"90^\circ", "90", True),
        ("90", r"90^\circ", True),
        (r"\$78", "78", True),
        (r"75\%", "75", True),
        (r"\left(\frac{17}{4},0\right)", r"\left( \frac{17}{4}, 0 \right)", True),
        (r"[-2, 7]", r"x \in [-2,7]", True),
        ("(2,5)", "(2,6)", False),
        ("4,6", "4,7", False),
        ("x=5", "y=6", False),
        ("0.75", r"75\%", False),
        (r"\begin{pmatrix} 2/5 \\ 1 \end{pmatrix}", r"\begin{pmatrix} \frac{2}{5} \\ 1 \end{pmatrix}", True),
        ("12", "13", False),
    ])
    def test_answer_pairs(self, answer, gold, expected):
        assert _compare_latex_raw(answer, gold) is expected

    @pytest.mark.parametrize("answer, gold", [("25%", "25"), ("25%", r"25\%"), ("100%", r"100\%")])
    def test_bare_percent_is_formatting(self, answer, gold):
        assert _compare_latex_raw(answer, gold) is True

    def test_matrix_row_break_ignores_spacing(self):
        spaced, tight = r"\begin{pmatrix} -6 \\ 6 \end{pmatrix}", r"\begin{pmatrix}-6\\6\end{pmatrix}"
        assert _compare_latex_raw(tight, spaced) is True
        assert normalizer._strip_format(spaced) == normalizer._strip_format(tight)

    def test_single_number_does_not_match_a_list(self):
        assert _compare_latex_raw("4", "4, 6") is False

    def test_lists_differing_in_one_entry(self):
        assert _compare_latex_raw("4, 6, 14, 15", "4,6,14,16") is False


class TestLatexFormatRules:
    """One class of answers per comparison rule, each with a case the rule must leave alone."""

    def test_numeric_fraction_stays_one_number(self):
        assert _compare_latex_raw(r"\frac{3}{2}x^2", r"\frac{3x^2}{2}") is True
        assert _compare_latex_raw(r"\frac{3\pi}{4}", r"\frac{3}{4}\pi") is True
        assert _compare_latex_raw(r"(\frac{3}{4}, 5)", r"\left(\frac34,5\right)") is True
        assert _compare_latex_raw(r"\frac{3}{2}x^2", r"\frac{2}{3}x^2") is False

    def test_mixed_number(self):
        assert _compare_latex_raw(r"6\frac{1}{5}", r"6\frac15") is True
        assert _compare_latex_raw(r"\frac{25}{13}", r"1\frac{12}{13}") is True
        assert _compare_latex_raw("6 1/5", r"\frac{31}{5}") is True
        assert _compare_latex_raw(r"6\frac{1}{5}", r"\frac{61}{5}") is False

    @pytest.mark.parametrize("answer, gold", [
        ("2√2", r"2\sqrt{2}"),
        ("-8x⁴", "-8x^4"),
        ("135°", r"135^\circ"),
        ("sqrt(2)", r"\sqrt{2}"),
        ("13?", "13"),
        (r"\frac{7}{5} + \frac{1}{5} i.", r"\frac{7}{5} + \frac{1}{5} i"),
        (r"440", r"440\text{ cm}^2"),
        (r"144", r"144 \mbox{ m}^3"),
        ("10,000", "10{,}000"),
        (r"\10", "10"),
        ("12th", "12"),
        (r"(2,+\infty)", r"(2,\infty)"),
    ])
    def test_spellings(self, answer, gold):
        assert _compare_latex_raw(answer, gold) is True

    def test_spellings_keep_values(self):
        assert _compare_latex_raw("2√3", r"2\sqrt{2}") is False
        assert _compare_latex_raw("11th", "12") is False

    def test_names(self):
        assert _compare_latex_raw("f(x) = 1-x", "1 - x") is True
        assert _compare_latex_raw(r"x_1 = 3", "3") is True
        assert _compare_latex_raw("-n(2n+1)", "-2n^2-n") is True
        assert _compare_latex_raw("f(x) = 1-x", "1 + x") is False
        assert _compare_latex_raw("f(x+1)", "fx+f") is False

    def test_structure(self):
        assert _compare_latex_raw("2, 4, -3", "2, -3, 4") is True
        assert _compare_latex_raw("(1, 4.5)", r"\left(1,\frac{9}{2}\right)") is True
        assert _compare_latex_raw(r"\pm\frac{3}{4}", r"\frac{3}{4}, -\frac{3}{4}") is True
        assert _compare_latex_raw(r"(1,2) \cup (3,4)", r"(3,4) \cup (1,2)") is True
        assert _compare_latex_raw(r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}", "(1,2)") is True
        assert _compare_latex_raw("(2,1)", "(1,2)") is False
        assert _compare_latex_raw("2, 4, -3", "2, 4, 3") is False
        assert _compare_latex_raw(r"\begin{pmatrix} 2 \\ 1 \end{pmatrix}", "(1,2)") is False

    def test_base_subscript_on_one_side(self):
        assert _compare_latex_raw("141", "141_{13}") is True
        assert _compare_latex_raw("144_{62}", "144") is True
        assert _compare_latex_raw("141_{13}", "141_{12}") is False
        assert _compare_latex_raw("142", "141_{13}") is False

    def test_words_and_choice_letters(self):
        assert _compare_latex_raw("Parabola", r"\text{parabola}") is True
        assert _compare_latex_raw("Saturday", r"\mbox{Saturday}") is True
        assert _compare_latex_raw("E", r"\text{(E)}") is True
        assert _compare_latex_raw("D", r"\text{(E)}") is False
        assert _compare_latex_raw("Sunday", r"\text{Saturday}") is False

    def test_closed_equality_uses_last_side(self):
        assert _compare_latex_raw("(-3)(5) = -15", "-15") is True
        assert _compare_latex_raw("(-3)(5) = -15", "15") is False
        assert _compare_latex_raw("y + 2 = 5", "7") is False

    def test_tiny_values_need_a_relative_match(self):
        assert _compare_latex_raw(r"\frac{1}{2005!}", r"\frac{1}{2004!}") is False
        assert normalizer._close(1e-12, 2e-12) is False
        assert normalizer._close(0.5, 0.5) is True


class TestLatexRuntime:
    @pytest.fixture
    def strict(self, monkeypatch):
        monkeypatch.setattr(normalizer, "_labels", {})
        normalizer.set_strict_labels(True)
        yield
        normalizer.set_strict_labels(False)

    def test_missing_label_raises_in_strict_mode(self, strict):
        with pytest.raises(normalizer.MissingLabel, match="build_math_labels"):
            answers_are_equivalent("x+1", "1+x", "latex")

    def test_label_wins_over_the_rules(self, monkeypatch):
        monkeypatch.setattr(normalizer, "_labels", {normalizer.label_key("3", "3"): False})
        assert answers_are_equivalent("3", "3", "latex") is False

    def test_live_use_computes_without_a_label(self, monkeypatch):
        monkeypatch.setattr(normalizer, "_labels", {})
        assert answers_are_equivalent("x+1", "1+x", "latex") is True

    def test_blocked_antlr_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "antlr4", None)
        monkeypatch.setattr(normalizer, "_labels", {})
        with pytest.raises(ImportError):
            answers_are_equivalent("x^2+2x+1", "(x+1)^2", "latex")

    def test_time_limit_scores_not_equivalent(self):
        assert compare_with_limit("x^2+2x+1", "(x+1)^2", seconds=1e-4) == (False, True)


@pytest.mark.skipif(not normalizer.LABEL_FILE.exists(), reason="label file is not in this install")
class TestLabelFile:
    def test_sha_and_versions(self):
        blob = json.loads(normalizer.LABEL_FILE.read_text())
        assert labels_digest(blob["labels"]) == blob["labels_sha256"] == label_sha()
        assert blob["versions"]["sympy"] == "1.13.1"
        assert blob["versions"]["antlr4"].startswith("4.11.")

    def test_keys_are_sorted(self):
        blob = json.loads(normalizer.LABEL_FILE.read_text())
        keys = list(blob["labels"])
        assert keys == sorted(keys)

    def test_matrix_row_break_pair_is_labelled_equivalent(self):
        blob = json.loads(normalizer.LABEL_FILE.read_text())
        key = normalizer.label_key(r"\begin{pmatrix}-6\\6\end{pmatrix}",
                                   r"\begin{pmatrix} -6 \\ 6 \end{pmatrix}")
        assert blob["labels"][key] is True


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
