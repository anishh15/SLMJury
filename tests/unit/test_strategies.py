"""Unit tests for slmjury.strategies — ensemble voting and persona definitions."""

import pytest

from slmjury.strategies.ensemble import (
    majority_vote,
    make_ensemble_name,
    get_ensemble_judges,
)
from slmjury.strategies.persona import PERSONAS, get_personas
from slmjury.strategies.debate import (
    _is_same_model_combo,
    make_combo_name,
    check_consensus,
    get_debate_prompt_template,
    MAX_DEBATE_ROUNDS,
    DEBATE_PROMPT_MATH,
    DEBATE_PROMPT_SCIENCE,
    DEBATE_PROMPT_GENERAL,
    MAD_COMBINATIONS,
)


# --- majority_vote ---


class TestMajorityVote:
    def test_all_correct(self):
        assert majority_vote(["Correct", "Correct", "Correct"]) == "Correct"

    def test_all_incorrect(self):
        assert majority_vote(["Incorrect", "Incorrect", "Incorrect"]) == "Incorrect"

    def test_two_correct_one_incorrect(self):
        assert majority_vote(["Correct", "Incorrect", "Correct"]) == "Correct"

    def test_two_incorrect_one_correct(self):
        assert majority_vote(["Incorrect", "Correct", "Incorrect"]) == "Incorrect"

    def test_two_correct_one_undefined(self):
        assert majority_vote(["Correct", "Undefined", "Correct"]) == "Correct"

    def test_all_undefined(self):
        assert majority_vote(["Undefined", "Undefined", "Undefined"]) == "Undefined"

    def test_mixed_no_majority(self):
        assert majority_vote(["Correct", "Incorrect", "Undefined"]) == "Undefined"


# --- make_ensemble_name ---


class TestMakeEnsembleName:
    def test_with_token_settings(self):
        models = ["qwen3-4b", "phi4mi-3.8b"]
        tokens = {"qwen3-4b": 10, "phi4mi-3.8b": 10}
        result = make_ensemble_name(models, tokens)
        assert result == "qwen3-4b-t10+phi4mi-3.8b-t10"

    def test_without_token_settings(self):
        models = ["model-a", "model-b"]
        result = make_ensemble_name(models)
        assert result == "model-a+model-b"


# --- get_ensemble_judges ---


class TestGetEnsembleJudges:
    def test_returns_dict(self):
        judges = get_ensemble_judges()
        assert isinstance(judges, dict)
        assert len(judges) == 5

    def test_has_expected_models(self):
        judges = get_ensemble_judges()
        assert "qwen3-4b" in judges
        assert "phi4mi-3.8b" in judges


# --- Persona definitions ---


class TestPersonas:
    def test_has_six_personas(self):
        assert len(PERSONAS) == 6

    def test_expected_persona_names(self):
        expected = {"Strict", "Lenient", "Industry", "Logic", "Safety", "Helpfulness"}
        assert set(PERSONAS.keys()) == expected

    def test_personas_are_non_empty_strings(self):
        for name, prompt in PERSONAS.items():
            assert isinstance(prompt, str), f"{name} is not a string"
            assert len(prompt.strip()) > 50, f"{name} prompt too short"

    def test_get_personas_returns_copy(self):
        p = get_personas()
        assert p == PERSONAS
        assert p is not PERSONAS  # should be a copy


# --- Debate utilities ---


class TestDebateUtils:
    def test_is_same_model_combo_true(self):
        assert _is_same_model_combo(["m1", "m1", "m1"]) is True

    def test_is_same_model_combo_false(self):
        assert _is_same_model_combo(["m1", "m2", "m3"]) is False

    def test_make_combo_name_different_models(self):
        result = make_combo_name(["m1", "m2", "m3"], [0, 0, 0])
        assert result == "m1+m2+m3"

    def test_make_combo_name_same_model(self):
        result = make_combo_name(["m1", "m1", "m1"], [0, 0.4, 0.9])
        assert result == "m1-t0+m1-t0.4+m1-t0.9"

    def test_max_debate_rounds(self):
        assert MAX_DEBATE_ROUNDS == 5

    def test_mad_combinations_count(self):
        assert len(MAD_COMBINATIONS) == 10

    def test_debate_prompts_have_placeholders(self):
        for name, prompt in [
            ("math", DEBATE_PROMPT_MATH),
            ("science", DEBATE_PROMPT_SCIENCE),
            ("general", DEBATE_PROMPT_GENERAL),
        ]:
            for ph in ["{agent_id}", "{question}", "{ground_truth}", "{student_answer}",
                       "{own_previous}", "{context}", "{round_num}"]:
                assert ph in prompt, f"{name} missing placeholder {ph}"

    def test_get_debate_prompt_math(self):
        assert get_debate_prompt_template("gsm8k") is DEBATE_PROMPT_MATH
        assert get_debate_prompt_template("math") is DEBATE_PROMPT_MATH

    def test_get_debate_prompt_science(self):
        assert get_debate_prompt_template("arc_easy") is DEBATE_PROMPT_SCIENCE
        assert get_debate_prompt_template("arc_challenge") is DEBATE_PROMPT_SCIENCE

    def test_get_debate_prompt_general(self):
        assert get_debate_prompt_template("hellaswag") is DEBATE_PROMPT_GENERAL
        assert get_debate_prompt_template("winogrande") is DEBATE_PROMPT_GENERAL
        assert get_debate_prompt_template("truthfulqa") is DEBATE_PROMPT_GENERAL

    def test_check_consensus_unanimous(self):
        state = {
            "p1": {
                "final_verdict": None,
                "resolved_at_round": None,
                "rounds": [
                    {"agent_0": {"verdict": "Correct"}, "agent_1": {"verdict": "Correct"}, "agent_2": {"verdict": "Correct"}}
                ],
            }
        }
        resolved, unresolved = check_consensus(state, 0)
        assert "p1" in resolved
        assert len(unresolved) == 0
        assert state["p1"]["final_verdict"] == "Correct"

    def test_check_consensus_disagreement(self):
        state = {
            "p1": {
                "final_verdict": None,
                "resolved_at_round": None,
                "rounds": [
                    {"agent_0": {"verdict": "Correct"}, "agent_1": {"verdict": "Incorrect"}, "agent_2": {"verdict": "Correct"}}
                ],
            }
        }
        resolved, unresolved = check_consensus(state, 0)
        assert len(resolved) == 0
        assert "p1" in unresolved
