"""Unit tests for slmjury.configs — YAML configuration loading."""

import pytest

from slmjury.configs import load_models_config


class TestLoadModelsConfig:
    def test_loads_successfully(self):
        config = load_models_config()
        assert isinstance(config, dict)

    def test_has_judge_models(self):
        config = load_models_config()
        assert "judge_models" in config
        assert len(config["judge_models"]) == 16

    def test_has_student_models(self):
        config = load_models_config()
        assert "student_models" in config
        assert len(config["student_models"]) == 8  # 2 main students + 6 for family bias

    def test_has_large_judge_models(self):
        config = load_models_config()
        large = config["large_judge_models"]
        assert len(large) == 7  # 6 large judges + the Qwen3-32B ladder judge
        assert [k for k, v in large.items() if v.get("ladder")] == ["qwen3-32b"]

    def test_get_judge_config_searches_both_sections(self):
        from slmjury.configs import get_judge_config
        assert get_judge_config("phi4-14b")["model"] == "microsoft/phi-4"
        assert get_judge_config("qwen3.5-122b")["model"] == "Qwen/Qwen3.5-122B-A10B-FP8"
        with pytest.raises(ValueError):
            get_judge_config("no-such-judge")

    def test_has_datasets(self):
        config = load_models_config()
        assert "datasets" in config
        expected = [
            "gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
            "hellaswag", "winogrande", "truthfulqa",
            "summeval", "mtbench",
        ]
        assert config["datasets"] == expected

    def test_has_token_settings(self):
        config = load_models_config()
        assert "token_settings" in config
        assert "quick" in config["token_settings"]
        assert "reasoned" in config["token_settings"]

    def test_has_ensemble_judges(self):
        config = load_models_config()
        assert "ensemble_judges" in config
        assert len(config["ensemble_judges"]) == 5

    def test_judge_model_has_required_fields(self):
        config = load_models_config()
        for key, model in config["judge_models"].items():
            assert "model" in model, f"{key} missing 'model' field"

    def test_student_model_has_required_fields(self):
        config = load_models_config()
        for key, model in config["student_models"].items():
            assert "model" in model, f"{key} missing 'model' field"

    def test_specific_judge_model(self):
        config = load_models_config()
        qwen = config["judge_models"]["qwen2.5-7b"]
        assert qwen["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert qwen["tensor_parallel_size"] == 2

    def test_thinking_models_flagged(self):
        config = load_models_config()
        qwen3 = config["judge_models"]["qwen3-4b"]
        assert qwen3.get("enable_thinking") is True

    def test_always_thinks_models(self):
        config = load_models_config()
        phi4r = config["judge_models"]["phi4r-14b"]
        assert phi4r.get("always_thinks") is True

    def test_has_reference_models(self):
        config = load_models_config()
        assert "reference_models" in config
        assert len(config["reference_models"]) == 2

    def test_reference_models_are_together_based(self):
        config = load_models_config()
        for key, ref in config["reference_models"].items():
            assert "model" in ref, f"{key} missing 'model' field"
            assert ref.get("provider") == "together", (
                f"{key} should use 'together' provider"
            )
            assert "api_key_env" in ref, f"{key} missing 'api_key_env'"

    def test_reference_model_ids(self):
        config = load_models_config()
        refs = config["reference_models"]
        assert refs["gpt-oss-120b"]["model"] == "openai/gpt-oss-120b"
        assert "Qwen3.5-397B" in refs["qwen3.5-397b"]["model"]


