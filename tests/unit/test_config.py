"""Unit tests for slmjury.configs — YAML configuration loading."""

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
        assert len(config["student_models"]) >= 2  # original 2 + E2-B additions

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

    def test_has_oracle_models(self):
        config = load_models_config()
        assert "oracle_models" in config
        assert len(config["oracle_models"]) == 2

    def test_oracle_models_are_together_based(self):
        config = load_models_config()
        for key, oracle in config["oracle_models"].items():
            assert "model" in oracle, f"{key} missing 'model' field"
            assert oracle.get("provider") == "together", (
                f"{key} should use 'together' provider"
            )
            assert "api_key_env" in oracle, f"{key} missing 'api_key_env'"

    def test_oracle_model_ids(self):
        config = load_models_config()
        oracles = config["oracle_models"]
        assert oracles["gpt-oss-120b"]["model"] == "openai/gpt-oss-120b"
        assert "Qwen3.5-397B" in oracles["qwen3.5-397b"]["model"]


