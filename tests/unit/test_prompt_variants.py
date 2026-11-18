"""Unit tests for slmjury/configs/prompt_variants.yaml."""

import fnmatch

import pytest

from slmjury.configs import CONFIG_DIR, PROMPT_VARIANTS_YAML, load_prompt_variants
from slmjury.core import judge, scoring_judge

VARIANTS = ["original", "minimal", "gpt_style", "rubric_based", "cot_style",
            "adversarial", "few_shot"]

VERDICT_FIELDS = ["question", "ground_truth", "student_answer"]
SUMMEVAL_FIELDS = ["source_text", "summary"]
MTBENCH_FIELDS = ["turn1_question", "turn1_response", "turn2_question", "turn2_response"]


@pytest.fixture(scope="module")
def variants():
    return load_prompt_variants()


def test_shipped_as_package_data():
    # pyproject.toml ships configs/*.yaml with the package.
    rel = PROMPT_VARIANTS_YAML.relative_to(CONFIG_DIR.parent).as_posix()
    assert fnmatch.fnmatch(rel, "configs/*.yaml")
    assert PROMPT_VARIANTS_YAML.exists()


def test_seven_variants_in_both_sections(variants):
    assert list(variants["verdict_prompts"]) == VARIANTS
    assert list(variants["scoring_prompts"]) == VARIANTS


def test_each_variant_has_every_style(variants):
    for v in VARIANTS:
        assert set(variants["verdict_prompts"][v]) == {"quick", "reasoned"}
        assert set(variants["scoring_prompts"][v]) == {"summeval", "mtbench"}


@pytest.mark.parametrize("variant", VARIANTS)
def test_templates_take_the_package_fields(variants, variant):
    v = variants["verdict_prompts"][variant]
    s = variants["scoring_prompts"][variant]
    # str.format raises if a template uses a field the package does not pass.
    for style in ("quick", "reasoned"):
        out = v[style].format(**{k: f"<{k}>" for k in VERDICT_FIELDS})
        assert all(f"<{k}>" in out for k in VERDICT_FIELDS)
    out = s["summeval"].format(**{k: f"<{k}>" for k in SUMMEVAL_FIELDS})
    assert all(f"<{k}>" in out for k in SUMMEVAL_FIELDS)
    out = s["mtbench"].format(**{k: f"<{k}>" for k in MTBENCH_FIELDS})
    assert all(f"<{k}>" in out for k in MTBENCH_FIELDS)


def test_original_matches_package_defaults(variants):
    # The runner sends no template for "original", so the package constants
    # are used. The YAML copies are the same text; three of them end with a
    # newline the constant does not have.
    v = variants["verdict_prompts"]["original"]
    s = variants["scoring_prompts"]["original"]
    pairs = [
        (v["quick"], judge.PROMPT_QUICK),
        (v["reasoned"], judge.PROMPT_REASONED),
        (s["summeval"], scoring_judge.PROMPT_SUMMEVAL),
        (s["mtbench"], scoring_judge.PROMPT_MTBENCH),
    ]
    for yaml_text, constant in pairs:
        assert yaml_text.rstrip("\n") == constant.rstrip("\n")


def test_other_variants_differ_from_defaults(variants):
    for v in VARIANTS[1:]:
        assert variants["verdict_prompts"][v]["quick"].rstrip("\n") != \
            judge.PROMPT_QUICK.rstrip("\n")
        assert variants["scoring_prompts"][v]["summeval"].rstrip("\n") != \
            scoring_judge.PROMPT_SUMMEVAL.rstrip("\n")
