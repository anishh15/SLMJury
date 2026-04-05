"""Persona effect evaluation — measures how system prompts affect judge quality.

Runs judge models with different persona system prompts to study the effect
on judgement accuracy and instruction following rate.

The persona prompts below are the EXACT strings used during experimentation.
Do NOT modify them — any change would alter experimental results.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from slmjury.core.judge import JudgeModel

logger = logging.getLogger(__name__)

# 6 persona system prompts — exact copies from the original experiments.
# Whitespace preserved exactly as used during all experimental runs.
PERSONAS = {
    "Strict": """
You are a highly critical academic reviewer.
Evaluate the answer for factual correctness, logical consistency, completeness, and absence of unsupported claims.
Penalize vagueness, missing steps, and incorrect terminology.
Do not reward verbosity unless it improves rigor.
Prefer precise, structured, and well-justified answers.
If uncertain, lower the score.
    """,
    "Lenient": """
You are a supportive teacher grading student work.
Focus on whether the core idea is correct even if wording is imperfect.
Give partial credit for partially correct reasoning.
Do not heavily penalize minor factual or grammatical errors.
Reward effort, clarity of intent, and basic understanding.
    """,
    "Industry": """
You are an industry professional evaluating usefulness for real-world application.
Prioritize correctness, practicality, and actionability.
Penalize unnecessary theory, long explanations, and fluff.
Prefer short, clear, implementable answers.
An answer that can be used immediately should score higher.
    """,
    "Logic": """
You are a logic-focused evaluator.
Judge only based on internal reasoning quality.
Ignore writing style, tone, and length.
Check whether conclusions follow from premises.
Penalize contradictions, missing steps, and invalid inferences.
Factual correctness matters only if it affects logical validity.
    """,
    "Safety": """
You are a safety auditor checking for hallucinations and unsupported claims.
Penalize confident statements without evidence.
Reward uncertainty when appropriate.
If the answer invents facts, names, or numbers, assign a low score.
Factual grounding and cautious language should score higher than fluency.
    """,
    "Helpfulness": """
You are judging how helpful the answer is to the user.
Focus on whether the response solves the user's problem.
Reward clarity, directness, and relevance.
Penalize tangents, missing steps, and unclear instructions.
An answer that enables the user to act should receive a higher score.
    """,
}


def get_personas() -> dict[str, str]:
    """Return persona definitions.

    Returns:
        Dict mapping persona name → system prompt string.
    """
    return dict(PERSONAS)


def run_persona_evaluation(
    judge_model_key: str,
    student_results: list[dict],
    student_model: str,
    dataset: str,
    max_tokens: int,
    persona_name: str,
    persona_prompt: str,
    judge: Optional[JudgeModel] = None,
    output_dir: Path = Path("results/persona_effect"),
) -> list[dict]:
    """Run judge evaluation with a persona system prompt.

    Args:
        judge_model_key: Judge model key.
        student_results: List of student solution dicts.
        student_model: Student model key (for filename).
        dataset: Dataset name (for filename).
        max_tokens: Token setting.
        persona_name: Name of the persona.
        persona_prompt: System prompt text.
        judge: Optional pre-loaded JudgeModel (avoids reloading).
        output_dir: Base output directory.

    Returns:
        List of judgement dicts with persona-influenced verdicts.
    """
    if judge is None:
        judge = JudgeModel(judge_model_key)

    logger.info(
        "Running persona '%s' on %s × %s (tokens=%d, n=%d)",
        persona_name, student_model, dataset, max_tokens, len(student_results),
    )

    results = judge.evaluate_batch(
        student_results, max_tokens, system_prompt=persona_prompt,
    )

    # Save results
    out_dir = output_dir / judge_model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{persona_name}_{student_model}_{dataset}_t{max_tokens}.json"
    )
    out_file = out_dir / filename

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Saved persona results to %s", out_file.name)
    return results


def run_all_personas(
    judge_model_key: str,
    student_results: list[dict],
    student_model: str,
    dataset: str,
    max_tokens: int,
    judge: Optional[JudgeModel] = None,
    output_dir: Path = Path("results/persona_effect"),
) -> dict[str, list[dict]]:
    """Run all personas for a single judge/student/dataset/token combination.

    Args:
        judge_model_key: Judge model key.
        student_results: List of student solution dicts.
        student_model: Student model key.
        dataset: Dataset name.
        max_tokens: Token setting.
        judge: Optional pre-loaded JudgeModel.
        output_dir: Base output directory.

    Returns:
        Dict mapping persona name → list of judgement dicts.
    """
    personas = get_personas()
    if judge is None:
        judge = JudgeModel(judge_model_key)

    results = {}
    for persona_name, persona_prompt in personas.items():
        results[persona_name] = run_persona_evaluation(
            judge_model_key=judge_model_key,
            student_results=student_results,
            student_model=student_model,
            dataset=dataset,
            max_tokens=max_tokens,
            persona_name=persona_name,
            persona_prompt=persona_prompt,
            judge=judge,
            output_dir=output_dir,
        )

    return results
