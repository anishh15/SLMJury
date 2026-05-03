"""CLI entry-point: score MT-Bench student responses using oracle models via API.

Uses OpenAI-compatible APIs:
  - GPT-OSS-120B via Together AI (https://api.together.ai/v1)
  - Qwen3-235B via Cerebras (https://api.cerebras.ai/v1)

Scores both turns separately — Turn 2 includes full conversation context.

Usage:
    python scripts/run_mtbench_oracle.py --student-responses results/mtbench_responses/llama3.1-8b.json
    python scripts/run_mtbench_oracle.py --student-responses results/mtbench_responses/ --oracle gpt-oss-120b
    python scripts/run_mtbench_oracle.py --student-responses results/mtbench_responses/ --oracle all
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from slmjury.configs import load_models_config
from slmjury.parsers.score import parse_score

# Load API keys from .env file (project root)
load_dotenv()

logger = logging.getLogger(__name__)

# Oracle scoring prompt for Turn 1 (same scale as SLM judges)
ORACLE_PROMPT_TURN1 = (
    "You are an expert evaluator. Rate the quality of the following response "
    "on a scale of 1-5.\n"
    "1 = Very poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent.\n\n"
    "[Question]:\n{question}\n\n"
    "[Response]:\n{response}\n\n"
    "Consider helpfulness, accuracy, depth, creativity, and level of detail.\n"
    "Provide your rating as: \\boxed{{SCORE}}"
)

# Oracle scoring prompt for Turn 2 (includes full conversation context)
ORACLE_PROMPT_TURN2 = (
    "You are an expert evaluator. Rate the quality of the assistant's "
    "Turn 2 response on a scale of 1-5.\n"
    "1 = Very poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent.\n\n"
    "[Turn 1 Question]:\n{turn1_question}\n\n"
    "[Turn 1 Response]:\n{turn1_response}\n\n"
    "[Turn 2 Question]:\n{turn2_question}\n\n"
    "[Turn 2 Response]:\n{turn2_response}\n\n"
    "Consider how well the Turn 2 response addresses the follow-up, "
    "maintains context from Turn 1, and demonstrates helpfulness, "
    "accuracy, and depth.\n"
    "Provide your rating as: \\boxed{{SCORE}}"
)


def _create_api_client(oracle_cfg: dict):
    """Create an OpenAI-compatible API client for the oracle model.

    Args:
        oracle_cfg: Oracle config dict from models.yaml.

    Returns:
        Configured OpenAI client.

    Raises:
        ValueError: If the required API key environment variable is not set.
    """
    from openai import OpenAI

    api_key_env = oracle_cfg["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"API key not set. Please set the '{api_key_env}' "
            f"environment variable.\n"
            f"  export {api_key_env}='your-api-key'"
        )

    return OpenAI(
        api_key=api_key,
        base_url=oracle_cfg["base_url"],
    )


def _score_single(
    client,
    model: str,
    prompt: str,
    max_retries: int = 3,
) -> tuple[Optional[int], str]:
    """Score a single prompt via API with retry logic.

    Args:
        client: OpenAI-compatible API client.
        model: Model ID string.
        prompt: Formatted scoring prompt.
        max_retries: Number of retry attempts on failure.

    Returns:
        Tuple of (score, raw_response_text).
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
            )
            text = response.choices[0].message.content.strip()
            score = parse_score(text)
            return score, text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    "API error (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error("API error after %d attempts: %s", max_retries, e)
                return None, f"API_ERROR: {e}"


def score_responses(
    responses_file: Path,
    oracle_key: str,
    oracle_cfg: dict,
    output_dir: Path,
) -> Path:
    """Score all responses in a single file using the oracle model via API.

    Args:
        responses_file: Path to student response JSON file.
        oracle_key: Oracle model key (e.g., 'gpt-oss-120b').
        oracle_cfg: Oracle config dict from models.yaml.
        output_dir: Directory for saving oracle scores.

    Returns:
        Path to the saved oracle scores file.
    """
    with open(responses_file) as f:
        responses = json.load(f)

    student_model = responses[0]["student_model"] if responses else "unknown"
    model_id = oracle_cfg["model"]
    provider = oracle_cfg.get("provider", "unknown")

    logger.info(
        "Scoring %d responses from %s using %s (%s via %s)",
        len(responses), student_model, oracle_key, model_id, provider,
    )

    client = _create_api_client(oracle_cfg)

    results = []
    parse_failures_t1 = 0
    parse_failures_t2 = 0

    for i, r in enumerate(responses):
        # --- Score Turn 1 ---
        t1_prompt = ORACLE_PROMPT_TURN1.format(
            question=r["turn1_question"],
            response=r["turn1_response"],
        )
        t1_score, t1_response = _score_single(client, model_id, t1_prompt)
        if t1_score is None:
            parse_failures_t1 += 1

        # --- Score Turn 2 ---
        t2_prompt = ORACLE_PROMPT_TURN2.format(
            turn1_question=r["turn1_question"],
            turn1_response=r["turn1_response"],
            turn2_question=r["turn2_question"],
            turn2_response=r["turn2_response"],
        )
        t2_score, t2_response = _score_single(client, model_id, t2_prompt)
        if t2_score is None:
            parse_failures_t2 += 1

        results.append({
            "problem_id": r["problem_id"],
            "question_id": r["question_id"],
            "category": r["category"],
            "turn1_question": r["turn1_question"],
            "turn1_response": r["turn1_response"],
            "turn2_question": r["turn2_question"],
            "turn2_response": r["turn2_response"],
            "student_model": r["student_model"],
            "oracle_model": oracle_key,
            "turn1_oracle_response": t1_response,
            "turn1_oracle_score": t1_score,
            "turn2_oracle_response": t2_response,
            "turn2_oracle_score": t2_score,
        })

        if (i + 1) % 10 == 0:
            logger.info("  Scored %d/%d questions", i + 1, len(responses))

    logger.info(
        "Oracle scoring complete. Parse failures: Turn1=%d/%d, Turn2=%d/%d",
        parse_failures_t1, len(responses),
        parse_failures_t2, len(responses),
    )

    # Save
    out_dir = output_dir / oracle_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{student_model}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Saved oracle scores to %s", out_file)
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Score MT-Bench student responses using oracle models via API.",
    )
    parser.add_argument(
        "--student-responses", required=True,
        help="Path to student response JSON file or directory of files",
    )
    parser.add_argument(
        "--oracle", default="all",
        help=(
            "Oracle model key from models.yaml, or 'all' to run both. "
            "Available: gpt-oss-120b, qwen3-235b (default: all)"
        ),
    )
    parser.add_argument(
        "--output-dir", default="results/mtbench_oracle",
        help="Directory for saving oracle scores",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load oracle config from models.yaml
    config = load_models_config()
    oracles = config.get("oracle_models", {})

    if args.oracle.lower() == "all":
        oracle_keys = list(oracles.keys())
    else:
        if args.oracle not in oracles:
            raise ValueError(
                f"Unknown oracle model: {args.oracle}. "
                f"Available: {list(oracles.keys())}"
            )
        oracle_keys = [args.oracle]

    responses_path = Path(args.student_responses)
    output_dir = Path(args.output_dir)

    # Collect response files
    if responses_path.is_file():
        response_files = [responses_path]
    elif responses_path.is_dir():
        response_files = sorted(responses_path.glob("*.json"))
    else:
        raise FileNotFoundError(f"Not found: {responses_path}")

    for oracle_key in oracle_keys:
        oracle_cfg = oracles[oracle_key]
        logger.info(
            "Oracle: %s (%s via %s)",
            oracle_key, oracle_cfg["model"], oracle_cfg.get("provider"),
        )

        for f in response_files:
            score_responses(f, oracle_key, oracle_cfg, output_dir)

    logger.info("All oracle scoring complete.")


if __name__ == "__main__":
    main()
