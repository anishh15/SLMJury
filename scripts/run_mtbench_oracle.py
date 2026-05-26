"""CLI entry-point: score MT-Bench student responses using oracle models via Together API.

Scores full 2-turn conversations holistically using large oracle models
hosted on Together AI. Uses the exact same PROMPT_MTBENCH as the SLM judges
to ensure fair correlation measurement. Includes configurable retry logic
and rate limiting.

Usage:
    python scripts/run_mtbench_oracle.py --oracle gpt-oss-120b \
        --student-responses results/mtbench_responses/llama3.1-8b.json
    python scripts/run_mtbench_oracle.py --oracle qwen3.5-397b \
        --student-responses results/mtbench_responses/
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

from slmjury.configs import load_models_config
from slmjury.core.scoring_judge import PROMPT_MTBENCH
from slmjury.parsers.score import parse_score


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_oracle_results(
    results: list[dict],
    output_dir: Path,
    oracle_key: str,
    student_model: str,
) -> Path:
    """Save oracle scoring results to JSON."""
    out_dir = output_dir / oracle_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{student_model}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Saved oracle scores to %s", out_file)
    return out_file


def _build_mtbench_prompt(response: dict) -> str:
    """Build the scoring prompt for one conversation.

    Uses the identical PROMPT_MTBENCH as the SLM scoring judges
    to ensure fair SLM-LLM correlation measurement.
    """
    return PROMPT_MTBENCH.format(
        turn1_question=response["turn1_question"],
        turn1_response=response["turn1_response"],
        turn2_question=response["turn2_question"],
        turn2_response=response["turn2_response"],
    )


def _extract_usage(completion) -> dict | None:
    """Extract token usage from a Together API completion response."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        k: getattr(usage, k)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, k)
    }


# ---------------------------------------------------------------------------
# Together API retry wrapper
# ---------------------------------------------------------------------------


def _create_together_completion_with_retries(
    client,
    *,
    model_id: str,
    messages: list[dict],
    cfg: dict,
    label: str,
):
    """Call Together chat completions with exponential backoff retries.

    Args:
        client: Together client instance.
        model_id: Together model identifier.
        messages: Chat messages to send.
        cfg: Oracle config dict (contains retry/token settings).
        label: Human-readable label for logging.

    Returns:
        Together API completion response.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    max_retries = int(cfg.get("max_retries", 6))
    initial_delay = float(cfg.get("retry_initial_delay", 2.0))
    max_delay = float(cfg.get("retry_max_delay", 60.0))
    max_tokens = int(cfg.get("max_tokens", 2048))

    request_kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": cfg.get("temperature", 0),
        "max_tokens": max_tokens,
    }
    if cfg.get("reasoning") is not None:
        request_kwargs["reasoning"] = cfg["reasoning"]
    if cfg.get("chat_template_kwargs"):
        request_kwargs["chat_template_kwargs"] = cfg["chat_template_kwargs"]

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**request_kwargs)
        except Exception as e:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Together request failed for {label} after "
                    f"{max_retries + 1} attempts"
                ) from e

            delay = min(max_delay, initial_delay * (2 ** attempt))
            delay += random.uniform(0, min(1.0, delay * 0.1))
            logger.warning(
                "Together request failed for %s (attempt %d/%d): %s. "
                "Retrying in %.1fs",
                label,
                attempt + 1,
                max_retries + 1,
                e,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"Unreachable retry state for {label}")


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def score_responses(
    responses_file: Path,
    oracle_key: str,
    oracle_cfg: dict,
    output_dir: Path,
) -> Path:
    """Score all responses in a file using an oracle model via Together API.

    Each conversation is scored individually using the identical
    PROMPT_MTBENCH used by SLM judges.

    Args:
        responses_file: Path to student response JSON file.
        oracle_key: Oracle model key (e.g., 'gpt-oss-120b').
        oracle_cfg: Oracle config dict from models.yaml.
        output_dir: Directory for saving oracle scores.

    Returns:
        Path to the saved oracle scores file.
    """
    from together import Together

    with open(responses_file) as f:
        responses = json.load(f)

    student_model = responses[0]["student_model"] if responses else "unknown"
    cfg = dict(oracle_cfg)
    model_id = cfg["model"]
    api_key_env = cfg.get("api_key_env", "TOGETHER_API_KEY")

    if not os.environ.get(api_key_env):
        raise RuntimeError(
            f"Missing required environment variable: {api_key_env}"
        )

    client = Together(api_key=os.environ[api_key_env])
    max_retries = int(cfg.get("max_retries", 6))
    request_delay = float(cfg.get("request_delay", 1.0))

    logger.info(
        "Scoring %d conversations from %s using Together oracle %s (%s), "
        "max_retries=%d",
        len(responses), student_model, oracle_key, model_id, max_retries,
    )

    results = []
    parse_failures = 0

    for i, r in enumerate(responses, start=1):
        label = f"{responses_file} item {i}/{len(responses)}"

        # Use the identical PROMPT_MTBENCH as SLM judges
        prompt = _build_mtbench_prompt(r)
        messages = [{"role": "user", "content": prompt}]

        # Try scoring with parse retries
        for parse_attempt in range(max_retries + 1):
            completion = _create_together_completion_with_retries(
                client,
                model_id=model_id,
                messages=messages,
                cfg=cfg,
                label=label,
            )
            response_text = (
                completion.choices[0].message.content or ""
            ).strip()
            usage = _extract_usage(completion)

            score = parse_score(response_text)
            if score is not None:
                break

            if parse_attempt >= max_retries:
                logger.warning(
                    "Failed to parse score for %s after %d attempts",
                    label, max_retries + 1,
                )
                parse_failures += 1
                break

            delay = min(
                float(cfg.get("retry_max_delay", 60.0)),
                float(cfg.get("retry_initial_delay", 2.0))
                * (2 ** parse_attempt),
            )
            logger.warning(
                "Could not parse score for %s (attempt %d/%d). "
                "Retrying in %.1fs",
                label, parse_attempt + 1, max_retries + 1, delay,
            )
            time.sleep(delay)

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
            "oracle_provider": "together",
            "oracle_response": response_text,
            "oracle_score": score,
            "oracle_usage": usage,
        })

        if i % 10 == 0:
            logger.info(
                "Together scored %d/%d from %s",
                i, len(responses), responses_file,
            )
        if request_delay > 0 and i < len(responses):
            time.sleep(request_delay)

    logger.info(
        "Oracle scoring complete. Parse failures: %d/%d",
        parse_failures, len(responses),
    )
    return _save_oracle_results(results, output_dir, oracle_key, student_model)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score MT-Bench student responses using oracle models "
            "via Together API."
        ),
    )
    parser.add_argument(
        "--student-responses", required=True,
        help="Path to student response JSON file or directory of files",
    )
    parser.add_argument(
        "--oracle", required=True,
        help=(
            "Oracle model key from models.yaml. "
            "Available: gpt-oss-120b, qwen3.5-397b"
        ),
    )
    parser.add_argument(
        "--output-dir", default="results/mtbench_oracle",
        help="Directory for saving oracle scores",
    )
    parser.add_argument(
        "--request-delay", type=float, default=None,
        help="Override request_delay from models.yaml (seconds between calls)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load oracle config from models.yaml
    config = load_models_config()
    oracles = config.get("oracle_models", {})

    if args.oracle not in oracles:
        raise ValueError(
            f"Unknown oracle model: {args.oracle}. "
            f"Available: {list(oracles.keys())}"
        )

    oracle_cfg = dict(oracles[args.oracle])
    responses_path = Path(args.student_responses)
    output_dir = Path(args.output_dir)

    # Apply CLI overrides
    if args.request_delay is not None:
        oracle_cfg["request_delay"] = args.request_delay

    # Collect response files
    if responses_path.is_file():
        response_files = [responses_path]
    elif responses_path.is_dir():
        response_files = sorted(responses_path.glob("*.json"))
    else:
        raise FileNotFoundError(f"Not found: {responses_path}")

    logger.info(
        "Oracle: %s (%s) — scoring %d student file(s) via Together API",
        args.oracle, oracle_cfg["model"], len(response_files),
    )

    for f in response_files:
        score_responses(f, args.oracle, oracle_cfg, output_dir)

    logger.info("All oracle scoring complete.")


if __name__ == "__main__":
    main()
