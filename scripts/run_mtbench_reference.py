"""CLI entry-point: score MT-Bench student responses with a reference model (Together API)."""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

from slmjury.analysis import paths
from slmjury.configs import load_models_config
from slmjury.core.scoring_judge import PROMPT_MTBENCH
from slmjury.parsers.score import parse_score

NOTES = """\
A large reference model, hosted on Together AI, scores each full 2-turn
conversation once. It gets the same PROMPT_MTBENCH as the SLM judges, so a
judge's correlation with the reference compares scores of the same prompt.
Requests are retried with exponential backoff and spaced by a configurable
delay. The reference models are listed under reference_models in
slmjury/configs/models.yaml.

examples:
  python scripts/run_mtbench_reference.py --reference gpt-oss-120b \\
      --solutions-dir results/core/mtbench_responses/llama3.1-8b.json
  python scripts/run_mtbench_reference.py --reference qwen3.5-397b

Writes results/core/mtbench_references/<reference>/<student>.json. Each record
uses the field names the scoring code reads.
"""


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_reference_results(
    results: list[dict],
    output_dir: Path,
    reference_key: str,
    student_model: str,
) -> Path:
    """Save the reference model's scores to JSON."""
    out_dir = output_dir / reference_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{student_model}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Saved reference scores to %s", out_file)
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
    """Call Together chat completions, retrying with exponential backoff.

    Retry and token settings come from ``cfg``, the reference model's config.
    Raises RuntimeError once every attempt has failed.
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
    reference_key: str,
    reference_cfg: dict,
    output_dir: Path,
) -> Path:
    """Score every conversation in ``responses_file`` with a reference model via the Together API.

    Each conversation is scored on its own, with the same PROMPT_MTBENCH the SLM
    judges use. Returns the path of the saved reference scores file.
    """
    from together import Together

    with open(responses_file) as f:
        responses = json.load(f)

    student_model = responses[0]["student_model"] if responses else "unknown"
    cfg = dict(reference_cfg)
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
        "Scoring %d conversations from %s with reference model %s (%s), "
        "max_retries=%d",
        len(responses), student_model, reference_key, model_id, max_retries,
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
            # Field names kept as the scoring code reads them.
            "oracle_model": reference_key,
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
        "Reference scoring complete. Parse failures: %d/%d",
        parse_failures, len(responses),
    )
    return _save_reference_results(results, output_dir, reference_key, student_model)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score MT-Bench student responses with a reference model "
            "via Together API."
        ),
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--solutions-dir", default=paths.default_dir("core", "mtbench_responses"),
        help="Directory of student MT-Bench responses, or one response file "
             "(default: core/mtbench_responses in the results folder)",
    )
    # Older spelling, kept working but not shown in --help.
    parser.add_argument("--student-responses", dest="solutions_dir",
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "--reference",
        help=(
            "Reference model key from reference_models in models.yaml. "
            "Available: gpt-oss-120b, qwen3.5-397b"
        ),
    )
    reference.add_argument("--oracle", dest="reference", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir", default=paths.default_dir("core", "mtbench_references"),
        help="Directory for saving the reference scores",
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

    # Load the reference model's config from models.yaml
    config = load_models_config()
    references = config.get("reference_models", {})

    if args.reference not in references:
        raise ValueError(
            f"Unknown reference model: {args.reference}. "
            f"Available: {list(references.keys())}"
        )

    reference_cfg = dict(references[args.reference])
    responses_path = Path(args.solutions_dir)
    output_dir = Path(args.output_dir)

    # Apply CLI overrides
    if args.request_delay is not None:
        reference_cfg["request_delay"] = args.request_delay

    # Collect response files
    if responses_path.is_file():
        response_files = [responses_path]
    elif responses_path.is_dir():
        response_files = sorted(responses_path.glob("*.json"))
    else:
        raise FileNotFoundError(f"Not found: {responses_path}")

    logger.info(
        "Reference model: %s (%s) — scoring %d student file(s) via Together API",
        args.reference, reference_cfg["model"], len(response_files),
    )

    for f in response_files:
        score_responses(f, args.reference, reference_cfg, output_dir)

    logger.info("All reference scoring complete.")


if __name__ == "__main__":
    main()
