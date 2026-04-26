"""CLI entry-point: score MT-Bench student responses using the oracle model.

This script is designed to be run by the supervisor on their setup where
the GPT-OSS-120B oracle model is available. It reads student response
files and produces oracle scores (1-5) for each response.

Reads oracle model config from models.yaml and reuses the shared
parse_score() parser — no duplicate logic.

Usage:
    python scripts/run_mtbench_oracle.py --student-responses results/mtbench_responses/llama3.1-8b.json
    python scripts/run_mtbench_oracle.py --student-responses results/mtbench_responses/
    python scripts/run_mtbench_oracle.py --student-responses results/mtbench_responses/ --tp 1 --gpu-mem 0.9
"""

import argparse
import gc
import json
import logging
import re
from pathlib import Path

from slmjury.configs import load_models_config
from slmjury.parsers.score import parse_score

logger = logging.getLogger(__name__)

# Oracle scoring prompt (same scale as SLM judges for fair comparison)
ORACLE_PROMPT = (
    "You are an expert evaluator. Rate the quality of the following response "
    "on a scale of 1-5.\n"
    "1 = Very poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent.\n\n"
    "[Question]:\n{question}\n\n"
    "[Response]:\n{response}\n\n"
    "Consider helpfulness, accuracy, depth, creativity, and level of detail.\n"
    "Provide your rating as: \\boxed{{SCORE}}"
)


def score_responses(responses_file: Path, oracle_cfg: dict, args) -> Path:
    """Score all responses in a single file using the oracle model."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    with open(responses_file) as f:
        responses = json.load(f)

    student_model = responses[0]["student_model"] if responses else "unknown"
    logger.info(
        "Scoring %d responses from %s (%s)",
        len(responses), student_model, responses_file.name,
    )

    # Apply CLI overrides
    cfg = dict(oracle_cfg)
    if args.tensor_parallel_size is not None:
        cfg["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        cfg["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        cfg["max_num_seqs"] = args.max_num_seqs

    model_name = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=cfg.get("tensor_parallel_size", 4),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_num_seqs=cfg.get("max_num_seqs", 16),
        enable_chunked_prefill=False,
        trust_remote_code=True,
        enforce_eager=True,
        dtype=cfg.get("dtype", "auto"),
    )

    # Build prompts
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ORACLE_PROMPT.format(
                question=r["question"],
                response=r["response"],
            )}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in responses
    ]

    params = SamplingParams(temperature=0, max_tokens=4096)
    outputs = llm.generate(prompts, params)

    # Parse scores using shared parser
    results = []
    parse_failures = 0
    for r, output in zip(responses, outputs):
        oracle_response = output.outputs[0].text.strip()
        # Strip thinking tags if present
        oracle_response = re.sub(
            r'<think>.*?</think>', '', oracle_response, flags=re.DOTALL,
        ).strip()

        score = parse_score(oracle_response)
        if score is None:
            parse_failures += 1

        results.append({
            "problem_id": r["problem_id"],
            "question_id": r["question_id"],
            "category": r["category"],
            "question": r["question"],
            "response": r["response"],
            "student_model": r["student_model"],
            "oracle_response": oracle_response,
            "oracle_score": score,
        })

    logger.info(
        "Oracle scoring complete. Parse failures: %d/%d (%.1f%%)",
        parse_failures, len(responses),
        100 * parse_failures / len(responses) if responses else 0,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{student_model}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Saved oracle scores to %s", out_file)

    # Cleanup
    del llm, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Score MT-Bench student responses using oracle model.",
    )
    parser.add_argument(
        "--student-responses", required=True,
        help="Path to student response JSON file or directory of files",
    )
    parser.add_argument(
        "--oracle", default="gpt-oss-120b",
        help="Oracle model key from models.yaml (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--output-dir", default="results/mtbench_oracle",
        help="Directory for saving oracle scores",
    )

    # Hardware overrides
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=None,
        help="Override tensor_parallel_size from models.yaml",
    )
    parser.add_argument(
        "--gpu-memory-utilization", "--gpu-mem", type=float, default=None,
        help="Override gpu_memory_utilization from models.yaml",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="Override max_num_seqs from models.yaml",
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
    oracle_cfg = oracles[args.oracle]

    responses_path = Path(args.student_responses)

    if responses_path.is_file():
        score_responses(responses_path, oracle_cfg, args)
    elif responses_path.is_dir():
        for f in sorted(responses_path.glob("*.json")):
            score_responses(f, oracle_cfg, args)
    else:
        raise FileNotFoundError(f"Not found: {responses_path}")

    logger.info("All oracle scoring complete.")


if __name__ == "__main__":
    main()
