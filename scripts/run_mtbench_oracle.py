"""CLI entry-point: score MT-Bench student responses using oracle models via vLLM.

Loads oracle models locally and scores full 2-turn conversations holistically.
Uses the same prompt as the SLM judges for fair correlation measurement.

Usage:
    python scripts/run_mtbench_oracle.py --oracle gpt-oss-120b \
        --student-responses results/mtbench_responses/llama3.1-8b.json
    python scripts/run_mtbench_oracle.py --oracle qwen3.5-397b \
        --student-responses results/mtbench_responses/ --tp 4 --gpu-mem 0.9
"""

import argparse
import gc
import json
import logging
from pathlib import Path

from slmjury.configs import load_models_config
from slmjury.parsers.score import parse_score


logger = logging.getLogger(__name__)

from slmjury.core.scoring_judge import PROMPT_MTBENCH


def score_responses(
    responses_file: Path,
    oracle_key: str,
    oracle_cfg: dict,
    output_dir: Path,
    tp_override: int | None = None,
    gpu_mem_override: float | None = None,
) -> Path:
    """Score all responses in a file using an oracle model via vLLM.

    Args:
        responses_file: Path to student response JSON file.
        oracle_key: Oracle model key (e.g., 'gpt-oss-120b').
        oracle_cfg: Oracle config dict from models.yaml.
        output_dir: Directory for saving oracle scores.
        tp_override: Override tensor_parallel_size.
        gpu_mem_override: Override gpu_memory_utilization.

    Returns:
        Path to the saved oracle scores file.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    with open(responses_file) as f:
        responses = json.load(f)

    student_model = responses[0]["student_model"] if responses else "unknown"
    cfg = dict(oracle_cfg)

    # Apply CLI overrides
    if tp_override is not None:
        cfg["tensor_parallel_size"] = tp_override
    if gpu_mem_override is not None:
        cfg["gpu_memory_utilization"] = gpu_mem_override

    model_id = cfg["model"]

    logger.info(
        "Scoring %d conversations from %s using oracle %s (%s)",
        len(responses), student_model, oracle_key, model_id,
    )

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    vllm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": cfg.get("tensor_parallel_size", 4),
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.9),
        "trust_remote_code": True,
        "enforce_eager": True,
        "dtype": cfg.get("dtype", "float16"),
    }
    if cfg.get("max_model_len"):
        vllm_kwargs["max_model_len"] = cfg["max_model_len"]
    if cfg.get("quantization"):
        vllm_kwargs["quantization"] = cfg["quantization"]
    if cfg.get("max_num_seqs"):
        vllm_kwargs["max_num_seqs"] = cfg["max_num_seqs"]

    llm = LLM(**vllm_kwargs)
    params = SamplingParams(temperature=0, max_tokens=4096)

    # Build prompts
    prompts = []
    for r in responses:
        user_content = PROMPT_MTBENCH.format(
            turn1_question=r["turn1_question"],
            turn1_response=r["turn1_response"],
            turn2_question=r["turn2_question"],
            turn2_response=r["turn2_response"],
        )
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(chat_prompt)

    # Batch generate
    logger.info("Running batch inference on %d prompts...", len(prompts))
    outputs = llm.generate(prompts, params)

    # Parse results
    results = []
    parse_failures = 0

    for r, output in zip(responses, outputs):
        response_text = output.outputs[0].text.strip()
        score = parse_score(response_text)
        if score is None:
            parse_failures += 1

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
            "oracle_response": response_text,
            "oracle_score": score,
        })

    logger.info(
        "Oracle scoring complete. Parse failures: %d/%d",
        parse_failures, len(responses),
    )

    # Cleanup GPU memory before next run
    del llm, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

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
        description="Score MT-Bench student responses using oracle models via vLLM.",
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

    # Hardware overrides
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=None,
        help="Override tensor_parallel_size from models.yaml",
    )
    parser.add_argument(
        "--gpu-memory-utilization", "--gpu-mem", type=float, default=None,
        help="Override gpu_memory_utilization from models.yaml",
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
    output_dir = Path(args.output_dir)

    # Collect response files
    if responses_path.is_file():
        response_files = [responses_path]
    elif responses_path.is_dir():
        response_files = sorted(responses_path.glob("*.json"))
    else:
        raise FileNotFoundError(f"Not found: {responses_path}")

    logger.info(
        "Oracle: %s (%s) — scoring %d student file(s)",
        args.oracle, oracle_cfg["model"], len(response_files),
    )

    for f in response_files:
        score_responses(
            f, args.oracle, oracle_cfg, output_dir,
            tp_override=args.tensor_parallel_size,
            gpu_mem_override=args.gpu_memory_utilization,
        )

    logger.info("All oracle scoring complete.")


if __name__ == "__main__":
    main()
