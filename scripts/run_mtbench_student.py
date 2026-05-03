"""CLI entry-point: generate student model responses for MT-Bench (2 turns).

MT-Bench is open-ended — the student generates free-form responses to 80
questions, each with 2 turns. Turn 2 is a follow-up that depends on the
model's Turn 1 response. These responses are then scored by oracle models
and SLM judges.

Usage:
    python scripts/run_mtbench_student.py --model llama3.1-8b
    python scripts/run_mtbench_student.py --model qwen2.5-32b --tp 2 --gpu-mem 0.85

    # Run both students:
    #   bash bash/run_mtbench_students.sh
"""

import argparse
import gc
import json
import logging
from pathlib import Path

from slmjury.configs import load_models_config
from slmjury.data import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate student responses for MT-Bench (2 turns).",
    )
    parser.add_argument(
        "--model", required=True,
        help="Student model key from models.yaml",
    )
    parser.add_argument(
        "--output-dir", default="results/mtbench_responses",
        help="Directory for saving responses",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-generate even if response file already exists",
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
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    out_file = output_dir / f"{args.model}.json"

    if out_file.exists() and not args.force:
        logger.info("SKIP %s — responses already exist (use --force)", args.model)
        return

    # Load model config
    config = load_models_config()
    students = config.get("student_models", {})
    if args.model not in students:
        raise ValueError(
            f"Unknown student model: {args.model}. "
            f"Available: {list(students.keys())}"
        )
    cfg = dict(students[args.model])

    # Apply CLI overrides
    if args.tensor_parallel_size is not None:
        cfg["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        cfg["gpu_memory_utilization"] = args.gpu_memory_utilization

    # Load MT-Bench prompts (2 turns per question)
    prompts = load_dataset("mtbench")
    logger.info("Loaded %d MT-Bench questions (2 turns each)", len(prompts))

    # Load model
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)

    vllm_kwargs = {
        "model": cfg["model"],
        "tensor_parallel_size": cfg.get("tensor_parallel_size", 4),
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.7),
        "trust_remote_code": True,
        "enforce_eager": True,
        "dtype": cfg.get("dtype", "float16"),
    }
    if cfg.get("max_model_len"):
        vllm_kwargs["max_model_len"] = cfg["max_model_len"]
    if cfg.get("quantization"):
        vllm_kwargs["quantization"] = cfg["quantization"]

    logger.info("Loading student model: %s", cfg["model"])
    llm = LLM(**vllm_kwargs)

    params = SamplingParams(temperature=0.7, max_tokens=2048, top_p=0.9)

    # --- Turn 1: Generate responses for all 80 questions ---
    logger.info("Generating Turn 1 responses...")
    turn1_chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["turn1"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    turn1_outputs = llm.generate(turn1_chat_prompts, params)
    turn1_responses = [
        output.outputs[0].text.strip() for output in turn1_outputs
    ]
    logger.info("Turn 1 complete: %d responses", len(turn1_responses))

    # --- Turn 2: Build multi-turn conversations and generate ---
    logger.info("Generating Turn 2 responses...")
    turn2_chat_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": p["turn1"]},
                {"role": "assistant", "content": t1_resp},
                {"role": "user", "content": p["turn2"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p, t1_resp in zip(prompts, turn1_responses)
    ]

    turn2_outputs = llm.generate(turn2_chat_prompts, params)
    turn2_responses = [
        output.outputs[0].text.strip() for output in turn2_outputs
    ]
    logger.info("Turn 2 complete: %d responses", len(turn2_responses))

    # --- Build results ---
    results = []
    for prompt_data, t1_resp, t2_resp in zip(
        prompts, turn1_responses, turn2_responses,
    ):
        results.append({
            "problem_id": prompt_data["problem_id"],
            "question_id": prompt_data["question_id"],
            "category": prompt_data["category"],
            "turn1_question": prompt_data["turn1"],
            "turn1_response": t1_resp,
            "turn2_question": prompt_data["turn2"],
            "turn2_response": t2_resp,
            "student_model": args.model,
        })

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Saved %d responses (2 turns each) to %s", len(results), out_file)

    # Cleanup
    del llm, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
