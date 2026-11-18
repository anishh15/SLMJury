"""Student solver: generates solutions with large student models served by vLLM.

Loads a student once and solves problems across datasets, with a prompt template per dataset type
(science multiple choice has its own).
"""

import gc
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from slmjury.configs import load_models_config
from slmjury.data import DATASET_LOADERS
from slmjury.parsers.answer import extract_answer, get_dataset_type
from slmjury.parsers.normalizer import answers_are_equivalent

logger = logging.getLogger(__name__)

# Prompt templates per dataset type
PROMPT_TEMPLATES = {
    "numeric": (
        "Think step-by-step to solve this math problem. "
        "Give your final answer after ####.\n\n"
        "Problem: {question}\n\nSolution:"
    ),
    "latex": (
        "Think step-by-step to solve this math problem. "
        "Put your final answer in \\boxed{{}}.\n\n"
        "Problem: {question}\n\nSolution:"
    ),
    "multiple_choice": (
        "Think step-by-step to answer this science question. "
        "Choose A, B, C, or D.\n\n{question}\n\n"
        "Think step-by-step, then give your final answer."
    ),
    "multiple_choice_general": (
        "Think step-by-step to answer this question. "
        "Choose the best option.\n\n{question}\n\n"
        "Think step-by-step, then give your final answer."
    ),
}

# Datasets that use science-specific MC prompt (all other MC datasets use general)
_SCIENCE_MC_DATASETS = frozenset({"arc_easy", "arc_challenge"})


def _get_prompt_key(dataset_name: str, dataset_type: str) -> str:
    """The prompt template key for a dataset.

    Science multiple choice (ARC) gets the science prompt; the other multiple-choice datasets (HellaSwag,
    WinoGrande, TruthfulQA) get the general one.
    """
    if dataset_type == "multiple_choice" and dataset_name not in _SCIENCE_MC_DATASETS:
        return "multiple_choice_general"
    return dataset_type


class StudentSolver:
    """Generates solutions with a large student model via vLLM.

    ``model_key`` is a key under student_models in models.yaml; ``model_config`` replaces the YAML lookup.
    """

    def __init__(
        self,
        model_key: str,
        output_dir: Path = Path("results/core/student_solutions"),
        model_config: Optional[dict] = None,
    ):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.model_key = model_key
        self.output_dir = Path(output_dir)

        if model_config:
            cfg = model_config
        else:
            config = load_models_config()
            students = config.get("student_models", {})
            if model_key not in students:
                raise ValueError(
                    f"Unknown student model: {model_key}. "
                    f"Available: {list(students.keys())}"
                )
            cfg = students[model_key]

        # Apply overrides from model_config if YAML was used as base
        if model_config and "model" not in model_config:
            config = load_models_config()
            students = config.get("student_models", {})
            if model_key in students:
                base = students[model_key]
                base.update(model_config)
                cfg = base

        self.model_name = cfg["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)

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

        logger.info("Loading student model: %s", self.model_name)
        self.llm = LLM(**vllm_kwargs)
        logger.info("Student model loaded: %s", self.model_name)

    def solve_batch(
        self,
        problems: list[dict],
        dataset_name: str,
        num_samples: Optional[int] = None,
    ) -> list[dict]:
        """Solve a dataset's problems (the first ``num_samples`` if given); returns one result dict per problem.

        ``dataset_name`` picks the prompt template.
        """
        from vllm import SamplingParams

        if num_samples:
            problems = problems[:num_samples]

        dataset_type = get_dataset_type(dataset_name)
        prompt_key = _get_prompt_key(dataset_name, dataset_type)
        template = PROMPT_TEMPLATES[prompt_key]

        logger.info("Solving %d %s problems...", len(problems), dataset_name)

        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": template.format(question=p["question"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in problems
        ]

        params = SamplingParams(temperature=0.7, max_tokens=1024, top_p=0.9)
        outputs = self.llm.generate(prompts, params)

        results = []
        for i, (problem, output) in enumerate(zip(problems, outputs)):
            response = output.outputs[0].text.strip()
            answer = extract_answer(response, dataset_type)
            results.append({
                "problem_id": problem["problem_id"],
                "question": problem["question"],
                "ground_truth_reasoning": problem["ground_truth_reasoning"],
                "ground_truth_answer": problem["ground_truth_answer"],
                "student_reasoning": response,
                "student_answer": answer,
                "dataset": dataset_name,
                "model": self.model_key,
            })
            if (i + 1) % 500 == 0:
                logger.info("  Processed %d/%d", i + 1, len(problems))

        logger.info("Solving complete for %s", dataset_name)
        return results

    def save_results(self, results: list[dict], dataset_name: str) -> Path:
        """Save student solutions to ``<output_dir>/<student>/<dataset>.json`` and return the path."""
        out_dir = self.output_dir / self.model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{dataset_name}.json"
        out_file = out_dir / filename

        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Saved %d solutions to %s", len(results), out_file.name)
        return out_file

    def cleanup(self, clear_hf_cache: bool = False):
        """Release GPU memory and clean up distributed processes.

        ``clear_hf_cache=True`` also deletes the model's HF cache; it defaults to False (see JudgeModel.cleanup).
        """
        model_name = getattr(self, "model_name", None)

        if hasattr(self, "llm"):
            del self.llm
        gc.collect()

        try:
            import torch
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.debug("Cleanup warning: %s", e)

        gc.collect()

        # Reset distributed env vars
        for key in list(os.environ.keys()):
            if key.startswith(("NCCL_", "RANK", "WORLD")):
                if key not in ("NCCL_DEBUG", "NCCL_IB_DISABLE"):
                    os.environ.pop(key, None)

        if clear_hf_cache and model_name:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir_name = f"models--{model_name.replace('/', '--')}"
            model_cache = cache_dir / model_dir_name
            if model_cache.exists():
                shutil.rmtree(model_cache, ignore_errors=True)
                logger.info("Cleared HF cache: %s", model_dir_name)

        logger.debug("Cleanup complete")


def calculate_accuracy(results: list[dict], dataset_name: str) -> dict:
    """Accuracy of a set of student results against their ground-truth answers.

    Returns correct, valid (an answer was extracted), total, accuracy and accuracy_pct; ``dataset_name``
    picks the comparison.
    """
    dataset_type = get_dataset_type(dataset_name)
    total = len(results)

    correct = sum(
        1 for r in results
        if answers_are_equivalent(r["student_answer"], r["ground_truth_answer"], dataset_type)
    )
    valid = sum(
        1 for r in results
        if r.get("student_answer", "").strip()
        and r["student_answer"].upper() != "UNABLE_TO_EXTRACT"
    )
    accuracy = correct / total if total > 0 else 0.0

    return {
        "correct": correct,
        "valid": valid,
        "total": total,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 2),
    }


def save_accuracy_summary(
    accuracy_data: dict,
    model_key: str,
    output_dir: Path = Path("results/core/summaries"),
) -> Path:
    """Merge one student's per-dataset accuracy into ``<output_dir>/student_accuracy.json`` and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "student_accuracy.json"

    # Load existing data if file exists
    if summary_file.exists():
        with open(summary_file) as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # Update with new data
    if model_key not in all_data:
        all_data[model_key] = {}
    all_data[model_key].update(accuracy_data)
    all_data[model_key]["last_updated"] = datetime.now().isoformat()

    with open(summary_file, "w") as f:
        json.dump(all_data, f, indent=2)

    logger.info("Accuracy summary saved to %s", summary_file)
    return summary_file


def recalculate_accuracy_from_files(
    solutions_dir: Path = Path("results/core/student_solutions"),
    output_dir: Path = Path("results/core/summaries"),
):
    """Rebuild ``<output_dir>/student_accuracy.json`` from the saved solution files, without rerunning inference.

    Reads ``<solutions_dir>/<student>/*.json``.
    """
    logger.info("Recalculating student accuracy from existing solutions...")

    # Clear existing summary
    summary_file = output_dir / "student_accuracy.json"
    if summary_file.exists():
        summary_file.unlink()

    for model_dir in sorted(solutions_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_key = model_dir.name
        logger.info("Processing %s...", model_key)

        accuracy_data = {}

        for solution_file in sorted(model_dir.glob("*.json")):
            with open(solution_file) as f:
                results = json.load(f)

            if not results:
                continue

            # Extract dataset name from file data or filename
            dataset = results[0].get("dataset", solution_file.stem)

            acc = calculate_accuracy(results, dataset)
            accuracy_data[dataset] = acc

            logger.info(
                "  %s: %d/%d correct, %d/%d valid",
                dataset, acc["correct"], acc["total"],
                acc["valid"], acc["total"],
            )

        if accuracy_data:
            save_accuracy_summary(accuracy_data, model_key, output_dir)

    logger.info("Accuracy summary recalculated: %s", summary_file)

