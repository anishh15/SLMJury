"""Student solver — generates solutions using large student models via vLLM.

Loads a student model once and solves problems across multiple datasets
efficiently. Supports configurable prompt templates per dataset type.
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
}


class StudentSolver:
    """Generates solutions using a large student model via vLLM.

    Args:
        model_key: Key from models.yaml student_models section.
        output_dir: Base directory for saving results.
        model_config: Optional config dict override (skips YAML lookup).
    """

    def __init__(
        self,
        model_key: str,
        output_dir: Path = Path("results/student_solutions"),
        model_config: Optional[dict] = None,
    ):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.model_key = model_key
        self.output_dir = output_dir

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
        """Solve a batch of problems from a dataset.

        Args:
            problems: List of problem dictionaries.
            dataset_name: Name of the dataset (for prompt template selection).
            num_samples: Optional limit on number of problems.

        Returns:
            List of result dictionaries with student solutions.
        """
        from vllm import SamplingParams

        if num_samples:
            problems = problems[:num_samples]

        dataset_type = get_dataset_type(dataset_name)
        template = PROMPT_TEMPLATES[dataset_type]

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
        """Save student solutions to JSON.

        Args:
            results: List of result dicts from solve_batch.
            dataset_name: Dataset name used in filename.

        Returns:
            Path to the saved JSON file.
        """
        out_dir = self.output_dir / self.model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.model_key}_{dataset_name}.json"
        out_file = out_dir / filename

        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Saved %d solutions to %s", len(results), out_file.name)
        return out_file

    def cleanup(self):
        """Release GPU memory and clean up distributed processes."""
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

        logger.debug("Cleanup complete")


def calculate_accuracy(results: list[dict], dataset_name: str) -> dict:
    """Calculate accuracy metrics for a set of student results.

    Args:
        results: List of result dicts with student_answer and ground_truth_answer.
        dataset_name: Dataset name for choosing comparison function.

    Returns:
        Dict with correct, valid, total, accuracy, and accuracy_pct.
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
    output_dir: Path = Path("results/summaries"),
) -> Path:
    """Save accuracy summary, appending to existing file if present.

    Args:
        accuracy_data: Dict mapping dataset name → accuracy metrics dict.
        model_key: Student model key.
        output_dir: Directory for the summary JSON.

    Returns:
        Path to the saved summary file.
    """
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
    solutions_dir: Path = Path("results/student_solutions"),
    output_dir: Path = Path("results/summaries"),
):
    """Recalculate accuracy summary from existing student solution files.

    Useful when the accuracy calculation logic changes — avoids re-running
    inference.

    Args:
        solutions_dir: Directory containing per-model solution subdirectories.
        output_dir: Directory for the summary JSON.
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
            dataset = results[0].get(
                "dataset", solution_file.stem.split("_")[-1],
            )

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

