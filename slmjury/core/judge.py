"""Judge model: evaluates student solutions with SLM judges served by vLLM.

Loads a judge once for many student-dataset pairs. Quick verdict (10 tokens) or reasoned verdict (8,192
tokens), with optional thinking mode for Qwen3 and Phi-4 reasoning models.
"""

import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from slmjury.configs import get_judge_config
from slmjury.parsers.judgement import parse_judgement

logger = logging.getLogger(__name__)

# Prompt for quick verdict (max_tokens=10)
PROMPT_QUICK = (
    "Your role is to compare the student's answer to the ground truth "
    "and determine correctness.\n\n"
    "Respond with exactly one word:\n"
    "- 'Correct': If the student's final answer matches the ground truth.\n"
    "- 'Incorrect': If the student's final answer does not match.\n\n"
    "Focus only on the final answer, ignore reasoning steps.\n\n"
    "Question: {question}\n\n"
    "Ground truth answer: {ground_truth}\n\n"
    "Student answer: {student_answer}"
)

# Prompt for reasoned verdict (max_tokens=8192)
PROMPT_REASONED = (
    "Your role is to compare the student's answer to the ground truth "
    "and determine correctness.\n\n"
    "First, explain your reasoning for the judgment.\n"
    "Then, on a new line, give the final verdict in the exact format:\n"
    "\\boxed{{CORRECT}} or \\boxed{{INCORRECT}}\n\n"
    "Focus only on the final answer when judging correctness. "
    "Ignore reasoning steps in the student's solution.\n\n"
    "Question: {question}\n\n"
    "Ground truth answer: {ground_truth}\n\n"
    "Student answer: {student_answer}\n"
)


class JudgeModel:
    """SLM judge for evaluating student solutions via vLLM.

    ``model_key`` is a key under judge_models or large_judge_models in models.yaml. ``model_config`` overrides
    its config (strategies use it for tensor_parallel_size, memory settings and so on).
    """

    def __init__(
        self,
        model_key: str,
        output_dir: Path = Path("results/core/judgements"),
        model_config: Optional[dict] = None,
    ):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.model_key = model_key
        self.output_dir = Path(output_dir)

        if model_config and "model" in model_config:
            cfg = model_config
        else:
            # Small judges, then the large and ladder judges (models.yaml).
            cfg = get_judge_config(model_key)
            if model_config:
                cfg.update(model_config)

        self.model_name = cfg["model"]
        self.enable_thinking = cfg.get("enable_thinking", False)
        self.always_thinks = cfg.get("always_thinks", False)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)

        vllm_kwargs = {
            "model": cfg["model"],
            "tensor_parallel_size": cfg.get("tensor_parallel_size", 4),
            "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.8),
            "max_num_seqs": cfg.get("max_num_seqs", 32),
            "enable_chunked_prefill": False,
            "trust_remote_code": True,
            "enforce_eager": True,
            "dtype": cfg.get("dtype", "float16"),
        }
        if cfg.get("max_model_len"):
            vllm_kwargs["max_model_len"] = cfg["max_model_len"]

        # Extra vLLM engine arguments from the model config (optional). Use it for
        # serving-stack settings that do not change sampling, for example
        #     vllm_extra: {attention_config: {flash_attn_version: 2}}
        # to pin FlashAttention 2 on GPUs where version 3 is unstable.
        for _k, _v in (cfg.get("vllm_extra") or {}).items():
            vllm_kwargs[_k] = _v

        logger.info("Loading judge model: %s", self.model_name)
        self.llm = LLM(**vllm_kwargs)
        logger.info("Judge model loaded: %s", self.model_name)

    def evaluate_batch(
        self,
        student_results: list[dict],
        max_tokens: int = 10,
        system_prompt: Optional[str] = None,
        prompt_style: Optional[str] = None,
        thinking: Optional[bool] = None,
        temperature: Optional[float] = None,
        prompt_template: Optional[str] = None,
    ) -> list[dict]:
        """Judge a batch of student solutions; returns judgement dicts with parsed verdicts.

        By default ``max_tokens`` sets the prompt (PROMPT_QUICK at 10, else PROMPT_REASONED) and thinking (at 8,192,
        for judges with enable_thinking); ``prompt_style``, ``thinking``, ``temperature`` and ``prompt_template`` override them.
        """
        from vllm import SamplingParams

        logger.info(
            "Evaluating %d solutions (max_tokens=%d)...",
            len(student_results), max_tokens,
        )

        # Resolve thinking mode
        # None = original auto-select (enable_thinking AND max_tokens == 8192)
        if thinking is None:
            enable_thinking = self.enable_thinking and max_tokens == 8192
        else:
            enable_thinking = thinking and self.enable_thinking

        # Resolve prompt template (an explicit prompt_template wins)
        # None = original auto-select (max_tokens == 10 → PROMPT_QUICK)
        if prompt_template is None:
            if prompt_style is None:
                prompt_template = PROMPT_QUICK if max_tokens == 10 else PROMPT_REASONED
            elif prompt_style == "quick":
                prompt_template = PROMPT_QUICK
            else:
                prompt_template = PROMPT_REASONED

        # Build chat messages
        messages_list = []
        for r in student_results:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            ground_truth = (
                r["ground_truth_reasoning"]
                if r["ground_truth_reasoning"]
                else r["ground_truth_answer"]
            )
            msgs.append({
                "role": "user",
                "content": prompt_template.format(
                    question=r["question"],
                    ground_truth=ground_truth,
                    student_answer=r["student_reasoning"],
                ),
            })
            messages_list.append(msgs)

        # Apply chat template
        extra_kwargs = (
            {"enable_thinking": enable_thinking}
            if "qwen3" in self.model_key
            else {}
        )
        prompts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **extra_kwargs,
            )
            for msgs in messages_list
        ]

        # Configure sampling
        # None = original auto-select
        if enable_thinking:
            _temp = temperature if temperature is not None else 0.6
            params = SamplingParams(
                temperature=_temp, top_p=0.95, top_k=20, min_p=0,
                max_tokens=max_tokens,
            )
        else:
            _temp = temperature if temperature is not None else 0
            params = SamplingParams(temperature=_temp, max_tokens=max_tokens)

        outputs = self.llm.generate(prompts, params)

        # Parse results
        results = []
        verdict_counts = {"Correct": 0, "Incorrect": 0, "Undefined": 0}

        for i, (student, output) in enumerate(zip(student_results, outputs)):
            response = output.outputs[0].text.strip()

            if enable_thinking:
                response = re.sub(
                    r'<think>.*?</think>', '', response, flags=re.DOTALL,
                ).strip()

            judgement = parse_judgement(response)
            verdict_counts[judgement] += 1

            results.append({
                "problem_id": student["problem_id"],
                "question": student["question"],
                "ground_truth_reasoning": student["ground_truth_reasoning"],
                "ground_truth_answer": student["ground_truth_answer"],
                "student_reasoning": student["student_reasoning"],
                "student_answer": student["student_answer"],
                "judge_response": response,
                "judgement": judgement,
            })

            if (i + 1) % 500 == 0:
                logger.info("  Processed %d/%d", i + 1, len(student_results))

        logger.info("Evaluation complete. Verdicts: %s", verdict_counts)
        return results

    def save_results(
        self,
        results: list[dict],
        student_model: str,
        dataset: str,
        max_tokens: int,
    ) -> Path:
        """Save judgements to ``<output_dir>/<judge>/<student>_<dataset>_t<max_tokens>.json`` and return the path."""
        out_dir = self.output_dir / self.model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{student_model}_{dataset}_t{max_tokens}.json"
        out_file = out_dir / filename

        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Saved judgements to %s", out_file.name)
        return out_file

    def cleanup(self, clear_hf_cache: bool = False):
        """Release GPU memory and clean up distributed processes.

        ``clear_hf_cache=True`` also deletes the model's HF cache to free disk space. It defaults to False because
        several processes often share one HF cache, and deleting it could remove weights another process is using.
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

        for key in list(os.environ.keys()):
            if key.startswith(("NCCL_", "RANK", "WORLD")):
                if key not in ("NCCL_DEBUG", "NCCL_IB_DISABLE"):
                    os.environ.pop(key, None)

        if clear_hf_cache and model_name:
            import shutil
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir_name = f"models--{model_name.replace('/', '--')}"
            model_cache = cache_dir / model_dir_name
            if model_cache.exists():
                shutil.rmtree(model_cache, ignore_errors=True)
                logger.info("Cleared HF cache: %s", model_dir_name)

        logger.debug("Cleanup complete")

