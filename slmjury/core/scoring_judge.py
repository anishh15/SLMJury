"""Scoring judge — evaluates open-ended tasks by assigning numerical scores.

Supports two modes:
- SummEval: 4-dimension single-call scoring (coherence, consistency, fluency, relevance)
- MT-Bench: Single score (1-5) for response quality

Reuses the same vLLM infrastructure as JudgeModel but with scoring prompts
instead of Correct/Incorrect verdict prompts.
"""

import gc
import json
import logging
import re
from pathlib import Path
from typing import Optional

from slmjury.configs import load_models_config
from slmjury.parsers.score import parse_score, parse_multi_score, SUMMEVAL_DIMENSIONS

logger = logging.getLogger(__name__)


# --- SummEval Prompt (4 dimensions, single call) ---

PROMPT_SUMMEVAL = (
    "Evaluate the following summary of a news article on four dimensions.\n"
    "Rate EACH dimension from 1 (very poor) to 5 (excellent).\n\n"
    "Source Article:\n{source_text}\n\n"
    "Summary to Evaluate:\n{summary}\n\n"
    "Dimensions:\n"
    "- Coherence: Is the summary well-organized and easy to follow?\n"
    "- Consistency: Does the summary accurately reflect the source article "
    "without contradictions?\n"
    "- Fluency: Is the summary grammatically correct and well-written?\n"
    "- Relevance: Does the summary capture the important information from "
    "the source?\n\n"
    "Provide your ratings as: "
    "\\boxed{{coherence=X, consistency=X, fluency=X, relevance=X}}"
)


# --- MT-Bench Prompt (single score) ---

PROMPT_MTBENCH = (
    "Rate the quality of the following response on a scale of 1-5.\n"
    "1 = Very poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent.\n\n"
    "[Question]:\n{question}\n\n"
    "[Response]:\n{response}\n\n"
    "Consider helpfulness, accuracy, depth, creativity, and level of detail.\n"
    "Provide your rating as: \\boxed{{SCORE}}"
)


class ScoringJudge:
    """SLM judge for scoring open-ended tasks via vLLM.

    Args:
        model_key: Key from models.yaml judge_models section.
        output_dir: Base directory for saving scoring results.
        model_config: Optional config dict override.
    """

    def __init__(
        self,
        model_key: str,
        output_dir: Path = Path("results/scoring"),
        model_config: Optional[dict] = None,
    ):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.model_key = model_key
        self.output_dir = Path(output_dir)

        config = load_models_config()
        judges = config.get("judge_models", {})
        if model_key not in judges:
            raise ValueError(
                f"Unknown judge model: {model_key}. "
                f"Available: {list(judges.keys())}"
            )
        cfg = dict(judges[model_key])

        # Apply CLI overrides
        if model_config:
            cfg.update(model_config)

        self.model_name = cfg["model"]
        self.enable_thinking = cfg.get("enable_thinking", False)
        self.always_thinks = cfg.get("always_thinks", False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"], trust_remote_code=True,
        )

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

        logger.info("Loading scoring judge: %s", self.model_name)
        self.llm = LLM(**vllm_kwargs)
        logger.info("Scoring judge loaded: %s", self.model_name)

    def score_summeval(
        self,
        summeval_data: list[dict],
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Score SummEval pairs on 4 dimensions.

        Args:
            summeval_data: List of SummEval dicts from data loader.
            max_tokens: Generation limit.
            system_prompt: Optional system prompt (e.g., persona).

        Returns:
            List of scored dicts with judge predictions per dimension.
        """
        from vllm import SamplingParams

        logger.info(
            "Scoring %d SummEval pairs (max_tokens=%d)...",
            len(summeval_data), max_tokens,
        )

        enable_thinking = self.enable_thinking and max_tokens >= 8192

        # Build chat messages
        messages_list = []
        for item in summeval_data:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({
                "role": "user",
                "content": PROMPT_SUMMEVAL.format(
                    source_text=item["source_text"],
                    summary=item["summary"],
                ),
            })
            messages_list.append(msgs)

        prompts = self._apply_chat_template(messages_list, enable_thinking)
        params = self._get_sampling_params(max_tokens, enable_thinking)
        outputs = self.llm.generate(prompts, params)

        results = []
        parse_failures = 0

        for i, (item, output) in enumerate(zip(summeval_data, outputs)):
            response = output.outputs[0].text.strip()
            response = self._strip_thinking(response, enable_thinking)

            scores = parse_multi_score(response)
            if all(v is None for v in scores.values()):
                parse_failures += 1

            results.append({
                "problem_id": item["problem_id"],
                "article_id": item["article_id"],
                "summary_id": item["summary_id"],
                "source_text": item["source_text"],
                "summary": item["summary"],
                "judge_response": response,
                # Ground truth human scores
                "human_coherence": item["coherence"],
                "human_consistency": item["consistency"],
                "human_fluency": item["fluency"],
                "human_relevance": item["relevance"],
                # Judge predicted scores
                "judge_coherence": scores["coherence"],
                "judge_consistency": scores["consistency"],
                "judge_fluency": scores["fluency"],
                "judge_relevance": scores["relevance"],
            })

            if (i + 1) % 200 == 0:
                logger.info("  Scored %d/%d", i + 1, len(summeval_data))

        logger.info(
            "SummEval scoring complete. Parse failures: %d/%d (%.1f%%)",
            parse_failures, len(summeval_data),
            100 * parse_failures / len(summeval_data) if summeval_data else 0,
        )
        return results

    def score_mtbench(
        self,
        mtbench_data: list[dict],
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Score MT-Bench responses on a 1-5 scale.

        Args:
            mtbench_data: List of dicts with 'question' and 'response' keys.
            max_tokens: Generation limit.
            system_prompt: Optional system prompt (e.g., persona).

        Returns:
            List of scored dicts with judge predicted score.
        """
        from vllm import SamplingParams

        logger.info(
            "Scoring %d MT-Bench responses (max_tokens=%d)...",
            len(mtbench_data), max_tokens,
        )

        enable_thinking = self.enable_thinking and max_tokens >= 8192

        messages_list = []
        for item in mtbench_data:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({
                "role": "user",
                "content": PROMPT_MTBENCH.format(
                    question=item["question"],
                    response=item["response"],
                ),
            })
            messages_list.append(msgs)

        prompts = self._apply_chat_template(messages_list, enable_thinking)
        params = self._get_sampling_params(max_tokens, enable_thinking)
        outputs = self.llm.generate(prompts, params)

        results = []
        parse_failures = 0

        for i, (item, output) in enumerate(zip(mtbench_data, outputs)):
            response = output.outputs[0].text.strip()
            response = self._strip_thinking(response, enable_thinking)

            score = parse_score(response)
            if score is None:
                parse_failures += 1

            result = {
                "problem_id": item["problem_id"],
                "question": item["question"],
                "response": item["response"],
                "judge_response": response,
                "judge_score": score,
            }
            # Include oracle score if present
            if "oracle_score" in item:
                result["oracle_score"] = item["oracle_score"]
            if "category" in item:
                result["category"] = item["category"]

            results.append(result)

        logger.info(
            "MT-Bench scoring complete. Parse failures: %d/%d (%.1f%%)",
            parse_failures, len(mtbench_data),
            100 * parse_failures / len(mtbench_data) if mtbench_data else 0,
        )
        return results

    def save_results(
        self,
        results: list[dict],
        dataset: str,
        suffix: str = "",
    ) -> Path:
        """Save scoring results to JSON.

        Args:
            results: List of scored dicts.
            dataset: Dataset name ('summeval' or 'mtbench').
            suffix: Optional filename suffix (e.g., persona name).

        Returns:
            Path to the saved JSON file.
        """
        out_dir = self.output_dir / self.model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        parts = [dataset]
        if suffix:
            parts.append(suffix)
        filename = "_".join(parts) + ".json"
        out_file = out_dir / filename

        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Saved scoring results to %s", out_file.name)
        return out_file

    def cleanup(self):
        """Release GPU memory and optionally clear HF cache."""
        logger.info("Cleaning up scoring judge: %s", self.model_key)
        del self.llm
        del self.tokenizer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Clear HF cache for this model
        import shutil
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            model_safe = self.model_name.replace("/", "--")
            for d in cache_dir.glob(f"models--{model_safe}*"):
                shutil.rmtree(d, ignore_errors=True)
                logger.info("Cleared cache: %s", d.name)

    # --- Internal helpers ---

    def _apply_chat_template(
        self,
        messages_list: list[list[dict]],
        enable_thinking: bool,
    ) -> list[str]:
        """Apply tokenizer chat template to message lists."""
        extra_kwargs = (
            {"enable_thinking": enable_thinking}
            if "qwen3" in self.model_key
            else {}
        )
        return [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                **extra_kwargs,
            )
            for msgs in messages_list
        ]

    def _get_sampling_params(self, max_tokens: int, enable_thinking: bool):
        """Build vLLM SamplingParams."""
        from vllm import SamplingParams

        if enable_thinking:
            return SamplingParams(
                temperature=0.6, top_p=0.95, top_k=20, min_p=0,
                max_tokens=max_tokens,
            )
        return SamplingParams(temperature=0, max_tokens=max_tokens)

    def _strip_thinking(self, response: str, enable_thinking: bool) -> str:
        """Remove <think> tags from reasoning model output."""
        if enable_thinking:
            return re.sub(
                r'<think>.*?</think>', '', response, flags=re.DOTALL,
            ).strip()
        return response
