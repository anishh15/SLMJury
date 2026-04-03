"""Multi-Agent Debate (MAD) — 3 SLM judges debate to reach consensus.

Implements RCR (Reflect-Critique-Refine) debate prompting where 3 agents
independently judge each problem, then debate for up to N rounds until
consensus. Unresolved problems fall back to majority vote.

Two variants:
- Variant A: Different models at temperature 0.0 (cross-architecture debate).
- Variant B: Same model at temperatures 0.0, 0.4, 0.9 (diversity via sampling).
"""

import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Optional

from slmjury.configs import load_models_config
from slmjury.parsers.answer import get_dataset_type
from slmjury.parsers.judgement import parse_judgement
from slmjury.strategies.ensemble import majority_vote

logger = logging.getLogger(__name__)

MAX_DEBATE_ROUNDS = 5

# Debate-specific max_num_seqs overrides (lower than standard judge runs
# because MAD keeps multiple agent contexts in GPU memory simultaneously).
_DEBATE_MAX_NUM_SEQS: dict[str, int] = {
    # 14B models: very tight memory
    "phi4-14b": 3, "phi4r-14b": 3, "phi4rp-14b": 3,
    # 7B–8B models
    "qwen2.5-7b": 4, "qwen3-8b": 4, "llama3.1-8b": 4,
    # 3B–4B models
    "qwen2.5-3b": 6, "qwen3-4b": 6, "llama3.2-3b": 6,
    "phi4mi-3.8b": 4, "phi4mr-3.8b": 4,
    # 1.5B–1.7B & smaller
    "qwen2.5-1.5b": 8, "qwen3-1.7b": 8, "qwen3-0.6b": 8,
    "llama3.2-1b": 8, "qwen3-14b": 3,
}

# Reasoned judgement prompt for Round 0 (same as judge.py)
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

# RCR debate prompt for math datasets
DEBATE_PROMPT_MATH = (
    "You are Agent {agent_id} in a multi-agent debate to judge whether "
    "a student's math answer is correct.\n\n"
    "Question: {question}\n\n"
    "Ground truth answer: {ground_truth}\n\n"
    "Student answer: {student_answer}\n\n"
    "{own_previous}\n\n"
    "Here are the judgements from other agents:\n{context}\n\n"
    "This is debate round {round_num}. Please carefully analyze all "
    "judgements—including your own—identify any errors in reasoning, "
    "and provide your revised judgement.\n"
    "- If you believe your previous verdict is correct, explain why and defend it.\n"
    "- If you believe you made an error, explain the error and provide a corrected judgement.\n"
    "- If you believe another agent's verdict is correct, explain why you agree with it.\n\n"
    "Your final verdict must be: \\boxed{{CORRECT}} or \\boxed{{INCORRECT}}"
)

# RCR debate prompt for science datasets
DEBATE_PROMPT_SCIENCE = (
    "You are Agent {agent_id} in a multi-agent debate to judge whether "
    "a student's science answer is correct.\n\n"
    "Question: {question}\n\n"
    "Ground truth answer: {ground_truth}\n\n"
    "Student answer: {student_answer}\n\n"
    "{own_previous}\n\n"
    "Here are the judgements from other agents:\n{context}\n\n"
    "This is debate round {round_num}. Please carefully analyze all "
    "judgements—including your own—identify any misconceptions or flawed "
    "scientific reasoning, and provide your revised judgement.\n"
    "- If you believe your previous verdict is correct, explain the scientific "
    "principles supporting your verdict.\n"
    "- If you believe you made an error, explain the scientific misconception "
    "and provide a corrected judgement.\n"
    "- If you believe another agent's verdict is correct, explain why their "
    "scientific reasoning is sound.\n\n"
    "Your final verdict must be: \\boxed{{CORRECT}} or \\boxed{{INCORRECT}}"
)

# Default MAD combinations
MAD_COMBINATIONS = [
    # Variant A: Different models, same temperature (cross-architecture debate)
    (["llama3.2-3b", "qwen2.5-1.5b", "qwen3-1.7b"], [0, 0, 0]),
    (["qwen2.5-1.5b", "qwen3-1.7b", "phi4mi-3.8b"], [0, 0, 0]),
    (["qwen3-4b", "phi4mi-3.8b", "qwen2.5-3b"], [0, 0, 0]),
    (["qwen3-4b", "qwen2.5-3b", "qwen3-1.7b"], [0, 0, 0]),
    (["llama3.2-3b", "qwen3-4b", "phi4mi-3.8b"], [0, 0, 0]),
    # Variant B: Same model, different temperatures (diversity via sampling)
    (["llama3.2-1b"] * 3, [0, 0.4, 0.9]),
    (["qwen2.5-1.5b"] * 3, [0, 0.4, 0.9]),
    (["qwen3-4b"] * 3, [0, 0.4, 0.9]),
    (["qwen3-0.6b"] * 3, [0, 0.4, 0.9]),
    (["phi4mi-3.8b"] * 3, [0, 0.4, 0.9]),
]

_MP_CTX = mp.get_context("spawn")


def _is_same_model_combo(combo_models: list[str]) -> bool:
    """Check if all 3 agents use the same model (Variant B)."""
    return combo_models[0] == combo_models[1] == combo_models[2]


def get_debate_prompt_template(dataset_name: str) -> str:
    """Return math or science debate prompt based on dataset type."""
    if get_dataset_type(dataset_name) == "multiple_choice":
        return DEBATE_PROMPT_SCIENCE
    return DEBATE_PROMPT_MATH


def build_debate_context(
    debate_state: dict, problem_id, agent_id: int, round_num: int,
) -> tuple[str, str]:
    """Format peer responses from the previous round for the debate prompt.

    Args:
        debate_state: Full debate state dict.
        problem_id: Problem ID to look up.
        agent_id: Current agent's index.
        round_num: Current round number.

    Returns:
        Tuple of (own_previous, context) strings for prompt placeholders.
    """
    prev_round = debate_state[problem_id]["rounds"][round_num - 1]

    own = prev_round[f"agent_{agent_id}"]
    own_previous = (
        f"Your previous verdict: {own['verdict']}\n"
        f"Your previous reasoning: {own['reasoning']}"
    )

    context_parts = []
    for idx in range(3):
        if idx != agent_id:
            peer = prev_round[f"agent_{idx}"]
            context_parts.append(
                f"Agent {idx}'s verdict: {peer['verdict']}\n"
                f"Agent {idx}'s reasoning: {peer['reasoning']}"
            )

    return own_previous, "\n\n".join(context_parts)


def check_consensus(debate_state: dict, round_num: int) -> tuple[list, list]:
    """Check if all 3 agents agree on each unresolved problem.

    Args:
        debate_state: Full debate state dict.
        round_num: Current round number.

    Returns:
        Tuple of (resolved_ids, unresolved_ids).
    """
    resolved_ids = []
    unresolved_ids = []

    for problem_id, state in debate_state.items():
        if state["final_verdict"] is not None:
            continue

        round_data = state["rounds"][round_num]
        verdicts = [round_data[f"agent_{i}"]["verdict"] for i in range(3)]

        if verdicts[0] == verdicts[1] == verdicts[2]:
            state["final_verdict"] = verdicts[0]
            state["resolved_at_round"] = round_num
            resolved_ids.append(problem_id)
        else:
            unresolved_ids.append(problem_id)

    return resolved_ids, unresolved_ids


def make_combo_name(combo_models: list[str], combo_temps: list[float]) -> str:
    """Create a readable combo name for directory/file naming."""
    if _is_same_model_combo(combo_models):
        return "+".join(f"{m}-t{t}" for m, t in zip(combo_models, combo_temps))
    return "+".join(combo_models)


# --- Subprocess isolation for vLLM model loading ---

def _run_in_subprocess(worker_fn, data: dict):
    """Serialize data → spawn worker in a fresh process → return results."""
    input_fd, input_path = tempfile.mkstemp(suffix="_mad_in.pkl")
    _, output_path = tempfile.mkstemp(suffix="_mad_out.pkl")

    try:
        with os.fdopen(input_fd, "wb") as f:
            pickle.dump(data, f)

        logger.debug("Spawning subprocess for %s", worker_fn.__name__)
        p = _MP_CTX.Process(target=worker_fn, args=(input_path, output_path))
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(
                f"Agent subprocess failed (exit code {p.exitcode})."
            )

        with open(output_path, "rb") as f:
            return pickle.load(f)
    finally:
        for path in (input_path, output_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _build_prompts(
    problems: list[dict],
    round_num: int,
    dataset_name: str,
    debate_state: Optional[dict] = None,
    agent_id: Optional[int] = None,
) -> list[str]:
    """Build raw prompt strings for a batch of problems."""
    prompts = []
    for p in problems:
        gt = (
            p["ground_truth_reasoning"]
            if p["ground_truth_reasoning"]
            else p["ground_truth_answer"]
        )

        if round_num == 0:
            prompts.append(PROMPT_REASONED.format(
                question=p["question"],
                ground_truth=gt,
                student_answer=p["student_reasoning"],
            ))
        else:
            own_prev, ctx = build_debate_context(
                debate_state, p["problem_id"], agent_id, round_num,
            )
            prompts.append(get_debate_prompt_template(dataset_name).format(
                agent_id=agent_id,
                question=p["question"],
                ground_truth=gt,
                student_answer=p["student_reasoning"],
                own_previous=own_prev,
                context=ctx,
                round_num=round_num,
            ))
    return prompts


def _run_inference(judge, model_key: str, prompts: list[str], temperature: float) -> list[str]:
    """Run vLLM inference, apply chat template, strip thinking tags."""
    from vllm import SamplingParams

    config = load_models_config()
    model_cfg = config.get("judge_models", {}).get(model_key, {})
    enable_thinking = model_cfg.get("enable_thinking", False)

    extra_kwargs = (
        {"enable_thinking": enable_thinking} if "qwen3" in model_key else {}
    )
    formatted = [
        judge.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, **extra_kwargs,
        )
        for p in prompts
    ]

    if enable_thinking:
        params = SamplingParams(
            temperature=temperature, top_p=0.95, top_k=20, min_p=0,
            max_tokens=8192,
        )
    else:
        params = SamplingParams(temperature=temperature, max_tokens=8192)

    outputs = judge.llm.generate(formatted, params)

    responses = []
    for output in outputs:
        response = output.outputs[0].text.strip()
        if enable_thinking:
            response = re.sub(
                r"<think>.*?</think>", "", response, flags=re.DOTALL,
            ).strip()
        responses.append(response)

    return responses


def _parse_responses(problems: list[dict], responses: list[str]) -> list[dict]:
    """Parse raw responses into structured judge result dicts."""
    results = []
    for problem, response in zip(problems, responses):
        results.append({
            "problem_id": problem["problem_id"],
            "question": problem["question"],
            "ground_truth_reasoning": problem["ground_truth_reasoning"],
            "ground_truth_answer": problem["ground_truth_answer"],
            "student_reasoning": problem["student_reasoning"],
            "student_answer": problem["student_answer"],
            "judge_response": response,
            "judgement": parse_judgement(response),
        })
    return results


def _worker_single_agent(input_path: str, output_path: str):
    """Subprocess entry-point: load one model, run inference for one agent, exit."""
    import pickle
    import warnings
    warnings.filterwarnings("ignore")

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    from slmjury.core.judge import JudgeModel

    model_key = data["model_key"]
    cfg_override = data.get("cfg_override")
    judge = JudgeModel(model_key, model_config=cfg_override)

    prompts = _build_prompts(
        data["problems"], data["round_num"], data["dataset_name"],
        data.get("debate_state"), data.get("agent_id"),
    )
    responses = _run_inference(judge, model_key, prompts, data["temperature"])
    results = _parse_responses(data["problems"], responses)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    judge.cleanup()


def _worker_same_model_agents(input_path: str, output_path: str):
    """Subprocess entry-point for Variant B: load one model, run 3 temps, exit."""
    import pickle
    import warnings
    warnings.filterwarnings("ignore")

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    from slmjury.core.judge import JudgeModel

    model_key = data["model_key"]
    cfg_override = data.get("cfg_override")
    judge = JudgeModel(model_key, model_config=cfg_override)

    all_results = []
    for agent_idx, temp in enumerate(data["temperatures"]):
        prompts = _build_prompts(
            data["problems"], data["round_num"], data["dataset_name"],
            data.get("debate_state"), agent_idx,
        )
        responses = _run_inference(judge, model_key, prompts, temp)
        all_results.append(_parse_responses(data["problems"], responses))

    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)

    judge.cleanup()


def run_agent_turn(
    model_key: str,
    temperature: float,
    problems: list[dict],
    round_num: int,
    dataset_name: str,
    debate_state: Optional[dict] = None,
    agent_id: Optional[int] = None,
    tp_size_override: Optional[int] = None,
) -> list[dict]:
    """Run one agent's inference in an isolated subprocess.

    Args:
        model_key: Judge model key.
        temperature: Sampling temperature.
        problems: List of problem dicts.
        round_num: Current debate round.
        dataset_name: Dataset name for prompt template selection.
        debate_state: Current debate state (for rounds > 0).
        agent_id: Agent index (0, 1, or 2).
        tp_size_override: Optional tensor_parallel_size override.

    Returns:
        List of judgement result dicts.
    """
    logger.info(
        "Agent %s (%s, temp=%.1f): %d problems, round %d",
        agent_id, model_key, temperature, len(problems), round_num,
    )

    config = load_models_config()
    cfg_override = dict(config.get("judge_models", {}).get(model_key, {}))
    if tp_size_override is not None:
        cfg_override["tensor_parallel_size"] = tp_size_override
    cfg_override["max_num_seqs"] = _DEBATE_MAX_NUM_SEQS.get(model_key, 6)

    return _run_in_subprocess(_worker_single_agent, {
        "model_key": model_key,
        "temperature": temperature,
        "problems": problems,
        "round_num": round_num,
        "dataset_name": dataset_name,
        "debate_state": debate_state,
        "agent_id": agent_id,
        "cfg_override": cfg_override,
    })


def run_agent_turn_same_model(
    model_key: str,
    temperatures: list[float],
    problems: list[dict],
    round_num: int,
    dataset_name: str,
    debate_state: Optional[dict] = None,
    tp_size_override: Optional[int] = None,
) -> list[list[dict]]:
    """Run 3 same-model agents (Variant B) in an isolated subprocess.

    Args:
        model_key: Judge model key (same for all 3 agents).
        temperatures: List of 3 temperatures.
        problems: List of problem dicts.
        round_num: Current debate round.
        dataset_name: Dataset name.
        debate_state: Current debate state.
        tp_size_override: Optional tensor_parallel_size override.

    Returns:
        List of 3 lists of judgement result dicts.
    """
    logger.info(
        "Same-model %s × 3 temps %s: %d problems, round %d",
        model_key, temperatures, len(problems), round_num,
    )

    config = load_models_config()
    cfg_override = dict(config.get("judge_models", {}).get(model_key, {}))
    if tp_size_override is not None:
        cfg_override["tensor_parallel_size"] = tp_size_override
    cfg_override["max_num_seqs"] = _DEBATE_MAX_NUM_SEQS.get(model_key, 6)

    return _run_in_subprocess(_worker_same_model_agents, {
        "model_key": model_key,
        "temperatures": temperatures,
        "problems": problems,
        "round_num": round_num,
        "dataset_name": dataset_name,
        "debate_state": debate_state,
        "cfg_override": cfg_override,
    })


def run_debate(
    combo_models: list[str],
    combo_temps: list[float],
    student_results: list[dict],
    dataset_name: str,
    tp_size_override: Optional[int] = None,
) -> dict:
    """Run the full debate loop for one combination.

    Args:
        combo_models: List of 3 model keys.
        combo_temps: List of 3 temperatures.
        student_results: List of student solution dicts.
        dataset_name: Dataset name.
        tp_size_override: Optional tensor_parallel_size override.

    Returns:
        Debate state dict with final verdicts for all problems.
    """
    same_model = _is_same_model_combo(combo_models)

    debate_state = {}
    for problem in student_results:
        debate_state[problem["problem_id"]] = {
            "student_result": problem,
            "rounds": [],
            "final_verdict": None,
            "resolved_at_round": None,
        }

    unresolved_ids = list(debate_state.keys())

    for round_num in range(MAX_DEBATE_ROUNDS + 1):
        if not unresolved_ids:
            break

        logger.info("Round %d: %d problems to judge", round_num, len(unresolved_ids))

        current_problems = (
            student_results if round_num == 0
            else [debate_state[pid]["student_result"] for pid in unresolved_ids]
        )

        for pid in unresolved_ids:
            debate_state[pid]["rounds"].append({})

        if same_model:
            all_agent_results = run_agent_turn_same_model(
                combo_models[0], combo_temps, current_problems,
                round_num, dataset_name, debate_state, tp_size_override,
            )
            for agent_idx, agent_results in enumerate(all_agent_results):
                for result in agent_results:
                    pid = result["problem_id"]
                    debate_state[pid]["rounds"][round_num][f"agent_{agent_idx}"] = {
                        "verdict": result["judgement"],
                        "reasoning": result["judge_response"],
                    }
        else:
            for agent_idx in range(3):
                agent_results = run_agent_turn(
                    combo_models[agent_idx], combo_temps[agent_idx],
                    current_problems, round_num, dataset_name,
                    debate_state, agent_idx, tp_size_override,
                )
                for result in agent_results:
                    pid = result["problem_id"]
                    debate_state[pid]["rounds"][round_num][f"agent_{agent_idx}"] = {
                        "verdict": result["judgement"],
                        "reasoning": result["judge_response"],
                    }

        resolved_ids, unresolved_ids = check_consensus(debate_state, round_num)
        logger.info(
            "Round %d: %d resolved, %d still debating",
            round_num, len(resolved_ids), len(unresolved_ids),
        )

    # Majority vote fallback
    for pid in unresolved_ids:
        last_round = debate_state[pid]["rounds"][-1]
        verdicts = [last_round[f"agent_{i}"]["verdict"] for i in range(3)]
        debate_state[pid]["final_verdict"] = majority_vote(verdicts)
        debate_state[pid]["resolved_at_round"] = "majority_vote"

    if unresolved_ids:
        logger.info("Majority vote fallback: %d problems", len(unresolved_ids))

    return debate_state


def save_debate_results(
    combo_name: str,
    debate_state: dict,
    student_model: str,
    dataset: str,
    output_dir: Path = Path("results/multi_agent_debate"),
) -> Path:
    """Convert debate state to final results and save to JSON.

    Args:
        combo_name: Readable combo name for subdirectory.
        debate_state: Debate state dict from run_debate.
        student_model: Student model key.
        dataset: Dataset name.
        output_dir: Base output directory.

    Returns:
        Path to the saved JSON file.
    """
    final_results = []
    for pid, state in debate_state.items():
        result = {**state["student_result"]}
        result["judgement"] = state["final_verdict"]
        result["resolved_at_round"] = state["resolved_at_round"]
        result["debate_rounds"] = state["rounds"]
        final_results.append(result)

    out_dir = output_dir / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{combo_name}_{student_model}_{dataset}.json"
    out_file = out_dir / filename

    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=4)

    logger.info("Saved debate results to %s", out_file.name)
    return out_file
