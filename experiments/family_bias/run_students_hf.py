#!/usr/bin/env python3
"""Generate family-bias student solutions with Hugging Face transformers."""

import argparse
import json
import logging
from pathlib import Path

from slmjury.analysis import paths
from slmjury.configs import load_models_config
from slmjury.core.solver import PROMPT_TEMPLATES, _get_prompt_key
from slmjury.data import load_dataset
from slmjury.parsers.answer import extract_answer, get_dataset_type

NOTES = """\
For students whose architecture the vLLM version used here could not load:
gemma4-31b-student (gemma4), glm-4.7-flash (glm4_moe_lite) and
qwen3.5-35b-student (qwen3_5_moe). These need transformers 5.x. The model is
sharded across all visible GPUs with device_map="auto".

The prompt templates, the routing between them and the answer extraction are
the package's own (slmjury.core.solver), and sampling matches
StudentSolver.solve_batch: temperature 0.7, top_p 0.9, 1,024 new tokens. The
output files have the same fields as the ones run_students.py writes, so the
judge step (scripts/run_judge.py) reads them unchanged.

The student is looked up under student_models in slmjury/configs/models.yaml
(model, dtype and trust_remote_code are used).

Writes results/family_bias/student_solutions/<student>/<dataset>.json

examples, from the repository root:
  python experiments/family_bias/run_students_hf.py --student gemma4-31b-student
  python experiments/family_bias/run_students_hf.py --student qwen3.5-35b-student \\
      --datasets arc_easy arc_challenge gsm8k winogrande truthfulqa
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_DATASETS = ["gsm8k", "gsm_plus", "math", "arc_easy", "arc_challenge",
                "hellaswag", "winogrande", "truthfulqa"]

# Sampling used by StudentSolver.solve_batch for every student.
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9


def student_config(model_key: str) -> dict:
    students = load_models_config().get("student_models", {})
    if model_key not in students:
        raise ValueError(f"Unknown student '{model_key}'. Available: {list(students)}")
    return students[model_key]


def main():
    ap = argparse.ArgumentParser(
        description="Generate family-bias student solutions with transformers.",
        epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    student = ap.add_mutually_exclusive_group(required=True)
    student.add_argument("--student", help="Student model key from student_models in models.yaml")
    # Older spelling, kept working but not shown in --help.
    student.add_argument("--model", dest="student", help=argparse.SUPPRESS)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--num-samples", type=int, default=None, help="Cap problems per dataset")
    ap.add_argument("--output-dir", default=paths.default_dir("family_bias", "student_solutions"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    # Every dataset loads before the model, so a bad download stops the run first.
    datasets = {name: load_dataset(name) for name in args.datasets}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = student_config(args.student)
    model_id = cfg["model"]
    trust = cfg.get("trust_remote_code", False)
    dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16

    logger.info("Loading %s (%s) via transformers device_map=auto ...", args.student, model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="auto", trust_remote_code=trust,
    )
    model.eval()

    out_base = Path(args.output_dir) / args.student
    out_base.mkdir(parents=True, exist_ok=True)

    for dataset_name, problems in datasets.items():
        if args.num_samples:
            problems = problems[: args.num_samples]
        dataset_type = get_dataset_type(dataset_name)
        template = PROMPT_TEMPLATES[_get_prompt_key(dataset_name, dataset_type)]
        logger.info("Solving %d %s problems (batch=%d)...",
                    len(problems), dataset_name, args.batch_size)

        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": template.format(question=p["question"])}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in problems
        ]

        results = []
        for start in range(0, len(problems), args.batch_size):
            bp = problems[start:start + args.batch_size]
            btxt = prompts[start:start + args.batch_size]
            enc = tok(btxt, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, top_p=TOP_P, pad_token_id=tok.pad_token_id,
                )
            for j, problem in enumerate(bp):
                text = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
                results.append({
                    "problem_id": problem["problem_id"],
                    "question": problem["question"],
                    "ground_truth_reasoning": problem["ground_truth_reasoning"],
                    "ground_truth_answer": problem["ground_truth_answer"],
                    "student_reasoning": text,
                    "student_answer": extract_answer(text, dataset_type),
                    "dataset": dataset_name,
                    "model": args.student,
                })
            logger.info("  %s: %d/%d", dataset_name,
                        min(start + args.batch_size, len(problems)), len(problems))

        out_file = out_base / f"{dataset_name}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved %d solutions -> %s", len(results), out_file)

    logger.info("All datasets complete for %s.", args.student)


if __name__ == "__main__":
    main()
