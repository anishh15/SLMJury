#!/usr/bin/env python3
"""Scaling and cost: accuracy against serving cost, and the cost-quality Pareto front."""

import argparse
import csv
import json

from slmjury.analysis import paths
from slmjury.analysis.metrics import pareto_front
from slmjury.configs import load_models_config

NOTES = """\
Accuracy is read, not recomputed, so this analysis and the large-judge
analysis always agree:
  small judges           results/core/summaries/individual/<judge>_t<budget>.json
  large/ladder judges    results/analysis/large_judges/small_vs_large.csv
                         (run experiments/large_judges/analyze.py first)

Cost model. Serving cost per judgement:

  $/judgement = (avg_tokens / throughput_tok_per_s) / 3600
                x n_gpus x $0.55 per GPU-hour

The same $0.55 per GPU-hour is applied to the small judges' 32 GB V100s and
the large judges' 44 GiB cards. n_gpus is each judge's tensor_parallel_size in
slmjury/configs/models.yaml (judge_models and large_judge_models), the GPU
count the cost model assumes for 16-bit serving. Under it the large judges
take 4 or 8 GPUs and most small judges 1 or 2, so the GPU count matters for
the cost ratios.

Writes results/analysis/scaling_cost/:
  scaling_pareto.csv           judge x budget -> accuracy, n_gpus, cost per 1k judgements
  pareto_front_t10.json        Pareto-optimal judges at 10 tokens
  pareto_front_t8192.json      Pareto-optimal judges at 8,192 tokens
  cost_ratio.csv               cost multiple and accuracy difference vs. phi4-14b at 10 tokens

Run it from the repository root.
"""

OUT_NAME = "scaling_cost"

# ── Cost-model constants ────────────────────────────────────────────────────
USD_PER_GPU_HOUR = 0.55
PROMPT_TOKENS = 200                       # shared judging prompt
AVG_TOKENS = {10: PROMPT_TOKENS + 10, 8192: PROMPT_TOKENS + 8192}

# Decode throughput (tokens/s) comes from one rule applied to every judge:
#
#     seconds per token = T_FIXED + T_PER_B * params_b
#
# a fixed per-token cost (scheduling, attention, tensor-parallel collectives)
# plus a term linear in total parameters, which bounds a weight-streaming
# decode. MoE judges are charged on total rather than active parameters,
# because every expert is resident and streamed.
#
# The two constants were fit to 17 per-model throughput values, minimising
# relative error so the fit holds across three orders of magnitude of model
# size. It is within 5.5% of those values on average and 20.5% at worst
# (GPT-OSS-120B).
#
# Judges served with max_num_seqs=3 instead of 32/64 get a batch penalty,
# fit the same way from the three Phi reasoning judges.
T_FIXED = 582e-6          # s/token
T_PER_B = 341e-6          # s/token per billion total parameters
SMALL_BATCH_PENALTY = 0.57
SMALL_BATCH_JUDGES = {"phi4r-14b", "phi4rp-14b", "phi4mr-3.8b"}

MODEL_PARAMS = {
    "qwen3-0.6b": 0.6, "qwen2.5-1.5b": 1.5, "qwen3-1.7b": 1.7,
    "llama3.2-1b": 1.0, "llama3.2-3b": 3.0, "qwen2.5-3b": 3.0,
    "qwen3-4b": 4.0, "phi4mr-3.8b": 3.8, "phi4mi-3.8b": 3.8,
    "qwen2.5-7b": 7.0, "qwen3-8b": 8.0, "llama3.1-8b": 8.0,
    "phi4-14b": 14.0, "phi4r-14b": 14.0, "phi4rp-14b": 14.0, "qwen3-14b": 14.0,
    "llama3.3-70b": 70.0, "gpt-oss-120b": 120.0, "qwen2.5-72b": 72.0,
    "deepseek-r1-distill-llama-70b": 70.0,
    "qwen3.5-122b": 122.0, "glm-4.5-air": 106.0,
    "qwen3-32b": 32.0,                       # ladder judge
}

# Filled by load_large_accuracy(). Ladder judges are priced like every other
# judge but keep their own type label rather than counting as large judges.
LADDER_MODELS: set[str] = set()

# Judges that cannot give a verdict within 10 tokens.
ALWAYS_THINKING = {"phi4r-14b", "phi4rp-14b", "phi4mr-3.8b", "gpt-oss-120b",
                   "deepseek-r1-distill-llama-70b", "glm-4.5-air"}

# Reference judge for the cost-multiple table.
REFERENCE_JUDGE = "phi4-14b"


def throughput(model: str) -> float | None:
    """Decode tokens/s for one judge, from the rule above."""
    p = MODEL_PARAMS.get(model)
    if p is None:
        return None
    tps = 1.0 / (T_FIXED + T_PER_B * p)
    if model in SMALL_BATCH_JUDGES:
        tps *= SMALL_BATCH_PENALTY
    return tps


def load_gpu_counts() -> dict[str, int]:
    """tensor_parallel_size per judge, from the model configuration."""
    cfg = load_models_config()
    counts = {}
    for section in ("judge_models", "large_judge_models"):
        for key, spec in (cfg.get(section) or {}).items():
            if isinstance(spec, dict) and spec.get("tensor_parallel_size"):
                counts[key] = int(spec["tensor_parallel_size"])
    return counts


def cost_per_1k(model: str, max_tokens: int, n_gpus: int) -> float | None:
    tps = throughput(model)
    if tps is None or n_gpus is None:
        return None
    hours = (AVG_TOKENS[max_tokens] / tps) / 3600
    return round(hours * n_gpus * USD_PER_GPU_HOUR * 1000, 6)


def load_slm_accuracy(model: str, max_tokens: int) -> float | None:
    path = paths.core("summaries") / "individual" / f"{model}_t{max_tokens}.json"
    if not path.exists():
        return None
    with open(path) as f:
        s = json.load(f)
    total = s.get("total_problems", 0)
    correct = s["total_correct"]
    return round(correct / total * 100, 4) if total else None


def load_large_accuracy() -> dict[str, dict[int, float]]:
    """Large and ladder judge accuracy from the large-judge analysis table."""
    table = paths.results_dir() / "analysis" / "large_judges" / "small_vs_large.csv"
    if not table.exists():
        raise SystemExit(
            f"{table} not found. Run experiments/large_judges/analyze.py first; "
            "this analysis reads its accuracies rather than recomputing them."
        )
    out = {}
    with open(table) as f:
        for row in csv.DictReader(f):
            if row["type"] not in ("Large", "Ladder"):
                continue
            # A run whose verdicts never parsed is a failed run, not a cheap
            # point on the frontier.
            if row.get("usable_ifr", "True") == "False":
                continue
            entry = {}
            for tok in (10, 8192):
                v = row.get(f"acc_t{tok}")
                if v not in (None, ""):
                    entry[tok] = float(v)
            if entry:
                out[row["judge"]] = entry
                if row["type"] == "Ladder":
                    LADDER_MODELS.add(row["judge"])
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Scaling curve and cost-quality Pareto front", epilog=NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-large", action="store_true",
                        help="small judges only; skip the large-judge rows")
    args = parser.parse_args()

    out_dir = paths.analysis_dir(OUT_NAME)
    gpus = load_gpu_counts()
    print("GPU counts from models.yaml (tensor_parallel_size):")
    for model in sorted(gpus, key=lambda m: MODEL_PARAMS.get(m, 0)):
        if model in MODEL_PARAMS:
            print(f"  {model:<32}{MODEL_PARAMS[model]:>6.1f}B  ->  {gpus[model]} GPU(s)")
    missing = [m for m in MODEL_PARAMS if m not in gpus]
    if missing:
        print(f"  (no config entry, cost omitted: {', '.join(sorted(missing))})")

    large_acc = {} if args.no_large else load_large_accuracy()

    rows = []
    for model, params in MODEL_PARAMS.items():
        is_large = model in large_acc
        if model not in gpus:
            continue
        if not is_large and model in ("llama3.3-70b", "gpt-oss-120b", "qwen2.5-72b",
                                      "deepseek-r1-distill-llama-70b"):
            continue  # large judge without usable accuracy
        for max_tokens in (10, 8192):
            if max_tokens == 10 and model in ALWAYS_THINKING:
                continue
            acc = (large_acc[model].get(max_tokens) if is_large
                   else load_slm_accuracy(model, max_tokens))
            rows.append({
                "model": model,
                "params_b": params,
                "type": ("Ladder" if model in LADDER_MODELS
                         else "Large" if is_large else "SLM"),
                "max_tokens": max_tokens,
                "accuracy": acc,
                "n_gpus": gpus[model],
                "cost_per_1k_usd": cost_per_1k(model, max_tokens, gpus[model]),
                "provider": "vllm",
            })

    # ── Pareto fronts ──────────────────────────────────────────────────────
    for max_tokens in (10, 8192):
        pts = [(r["cost_per_1k_usd"], r["accuracy"], r["model"]) for r in rows
               if r["max_tokens"] == max_tokens
               and r["cost_per_1k_usd"] is not None and r["accuracy"] is not None]
        front = pareto_front(pts)
        print(f"\nPareto front (t{max_tokens}):")
        for cost, acc, model in front:
            print(f"  {model:<24} ${cost:>10.4f}/1k   {acc:.2f}%")
        with open(out_dir / f"pareto_front_t{max_tokens}.json", "w") as f:
            json.dump([{"model": m, "cost": c, "accuracy": a} for c, a, m in front],
                      f, indent=4)

    # ── Scaling table ──────────────────────────────────────────────────────
    print("\nScaling table:")
    print(f"  {'Model':<32}{'Params':>8}{'GPUs':>6}{'Acc@t10':>10}{'$@t10':>10}"
          f"{'Acc@t8192':>11}{'$@t8192':>11}")
    print("  " + "-" * 88)
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["max_tokens"]] = r
    for model in sorted(by_model, key=lambda m: MODEL_PARAMS.get(m, 0)):
        d = by_model[model]
        t10, t8k = d.get(10, {}), d.get(8192, {})

        def cell(src, key, fmt, width):
            v = src.get(key)
            return f"{(fmt.format(v) if v is not None else '-'):>{width}}"

        print(f"  {model:<32}{MODEL_PARAMS[model]:>7.1f}B"
              f"{(t10 or t8k).get('n_gpus', '-'):>6}"
              + cell(t10, "accuracy", "{:.2f}", 10)
              + cell(t10, "cost_per_1k_usd", "${:.3f}", 10)
              + cell(t8k, "accuracy", "{:.2f}", 11)
              + cell(t8k, "cost_per_1k_usd", "${:.3f}", 11))

    out_csv = out_dir / "scaling_pareto.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "params_b", "type", "max_tokens", "accuracy",
                           "n_gpus", "cost_per_1k_usd", "provider"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")

    # ── Cost multiples versus the reference judge ──────────────────────────
    ref = by_model.get(REFERENCE_JUDGE, {}).get(10)
    ratio_rows = []
    if ref and ref["cost_per_1k_usd"]:
        print(f"\nCost relative to {REFERENCE_JUDGE} at B=10 "
              f"(${ref['cost_per_1k_usd']:.3f}/1k, {ref['accuracy']:.2f}%):")
        for model in sorted(by_model, key=lambda m: MODEL_PARAMS.get(m, 0)):
            for tok in (10, 8192):
                r = by_model[model].get(tok)
                if not r or r["cost_per_1k_usd"] is None or r["accuracy"] is None:
                    continue
                mult = r["cost_per_1k_usd"] / ref["cost_per_1k_usd"]
                ratio_rows.append({
                    "model": model, "type": r["type"], "params_b": r["params_b"],
                    "max_tokens": tok, "accuracy": r["accuracy"],
                    "cost_per_1k_usd": r["cost_per_1k_usd"],
                    "cost_multiple_vs_ref": round(mult, 2),
                    "accuracy_delta_vs_ref": round(r["accuracy"] - ref["accuracy"], 2),
                })
                if r["type"] == "Large":
                    print(f"  {model:<32} t{tok:<5} ${r['cost_per_1k_usd']:>9.3f}/1k  "
                          f"{mult:>7.1f}x cost   {r['accuracy'] - ref['accuracy']:+.2f} pp accuracy")
        ratio_csv = out_dir / "cost_ratio.csv"
        with open(ratio_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ratio_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ratio_rows)
        print(f"\nSaved: {ratio_csv}")


if __name__ == "__main__":
    main()
