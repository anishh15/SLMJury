#!/bin/bash
# ============================================================
#  Run Scoring Evaluation for Open-Ended Datasets
#  Output: results/scoring/, results/scoring_evaluations/
# ============================================================

# --- Configuration ---
GPUS="2,3"
TP_SIZE=2
GPU_MEM_UTIL=0.9
DATASET="summeval"      # summeval or mtbench

MODELS=(
    "qwen2.5-1.5b"
    "qwen2.5-3b"
    "qwen2.5-7b"
    "qwen3-0.6b"
    "qwen3-1.7b"
    "qwen3-4b"
    "qwen3-8b"
    "qwen3-14b"
    "llama3.2-1b"
    "llama3.2-3b"
    "llama3.1-8b"
    "phi4-14b"
    "phi4r-14b"
    "phi4rp-14b"
    "phi4mi-3.8b"
    "phi4mr-3.8b"
)
# ----------------------

set -e

TOTAL=${#MODELS[@]}
COUNT=0

for MODEL in "${MODELS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "============================================================"
    echo "  [$COUNT/$TOTAL] Scoring: $MODEL × $DATASET"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS python scripts/run_scoring_judge.py \
        --judge "$MODEL" \
        --dataset "$DATASET" \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL
done

echo ""
echo "All scoring evaluations complete."
