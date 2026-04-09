#!/bin/bash
# ============================================================
#  Run Persona Effect Evaluation
#  Output: results/persona_effect/
# ============================================================

# --- Configuration ---
GPUS="2,3"
TP_SIZE=2
GPU_MEM_UTIL=0.9

MODELS=(
    "qwen3-4b"
    "phi4mi-3.8b"
    "llama3.1-8b"
    "qwen3-14b"
    "phi4-14b"
)
# ----------------------

set -e

TOTAL=${#MODELS[@]}
COUNT=0

for MODEL in "${MODELS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "============================================================"
    echo "  [$COUNT/$TOTAL] Persona: $MODEL"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS python scripts/run_personas.py \
        --judge "$MODEL" \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL
done

echo ""
echo "All persona evaluations complete."
