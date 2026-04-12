#!/bin/bash
# ============================================================
#  Run Student Model Inference
#  Output: results/student_solutions/
# ============================================================

# --- Configuration ---
GPUS="2,3"
TP_SIZE=2
GPU_MEM_UTIL=0.9

MODELS=(
    "qwen2.5-32b"
    "llama3.1-8b"
)
# ----------------------

set -e

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Student solver: $MODEL"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS python scripts/run_student.py \
        --model "$MODEL" \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL
done

echo ""
echo "All student solutions complete."
