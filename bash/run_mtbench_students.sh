#!/bin/bash
# ============================================================
#  Run MT-Bench Student Responses (2 turns)
#  Output: results/mtbench_responses/
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
    echo "  MT-Bench Student: $MODEL (2 turns)"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS python scripts/run_mtbench_student.py \
        --model "$MODEL" \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL
done

echo ""
echo "All MT-Bench student responses complete."
