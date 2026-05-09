#!/bin/bash
# ============================================================
#  Run MT-Bench Oracle Scoring (vLLM)
#  Output: results/mtbench_oracle/{oracle}/{student}.json
# ============================================================

# --- Configuration ---
GPUS="0,1,2,3"
TP_SIZE=4
GPU_MEM_UTIL=0.9

ORACLES=(
    "gpt-oss-120b"
    "qwen3.5-397b"
)
# ----------------------

set -e

for ORACLE in "${ORACLES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  MT-Bench Oracle: $ORACLE"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS python scripts/run_mtbench_oracle.py \
        --oracle "$ORACLE" \
        --student-responses results/mtbench_responses/ \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL
done

echo ""
echo "All MT-Bench oracle scoring complete."
