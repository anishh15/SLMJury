#!/bin/bash
# ============================================================
#  Run Multi-Agent Debate (MAD)
#  Output: results/multi_agent_debate/
# ============================================================

# --- Configuration ---
GPUS="2,3"
TP_SIZE=1
GPU_MEM_UTIL=0.8

# Combo indices to run (0-9, or "all")
COMBOS=(0 1 2 3 4 5 6 7 8 9)
# ----------------------

set -e

TOTAL=${#COMBOS[@]}
COUNT=0

for IDX in "${COMBOS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "============================================================"
    echo "  [$COUNT/$TOTAL] MAD combo index: $IDX"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS python scripts/run_mad.py \
        --combo-index "$IDX" \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL
done

echo ""
echo "All debate experiments complete."
