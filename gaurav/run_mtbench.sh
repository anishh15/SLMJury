#!/bin/bash
# ============================================================
#  Gaurav's Unified MT-Bench Pipeline
#
#  Runs everything needed for MT-Bench locally:
#    1. Student responses (2 turns) for both student models
#    2. Oracle scoring for both oracle models
#
#  Output:
#    results/mtbench_responses/{student}.json
#    results/mtbench_oracle/{oracle}/{student}.json
#
#  Usage:
#    CUDA_VISIBLE_DEVICES=0,1,2,3 bash gaurav/run_mtbench.sh
# ============================================================

# --- Configuration ---
TP_SIZE=4
GPU_MEM_UTIL=0.9

STUDENT_MODELS=(
    "qwen2.5-32b"
    "llama3.1-8b"
)

ORACLE_MODELS=(
    "gpt-oss-120b"
    "qwen3-235b"
)
# ----------------------

set -e

echo "============================================================"
echo "  MT-Bench Pipeline — Gaurav"
echo "  Students: ${STUDENT_MODELS[*]}"
echo "  Oracles:  ${ORACLE_MODELS[*]}"
echo "  TP: $TP_SIZE  |  GPU Mem: $GPU_MEM_UTIL"
echo "============================================================"

# ── Step 1: Generate student responses ──
echo ""
echo ">>> STEP 1: Student Responses"
echo ""

for MODEL in "${STUDENT_MODELS[@]}"; do
    echo "------------------------------------------------------------"
    echo "  Student: $MODEL (2 turns)"
    echo "------------------------------------------------------------"
    python scripts/run_mtbench_student.py \
        --model "$MODEL" \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL
done

echo ""
echo ">>> Student responses complete."
echo ""

# ── Step 2: Oracle scoring ──
echo ">>> STEP 2: Oracle Scoring"
echo ""

for ORACLE in "${ORACLE_MODELS[@]}"; do
    echo "------------------------------------------------------------"
    echo "  Oracle: $ORACLE"
    echo "------------------------------------------------------------"
    python scripts/run_mtbench_oracle.py \
        --oracle "$ORACLE" \
        --student-responses results/mtbench_responses/ \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $GPU_MEM_UTIL
done

echo ""
echo "============================================================"
echo "  MT-Bench Pipeline Complete!"
echo ""
echo "  Student responses: results/mtbench_responses/"
echo "  Oracle scores:     results/mtbench_oracle/"
echo "============================================================"
