#!/bin/bash
# ============================================================
#  Run MT-Bench Oracle Scoring (Together API)
#  Output: results/mtbench_oracle/{oracle}/{student}.json
#
#  Requires: TOGETHER_API_KEY environment variable
# ============================================================

ORACLES=(
    "gpt-oss-120b"
    "qwen3.5-397b"
)
# ----------------------

set -e

for ORACLE in "${ORACLES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  MT-Bench Oracle: $ORACLE (Together API)"
    echo "============================================================"
    python scripts/run_mtbench_oracle.py \
        --oracle "$ORACLE" \
        --student-responses results/mtbench_responses/
done

echo ""
echo "All MT-Bench oracle scoring complete."
