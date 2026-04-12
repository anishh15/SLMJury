#!/bin/bash
# ============================================================
#  Run Evaluation + Summary Generation
#  Output: results/evaluations/, results/summaries/
# ============================================================

set -e

echo "Running evaluations with summary generation..."
python scripts/run_evaluation.py --summary

echo ""
echo "All evaluations and summaries complete."
