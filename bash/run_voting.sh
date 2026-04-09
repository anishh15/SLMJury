#!/bin/bash
# ============================================================
#  Run Majority Voting Ensembles
#  Output: results/majority_voting/
# ============================================================

set -e

echo "Generating all C(5,3) = 10 majority voting ensembles..."
python -c "
from slmjury.strategies.ensemble import generate_all_ensembles
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
generate_all_ensembles()
"

echo ""
echo "All majority voting ensembles complete."
