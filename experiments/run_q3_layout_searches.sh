#!/usr/bin/env bash
# Run the Q3 layout searches in order. Each step is resumable: if this stops for any
# reason, run the same command again and every script continues from its checkpoints.
#   bash experiments/run_q3_layout_searches.sh            (about 2-2.5 hours on 2 cores)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/q3_structured_layouts.py
python experiments/q3_adjoint_search.py
python experiments/q3_seeded_search.py
echo "All Q3 layout searches done."
