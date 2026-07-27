#!/usr/bin/env bash
set -euo pipefail

SEEDS=(${SEEDS:-42 43 44})
STAGES=(${STAGES:-baseline training refine teacher input})

for seed in "${SEEDS[@]}"; do
  for stage in "${STAGES[@]}"; do
    echo "Starting segmentator ablation stage=$stage seed=$seed"
    STAGE="$stage" SEED="$seed" bash scripts/run_segmentator_epoch7_ablation.sh
  done
done
