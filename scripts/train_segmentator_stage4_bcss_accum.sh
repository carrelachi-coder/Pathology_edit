#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_ROOT="${BCSS_PATCHES_ROOT:-/data/wqx/flowedit/data/BCSS_PATCHES}"
UNI2H_REPO_PATH="${UNI2H_REPO:-./UNI-2h}"
DISABLE_CUDNN_ARGS=()
if [[ "${SEGMENTATOR_DISABLE_CUDNN:-0}" == "1" ]]; then
  DISABLE_CUDNN_ARGS+=(--disable-cudnn)
fi

python -m segmentator.cli \
  --dataset-root "$DATASET_ROOT" \
  --uni2h-repo "$UNI2H_REPO_PATH" \
  --output-dir "segmentator_runs/stage4_bcss_1000_200_accum" \
  --image-size 512 \
  --batch-size 1 \
  --grad-accum-steps 2 \
  --train-split 1000 \
  --val-split 200 \
  --epochs 20 \
  --class-weighting none \
  --amp \
  "${DISABLE_CUDNN_ARGS[@]}"
