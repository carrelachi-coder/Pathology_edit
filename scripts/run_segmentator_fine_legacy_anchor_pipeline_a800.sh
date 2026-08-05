#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS:-0,2}"
FINE_OUTPUT_DIR="${FINE_OUTPUT_DIR:-/data1/zhao/wqx/segmentator_fine/legacy_anchor_fine_seed42}"
JOINT_OUTPUT_DIR="${JOINT_OUTPUT_DIR:-/data1/zhao/wqx/segmentator_fine/legacy_anchor_joint_seed42}"
PIPELINE_STATE_DIR="${PIPELINE_STATE_DIR:-/data1/zhao/wqx/segmentator_fine/legacy_anchor_pipeline_seed42}"
FINE_CHECKPOINT="${FINE_CHECKPOINT:-$FINE_OUTPUT_DIR/best_composite.pt}"

mkdir -p "$PIPELINE_STATE_DIR"

echo "[$(date --iso-8601=seconds)] stage=fine status=starting" | tee "$PIPELINE_STATE_DIR/status.txt"
GPU_IDS="$GPU_IDS" \
OUTPUT_DIR="$FINE_OUTPUT_DIR" \
bash scripts/train_segmentator_fine_legacy_anchor_a800.sh

if [[ ! -f "$FINE_CHECKPOINT" ]]; then
  echo "Fine stage completed without the expected checkpoint: $FINE_CHECKPOINT" >&2
  exit 2
fi
echo "[$(date --iso-8601=seconds)] stage=fine status=complete checkpoint=$FINE_CHECKPOINT" \
  | tee "$PIPELINE_STATE_DIR/status.txt"

echo "[$(date --iso-8601=seconds)] stage=joint status=starting" | tee "$PIPELINE_STATE_DIR/status.txt"
GPU_IDS="$GPU_IDS" \
OUTPUT_DIR="$JOINT_OUTPUT_DIR" \
FINE_CHECKPOINT="$FINE_CHECKPOINT" \
bash scripts/train_segmentator_fine_legacy_anchor_joint_a800.sh

echo "[$(date --iso-8601=seconds)] stage=joint status=complete" | tee "$PIPELINE_STATE_DIR/status.txt"
