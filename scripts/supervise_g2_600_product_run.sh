#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_ROOT MASK_PID [PHYSICAL_GPU]" >&2
  exit 2
fi

RUN_ROOT=$1
MASK_PID=$2
PHYSICAL_GPU=${3:-1}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PATHOLOGY_PHASE5_PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}

: "${PATHOLOGY_INPAINT_CHECKPOINT:=/home/lyw/wqx-DL/flow-edit/hf_generation_release/pathology-inpaint-controlnet}"
: "${PATHOLOGY_CROSS_V1_CHECKPOINT:=/home/lyw/wqx-DL/flow-edit/hf_generation_release/pathology-cross-v1-pix2pix/cross_v1}"
: "${PATHOLOGY_PIX2PIX_CHECKPOINT:=/home/lyw/wqx-DL/flow-edit/hf_generation_release/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt}"
: "${PATHOLOGY_PROBNET_CHECKPOINT:=/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt}"
: "${PATHOLOGY_CELLVIT_ROOT:=$REPO_ROOT/CellViT-plus-plus-main/CellViT-plus-plus-main}"
: "${PATHOLOGY_CELLVIT_MODEL:=$PATHOLOGY_CELLVIT_ROOT/checkpoints/CellViT-SAM-H-x40-AMP-001.pth}"
: "${PATHOLOGY_CELLVIT_PYTHON:=$PYTHON}"
: "${PATHOLOGY_SEGMENTATOR_CHECKPOINT:=/data1/zhao/wqx/segmentator_fine/legacy_anchor_fine_seed42/best_composite.pt}"
: "${PATHOLOGY_SEGMENTATOR_PYTHON:=/home/lyw/anaconda3/envs/pathology-segmentator-mmseg/bin/python}"
export PATHOLOGY_INPAINT_CHECKPOINT
export PATHOLOGY_CROSS_V1_CHECKPOINT
export PATHOLOGY_PIX2PIX_CHECKPOINT
export PATHOLOGY_PROBNET_CHECKPOINT
export PATHOLOGY_CELLVIT_ROOT
export PATHOLOGY_CELLVIT_MODEL
export PATHOLOGY_CELLVIT_PYTHON
export PATHOLOGY_SEGMENTATOR_CHECKPOINT
export PATHOLOGY_SEGMENTATOR_PYTHON

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "G2 run root does not exist: $RUN_ROOT" >&2
  exit 1
fi
if ! kill -0 "$MASK_PID" 2>/dev/null; then
  echo "Mask process is not running: $MASK_PID" >&2
  exit 1
fi
if [[ -z ${OPENAI_API_KEY:-} ]]; then
  echo "OPENAI_API_KEY is required for alternate generation routes" >&2
  exit 1
fi
test -x "$PATHOLOGY_SEGMENTATOR_PYTHON"

cd "$REPO_ROOT"
"$PYTHON" - <<'PY'
from controlnet_train.inference.model_paths import (
    DEFAULT_CELLVIT_MODEL,
    DEFAULT_CROSS_V1_CHECKPOINT,
    DEFAULT_INPAINT_CHECKPOINT,
    DEFAULT_PIX2PIX_CHECKPOINT,
    DEFAULT_PROBNET_CHECKPOINT,
    validate_frozen_cellvit_checkpoint,
    validate_frozen_probnet_checkpoint,
    validate_production_controlnet_checkpoint,
    validate_production_pix2pix_checkpoint,
)
from segmentator.release import load_segmentator_release

validate_production_controlnet_checkpoint(
    DEFAULT_INPAINT_CHECKPOINT, mode="inpaint"
)
validate_production_controlnet_checkpoint(
    DEFAULT_CROSS_V1_CHECKPOINT, mode="cross-v1"
)
validate_production_pix2pix_checkpoint(DEFAULT_PIX2PIX_CHECKPOINT)
validate_frozen_probnet_checkpoint(DEFAULT_PROBNET_CHECKPOINT)
validate_frozen_cellvit_checkpoint(DEFAULT_CELLVIT_MODEL)
load_segmentator_release(
    "benchmark_configs/releases/segmentator_fine_legacy_anchor.json",
    verify_checkpoint=True,
)
print("frozen_model_preflight=passed")
PY

echo "[$(date -Is)] waiting for mask pid=$MASK_PID"
while kill -0 "$MASK_PID" 2>/dev/null; do
  sleep 60
done

FROZEN_MANIFEST=$RUN_ROOT/g2_600_frozen_product_manifest.json
APPROVED_MASK=$RUN_ROOT/approved_mask_stage_manifest.json
APPROVED_NUCLEI=$RUN_ROOT/approved_nuclei_stage_manifest.json
test -s "$FROZEN_MANIFEST"
test -s "$APPROVED_MASK"

echo "[$(date -Is)] starting nuclei stage on physical GPU $PHYSICAL_GPU"
CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU "$PYTHON" scripts/run_g2_600.py nuclei \
  --manifest "$FROZEN_MANIFEST" \
  --approved-mask-manifest "$APPROVED_MASK" \
  --output "$RUN_ROOT" \
  --expected-count 600
test -s "$APPROVED_NUCLEI"

echo "[$(date -Is)] starting image/evaluator stage on physical GPU $PHYSICAL_GPU"
CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU "$PYTHON" scripts/run_g2_600.py image \
  --manifest "$FROZEN_MANIFEST" \
  --approved-mask-manifest "$APPROVED_MASK" \
  --approved-nuclei-manifest "$APPROVED_NUCLEI" \
  --output "$RUN_ROOT" \
  --expected-count 600

echo "[$(date -Is)] G2-600 product run completed"
