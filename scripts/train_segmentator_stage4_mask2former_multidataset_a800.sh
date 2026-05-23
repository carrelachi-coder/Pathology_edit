#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASETS_ROOT="${DATASETS_ROOT:-/data/wqx/flowedit/data}"
UNI2H_REPO_PATH="${UNI2H_REPO:-./UNI-2h}"
MANIFEST_PATH="${SEGMENTATOR_MANIFEST:-segmentator_runs/stage4_multidataset_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-segmentator_runs/stage4_mask2former_multidataset_a800}"

DATASETS=(${SEGMENTATOR_DATASETS:-bcss glas ignite orca panda puma})
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
EPOCHS="${EPOCHS:-20}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
MAX_PER_DATASET_ARGS=()
if [[ -n "${MAX_PER_DATASET:-}" ]]; then
  MAX_PER_DATASET_ARGS+=(--max-per-dataset "$MAX_PER_DATASET")
fi

DISABLE_CUDNN_ARGS=()
if [[ "${SEGMENTATOR_DISABLE_CUDNN:-0}" == "1" ]]; then
  DISABLE_CUDNN_ARGS+=(--disable-cudnn)
fi

mkdir -p "$(dirname "$MANIFEST_PATH")" "$OUTPUT_DIR"

if [[ "${REBUILD_MANIFEST:-0}" == "1" || ! -f "$MANIFEST_PATH" ]]; then
  python scripts/build_segmentator_multidataset_manifest.py \
    --datasets-root "$DATASETS_ROOT" \
    --datasets "${DATASETS[@]}" \
    --val-fraction "$VAL_FRACTION" \
    --output "$MANIFEST_PATH" \
    "${MAX_PER_DATASET_ARGS[@]}"
else
  echo "Using existing manifest: $MANIFEST_PATH"
fi

python -m segmentator.cli \
  --dataset-root "$DATASETS_ROOT" \
  --uni2h-repo "$UNI2H_REPO_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --manifest "$MANIFEST_PATH" \
  --decoder mask2former \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --epochs "$EPOCHS" \
  --balanced-datasets \
  --class-weighting none \
  --no-amp \
  "${DISABLE_CUDNN_ARGS[@]}"
