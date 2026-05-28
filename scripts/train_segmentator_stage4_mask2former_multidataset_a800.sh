#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,4}"
if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS="${#VISIBLE_GPU_LIST[@]}"
fi

DATASETS_ROOT="${DATASETS_ROOT:-/data/wqx/flowedit/data}"
UNI2H_REPO_PATH="${UNI2H_REPO:-./UNI-2h}"
MANIFEST_PATH="${SEGMENTATOR_MANIFEST:-segmentator_runs/stage4_multidataset_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-segmentator_runs/stage4_mask2former_multidataset_a800}"

DATASETS=(${SEGMENTATOR_DATASETS:-bcss glas ignite orca panda puma})
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-6}"
if [[ -z "${GRAD_ACCUM_STEPS:-}" ]]; then
  ACCUM_DENOM=$((BATCH_SIZE * NUM_GPUS))
  GRAD_ACCUM_STEPS=$(((TARGET_EFFECTIVE_BATCH_SIZE + ACCUM_DENOM - 1) / ACCUM_DENOM))
  if (( GRAD_ACCUM_STEPS < 1 )); then
    GRAD_ACCUM_STEPS=1
  fi
fi
EPOCHS="${EPOCHS:-20}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
MAX_PER_DATASET_ARGS=()
if [[ -n "${MAX_PER_DATASET:-}" ]]; then
  MAX_PER_DATASET_ARGS+=(--max-per-dataset "$MAX_PER_DATASET")
fi

DISABLE_CUDNN_ARGS=()
if [[ "${SEGMENTATOR_DISABLE_CUDNN:-0}" == "1" ]]; then
  DISABLE_CUDNN_ARGS+=(--disable-cudnn)
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_CHECKPOINT}" != "0" && "${RESUME_FROM_CHECKPOINT}" != "none" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
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

TRAIN_ARGS=(
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
  "${RESUME_ARGS[@]}" \
  "${DISABLE_CUDNN_ARGS[@]}"
)

echo "Launching Mask2Former: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS batch_per_gpu=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS effective_batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS)) resume=${RESUME_FROM_CHECKPOINT:-none}"
if (( NUM_GPUS > 1 )); then
  torchrun --standalone --nnodes=1 --nproc_per_node "$NUM_GPUS" -m segmentator.cli "${TRAIN_ARGS[@]}"
else
  python -m segmentator.cli "${TRAIN_ARGS[@]}"
fi
