#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS:-1,2,4}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS="${#VISIBLE_GPU_LIST[@]}"
fi

DATASETS_ROOT="${DATASETS_ROOT:-/data/wqx/flowedit/data}"
UNI2H_REPO_PATH="${UNI2H_REPO:-./UNI-2h}"
MANIFEST_PATH="${SEGMENTATOR_MANIFEST:-segmentator_runs/stage4_multidataset_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-segmentator_runs/stage4_mask2former_multidataset_a800}"
LABEL_SPACE_SUMMARY="${LABEL_SPACE_SUMMARY:-}"

DATASETS=(${SEGMENTATOR_DATASETS:-bcss glas ignite orca panda puma})
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
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
TEST_FRACTION="${TEST_FRACTION:-0.0}"
SEED="${SEED:-42}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-none}"
DATASET_SAMPLING_TEMPERATURE="${DATASET_SAMPLING_TEMPERATURE:-0.0}"
RARE_CLASS_SAMPLING="${RARE_CLASS_SAMPLING:-0}"
RARE_CLASS_SAMPLE_BOOST="${RARE_CLASS_SAMPLE_BOOST:-2.0}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
BACKBONE_UNFREEZE_EPOCH="${BACKBONE_UNFREEZE_EPOCH:--1}"
BACKBONE_UNFREEZE_BLOCKS="${BACKBONE_UNFREEZE_BLOCKS:-0}"
MIN_FREE_GPU_MEMORY_GB_BEFORE_UNFREEZE="${MIN_FREE_GPU_MEMORY_GB_BEFORE_UNFREEZE:-0}"
GPU_MEMORY_POLL_SECONDS="${GPU_MEMORY_POLL_SECONDS:-60}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-4}"
SYMMETRIC_PADDING="${SYMMETRIC_PADDING:-0}"
BOUNDARY_REFINEMENT="${BOUNDARY_REFINEMENT:-0}"
CELLVIT_MODE="${CELLVIT_MODE:-none}"
CELL_PRIOR_DROPOUT="${CELL_PRIOR_DROPOUT:-0.2}"
CELL_AUX_LOSS_WEIGHT="${CELL_AUX_LOSS_WEIGHT:-0.2}"
AUGMENT_VFLIP="${AUGMENT_VFLIP:-0}"
AUGMENT_ROT90="${AUGMENT_ROT90:-0}"
AUGMENT_SCALE_CROP="${AUGMENT_SCALE_CROP:-0.0}"
STAIN_AUGMENTATION="${STAIN_AUGMENTATION:-randstainna}"
STAIN_AUGMENTATION_PROB="${STAIN_AUGMENTATION_PROB:-0.7}"
RANDSTAINNA_ROOT="${RANDSTAINNA_ROOT:-third_party/RandStainNA}"
RANDSTAINNA_STD_HYPER="${RANDSTAINNA_STD_HYPER:--0.3}"
RANDSTAINNA_DISTRIBUTION="${RANDSTAINNA_DISTRIBUTION:-normal}"
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

LABEL_SPACE_SUMMARY_ARGS=()
if [[ -n "$LABEL_SPACE_SUMMARY" ]]; then
  LABEL_SPACE_SUMMARY_ARGS+=(--label-space-summary "$LABEL_SPACE_SUMMARY")
fi

mkdir -p "$(dirname "$MANIFEST_PATH")" "$OUTPUT_DIR"

if [[ "${REBUILD_MANIFEST:-0}" == "1" || ! -f "$MANIFEST_PATH" ]]; then
  python scripts/build_segmentator_multidataset_manifest.py \
    --datasets-root "$DATASETS_ROOT" \
    --datasets "${DATASETS[@]}" \
    --val-fraction "$VAL_FRACTION" \
    --test-fraction "$TEST_FRACTION" \
    --seed "$SEED" \
    --output "$MANIFEST_PATH" \
    "${MAX_PER_DATASET_ARGS[@]}"
else
  echo "Using existing manifest: $MANIFEST_PATH"
fi

OPTIONAL_TRAIN_ARGS=()
[[ "$RARE_CLASS_SAMPLING" == "1" ]] && OPTIONAL_TRAIN_ARGS+=(--rare-class-sampling)
[[ "$SYMMETRIC_PADDING" == "1" ]] && OPTIONAL_TRAIN_ARGS+=(--symmetric-padding)
[[ "$BOUNDARY_REFINEMENT" == "1" ]] && OPTIONAL_TRAIN_ARGS+=(--boundary-refinement)
[[ "$AUGMENT_VFLIP" == "1" ]] && OPTIONAL_TRAIN_ARGS+=(--augment-vflip)
[[ "$AUGMENT_ROT90" == "1" ]] && OPTIONAL_TRAIN_ARGS+=(--augment-rot90)

TRAIN_ARGS=(
  --dataset-root "$DATASETS_ROOT" \
  --uni2h-repo "$UNI2H_REPO_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --manifest "$MANIFEST_PATH" \
  --decoder mask2former \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --balanced-datasets \
  --dataset-sampling-temperature "$DATASET_SAMPLING_TEMPERATURE" \
  --rare-class-sample-boost "$RARE_CLASS_SAMPLE_BOOST" \
  --class-weighting "$CLASS_WEIGHTING" \
  --lr "$LR" \
  --backbone-lr "$BACKBONE_LR" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --lr-scheduler "$LR_SCHEDULER" \
  --backbone-unfreeze-epoch "$BACKBONE_UNFREEZE_EPOCH" \
  --backbone-unfreeze-blocks "$BACKBONE_UNFREEZE_BLOCKS" \
  --min-free-gpu-memory-gb-before-unfreeze "$MIN_FREE_GPU_MEMORY_GB_BEFORE_UNFREEZE" \
  --gpu-memory-poll-seconds "$GPU_MEMORY_POLL_SECONDS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --cellvit-mode "$CELLVIT_MODE" \
  --cell-prior-dropout "$CELL_PRIOR_DROPOUT" \
  --cell-aux-loss-weight "$CELL_AUX_LOSS_WEIGHT" \
  --augment-scale-crop "$AUGMENT_SCALE_CROP" \
  --stain-augmentation "$STAIN_AUGMENTATION" \
  --stain-augmentation-prob "$STAIN_AUGMENTATION_PROB" \
  --randstainna-root "$RANDSTAINNA_ROOT" \
  --randstainna-std-hyper "$RANDSTAINNA_STD_HYPER" \
  --randstainna-distribution "$RANDSTAINNA_DISTRIBUTION" \
  --no-amp \
  "${RESUME_ARGS[@]}" \
  "${LABEL_SPACE_SUMMARY_ARGS[@]}" \
  "${OPTIONAL_TRAIN_ARGS[@]}" \
  "${DISABLE_CUDNN_ARGS[@]}"
)

echo "Launching Mask2Former: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS batch_per_gpu=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS effective_batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS)) resume=${RESUME_FROM_CHECKPOINT:-none} stain=${STAIN_AUGMENTATION} stain_prob=${STAIN_AUGMENTATION_PROB}"
if (( NUM_GPUS > 1 )); then
  torchrun --standalone --nnodes=1 --nproc_per_node "$NUM_GPUS" -m segmentator.cli "${TRAIN_ARGS[@]}"
else
  python -m segmentator.cli "${TRAIN_ARGS[@]}"
fi
