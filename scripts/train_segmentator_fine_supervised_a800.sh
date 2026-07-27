#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS:-3,5}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPU_LIST[@]}}"

DATASETS_ROOT="${DATASETS_ROOT:-/data/wqx/flowedit/data}"
UNI2H_REPO_PATH="${UNI2H_REPO:-./UNI-2h}"
MANIFEST_PATH="${SEGMENTATOR_MANIFEST:-/data1/zhao/wqx/segmentator_manifests/grouped_seed42.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/zhao/wqx/segmentator_fine/fine_supervised_sampling_seed42}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-/data1/zhao/wqx/segmentator_fine/fine_probe_single_scale_seed42/best_fine_mIoU.pt}"
LABEL_SPACE_SUMMARY="${LABEL_SPACE_SUMMARY:-/data1/zhao/wqx/segmentator_improved/baseline_seed42/config.json}"

IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-8}"
ACCUM_DENOM=$((BATCH_SIZE * NUM_GPUS))
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$(((TARGET_EFFECTIVE_BATCH_SIZE + ACCUM_DENOM - 1) / ACCUM_DENOM))}"
EPOCHS="${EPOCHS:-8}"
SEED="${SEED:-42}"
LR="${LR:-5e-5}"
FINE_LOSS_WEIGHT="${FINE_LOSS_WEIGHT:-1.0}"
FINE_CLASS_WEIGHT_MIN="${FINE_CLASS_WEIGHT_MIN:-0.75}"
FINE_CLASS_WEIGHT_MAX="${FINE_CLASS_WEIGHT_MAX:-2.0}"
FINE_SAMPLING_TEMPERATURE="${FINE_SAMPLING_TEMPERATURE:-0.5}"
FINE_SAMPLING_RARE_BOOST="${FINE_SAMPLING_RARE_BOOST:-4.0}"
FINE_SAMPLING_MIN_PIXELS="${FINE_SAMPLING_MIN_PIXELS:-256}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"
# The Fine Probe freezes every shared tensor, so gate against the metrics of its
# actual initialization checkpoint instead of combining maxima from two epochs.
COARSE_MIOU_FLOOR="${COARSE_MIOU_FLOOR:-0.6159}"
COARSE_BOUNDARY_F1_4_FLOOR="${COARSE_BOUNDARY_F1_4_FLOOR:-0.5106}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
DDP_TIMEOUT_SECONDS="${DDP_TIMEOUT_SECONDS:-7200}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

mkdir -p "$OUTPUT_DIR"

START_ARGS=()
if [[ "$RESUME_FROM_CHECKPOINT" == "latest" && -f "$OUTPUT_DIR/checkpoint_last.pt" ]]; then
  START_ARGS+=(--resume-from-checkpoint latest)
elif [[ -n "$RESUME_FROM_CHECKPOINT" && "$RESUME_FROM_CHECKPOINT" != "latest" && "$RESUME_FROM_CHECKPOINT" != "none" ]]; then
  START_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
elif [[ -f "$PROBE_CHECKPOINT" ]]; then
  START_ARGS+=(--init-from-checkpoint "$PROBE_CHECKPOINT")
else
  echo "Eligible Fine Probe checkpoint not found: $PROBE_CHECKPOINT" >&2
  exit 2
fi

TRAIN_ARGS=(
  --dataset-root "$DATASETS_ROOT"
  --uni2h-repo "$UNI2H_REPO_PATH"
  --output-dir "$OUTPUT_DIR"
  --manifest "$MANIFEST_PATH"
  --decoder mask2former
  --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --dataset-sampling-temperature "$FINE_SAMPLING_TEMPERATURE"
  --fine-supervision-sampling
  --fine-sampling-rare-class-boost "$FINE_SAMPLING_RARE_BOOST"
  --fine-sampling-min-valid-pixels "$FINE_SAMPLING_MIN_PIXELS"
  --class-weighting none
  --fine-class-weighting inverse_sqrt
  --fine-class-weight-min "$FINE_CLASS_WEIGHT_MIN"
  --fine-class-weight-max "$FINE_CLASS_WEIGHT_MAX"
  --label-space-summary "$LABEL_SPACE_SUMMARY"
  --lr "$LR"
  --backbone-lr 1e-5
  --warmup-epochs 0
  --lr-scheduler cosine
  --backbone-unfreeze-epoch -1
  --backbone-unfreeze-blocks 0
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --checkpoint-mode fine_dataset_macro
  --checkpoint-coarse-miou-floor "$COARSE_MIOU_FLOOR"
  --checkpoint-coarse-boundary-f1-4-floor "$COARSE_BOUNDARY_F1_4_FLOOR"
  --ddp-timeout-seconds "$DDP_TIMEOUT_SECONDS"
  --rank-zero-validation
  --hierarchical-fine
  --freeze-shared-for-fine
  --fine-only-loss
  --fine-loss-weight "$FINE_LOSS_WEIGHT"
  --cellvit-mode none
  --augment-scale-crop 0.0
  --stain-augmentation randstainna
  --stain-augmentation-prob 0.7
  --randstainna-root third_party/RandStainNA
  --randstainna-std-hyper -0.3
  --randstainna-distribution normal
  --augment-vflip
  --augment-rot90
  --no-amp
  "${START_ARGS[@]}"
)
if [[ -n "$SAMPLES_PER_EPOCH" ]]; then
  TRAIN_ARGS+=(--samples-per-epoch "$SAMPLES_PER_EPOCH")
fi

echo "Launching fine-supervision-aware Segmentator on GPUs $GPU_IDS (world_size=$NUM_GPUS, effective_batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS)))"
if (( NUM_GPUS > 1 )); then
  torchrun --standalone --nnodes=1 --nproc_per_node "$NUM_GPUS" -m segmentator.cli "${TRAIN_ARGS[@]}"
else
  python -m segmentator.cli "${TRAIN_ARGS[@]}"
fi
