#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS:-1,2}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPU_LIST[@]}}"

DATASETS_ROOT="${DATASETS_ROOT:-/data/wqx/flowedit/data}"
UNI2H_REPO_PATH="${UNI2H_REPO:-./UNI-2h}"
MANIFEST_PATH="${SEGMENTATOR_MANIFEST:-/data1/zhao/wqx/segmentator_manifests/grouped_seed42.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/zhao/wqx/segmentator_fine/fine_boundary_v3_seed42}"
FINE_CHECKPOINT="${FINE_CHECKPOINT:-/data1/zhao/wqx/segmentator_fine/fine_supervised_sampling_seed42/best_composite.pt}"
LABEL_SPACE_SUMMARY="${LABEL_SPACE_SUMMARY:-/data1/zhao/wqx/segmentator_improved/baseline_seed42/config.json}"

IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-8}"
ACCUM_DENOM=$((BATCH_SIZE * NUM_GPUS))
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$(((TARGET_EFFECTIVE_BATCH_SIZE + ACCUM_DENOM - 1) / ACCUM_DENOM))}"
EPOCHS="${EPOCHS:-8}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-39404}"
LR="${LR:-3e-5}"
SEED="${SEED:-42}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
DDP_TIMEOUT_SECONDS="${DDP_TIMEOUT_SECONDS:-7200}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

mkdir -p "$OUTPUT_DIR"

START_ARGS=()
if [[ "$RESUME_FROM_CHECKPOINT" == "latest" && -f "$OUTPUT_DIR/checkpoint_last.pt" ]]; then
  START_ARGS+=(--resume-from-checkpoint latest)
elif [[ -n "$RESUME_FROM_CHECKPOINT" && "$RESUME_FROM_CHECKPOINT" != "latest" && "$RESUME_FROM_CHECKPOINT" != "none" ]]; then
  START_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
elif [[ -f "$FINE_CHECKPOINT" ]]; then
  START_ARGS+=(--init-from-checkpoint "$FINE_CHECKPOINT")
else
  echo "Fine Segmentator checkpoint not found: $FINE_CHECKPOINT" >&2
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
  --samples-per-epoch "$SAMPLES_PER_EPOCH"
  --seed "$SEED"
  --balanced-datasets
  --dataset-sampling-temperature 0.5
  --boundary-aware-sampling
  --boundary-sampling-mode dataset_quantile
  --boundary-sampling-boost 3.0
  --boundary-sampling-min-pixels 4096
  --boundary-sampling-width 4
  --class-weighting none
  --fine-class-weighting inverse_sqrt
  --fine-class-weight-min 0.75
  --fine-class-weight-max 2.0
  --label-space-summary "$LABEL_SPACE_SUMMARY"
  --lr "$LR"
  --backbone-lr 1e-5
  --warmup-epochs 0
  --lr-scheduler cosine
  --backbone-unfreeze-epoch -1
  --backbone-unfreeze-blocks 0
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --checkpoint-mode boundary_f1_4
  --checkpoint-coarse-miou-floor 0.6678
  --checkpoint-fine-dataset-macro-floor 0.6370
  --ddp-timeout-seconds "$DDP_TIMEOUT_SECONDS"
  --rank-zero-validation
  --hierarchical-fine
  --boundary-refinement
  --trainable-scope boundary
  --refinement-only-loss
  --refinement-loss-weight 0.25
  --refinement-boundary-weight 1.0
  --refinement-boundary-widths 2 4 8
  --refinement-boundary-ce-weight 1.0
  --refinement-consistency-weight 2.0
  --refinement-gate-width 4
  --refinement-gate-threshold 0.15
  --refinement-gate-mode learned_soft
  --refinement-gate-loss-weight 0.5
  --refinement-gate-target-width 8
  --cellvit-mode none
  --stain-augmentation randstainna
  --stain-augmentation-prob 0.7
  --randstainna-root third_party/RandStainNA
  --randstainna-std-hyper -0.3
  --randstainna-distribution normal
  --augment-vflip
  --augment-rot90
  --augment-scale-crop 0.0
  --no-amp
  "${START_ARGS[@]}"
)

echo "Launching Fine Boundary V3 on GPUs $GPU_IDS (world_size=$NUM_GPUS)"
if (( NUM_GPUS > 1 )); then
  torchrun --standalone --nnodes=1 --nproc_per_node "$NUM_GPUS" -m segmentator.cli "${TRAIN_ARGS[@]}"
else
  python -m segmentator.cli "${TRAIN_ARGS[@]}"
fi
