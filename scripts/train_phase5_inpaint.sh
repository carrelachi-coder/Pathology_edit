#!/bin/bash
set -euo pipefail

# Phase 5 synthetic inpaint dataset build + ControlNet training.
# Update the paths below for your server before running.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export HF_HOME="${HF_HOME:-/data/huggingface}"

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/Pathology_edit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-black-forest-labs/FLUX.1-dev}"

BCSS_ROOT="${BCSS_ROOT:-/data/datasets/BCSS/BCSS_PATCHES}"
PANDA_ROOT="${PANDA_ROOT:-/data/datasets/PANDA/PANDA_PATCHES}"
GLAS_ROOT="${GLAS_ROOT:-/data/datasets/GlaS/GlaS_PATCHES}"
IGNITE_ROOT="${IGNITE_ROOT:-/data/datasets/IGNITE_PATCHES}"
ORCA_ROOT="${ORCA_ROOT:-/data/datasets/ORCA/ORCA_PATCHES}"
PUMA_ROOT="${PUMA_ROOT:-/data/datasets/PUMA/PUMA_PATCHES}"

INPAINT_META_DIR="${INPAINT_META_DIR:-${PROJECT_ROOT}/phase5_runs/inpaint_meta}"
INPAINT_OUTPUT_DIR="${INPAINT_OUTPUT_DIR:-${PROJECT_ROOT}/phase5_runs/controlnet_inpaint}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

RUN_DATASET_BUILD="${RUN_DATASET_BUILD:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"

cd "${PROJECT_ROOT}"

if [[ "${RUN_DATASET_BUILD}" == "1" ]]; then
  "${PYTHON_BIN}" controlnet_train/cli/build_inpaint_dataset.py \
    --dataset-root "BCSS=${BCSS_ROOT}" \
    --dataset-root "PANDA=${PANDA_ROOT}" \
    --dataset-root "GlaS=${GLAS_ROOT}" \
    --dataset-root "IGNITE=${IGNITE_ROOT}" \
    --dataset-root "ORCA=${ORCA_ROOT}" \
    --dataset-root "PUMA=${PUMA_ROOT}" \
    --output-dir "${INPAINT_META_DIR}" \
    --val-ratio 0.1 \
    --seed 42 \
    --forced-mode mixed \
    --max-attempts-per-sample 5
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  TRAIN_OPTIMIZER_ARGS=()
  if [[ "${USE_8BIT_ADAM}" == "1" ]]; then
    TRAIN_OPTIMIZER_ARGS+=(--use-8bit-adam)
  fi

  accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" \
    controlnet_train/cli/train_controlnet_flux_inpaint.py \
    --pretrained_model_name_or_path "${MODEL_DIR}" \
    --train-metadata "${INPAINT_META_DIR}/metadata_inpaint_train.jsonl" \
    --output-dir "${INPAINT_OUTPUT_DIR}" \
    --logging-dir logs \
    --seed 42 \
    --train-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --num-train-epochs 10 \
    --learning-rate 1e-5 \
    --lr-scheduler cosine \
    --lr-warmup-steps 500 \
    --checkpointing-steps 1000 \
    --checkpoints-total-limit 3 \
    --mixed-precision "${MIXED_PRECISION}" \
    --gradient-checkpointing \
    "${TRAIN_OPTIMIZER_ARGS[@]}" \
    --allow-tf32 \
    --dataloader-num-workers 8 \
    --num-double-layers 4 \
    --num-single-layers 4 \
    --guidance-scale 3.5 \
    --report-to tensorboard \
    --tracker-project-name flux_controlnet_phase5_inpaint
fi
