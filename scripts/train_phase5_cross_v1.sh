#!/bin/bash
set -euo pipefail

# Phase 5.3 Cross V1 training — IP-Adapter reference attention for Flux ControlNet.
# Update paths below before running.

GPU_IDS="${GPU_IDS:-1,2,4}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export HF_HOME="${HF_HOME:-/data/huggingface}"

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/Pathology_edit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-black-forest-labs/FLUX.1-dev}"

# UNI2-h checkpoint — absolute path to pytorch_model.bin
UNI_CHECKPOINT="${UNI_CHECKPOINT:-${PROJECT_ROOT}/UNI-2h/pytorch_model.bin}"

# Cross V1 training metadata (165K pairs, already built)
CROSS_META="${CROSS_META:-${PROJECT_ROOT}/datasets/phase5_runs/cross_meta/metadata_cross_train.json}"
CROSS_V1_OUTPUT_DIR="${CROSS_V1_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v1}"

MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"

cd "${PROJECT_ROOT}"

TRAIN_OPTIMIZER_ARGS=()
if [[ "${USE_8BIT_ADAM}" == "1" ]]; then
  TRAIN_OPTIMIZER_ARGS+=(--use-8bit-adam)
fi

accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" --gpu_ids="${GPU_IDS}" \
  controlnet_train/cli/train_controlnet_flux_cross_v1.py \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --train-metadata "${CROSS_META}" \
  --uni-checkpoint-path "${UNI_CHECKPOINT}" \
  --output-dir "${CROSS_V1_OUTPUT_DIR}" \
  --logging-dir logs \
  --seed 42 \
  --train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 10 \
  --max-train-steps 20000 \
  --self-reconstruction-warmup-steps 500 \
  --learning-rate 1e-5 \
  --lr-scheduler cosine \
  --lr-warmup-steps 500 \
  --checkpointing-steps 2000 \
  --checkpoints-total-limit 3 \
  --mixed-precision "${MIXED_PRECISION}" \
  --gradient-checkpointing \
  "${TRAIN_OPTIMIZER_ARGS[@]}" \
  --allow-tf32 \
  --dataloader-num-workers 8 \
  --num-double-layers 4 \
  --num-single-layers 4 \
  --cross-v1-spatial-mode target_only \
  --disable-reference-perceiver-self-attn \
  --guidance-scale 3.5 \
  --report-to tensorboard \
  --tracker-project-name flux_controlnet_phase5_cross_v1 \
  --prompt-source dataset
