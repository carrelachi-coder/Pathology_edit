#!/bin/bash
set -euo pipefail

# Phase 5.4 Cross V4.
# Target structure still enters ControlNet only. Reference local tokens, route
# anchors, per-class prior tokens, and mask-guided correspondence bias enter
# FLUX joint attention.

GPU_IDS="${GPU_IDS:-1,2,4}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export HF_HOME="${HF_HOME:-/data/huggingface}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
MODEL_DIR="${MODEL_DIR:-/data/huggingface/FLUX.1-dev}"
CROSS_META="${CROSS_META:-${PROJECT_ROOT}/phase5_runs/cross_meta/metadata_cross_train.json}"
CROSS_V4_OUTPUT_DIR="${CROSS_V4_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v4_mask_guided}"

CONTROLNET_CHECKPOINT="${CONTROLNET_CHECKPOINT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LOAD_CONDITIONING_FROM_CHECKPOINT="${LOAD_CONDITIONING_FROM_CHECKPOINT:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

CONTROLNET_LEARNING_RATE="${CONTROLNET_LEARNING_RATE:-1e-6}"
CONTROLNET_TRAIN_MODE="${CONTROLNET_TRAIN_MODE:-outputs}"
CONTROLNET_TRAIN_X_EMBEDDER="${CONTROLNET_TRAIN_X_EMBEDDER:-1}"
CONDITIONING_LEARNING_RATE="${CONDITIONING_LEARNING_RATE:-5e-7}"

TARGET_TISSUE_ENCODING="${TARGET_TISSUE_ENCODING:-one_hot}"
TARGET_ONE_HOT_SCALE="${TARGET_ONE_HOT_SCALE:-4.0}"
REFERENCE_ROUTE_ANCHOR_MODE="${REFERENCE_ROUTE_ANCHOR_MODE:-coarse}"
REFERENCE_TOKEN_OUTPUT_INIT_STD="${REFERENCE_TOKEN_OUTPUT_INIT_STD:-0.02}"

CROSS_V4_TISSUE_PRIOR_TOKENS_PER_CLASS="${CROSS_V4_TISSUE_PRIOR_TOKENS_PER_CLASS:-4}"
CROSS_V4_CELL_PRIOR_TOKENS_PER_CLASS="${CROSS_V4_CELL_PRIOR_TOKENS_PER_CLASS:-0}"
CROSS_V4_GLOBAL_STYLE_TOKENS="${CROSS_V4_GLOBAL_STYLE_TOKENS:-0}"
CROSS_V4_BIASED_DOUBLE_BLOCKS="${CROSS_V4_BIASED_DOUBLE_BLOCKS:-last}"
CROSS_V4_BIAS_SCALE="${CROSS_V4_BIAS_SCALE:-1.0}"
CROSS_V4_BIAS_WARMUP_STEPS="${CROSS_V4_BIAS_WARMUP_STEPS:-1000}"

SELF_RECONSTRUCTION_WARMUP_STEPS="${SELF_RECONSTRUCTION_WARMUP_STEPS:-500}"
SELF_RECONSTRUCTION_SAMPLE_PROB="${SELF_RECONSTRUCTION_SAMPLE_PROB:-0.05}"
REFERENCE_STYLE_LOSS_WEIGHT="${REFERENCE_STYLE_LOSS_WEIGHT:-1.0}"
REFERENCE_STYLE_LOSS_INTERVAL="${REFERENCE_STYLE_LOSS_INTERVAL:-1}"
REF_SWAP_LOSS_WEIGHT="${REF_SWAP_LOSS_WEIGHT:-0.1}"
REF_SWAP_VARIANTS="${REF_SWAP_VARIANTS:-zero}"
REF_SWAP_LOSS_INTERVAL="${REF_SWAP_LOSS_INTERVAL:-1}"
REF_CHECK_STEP="${REF_CHECK_STEP:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"

cd "${PROJECT_ROOT}"

TRAIN_OPTIMIZER_ARGS=()
if [[ "${USE_8BIT_ADAM}" == "1" ]]; then
  TRAIN_OPTIMIZER_ARGS+=(--use-8bit-adam)
fi

TRAIN_MEMORY_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  TRAIN_MEMORY_ARGS+=(--gradient-checkpointing)
fi

TRAIN_CONTROLNET_ARGS=(--controlnet-train-mode "${CONTROLNET_TRAIN_MODE}")
if [[ "${CONTROLNET_TRAIN_X_EMBEDDER}" == "1" ]]; then
  TRAIN_CONTROLNET_ARGS+=(--controlnet-train-x-embedder)
fi

TRAIN_CHECKPOINT_ARGS=()
if [[ -n "${CONTROLNET_CHECKPOINT}" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(--controlnet_model_name_or_path "${CONTROLNET_CHECKPOINT}")
  if [[ "${LOAD_CONDITIONING_FROM_CHECKPOINT}" == "1" ]]; then
    TRAIN_CHECKPOINT_ARGS+=(--load-conditioning-from-checkpoint)
  fi
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" --gpu_ids="${GPU_IDS}" \
  controlnet_train/cli/train_controlnet_flux_cross_v4.py \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --train-metadata "${CROSS_META}" \
  "${TRAIN_CHECKPOINT_ARGS[@]}" \
  --output-dir "${CROSS_V4_OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}" \
  --logging-dir logs \
  --seed 42 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 10 \
  --max-train-steps "${MAX_TRAIN_STEPS}" \
  --self-reconstruction-warmup-steps "${SELF_RECONSTRUCTION_WARMUP_STEPS}" \
  --self-reconstruction-sample-prob "${SELF_RECONSTRUCTION_SAMPLE_PROB}" \
  --reference-token-output-init-std "${REFERENCE_TOKEN_OUTPUT_INIT_STD}" \
  --reference-route-anchor-mode "${REFERENCE_ROUTE_ANCHOR_MODE}" \
  --reference-style-loss-weight "${REFERENCE_STYLE_LOSS_WEIGHT}" \
  --reference-style-loss-interval "${REFERENCE_STYLE_LOSS_INTERVAL}" \
  --ref-swap-loss-weight "${REF_SWAP_LOSS_WEIGHT}" \
  --ref-swap-loss-interval "${REF_SWAP_LOSS_INTERVAL}" \
  --ref-swap-variants "${REF_SWAP_VARIANTS}" \
  --ref-check-step "${REF_CHECK_STEP}" \
  --target-tissue-encoding "${TARGET_TISSUE_ENCODING}" \
  --target-one-hot-scale "${TARGET_ONE_HOT_SCALE}" \
  --cross-v4-tissue-prior-tokens-per-class "${CROSS_V4_TISSUE_PRIOR_TOKENS_PER_CLASS}" \
  --cross-v4-cell-prior-tokens-per-class "${CROSS_V4_CELL_PRIOR_TOKENS_PER_CLASS}" \
  --cross-v4-global-style-tokens "${CROSS_V4_GLOBAL_STYLE_TOKENS}" \
  --cross-v4-biased-double-blocks "${CROSS_V4_BIASED_DOUBLE_BLOCKS}" \
  --cross-v4-bias-scale "${CROSS_V4_BIAS_SCALE}" \
  --cross-v4-bias-warmup-steps "${CROSS_V4_BIAS_WARMUP_STEPS}" \
  --learning-rate "${CONTROLNET_LEARNING_RATE}" \
  "${TRAIN_CONTROLNET_ARGS[@]}" \
  --conditioning-learning-rate "${CONDITIONING_LEARNING_RATE}" \
  --lr-scheduler constant_with_warmup \
  --lr-warmup-steps 500 \
  --checkpointing-steps "${CHECKPOINTING_STEPS}" \
  --checkpoints-total-limit 5 \
  --mixed-precision "${MIXED_PRECISION}" \
  "${TRAIN_OPTIMIZER_ARGS[@]}" \
  "${TRAIN_MEMORY_ARGS[@]}" \
  --allow-tf32 \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader-prefetch-factor "${DATALOADER_PREFETCH_FACTOR}" \
  --num-double-layers 4 \
  --num-single-layers 4 \
  --guidance-scale 3.5 \
  --report-to tensorboard \
  --tracker-project-name flux_controlnet_phase5_cross_v4_mask_guided
