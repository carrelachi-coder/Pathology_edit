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
SOURCE_CROSS_V1_OUTPUT_DIR="${SOURCE_CROSS_V1_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v1}"
CONTROLNET_CHECKPOINT="${CONTROLNET_CHECKPOINT:-${SOURCE_CROSS_V1_OUTPUT_DIR}/checkpoint-20000}"
CROSS_V1_OUTPUT_DIR="${CROSS_V1_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v1_single10_uni}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# Reference-aware auxiliary losses. Perceptual UNI loss is the main appearance
# signal; style is kept as a lower-weight stain/color auxiliary.
PERCEPTUAL_LOSS_WEIGHT="${PERCEPTUAL_LOSS_WEIGHT:-0.5}"
PERCEPTUAL_LOSS_INTERVAL="${PERCEPTUAL_LOSS_INTERVAL:-1}"
REFERENCE_STYLE_LOSS_WEIGHT="${REFERENCE_STYLE_LOSS_WEIGHT:-5}"
REFERENCE_STYLE_TISSUE_WEIGHT="${REFERENCE_STYLE_TISSUE_WEIGHT:-1.0}"
REFERENCE_STYLE_NUCLEI_WEIGHT="${REFERENCE_STYLE_NUCLEI_WEIGHT:-1.0}"
REFERENCE_STYLE_MEAN_WEIGHT="${REFERENCE_STYLE_MEAN_WEIGHT:-1.0}"
REFERENCE_STYLE_STD_WEIGHT="${REFERENCE_STYLE_STD_WEIGHT:-1.0}"
REFERENCE_STYLE_COV_WEIGHT="${REFERENCE_STYLE_COV_WEIGHT:-0.25}"
REFERENCE_STYLE_MIN_PIXELS="${REFERENCE_STYLE_MIN_PIXELS:-32}"
REFERENCE_STYLE_LOSS_INTERVAL="${REFERENCE_STYLE_LOSS_INTERVAL:-4}"
REF_SWAP_LOSS_WEIGHT="${REF_SWAP_LOSS_WEIGHT:-0.1}"
REF_SWAP_MARGIN="${REF_SWAP_MARGIN:-0.2}"
REF_SWAP_VARIANTS="${REF_SWAP_VARIANTS:-random}"
REF_SWAP_LOSS_INTERVAL="${REF_SWAP_LOSS_INTERVAL:-2}"
SELF_RECONSTRUCTION_SAMPLE_PROB="${SELF_RECONSTRUCTION_SAMPLE_PROB:-0.0}"
SELF_RECONSTRUCTION_L1_WEIGHT="${SELF_RECONSTRUCTION_L1_WEIGHT:-0.0}"
IP_REF_LEARNING_RATE="${IP_REF_LEARNING_RATE:-3e-5}"
IP_SINGLE_LEARNING_RATE="${IP_SINGLE_LEARNING_RATE:-1e-4}"
IP_SINGLE_NUM_LAYERS="${IP_SINGLE_NUM_LAYERS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"

cd "${PROJECT_ROOT}"

TRAIN_OPTIMIZER_ARGS=()
if [[ "${USE_8BIT_ADAM}" == "1" ]]; then
  TRAIN_OPTIMIZER_ARGS+=(--use-8bit-adam)
fi

TRAIN_MEMORY_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  TRAIN_MEMORY_ARGS+=(--gradient-checkpointing)
fi

CLI_HELP="$("${PYTHON_BIN}" controlnet_train/cli/train_controlnet_flux_cross_v1.py --help 2>&1 || true)"

TRAIN_DATALOADER_ARGS=(--dataloader-num-workers "${DATALOADER_NUM_WORKERS}")
if grep -q -- "--dataloader-prefetch-factor" <<< "${CLI_HELP}"; then
  TRAIN_DATALOADER_ARGS+=(--dataloader-prefetch-factor "${DATALOADER_PREFETCH_FACTOR}")
else
  echo "Warning: CLI does not support --dataloader-prefetch-factor; skipping it." >&2
fi

TRAIN_AUX_INTERVAL_ARGS=()
if grep -q -- "--reference-style-loss-interval" <<< "${CLI_HELP}"; then
  TRAIN_AUX_INTERVAL_ARGS+=(--reference-style-loss-interval "${REFERENCE_STYLE_LOSS_INTERVAL}")
else
  echo "Warning: CLI does not support --reference-style-loss-interval; style loss will run every step." >&2
fi
if grep -q -- "--ref-swap-loss-interval" <<< "${CLI_HELP}"; then
  TRAIN_AUX_INTERVAL_ARGS+=(--ref-swap-loss-interval "${REF_SWAP_LOSS_INTERVAL}")
else
  echo "Warning: CLI does not support --ref-swap-loss-interval; ref-swap loss will run every step." >&2
fi
if grep -q -- "--perceptual-loss-interval" <<< "${CLI_HELP}"; then
  TRAIN_AUX_INTERVAL_ARGS+=(--perceptual-loss-interval "${PERCEPTUAL_LOSS_INTERVAL}")
else
  echo "Warning: CLI does not support --perceptual-loss-interval; perceptual loss will use its default." >&2
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

TRAIN_CHECKPOINT_ARGS=()
if [[ -n "${CONTROLNET_CHECKPOINT}" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(
    --controlnet_model_name_or_path "${CONTROLNET_CHECKPOINT}"
    --ip-adapter-checkpoint "${CONTROLNET_CHECKPOINT}"
    --a1-lite
    --a1-lite-load-ref-encoder
  )
fi

accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" --gpu_ids="${GPU_IDS}" \
  controlnet_train/cli/train_controlnet_flux_cross_v1.py \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --train-metadata "${CROSS_META}" \
  --uni-checkpoint-path "${UNI_CHECKPOINT}" \
  "${TRAIN_CHECKPOINT_ARGS[@]}" \
  --output-dir "${CROSS_V1_OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}" \
  --logging-dir logs \
  --seed 42 \
  --train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 10 \
  --max-train-steps 20000 \
  --self-reconstruction-warmup-steps 0 \
  --self-reconstruction-sample-prob "${SELF_RECONSTRUCTION_SAMPLE_PROB}" \
  --self-reconstruction-l1-weight "${SELF_RECONSTRUCTION_L1_WEIGHT}" \
  --learning-rate 1e-5 \
  --ip-ref-learning-rate "${IP_REF_LEARNING_RATE}" \
  --ip-single-learning-rate "${IP_SINGLE_LEARNING_RATE}" \
  --lr-scheduler cosine \
  --lr-warmup-steps 500 \
  --checkpointing-steps 2000 \
  --checkpoints-total-limit 3 \
  --mixed-precision "${MIXED_PRECISION}" \
  "${TRAIN_OPTIMIZER_ARGS[@]}" \
  "${TRAIN_MEMORY_ARGS[@]}" \
  --allow-tf32 \
  "${TRAIN_DATALOADER_ARGS[@]}" \
  --num-double-layers 4 \
  --num-single-layers 4 \
  --ip-init-gain 0.1 \
  --ip-single-num-layers "${IP_SINGLE_NUM_LAYERS}" \
  --cross-v1-spatial-mode target_only \
  --skip-reference-perceiver \
  --disable-reference-perceiver-self-attn \
  --perceptual-loss-weight "${PERCEPTUAL_LOSS_WEIGHT}" \
  --reference-style-loss-weight "${REFERENCE_STYLE_LOSS_WEIGHT}" \
  --reference-style-tissue-weight "${REFERENCE_STYLE_TISSUE_WEIGHT}" \
  --reference-style-nuclei-weight "${REFERENCE_STYLE_NUCLEI_WEIGHT}" \
  --reference-style-mean-weight "${REFERENCE_STYLE_MEAN_WEIGHT}" \
  --reference-style-std-weight "${REFERENCE_STYLE_STD_WEIGHT}" \
  --reference-style-cov-weight "${REFERENCE_STYLE_COV_WEIGHT}" \
  --reference-style-min-pixels "${REFERENCE_STYLE_MIN_PIXELS}" \
  --ref-swap-loss-weight "${REF_SWAP_LOSS_WEIGHT}" \
  --ref-swap-margin "${REF_SWAP_MARGIN}" \
  --ref-swap-variants "${REF_SWAP_VARIANTS}" \
  "${TRAIN_AUX_INTERVAL_ARGS[@]}" \
  --guidance-scale 3.5 \
  --report-to tensorboard \
  --tracker-project-name flux_controlnet_phase5_cross_v1 \
  --prompt-source dataset
