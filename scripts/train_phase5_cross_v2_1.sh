#!/bin/bash
set -euo pipefail

# Phase 5.3 Cross V2.1.
# Simple route: no IP-Adapter, no UNI reference encoder. ControlNet condition is:
# [z_ref, ref_tissue_feat, ref_nuclei_feat, tar_tissue_feat, tar_nuclei_feat]

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
CROSS_V2_1_OUTPUT_DIR="${CROSS_V2_1_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v2_1_nohed_selfrec}"

# Optional warm start from Cross V1 or Cross V2.1. V1 mask weights are remapped
# after the z_ref channels; the new z_ref projection starts at zero.
CONTROLNET_CHECKPOINT="${CONTROLNET_CHECKPOINT:-}"
CONDITIONING_CHECKPOINT="${CONDITIONING_CHECKPOINT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LOAD_CONDITIONING_FROM_CHECKPOINT="${LOAD_CONDITIONING_FROM_CHECKPOINT:-1}"
RESOLVE_CONTROLNET_CHECKPOINT_LATEST="${RESOLVE_CONTROLNET_CHECKPOINT_LATEST:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

CONTROLNET_LEARNING_RATE="${CONTROLNET_LEARNING_RATE:-1e-6}"
CONTROLNET_TRAIN_MODE="${CONTROLNET_TRAIN_MODE:-outputs}"
CONTROLNET_TRAIN_X_EMBEDDER="${CONTROLNET_TRAIN_X_EMBEDDER:-1}"
CONTROLNET_TRAIN_LAST_N_BLOCKS="${CONTROLNET_TRAIN_LAST_N_BLOCKS:-0}"
CONTROLNET_TRAIN_LAST_N_SINGLE_BLOCKS="${CONTROLNET_TRAIN_LAST_N_SINGLE_BLOCKS:-0}"
CONDITIONING_LEARNING_RATE="${CONDITIONING_LEARNING_RATE:-5e-7}"

STAIN_AUGMENTATION="${STAIN_AUGMENTATION:-none}"
STAIN_COUNTERFACTUAL_PROB="${STAIN_COUNTERFACTUAL_PROB:-0}"
HED_SIGMA="${HED_SIGMA:-0.1}"
HED_BETA="${HED_BETA:-0.01}"
HED_STRONG_ALPHA_SAMPLING="${HED_STRONG_ALPHA_SAMPLING:-0}"
HED_ALPHA_MIN="${HED_ALPHA_MIN:-0.8}"
HED_ALPHA_LOW="${HED_ALPHA_LOW:-0.95}"
HED_ALPHA_HIGH="${HED_ALPHA_HIGH:-1.05}"
HED_ALPHA_MAX="${HED_ALPHA_MAX:-1.2}"

SELF_RECONSTRUCTION_WARMUP_STEPS="${SELF_RECONSTRUCTION_WARMUP_STEPS:-500}"
SELF_RECONSTRUCTION_SAMPLE_PROB="${SELF_RECONSTRUCTION_SAMPLE_PROB:-0.1}"
REFERENCE_REGION_LOSS_WEIGHT="${REFERENCE_REGION_LOSS_WEIGHT:-0.05}"
REFERENCE_REGION_LOSS_WARMUP_STEPS="${REFERENCE_REGION_LOSS_WARMUP_STEPS:-500}"
REFERENCE_REGION_LOSS_INTERVAL="${REFERENCE_REGION_LOSS_INTERVAL:-1}"
REFERENCE_REGION_LOSS_MIN_SIGMA="${REFERENCE_REGION_LOSS_MIN_SIGMA:-0.0}"
REFERENCE_REGION_LOSS_MAX_SIGMA="${REFERENCE_REGION_LOSS_MAX_SIGMA:-0.6}"
REFERENCE_REGION_MIN_PIXELS="${REFERENCE_REGION_MIN_PIXELS:-32}"
UNI_CHECKPOINT_PATH="${UNI_CHECKPOINT_PATH:-}"
SAME_WSI_APPEARANCE_CHECKPOINT="${SAME_WSI_APPEARANCE_CHECKPOINT:-/data/wqx/flowedit/same_wsi_appearance/best.pt}"
REFERENCE_PERCEPTUAL_BACKEND="${REFERENCE_PERCEPTUAL_BACKEND:-vgg}"
REFERENCE_VGG_WEIGHTS="${REFERENCE_VGG_WEIGHTS:-imagenet}"
REFERENCE_VGG_WEIGHTS_PATH="${REFERENCE_VGG_WEIGHTS_PATH:-}"
REFERENCE_VGG_LAYERS="${REFERENCE_VGG_LAYERS:-relu1_1,relu1_2,relu2_1,relu2_2}"
REFERENCE_VGG_LOSS_TYPE="${REFERENCE_VGG_LOSS_TYPE:-gram}"
REFERENCE_VGG_INPUT_SIZE="${REFERENCE_VGG_INPUT_SIZE:-256}"
REFERENCE_PERCEPTUAL_LOSS_WEIGHT="${REFERENCE_PERCEPTUAL_LOSS_WEIGHT:-0.05}"
REFERENCE_PERCEPTUAL_LOSS_WARMUP_STEPS="${REFERENCE_PERCEPTUAL_LOSS_WARMUP_STEPS:-500}"
REFERENCE_PERCEPTUAL_LOSS_INTERVAL="${REFERENCE_PERCEPTUAL_LOSS_INTERVAL:-1}"
REFERENCE_PERCEPTUAL_LOSS_MIN_SIGMA="${REFERENCE_PERCEPTUAL_LOSS_MIN_SIGMA:-0.0}"
REFERENCE_PERCEPTUAL_LOSS_MAX_SIGMA="${REFERENCE_PERCEPTUAL_LOSS_MAX_SIGMA:-0.4}"
REFERENCE_PERCEPTUAL_MIN_PIXELS="${REFERENCE_PERCEPTUAL_MIN_PIXELS:-8}"
REFERENCE_GRAD_RATIO_INTERVAL="${REFERENCE_GRAD_RATIO_INTERVAL:-50}"
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

TRAIN_CONTROLNET_ARGS=(
  --controlnet-train-mode "${CONTROLNET_TRAIN_MODE}"
  --controlnet-train-last-n-blocks "${CONTROLNET_TRAIN_LAST_N_BLOCKS}"
  --controlnet-train-last-n-single-blocks "${CONTROLNET_TRAIN_LAST_N_SINGLE_BLOCKS}"
)
if [[ "${CONTROLNET_TRAIN_X_EMBEDDER}" == "1" ]]; then
  TRAIN_CONTROLNET_ARGS+=(--controlnet-train-x-embedder)
fi

TRAIN_HED_ARGS=(
  --stain-augmentation "${STAIN_AUGMENTATION}"
  --stain-counterfactual-prob "${STAIN_COUNTERFACTUAL_PROB}"
  --hed-sigma "${HED_SIGMA}"
  --hed-beta "${HED_BETA}"
  --hed-alpha-min "${HED_ALPHA_MIN}"
  --hed-alpha-low "${HED_ALPHA_LOW}"
  --hed-alpha-high "${HED_ALPHA_HIGH}"
  --hed-alpha-max "${HED_ALPHA_MAX}"
)
if [[ "${HED_STRONG_ALPHA_SAMPLING}" == "1" ]]; then
  TRAIN_HED_ARGS+=(--hed-strong-alpha-sampling)
fi

TRAIN_REFERENCE_REGION_ARGS=(
  --reference-region-loss-weight "${REFERENCE_REGION_LOSS_WEIGHT}"
  --reference-region-loss-warmup-steps "${REFERENCE_REGION_LOSS_WARMUP_STEPS}"
  --reference-region-loss-interval "${REFERENCE_REGION_LOSS_INTERVAL}"
  --reference-region-loss-min-sigma "${REFERENCE_REGION_LOSS_MIN_SIGMA}"
  --reference-region-loss-max-sigma "${REFERENCE_REGION_LOSS_MAX_SIGMA}"
  --reference-region-min-pixels "${REFERENCE_REGION_MIN_PIXELS}"
)
if [[ "${REFERENCE_PERCEPTUAL_BACKEND}" == "vgg" ]]; then
  TRAIN_REFERENCE_REGION_ARGS+=(
    --reference-perceptual-backend vgg
    --reference-vgg-weights "${REFERENCE_VGG_WEIGHTS}"
    --reference-vgg-layers "${REFERENCE_VGG_LAYERS}"
    --reference-vgg-loss-type "${REFERENCE_VGG_LOSS_TYPE}"
    --reference-vgg-input-size "${REFERENCE_VGG_INPUT_SIZE}"
    --reference-perceptual-loss-weight "${REFERENCE_PERCEPTUAL_LOSS_WEIGHT}"
    --reference-perceptual-loss-warmup-steps "${REFERENCE_PERCEPTUAL_LOSS_WARMUP_STEPS}"
    --reference-perceptual-loss-interval "${REFERENCE_PERCEPTUAL_LOSS_INTERVAL}"
    --reference-perceptual-loss-min-sigma "${REFERENCE_PERCEPTUAL_LOSS_MIN_SIGMA}"
    --reference-perceptual-loss-max-sigma "${REFERENCE_PERCEPTUAL_LOSS_MAX_SIGMA}"
    --reference-perceptual-min-pixels "${REFERENCE_PERCEPTUAL_MIN_PIXELS}"
    --reference-grad-ratio-interval "${REFERENCE_GRAD_RATIO_INTERVAL}"
  )
  if [[ -n "${REFERENCE_VGG_WEIGHTS_PATH}" ]]; then
    TRAIN_REFERENCE_REGION_ARGS+=(--reference-vgg-weights-path "${REFERENCE_VGG_WEIGHTS_PATH}")
  fi
elif [[ "${REFERENCE_PERCEPTUAL_BACKEND}" == "same_wsi" && -n "${SAME_WSI_APPEARANCE_CHECKPOINT}" ]]; then
  TRAIN_REFERENCE_REGION_ARGS+=(
    --reference-perceptual-backend same_wsi
    --same-wsi-appearance-checkpoint "${SAME_WSI_APPEARANCE_CHECKPOINT}"
    --reference-perceptual-loss-weight "${REFERENCE_PERCEPTUAL_LOSS_WEIGHT}"
    --reference-perceptual-loss-warmup-steps "${REFERENCE_PERCEPTUAL_LOSS_WARMUP_STEPS}"
    --reference-perceptual-loss-interval "${REFERENCE_PERCEPTUAL_LOSS_INTERVAL}"
    --reference-perceptual-loss-min-sigma "${REFERENCE_PERCEPTUAL_LOSS_MIN_SIGMA}"
    --reference-perceptual-loss-max-sigma "${REFERENCE_PERCEPTUAL_LOSS_MAX_SIGMA}"
    --reference-perceptual-min-pixels "${REFERENCE_PERCEPTUAL_MIN_PIXELS}"
    --reference-grad-ratio-interval "${REFERENCE_GRAD_RATIO_INTERVAL}"
  )
elif [[ "${REFERENCE_PERCEPTUAL_BACKEND}" == "uni" && -n "${UNI_CHECKPOINT_PATH}" ]]; then
  TRAIN_REFERENCE_REGION_ARGS+=(
    --reference-perceptual-backend uni
    --uni-checkpoint-path "${UNI_CHECKPOINT_PATH}"
    --reference-perceptual-loss-weight "${REFERENCE_PERCEPTUAL_LOSS_WEIGHT}"
    --reference-perceptual-loss-warmup-steps "${REFERENCE_PERCEPTUAL_LOSS_WARMUP_STEPS}"
    --reference-perceptual-loss-interval "${REFERENCE_PERCEPTUAL_LOSS_INTERVAL}"
    --reference-perceptual-loss-min-sigma "${REFERENCE_PERCEPTUAL_LOSS_MIN_SIGMA}"
    --reference-perceptual-loss-max-sigma "${REFERENCE_PERCEPTUAL_LOSS_MAX_SIGMA}"
    --reference-grad-ratio-interval "${REFERENCE_GRAD_RATIO_INTERVAL}"
  )
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

TRAIN_CHECKPOINT_ARGS=()
if [[ -n "${CONTROLNET_CHECKPOINT}" ]]; then
  if [[ "${RESOLVE_CONTROLNET_CHECKPOINT_LATEST}" == "1" && -d "${CONTROLNET_CHECKPOINT}" ]]; then
    if [[ ! -f "${CONTROLNET_CHECKPOINT}/config.json" && ! -f "${CONTROLNET_CHECKPOINT}/diffusion_pytorch_model.safetensors" ]]; then
      LATEST_CONTROLNET_CHECKPOINT="$(find "${CONTROLNET_CHECKPOINT}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -exec basename {} \; 2>/dev/null | sort -V | tail -n 1)"
      if [[ -n "${LATEST_CONTROLNET_CHECKPOINT}" ]]; then
        CONTROLNET_CHECKPOINT="${CONTROLNET_CHECKPOINT}/${LATEST_CONTROLNET_CHECKPOINT}"
      fi
    fi
  fi
  TRAIN_CHECKPOINT_ARGS+=(--controlnet_model_name_or_path "${CONTROLNET_CHECKPOINT}")
  if [[ "${LOAD_CONDITIONING_FROM_CHECKPOINT}" == "1" ]]; then
    if [[ -n "${CONDITIONING_CHECKPOINT}" ]]; then
      TRAIN_CHECKPOINT_ARGS+=(--conditioning-checkpoint "${CONDITIONING_CHECKPOINT}")
    fi
    TRAIN_CHECKPOINT_ARGS+=(--load-conditioning-from-checkpoint)
  fi
fi

echo "Cross V2.1 warm-start ControlNet checkpoint: ${CONTROLNET_CHECKPOINT:-<none>}"
echo "Cross V2.1 conditioning checkpoint: ${CONDITIONING_CHECKPOINT:-<none>} load=${LOAD_CONDITIONING_FROM_CHECKPOINT}"
echo "Cross V2.1 output dir: ${CROSS_V2_1_OUTPUT_DIR}"
echo "Cross V2.1 reference perceptual: backend=${REFERENCE_PERCEPTUAL_BACKEND} vgg_weights=${REFERENCE_VGG_WEIGHTS} vgg_weights_path=${REFERENCE_VGG_WEIGHTS_PATH:-<torchvision>} vgg_layers=${REFERENCE_VGG_LAYERS} vgg_loss_type=${REFERENCE_VGG_LOSS_TYPE} same_wsi=${SAME_WSI_APPEARANCE_CHECKPOINT:-<none>} uni=${UNI_CHECKPOINT_PATH:-<none>} weight=${REFERENCE_PERCEPTUAL_LOSS_WEIGHT} sigma=[${REFERENCE_PERCEPTUAL_LOSS_MIN_SIGMA},${REFERENCE_PERCEPTUAL_LOSS_MAX_SIGMA}]"
echo "Cross V2.1 reference grad-ratio interval: ${REFERENCE_GRAD_RATIO_INTERVAL}"

accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" --gpu_ids="${GPU_IDS}" \
  controlnet_train/cli/train_controlnet_flux_cross_v2_1.py \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --train-metadata "${CROSS_META}" \
  "${TRAIN_HED_ARGS[@]}" \
  "${TRAIN_CHECKPOINT_ARGS[@]}" \
  --output-dir "${CROSS_V2_1_OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}" \
  --logging-dir logs \
  --seed 42 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 10 \
  --max-train-steps "${MAX_TRAIN_STEPS}" \
  --self-reconstruction-warmup-steps "${SELF_RECONSTRUCTION_WARMUP_STEPS}" \
  --self-reconstruction-sample-prob "${SELF_RECONSTRUCTION_SAMPLE_PROB}" \
  "${TRAIN_REFERENCE_REGION_ARGS[@]}" \
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
  --tracker-project-name flux_controlnet_phase5_cross_v2_1_nohed_selfrec \
  --prompt-source dataset
