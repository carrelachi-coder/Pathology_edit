#!/bin/bash
set -euo pipefail

# Phase 5.3 Cross V1 Experiment B continuation.
# Load the 20k Cross V1 checkpoint, train only ControlNet residual output
# projections plus spatial conditioning/IP/ref, and continue with weak HED
# counterfactual + sparse self-reconstruction. Do not add swap/style/single-stream IP.

GPU_IDS="${GPU_IDS:-1,2,4}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export HF_HOME="${HF_HOME:-/data/huggingface}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-/data/huggingface/FLUX.1-dev}"

# UNI2-h checkpoint — absolute path to pytorch_model.bin
UNI_CHECKPOINT="${UNI_CHECKPOINT:-${PROJECT_ROOT}/UNI-2h/pytorch_model.bin}"

# Cross V1 training metadata (165K pairs, already built)
CROSS_META="${CROSS_META:-${PROJECT_ROOT}/phase5_runs/cross_meta/metadata_cross_train.json}"
SOURCE_CROSS_V1_OUTPUT_DIR="${SOURCE_CROSS_V1_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v1_skip_perceiver_20k}"
CONTROLNET_CHECKPOINT="${CONTROLNET_CHECKPOINT:-${SOURCE_CROSS_V1_OUTPUT_DIR}/checkpoint-20000}"
CROSS_V1_OUTPUT_DIR="${CROSS_V1_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v1_expB_outputs_weak_hed}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LOAD_REF_ENCODER="${LOAD_REF_ENCODER:-1}"
LOAD_IP_ADAPTER="${LOAD_IP_ADAPTER:-1}"
LOAD_SINGLE_IP_FROM_CHECKPOINT="${LOAD_SINGLE_IP_FROM_CHECKPOINT:-0}"

MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# Reference-usage sanity losses. Keep style/swap off for a clean attribution test.
PERCEPTUAL_LOSS_WEIGHT="${PERCEPTUAL_LOSS_WEIGHT:-0.5}"
PERCEPTUAL_LOSS_INTERVAL="${PERCEPTUAL_LOSS_INTERVAL:-1}"
REFERENCE_STYLE_LOSS_WEIGHT="${REFERENCE_STYLE_LOSS_WEIGHT:-0}"
REFERENCE_STYLE_TISSUE_WEIGHT="${REFERENCE_STYLE_TISSUE_WEIGHT:-1.0}"
REFERENCE_STYLE_NUCLEI_WEIGHT="${REFERENCE_STYLE_NUCLEI_WEIGHT:-1.0}"
REFERENCE_STYLE_MEAN_WEIGHT="${REFERENCE_STYLE_MEAN_WEIGHT:-1.0}"
REFERENCE_STYLE_STD_WEIGHT="${REFERENCE_STYLE_STD_WEIGHT:-1.0}"
REFERENCE_STYLE_COV_WEIGHT="${REFERENCE_STYLE_COV_WEIGHT:-0.25}"
REFERENCE_STYLE_MIN_PIXELS="${REFERENCE_STYLE_MIN_PIXELS:-32}"
REFERENCE_STYLE_LOSS_INTERVAL="${REFERENCE_STYLE_LOSS_INTERVAL:-4}"
REF_SWAP_LOSS_WEIGHT="${REF_SWAP_LOSS_WEIGHT:-0}"
REF_SWAP_MARGIN="${REF_SWAP_MARGIN:-0.02}"
REF_SWAP_VARIANTS="${REF_SWAP_VARIANTS:-random}"
REF_SWAP_LOSS_INTERVAL="${REF_SWAP_LOSS_INTERVAL:-2}"
SELF_RECONSTRUCTION_WARMUP_STEPS="${SELF_RECONSTRUCTION_WARMUP_STEPS:-0}"
SELF_RECONSTRUCTION_SAMPLE_PROB="${SELF_RECONSTRUCTION_SAMPLE_PROB:-0.15}"
SELF_RECONSTRUCTION_L1_WEIGHT="${SELF_RECONSTRUCTION_L1_WEIGHT:-0.05}"
CONTROLNET_LEARNING_RATE="${CONTROLNET_LEARNING_RATE:-1e-6}"
CONTROLNET_TRAIN_MODE="${CONTROLNET_TRAIN_MODE:-outputs}"
CONTROLNET_TRAIN_X_EMBEDDER="${CONTROLNET_TRAIN_X_EMBEDDER:-0}"
CONTROLNET_TRAIN_LAST_N_BLOCKS="${CONTROLNET_TRAIN_LAST_N_BLOCKS:-0}"
CONTROLNET_TRAIN_LAST_N_SINGLE_BLOCKS="${CONTROLNET_TRAIN_LAST_N_SINGLE_BLOCKS:-0}"
CONDITIONING_LEARNING_RATE="${CONDITIONING_LEARNING_RATE:-5e-7}"
IP_REF_LEARNING_RATE="${IP_REF_LEARNING_RATE:-1e-5}"
IP_SINGLE_LEARNING_RATE="${IP_SINGLE_LEARNING_RATE:-0}"
IP_SINGLE_NUM_LAYERS="${IP_SINGLE_NUM_LAYERS:-0}"
STAIN_AUGMENTATION="${STAIN_AUGMENTATION:-hed_aggressive}"
STAIN_COUNTERFACTUAL_PROB="${STAIN_COUNTERFACTUAL_PROB:-0.5}"
HED_SIGMA="${HED_SIGMA:-0.1}"
HED_BETA="${HED_BETA:-0.01}"
HED_STRONG_ALPHA_SAMPLING="${HED_STRONG_ALPHA_SAMPLING:-0}"
HED_ALPHA_MIN="${HED_ALPHA_MIN:-0.8}"
HED_ALPHA_LOW="${HED_ALPHA_LOW:-0.95}"
HED_ALPHA_HIGH="${HED_ALPHA_HIGH:-1.05}"
HED_ALPHA_MAX="${HED_ALPHA_MAX:-1.2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"

cd "${PROJECT_ROOT}"

resolve_saved_single_ip_layers() {
  local checkpoint_dir="$1"
  local state_path="${checkpoint_dir}/phase5_ip_adapter.pt"
  if [[ -z "${checkpoint_dir}" || ! -f "${state_path}" ]]; then
    echo "0"
    return
  fi
  "${PYTHON_BIN}" -c 'import sys
from pathlib import Path
import torch

state_path = Path(sys.argv[1])
try:
    state = torch.load(state_path, map_location="cpu", weights_only=True)
except TypeError:
    state = torch.load(state_path, map_location="cpu")

if "num_single_layers" in state:
    print(int(state["num_single_layers"]))
else:
    indices = {
        int(key.split("_")[2])
        for key in state
        if key.startswith("single_block_")
        and key.endswith(("_to_k_ip", "_to_v_ip"))
    }
    print(len(indices))
' "${state_path}" 2>/dev/null || echo "0"
}

if [[ "${IP_SINGLE_NUM_LAYERS}" == "auto" ]]; then
  if [[ "${LOAD_IP_ADAPTER}" == "1" ]]; then
    IP_SINGLE_NUM_LAYERS="$(resolve_saved_single_ip_layers "${CONTROLNET_CHECKPOINT}")"
  else
    IP_SINGLE_NUM_LAYERS=0
  fi
fi
if ! [[ "${IP_SINGLE_NUM_LAYERS}" =~ ^[0-9]+$ ]]; then
  echo "Invalid IP_SINGLE_NUM_LAYERS=${IP_SINGLE_NUM_LAYERS}; expected auto or a non-negative integer." >&2
  exit 2
fi
if [[ "${LOAD_SINGLE_IP_FROM_CHECKPOINT}" == "auto" ]]; then
  if [[ "${LOAD_IP_ADAPTER}" == "1" && "${IP_SINGLE_NUM_LAYERS}" -gt 0 ]]; then
    LOAD_SINGLE_IP_FROM_CHECKPOINT=1
  else
    LOAD_SINGLE_IP_FROM_CHECKPOINT=0
  fi
fi

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

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

TRAIN_CHECKPOINT_ARGS=()
if [[ -n "${CONTROLNET_CHECKPOINT}" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(
    --controlnet_model_name_or_path "${CONTROLNET_CHECKPOINT}"
    --load-conditioning-from-checkpoint
  )
fi
if [[ "${LOAD_REF_ENCODER}" == "1" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(--load-ref-encoder-from-checkpoint)
fi
if [[ "${LOAD_IP_ADAPTER}" == "0" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(--no-load-ip-adapter-from-controlnet)
fi
if [[ "${LOAD_SINGLE_IP_FROM_CHECKPOINT}" == "1" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(--load-single-ip-from-checkpoint)
fi

accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" --gpu_ids="${GPU_IDS}" \
  controlnet_train/cli/train_controlnet_flux_cross_v1.py \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --train-metadata "${CROSS_META}" \
  "${TRAIN_HED_ARGS[@]}" \
  --uni-checkpoint-path "${UNI_CHECKPOINT}" \
  "${TRAIN_CHECKPOINT_ARGS[@]}" \
  --output-dir "${CROSS_V1_OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}" \
  --logging-dir logs \
  --seed 42 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 10 \
  --max-train-steps "${MAX_TRAIN_STEPS}" \
  --self-reconstruction-warmup-steps "${SELF_RECONSTRUCTION_WARMUP_STEPS}" \
  --self-reconstruction-sample-prob "${SELF_RECONSTRUCTION_SAMPLE_PROB}" \
  --self-reconstruction-l1-weight "${SELF_RECONSTRUCTION_L1_WEIGHT}" \
  --learning-rate "${CONTROLNET_LEARNING_RATE}" \
  "${TRAIN_CONTROLNET_ARGS[@]}" \
  --conditioning-learning-rate "${CONDITIONING_LEARNING_RATE}" \
  --ip-ref-learning-rate "${IP_REF_LEARNING_RATE}" \
  --ip-single-learning-rate "${IP_SINGLE_LEARNING_RATE}" \
  --lr-scheduler constant_with_warmup \
  --lr-warmup-steps 500 \
  --checkpointing-steps "${CHECKPOINTING_STEPS}" \
  --checkpoints-total-limit 5 \
  --mixed-precision "${MIXED_PRECISION}" \
  "${TRAIN_OPTIMIZER_ARGS[@]}" \
  "${TRAIN_MEMORY_ARGS[@]}" \
  --allow-tf32 \
  "${TRAIN_DATALOADER_ARGS[@]}" \
  --num-double-layers 4 \
  --num-single-layers 4 \
  --ip-init-gain 0.1 \
  --ip-single-num-layers "${IP_SINGLE_NUM_LAYERS}" \
  --cross-v1-spatial-mode reference_target \
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
  --tracker-project-name flux_controlnet_phase5_cross_v1_expB_outputs_weak_hed \
  --prompt-source dataset
