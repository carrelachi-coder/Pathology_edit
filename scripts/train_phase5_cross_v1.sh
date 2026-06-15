#!/bin/bash
set -euo pipefail

# Phase 5.3 Cross V1 global-soft-bias IP Phase 1.
# Default path: freeze ControlNet/conditioning with --a1-lite, train only the
# stats-token reference path with pure denoise loss, and keep fixed probe
# manifests explicit so following curves stay directly comparable.

GPU_IDS="${GPU_IDS:-1,2,4}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-/data/huggingface}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-/data/huggingface/FLUX.1-dev}"

# UNI2-h checkpoint — absolute path to pytorch_model.bin
UNI_CHECKPOINT="${UNI_CHECKPOINT:-${PROJECT_ROOT}/UNI-2h/pytorch_model.bin}"
CONCH_ROOT="${CONCH_ROOT:-${PROJECT_ROOT}/CONCH-main}"
CONCH_CHECKPOINT="${CONCH_CHECKPOINT:-${CONCH_ROOT}/checkpoints/conch/pytorch_model.bin}"
if [[ ! -f "${CONCH_CHECKPOINT}" && -f "${CONCH_ROOT}/checkpoints/pytorch_model.bin" ]]; then
  CONCH_CHECKPOINT="${CONCH_ROOT}/checkpoints/pytorch_model.bin"
fi

# Cross V1 training metadata (165K pairs, already built)
CROSS_META="${CROSS_META:-${PROJECT_ROOT}/phase5_runs/cross_meta/metadata_cross_train.json}"
SOURCE_CROSS_V1_OUTPUT_DIR="${SOURCE_CROSS_V1_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v1_expB_outputs_weak_hed}"
CONTROLNET_CHECKPOINT="${CONTROLNET_CHECKPOINT:-${SOURCE_CROSS_V1_OUTPUT_DIR}/checkpoint-66000}"
CONDITIONING_CHECKPOINT="${CONDITIONING_CHECKPOINT:-${SOURCE_CROSS_V1_OUTPUT_DIR}}"
CROSS_V1_OUTPUT_DIR="${CROSS_V1_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v1_global_soft_bias_stats_conch_region_loss}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${CROSS_V1_OUTPUT_DIR}/logs}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
CROSS_V1_IP_ARCHITECTURE="${CROSS_V1_IP_ARCHITECTURE:-global_soft_bias}"
REGIONAL_IP_ADAPTER="${REGIONAL_IP_ADAPTER:-1}"
REGIONAL_IP_STRICT="${REGIONAL_IP_STRICT:-0}"
REGIONAL_IP_TOKEN_MODE="${REGIONAL_IP_TOKEN_MODE:-stats}"
REGIONAL_IP_LABEL_MODE="${REGIONAL_IP_LABEL_MODE:-coarse-tissue}"
REGIONAL_IP_SOFT_BIAS_INIT="${REGIONAL_IP_SOFT_BIAS_INIT:-4.0}"
PROBE_SELECTION_MANIFEST="${PROBE_SELECTION_MANIFEST:-${SELECTION_MANIFEST:-}}"
A1_LITE="${A1_LITE:-1}"
LOAD_REF_ENCODER="${LOAD_REF_ENCODER:-1}"
LOAD_IP_ADAPTER="${LOAD_IP_ADAPTER:-0}"
IP_ADAPTER_CHECKPOINT="${IP_ADAPTER_CHECKPOINT:-}"
LOAD_SINGLE_IP_FROM_CHECKPOINT="${LOAD_SINGLE_IP_FROM_CHECKPOINT:-0}"
SKIP_REFERENCE_PERCEIVER="${SKIP_REFERENCE_PERCEIVER:-0}"
REFERENCE_PERCEIVER_CROSS_GATE_INIT="${REFERENCE_PERCEIVER_CROSS_GATE_INIT:-none}"

MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# Reference-usage sanity losses. Keep RGB style/swap off; use decoded-RGB -> frozen-CONCH
# region stats so the scorer is independent from the UNI stats-token injection path.
PERCEPTUAL_LOSS_WEIGHT="${PERCEPTUAL_LOSS_WEIGHT:-0}"
PERCEPTUAL_LOSS_INTERVAL="${PERCEPTUAL_LOSS_INTERVAL:-1}"
REFERENCE_REGION_LOSS_WEIGHT="${REFERENCE_REGION_LOSS_WEIGHT:-0.4}"
REFERENCE_REGION_LOSS_BACKEND="${REFERENCE_REGION_LOSS_BACKEND:-conch}"
REFERENCE_REGION_LOSS_INTERVAL="${REFERENCE_REGION_LOSS_INTERVAL:-4}"
REFERENCE_REGION_LOSS_MIN_SIGMA="${REFERENCE_REGION_LOSS_MIN_SIGMA:-0.0}"
REFERENCE_REGION_LOSS_MAX_SIGMA="${REFERENCE_REGION_LOSS_MAX_SIGMA:-0.6}"
REFERENCE_REGION_TISSUE_WEIGHT="${REFERENCE_REGION_TISSUE_WEIGHT:-1.0}"
REFERENCE_REGION_NUCLEI_WEIGHT="${REFERENCE_REGION_NUCLEI_WEIGHT:-0.25}"
REFERENCE_REGION_COMPOSITE_WEIGHT="${REFERENCE_REGION_COMPOSITE_WEIGHT:-0}"
REFERENCE_REGION_MEAN_WEIGHT="${REFERENCE_REGION_MEAN_WEIGHT:-1.0}"
REFERENCE_REGION_STD_WEIGHT="${REFERENCE_REGION_STD_WEIGHT:-0.5}"
REFERENCE_REGION_FFT_WEIGHT="${REFERENCE_REGION_FFT_WEIGHT:-0.25}"
REFERENCE_REGION_FFT_BINS="${REFERENCE_REGION_FFT_BINS:-6}"
REFERENCE_REGION_FFT_SIZE="${REFERENCE_REGION_FFT_SIZE:-64}"
REFERENCE_REGION_COSINE_WEIGHT="${REFERENCE_REGION_COSINE_WEIGHT:-0.25}"
REFERENCE_REGION_MIN_PIXELS="${REFERENCE_REGION_MIN_PIXELS:-32}"
REFERENCE_REGION_MIN_TOKENS="${REFERENCE_REGION_MIN_TOKENS:-2}"
REFERENCE_REGION_MAX_REGIONS_PER_SAMPLE="${REFERENCE_REGION_MAX_REGIONS_PER_SAMPLE:-}"
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
SELF_RECONSTRUCTION_SAMPLE_PROB="${SELF_RECONSTRUCTION_SAMPLE_PROB:-0}"
SELF_RECONSTRUCTION_L1_WEIGHT="${SELF_RECONSTRUCTION_L1_WEIGHT:-0}"
CONTROLNET_LEARNING_RATE="${CONTROLNET_LEARNING_RATE:-1e-6}"
CONTROLNET_TRAIN_MODE="${CONTROLNET_TRAIN_MODE:-outputs}"
CONTROLNET_TRAIN_X_EMBEDDER="${CONTROLNET_TRAIN_X_EMBEDDER:-0}"
CONTROLNET_TRAIN_LAST_N_BLOCKS="${CONTROLNET_TRAIN_LAST_N_BLOCKS:-0}"
CONTROLNET_TRAIN_LAST_N_SINGLE_BLOCKS="${CONTROLNET_TRAIN_LAST_N_SINGLE_BLOCKS:-0}"
CONDITIONING_LEARNING_RATE="${CONDITIONING_LEARNING_RATE:-5e-7}"
IP_REF_LEARNING_RATE="${IP_REF_LEARNING_RATE:-1e-4}"
IP_SINGLE_LEARNING_RATE="${IP_SINGLE_LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine_with_min_lr}"
LR_WARMUP_STEPS_REQUESTED="${LR_WARMUP_STEPS:-}"
LR_NUM_CYCLES="${LR_NUM_CYCLES:-0.5}"
LR_MIN_FACTOR="${LR_MIN_FACTOR:-0.1}"
LR_DECAY_START_STEP="${LR_DECAY_START_STEP:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_DEVICE="${EMA_DEVICE:-cpu}"
IP_SINGLE_NUM_LAYERS="${IP_SINGLE_NUM_LAYERS:-0}"
IP_HEALTH_DEBUG_INTERVAL="${IP_HEALTH_DEBUG_INTERVAL:-100}"
IP_HEALTH_DEBUG_WARMUP_STEPS="${IP_HEALTH_DEBUG_WARMUP_STEPS:-100}"
IP_HEALTH_MIN_REF_L2="${IP_HEALTH_MIN_REF_L2:-1e-6}"
IP_HEALTH_MIN_SWAP_LOSS_GAP="${IP_HEALTH_MIN_SWAP_LOSS_GAP:-0}"
IP_HEALTH_MAX_IP_RATIO="${IP_HEALTH_MAX_IP_RATIO:-1.0}"
IP_HEALTH_MIN_IP_RATIO="${IP_HEALTH_MIN_IP_RATIO:-1e-8}"
TEXT_DROPOUT_PROB="${TEXT_DROPOUT_PROB:-0.2}"
MASK_AUGMENTATION="${MASK_AUGMENTATION:-affine_coarse}"
MASK_AUGMENT_PROB="${MASK_AUGMENT_PROB:-1}"
MASK_AUGMENT_TRANSLATE="${MASK_AUGMENT_TRANSLATE:-0.03}"
MASK_AUGMENT_SCALE="${MASK_AUGMENT_SCALE:-0.04}"
MASK_AUGMENT_ROTATE_DEGREES="${MASK_AUGMENT_ROTATE_DEGREES:-3.0}"
MASK_AUGMENT_BOUNDARY_JITTER="${MASK_AUGMENT_BOUNDARY_JITTER:-0}"
MASK_AUGMENT_BOUNDARY_GRID="${MASK_AUGMENT_BOUNDARY_GRID:-8}"
MASK_AUGMENT_COARSE_PROB="${MASK_AUGMENT_COARSE_PROB:-0.5}"
MASK_AUGMENT_COARSE_FACTOR="${MASK_AUGMENT_COARSE_FACTOR:-4}"
STAIN_AUGMENTATION="${STAIN_AUGMENTATION:-none}"
STAIN_COUNTERFACTUAL_PROB="${STAIN_COUNTERFACTUAL_PROB:-0}"
NOISING_DEGRADATION="${NOISING_DEGRADATION:-none}"
DEGRADED_NOISING_MIN_SIGMA="${DEGRADED_NOISING_MIN_SIGMA:-0}"
TEXTURE_BLUR_PROB="${TEXTURE_BLUR_PROB:-0.7}"
TEXTURE_BLUR_SIGMA_MIN="${TEXTURE_BLUR_SIGMA_MIN:-0.4}"
TEXTURE_BLUR_SIGMA_MAX="${TEXTURE_BLUR_SIGMA_MAX:-1.2}"
TEXTURE_DOWNSAMPLE_PROB="${TEXTURE_DOWNSAMPLE_PROB:-0.7}"
TEXTURE_DOWNSAMPLE_SCALE_MIN="${TEXTURE_DOWNSAMPLE_SCALE_MIN:-0.45}"
TEXTURE_DOWNSAMPLE_SCALE_MAX="${TEXTURE_DOWNSAMPLE_SCALE_MAX:-0.8}"
TEXTURE_NOISE_PROB="${TEXTURE_NOISE_PROB:-0.25}"
TEXTURE_NOISE_STD_MIN="${TEXTURE_NOISE_STD_MIN:-0.005}"
TEXTURE_NOISE_STD_MAX="${TEXTURE_NOISE_STD_MAX:-0.02}"
HED_SIGMA="${HED_SIGMA:-0.1}"
HED_BETA="${HED_BETA:-0.01}"
HED_STRONG_ALPHA_SAMPLING="${HED_STRONG_ALPHA_SAMPLING:-0}"
HED_ALPHA_MIN="${HED_ALPHA_MIN:-0.8}"
HED_ALPHA_LOW="${HED_ALPHA_LOW:-0.95}"
HED_ALPHA_HIGH="${HED_ALPHA_HIGH:-1.05}"
HED_ALPHA_MAX="${HED_ALPHA_MAX:-1.2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"

if [[ -z "${TRAIN_LOG_FILE}" ]]; then
  TRAIN_LOG_FILE="${TRAIN_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
fi

if [[ -n "${LR_WARMUP_STEPS_REQUESTED}" ]]; then
  LR_WARMUP_STEPS="${LR_WARMUP_STEPS_REQUESTED}"
elif [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  LR_WARMUP_STEPS=0
else
  LR_WARMUP_STEPS=500
fi
mkdir -p "${TRAIN_LOG_DIR}" "$(dirname "${TRAIN_LOG_FILE}")"
ln -sfn "${TRAIN_LOG_FILE}" "${TRAIN_LOG_DIR}/latest.log"
echo "Writing console log to ${TRAIN_LOG_FILE}"
exec > >(tee -a "${TRAIN_LOG_FILE}") 2>&1
echo "==== Phase 5 Cross V1 run started $(date '+%Y-%m-%d %H:%M:%S') ===="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CROSS_V1_OUTPUT_DIR=${CROSS_V1_OUTPUT_DIR}"
echo "CONTROLNET_CHECKPOINT=${CONTROLNET_CHECKPOINT}"
echo "CONDITIONING_CHECKPOINT=${CONDITIONING_CHECKPOINT}"
echo "UNI_CHECKPOINT=${UNI_CHECKPOINT}"
echo "CONCH_ROOT=${CONCH_ROOT}"
echo "CONCH_CHECKPOINT=${CONCH_CHECKPOINT}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "LOAD_IP_ADAPTER=${LOAD_IP_ADAPTER}"
echo "IP_ADAPTER_CHECKPOINT=${IP_ADAPTER_CHECKPOINT:-<auto>}"
echo "CROSS_V1_IP_ARCHITECTURE=${CROSS_V1_IP_ARCHITECTURE}"
echo "REGIONAL_IP_TOKEN_MODE=${REGIONAL_IP_TOKEN_MODE}"
echo "REGIONAL_IP_LABEL_MODE=${REGIONAL_IP_LABEL_MODE}"
echo "REGIONAL_IP_SOFT_BIAS_INIT=${REGIONAL_IP_SOFT_BIAS_INIT}"
echo "PROBE_SELECTION_MANIFEST=${PROBE_SELECTION_MANIFEST:-<unset>}"
if [[ -n "${PROBE_SELECTION_MANIFEST}" ]]; then
  echo "Frozen probe set: using existing 128 held-out selection manifest ${PROBE_SELECTION_MANIFEST}"
else
  echo "Frozen probe set: PROBE_SELECTION_MANIFEST is unset; diagnostics must pass --selection-manifest explicitly." >&2
fi
echo "REFERENCE_PERCEIVER_CROSS_GATE_INIT=${REFERENCE_PERCEIVER_CROSS_GATE_INIT}"
echo "TEXT_DROPOUT_PROB=${TEXT_DROPOUT_PROB}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "EFFECTIVE_BATCH_SIZE=$((TRAIN_BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))"
echo "LR_SCHEDULER=${LR_SCHEDULER}"
echo "LR_WARMUP_STEPS=${LR_WARMUP_STEPS}"
echo "LR_NUM_CYCLES=${LR_NUM_CYCLES}"
echo "LR_MIN_FACTOR=${LR_MIN_FACTOR}"
echo "LR_DECAY_START_STEP=${LR_DECAY_START_STEP}"
echo "EMA_DECAY=${EMA_DECAY}"
echo "EMA_DEVICE=${EMA_DEVICE}"
echo "REFERENCE_REGION_LOSS_WEIGHT=${REFERENCE_REGION_LOSS_WEIGHT}"
echo "REFERENCE_REGION_LOSS_BACKEND=${REFERENCE_REGION_LOSS_BACKEND}"
echo "REFERENCE_REGION_LOSS_INTERVAL=${REFERENCE_REGION_LOSS_INTERVAL}"
echo "REFERENCE_REGION_LOSS_SIGMA=[${REFERENCE_REGION_LOSS_MIN_SIGMA},${REFERENCE_REGION_LOSS_MAX_SIGMA}]"
echo "MASK_AUGMENTATION=${MASK_AUGMENTATION}"
echo "MASK_AUGMENT_BOUNDARY_JITTER=${MASK_AUGMENT_BOUNDARY_JITTER}"
echo "MASK_AUGMENT_COARSE_PROB=${MASK_AUGMENT_COARSE_PROB}"
echo "TRAIN_LOG_FILE=${TRAIN_LOG_FILE}"

RESUME_CHECKPOINT_PATH=""
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ "${RESUME_FROM_CHECKPOINT}" == "latest" ]]; then
    RESUME_CHECKPOINT_PATH="$(find "${CROSS_V1_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  elif [[ "${RESUME_FROM_CHECKPOINT}" = /* ]]; then
    RESUME_CHECKPOINT_PATH="${RESUME_FROM_CHECKPOINT}"
  else
    RESUME_CHECKPOINT_PATH="${CROSS_V1_OUTPUT_DIR}/${RESUME_FROM_CHECKPOINT}"
  fi
  if [[ -z "${RESUME_CHECKPOINT_PATH}" || ! -d "${RESUME_CHECKPOINT_PATH}" ]]; then
    echo "ERROR: RESUME_FROM_CHECKPOINT does not exist: ${RESUME_CHECKPOINT_PATH:-${RESUME_FROM_CHECKPOINT}}" >&2
    echo "Available checkpoints under ${CROSS_V1_OUTPUT_DIR}:" >&2
    find "${CROSS_V1_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 20 >&2 || true
    exit 2
  fi
  echo "RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH}"
fi

if [[ "${LOAD_IP_ADAPTER}" == "1" && -z "${IP_ADAPTER_CHECKPOINT}" && -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  if [[ -f "${RESUME_CHECKPOINT_PATH}/phase5_ip_adapter.pt" ]]; then
    IP_ADAPTER_CHECKPOINT="${RESUME_CHECKPOINT_PATH}"
    echo "Resolved IP_ADAPTER_CHECKPOINT from resume checkpoint: ${IP_ADAPTER_CHECKPOINT}"
  else
    echo "Warning: resume checkpoint has no phase5_ip_adapter.pt; IP-Adapter will fall back to CONTROLNET_CHECKPOINT." >&2
  fi
fi

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
    IP_SINGLE_NUM_LAYERS="$(resolve_saved_single_ip_layers "${IP_ADAPTER_CHECKPOINT:-${CONTROLNET_CHECKPOINT}}")"
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
  --noising-degradation "${NOISING_DEGRADATION}"
  --degraded-noising-min-sigma "${DEGRADED_NOISING_MIN_SIGMA}"
  --texture-blur-prob "${TEXTURE_BLUR_PROB}"
  --texture-blur-sigma-min "${TEXTURE_BLUR_SIGMA_MIN}"
  --texture-blur-sigma-max "${TEXTURE_BLUR_SIGMA_MAX}"
  --texture-downsample-prob "${TEXTURE_DOWNSAMPLE_PROB}"
  --texture-downsample-scale-min "${TEXTURE_DOWNSAMPLE_SCALE_MIN}"
  --texture-downsample-scale-max "${TEXTURE_DOWNSAMPLE_SCALE_MAX}"
  --texture-noise-prob "${TEXTURE_NOISE_PROB}"
  --texture-noise-std-min "${TEXTURE_NOISE_STD_MIN}"
  --texture-noise-std-max "${TEXTURE_NOISE_STD_MAX}"
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
if [[ -n "${CONDITIONING_CHECKPOINT}" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(--conditioning-checkpoint "${CONDITIONING_CHECKPOINT}")
fi
if [[ "${LOAD_REF_ENCODER}" == "1" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(--load-ref-encoder-from-checkpoint)
fi
if [[ "${LOAD_IP_ADAPTER}" == "0" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(--no-load-ip-adapter-from-controlnet)
fi
if [[ "${LOAD_IP_ADAPTER}" == "1" && -n "${IP_ADAPTER_CHECKPOINT}" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(--ip-adapter-checkpoint "${IP_ADAPTER_CHECKPOINT}")
fi
if [[ "${LOAD_SINGLE_IP_FROM_CHECKPOINT}" == "1" ]]; then
  TRAIN_CHECKPOINT_ARGS+=(--load-single-ip-from-checkpoint)
fi

TRAIN_MODE_ARGS=(--cross-v1-ip-architecture "${CROSS_V1_IP_ARCHITECTURE}")
if [[ "${A1_LITE}" == "1" ]]; then
  TRAIN_MODE_ARGS+=(--a1-lite)
fi
if [[ "${REGIONAL_IP_ADAPTER}" == "1" ]]; then
  TRAIN_MODE_ARGS+=(
    --regional-ip-adapter
    --regional-ip-token-mode "${REGIONAL_IP_TOKEN_MODE}"
    --regional-ip-label-mode "${REGIONAL_IP_LABEL_MODE}"
    --regional-ip-soft-bias-init "${REGIONAL_IP_SOFT_BIAS_INIT}"
  )
fi
if [[ "${REGIONAL_IP_STRICT}" != "1" ]]; then
  TRAIN_MODE_ARGS+=(--no-regional-ip-strict)
fi
if [[ "${SKIP_REFERENCE_PERCEIVER}" == "1" ]]; then
  TRAIN_MODE_ARGS+=(--skip-reference-perceiver)
fi
if [[ -n "${REFERENCE_PERCEIVER_CROSS_GATE_INIT}" && "${REFERENCE_PERCEIVER_CROSS_GATE_INIT}" != "none" ]]; then
  TRAIN_MODE_ARGS+=(--reference-perceiver-cross-gate-init "${REFERENCE_PERCEIVER_CROSS_GATE_INIT}")
fi

TRAIN_REGION_LOSS_ARGS=(
  --reference-region-loss-weight "${REFERENCE_REGION_LOSS_WEIGHT}"
  --reference-region-loss-backend "${REFERENCE_REGION_LOSS_BACKEND}"
  --reference-region-loss-interval "${REFERENCE_REGION_LOSS_INTERVAL}"
  --reference-region-loss-min-sigma "${REFERENCE_REGION_LOSS_MIN_SIGMA}"
  --reference-region-loss-max-sigma "${REFERENCE_REGION_LOSS_MAX_SIGMA}"
  --reference-region-tissue-weight "${REFERENCE_REGION_TISSUE_WEIGHT}"
  --reference-region-nuclei-weight "${REFERENCE_REGION_NUCLEI_WEIGHT}"
  --reference-region-composite-weight "${REFERENCE_REGION_COMPOSITE_WEIGHT}"
  --reference-region-mean-weight "${REFERENCE_REGION_MEAN_WEIGHT}"
  --reference-region-std-weight "${REFERENCE_REGION_STD_WEIGHT}"
  --reference-region-fft-weight "${REFERENCE_REGION_FFT_WEIGHT}"
  --reference-region-fft-bins "${REFERENCE_REGION_FFT_BINS}"
  --reference-region-fft-size "${REFERENCE_REGION_FFT_SIZE}"
  --reference-region-cosine-weight "${REFERENCE_REGION_COSINE_WEIGHT}"
  --reference-region-min-pixels "${REFERENCE_REGION_MIN_PIXELS}"
  --reference-region-min-tokens "${REFERENCE_REGION_MIN_TOKENS}"
)
if [[ -n "${REFERENCE_REGION_MAX_REGIONS_PER_SAMPLE}" ]]; then
  TRAIN_REGION_LOSS_ARGS+=(--reference-region-max-regions-per-sample "${REFERENCE_REGION_MAX_REGIONS_PER_SAMPLE}")
fi

accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" --gpu_ids="${GPU_IDS}" \
  controlnet_train/cli/train_controlnet_flux_cross_v1.py \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --train-metadata "${CROSS_META}" \
  "${TRAIN_MODE_ARGS[@]}" \
  "${TRAIN_HED_ARGS[@]}" \
  --uni-checkpoint-path "${UNI_CHECKPOINT}" \
  --conch-root "${CONCH_ROOT}" \
  --conch-checkpoint-path "${CONCH_CHECKPOINT}" \
  "${TRAIN_CHECKPOINT_ARGS[@]}" \
  --output-dir "${CROSS_V1_OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}" \
  --logging-dir logs \
  --seed 42 \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
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
  --lr-scheduler "${LR_SCHEDULER}" \
  --lr-warmup-steps "${LR_WARMUP_STEPS}" \
  --lr-num-cycles "${LR_NUM_CYCLES}" \
  --lr-min-factor "${LR_MIN_FACTOR}" \
  --lr-decay-start-step "${LR_DECAY_START_STEP}" \
  --ema-decay "${EMA_DECAY}" \
  --ema-device "${EMA_DEVICE}" \
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
  --ip-health-debug-interval "${IP_HEALTH_DEBUG_INTERVAL}" \
  --ip-health-debug-warmup-steps "${IP_HEALTH_DEBUG_WARMUP_STEPS}" \
  --ip-health-min-ref-l2 "${IP_HEALTH_MIN_REF_L2}" \
  --ip-health-min-swap-loss-gap "${IP_HEALTH_MIN_SWAP_LOSS_GAP}" \
  --ip-health-max-ip-ratio "${IP_HEALTH_MAX_IP_RATIO}" \
  --ip-health-min-ip-ratio "${IP_HEALTH_MIN_IP_RATIO}" \
  --cross-v1-spatial-mode reference_target \
  --disable-reference-perceiver-self-attn \
  --perceptual-loss-weight "${PERCEPTUAL_LOSS_WEIGHT}" \
  "${TRAIN_REGION_LOSS_ARGS[@]}" \
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
  --proportion-empty-prompts "${TEXT_DROPOUT_PROB}" \
  --mask-augmentation "${MASK_AUGMENTATION}" \
  --mask-augment-prob "${MASK_AUGMENT_PROB}" \
  --mask-augment-translate "${MASK_AUGMENT_TRANSLATE}" \
  --mask-augment-scale "${MASK_AUGMENT_SCALE}" \
  --mask-augment-rotate-degrees "${MASK_AUGMENT_ROTATE_DEGREES}" \
  --mask-augment-boundary-jitter "${MASK_AUGMENT_BOUNDARY_JITTER}" \
  --mask-augment-boundary-grid "${MASK_AUGMENT_BOUNDARY_GRID}" \
  --mask-augment-coarse-prob "${MASK_AUGMENT_COARSE_PROB}" \
  --mask-augment-coarse-factor "${MASK_AUGMENT_COARSE_FACTOR}" \
  --report-to tensorboard \
  --tracker-project-name flux_controlnet_phase5_cross_v1_global_soft_bias_conch_region_loss \
  --prompt-source dataset
