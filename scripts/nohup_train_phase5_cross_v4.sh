#!/bin/bash
set -euo pipefail

# One-command background launcher for Phase 5.4 Cross V4 training.
# Usage:
#   bash scripts/nohup_train_phase5_cross_v4.sh
#
# Optional env overrides are forwarded to train_phase5_cross_v4.sh, for example:
#   GPU_IDS=1,2,4 MAX_TRAIN_STEPS=2000 bash scripts/nohup_train_phase5_cross_v4.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_phase5_cross_v4.sh}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
CROSS_V4_OUTPUT_DIR="${CROSS_V4_OUTPUT_DIR:-/data/wqx/flowedit/controlnet_cross_v4_mask_guided}"
LOG_ROOT="${LOG_ROOT:-${CROSS_V4_OUTPUT_DIR}/nohup_logs}"
RUN_NAME="${RUN_NAME:-cross_v4_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${RUN_NAME}.pid}"
MANIFEST_FILE="${MANIFEST_FILE:-${LOG_ROOT}/${RUN_NAME}.manifest.json}"

GPU_IDS="${GPU_IDS:-1,2,4}"
MAX_CUDA_MEMORY_GB="${MAX_CUDA_MEMORY_GB:-80}"
CROSS_V4_DIAGNOSE_STEPS="${CROSS_V4_DIAGNOSE_STEPS:-1,10,100,500,1000,1500,2000,3000,4000}"
CROSS_V4_DIAGNOSE_JSONL="${CROSS_V4_DIAGNOSE_JSONL:-${CROSS_V4_OUTPUT_DIR}/cross_v4_diagnostics.jsonl}"
SELF_RECONSTRUCTION_WARMUP_STEPS="${SELF_RECONSTRUCTION_WARMUP_STEPS:-0}"
SELF_RECONSTRUCTION_SAMPLE_PROB="${SELF_RECONSTRUCTION_SAMPLE_PROB:-0}"
REF_SWAP_LOSS_WEIGHT="${REF_SWAP_LOSS_WEIGHT:-0.05}"
REF_SWAP_LOSS_INTERVAL="${REF_SWAP_LOSS_INTERVAL:-100}"
REF_SWAP_VARIANTS="${REF_SWAP_VARIANTS:-same_class}"
CROSS_V4_BIAS_WARMUP_STEPS="${CROSS_V4_BIAS_WARMUP_STEPS:-1000}"
CROSS_V4_EXTREME_BIAS_SMOKE="${CROSS_V4_EXTREME_BIAS_SMOKE:-0}"
PAIR_DIFFICULTY_SAMPLER="${PAIR_DIFFICULTY_SAMPLER:-1}"
PAIR_DIFFICULTY_TARGET_FULL="${PAIR_DIFFICULTY_TARGET_FULL:-0.70}"
PAIR_DIFFICULTY_TARGET_PARTIAL="${PAIR_DIFFICULTY_TARGET_PARTIAL:-0.25}"
PAIR_DIFFICULTY_TARGET_LOW="${PAIR_DIFFICULTY_TARGET_LOW:-0.05}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Missing train script: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${CROSS_V4_OUTPUT_DIR}"

cat > "${MANIFEST_FILE}" <<EOF
{
  "run_name": "${RUN_NAME}",
  "repo_root": "${REPO_ROOT}",
  "project_root": "${PROJECT_ROOT}",
  "train_script": "${TRAIN_SCRIPT}",
  "output_dir": "${CROSS_V4_OUTPUT_DIR}",
  "log_file": "${LOG_FILE}",
  "pid_file": "${PID_FILE}",
  "diagnostics_jsonl": "${CROSS_V4_DIAGNOSE_JSONL}",
  "gpu_ids": "${GPU_IDS}",
  "max_cuda_memory_gb": "${MAX_CUDA_MEMORY_GB}",
  "diagnose_steps": "${CROSS_V4_DIAGNOSE_STEPS}",
  "self_reconstruction_warmup_steps": "${SELF_RECONSTRUCTION_WARMUP_STEPS}",
  "self_reconstruction_sample_prob": "${SELF_RECONSTRUCTION_SAMPLE_PROB}",
  "ref_swap_loss_weight": "${REF_SWAP_LOSS_WEIGHT}",
  "ref_swap_loss_interval": "${REF_SWAP_LOSS_INTERVAL}",
  "ref_swap_variants": "${REF_SWAP_VARIANTS}",
  "cross_v4_bias_warmup_steps": "${CROSS_V4_BIAS_WARMUP_STEPS}",
  "cross_v4_extreme_bias_smoke": "${CROSS_V4_EXTREME_BIAS_SMOKE}",
  "pair_difficulty_sampler": "${PAIR_DIFFICULTY_SAMPLER}",
  "pair_difficulty_target_full": "${PAIR_DIFFICULTY_TARGET_FULL}",
  "pair_difficulty_target_partial": "${PAIR_DIFFICULTY_TARGET_PARTIAL}",
  "pair_difficulty_target_low": "${PAIR_DIFFICULTY_TARGET_LOW}",
  "started_at": "$(date -Is)"
}
EOF

export PROJECT_ROOT
export CROSS_V4_OUTPUT_DIR
export GPU_IDS
export MAX_CUDA_MEMORY_GB
export CROSS_V4_DIAGNOSE_STEPS
export CROSS_V4_DIAGNOSE_JSONL
export SELF_RECONSTRUCTION_WARMUP_STEPS
export SELF_RECONSTRUCTION_SAMPLE_PROB
export REF_SWAP_LOSS_WEIGHT
export REF_SWAP_LOSS_INTERVAL
export REF_SWAP_VARIANTS
export CROSS_V4_BIAS_WARMUP_STEPS
export CROSS_V4_EXTREME_BIAS_SMOKE
export PAIR_DIFFICULTY_SAMPLER
export PAIR_DIFFICULTY_TARGET_FULL
export PAIR_DIFFICULTY_TARGET_PARTIAL
export PAIR_DIFFICULTY_TARGET_LOW

nohup bash "${TRAIN_SCRIPT}" > "${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started Cross V4 training with nohup."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Diagnostics: ${CROSS_V4_DIAGNOSE_JSONL}"
echo "Manifest: ${MANIFEST_FILE}"
echo "Tail: tail -f ${LOG_FILE}"
