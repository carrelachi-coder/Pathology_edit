#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CPU_ID="${BENCHMARK_CPU_ID:-120}"
MAX_LOAD_PER_CPU="${MAX_LOAD_PER_CPU:-0.50}"
INTENTS="${INTENTS:-$ROOT/runs/benchmark_v1/manifests/mask_semantic_intents.jsonl}"
OUTPUT="${OUTPUT:-$ROOT/runs/benchmark_v1/manifests/prompts}"

if [[ -n "${BENCHMARK_ENV_FILE:-}" ]]; then
  set -a
  source "$BENCHMARK_ENV_FILE"
  set +a
fi

read -r load1 _ < /proc/loadavg
cpus=$(nproc)
awk -v load="$load1" -v cpus="$cpus" -v limit="$MAX_LOAD_PER_CPU" 'BEGIN { exit !(load / cpus <= limit) }' || {
  echo "Health guard: load1=$load1 across $cpus CPUs exceeds ratio $MAX_LOAD_PER_CPU" >&2
  exit 3
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
exec nice -n 15 ionice -c 3 taskset -c "$CPU_ID" "$PYTHON" \
  -m phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts \
  --intents "$INTENTS" \
  --output "$OUTPUT" \
  --use-llm-generator \
  --generator-model "${GENERATOR_MODEL:-gpt-4.1-mini}" \
  --generator-api-base-url "${GENERATOR_API_BASE_URL:-https://api.cursorai.art/v1}" \
  --generator-api-key-env "${GENERATOR_API_KEY_ENV:-VISION_API_KEY}" \
  --use-llm-checker \
  --checker-model "${CHECKER_MODEL:-gpt-4.1-mini}" \
  --checker-api-base-url "${CHECKER_API_BASE_URL:-https://api.cursorai.art/v1}" \
  --checker-api-key-env "${CHECKER_API_KEY_ENV:-VISION_API_KEY}" \
  --manual-review-per-group "${MANUAL_REVIEW_PER_GROUP:-3}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  --resume \
  --print-summary
