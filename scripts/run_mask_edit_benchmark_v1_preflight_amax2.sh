#!/usr/bin/env bash
set -euo pipefail
umask 027

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CPU_ID="${BENCHMARK_CPU_ID:-120}"
SOURCE_INTENTS="${SOURCE_INTENTS:-$ROOT/runs/benchmark_v1/manifests/mask_semantic_intents.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/zhao/wqx/benchmark_v1/preflight_20260713}"
PREFLIGHT_INTENTS="$OUTPUT_ROOT/intents/preflight_intents.jsonl"
PROMPT_DIR="$OUTPUT_ROOT/prompts"
PROMPTS="$PROMPT_DIR/benchmark_prompts.accepted.csv"

API_BASE_URL="${API_BASE_URL:-https://api.cursorai.art/v1}"
API_MODEL="${API_MODEL:-gpt-4.1-mini}"
API_KEY_ENV="${API_KEY_ENV:-VISION_API_KEY}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-500}"
SEED="${SEED:-13}"

MAX_LOAD_PER_CPU="${MAX_LOAD_PER_CPU:-0.50}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-64}"
MIN_OUTPUT_FREE_GIB="${MIN_OUTPUT_FREE_GIB:-500}"
HEALTH_CHECK_SECONDS="${HEALTH_CHECK_SECONDS:-10}"
HEALTHY_STREAK_REQUIRED="${HEALTHY_STREAK_REQUIRED:-3}"
MAX_HEALTH_WAIT_SECONDS="${MAX_HEALTH_WAIT_SECONDS:-1800}"

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Required API key environment variable is not set: $API_KEY_ENV" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

run_low() {
  nice -n 15 ionice -c 3 taskset -c "$CPU_ID" "$PYTHON" "$@"
}

wait_for_health() {
  local stage="$1"
  local started now healthy_streak load1 cpus available_bytes output_free_bytes
  started=$(date +%s)
  healthy_streak=0
  while true; do
    read -r load1 _ < /proc/loadavg
    cpus=$(nproc --all)
    available_bytes=$(free -b | awk '/^Mem:/ {print $7}')
    output_free_bytes=$(df -PB1 "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
    if awk -v load="$load1" -v cpus="$cpus" -v limit="$MAX_LOAD_PER_CPU" \
        -v available="$available_bytes" -v min_available="$((MIN_AVAILABLE_GIB * 1024 * 1024 * 1024))" \
        -v output_free="$output_free_bytes" -v min_output="$((MIN_OUTPUT_FREE_GIB * 1024 * 1024 * 1024))" \
        'BEGIN { exit !((load / cpus <= limit) && (available >= min_available) && (output_free >= min_output)) }'; then
      healthy_streak=$((healthy_streak + 1))
    else
      healthy_streak=0
    fi
    echo "health stage=$stage load1=$load1 cpus=$cpus available_gib=$((available_bytes / 1024 / 1024 / 1024)) output_free_gib=$((output_free_bytes / 1024 / 1024 / 1024)) streak=$healthy_streak/$HEALTHY_STREAK_REQUIRED"
    if (( healthy_streak >= HEALTHY_STREAK_REQUIRED )); then
      return 0
    fi
    now=$(date +%s)
    if (( now - started >= MAX_HEALTH_WAIT_SECONDS )); then
      echo "Health guard timed out before stage: $stage" >&2
      return 3
    fi
    sleep "$HEALTH_CHECK_SECONDS"
  done
}

echo "preflight_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait_for_health select
run_low -m phase3_mask_edit.cli.select_mask_edit_benchmark_preflight \
  --intents "$SOURCE_INTENTS" \
  --output "$PREFLIGHT_INTENTS" \
  --per-cell 1 \
  --seed "$SEED" \
  --print-summary

wait_for_health prompts
run_low -m phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts \
  --intents "$PREFLIGHT_INTENTS" \
  --output "$PROMPT_DIR" \
  --use-llm-generator \
  --generator-model "$API_MODEL" \
  --generator-api-base-url "$API_BASE_URL" \
  --generator-api-key-env "$API_KEY_ENV" \
  --use-llm-checker \
  --checker-model "$API_MODEL" \
  --checker-api-base-url "$API_BASE_URL" \
  --checker-api-key-env "$API_KEY_ENV" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --max-retries 3 \
  --checker-repair-attempts 2 \
  --retry-rejected \
  --resume \
  --print-summary

run_mode() {
  local mode="$1"
  local output="$OUTPUT_ROOT/runs/$mode"
  local args=(
    -m phase3_mask_edit.cli.run_mask_edit_benchmark
    --intents "$PREFLIGHT_INTENTS"
    --output "$output"
    --modes "$mode"
    --contour-provider api-vision
    --contour-model "$API_MODEL"
    --contour-api-base-url "$API_BASE_URL"
    --contour-api-key-env "$API_KEY_ENV"
    --api-image-detail high
    --max-attempts "$MAX_ATTEMPTS"
    --coordinate-tolerance-px 16
    --bootstrap-iterations "$BOOTSTRAP_ITERATIONS"
    --seed "$SEED"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --resume
    --print-summary
  )
  if [[ "$mode" != "gt" ]]; then
    args+=(
      --prompts "$PROMPTS"
      --parser-model "$API_MODEL"
      --parser-api-base-url "$API_BASE_URL"
      --parser-api-key-env "$API_KEY_ENV"
    )
  fi
  wait_for_health "$mode"
  run_low "${args[@]}"
}

run_mode gt
run_mode prompt
run_mode instruction

touch "$OUTPUT_ROOT/PREFLIGHT_COMPLETE"
echo "preflight_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
