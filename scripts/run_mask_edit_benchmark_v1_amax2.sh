#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CPU_ID="${BENCHMARK_CPU_ID:-120}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/zhao/wqx/benchmark_v1}"
MODE="${MODE:-gt}"
INTENTS="${INTENTS:-$ROOT/runs/benchmark_v1/manifests/mask_semantic_intents.jsonl}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
PROMPTS="${PROMPTS:-$ROOT/runs/benchmark_v1/manifests/prompts/benchmark_prompts.accepted.csv}"
RUN_NAME="${RUN_NAME:-mask_semantic_${MODE}}"
OUTPUT="$OUTPUT_ROOT/$RUN_NAME"

if [[ -n "${BENCHMARK_ENV_FILE:-}" ]]; then
  set -a
  source "$BENCHMARK_ENV_FILE"
  set +a
fi

MAX_LOAD_PER_CPU="${MAX_LOAD_PER_CPU:-0.50}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-64}"
MIN_OUTPUT_FREE_GIB="${MIN_OUTPUT_FREE_GIB:-500}"

mkdir -p "$OUTPUT_ROOT"

read -r load1 _ < /proc/loadavg
cpus=$(nproc)
available_bytes=$(free -b | awk '/^Mem:/ {print $7}')
output_free_bytes=$(df -PB1 "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')

awk -v load="$load1" -v cpus="$cpus" -v limit="$MAX_LOAD_PER_CPU" 'BEGIN { exit !(load / cpus <= limit) }' || {
  echo "Health guard: load1=$load1 across $cpus CPUs exceeds ratio $MAX_LOAD_PER_CPU" >&2
  exit 3
}
(( available_bytes >= MIN_AVAILABLE_GIB * 1024 * 1024 * 1024 )) || {
  echo "Health guard: available memory is below ${MIN_AVAILABLE_GIB} GiB" >&2
  exit 3
}
(( output_free_bytes >= MIN_OUTPUT_FREE_GIB * 1024 * 1024 * 1024 )) || {
  echo "Health guard: output filesystem has less than ${MIN_OUTPUT_FREE_GIB} GiB free" >&2
  exit 3
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

args=(
  -m phase3_mask_edit.cli.run_mask_edit_benchmark
  --intents "$INTENTS"
  --output "$OUTPUT"
  --modes "$MODE"
  --contour-provider api-vision
  --contour-model "${CONTOUR_MODEL:-gpt-4.1-mini}"
  --contour-api-base-url "${CONTOUR_API_BASE_URL:-https://api.cursorai.art/v1}"
  --contour-api-key-env "${CONTOUR_API_KEY_ENV:-VISION_API_KEY}"
  --api-image-detail high
  --max-attempts "${MAX_ATTEMPTS:-10}"
  --coordinate-tolerance-px "${COORDINATE_TOLERANCE_PX:-16}"
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-2000}"
  --seed "${SEED:-13}"
  --checkpoint-every "${CHECKPOINT_EVERY:-25}"
  --resume
)

if [[ -n "$SAMPLE_IDS" ]]; then
  args+=(--sample-ids "$SAMPLE_IDS")
fi

if [[ "$MODE" != "gt" ]]; then
  args+=(
    --prompts "$PROMPTS"
    --parser-model "${PARSER_MODEL:-gpt-4.1-mini}"
    --parser-api-base-url "${PARSER_API_BASE_URL:-https://api.cursorai.art/v1}"
    --parser-api-key-env "${PARSER_API_KEY_ENV:-VISION_API_KEY}"
  )
fi

cd "$ROOT"
exec nice -n 15 ionice -c 3 taskset -c "$CPU_ID" "$PYTHON" "${args[@]}"
