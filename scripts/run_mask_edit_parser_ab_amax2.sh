#!/usr/bin/env bash
set -euo pipefail
umask 027

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CPU_ID="${BENCHMARK_CPU_ID:-120}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-/data1/zhao/wqx/benchmark_v1/preflight_20260713}"
INTENTS="${INTENTS:-$PREFLIGHT_ROOT/intents/preflight_intents.jsonl}"
PROMPTS="${PROMPTS:-$PREFLIGHT_ROOT/prompts/benchmark_prompts.accepted.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PREFLIGHT_ROOT/parser_ab_v0_2}"
API_BASE_URL="${API_BASE_URL:-https://api.cursorai.art/v1}"
API_KEY_ENV="${API_KEY_ENV:-VISION_API_KEY}"
MODELS=(
  "${MODEL_A:-gpt-4.1-mini}"
  "${MODEL_B:-gpt-5.4-mini-2026-03-17}"
)

EXPECTED_INTENTS_SHA="5c3e10b7f28be1968dfcaca04deae24f3d6550aaedee7bee036258cef1c6e04f"
EXPECTED_PROMPTS_SHA="9a8ed46555333c72a96916b8a21eac2efec0c0c5124a9fa8e9fae4aec6fce7bd"

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
cd "$ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

run_low() {
  nice -n 15 ionice -c 3 taskset -c "$CPU_ID" "$PYTHON" "$@"
}

verify_frozen_input() {
  local path="$1"
  local expected="$2"
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "Frozen input hash mismatch: $path expected=$expected actual=$actual" >&2
    exit 4
  fi
}

wait_for_health() {
  local stage="$1"
  local started now streak load1 cpus available_bytes output_free_bytes
  started=$(date +%s)
  streak=0
  while true; do
    read -r load1 _ < /proc/loadavg
    cpus=$(nproc --all)
    available_bytes=$(free -b | awk '/^Mem:/ {print $7}')
    output_free_bytes=$(df -PB1 "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
    if awk -v load="$load1" -v cpus="$cpus" -v limit="$MAX_LOAD_PER_CPU" \
        -v available="$available_bytes" -v min_available="$((MIN_AVAILABLE_GIB * 1024 * 1024 * 1024))" \
        -v output_free="$output_free_bytes" -v min_output="$((MIN_OUTPUT_FREE_GIB * 1024 * 1024 * 1024))" \
        'BEGIN { exit !((load / cpus <= limit) && (available >= min_available) && (output_free >= min_output)) }'; then
      streak=$((streak + 1))
    else
      streak=0
    fi
    echo "health stage=$stage load1=$load1 cpus=$cpus available_gib=$((available_bytes / 1024 / 1024 / 1024)) output_free_gib=$((output_free_bytes / 1024 / 1024 / 1024)) streak=$streak/$HEALTHY_STREAK_REQUIRED"
    if (( streak >= HEALTHY_STREAK_REQUIRED )); then
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

verify_frozen_input "$INTENTS" "$EXPECTED_INTENTS_SHA"
verify_frozen_input "$PROMPTS" "$EXPECTED_PROMPTS_SHA"

echo "parser_ab_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_model() {
  local model="$1"
  local model_slug output
  model_slug=${model//\//_}
  output="$OUTPUT_ROOT/$model_slug"
  wait_for_health "parser_ab_$model_slug"
  run_low -m phase3_mask_edit.cli.run_mask_edit_parser_ab \
    --intents "$INTENTS" \
    --prompts "$PROMPTS" \
    --output "$output" \
    --parser-model "$model" \
    --parser-api-base-url "$API_BASE_URL" \
    --parser-api-key-env "$API_KEY_ENV" \
    --modes prompt instruction \
    --checkpoint-every 1 \
    --resume \
    --print-summary
  touch "$output/PARSER_COMPLETE"
}

pids=()
for model in "${MODELS[@]}"; do
  model_slug=${model//\//_}
  mkdir -p "$OUTPUT_ROOT/$model_slug"
  run_model "$model" >"$OUTPUT_ROOT/$model_slug/runner.log" 2>&1 &
  pids+=("$!")
done

run_status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    run_status=1
  fi
done
if (( run_status != 0 )); then
  echo "One or more parser A/B arms failed; see per-model runner.log files." >&2
  exit 6
fi

run_low - "$OUTPUT_ROOT" "${MODELS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = sys.argv[2:]
reports = {}
for model in models:
    slug = model.replace("/", "_")
    reports[model] = json.loads((root / slug / "parser_report.json").read_text())
comparison = {
    "parser_only": True,
    "contour_calls": 0,
    "models": reports,
    "all_models_passed": all(report["gate"]["passed"] for report in reports.values()),
}
(root / "comparison.json").write_text(json.dumps(comparison, indent=2))
print(json.dumps(comparison, indent=2))
PY

touch "$OUTPUT_ROOT/AB_COMPLETE"
echo "parser_ab_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
