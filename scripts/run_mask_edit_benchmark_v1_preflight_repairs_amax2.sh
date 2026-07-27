#!/usr/bin/env bash
set -euo pipefail
umask 027

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CPU_ID="${BENCHMARK_CPU_ID:-120}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/zhao/wqx/benchmark_v1/preflight_20260713}"
INTENTS="$OUTPUT_ROOT/intents/preflight_intents.jsonl"
PROMPT_DIR="$OUTPUT_ROOT/prompts"
PROMPTS="$PROMPT_DIR/benchmark_prompts.accepted.csv"
REPAIR_ROOT="$OUTPUT_ROOT/repairs/v2"
RERUN_IDS="$REPAIR_ROOT/rerun_sample_ids.txt"
API_BASE_URL="${API_BASE_URL:-https://api.cursorai.art/v1}"
API_MODEL="${API_MODEL:-gpt-4.1-mini}"
API_KEY_ENV="${API_KEY_ENV:-VISION_API_KEY}"

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

mkdir -p "$REPAIR_ROOT"

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

cd "$ROOT"
echo "repair_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

run_low - "$INTENTS" "$OUTPUT_ROOT/runs" "$RERUN_IDS" <<'PY'
import csv
import sys
from pathlib import Path

from phase3_mask_edit.benchmark.models import read_intents_jsonl

intents_path = Path(sys.argv[1])
runs_root = Path(sys.argv[2])
output_path = Path(sys.argv[3])
selected = {
    item.sample_id
    for item in read_intents_jsonl(intents_path)
    if item.specialized or item.primitive == "intratumoral_immune_infiltration"
}
for mode in ("prompt", "instruction"):
    eval_path = runs_root / mode / "benchmark_eval_results.csv"
    with eval_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("status") != "completed":
                selected.add(str(row["sample_id"]))
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("".join(f"{sample_id}\n" for sample_id in sorted(selected)))
print(f"repair_sample_count={len(selected)}")
PY

specialized_primitives=(
  gleason_upgrade_3to4
  gleason_upgrade_4to5
  gleason_downgrade_4to3
  benign_to_gleason3
  benign_atrophy
  normal_to_adenomatous
  adenoma_to_carcinoma
  grade_upgrade
  treatment_dedifferentiation
)

if [[ ! -f "$REPAIR_ROOT/PROMPTS_COMPLETE" ]]; then
  wait_for_health repair_prompts
  run_low -m phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts \
    --intents "$INTENTS" \
    --output "$PROMPT_DIR" \
    --primitives "${specialized_primitives[@]}" \
    --use-llm-generator \
    --generator-model "$API_MODEL" \
    --generator-api-base-url "$API_BASE_URL" \
    --generator-api-key-env "$API_KEY_ENV" \
    --use-llm-checker \
    --checker-model "$API_MODEL" \
    --checker-api-base-url "$API_BASE_URL" \
    --checker-api-key-env "$API_KEY_ENV" \
    --checkpoint-every 1 \
    --max-retries 3 \
    --checker-repair-attempts 2 \
    --retry-existing \
    --resume \
    --print-summary
  touch "$REPAIR_ROOT/PROMPTS_COMPLETE"
fi

run_repair_mode() {
  local mode="$1"
  local output="$REPAIR_ROOT/$mode"
  if [[ -f "$REPAIR_ROOT/${mode^^}_COMPLETE" ]]; then
    return 0
  fi
  wait_for_health "repair_$mode"
  run_low -m phase3_mask_edit.cli.run_mask_edit_benchmark \
    --intents "$INTENTS" \
    --sample-ids "$RERUN_IDS" \
    --prompts "$PROMPTS" \
    --output "$output" \
    --modes "$mode" \
    --parser-model "$API_MODEL" \
    --parser-api-base-url "$API_BASE_URL" \
    --parser-api-key-env "$API_KEY_ENV" \
    --contour-provider api-vision \
    --contour-model "$API_MODEL" \
    --contour-api-base-url "$API_BASE_URL" \
    --contour-api-key-env "$API_KEY_ENV" \
    --api-image-detail high \
    --max-attempts 10 \
    --coordinate-tolerance-px 16 \
    --bootstrap-iterations 500 \
    --seed 13 \
    --checkpoint-every 1 \
    --resume \
    --print-summary
  touch "$REPAIR_ROOT/${mode^^}_COMPLETE"
}

run_repair_mode prompt
run_repair_mode instruction

for mode in prompt instruction; do
  run_low -m phase3_mask_edit.cli.merge_mask_edit_benchmark_results \
    --base "$OUTPUT_ROOT/runs/$mode/benchmark_eval_results.csv" \
    --rerun "$REPAIR_ROOT/$mode/benchmark_eval_results.csv" \
    --output "$OUTPUT_ROOT/runs_repaired_v2/$mode" \
    --bootstrap-iterations 500 \
    --seed 13 \
    --print-summary
done

touch "$REPAIR_ROOT/REPAIR_COMPLETE"
echo "repair_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
