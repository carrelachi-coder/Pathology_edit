#!/usr/bin/env bash
set -euo pipefail
umask 027

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
FULL_ROOT="${FULL_ROOT:-/data1/zhao/wqx/benchmark_v1/full_v0_2}"
INTENTS="${INTENTS:-$ROOT/runs/benchmark_v1/manifests/mask_semantic_intents.jsonl}"
PROMPTS="${PROMPTS:-$FULL_ROOT/prompts/benchmark_prompts.accepted.csv}"
CPU_ID="${BENCHMARK_CPU_ID:-120}"
API_BASE_URL="${API_BASE_URL:-https://api.cursorai.art/v1}"
API_KEY_ENV="${API_KEY_ENV:-VISION_API_KEY}"
PARSER_MODEL="${PARSER_MODEL:-gpt-4.1-mini}"
CONTOUR_MODEL="${CONTOUR_MODEL:-gpt-4.1-mini}"
SEMANTIC_REPAIR_ATTEMPTS="${SEMANTIC_REPAIR_ATTEMPTS:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CURRENT_PID="${CURRENT_PID:-}"

WORK_ROOT="$FULL_ROOT/agentic_finalize"
BACKUP_ROOT="$WORK_ROOT/pre_repair_backup"
LOG="$WORK_ROOT/finalize.log"

mkdir -p "$WORK_ROOT" "$BACKUP_ROOT"
exec 9>"$WORK_ROOT/.finalize.lock"
if ! flock -n 9; then
  echo "Agentic benchmark finalization is already running."
  exit 0
fi

if [[ -n "$CURRENT_PID" ]]; then
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
fi
while [[ ! -f "$FULL_ROOT/FULL_COMPLETE" ]]; do
  sleep "$POLL_SECONDS"
done

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Required API key environment variable is not set: $API_KEY_ENV" >&2
  exit 2
fi

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

wait_for_health() {
  local load1 cpus available_bytes
  while true; do
    read -r load1 _ < /proc/loadavg
    cpus=$(nproc --all)
    available_bytes=$(free -b | awk '/^Mem:/ {print $7}')
    if awk -v load="$load1" -v cpus="$cpus" -v available="$available_bytes" \
      'BEGIN { exit !((load / cpus <= 0.50) && (available >= 68719476736)) }'; then
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

backup_mode() {
  local mode="$1"
  local source="$FULL_ROOT/runs/$mode"
  local target="$BACKUP_ROOT/$mode"
  mkdir -p "$target"
  for name in benchmark_eval_results.csv benchmark_report.json benchmark_report.csv run_manifest.json; do
    if [[ -f "$source/$name" && ! -f "$target/$name" ]]; then
      cp -a "$source/$name" "$target/$name"
    fi
  done
}

select_repairable_failures() {
  local mode="$1"
  local eval_path="$FULL_ROOT/runs/$mode/benchmark_eval_results.csv"
  local output="$WORK_ROOT/${mode}_repairable_failures.txt"
  run_low - "$eval_path" "$output" <<'PY'
import csv
import sys
from pathlib import Path

eval_path = Path(sys.argv[1])
output = Path(sys.argv[2])
selected = []
if eval_path.is_file():
    with eval_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            error = str(row.get("error") or "")
            if error.startswith("planner produced no executable intents") or (
                error.startswith("API request failed")
                and ("HTTP 401" in error or "HTTP 429" in error)
            ):
                selected.append(str(row["sample_id"]))
output.write_text("".join(f"{sample_id}\n" for sample_id in sorted(set(selected))))
print(f"mode={eval_path.parent.name} repairable_failures={len(set(selected))}")
PY
}

run_mode() {
  local mode="$1"
  local sample_ids="$WORK_ROOT/${mode}_repairable_failures.txt"
  local output="$FULL_ROOT/runs/$mode"
  local args=(
    -m phase3_mask_edit.cli.run_mask_edit_benchmark
    --intents "$INTENTS"
    --prompts "$PROMPTS"
    --output "$output"
    --modes "$mode"
    --parser-model "$PARSER_MODEL"
    --parser-api-base-url "$API_BASE_URL"
    --parser-api-key-env "$API_KEY_ENV"
    --contour-provider api-vision
    --contour-model "$CONTOUR_MODEL"
    --contour-api-base-url "$API_BASE_URL"
    --contour-api-key-env "$API_KEY_ENV"
    --api-image-detail high
    --max-attempts "$MAX_ATTEMPTS"
    --semantic-repair-attempts "$SEMANTIC_REPAIR_ATTEMPTS"
    --coordinate-tolerance-px 16
    --bootstrap-iterations 2000
    --seed 13
    --checkpoint-every 25
    --resume
  )
  if [[ -s "$sample_ids" ]]; then
    args+=(--sample-ids "$sample_ids")
  else
    args+=(--limit 0 --no-retry-failed)
  fi
  wait_for_health
  run_low "${args[@]}" >>"$LOG" 2>&1
}

for mode in gt prompt instruction; do
  backup_mode "$mode"
  select_repairable_failures "$mode" >>"$LOG" 2>&1
  run_mode "$mode"
done

touch "$FULL_ROOT/AGENTIC_METRICS_COMPLETE"
printf '%s agentic_finalize_complete\n' "$(date -Is)" >>"$LOG"
