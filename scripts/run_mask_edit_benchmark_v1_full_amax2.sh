#!/usr/bin/env bash
set -euo pipefail
umask 027

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CPU_ID="${BENCHMARK_CPU_ID:-120}"
INTENTS="${INTENTS:-$ROOT/runs/benchmark_v1/manifests/mask_semantic_intents.jsonl}"
FULL_ROOT="${FULL_ROOT:-/data1/zhao/wqx/benchmark_v1/full_v0_2}"
PROMPT_DIR="${PROMPT_DIR:-$FULL_ROOT/prompts}"
PROMPTS="${PROMPTS:-$PROMPT_DIR/benchmark_prompts.accepted.csv}"
PARSER_AB_ROOT="${PARSER_AB_ROOT:-/data1/zhao/wqx/benchmark_v1/preflight_20260713/parser_ab_v0_2}"

API_BASE_URL="${API_BASE_URL:-https://api.cursorai.art/v1}"
API_KEY_ENV="${API_KEY_ENV:-VISION_API_KEY}"
PARSER_MODEL="${PARSER_MODEL:-gpt-4.1-mini}"
GENERATOR_MODEL="${GENERATOR_MODEL:-gpt-4.1-mini}"
CHECKER_MODEL="${CHECKER_MODEL:-gpt-4.1-mini}"
CONTOUR_MODEL="${CONTOUR_MODEL:-gpt-4.1-mini}"

EXPECTED_INTENTS_SHA="8d0688d20b7344aa140560c42b223e7cf12160ea3731f788ff791857ef86a3dc"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
SEMANTIC_REPAIR_ATTEMPTS="${SEMANTIC_REPAIR_ATTEMPTS:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-2000}"
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

mkdir -p "$FULL_ROOT" "$PROMPT_DIR" "$FULL_ROOT/runs"
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
  local stage="$1"
  local started now streak load1 cpus available_bytes output_free_bytes
  started=$(date +%s)
  streak=0
  while true; do
    read -r load1 _ < /proc/loadavg
    cpus=$(nproc --all)
    available_bytes=$(free -b | awk '/^Mem:/ {print $7}')
    output_free_bytes=$(df -PB1 "$FULL_ROOT" | awk 'NR==2 {print $4}')
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

actual_intents_sha=$(sha256sum "$INTENTS" | awk '{print $1}')
if [[ "$actual_intents_sha" != "$EXPECTED_INTENTS_SHA" ]]; then
  echo "Frozen intent hash mismatch: expected=$EXPECTED_INTENTS_SHA actual=$actual_intents_sha" >&2
  exit 4
fi

if [[ ! -f "$PARSER_AB_ROOT/AB_COMPLETE" ]]; then
  echo "Parser A/B is not complete: $PARSER_AB_ROOT/AB_COMPLETE" >&2
  exit 5
fi

run_low - "$PARSER_AB_ROOT/$PARSER_MODEL/parser_report.json" "$PARSER_MODEL" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
expected_model = sys.argv[2]
report = json.loads(report_path.read_text())
assert report["parser_only"] is True
assert report["contour_calls"] == 0
assert report["model"] == expected_model
assert report["modes"]["prompt"]["primitive_exact_rate"] >= 0.98
assert report["modes"]["instruction"]["primitive_exact_rate"] == 1.0
assert report["gate"]["passed"] is True
PY

run_low - "$FULL_ROOT/run_config.json" "$INTENTS" "$actual_intents_sha" \
  "$PARSER_MODEL" "$GENERATOR_MODEL" "$CHECKER_MODEL" "$CONTOUR_MODEL" "$CPU_ID" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "semantic_diff_schema_version": "0.2",
    "intents": {"path": sys.argv[2], "sha256": sys.argv[3]},
    "parser_model": sys.argv[4],
    "prompt_generator_model": sys.argv[5],
    "prompt_checker_model": sys.argv[6],
    "contour_model": sys.argv[7],
    "cpu_id": int(sys.argv[8]),
    "thread_limit": 1,
}
path.write_text(json.dumps(payload, indent=2))
PY

echo "full_benchmark_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
wait_for_health prompts
run_low -m phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts \
  --intents "$INTENTS" \
  --output "$PROMPT_DIR" \
  --use-llm-generator \
  --generator-model "$GENERATOR_MODEL" \
  --generator-api-base-url "$API_BASE_URL" \
  --generator-api-key-env "$API_KEY_ENV" \
  --use-llm-checker \
  --checker-model "$CHECKER_MODEL" \
  --checker-api-base-url "$API_BASE_URL" \
  --checker-api-key-env "$API_KEY_ENV" \
  --manual-review-per-group 3 \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --max-retries 3 \
  --checker-repair-attempts 2 \
  --retry-rejected \
  --resume \
  --print-summary

run_low - "$INTENTS" "$PROMPTS" <<'PY'
import csv
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    intent_ids = {json.loads(line)["sample_id"] for line in stream if line.strip()}
with open(sys.argv[2], encoding="utf-8", newline="") as stream:
    prompt_rows = list(csv.DictReader(stream))
accepted_ids = {
    row["sample_id"]
    for row in prompt_rows
    if row.get("checker_status", "").lower() == "accepted"
}
missing = intent_ids - accepted_ids
extra = accepted_ids - intent_ids
if missing or extra:
    raise SystemExit(
        f"Prompt bank mismatch: intents={len(intent_ids)} accepted={len(accepted_ids)} "
        f"missing={len(missing)} extra={len(extra)}"
    )
PY
touch "$FULL_ROOT/PROMPTS_COMPLETE"

run_mode() {
  local mode="$1"
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
    --bootstrap-iterations "$BOOTSTRAP_ITERATIONS"
    --seed "$SEED"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --resume
    --print-summary
  )
  wait_for_health "$mode"
  run_low "${args[@]}"
  touch "$FULL_ROOT/${mode^^}_COMPLETE"
}

run_mode gt
run_mode prompt
run_mode instruction

touch "$FULL_ROOT/FULL_COMPLETE"
echo "full_benchmark_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
