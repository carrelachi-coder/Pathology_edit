#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
LOG_ROOT="${LOG_ROOT:-/data1/zhao/wqx/benchmark_v1/logs}"
MANIFEST_DIR="${MANIFEST_DIR:-$ROOT/runs/benchmark_v1/manifests}"
MAX_LOAD_PER_CPU="${MAX_LOAD_PER_CPU:-0.50}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-21600}"
POLL_SEC="${POLL_SEC:-30}"
REQUIRED_HEALTHY_CHECKS="${REQUIRED_HEALTHY_CHECKS:-3}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CPU_ID="${BENCHMARK_CPU_ID:-120}"

mkdir -p "$LOG_ROOT"
started_at=$(date +%s)
healthy_checks=0

while true; do
  read -r load1 _ < /proc/loadavg
  cpus=$(nproc)
  if awk -v load="$load1" -v cpus="$cpus" -v limit="$MAX_LOAD_PER_CPU" \
    'BEGIN { exit !(load / cpus <= limit) }'; then
    healthy_checks=$((healthy_checks + 1))
    if (( healthy_checks >= REQUIRED_HEALTHY_CHECKS )); then
      break
    fi
  else
    healthy_checks=0
  fi
  now=$(date +%s)
  if (( now - started_at >= MAX_WAIT_SEC )); then
    echo "Timed out waiting for server load below ratio $MAX_LOAD_PER_CPU" >&2
    exit 4
  fi
  sleep "$POLL_SEC"
done

cd "$ROOT"
scripts/build_mask_edit_benchmark_v1_amax2.sh

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 15 ionice -c 3 taskset -c "$CPU_ID" "$PYTHON" \
  -m phase3_mask_edit.cli.validate_mask_edit_benchmark_manifest \
  --intents "$MANIFEST_DIR/mask_semantic_intents.jsonl" \
  --shortfalls "$MANIFEST_DIR/shortfalls.csv" \
  --output "$MANIFEST_DIR/validation_report.json" \
  --expected-per-cell 100 \
  --max-per-wsi-per-cell 10 \
  --require-all-specialized \
  --print-summary

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 15 ionice -c 3 taskset -c "$CPU_ID" "$PYTHON" \
  -m phase3_mask_edit.cli.generate_mask_edit_benchmark_prompts \
  --intents "$MANIFEST_DIR/mask_semantic_intents.jsonl" \
  --output "$MANIFEST_DIR/prompts_template" \
  --checkpoint-every 100 \
  --resume \
  --print-summary
