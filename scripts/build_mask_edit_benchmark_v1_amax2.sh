#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
MANIFEST_DIR="${MANIFEST_DIR:-$ROOT/runs/benchmark_v1/manifests}"
CPU_ID="${BENCHMARK_CPU_ID:-120}"
MAX_LOAD_PER_CPU="${MAX_LOAD_PER_CPU:-0.50}"

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
  -m phase3_mask_edit.cli.build_mask_edit_benchmark \
  --config benchmark_configs/benchmark_v1_amax2.yaml \
  --output "$MANIFEST_DIR" \
  --print-summary
