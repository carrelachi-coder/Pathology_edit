#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CHECKPOINT="${CHECKPOINT:-/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211}"
LIBRARY_ROOT="${LIBRARY_ROOT:-/data1/zhao/wqx/benchmarks/runs/probnet_compact_20260717/provenance/frozen_training_libraries}"
HELDOUT_MANIFEST="${HELDOUT_MANIFEST:-/data1/zhao/wqx/probnet_density/heldout_test_9950_seed20260723/manifest_9950.jsonl}"
RUN_ROOT="${RUN_ROOT:-/data1/zhao/wqx/benchmarks/runs/probnet_spatial_epoch29_strict_geometry_20260724}"
P1_PID_FILE="${P1_PID_FILE:-${RUN_ROOT}/launcher.pid}"
ENDPOINT_REUSE="${ENDPOINT_REUSE:-}"
GPU="${GPU:-0}"

cd "${REPO_ROOT}"
P2_ROOT="${RUN_ROOT}/p2_geometry_endpoint"
ENDPOINT="${P2_ROOT}/endpoint_120.jsonl"
LAYOUT_ROOT="${P2_ROOT}/planned_layouts"
mkdir -p "${RUN_ROOT}/logs" "${P2_ROOT}"

if [[ ! -f "${P2_ROOT}/endpoint_120.summary.json" ]]; then
  if [[ -n "${ENDPOINT_REUSE}" ]]; then
    cp "${ENDPOINT_REUSE}" "${ENDPOINT}"
    cp "${ENDPOINT_REUSE%.jsonl}.summary.json" \
      "${P2_ROOT}/endpoint_120.summary.json"
    if [[ -f "${ENDPOINT_REUSE%.jsonl}.eligibility.json" ]]; then
      cp "${ENDPOINT_REUSE%.jsonl}.eligibility.json" \
        "${P2_ROOT}/endpoint_120.eligibility.json"
    fi
  else
    "${PYTHON_BIN}" scripts/prepare_probnet_spatial_p2.py select \
      --source-manifest "${HELDOUT_MANIFEST}" \
      --library-root "${LIBRARY_ROOT}" \
      --output-manifest "${ENDPOINT}" \
      --expected-source-samples 9950 \
      --per-bin 5 \
      --seed 20260724 \
      > "${RUN_ROOT}/logs/p2_select.log" 2>&1
  fi
fi

if [[ -f "${P1_PID_FILE}" ]]; then
  p1_pid=$(cat "${P1_PID_FILE}")
  while kill -0 "${p1_pid}" 2>/dev/null; do
    sleep 30
  done
fi
if [[ ! -f "${RUN_ROOT}/p1_spatial_ablation/complete" ]]; then
  echo "P1 did not complete; refusing to start P2 geometry endpoint" >&2
  exit 2
fi

if [[ ! -f "${LAYOUT_ROOT}/validation.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
    scripts/prepare_probnet_spatial_p2.py layouts \
      --manifest "${ENDPOINT}" \
      --checkpoint "${CHECKPOINT}" \
      --expected-checkpoint-sha256 "${CHECKPOINT_SHA256}" \
      --library-root "${LIBRARY_ROOT}" \
      --output-root "${LAYOUT_ROOT}" \
      --expected-samples 120 \
      --seed 20260724 \
      --device cuda:0 \
      > "${RUN_ROOT}/logs/p2_geometry.log" 2>&1
fi

"${PYTHON_BIN}" - "${LAYOUT_ROOT}/validation.json" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
if not payload.get("complete"):
    raise SystemExit("P2 geometry endpoint is incomplete")
print(json.dumps(payload["overall"], indent=2, ensure_ascii=False))
print("geometry_safety_gate_passed=", payload["geometry_safety_gate_passed"])
PY

"${PYTHON_BIN}" scripts/write_probnet_spatial_geometry_report.py \
  --run-root "${RUN_ROOT}" \
  > "${RUN_ROOT}/logs/final_geometry_report.log" 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ > "${P2_ROOT}/complete"
echo "P2 geometry complete: ${P2_ROOT}"
