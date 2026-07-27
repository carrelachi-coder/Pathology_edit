#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CHECKPOINT="${CHECKPOINT:-/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211}"
LIBRARY_ROOT="${LIBRARY_ROOT:-/data1/zhao/wqx/benchmarks/runs/probnet_compact_20260717/provenance/frozen_training_libraries}"
P1_MANIFEST="${P1_MANIFEST:-/data1/zhao/wqx/benchmarks/runs/probnet_hierarchical_epoch1_boundary_backfill_same_schema_20260722/evaluation/manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/zhao/wqx/benchmarks/runs/probnet_spatial_epoch29_strict_geometry_20260724}"
GPUS_CSV="${GPUS_CSV:-2,5}"
EXPECTED_CASES="${EXPECTED_CASES:-1200}"
REPLICATES="${REPLICATES:-5}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-5000}"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
SHARDS="${#GPUS[@]}"
if [[ "${SHARDS}" -lt 1 ]]; then
  echo "No GPUs configured" >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"

for path in "${PYTHON_BIN}" "${CHECKPOINT}" "${P1_MANIFEST}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done
actual_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${CHECKPOINT_SHA256}" ]]; then
  echo "Checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 2
fi
for dataset in BCSS GlaS IGNITE ORCA PANDA PUMA; do
  if [[ ! -f "${LIBRARY_ROOT}/${dataset}/statistics.json" ]]; then
    echo "Missing frozen library: ${LIBRARY_ROOT}/${dataset}" >&2
    exit 2
  fi
done

printf '%s\n' \
  "ProbNet spatial-only benchmark" \
  "  checkpoint=${CHECKPOINT}" \
  "  checkpoint_sha256=${actual_sha256}" \
  "  manifest=${P1_MANIFEST}" \
  "  output=${OUTPUT_ROOT}" \
  "  physical_gpus=${GPUS_CSV}" \
  "  shards=${SHARDS}"

pids=()
for shard_index in "${!GPUS[@]}"; do
  physical_gpu="${GPUS[${shard_index}]}"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" scripts/run_probnet_spatial_benchmark.py run-shard \
      --manifest "${P1_MANIFEST}" \
      --checkpoint "${CHECKPOINT}" \
      --expected-checkpoint-sha256 "${CHECKPOINT_SHA256}" \
      --library-root "${LIBRARY_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --expected-cases "${EXPECTED_CASES}" \
      --replicates "${REPLICATES}" \
      --shards "${SHARDS}" \
      --shard-index "${shard_index}" \
      --device cuda:0 \
      > "${OUTPUT_ROOT}/logs/p1_shard_${shard_index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != 0 ]]; then
  echo "At least one P1 shard failed; see ${OUTPUT_ROOT}/logs" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/run_probnet_spatial_benchmark.py report \
  --output-root "${OUTPUT_ROOT}" \
  --expected-cases "${EXPECTED_CASES}" \
  --replicates "${REPLICATES}" \
  --shards "${SHARDS}" \
  --bootstrap-repeats "${BOOTSTRAP_REPEATS}" \
  > "${OUTPUT_ROOT}/logs/p1_report.log" 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_ROOT}/p1_spatial_ablation/complete"
echo "P1 complete: ${OUTPUT_ROOT}/p1_spatial_ablation"
