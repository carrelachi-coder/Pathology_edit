#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:?usage: build_probnet_frozen_libraries.sh OUTPUT_ROOT [DATA_ROOT]}"
DATA_ROOT="${2:-/data/wqx/flowedit/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
SOURCE_ROOT="${OUTPUT_ROOT}/provenance/library_sources"
LIBRARY_ROOT="${OUTPUT_ROOT}/provenance/frozen_training_libraries"

mkdir -p "${LIBRARY_ROOT}"
cd "${REPO_ROOT}"

for dataset in BCSS GlaS IGNITE ORCA PANDA PUMA; do
  manifest="${SOURCE_ROOT}/${dataset}.jsonl"
  dataset_root="${DATA_ROOT}/${dataset}_PATCHES"
  output="${LIBRARY_ROOT}/${dataset}"
  done_file="${output}/.complete"
  if [[ -f "${done_file}" ]]; then
    echo "[skip] ${dataset}: ${done_file} exists"
    continue
  fi
  rm -rf "${output}"
  mkdir -p "${output}"
  echo "[build] ${dataset}"
  "${PYTHON_BIN}" inpaint_cells/nuclei_library/build_library.py \
    --dataset "${dataset}" \
    --gt-dir "${dataset_root}" \
    --output-dir "${output}" \
    --format layered \
    --source-manifest "${manifest}" \
    --min-area 10 \
    --max-area 5000 \
    --max-instances-per-bucket 10000
  date -u +%Y-%m-%dT%H:%M:%SZ > "${done_file}"
done

echo "Frozen training-only libraries complete: ${LIBRARY_ROOT}"
