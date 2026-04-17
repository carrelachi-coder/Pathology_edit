#!/usr/bin/env bash
set -euo pipefail

# Prepare six Phase 4 datasets, build per-dataset nuclei libraries, then train
# one ProbNet across all datasets.
#
# Usage:
#   bash scripts/phase4_probnet_workflow_all.sh [EDIT_DATASETS_ROOT] [PHASE4_ROOT]
#
# Example:
#   bash scripts/phase4_probnet_workflow_all.sh edit_datasets phase4_runs/all6

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EDIT_DATASETS_ROOT="${1:-${EDIT_DATASETS_ROOT:-${REPO_ROOT}/edit_datasets}}"
PHASE4_ROOT="${2:-${PHASE4_ROOT:-${REPO_ROOT}/phase4_runs/all6}}"

DATASETS=(${DATASETS:-BCSS GlaS IGNITE ORCA PANDA PUMA})

IMG_SIZE="${IMG_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
N_AUGMENTATIONS="${N_AUGMENTATIONS:-3}"
CROP_MODE="${CROP_MODE:-mask}"

PROBNET_RUN_DIR="${PROBNET_RUN_DIR:-${PHASE4_ROOT}/probnet_run}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-${PHASE4_ROOT}/generated_masks}"

GAMMA_VALUES="${GAMMA_VALUES:-1.0,2.0,3.0}"
PROB_COUNT_WEIGHT="${PROB_COUNT_WEIGHT:-0.7}"
DENSITY_SCALE="${DENSITY_SCALE:-1.0}"
MAX_DENSITY_PER_10K="${MAX_DENSITY_PER_10K:-900}"
MAX_COUNT_FACTOR="${MAX_COUNT_FACTOR:-2.5}"

mkdir -p "${PHASE4_ROOT}" "${GEN_OUTPUT_DIR}"

echo "== Phase 4 all-dataset ProbNet workflow =="
echo "Edit datasets root: ${EDIT_DATASETS_ROOT}"
echo "Phase4 root: ${PHASE4_ROOT}"
echo "Datasets: ${DATASETS[*]}"

TRAIN_SPECS=()

for DATASET in "${DATASETS[@]}"; do
  RAW_PATCH_DIR="${EDIT_DATASETS_ROOT}/${DATASET}"
  PROBNET_DATA_DIR="${PHASE4_ROOT}/probnet_data/${DATASET}"
  NUCLEI_LIBRARY_DIR="${PHASE4_ROOT}/nuclei_library/${DATASET}"

  if [[ ! -d "${RAW_PATCH_DIR}" ]]; then
    echo "ERROR: missing dataset directory: ${RAW_PATCH_DIR}" >&2
    exit 2
  fi

  echo
  echo "== ${DATASET}: prepare ProbNet training data =="
  python inpaint_cells/data/prepare_dataset.py \
    --dataset "${DATASET}" \
    --input-dir "${RAW_PATCH_DIR}" \
    --output-dir "${PROBNET_DATA_DIR}" \
    --format auto \
    --n-augmentations "${N_AUGMENTATIONS}" \
    --val-ratio 0.1 \
    --seed 42

  echo
  echo "== ${DATASET}: build nuclei instance library =="
  python inpaint_cells/nuclei_library/build_library.py \
    --dataset "${DATASET}" \
    --gt-dir "${RAW_PATCH_DIR}" \
    --output-dir "${NUCLEI_LIBRARY_DIR}" \
    --format auto \
    --min-area 10 \
    --max-area 5000

  TRAIN_SPECS+=("${DATASET}:${PROBNET_DATA_DIR}")
done

echo
echo "== Train one ProbNet on all datasets =="
python inpaint_cells/train.py \
  --mode train \
  --datasets "${TRAIN_SPECS[@]}" \
  --output-dir "${PROBNET_RUN_DIR}" \
  --img-size "${IMG_SIZE}" \
  --crop-mode "${CROP_MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --resume-from-checkpoint latest

CKPT="${PROBNET_RUN_DIR}/checkpoints/best.pt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(ls -1 "${PROBNET_RUN_DIR}"/checkpoints/epoch_*.pt | sort | tail -n 1)"
fi
echo "Using checkpoint: ${CKPT}"

echo
echo "== Batch inference smoke test on each validation split =="
for DATASET in "${DATASETS[@]}"; do
  python inpaint_cells/generate.py \
    --dataset "${DATASET}" \
    --ckpt "${CKPT}" \
    --library "${PHASE4_ROOT}/nuclei_library/${DATASET}" \
    --test-dir "${PHASE4_ROOT}/probnet_data/${DATASET}" \
    --output-dir "${GEN_OUTPUT_DIR}/${DATASET}/val" \
    --vis-dir "${GEN_OUTPUT_DIR}/${DATASET}/val/vis" \
    --n 16 \
    --gamma-values "${GAMMA_VALUES}" \
    --prob-count-weight "${PROB_COUNT_WEIGHT}" \
    --density-scale "${DENSITY_SCALE}" \
    --max-density-per-10k "${MAX_DENSITY_PER_10K}" \
    --max-count-factor "${MAX_COUNT_FACTOR}" \
    --min-distance-mode adaptive \
    --min-distance-scale 0.75 \
    --oversample-base 3.0 \
    --oversample-gamma-scale 0.35
done

echo
echo "Done."
echo "Checkpoint: ${CKPT}"
echo "Generated masks: ${GEN_OUTPUT_DIR}"
