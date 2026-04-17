#!/usr/bin/env bash
set -euo pipefail

# Phase 4 ProbNet workflow:
#   1. prepare ProbNet training samples from layered masks
#   2. build nuclei instance library
#   3. train ProbNet
#   4. generate new nuclei masks with ProbNet-centered weighted Poisson sampling
#
# Usage:
#   bash scripts/phase4_probnet_workflow.sh DATASET RAW_PATCH_DIR [PHASE4_ROOT]
#
# Example:
#   bash scripts/phase4_probnet_workflow.sh BCSS edit_datasets/BCSS phase4_runs/BCSS
#
# Most knobs can still be overridden with environment variables.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# -------------------------
# Positional paths
# -------------------------
DATASET="${1:-${DATASET:-BCSS}}"

# Input patch directory. Supported structures:
#   A) ${RAW_PATCH_DIR}/tissue_masks/*.png + nuclei_masks/*.png
#   B) ${RAW_PATCH_DIR}/{sample}/tissue_mask.png + nuclei_mask.png
#   C) ${RAW_PATCH_DIR}/gt_tissue/*.png + gt_nuclei/*.png
RAW_PATCH_DIR="${2:-${RAW_PATCH_DIR:-${REPO_ROOT}/edit_datasets/${DATASET}}}"

# Optional nuclei annotation sources for datasets without layered nuclei masks.
# Set one of these when build_library.py cannot auto-detect nuclei_mask.png:
#   CELLVIT_DIR=/path/to/cellvit_json_outputs
#   GEOJSON_DIR=/path/to/puma_geojson
CELLVIT_DIR="${CELLVIT_DIR:-}"
GEOJSON_DIR="${GEOJSON_DIR:-}"

# Output workspace for Phase 4 artifacts.
PHASE4_ROOT="${3:-${PHASE4_ROOT:-${REPO_ROOT}/phase4_runs/${DATASET}}}"
PROBNET_DATA_DIR="${PROBNET_DATA_DIR:-${PHASE4_ROOT}/probnet_data}"
NUCLEI_LIBRARY_DIR="${NUCLEI_LIBRARY_DIR:-${PHASE4_ROOT}/nuclei_library}"
PROBNET_RUN_DIR="${PROBNET_RUN_DIR:-${PHASE4_ROOT}/probnet_run}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-${PHASE4_ROOT}/generated_masks}"

# Optional single edited-mask inference inputs. Leave empty to skip.
SINGLE_EDITED_TISSUE="${SINGLE_EDITED_TISSUE:-}"
SINGLE_INPUT_NUCLEI="${SINGLE_INPUT_NUCLEI:-}"
SINGLE_EDIT_REGION="${SINGLE_EDIT_REGION:-}"

# Training knobs.
IMG_SIZE="${IMG_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
N_AUGMENTATIONS="${N_AUGMENTATIONS:-3}"

# Generation knobs.
PROFILE_DIR="${PROFILE_DIR:-${REPO_ROOT}/inpaint_cells/configs}"
PROFILE_JSON="${PROFILE_JSON:-${PROFILE_DIR}/generation_profiles.json}"

eval "$(
  python scripts/phase4_generation_profile.py \
    --dataset "${DATASET}" \
    --profile-json "${PROFILE_JSON}" \
    --profile-dir "${PROFILE_DIR}"
)"

GAMMA_VALUES="${GAMMA_VALUES:-${PROFILE_GAMMA_VALUES}}"
PROB_COUNT_WEIGHT="${PROB_COUNT_WEIGHT:-${PROFILE_PROB_COUNT_WEIGHT}}"
DENSITY_SCALE="${DENSITY_SCALE:-${PROFILE_DENSITY_SCALE}}"
DENSITY_SCALE_JSON="${DENSITY_SCALE_JSON:-${PROFILE_DENSITY_SCALE_JSON}}"
MAX_DENSITY_PER_10K="${MAX_DENSITY_PER_10K:-${PROFILE_MAX_DENSITY_PER_10K}}"
MAX_COUNT_FACTOR="${MAX_COUNT_FACTOR:-${PROFILE_MAX_COUNT_FACTOR}}"
MIN_DISTANCE_SCALE="${MIN_DISTANCE_SCALE:-${PROFILE_MIN_DISTANCE_SCALE}}"

mkdir -p "${PHASE4_ROOT}" "${GEN_OUTPUT_DIR}"

if [[ ! -d "${RAW_PATCH_DIR}" ]]; then
  echo "ERROR: RAW_PATCH_DIR does not exist: ${RAW_PATCH_DIR}" >&2
  echo "Usage: bash scripts/phase4_probnet_workflow.sh DATASET RAW_PATCH_DIR [PHASE4_ROOT]" >&2
  exit 2
fi

echo "== Phase 4 ProbNet workflow =="
echo "Dataset: ${DATASET}"
echo "Raw patches: ${RAW_PATCH_DIR}"
echo "Phase4 root: ${PHASE4_ROOT}"
echo "Generation profile: ${PROFILE_DIR}/GENERATION_PROFILES.md"
echo "Generation profile JSON: ${PROFILE_JSON}"
echo "  gamma=${GAMMA_VALUES}, prob_count_weight=${PROB_COUNT_WEIGHT}, density_scale=${DENSITY_SCALE}"
echo "  max_density_per_10k=${MAX_DENSITY_PER_10K}, max_count_factor=${MAX_COUNT_FACTOR}, min_distance_scale=${MIN_DISTANCE_SCALE}"
[[ -n "${DENSITY_SCALE_JSON}" ]] && echo "  density_scale_json=${DENSITY_SCALE_JSON}"

echo
echo "== 0. Optional environment setup =="
echo "Create once with:"
echo "  conda env create -f envs/phase4_probnet.yaml"
echo "  conda activate pathology-phase4"

echo
echo "== 1. Prepare ProbNet training data =="
python inpaint_cells/data/prepare_dataset.py \
  --dataset "${DATASET}" \
  --input-dir "${RAW_PATCH_DIR}" \
  --output-dir "${PROBNET_DATA_DIR}" \
  --format auto \
  --n-augmentations "${N_AUGMENTATIONS}" \
  --val-ratio 0.1 \
  --seed 42

echo
echo "== 2. Build nuclei instance library =="
BUILD_LIBRARY_ARGS=(
  --dataset "${DATASET}"
  --gt-dir "${RAW_PATCH_DIR}"
  --output-dir "${NUCLEI_LIBRARY_DIR}"
  --min-area 10
  --max-area 5000
)
if [[ -n "${CELLVIT_DIR}" ]]; then
  BUILD_LIBRARY_ARGS+=(--format cellvit-json --cellvit-dir "${CELLVIT_DIR}")
elif [[ -n "${GEOJSON_DIR}" ]]; then
  BUILD_LIBRARY_ARGS+=(--format geojson --geojson-dir "${GEOJSON_DIR}")
else
  BUILD_LIBRARY_ARGS+=(--format auto)
fi
python inpaint_cells/nuclei_library/build_library.py "${BUILD_LIBRARY_ARGS[@]}"

echo
echo "== 3. Train ProbNet =="
python inpaint_cells/train.py \
  --mode train \
  --datasets "${DATASET}:${PROBNET_DATA_DIR}" \
  --output-dir "${PROBNET_RUN_DIR}" \
  --img-size "${IMG_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --resume-from-checkpoint latest

CKPT="${PROBNET_RUN_DIR}/checkpoints/best.pt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(ls -1 "${PROBNET_RUN_DIR}"/checkpoints/epoch_*.pt | sort | tail -n 1)"
fi
echo "Using checkpoint: ${CKPT}"

echo
echo "== 4. Batch inference on validation split with gamma comparison =="
DENSITY_JSON_ARGS=()
if [[ -n "${DENSITY_SCALE_JSON}" ]]; then
  DENSITY_JSON_ARGS+=(--density-scale-json "${DENSITY_SCALE_JSON}")
fi
python inpaint_cells/generate.py \
  --dataset "${DATASET}" \
  --ckpt "${CKPT}" \
  --library "${NUCLEI_LIBRARY_DIR}" \
  --test-dir "${PROBNET_DATA_DIR}" \
  --output-dir "${GEN_OUTPUT_DIR}/val" \
  --vis-dir "${GEN_OUTPUT_DIR}/val/vis" \
  --n 16 \
  --gamma-values "${GAMMA_VALUES}" \
  --prob-count-weight "${PROB_COUNT_WEIGHT}" \
  --density-scale "${DENSITY_SCALE}" \
  "${DENSITY_JSON_ARGS[@]}" \
  --max-density-per-10k "${MAX_DENSITY_PER_10K}" \
  --max-count-factor "${MAX_COUNT_FACTOR}" \
  --min-distance-mode adaptive \
  --min-distance-scale "${MIN_DISTANCE_SCALE}" \
  --oversample-base 3.0 \
  --oversample-gamma-scale 0.35

if [[ -n "${SINGLE_EDITED_TISSUE}" && -n "${SINGLE_EDIT_REGION}" ]]; then
  echo
  echo "== 5. Single edited tissue-mask inference =="
  SINGLE_ARGS=(
    --dataset "${DATASET}"
    --ckpt "${CKPT}"
    --library "${NUCLEI_LIBRARY_DIR}"
    --input-tissue "${SINGLE_EDITED_TISSUE}"
    --edit-region "${SINGLE_EDIT_REGION}"
    --output "${GEN_OUTPUT_DIR}/single_nuclei_mask.png"
    --vis-dir "${GEN_OUTPUT_DIR}/single_vis"
    --gamma-values "${GAMMA_VALUES}"
    --prob-count-weight "${PROB_COUNT_WEIGHT}"
    --density-scale "${DENSITY_SCALE}"
    "${DENSITY_JSON_ARGS[@]}"
    --max-density-per-10k "${MAX_DENSITY_PER_10K}"
    --max-count-factor "${MAX_COUNT_FACTOR}"
    --min-distance-mode adaptive
    --min-distance-scale "${MIN_DISTANCE_SCALE}"
  )
  if [[ -n "${SINGLE_INPUT_NUCLEI}" ]]; then
    SINGLE_ARGS+=(--input-nuclei "${SINGLE_INPUT_NUCLEI}")
  fi
  python inpaint_cells/generate.py "${SINGLE_ARGS[@]}"
fi

echo
echo "Done."
echo "Generated masks: ${GEN_OUTPUT_DIR}"
echo "Gamma comparison images: ${GEN_OUTPUT_DIR}/val/vis"
