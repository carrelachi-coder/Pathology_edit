#!/usr/bin/env bash
set -euo pipefail

# Prepare six Phase 4 datasets, build per-dataset nuclei libraries, then train
# one ProbNet across all datasets.
#
# Usage:
#   bash scripts/phase4_probnet_workflow_all.sh [ACTION] [EDIT_DATASETS_ROOT] [PHASE4_ROOT]
#
# Example:
#   bash scripts/phase4_probnet_workflow_all.sh prepare_dataset
#   bash scripts/phase4_probnet_workflow_all.sh all edit_datasets phase4_runs/all6
#
# ACTION:
#   prepare_dataset  Prepare ProbNet train/val data only.
#   build_library    Build nuclei libraries only.
#   train            Train ProbNet from existing prepared data.
#   generate         Run validation smoke-test generation from an existing checkpoint.
#   all              Run the full workflow. This is the default.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ACTION="${1:-all}"
case "${ACTION}" in
  prepare_dataset|build_library|train|generate|all)
    shift || true
    ;;
  *)
    # Backward compatible form:
    #   bash scripts/phase4_probnet_workflow_all.sh edit_datasets phase4_runs/all6
    ACTION="all"
    ;;
esac

EDIT_DATASETS_ROOT="${1:-${EDIT_DATASETS_ROOT:-${REPO_ROOT}/edit_datasets}}"
PHASE4_ROOT="${2:-${PHASE4_ROOT:-${REPO_ROOT}/phase4_runs/all6}}"

DATASETS=(${DATASETS:-BCSS GlaS IGNITE ORCA PANDA PUMA})

IMG_SIZE="${IMG_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
N_AUGMENTATIONS="${N_AUGMENTATIONS:-3}"
CROP_MODE="${CROP_MODE:-mask}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"

PROBNET_RUN_DIR="${PROBNET_RUN_DIR:-${PHASE4_ROOT}/probnet_run}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-${PHASE4_ROOT}/generated_masks}"

PROFILE_DIR="${PROFILE_DIR:-${REPO_ROOT}/inpaint_cells/configs}"
PROFILE_JSON="${PROFILE_JSON:-${PROFILE_DIR}/generation_profiles.json}"

profile_for_dataset() {
  local dataset="$1"

  eval "$(
    python scripts/phase4_generation_profile.py \
      --dataset "${dataset}" \
      --profile-json "${PROFILE_JSON}" \
      --profile-dir "${PROFILE_DIR}"
  )"

  PROFILE_GAMMA_VALUES="${GAMMA_VALUES:-${PROFILE_GAMMA_VALUES}}"
  PROFILE_PROB_COUNT_WEIGHT="${PROB_COUNT_WEIGHT:-${PROFILE_PROB_COUNT_WEIGHT}}"
  PROFILE_DENSITY_SCALE="${DENSITY_SCALE:-${PROFILE_DENSITY_SCALE}}"
  PROFILE_DENSITY_SCALE_JSON="${DENSITY_SCALE_JSON:-${PROFILE_DENSITY_SCALE_JSON}}"
  PROFILE_MAX_DENSITY_PER_10K="${MAX_DENSITY_PER_10K:-${PROFILE_MAX_DENSITY_PER_10K}}"
  PROFILE_MAX_COUNT_FACTOR="${MAX_COUNT_FACTOR:-${PROFILE_MAX_COUNT_FACTOR}}"
  PROFILE_MIN_DISTANCE_SCALE="${MIN_DISTANCE_SCALE:-${PROFILE_MIN_DISTANCE_SCALE}}"
}

mkdir -p "${PHASE4_ROOT}" "${GEN_OUTPUT_DIR}"

echo "== Phase 4 all-dataset ProbNet workflow =="
echo "Action: ${ACTION}"
echo "Repo root: ${REPO_ROOT}"
echo "Edit datasets root: ${EDIT_DATASETS_ROOT}"
echo "Phase4 root: ${PHASE4_ROOT}"
echo "Datasets: ${DATASETS[*]}"
echo "Generation profiles: ${PROFILE_DIR}/GENERATION_PROFILES.md"
echo "Generation profile JSON: ${PROFILE_JSON}"

TRAIN_SPECS=()

prepare_probnet_data() {
  local dataset="$1"
  local raw_patch_dir="$2"
  local probnet_data_dir="$3"

  echo
  echo "== ${dataset}: prepare ProbNet training data =="
  SPLIT_ARGS=()
  if [[ -n "${SPLIT_MANIFEST}" ]]; then
    SPLIT_ARGS+=(--split-manifest "${SPLIT_MANIFEST}")
  fi
  python inpaint_cells/data/prepare_dataset.py \
    --dataset "${dataset}" \
    --input-dir "${raw_patch_dir}" \
    --output-dir "${probnet_data_dir}" \
    --format auto \
    --n-augmentations "${N_AUGMENTATIONS}" \
    --val-ratio 0.1 \
    --seed 42 \
    "${SPLIT_ARGS[@]}"
}

build_nuclei_library() {
  local dataset="$1"
  local raw_patch_dir="$2"
  local nuclei_library_dir="$3"

  echo
  echo "== ${dataset}: build nuclei instance library =="
  python inpaint_cells/nuclei_library/build_library.py \
    --dataset "${dataset}" \
    --gt-dir "${raw_patch_dir}" \
    --output-dir "${nuclei_library_dir}" \
    --format auto \
    --min-area 10 \
    --max-area 5000
}


for DATASET in "${DATASETS[@]}"; do
  RAW_PATCH_DIR="${EDIT_DATASETS_ROOT}/${DATASET}"
  PROBNET_DATA_DIR="${PHASE4_ROOT}/probnet_data/${DATASET}"
  NUCLEI_LIBRARY_DIR="${PHASE4_ROOT}/nuclei_library/${DATASET}"

  if [[ ! -d "${RAW_PATCH_DIR}" ]]; then
    echo "ERROR: missing dataset directory: ${RAW_PATCH_DIR}" >&2
    exit 2
  fi

  if [[ "${ACTION}" == "prepare_dataset" || "${ACTION}" == "all" ]]; then
    prepare_probnet_data "${DATASET}" "${RAW_PATCH_DIR}" "${PROBNET_DATA_DIR}"
  fi

  if [[ "${ACTION}" == "build_library" || "${ACTION}" == "all" ]]; then
    build_nuclei_library "${DATASET}" "${RAW_PATCH_DIR}" "${NUCLEI_LIBRARY_DIR}"
  fi

  TRAIN_SPECS+=("${DATASET}:${PROBNET_DATA_DIR}")
done

if [[ "${ACTION}" == "prepare_dataset" ]]; then
  echo
  echo "Done. Prepared ProbNet data: ${PHASE4_ROOT}/probnet_data"
  exit 0
fi

if [[ "${ACTION}" == "build_library" ]]; then
  echo
  echo "Done. Nuclei libraries: ${PHASE4_ROOT}/nuclei_library"
  exit 0
fi

if [[ "${ACTION}" == "train" || "${ACTION}" == "all" ]]; then
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
fi

CKPT="${PROBNET_RUN_DIR}/checkpoints/best.pt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(ls -1 "${PROBNET_RUN_DIR}"/checkpoints/epoch_*.pt | sort | tail -n 1)"
fi
echo "Using checkpoint: ${CKPT}"

if [[ "${ACTION}" == "train" ]]; then
  echo
  echo "Done. Checkpoint: ${CKPT}"
  exit 0
fi

echo
echo "== Batch inference smoke test on each validation split =="
for DATASET in "${DATASETS[@]}"; do
  profile_for_dataset "${DATASET}"
  DENSITY_JSON_ARGS=()
  if [[ -n "${PROFILE_DENSITY_SCALE_JSON}" ]]; then
    DENSITY_JSON_ARGS+=(--density-scale-json "${PROFILE_DENSITY_SCALE_JSON}")
  fi

  echo
  echo "== ${DATASET}: generation profile =="
  echo "  gamma=${PROFILE_GAMMA_VALUES}, prob_count_weight=${PROFILE_PROB_COUNT_WEIGHT}, density_scale=${PROFILE_DENSITY_SCALE}"
  echo "  max_density_per_10k=${PROFILE_MAX_DENSITY_PER_10K}, max_count_factor=${PROFILE_MAX_COUNT_FACTOR}, min_distance_scale=${PROFILE_MIN_DISTANCE_SCALE}"
  [[ -n "${PROFILE_DENSITY_SCALE_JSON}" ]] && echo "  density_scale_json=${PROFILE_DENSITY_SCALE_JSON}"

  python inpaint_cells/generate.py \
    --dataset "${DATASET}" \
    --ckpt "${CKPT}" \
    --library "${PHASE4_ROOT}/nuclei_library/${DATASET}" \
    --test-dir "${PHASE4_ROOT}/probnet_data/${DATASET}" \
    --output-dir "${GEN_OUTPUT_DIR}/${DATASET}/val" \
    --vis-dir "${GEN_OUTPUT_DIR}/${DATASET}/val/vis" \
    --n 16 \
    --gamma-values "${PROFILE_GAMMA_VALUES}" \
    --prob-count-weight "${PROFILE_PROB_COUNT_WEIGHT}" \
    --density-scale "${PROFILE_DENSITY_SCALE}" \
    "${DENSITY_JSON_ARGS[@]}" \
    --max-density-per-10k "${PROFILE_MAX_DENSITY_PER_10K}" \
    --max-count-factor "${PROFILE_MAX_COUNT_FACTOR}" \
    --min-distance-mode adaptive \
    --min-distance-scale "${PROFILE_MIN_DISTANCE_SCALE}" \
    --oversample-base 3.0 \
    --oversample-gamma-scale 0.35
done

echo
echo "Done."
echo "Checkpoint: ${CKPT}"
echo "Generated masks: ${GEN_OUTPUT_DIR}"
