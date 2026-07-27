#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "${SEGMENTATOR_CONDA_ENV:-pathology-segmentator-mmseg}"
fi

STAGE="${STAGE:-baseline}"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-/data1/zhao/wqx/segmentator_improved}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/data1/zhao/wqx/segmentator_manifests}"

case "$STAGE" in
  baseline)
    CLASS_WEIGHTING=none
    DATASET_SAMPLING_TEMPERATURE=0.0
    RARE_CLASS_SAMPLING=0
    BACKBONE_UNFREEZE_EPOCH=-1
    BACKBONE_UNFREEZE_BLOCKS=0
    LR_SCHEDULER=none
    EARLY_STOPPING_PATIENCE=0
    SYMMETRIC_PADDING=0
    BOUNDARY_REFINEMENT=0
    CELLVIT_MODE=none
    AUGMENT_VFLIP=0
    AUGMENT_ROT90=0
    AUGMENT_SCALE_CROP=0.0
    ;;
  training)
    CLASS_WEIGHTING=inverse_sqrt
    DATASET_SAMPLING_TEMPERATURE=0.5
    RARE_CLASS_SAMPLING=1
    BACKBONE_UNFREEZE_EPOCH=2
    BACKBONE_UNFREEZE_BLOCKS=6
    LR_SCHEDULER=cosine
    EARLY_STOPPING_PATIENCE=4
    SYMMETRIC_PADDING=1
    BOUNDARY_REFINEMENT=0
    CELLVIT_MODE=none
    AUGMENT_VFLIP=1
    AUGMENT_ROT90=1
    AUGMENT_SCALE_CROP=0.1
    ;;
  refine|teacher|input)
    CLASS_WEIGHTING=inverse_sqrt
    DATASET_SAMPLING_TEMPERATURE=0.5
    RARE_CLASS_SAMPLING=1
    BACKBONE_UNFREEZE_EPOCH=2
    BACKBONE_UNFREEZE_BLOCKS=6
    LR_SCHEDULER=cosine
    EARLY_STOPPING_PATIENCE=4
    SYMMETRIC_PADDING=1
    BOUNDARY_REFINEMENT=1
    CELLVIT_MODE=none
    [[ "$STAGE" == "teacher" ]] && CELLVIT_MODE=teacher
    [[ "$STAGE" == "input" ]] && CELLVIT_MODE=input
    AUGMENT_VFLIP=1
    AUGMENT_ROT90=1
    AUGMENT_SCALE_CROP=0.1
    ;;
  *)
    echo "Unknown STAGE=$STAGE; expected baseline, training, refine, teacher, or input" >&2
    exit 2
    ;;
esac

export SEED CLASS_WEIGHTING DATASET_SAMPLING_TEMPERATURE RARE_CLASS_SAMPLING
export BACKBONE_UNFREEZE_EPOCH BACKBONE_UNFREEZE_BLOCKS SYMMETRIC_PADDING
export LR_SCHEDULER EARLY_STOPPING_PATIENCE
export BOUNDARY_REFINEMENT CELLVIT_MODE AUGMENT_VFLIP AUGMENT_ROT90 AUGMENT_SCALE_CROP
export TEST_FRACTION="${TEST_FRACTION:-0.1}"
export EPOCHS="${EPOCHS:-12}"
export SEGMENTATOR_MANIFEST="${SEGMENTATOR_MANIFEST:-$MANIFEST_ROOT/grouped_seed${SEED}.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/${STAGE}_seed${SEED}}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"

mkdir -p "$MANIFEST_ROOT" "$RUN_ROOT"
exec bash scripts/train_segmentator_stage4_mask2former_multidataset_a800.sh
