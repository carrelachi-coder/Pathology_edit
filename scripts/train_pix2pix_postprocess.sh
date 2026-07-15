#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wqx/flowedit/pix2pix_texture_transfer_lazy_ver4_wsi_identity_i0_local_full_pyramid_v3_ft}"
RESUME="${RESUME:-/data/wqx/flowedit/pix2pix_texture_transfer_lazy_ver4_wsi_identity_i0_steered_texture_v2_ft/ckpt/pilot_step002000.pt}"
TRAIN_METADATA="${TRAIN_METADATA:-${REPO_DIR}/phase5_runs/cross_meta/metadata_cross_train.json}"
VAL_METADATA="${VAL_METADATA:-${REPO_DIR}/phase5_runs/cross_meta/metadata_cross_val.json}"
I0_CACHE_DIR="${I0_CACHE_DIR:-/data/wqx/flowedit/pix2pix_i0_lazy_cache/train}"
VAL_I0_CACHE_DIR="${VAL_I0_CACHE_DIR:-/data/wqx/flowedit/pix2pix_i0_lazy_cache/val}"
GPU_IDS="${GPU_IDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_CONTINUATION_STEPS="${MAX_CONTINUATION_STEPS:-1000}"
EPOCHS="${EPOCHS:-27}"
MASTER_PORT="${MASTER_PORT:-29571}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-steered_full_pyramid}"
HIGHRES_LR="${HIGHRES_LR:-0.000005}"
CROSS4_LR="${CROSS4_LR:-0.000002}"
MIDCROSS_LR="${MIDCROSS_LR:-0.000001}"
BACKBONE_LR="${BACKBONE_LR:-0.0000005}"
IDENTITY_LR="${IDENTITY_LR:-0.0000005}"
REFERENCE_ROTATION_PAIR_PROB="${REFERENCE_ROTATION_PAIR_PROB:-0.0}"
LAMBDA_REF_ORIENTATION_CONSISTENCY="${LAMBDA_REF_ORIENTATION_CONSISTENCY:-0.0}"
LAMBDA_REF_ROTATION_STYLE="${LAMBDA_REF_ROTATION_STYLE:-0.0}"
DETAIL_DROPOUT_PROB="${DETAIL_DROPOUT_PROB:-0.0}"
MAIN_REF_RANDOM_ROTATION_PROB="${MAIN_REF_RANDOM_ROTATION_PROB:-0.0}"
MAIN_REF_RANDOM_ROTATION_MIN_DEGREES="${MAIN_REF_RANDOM_ROTATION_MIN_DEGREES:-15}"
MAIN_REF_RANDOM_ROTATION_MAX_DEGREES="${MAIN_REF_RANDOM_ROTATION_MAX_DEGREES:-180}"
MAIN_REF_RANDOM_ROTATION_RAMP_STEPS="${MAIN_REF_RANDOM_ROTATION_RAMP_STEPS:-0}"
ROTATED_REF_L1_SCALE="${ROTATED_REF_L1_SCALE:-2.0}"
ROTATED_REF_CONTENT_SCALE="${ROTATED_REF_CONTENT_SCALE:-2.0}"
ROTATED_REF_GRAM_SCALE="${ROTATED_REF_GRAM_SCALE:-0.5}"
ROTATED_REF_CONTEXTUAL_SCALE="${ROTATED_REF_CONTEXTUAL_SCALE:-0.25}"
LAMBDA_TARGET_ORIENTATION="${LAMBDA_TARGET_ORIENTATION:-0.0}"
LAMBDA_TARGET_ANISOTROPY="${LAMBDA_TARGET_ANISOTROPY:-0.0}"
TARGET_ORIENTATION_MIN_COHERENCE="${TARGET_ORIENTATION_MIN_COHERENCE:-0.20}"
TARGET_ORIENTATION_MIN_TRUST="${TARGET_ORIENTATION_MIN_TRUST:-0.50}"
TARGET_ORIENTATION_BOUNDARY_RADIUS="${TARGET_ORIENTATION_BOUNDARY_RADIUS:-1}"
TARGET_ORIENTATION_NUCLEI_RADIUS="${TARGET_ORIENTATION_NUCLEI_RADIUS:-2}"
LAMBDA_I0_WINDOW_ORIENTATION="${LAMBDA_I0_WINDOW_ORIENTATION:-0.20}"
LAMBDA_I0_WINDOW_DIRECTIONALITY="${LAMBDA_I0_WINDOW_DIRECTIONALITY:-0.02}"
LAMBDA_I0_RESIDUAL_ORIENTATION="${LAMBDA_I0_RESIDUAL_ORIENTATION:-0.10}"
LAMBDA_I0_TEXTURE_ENERGY="${LAMBDA_I0_TEXTURE_ENERGY:-0.35}"
I0_TEXTURE_ENERGY_FLOOR_RATIO="${I0_TEXTURE_ENERGY_FLOOR_RATIO:-0.95}"
I0_ORIENTATION_WINDOW_SIZES="${I0_ORIENTATION_WINDOW_SIZES:-32,64}"
I0_ORIENTATION_WINDOW_STRIDES="${I0_ORIENTATION_WINDOW_STRIDES:-16,32}"
I0_ORIENTATION_MIN_COHERENCE="${I0_ORIENTATION_MIN_COHERENCE:-0.20}"
I0_ORIENTATION_MIN_RELATIVE_ENERGY="${I0_ORIENTATION_MIN_RELATIVE_ENERGY:-0.50}"
I0_ORIENTATION_MIN_WINDOW_FRACTION="${I0_ORIENTATION_MIN_WINDOW_FRACTION:-0.25}"
I0_ORIENTATION_MIN_RESULTANT="${I0_ORIENTATION_MIN_RESULTANT:-0.15}"
I0_ORIENTATION_DIRECTIONALITY_FLOOR_RATIO="${I0_ORIENTATION_DIRECTIONALITY_FLOOR_RATIO:-0.50}"
I0_ORIENTATION_MIN_TRUST="${I0_ORIENTATION_MIN_TRUST:-0.50}"
I0_ORIENTATION_BOUNDARY_RADIUS="${I0_ORIENTATION_BOUNDARY_RADIUS:-3}"
I0_ORIENTATION_NUCLEI_RADIUS="${I0_ORIENTATION_NUCLEI_RADIUS:-5}"
I0_ORIENTATION_RAMP_STEPS="${I0_ORIENTATION_RAMP_STEPS:-200}"
CROSS4_TEXTURE_STEERING="${CROSS4_TEXTURE_STEERING:-1}"
CROSS4_STEERING_ANGLES="${CROSS4_STEERING_ANGLES:-0,30,60,90,120,150}"
CROSS4_STEERING_SMOOTHING_SIGMA="${CROSS4_STEERING_SMOOTHING_SIGMA:-8.0}"
CROSS4_STEERING_MIN_COHERENCE="${CROSS4_STEERING_MIN_COHERENCE:-0.20}"
CROSS4_STEERING_MIN_RELATIVE_ENERGY="${CROSS4_STEERING_MIN_RELATIVE_ENERGY:-0.50}"
CROSS4_STEERING_MIN_RESULTANT="${CROSS4_STEERING_MIN_RESULTANT:-0.15}"
CROSS4_STEERING_MINIMUM_STRENGTH="${CROSS4_STEERING_MINIMUM_STRENGTH:-0.70}"
CROSS4_STEERING_MINIMUM_SUPPORT="${CROSS4_STEERING_MINIMUM_SUPPORT:-0.05}"
CROSS4_STEERING_TEMPERATURE="${CROSS4_STEERING_TEMPERATURE:-0.06}"
CROSS4_STEERING_REFERENCE_MODE="${CROSS4_STEERING_REFERENCE_MODE:-local_histogram}"
CROSS4_STEERING_LOCAL_BINS="${CROSS4_STEERING_LOCAL_BINS:-36}"
CROSS4_STEERING_LOCAL_KAPPA="${CROSS4_STEERING_LOCAL_KAPPA:-8.0}"
CROSS4_STEERING_SCALES="${CROSS4_STEERING_SCALES:-1/1,1/2,1/4,1/8,1/16}"
CROSS4_STEERING_GAIN="${CROSS4_STEERING_GAIN:-1.5}"
CROSS8_STEERING_GAIN="${CROSS8_STEERING_GAIN:-2.0}"
CROSS16_STEERING_GAIN="${CROSS16_STEERING_GAIN:-2.0}"
CROSS2_STEERING_GAIN="${CROSS2_STEERING_GAIN:-1.0}"
CROSS1_STEERING_GAIN="${CROSS1_STEERING_GAIN:-1.0}"
FULL_PYRAMID_TEXTURE_STEERING="${FULL_PYRAMID_TEXTURE_STEERING:-1}"
STEERING_HIGHRES_REFERENCE_SIZE="${STEERING_HIGHRES_REFERENCE_SIZE:-8}"
LAMBDA_ANCHOR_TEACHER_CONSISTENCY="${LAMBDA_ANCHOR_TEACHER_CONSISTENCY:-0.0}"
ROTATED_I0_DROPOUT_PROB="${ROTATED_I0_DROPOUT_PROB:-0.0}"
ROTATED_I0_DROPOUT_MIN_DIAMETER="${ROTATED_I0_DROPOUT_MIN_DIAMETER:-64}"
ROTATED_I0_DROPOUT_MAX_DIAMETER="${ROTATED_I0_DROPOUT_MAX_DIAMETER:-128}"
ROTATED_I0_DROPOUT_SIGMA_MIN="${ROTATED_I0_DROPOUT_SIGMA_MIN:-2.0}"
ROTATED_I0_DROPOUT_SIGMA_MAX="${ROTATED_I0_DROPOUT_SIGMA_MAX:-3.0}"
ROTATION_MONITOR_EVERY_STEPS="${ROTATION_MONITOR_EVERY_STEPS:-0}"
ROTATION_MONITOR_STOP_PATIENCE="${ROTATION_MONITOR_STOP_PATIENCE:-2}"
ROTATION_MONITOR_MAX_CLEAN_DRIFT="${ROTATION_MONITOR_MAX_CLEAN_DRIFT:-0.03}"
ROTATION_MONITOR_MAX_REF_DISTANCE_RATIO="${ROTATION_MONITOR_MAX_REF_DISTANCE_RATIO:-1.10}"
ROTATION_MONITOR_MAX_BOUNDARY_SEAM_RATIO="${ROTATION_MONITOR_MAX_BOUNDARY_SEAM_RATIO:-1.10}"
ROTATION_MONITOR_MIN_NUCLEI_BAND_RATIO="${ROTATION_MONITOR_MIN_NUCLEI_BAND_RATIO:-0.85}"
ROTATION_MONITOR_MAX_NUCLEI_BAND_RATIO="${ROTATION_MONITOR_MAX_NUCLEI_BAND_RATIO:-1.15}"

mkdir -p "${OUTPUT_DIR}/torchrun_logs"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
LOG_FILE="${OUTPUT_DIR}/torchrun_logs/production_full_pyramid_${MAX_CONTINUATION_STEPS}.log"
STEERING_ARGS=()
if [[ "${CROSS4_TEXTURE_STEERING}" == "1" ]]; then
  STEERING_ARGS+=(--cross4-texture-steering)
fi
if [[ "${FULL_PYRAMID_TEXTURE_STEERING}" == "1" ]]; then
  STEERING_ARGS+=(--full-pyramid-texture-steering)
fi

echo "[pix2pix-production] repo=${REPO_DIR}"
echo "[pix2pix-production] output=${OUTPUT_DIR}"
echo "[pix2pix-production] resume=${RESUME}"
echo "[pix2pix-production] gpus=${GPU_IDS} nproc=${NPROC} batch_size=${BATCH_SIZE} epochs=${EPOCHS} steps=${MAX_CONTINUATION_STEPS}"
echo "[pix2pix-production] steering=${CROSS4_STEERING_REFERENCE_MODE}:${CROSS4_STEERING_SCALES} gains=${CROSS1_STEERING_GAIN}/${CROSS2_STEERING_GAIN}/${CROSS4_STEERING_GAIN}/${CROSS8_STEERING_GAIN}/${CROSS16_STEERING_GAIN}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  -m controlnet_train.pix2pix_transfer.train \
  --metadata "${TRAIN_METADATA}" \
  --val-metadata "${VAL_METADATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --i0-cache-dir "${I0_CACHE_DIR}" \
  --val-i0-cache-dir "${VAL_I0_CACHE_DIR}" \
  --image-size 256 \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --max-continuation-steps "${MAX_CONTINUATION_STEPS}" \
  --trainable-scope "${TRAINABLE_SCOPE}" \
  --highres-lr "${HIGHRES_LR}" \
  --cross4-lr "${CROSS4_LR}" \
  --midcross-lr "${MIDCROSS_LR}" \
  --backbone-lr "${BACKBONE_LR}" \
  --identity-lr "${IDENTITY_LR}" \
  --weight-decay 0.0 \
  --base-channels 64 \
  --num-heads 4 \
  --upsample-mode bilinear \
  --cross-attn-scales "1/4,1/8,1/16" \
  --region-label-mode tissue_nuclei \
  --wsi-identity-adapter \
  --identity-gamma-max 0.30 \
  --identity-gamma-init 0.10 \
  --identity-min-tissue-pixels 256 \
  --identity-min-nuclei-pixels 64 \
  --hard-pair-sampling \
  --hard-pair-full-mass 0.40 \
  --hard-pair-hard-mass 0.30 \
  --cross-wsi-style-prob 0.30 \
  --detail-dropout-prob "${DETAIL_DROPOUT_PROB}" \
  --detail-dropout-min-diameter 32 \
  --detail-dropout-max-diameter 96 \
  --detail-dropout-sigma-min 1.2 \
  --detail-dropout-sigma-max 2.5 \
  --detail-dropout-feather-radius 5 \
  --lambda-detail-fine 0.10 \
  --lambda-detail-mid 0.10 \
  --lambda-baseline-consistency 0.50 \
  --lambda-anchor-teacher-consistency "${LAMBDA_ANCHOR_TEACHER_CONSISTENCY}" \
  --rotated-i0-dropout-prob "${ROTATED_I0_DROPOUT_PROB}" \
  --rotated-i0-dropout-min-diameter "${ROTATED_I0_DROPOUT_MIN_DIAMETER}" \
  --rotated-i0-dropout-max-diameter "${ROTATED_I0_DROPOUT_MAX_DIAMETER}" \
  --rotated-i0-dropout-sigma-min "${ROTATED_I0_DROPOUT_SIGMA_MIN}" \
  --rotated-i0-dropout-sigma-max "${ROTATED_I0_DROPOUT_SIGMA_MAX}" \
  --reference-rotation-pair-prob "${REFERENCE_ROTATION_PAIR_PROB}" \
  --lambda-ref-orientation-consistency "${LAMBDA_REF_ORIENTATION_CONSISTENCY}" \
  --lambda-ref-rotation-style "${LAMBDA_REF_ROTATION_STYLE}" \
  --main-ref-random-rotation-prob "${MAIN_REF_RANDOM_ROTATION_PROB}" \
  --main-ref-random-rotation-min-degrees "${MAIN_REF_RANDOM_ROTATION_MIN_DEGREES}" \
  --main-ref-random-rotation-max-degrees "${MAIN_REF_RANDOM_ROTATION_MAX_DEGREES}" \
  --main-ref-random-rotation-ramp-steps "${MAIN_REF_RANDOM_ROTATION_RAMP_STEPS}" \
  --rotated-ref-l1-scale "${ROTATED_REF_L1_SCALE}" \
  --rotated-ref-content-scale "${ROTATED_REF_CONTENT_SCALE}" \
  --rotated-ref-gram-scale "${ROTATED_REF_GRAM_SCALE}" \
  --rotated-ref-contextual-scale "${ROTATED_REF_CONTEXTUAL_SCALE}" \
  --lambda-target-orientation "${LAMBDA_TARGET_ORIENTATION}" \
  --lambda-target-anisotropy "${LAMBDA_TARGET_ANISOTROPY}" \
  --target-orientation-min-coherence "${TARGET_ORIENTATION_MIN_COHERENCE}" \
  --target-orientation-min-trust "${TARGET_ORIENTATION_MIN_TRUST}" \
  --target-orientation-boundary-radius "${TARGET_ORIENTATION_BOUNDARY_RADIUS}" \
  --target-orientation-nuclei-radius "${TARGET_ORIENTATION_NUCLEI_RADIUS}" \
  --lambda-i0-window-orientation "${LAMBDA_I0_WINDOW_ORIENTATION}" \
  --lambda-i0-window-directionality "${LAMBDA_I0_WINDOW_DIRECTIONALITY}" \
  --lambda-i0-residual-orientation "${LAMBDA_I0_RESIDUAL_ORIENTATION}" \
  --lambda-i0-texture-energy "${LAMBDA_I0_TEXTURE_ENERGY}" \
  --i0-texture-energy-floor-ratio "${I0_TEXTURE_ENERGY_FLOOR_RATIO}" \
  --i0-orientation-window-sizes "${I0_ORIENTATION_WINDOW_SIZES}" \
  --i0-orientation-window-strides "${I0_ORIENTATION_WINDOW_STRIDES}" \
  --i0-orientation-min-coherence "${I0_ORIENTATION_MIN_COHERENCE}" \
  --i0-orientation-min-relative-energy "${I0_ORIENTATION_MIN_RELATIVE_ENERGY}" \
  --i0-orientation-min-window-fraction "${I0_ORIENTATION_MIN_WINDOW_FRACTION}" \
  --i0-orientation-min-resultant "${I0_ORIENTATION_MIN_RESULTANT}" \
  --i0-orientation-directionality-floor-ratio "${I0_ORIENTATION_DIRECTIONALITY_FLOOR_RATIO}" \
  --i0-orientation-min-trust "${I0_ORIENTATION_MIN_TRUST}" \
  --i0-orientation-boundary-radius "${I0_ORIENTATION_BOUNDARY_RADIUS}" \
  --i0-orientation-nuclei-radius "${I0_ORIENTATION_NUCLEI_RADIUS}" \
  --i0-orientation-ramp-steps "${I0_ORIENTATION_RAMP_STEPS}" \
  "${STEERING_ARGS[@]}" \
  --cross4-steering-angles "${CROSS4_STEERING_ANGLES}" \
  --cross4-steering-smoothing-sigma "${CROSS4_STEERING_SMOOTHING_SIGMA}" \
  --cross4-steering-min-coherence "${CROSS4_STEERING_MIN_COHERENCE}" \
  --cross4-steering-min-relative-energy "${CROSS4_STEERING_MIN_RELATIVE_ENERGY}" \
  --cross4-steering-min-resultant "${CROSS4_STEERING_MIN_RESULTANT}" \
  --cross4-steering-minimum-strength "${CROSS4_STEERING_MINIMUM_STRENGTH}" \
  --cross4-steering-minimum-support "${CROSS4_STEERING_MINIMUM_SUPPORT}" \
  --cross4-steering-temperature "${CROSS4_STEERING_TEMPERATURE}" \
  --cross4-steering-reference-mode "${CROSS4_STEERING_REFERENCE_MODE}" \
  --cross4-steering-local-bins "${CROSS4_STEERING_LOCAL_BINS}" \
  --cross4-steering-local-kappa "${CROSS4_STEERING_LOCAL_KAPPA}" \
  --cross4-steering-scales "${CROSS4_STEERING_SCALES}" \
  --cross4-steering-gain "${CROSS4_STEERING_GAIN}" \
  --cross8-steering-gain "${CROSS8_STEERING_GAIN}" \
  --cross16-steering-gain "${CROSS16_STEERING_GAIN}" \
  --cross2-steering-gain "${CROSS2_STEERING_GAIN}" \
  --cross1-steering-gain "${CROSS1_STEERING_GAIN}" \
  --steering-highres-reference-size "${STEERING_HIGHRES_REFERENCE_SIZE}" \
  --lambda-l1 0.25 \
  --lambda-perc 0.5 \
  --lambda-gram 1.0 \
  --lambda-contextual 1.0 \
  --lambda-identity-od 0.10 \
  --lambda-identity-feature 0.25 \
  --lambda-identity-band 0.15 \
  --lambda-identity-rank 0.10 \
  --cross-lambda-identity-od 0.20 \
  --cross-lambda-identity-feature 0.40 \
  --cross-lambda-identity-band 0.20 \
  --cross-lambda-identity-rank 0.15 \
  --cross-lambda-gram 0.50 \
  --cross-lambda-contextual 0.50 \
  --lambda-structure-gray 0.50 \
  --lambda-structure-edge 0.50 \
  --identity-rank-margin 0.10 \
  --lambda-adv 0.03 \
  --adv-warmup-steps 1500 \
  --adv-mask-mode non_background \
  --d-lr 0.00001 \
  --d-weight-decay 0.0 \
  --d-base-channels 64 \
  --d-max-channels 512 \
  --d-num-layers 3 \
  --boundary-adv-floor 0.20 \
  --condition-mismatch-d-weight 0.10 \
  --context-ramp-steps 500 \
  --rotation-monitor-every-steps "${ROTATION_MONITOR_EVERY_STEPS}" \
  --rotation-monitor-stop-patience "${ROTATION_MONITOR_STOP_PATIENCE}" \
  --rotation-monitor-max-clean-drift "${ROTATION_MONITOR_MAX_CLEAN_DRIFT}" \
  --rotation-monitor-max-ref-distance-ratio "${ROTATION_MONITOR_MAX_REF_DISTANCE_RATIO}" \
  --rotation-monitor-max-boundary-seam-ratio "${ROTATION_MONITOR_MAX_BOUNDARY_SEAM_RATIO}" \
  --rotation-monitor-min-nuclei-band-ratio "${ROTATION_MONITOR_MIN_NUCLEI_BAND_RATIO}" \
  --rotation-monitor-max-nuclei-band-ratio "${ROTATION_MONITOR_MAX_NUCLEI_BAND_RATIO}" \
  --l1-blur-sigma 2.0 \
  --content-layers "15,22" \
  --gram-layers "3,8,15" \
  --contextual-layers "8,15" \
  --texture-min-pixels 8 \
  --contextual-max-samples 256 \
  --contextual-temperature 0.1 \
  --loss-normalization-decay 0.99 \
  --loss-normalization-steps 200 \
  --vgg-weights imagenet \
  --mixed-precision bf16 \
  --grad-clip 1.0 \
  --ref-trust-gate \
  --ref-fallback-scale 0.05 \
  --ref-soft-context-scale 0.20 \
  --ref-nuclei-context-scale 0.00 \
  --ref-soft-context-radius 5 \
  --matched-tissue-trust-floor 0.65 \
  --matched-nuclei-trust-floor 0.70 \
  --boundary-feather-radius 5 \
  --lambda-boundary-hf 2.0 \
  --lambda-lowtrust-hf 0.5 \
  --ood-diagnose-every-epochs 0 \
  --log-every 25 \
  --sample-every 100 \
  --eval-every-epochs 0 \
  --eval-num-samples 5 \
  --eval-batch-size 5 \
  --eval-seed 123 \
  --save-every 1 \
  --resume "${RESUME}" \
  --seed 42 \
  --device cuda \
  2>&1 | tee -a "${LOG_FILE}"

printf -v STEP_SUFFIX '%06d' "${MAX_CONTINUATION_STEPS}"
EXPECTED_CHECKPOINT="${OUTPUT_DIR}/ckpt/pilot_step${STEP_SUFFIX}.pt"
test -f "${EXPECTED_CHECKPOINT}"
echo "[pix2pix-production] completed checkpoint=${EXPECTED_CHECKPOINT}"
