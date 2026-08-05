#!/usr/bin/env bash
set -euo pipefail
umask 027

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
CROSS_ROOT="${CROSS_ROOT:-/data1/zhao/wqx/benchmarks/runs/cross_v1_epoch26_autocond_strict_oral_1454_20260802}"
PRIOR_ROOT="${PRIOR_ROOT:-/data1/zhao/wqx/benchmarks/runs}"
MANIFEST="${MANIFEST:-$PRIOR_ROOT/pathokid_conch_generation_v3_plus_unipath_common1421_strict_oral_20260724/evaluation_manifest_1421.json}"
CONCH_OUTPUT="${CONCH_OUTPUT:-$PRIOR_ROOT/pathokid_conch_generation_v3_plus_cross_common1421_strict_oral_20260802}"
UNI2H_OUTPUT="${UNI2H_OUTPUT:-$PRIOR_ROOT/pathokid_uni2h_generation_v3_plus_cross_common1421_strict_oral_20260802}"
GPU_ID="${GPU_ID:-7}"
EXPECTED_MANIFEST_SHA256="8cc8e2086b1c218ffd2a8328d6be3c6105113994dac7e6505cc59a9387e6d3d8"

BASELINE_ROOT="$PRIOR_ROOT/generation_baselines_mpp_corrected_v3_1500_20260716"
PATHDIFF_TEXT_ROOT="$PRIOR_ROOT/generation_baselines_pathdiff_text_20x_normalized_strict_oral_1454_20260723/pathdiff_text"
UNIPATH_ROOT="$PRIOR_ROOT/generation_baselines_unipath_rag_normalized_strict_oral_1454_20260724/unipath_7b"
CONCH_PRIOR="$PRIOR_ROOT/pathokid_conch_generation_v3_plus_unipath_common1421_strict_oral_20260724"
UNI2H_PRIOR="$PRIOR_ROOT/pathokid_uni2h_generation_v3_plus_unipath_common1421_strict_oral_20260724"

cd "$ROOT"
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$CROSS_ROOT/GENERATION_COMPLETE" ]]; then
  echo "Cross generation is not complete: $CROSS_ROOT" >&2
  exit 2
fi
actual_manifest_sha256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
if [[ "$actual_manifest_sha256" != "$EXPECTED_MANIFEST_SHA256" ]]; then
  echo "Patho-KID common-set manifest SHA256 mismatch: $actual_manifest_sha256" >&2
  exit 3
fi

declare -a model_roots=(
  --model-root "cross_v1_epoch26_autocond=$CROSS_ROOT"
  --model-root "pixcell_controlnet=$BASELINE_ROOT/pixcell_controlnet"
  --model-root "pathdiff_conic=$BASELINE_ROOT/pathdiff_conic"
  --model-root "mupad_text=$BASELINE_ROOT/mupad_text"
  --model-root "mupad_image_auxiliary=$BASELINE_ROOT/mupad_image_auxiliary"
  --model-root "pathdiff_text_20x=$PATHDIFF_TEXT_ROOT"
  --model-root "unipath_7b=$UNIPATH_ROOT"
)

seed_cache() {
  local extractor="$1"
  local prior="$2"
  local output="$3"
  mkdir -p "$output/cache"
  if [[ ! -d "$output/cache/$extractor" ]]; then
    cp -al "$prior/cache/$extractor" "$output/cache/$extractor"
  fi
}

run_pathokid() {
  local extractor="$1"
  local output="$2"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
    phase3_mask_edit/cli/run_pathokid_benchmark.py \
    --config benchmark_configs/pathokid.yaml \
    --manifest "$MANIFEST" \
    --output-root "$output" \
    "${model_roots[@]}" \
    --feature-extractors "$extractor" \
    --bootstrap-group-field pair_id \
    --device cuda \
    --dtype bf16 \
    --batch-size 16 \
    --subset-size 1000 \
    --subset-repeats 100 \
    --bootstrap-repeats 100 \
    --seed 20260715 \
    > "$output/run.log" 2>&1
}

seed_cache conch "$CONCH_PRIOR" "$CONCH_OUTPUT"
run_pathokid conch "$CONCH_OUTPUT"

seed_cache uni2h "$UNI2H_PRIOR" "$UNI2H_OUTPUT"
run_pathokid uni2h "$UNI2H_OUTPUT"

touch "$CONCH_OUTPUT/PATHOKID_COMPLETE" "$UNI2H_OUTPUT/PATHOKID_COMPLETE"
echo "Cross-inclusive CONCH/UNI-2h Patho-KID complete"
