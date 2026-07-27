#!/usr/bin/env bash
set -euo pipefail
umask 027

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data1/zhao/wqx/benchmarks/data/representation_two_primitive_bcss_paired300_native_v1_sampling_v3_epoch29}"
RUN_ROOT="${RUN_ROOT:-/data1/zhao/wqx/benchmarks/runs/representation_two_primitive_bcss_paired300_native_v1_sampling_v3_epoch29}"
MANIFEST="${MANIFEST:-$DATA_ROOT/manifests/combined_nuclei_manifest.jsonl}"
SMOKE_MANIFEST="${SMOKE_MANIFEST:-$DATA_ROOT/manifests/smoke_nuclei_manifest.jsonl}"
NUCLEI_ROOT="${NUCLEI_ROOT:-$DATA_ROOT/nuclei}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/data/huggingface/FLUX.1-dev}"
INPAINT_CHECKPOINT="${INPAINT_CHECKPOINT:-$ROOT/phase5_runs/controlnet_inpaint_all}"
CROSS_CHECKPOINT="${CROSS_CHECKPOINT:-$ROOT/phase5_runs/controlnet_cross_v1}"
PIX2PIX_CHECKPOINT="${PIX2PIX_CHECKPOINT:-/data/wqx/flowedit/pix2pix_texture_transfer_lazy_ver4_wsi_identity_i0_local_full_pyramid_v3_ft/ckpt/pilot_step001000.pt}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
GPU_CSV="${GPU_CSV:-1,2,7}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1200}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-20000}"

IFS=',' read -r -a GPUS <<<"$GPU_CSV"
NUM_SHARDS="${#GPUS[@]}"
if (( NUM_SHARDS < 1 )); then
  echo "GPU_CSV must contain at least one GPU" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/reports"
cd "$ROOT"
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

if [[ -n "$WAIT_FOR_PID" ]]; then
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 20
  done
fi

"$PYTHON" scripts/audit_two_primitive_nuclei.py \
  --manifest "$MANIFEST" \
  --output "$DATA_ROOT/full_nuclei_audit.json" \
  --expected-count "$EXPECTED_COUNT" \
  --minimum-placement-completion 0.98 \
  > "$RUN_ROOT/logs/nuclei_audit.log" 2>&1
"$PYTHON" scripts/split_two_primitive_nuclei_manifests.py \
  --manifest "$MANIFEST" \
  --output-root "$DATA_ROOT/manifests" \
  --expected-count "$EXPECTED_COUNT" \
  --expected-group-count 300 \
  > "$RUN_ROOT/logs/split_nuclei_manifests.log" 2>&1
touch "$DATA_ROOT/NUCLEI_COMPLETE"

for path in \
  "$PRETRAINED_MODEL" \
  "$INPAINT_CHECKPOINT" \
  "$CROSS_CHECKPOINT" \
  "$PIX2PIX_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "required generation artifact missing: $path" >&2
    exit 3
  fi
done

wait_for_gpu_capacity() {
  local gpu used all_ready
  while true; do
    all_ready=1
    for gpu in "${GPUS[@]}"; do
      used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      if (( used > MAX_GPU_USED_MIB )); then
        all_ready=0
      fi
    done
    if (( all_ready == 1 )); then
      return 0
    fi
    sleep 30
  done
}

run_smoke() {
  local backend gpu
  gpu="${GPUS[0]}"
  wait_for_gpu_capacity
  for backend in inpaint cross-v1; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      scripts/run_embedding_utility_generation.py \
      --cohort-manifest "$SMOKE_MANIFEST" \
      --output-root "$RUN_ROOT/generated" \
      --backend "$backend" \
      --pretrained-model "$PRETRAINED_MODEL" \
      --inpaint-checkpoint "$INPAINT_CHECKPOINT" \
      --cross-v1-checkpoint "$CROSS_CHECKPOINT" \
      --pix2pix-checkpoint "$PIX2PIX_CHECKPOINT" \
      --device cuda \
      --dtype bf16 \
      --num-inference-steps 28 \
      --guidance-scale 3.5 \
      --controlnet-conditioning-scale 1.0 \
      --expected-count 4 \
      --resume \
      > "$RUN_ROOT/logs/${backend}_smoke.log" 2>&1
  done
  "$PYTHON" scripts/audit_two_primitive_generation.py \
    --manifest "$SMOKE_MANIFEST" \
    --generated-root "$RUN_ROOT/generated" \
    --expected-nuclei-root "$NUCLEI_ROOT" \
    --expected-pix2pix-checkpoint "$PIX2PIX_CHECKPOINT" \
    --expected-count 4 \
    --expected-size 512 \
    --output "$RUN_ROOT/reports/smoke_generation_audit.json" \
    > "$RUN_ROOT/logs/smoke_generation_audit.log" 2>&1
  touch "$RUN_ROOT/GENERATION_SMOKE_COMPLETE"
}

run_backend() {
  local backend="$1"
  local shard gpu pid status count
  local -a pids=()
  wait_for_gpu_capacity
  for shard in "${!GPUS[@]}"; do
    gpu="${GPUS[$shard]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      scripts/run_embedding_utility_generation.py \
      --cohort-manifest "$MANIFEST" \
      --output-root "$RUN_ROOT/generated" \
      --backend "$backend" \
      --pretrained-model "$PRETRAINED_MODEL" \
      --inpaint-checkpoint "$INPAINT_CHECKPOINT" \
      --cross-v1-checkpoint "$CROSS_CHECKPOINT" \
      --pix2pix-checkpoint "$PIX2PIX_CHECKPOINT" \
      --device cuda \
      --dtype bf16 \
      --num-inference-steps 28 \
      --guidance-scale 3.5 \
      --controlnet-conditioning-scale 1.0 \
      --expected-count "$EXPECTED_COUNT" \
      --shard-index "$shard" \
      --num-shards "$NUM_SHARDS" \
      --resume \
      > "$RUN_ROOT/logs/${backend}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "$backend generation has a failed shard" >&2
    return 4
  fi
  count="$(
    find "$RUN_ROOT/generated/$backend" \
      -mindepth 2 -maxdepth 2 -name generated_image.png -type f |
      wc -l |
      tr -d ' '
  )"
  if [[ "$count" != "$EXPECTED_COUNT" ]]; then
    echo "$backend generated count $count != $EXPECTED_COUNT" >&2
    return 5
  fi
}

run_smoke
run_backend inpaint
touch "$RUN_ROOT/INPAINT_COMPLETE"
run_backend cross-v1
touch "$RUN_ROOT/CROSS_COMPLETE"

"$PYTHON" scripts/audit_two_primitive_generation.py \
  --manifest "$MANIFEST" \
  --generated-root "$RUN_ROOT/generated" \
  --expected-nuclei-root "$NUCLEI_ROOT" \
  --expected-pix2pix-checkpoint "$PIX2PIX_CHECKPOINT" \
  --expected-count "$EXPECTED_COUNT" \
  --expected-size 512 \
  --output "$RUN_ROOT/reports/full_generation_audit.json" \
  > "$RUN_ROOT/logs/full_generation_audit.log" 2>&1

touch "$RUN_ROOT/GENERATION_COMPLETE"
echo "two-primitive generation complete: targets=$EXPECTED_COUNT backends=2"
