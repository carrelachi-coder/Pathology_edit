#!/usr/bin/env bash
set -euo pipefail
umask 027

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
MANIFEST="${MANIFEST:-/data1/zhao/wqx/benchmarks/data/complex_paired_v3_1500/paired_directions_strict_oral_1454.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data1/zhao/wqx/benchmarks/runs/cross_v1_epoch26_autocond_strict_oral_1454_20260802}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/data/huggingface/FLUX.1-dev}"
CROSS_CHECKPOINT="${CROSS_CHECKPOINT:-/home/lyw/wqx-DL/flow-edit/hf_generation_release/pathology-cross-v1-pix2pix/cross_v1}"
PIX2PIX_CHECKPOINT="${PIX2PIX_CHECKPOINT:-/home/lyw/wqx-DL/flow-edit/hf_generation_release/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt}"
GPU_CSV="${GPU_CSV:-1,2,3,7}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1454}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-5000}"
MAX_PASSES="${MAX_PASSES:-3}"
EXPECTED_PIX2PIX_SHA256="be5fe9376efdb5620a57481082f6d5738b6353796fb00fe6e58f6b212ba7c2ac"

IFS=',' read -r -a GPUS <<<"$GPU_CSV"
NUM_SHARDS="${#GPUS[@]}"
if (( NUM_SHARDS < 1 )); then
  echo "GPU_CSV must contain at least one GPU" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/state"
cd "$ROOT"
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATHOLOGY_CROSS_V1_CHECKPOINT="$CROSS_CHECKPOINT"
export PATHOLOGY_PIX2PIX_CHECKPOINT="$PIX2PIX_CHECKPOINT"

for path in \
  "$PYTHON" \
  "$MANIFEST" \
  "$PRETRAINED_MODEL" \
  "$CROSS_CHECKPOINT" \
  "$PIX2PIX_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "required artifact missing: $path" >&2
    exit 3
  fi
done

actual_sha256="$(sha256sum "$PIX2PIX_CHECKPOINT" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_PIX2PIX_SHA256" ]]; then
  echo "pix2pix SHA256 mismatch: $actual_sha256" >&2
  exit 4
fi

wait_for_gpu_capacity() {
  local all_ready gpu used
  while true; do
    all_ready=1
    for gpu in "${GPUS[@]}"; do
      used="$(
        nvidia-smi --id="$gpu" --query-gpu=memory.used \
          --format=csv,noheader,nounits | tr -d ' '
      )"
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

status=1
for pass_index in $(seq 1 "$MAX_PASSES"); do
  wait_for_gpu_capacity
  declare -a pids=()
  for shard in "${!GPUS[@]}"; do
    gpu="${GPUS[$shard]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      scripts/run_cross_v1_pathokid_batch.py \
      --manifest "$MANIFEST" \
      --output-root "$OUTPUT_ROOT" \
      --pretrained-model "$PRETRAINED_MODEL" \
      --cross-v1-checkpoint "$CROSS_CHECKPOINT" \
      --pix2pix-checkpoint "$PIX2PIX_CHECKPOINT" \
      --device cuda \
      --dtype bf16 \
      --num-inference-steps 28 \
      --guidance-scale 3.5 \
      --controlnet-conditioning-scale 1.0 \
      --expected-count "$EXPECTED_COUNT" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --resume \
      > "$OUTPUT_ROOT/logs/pass${pass_index}_shard${shard}of${NUM_SHARDS}.log" 2>&1 &
    pids+=("$!")
    echo "$!" > "$OUTPUT_ROOT/state/shard${shard}of${NUM_SHARDS}.pid"
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status == 0 )); then
    break
  fi
  echo "Cross generation pass $pass_index had failures; resuming missing rows" >&2
  sleep 30
done
if (( status != 0 )); then
  echo "Cross generation still has failed shards after $MAX_PASSES passes" >&2
  exit 6
fi

"$PYTHON" scripts/run_cross_v1_pathokid_batch.py \
  --manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --pretrained-model "$PRETRAINED_MODEL" \
  --cross-v1-checkpoint "$CROSS_CHECKPOINT" \
  --pix2pix-checkpoint "$PIX2PIX_CHECKPOINT" \
  --expected-count "$EXPECTED_COUNT" \
  --audit-only \
  > "$OUTPUT_ROOT/logs/final_audit.log" 2>&1

touch "$OUTPUT_ROOT/GENERATION_COMPLETE"
echo "Cross V1 epoch-26 AutoCond generation complete: $OUTPUT_ROOT"
