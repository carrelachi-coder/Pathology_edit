#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
PYTHON="${PYTHON:-/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python}"
RUN_ROOT="${RUN_ROOT:-/data1/zhao/wqx/benchmarks/runs/representation_two_primitive_bcss_paired300_native_v1_sampling_v3_epoch29}"
COUNT="${COUNT:-300}"

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/embeddings" "$RUN_ROOT/dose_response"

cells=(u1_moderate u1_significant u2_moderate u2_significant)
encoders=(uni2h conch)
pids=()
labels=()

for encoder_index in "${!encoders[@]}"; do
  encoder="${encoders[$encoder_index]}"
  for cell_index in "${!cells[@]}"; do
    cell="${cells[$cell_index]}"
    gpu=$((encoder_index * 4 + cell_index))
    manifest="$RUN_ROOT/manifests/${cell}_evaluation_manifest.jsonl"
    output="$RUN_ROOT/embeddings/${encoder}/${cell}"
    log="$RUN_ROOT/logs/embedding_${encoder}_${cell}.log"
    mkdir -p "$output"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$PYTHON" \
      -m phase3_mask_edit.cli.run_embedding_utility_benchmark \
      --feature-extractor "$encoder" \
      --manifest "$manifest" \
      --output-root "$output" \
      --device cuda \
      --expected-count "$COUNT" \
      --bootstrap-repeats 1000 \
      >"$log" 2>&1 &
    pids+=("$!")
    labels+=("${encoder}/${cell}/gpu${gpu}")
  done
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf 'complete %s\n' "${labels[$index]}"
  else
    code="$?"
    printf 'failed %s exit=%s\n' "${labels[$index]}" "$code" >&2
    status="$code"
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

for encoder in "${encoders[@]}"; do
  for primitive in u1 u2; do
    PYTHONPATH=. "$PYTHON" \
      -m phase3_mask_edit.cli.run_embedding_utility_dose_response \
      --moderate-manifest \
        "$RUN_ROOT/manifests/${primitive}_moderate_evaluation_manifest.jsonl" \
      --moderate-cache-root \
        "$RUN_ROOT/embeddings/${encoder}/${primitive}_moderate/cache/${encoder}" \
      --significant-manifest \
        "$RUN_ROOT/manifests/${primitive}_significant_evaluation_manifest.jsonl" \
      --significant-cache-root \
        "$RUN_ROOT/embeddings/${encoder}/${primitive}_significant/cache/${encoder}" \
      --output-root "$RUN_ROOT/dose_response/${encoder}/${primitive}" \
      --feature-extractor-name "$encoder" \
      --dose-field realized_dose_fraction \
      --moderate-dose-field moderate_realized_dose_fraction \
      --expected-count "$COUNT" \
      --bootstrap-repeats 1000 \
      >"$RUN_ROOT/logs/dose_response_${encoder}_${primitive}.log" 2>&1
  done
done

touch "$RUN_ROOT/EMBEDDINGS_COMPLETE"
printf 'two-primitive embeddings and within-primitive dose responses complete\n'
