#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 FIRST_PASS_PID DOWNLOAD_PID [DOWNLOAD_PID ...]" >&2
  exit 2
fi

FIRST_PASS_PID=$1
shift
DOWNLOAD_PIDS=("$@")

BASE=/data1/zhao/wqx/benchmarks
RAG_ROOT="$BASE/weights/unipath/UniPath-68K/RAG_8K"
WSI_ROOT="$BASE/weights/unipath/TCGA_WSI"
PYTHON="$BASE/envs/unipath/bin/python"
EXTRACTOR="$BASE/scripts/extract_unipath_patches_from_h5.py"
LOG="$BASE/logs/extract_unipath_rag_patches.log"

ORIGINAL_MD5=407dd679e308628f516b5fd1ac380536
ORIGINAL_FILE="$WSI_ROOT/69d9593e-6189-453a-9042-28e6485f1a93/TCGA-49-6761-01Z-00-DX1.10c577a2-f65c-4517-8a25-83d44e380f8f.svs"

REPLACEMENT_MD5=70f40f5c4d54676d00cb13359f75273a
REPLACEMENT_DIR="$WSI_ROOT/3f343c82-0bc4-4eff-9352-8fbbb3e88758"
REPLACEMENT_FILE="$REPLACEMENT_DIR/TCGA-YU-A94I-01Z-00-DX1.9A66760C-01EB-4722-B462-0BDE80EFB6EA.svs"
ALIAS_FILE="$REPLACEMENT_DIR/TCGA-YU-A94I-01Z-00-DX1.CF7FD72C-A39C-48BD-9E93-3EBAAA3457BA.svs"

wait_for_pid() {
  local pid=$1
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

echo "$(date -Is) waiting for first extraction PID $FIRST_PASS_PID"
wait_for_pid "$FIRST_PASS_PID"

for pid in "${DOWNLOAD_PIDS[@]}"; do
  echo "$(date -Is) waiting for download PID $pid"
  wait_for_pid "$pid"
done

echo "$(date -Is) verifying downloaded WSI files"
echo "$ORIGINAL_MD5  $ORIGINAL_FILE" | md5sum --check --status
echo "$REPLACEMENT_MD5  $REPLACEMENT_FILE" | md5sum --check --status

if [ ! -e "$ALIAS_FILE" ]; then
  ln -s "$REPLACEMENT_FILE" "$ALIAS_FILE"
fi

echo "$(date -Is) running final extraction pass"
nice -n 10 "$PYTHON" -u "$EXTRACTOR" \
  --h5 "$RAG_ROOT/selected_8k.h5" \
  --wsi-dir "$WSI_ROOT" \
  >>"$LOG" 2>&1

IMAGE_COUNT=$(find "$RAG_ROOT/images" -maxdepth 1 -type f -name "*.png" | wc -l)
echo "$(date -Is) final image count: $IMAGE_COUNT"
test "$IMAGE_COUNT" -eq 8000
