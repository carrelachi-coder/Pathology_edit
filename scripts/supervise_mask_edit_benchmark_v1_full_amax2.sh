#!/usr/bin/env bash
set -u

REPO_ROOT="${REPO_ROOT:-/home/lyw/wqx-DL/flow-edit/FlowEdit-main}"
FULL_ROOT="${FULL_ROOT:-/data1/zhao/wqx/benchmark_v1/full_v0_2}"
CURRENT_PID="${CURRENT_PID:-}"
MAX_RECOVERY_ATTEMPTS="${MAX_RECOVERY_ATTEMPTS:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG="$FULL_ROOT/full_benchmark_recovery.log"

mkdir -p "$FULL_ROOT"
exec 9>"$FULL_ROOT/.full_benchmark_recovery.lock"
if ! flock -n 9; then
  echo "A full benchmark recovery supervisor is already running."
  exit 0
fi

cd "$REPO_ROOT"

if [[ -n "$CURRENT_PID" ]]; then
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
fi

for ((attempt = 1; attempt <= MAX_RECOVERY_ATTEMPTS; attempt++)); do
  if [[ -f "$FULL_ROOT/FULL_COMPLETE" ]]; then
    exit 0
  fi

  printf '%s recovery_attempt=%d\n' "$(date -Is)" "$attempt" >>"$LOG"
  bash scripts/run_mask_edit_benchmark_v1_full_amax2.sh >>"$LOG" 2>&1
  status=$?
  printf '%s recovery_exit=%d\n' "$(date -Is)" "$status" >>"$LOG"

  if [[ -f "$FULL_ROOT/FULL_COMPLETE" ]]; then
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

exit 1
