#!/usr/bin/env python3
"""Run one frozen command after the same physical GPU is observed idle twice."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def parse_rows(text: str, *, maximum_utilization: int, maximum_memory_mib: int) -> list[int]:
    eligible: list[int] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise ValueError(f"unexpected nvidia-smi row: {raw}")
        index, memory, utilization = (int(value) for value in parts)
        if utilization <= maximum_utilization and memory <= maximum_memory_mib:
            eligible.append(index)
    if len(eligible) != len(set(eligible)):
        raise ValueError("nvidia-smi returned duplicate physical GPU indices")
    return sorted(eligible)


def query_gpus() -> tuple[str, list[str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return datetime.now(timezone.utc).isoformat(), completed.stdout.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--maximum-utilization", type=int, default=10)
    parser.add_argument("--maximum-memory-mib", type=int, default=20000)
    parser.add_argument("--observation-interval-seconds", type=int, default=30)
    parser.add_argument("--poll-interval-seconds", type=int, default=15)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != "--" or len(args.command) == 1:
        raise ValueError("a command must follow --")
    if args.gate_receipt.exists():
        raise ValueError("gate receipt path already exists")
    if min(args.observation_interval_seconds, args.poll_interval_seconds) <= 0:
        raise ValueError("intervals must be positive")

    attempts = 0
    while True:
        first_utc, first_rows = query_gpus()
        first = parse_rows(
            "\n".join(first_rows),
            maximum_utilization=args.maximum_utilization,
            maximum_memory_mib=args.maximum_memory_mib,
        )
        attempts += 1
        if not first:
            time.sleep(args.poll_interval_seconds)
            continue
        time.sleep(args.observation_interval_seconds)
        second_utc, second_rows = query_gpus()
        second = parse_rows(
            "\n".join(second_rows),
            maximum_utilization=args.maximum_utilization,
            maximum_memory_mib=args.maximum_memory_mib,
        )
        stable = sorted(set(first) & set(second))
        if stable:
            selected = stable[0]
            break

    receipt = {
        "schema_version": "idle-gpu-single-command-gate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "first_observation": {"observed_utc": first_utc, "raw_rows": first_rows},
        "second_observation": {"observed_utc": second_utc, "raw_rows": second_rows},
        "selected_physical_gpu_index": selected,
        "maximum_utilization": args.maximum_utilization,
        "maximum_memory_mib": args.maximum_memory_mib,
        "observation_interval_seconds": args.observation_interval_seconds,
        "poll_attempts": attempts,
        "automatic_retries": 0,
        "command": args.command[1:],
    }
    args.gate_receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.gate_receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(selected)
    return subprocess.run(args.command[1:], env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
