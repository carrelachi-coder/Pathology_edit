#!/usr/bin/env python3
"""Launch a frozen compilation batch only on repeatedly observed idle GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable


def parse_rows(text: str, *, maximum_utilization: int, maximum_memory_mib: int) -> list[int]:
    eligible: list[int] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise ValueError(f"unexpected nvidia-smi row: {raw}")
        index, utilization, memory = (int(value) for value in parts)
        if utilization <= maximum_utilization and memory <= maximum_memory_mib:
            eligible.append(index)
    if len(eligible) != len(set(eligible)):
        raise ValueError("nvidia-smi returned duplicate physical GPU indices")
    return sorted(eligible)


def observe(
    query: Callable[[], str], *, maximum_utilization: int, maximum_memory_mib: int
) -> dict:
    text = query()
    return {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": text.splitlines(),
        "eligible_physical_gpu_indices": parse_rows(
            text,
            maximum_utilization=maximum_utilization,
            maximum_memory_mib=maximum_memory_mib,
        ),
    }


def stable_devices(first: dict, second: dict, maximum_devices: int) -> list[int]:
    return sorted(
        set(first["eligible_physical_gpu_indices"])
        & set(second["eligible_physical_gpu_indices"])
    )[:maximum_devices]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--case-index", type=int, action="append", required=True)
    parser.add_argument("--seconds-per-case", type=int, default=7200)
    parser.add_argument("--minimum-devices", type=int, default=1)
    parser.add_argument("--maximum-devices", type=int, default=8)
    parser.add_argument("--maximum-utilization", type=int, default=10)
    parser.add_argument("--maximum-memory-mib", type=int, default=50000)
    parser.add_argument("--observation-interval-seconds", type=int, default=60)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists() or args.gate_receipt.exists():
        raise ValueError("output and gate receipt must be new append-only paths")
    if not args.controller.is_file() or not args.cases.is_file() or not args.candidate_manifest.is_file():
        raise ValueError("a frozen controller input is missing")
    if not 1 <= args.minimum_devices <= args.maximum_devices:
        raise ValueError("invalid device count bounds")
    if args.observation_interval_seconds <= 0 or args.poll_interval_seconds <= 0:
        raise ValueError("observation intervals must be positive")
    selected_indices = sorted(set(args.case_index))
    if len(selected_indices) != len(args.case_index) or selected_indices[0] < 0:
        raise ValueError("case indices must be unique and non-negative")

    def query() -> str:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    attempts = 0
    while True:
        first = observe(
            query,
            maximum_utilization=args.maximum_utilization,
            maximum_memory_mib=args.maximum_memory_mib,
        )
        if len(first["eligible_physical_gpu_indices"]) < args.minimum_devices:
            time.sleep(args.poll_interval_seconds)
            attempts += 1
            continue
        time.sleep(args.observation_interval_seconds)
        second = observe(
            query,
            maximum_utilization=args.maximum_utilization,
            maximum_memory_mib=args.maximum_memory_mib,
        )
        devices = stable_devices(first, second, args.maximum_devices)
        attempts += 1
        if len(devices) < args.minimum_devices:
            continue
        break

    gate = {
        "schema_version": "reserved-four-target-idle-gpu-gate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "first_observation": first,
        "second_observation": second,
        "selected_physical_gpu_indices": devices,
        "maximum_utilization": args.maximum_utilization,
        "maximum_memory_mib": args.maximum_memory_mib,
        "observation_interval_seconds": args.observation_interval_seconds,
        "poll_attempts": attempts,
        "case_indices": selected_indices,
        "automatic_retries": 0,
    }
    args.gate_receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.gate_receipt.open("x", encoding="utf-8", errors="strict") as handle:
        handle.write(json.dumps(gate, indent=2) + "\n")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    command = [
        sys.executable,
        str(args.controller.resolve()),
        "--cases",
        str(args.cases.resolve()),
        "--candidate-manifest",
        str(args.candidate_manifest.resolve()),
        "--output",
        str(args.output.resolve()),
        "--seconds-per-case",
        str(args.seconds_per_case),
    ]
    for visible_index in range(len(devices)):
        command.extend(["--device", f"cuda:{visible_index}"])
    for case_index in selected_indices:
        command.extend(["--case-index", str(case_index)])
    completed = subprocess.run(command, env=environment, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
