#!/usr/bin/env python3
"""Run a resumable sequence of generation baselines on one GPU shard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase3_mask_edit.benchmark.generation_models import (
    load_generation_model_configs,
)


DEFAULT_MODELS = [
    "pixcell_controlnet",
    "mupad_text",
    "mupad_image_auxiliary",
    "pathldm_plip",
    "pathdiff_conic",
]
FORBIDDEN_MODELS = {"cross_v1_project", "unipath_7b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("benchmark_configs/models"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--cuda-visible-device", required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def build_commands(args: argparse.Namespace) -> list[tuple[str, str]]:
    configs = load_generation_model_configs(args.config_dir)
    unknown = sorted(set(args.models) - set(configs))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    forbidden = sorted(set(args.models) & FORBIDDEN_MODELS)
    if forbidden:
        raise ValueError(f"models explicitly excluded from this run: {forbidden}")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    return [
        (
            model_id,
            configs[model_id].build_remote_command(
                manifest=str(args.manifest),
                output_root=str(args.output_root),
                device="cuda:0",
                num_shards=args.num_shards,
                shard_index=args.shard_index,
            ),
        )
        for model_id in args.models
    ]


def main() -> int:
    args = parse_args()
    commands = build_commands(args)
    if args.dry_run:
        for model_id, command in commands:
            print(f"[{model_id}] CUDA_VISIBLE_DEVICES={args.cuda_visible_device} {command}")
        return 0

    state_path = args.state_root / (
        f"queue_shard{args.shard_index}of{args.num_shards}.json"
    )
    state = {
        "schema_version": 1,
        "status": "running",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "cuda_visible_device": str(args.cuda_visible_device),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "models": args.models,
        "excluded_models": ["cross_v1_project", "unipath_7b"],
        "deferred_models": ["pathdiff_text"],
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "model_runs": {},
    }
    write_json_atomic(state_path, state)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_device)
    failed_models = []
    for model_id, command in commands:
        model_state = {"status": "running", "attempts": [], "started_at_utc": utc_now()}
        state["model_runs"][model_id] = model_state
        state["current_model"] = model_id
        state["updated_at_utc"] = utc_now()
        write_json_atomic(state_path, state)
        succeeded = False
        for attempt in range(1, args.attempts + 1):
            started = time.time()
            return_code = subprocess.run(
                ["bash", "-lc", command], env=environment, check=False
            ).returncode
            model_state["attempts"].append(
                {
                    "attempt": attempt,
                    "return_code": return_code,
                    "runtime_seconds": round(time.time() - started, 3),
                    "completed_at_utc": utc_now(),
                }
            )
            state["updated_at_utc"] = utc_now()
            write_json_atomic(state_path, state)
            if return_code == 0:
                succeeded = True
                break
            if attempt < args.attempts:
                time.sleep(args.retry_delay_seconds)
        model_state["status"] = "completed" if succeeded else "failed"
        model_state["completed_at_utc"] = utc_now()
        if not succeeded:
            failed_models.append(model_id)
        write_json_atomic(state_path, state)

    state.pop("current_model", None)
    state["status"] = "failed" if failed_models else "completed"
    state["failed_models"] = failed_models
    state["completed_at_utc"] = utc_now()
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    return 1 if failed_models else 0


if __name__ == "__main__":
    raise SystemExit(main())
