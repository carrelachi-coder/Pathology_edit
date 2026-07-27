#!/usr/bin/env python3
"""Wait for generation shards, normalize to 512, and validate the complete run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_MODELS = [
    "pixcell_controlnet",
    "mupad_text",
    "mupad_image_auxiliary",
    "pathldm_plip",
    "pathdiff_conic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--cross-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def queue_states(state_root: Path, num_shards: int) -> list[dict] | None:
    states = []
    for shard_index in range(num_shards):
        path = state_root / f"queue_shard{shard_index}of{num_shards}.json"
        if not path.exists():
            return None
        states.append(json.loads(path.read_text(encoding="utf-8")))
    return states


def main() -> int:
    args = parse_args()
    state_path = args.state_root / "finalization.json"
    state = {
        "schema_version": 1,
        "status": "waiting_for_generation",
        "manifest": str(args.manifest),
        "raw_root": str(args.raw_root),
        "normalized_root": str(args.normalized_root),
        "models": args.models,
        "evaluation_frame": {"image_size": [512, 512], "mpp": 0.25, "fov_um": 128.0},
        "deferred_models": ["pathdiff_text"],
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
    }
    write_json_atomic(state_path, state)
    while True:
        queues = queue_states(args.state_root, args.num_shards)
        statuses = [] if queues is None else [queue.get("status") for queue in queues]
        state["queue_statuses"] = statuses
        state["updated_at_utc"] = utc_now()
        write_json_atomic(state_path, state)
        if queues is not None and statuses == ["completed"] * args.num_shards:
            break
        time.sleep(args.poll_seconds)

    state["status"] = "normalizing"
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    normalize_command = [
        sys.executable,
        str(args.repo_root / "scripts/normalize_generation_mpp.py"),
        "--manifest",
        str(args.manifest),
        "--cross-root",
        str(args.cross_root),
        "--baseline-root",
        str(args.raw_root),
        "--output-root",
        str(args.normalized_root),
        "--target-resolution",
        "512",
        "--target-mpp",
        "0.25",
        "--models",
        *args.models,
    ]
    subprocess.run(normalize_command, cwd=args.repo_root, check=True)

    state["status"] = "validating"
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    validate_command = [
        sys.executable,
        str(args.repo_root / "scripts/validate_generation_baselines.py"),
        "--manifest",
        str(args.manifest),
        "--output-root",
        str(args.raw_root),
        "--models",
        *args.models,
        "--normalized-root",
        str(args.normalized_root),
        "--write-report",
    ]
    subprocess.run(validate_command, cwd=args.repo_root, check=True)

    state["status"] = "completed"
    state["validation_report"] = str(args.normalized_root / "validation.json")
    state["completed_at_utc"] = utc_now()
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
