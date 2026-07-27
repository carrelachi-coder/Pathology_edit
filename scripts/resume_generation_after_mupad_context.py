#!/usr/bin/env python3
"""Prepare MuPaD WSI contexts, then resume the deferred generation queues."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


DEFAULT_MODELS = ["mupad_image_auxiliary", "pathldm_plip", "pathdiff_conic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--wsi-root", type=Path, action="append", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--prerequisite-state-root", type=Path)
    parser.add_argument("--context-python", type=Path, required=True)
    parser.add_argument("--queue-python", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--superseded-queue-pids", nargs="*", type=int, default=[])
    parser.add_argument("--poll-seconds", type=int, default=600)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def load_manifest_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    return len(records)


def generated_count(root: Path, model_id: str) -> int:
    model_root = root / model_id
    return sum(1 for _ in model_root.rglob("generated.png")) if model_root.exists() else 0


def context_ready(summary: dict) -> bool:
    prepared = int(summary.get("completed_this_run", 0)) + int(
        summary.get("skipped_complete", 0)
    )
    return (
        int(summary.get("missing_wsi_count", -1)) == 0
        and not summary.get("missing_context_records")
        and not summary.get("failures")
        and prepared == int(summary.get("eligible_direction_count", -1))
    )


def prerequisite_queues_complete(state_root: Path | None, num_shards: int) -> bool:
    if state_root is None:
        return True
    for shard_index in range(num_shards):
        state_path = state_root / f"queue_shard{shard_index}of{num_shards}.json"
        if not state_path.exists():
            return False
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "completed":
            return False
    return True


def queue_has_live_children(pid: int) -> bool:
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    if not children_path.exists():
        return False
    for child in children_path.read_text(encoding="utf-8").split():
        stat_path = Path(f"/proc/{child}/stat")
        if stat_path.exists() and stat_path.read_text(encoding="utf-8").split()[2] != "Z":
            return True
    return False


def terminate_superseded_queue(pid: int, manifest: Path, raw_root: Path) -> str:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return "already_absent"
    cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
        "utf-8", errors="replace"
    )
    required = ("run_generation_baseline_queue.py", str(manifest), str(raw_root))
    if not all(token in cmdline for token in required):
        raise RuntimeError(f"refusing to terminate unexpected process {pid}: {cmdline}")
    if queue_has_live_children(pid):
        raise RuntimeError(f"superseded queue {pid} still has a live child")
    os.kill(pid, signal.SIGKILL)
    return "terminated"


def prepare_contexts(args: argparse.Namespace, log_handle) -> dict:
    command = [
        str(args.context_python),
        str(args.repo_root / "scripts/prepare_mupad_wsi_context.py"),
        "--generation-manifest",
        str(args.manifest),
        "--patch-manifest",
        str(args.patch_manifest),
        "--output-root",
        str(args.context_root),
        "--allow-missing",
    ]
    for root in args.wsi_root:
        command.extend(["--wsi-root", str(root)])
    result = subprocess.run(
        command,
        cwd=args.repo_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary = json.loads(
        (args.context_root / "preparation_summary.json").read_text(encoding="utf-8")
    )
    summary["prepare_return_code"] = result.returncode
    return summary


def start_queues(args: argparse.Namespace, log_handles: list) -> list[subprocess.Popen]:
    processes = []
    num_shards = len(args.cuda_visible_devices)
    for shard_index, cuda_device in enumerate(args.cuda_visible_devices):
        command = [
            str(args.queue_python),
            str(args.repo_root / "scripts/run_generation_baseline_queue.py"),
            "--config-dir",
            str(args.repo_root / "benchmark_configs/models"),
            "--manifest",
            str(args.manifest),
            "--output-root",
            str(args.raw_root),
            "--state-root",
            str(args.state_root),
            "--models",
            *args.models,
            "--cuda-visible-device",
            str(cuda_device),
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(shard_index),
        ]
        processes.append(
            subprocess.Popen(
                command,
                cwd=args.repo_root,
                stdout=log_handles[shard_index],
                stderr=subprocess.STDOUT,
            )
        )
    return processes


def main() -> int:
    args = parse_args()
    if len(args.cuda_visible_devices) < 1:
        raise ValueError("at least one CUDA device is required")
    args.state_root.mkdir(parents=True, exist_ok=True)
    state_path = args.state_root / "mupad_context_resume.json"
    context_log_path = args.state_root / "mupad_context_preparation.log"
    expected_text_count = load_manifest_count(args.manifest)
    state = {
        "schema_version": 1,
        "status": "waiting_for_mupad_context",
        "pid": os.getpid(),
        "manifest": str(args.manifest),
        "context_root": str(args.context_root),
        "raw_root": str(args.raw_root),
        "models": args.models,
        "cuda_visible_devices": args.cuda_visible_devices,
        "superseded_queue_pids": args.superseded_queue_pids,
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
    }
    write_json_atomic(state_path, state)

    with context_log_path.open("a", encoding="utf-8") as context_log:
        while True:
            summary = prepare_contexts(args, context_log)
            text_count = generated_count(args.raw_root, "mupad_text")
            live_children = {
                str(pid): queue_has_live_children(pid)
                for pid in args.superseded_queue_pids
            }
            prerequisite_complete = prerequisite_queues_complete(
                args.prerequisite_state_root, len(args.cuda_visible_devices)
            )
            state.update(
                {
                    "status": "waiting_for_mupad_context",
                    "context_summary": {
                        "available_wsi_count": summary["available_wsi_count"],
                        "missing_wsi_count": summary["missing_wsi_count"],
                        "eligible_direction_count": summary[
                            "eligible_direction_count"
                        ],
                        "combined_exclusion_count": summary[
                            "combined_exclusion_count"
                        ],
                        "prepared_context_count": summary["completed_this_run"]
                        + summary["skipped_complete"],
                        "failure_count": len(summary["failures"]),
                    },
                    "mupad_text": {
                        "generated_count": text_count,
                        "expected_count": expected_text_count,
                    },
                    "superseded_queue_live_children": live_children,
                    "prerequisite_queues_complete": prerequisite_complete,
                    "updated_at_utc": utc_now(),
                }
            )
            write_json_atomic(state_path, state)
            if (
                context_ready(summary)
                and text_count == expected_text_count
                and not any(live_children.values())
                and prerequisite_complete
            ):
                break
            time.sleep(args.poll_seconds)

    state["status"] = "replacing_superseded_queues"
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    state["superseded_queue_results"] = {
        str(pid): terminate_superseded_queue(pid, args.manifest, args.raw_root)
        for pid in args.superseded_queue_pids
    }

    log_handles = [
        (args.state_root / f"queue_shard{index}of{len(args.cuda_visible_devices)}.log").open(
            "a", encoding="utf-8"
        )
        for index in range(len(args.cuda_visible_devices))
    ]
    try:
        state["status"] = "running_generation_queues"
        state["updated_at_utc"] = utc_now()
        write_json_atomic(state_path, state)
        processes = start_queues(args, log_handles)
        state["queue_pids"] = [process.pid for process in processes]
        state["updated_at_utc"] = utc_now()
        write_json_atomic(state_path, state)
        return_codes = [process.wait() for process in processes]
    finally:
        for handle in log_handles:
            handle.close()

    state["queue_return_codes"] = return_codes
    state["status"] = (
        "completed" if all(code == 0 for code in return_codes) else "failed"
    )
    state["completed_at_utc"] = utc_now()
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
