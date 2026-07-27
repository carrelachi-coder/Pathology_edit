#!/usr/bin/env python3
"""Wait for single-baseline workers, normalize, and validate outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _pid_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            if stat_path.read_text(encoding="utf-8").split()[2] == "Z":
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _completion_counts(model_root: Path) -> tuple[int, int]:
    complete = 0
    errors = 0
    for metadata in model_root.glob("*/*/metadata.json"):
        if (metadata.parent / "generated.png").is_file():
            complete += 1
    errors = sum(1 for _ in model_root.glob("*/*/error.json"))
    return complete, errors


def _run(command: list[str], cwd: Path) -> dict:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "output": result.stdout[-20000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--model-id", default="pathdiff_text")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    launch_path = args.state_root / "launch_state_strict_oral.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    pids = [int(job["pid"]) for job in launch["jobs"]]
    state_path = args.state_root / "finalization_strict_oral.json"
    model_root = args.raw_root / args.model_id
    state = {
        "status": "waiting_for_generation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_pids": pids,
        "expected": args.expected,
    }
    while any(_pid_alive(pid) for pid in pids):
        complete, errors = _completion_counts(model_root)
        state.update(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "complete": complete,
                "error_files": errors,
                "alive_pids": [pid for pid in pids if _pid_alive(pid)],
            }
        )
        _atomic_json(state_path, state)
        time.sleep(args.poll_seconds)

    complete, errors = _completion_counts(model_root)
    state.update(
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": complete,
            "error_files": errors,
            "alive_pids": [],
        }
    )
    if complete != args.expected or errors:
        state["status"] = "generation_incomplete"
        _atomic_json(state_path, state)
        return 1

    normalization_summary_path = args.normalized_root / "normalization_summary.json"
    previous_summary = None
    if normalization_summary_path.is_file():
        previous_summary = json.loads(normalization_summary_path.read_text())
    normalize = _run(
        [
            sys.executable,
            str(args.repo_root / "scripts/normalize_generation_mpp.py"),
            "--manifest",
            str(args.manifest),
            "--cross-root",
            str(args.raw_root),
            "--baseline-root",
            str(args.raw_root),
            "--output-root",
            str(args.normalized_root),
            "--models",
            args.model_id,
        ],
        args.repo_root,
    )
    state["normalization"] = normalize
    if previous_summary is not None and normalization_summary_path.is_file():
        pathdiff_summary = json.loads(normalization_summary_path.read_text())
        previous_summary.setdefault("counts", {}).update(pathdiff_summary["counts"])
        previous_summary["failures"] = [
            failure
            for failure in previous_summary.get("failures", [])
            if failure.get("model_id") != args.model_id
        ] + pathdiff_summary.get("failures", [])
        previous_summary["valid"] = not previous_summary["failures"]
        _atomic_json(normalization_summary_path, previous_summary)

    validate = _run(
        [
            sys.executable,
            str(args.repo_root / "scripts/validate_generation_baselines.py"),
            "--manifest",
            str(args.manifest),
            "--output-root",
            str(args.raw_root),
            "--models",
            args.model_id,
            "--normalized-root",
            str(args.normalized_root),
        ],
        args.repo_root,
    )
    state["validation"] = validate
    state["status"] = (
        "completed"
        if normalize["returncode"] == 0 and validate["returncode"] == 0
        else "finalization_failed"
    )
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(state_path, state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
