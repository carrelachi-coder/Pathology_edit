#!/usr/bin/env python3
"""Run bounded four-target candidate compilation without automatic approval."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_process(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    seconds: int,
    env: dict[str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    return {
        "controller_exit_code": code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
    }


def validate_inputs(cases_path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    cases = json.loads(cases_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "reserved-four-target-candidate-freeze-v1":
        raise ValueError("candidate manifest has an unsupported schema")
    if manifest.get("formal_qualification_updated") is not False:
        raise ValueError("candidate manifest improperly claims formal qualification")
    if manifest.get("cases", {}).get("sha256") != sha256_file(cases_path):
        raise ValueError("candidate cases do not match the frozen manifest")
    if not isinstance(cases, list) or len(cases) != 4 * int(
        manifest.get("unique_reserved_topology_pass_patients", -1)
    ):
        raise ValueError("candidate case count is not four per reserved patient")
    expected_indices: list[int] = []
    for row in manifest.get("selected_candidates", []):
        if row.get("status") != "topology_pass_candidate_only":
            raise ValueError("candidate row has an invalid admission status")
        expected_indices.extend(int(value) for value in row.get("case_indices", []))
    if expected_indices != list(range(len(cases))):
        raise ValueError("candidate case indices are incomplete or reordered")
    case_ids = [str(row.get("case_id")) for row in cases]
    if any(not value for value in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("candidate case IDs must be unique and non-empty")
    return cases, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--compiler",
        type=Path,
        default=Path(__file__).with_name("compile_manuscript_joint_technical.py"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--seconds-per-case", type=int, default=7200)
    parser.add_argument("--case-index", type=int, action="append")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; compilation runs are append-only")
    if args.seconds_per_case <= 0 or not args.device:
        raise ValueError("invalid compilation resource bounds")
    cases, candidate_manifest = validate_inputs(args.cases, args.candidate_manifest)
    selected_indices = (
        sorted(set(args.case_index)) if args.case_index is not None else list(range(len(cases)))
    )
    if not selected_indices or selected_indices[0] < 0 or selected_indices[-1] >= len(cases):
        raise ValueError("case selection is empty or out of range")
    compiler = args.compiler.resolve()
    if not compiler.is_file():
        raise ValueError(f"compiler does not exist: {compiler}")
    repository_root = compiler.parent.parent
    args.output.mkdir(parents=True)
    (args.output / "logs").mkdir()
    (args.output / "receipts").mkdir()
    (args.output / "compiled").mkdir()
    run_identity = {
        "schema_version": "reserved-four-target-compilation-run-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cases": identity(args.cases),
        "candidate_manifest": identity(args.candidate_manifest),
        "compiler": identity(compiler),
        "controller": identity(Path(__file__)),
        "selected_case_indices": selected_indices,
        "devices": list(args.device),
        "workers": len(args.device),
        "seconds_per_case": args.seconds_per_case,
        "automatic_retries": 0,
        "scope": "candidate condition compilation only; no mask approval, H&E generation, or formal cohort admission",
        "topology_pass_patients": candidate_manifest[
            "unique_reserved_topology_pass_patients"
        ],
    }
    (args.output / "run_identity.json").write_text(json.dumps(run_identity, indent=2) + "\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )

    queues = [[] for _ in args.device]
    for position, case_index in enumerate(selected_indices):
        queues[position % len(args.device)].append(case_index)

    def run_queue(device: str, indices: list[int]) -> list[dict[str, Any]]:
        receipts = []
        for case_index in indices:
            case_id = str(cases[case_index]["case_id"])
            command = [
                sys.executable,
                str(compiler),
                "--cases",
                str(args.cases.resolve()),
                "--case-index",
                str(case_index),
                "--output",
                str((args.output / "compiled").resolve()),
                "--device",
                device,
            ]
            receipt = {
                "case_index": case_index,
                "case_id": case_id,
                "device": device,
                "command": command,
                "automatic_retries": 0,
            }
            receipt.update(
                run_process(
                    command,
                    cwd=repository_root,
                    log_path=args.output / "logs" / f"case_{case_index:04d}.log",
                    seconds=args.seconds_per_case,
                    env=environment,
                )
            )
            summary_path = args.output / "compiled" / f"{case_id}_summary.json"
            if receipt["controller_exit_code"] == 0 and summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                if summary.get("case_id") != case_id:
                    receipt["terminal_status"] = "indeterminate_identity"
                else:
                    receipt["terminal_status"] = str(summary.get("status"))
                    receipt["summary"] = identity(summary_path)
                    receipt["selected_candidate_id"] = summary.get("selected_candidate_id")
                    receipt["abstain_reasons"] = summary.get("reasons")
            elif receipt["timed_out"]:
                receipt["terminal_status"] = "timeout"
            else:
                receipt["terminal_status"] = "execution_error"
            receipt_path = args.output / "receipts" / f"case_{case_index:04d}.json"
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            receipts.append(receipt)
        return receipts

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.device)) as executor:
        futures = {
            executor.submit(run_queue, device, indices): device
            for device, indices in zip(args.device, queues)
            if indices
        }
        for future in as_completed(futures):
            completed.extend(future.result())
            progress = sorted(completed, key=lambda row: row["case_index"])
            (args.output / "compilation_progress.json").write_text(
                json.dumps(progress, indent=2) + "\n"
            )
    final = sorted(completed, key=lambda row: row["case_index"])
    (args.output / "compilation_receipts.json").write_text(
        json.dumps(final, indent=2) + "\n"
    )
    indeterminate = {"timeout", "execution_error", "indeterminate_identity"}
    return 78 if any(row["terminal_status"] in indeterminate for row in final) else 0


if __name__ == "__main__":
    raise SystemExit(main())
