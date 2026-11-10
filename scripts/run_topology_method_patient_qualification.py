#!/usr/bin/env python3
"""Patient-ordered source topology qualification with fail-closed stopping."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_mask_edit.benchmark.experiment_identity import asset_identity


def group_patient_sources(
    freeze: dict,
    *,
    excluded_patients: set[str],
) -> list[tuple[str, list[tuple[int, dict]]]]:
    patients = [
        str(value)
        for value in freeze["patient_aliases"]
        if str(value) not in excluded_patients
    ]
    grouped = {patient: [] for patient in patients}
    for index, source in enumerate(freeze["candidate_sources"]):
        patient = str(source["patient_id"])
        if patient in grouped:
            grouped[patient].append((index, source))
    for patient, rows in grouped.items():
        if not rows:
            raise ValueError(f"patient has no source rows: {patient}")
        ranks = [int(source["patient_source_rank"]) for _index, source in rows]
        if ranks != list(range(len(rows))):
            raise ValueError(f"patient source order is not contiguous: {patient}")
    return [(patient, grouped[patient]) for patient in patients]


def run_one(command: list[str], *, log_path: Path, seconds: int) -> dict:
    start = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
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
        "elapsed_seconds": time.monotonic() - start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds-per-source", type=int, default=1200)
    parser.add_argument("--exclude-patient", action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or args.seconds_per_source <= 0:
        raise ValueError("invalid resource bounds")
    freeze = json.loads(args.source_freeze.read_text())
    excluded = {str(value) for value in args.exclude_patient}
    unknown_exclusions = excluded - set(freeze["patient_aliases"])
    if unknown_exclusions:
        raise ValueError(f"excluded patient is absent: {sorted(unknown_exclusions)}")
    groups = group_patient_sources(freeze, excluded_patients=excluded)
    args.output.mkdir(parents=True, exist_ok=False)
    probe = Path(__file__).with_name("probe_manuscript_joint_capacity.py")
    identity = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_freeze": asset_identity(args.source_freeze),
        "case_templates": asset_identity(args.cases),
        "probe_script": asset_identity(probe),
        "controller_script": asset_identity(__file__),
        "partition": freeze["partition"],
        "requested_patients": len(groups),
        "available_sources": sum(len(rows) for _patient, rows in groups),
        "excluded_patients": sorted(excluded),
        "workers": args.workers,
        "seconds_per_source": args.seconds_per_source,
        "automatic_retries": 0,
        "selection_rule": "first passing source in frozen patient-local order; stop patient on unknown execution state",
        "scope": "source-only topology method qualification; no target approval or H&E generation",
    }
    (args.output / "controller_identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )

    def screen_patient(patient_order: int, patient: str, rows: list[tuple[int, dict]]):
        attempts = []
        patient_dir = args.output / f"patient_{patient_order:03d}_{patient}"
        patient_dir.mkdir()
        status = "exhausted_without_pass"
        selected_reference = None
        for source_index, source in rows:
            rank = int(source["patient_source_rank"])
            output = patient_dir / f"source_{rank:03d}"
            command = [
                sys.executable,
                str(probe),
                "--source-freeze",
                str(args.source_freeze),
                "--cases",
                str(args.cases),
                "--output",
                str(output),
                "--start-index",
                str(source_index),
                "--limit",
                "1",
                "--topology-witness",
            ]
            receipt = run_one(
                command,
                log_path=patient_dir / f"source_{rank:03d}.log",
                seconds=args.seconds_per_source,
            )
            receipt.update(
                source_index=source_index,
                patient_source_rank=rank,
                reference_id=source["reference_id"],
            )
            result_path = output / "capacity_screen.json"
            if receipt["controller_exit_code"] != 0 or not result_path.is_file():
                status = "indeterminate_execution"
                attempts.append(receipt)
                break
            records = json.loads(result_path.read_text())
            if (
                len(records) != 1
                or records[0]["reference_id"] != source["reference_id"]
                or records[0]["patient_id"] != patient
            ):
                receipt["identity_mismatch"] = True
                attempts.append(receipt)
                status = "indeterminate_identity"
                break
            receipt["result"] = records[0]
            receipt["result_asset"] = asset_identity(result_path)
            attempts.append(receipt)
            if records[0].get("both_significant_topology_pass") is True:
                status = "first_source_passed"
                selected_reference = source["reference_id"]
                break
        patient_receipt = {
            "patient_order": patient_order,
            "patient_id": patient,
            "status": status,
            "selected_reference_id": selected_reference,
            "attempted_sources": len(attempts),
            "available_sources": len(rows),
            "attempts": attempts,
        }
        (patient_dir / "patient_receipt.json").write_text(
            json.dumps(patient_receipt, indent=2) + "\n"
        )
        return patient_receipt

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(screen_patient, order, patient, rows): patient
            for order, (patient, rows) in enumerate(groups)
        }
        for future in as_completed(futures):
            completed.append(future.result())
            ordered = sorted(completed, key=lambda row: row["patient_order"])
            (args.output / "patient_progress.json").write_text(
                json.dumps(ordered, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "completed_patients": len(completed),
                        "requested_patients": len(groups),
                        "passed_patients": sum(
                            row["status"] == "first_source_passed"
                            for row in completed
                        ),
                        "indeterminate_patients": sum(
                            row["status"].startswith("indeterminate")
                            for row in completed
                        ),
                    }
                ),
                flush=True,
            )
    final = sorted(completed, key=lambda row: row["patient_order"])
    (args.output / "patient_qualification.json").write_text(
        json.dumps(final, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
