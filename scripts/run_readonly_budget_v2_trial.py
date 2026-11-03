#!/usr/bin/env python3
"""One approved fixed synthetic trial only. No generator, publisher or S07 API."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import errno
import shutil
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostic_storage.validation_budget_policy_v2 import (
    budget_for,
    file_sha,
    policy_identity,
    verify_legacy_files,
    COMPATIBILITY_ID,
)
from diagnostic_storage.validator_process_gate_v2 import run_validation, request_value

KIND = "readonly-budget-v2-fixed-synthetic-trial"


def load(path, limit=1024 * 1024):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("control file limit")
    return json.loads(raw), raw


def authorize(manifest_raw, approval):
    m = json.loads(manifest_raw)
    digest = hashlib.sha256(manifest_raw).hexdigest()
    if m.get("kind") != KIND or m.get("permissions") != {
        "production": False,
        "S07": False,
        "generation": False,
        "publication": False,
    }:
        raise PermissionError("read-only synthetic scope required")
    if (
        approval.get("kind") != KIND
        or approval.get("approved") is not True
        or approval.get("manifest_sha256") != digest
        or approval.get("policy") != policy_identity()
        or approval.get("compatibility_profile") != COMPATIBILITY_ID
        or approval.get("compatibility_reviewed") is not True
    ):
        raise PermissionError("separate reviewed manifest-bound trial approval absent")
    jobs = m.get("jobs", [])
    if len(jobs) != 3:
        raise ValueError("exactly three fixed measurements")
    for job, (label, kind) in zip(
        jobs,
        [("B_capture", "capture"), ("C_capture", "capture"), ("BC_compare", "compare")],
    ):
        if (
            job["label"] != label
            or job["request"]["kind"] != kind
            or job["hard_seconds"] != budget_for(kind)
            or job["measurements"] != 1
        ):
            raise ValueError("unreviewed task/order/count/budget")
        raw = json.dumps(
            job["request"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        request_value(raw)
        if hashlib.sha256(raw).hexdigest() != job["request_sha256"]:
            raise ValueError("request bytes drift")
        if (
            job["request"]["identity"]["reference_id"] != "synthetic-source"
            or job["request"]["identity"]["patient_id"] != "synthetic-patient"
        ):
            raise ValueError("fixed synthetic identity only")
    if (
        jobs[0]["request"]["mode"] != "hash_only"
        or jobs[1]["request"]["mode"] != "full_capture"
    ):
        raise ValueError("mode order")
    pair = jobs[2]["request"]
    for left, right, key in [
        ("left", "capture", 0),
        ("right", "capture", 1),
        ("left_receipt", "receipt", 0),
        ("right_receipt", "receipt", 1),
    ]:
        if pair[left] != jobs[key]["request"][right]:
            raise ValueError("pair input changed")
    if m.get("automatic_retries") != 0:
        raise ValueError("no retries")
    return m, digest


def preflight(m):
    start = time.monotonic()

    def check():
        if time.monotonic() - start >= m["setup_watchdog_seconds"]:
            raise TimeoutError("setup hold")

    verify_legacy_files()
    for path, expected in m["runtime_code_files"].items():
        check()
        if file_sha(path) != expected:
            raise ValueError("runtime code drift")
    runtime = m["runtime"]
    if (
        sys.executable != runtime["executable"]
        or file_sha(sys.executable) != runtime["executable_sha256"]
    ):
        raise ValueError("interpreter drift")
    if any(os.environ.get(k) != v for k, v in runtime["environment"].items()):
        raise ValueError("environment drift")
    packages = []
    for d in importlib.metadata.distributions():
        check()
        record = Path(d._path) / "RECORD"
        packages.append(
            {
                "name": d.metadata["Name"],
                "version": d.version,
                "record_sha256": file_sha(record) if record.is_file() else None,
            }
        )
    packages.sort(key=lambda x: (x["name"] or "").lower())
    raw = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(raw).hexdigest() != runtime["packages_sha256"]:
        raise ValueError("distribution identity drift")
    for path, expected in runtime["native_files_sha256"].items():
        check()
        if file_sha(path) != expected:
            raise ValueError("native library drift")
    invpath = Path(m["input_inventory"]["path"])
    if file_sha(invpath) != m["input_inventory"]["sha256"]:
        raise ValueError("input inventory drift")
    inventory, _ = load(invpath)
    root = Path(m["input_root"])
    if root.is_symlink():
        raise ValueError("input root symlink")
    root = root.resolve(strict=True)
    if len(inventory["files"]) != 809:
        raise ValueError("input count drift")
    for name, entry in inventory["files"].items():
        check()
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("invalid relative inventory")
        path = root / rel
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("input escaped frozen root")
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_sha(path) != entry["sha256"]
        ):
            raise ValueError("input bytes drift")
    output = Path(m["output_root"])
    if shutil.disk_usage(output).free < m["minimum_free_output_bytes"]:
        raise OSError("output storage hold")
    reservation = m["existing_reservation"]
    stat = Path(reservation["path"]).stat()
    if (
        stat.st_size != reservation["bytes"]
        or stat.st_blocks * 512 < reservation["bytes"]
    ):
        raise OSError("reservation missing; physical allocation is not quota")
    # Probe the actual new output directory, never the immutable capture roots.
    pending, final = output / "link_probe_pending", output / "link_probe_final"
    with pending.open("xb") as stream:
        stream.write(b"bounded-readonly-trial-preflight")
    os.link(pending, final)
    if pending.stat().st_ino != final.stat().st_ino:
        raise OSError("hardlink identity failed")
    try:
        os.link(pending, final)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    else:
        raise OSError("no-replace failed")
    check()
    return {
        "filesystem_no_replace_checked_at_actual_output": True,
        "reservation_allocated_not_quota": True,
        "input_files_verified": 809,
        "elapsed_seconds": time.monotonic() - start,
        "code_and_runtime_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    approval, _ = load(args.approval)
    _, raw = load(args.manifest)
    m, digest = authorize(raw, approval)
    root = Path(m["output_root"])
    root.mkdir(mode=0o700, exist_ok=False)
    rows = []
    result = {
        "manifest_sha256": digest,
        "status": "HOLD",
        "steps": rows,
        "execution_permitted": False,
        "real_source_calls": 0,
        "capture_markers_published": False,
    }
    try:
        result["preflight"] = preflight(m)
        for job in m["jobs"]:
            request = json.dumps(
                job["request"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            row = {"label": job["label"], "result": run_validation(request)}
            rows.append(row)
            with (root / (job["label"] + "_receipt.json")).open("x") as stream:
                json.dump(row, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            if not row["result"]["validation_ok"]:
                break
        if len(rows) == 3 and all(row["result"]["validation_ok"] for row in rows):
            result["status"] = "fixed_synthetic_budget_trial_pass"
    except Exception as exc:
        result["failure_type"] = type(exc).__name__
    with (root / "trial_result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {"status": result["status"], "steps": len(rows), "real_source_calls": 0}
        )
    )
    return 0 if result["status"] == "fixed_synthetic_budget_trial_pass" else 78


if __name__ == "__main__":
    raise SystemExit(main())
