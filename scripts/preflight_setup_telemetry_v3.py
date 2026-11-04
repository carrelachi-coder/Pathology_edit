"""Review-only setup preparation; no validator, S07 or launch entrypoint.

Preserves v2 identity checks, traversal and deadline-check locations. Telemetry
adds bounded I/O/time overhead; it does not extend the clock or select inputs.
"""
from __future__ import annotations
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from contextlib import contextmanager

from diagnostic_storage import validation_budget_policy_v2 as policy
from scripts.run_readonly_budget_v2_trial import load

STAGES = (
    "legacy_code",
    "runtime_code",
    "interpreter_environment",
    "distributions",
    "native_binaries",
    "input_inventory",
    "storage",
)


class SetupTelemetry:
    MAX_BYTES = 1048576
    MAX_RECORDS = 2048
    MAX_RECORD_BYTES = 2048

    def __init__(self, path, *, clock=time.monotonic):
        self.clock = clock
        self.origin = clock()
        self.stages = []
        self.current = None
        self.last_complete = None
        self.faults = []
        self.records = self.written = 0
        self.stream = None
        self.finalized = False
        try:
            self.stream = Path(path).open("xb", buffering=0)
        except Exception as exc:
            self._fault(exc)

    def _fault(self, exc):
        if len(self.faults) < 8:
            self.faults.append(type(exc).__name__)

    def _state(self):
        if self.current is None:
            return {}
        s = dict(self.current)
        s["elapsed_ms"] = max(0, (self.clock() - s.pop("_start")) * 1000)
        return s

    def _emit(self, event):
        if self.faults:
            return
        try:
            row = {"sequence": self.records, "event": event, **self._state()}
            raw = (
                json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
            if (
                len(raw) > self.MAX_RECORD_BYTES
                or self.records >= self.MAX_RECORDS
                or self.written + len(raw) > self.MAX_BYTES
            ):
                raise ValueError("bounded setup telemetry exceeded")
            if self.stream.write(raw) != len(raw):
                raise OSError("partial telemetry write")
            self.records += 1
            self.written += len(raw)
        except Exception as exc:
            self._fault(exc)

    @contextmanager
    def stage(self, name):
        if (
            name not in STAGES
            or self.current is not None
            or len(self.stages) >= len(STAGES)
        ):
            raise ValueError("internal stage contract")
        self.current = {
            "stage": name,
            "stage_started": True,
            "stage_completed": False,
            "elapsed_ms": 0,
            "items_completed": 0,
            "bytes_processed": 0,
            "last_completed_item_index": None,
            "_start": self.clock(),
        }
        self._emit("stage_started")
        try:
            yield
        except BaseException:
            self._emit("stage_interrupted")
            self.stages.append(self._state())
            self.current = None
            raise
        else:
            self.current["stage_completed"] = True
            self._emit("stage_completed")
            self.stages.append(self._state())
            self.last_complete = name
            self.current = None

    def hashed(self, n):
        # Counts only fully read hash inputs; interrupted hash bytes are unknown.
        if self.current is not None:
            self.current["bytes_processed"] += n

    def completed(self):
        if self.current is not None:
            self.current["last_completed_item_index"] = self.current["items_completed"]
            self.current["items_completed"] += 1
            self._emit("item_completed")

    def finish(self, identity_passed, failure_type=None):
        if self.stream is not None:
            try:
                self.stream.flush()
                os.fsync(self.stream.fileno())
            except Exception as exc:
                self._fault(exc)
            try:
                self.stream.close()
            except Exception as exc:
                self._fault(exc)
        self.finalized = True
        return {
            "format": "bounded-setup-telemetry-v3",
            "telemetry_valid": not self.faults,
            "identity_check_passed": identity_passed,
            "preparation_accepted": identity_passed and not self.faults,
            "failure_type": failure_type,
            "fault_types": list(self.faults),
            "elapsed_ms": max(0, (self.clock() - self.origin) * 1000),
            "last_completed_stage": self.last_complete,
            "stages": list(self.stages),
            "records": self.records,
            "telemetry_bytes": self.written,
            "execution_permitted": False,
            "bytes_definition": "actual bytes fed to completed SHA256 reads; failed or interrupted partial reads uncounted; no second file scan",
            "index_definition": "zero-based completed loop-item index; distributions are hashed/enumerated here and accepted only by aggregate hash at stage completion; failing/in-flight item uncounted",
            "abrupt_kill_limit": "SIGKILL may leave only a bounded journal prefix; absence of final summary is incomplete, never acceptance",
        }


def hash_file(path, telemetry):
    result = hashlib.sha256()
    n = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
            n += len(chunk)
    telemetry.hashed(n)
    return result.hexdigest()


def preflight(m, telemetry):
    start = time.monotonic()

    def check():
        if time.monotonic() - start >= m["setup_watchdog_seconds"]:
            raise TimeoutError("setup hold")

    def file_sha(path):
        return hash_file(path, telemetry)

    with telemetry.stage("legacy_code"):
        root = Path(policy.__file__).parent
        pins = dict(policy.PRODUCER_PINS)
        pins["validator_process_gate_v1.py"] = policy.LEGACY_GATE_PIN
        for name, expected in pins.items():
            if file_sha(root / name) != expected:
                raise ValueError("unreviewed legacy component drift: " + name)
            telemetry.completed()
    with telemetry.stage("runtime_code"):
        for path, expected in m["runtime_code_files"].items():
            check()
            if file_sha(path) != expected:
                raise ValueError("runtime code drift")
            telemetry.completed()
    runtime = m["runtime"]
    with telemetry.stage("interpreter_environment"):
        if (
            sys.executable != runtime["executable"]
            or file_sha(sys.executable) != runtime["executable_sha256"]
        ):
            raise ValueError("interpreter drift")
        if any(os.environ.get(k) != v for k, v in runtime["environment"].items()):
            raise ValueError("environment drift")
        telemetry.completed()
    with telemetry.stage("distributions"):
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
            telemetry.completed()
        packages.sort(key=lambda x: (x["name"] or "").lower())
        raw = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(raw).hexdigest() != runtime["packages_sha256"]:
            raise ValueError("distribution identity drift")
    with telemetry.stage("native_binaries"):
        for path, expected in runtime["native_files_sha256"].items():
            check()
            if file_sha(path) != expected:
                raise ValueError("native library drift")
            telemetry.completed()
    with telemetry.stage("input_inventory"):
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
            telemetry.completed()
    with telemetry.stage("storage"):
        output = Path(m["output_root"])
        if shutil.disk_usage(output).free < m["minimum_free_output_bytes"]:
            raise OSError("output storage hold")
        telemetry.completed()
        reservation = m["existing_reservation"]
        stat = Path(reservation["path"]).stat()
        if (
            stat.st_size != reservation["bytes"]
            or stat.st_blocks * 512 < reservation["bytes"]
        ):
            raise OSError("reservation missing; physical allocation is not quota")
        telemetry.completed()
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
        telemetry.completed()
        check()
    return {
        "filesystem_no_replace_checked_at_actual_output": True,
        "reservation_allocated_not_quota": True,
        "input_files_verified": 809,
        "elapsed_seconds": time.monotonic() - start,
        "code_and_runtime_match": True,
    }


def capture_preflight(m, output):
    """New preparation-only tool. Refuses reuse. Never invokes a helper or grant."""
    output = Path(output)
    if output != Path(m["output_root"]):
        raise ValueError("telemetry and storage checks must use the same new output")
    output.mkdir(mode=0o700, exist_ok=False)
    telemetry = SetupTelemetry(output / "setup_telemetry.jsonl")
    passed = False
    failure = None
    identity_result = None
    try:
        identity_result = preflight(m, telemetry)
        passed = True
    except (Exception, KeyboardInterrupt) as exc:
        failure = type(exc).__name__
    report = telemetry.finish(passed, failure)
    report["identity_result"] = identity_result
    report["status"] = (
        "preparation_only_pass" if report["preparation_accepted"] else "HOLD"
    )
    try:
        with (output / "setup_summary.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # No final summary means invalid evidence. No downstream action exists.
        raise
    return report
