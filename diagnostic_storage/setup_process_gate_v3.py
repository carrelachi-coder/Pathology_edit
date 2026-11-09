"""Once-only preparation gate. Task policy is fixed to sixty seconds."""
import hashlib
import json
from pathlib import Path
import sys
import time

from .validator_process_gate_v2 import _run_process
from .validation_budget_policy_v2 import file_sha

KIND = "setup-v3-target-trial"
SECONDS = 60


def source_identity():
    root = Path(__file__).resolve().parents[1]
    names = [
        "diagnostic_storage/setup_process_gate_v3.py",
        "diagnostic_storage/setup_result_verifier_v3.py",
        "scripts/setup_preparation_worker_v3.py",
        "scripts/preflight_setup_telemetry_v3.py",
    ]
    return {str(root / name): file_sha(root / name) for name in names}


def run_setup(manifest_path, approval, *, python_executable=sys.executable):
    started = time.monotonic()
    path = Path(manifest_path).resolve(strict=True)
    with path.open("rb") as stream:
        raw = stream.read(1048577)
    if len(raw) > 1048576:
        raise ValueError("manifest limit")
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if (
        approval.get("kind") != KIND
        or approval.get("approved") is not True
        or approval.get("manifest_sha256") != digest
        or approval.get("scope") != "setup_only"
    ):
        raise PermissionError("manifest-bound setup-only authorization required")
    if (
        manifest.get("kind") != KIND
        or manifest.get("setup_watchdog_seconds") != SECONDS
    ):
        raise ValueError("frozen setup policy required")
    if manifest.get("permissions") != {
        "setup": True,
        "validation": False,
        "S07": False,
        "generation": False,
        "publication": False,
    }:
        raise ValueError("scope mismatch")
    sources = source_identity()
    if any(manifest["runtime_code_files"].get(p) != s for p, s in sources.items()):
        raise ValueError("execution source identity mismatch")
    output = Path(manifest["output_root"])
    if output.exists():
        raise FileExistsError("output already exists")
    claim = Path(str(output) + ".launch_claim")
    with claim.open("x") as stream:
        json.dump({"manifest_sha256": digest, "authorization_consumed": True}, stream)
    request = json.dumps(
        {"manifest_path": str(path), "manifest_sha256": digest}, sort_keys=True
    ).encode()
    request_sha = hashlib.sha256(request).hexdigest()
    result = {
        "status": "HOLD",
        "manifest_sha256": digest,
        "authorization_consumed": True,
        "attempts": 1,
        "automatic_retries": 0,
        "execution_permitted": False,
    }
    remaining = SECONDS - (time.monotonic() - started)
    if remaining <= 0:
        result["reason"] = "timeout_before_spawn"
        return result
    worker = (
        Path(__file__).resolve().parents[1] / "scripts/setup_preparation_worker_v3.py"
    )
    try:
        transport = _run_process(
            [str(python_executable), str(worker)], request, remaining, kind="compare"
        )
    except Exception as exc:
        result["reason"] = "transport_error"
        result["error_type"] = type(exc).__name__
        return result
    wire = transport.pop("stdout")
    result["transport"] = transport
    if not transport["transport_ok"]:
        result["reason"] = transport["reason"]
        return result
    try:
        response = json.loads(wire)
        if (
            response["request_sha256"] != request_sha
            or response["manifest_sha256"] != digest
            or response["source_identity"] != sources
            or response["status"] != "setup_evidence_verified"
            or response["execution_permitted"] is not False
        ):
            raise ValueError("worker binding")
        if time.monotonic() - started >= SECONDS:
            raise TimeoutError("late verification")
    except Exception as exc:
        result["reason"] = type(exc).__name__
        return result
    result.update(status="setup_verified", worker_receipt=response)
    return result
