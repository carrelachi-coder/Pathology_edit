import hashlib
import importlib.metadata
import json
from pathlib import Path
import pytest
from diagnostic_storage import setup_process_gate_v3 as gate
from tests.test_setup_telemetry_v3 import fixture, sha


def prepared(tmp_path, monkeypatch):
    m = fixture(tmp_path, monkeypatch)
    monkeypatch.undo()
    packages = []
    for d in importlib.metadata.distributions():
        record = Path(d._path) / "RECORD"
        packages.append(
            {
                "name": d.metadata["Name"],
                "version": d.version,
                "record_sha256": sha(record) if record.is_file() else None,
            }
        )
    packages.sort(key=lambda x: (x["name"] or "").lower())
    m["runtime"]["packages_sha256"] = hashlib.sha256(
        json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    m["runtime_code_files"].update(gate.source_identity())
    m["kind"] = gate.KIND
    m["permissions"] = {
        "setup": True,
        "validation": False,
        "S07": False,
        "generation": False,
        "publication": False,
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    approval = {
        "kind": gate.KIND,
        "approved": True,
        "manifest_sha256": sha(p),
        "scope": "setup_only",
    }
    return m, p, approval


def test_real_worker_preparation_and_verifier_end_to_end(tmp_path, monkeypatch):
    m, p, a = prepared(tmp_path, monkeypatch)
    result = gate.run_setup(p, a)
    assert result["status"] == "setup_verified", result
    assert result["transport"]["child_reaped"]
    assert result["transport"]["elapsed_seconds"] < 60
    assert result["worker_receipt"]["source_identity"] == gate.source_identity()
    assert result["execution_permitted"] is False
    assert (Path(m["output_root"]) / "setup_binding.json").exists()
    with pytest.raises(FileExistsError):
        gate.run_setup(p, a)


def test_consumed_legacy_approval_rejected_before_spawn(tmp_path, monkeypatch):
    m, p, a = prepared(tmp_path, monkeypatch)
    a["kind"] = "readonly-budget-v2-fixed-synthetic-trial"
    with pytest.raises(PermissionError):
        gate.run_setup(p, a)
    assert not Path(m["output_root"]).exists()


def test_real_worker_identity_failure_holds_without_binding(tmp_path, monkeypatch):
    m, p, a = prepared(tmp_path, monkeypatch)
    (Path(m["input_root"]) / "0").write_bytes(b"corrupt")
    result = gate.run_setup(p, a)
    assert result["status"] == "HOLD"
    assert (
        result["transport"]["exit_code"] == 78 and result["transport"]["child_reaped"]
    )
    assert not (Path(m["output_root"]) / "setup_binding.json").exists()
    with pytest.raises(FileExistsError):
        gate.run_setup(p, a)


def test_transport_timeout_consumes_attempt_without_retry(tmp_path, monkeypatch):
    m, p, a = prepared(tmp_path, monkeypatch)
    calls = []

    def timeout(argv, raw, seconds, *, kind):
        calls.append((seconds, kind))
        return {
            "transport_ok": False,
            "reason": "timeout",
            "stdout": b"",
            "child_reaped": True,
            "timed_out": True,
        }

    monkeypatch.setattr(gate, "_run_process", timeout)
    result = gate.run_setup(p, a)
    assert result["reason"] == "timeout" and len(calls) == 1 and 0 < calls[0][0] <= 60
    with pytest.raises(FileExistsError):
        gate.run_setup(p, a)
    assert len(calls) == 1


def test_transport_exception_holds_and_consumes_claim(tmp_path, monkeypatch):
    m, p, a = prepared(tmp_path, monkeypatch)

    def failed(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(gate, "_run_process", failed)
    result = gate.run_setup(p, a)
    assert result["status"] == "HOLD" and result["reason"] == "transport_error"
    with pytest.raises(FileExistsError):
        gate.run_setup(p, a)
