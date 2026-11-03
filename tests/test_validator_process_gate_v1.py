"""OS process gate tests. No real-source/S07/controller invocation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest import mock

import pytest
from diagnostic_storage import validator_process_gate_v1 as gate
from tests.test_observer_codec_adapter_v1 import complete, identity


def request(core, rp):
    return json.dumps(
        {
            "kind": "capture",
            "identity": identity(),
            "capture": str(core.root),
            "receipt": str(rp),
            "mode": core.mode,
        }
    ).encode()


def command(code):
    return [sys.executable, "-c", code]


def test_actual_readonly_child_validates_without_publishing(tmp_path):
    core, rp = complete(tmp_path)
    before = {p.name: p.stat().st_ino for p in core.root.iterdir()}
    result = gate.run_validation(request(core, rp))
    assert result["validation_ok"] and result["status"] == "validated_readonly"
    assert result["transport"]["child_reaped"] and result["transport"]["exit_code"] == 0
    assert not result["interpretation_permitted"] and not result["execution_permitted"]
    assert not (core.root / "adapter_complete.json").exists()
    assert before == {p.name: p.stat().st_ino for p in core.root.iterdir()}


def test_actual_pair_readonly_process(tmp_path):
    b, br = complete(tmp_path, "B")
    c, cr = complete(tmp_path, "C", "full_capture")
    raw = json.dumps(
        {
            "kind": "compare",
            "identity": identity(),
            "left": str(b.root),
            "right": str(c.root),
            "left_receipt": str(br),
            "right_receipt": str(cr),
        }
    ).encode()
    result = gate.run_validation(raw)
    assert (
        result["validation_ok"]
        and result["worker_receipt"]["verdict"]["status"] == "semantic_match"
    )
    assert (
        not (b.root / "adapter_complete.json").exists()
        and not (c.root / "adapter_complete.json").exists()
    )


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "nonzero",
        "signal",
        "stdout_overflow",
        "stderr_overflow",
        "late_valid_output",
    ],
)
def test_os_failure_is_hold_and_never_late_acceptance(tmp_path, failure):
    scripts = {
        "timeout": "import time;time.sleep(2)",
        "late_valid_output": 'import time;print("valid-looking",flush=True);time.sleep(2)',
        "nonzero": "raise SystemExit(7)",
        "signal": "import os,signal;os.kill(os.getpid(),signal.SIGTERM)",
        "stdout_overflow": 'print("x"*40000,flush=True)',
        "stderr_overflow": 'import sys;print("x"*10000,file=sys.stderr,flush=True)',
    }
    result = gate._run_process(
        command(scripts[failure]),
        b"{}",
        0.1 if failure in ("timeout", "late_valid_output") else 2,
    )
    assert not result["transport_ok"] and result["child_reaped"]
    assert result["timed_out"] == (failure in ("timeout", "late_valid_output"))
    assert (
        len(result["stdout"]) <= gate.MAX_STDOUT
        and result["stderr_bytes"] <= gate.MAX_STDERR
    )


def test_timeout_kills_descendant_that_holds_pipes_and_could_write_later(tmp_path):
    marker = tmp_path / "forbidden_late_marker"
    code = (
        "import os,time;pid=os.fork();time.sleep(1);open("
        + repr(str(marker))
        + ',"w").write("late")'
    )
    result = gate._run_process(command(code), b"{}", 0.1)
    assert result["timed_out"] and result["child_reaped"]
    time.sleep(1.1)
    assert not marker.exists()


def test_external_cancel_kills_and_reaps_without_swallowing(monkeypatch):
    original = gate.selectors.DefaultSelector
    seen = []

    class Interrupted:
        def __init__(self):
            self.inner = original()

        def register(self, *a, **kw):
            return self.inner.register(*a, **kw)

        def get_map(self):
            return self.inner.get_map()

        def close(self):
            self.inner.close()

        def select(self, *a):
            raise KeyboardInterrupt("external cancel")

    popen = gate.subprocess.Popen

    def capture(*a, **kw):
        p = popen(*a, **kw)
        seen.append(p)
        return p

    monkeypatch.setattr(gate.selectors, "DefaultSelector", Interrupted)
    monkeypatch.setattr(gate.subprocess, "Popen", capture)
    with pytest.raises(KeyboardInterrupt):
        gate._run_process(command("import time;time.sleep(2)"), b"{}", 2)
    assert len(seen) == 1 and seen[0].poll() is not None


@pytest.mark.parametrize(
    "mutate",
    ["wrong_binding", "bad_verdict", "business_import", "torch_import", "invalid_json"],
)
def test_even_exit0_requires_exact_bound_verdict(tmp_path, monkeypatch, mutate):
    core, rp = complete(tmp_path)
    raw = request(core, rp)
    payload = {
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "kind": "capture",
        "worker_readonly": True,
        "business_modules_loaded": False,
        "torch_loaded": False,
        "verdict": {
            "candidate_validated": True,
            "interpretation_permitted": False,
            "qualification_updates_permitted": False,
        },
    }
    if mutate == "wrong_binding":
        payload["request_sha256"] = "0" * 64
    if mutate == "bad_verdict":
        payload["verdict"]["candidate_validated"] = False
    if mutate == "business_import":
        payload["business_modules_loaded"] = True
    if mutate == "torch_import":
        payload["torch_loaded"] = True
    wire = json.dumps(payload).encode() if mutate != "invalid_json" else b"not json"
    monkeypatch.setattr(
        gate,
        "_run_process",
        lambda *a: {
            "transport_ok": True,
            "reason": None,
            "stdout": wire,
            "exit_code": 0,
            "child_reaped": True,
        },
    )
    result = gate.run_validation(raw)
    assert not result["validation_ok"] and result["status"] == "HOLD"


def test_failure_stops_before_next_validation_step_without_retry(monkeypatch):
    seen = []

    def fail(raw, **kw):
        seen.append(raw)
        return {"validation_ok": False, "status": "HOLD"}

    monkeypatch.setattr(gate, "run_validation", fail)
    result = gate.validate_in_order([("B", b"first"), ("C", b"forbidden")])
    assert (
        seen == [b"first"]
        and result["status"] == "HOLD"
        and not result["remaining_steps_started"]
    )


def test_spawn_failure_is_hold_not_exception(tmp_path):
    core, rp = complete(tmp_path)
    result = gate.run_validation(
        request(core, rp), python_executable=tmp_path / "missing_python"
    )
    assert (
        result["status"] == "HOLD"
        and result["transport"]["error_type"] == "FileNotFoundError"
    )


@pytest.mark.parametrize(
    "raw", [b"", b"[]", b"{}", b"x" * 32769, b'{"kind":"capture","kind":"compare"}']
)
def test_bad_request_rejected_before_spawn(raw, monkeypatch):
    monkeypatch.setattr(
        gate.subprocess,
        "Popen",
        mock.Mock(side_effect=AssertionError("must not launch")),
    )
    with pytest.raises(ValueError):
        gate.run_validation(raw)


def test_timeout_cannot_be_extended_and_parent_import_is_clean():
    for seconds in (0, 31, float("nan"), True):
        with pytest.raises(ValueError):
            gate._run_process(command("pass"), b"{}", seconds)
    subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys;import diagnostic_storage.validator_process_gate_v1;assert "numpy" not in sys.modules;assert "torch" not in sys.modules;assert not any(n.startswith("phase3_") for n in sys.modules)',
        ],
        check=True,
    )


def test_malformed_first_step_is_hold_and_never_starts_next():
    result = gate.validate_in_order([("B", b"bad request"), ("C", b"not allowed")])
    assert (
        result["status"] == "HOLD"
        and len(result["steps"]) == 1
        and not result["remaining_steps_started"]
    )


def test_exit0_verdict_arriving_after_parent_deadline_is_hold(tmp_path, monkeypatch):
    core, rp = complete(tmp_path)
    raw = request(core, rp)
    payload = {
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "kind": "capture",
        "worker_readonly": True,
        "business_modules_loaded": False,
        "torch_loaded": False,
        "verdict": {
            "candidate_validated": True,
            "interpretation_permitted": False,
            "qualification_updates_permitted": False,
        },
    }
    monkeypatch.setattr(
        gate,
        "_run_process",
        lambda *a: {
            "transport_ok": True,
            "reason": None,
            "stdout": json.dumps(payload).encode(),
            "exit_code": 0,
            "child_reaped": True,
        },
    )
    ticks = iter([0, 31])
    monkeypatch.setattr(gate.time, "monotonic", lambda: next(ticks))
    result = gate.run_validation(raw)
    assert result["status"] == "HOLD" and result["reason"] == "timeout"


def test_existing_controller_sequence_does_not_start_C_after_gate_timeout(tmp_path):
    from scripts.run_limited_topology_controller import sequence

    core, rp = complete(tmp_path)
    raw = request(core, rp)
    seen = []

    def execute(arm, timeout):
        seen.append(arm)
        return {"exit_code": 0, "timed_out": False}

    def assess(arm):
        return (
            True
            if arm == "A"
            else gate.run_validation(raw, timeout_seconds=0.001)["validation_ok"]
        )

    result = sequence(execute, assess, emit=lambda _: None)
    assert result["status"] == "mismatch_hold" and seen == ["A", "B"]
    assert not (core.root / "adapter_complete.json").exists()
