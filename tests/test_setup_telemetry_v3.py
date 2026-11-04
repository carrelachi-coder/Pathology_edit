import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest import mock
import pytest

from diagnostic_storage import validation_budget_policy_v2 as policy
from scripts import run_readonly_budget_v2_trial as old
from scripts import preflight_setup_telemetry_v3 as new


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    root.mkdir()
    files = {}
    for i in range(809):
        p = root / str(i)
        p.write_bytes(f"input-{i}".encode())
        files[str(i)] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    inv = tmp_path / "inventory.json"
    inv.write_text(json.dumps({"files": files}))
    distributions = []
    packages = []
    for i in range(2):
        p = tmp_path / f"dist{i}"
        p.mkdir()
        (p / "RECORD").write_text(f"record{i}")
        distributions.append(
            SimpleNamespace(_path=p, metadata={"Name": f"d{i}"}, version="1")
        )
        packages.append(
            {"name": f"d{i}", "version": "1", "record_sha256": sha(p / "RECORD")}
        )
    monkeypatch.setattr(
        new.importlib.metadata, "distributions", lambda: iter(distributions)
    )
    native = tmp_path / "native"
    native.write_bytes(b"native")
    code = tmp_path / "code"
    code.write_bytes(b"code")
    reserve = tmp_path / "reserve"
    reserve.write_bytes(b"reserved")
    output = tmp_path / "output"
    return {
        "setup_watchdog_seconds": 60,
        "runtime_code_files": {str(code): sha(code)},
        "runtime": {
            "executable": sys.executable,
            "executable_sha256": sha(sys.executable),
            "environment": {},
            "packages_sha256": hashlib.sha256(
                json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "native_files_sha256": {str(native): sha(native)},
        },
        "input_inventory": {"path": str(inv), "sha256": sha(inv)},
        "input_root": str(root),
        "output_root": str(output),
        "minimum_free_output_bytes": 0,
        "existing_reservation": {"path": str(reserve), "bytes": 8},
    }


def run(m, path):
    t = new.SetupTelemetry(path)
    try:
        r = new.preflight(m, t)
    except BaseException as exc:
        state = t.finish(False, type(exc).__name__)
        return None, state, exc
    return r, t.finish(True), None


def test_full_identity_order_matches_frozen_v2_and_progress_is_bounded(
    tmp_path, monkeypatch
):
    m = fixture(tmp_path, monkeypatch)
    output = Path(m["output_root"])
    output.mkdir()
    old_trace, new_trace = [], []
    original = policy.file_sha

    def old_hash(path):
        old_trace.append(str(path))
        return original(path)

    monkeypatch.setattr(old, "file_sha", old_hash)
    monkeypatch.setattr(policy, "file_sha", old_hash)
    before = random.getstate()
    a = old.preflight(m)
    for name in ["link_probe_pending", "link_probe_final"]:
        (output / name).unlink()
    original_new = new.hash_file

    def new_hash(path, t):
        new_trace.append(str(path))
        return original_new(path, t)

    monkeypatch.setattr(new, "hash_file", new_hash)
    b, state, exc = run(m, tmp_path / "journal")
    assert exc is None and old_trace == new_trace
    assert {k: v for k, v in a.items() if k != "elapsed_seconds"} == {
        k: v for k, v in b.items() if k != "elapsed_seconds"
    }
    assert random.getstate() == before
    assert [s["stage"] for s in state["stages"]] == list(new.STAGES)
    assert [s["items_completed"] for s in state["stages"]] == [5, 1, 1, 2, 1, 809, 3]
    assert all(s["stage_completed"] for s in state["stages"])
    assert state["last_completed_stage"] == "storage" and state["telemetry_valid"]
    assert state["telemetry_bytes"] <= new.SetupTelemetry.MAX_BYTES
    rows = [json.loads(x) for x in (tmp_path / "journal").read_text().splitlines()]
    assert [x["sequence"] for x in rows] == list(range(len(rows)))
    assert not any(
        "input-" in json.dumps(x) or str(tmp_path) in json.dumps(x) for x in rows
    )
    invstage = state["stages"][5]
    inventory = json.loads(Path(m["input_inventory"]["path"]).read_text())
    assert invstage["bytes_processed"] == Path(
        m["input_inventory"]["path"]
    ).stat().st_size + sum(v["bytes"] for v in inventory["files"].values())


@pytest.mark.parametrize(
    "stage,trigger",
    [
        ("legacy_code", 1),
        ("runtime_code", 1),
        ("distributions", 1),
        ("native_binaries", 1),
        ("input_inventory", 4),
    ],
)
def test_slow_hash_timeout_matches_old_verdict_and_read_order(
    tmp_path, monkeypatch, stage, trigger
):
    m = fixture(tmp_path, monkeypatch)
    Path(m["output_root"]).mkdir()
    now = [0.0]
    monkeypatch.setattr(new.time, "monotonic", lambda: now[0])
    trace = []
    slowpaths = {
        "legacy_code": str(
            Path(policy.__file__).parent / next(iter(policy.PRODUCER_PINS))
        ),
        "runtime_code": next(iter(m["runtime_code_files"])),
        "distributions": str(tmp_path / "dist0/RECORD"),
        "native_binaries": next(iter(m["runtime"]["native_files_sha256"])),
        "input_inventory": str(Path(m["input_root"]) / "3"),
    }
    original = policy.file_sha

    def old_hash(path):
        trace.append(str(path))
        digest = original(path)
        if str(path) == slowpaths[stage]:
            now[0] = 61
        return digest

    monkeypatch.setattr(old, "file_sha", old_hash)
    monkeypatch.setattr(policy, "file_sha", old_hash)
    with pytest.raises(TimeoutError):
        old.preflight(m)
    baseline = list(trace)
    now[0] = 0
    trace.clear()
    h = new.hash_file

    def observed(path, t):
        trace.append(str(path))
        digest = h(path, t)
        if str(path) == slowpaths[stage]:
            now[0] = 61
        return digest

    monkeypatch.setattr(new, "hash_file", observed)
    _, s, exc = run(m, tmp_path / "journal")
    assert isinstance(exc, TimeoutError) and trace == baseline
    current = s["stages"][-1]
    assert not current["stage_completed"]
    assert current["elapsed_ms"] >= 0 and not s["preparation_accepted"]
    if stage == "input_inventory":
        assert current["stage"] == "input_inventory" and current["items_completed"] == 4
        assert current["last_completed_item_index"] == 3
        assert s["last_completed_stage"] == "native_binaries"


@pytest.mark.parametrize("operation", ["mismatch", "read_error", "keyboard_interrupt"])
def test_partial_hash_failures_preserve_primary_error_and_last_complete_item(
    tmp_path, monkeypatch, operation
):
    m = fixture(tmp_path, monkeypatch)
    Path(m["output_root"]).mkdir()
    target = str(Path(m["input_root"]) / "4")
    h = new.hash_file

    def failed(path, t):
        if str(path) == target:
            if operation == "mismatch":
                return "0" * 64
            if operation == "read_error":
                raise OSError("read failed")
            raise KeyboardInterrupt()
        return h(path, t)

    monkeypatch.setattr(new, "hash_file", failed)
    _, s, exc = run(m, tmp_path / "journal")
    assert isinstance(
        exc,
        {
            "mismatch": ValueError,
            "read_error": OSError,
            "keyboard_interrupt": KeyboardInterrupt,
        }[operation],
    )
    current = s["stages"][-1]
    assert (
        current["stage"] == "input_inventory"
        and current["items_completed"] == 4
        and current["last_completed_item_index"] == 3
    )
    assert s["telemetry_valid"] and not s["identity_check_passed"]
    assert not current["stage_completed"]


@pytest.mark.parametrize(
    "fault", ["open", "write", "enospc", "partial", "flush", "close", "limit"]
)
def test_telemetry_fault_does_not_change_identity_results_but_invalidates_evidence(
    tmp_path, monkeypatch, fault
):
    m = fixture(tmp_path, monkeypatch)
    Path(m["output_root"]).mkdir()
    path = tmp_path / "journal"
    if fault == "open":
        path.mkdir()
    t = new.SetupTelemetry(path)
    if fault not in ("open", "limit"):
        t.stream.close()
        stream = mock.Mock()
        stream.write.side_effect = lambda b: len(b)
        stream.fileno.return_value = 1
        if fault == "write":
            stream.write.side_effect = OSError("write")
        if fault == "enospc":
            stream.write.side_effect = OSError(28, "no space")
        if fault == "partial":
            stream.write.side_effect = lambda b: 1
        if fault in ("flush", "close"):
            getattr(stream, fault).side_effect = OSError(fault)
        t.stream = stream
        monkeypatch.setattr(new.os, "fsync", lambda n: None)
    if fault == "limit":
        t.MAX_RECORDS = 2
    r = new.preflight(m, t)
    s = t.finish(True)
    assert r["code_and_runtime_match"] and s["identity_check_passed"]
    assert not s["telemetry_valid"] and not s["preparation_accepted"]
    assert s["fault_types"]


def test_preparation_wrapper_holds_and_refuses_directory_reuse(tmp_path, monkeypatch):
    m = fixture(tmp_path, monkeypatch)

    def interrupted(m, t):
        with t.stage("input_inventory"):
            t.completed()
            raise KeyboardInterrupt()

    monkeypatch.setattr(new, "preflight", interrupted)
    result = new.capture_preflight(m, m["output_root"])
    assert result["status"] == "HOLD" and result["failure_type"] == "KeyboardInterrupt"
    saved = json.loads((Path(m["output_root"]) / "setup_summary.json").read_text())
    assert saved["stages"][0]["items_completed"] == 1
    with pytest.raises(FileExistsError):
        new.capture_preflight(m, m["output_root"])


def test_journal_prefix_survives_abrupt_process_exit_without_valid_summary(tmp_path):
    import subprocess
    import os

    target = tmp_path / "journal"
    code = """import os
from scripts.preflight_setup_telemetry_v3 import SetupTelemetry
t=SetupTelemetry(PATH)
with t.stage('input_inventory'):
 t.completed()
 os._exit(9)
""".replace(
        "PATH", repr(str(target))
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert p.returncode == 9
    rows = [json.loads(x) for x in target.read_text().splitlines()]
    assert (
        rows[-1]["event"] == "item_completed"
        and rows[-1]["last_completed_item_index"] == 0
    )
    assert not (tmp_path / "setup_summary.json").exists()


def test_trace_fault_plus_identity_failure_preserves_original_value_error(
    tmp_path, monkeypatch
):
    m = fixture(tmp_path, monkeypatch)
    Path(m["output_root"]).mkdir()
    m["runtime"]["packages_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="distribution identity drift"):
        old.preflight(m)
    journal = tmp_path / "not-a-file"
    journal.mkdir()
    _, state, exc = run(m, journal)
    assert isinstance(exc, ValueError) and str(exc) == "distribution identity drift"
    assert not state["telemetry_valid"] and not state["preparation_accepted"]
    current = state["stages"][-1]
    assert current["items_completed"] == 2 and not current["stage_completed"]
    assert state["last_completed_stage"] == "interpreter_environment"


def test_storage_slow_operation_preserves_final_check_timeout(tmp_path, monkeypatch):
    m = fixture(tmp_path, monkeypatch)
    Path(m["output_root"]).mkdir()
    clock = [0.0]
    monkeypatch.setattr(new.time, "monotonic", lambda: clock[0])
    original = new.shutil.disk_usage

    def slow(path):
        result = original(path)
        clock[0] = 61
        return result

    monkeypatch.setattr(new.shutil, "disk_usage", slow)
    with pytest.raises(TimeoutError):
        old.preflight(m)
    for n in ["link_probe_pending", "link_probe_final"]:
        (Path(m["output_root"]) / n).unlink()
    clock[0] = 0
    _, state, exc = run(m, tmp_path / "journal")
    assert isinstance(exc, TimeoutError)
    assert state["stages"][-1]["stage"] == "storage"
    assert state["stages"][-1]["items_completed"] == 3
    assert not state["stages"][-1]["stage_completed"]
    assert state["last_completed_stage"] == "input_inventory"


def test_220_distribution_workload_stays_within_fixed_journal_budget(
    tmp_path, monkeypatch
):
    m = fixture(tmp_path, monkeypatch)
    base = list(new.importlib.metadata.distributions())
    monkeypatch.setattr(
        new.importlib.metadata, "distributions", lambda: iter(base * 110)
    )
    packages = [
        {
            "name": d.metadata["Name"],
            "version": d.version,
            "record_sha256": sha(Path(d._path) / "RECORD"),
        }
        for d in base * 110
    ]
    packages.sort(key=lambda x: x["name"].lower())
    m["runtime"]["packages_sha256"] = hashlib.sha256(
        json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = new.capture_preflight(m, m["output_root"])
    assert result["status"] == "preparation_only_pass"
    assert result["stages"][3]["items_completed"] == 220
    assert result["records"] < 2048 and result["telemetry_bytes"] < 1048576
    assert result["execution_permitted"] is False
    assert sorted(p.name for p in Path(m["output_root"]).iterdir()) == [
        "link_probe_final",
        "link_probe_pending",
        "setup_summary.json",
        "setup_telemetry.jsonl",
    ]


def test_final_summary_write_failure_cannot_return_acceptance(tmp_path, monkeypatch):
    m = fixture(tmp_path, monkeypatch)
    original = Path.open

    def opened(path, *a, **kw):
        if path.name == "setup_summary.json":
            raise OSError(28, "full")
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(OSError):
        new.capture_preflight(m, m["output_root"])
    assert not (Path(m["output_root"]) / "setup_summary.json").exists()


def test_partial_file_read_error_never_counts_unfinished_hash_bytes(
    tmp_path, monkeypatch
):
    t = new.SetupTelemetry(tmp_path / "journal")
    stream = mock.MagicMock()
    stream.__enter__.return_value = stream
    stream.read.side_effect = [b"a" * 16, OSError("interrupted read")]
    with t.stage("input_inventory"):
        with mock.patch.object(Path, "open", return_value=stream):
            with pytest.raises(OSError):
                new.hash_file(tmp_path / "dummy", t)
    s = t.finish(False, "OSError")
    assert s["stages"][0]["bytes_processed"] == 0
    assert s["stages"][0]["last_completed_item_index"] is None
