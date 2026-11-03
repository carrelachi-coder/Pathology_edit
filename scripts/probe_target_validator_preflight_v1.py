#!/usr/bin/env python3
"""Target-host technical profile only: fixed synthetic cases, no real source API.

Creates fresh synthetic B/C without outer acceptance, then uses independent
read-only workers under a30s OS bound. No live Controller or S07 entrypoint.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from diagnostic_storage.observer_adapter_v1 import ObserverAdapter
from diagnostic_storage.validator_process_gate_v1 import (
    run_validation,
    validate_in_order,
    _run_process,
)

EVENTS = 2100000
ARRAYS = 683


def load_fixture():
    path = Path(__file__).with_name("stress_observer_adapter_v1.py")
    spec = importlib.util.spec_from_file_location("_accepted_synthetic_driver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture_request(core, receipt, fixture):
    return json.dumps(
        {
            "kind": "capture",
            "identity": fixture.identity(),
            "capture": str(core.root),
            "receipt": str(receipt),
            "mode": core.mode,
        },
        separators=(",", ":"),
    ).encode()


def faults(root, fixture):
    root.mkdir(exist_ok=False)
    core = ObserverAdapter(
        root / "small", mode="hash_only", identity=fixture.identity()
    )
    for op in fixture.identity()["operations"]:
        core.sink.record("qualification.begin", {"case": fixture.Case(**op)})
        core.sink.record(
            "qualification.complete",
            {
                "failures": ["synthetic-negative"],
                "witness": {"errors": ["synthetic-negative"]},
            },
        )
    core.sink.close()
    receipt = fixture.receipt(root / "receipt.json")
    core.finalize(normal_invocation_end=True, business_receipt_path=receipt)
    raw = capture_request(core, receipt, fixture)
    result = validate_in_order([("B", raw), ("C", raw)], timeout_seconds=0.001)
    assert result["status"] == "HOLD" and len(result["steps"]) == 1
    assert not (core.root / "adapter_complete.json").exists()
    marker = root / "forbidden_late_write"
    code = (
        'import time;print("valid-looking",flush=True);time.sleep(1);open('
        + repr(str(marker))
        + ',"w").write("late")'
    )
    late = _run_process([sys.executable, "-c", code], b"{}", 0.05)
    assert late["timed_out"] and late["child_reaped"]
    time.sleep(1.1)
    assert not marker.exists()
    late.pop("stdout")
    return {
        "forced_validation_timeout": result,
        "late_exit_probe": late,
        "capture_marker_exists": False,
        "late_marker_exists": False,
        "real_witness_calls": 0,
    }


def profile(root, fixture):
    root.mkdir(exist_ok=False)
    initial = fixture.globals_sha()
    rows = []
    started = []
    for arm, mode in [("B", "hash_only"), ("C", "full_capture")]:
        started.append(arm)
        core = ObserverAdapter(root / arm, mode=mode, identity=fixture.identity())
        assert core.valid
        generator = np.random.default_rng(20261103)
        guard = 0
        snapshots = 0
        interval = (EVENTS - 4) // (ARRAYS + 1)
        begin = time.perf_counter()
        for i in range(EVENTS):
            if i in (0, EVENTS // 2):
                op = fixture.identity()["operations"][0 if i == 0 else 1]
                core.sink.record("qualification.begin", {"case": fixture.Case(**op)})
            elif i in (EVENTS // 2 - 1, EVENTS - 1):
                core.sink.record(
                    "qualification.complete",
                    {
                        "failures": ["synthetic-negative"],
                        "witness": {"errors": ["synthetic-negative"]},
                    },
                )
            else:
                guard += 1
                values = {
                    "row": (i // 512) % 512,
                    "col": i % 512,
                    "accepted": i % 3 == 0,
                }
                if guard % interval == 0 and snapshots < ARRAYS:
                    values["snapshot"] = generator.random((512, 512), dtype=np.float64)
                    snapshots += 1
                core.sink.record("guard.stage", values)
            if core.sink.failed or not core.valid:
                raise RuntimeError("synthetic capture invalid")
        assert snapshots == ARRAYS
        core.sink.close()
        receipt = fixture.receipt(root / (arm + "_receipt.json"))
        core.finalize(normal_invocation_end=True, business_receipt_path=receipt)
        assert core.valid
        encode = time.perf_counter() - begin
        verdict = run_validation(capture_request(core, receipt, fixture))
        rows.append(
            {
                "arm": arm,
                "events": EVENTS,
                "snapshots": snapshots,
                "encode_seconds": encode,
                "validation_gate": verdict,
                "storage": fixture.allocated(core.root),
            }
        )
        assert not (core.root / "adapter_complete.json").exists()
        if not verdict["validation_ok"]:
            return {
                "status": "HOLD",
                "started_synthetic_arms": started,
                "arms": rows,
                "next_synthetic_arm_started": False,
                "comparison": "not_run_hold",
                "automatic_retries": 0,
            }
    arrays = fixture.independent_snapshots(root / "C", ARRAYS)
    raw = json.dumps(
        {
            "kind": "compare",
            "identity": fixture.identity(),
            "left": str(root / "B"),
            "right": str(root / "C"),
            "left_receipt": str(root / "B_receipt.json"),
            "right_receipt": str(root / "C_receipt.json"),
        },
        separators=(",", ":"),
    ).encode()
    compared = run_validation(raw)
    assert fixture.globals_sha() == initial
    assert (
        not (root / "B/adapter_complete.json").exists()
        and not (root / "C/adapter_complete.json").exists()
    )
    return {
        "status": "target_synthetic_readonly_pass"
        if compared["validation_ok"]
        else "HOLD",
        "started_synthetic_arms": started,
        "arms": rows,
        "independent_arrays": arrays,
        "comparison_gate": compared,
        "global_python_numpy_rng_unchanged": True,
        "outer_acceptance_markers_published": False,
        "automatic_retries": 0,
        "real_witness_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = load_fixture()
    safety = faults(args.output / "separate_safety_fixture", fixture)
    (args.output / "safety_report.json").write_text(json.dumps(safety, indent=2) + "\n")
    result = profile(args.output / "fixed_2p1m_profile", fixture)
    result.update(
        {
            "scope": "isolated target synthetic technical preflight; not production integration or S07",
            "business_modules_loaded": any(
                n.startswith(("phase3_", "inpaint_cells")) for n in sys.modules
            ),
            "torch_loaded": "torch" in sys.modules,
            "execution_permitted": False,
        }
    )
    (args.output / "target_profile_report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "safety_fixture_passed": True,
                "real_witness_calls": 0,
            }
        ),
        flush=True,
    )
