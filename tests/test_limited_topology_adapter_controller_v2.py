import json
import random
from pathlib import Path
import pytest
from diagnostic_storage.observer_adapter_v2 import ObserverAdapter
from diagnostic_storage.observer_invocation_bridge_v1 import invoke_with_adapter
from diagnostic_storage.observer_adapter_validation_v3_budget import compare_captures
from phase3_mask_edit_refine.passive_observer import observe
from scripts import run_limited_topology_adapter_controller_v2 as controller
from tests.test_observer_codec_adapter_v2_budget import identity, receipt, events


def test_actual_codec_bridge_and_full_comparison(tmp_path):
    before = random.getstate()
    for arm, mode in [("B", "hash_only"), ("C", "full_capture")]:
        folder = tmp_path / arm
        folder.mkdir()
        rp = receipt(folder / "receipt.json")
        core = ObserverAdapter(folder / "observer", mode=mode, identity=identity())

        def invoke():
            for event, values in events():
                observe(event, **values)
            return "same business result"

        assert (
            invoke_with_adapter(invoke, core, business_receipt_path=rp)
            == "same business result"
        )
        assert core.status()["phase"] == "SEALED_PENDING_VALIDATION"
    result = compare_captures(
        tmp_path / "B/observer",
        tmp_path / "C/observer",
        identity=identity(),
        left_receipt=tmp_path / "B/receipt.json",
        right_receipt=tmp_path / "C/receipt.json",
    )
    assert result["status"] == "semantic_match" and result["events"] == 6
    assert result["excluded_semantic_fields"] == [] and random.getstate() == before


def test_old_controller_authorization_cannot_open_adapter_controller():
    with pytest.raises(PermissionError, match="new Adapter"):
        controller.approval_gate({}, "a", {"approved": True, "manifest_sha256": "a"})


def test_previous_adapter_authorization_cannot_open_v2():
    plan = {
        "controller_kind": "limited-topology-adapter-v2-budget",
        "execution_permitted": True,
        "rng_decision_approved": True,
        "preflight_complete": True,
        "rng_decision": {"sha256": "r" * 64},
        "rng_policy": "common_parent_explicit_postfork_restore_v1",
        "limits": {
            "source_processes": 3,
            "witness_calls": 6,
            "seconds_per_source_pair": 1200,
            "retries": 0,
        },
    }
    previous = {
        "kind": "limited-topology-adapter-v1",
        "approved": True,
        "manifest_sha256": "m" * 64,
        "rng_decision_sha256": "r" * 64,
    }
    with pytest.raises(PermissionError, match="new Adapter"):
        controller.approval_gate(plan, "m" * 64, previous)
    current = {**previous, "kind": "limited-topology-adapter-v2-budget"}
    controller.approval_gate(plan, "m" * 64, current)
    with pytest.raises(PermissionError, match="authorization absent"):
        controller.approval_gate(plan, "n" * 64, current)


def test_legacy_sequence_stops_before_C_when_B_invalid():
    launches = []

    def execute(arm, timeout):
        launches.append(arm)
        return {"timed_out": False, "exit_code": 0}

    result = controller.sequence(execute, lambda arm: arm != "B", emit=lambda row: None)
    assert result["status"] == "mismatch_hold" and launches == ["A", "B"]
    assert result["reserved_witness_calls"] == 4


def test_controller_assessment_uses_actual_process_gate_and_full_receipts(tmp_path):
    import sys

    ident = identity()
    for arm in ["A", "B", "C"]:
        folder = tmp_path / arm / "business"
        folder.mkdir(parents=True)
        receipt(folder / "capacity_screen.json")
        if arm == "A":
            continue
        core = ObserverAdapter(
            tmp_path / arm / "observer",
            mode="hash_only" if arm == "B" else "full_capture",
            identity=ident,
        )

        def invoke():
            for event, values in events():
                observe(event, **values)

        invoke_with_adapter(
            invoke, core, business_receipt_path=folder / "capacity_screen.json"
        )
    row = json.loads((tmp_path / "A/business/capacity_screen.json").read_text())[0]
    historical = tmp_path / "historical.json"
    historical.write_text(
        json.dumps([{"reference_id": ident["reference_id"], "result": row}])
    )
    plan = {
        "output_root": str(tmp_path),
        "reference_id": ident["reference_id"],
        "historical_receipt": str(historical),
        "adapter_identity": ident,
        "adapter_snapshot": str(Path(__file__).resolve().parents[1]),
        "runtime": {"executable": sys.executable},
    }
    assert controller.assess_real(plan, "A")
    parent_modules_before = set(sys.modules)
    parent_path_before = list(sys.path)
    assert controller.assess_real(plan, "B")
    assert set(sys.modules) == parent_modules_before
    assert sys.path == parent_path_before
    assert controller.assess_real(plan, "C")
    assert set(sys.modules) == parent_modules_before
    assert sys.path == parent_path_before
    b = json.loads((tmp_path / "B/validation_gate_receipt.json").read_text())
    c = json.loads((tmp_path / "C/validation_gate_receipt.json").read_text())
    assert b["validation_budget_seconds"] == 30 and c["validation_budget_seconds"] == 60
    assert c["worker_receipt"]["verdict"]["status"] == "semantic_match"
    assert not (tmp_path / "C/observer/adapter_complete.json").exists()


def test_scope_gate_binds_adapter_cases_to_actual_probe_construction(tmp_path):
    sources = tmp_path / "sources.json"
    cases = tmp_path / "cases.json"
    sources.write_text(
        json.dumps(
            {
                "candidate_sources": [{}] * 7
                + [{"reference_id": "source", "patient_id": "patient"}]
            }
        )
    )
    cases.write_text(
        json.dumps([{}, {"primitive_id": "u1"}, {}, {"primitive_id": "u2"}])
    )
    args = [
        "--source-freeze",
        str(sources),
        "--cases",
        str(cases),
        "--start-index",
        "7",
        "--limit",
        "1",
        "--topology-witness",
    ]
    plan = {
        "source_subset": {"path": str(sources)},
        "case_templates": {"path": str(cases)},
        "reference_id": "source",
        "arms": {
            a: {
                "probe": "scripts/probe_manuscript_joint_capacity.py",
                "probe_args": args,
                "mode": m,
            }
            for a, m in [("A", None), ("B", "hash_only"), ("C", "full_capture")]
        },
        "adapter_identity": {
            "reference_id": "source",
            "patient_id": "patient",
            "operations": [
                {"case_id": "source__u1__capacity", "primitive_id": "u1"},
                {"case_id": "source__u2__capacity", "primitive_id": "u2"},
            ],
        },
    }
    controller.scope_gate(plan)
    plan["adapter_identity"]["operations"][0]["case_id"] = "different"
    with pytest.raises(ValueError, match="adapter identity"):
        controller.scope_gate(plan)
