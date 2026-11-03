"""Candidate30/60 policy tests on small local synthetic fixtures, no target trial."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
from unittest import mock

import pytest
from diagnostic_storage import validation_budget_policy_v2 as policy
from diagnostic_storage import validator_process_gate_v1 as gate1
from diagnostic_storage import validator_process_gate_v2 as gate2
from diagnostic_storage import observer_adapter_validation_v1 as validator1
from diagnostic_storage import observer_adapter_validation_v2 as validator2
from tests.test_observer_codec_adapter_v1 import complete, identity
from tests.test_validator_process_gate_v1 import request, command


def pair_request(b, br, c, cr):
    return json.dumps(
        {
            "kind": "compare",
            "identity": identity(),
            "left": str(b.root),
            "right": str(c.root),
            "left_receipt": str(br),
            "right_receipt": str(cr),
        }
    ).encode()


def test_trusted_policy_and_legacy_files_are_pinned():
    assert policy.budget_for("capture") == 30 and policy.budget_for("compare") == 60
    with pytest.raises(TypeError):
        policy.BUDGETS["compare"] = 999
    assert policy.verify_legacy_files() == dict(policy.PRODUCER_PINS)
    assert (
        policy.policy_identity()["policy_id"] == "readonly-validation-budget-v2-30-60"
    )
    for kind in ("unknown", 60, None):
        with pytest.raises(ValueError):
            policy.budget_for(kind)
    with pytest.raises(ValueError):
        validator1._deadline(60)
    with pytest.raises(ValueError):
        gate1._run_process(command("pass"), b"{}", 60)


@pytest.mark.parametrize(
    "kind,seconds",
    [("capture", 31), ("compare", 61), ("compare", float("inf")), ("capture", True)],
)
def test_no_arbitrary_timeout_from_any_layer(kind, seconds):
    with pytest.raises(ValueError):
        validator2._deadline(kind, seconds)
    with pytest.raises(ValueError):
        gate2._run_process(command("pass"), b"{}", seconds, kind=kind)


def test_data_request_cannot_override_policy(tmp_path, monkeypatch):
    b, br = complete(tmp_path)
    value = json.loads(request(b, br))
    value["deadline_seconds"] = 600
    monkeypatch.setattr(
        gate2.subprocess,
        "Popen",
        mock.Mock(side_effect=AssertionError("must not launch")),
    )
    with pytest.raises(ValueError):
        gate2.run_validation(json.dumps(value).encode())


@pytest.mark.parametrize("kind,expected", [("capture", 30), ("compare", 60)])
def test_real_worker_explicit_budget_reaches_library_and_parent(
    tmp_path, kind, expected
):
    b, br = complete(tmp_path, "B")
    raw = request(b, br)
    if kind == "compare":
        c, cr = complete(tmp_path, "C", "full_capture")
        raw = pair_request(b, br, c, cr)
    result = gate2.run_validation(raw)
    assert result["validation_ok"], result
    assert result["transport"]["hard_timeout_seconds"] == expected
    child = result["worker_receipt"]
    assert (
        child["validation_budget_seconds"]
        == child["verdict"]["validation_budget_seconds"]
        == expected
    )
    assert child["verdict"]["producer_component_sha256"] == dict(policy.PRODUCER_PINS)
    assert child["verdict"]["consumer_component_sha256"] == policy.consumer_identity()
    assert not result["execution_permitted"] and not result["capture_marker_published"]
    assert not (b.root / "adapter_complete.json").exists()


def test_compare_uses_v2_clock_beyond30_not_old_hidden_deadline(tmp_path, monkeypatch):
    b, br = complete(tmp_path, "B")
    c, cr = complete(tmp_path, "C", "full_capture")
    before = {
        p: p.read_bytes()
        for root in (b.root, c.root)
        for p in root.iterdir()
        if p.is_file()
    }
    ticks = [0]

    def clock():
        ticks[0] += 1
        return 0 if ticks[0] == 1 else 31

    monkeypatch.setattr(validator2.time, "monotonic", clock)
    monkeypatch.setattr(
        validator1,
        "_deadline",
        mock.Mock(side_effect=AssertionError("legacy30s must not be used")),
    )
    result = validator2.compare_captures(
        b.root, c.root, identity=identity(), left_receipt=br, right_receipt=cr
    )
    assert result["status"] == "semantic_match"
    assert all(p.read_bytes() == raw for p, raw in before.items())
    with pytest.raises(TimeoutError):
        validator2.validate_capture(
            b.root,
            identity=identity(),
            business_receipt_path=br,
            expected_mode="hash_only",
            deadline_seconds=0,
        )


def test_compatibility_does_not_rewrite_or_ignore_producer_identity(tmp_path):
    b, br = complete(tmp_path)
    p = b.root / "adapter_manifest.json"
    value = json.loads(p.read_text())
    value["code_sha256"]["observer_adapter_validation_v1.py"] = "f" * 64
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validator2.validate_capture(
            b.root,
            identity=identity(),
            business_receipt_path=br,
            expected_mode="hash_only",
        )
    assert not (b.root / "adapter_complete.json").exists()


def test_legacy_code_drift_rejected_before_spawn(tmp_path, monkeypatch):
    b, br = complete(tmp_path)
    monkeypatch.setattr(policy, "file_sha", lambda p: "0" * 64)
    monkeypatch.setattr(
        gate2.subprocess,
        "Popen",
        mock.Mock(side_effect=AssertionError("must not launch")),
    )
    with pytest.raises(ValueError):
        gate2.run_validation(request(b, br))


@pytest.mark.parametrize(
    "mutation", ["top_budget", "inner_budget", "inner_policy", "consumer_code"]
)
def test_worker_cannot_merely_claim_outer60_while_running_old_inner30(
    tmp_path, monkeypatch, mutation
):
    b, br = complete(tmp_path)
    raw = request(b, br)
    verdict = {
        "candidate_validated": True,
        "interpretation_permitted": False,
        "qualification_updates_permitted": False,
        **policy.policy_identity(),
        "validation_budget_seconds": 30,
        "consumer_component_sha256": policy.consumer_identity(),
    }
    payload = {
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "kind": "capture",
        "worker_readonly": True,
        "business_modules_loaded": False,
        "torch_loaded": False,
        **policy.policy_identity(),
        "validation_budget_seconds": 30,
        "verdict": verdict,
    }
    if mutation == "top_budget":
        payload["validation_budget_seconds"] = 60
    if mutation == "inner_budget":
        verdict["validation_budget_seconds"] = 60
    if mutation == "inner_policy":
        verdict["policy_sha256"] = "0" * 64
    if mutation == "consumer_code":
        verdict["consumer_component_sha256"] = {}
    monkeypatch.setattr(
        gate2,
        "_run_process",
        lambda *a, **kw: {
            "transport_ok": True,
            "reason": None,
            "stdout": json.dumps(payload).encode(),
            "exit_code": 0,
            "child_reaped": True,
        },
    )
    assert gate2.run_validation(raw)["status"] == "HOLD"


def test_v2_preserves_exact_transport_ast_except_task_cap():
    old = ast.parse(textwrap.dedent(inspect.getsource(gate1._run_process))).body[0]
    new = ast.parse(textwrap.dedent(inspect.getsource(gate2._run_process))).body[0]
    old.args.kwonlyargs = [ast.arg(arg="kind")]
    old.args.kw_defaults = [None]

    class Cap(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "HARD_SECONDS":
                return ast.Call(
                    func=ast.Name(id="budget_for", ctx=ast.Load()),
                    args=[ast.Name(id="kind", ctx=ast.Load())],
                    keywords=[],
                )
            return node

        def visit_Constant(self, node):
            if node.value == "hard timeout may not exceed30s":
                node.value = "hard timeout exceeds trusted task policy"
            return node

    old = Cap().visit(old)
    assert ast.dump(old, include_attributes=False) == ast.dump(
        new, include_attributes=False
    )


def test_actual_short_hard_timeout_still_holds_and_reaps_under_v2():
    result = gate2._run_process(
        command("import time;time.sleep(2)"), b"{}", 0.05, kind="compare"
    )
    assert result["timed_out"] and result["child_reaped"] and not result["transport_ok"]


def test_v2_still_checks_all_events_arrays_and_receipts_in_one_pass(
    tmp_path, monkeypatch
):
    from diagnostic_storage import guard_blocks_v2 as codec

    b, br = complete(tmp_path, "B")
    c, cr = complete(tmp_path, "C", "full_capture")
    calls = []
    original = codec.Reader.iter_events

    def count(reader, *a, **kw):
        calls.append(str(reader.root))
        yield from original(reader, *a, **kw)

    monkeypatch.setattr(codec.Reader, "iter_events", count)
    result = validator2.compare_captures(
        b.root, c.root, identity=identity(), left_receipt=br, right_receipt=cr
    )
    assert (
        result["excluded_semantic_fields"] == []
        and len(calls) == 2
        and len(set(calls)) == 2
    )
    path = next((c.root / "codec").glob("array_*.bin"))
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"X")
    with pytest.raises(ValueError):
        validator2.compare_captures(
            b.root, c.root, identity=identity(), left_receipt=br, right_receipt=cr
        )


def trial_manifest():
    from scripts.run_readonly_budget_v2_trial import KIND

    common = {"identity": identity()}
    values = [
        (
            "B_capture",
            {
                "kind": "capture",
                **common,
                "capture": "B",
                "receipt": "Br",
                "mode": "hash_only",
            },
        ),
        (
            "C_capture",
            {
                "kind": "capture",
                **common,
                "capture": "C",
                "receipt": "Cr",
                "mode": "full_capture",
            },
        ),
        (
            "BC_compare",
            {
                "kind": "compare",
                **common,
                "left": "B",
                "right": "C",
                "left_receipt": "Br",
                "right_receipt": "Cr",
            },
        ),
    ]
    m = {
        "kind": KIND,
        "permissions": {
            "production": False,
            "S07": False,
            "generation": False,
            "publication": False,
        },
        "automatic_retries": 0,
        "jobs": [
            {
                "label": label,
                "request": value,
                "hard_seconds": policy.budget_for(value["kind"]),
                "measurements": 1,
                "request_sha256": hashlib.sha256(
                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for label, value in values
        ],
    }
    raw = json.dumps(m).encode()
    approval = {
        "kind": KIND,
        "approved": True,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "policy": policy.policy_identity(),
        "compatibility_profile": policy.COMPATIBILITY_ID,
        "compatibility_reviewed": True,
    }
    return raw, approval


@pytest.mark.parametrize(
    "change",
    [
        "old_S07_approval",
        "missing_compatibility_review",
        "wrong_manifest",
        "wrong_policy",
    ],
)
def test_target_trial_cannot_run_before_exact_reviewed_approval(change):
    from scripts.run_readonly_budget_v2_trial import authorize

    raw, approval = trial_manifest()
    if change == "old_S07_approval":
        approval = {"approved": True, "manifest_sha256": approval["manifest_sha256"]}
    if change == "missing_compatibility_review":
        approval["compatibility_reviewed"] = False
    if change == "wrong_manifest":
        approval["manifest_sha256"] = "0" * 64
    if change == "wrong_policy":
        approval["policy"] = {}
    with pytest.raises(PermissionError):
        authorize(raw, approval)


def test_target_trial_count_and_scope_are_fixed_before_measurement():
    from scripts.run_readonly_budget_v2_trial import authorize

    raw, approval = trial_manifest()
    assert authorize(raw, approval)[0]["automatic_retries"] == 0
    m = json.loads(raw)
    m["jobs"][-1]["measurements"] = 2
    raw = json.dumps(m).encode()
    approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        authorize(raw, approval)
