"""Independent Adapter synthetic tests, never a BCSS/S07 witness invocation."""
import ast
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

from diagnostic_storage import guard_blocks_v2 as codec
from diagnostic_storage import observer_adapter_v2 as adapter
from diagnostic_storage import observer_adapter_contract_v2 as contract
from diagnostic_storage import observer_adapter_validation_v2_budget as validator
from phase3_mask_edit_refine import passive_observer as legacy


@dataclass
class Case:
    case_id: str
    primitive_id: str = ""


def identity():
    return {
        "reference_id": "synthetic-source",
        "patient_id": "synthetic-patient",
        "operations": [
            {"case_id": "synthetic-U1", "primitive_id": "synthetic-primitive1"},
            {"case_id": "synthetic-U2", "primitive_id": "synthetic-primitive2"},
        ],
        **{
            k: "1" * 64
            for k in (
                "run_manifest_sha256",
                "input_identity_sha256",
                "case_mapping_sha256",
                "contract_sha256",
                "startup_identity_sha256",
            )
        },
    }


def receipt(path, ident=None):
    ident = ident or identity()
    row = {
        "reference_id": ident["reference_id"],
        "patient_id": ident["patient_id"],
        "source_only": True,
        "target_mask_created": False,
        "source_content_verified_against_census": True,
        "checks": [
            {
                "primitive_id": op["primitive_id"],
                "passed_necessary_capacity": True,
                "passed_topology_witness": False,
                "failure_reasons": ["synthetic-negative"],
                "metrics": {
                    "whole_mask_topology_witness": {
                        "errors": ["unchanged synthetic error"]
                    }
                },
            }
            for op in ident["operations"]
        ],
        "both_significant_capacity_pass": True,
        "both_significant_topology_pass": False,
    }
    path.write_text(json.dumps([row], indent=2) + "\n")
    return path


def events(values=None):
    for op in identity()["operations"]:
        yield "qualification.begin", {
            "case": Case(op["case_id"], op["primitive_id"]),
            "source": np.arange(12, dtype=">i4").reshape(3, 4),
        }
        yield "guard.stage", values if values is not None else {
            "row": 2,
            "col": 3,
            "result": False,
        }
        yield "qualification.complete", {
            "failures": ["synthetic-negative"],
            "witness": {"errors": ["unchanged synthetic error"]},
        }


def emit(sink, values=None):
    with legacy.observation_session(sink):
        for name, value in events(values):
            sink.record(name, value)


def complete(tmp_path, name="B", mode="hash_only", values=None):
    core = adapter.ObserverAdapter(tmp_path / name, mode=mode, identity=identity())
    assert core.valid
    emit(core.sink, values)
    assert (
        core.phase == "QUIESCED"
        and not (core.root / "codec/observer_complete.json").exists()
    )
    rp = receipt(tmp_path / (name + "_receipt.json"))
    core.finalize(normal_invocation_end=True, business_receipt_path=rp)
    assert core.valid, core.status()
    return core, rp


def test_legacy_bytes_rng_checkpoint_and_single_copy_differential(
    tmp_path, monkeypatch
):
    source = np.arange(15, dtype=">i4").reshape(3, 5)[:, ::-1]
    local = np.random.default_rng(123)
    exc = ValueError("same exception")
    exc.context = {"x": 1}
    values = {
        "ordered": {3: "类型", "a": True},
        "tuples": (None, False, np.int32(7)),
        "sets": {2, 4},
        "frozen": frozenset({3, 5}),
        "exception": exc,
        "local": local,
        "numbers": [float("nan"), float("inf"), -float("inf"), -0.0],
        "arrays": [source, source, np.empty((0, 3), dtype="u2")],
    }
    original = source.copy()
    rng_before = (
        random.getstate(),
        np.random.get_state(),
        json.dumps(local.bit_generator.state),
    )
    old = legacy.PassiveObserver(tmp_path / "legacy", mode="hash_only")
    emit(old, values)
    core = adapter.ObserverAdapter(
        tmp_path / "new", mode="full_capture", identity=identity()
    )
    copy = mock.Mock(wraps=core.writer._copy_array)
    monkeypatch.setattr(core.writer, "_copy_array", copy)
    emit(core.sink, values)
    rp = receipt(tmp_path / "receipt.json")
    core.finalize(normal_invocation_end=True, business_receipt_path=rp)
    raw = b"".join(codec.Reader(core.root / "codec").iter_events())
    assert raw == (tmp_path / "legacy/semantic.jsonl").read_bytes()
    summary = json.loads((core.root / "codec/summary.json").read_text())
    assert copy.call_count == summary["arrays"]
    assert np.array_equal(source, original) and source.flags.writeable
    assert random.getstate() == rng_before[0]
    assert (
        np.array_equal(np.random.get_state()[1], rng_before[1][1])
        and np.random.get_state()[2:] == rng_before[1][2:]
    )
    assert json.dumps(local.bit_generator.state) == rng_before[2]
    assert adapter._CHECKPOINTS == legacy._CHECKPOINTS


def test_only_external_validator_publishes_acceptance_and_negative_is_complete(
    tmp_path,
):
    core, rp = complete(tmp_path)
    assert (
        not (core.root / "adapter_complete.json").exists()
        and core.status()["interpretation_permitted"] is False
    )
    result = validator.validate_capture(
        core.root,
        identity=identity(),
        business_receipt_path=rp,
        expected_mode="hash_only",
    )
    assert result["candidate_validated"] and not result["interpretation_permitted"]
    result = validator.validate_capture(
        core.root,
        identity=identity(),
        business_receipt_path=rp,
        expected_mode="hash_only",
        publish=True,
    )
    assert (
        result["interpretation_permitted"]
        and not result["qualification_updates_permitted"]
    )
    assert validator.validate_capture(
        core.root,
        identity=identity(),
        business_receipt_path=rp,
        expected_mode="hash_only",
    )["interpretation_permitted"]
    with pytest.raises(contract.AdapterError):
        validator.validate_capture(
            core.root,
            identity=identity(),
            business_receipt_path=rp,
            expected_mode="hash_only",
            publish=True,
        )


def test_hash_full_stream_compare_preserves_all_fields(tmp_path):
    b, br = complete(tmp_path, "B")
    c, cr = complete(tmp_path, "C", "full_capture")
    assert (
        validator.compare_captures(
            b.root, c.root, identity=identity(), left_receipt=br, right_receipt=cr
        )["status"]
        == "semantic_match"
    )
    assert not (b.root / "adapter_complete.json").exists()
    with pytest.raises(contract.AdapterError):
        validator.compare_captures(
            c.root, b.root, identity=identity(), left_receipt=cr, right_receipt=br
        )


@pytest.mark.parametrize(
    "fault",
    [
        "record",
        "encode",
        "rng",
        "queue",
        "copy",
        "write",
        "enospc",
        "metadata",
        "partial_array",
        "record_and_invalidate",
        "quiesce_and_invalidate",
    ],
)
def test_recorder_and_double_faults_preserve_business_exception(
    tmp_path, monkeypatch, fault
):
    core = adapter.ObserverAdapter(
        tmp_path / fault, mode="full_capture", identity=identity()
    )
    original = np.arange(8, dtype="u1")
    before = original.copy()

    def broken(*a, **kw):
        if fault == "copy":
            raise MemoryError("injected copy fault")
        if fault == "enospc":
            raise OSError(28, "injected disk full")
        raise RuntimeError("injected recorder fault")

    target = {
        "record": "record",
        "encode": "_encode",
        "rng": "_rng",
        "record_and_invalidate": "record",
        "quiesce_and_invalidate": "quiesce",
    }
    if fault in target:
        monkeypatch.setattr(core, target[fault], broken)
    elif fault == "queue":
        monkeypatch.setattr(core._queue, "put_nowait", broken)
    elif fault == "copy":
        monkeypatch.setattr(core.writer, "_copy_array", broken)
    elif fault in ("write", "enospc"):
        monkeypatch.setattr(core.writer, "write_event", broken)
    elif fault == "metadata":
        core.max_event_metadata_bytes = 1
    if fault in ("record_and_invalidate", "quiesce_and_invalidate"):
        monkeypatch.setattr(core, "invalidate", broken)
    error = ValueError("original planner failure")
    with pytest.raises(ValueError) as caught:
        with legacy.observation_session(core.sink):
            legacy.observe(
                "qualification.begin", case=Case("synthetic-U1"), source=original
            )
            if fault == "partial_array":
                legacy.observe("bad", value=[original, object()])
            raise error
    assert (
        caught.value is error
        and np.array_equal(original, before)
        and original.flags.writeable
    )
    core.finalize(
        normal_invocation_end=False, business_receipt_path=tmp_path / "missing"
    )
    assert not core.valid and not (core.root / "adapter_complete.json").exists()
    assert not (core.root / "codec/observer_complete.json").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_patient",
        "wrong_source",
        "wrong_primitive",
        "missing_check",
        "caught_error_exit0",
        "contradictory",
        "missing_receipt",
        "missing_events",
        "normal_false",
    ],
)
def test_receipt_or_process_return_cannot_manufacture_completion(tmp_path, fault):
    core = adapter.ObserverAdapter(
        tmp_path / "B", mode="hash_only", identity=identity()
    )
    if fault == "missing_events":
        with legacy.observation_session(core.sink):
            legacy.observe("guard.stage", result=False)
    else:
        emit(core.sink)
    rp = receipt(tmp_path / "receipt.json")
    rows = json.loads(rp.read_text())
    if fault == "wrong_patient":
        rows[0]["patient_id"] = "different"
    if fault == "wrong_source":
        rows[0]["reference_id"] = "different"
    if fault == "wrong_primitive":
        rows[0]["checks"][1]["primitive_id"] = "other"
    if fault == "missing_check":
        rows[0]["checks"].pop()
    if fault == "caught_error_exit0":
        rows[0]["checks"][0] = {
            "primitive_id": "synthetic-primitive1",
            "error": "caught",
        }
    if fault == "contradictory":
        rows[0]["both_significant_topology_pass"] = True
    rp.write_text(json.dumps(rows))
    if fault == "missing_receipt":
        rp.unlink()
    core.finalize(
        normal_invocation_end=fault != "normal_false", business_receipt_path=rp
    )
    assert (
        not core.valid
        and not (core.root / "adapter_complete.json").exists()
        and not (core.root / "codec/observer_complete.json").exists()
    )


@pytest.mark.parametrize(
    "fault",
    [
        "finish",
        "status_close",
        "validation_close",
        "marker_close",
        "marker_race",
        "timeout",
    ],
)
def test_partial_sealing_and_publication_cannot_gain_outer_acceptance(
    tmp_path, monkeypatch, fault
):
    core = adapter.ObserverAdapter(
        tmp_path / "B", mode="hash_only", identity=identity()
    )
    emit(core.sink)
    rp = receipt(tmp_path / "r.json")
    original = Path.open
    link = contract.os.link

    class BadClose:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self.f

        def __exit__(self, *args):
            self.f.close()
            raise OSError("injected partial close")

    def altered(path, *a, **kw):
        f = original(path, *a, **kw)
        targets = {
            "status_close": "adapter_status.json.pending",
            "validation_close": "adapter_validation.json.pending",
            "marker_close": "adapter_complete.json.pending",
        }
        if a and a[0] == "xb" and path.name == targets.get(fault):
            return BadClose(f)
        return f

    def raced(source, target):
        if Path(target).name == "adapter_complete.json":
            Path(target).write_bytes(b"competing bytes")
        return link(source, target)

    if fault == "finish":
        monkeypatch.setattr(
            core.writer, "finish", mock.Mock(side_effect=OSError("finish"))
        )
    monkeypatch.setattr(Path, "open", altered)
    if fault == "marker_race":
        monkeypatch.setattr(contract.os, "link", raced)
    core.finalize(normal_invocation_end=True, business_receipt_path=rp)
    with pytest.raises((ValueError, OSError, TimeoutError)):
        validator.validate_capture(
            core.root,
            identity=identity(),
            business_receipt_path=rp,
            expected_mode="hash_only",
            publish=True,
            deadline_seconds=0 if fault == "timeout" else 30,
        )
    if fault == "marker_race":
        assert (core.root / "adapter_complete.json").read_bytes() == b"competing bytes"
    else:
        assert not (core.root / "adapter_complete.json").exists()
    # close failures never hide behind an inner Codec marker.
    with pytest.raises((ValueError, OSError, TimeoutError)):
        validator.validate_capture(
            core.root,
            identity=identity(),
            business_receipt_path=rp,
            expected_mode="hash_only",
            deadline_seconds=0 if fault == "timeout" else 30,
        )
    if fault == "finish":
        # Fault stub deliberately bypassed Codec cleanup; test owns these FDs.
        core.writer.bi.close()
        core.writer.ai.close()


@pytest.mark.parametrize(
    "fault",
    [
        "stale_receipt",
        "wrong_identity",
        "wrong_mode",
        "orphan_outer",
        "missing_raster",
        "forged_outer",
        "modified_status",
    ],
)
def test_external_validator_rejects_binding_and_inventory_failures(tmp_path, fault):
    core, rp = complete(tmp_path, "C", "full_capture")
    expected = identity()
    mode = "full_capture"
    if fault == "stale_receipt":
        rp.write_text(rp.read_text() + " ")
    if fault == "wrong_identity":
        expected["run_manifest_sha256"] = "2" * 64
    if fault == "wrong_mode":
        mode = "hash_only"
    if fault == "orphan_outer":
        (core.root / "orphan").write_text("extra")
    if fault == "missing_raster":
        next((core.root / "codec").glob("array_*.bin")).unlink()
    if fault == "forged_outer":
        (core.root / "adapter_complete.json").write_text("{}")
    if fault == "modified_status":
        p = core.root / "adapter_status.json"
        x = json.loads(p.read_text())
        x["sink_fault_free"] = False
        p.write_text(json.dumps(x))
    with pytest.raises((ValueError, OSError)):
        validator.validate_capture(
            core.root,
            identity=expected,
            business_receipt_path=rp,
            expected_mode=mode,
            publish=True,
        )


def test_constructor_collision_identity_and_resource_failures_do_not_overwrite(
    tmp_path,
):
    root = tmp_path / "existing"
    root.mkdir()
    p = root / "keep"
    p.write_text("original")
    core = adapter.ObserverAdapter(root, mode="hash_only", identity=identity())
    core.sink.close()
    core.finalize(normal_invocation_end=True, business_receipt_path=tmp_path / "none")
    assert (
        p.read_text() == "original"
        and len(list(root.iterdir())) == 1
        and not core.valid
    )
    invalid = adapter.ObserverAdapter(
        tmp_path / "invalid", mode="hash_only", identity={}
    )
    assert not invalid.valid and not (tmp_path / "invalid").exists()
    invalid = adapter.ObserverAdapter(
        tmp_path / "over",
        mode="hash_only",
        identity=identity(),
        total_limit_bytes=contract.TOTAL + 1,
    )
    assert not invalid.valid and not (tmp_path / "over").exists()


def test_defaults_do_not_activate_or_import_torch_and_codec_is_unchanged(tmp_path):
    assert (
        codec.file_hash("diagnostic_storage/guard_blocks_v2.py") == contract.CODEC_PIN
    )
    script = "import sys;from diagnostic_storage.observer_adapter_v2 import ObserverAdapter;assert 'torch' not in sys.modules;from phase3_mask_edit_refine.passive_observer import _ACTIVE,observe;assert _ACTIVE.get() is None;observe('disabled',value=object())"
    subprocess.run([sys.executable, "-c", script], check=True)
    assert not list(tmp_path.iterdir())


def test_actual_synthetic_executor_ownership_guard_and_replan_noninterference(
    tmp_path, monkeypatch
):
    from tests.test_passive_topology_observer import (
        _fixture,
        _compile,
        _planner_failure_run,
    )
    from scripts.review_passive_topology_observer import review

    assert review(Path(__file__).resolve().parents[1])["passed"]
    fixture = _fixture()
    before = fixture[0].copy()
    expected = _compile(fixture)
    rng = (random.getstate(), np.random.get_state())
    executor_cores = []
    for mode in ("hash_only", "full_capture"):
        core = adapter.ObserverAdapter(tmp_path / mode, mode=mode, identity=identity())
        assert _compile(fixture, core.sink) == expected
        assert np.array_equal(fixture[0], before) and random.getstate() == rng[0]
        assert (
            np.array_equal(np.random.get_state()[1], rng[1][1])
            and np.random.get_state()[2:] == rng[1][2:]
        )
        core.finalize(
            normal_invocation_end=False, business_receipt_path=tmp_path / "none"
        )
        executor_cores.append(core)
    from itertools import zip_longest

    count = 0
    for a, b in zip_longest(
        codec.Reader(executor_cores[0].root / "codec").iter_events(
            require_business_complete=False
        ),
        codec.Reader(executor_cores[1].root / "codec").iter_events(
            require_business_complete=False
        ),
    ):
        assert a == b
        count += 1
    assert count > 0
    baseline = _planner_failure_run()
    replan_cores = []
    for mode in ("hash_only", "full_capture"):
        observed = adapter.ObserverAdapter(
            tmp_path / ("replan_" + mode), mode=mode, identity=identity()
        )
        assert _planner_failure_run(observed.sink) == baseline
        observed.finalize(
            normal_invocation_end=False, business_receipt_path=tmp_path / "none"
        )
        replan_cores.append(observed)
    for a, b in zip_longest(
        codec.Reader(replan_cores[0].root / "codec").iter_events(
            require_business_complete=False
        ),
        codec.Reader(replan_cores[1].root / "codec").iter_events(
            require_business_complete=False
        ),
    ):
        assert a == b
    core = adapter.ObserverAdapter(
        tmp_path / "replan", mode="hash_only", identity=identity()
    )
    monkeypatch.setattr(
        core, "record", mock.Mock(side_effect=RuntimeError("record failed"))
    )
    monkeypatch.setattr(
        core, "invalidate", mock.Mock(side_effect=RuntimeError("handler failed"))
    )
    assert _planner_failure_run(core.sink) == baseline and len(baseline[0]) == 3
    core.finalize(normal_invocation_end=False, business_receipt_path=tmp_path / "none")
    assert core.sink.failed and not (core.root / "adapter_complete.json").exists()


def test_bounded_controls_reject_before_serialization_and_arrays_reject_before_copy(
    tmp_path, monkeypatch
):
    with mock.patch.object(
        contract.codec, "dump", side_effect=AssertionError("must precheck")
    ):
        with pytest.raises(contract.AdapterError):
            contract.bounded_json({"huge": "x" * contract.CONTROL})
    core = adapter.ObserverAdapter(
        tmp_path / "B", mode="full_capture", identity=identity()
    )
    spy = mock.Mock(side_effect=AssertionError("must not copy"))
    monkeypatch.setattr(core.writer, "_copy_array", spy)
    with legacy.observation_session(core.sink):
        legacy.observe(
            "large",
            array=np.broadcast_to(np.zeros(1, dtype="u1"), (codec.MAX_ARRAY + 1,)),
        )
    spy.assert_not_called()
    assert core.sink.failed
    core.finalize(normal_invocation_end=False, business_receipt_path=tmp_path / "none")


def test_ported_encoder_ast_has_only_array_delegation_difference():
    import inspect
    import textwrap

    old = ast.parse(textwrap.dedent(inspect.getsource(legacy.PassiveObserver._encode)))
    new = ast.parse(textwrap.dedent(inspect.getsource(adapter.ObserverAdapter._encode)))
    for tree in (old, new):
        branch = next(
            n
            for n in tree.body[0].body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "type(value) is np.ndarray"
        )
        branch.body = [ast.Pass()]
    assert ast.dump(old, include_attributes=False) == ast.dump(
        new, include_attributes=False
    )
    for name in ("_reserve_metadata", "_rng"):
        a = ast.parse(
            textwrap.dedent(inspect.getsource(getattr(legacy.PassiveObserver, name)))
        )
        b = ast.parse(
            textwrap.dedent(inspect.getsource(getattr(adapter.ObserverAdapter, name)))
        )
        assert ast.dump(a, include_attributes=False) == ast.dump(
            b, include_attributes=False
        )


@pytest.mark.parametrize("field", ["case_primitive", "failure_reasons", "witness"])
def test_terminal_event_and_receipt_must_correspond(tmp_path, field):
    core = adapter.ObserverAdapter(
        tmp_path / "B", mode="hash_only", identity=identity()
    )
    with legacy.observation_session(core.sink):
        for name, values in events():
            if field == "case_primitive" and name == "qualification.begin":
                values["case"].primitive_id = "other"
            core.sink.record(name, values)
    rp = receipt(tmp_path / "r.json")
    rows = json.loads(rp.read_text())
    if field == "failure_reasons":
        rows[0]["checks"][0]["failure_reasons"] = ["different-negative"]
    if field == "witness":
        rows[0]["checks"][0]["metrics"]["whole_mask_topology_witness"]["errors"] = [
            "different-error"
        ]
    rp.write_text(json.dumps(rows))
    core.finalize(normal_invocation_end=True, business_receipt_path=rp)
    assert core.valid and (core.root / "codec/observer_complete.json").exists()
    with pytest.raises(contract.AdapterError, match="binding|payload"):
        validator.validate_capture(
            core.root,
            identity=identity(),
            business_receipt_path=rp,
            expected_mode="hash_only",
            publish=True,
        )
    assert not (core.root / "adapter_complete.json").exists()


@pytest.mark.parametrize(
    "value", [np.zeros(1, dtype=object), np.zeros(1, dtype=[("x", "i4")]), object()]
)
def test_unsupported_types_fail_without_coercion_or_iteration(tmp_path, value):
    core = adapter.ObserverAdapter(
        tmp_path / "B", mode="full_capture", identity=identity()
    )
    with legacy.observation_session(core.sink):
        legacy.observe("unsupported", value=value)
    assert core.sink.failed and not core.valid
    core.finalize(normal_invocation_end=False, business_receipt_path=tmp_path / "none")
    assert not (core.root / "adapter_complete.json").exists()


def test_reference_identity_snapshot_and_finalize_are_not_resume(tmp_path):
    ident = identity()
    core = adapter.ObserverAdapter(tmp_path / "B", mode="hash_only", identity=ident)
    ident["operations"][0]["case_id"] = "changed-after-creation"
    emit(core.sink)
    rp = receipt(tmp_path / "r.json")
    first = core.finalize(normal_invocation_end=True, business_receipt_path=rp)
    before = (core.root / "adapter_status.json").read_bytes()
    assert core.finalize(normal_invocation_end=True, business_receipt_path=rp) == first
    assert (core.root / "adapter_status.json").read_bytes() == before
    validator.validate_capture(
        core.root,
        identity=identity(),
        business_receipt_path=rp,
        expected_mode="hash_only",
    )


def test_interrupted_process_never_has_outer_acceptance(tmp_path):
    ident = identity()
    root = tmp_path / "killed"
    code = (
        "from diagnostic_storage.observer_adapter_v2 import ObserverAdapter;import os;core=ObserverAdapter("
        + repr(str(root))
        + ',mode="hash_only",identity='
        + repr(ident)
        + ');core.sink.record("guard.stage",{"row":0});os._exit(9)'
    )
    result = subprocess.run([sys.executable, "-c", code])
    assert result.returncode == 9 and not (root / "adapter_complete.json").exists()
    with pytest.raises((ValueError, OSError)):
        validator.validate_capture(
            root,
            identity=ident,
            business_receipt_path=tmp_path / "none",
            expected_mode="hash_only",
            publish=True,
        )


def test_full_business_receipt_comparison_has_no_extra_field_exclusions(tmp_path):
    captures = []
    for mode, name in [("hash_only", "B"), ("full_capture", "C")]:
        core = adapter.ObserverAdapter(tmp_path / name, mode=mode, identity=identity())
        emit(core.sink)
        rp = receipt(tmp_path / (name + ".json"))
        if name == "C":
            rows = json.loads(rp.read_text())
            rows[0]["auxiliary_field"] = {"do_not_drop": 1}
            rp.write_text(json.dumps(rows))
        core.finalize(normal_invocation_end=True, business_receipt_path=rp)
        captures.append((core, rp))
    with pytest.raises(contract.AdapterError, match="full business receipt mismatch"):
        validator.compare_captures(
            captures[0][0].root,
            captures[1][0].root,
            identity=identity(),
            left_receipt=captures[0][1],
            right_receipt=captures[1][1],
        )


def test_pressure_report_counts_index_hardlinks_once(tmp_path):
    from scripts.stress_observer_adapter_v1 import allocated

    core, rp = complete(tmp_path)
    expected = sum(
        p.stat().st_blocks * 512
        for directory in (core.root, core.root / "codec")
        for p in directory.iterdir()
        if p.is_file() and not p.name.endswith(".pending")
    )
    assert allocated(core.root)["unique_file_allocated_bytes"] == expected
    assert (core.root / "codec/arrays.index.pending").stat().st_ino == (
        core.root / "codec/arrays.index.jsonl"
    ).stat().st_ino


def test_comparison_uses_one_reader_pass_per_capture_and_never_cached_proofs(
    tmp_path, monkeypatch
):
    b, br = complete(tmp_path, "B")
    c, cr = complete(tmp_path, "C", "full_capture")
    calls = []
    original = codec.Reader.iter_events

    def counted(reader, *args, **kwargs):
        calls.append(str(reader.root))
        yield from original(reader, *args, **kwargs)

    monkeypatch.setattr(codec.Reader, "iter_events", counted)
    monkeypatch.setattr(
        codec,
        "compare",
        mock.Mock(side_effect=AssertionError("must not re-read streams")),
    )
    result = validator.compare_captures(
        b.root, c.root, identity=identity(), left_receipt=br, right_receipt=cr
    )
    assert (
        result["status"] == "semantic_match"
        and len(calls) == 2
        and len(set(calls)) == 2
    )


def test_onepass_comparison_rechecks_arrays_even_after_prior_acceptance(tmp_path):
    b, br = complete(tmp_path, "B")
    c, cr = complete(tmp_path, "C", "full_capture")
    validator.validate_capture(
        c.root,
        identity=identity(),
        business_receipt_path=cr,
        expected_mode="full_capture",
        publish=True,
    )
    path = next((c.root / "codec").glob("array_*.bin"))
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"X")
    with pytest.raises(ValueError):
        validator.compare_captures(
            b.root, c.root, identity=identity(), left_receipt=br, right_receipt=cr
        )


def test_receipt_change_during_stream_cannot_pass_final_binding(tmp_path, monkeypatch):
    core, rp = complete(tmp_path)
    original = codec.Reader.iter_events

    def mutate(reader, *args, **kwargs):
        for i, line in enumerate(original(reader, *args, **kwargs)):
            if i == 1:
                rp.write_text(rp.read_text() + " ")
            yield line

    monkeypatch.setattr(codec.Reader, "iter_events", mutate)
    with pytest.raises(
        contract.AdapterError, match="receipt changed during validation"
    ):
        validator.validate_capture(
            core.root,
            identity=identity(),
            business_receipt_path=rp,
            expected_mode="hash_only",
            publish=True,
        )
    assert not (core.root / "adapter_complete.json").exists()


def test_onepass_comparison_preserves_unknown_fields_and_midstream_deadline(
    tmp_path, monkeypatch
):
    b, br = complete(tmp_path, "B", values={"extra": "left"})
    c, cr = complete(tmp_path, "C", "full_capture", values={"extra": "right"})
    with pytest.raises(contract.AdapterError, match="logical event differs"):
        validator.compare_captures(
            b.root, c.root, identity=identity(), left_receipt=br, right_receipt=cr
        )
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 1000 if calls > 6 else 0

    monkeypatch.setattr(validator.time, "monotonic", clock)
    with pytest.raises(TimeoutError):
        validator.compare_captures(
            b.root, c.root, identity=identity(), left_receipt=br, right_receipt=cr
        )
