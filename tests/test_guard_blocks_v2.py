"""Standalone codec tests only; no pathology module or source invocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
from diagnostic_storage import guard_blocks_v2 as codec


def line(i, event="guard.stage", **fields):
    # Alternating whitespace/Unicode are retained byte-for-byte, not normalized.
    return (
        json.dumps(
            {"sequence": i, "event": event, **fields},
            ensure_ascii=i % 2 == 0,
            separators=(",", ":") if i % 2 == 0 else (", ", ": "),
        )
        + "\n"
    ).encode()


def write_complete(path, *, mode="hash_only", arrays=False, target=200):
    w = codec.Writer(path, mode=mode, target_block_bytes=target)
    data = []
    entries = [
        ("qualification.begin", "U1"),
        ("guard.stage", None),
        ("qualification.complete", "U1"),
        ("qualification.begin", "U2"),
        ("guard.reject", None),
        ("qualification.complete", "U2"),
    ]
    for i, (kind, op) in enumerate(entries):
        fields = {
            "payload": {
                "unicode": "不丢字段\\n",
                "nonfinite": {"tag": "+inf"},
                "ordered": [1, True, None],
            }
        }
        if op:
            fields["operation_id"] = op
        if arrays and i in (1, 4):
            fields["snapshot"] = w.add_array(np.arange(15, dtype=">i4").reshape(3, 5))
        raw = line(i, kind, **fields)
        data.append(raw)
        w.write_event(raw)
    summary = w.finish(
        normal_end=True,
        receipt={
            "normal_end": True,
            "operation_ids": ["U1", "U2"],
            "payload": {"arbitrary": "retained"},
        },
    )
    return w, b"".join(data), summary


def test_byte_exact_roundtrip_streaming_and_modes(tmp_path):
    _, original, s = write_complete(tmp_path / "B", arrays=True)
    _, other, _ = write_complete(tmp_path / "C", mode="full_capture", arrays=True)
    assert original == other and s["events"] == 6
    reader = codec.Reader(tmp_path / "C")
    iterator = reader.iter_events()
    assert iter(iterator) is iterator and reader.last_validation is None
    assert b"".join(iterator) == original
    assert reader.last_validation["business_process_complete"]
    assert codec.compare(tmp_path / "B", tmp_path / "C")["strict_match"]


def test_no_whole_file_read_and_no_event_object_list(tmp_path, monkeypatch):
    _, original, _ = write_complete(tmp_path / "x", mode="full_capture", arrays=True)
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("whole file read"))
    assert b"".join(codec.Reader(tmp_path / "x").iter_events()) == original


def test_complete_blocks_do_not_upgrade_missing_business_tail(tmp_path):
    w = codec.Writer(tmp_path / "prefix")
    raw = line(0, "qualification.begin", operation_id="U1")
    w.write_event(raw)
    s = w.finish(
        normal_end=True, receipt={"normal_end": True, "operation_ids": ["U1", "U2"]}
    )
    assert (
        s["block_integrity_complete"] is True
        and s["business_process_complete"] is False
    )
    assert not (tmp_path / "prefix/observer_complete.json").exists()
    with pytest.raises(codec.CodecError):
        codec.Reader(tmp_path / "prefix").validate()
    result = codec.Reader(tmp_path / "prefix").validate(require_business_complete=False)
    assert (
        result["block_integrity_complete"] and not result["business_process_complete"]
    )


@pytest.mark.parametrize(
    "case",
    [
        "sequence",
        "duplicate_key",
        "raw_nonfinite",
        "unterminated",
        "array_without_registration",
        "operation_order",
    ],
)
def test_invalid_events_fail_closed(tmp_path, case):
    w = codec.Writer(tmp_path / "x")
    raw = {
        "sequence": line(1),
        "duplicate_key": b'{"sequence":0,"sequence":1,"event":"x"}\n',
        "raw_nonfinite": b'{"sequence":0,"event":"x","value":NaN}\n',
        "unterminated": b'{"sequence":0,"event":"x"}',
        "array_without_registration": line(
            0, snapshot={"array": {"dtype": "|u1", "shape": [1], "content_sha256": "x"}}
        ),
        "operation_order": line(0, "qualification.begin", operation_id="U2"),
    }[case]
    with pytest.raises((codec.CodecError, ValueError)):
        w.write_event(raw)
    assert w.faults
    w.finish(normal_end=False, receipt={})
    assert not (w.root / "observer_complete.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "payload",
        "header",
        "index_missing",
        "duplicate_entry",
        "orphan",
        "symlink",
        "missing_array",
        "duplicate_array",
        "completion",
        "partial_index",
        "trailing_compressed",
    ],
)
def test_storage_corruption_not_accepted(tmp_path, mutation):
    root = tmp_path / "x"
    write_complete(root, mode="full_capture", arrays=True)
    block = root / "block_000000000000.bin"
    if mutation in ("payload", "header", "trailing_compressed"):
        with block.open("r+b") as f:
            if mutation == "trailing_compressed":
                f.seek(0, 2)
                f.write(b"extra")
            else:
                f.seek(
                    -1 if mutation == "payload" else 1,
                    2 if mutation == "payload" else 0,
                )
                f.write(b"X")
    elif mutation == "index_missing":
        (root / "blocks.index.jsonl").unlink()
    elif mutation == "duplicate_entry":
        p = root / "blocks.index.jsonl"
        with p.open("ab") as f:
            f.write(p.open("rb").readline())
    elif mutation == "orphan":
        (root / "block_999999999999.bin").write_text("orphan")
    elif mutation == "symlink":
        block.unlink()
        block.symlink_to(root / "block_000000000001.bin")
    elif mutation == "missing_array":
        (root / "array_000000000000.bin").unlink()
    elif mutation == "duplicate_array":
        p = root / "arrays.index.jsonl"
        with p.open("ab") as f:
            f.write(p.open("rb").readline())
    elif mutation == "completion":
        (root / "observer_complete.json").write_text("{}")
    else:
        with (root / "blocks.index.jsonl").open("ab") as f:
            f.write(b"{partial")
    with pytest.raises((codec.CodecError, FileNotFoundError, ValueError)):
        codec.Reader(root).validate()


@pytest.mark.parametrize(
    "fault",
    [
        "write",
        "enospc",
        "block_close",
        "index_close",
        "summary_close",
        "marker_close",
        "publish_race",
    ],
)
def test_faults_cannot_publish_success(tmp_path, monkeypatch, fault):
    root = tmp_path / "x"
    w = codec.Writer(root, target_block_bytes=200)
    original_open = Path.open
    original_link = codec.os.link

    class FailClose:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, *args):
            self.stream.close()
            raise OSError("injected after bytes written")

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, *args):
            raise OSError(28, "injected ENOSPC")

    def altered(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if args and args[0] == "xb":
            if fault in ("write", "enospc") and path.name.startswith("block_"):
                return FailingStream(stream)
            if fault == "block_close" and path.name.startswith("block_"):
                return FailClose(stream)
            if fault == "summary_close" and path.name == "summary.json.pending":
                return FailClose(stream)
            if (
                fault == "marker_close"
                and path.name == "observer_complete.json.pending"
            ):
                return FailClose(stream)
        return stream

    def raced(source, target):
        if Path(target).name.startswith("block_"):
            with original_open(Path(target), "xb") as f:
                f.write(b"competing publisher")
        return original_link(source, target)

    with monkeypatch.context() as m:
        m.setattr(Path, "open", altered)
        if fault == "publish_race":
            m.setattr(codec.os, "link", raced)
        if fault == "index_close":

            class BadIndex:
                def __init__(self, inner):
                    self.inner = inner

                def __getattr__(self, name):
                    return getattr(self.inner, name)

                def close(self):
                    self.inner.close()
                    raise OSError("index close")

            w.bi = BadIndex(w.bi)
        try:
            for i, (kind, op) in enumerate(
                [
                    ("qualification.begin", "U1"),
                    ("qualification.complete", "U1"),
                    ("qualification.begin", "U2"),
                    ("qualification.complete", "U2"),
                ]
            ):
                w.write_event(line(i, kind, operation_id=op))
            w.finish(
                normal_end=True,
                receipt={"normal_end": True, "operation_ids": ["U1", "U2"]},
            )
        except OSError:
            pass
    assert w.faults and not (root / "observer_complete.json").exists()
    with pytest.raises((codec.CodecError, FileNotFoundError)):
        codec.Reader(root).validate()
    for stream in (w.bi, w.ai):
        try:
            stream.close()
        except OSError:
            pass


def test_pre_copy_array_budget_and_detached_bytes(tmp_path, monkeypatch):
    w = codec.Writer(
        tmp_path / "limited",
        mode="full_capture",
        max_storage_bytes=262144,
        max_array_bytes=8,
    )
    spy = mock.Mock(side_effect=AssertionError("must not copy"))
    monkeypatch.setattr(w, "_copy_array", spy)
    with pytest.raises(codec.BudgetError):
        w.add_array(np.broadcast_to(np.zeros(1, dtype=np.uint8), (100, 100)))
    spy.assert_not_called()
    w.finish(normal_end=False, receipt={})
    w = codec.Writer(tmp_path / "copy", mode="full_capture")
    source = np.arange(6, dtype=">i4")
    original = source.copy()
    w.write_event(line(0, "qualification.begin", operation_id="U1"))
    token = w.add_array(source)
    source[:] = 0
    w.write_event(line(1, snapshot=token))
    w.write_event(line(2, "qualification.complete", operation_id="U1"))
    w.write_event(line(3, "qualification.begin", operation_id="U2"))
    w.write_event(line(4, "qualification.complete", operation_id="U2"))
    w.finish(
        normal_end=True, receipt={"normal_end": True, "operation_ids": ["U1", "U2"]}
    )
    assert codec.Reader(w.root).validate()["array_occurrences"] == 1
    assert (
        token["array"]["content_sha256"]
        == hashlib.sha256(original.tobytes()).hexdigest()
    )


def test_unified_storage_budget_counts_arrays_and_controls(tmp_path):
    w = codec.Writer(tmp_path / "x", mode="full_capture", max_storage_bytes=300000)
    w.write_event(line(0, "qualification.begin", operation_id="U1"))
    with pytest.raises(codec.BudgetError):
        w.add_array(np.zeros(300000, dtype=np.uint8))
    assert w.faults and not (w.root / "observer_complete.json").exists()
    w.finish(normal_end=False, receipt={})


def test_process_interruption_has_no_complete_marker(tmp_path):
    root = tmp_path / "killed"
    code = (
        "from diagnostic_storage.guard_blocks_v2 import Writer;import os;w=Writer("
        + repr(str(root))
        + ');w.write_event(b\'{"sequence":0,"event":"guard.stage"}\\n\');os._exit(9)'
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[1])
    )
    assert result.returncode == 9 and not (root / "observer_complete.json").exists()
    with pytest.raises(FileNotFoundError):
        codec.Reader(root).validate()


def test_partial_iterator_does_not_claim_validation(tmp_path):
    write_complete(tmp_path / "x")
    reader = codec.Reader(tmp_path / "x")
    events = reader.iter_events()
    next(events)
    events.close()
    assert reader.last_validation is None


def resign(root):
    """Test-only self-consistent altered metadata, to exercise semantic checks."""
    summary = json.loads((root / "summary.json").read_text())
    summary["block_index_sha256"] = codec.file_hash(root / "blocks.index.jsonl")
    (root / "summary.json").write_bytes(codec.dump(summary) + b"\n")
    (root / "observer_complete.json").write_bytes(
        codec.dump({"summary_sha256": codec.file_hash(root / "summary.json")}) + b"\n"
    )


def test_incomplete_business_cannot_be_promoted_by_valid_looking_footer(tmp_path):
    root = tmp_path / "prefix"
    w = codec.Writer(root)
    w.write_event(line(0, "qualification.begin", operation_id="U1"))
    w.finish(
        normal_end=True, receipt={"normal_end": True, "operation_ids": ["U1", "U2"]}
    )
    summary = json.loads((root / "summary.json").read_text())
    summary["business_process_complete"] = True
    (root / "summary.json").write_bytes(codec.dump(summary) + b"\n")
    pending = root / "observer_complete.json.pending"
    pending.write_bytes(
        codec.dump({"summary_sha256": codec.file_hash(root / "summary.json")}) + b"\n"
    )
    os.link(pending, root / "observer_complete.json")
    with pytest.raises(codec.CodecError, match="missing business boundaries"):
        codec.Reader(root).validate()


@pytest.mark.parametrize("mutation", ["oversized", "bounded_expansion"])
def test_self_consistent_block_cannot_bypass_decompression_limits(tmp_path, mutation):
    root = tmp_path / "x"
    write_complete(root, target=4096)
    path = root / "block_000000000000.bin"
    header, _, offset = codec._file_header(root, path.name, path.name)
    payload = path.read_bytes()[offset:]
    header["decoded_bytes"] = codec.MAX_RECORD + 1 if mutation == "oversized" else 1
    h = codec.dump(header)
    new = codec.MAGIC + codec.struct.pack(">I", len(h)) + h + payload
    path.write_bytes(new)
    entry = json.loads((root / "blocks.index.jsonl").read_text())
    entry["header_sha256"] = codec.digest(h)
    entry["file_sha256"] = codec.digest(new)
    (root / "blocks.index.jsonl").write_bytes(codec.dump(entry) + b"\n")
    resign(root)
    with pytest.raises(codec.CodecError, match="bounded"):
        codec.Reader(root).validate()


@pytest.mark.parametrize("dtype", ["|b1", ">i4", ">f8", "<c16", "<U2"])
def test_array_dtype_endian_shape_and_bits_preserved(tmp_path, dtype):
    source = np.array([[0, 1], [2, 3]], dtype=dtype)
    root = tmp_path / "x"
    w = codec.Writer(root, mode="full_capture")
    w.write_event(line(0, "qualification.begin", operation_id="U1"))
    token = w.add_array(source[:, ::-1])
    expected = np.array(source[:, ::-1], copy=True, order="C")
    w.write_event(line(1, snapshot=token))
    w.write_event(line(2, "qualification.complete", operation_id="U1"))
    w.write_event(line(3, "qualification.begin", operation_id="U2"))
    w.write_event(line(4, "qualification.complete", operation_id="U2"))
    w.finish(
        normal_end=True, receipt={"normal_end": True, "operation_ids": ["U1", "U2"]}
    )
    assert codec.Reader(root).validate()["array_occurrences"] == 1
    path = root / "array_000000000000.bin"
    header, _, offset = codec._file_header(root, path.name, path.name)
    assert header["descriptor"]["dtype"] == expected.dtype.str and header["descriptor"][
        "shape"
    ] == [2, 2]
    assert path.read_bytes()[offset:] == expected.tobytes()


def test_strict_comparison_rejects_unknown_field_difference(tmp_path):
    write_complete(tmp_path / "a")
    w = codec.Writer(tmp_path / "b")
    for i, (kind, op) in enumerate(
        [
            ("qualification.begin", "U1"),
            ("qualification.complete", "U1"),
            ("qualification.begin", "U2"),
            ("qualification.complete", "U2"),
        ]
    ):
        w.write_event(line(i, kind, operation_id=op, extra="retained"))
    w.finish(
        normal_end=True, receipt={"normal_end": True, "operation_ids": ["U1", "U2"]}
    )
    with pytest.raises(codec.CodecError):
        codec.compare(tmp_path / "a", tmp_path / "b")


@pytest.mark.parametrize(
    "limits",
    [
        {"max_array_bytes": codec.MAX_ARRAY + 1},
        {"max_array_bytes": 0},
        {"max_array_occurrences": 100001},
        {"max_array_occurrences": True},
    ],
)
def test_caller_cannot_expand_frozen_array_limits(tmp_path, limits):
    with pytest.raises(codec.CodecError, match="array resource limits"):
        codec.Writer(tmp_path / "x", **limits)
    assert not (tmp_path / "x").exists()


def test_operation_summary_must_match_actual_events(tmp_path):
    root = tmp_path / "x"
    write_complete(root)
    summary = json.loads((root / "summary.json").read_text())
    summary["completed_operations"] = []
    (root / "summary.json").write_bytes(codec.dump(summary) + b"\n")
    resign(root)
    with pytest.raises(codec.CodecError, match="operation summary mismatch"):
        codec.Reader(root).validate()


def test_completion_race_preserves_competing_bytes(tmp_path, monkeypatch):
    original = codec.os.link
    foreign = b"concurrent publisher, never overwrite"

    def race(source, target):
        if Path(target).name == "observer_complete.json":
            Path(target).write_bytes(foreign)
        original(source, target)

    monkeypatch.setattr(codec.os, "link", race)
    with pytest.raises(FileExistsError):
        write_complete(tmp_path / "x")
    assert (tmp_path / "x/observer_complete.json").read_bytes() == foreign
    with pytest.raises(codec.CodecError):
        codec.Reader(tmp_path / "x").validate()


def test_constructor_closes_first_index_if_second_open_fails(tmp_path, monkeypatch):
    original = Path.open
    opened = []

    def fail_second(path, *args, **kwargs):
        if path.name == "arrays.index.pending":
            raise OSError("index initialization fault")
        stream = original(path, *args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", fail_second)
    with pytest.raises(OSError):
        codec.Writer(tmp_path / "x")
    assert opened and all(stream.closed for stream in opened)


def test_real_64mib_logical_array_ceiling_is_checked_before_copy(tmp_path, monkeypatch):
    writer = codec.Writer(tmp_path / "x", mode="full_capture")
    source = np.broadcast_to(np.zeros(1, dtype="u1"), (codec.MAX_ARRAY + 1,))
    copy = mock.Mock(side_effect=AssertionError("oversized view must not be copied"))
    monkeypatch.setattr(writer, "_copy_array", copy)
    with pytest.raises(codec.BudgetError):
        writer.add_array(source)
    copy.assert_not_called()
    writer.finish(normal_end=False, receipt={})


def test_large_single_record_gets_dedicated_bounded_block(tmp_path):
    writer = codec.Writer(tmp_path / "x")
    raws = [
        line(0, "qualification.begin", operation_id="U1"),
        line(1, payload="x" * (codec.TARGET + 1)),
        line(2, "qualification.complete", operation_id="U1"),
        line(3, "qualification.begin", operation_id="U2"),
        line(4, "qualification.complete", operation_id="U2"),
    ]
    for raw in raws:
        writer.write_event(raw)
    summary = writer.finish(
        normal_end=True, receipt={"normal_end": True, "operation_ids": ["U1", "U2"]}
    )
    assert summary["blocks"] == 3
    assert list(codec.Reader(writer.root).iter_events()) == raws


def test_raw_nonfinite_array_bits_are_not_canonicalized(tmp_path):
    writer = codec.Writer(tmp_path / "x", mode="full_capture")
    source = np.array(
        [0x7FF8000000000001, 0x7FF8000000000042, 0x8000000000000000], dtype="u8"
    ).view("f8")
    writer.write_event(line(0, "qualification.begin", operation_id="U1"))
    token = writer.add_array(source)
    writer.write_event(line(1, snapshot=token))
    writer.write_event(line(2, "qualification.complete", operation_id="U1"))
    writer.write_event(line(3, "qualification.begin", operation_id="U2"))
    writer.write_event(line(4, "qualification.complete", operation_id="U2"))
    writer.finish(
        normal_end=True, receipt={"normal_end": True, "operation_ids": ["U1", "U2"]}
    )
    codec.Reader(writer.root).validate()
    assert (
        token["array"]["content_sha256"] == hashlib.sha256(source.tobytes()).hexdigest()
    )


def test_index_preallocation_is_not_forgotten_on_next_append(tmp_path):
    budget = codec.Budget(8 * 1024**2)
    path = tmp_path / "arrays.index.pending"
    budget.claim(path, 100000)
    # Emulate an observed filesystem extent larger than current logical length.
    budget.total += 500000 - budget.charges[str(path)]
    budget.charges[str(path)] = 500000
    budget.claim_many([(path, 100001)])
    assert budget.charges[str(path)] == 500000
    assert (
        budget.reservation(tmp_path / "blocks.index.pending", 100000)
        >= 2 * 100000 + 65536
    )


def test_budget_exhaustion_preserves_room_for_invalid_summary(tmp_path):
    writer = codec.Writer(
        tmp_path / "x", mode="full_capture", max_storage_bytes=1024**2
    )
    writer.write_event(line(0, "qualification.begin", operation_id="U1"))
    copies = mock.Mock(wraps=writer._copy_array)
    writer._copy_array = copies
    while True:
        before = copies.call_count
        try:
            token = writer.add_array(np.zeros(16384, dtype="u1"))
        except codec.BudgetError:
            assert copies.call_count == before
            break
        writer.write_event(line(writer.events, snapshot=token))
    summary = writer.finish(normal_end=False, receipt={"normal_end": False})
    assert summary["faults"] and not summary["business_process_complete"]
    assert (writer.root / "summary.json").exists()
    assert not (writer.root / "observer_complete.json").exists()
    with pytest.raises(codec.CodecError):
        codec.Reader(writer.root).validate()
