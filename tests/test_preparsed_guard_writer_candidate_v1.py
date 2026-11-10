"""Standalone differential checks; no pathology business module is imported."""
import json

import numpy as np
import pytest

from diagnostic_storage import guard_blocks_v2 as codec
from diagnostic_storage.preparsed_guard_writer_candidate_v1 import (
    PreparsedWriterCandidate,
)


def _line(record):
    return (
        json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode()


@pytest.mark.parametrize("mode", ["hash_only", "full_capture"])
def test_same_bytes_and_reader_result_as_existing_writer(tmp_path, mode):
    original = codec.Writer(tmp_path / "original", mode=mode, target_block_bytes=256)
    candidate = PreparsedWriterCandidate(
        tmp_path / "candidate", mode=mode, target_block_bytes=256
    )
    raw = []
    names = ["qualification.begin"] + ["guard.stage"] * 40 + [
        "qualification.complete",
        "qualification.begin",
        "guard.accept",
        "qualification.complete",
    ]
    for i, name in enumerate(names):
        record = {
            "sequence": i,
            "event": name,
            "values": {"mapping": [["row", i], ["text", "组织\\n"]]},
        }
        if name in ("qualification.begin", "qualification.complete"):
            record["operation_id"] = "U1" if i < 42 else "U2"
        if i == 20:
            data = np.arange(12, dtype=">i4").reshape(3, 4)
            token_a = original.add_array(data)
            token_b = candidate.add_array(data)
            assert token_a == token_b
            record["snapshot"] = token_a
        line = _line(record)
        original.write_event(line)
        candidate.write_record(record)
        raw.append(line)
    receipt = {"normal_end": True, "operation_ids": ["U1", "U2"]}
    left = original.finish(normal_end=True, receipt=receipt)
    right = candidate.finish(normal_end=True, receipt=receipt)
    assert left["business_process_complete"] is True
    assert right["business_process_complete"] is True
    assert left["stream_sha256"] == right["stream_sha256"]
    assert b"".join(codec.Reader(tmp_path / "candidate").iter_events()) == b"".join(raw)
    assert codec.compare(tmp_path / "original", tmp_path / "candidate")["strict_match"]


@pytest.mark.parametrize(
    "record",
    [
        {"sequence": 1, "event": "guard.stage"},
        {"sequence": 0, "event": "guard.stage", "bad": float("nan")},
        {"sequence": 0, "event": "guard.stage", "snapshot": {"array": {}}},
    ],
)
def test_invalid_record_faults_writer_without_completion(tmp_path, record):
    writer = PreparsedWriterCandidate(tmp_path / "invalid")
    with pytest.raises((ValueError, TypeError)):
        writer.write_record(record)
    assert writer.faults
    assert not (tmp_path / "invalid/observer_complete.json").exists()
