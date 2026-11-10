import hashlib
import json

import pytest

from diagnostic_storage.guard_blocks_v2 import Writer
from scripts.analyze_s07_prefix_attempts_v1 import analyze


def _write(writer, event, values):
    writer.write_event(
        (
            json.dumps(
                {"sequence": writer.events, "event": event, "values": {"mapping": list(values.items())}},
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )


def test_summarizes_only_committed_prefix_and_refuses_changed_index(tmp_path):
    root = tmp_path / "codec"
    writer = Writer(root, expected_operations=["U1", "U2"], target_block_bytes=512)
    _write(writer, "qualification.begin", {"case": {"fields": [["case_id", "U1"]]}})
    _write(writer, "compiler.attempt", {"attempt": 1})
    _write(writer, "front.contribution", {
        "index": 0,
        "pass_index": 0,
        "requested": 100,
        "audit": {"fields": [
            ["realized_pixels", 70],
            ["rejected_source_connectivity", 3],
            ["rejected_source_hole_change", 2],
            ["rejected_target_hole_change", 1],
            ["rejected_target_island", 0],
            ["rejected_unselected_target_contact", 0],
            ["rejected_source_retention", 0],
        ]},
    })
    topology = {"mapping": [["passed", True], ["source_components_before", 1], ["source_components_after", 1], ["source_holes_before", 0], ["source_holes_after", 0], ["target_components_before", 1], ["target_components_after", 1], ["target_holes_before", 0], ["target_holes_after", 0], ["changed_component_area_passed", True]]}
    _write(writer, "probe.complete", {"total": 100, "realized": 70, "hard_min_pixels": 100, "topology": topology})
    _write(writer, "probe.complete", {"total": 100, "realized": 110, "hard_min_pixels": 100, "topology": {"mapping": [[k, False if k == "passed" else v] for k, v in topology["mapping"]]}})
    _write(writer, "compiler.exception", {"exception": {"exception_type": "Underfill"}})
    _write(writer, "compiler.attempt", {"attempt": 2})
    writer._flush()  # Simulate a process that stops after a committed block.
    index = root / "blocks.index.pending"
    sha = hashlib.sha256(index.read_bytes()).hexdigest()
    result = analyze(root, sha)
    assert result["committed_events"] == 7
    assert result["attempts"][0]["max_topology_passing_realized_pixels"] == 70
    assert result["attempts"][0]["max_topology_failing_realized_pixels"] == 110
    assert result["attempts"][0]["probe_topology"][0]["source_holes_after"] == 0
    assert result["attempts"][0]["probe_topology"][0]["front_contributions"] == [{
        "work_index": 0,
        "pass_index": 0,
        "requested_pixels": 100,
        "realized_pixels": 70,
        "rejected_source_connectivity": 3,
        "rejected_source_hole_change": 2,
        "rejected_target_hole_change": 1,
        "rejected_target_island": 0,
        "rejected_unselected_target_contact": 0,
        "rejected_source_retention": 0,
    }]
    assert result["attempts"][0]["exception_type"] == "Underfill"
    assert len(result["attempts"][0]["probe_values_sha256"]) == 2
    assert len(result["attempts"][0]["probe_rng_sha256"]) == 2
    assert result["attempts"][0]["attempt_values_sha256"] != result["attempts"][1]["attempt_values_sha256"]
    assert result["attempts"][1]["end_sequence"] is None
    assert result["qualification_updates_permitted"] is False
    with pytest.raises(ValueError, match="index identity"):
        analyze(root, "0" * 64)
