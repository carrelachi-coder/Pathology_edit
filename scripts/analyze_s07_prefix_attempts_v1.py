#!/usr/bin/env python3
"""Analyze every committed S07 B event block without upgrading partial evidence.

This script reads only an archived, incomplete Codec directory. It verifies
committed block files and decoded bytes, then summarizes completed U1 trials.
It never calls a pathology planner or publishes an acceptance marker.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import zlib


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mapping(value):
    return dict(value.get("mapping", []))


def fingerprint(value) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def analyze(codec_dir: Path, expected_index_sha256: str) -> dict:
    index = codec_dir / "blocks.index.pending"
    raw_index = index.read_bytes()
    if sha(raw_index) != expected_index_sha256:
        raise ValueError("archived pending index identity changed")
    rows = [json.loads(line) for line in raw_index.splitlines()]
    attempts = []
    current = None
    previous = "0" * 64
    next_sequence = 0
    total_events = 0
    for block_number, row in enumerate(rows):
        if (
            row["index"] != block_number
            or row["first_sequence"] != next_sequence
            or row["previous_header_sha256"] != previous
        ):
            raise ValueError("committed block chain or sequence gap")
        raw = (codec_dir / row["file"]).read_bytes()
        if sha(raw) != row["file_sha256"] or raw[:8] != b"GTRACE2\0":
            raise ValueError("committed block bytes changed")
        header_length = struct.unpack(">I", raw[8:12])[0]
        header_raw = raw[12 : 12 + header_length]
        header = json.loads(header_raw)
        if sha(header_raw) != row["header_sha256"]:
            raise ValueError("block header identity changed")
        decoded = zlib.decompress(raw[12 + header_length :])
        if sha(decoded) != header["decoded_sha256"]:
            raise ValueError("decoded block bytes changed")
        for line in decoded.splitlines():
            total_events += 1
            if b'"event":"guard.' in line:
                if current is not None:
                    if b'"event":"guard.stage"' in line:
                        current["guard_stage_events"] += 1
                    elif b'"event":"guard.reject"' in line:
                        current["guard_reject_events"] += 1
                    elif b'"event":"guard.accept"' in line:
                        current["guard_accept_events"] += 1
                    elif b'"event":"guard.local_result"' in line:
                        current["guard_local_result_events"] += 1
                continue
            event = json.loads(line)
            name = event["event"]
            values = mapping(event.get("values", {}))
            if name == "compiler.attempt":
                current = {
                    "ordinal": len(attempts) + 1,
                    "attempt_label": values.get("attempt"),
                    "start_sequence": event["sequence"],
                    "end_sequence": None,
                    "probe_count": 0,
                    "topology_passing_probe_count": 0,
                    "max_topology_passing_realized_pixels": None,
                    "max_topology_failing_realized_pixels": None,
                    "hard_min_pixels": None,
                    "grow_begin_count": 0,
                    "grow_end_count": 0,
                    "guard_stage_events": 0,
                    "guard_reject_events": 0,
                    "guard_accept_events": 0,
                    "guard_local_result_events": 0,
                    "exception_type": None,
                    "attempt_values_sha256": fingerprint(values),
                    "attempt_rng_sha256": fingerprint(event.get("rng")),
                    "probe_values_sha256": [],
                    "probe_rng_sha256": [],
                    "planner_selection_values_sha256": [],
                    "probe_topology": [],
                }
                attempts.append(current)
            elif current is not None:
                if name == "probe.complete":
                    current["probe_count"] += 1
                    current["probe_values_sha256"].append(fingerprint(values))
                    current["probe_rng_sha256"].append(fingerprint(event.get("rng")))
                    current["hard_min_pixels"] = values["hard_min_pixels"]
                    topology = mapping(values["topology"])
                    current["probe_topology"].append({
                        "requested_pixels": values["total"],
                        "realized_pixels": values["realized"],
                        "hard_min_pixels": values["hard_min_pixels"],
                        "passed": topology["passed"],
                        "source_components_before": topology["source_components_before"],
                        "source_components_after": topology["source_components_after"],
                        "source_holes_before": topology["source_holes_before"],
                        "source_holes_after": topology["source_holes_after"],
                        "target_components_before": topology["target_components_before"],
                        "target_components_after": topology["target_components_after"],
                        "target_holes_before": topology["target_holes_before"],
                        "target_holes_after": topology["target_holes_after"],
                        "changed_component_area_passed": topology["changed_component_area_passed"],
                    })
                    key = (
                        "max_topology_passing_realized_pixels"
                        if topology["passed"] is True
                        else "max_topology_failing_realized_pixels"
                    )
                    if topology["passed"] is True:
                        current["topology_passing_probe_count"] += 1
                    realized = values["realized"]
                    current[key] = max(current[key] or 0, realized)
                elif name == "grow.begin":
                    current["grow_begin_count"] += 1
                elif name == "planner.selection":
                    current["planner_selection_values_sha256"].append(fingerprint(values))
                elif name == "grow.end":
                    current["grow_end_count"] += 1
                elif name == "compiler.exception":
                    current["exception_type"] = values["exception"]["exception_type"]
                    current["end_sequence"] = event["sequence"]
        previous = row["header_sha256"]
        next_sequence = row["last_sequence"] + 1
    if total_events != next_sequence:
        raise ValueError("decoded event count differs from committed index")
    return {
        "kind": "s07-B-committed-prefix-attempt-summary-v1",
        "status": "partial_prefix_only_business_process_incomplete",
        "pending_index_sha256": expected_index_sha256,
        "committed_blocks": len(rows),
        "committed_events": total_events,
        "attempts": attempts,
        "completed_exception_types": dict(
            Counter(a["exception_type"] for a in attempts if a["end_sequence"] is not None)
        ),
        "qualification_updates_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.codec_dir, args.expected_index_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
