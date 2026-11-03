#!/usr/bin/env python3
"""Pure synthetic codec pressure runner. No pathology imports or S07 replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import zlib

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostic_storage.guard_blocks_v2 import (
    Writer,
    Reader,
    BudgetError,
    CodecError,
    file_hash,
)


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def storage(root):
    seen = set()
    logical = 0
    allocated = 0
    files = 0
    # Benchmark-only inventory; producer and decoder do not accumulate this set.
    for path in root.iterdir():
        st = path.stat()
        files += 1
        if (st.st_dev, st.st_ino) not in seen:
            seen.add((st.st_dev, st.st_ino))
            logical += st.st_size
            allocated += st.st_blocks * 512
    return {
        "directory_entries": files,
        "unique_inodes": len(seen),
        "file_content_bytes_unique_inodes": logical,
        "allocated_bytes_unique_inodes": allocated,
    }


def event(i, n):
    op = "U1" if i < n // 2 else "U2"
    if i in (0, n // 2):
        kind = "qualification.begin"
    elif i in (n // 2 - 1, n - 1):
        kind = "qualification.complete"
    else:
        return (
            b'{"sequence":'
            + str(i).encode()
            + b',"event":"guard.stage","values":{"mapping":[["stage","source_present"],["row",'
            + str((i // 512) % 512).encode()
            + b'],["col",'
            + str(i % 512).encode()
            + b'],["value",'
            + str(i % 4).encode()
            + b"]]}}\n"
        )
    return (
        json.dumps(
            {"sequence": i, "event": kind, "operation_id": op}, separators=(",", ":")
        )
        + "\n"
    ).encode()


def decode(directory):
    reader = Reader(directory)
    digest = hashlib.sha256()
    count = 0
    started = time.perf_counter()
    for raw in reader.iter_events():
        digest.update(raw)
        count += 1
    return {
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": rss(),
        "events": count,
        "independent_decoded_sha256": digest.hexdigest(),
        "validation": reader.last_validation,
    }


def run_case(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    data = root / "stream"
    source_files = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "diagnostic_storage/guard_blocks_v2.py",
    ]
    identity = {
        str(p.relative_to(source_files[0].parents[1])): file_hash(p)
        for p in source_files
    }
    if args.kind == "array_budget":
        result = array_budget_case(root)
        result["implementation_sha256"] = identity
        (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    writer = Writer(data, mode="full_capture" if args.arrays else "hash_only")
    expected = hashlib.sha256()
    source_bytes = 0
    count = args.events
    array_count = 0
    interval = max(1, count // (args.arrays + 1))
    if args.arrays:
        import numpy as np

        rng = np.random.default_rng(20261103)
    started = time.perf_counter()
    for i in range(count):
        raw = event(i, count)
        if (
            args.arrays
            and i % interval == 0
            and i not in (0, count // 2, count // 2 - 1, count - 1)
            and array_count < args.arrays
        ):
            snapshot = rng.random((512, 512), dtype=np.float64)
            token = writer.add_array(snapshot)
            parsed = json.loads(raw)
            parsed["snapshot"] = token
            raw = (json.dumps(parsed, separators=(",", ":")) + "\n").encode()
            array_count += 1
        writer.write_event(raw)
        expected.update(raw)
        source_bytes += len(raw)
    if array_count != args.arrays:
        raise RuntimeError("requested synthetic snapshot count not achieved")
    summary = writer.finish(
        normal_end=True,
        receipt={
            "normal_end": True,
            "operation_ids": ["U1", "U2"],
            "scope": "synthetic fixture only",
        },
    )
    encode_seconds = time.perf_counter() - started
    producer_peak = rss()
    # Fresh interpreter isolates streaming decoder's peak from writer allocation.
    done = subprocess.run(
        [sys.executable, __file__, "--decode", str(data)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    decoded = json.loads(done.stdout)
    assert (
        decoded["events"] == count
        and decoded["independent_decoded_sha256"]
        == expected.hexdigest()
        == summary["stream_sha256"]
    )
    occupied = storage(data)
    assert occupied["allocated_bytes_unique_inodes"] <= 4 * 1024**3
    result = {
        "scope": "pure synthetic codec stress; no pathology planner/guard/witness invocation",
        "kind": args.kind,
        "logical_events": count,
        "physical_blocks": summary["blocks"],
        "array_occurrences": array_count,
        "mode": summary["mode"],
        "block_integrity_complete": summary["block_integrity_complete"],
        "business_process_complete": summary["business_process_complete"],
        "raw_event_bytes": source_bytes,
        "raw_array_payload_bytes": array_count * 512 * 512 * 8,
        "encode_seconds_including_generation_and_reference_hash": encode_seconds,
        "encode_logical_events_per_second": count / encode_seconds,
        "writer_peak_rss_bytes": producer_peak,
        "writer_active_budget_entries_at_end": len(writer.budget.charges),
        "decoder": decoded,
        "storage": occupied,
        "all_events_and_bytes_verified": True,
        "source_hash": expected.hexdigest(),
        "new_witness_calls": 0,
        "production_acceptance": False,
        "host": {"platform": platform.platform(), "python": sys.version},
    }
    assert identity == {
        str(p.relative_to(source_files[0].parents[1])): file_hash(p)
        for p in source_files
    }, "implementation changed during stress run"
    result["implementation_sha256"] = identity
    result["host"]["zlib_runtime"] = zlib.ZLIB_RUNTIME_VERSION
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def array_budget_case(root):
    import numpy as np

    rng = np.random.default_rng(20261103)
    writer = Writer(root / "stream", mode="full_capture")
    writer.write_event(event(0, 100000))
    copies = 0
    original_copy = writer._copy_array

    def snapshot(array):
        nonlocal copies
        copies += 1
        return original_copy(array)

    writer._copy_array = snapshot
    count = 0
    started = time.perf_counter()
    before_failed_copy = None
    reason = None
    while True:
        array = rng.random((512, 512), dtype=np.float64)
        before = copies
        try:
            token = writer.add_array(array)
        except BudgetError as exc:
            before_failed_copy = copies == before
            reason = str(exc)
            break
        count += 1
        raw = (
            json.dumps(
                {"sequence": count, "event": "guard.stage", "snapshot": token},
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        writer.write_event(raw)
    summary = writer.finish(
        normal_end=False,
        receipt={
            "normal_end": False,
            "operation_ids": ["U1", "U2"],
            "scope": "synthetic budget exhaustion, not source evidence",
        },
    )
    occupied = storage(writer.root)
    assert (
        before_failed_copy
        and occupied["allocated_bytes_unique_inodes"] <= 4 * 1024**3
        and not (writer.root / "observer_complete.json").exists()
    )
    try:
        Reader(writer.root).validate()
    except CodecError:
        rejected = True
    else:
        raise AssertionError("budget failure accepted")
    result = {
        "scope": "synthetic actual disk allocation stress, not source capacity",
        "kind": "array_budget",
        "accepted_snapshots": count,
        "raw_array_payload_bytes": count * 512 * 512 * 8,
        "max_storage_bytes": 4 * 1024**3,
        "storage": occupied,
        "budget_rejection": reason,
        "failed_snapshot_rejected_before_copy": before_failed_copy,
        "array_copy_calls": copies,
        "block_integrity_complete": summary["block_integrity_complete"],
        "business_process_complete": summary["business_process_complete"],
        "complete_marker_exists": False,
        "strict_reader_rejected": rejected,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": rss(),
        "budget_total_conservative": writer.budget.total,
        "new_witness_calls": 0,
    }
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kind", choices=["events", "full_arrays", "array_budget"], default="events"
    )
    p.add_argument("--events", type=int, default=2000000)
    p.add_argument("--arrays", type=int, default=0)
    p.add_argument("--output", type=Path)
    p.add_argument("--decode", type=Path)
    args = p.parse_args()
    if args.decode:
        print(json.dumps(decode(args.decode)))
    else:
        if args.output is None or args.events < 4:
            p.error("fresh output and at least4 events required")
        if (args.kind == "full_arrays") != (args.arrays > 0) or args.arrays < 0:
            p.error(
                "full_arrays requires a positive array count; other kinds require zero"
            )
        print(json.dumps(run_case(args), indent=2))
