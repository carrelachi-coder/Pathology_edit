#!/usr/bin/env python3
"""Synthetic Adapter pressure only. No pathology planner, guard or witness call.

Creates fresh hash/full streams >2M events and real array blobs, or exhausts the
same4GiB including the outer envelope. Never runs the production controller.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import random
import resource
import struct
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from diagnostic_storage import guard_blocks_v2 as codec
from diagnostic_storage.observer_adapter_v1 import ObserverAdapter
from diagnostic_storage.observer_adapter_validation_v1 import (
    validate_capture,
    compare_captures,
)


@dataclass
class Case:
    case_id: str
    primitive_id: str


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


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def globals_sha():
    return hashlib.sha256(
        pickle.dumps((random.getstate(), np.random.get_state()))
    ).hexdigest()


def receipt(path):
    row = {
        "reference_id": "synthetic-source",
        "patient_id": "synthetic-patient",
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
                    "whole_mask_topology_witness": {"errors": ["synthetic-negative"]}
                },
            }
            for op in identity()["operations"]
        ],
        "both_significant_capacity_pass": True,
        "both_significant_topology_pass": False,
    }
    path.write_text(json.dumps([row], indent=2) + "\n")
    return path


def allocated(root):
    count = total = 0
    # Benchmark-only inode inventory. Adapter/reader never accumulate this set.
    # Codec indices use .index.pending / .index.jsonl (unlike blob suffixes).
    seen = set()
    for directory in (root, root / "codec"):
        for p in directory.iterdir():
            if p.is_dir():
                continue
            stat = p.stat()
            key = stat.st_dev, stat.st_ino
            if key in seen:
                continue
            seen.add(key)
            total += stat.st_blocks * 512
            count += 1
    return {"unique_file_allocated_bytes": total, "unique_file_count": count}


def independent_snapshots(root, count):
    """Regenerate original payloads without Codec decoding or adapter encoding."""
    rng = np.random.default_rng(20261103)
    seen = 0
    chain = hashlib.sha256()
    with (root / "codec/arrays.index.jsonl").open("rb") as index:
        for raw in index:
            entry = json.loads(raw)
            descriptor = entry["header"]["descriptor"]
            if (
                descriptor["shape"] != [512, 512]
                or descriptor["dtype"] != np.dtype("float64").str
            ):
                continue
            expected = rng.random((512, 512), dtype=np.float64).tobytes()
            with (root / "codec" / entry["file"]).open("rb") as blob:
                assert blob.read(8) == bytes.fromhex("4754524143453200")
                size = struct.unpack(">I", blob.read(4))[0]
                assert size <= 16384
                header = json.loads(blob.read(size))
                assert header == entry["header"]
                assert blob.read(len(expected) + 1) == expected
            chain.update(hashlib.sha256(expected).digest())
            seen += 1
    assert seen == count
    return {
        "snapshots": seen,
        "raw_payload_bytes": seen * 512 * 512 * 8,
        "every_original_payload_byte_equal": True,
        "ordered_reference_hash": chain.hexdigest(),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    if args.kind == "budget":
        return budget(args.output)
    started = time.perf_counter()
    initial = globals_sha()
    reports = []
    for mode, arm in [("hash_only", "B"), ("full_capture", "C")]:
        core = ObserverAdapter(args.output / arm, mode=mode, identity=identity())
        assert core.valid
        rng = np.random.default_rng(20261103)
        snapshot_count = 0
        interval = (args.events - 4) // (args.arrays + 1)
        guard_ordinal = 0
        begin = time.perf_counter()
        for i in range(args.events):
            operation = (
                identity()["operations"][0 if i < args.events // 2 else 1]
                if i in (0, args.events // 2)
                else None
            )
            if operation:
                core.sink.record("qualification.begin", {"case": Case(**operation)})
            elif i in (args.events // 2 - 1, args.events - 1):
                core.sink.record(
                    "qualification.complete",
                    {
                        "failures": ["synthetic-negative"],
                        "witness": {"errors": ["synthetic-negative"]},
                    },
                )
            else:
                guard_ordinal += 1
                values = {
                    "row": (i // 512) % 512,
                    "col": i % 512,
                    "accepted": i % 3 == 0,
                }
                if guard_ordinal % interval == 0 and snapshot_count < args.arrays:
                    values["snapshot"] = rng.random((512, 512), dtype=np.float64)
                    snapshot_count += 1
                core.sink.record("guard.stage", values)
            if core.sink.failed or not core.valid:
                raise RuntimeError(core.status())
        assert snapshot_count == args.arrays
        core.sink.close()
        rp = receipt(args.output / (arm + "_receipt.json"))
        core.finalize(normal_invocation_end=True, business_receipt_path=rp)
        assert core.valid
        encode = time.perf_counter() - begin
        peak = rss()
        # Fresh validator process measures bounded memory separately from writer.
        child = subprocess.run(
            [
                sys.executable,
                __file__,
                "--validate",
                str(core.root),
                "--receipt",
                str(rp),
                "--mode",
                mode,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        decoded = json.loads(child.stdout)
        assert decoded["verdict"]["stream"]["logical_events"] == args.events
        reports.append(
            {
                "arm": arm,
                "mode": mode,
                "events": args.events,
                "snapshots": snapshot_count,
                "encode_seconds": encode,
                "writer_process_peak_rss_bytes": peak,
                "validator": decoded,
                "storage": allocated(core.root),
            }
        )
    assert globals_sha() == initial
    verification = independent_snapshots(args.output / "C", args.arrays)
    compare_started = time.perf_counter()
    try:
        compared = compare_captures(
            args.output / "B",
            args.output / "C",
            identity=identity(),
            left_receipt=args.output / "B_receipt.json",
            right_receipt=args.output / "C_receipt.json",
        )
        compare_result = {"status": "semantic_match", "result": compared}
    except TimeoutError:
        compare_result = {
            "status": "timeout_hold",
            "reason": "unchanged30s cooperative wrapper budget exceeded; not accepted, no retry or limit increase",
        }
    result = {
        "scope": "pure synthetic Adapter chain, not live Observer/S07",
        "arms": reports,
        "independent_snapshots": verification,
        "global_python_numpy_rng_unchanged": True,
        "comparison": compare_result,
        "comparison_seconds": time.perf_counter() - compare_started,
        "elapsed_seconds": time.perf_counter() - started,
        "new_witness_calls": 0,
        "production_integration": False,
        "code_sha256": {
            name: codec.file_hash(Path("diagnostic_storage") / name)
            for name in (
                "guard_blocks_v2.py",
                "observer_adapter_v1.py",
                "observer_adapter_contract_v1.py",
                "observer_adapter_validation_v1.py",
            )
        },
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def budget(root):
    core = ObserverAdapter(root / "capture", mode="full_capture", identity=identity())
    assert core.valid
    core.sink.record(
        "qualification.begin", {"case": Case(**identity()["operations"][0])}
    )
    copies = 0
    original = core.writer._copy_array

    def copy(array):
        nonlocal copies
        copies += 1
        return original(array)

    core.writer._copy_array = copy
    rng = np.random.default_rng(20261103)
    accepted = 0
    started = time.perf_counter()
    while not core.sink.failed and core.valid:
        before = copies
        core.sink.record(
            "guard.stage", {"snapshot": rng.random((512, 512), dtype=np.float64)}
        )
        if not core.sink.failed and core.valid:
            accepted += 1
    rejected_before_copy = copies == before
    core.sink.close()
    rp = receipt(root / "receipt.json")
    core.finalize(normal_invocation_end=True, business_receipt_path=rp)
    size = allocated(core.root)
    assert rejected_before_copy and size["unique_file_allocated_bytes"] <= 4 * 1024**3
    assert (
        not (core.root / "adapter_complete.json").exists()
        and not (core.root / "codec/observer_complete.json").exists()
    )
    try:
        validate_capture(
            core.root,
            identity=identity(),
            business_receipt_path=rp,
            expected_mode="full_capture",
            publish=True,
        )
    except ValueError:
        rejected = True
    else:
        raise AssertionError("budget failure accepted")
    result = {
        "scope": "synthetic actual whole-tree4GiB boundary including outer controls",
        "snapshots_accepted": accepted,
        "failed_snapshot_rejected_before_copy": rejected_before_copy,
        "status": core.status(),
        "storage": size,
        "limit_bytes": 4 * 1024**3,
        "strict_validator_rejected": rejected,
        "outer_complete_marker": False,
        "inner_complete_marker": False,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": rss(),
        "new_witness_calls": 0,
    }
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["pressure", "budget"], default="pressure")
    parser.add_argument("--events", type=int, default=2100000)
    parser.add_argument("--arrays", type=int, default=683)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--mode", choices=["hash_only", "full_capture"])
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("validation requires assertions enabled")
    if args.validate:
        started = time.perf_counter()
        verdict = validate_capture(
            args.validate,
            identity=identity(),
            business_receipt_path=args.receipt,
            expected_mode=args.mode,
            publish=True,
        )
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "seconds": time.perf_counter() - started,
                    "peak_rss_bytes": rss(),
                }
            )
        )
    else:
        if (
            args.output is None
            or args.events < 4
            or not 0 < args.arrays < args.events // 4
        ):
            parser.error("fresh output and valid event/array sizes required")
        print(json.dumps(run(args), indent=2))
