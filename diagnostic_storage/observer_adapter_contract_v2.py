"""Bounded file/receipt contract shared by the independent adapter and validator.

No production activation, model import or qualification call. Not an OS quota.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re

from . import guard_blocks_v2 as codec

FORMAT = "observer-codec-adapter-v2-budget"
TOTAL = 4 * 1024**3
OUTER = 256 * 1024
CONTROL = 32 * 1024
RECEIPT = 8 * 1024**2
CODEC_PIN = "dd4103b00b7eebf2ac4018a7997a61a01826840a0647cc3c9d19dafd53ecbef4"
BUDGET_PIN = "d4ac09d03509e9d149b96e184d0abdebab08462730b7e5a494af7432df03cf64"
CONTROLS = (
    "adapter_manifest.json",
    "adapter_status.json",
    "adapter_validation.json",
    "adapter_complete.json",
)


class AdapterError(ValueError):
    pass


def bounded_json(value, limit=CONTROL):
    """Reject excessive metadata BEFORE allocating an escaped JSON string."""
    remaining = limit

    def walk(item):
        nonlocal remaining
        remaining -= 16
        if remaining < 0:
            raise AdapterError("metadata limit")
        if type(item) is str:
            remaining -= 12 * len(item)
        elif item is None or type(item) is bool:
            pass
        elif type(item) is int:
            remaining -= item.bit_length() // 3 + 2
        elif type(item) is float:
            if not math.isfinite(item):
                raise AdapterError("nonfinite control metadata")
        elif type(item) in (list, tuple):
            if len(item) > remaining // 16:
                raise AdapterError("metadata limit")
            for v in item:
                walk(v)
        elif type(item) is dict:
            if len(item) > remaining // 32:
                raise AdapterError("metadata limit")
            for k, v in item.items():
                if type(k) is not str:
                    raise AdapterError("control keys must be strings")
                walk(k)
                walk(v)
        else:
            raise AdapterError("unsupported control type")
        if remaining < 0:
            raise AdapterError("metadata limit")

    walk(value)
    raw = codec.dump(value) + b"\n"
    if len(raw) > limit:
        raise AdapterError("metadata byte limit")
    return raw


def identity_copy(value):
    identity = codec.parse(bounded_json(value))
    required = {
        "reference_id",
        "patient_id",
        "operations",
        "run_manifest_sha256",
        "input_identity_sha256",
        "case_mapping_sha256",
        "contract_sha256",
        "startup_identity_sha256",
    }
    if set(identity) != required:
        raise AdapterError("identity fields")
    for key in required - {"reference_id", "patient_id", "operations"}:
        if type(identity[key]) is not str or not re.fullmatch(
            "[0-9a-f]{64}", identity[key]
        ):
            raise AdapterError("identity SHA")
    for key in ("reference_id", "patient_id"):
        if type(identity[key]) is not str or not identity[key]:
            raise AdapterError("source identity")
    operations = identity["operations"]
    if type(operations) is not list or len(operations) != 2:
        raise AdapterError("two operations required")
    for op in operations:
        if (
            type(op) is not dict
            or set(op) != {"case_id", "primitive_id"}
            or any(type(v) is not str or not v or len(v) > 512 for v in op.values())
        ):
            raise AdapterError("operation mapping")
    if (
        len({op["case_id"] for op in operations}) != 2
        or len({op["primitive_id"] for op in operations}) != 2
    ):
        raise AdapterError("duplicate operation identity")
    return identity


def code_identity():
    root = Path(__file__).parent
    names = (
        "guard_blocks_v2.py",
        "guard_budget_candidate_v1.py",
        "observer_adapter_contract_v2.py",
        "observer_adapter_v2.py",
        "observer_adapter_validation_v2_budget.py",
    )
    result = {name: codec.file_hash(root / name) for name in names}
    if result["guard_blocks_v2.py"] != CODEC_PIN:
        raise AdapterError("Codec baseline changed")
    if result["guard_budget_candidate_v1.py"] != BUDGET_PIN:
        raise AdapterError("Budget candidate changed")
    return result


def read_json(path, limit=CONTROL):
    path = Path(path)
    if path.is_symlink():
        raise AdapterError("symlink input")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise AdapterError("input byte limit")
    return codec.parse(raw), raw


def aliases(root, name):
    final, pending = root / name, root / (name + ".pending")
    if (
        final.is_symlink()
        or pending.is_symlink()
        or not final.is_file()
        or not pending.is_file()
    ):
        raise AdapterError("missing/nonregular control pair")
    a, b = final.stat(), pending.stat()
    if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        raise AdapterError("control alias mismatch")


def outer_allocated(root):
    # At most four control pairs. No accumulating list of Codec artifacts.
    seen = set()
    total = 0
    for entry in os.scandir(root):
        if entry.name == "codec":
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise AdapterError("invalid Codec directory")
            continue
        if entry.name not in CONTROLS and entry.name not in tuple(
            x + ".pending" for x in CONTROLS
        ):
            raise AdapterError("foreign outer artifact")
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise AdapterError("nonregular control")
        stat = entry.stat(follow_symlinks=False)
        key = stat.st_dev, stat.st_ino
        if key not in seen:
            seen.add(key)
            total += stat.st_blocks * 512
    if total > OUTER:
        raise AdapterError("outer control allocation limit")
    return total


def publish_control(root, name, value, *, before_link=None):
    if name not in CONTROLS:
        raise AdapterError("unknown control file")
    raw = bounded_json(value)
    pending, final = root / (name + ".pending"), root / name
    reserve = ((len(raw) + 4095) // 4096 + 1) * 4096
    if outer_allocated(root) + reserve > OUTER:
        raise AdapterError("outer budget before write")
    with pending.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    outer_allocated(root)
    result = codec.digest(raw)
    if before_link is not None:
        before_link()
    os.link(pending, final)  # Atomic no-replace. No fallible work after publication.
    return result


def business_receipt(path, identity):
    rows, raw = read_json(path, RECEIPT)
    if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
        raise AdapterError("one source receipt required")
    row = rows[0]
    if (
        row.get("reference_id") != identity["reference_id"]
        or row.get("patient_id") != identity["patient_id"]
    ):
        raise AdapterError("receipt source mapping mismatch")
    if (
        row.get("source_only") is not True
        or row.get("target_mask_created") is not False
        or row.get("source_content_verified_against_census") is not True
    ):
        raise AdapterError("receipt source contract")
    checks = row.get("checks")
    if type(checks) is not list or len(checks) != 2:
        raise AdapterError("two checks required")
    for check, op in zip(checks, identity["operations"]):
        if (
            type(check) is not dict
            or check.get("primitive_id") != op["primitive_id"]
            or check.get("passed_necessary_capacity") is not True
            or type(check.get("passed_topology_witness")) is not bool
            or type(check.get("failure_reasons")) is not list
            or type(check.get("metrics")) is not dict
            or type(check["metrics"].get("whole_mask_topology_witness")) is not dict
        ):
            raise AdapterError("incomplete operation receipt")
        if check["passed_topology_witness"] != (not check["failure_reasons"]):
            raise AdapterError("contradictory operation receipt")
    if row.get("both_significant_capacity_pass") is not True or row.get(
        "both_significant_topology_pass"
    ) is not all(c["passed_topology_witness"] for c in checks):
        raise AdapterError("receipt aggregate mismatch")
    return row, {"sha256": codec.digest(raw), "bytes": len(raw)}
