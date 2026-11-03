"""Offline acceptance validator. No live adapter instance or production runner.

Trusted identity, receipt location and arm mode must come from the external
harness. A Codec marker alone never establishes Observer acceptance.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

from . import guard_blocks_v2 as codec
from .observer_adapter_contract_v1 import (
    AdapterError,
    FORMAT,
    TOTAL,
    OUTER,
    CONTROLS,
    aliases,
    bounded_json,
    business_receipt,
    code_identity,
    identity_copy,
    outer_allocated,
    publish_control,
    read_json,
)


def _deadline(seconds):
    if type(seconds) not in (int, float) or not 0 <= seconds <= 30:
        raise AdapterError("validation deadline must be within unchanged30s reference")
    end = time.monotonic() + seconds

    def check():
        if time.monotonic() >= end:
            raise TimeoutError("adapter validation deadline")

    return check


def _inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise AdapterError("invalid adapter root")
    names = set()
    for entry in os.scandir(root):
        names.add(entry.name)
        if len(names) > 9:  # Four control pairs plus the Codec directory.
            raise AdapterError("excess outer entries")
    base = {"codec"} | {n + suffix for n in CONTROLS[:2] for suffix in ("", ".pending")}
    accepted = base | {n + suffix for n in CONTROLS[2:] for suffix in ("", ".pending")}
    if names not in (base, accepted):
        raise AdapterError("partial or foreign outer envelope")
    for name in CONTROLS if names == accepted else CONTROLS[:2]:
        aliases(root, name)
    return names == accepted


def _json_token(value):
    """Legacy typed encoding of already-parsed JSON only; no live callbacks."""
    if type(value) is dict:
        return {"mapping": [[k, _json_token(v)] for k, v in value.items()]}
    if type(value) is list:
        return {"sequence_type": "list", "items": [_json_token(v) for v in value]}
    return value


def _prepare(directory, identity, business_receipt_path, expected_mode, publish):
    expected = identity_copy(identity)
    if expected_mode not in ("hash_only", "full_capture"):
        raise AdapterError("explicit expected mode required")
    root = Path(directory)
    accepted = _inventory(root)
    if publish and accepted:
        raise AdapterError("refuse republish/resume")
    manifest, manifest_raw = read_json(root / "adapter_manifest.json")
    status, status_raw = read_json(root / "adapter_status.json")
    if (
        manifest.get("format") != FORMAT
        or manifest.get("identity") != expected
        or manifest.get("mode") != expected_mode
        or manifest.get("code_sha256") != code_identity()
    ):
        raise AdapterError("manifest identity/mode/code mismatch")
    limits = manifest["limits"]
    total = limits["total_bytes"]
    if (
        type(total) is not int
        or not 2 * OUTER <= total <= TOTAL
        or limits
        != {"total_bytes": total, "outer_bytes": OUTER, "codec_bytes": total - OUTER}
    ):
        raise AdapterError("resource partition mismatch")
    if (
        status.get("format") != FORMAT
        or status.get("phase") != "SEALED_PENDING_VALIDATION"
        or status.get("core_fault_free") is not True
        or status.get("sink_fault_free") is not True
        or status.get("sink_closed") is not True
        or status.get("finalized") is not True
        or status.get("interpretation_permitted") is not False
        or status.get("qualification_updates_permitted") is not False
        or status.get("faults")
    ):
        raise AdapterError("terminal adapter fault or incomplete session")
    identity_sha = codec.digest(bounded_json(expected))
    receipt_row, receipt_binding = business_receipt(business_receipt_path, expected)
    if (
        status.get("receipt") != receipt_binding
        or status.get("identity_sha256") != identity_sha
    ):
        raise AdapterError("stale receipt or identity binding")
    summary, summary_raw = read_json(root / "codec/summary.json")
    if (
        summary.get("mode") != expected_mode
        or summary.get("expected_operations")
        != [op["case_id"] for op in expected["operations"]]
        or summary.get("limits", {}).get("storage_bytes") != total - OUTER
    ):
        raise AdapterError("Codec mode/case/budget binding")
    envelope, _ = read_json(root / "codec/business_receipt.json", 8192)
    if envelope != {
        "normal_end": True,
        "operation_ids": [op["case_id"] for op in expected["operations"]],
        "payload": {"receipt": receipt_binding, "identity_sha256": identity_sha},
    }:
        raise AdapterError("Codec business envelope mismatch")
    return dict(
        root=root,
        expected=expected,
        expected_mode=expected_mode,
        accepted=accepted,
        total=total,
        status=status,
        identity_sha=identity_sha,
        receipt_row=receipt_row,
        receipt_binding=receipt_binding,
        receipt_path=business_receipt_path,
        manifest_raw=manifest_raw,
        status_raw=status_raw,
        summary_raw=summary_raw,
    )


def _checked_events(context, check):
    root, expected = context["root"], context["expected"]
    receipt_row = context["receipt_row"]
    reader = codec.Reader(root / "codec")
    completed = 0
    for raw in reader.iter_events():
        check()
        # Frozen adapter layout puts event after sequence. Reader already parses
        # and validates EVERY event; only terminal payloads need a second parse.
        if b'"event":"qualification.' in raw[:128]:
            event = codec.parse(raw)
            if event["event"] == "qualification.begin":
                values = dict(event["values"]["mapping"])
                case = dict(values["case"]["fields"])
                if (
                    case.get("case_id") != expected["operations"][completed]["case_id"]
                    or case.get("primitive_id")
                    != expected["operations"][completed]["primitive_id"]
                ):
                    raise AdapterError("event case/primitive binding mismatch")
            elif event["event"] == "qualification.complete":
                values = dict(event["values"]["mapping"])
                receipt_check = receipt_row["checks"][completed]
                if values != {
                    "failures": _json_token(receipt_check["failure_reasons"]),
                    "witness": _json_token(
                        receipt_check["metrics"]["whole_mask_topology_witness"]
                    ),
                }:
                    raise AdapterError("terminal event/receipt payload mismatch")
                completed += 1
        yield raw
    context["completed"] = completed
    context["verdict"] = reader.last_validation


def _finish(context, check, publish):
    root, expected_mode = context["root"], context["expected_mode"]
    completed, status = context["completed"], context["status"]
    total, accepted = context["total"], context["accepted"]
    identity_sha, receipt_binding = context["identity_sha"], context["receipt_binding"]
    manifest_raw, status_raw, summary_raw = (
        context["manifest_raw"],
        context["status_raw"],
        context["summary_raw"],
    )
    # No cached acceptance: rebind closed controls/receipt after stream exhaustion.
    for name, expected_raw in [
        ("adapter_manifest.json", manifest_raw),
        ("adapter_status.json", status_raw),
        ("codec/summary.json", summary_raw),
    ]:
        _, actual_raw = read_json(root / name)
        if actual_raw != expected_raw:
            raise AdapterError("control changed during validation")
    _, current_binding = business_receipt(context["receipt_path"], context["expected"])
    if current_binding != receipt_binding:
        raise AdapterError("receipt changed during validation")
    verdict = context["verdict"]
    if (
        completed != 2
        or status.get("events") != verdict["logical_events"]
        or not verdict["block_integrity_complete"]
        or not verdict["business_process_complete"]
    ):
        raise AdapterError("stream or event count incomplete")
    allocated = outer_allocated(root)
    for entry in os.scandir(root / "codec"):
        # Reader already required exact pending/final alias identities.
        if not entry.name.endswith(".pending"):
            allocated += entry.stat(follow_symlinks=False).st_blocks * 512
    if allocated > total:
        raise AdapterError("whole-tree allocation exceeded")
    bindings = {
        "manifest_sha256": codec.digest(manifest_raw),
        "status_sha256": codec.digest(status_raw),
        "codec_summary_sha256": codec.digest(summary_raw),
        "business_receipt": receipt_binding,
    }
    validation = {
        "format": FORMAT,
        "scope": "capture completeness only; not qualification or cross-arm acceptance",
        "identity_sha256": identity_sha,
        "mode": expected_mode,
        "bindings": bindings,
        "stream": verdict,
        "qualification_updates_permitted": False,
    }
    if accepted:
        saved, saved_raw = read_json(root / "adapter_validation.json")
        marker, _ = read_json(root / "adapter_complete.json")
        if saved != validation or marker != {
            "format": FORMAT,
            "validation_sha256": codec.digest(saved_raw),
        }:
            raise AdapterError("outer acceptance proof mismatch")
    check()
    result = {
        "candidate_validated": True,
        "interpretation_permitted": accepted or publish,
        "qualification_updates_permitted": False,
        "mode": expected_mode,
        "stream": verdict,
        "allocated_bytes_before_optional_outer_publication": allocated,
        "bindings": bindings,
    }
    if publish:
        validation_sha = publish_control(
            root, "adapter_validation.json", validation, before_link=check
        )
        publish_control(
            root,
            "adapter_complete.json",
            {"format": FORMAT, "validation_sha256": validation_sha},
            before_link=check,
        )
        # No fallible verification/cleanup after the final atomic publication.
    return result


def validate_capture(
    directory,
    *,
    identity,
    business_receipt_path,
    expected_mode,
    publish=False,
    deadline_seconds=30,
):
    """Complete one streaming validation; explicit external publication only.

    Cooperative checks do not replace a future hard helper process timeout.
    """
    check = _deadline(deadline_seconds)
    check()
    context = _prepare(
        directory, identity, business_receipt_path, expected_mode, publish
    )
    for _ in _checked_events(context, check):
        pass
    return _finish(context, check, publish)


def compare_captures(
    left, right, *, identity, left_receipt, right_receipt, deadline_seconds=30
):
    """One pass per stream: all evidence/terminal/receipt checks AND byte match.

    Uses only public Codec.Reader; no repeated full validation or cached proof
    trust. Every array/hash/boundary is rechecked; no exclusions or normalization.
    Cooperative per-event checks still require an external hard process deadline.
    """
    check = _deadline(deadline_seconds)
    check()
    a = _prepare(left, identity, left_receipt, "hash_only", False)
    b = _prepare(right, identity, right_receipt, "full_capture", False)
    if a["receipt_row"] != b["receipt_row"]:
        raise AdapterError("full business receipt mismatch")
    ia, ib = iter(_checked_events(a, check)), iter(_checked_events(b, check))
    sentinel = object()
    count = 0
    while True:
        check()
        x, y = next(ia, sentinel), next(ib, sentinel)
        if x is sentinel or y is sentinel:
            if x is not y:
                raise AdapterError("logical event count mismatch")
            break
        if x != y:
            raise AdapterError("logical event differs at " + str(count))
        count += 1
    _finish(a, check, False)
    _finish(b, check, False)
    check()
    return {
        "status": "semantic_match",
        "events": count,
        "excluded_semantic_fields": [],
        "qualification_updates_permitted": False,
        "scope": "synthetic/offline B/C capture only; A/history requires separate full receipt gate",
    }
