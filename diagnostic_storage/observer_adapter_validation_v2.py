"""Candidate read-only v2 consumer of EXACT pinned v1 capture primitives.

Producer identity and consumer identity are distinct. No manifest rewrite, monkey
patch of v1 clocks, new serialization, field exclusion, marker publication or
change to scientific/recording budgets. Compatibility approval remains pending.
"""
import math
import time
from . import observer_adapter_validation_v1 as legacy
from .validation_budget_policy_v2 import (
    COMPATIBILITY_ID,
    budget_for,
    consumer_identity,
    policy_identity,
    verify_legacy_files,
)


def _deadline(kind, seconds):
    maximum = budget_for(kind)
    seconds = maximum if seconds is None else seconds
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 <= seconds <= maximum
    ):
        raise ValueError("deadline exceeds trusted task budget")
    end = time.monotonic() + seconds

    def check():
        if time.monotonic() >= end:
            raise TimeoutError("v2 read-only validation deadline")

    return check


def _provenance(kind):
    return {
        **policy_identity(),
        "compatibility_profile": COMPATIBILITY_ID,
        "validation_budget_seconds": budget_for(kind),
        "producer_component_sha256": verify_legacy_files(),
        "consumer_component_sha256": consumer_identity(),
    }


def validate_capture(
    directory, *, identity, business_receipt_path, expected_mode, deadline_seconds=None
):
    check = _deadline("capture", deadline_seconds)
    check()
    provenance = _provenance("capture")
    context = legacy._prepare(
        directory, identity, business_receipt_path, expected_mode, False
    )
    for _ in legacy._checked_events(context, check):
        pass
    result = legacy._finish(context, check, False)
    # Source proof (if present) stays checked/unchanged, not newly authorized.
    result["source_had_outer_proof"] = context["accepted"]
    result["interpretation_permitted"] = False
    result.update(provenance)
    check()
    return result


def compare_captures(
    left, right, *, identity, left_receipt, right_receipt, deadline_seconds=None
):
    check = _deadline("compare", deadline_seconds)
    check()
    provenance = _provenance("compare")
    a = legacy._prepare(left, identity, left_receipt, "hash_only", False)
    b = legacy._prepare(right, identity, right_receipt, "full_capture", False)
    if a["receipt_row"] != b["receipt_row"]:
        raise ValueError("full business receipt mismatch")
    ia, ib = iter(legacy._checked_events(a, check)), iter(
        legacy._checked_events(b, check)
    )
    sentinel = object()
    count = 0
    while True:
        check()
        x, y = next(ia, sentinel), next(ib, sentinel)
        if x is sentinel or y is sentinel:
            if x is not y:
                raise ValueError("logical event count mismatch")
            break
        if x != y:
            raise ValueError("logical event differs at " + str(count))
        count += 1
    legacy._finish(a, check, False)
    legacy._finish(b, check, False)
    check()
    return {
        "status": "semantic_match",
        "events": count,
        "excluded_semantic_fields": [],
        "interpretation_permitted": False,
        "qualification_updates_permitted": False,
        "scope": "read-only versioned consumer; no source qualification or new capture proof",
        **provenance,
    }
