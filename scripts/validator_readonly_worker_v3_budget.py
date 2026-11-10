#!/usr/bin/env python3
"""Candidate v2 read-only helper with trusted per-task budget. No publish switch, no source/witness/model entrypoint."""
import hashlib
import json
from pathlib import Path
import resource
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostic_storage.validator_process_gate_v3_budget import MAX_REQUEST, request_value


def main():
    raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
    request = request_value(raw)
    from diagnostic_storage.validation_budget_policy_v3_budget import (
        budget_for,
        policy_identity,
    )

    seconds = budget_for(request["kind"])
    # Imported in the isolated worker only, never the gate's common parent.
    from diagnostic_storage.observer_adapter_validation_v3_budget import (
        validate_capture,
        compare_captures,
    )

    if request["kind"] == "capture":
        verdict = validate_capture(
            request["capture"],
            identity=request["identity"],
            business_receipt_path=request["receipt"],
            expected_mode=request["mode"],
            deadline_seconds=seconds,
        )
    else:
        verdict = compare_captures(
            request["left"],
            request["right"],
            identity=request["identity"],
            left_receipt=request["left_receipt"],
            right_receipt=request["right_receipt"],
            deadline_seconds=seconds,
        )
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "kind": request["kind"],
        "worker_readonly": True,
        "validation_budget_seconds": seconds,
        **policy_identity(),
        "verdict": verdict,
        "peak_rss_bytes": int(usage if sys.platform == "darwin" else usage * 1024),
        "business_modules_loaded": any(
            n.startswith(("phase3_", "inpaint_cells")) for n in sys.modules
        ),
        "torch_loaded": "torch" in sys.modules,
    }
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # No exception str/repr or raw private capture values on stderr.
        print(type(exc).__name__, file=sys.stderr)
        raise SystemExit(78)
