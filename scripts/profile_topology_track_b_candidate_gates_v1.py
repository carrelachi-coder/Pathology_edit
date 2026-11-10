#!/usr/bin/env python3
"""Run the unchanged source probe with bounded candidate-gate telemetry."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_mask_edit_refine.gates import GateRegistry


def _install_gate_timing() -> None:
    original = GateRegistry.run
    started = time.monotonic()
    call_index = 0

    def timed(self, context):
        nonlocal call_index
        call_index += 1
        report = original(self, context)
        failed = [
            {
                "check_id": item.check_id,
                "detail": item.detail[:500],
            }
            for item in report.checks
            if not item.passed
        ]
        print(
            "GATE_RESULT "
            + json.dumps(
                {
                    "call": call_index,
                    "elapsed_from_start_seconds": time.monotonic() - started,
                    "candidate_id": context.candidate.candidate_id,
                    "tool_name": context.candidate.tool_name,
                    "changed_pixels": int(context.candidate.change_region.sum()),
                    "passed": bool(report.passed),
                    "failed": failed,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return report

    GateRegistry.run = timed


def main() -> int:
    _install_gate_timing()
    probe = Path(__file__).with_name("probe_manuscript_joint_capacity.py")
    sys.argv[0] = str(probe)
    runpy.run_path(str(probe), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
