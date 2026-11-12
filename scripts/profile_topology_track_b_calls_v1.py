#!/usr/bin/env python3
"""Run the source probe with bounded per-grow timing telemetry."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_mask_edit_refine import candidates, execution, topology


def _install_growth_timing() -> None:
    original = topology.topology_safe_priority_grow
    call_index = 0

    def timed(*args, **kwargs):
        nonlocal call_index
        call_index += 1
        current = call_index
        allowed = kwargs.get("allowed", args[0] if args else None)
        started = time.monotonic()
        print(
            "GROW_BEGIN "
            + json.dumps(
                {
                    "call": current,
                    "target_pixels": int(kwargs["target_pixels"]),
                    "allowed_pixels": int(np.count_nonzero(allowed)),
                    "seed": int(kwargs["seed"]),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            print(
                "GROW_ERROR "
                + json.dumps(
                    {
                        "call": current,
                        "elapsed_seconds": time.monotonic() - started,
                        "exception": type(exc).__name__,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            raise
        print(
            "GROW_END "
            + json.dumps(
                {
                    "call": current,
                    "elapsed_seconds": time.monotonic() - started,
                    "realized_pixels": int(result[1].realized_pixels),
                    "requested_pixels": int(result[1].requested_pixels),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return result

    topology.topology_safe_priority_grow = timed
    execution.topology_safe_priority_grow = timed
    candidates.topology_safe_priority_grow = timed


def main() -> int:
    _install_growth_timing()
    probe = Path(__file__).with_name("probe_manuscript_joint_capacity.py")
    sys.argv[0] = str(probe)
    runpy.run_path(str(probe), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
