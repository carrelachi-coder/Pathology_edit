#!/usr/bin/env python3
"""Run the unchanged source probe while exposing read-only SIGUSR1 stacks."""

from __future__ import annotations

import faulthandler
import runpy
import signal
import sys
from pathlib import Path


def main() -> int:
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    probe = Path(__file__).with_name("probe_manuscript_joint_capacity.py")
    sys.argv[0] = str(probe)
    runpy.run_path(str(probe), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
