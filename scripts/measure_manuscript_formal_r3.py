#!/usr/bin/env python3
"""Measure the frozen formal R3 cohort using the validated six-image chain."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_manuscript_joint_technical import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--formal"]))
