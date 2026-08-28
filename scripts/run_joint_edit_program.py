#!/usr/bin/env python3
"""Run the primitive-free ordered joint mask-edit program CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.program_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
