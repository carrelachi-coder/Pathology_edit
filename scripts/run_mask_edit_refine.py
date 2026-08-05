#!/usr/bin/env python3
"""Independent entrypoint for phase3_mask_edit_refine."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase3_mask_edit_refine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
