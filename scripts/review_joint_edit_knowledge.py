#!/usr/bin/env python3
"""Print/export the draft joint knowledge review without promoting skills."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.skills.review import (
    build_joint_knowledge_review,
    export_capability_matrix,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matrix-output", type=Path)
    args = parser.parse_args()
    review = build_joint_knowledge_review()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(review, encoding="utf-8")
    else:
        print(review, end="")
    if args.matrix_output:
        args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
        export_capability_matrix(args.matrix_output)
