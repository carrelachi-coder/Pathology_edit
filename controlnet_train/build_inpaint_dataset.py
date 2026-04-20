"""Normalize Phase 5 inpaint metadata into train/val jsonl files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.data.inpaint import build_inpaint_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized metadata for ControlNet inpaint training.")
    parser.add_argument(
        "--input-jsonl",
        action="append",
        required=True,
        help="Path to an input jsonl file describing inpaint samples. Repeat for multiple sources.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for metadata_inpaint_{train,val}.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, val_path = build_inpaint_metadata(
        input_jsonl_paths=args.input_jsonl,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"train metadata: {train_path}")
    print(f"val metadata:   {val_path}")


if __name__ == "__main__":
    main()
