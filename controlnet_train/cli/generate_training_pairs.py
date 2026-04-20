"""Build Phase 5 same-WSI cross-reconstruction metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controlnet_train.data.cross import build_cross_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build same-WSI cross-reconstruction pairs.")
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        help="Dataset root in the form DATASET=PATH. Repeat for multiple datasets.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for metadata_cross_{train,val}.json")
    parser.add_argument("--num-ref-per-target", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = {}
    for spec in args.dataset_root:
        if "=" not in spec:
            raise ValueError(f"Invalid --dataset-root spec '{spec}'. Expected DATASET=PATH.")
        dataset_name, path = spec.split("=", 1)
        dataset_roots[dataset_name] = path

    train_path, val_path = build_cross_metadata(
        dataset_roots=dataset_roots,
        output_dir=args.output_dir,
        num_ref_per_target=args.num_ref_per_target,
        top_k=args.top_k,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"train metadata: {train_path}")
    print(f"val metadata:   {val_path}")


if __name__ == "__main__":
    main()
