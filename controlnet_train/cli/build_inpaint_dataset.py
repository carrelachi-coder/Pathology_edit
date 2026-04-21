"""Normalize Phase 5 inpaint metadata into train/val jsonl files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controlnet_train.data import build_inpaint_metadata, build_synthetic_inpaint_metadata
from controlnet_train.data.inpaint_synthesis import _VALID_FORCED_MODES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build normalized metadata for ControlNet inpaint training.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input-jsonl",
        action="append",
        type=Path,
        help="Path to an input jsonl file describing inpaint samples. Repeat for multiple sources.",
    )
    source_group.add_argument(
        "--dataset-root",
        action="append",
        metavar="DATASET=PATH",
        help="Dataset root pair for synthetic GT generation. Repeat for multiple datasets.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for metadata_inpaint_{train,val}.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--forced-mode",
        default="identity",
        choices=sorted(_VALID_FORCED_MODES),
        help="Synthetic GT edit mode for --dataset-root. Defaults to identity.",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=None)
    parser.add_argument("--max-attempts-per-sample", type=int, default=None)
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def _parse_dataset_roots(dataset_root_args: list[str]) -> dict[str, Path]:
    dataset_roots: dict[str, Path] = {}
    for item in dataset_root_args:
        dataset_name, separator, dataset_path = item.partition("=")
        if not separator or not dataset_name or not dataset_path:
            raise ValueError(f"Invalid --dataset-root value: {item!r}. Expected DATASET=PATH.")
        dataset_roots[dataset_name.upper()] = Path(dataset_path)
    return dataset_roots


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_jsonl:
        if (
            args.samples_per_dataset is not None
            or args.max_attempts_per_sample is not None
            or args.forced_mode != "identity"
        ):
            parser.error(
                "--samples-per-dataset, --max-attempts-per-sample, and --forced-mode are only supported with --dataset-root."
            )
        train_path, val_path = build_inpaint_metadata(
            input_jsonl_paths=args.input_jsonl,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        if args.samples_per_dataset is not None and args.samples_per_dataset <= 0:
            parser.error(f"--samples-per-dataset must be positive, got {args.samples_per_dataset}")
        if args.max_attempts_per_sample is not None and args.max_attempts_per_sample <= 0:
            parser.error(
                f"--max-attempts-per-sample must be positive, got {args.max_attempts_per_sample}"
            )
        try:
            dataset_roots = _parse_dataset_roots(args.dataset_root)
        except ValueError as exc:
            parser.error(str(exc))
        train_path, val_path = build_synthetic_inpaint_metadata(
            dataset_roots=dataset_roots,
            output_dir=args.output_dir,
            forced_mode=args.forced_mode,
            val_ratio=args.val_ratio,
            seed=args.seed,
            samples_per_dataset=args.samples_per_dataset,
            max_attempts_per_sample=args.max_attempts_per_sample,
        )
    print(f"train metadata: {train_path}")
    print(f"val metadata:   {val_path}")


if __name__ == "__main__":
    main()
