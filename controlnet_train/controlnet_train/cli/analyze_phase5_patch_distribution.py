"""Summarize layered Phase 5 patch composition for dataset planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_config import get_config


DEFAULT_DATASET_ROOTS = {
    "BCSS": Path(r"D:\WQX\datasets\BCSS\BCSS_PATCHES"),
    "PANDA": Path(r"D:\WQX\datasets\PANDA\PANDA_PATCHES"),
    "GLAS": Path(r"D:\WQX\datasets\GlaS\GlaS_PATCHES"),
    "IGNITE": Path(r"D:\WQX\datasets\IGNITE_PATCHES"),
    "ORCA": Path(r"D:\WQX\datasets\ORCA\ORCA_PATCHES"),
    "PUMA": Path(r"D:\WQX\datasets\PUMA\PUMA_PATCHES"),
}
DEFAULT_OUTPUT_PATH = Path("phase5_runs/patch_analysis/phase5_patch_distribution_summary.json")
DEFAULT_TUMOR_RICH_THRESHOLD = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze layered patch composition for Phase 5 inpaint planning.")
    parser.add_argument(
        "--dataset-root",
        action="append",
        metavar="DATASET=PATH",
        help="Dataset root pair. Repeat for multiple datasets. Defaults to the known six layered patch roots.",
    )
    parser.add_argument(
        "--tumor-rich-threshold",
        type=float,
        default=DEFAULT_TUMOR_RICH_THRESHOLD,
        help="Minimum tumor pixel ratio to count a patch as tumor-rich.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the summary JSON.",
    )
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def _parse_dataset_roots(dataset_root_args: list[str] | None) -> dict[str, Path]:
    if not dataset_root_args:
        return dict(DEFAULT_DATASET_ROOTS)

    dataset_roots: dict[str, Path] = {}
    for item in dataset_root_args:
        dataset_name, separator, dataset_path = item.partition("=")
        if not separator or not dataset_name or not dataset_path:
            raise ValueError(f"Invalid --dataset-root value: {item!r}. Expected DATASET=PATH.")
        dataset_roots[dataset_name.upper()] = Path(dataset_path)
    return dataset_roots


def summarize_patch_categories(
    dataset_name: str,
    tissue_mask: np.ndarray,
    *,
    tumor_rich_threshold: float = DEFAULT_TUMOR_RICH_THRESHOLD,
) -> dict[str, object]:
    config = get_config(dataset_name)
    mask = np.asarray(tissue_mask)
    total_pixels = int(mask.size)

    foreground_labels = sorted(
        int(label) for label in np.unique(mask) if int(label) not in config.skip_tissues
    )
    foreground_pixels = int(np.count_nonzero(~np.isin(mask, list(config.skip_tissues))))
    tumor_pixels = int(np.count_nonzero(np.isin(mask, list(config.tumor_ids))))
    stroma_pixels = int(np.count_nonzero(np.isin(mask, list(config.stroma_ids))))

    tumor_ratio = float(tumor_pixels / total_pixels) if total_pixels else 0.0
    stroma_ratio = float(stroma_pixels / total_pixels) if total_pixels else 0.0
    foreground_ratio = float(foreground_pixels / total_pixels) if total_pixels else 0.0

    single_tissue_patch = len(foreground_labels) == 1 and foreground_pixels > 0
    pure_stroma_patch = single_tissue_patch and bool(config.stroma_ids) and all(
        label in config.stroma_ids for label in foreground_labels
    )
    mixed_patch = len(foreground_labels) >= 2
    tumor_rich_patch = tumor_ratio > tumor_rich_threshold

    return {
        "foreground_labels": foreground_labels,
        "foreground_ratio": foreground_ratio,
        "tumor_ratio": tumor_ratio,
        "stroma_ratio": stroma_ratio,
        "single_tissue_patch": single_tissue_patch,
        "pure_stroma_patch": pure_stroma_patch,
        "mixed_patch": mixed_patch,
        "tumor_rich_patch": tumor_rich_patch,
    }


def analyze_dataset_root(
    dataset_name: str,
    dataset_root: str | Path,
    *,
    tumor_rich_threshold: float = DEFAULT_TUMOR_RICH_THRESHOLD,
) -> dict[str, object]:
    dataset_name = dataset_name.upper()
    dataset_root = Path(dataset_root)
    tissue_mask_paths = sorted((dataset_root / "tissue_masks").glob("*.png"))

    patch_count = len(tissue_mask_paths)
    paired_patch_count = 0
    missing_image_count = 0
    missing_nuclei_count = 0
    single_tissue_patch_count = 0
    pure_stroma_patch_count = 0
    mixed_patch_count = 0
    tumor_rich_patch_count = 0

    label_histogram: dict[str, int] = {}

    for tissue_mask_path in tissue_mask_paths:
        sample_id = tissue_mask_path.stem
        image_path = dataset_root / "images" / f"{sample_id}.png"
        nuclei_mask_path = dataset_root / "nuclei_masks" / f"{sample_id}.png"

        image_exists = image_path.exists()
        nuclei_exists = nuclei_mask_path.exists()
        if image_exists and nuclei_exists:
            paired_patch_count += 1
        if not image_exists:
            missing_image_count += 1
        if not nuclei_exists:
            missing_nuclei_count += 1

        summary = summarize_patch_categories(
            dataset_name,
            load_mask_array(tissue_mask_path),
            tumor_rich_threshold=tumor_rich_threshold,
        )

        for label in summary["foreground_labels"]:
            label_histogram[str(label)] = label_histogram.get(str(label), 0) + 1

        if summary["single_tissue_patch"]:
            single_tissue_patch_count += 1
        if summary["pure_stroma_patch"]:
            pure_stroma_patch_count += 1
        if summary["mixed_patch"]:
            mixed_patch_count += 1
        if summary["tumor_rich_patch"]:
            tumor_rich_patch_count += 1

    def ratio(count: int) -> float:
        return float(count / patch_count) if patch_count else 0.0

    return {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "patch_count": patch_count,
        "paired_patch_count": paired_patch_count,
        "missing_image_count": missing_image_count,
        "missing_nuclei_count": missing_nuclei_count,
        "single_tissue_patch_count": single_tissue_patch_count,
        "single_tissue_patch_ratio": ratio(single_tissue_patch_count),
        "pure_stroma_patch_count": pure_stroma_patch_count,
        "pure_stroma_patch_ratio": ratio(pure_stroma_patch_count),
        "mixed_patch_count": mixed_patch_count,
        "mixed_patch_ratio": ratio(mixed_patch_count),
        "tumor_rich_patch_count": tumor_rich_patch_count,
        "tumor_rich_patch_ratio": ratio(tumor_rich_patch_count),
        "foreground_label_patch_histogram": dict(sorted(label_histogram.items(), key=lambda item: int(item[0]))),
    }


def build_summary(
    dataset_roots: dict[str, Path],
    *,
    tumor_rich_threshold: float = DEFAULT_TUMOR_RICH_THRESHOLD,
) -> dict[str, object]:
    dataset_summaries = [
        analyze_dataset_root(dataset_name, dataset_root, tumor_rich_threshold=tumor_rich_threshold)
        for dataset_name, dataset_root in dataset_roots.items()
    ]
    total_patch_count = sum(summary["patch_count"] for summary in dataset_summaries)

    return {
        "tumor_rich_threshold": tumor_rich_threshold,
        "total_patch_count": total_patch_count,
        "datasets": dataset_summaries,
    }


def write_summary(path: str | Path, payload: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    return path


def load_mask_array(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path))


def _print_summary(payload: dict[str, object]) -> None:
    print(f"tumor-rich threshold: {payload['tumor_rich_threshold']:.2f}")
    print(f"total patches: {payload['total_patch_count']}")
    for summary in payload["datasets"]:
        print(
            f"{summary['dataset']}: patches={summary['patch_count']} "
            f"single={summary['single_tissue_patch_ratio']:.3f} "
            f"pure_stroma={summary['pure_stroma_patch_ratio']:.3f} "
            f"mixed={summary['mixed_patch_ratio']:.3f} "
            f"tumor_rich={summary['tumor_rich_patch_ratio']:.3f}"
        )


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataset_roots = _parse_dataset_roots(args.dataset_root)
    except ValueError as exc:
        parser.error(str(exc))

    summary = build_summary(
        dataset_roots,
        tumor_rich_threshold=args.tumor_rich_threshold,
    )
    output_path = write_summary(args.output_json, summary)
    _print_summary(summary)
    print(f"summary json: {output_path}")


if __name__ == "__main__":
    main()
