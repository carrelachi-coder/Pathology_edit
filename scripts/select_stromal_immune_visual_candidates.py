#!/usr/bin/env python3
"""Select and package stromal-immune visual candidates from the paired bank."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


BACKENDS = (("inpaint", "inpaint_image"), ("cross", "cross_image"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-per-wsi", type=int, default=1)
    parser.add_argument("--metric-size", type=int, default=128)
    parser.add_argument("--exclude-ids", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None:
            image = image.resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(image) > 0


def grayscale(image: np.ndarray) -> np.ndarray:
    return (
        0.299 * image[..., 0]
        + 0.587 * image[..., 1]
        + 0.114 * image[..., 2]
    )


def laplacian_variance(image: np.ndarray) -> float:
    gray = grayscale(image)
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.var(laplacian))


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_gray = grayscale(first).reshape(-1)
    second_gray = grayscale(second).reshape(-1)
    first_gray = first_gray - first_gray.mean()
    second_gray = second_gray - second_gray.mean()
    denominator = float(
        np.linalg.norm(first_gray) * np.linalg.norm(second_gray)
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_gray, second_gray) / denominator)


def mean_absolute(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    difference = np.abs(first - second).mean(axis=2)
    if mask is None or not np.any(mask):
        return float(difference.mean())
    return float(difference[mask].mean())


def color_shift(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first.mean(axis=(0, 1)) - second.mean(axis=(0, 1))))


def percentile_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    if len(order) == 1:
        ranks[order] = 1.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, len(order))
    return ranks


def quality_metrics(
    reference: np.ndarray,
    moderate: np.ndarray,
    significant: np.ndarray,
    moderate_mask: np.ndarray,
    significant_mask: np.ndarray,
) -> dict[str, float]:
    reference_sharpness = max(laplacian_variance(reference), 1e-10)
    moderate_ratio = laplacian_variance(moderate) / reference_sharpness
    significant_ratio = laplacian_variance(significant) / reference_sharpness
    sharpness_quality = 0.5 * (
        math.exp(-abs(math.log(max(moderate_ratio, 1e-6))) / 0.55)
        + math.exp(-abs(math.log(max(significant_ratio, 1e-6))) / 0.55)
    )
    structure = 0.5 * (
        correlation(reference, moderate)
        + correlation(reference, significant)
    )
    drift = 0.5 * (
        color_shift(reference, moderate)
        + color_shift(reference, significant)
    )
    inside_response = 0.5 * (
        mean_absolute(reference, moderate, moderate_mask)
        + mean_absolute(reference, significant, significant_mask)
    )
    progression = mean_absolute(
        moderate,
        significant,
        moderate_mask | significant_mask,
    )
    return {
        "moderate_sharpness_ratio": moderate_ratio,
        "significant_sharpness_ratio": significant_ratio,
        "sharpness_quality": sharpness_quality,
        "structure_correlation": structure,
        "color_shift": drift,
        "inside_response": inside_response,
        "moderate_significant_visible_change": progression,
    }


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package_backend(
    rows: list[dict],
    output_root: Path,
    backend: str,
) -> None:
    backend_root = output_root / backend
    packaged = []
    for rank, row in enumerate(rows, start=1):
        display_id = row["reference_id"].rsplit("_", 1)[-1]
        case_dir_name = f"{rank:02d}_{display_id}"
        case_dir = backend_root / case_dir_name
        copy_required(Path(row["reference_image"]), case_dir / "original.png")
        copy_required(
            Path(row["moderate_change_region"]),
            case_dir / "moderate_change_region.png",
        )
        copy_required(
            Path(row["significant_change_region"]),
            case_dir / "significant_change_region.png",
        )
        copy_required(
            Path(row["moderate_generated"]),
            case_dir / "moderate_generated.png",
        )
        copy_required(
            Path(row["significant_generated"]),
            case_dir / "significant_generated.png",
        )
        record = dict(row)
        record.update(
            {
                "rank": rank,
                "backend": backend,
                "case_dir": case_dir_name,
                "display_id": display_id,
                "pair_id": row["moderate_sample_id"],
                "sample_id": row["significant_sample_id"],
                "moderate_fraction": row["moderate_fraction"],
                "significant_fraction": row["significant_fraction"],
                "dose_increase": (
                    row["significant_fraction"] - row["moderate_fraction"]
                ),
                "incremental_visible_change": row["incremental_fraction"],
                "sharpness_ratio": 0.5
                * (
                    row["moderate_sharpness_ratio"]
                    + row["significant_sharpness_ratio"]
                ),
            }
        )
        packaged.append(record)
    backend_root.mkdir(parents=True, exist_ok=True)
    (backend_root / "selection_manifest.json").write_text(
        json.dumps(packaged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    exclude_ids = {
        value.strip() for value in args.exclude_ids.split(",") if value.strip()
    }
    manifest_root = args.run_root / "manifests"
    moderate_rows = read_jsonl(
        manifest_root / "u2_moderate_evaluation_manifest.jsonl"
    )
    significant_rows = read_jsonl(
        manifest_root / "u2_significant_evaluation_manifest.jsonl"
    )
    significant_by_reference = {
        row["reference_id"]: row for row in significant_rows
    }
    base_rows: list[dict] = []
    for moderate in moderate_rows:
        reference_id = moderate["reference_id"]
        if reference_id in exclude_ids:
            continue
        significant = significant_by_reference[reference_id]
        moderate_mask_full = load_mask(Path(moderate["change_region"]))
        significant_mask_full = load_mask(Path(significant["change_region"]))
        moderate_fraction = float(moderate_mask_full.mean())
        significant_fraction = float(significant_mask_full.mean())
        incremental_fraction = float(
            np.mean(significant_mask_full & ~moderate_mask_full)
        )
        containment = float(
            np.count_nonzero(moderate_mask_full & significant_mask_full)
            / max(np.count_nonzero(moderate_mask_full), 1)
        )
        if (
            moderate_fraction < 0.075
            or significant_fraction < 0.125
            or significant_fraction <= moderate_fraction + 0.018
            or incremental_fraction < 0.025
        ):
            continue
        base_rows.append(
            {
                "reference_id": reference_id,
                "reference_image": moderate["reference_image"],
                "wsi_id": moderate["wsi_id"],
                "patient_id": moderate["patient_id"],
                "primitive": moderate["primitive"],
                "moderate_sample_id": moderate["sample_id"],
                "significant_sample_id": significant["sample_id"],
                "moderate_change_region": moderate["change_region"],
                "significant_change_region": significant["change_region"],
                "moderate_fraction": moderate_fraction,
                "significant_fraction": significant_fraction,
                "incremental_fraction": incremental_fraction,
                "moderate_containment_in_significant": containment,
            }
        )

    if len(base_rows) < args.count:
        raise ValueError(
            f"only {len(base_rows)} mask-eligible rows for {args.count} slots"
        )

    summary = {
        "primitive": "stromal_immune_infiltration",
        "input_references": len(moderate_rows),
        "mask_eligible_references": len(base_rows),
        "selection_count_per_backend": args.count,
        "max_per_wsi": args.max_per_wsi,
        "backends": {},
    }
    for backend, manifest_image_field in BACKENDS:
        metric_rows = []
        for base in base_rows:
            reference_id = base["reference_id"]
            moderate = next(
                row
                for row in moderate_rows
                if row["reference_id"] == reference_id
            )
            significant = significant_by_reference[reference_id]
            reference_image = load_rgb(
                Path(base["reference_image"]), args.metric_size
            )
            moderate_image = load_rgb(
                Path(moderate[manifest_image_field]), args.metric_size
            )
            significant_image = load_rgb(
                Path(significant[manifest_image_field]), args.metric_size
            )
            moderate_mask = load_mask(
                Path(base["moderate_change_region"]), args.metric_size
            )
            significant_mask = load_mask(
                Path(base["significant_change_region"]), args.metric_size
            )
            row = dict(base)
            row.update(
                {
                    "moderate_generated": moderate[manifest_image_field],
                    "significant_generated": significant[manifest_image_field],
                }
            )
            row.update(
                quality_metrics(
                    reference_image,
                    moderate_image,
                    significant_image,
                    moderate_mask,
                    significant_mask,
                )
            )
            metric_rows.append(row)

        mask_ranks = percentile_ranks(
            [
                0.45 * row["moderate_fraction"]
                + 0.55 * row["significant_fraction"]
                for row in metric_rows
            ]
        )
        increment_ranks = percentile_ranks(
            [row["incremental_fraction"] for row in metric_rows]
        )
        quality_ranks = percentile_ranks(
            [
                0.45 * row["sharpness_quality"]
                + 0.40 * row["structure_correlation"]
                - 0.15 * row["color_shift"]
                for row in metric_rows
            ]
        )
        response_ranks = percentile_ranks(
            [row["inside_response"] for row in metric_rows]
        )
        progression_ranks = percentile_ranks(
            [
                row["moderate_significant_visible_change"]
                for row in metric_rows
            ]
        )
        containment_ranks = percentile_ranks(
            [
                row["moderate_containment_in_significant"]
                for row in metric_rows
            ]
        )
        for index, row in enumerate(metric_rows):
            response_target = max(
                0.0, 1.0 - abs(float(response_ranks[index]) - 0.78) / 0.78
            )
            progression_target = max(
                0.0,
                1.0 - abs(float(progression_ranks[index]) - 0.82) / 0.82,
            )
            row["selection_score"] = float(
                0.26 * mask_ranks[index]
                + 0.19 * increment_ranks[index]
                + 0.27 * quality_ranks[index]
                + 0.12 * response_target
                + 0.11 * progression_target
                + 0.05 * containment_ranks[index]
            )

        selected = []
        wsi_counts: dict[str, int] = {}
        for row in sorted(
            metric_rows,
            key=lambda item: (-item["selection_score"], item["reference_id"]),
        ):
            wsi_id = row["wsi_id"]
            if wsi_counts.get(wsi_id, 0) >= args.max_per_wsi:
                continue
            selected.append(row)
            wsi_counts[wsi_id] = wsi_counts.get(wsi_id, 0) + 1
            if len(selected) == args.count:
                break
        if len(selected) != args.count:
            raise ValueError(
                f"{backend}: selected {len(selected)} of {args.count} cases"
            )
        package_backend(selected, args.output_root, backend)
        summary["backends"][backend] = {
            "selected_count": len(selected),
            "unique_wsis": len({row["wsi_id"] for row in selected}),
            "selection_manifest": str(
                args.output_root / backend / "selection_manifest.json"
            ),
            "top_reference_ids": [row["reference_id"] for row in selected],
        }

    (args.output_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
