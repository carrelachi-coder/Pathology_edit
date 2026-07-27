#!/usr/bin/env python3
"""Prepare real WSI context images for the MuPaD image-conditioned baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image


GDC_FILES_URL = "https://api.gdc.cancer.gov/files"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path)
    parser.add_argument("--wsi-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gdc-manifest", type=Path)
    parser.add_argument("--source-inner-offset", type=int, default=80)
    parser.add_argument("--source-patch-size", type=int, default=512)
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--model-resolution", type=int, default=512)
    parser.add_argument("--central-max-absolute-error", type=int, default=1)
    parser.add_argument("--central-max-mae", type=float, default=1e-4)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_generation_manifest(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["records"], payload.get("provenance", {})


def load_patch_rows(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {row["annotation_id"]: row for row in rows}


def box_from_patch(row: dict, patch_size: int, inner_offset: int) -> list[int]:
    left = int(row["x"]) + inner_offset
    top = int(row["y"]) + inner_offset
    return [left, top, left + patch_size, top + patch_size]


def context_box_from_patch(
    row: dict, patch_size: int, context_size: int, inner_offset: int
) -> list[int]:
    patch_box = box_from_patch(row, patch_size, inner_offset)
    center_x = (patch_box[0] + patch_box[2]) // 2
    center_y = (patch_box[1] + patch_box[3]) // 2
    half = context_size // 2
    return [center_x - half, center_y - half, center_x + half, center_y + half]


def boxes_overlap(left: list[int], right: list[int]) -> bool:
    return (
        left[0] < right[2]
        and left[2] > right[0]
        and left[1] < right[3]
        and left[3] > right[1]
    )


def discover_wsis(roots: list[Path], required_names: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.svs"):
            if path.name in required_names and path.name not in found:
                found[path.name] = path
    return found


def query_gdc(names: list[str]) -> list[dict]:
    hits = []
    for start in range(0, len(names), 25):
        batch = names[start : start + 25]
        filters = {
            "op": "in",
            "content": {"field": "files.file_name", "value": batch},
        }
        query = urllib.parse.urlencode(
            {
                "filters": json.dumps(filters, separators=(",", ":")),
                "fields": "file_id,file_name,md5sum,file_size,state",
                "format": "JSON",
                "size": str(len(batch) + 5),
            }
        )
        with urllib.request.urlopen(f"{GDC_FILES_URL}?{query}", timeout=120) as response:
            hits.extend(json.load(response)["data"]["hits"])
    by_name = {str(hit["file_name"]): hit for hit in hits}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise RuntimeError(f"GDC did not resolve {len(missing)} WSI files: {missing[:5]}")
    return [by_name[name] for name in names]


def write_gdc_manifest(path: Path, names: list[str]) -> None:
    hits = query_gdc(names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "filename", "md5", "size", "state"])
        for hit in hits:
            writer.writerow(
                [
                    hit.get("file_id") or hit["id"],
                    hit["file_name"],
                    hit["md5sum"],
                    hit["file_size"],
                    hit["state"],
                ]
            )


def read_slide_metadata(slide) -> dict:
    return {
        "dimensions": list(slide.dimensions),
        "mpp_x": float(slide.properties.get("openslide.mpp-x", "nan")),
        "mpp_y": float(slide.properties.get("openslide.mpp-y", "nan")),
        "objective_power": slide.properties.get("openslide.objective-power"),
    }


def central_match_is_valid(
    absolute_error: np.ndarray, max_absolute_error: int, max_mae: float
) -> bool:
    return bool(
        int(absolute_error.max()) <= max_absolute_error
        and float(absolute_error.mean()) <= max_mae
    )


def main() -> int:
    args = parse_args()
    records, provenance = load_generation_manifest(args.generation_manifest)
    patch_manifest = args.patch_manifest or Path(provenance["patch_manifest"])
    patches = load_patch_rows(patch_manifest)
    required_names = {f"{record['wsi_id']}.svs" for record in records}
    wsi_paths = discover_wsis(args.wsi_root, required_names)
    missing_names = sorted(required_names - set(wsi_paths))
    if args.gdc_manifest is not None:
        write_gdc_manifest(args.gdc_manifest, missing_names)

    args.output_root.mkdir(parents=True, exist_ok=True)
    context_root = args.output_root / "contexts"
    exclusions = []
    eligible = []
    for record in records:
        reference = patches[record["reference_annotation_id"]]
        target = patches[record["target_annotation_id"]]
        context_box = context_box_from_patch(
            reference,
            args.source_patch_size,
            args.context_size,
            args.source_inner_offset,
        )
        target_box = box_from_patch(
            target, args.source_patch_size, args.source_inner_offset
        )
        if boxes_overlap(context_box, target_box):
            exclusions.append(record["sample_id"])
        else:
            eligible.append((record, reference, target, context_box, target_box))

    overlap_exclusion_path = args.output_root / "excluded_target_overlap.txt"
    overlap_exclusion_path.write_text(
        "".join(f"{item}\n" for item in exclusions), encoding="utf-8"
    )

    import openslide

    completed = 0
    skipped = 0
    missing_records = []
    boundary_exclusions = []
    failures = []
    for record, reference, target, context_box, target_box in eligible:
        sample_dir = context_root / record["organ"] / record["reference_annotation_id"]
        output_path = sample_dir / "context.png"
        metadata_path = sample_dir / "metadata.json"
        if output_path.exists() and metadata_path.exists() and not args.overwrite:
            skipped += 1
            continue
        wsi_name = f"{record['wsi_id']}.svs"
        wsi_path = wsi_paths.get(wsi_name)
        if wsi_path is None:
            missing_records.append(record["sample_id"])
            continue
        slide = None
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            slide_metadata = read_slide_metadata(slide)
            left, top, right, bottom = context_box
            if (
                left < 0
                or top < 0
                or right > slide.dimensions[0]
                or bottom > slide.dimensions[1]
            ):
                boundary_exclusions.append(record["sample_id"])
                continue
            region = slide.read_region(
                (left, top), 0, (args.context_size, args.context_size)
            ).convert("RGB")
            margin = (args.context_size - args.source_patch_size) // 2
            central = region.crop(
                (
                    margin,
                    margin,
                    margin + args.source_patch_size,
                    margin + args.source_patch_size,
                )
            )
            reference_image = Image.open(record["reference_image"]).convert("RGB")
            central_array = np.asarray(central, dtype=np.int16)
            reference_array = np.asarray(reference_image, dtype=np.int16)
            absolute_error = np.abs(central_array - reference_array)
            central_exact = bool(np.array_equal(central_array, reference_array))
            central_verified = central_match_is_valid(
                absolute_error,
                args.central_max_absolute_error,
                args.central_max_mae,
            )
            if not central_verified:
                raise ValueError(
                    "central WSI crop exceeds reference tolerance: "
                    f"mae={absolute_error.mean():.6f}, max={absolute_error.max()}"
                )
            condition = region.resize(
                (args.model_resolution, args.model_resolution), Image.Resampling.LANCZOS
            )
            condition = Image.fromarray(np.asarray(condition, dtype=np.uint8), mode="RGB")
            sample_dir.mkdir(parents=True, exist_ok=True)
            condition.save(output_path)
            source_mpp = float(record.get("source_mpp", 0.25))
            metadata = {
                "status": "completed",
                "sample_id": record["sample_id"],
                "reference_annotation_id": record["reference_annotation_id"],
                "target_annotation_id": record["target_annotation_id"],
                "organ": record["organ"],
                "wsi_id": record["wsi_id"],
                "wsi_path": str(wsi_path),
                "wsi": slide_metadata,
                "manifest_coordinate_xy": [int(reference["x"]), int(reference["y"])],
                "source_inner_offset_xy": [args.source_inner_offset] * 2,
                "reference_patch_box_level0_xyxy": box_from_patch(
                    reference, args.source_patch_size, args.source_inner_offset
                ),
                "source_context_box_level0_xyxy": context_box,
                "target_patch_box_level0_xyxy": target_box,
                "target_overlap": False,
                "source_context_resolution": [args.context_size, args.context_size],
                "source_context_mpp_nominal": source_mpp,
                "model_resolution": [args.model_resolution, args.model_resolution],
                "model_mpp_nominal": source_mpp
                * args.context_size
                / args.model_resolution,
                "context_operation": "real_wsi_centered_crop_then_downsample",
                "central_reference_exact": central_exact,
                "central_reference_verified": central_verified,
                "central_reference_mae": float(absolute_error.mean()),
                "central_reference_max_absolute_error": int(absolute_error.max()),
                "central_reference_tolerance": {
                    "max_absolute_error": args.central_max_absolute_error,
                    "max_mae": args.central_max_mae,
                },
                "resize_filter": "LANCZOS",
                "output": str(output_path),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            completed += 1
        except Exception as exc:
            failures.append(
                {
                    "sample_id": record["sample_id"],
                    "reference_annotation_id": record["reference_annotation_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if slide is not None:
                slide.close()

    combined_exclusions = sorted(set(exclusions + boundary_exclusions))
    boundary_exclusion_path = args.output_root / "excluded_wsi_boundary.txt"
    boundary_exclusion_path.write_text(
        "".join(f"{item}\n" for item in sorted(boundary_exclusions)),
        encoding="utf-8",
    )
    combined_exclusion_path = args.output_root / "excluded_mupad_image.txt"
    combined_exclusion_path.write_text(
        "".join(f"{item}\n" for item in combined_exclusions), encoding="utf-8"
    )

    summary = {
        "schema_version": 1,
        "generation_manifest": str(args.generation_manifest),
        "generation_manifest_sha256": sha256(args.generation_manifest),
        "patch_manifest": str(patch_manifest),
        "patch_manifest_sha256": sha256(patch_manifest),
        "wsi_roots": [str(path) for path in args.wsi_root],
        "required_wsi_count": len(required_names),
        "available_wsi_count": len(wsi_paths),
        "missing_wsi_count": len(missing_names),
        "missing_wsi_filenames": missing_names,
        "total_direction_count": len(records),
        "excluded_target_overlap_count": len(exclusions),
        "excluded_target_overlap_file": str(overlap_exclusion_path),
        "excluded_wsi_boundary_count": len(boundary_exclusions),
        "excluded_wsi_boundary_file": str(boundary_exclusion_path),
        "combined_exclusion_count": len(combined_exclusions),
        "combined_exclusion_file": str(combined_exclusion_path),
        "eligible_direction_count": len(records) - len(combined_exclusions),
        "completed_this_run": completed,
        "skipped_complete": skipped,
        "missing_context_records": missing_records,
        "failures": failures,
        "valid": not failures and not missing_records,
    }
    (args.output_root / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        key: summary[key]
        for key in (
            "required_wsi_count",
            "available_wsi_count",
            "missing_wsi_count",
            "total_direction_count",
            "excluded_target_overlap_count",
            "excluded_wsi_boundary_count",
            "combined_exclusion_count",
            "eligible_direction_count",
            "completed_this_run",
            "skipped_complete",
            "valid",
        )
    }
    report["missing_context_count"] = len(missing_records)
    report["failure_count"] = len(failures)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if failures:
        return 1
    if missing_records and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
