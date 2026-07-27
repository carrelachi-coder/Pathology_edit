#!/usr/bin/env python3
"""Normalize generation outputs to the shared 512-pixel evaluation grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


MODEL_SPECS = {
    "cross_v1_project": {
        "source": "cross",
        "filename": "stage2_pix2pix_pilot_step001000_latest.png",
        "native_resolution": 512,
        "native_mpp": 0.25,
    },
    "pixcell_controlnet": {
        "source": "baseline",
        "filename": "generated.png",
        "native_resolution": 256,
        "native_mpp": 0.5,
    },
    "pathdiff_conic": {
        "source": "baseline",
        "filename": "generated.png",
        "native_resolution": 256,
        "native_mpp": 0.5,
    },
    "pathdiff_text": {
        "source": "baseline",
        "filename": "generated.png",
        "native_resolution": 256,
        "native_mpp": 0.5,
        "normalization_strategy": "physical_fov_center_crop",
        "physical_scale_status": "prompt_conditioned_nominal_scale",
    },
    "pathldm_plip": {
        "source": "baseline",
        "filename": "generated.png",
        "native_resolution": 256,
        "native_mpp": 1.0,
        "organs": {"breast"},
    },
    "unipath_7b": {
        "source": "baseline",
        "filename": "generated.png",
        "native_resolution": 384,
        "native_mpp": 0.5,
    },
    "mupad_text": {
        "source": "baseline",
        "filename": "generated.png",
        "native_resolution": 512,
        "native_mpp": 0.5,
    },
    "mupad_image_auxiliary": {
        "source": "baseline",
        "filename": "generated.png",
        "native_resolution": 512,
        "native_mpp": 0.5,
        "exclude_sample_ids": (
            "/data1/zhao/wqx/benchmarks/data/complex_paired_v3_1500/"
            "mupad_wsi_context/excluded_mupad_image.txt"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cross-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-resolution", type=int, default=512)
    parser.add_argument("--target-mpp", type=float, default=0.25)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["records"] if isinstance(payload, dict) else payload


def source_path(
    record: dict, model_id: str, spec: dict, cross_root: Path, baseline_root: Path
) -> Path:
    root = cross_root if spec["source"] == "cross" else baseline_root / model_id
    return root / record["organ"] / record["sample_id"] / spec["filename"]


def records_for_spec(records: list[dict], spec: dict) -> list[dict]:
    selected = records
    if spec.get("organs"):
        selected = [record for record in selected if record["organ"] in spec["organs"]]
    exclusion_path = spec.get("exclude_sample_ids")
    if exclusion_path:
        excluded = {
            line.strip()
            for line in Path(exclusion_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        selected = [record for record in selected if record["sample_id"] not in excluded]
    return selected


def normalize_image(
    image: Image.Image,
    native_mpp: float | None,
    target_resolution: int,
    target_mpp: float,
    strategy: str = "physical_fov_center_crop",
) -> tuple[Image.Image, list[int]]:
    if strategy == "full_frame_resize_unknown_scale":
        box = [0, 0, image.width, image.height]
        normalized = image
        if normalized.size != (target_resolution, target_resolution):
            normalized = normalized.resize(
                (target_resolution, target_resolution), LANCZOS
            )
        return normalized, box
    if strategy != "physical_fov_center_crop":
        raise ValueError(f"Unknown normalization strategy: {strategy}")
    if native_mpp is None:
        raise ValueError("native_mpp is required for physical FOV normalization")
    target_fov_um = target_resolution * target_mpp
    crop_pixels = round(target_fov_um / native_mpp)
    if crop_pixels > min(image.size):
        raise ValueError(
            f"Native image {image.size} at {native_mpp} MPP cannot cover "
            f"the target {target_fov_um} um field"
        )
    left = (image.width - crop_pixels) // 2
    top = (image.height - crop_pixels) // 2
    box = [left, top, left + crop_pixels, top + crop_pixels]
    cropped = image.crop(tuple(box))
    if cropped.size != (target_resolution, target_resolution):
        cropped = cropped.resize(
            (target_resolution, target_resolution), LANCZOS
        )
    return cropped, box


def main() -> int:
    args = parse_args()
    unknown = sorted(set(args.models) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown model IDs: {unknown}")
    records = load_records(args.manifest)
    failures = []
    counts = {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for model_id in args.models:
        spec = MODEL_SPECS[model_id]
        count = 0
        for record in records_for_spec(records, spec):
            source = source_path(
                record, model_id, spec, args.cross_root, args.baseline_root
            )
            sample_dir = (
                args.output_root / model_id / record["organ"] / record["sample_id"]
            )
            output = sample_dir / "generated.png"
            metadata_path = sample_dir / "normalization.json"
            if output.exists() and metadata_path.exists() and not args.overwrite:
                count += 1
                continue
            try:
                if not source.exists():
                    raise FileNotFoundError(source)
                image = Image.open(source).convert("RGB")
                expected = (spec["native_resolution"], spec["native_resolution"])
                if image.size != expected:
                    raise ValueError(
                        f"Unexpected native resolution {image.size}; expected {expected}"
                    )
                normalized, crop_box = normalize_image(
                    image,
                    spec["native_mpp"],
                    args.target_resolution,
                    args.target_mpp,
                    spec.get(
                        "normalization_strategy", "physical_fov_center_crop"
                    ),
                )
                sample_dir.mkdir(parents=True, exist_ok=True)
                normalized.save(output)
                scale_is_known = spec["native_mpp"] is not None
                metadata = {
                    "status": "completed",
                    "model_id": model_id,
                    "sample_id": record["sample_id"],
                    "organ": record["organ"],
                    "source": str(source),
                    "native_resolution": list(image.size),
                    "native_mpp": spec["native_mpp"],
                    "native_fov_um": image.width * spec["native_mpp"]
                    if scale_is_known
                    else None,
                    "center_crop_box_xyxy": crop_box,
                    "crop_applied": crop_box != [0, 0, image.width, image.height],
                    "normalization_strategy": spec.get(
                        "normalization_strategy", "physical_fov_center_crop"
                    ),
                    "physical_scale_status": spec.get(
                        "physical_scale_status",
                        "known"
                        if scale_is_known
                        else "unknown_pathcap_no_mpp_normalization",
                    ),
                    "target_resolution": [args.target_resolution, args.target_resolution],
                    "target_mpp": args.target_mpp if scale_is_known else None,
                    "target_fov_um": args.target_resolution * args.target_mpp
                    if scale_is_known
                    else None,
                    "requested_target_mpp": args.target_mpp,
                    "requested_target_fov_um": args.target_resolution
                    * args.target_mpp,
                    "resize_filter": "LANCZOS",
                    "output": str(output),
                }
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                count += 1
            except Exception as exc:
                failures.append(
                    {
                        "model_id": model_id,
                        "sample_id": record["sample_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        counts[model_id] = count
        print(f"{model_id}: normalized={count}", flush=True)

    summary = {
        "manifest": str(args.manifest),
        "cross_root": str(args.cross_root),
        "baseline_root": str(args.baseline_root),
        "output_root": str(args.output_root),
        "target_resolution": args.target_resolution,
        "target_mpp": args.target_mpp,
        "target_fov_um": args.target_resolution * args.target_mpp,
        "counts": counts,
        "failures": failures,
        "valid": not failures,
    }
    (args.output_root / "normalization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
