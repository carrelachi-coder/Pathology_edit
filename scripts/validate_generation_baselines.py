#!/usr/bin/env python3
"""Validate raw generation provenance and MPP-normalized preview outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


MUPAD_IMAGE_EXCLUSIONS = (
    "/data1/zhao/wqx/benchmarks/data/complex_paired_v3_1500/"
    "mupad_wsi_context/excluded_mupad_image.txt"
)


RAW_SPECS = {
    "pixcell_controlnet": {
        "inputs": ["reference_image", "target_nuclei_mask"],
        "resolution": 256,
        "mpp": 0.5,
        "steps": 20,
        "guidance": 2.5,
    },
    "pathdiff_conic": {
        "inputs": ["prompt", "target_conic_instance_type_mask"],
        "resolution": 256,
        "mpp": 0.5,
        "steps": 200,
        "guidance": 1.75,
    },
    "pathdiff_text": {
        "inputs": ["prompt"],
        "resolution": 256,
        "mpp": 0.5,
        "steps": 200,
        "guidance": 1.75,
    },
    "pathldm_plip": {
        "inputs": ["prompt"],
        "resolution": 256,
        "mpp": 1.0,
        "steps": 50,
        "guidance": 1.5,
        "organs": {"breast"},
    },
    "unipath_7b": {
        "inputs": ["prompt"],
        "resolution": 384,
        "mpp": 0.5,
        "steps": 30,
        "guidance": 3.0,
    },
    "mupad_text": {
        "inputs": ["prompt"],
        "resolution": 512,
        "mpp": 0.5,
        "steps": 250,
        "guidance": 2.5,
    },
    "mupad_image_auxiliary": {
        "inputs": ["reference_wsi_context"],
        "resolution": 512,
        "mpp": 0.5,
        "steps": 250,
        "guidance": 2.5,
        "exclude_sample_ids": MUPAD_IMAGE_EXCLUSIONS,
    },
}

NORMALIZED_SPECS = {
    "cross_v1_project": {"crop": [0, 0, 512, 512]},
    "pixcell_controlnet": {"crop": [0, 0, 256, 256]},
    "pathdiff_conic": {"crop": [0, 0, 256, 256]},
    "pathdiff_text": {
        "crop": [0, 0, 256, 256],
        "normalization_strategy": "physical_fov_center_crop",
        "physical_scale_status": "prompt_conditioned_nominal_scale",
    },
    "pathldm_plip": {"crop": [64, 64, 192, 192], "organs": {"breast"}},
    "unipath_7b": {"crop": [64, 64, 320, 320]},
    "mupad_text": {"crop": [128, 128, 384, 384]},
    "mupad_image_auxiliary": {
        "crop": [128, 128, 384, 384],
        "exclude_sample_ids": MUPAD_IMAGE_EXCLUSIONS,
    },
}

LATEST_CROSS_CHECKPOINT = (
    "/data/wqx/flowedit/"
    "pix2pix_texture_transfer_lazy_ver4_wsi_identity_i0_local_full_pyramid_v3_ft/"
    "ckpt/pilot_step001000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(RAW_SPECS))
    parser.add_argument("--cross-root", type=Path)
    parser.add_argument("--normalized-root", type=Path)
    parser.add_argument("--preview-root", type=Path)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["records"] if isinstance(payload, dict) else payload


def records_for_spec(records: list[dict], spec: dict) -> list[dict]:
    organs = spec.get("organs")
    selected = [record for record in records if not organs or record["organ"] in organs]
    exclusion_path = spec.get("exclude_sample_ids")
    if exclusion_path:
        excluded = {
            line.strip()
            for line in Path(exclusion_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        selected = [record for record in selected if record["sample_id"] not in excluded]
    return selected


def validate_raw_model(
    model_id: str, records: list[dict], output_root: Path
) -> dict:
    spec = RAW_SPECS[model_id]
    expected_records = records_for_spec(records, spec)
    expected = {record["sample_id"]: record for record in expected_records}
    failures = []
    found = set()
    organ_counts = {}
    model_root = output_root / model_id
    present = {
        path.parent.name for path in model_root.glob("*/*/metadata.json")
    }
    extras = sorted(present - set(expected))
    if extras:
        failures.append(f"unexpected samples: {extras}")

    for sample_id, record in expected.items():
        sample_dir = model_root / record["organ"] / sample_id
        image_path = sample_dir / "generated.png"
        metadata_path = sample_dir / "metadata.json"
        if not image_path.exists() or not metadata_path.exists():
            failures.append(f"{sample_id}: missing image or metadata")
            continue
        try:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_shape = (spec["resolution"], spec["resolution"], 3)
            if image.shape != expected_shape or float(image.std()) < 1.0:
                failures.append(
                    f"{sample_id}: invalid image shape/std {image.shape}/{image.std():.3f}"
                )
            if metadata.get("status") != "completed":
                failures.append(f"{sample_id}: metadata status is not completed")
            if metadata.get("sample_id") != sample_id:
                failures.append(f"{sample_id}: metadata sample_id mismatch")
            if metadata.get("target_image_used_for_generation") is not False:
                failures.append(f"{sample_id}: target RGB provenance violation")
            if metadata.get("allowed_generation_inputs") != spec["inputs"]:
                failures.append(f"{sample_id}: generation input contract mismatch")
            if metadata.get("native_output_resolution") != [
                spec["resolution"],
                spec["resolution"],
            ]:
                failures.append(f"{sample_id}: native resolution metadata mismatch")
            if metadata.get("native_output_mpp") != spec["mpp"]:
                failures.append(f"{sample_id}: native MPP metadata mismatch")
            if metadata.get("steps") != spec["steps"]:
                failures.append(f"{sample_id}: sampling steps mismatch")
            if metadata.get("guidance") != spec["guidance"]:
                failures.append(f"{sample_id}: guidance mismatch")
            if not metadata.get("model_checkpoint") or not metadata.get("command"):
                failures.append(f"{sample_id}: missing execution provenance")
            if model_id == "pixcell_controlnet":
                prep = metadata.get("target_mask_preprocessing", {})
                if prep.get("crop_applied") is not False or prep.get("model_mpp") != 0.5:
                    failures.append(f"{sample_id}: invalid PixCell mask preprocessing")
            if model_id == "pathdiff_conic":
                prep = metadata.get("target_condition_preprocessing", {})
                conic_path = Path(metadata.get("conic_mask", ""))
                if prep.get("crop_applied") is not False or prep.get("model_mpp") != 0.5:
                    failures.append(f"{sample_id}: invalid PathDiff preprocessing")
                if not conic_path.exists() or np.load(conic_path).shape != (256, 256, 2):
                    failures.append(f"{sample_id}: invalid CoNIC cell condition")
                else:
                    conic_metadata = json.loads(
                        (conic_path.parent / "metadata.json").read_text(encoding="utf-8")
                    )
                    if (
                        conic_metadata.get("source_mpp") != 0.25
                        or conic_metadata.get("model_mpp") != 0.5
                        or conic_metadata.get("crop_applied") is not False
                    ):
                        failures.append(f"{sample_id}: invalid CoNIC MPP provenance")
            if model_id == "pathdiff_text":
                condition = metadata.get("pathdiff_control_condition", {})
                if metadata.get("pathdiff_inference_mode") != "t2i":
                    failures.append(f"{sample_id}: PathDiff text mode mismatch")
                if metadata.get("pathdiff_official_entrypoint") != (
                    "sampling.py::sample_one(mode='t2i')"
                ):
                    failures.append(f"{sample_id}: PathDiff text entrypoint mismatch")
                if condition != {
                    "kind": "official_null_mask",
                    "value": 10,
                    "shape": [256, 256, 6],
                }:
                    failures.append(f"{sample_id}: invalid PathDiff NULL_MASK")
                scale_condition = metadata.get("pathdiff_text_scale_condition", {})
                if metadata.get("native_scale_status") != (
                    "prompt_conditioned_nominal_scale"
                ):
                    failures.append(f"{sample_id}: invalid PathDiff scale provenance")
                if scale_condition.get("kind") != (
                    "prompt_conditioned_objective_magnification"
                ):
                    failures.append(f"{sample_id}: missing PathDiff text scale condition")
                if (
                    scale_condition.get("objective_magnification") != 20.0
                    or scale_condition.get("nominal_mpp") != 0.5
                    or scale_condition.get("prompt_position") != "prefix"
                ):
                    failures.append(f"{sample_id}: invalid PathDiff text scale condition")
                effective_prompt = metadata.get("effective_prompt", "")
                if not effective_prompt.startswith(
                    "H&E-stained histopathology at 20x objective magnification."
                ):
                    failures.append(f"{sample_id}: PathDiff scale prefix missing")
                if metadata.get("conic_mask") is not None:
                    failures.append(f"{sample_id}: text-only run used a CoNIC mask")
            if model_id == "pathldm_plip":
                tumor_level = metadata.get("pathldm_tumor_level")
                til_level = metadata.get("pathldm_til_level")
                prefix = metadata.get("pathldm_conditioning_prefix")
                if tumor_level not in {"low", "high"} or til_level not in {
                    "low",
                    "high",
                }:
                    failures.append(f"{sample_id}: invalid PathLDM condition levels")
                else:
                    expected_prefix = (
                        f"{tumor_level.capitalize()} tumor; {til_level} TIL;"
                    )
                    if prefix != expected_prefix:
                        failures.append(f"{sample_id}: PathLDM prefix mismatch")
                    if metadata.get("effective_prompt") != prefix + metadata.get(
                        "prompt", ""
                    ):
                        failures.append(
                            f"{sample_id}: PathLDM effective prompt mismatch"
                        )
            if model_id == "mupad_image_auxiliary":
                prep = metadata.get("reference_preprocessing", {})
                central_verified = prep.get(
                    "central_reference_verified", prep.get("central_reference_exact")
                )
                if (
                    prep.get("source_context_resolution") != [1024, 1024]
                    or prep.get("model_resolution") != [512, 512]
                    or prep.get("context_operation")
                    != "real_wsi_centered_crop_then_downsample"
                    or prep.get("model_mpp_nominal") != 0.5
                    or central_verified is not True
                    or prep.get("target_overlap") is not False
                    or not Path(prep.get("wsi_path", "")).is_file()
                ):
                    failures.append(f"{sample_id}: invalid MuPaD-I2I preprocessing")
            found.add(sample_id)
            organ = record["organ"]
            organ_counts[organ] = organ_counts.get(organ, 0) + 1
        except Exception as exc:
            failures.append(f"{sample_id}: {type(exc).__name__}: {exc}")

    failures.extend(
        f"unexpected error file: {path}"
        for path in model_root.glob("*/*/error.json")
    )
    return {
        "expected": len(expected),
        "completed": len(found),
        "missing": sorted(set(expected) - found),
        "organ_counts": dict(sorted(organ_counts.items())),
        "failures": failures,
        "valid": len(found) == len(expected) and not failures,
    }


def validate_cross(records: list[dict], cross_root: Path) -> dict:
    failures = []
    completed = 0
    for record in records:
        sample_dir = cross_root / record["organ"] / record["sample_id"]
        image_path = sample_dir / "stage2_pix2pix_pilot_step001000_latest.png"
        summary_path = sample_dir / "pix2pix_pilot_step001000_latest_summary.json"
        try:
            if Image.open(image_path).size != (512, 512):
                failures.append(f"{record['sample_id']}: invalid Cross image resolution")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint = summary.get("checkpoint", {})
            protocol = summary.get("inference_protocol", {})
            if checkpoint.get("checkpoint") != LATEST_CROSS_CHECKPOINT:
                failures.append(f"{record['sample_id']}: stale Cross checkpoint")
            if checkpoint.get("use_wsi_identity") is not True:
                failures.append(f"{record['sample_id']}: WSI identity disabled")
            if checkpoint.get("trust_gate") != "nuclei_reference_support_v2":
                failures.append(f"{record['sample_id']}: nuclei trust gate mismatch")
            if checkpoint.get("nuclei_reference_trust", {}).get("enabled") is not True:
                failures.append(f"{record['sample_id']}: nuclei trust disabled")
            if protocol.get("loads_ip_adapter") is not False:
                failures.append(f"{record['sample_id']}: IP-Adapter was loaded")
            if protocol.get("loads_uni") is not False:
                failures.append(f"{record['sample_id']}: UNI was loaded")
            completed += 1
        except Exception as exc:
            failures.append(f"{record['sample_id']}: {type(exc).__name__}: {exc}")
    return {
        "expected": len(records),
        "completed": completed,
        "failures": failures,
        "valid": completed == len(records) and not failures,
    }


def validate_normalized(
    records: list[dict], normalized_root: Path, models: list[str]
) -> dict:
    failures = []
    counts = {}
    for model_id in models:
        spec = NORMALIZED_SPECS[model_id]
        model_records = records_for_spec(records, spec)
        count = 0
        for record in model_records:
            sample_dir = (
                normalized_root / model_id / record["organ"] / record["sample_id"]
            )
            try:
                image = Image.open(sample_dir / "generated.png")
                metadata = json.loads(
                    (sample_dir / "normalization.json").read_text(encoding="utf-8")
                )
                if image.size != (512, 512):
                    failures.append(f"{model_id}/{record['sample_id']}: not 512x512")
                expected_target_mpp = spec.get("target_mpp", 0.25)
                expected_target_fov = spec.get("target_fov_um", 128.0)
                if metadata.get("target_mpp") != expected_target_mpp:
                    failures.append(f"{model_id}/{record['sample_id']}: target MPP mismatch")
                if metadata.get("target_fov_um") != expected_target_fov:
                    failures.append(f"{model_id}/{record['sample_id']}: target FOV mismatch")
                if metadata.get("center_crop_box_xyxy") != spec["crop"]:
                    failures.append(f"{model_id}/{record['sample_id']}: crop box mismatch")
                if "normalization_strategy" in spec and metadata.get(
                    "normalization_strategy"
                ) != spec["normalization_strategy"]:
                    failures.append(
                        f"{model_id}/{record['sample_id']}: normalization strategy mismatch"
                    )
                if "physical_scale_status" in spec and metadata.get(
                    "physical_scale_status"
                ) != spec["physical_scale_status"]:
                    failures.append(
                        f"{model_id}/{record['sample_id']}: scale status mismatch"
                    )
                count += 1
            except Exception as exc:
                failures.append(
                    f"{model_id}/{record['sample_id']}: {type(exc).__name__}: {exc}"
                )
        counts[model_id] = count
    return {"counts": counts, "failures": failures, "valid": not failures}


def validate_preview(preview_root: Path) -> dict:
    failures = []
    try:
        payload = json.loads(
            (preview_root / "preview_manifest.json").read_text(encoding="utf-8")
        )
        records = payload["records"]
        counts = {}
        for record in records:
            counts[record["organ"]] = counts.get(record["organ"], 0) + 1
            if record["organ"] != "breast" and any(
                "PathLDM" in column for column in record["columns"]
            ):
                failures.append(f"{record['sample_id']}: PathLDM outside breast")
        expected = {organ: (10 if organ == "breast" else 3) for organ in counts}
        if counts != expected:
            failures.append(f"preview counts mismatch: {counts} != {expected}")
        for sheet in preview_root.glob("*.jpg"):
            image = Image.open(sheet)
            if min(image.size) <= 0:
                failures.append(f"invalid sheet: {sheet}")
        if len(list(preview_root.glob("*.jpg"))) != 6:
            failures.append("expected six organ sheets")
    except Exception as exc:
        failures.append(f"preview: {type(exc).__name__}: {exc}")
        counts = {}
    return {"organ_counts": counts, "failures": failures, "valid": not failures}


def main() -> int:
    args = parse_args()
    records = load_records(args.manifest)
    report = {
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "models": {},
        "valid": True,
    }
    for model_id in args.models:
        if model_id not in RAW_SPECS:
            raise ValueError(f"Unknown model ID: {model_id}")
        model_report = validate_raw_model(model_id, records, args.output_root)
        report["models"][model_id] = model_report
        report["valid"] = report["valid"] and model_report["valid"]
        print(
            f"{model_id}: {model_report['completed']}/{model_report['expected']}, "
            f"failures={len(model_report['failures'])}",
            flush=True,
        )
    if args.cross_root:
        report["cross"] = validate_cross(records, args.cross_root)
        report["valid"] = report["valid"] and report["cross"]["valid"]
    if args.normalized_root:
        report["normalized"] = validate_normalized(
            records, args.normalized_root, args.models
        )
        report["valid"] = report["valid"] and report["normalized"]["valid"]
    if args.preview_root:
        report["preview"] = validate_preview(args.preview_root)
        report["valid"] = report["valid"] and report["preview"]["valid"]
    if args.write_report:
        report_path = (args.normalized_root or args.output_root) / "validation.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(report_path, flush=True)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
