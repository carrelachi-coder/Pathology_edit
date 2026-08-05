#!/usr/bin/env python3
"""Generate the paired AutoCond Patho-KID cohort with production Cross V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_PIX2PIX_SHA256 = (
    "be5fe9376efdb5620a57481082f6d5738b6353796fb00fe6e58f6b212ba7c2ac"
)
EXPECTED_PIX2PIX_EPOCH = 26
EXPECTED_PIX2PIX_GLOBAL_STEP = 214895
EXPECTED_TRUST_GATE = "nuclei_reference_support_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--cross-v1-checkpoint", type=Path, required=True)
    parser.add_argument("--pix2pix-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--expected-count", type=int, default=1454)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"Unsupported manifest structure: {path}")
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def image_stats(image: Image.Image) -> dict[str, Any]:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pixels = array.reshape(-1, 3)
    return {
        "size": list(image.size),
        "mean": [round(float(value), 3) for value in pixels.mean(axis=0)],
        "std": [round(float(value), 3) for value in pixels.std(axis=0)],
        "min": [int(value) for value in pixels.min(axis=0)],
        "max": [int(value) for value in pixels.max(axis=0)],
    }


def validate_manifest(records: list[dict[str, Any]], expected_count: int) -> None:
    if len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} strict-Oral directions, found {len(records)}"
        )
    sample_ids = [str(record.get("sample_id", "")) for record in records]
    if not all(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Manifest sample_id values must be non-empty and unique")
    target_images = [str(record.get("target_image", "")) for record in records]
    if not all(target_images) or len(set(target_images)) != len(target_images):
        raise ValueError("Manifest target images must be non-empty and unique")
    pair_directions: dict[str, set[str]] = {}
    for record in records:
        pair_directions.setdefault(str(record.get("pair_id", "")), set()).add(
            str(record.get("direction", ""))
        )
        if not str(record.get("prompt", "")).strip():
            raise ValueError(f"{record['sample_id']}: prompt is empty")
        for field in (
            "reference_image",
            "reference_tissue_mask",
            "reference_nuclei_mask",
            "target_tissue_mask",
            "target_nuclei_mask",
        ):
            value = record.get(field)
            if not value or not Path(value).is_file():
                raise FileNotFoundError(
                    f"{record['sample_id']}: missing {field}: {value}"
                )
    invalid_pairs = {
        pair_id: sorted(directions)
        for pair_id, directions in pair_directions.items()
        if directions != {"a_to_b", "b_to_a"}
    }
    if invalid_pairs:
        raise ValueError(f"Manifest contains incomplete direction pairs: {invalid_pairs}")


def validate_release(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.manifest,
        args.pretrained_model,
        args.cross_v1_checkpoint,
        args.pix2pix_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    pix2pix_sha256 = sha256_file(args.pix2pix_checkpoint)
    if pix2pix_sha256 != EXPECTED_PIX2PIX_SHA256:
        raise ValueError(
            "Refusing non-release pix2pix checkpoint: "
            f"{pix2pix_sha256} != {EXPECTED_PIX2PIX_SHA256}"
        )
    release_manifest_path = args.cross_v1_checkpoint.parent / "manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    metadata = release_manifest.get("model_metadata", {})
    if (
        metadata.get("pix2pix_epoch") != EXPECTED_PIX2PIX_EPOCH
        or metadata.get("pix2pix_global_step") != EXPECTED_PIX2PIX_GLOBAL_STEP
        or metadata.get("trust_gate") != EXPECTED_TRUST_GATE
    ):
        raise ValueError(f"Cross release manifest is not the epoch-26 release: {metadata}")
    return {
        "release_manifest": str(release_manifest_path),
        "release_git_commit": release_manifest.get("git_commit"),
        "cross_v1_checkpoint": str(args.cross_v1_checkpoint),
        "pix2pix_checkpoint": str(args.pix2pix_checkpoint),
        "pix2pix_sha256": pix2pix_sha256,
        "pix2pix_epoch": EXPECTED_PIX2PIX_EPOCH,
        "pix2pix_global_step": EXPECTED_PIX2PIX_GLOBAL_STEP,
        "trust_gate": EXPECTED_TRUST_GATE,
    }


def output_is_complete(sample_dir: Path) -> bool:
    image_path = sample_dir / "generated.png"
    metadata_path = sample_dir / "metadata.json"
    if not image_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with Image.open(image_path) as image:
            if image.size != (512, 512):
                return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = metadata["generation"]["pix2pix_v2"]
        return bool(
            metadata.get("status") == "completed"
            and metadata.get("target_image_used_for_generation") is False
            and checkpoint.get("epoch") == EXPECTED_PIX2PIX_EPOCH
            and checkpoint.get("global_step") == EXPECTED_PIX2PIX_GLOBAL_STEP
            and checkpoint.get("trust_gate") == EXPECTED_TRUST_GATE
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def link_final_output(stage2_path: Path, generated_path: Path) -> None:
    temporary = generated_path.with_suffix(".png.tmp")
    temporary.unlink(missing_ok=True)
    os.link(stage2_path, temporary)
    temporary.replace(generated_path)


def audit_outputs(
    records: list[dict[str, Any]], output_root: Path
) -> dict[str, Any]:
    failures = []
    completed = 0
    organ_counts: dict[str, int] = {}
    for record in records:
        sample_dir = output_root / record["organ"] / record["sample_id"]
        if output_is_complete(sample_dir):
            completed += 1
            organ = str(record["organ"])
            organ_counts[organ] = organ_counts.get(organ, 0) + 1
        else:
            failures.append(record["sample_id"])
    report = {
        "schema_version": 1,
        "status": "complete" if not failures else "incomplete",
        "expected": len(records),
        "completed": completed,
        "missing_or_invalid": failures,
        "organ_counts": dict(sorted(organ_counts.items())),
        "audited_at_utc": utc_now(),
    }
    write_json_atomic(output_root / "validation.json", report)
    return report


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    if args.num_inference_steps != 28:
        raise ValueError("Formal Cross V1 Patho-KID generation requires 28 steps")
    release = validate_release(args)
    records = read_records(args.manifest)
    validate_manifest(records, args.expected_count)
    if args.audit_only:
        report = audit_outputs(records, args.output_root)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["status"] == "complete" else 2

    shard_records = [
        record
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_items is not None:
        shard_records = shard_records[: args.max_items]
    pending_records = []
    skipped = 0
    for record in shard_records:
        sample_dir = args.output_root / record["organ"] / record["sample_id"]
        if args.resume and output_is_complete(sample_dir):
            skipped += 1
        else:
            pending_records.append(record)

    shard_name = f"shard{args.shard_index:02d}-of-{args.num_shards:02d}"
    summary_path = args.output_root / "state" / f"{shard_name}.json"
    base_summary = {
        "schema_version": 1,
        "status": "running",
        "model_id": "cross_v1_no_ip_pix2pix_epoch26",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "output_root": str(args.output_root),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "requested": len(shard_records),
        "pending_at_start": len(pending_records),
        "skipped_complete": skipped,
        "release": release,
        "inference": {
            "loads_ip_adapter": False,
            "loads_uni": False,
            "stage1_reference_mask_mode": "target",
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
            "native_resolution": [512, 512],
            "native_mpp": 0.25,
        },
        "started_at_utc": utc_now(),
    }
    write_json_atomic(summary_path, base_summary)
    if not pending_records:
        base_summary.update(
            {"status": "complete", "completed_this_run": 0, "failures": []}
        )
        write_json_atomic(summary_path, base_summary)
        print(json.dumps(base_summary, indent=2), flush=True)
        return 0

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    from scripts.run_phase3_inpaint_pipeline import _run_cross_v1_no_ip_generation

    completed = 0
    failures = []
    for index, record in enumerate(pending_records, start=1):
        sample_dir = args.output_root / record["organ"] / record["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        error_path = sample_dir / "error.json"
        started = time.time()
        try:
            final_image, _, generation = _run_cross_v1_no_ip_generation(
                pretrained_model_name_or_path=args.pretrained_model,
                checkpoint_path=args.cross_v1_checkpoint,
                pix2pix_checkpoint_path=args.pix2pix_checkpoint,
                reference_image_path=record["reference_image"],
                reference_tissue_mask_path=record["reference_tissue_mask"],
                reference_nuclei_mask_path=record["reference_nuclei_mask"],
                target_tissue_mask_path=record["target_tissue_mask"],
                target_nuclei_mask_path=record["target_nuclei_mask"],
                generation_change_region_path=None,
                prompt=str(record["prompt"]),
                output_dir=sample_dir,
                device=args.device,
                torch_dtype=dtype,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                seed=int(record["seed"]),
            )
            pix2pix_info = generation["pix2pix_v2"]
            if (
                pix2pix_info.get("epoch") != EXPECTED_PIX2PIX_EPOCH
                or pix2pix_info.get("global_step")
                != EXPECTED_PIX2PIX_GLOBAL_STEP
                or pix2pix_info.get("trust_gate") != EXPECTED_TRUST_GATE
            ):
                raise RuntimeError(f"Unexpected pix2pix runtime: {pix2pix_info}")
            if final_image.size != (512, 512):
                raise RuntimeError(f"Unexpected output frame: {final_image.size}")
            generated_path = sample_dir / "generated.png"
            link_final_output(Path(generation["pix2pix_output"]), generated_path)
            metadata = {
                "schema_version": 1,
                "status": "completed",
                "model_id": "cross_v1_no_ip_pix2pix_epoch26",
                "sample_id": record["sample_id"],
                "pair_id": record["pair_id"],
                "direction": record["direction"],
                "organ": record["organ"],
                "wsi_id": record["wsi_id"],
                "seed": int(record["seed"]),
                "prompt": record["prompt"],
                "allowed_generation_inputs": [
                    "reference_image",
                    "reference_tissue_mask",
                    "reference_nuclei_mask",
                    "target_tissue_mask",
                    "target_nuclei_mask",
                    "prompt",
                ],
                "target_image_used_for_generation": False,
                "reference_image": record["reference_image"],
                "reference_tissue_mask": record["reference_tissue_mask"],
                "reference_nuclei_mask": record["reference_nuclei_mask"],
                "target_tissue_mask": record["target_tissue_mask"],
                "target_nuclei_mask": record["target_nuclei_mask"],
                "output": str(generated_path),
                "runtime_seconds": round(time.time() - started, 3),
                "image_stats": image_stats(final_image),
                "release": release,
                "generation": generation,
                "completed_at_utc": utc_now(),
            }
            write_json_atomic(sample_dir / "metadata.json", metadata)
            error_path.unlink(missing_ok=True)
            completed += 1
            print(
                f"[{index}/{len(pending_records)}] done {record['sample_id']} "
                f"({metadata['runtime_seconds']}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - isolate failures in a 1,454-case run
            failure = {
                "status": "failed",
                "sample_id": record.get("sample_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "failed_at_utc": utc_now(),
            }
            write_json_atomic(error_path, failure)
            failures.append(failure)
            print(
                f"[{index}/{len(pending_records)}] FAIL "
                f"{record.get('sample_id')}: {failure['error']}",
                flush=True,
            )
        base_summary.update(
            {
                "completed_this_run": completed,
                "failure_count": len(failures),
                "last_processed": record.get("sample_id"),
                "updated_at_utc": utc_now(),
            }
        )
        write_json_atomic(summary_path, base_summary)

    base_summary.update(
        {
            "status": "complete" if not failures else "incomplete",
            "completed_this_run": completed,
            "failure_count": len(failures),
            "failures": [failure["sample_id"] for failure in failures],
            "completed_at_utc": utc_now(),
        }
    )
    write_json_atomic(summary_path, base_summary)
    print(json.dumps(base_summary, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
