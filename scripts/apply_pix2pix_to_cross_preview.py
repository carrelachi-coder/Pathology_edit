#!/usr/bin/env python3
"""Apply one pix2pix checkpoint to an existing Cross V1 preview run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import torch
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-name", default="stage2_pix2pix_candidate.png")
    parser.add_argument("--summary-name", default="pix2pix_candidate_summary.json")
    parser.add_argument("--batch-summary-name", default="pix2pix_candidate_batch_summary.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from controlnet_train.pix2pix_transfer.inference import (
        load_pix2pix_postprocessor,
        run_pix2pix_postprocess,
    )

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.torch_dtype]
    payload = json.loads(args.metadata.read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    if args.max_items > 0:
        records = records[: args.max_items]
    bundle = load_pix2pix_postprocessor(
        args.checkpoint,
        device=args.device,
        torch_dtype=dtype,
    )
    completed = 0
    skipped = 0
    failures = []
    for index, record in enumerate(records):
        sample_dir = args.run_dir / record["organ"] / record["sample_id"]
        stage1_path = sample_dir / "stage1_no_ip.png"
        output_path = sample_dir / args.output_name
        summary_path = sample_dir / args.summary_name
        if output_path.is_file() and summary_path.is_file():
            skipped += 1
            print(f"[{index + 1}/{len(records)}] skip {record['sample_id']}", flush=True)
            continue
        started = time.time()
        try:
            with Image.open(stage1_path) as image:
                prediction, info = run_pix2pix_postprocess(
                    bundle=bundle,
                    i0_image=image.convert("RGB"),
                    reference_image_path=record["reference_image"],
                    target_tissue_mask_path=record["target_tissue_mask"],
                    target_nuclei_mask_path=record["target_nuclei_mask"],
                    reference_tissue_mask_path=record["reference_tissue_mask"],
                    reference_nuclei_mask_path=record["reference_nuclei_mask"],
                    image_size=512,
                    device=args.device,
                    torch_dtype=dtype,
                )
            prediction.save(output_path)
            summary = {
                "sample_id": record["sample_id"],
                "elapsed_seconds": time.time() - started,
                "input_stage1": str(stage1_path),
                "output": str(output_path),
                "checkpoint": info,
                "inference_protocol": {
                    "loads_ip_adapter": False,
                    "loads_uni": False,
                    "trust_gate": info["trust_gate"],
                },
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            completed += 1
            print(
                f"[{index + 1}/{len(records)}] complete {record['sample_id']} "
                f"elapsed={summary['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "sample_id": record["sample_id"],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(failure["traceback"], flush=True)
    batch_summary = {
        "records": len(records),
        "completed_this_run": completed,
        "skipped_complete": skipped,
        "failures": failures,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": bundle.epoch,
        "checkpoint_global_step": bundle.global_step,
        "use_wsi_identity": bundle.config.use_wsi_identity,
        "trust_gate": (
            "nuclei_reference_support_v2"
            if bundle.config.highres_nuclei_trust_enabled
            and bundle.config.full_pyramid_texture_steering
            else "removed_from_production_inference"
        ),
    }
    summary_output = args.run_dir / args.batch_summary_name
    summary_output.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
