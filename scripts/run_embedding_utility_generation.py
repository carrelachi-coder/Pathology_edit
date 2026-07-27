#!/usr/bin/env python3
"""Generate one forced backend for the exploratory embedding-utility cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from controlnet_train.inference import (
    DEFAULT_CROSS_V1_CHECKPOINT,
    DEFAULT_INPAINT_CHECKPOINT,
    DEFAULT_PIX2PIX_CHECKPOINT,
)
from scripts.run_phase3_inpaint_pipeline import _run_generation_stage


DEFAULT_FLUX = Path("/data/huggingface/FLUX.1-dev")


def _generation_change_region(row: dict) -> Path:
    """Return the finalized generation support, with legacy fallback."""

    for field in (
        "inpaint_change_region",
        "generation_change_region",
        "change_region",
    ):
        value = row.get(field)
        if value:
            return Path(value)
    raise ValueError(
        f"{row.get('sample_id', 'unknown')}: no generation change-region path"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backend", required=True, choices=("inpaint", "cross-v1"))
    parser.add_argument("--pretrained-model", type=Path, default=DEFAULT_FLUX)
    parser.add_argument(
        "--inpaint-checkpoint", type=Path, default=Path(DEFAULT_INPAINT_CHECKPOINT)
    )
    parser.add_argument(
        "--cross-v1-checkpoint", type=Path, default=Path(DEFAULT_CROSS_V1_CHECKPOINT)
    )
    parser.add_argument(
        "--pix2pix-checkpoint", type=Path, default=Path(DEFAULT_PIX2PIX_CHECKPOINT)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--expected-count", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_rows(rows: list[dict]) -> None:
    required_paths = (
        "reference_image",
        "reference_tissue_mask",
        "reference_nuclei_mask",
        "target_tissue_mask",
        "target_nuclei_mask",
    )
    for row in rows:
        if not row.get("sample_id") or not row.get("wsi_id"):
            raise ValueError("every cohort row requires sample_id and wsi_id")
        for field in required_paths:
            value = row.get(field)
            if not value or not Path(value).is_file():
                raise FileNotFoundError(
                    f"{row.get('sample_id', 'unknown')}: missing {field}: {value}"
                )
        change_region = _generation_change_region(row)
        if not change_region.is_file():
            raise FileNotFoundError(
                f"{row.get('sample_id', 'unknown')}: missing generation "
                f"change region: {change_region}"
            )


def _pipeline_args(args: argparse.Namespace, row: dict, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        profile="BCSS",
        reference_image=Path(row["reference_image"]),
        reference_tissue_mask=Path(row["reference_tissue_mask"]),
        reference_nuclei_mask=Path(row["reference_nuclei_mask"]),
        generation_mode=args.backend,
        cross_backend="cross-v1",
        route_threshold=0.35,
        pretrained_model_name_or_path=args.pretrained_model,
        inpaint_checkpoint=args.inpaint_checkpoint,
        cross_v1_checkpoint=args.cross_v1_checkpoint,
        pix2pix_checkpoint=args.pix2pix_checkpoint,
        device=args.device,
        prompt=None,
        prompt_source="dataset",
        torch_dtype=args.dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        seed=int(row.get("generation_seed", 42)),
        color_match="none",
        output=output,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    all_rows = _read_jsonl(args.cohort_manifest)
    if args.limit is None and len(all_rows) != args.expected_count:
        raise ValueError(
            f"frozen cohort requires {args.expected_count} rows, found {len(all_rows)}"
        )
    rows = all_rows[: args.limit] if args.limit is not None else all_rows
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    _validate_rows(rows)
    for path in (
        args.pretrained_model,
        args.inpaint_checkpoint if args.backend == "inpaint" else args.cross_v1_checkpoint,
        args.pix2pix_checkpoint if args.backend == "cross-v1" else None,
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)

    backend_root = args.output_root / args.backend
    backend_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures = []
    for index, row in enumerate(rows, start=1):
        sample_dir = backend_root / row["sample_id"]
        generated_path = sample_dir / "generated_image.png"
        info_path = sample_dir / "generation_info.json"
        if args.resume and generated_path.is_file() and info_path.is_file():
            completed += 1
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            generation_change_region = _generation_change_region(row)
            with Image.open(row["reference_image"]) as image:
                reference_image = np.asarray(image.convert("RGB"))
            with Image.open(generation_change_region) as image:
                change_region = np.asarray(image.convert("L")) > 128
            output_path, info = _run_generation_stage(
                args=_pipeline_args(args, row, sample_dir),
                output_dir=sample_dir,
                reference_image=reference_image,
                change_region=change_region,
                target_tissue_path=Path(row["target_tissue_mask"]),
                target_nuclei_path=Path(row["target_nuclei_mask"]),
            )
            if output_path != generated_path or not generated_path.is_file():
                raise RuntimeError(f"unexpected generation output: {output_path}")
            provenance = {
                "sample_id": row["sample_id"],
                "wsi_id": row["wsi_id"],
                "backend": args.backend,
                "reference_image": row["reference_image"],
                "target_tissue_mask": row["target_tissue_mask"],
                "target_nuclei_mask": row["target_nuclei_mask"],
                "generation_change_region": str(generation_change_region),
                "generation_seed": int(row.get("generation_seed", 42)),
                "generation": info,
            }
            (sample_dir / "utility_generation.json").write_text(
                json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            completed += 1
        except Exception as exc:
            failure = {
                "sample_id": row.get("sample_id"),
                "backend": args.backend,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            (sample_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if index % 5 == 0 or index == len(rows):
            print(
                f"backend={args.backend} completed={completed} failed={len(failures)} "
                f"processed={index}/{len(rows)} shard={args.shard_index}/{args.num_shards}",
                flush=True,
            )
    summary = {
        "status": "complete" if not failures and completed == len(rows) else "incomplete",
        "backend": args.backend,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "requested_count": len(rows),
        "completed_count": completed,
        "failure_count": len(failures),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    suffix = f"shard{args.shard_index:02d}-of-{args.num_shards:02d}"
    (backend_root / f"generation_summary_{suffix}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (backend_root / f"generation_failures_{suffix}.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
