"""Precompute ControlNet I0 images for pix2pix texture-transfer training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controlnet_train.pix2pix_transfer.dataset import (
    i0_cache_path,
    metadata_cache_id,
    read_metadata,
    resolve_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True, help="I0 cache directory.")
    parser.add_argument("--metadata-root", default=None)
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V1 ControlNet checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument("--source-latent-init-strength", type=float, default=0.0)
    parser.add_argument("--mask-chord-scale", type=float, default=0.0)
    parser.add_argument("--mask-chord-use-gate", action="store_true")
    parser.add_argument("--mask-chord-gate-dilate-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-feather-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-outside-scale", type=float, default=0.0)
    parser.add_argument("--prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0, help="Start from this metadata row.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _dtype_by_name(name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[str(name)]


def _resolve_record_path(record: dict[str, Any], field: str, metadata_root: Path) -> Path:
    if field not in record or not record[field]:
        raise KeyError(f"Metadata record is missing required field {field!r}")
    return resolve_path(record[field], metadata_root=metadata_root)


def _resolve_prompt(
    *,
    record: dict[str, Any],
    prompt_override: str | None,
    prompt_source: str,
    default_prompt_for_dataset,
) -> str:
    if prompt_override:
        return prompt_override
    if prompt_source == "metadata":
        prompt = record.get("prompt")
        if prompt:
            return str(prompt)
    if prompt_source == "dataset":
        dataset = record.get("dataset")
        if dataset:
            return default_prompt_for_dataset(str(dataset))
    return str(record.get("prompt") or "H&E stained cancer histopathology at 40x magnification")


def _record_identity(record: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "metadata_index": metadata_cache_id(record, index),
        "sample_id": str(record.get("sample_id") or ""),
        "reference_sample_id": str(record.get("reference_sample_id") or ""),
        "dataset": str(record.get("dataset") or ""),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata)
    metadata_root = Path(args.metadata_root) if args.metadata_root else metadata_path.parent
    records = read_metadata(metadata_path)
    for original_index, record in enumerate(records):
        record.setdefault("metadata_index", original_index)

    start = max(0, int(args.start))
    stop = len(records)
    if args.max_samples is not None:
        stop = min(stop, start + int(args.max_samples))
    selected = list(enumerate(records))[start:stop]

    import torch

    from controlnet_train.data.common import (
        default_prompt_for_dataset,
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v1 import (
        load_cross_v1_bundle,
        run_cross_v1_bundle,
    )

    bundle = load_cross_v1_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        uni_checkpoint_path=args.uni_checkpoint_path,
        device=args.device,
        torch_dtype=_dtype_by_name(args.torch_dtype),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=args.ip_scale,
    )

    manifest_path = output_dir / "manifest.jsonl"
    written = 0
    skipped = 0
    failed = 0
    with manifest_path.open("a", encoding="utf8") as manifest:
        for offset, (index, record) in enumerate(selected, start=1):
            out_path = i0_cache_path(output_dir, record, index)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                if offset % max(1, int(args.save_every)) == 0:
                    print(
                        f"[{offset}/{len(selected)}] skipped={skipped} "
                        f"written={written} failed={failed}"
                    )
                continue

            try:
                reference_image = load_image_tensor(
                    _resolve_record_path(record, "reference_image", metadata_root)
                )
                reference_tissue_mask = load_tissue_mask(
                    _resolve_record_path(record, "reference_tissue_mask", metadata_root)
                )
                reference_nuclei_mask = load_nuclei_mask(
                    _resolve_record_path(record, "reference_nuclei_mask", metadata_root)
                )
                target_tissue_mask = load_tissue_mask(
                    _resolve_record_path(record, "target_tissue_mask", metadata_root)
                )
                target_nuclei_mask = load_nuclei_mask(
                    _resolve_record_path(record, "target_nuclei_mask", metadata_root)
                )
                prompt = _resolve_prompt(
                    record=record,
                    prompt_override=args.prompt,
                    prompt_source=args.prompt_source,
                    default_prompt_for_dataset=default_prompt_for_dataset,
                )
                with torch.no_grad():
                    image = run_cross_v1_bundle(
                        bundle,
                        reference_image=reference_image,
                        reference_tissue_mask=reference_tissue_mask,
                        reference_nuclei_mask=reference_nuclei_mask,
                        target_tissue_mask=target_tissue_mask,
                        target_nuclei_mask=target_nuclei_mask,
                        prompt=prompt,
                        source_latent_init_strength=args.source_latent_init_strength,
                        mask_chord_scale=args.mask_chord_scale,
                        mask_chord_use_gate=args.mask_chord_use_gate,
                        mask_chord_gate_dilate_radius=args.mask_chord_gate_dilate_radius,
                        mask_chord_gate_feather_radius=args.mask_chord_gate_feather_radius,
                        mask_chord_gate_outside_scale=args.mask_chord_gate_outside_scale,
                        seed=int(args.seed) + int(metadata_cache_id(record, index)),
                    )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(out_path)
                manifest.write(
                    json.dumps(
                        {
                            **_record_identity(record, index),
                            "path": str(out_path),
                            "prompt": prompt,
                            "seed": int(args.seed) + int(metadata_cache_id(record, index)),
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                written += 1
            except Exception as exc:
                failed += 1
                manifest.write(
                    json.dumps(
                        {
                            **_record_identity(record, index),
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                        allow_nan=True,
                    )
                    + "\n"
                )
                print(f"[error] metadata row {index}: {exc!r}")

            if offset % max(1, int(args.save_every)) == 0:
                print(
                    f"[{offset}/{len(selected)}] skipped={skipped} "
                    f"written={written} failed={failed}"
                )

    summary = {
        "metadata": str(metadata_path),
        "output_dir": str(output_dir),
        "start": start,
        "stop": stop,
        "selected": len(selected),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "checkpoint": str(args.checkpoint),
        "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "controlnet_conditioning_scale": float(args.controlnet_conditioning_scale),
        "ip_scale": float(args.ip_scale),
        "source_latent_init_strength": float(args.source_latent_init_strength),
        "mask_chord_scale": float(args.mask_chord_scale),
        "mask_chord_use_gate": bool(args.mask_chord_use_gate),
        "mask_chord_gate_dilate_radius": int(args.mask_chord_gate_dilate_radius),
        "mask_chord_gate_feather_radius": int(args.mask_chord_gate_feather_radius),
        "mask_chord_gate_outside_scale": float(args.mask_chord_gate_outside_scale),
        "has_failures": bool(failed),
        "failure_rate": float(failed / max(1, len(selected))) if selected else math.nan,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    print(
        f"wrote I0 cache to {output_dir} | written={written} "
        f"skipped={skipped} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
