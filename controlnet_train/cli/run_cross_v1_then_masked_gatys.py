"""Generate I0 with Cross V1, then run masked Gatys post-processing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controlnet_train.data.common import default_prompt_for_dataset, load_image_tensor, load_nuclei_mask, load_tissue_mask
from controlnet_train.inference.pipeline_cross_v1 import CrossV1InferenceBundle, load_cross_v1_bundle, run_cross_v1_bundle
from controlnet_train.postprocess import GatysTransferConfig, MaskedGatysStyleTransfer
from controlnet_train.postprocess.masked_gatys import (
    _gatys_feature_layers,
    _resolve_device,
    _resolve_torch_dtype,
    overlay_nuclei_on_tissue_mask,
    save_label_mask,
    tensor_to_pil,
)


CrossGenerator = Callable[..., Image.Image]


@dataclass(frozen=True)
class CrossThenGatysResult:
    i0_path: Path
    final_path: Path
    summary_path: Path
    target_gt_path: Path | None
    pre_color_match_path: Path | None
    i0: Image.Image
    final: Image.Image
    gatys_history: list[dict[str, float]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross V1 -> masked Gatys pipeline.")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default=os.environ.get("MODEL_DIR"),
        required=False if os.environ.get("MODEL_DIR") else True,
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CONTROLNET_CHECKPOINT"),
        required=False if os.environ.get("CONTROLNET_CHECKPOINT") else True,
        help="Cross V1 checkpoint dir, e.g. checkpoint-66000.",
    )
    parser.add_argument(
        "--uni-checkpoint-path",
        default=os.environ.get("UNI_CHECKPOINT"),
        required=False if os.environ.get("UNI_CHECKPOINT") else True,
    )
    parser.add_argument("--reference-image", default=None)
    parser.add_argument("--reference-tissue-mask", default=None)
    parser.add_argument("--reference-nuclei-mask", default=None)
    parser.add_argument("--target-image", default=None)
    parser.add_argument("--target-tissue-mask", default=None)
    parser.add_argument("--target-nuclei-mask", default=None)
    parser.add_argument("--metadata", default=None, help="Optional metadata_cross_val.json to batch over.")
    parser.add_argument("--num-samples", type=int, default=1, help="How many pairs to run from --metadata.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--gatys-torch-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="fp32",
        help="Torch dtype for masked Gatys pixel optimization. Keep fp32 unless memory forces lower precision.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument("--regional-ip-soft-bias", type=float, default=None)
    parser.add_argument("--source-latent-init-strength", type=float, default=0.0)
    parser.add_argument("--mask-chord-scale", type=float, default=0.0)
    parser.add_argument("--mask-chord-use-gate", action="store_true")
    parser.add_argument("--mask-chord-gate-dilate-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-feather-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-outside-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gatys-output-name", default="masked_gatys.png")
    parser.add_argument("--i0-output-name", default="i0.png")
    parser.add_argument("--regions", default=None, help="Comma-separated tissue labels to transfer.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--adam-lr", type=float, default=0.02)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--no-content-loss", action="store_true", help="Disable the VGG content anchor during masked Gatys.")
    parser.add_argument("--style-weight", type=float, default=1e4)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--style-layers", default="conv1_1,conv2_1,conv3_1,conv4_1,conv5_1")
    parser.add_argument("--layer-weights", default="conv1_1=1.0,conv2_1=1.0,conv3_1=0.5,conv4_1=0.25,conv5_1=0.0")
    parser.add_argument("--content-layer", default="conv4_2")
    parser.add_argument("--min-region-pixels", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-preserve-background", action="store_true")
    parser.add_argument("--background-label", type=int, default=0)
    parser.add_argument("--optimize-background", action="store_true", help="Allow label 0/background pixels to be optimized too.")
    parser.add_argument(
        "--missing-region-fallback",
        choices=["pooled", "skip"],
        default="pooled",
        help="How to style target labels missing from the reference mask.",
    )
    parser.add_argument("--no-save-mask-debug", action="store_true")
    parser.add_argument(
        "--pre-gatys-color-match",
        choices=["none", "lab", "macenko"],
        default="none",
        help="Apply stain/color match to I0 before masked Gatys.",
    )
    parser.add_argument(
        "--color-match-scope",
        choices=["region", "global"],
        default="region",
        help="For Macenko, match each composite mask label separately or match globally.",
    )
    parser.add_argument("--color-match-strength", type=float, default=1.0)
    parser.add_argument("--macenko-io", type=float, default=240.0)
    parser.add_argument("--macenko-beta", type=float, default=0.15)
    parser.add_argument("--macenko-alpha", type=float, default=1.0)
    parser.add_argument("--vgg-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--vgg-weights-path", default=None)
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def read_cross_metadata(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    if isinstance(payload, dict):
        records = payload.get("pairs")
        if not isinstance(records, list):
            raise ValueError("cross metadata dict must contain a 'pairs' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported cross metadata payload type: {type(payload)!r}")


def select_eval_records(
    records: list[dict[str, Any]],
    *,
    num_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if num_samples is None or num_samples <= 0 or num_samples >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    return [records[int(index)] for index in indices[:num_samples]]


def run_cross_v1_then_masked_gatys(
    *,
    bundle: CrossV1InferenceBundle,
    reference_image_path: str | Path,
    reference_tissue_mask_path: str | Path,
    reference_nuclei_mask_path: str | Path,
    target_tissue_mask_path: str | Path,
    target_nuclei_mask_path: str | Path,
    output_dir: str | Path,
    prompt: str,
    generate_i0: CrossGenerator = run_cross_v1_bundle,
    gatys_runner: MaskedGatysStyleTransfer | None = None,
    gatys_config: GatysTransferConfig | None = None,
    seed: int = 42,
    source_latent_init_strength: float = 0.0,
    mask_chord_scale: float = 0.0,
    mask_chord_use_gate: bool = False,
    mask_chord_gate_dilate_radius: int = 0,
    mask_chord_gate_feather_radius: int = 0,
    mask_chord_gate_outside_scale: float = 0.0,
    i0_output_name: str = "i0.png",
    gatys_output_name: str = "masked_gatys.png",
    target_image_path: str | Path | None = None,
    target_gt_output_name: str = "target_gt.png",
    regions: tuple[int, ...] | None = None,
) -> CrossThenGatysResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_gt_path = _copy_target_gt(target_image_path, output_dir / target_gt_output_name)

    reference_image = load_image_tensor(reference_image_path)
    reference_tissue_mask = load_tissue_mask(reference_tissue_mask_path)
    reference_nuclei_mask = load_nuclei_mask(reference_nuclei_mask_path)
    target_tissue_mask = load_tissue_mask(target_tissue_mask_path)
    target_nuclei_mask = load_nuclei_mask(target_nuclei_mask_path)
    raw_reference_nuclei_mask = load_nuclei_mask(reference_nuclei_mask_path, remap=False)
    raw_target_nuclei_mask = load_nuclei_mask(target_nuclei_mask_path, remap=False)
    gatys_target_mask = overlay_nuclei_on_tissue_mask(
        target_tissue_mask,
        raw_target_nuclei_mask,
    )
    gatys_reference_mask = overlay_nuclei_on_tissue_mask(
        reference_tissue_mask,
        raw_reference_nuclei_mask,
    )
    target_gatys_mask_path = save_label_mask(gatys_target_mask, output_dir / "target_gatys_composite_mask.png")
    reference_gatys_mask_path = save_label_mask(
        gatys_reference_mask,
        output_dir / "reference_gatys_composite_mask.png",
    )

    i0 = generate_i0(
        bundle,
        reference_image=reference_image,
        reference_tissue_mask=reference_tissue_mask,
        reference_nuclei_mask=reference_nuclei_mask,
        target_tissue_mask=target_tissue_mask,
        target_nuclei_mask=target_nuclei_mask,
        prompt=prompt,
        source_latent_init_strength=source_latent_init_strength,
        mask_chord_scale=mask_chord_scale,
        mask_chord_use_gate=mask_chord_use_gate,
        mask_chord_gate_dilate_radius=mask_chord_gate_dilate_radius,
        mask_chord_gate_feather_radius=mask_chord_gate_feather_radius,
        mask_chord_gate_outside_scale=mask_chord_gate_outside_scale,
        seed=seed,
    )
    i0_path = output_dir / i0_output_name
    i0.save(i0_path)

    if gatys_runner is None:
        device = _resolve_device(gatys_config.device if gatys_config is not None else "cuda")
        config = gatys_config or GatysTransferConfig()
        extractor = _build_gatys_extractor(config, device=device)
        gatys_runner = MaskedGatysStyleTransfer(extractor, config)
    else:
        config = gatys_config or GatysTransferConfig()

    final = gatys_runner.run(
        initial_image=_pil_to_tensor(i0),
        reference_image=load_image_tensor(reference_image_path),
        target_mask=gatys_target_mask,
        reference_mask=gatys_reference_mask,
        regions=regions,
        output_dir=output_dir,
        output_name=gatys_output_name,
    )
    pre_color_match_path = getattr(final, "pre_color_match_path", None)

    summary = {
        "reference_image": str(reference_image_path),
        "target_image": str(target_image_path) if target_image_path is not None else None,
        "reference_tissue_mask": str(reference_tissue_mask_path),
        "reference_nuclei_mask": str(reference_nuclei_mask_path),
        "target_tissue_mask": str(target_tissue_mask_path),
        "target_nuclei_mask": str(target_nuclei_mask_path),
        "prompt": prompt,
        "seed": int(seed),
        "controlnet": {
            "checkpoint": str(bundle.checkpoint_path),
            "pretrained_model_name_or_path": str(bundle.pretrained_model_name_or_path),
            "num_inference_steps": int(bundle.num_inference_steps),
            "guidance_scale": float(bundle.guidance_scale),
            "controlnet_conditioning_scale": float(bundle.controlnet_conditioning_scale),
            "ip_adapter_scale": float(bundle.ip_adapter_scale),
        },
        "gatys": {
            **asdict(config),
            "regions": list(regions) if regions is not None else None,
        },
        "artifacts": {
            "target_gt": str(target_gt_path) if target_gt_path is not None else None,
            "i0": str(i0_path),
            "target_gatys_composite_mask": str(target_gatys_mask_path),
            "reference_gatys_composite_mask": str(reference_gatys_mask_path),
            "pre_gatys_color_matched": str(pre_color_match_path) if pre_color_match_path is not None else None,
            "final": str(final.output_path or output_dir / gatys_output_name),
            "metrics": str(final.metrics_path) if final.metrics_path is not None else None,
        },
        "history": final.history,
    }
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf8")
    return CrossThenGatysResult(
        i0_path=i0_path,
        final_path=final.output_path or (output_dir / gatys_output_name),
        summary_path=summary_path,
        target_gt_path=target_gt_path,
        pre_color_match_path=pre_color_match_path,
        i0=i0,
        final=final.image,
        gatys_history=final.history,
    )


def run_cross_v1_then_masked_gatys_from_record(
    *,
    record: dict[str, Any],
    bundle: CrossV1InferenceBundle,
    output_dir: str | Path,
    prompt_override: str | None = None,
    generate_i0: CrossGenerator = run_cross_v1_bundle,
    gatys_runner: MaskedGatysStyleTransfer | None = None,
    gatys_config: GatysTransferConfig | None = None,
    seed: int = 42,
    source_latent_init_strength: float = 0.0,
    mask_chord_scale: float = 0.0,
    mask_chord_use_gate: bool = False,
    mask_chord_gate_dilate_radius: int = 0,
    mask_chord_gate_feather_radius: int = 0,
    mask_chord_gate_outside_scale: float = 0.0,
    i0_output_name: str = "i0.png",
    gatys_output_name: str = "masked_gatys.png",
    target_gt_output_name: str = "target_gt.png",
    regions: tuple[int, ...] | None = None,
) -> CrossThenGatysResult:
    prompt = (
        str(prompt_override)
        if prompt_override is not None and str(prompt_override).strip()
        else str(record.get("prompt") or default_prompt_for_dataset(str(record.get("dataset") or "BCSS")))
    )
    return run_cross_v1_then_masked_gatys(
        bundle=bundle,
        reference_image_path=record["reference_image"],
        reference_tissue_mask_path=record["reference_tissue_mask"],
        reference_nuclei_mask_path=record["reference_nuclei_mask"],
        target_tissue_mask_path=record["target_tissue_mask"],
        target_nuclei_mask_path=record["target_nuclei_mask"],
        output_dir=output_dir,
        prompt=prompt,
        generate_i0=generate_i0,
        gatys_runner=gatys_runner,
        gatys_config=gatys_config,
        seed=seed,
        source_latent_init_strength=source_latent_init_strength,
        mask_chord_scale=mask_chord_scale,
        mask_chord_use_gate=mask_chord_use_gate,
        mask_chord_gate_dilate_radius=mask_chord_gate_dilate_radius,
        mask_chord_gate_feather_radius=mask_chord_gate_feather_radius,
        mask_chord_gate_outside_scale=mask_chord_gate_outside_scale,
        i0_output_name=i0_output_name,
        gatys_output_name=gatys_output_name,
        target_image_path=record.get("target_image"),
        target_gt_output_name=target_gt_output_name,
        regions=regions,
    )


def run_cross_v1_then_masked_gatys_batch(
    *,
    records: list[dict[str, Any]],
    bundle: CrossV1InferenceBundle,
    output_dir: str | Path,
    prompt_override: str | None = None,
    generate_i0: CrossGenerator = run_cross_v1_bundle,
    gatys_runner: MaskedGatysStyleTransfer | None = None,
    gatys_config: GatysTransferConfig | None = None,
    seed: int = 42,
    source_latent_init_strength: float = 0.0,
    mask_chord_scale: float = 0.0,
    mask_chord_use_gate: bool = False,
    mask_chord_gate_dilate_radius: int = 0,
    mask_chord_gate_feather_radius: int = 0,
    mask_chord_gate_outside_scale: float = 0.0,
    i0_output_name: str = "i0.png",
    gatys_output_name: str = "masked_gatys.png",
    target_gt_output_name: str = "target_gt.png",
    regions: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    config = gatys_config or GatysTransferConfig()
    if gatys_runner is None:
        device = _resolve_device(config.device)
        extractor = _build_gatys_extractor(config, device=device)
        gatys_runner = MaskedGatysStyleTransfer(extractor, config)

    manifest_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample_id = str(record.get("sample_id") or f"sample_{index}")
        ref_id = str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)
        sample_dir = output_root / f"{index:04d}_{_safe_name(sample_id)}__ref_{_safe_name(ref_id)}"
        result = run_cross_v1_then_masked_gatys_from_record(
            record=record,
            bundle=bundle,
            output_dir=sample_dir,
            prompt_override=prompt_override,
            generate_i0=generate_i0,
            gatys_runner=gatys_runner,
            gatys_config=config,
            seed=seed,
            source_latent_init_strength=source_latent_init_strength,
            mask_chord_scale=mask_chord_scale,
            mask_chord_use_gate=mask_chord_use_gate,
            mask_chord_gate_dilate_radius=mask_chord_gate_dilate_radius,
            mask_chord_gate_feather_radius=mask_chord_gate_feather_radius,
            mask_chord_gate_outside_scale=mask_chord_gate_outside_scale,
            i0_output_name=i0_output_name,
            gatys_output_name=gatys_output_name,
            target_gt_output_name=target_gt_output_name,
            regions=regions,
        )
        manifest_rows.append(
            {
                "index": index,
                "sample_id": sample_id,
                "reference_sample_id": ref_id,
                "prompt": str(prompt_override) if prompt_override is not None else str(record.get("prompt") or ""),
                "target_image": str(record.get("target_image") or ""),
                "output_dir": str(sample_dir),
                "target_gt_path": str(result.target_gt_path) if result.target_gt_path is not None else None,
                "i0_path": str(result.i0_path),
                "target_gatys_composite_mask_path": str(sample_dir / "target_gatys_composite_mask.png"),
                "reference_gatys_composite_mask_path": str(sample_dir / "reference_gatys_composite_mask.png"),
                "pre_gatys_color_matched_path": (
                    str(result.pre_color_match_path) if result.pre_color_match_path is not None else None
                ),
                "final_path": str(result.final_path),
                "summary_path": str(result.summary_path),
            }
        )
        print(f"[{index + 1}/{len(records)}] wrote {result.final_path}")

    (output_root / "batch_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    print(f"wrote {output_root / 'batch_manifest.json'}")
    return manifest_rows


def main(argv=None) -> int:
    args = parse_args(argv)
    if bool(args.metadata):
        if args.reference_image or args.reference_tissue_mask or args.reference_nuclei_mask or args.target_image or args.target_tissue_mask or args.target_nuclei_mask:
            raise SystemExit("--metadata mode does not use single-pair image/mask arguments")
    else:
        required = {
            "--reference-image": args.reference_image,
            "--reference-tissue-mask": args.reference_tissue_mask,
            "--reference-nuclei-mask": args.reference_nuclei_mask,
            "--target-tissue-mask": args.target_tissue_mask,
            "--target-nuclei-mask": args.target_nuclei_mask,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit("missing required arguments: " + ", ".join(missing))

    bundle = load_cross_v1_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        uni_checkpoint_path=args.uni_checkpoint_path,
        device=args.device,
        torch_dtype=_resolve_torch_dtype(args.torch_dtype),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=args.ip_scale,
    )
    if args.prompt:
        prompt = args.prompt
    elif args.dataset:
        prompt = default_prompt_for_dataset(args.dataset)
    else:
        prompt = "H&E stained cancer histopathology at 40x magnification"
    gatys_config = GatysTransferConfig(
        steps=args.steps,
        optimizer=args.optimizer,
        lr=args.lr,
        adam_lr=args.adam_lr,
        content_weight=args.content_weight,
        use_content_loss=not args.no_content_loss,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        style_layers=_parse_csv(args.style_layers),
        content_layer=args.content_layer,
        layer_weights=_parse_layer_weights(args.layer_weights),
        min_region_pixels=args.min_region_pixels,
        log_every=args.log_every,
        save_every=args.save_every,
        preserve_background=not args.no_preserve_background,
        background_label=args.background_label,
        optimize_background=args.optimize_background,
        missing_region_fallback=args.missing_region_fallback,
        save_mask_debug=not args.no_save_mask_debug,
        pre_color_match=args.pre_gatys_color_match,
        color_match_scope=args.color_match_scope,
        color_match_strength=args.color_match_strength,
        macenko_io=args.macenko_io,
        macenko_beta=args.macenko_beta,
        macenko_alpha=args.macenko_alpha,
        device=args.device,
        torch_dtype=args.gatys_torch_dtype,
        vgg_weights=args.vgg_weights,
        vgg_weights_path=args.vgg_weights_path,
    )
    if args.metadata:
        records = select_eval_records(read_cross_metadata(args.metadata), num_samples=args.num_samples, seed=args.seed)
        if not records:
            raise SystemExit(f"no records found in {args.metadata}")
        run_cross_v1_then_masked_gatys_batch(
            records=records,
            bundle=bundle,
            output_dir=args.output_dir,
            prompt_override=args.prompt,
            generate_i0=run_cross_v1_bundle,
            gatys_runner=None,
            gatys_config=gatys_config,
            seed=args.seed,
            source_latent_init_strength=args.source_latent_init_strength,
            mask_chord_scale=args.mask_chord_scale,
            mask_chord_use_gate=args.mask_chord_use_gate,
            mask_chord_gate_dilate_radius=args.mask_chord_gate_dilate_radius,
            mask_chord_gate_feather_radius=args.mask_chord_gate_feather_radius,
            mask_chord_gate_outside_scale=args.mask_chord_gate_outside_scale,
            i0_output_name=args.i0_output_name,
            gatys_output_name=args.gatys_output_name,
            target_gt_output_name="target_gt.png",
            regions=_parse_regions(args.regions),
        )
        return 0

    result = run_cross_v1_then_masked_gatys(
        bundle=bundle,
        reference_image_path=args.reference_image,
        reference_tissue_mask_path=args.reference_tissue_mask,
        reference_nuclei_mask_path=args.reference_nuclei_mask,
        target_tissue_mask_path=args.target_tissue_mask,
        target_nuclei_mask_path=args.target_nuclei_mask,
        output_dir=args.output_dir,
        prompt=prompt,
        seed=args.seed,
        source_latent_init_strength=args.source_latent_init_strength,
        mask_chord_scale=args.mask_chord_scale,
        mask_chord_use_gate=args.mask_chord_use_gate,
        mask_chord_gate_dilate_radius=args.mask_chord_gate_dilate_radius,
        mask_chord_gate_feather_radius=args.mask_chord_gate_feather_radius,
        mask_chord_gate_outside_scale=args.mask_chord_gate_outside_scale,
        i0_output_name=args.i0_output_name,
        gatys_output_name=args.gatys_output_name,
        target_image_path=args.target_image,
        target_gt_output_name="target_gt.png",
        regions=_parse_regions(args.regions),
        gatys_config=gatys_config,
    )
    print(f"wrote {result.i0_path}")
    print(f"wrote {result.final_path}")
    print(f"wrote {result.summary_path}")
    return 0


def _build_gatys_extractor(config: GatysTransferConfig, *, device):
    from controlnet_train.postprocess.masked_gatys import build_vgg19_feature_extractor

    return build_vgg19_feature_extractor(
        layers=_gatys_feature_layers(config),
        weights=config.vgg_weights,
        weights_path=config.vgg_weights_path,
        device=device,
        dtype=_resolve_torch_dtype(config.torch_dtype),
    )


def _pil_to_tensor(image: Image.Image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    import torch

    return torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)


def _copy_target_gt(target_image_path: str | Path | None, output_path: Path) -> Path | None:
    if target_image_path is None or not str(target_image_path).strip():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_image_path, output_path)
    return output_path


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_layer_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        name, raw_weight = item.split("=", 1)
        weights[name.strip()] = float(raw_weight)
    return weights


def _parse_regions(value: str | None) -> tuple[int, ...] | None:
    if value is None or not str(value).strip():
        return None
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _safe_name(value: str) -> str:
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"_", "-", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


if __name__ == "__main__":
    raise SystemExit(main())
