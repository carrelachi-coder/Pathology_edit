#!/usr/bin/env python3
"""Probe UNI/CONCH sensitivity to controlled texture/color perturbations.

This diagnostic keeps image identity, tissue label, and semantics fixed, then
applies controlled low-pass texture filters or global color/stain changes to
the same RGB patch. It measures whether frozen encoder regional tokens move
when only one appearance factor changes.

It can also apply global color/stain perturbations that preserve spatial
structure, which is useful for checking whether an encoder can carry reference
stain/color cues.

The key comparison is:

    original RGB -> frozen encoder tokens -> same-label region descriptor
    blurred RGB  -> frozen encoder tokens -> same-label region descriptor

and the scale reference is natural same-label distance between different
original images. If texture distances are tiny relative to natural same-label
distances, the encoder is probably not carrying much fine texture signal in
the final regional tokens. For color perturbations, the same ratio tells us
whether color/stain changes are visible to the encoder at that layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe_uni_target_reference_region_separability import (  # noqa: E402
    build_coarse_lookup,
    build_label_name_lookup,
    canonical_label_name,
    cosine_distance,
    describe_values,
    greater_than_probability,
    mean_difference,
    normalize_label_mode,
    parse_label,
    parse_torch_dtype,
    read_metadata,
    remap_fine_to_coarse,
    resolve_device,
    resolve_metadata_path,
    write_json,
)


@dataclass(frozen=True)
class ImageEntry:
    index: int
    dataset: str
    sample_id: str
    image_path: Path
    tissue_mask_path: Path


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    module: Any
    uni_layer: int | None = None
    conch_layer: int | None = None

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.name == "uni":
            return self.module.extract_uni_features(images).float().cpu()
        if self.name.startswith("uni_layer_"):
            if self.uni_layer is None:
                raise ValueError(f"{self.name} is missing uni_layer")
            return extract_uni_intermediate_features(self.module, images, layer_number=self.uni_layer).float().cpu()
        if self.name == "conch":
            return self.module.extract_features(images).float().cpu()
        if self.name.startswith("conch_layer_"):
            if self.conch_layer is None:
                raise ValueError(f"{self.name} is missing conch_layer")
            return extract_conch_intermediate_features(
                self.module,
                images,
                layer_number=self.conch_layer,
            ).float().cpu()
        raise ValueError(f"unsupported encoder backend: {self.name}")


@dataclass
class RegionDescriptor:
    backend: str
    label_id: int
    label_name: str
    entry: ImageEntry
    token_count: int
    token_fraction: float
    mean: torch.Tensor
    std: torch.Tensor

    @property
    def concat(self) -> torch.Tensor:
        return torch.cat([self.mean, self.std], dim=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure frozen UNI/CONCH regional token sensitivity to texture-only filters."
    )
    parser.add_argument("--metadata", required=True, help="Cross metadata JSON/JSONL with image/mask fields.")
    parser.add_argument("--metadata-base-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backend",
        choices=("uni", "conch", "both"),
        default="both",
        help="Encoder backend to probe.",
    )
    parser.add_argument("--uni-checkpoint-path", default=None)
    parser.add_argument(
        "--uni-layer",
        type=int,
        action="append",
        default=None,
        help=(
            "Probe a 1-based UNI transformer block output instead of only final tokens. "
            "Repeatable, e.g. --uni-layer 6 --uni-layer 12 --uni-layer 18 --uni-layer 24."
        ),
    )
    parser.add_argument(
        "--uni-include-final",
        action="store_true",
        help="When --uni-layer is set, also include the standard final UNI tokens for comparison.",
    )
    parser.add_argument("--conch-checkpoint-path", default=None)
    parser.add_argument(
        "--conch-layer",
        type=int,
        action="append",
        default=None,
        help=(
            "Probe a 1-based CONCH visual transformer block output instead of only final tokens. "
            "Repeatable, e.g. --conch-layer 3 --conch-layer 6 --conch-layer 9 --conch-layer 12."
        ),
    )
    parser.add_argument(
        "--conch-include-final",
        action="store_true",
        help="When --conch-layer is set, also include the standard final CONCH tokens for comparison.",
    )
    parser.add_argument("--conch-root", default=None)
    parser.add_argument("--conch-model-cfg", default="conch_ViT-B-16")
    parser.add_argument("--image-field", default="reference_image")
    parser.add_argument("--mask-field", default="reference_tissue_mask")
    parser.add_argument("--sample-id-field", default="reference_sample_id")
    parser.add_argument(
        "--label-mode",
        choices=("coarse_tissue", "coarse", "fine", "tissue"),
        default="coarse_tissue",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label name/id to probe. Repeatable. Defaults to tumor and stroma.",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--samples-per-label", type=int, default=64)
    parser.add_argument("--min-region-tokens", type=int, default=2)
    parser.add_argument("--min-region-fraction", type=float, default=0.0)
    parser.add_argument(
        "--perturbation-mode",
        choices=("texture", "color", "both"),
        default="texture",
        help="Which perturbation family to run. Defaults to the original texture-only probe.",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        action="append",
        default=None,
        help="Gaussian blur sigma in source image pixels. Repeatable. Defaults to 0.75, 1.5, 3.0.",
    )
    parser.add_argument(
        "--downup-scale",
        type=float,
        action="append",
        default=None,
        help="Optional downsample-then-upsample scale, e.g. 0.5 or 0.25. Repeatable.",
    )
    parser.add_argument(
        "--hed-alpha",
        action="append",
        default=None,
        help=(
            "Deterministic HED stain alpha as H,E or H,E,D. Repeatable. "
            "Example: --hed-alpha 1.15,1.0 --hed-alpha 0.85,1.1."
        ),
    )
    parser.add_argument(
        "--rgb-gain",
        action="append",
        default=None,
        help="Per-channel RGB gain as R,G,B. Repeatable. Example: --rgb-gain 1.05,1.0,0.95.",
    )
    parser.add_argument(
        "--saturation-scale",
        type=float,
        action="append",
        default=None,
        help="Scale chroma around luminance while preserving spatial structure. Repeatable.",
    )
    parser.add_argument(
        "--brightness-scale",
        type=float,
        action="append",
        default=None,
        help="Global RGB brightness multiplier. Repeatable.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--std-weight", type=float, default=0.5)
    parser.add_argument("--pooled-cosine-weight", type=float, default=0.25)
    parser.add_argument("--max-natural-pairs", type=int, default=20000)
    parser.add_argument(
        "--max-color-mean-l1-for-texture-only",
        type=float,
        default=0.02,
        help="Color mean drift threshold used only for the summary note.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.label is None:
        args.label = ["tumor", "stroma"]
    if args.backend in {"uni", "both"} and not args.uni_checkpoint_path:
        raise ValueError("--uni-checkpoint-path is required for --backend uni/both")
    if args.backend in {"conch", "both"} and not args.conch_checkpoint_path:
        raise ValueError("--conch-checkpoint-path is required for --backend conch/both")
    if args.candidate_pool_size <= 0:
        raise ValueError("--candidate-pool-size must be positive")
    if args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be positive")
    if args.min_region_tokens <= 0:
        raise ValueError("--min-region-tokens must be positive")
    if not 0.0 <= args.min_region_fraction <= 1.0:
        raise ValueError("--min-region-fraction must be in [0, 1]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_natural_pairs < 0:
        raise ValueError("--max-natural-pairs must be non-negative")
    for sigma in args.blur_sigma or []:
        if sigma <= 0:
            raise ValueError("--blur-sigma values must be positive")
    for layer in args.uni_layer or []:
        if layer <= 0:
            raise ValueError("--uni-layer is 1-based and must be positive")
    for layer in args.conch_layer or []:
        if layer <= 0:
            raise ValueError("--conch-layer is 1-based and must be positive")
    for scale in args.downup_scale or []:
        if not 0.0 < scale < 1.0:
            raise ValueError("--downup-scale values must be in (0, 1)")
    for value in args.hed_alpha or []:
        parsed = parse_float_tuple(value, expected=(2, 3), name="--hed-alpha")
        if any(part <= 0 for part in parsed):
            raise ValueError("--hed-alpha values must be positive")
    for value in args.rgb_gain or []:
        parsed = parse_float_tuple(value, expected=(3,), name="--rgb-gain")
        if any(part <= 0 for part in parsed):
            raise ValueError("--rgb-gain values must be positive")
    for value in args.saturation_scale or []:
        if value <= 0:
            raise ValueError("--saturation-scale values must be positive")
    for value in args.brightness_scale or []:
        if value <= 0:
            raise ValueError("--brightness-scale values must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from controlnet_train.data.common import load_image_tensor, load_tissue_mask
    from controlnet_train.modules.reference_image_encoder import (
        ReferenceImageEncoder,
        resize_mask_to_token_labels,
    )
    from dataset_config import COARSE_LABELS, FINE_LABELS, FINE_TO_PARENT

    metadata_path = Path(args.metadata)
    base_dir = Path(args.metadata_base_dir) if args.metadata_base_dir else metadata_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_mode = normalize_label_mode(args.label_mode)
    label_lookup = build_label_name_lookup(label_mode, COARSE_LABELS, FINE_LABELS)
    label_ids = [parse_label(value, label_lookup) for value in args.label]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError(f"--label values must resolve to unique ids, got {label_ids}")

    records = read_metadata(metadata_path)
    entries = build_entries(
        records,
        base_dir=base_dir,
        image_field=str(args.image_field),
        mask_field=str(args.mask_field),
        sample_id_field=str(args.sample_id_field),
    )
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    entries = entries[: int(args.candidate_pool_size)]

    remap_lookup = build_coarse_lookup(FINE_TO_PARENT, device=torch.device("cpu"))
    selected_entries = select_entries_by_label(
        entries,
        label_ids=label_ids,
        label_mode=label_mode,
        remap_lookup=remap_lookup,
        load_tissue_mask=load_tissue_mask,
        samples_per_label=int(args.samples_per_label),
        min_region_fraction=float(args.min_region_fraction),
    )
    if not selected_entries:
        raise RuntimeError("No eligible image entries found for requested labels.")

    device = resolve_device(args.device)
    dtype = parse_torch_dtype(args.torch_dtype)
    encoders = load_encoders(args, device=device, dtype=dtype)
    perturbations = build_perturbation_specs(
        mode=str(args.perturbation_mode),
        blur_sigmas=args.blur_sigma,
        downup_scales=args.downup_scale,
        hed_alpha_values=args.hed_alpha,
        rgb_gain_values=args.rgb_gain,
        saturation_scales=args.saturation_scale,
        brightness_scales=args.brightness_scale,
    )
    if not perturbations:
        raise RuntimeError("No perturbations configured.")

    texture_rows, descriptors, skipped = collect_texture_rows(
        selected_entries,
        encoders=encoders,
        perturbations=perturbations,
        label_ids=label_ids,
        label_lookup=label_lookup,
        label_mode=label_mode,
        remap_lookup=remap_lookup,
        load_image_tensor=load_image_tensor,
        load_tissue_mask=load_tissue_mask,
        resize_mask_to_token_labels=resize_mask_to_token_labels,
        batch_size=int(args.batch_size),
        device=device,
        dtype=dtype,
        min_region_tokens=int(args.min_region_tokens),
        min_region_fraction=float(args.min_region_fraction),
        mean_weight=float(args.mean_weight),
        std_weight=float(args.std_weight),
        pooled_cosine_weight=float(args.pooled_cosine_weight),
        progress_every=int(args.progress_every),
    )

    natural_rows = build_natural_same_label_rows(
        descriptors,
        rng=random.Random(args.seed + 1009),
        max_pairs=int(args.max_natural_pairs),
        mean_weight=float(args.mean_weight),
        std_weight=float(args.std_weight),
        pooled_cosine_weight=float(args.pooled_cosine_weight),
    )
    summary = build_summary(
        texture_rows,
        natural_rows,
        label_ids=label_ids,
        label_lookup=label_lookup,
        backends=[encoder.name for encoder in encoders],
        perturbation_names=[name for name, _ in perturbations],
        max_color_mean_l1=float(args.max_color_mean_l1_for_texture_only),
    )
    summary.update(
        {
            "metadata": str(metadata_path),
            "metadata_base_dir": str(base_dir),
            "image_field": str(args.image_field),
            "mask_field": str(args.mask_field),
            "sample_id_field": str(args.sample_id_field),
            "backend": str(args.backend),
            "perturbation_mode": str(args.perturbation_mode),
            "uni_checkpoint_path": str(args.uni_checkpoint_path) if args.uni_checkpoint_path else None,
            "uni_layers": [int(layer) for layer in args.uni_layer or []],
            "uni_include_final": bool(args.uni_include_final),
            "conch_checkpoint_path": str(args.conch_checkpoint_path) if args.conch_checkpoint_path else None,
            "conch_root": str(args.conch_root) if args.conch_root else None,
            "conch_model_cfg": str(args.conch_model_cfg),
            "conch_layers": [int(layer) for layer in args.conch_layer or []],
            "conch_include_final": bool(args.conch_include_final),
            "hed_alpha": list(args.hed_alpha or []),
            "rgb_gain": list(args.rgb_gain or []),
            "saturation_scale": [float(value) for value in args.saturation_scale or []],
            "brightness_scale": [float(value) for value in args.brightness_scale or []],
            "label_mode": label_mode,
            "labels": [
                {"id": int(label_id), "name": canonical_label_name(label_id, label_lookup, fallback=str(label_id))}
                for label_id in label_ids
            ],
            "candidate_entries": len(entries),
            "selected_entries": len(selected_entries),
            "texture_row_count": len(texture_rows),
            "natural_pair_row_count": len(natural_rows),
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:100],
            "outputs": {
                "texture_rows_csv": "encoder_texture_sensitivity_rows.csv",
                "natural_pairs_csv": "encoder_texture_natural_pairs.csv",
                "summary_json": "encoder_texture_sensitivity_summary.json",
            },
        }
    )

    write_csv_rows(output_dir / "encoder_texture_sensitivity_rows.csv", texture_rows)
    write_csv_rows(output_dir / "encoder_texture_natural_pairs.csv", natural_rows)
    write_json(output_dir / "encoder_texture_sensitivity_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


def load_encoders(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype) -> list[EncoderSpec]:
    encoders: list[EncoderSpec] = []
    if args.backend in {"uni", "both"}:
        from controlnet_train.modules.reference_image_encoder import ReferenceImageEncoder

        uni = ReferenceImageEncoder(args.uni_checkpoint_path, skip_perceiver=True)
        uni.to(device=device, dtype=dtype)
        uni.eval()
        uni.requires_grad_(False)
        uni_layers = [int(layer) for layer in args.uni_layer or []]
        validate_uni_layers(uni, uni_layers)
        if not uni_layers or bool(args.uni_include_final):
            encoders.append(EncoderSpec("uni", uni))
        for layer in uni_layers:
            encoders.append(EncoderSpec(f"uni_layer_{layer:02d}", uni, uni_layer=layer))
    if args.backend in {"conch", "both"}:
        from controlnet_train.modules.conch_feature_encoder import ConchFeatureEncoder

        conch = ConchFeatureEncoder(
            args.conch_checkpoint_path,
            conch_root=args.conch_root,
            model_cfg=args.conch_model_cfg,
        )
        conch.to(device=device, dtype=dtype)
        conch.eval()
        conch.requires_grad_(False)
        conch_layers = [int(layer) for layer in getattr(args, "conch_layer", None) or []]
        validate_conch_layers(conch, conch_layers)
        if not conch_layers or bool(getattr(args, "conch_include_final", False)):
            encoders.append(EncoderSpec("conch", conch))
        for layer in conch_layers:
            encoders.append(EncoderSpec(f"conch_layer_{layer:02d}", conch, conch_layer=layer))
    return encoders


def validate_uni_layers(encoder: Any, layers: list[int]) -> None:
    if not layers:
        return
    if not hasattr(encoder.uni, "get_intermediate_layers"):
        raise RuntimeError("UNI backbone does not expose get_intermediate_layers(); cannot probe --uni-layer")
    depth = len(getattr(encoder.uni, "blocks", []))
    if depth <= 0:
        return
    invalid = [layer for layer in layers if layer > depth]
    if invalid:
        raise ValueError(f"--uni-layer exceeds UNI depth {depth}: {invalid}")


def extract_uni_intermediate_features(
    encoder: Any,
    images: torch.Tensor,
    *,
    layer_number: int,
) -> torch.Tensor:
    """Return patch tokens from a 1-based UNI transformer block."""
    if layer_number <= 0:
        raise ValueError(f"layer_number must be 1-based and positive, got {layer_number}")
    if not hasattr(encoder.uni, "get_intermediate_layers"):
        raise RuntimeError("UNI backbone does not expose get_intermediate_layers()")
    encoder._lock_uni_backbone()
    x = encoder._prepare_uni_input(images)
    block_index = int(layer_number) - 1
    outputs = encoder.uni.get_intermediate_layers(
        x,
        n=[block_index],
        reshape=False,
    )
    if not outputs:
        raise RuntimeError(f"UNI get_intermediate_layers returned no output for layer {layer_number}")
    features = outputs[0]
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim == 4:
        features = features.flatten(2).transpose(1, 2)
    if features.ndim != 3:
        raise ValueError(
            f"UNI intermediate features must be BxTxC or BxCxHxW, got {tuple(features.shape)} "
            f"for layer {layer_number}"
        )
    patch_size = int(getattr(encoder, "uni_patch_size", 14))
    if hasattr(encoder, "num_spatial_tokens"):
        num_patch_tokens = int(encoder.num_spatial_tokens)
    else:
        num_patch_tokens = (x.shape[-2] // patch_size) * (x.shape[-1] // patch_size)
    if features.shape[1] > num_patch_tokens:
        features = features[:, -num_patch_tokens:, :]
    return features


def validate_conch_layers(encoder: Any, layers: list[int]) -> None:
    if not layers:
        return
    trunk = encoder.model.visual.trunk
    if hasattr(trunk, "get_intermediate_layers"):
        blocks = _find_conch_blocks(trunk, required=False)
        depth = len(blocks) if blocks is not None else 0
        if depth > 0:
            invalid = [layer for layer in layers if layer > depth]
            if invalid:
                raise ValueError(f"--conch-layer exceeds CONCH depth {depth}: {invalid}")
        return
    blocks = _find_conch_blocks(trunk, required=True)
    invalid = [layer for layer in layers if layer > len(blocks)]
    if invalid:
        raise ValueError(f"--conch-layer exceeds CONCH depth {len(blocks)}: {invalid}")


def extract_conch_intermediate_features(
    encoder: Any,
    images: torch.Tensor,
    *,
    layer_number: int,
) -> torch.Tensor:
    """Return patch tokens from a 1-based CONCH visual transformer block."""
    if layer_number <= 0:
        raise ValueError(f"layer_number must be 1-based and positive, got {layer_number}")
    encoder.model.eval()
    trunk = encoder.model.visual.trunk
    x = encoder._prepare_input(images)
    block_index = int(layer_number) - 1
    if hasattr(trunk, "get_intermediate_layers"):
        features = _call_get_intermediate_layers(trunk, x, block_index=block_index, layer_number=layer_number)
    else:
        features = _extract_conch_layer_with_hook(trunk, x, block_index=block_index, layer_number=layer_number)
    return _normalize_conch_token_output(encoder, features)


def _call_get_intermediate_layers(
    trunk: Any,
    x: torch.Tensor,
    *,
    block_index: int,
    layer_number: int,
) -> torch.Tensor:
    attempts = (
        {"n": [block_index], "reshape": False, "return_prefix_tokens": False},
        {"n": [block_index], "reshape": False},
        {"n": [block_index]},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            outputs = trunk.get_intermediate_layers(x, **kwargs)
            break
        except TypeError as exc:
            last_error = exc
    else:
        raise RuntimeError("CONCH trunk.get_intermediate_layers call failed") from last_error
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)) and outputs:
        features = outputs[0]
        if isinstance(features, (tuple, list)) and features:
            features = features[0]
        if torch.is_tensor(features):
            return features
    raise RuntimeError(f"CONCH get_intermediate_layers returned no tensor for layer {layer_number}")


def _extract_conch_layer_with_hook(
    trunk: Any,
    x: torch.Tensor,
    *,
    block_index: int,
    layer_number: int,
) -> torch.Tensor:
    blocks = _find_conch_blocks(trunk, required=True)
    if block_index >= len(blocks):
        raise ValueError(f"CONCH layer {layer_number} exceeds available depth {len(blocks)}")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(tensor):
            raise TypeError(f"CONCH block {layer_number} hook output is not a tensor: {type(output)!r}")
        captured["features"] = tensor

    handle = blocks[block_index].register_forward_hook(hook)
    try:
        _ = trunk(x)
    finally:
        handle.remove()
    if "features" not in captured:
        raise RuntimeError(f"CONCH block hook did not capture layer {layer_number}")
    return captured["features"]


def _find_conch_blocks(trunk: Any, *, required: bool) -> Any:
    candidates = (
        getattr(trunk, "blocks", None),
        getattr(getattr(trunk, "transformer", None), "resblocks", None),
        getattr(trunk, "resblocks", None),
        getattr(getattr(trunk, "model", None), "blocks", None),
    )
    for blocks in candidates:
        if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
            return blocks
    if required:
        raise RuntimeError(
            "Could not find CONCH visual transformer blocks. Expected trunk.blocks, "
            "trunk.transformer.resblocks, trunk.resblocks, or trunk.model.blocks."
        )
    return None


def _normalize_conch_token_output(encoder: Any, features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        features = features.flatten(2).transpose(1, 2)
    if features.ndim != 3:
        raise ValueError(f"CONCH intermediate features must be BxTxC or BxCxHxW, got {tuple(features.shape)}")
    num_patch_tokens = int(encoder.num_spatial_tokens)
    if features.shape[1] > num_patch_tokens:
        features = features[:, -num_patch_tokens:, :]
    return features


def build_entries(
    records: list[dict[str, Any]],
    *,
    base_dir: Path,
    image_field: str,
    mask_field: str,
    sample_id_field: str,
) -> list[ImageEntry]:
    entries: list[ImageEntry] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if not record.get(image_field) or not record.get(mask_field):
            continue
        image_path = resolve_metadata_path(record[image_field], base_dir)
        mask_path = resolve_metadata_path(record[mask_field], base_dir)
        key = (str(image_path), str(mask_path))
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            ImageEntry(
                index=index,
                dataset=str(record.get("dataset") or "unknown"),
                sample_id=str(record.get(sample_id_field) or record.get("sample_id") or image_path.stem),
                image_path=image_path,
                tissue_mask_path=mask_path,
            )
        )
    return entries


def select_entries_by_label(
    entries: list[ImageEntry],
    *,
    label_ids: list[int],
    label_mode: str,
    remap_lookup: torch.Tensor,
    load_tissue_mask,
    samples_per_label: int,
    min_region_fraction: float,
) -> list[ImageEntry]:
    selected: list[ImageEntry] = []
    counts = {int(label_id): 0 for label_id in label_ids}
    seen_paths: set[Path] = set()

    def enough() -> bool:
        return all(counts[int(label_id)] >= samples_per_label for label_id in label_ids)

    for entry in entries:
        if enough():
            break
        try:
            mask = load_tissue_mask(entry.tissue_mask_path)
        except Exception:
            continue
        label_mask = prepare_label_mask(mask, label_mode=label_mode, remap_lookup=remap_lookup)
        usable_labels: list[int] = []
        total_pixels = max(1, int(label_mask.numel()))
        for label_id in label_ids:
            label_id = int(label_id)
            if counts[label_id] >= samples_per_label:
                continue
            pixel_count = int((label_mask == label_id).sum().item())
            fraction = float(pixel_count / total_pixels)
            if pixel_count > 0 and fraction >= min_region_fraction:
                usable_labels.append(label_id)
        if not usable_labels:
            continue
        if entry.image_path not in seen_paths:
            selected.append(entry)
            seen_paths.add(entry.image_path)
        for label_id in usable_labels:
            counts[label_id] += 1
    return selected


def build_perturbation_specs(
    *,
    mode: str,
    blur_sigmas: list[float] | None,
    downup_scales: list[float] | None,
    hed_alpha_values: list[str] | None,
    rgb_gain_values: list[str] | None,
    saturation_scales: list[float] | None,
    brightness_scales: list[float] | None,
) -> list[tuple[str, dict[str, Any]]]:
    specs: list[tuple[str, dict[str, Any]]] = []
    include_texture = mode in {"texture", "both"}
    include_color = mode in {"color", "both"}
    if include_texture:
        for sigma in blur_sigmas or [0.75, 1.5, 3.0]:
            name = f"gaussian_blur_sigma_{float(sigma):g}"
            specs.append((name, {"family": "texture", "kind": "gaussian_blur", "sigma": float(sigma)}))
        for scale in downup_scales or []:
            name = f"downup_scale_{float(scale):g}"
            specs.append((name, {"family": "texture", "kind": "downup", "scale": float(scale)}))
    if include_color:
        explicit_color = any(
            values
            for values in (
                hed_alpha_values,
                rgb_gain_values,
                saturation_scales,
                brightness_scales,
            )
        )
        hed_values = hed_alpha_values or (
            ["1.15,1.0", "0.85,1.0", "1.0,1.15", "1.0,0.85"] if not explicit_color else []
        )
        saturation_values = saturation_scales or ([0.75, 1.25] if not explicit_color else [])
        for raw_value in hed_values:
            alpha = parse_float_tuple(raw_value, expected=(2, 3), name="--hed-alpha")
            if len(alpha) == 2:
                alpha = (alpha[0], alpha[1], 1.0)
            name = "hed_alpha_h_{}_e_{}_d_{}".format(
                format_float_for_name(alpha[0]),
                format_float_for_name(alpha[1]),
                format_float_for_name(alpha[2]),
            )
            specs.append(
                (
                    name,
                    {
                        "family": "color",
                        "kind": "hed_alpha",
                        "alpha_h": float(alpha[0]),
                        "alpha_e": float(alpha[1]),
                        "alpha_d": float(alpha[2]),
                    },
                )
            )
        for raw_value in rgb_gain_values or []:
            gain = parse_float_tuple(raw_value, expected=(3,), name="--rgb-gain")
            name = "rgb_gain_r_{}_g_{}_b_{}".format(
                format_float_for_name(gain[0]),
                format_float_for_name(gain[1]),
                format_float_for_name(gain[2]),
            )
            specs.append(
                (
                    name,
                    {
                        "family": "color",
                        "kind": "rgb_gain",
                        "gain_r": float(gain[0]),
                        "gain_g": float(gain[1]),
                        "gain_b": float(gain[2]),
                    },
                )
            )
        for scale in saturation_values:
            name = f"saturation_scale_{format_float_for_name(float(scale))}"
            specs.append((name, {"family": "color", "kind": "saturation", "scale": float(scale)}))
        for scale in brightness_scales or []:
            name = f"brightness_scale_{format_float_for_name(float(scale))}"
            specs.append((name, {"family": "color", "kind": "brightness", "scale": float(scale)}))
    return specs


def collect_texture_rows(
    entries: list[ImageEntry],
    *,
    encoders: list[EncoderSpec],
    perturbations: list[tuple[str, dict[str, Any]]],
    label_ids: list[int],
    label_lookup: dict[str, int],
    label_mode: str,
    remap_lookup: torch.Tensor,
    load_image_tensor,
    load_tissue_mask,
    resize_mask_to_token_labels,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    min_region_tokens: int,
    min_region_fraction: float,
    mean_weight: float,
    std_weight: float,
    pooled_cosine_weight: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[RegionDescriptor], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    descriptors: list[RegionDescriptor] = []
    skipped: list[dict[str, Any]] = []
    pending_entries: list[ImageEntry] = []
    pending_images: list[torch.Tensor] = []
    pending_masks: list[torch.Tensor] = []
    processed = 0

    def flush() -> None:
        nonlocal processed
        if not pending_entries:
            return
        image_cpu = torch.stack(pending_images)
        mask_cpu = torch.stack(pending_masks)
        image_batch = image_cpu.to(device=device, dtype=dtype)
        for encoder in encoders:
            with torch.no_grad():
                original_features = encoder.extract_features(image_batch)
            token_labels = resize_mask_to_token_labels(mask_cpu, int(original_features.shape[1]))
            if label_mode == "coarse_tissue":
                token_labels = remap_fine_to_coarse(token_labels, remap_lookup)

            original_by_label: dict[tuple[int, int], RegionDescriptor] = {}
            for batch_index, entry in enumerate(pending_entries):
                for label_id in label_ids:
                    label_id = int(label_id)
                    region = token_labels[batch_index] == label_id
                    token_count = int(region.sum().item())
                    token_fraction = float(token_count / max(1, int(region.numel())))
                    if token_count < min_region_tokens or token_fraction < min_region_fraction:
                        continue
                    region_tokens = original_features[batch_index, region]
                    descriptor = make_region_descriptor(
                        backend=encoder.name,
                        label_id=label_id,
                        label_lookup=label_lookup,
                        entry=entry,
                        token_count=token_count,
                        token_fraction=token_fraction,
                        tokens=region_tokens,
                    )
                    descriptors.append(descriptor)
                    original_by_label[(batch_index, label_id)] = descriptor

            for perturbation_name, perturbation_spec in perturbations:
                perturbed_cpu = apply_perturbation(image_cpu, perturbation_spec)
                perturbed_batch = perturbed_cpu.to(device=device, dtype=dtype)
                with torch.no_grad():
                    perturbed_features = encoder.extract_features(perturbed_batch)
                for batch_index, entry in enumerate(pending_entries):
                    pixel_label_mask = prepare_label_mask(
                        mask_cpu[batch_index],
                        label_mode=label_mode,
                        remap_lookup=remap_lookup,
                    )
                    for label_id in label_ids:
                        label_id = int(label_id)
                        original_descriptor = original_by_label.get((batch_index, label_id))
                        if original_descriptor is None:
                            continue
                        region = token_labels[batch_index] == label_id
                        original_tokens = original_features[batch_index, region]
                        perturbed_tokens = perturbed_features[batch_index, region]
                        metrics = token_pair_metrics(
                            original_tokens,
                            perturbed_tokens,
                            mean_weight=mean_weight,
                            std_weight=std_weight,
                            pooled_cosine_weight=pooled_cosine_weight,
                        )
                        color_metrics = regional_color_metrics(
                            image_cpu[batch_index],
                            perturbed_cpu[batch_index],
                            pixel_label_mask == label_id,
                        )
                        rows.append(
                            {
                                "backend": encoder.name,
                                "perturbation": perturbation_name,
                                "perturbation_family": str(perturbation_spec.get("family", "unknown")),
                                "dataset": entry.dataset,
                                "sample_id": entry.sample_id,
                                "metadata_index": entry.index,
                                "image_path": str(entry.image_path),
                                "tissue_mask_path": str(entry.tissue_mask_path),
                                "label_id": label_id,
                                "label_name": original_descriptor.label_name,
                                "token_count": original_descriptor.token_count,
                                "token_fraction": original_descriptor.token_fraction,
                                **metrics,
                                **color_metrics,
                            }
                        )
        processed += len(pending_entries)
        if progress_every > 0 and processed % progress_every < len(pending_entries):
            print(f"[texture-sensitivity] processed={processed}/{len(entries)}", flush=True)
        pending_entries.clear()
        pending_images.clear()
        pending_masks.clear()

    for entry in entries:
        try:
            pending_images.append(load_image_tensor(entry.image_path))
            pending_masks.append(load_tissue_mask(entry.tissue_mask_path))
            pending_entries.append(entry)
        except Exception as exc:  # noqa: BLE001 - diagnostic should keep going.
            skipped.append(
                {
                    "sample_id": entry.sample_id,
                    "image_path": str(entry.image_path),
                    "mask_path": str(entry.tissue_mask_path),
                    "reason": f"load_failed:{type(exc).__name__}",
                    "detail": str(exc),
                }
            )
            continue
        if len(pending_entries) >= batch_size:
            flush()
    flush()
    return rows, descriptors, skipped


def make_region_descriptor(
    *,
    backend: str,
    label_id: int,
    label_lookup: dict[str, int],
    entry: ImageEntry,
    token_count: int,
    token_fraction: float,
    tokens: torch.Tensor,
) -> RegionDescriptor:
    return RegionDescriptor(
        backend=backend,
        label_id=int(label_id),
        label_name=canonical_label_name(label_id, label_lookup, fallback=str(label_id)),
        entry=entry,
        token_count=int(token_count),
        token_fraction=float(token_fraction),
        mean=tokens.float().mean(dim=0).cpu(),
        std=torch.sqrt(tokens.float().var(dim=0, unbiased=False).cpu() + 1e-6),
    )


def prepare_label_mask(mask: torch.Tensor, *, label_mode: str, remap_lookup: torch.Tensor) -> torch.Tensor:
    labels = mask.long().cpu()
    if label_mode == "coarse_tissue":
        labels = remap_fine_to_coarse(labels, remap_lookup)
    return labels


def apply_perturbation(images: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    kind = str(spec["kind"])
    if kind == "gaussian_blur":
        return gaussian_blur(images, sigma=float(spec["sigma"]))
    if kind == "downup":
        return downsample_upsample(images, scale=float(spec["scale"]))
    if kind == "hed_alpha":
        alpha = (
            float(spec["alpha_h"]),
            float(spec["alpha_e"]),
            float(spec.get("alpha_d", 1.0)),
        )
        return hed_alpha_perturb(images, alpha=alpha)
    if kind == "rgb_gain":
        gain = (
            float(spec["gain_r"]),
            float(spec["gain_g"]),
            float(spec["gain_b"]),
        )
        return rgb_gain(images, gain=gain)
    if kind == "saturation":
        return saturation_scale(images, scale=float(spec["scale"]))
    if kind == "brightness":
        return brightness_scale(images, scale=float(spec["scale"]))
    raise ValueError(f"unsupported perturbation kind: {kind}")


def gaussian_blur(images: torch.Tensor, *, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return images.clone()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    channels = int(images.shape[1])
    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.pad(images, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel_x, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel_y, groups=channels)
    return x.clamp(0.0, 1.0)


def downsample_upsample(images: torch.Tensor, *, scale: float) -> torch.Tensor:
    h, w = int(images.shape[-2]), int(images.shape[-1])
    small_h = max(1, int(round(h * scale)))
    small_w = max(1, int(round(w * scale)))
    small = F.interpolate(images, size=(small_h, small_w), mode="bicubic", align_corners=False)
    restored = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
    return restored.clamp(0.0, 1.0)


def hed_alpha_perturb(images: torch.Tensor, *, alpha: tuple[float, float, float]) -> torch.Tensor:
    """Deterministically scale H/E/D optical-density concentrations."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected BCHW RGB tensor, got {tuple(images.shape)}")
    dtype = images.dtype
    device = images.device
    eps = 1.0 / 255.0
    stain_rows = torch.tensor(
        [
            [0.650, 0.704, 0.286],
            [0.072, 0.990, 0.105],
            [0.268, 0.570, 0.776],
        ],
        dtype=torch.float32,
        device=device,
    )
    stain_rows = stain_rows / stain_rows.norm(dim=1, keepdim=True).clamp_min(eps)
    inv_stain_rows = torch.linalg.inv(stain_rows)
    rgb = images.float().clamp(eps, 1.0)
    b, _c, h, w = rgb.shape
    rgb_flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    od = -torch.log(rgb_flat)
    concentrations = od @ inv_stain_rows
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device)
    concentrations = (concentrations * alpha_tensor).clamp_min(0.0)
    perturbed_od = concentrations @ stain_rows
    perturbed = torch.exp(-perturbed_od).reshape(b, h, w, 3).permute(0, 3, 1, 2)
    return perturbed.clamp(0.0, 1.0).to(dtype=dtype)


def rgb_gain(images: torch.Tensor, *, gain: tuple[float, float, float]) -> torch.Tensor:
    gain_tensor = torch.tensor(gain, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * gain_tensor).clamp(0.0, 1.0)


def saturation_scale(images: torch.Tensor, *, scale: float) -> torch.Tensor:
    weights = images.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    luminance = (images * weights).sum(dim=1, keepdim=True)
    return (luminance + (images - luminance) * float(scale)).clamp(0.0, 1.0)


def brightness_scale(images: torch.Tensor, *, scale: float) -> torch.Tensor:
    return (images * float(scale)).clamp(0.0, 1.0)


def token_pair_metrics(
    original_tokens: torch.Tensor,
    perturbed_tokens: torch.Tensor,
    *,
    mean_weight: float,
    std_weight: float,
    pooled_cosine_weight: float,
) -> dict[str, float]:
    left = original_tokens.float()
    right = perturbed_tokens.float()
    left_mean = left.mean(dim=0)
    right_mean = right.mean(dim=0)
    left_std = torch.sqrt(left.var(dim=0, unbiased=False) + 1e-6)
    right_std = torch.sqrt(right.var(dim=0, unbiased=False) + 1e-6)
    mean_l1 = float(F.l1_loss(left_mean, right_mean).item())
    std_l1 = float(F.l1_loss(left_std, right_std).item())
    mean_cos = cosine_distance(left_mean, right_mean)
    std_cos = cosine_distance(left_std, right_std)
    concat_cos = cosine_distance(torch.cat([left_mean, left_std]), torch.cat([right_mean, right_std]))
    total_weight = float(mean_weight) + float(std_weight) + float(pooled_cosine_weight)
    weighted = float(mean_weight) * mean_l1
    weighted += float(std_weight) * std_l1
    weighted += float(pooled_cosine_weight) * mean_cos
    region_loss_style = weighted / total_weight if total_weight > 0 else weighted

    token_cos = 1.0 - (
        F.normalize(left, dim=-1, eps=1e-6) * F.normalize(right, dim=-1, eps=1e-6)
    ).sum(dim=-1)
    return {
        "aligned_token_l1_distance": float(F.l1_loss(left, right).item()),
        "aligned_token_cosine_distance": float(token_cos.mean().item()),
        "mean_l1_distance": mean_l1,
        "std_l1_distance": std_l1,
        "mean_cosine_distance": mean_cos,
        "std_cosine_distance": std_cos,
        "concat_cosine_distance": concat_cos,
        "two_token_average_cosine_distance": float((mean_cos + std_cos) * 0.5),
        "region_loss_style_distance": float(region_loss_style),
    }


def descriptor_pair_metrics(
    left: RegionDescriptor,
    right: RegionDescriptor,
    *,
    mean_weight: float,
    std_weight: float,
    pooled_cosine_weight: float,
) -> dict[str, float]:
    mean_l1 = float(F.l1_loss(left.mean, right.mean).item())
    std_l1 = float(F.l1_loss(left.std, right.std).item())
    mean_cos = cosine_distance(left.mean, right.mean)
    std_cos = cosine_distance(left.std, right.std)
    concat_cos = cosine_distance(left.concat, right.concat)
    total_weight = float(mean_weight) + float(std_weight) + float(pooled_cosine_weight)
    weighted = float(mean_weight) * mean_l1
    weighted += float(std_weight) * std_l1
    weighted += float(pooled_cosine_weight) * mean_cos
    region_loss_style = weighted / total_weight if total_weight > 0 else weighted
    return {
        "mean_l1_distance": mean_l1,
        "std_l1_distance": std_l1,
        "mean_cosine_distance": mean_cos,
        "std_cosine_distance": std_cos,
        "concat_cosine_distance": concat_cos,
        "two_token_average_cosine_distance": float((mean_cos + std_cos) * 0.5),
        "region_loss_style_distance": float(region_loss_style),
    }


def regional_color_metrics(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    region: torch.Tensor,
) -> dict[str, float]:
    if not bool(region.any().item()):
        return {
            "rgb_region_mean_l1": math.nan,
            "rgb_region_std_l1": math.nan,
            "rgb_region_pixel_l1": math.nan,
            "rgb_region_pixel_count": 0,
        }
    left = original[:, region].float()
    right = perturbed[:, region].float()
    return {
        "rgb_region_mean_l1": float(F.l1_loss(left.mean(dim=1), right.mean(dim=1)).item()),
        "rgb_region_std_l1": float(F.l1_loss(left.std(dim=1, unbiased=False), right.std(dim=1, unbiased=False)).item()),
        "rgb_region_pixel_l1": float(F.l1_loss(left, right).item()),
        "rgb_region_pixel_count": int(region.sum().item()),
    }


def build_natural_same_label_rows(
    descriptors: list[RegionDescriptor],
    *,
    rng: random.Random,
    max_pairs: int,
    mean_weight: float,
    std_weight: float,
    pooled_cosine_weight: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[RegionDescriptor]] = {}
    for descriptor in descriptors:
        grouped.setdefault((descriptor.backend, descriptor.label_id), []).append(descriptor)

    rows: list[dict[str, Any]] = []
    for (backend, label_id), items in grouped.items():
        pair_indices = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]
        rng.shuffle(pair_indices)
        if max_pairs > 0:
            pair_indices = pair_indices[:max_pairs]
        for i, j in pair_indices:
            left = items[i]
            right = items[j]
            metrics = descriptor_pair_metrics(
                left,
                right,
                mean_weight=mean_weight,
                std_weight=std_weight,
                pooled_cosine_weight=pooled_cosine_weight,
            )
            rows.append(
                {
                    "backend": backend,
                    "label_id": int(label_id),
                    "label_name": left.label_name,
                    "left_sample_id": left.entry.sample_id,
                    "right_sample_id": right.entry.sample_id,
                    "left_image_path": str(left.entry.image_path),
                    "right_image_path": str(right.entry.image_path),
                    "left_token_count": left.token_count,
                    "right_token_count": right.token_count,
                    **metrics,
                }
            )
    return rows


def build_summary(
    texture_rows: list[dict[str, Any]],
    natural_rows: list[dict[str, Any]],
    *,
    label_ids: list[int],
    label_lookup: dict[str, int],
    backends: list[str],
    perturbation_names: list[str],
    max_color_mean_l1: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_backend_label_perturbation": {}}
    metric_names = (
        "aligned_token_l1_distance",
        "aligned_token_cosine_distance",
        "mean_l1_distance",
        "std_l1_distance",
        "mean_cosine_distance",
        "std_cosine_distance",
        "concat_cosine_distance",
        "two_token_average_cosine_distance",
        "region_loss_style_distance",
        "rgb_region_mean_l1",
        "rgb_region_std_l1",
        "rgb_region_pixel_l1",
    )
    natural_metric_names = (
        "mean_l1_distance",
        "std_l1_distance",
        "mean_cosine_distance",
        "std_cosine_distance",
        "concat_cosine_distance",
        "two_token_average_cosine_distance",
        "region_loss_style_distance",
    )
    primary = "region_loss_style_distance"
    for backend in backends:
        backend_summary: dict[str, Any] = {}
        for label_id in label_ids:
            label_key = f"{int(label_id)}:{canonical_label_name(label_id, label_lookup, fallback=str(label_id))}"
            label_summary: dict[str, Any] = {}
            natural_for_label = [
                row for row in natural_rows if row["backend"] == backend and int(row["label_id"]) == int(label_id)
            ]
            natural_values = [float(row[primary]) for row in natural_for_label]
            natural_stats = {
                metric: describe_values(torch.tensor([float(row[metric]) for row in natural_for_label]))
                for metric in natural_metric_names
            }
            for perturbation_name in perturbation_names:
                rows = [
                    row
                    for row in texture_rows
                    if row["backend"] == backend
                    and int(row["label_id"]) == int(label_id)
                    and row["perturbation"] == perturbation_name
                ]
                texture_values = [float(row[primary]) for row in rows]
                family = first_nonempty([str(row.get("perturbation_family", "")) for row in rows]) or "unknown"
                texture_stats = {
                    metric: describe_values(torch.tensor([float(row[metric]) for row in rows]))
                    for metric in metric_names
                }
                texture_mean = mean_of(texture_values)
                natural_mean = mean_of(natural_values)
                ratio = (
                    float(texture_mean / natural_mean)
                    if texture_mean is not None and natural_mean not in (None, 0.0)
                    else None
                )
                color_mean = mean_of([float(row["rgb_region_mean_l1"]) for row in rows])
                label_summary[perturbation_name] = {
                    "texture_stats": texture_stats,
                    "natural_same_label_stats": natural_stats,
                    "comparisons": {
                        "primary_metric": primary,
                        "perturbation_family": family,
                        "perturbation_minus_natural_mean": mean_difference(texture_values, natural_values),
                        "perturbation_over_natural_mean": ratio,
                        "perturbation_greater_than_natural_probability": greater_than_probability(
                            texture_values,
                            natural_values,
                        ),
                        "natural_greater_than_perturbation_probability": greater_than_probability(
                            natural_values,
                            texture_values,
                        ),
                        "texture_minus_natural_mean": mean_difference(texture_values, natural_values),
                        "texture_over_natural_mean": ratio,
                        "texture_greater_than_natural_probability": greater_than_probability(
                            texture_values,
                            natural_values,
                        ),
                        "natural_greater_than_texture_probability": greater_than_probability(
                            natural_values,
                            texture_values,
                        ),
                    },
                    "color_preservation": {
                        "rgb_region_mean_l1_mean": color_mean,
                        "max_color_mean_l1_for_texture_only": float(max_color_mean_l1),
                        "reading": (
                            "color mean drift is small"
                            if color_mean is not None and color_mean <= max_color_mean_l1
                            else "color mean drift is not negligible; interpret as mixed color+texture perturbation"
                        ),
                    },
                    "color_shift": {
                        "rgb_region_mean_l1_mean": color_mean,
                        "rgb_region_std_l1_mean": mean_of([float(row["rgb_region_std_l1"]) for row in rows]),
                        "rgb_region_pixel_l1_mean": mean_of([float(row["rgb_region_pixel_l1"]) for row in rows]),
                    },
                    "texture_sensitivity_reading": texture_sensitivity_reading(ratio),
                    "perturbation_sensitivity_reading": perturbation_sensitivity_reading(family, ratio),
                }
            backend_summary[label_key] = label_summary
        summary["by_backend_label_perturbation"][backend] = backend_summary
    return summary


def texture_sensitivity_reading(texture_over_natural: float | None) -> str:
    if texture_over_natural is None:
        return "insufficient natural or texture rows"
    if texture_over_natural < 0.10:
        return "texture perturbation is tiny relative to natural same-label patch differences"
    if texture_over_natural < 0.25:
        return "texture perturbation is visible but weak relative to natural same-label patch differences"
    return "texture perturbation produces a nontrivial encoder-space shift"


def perturbation_sensitivity_reading(family: str, perturbation_over_natural: float | None) -> str:
    if perturbation_over_natural is None:
        return "insufficient natural or perturbation rows"
    noun = "color/stain" if family == "color" else "texture" if family == "texture" else "perturbation"
    if perturbation_over_natural < 0.10:
        return f"{noun} perturbation is tiny relative to natural same-label patch differences"
    if perturbation_over_natural < 0.25:
        return f"{noun} perturbation is visible but weak relative to natural same-label patch differences"
    return f"{noun} perturbation produces a nontrivial encoder-space shift"


def parse_float_tuple(value: str, *, expected: tuple[int, ...], name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) not in expected:
        expected_text = " or ".join(str(count) for count in expected)
        raise ValueError(f"{name} expects {expected_text} comma-separated floats, got {value!r}")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{name} expects comma-separated floats, got {value!r}") from exc


def format_float_for_name(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def first_nonempty(values: list[str]) -> str | None:
    for value in values:
        if value:
            return value
    return None


def mean_of(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
