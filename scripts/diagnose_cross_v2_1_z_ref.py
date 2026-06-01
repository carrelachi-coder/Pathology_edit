"""Diagnose Cross V2.1 z_ref input capacity and x-embedder gate strength."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Cross V2.1 z_ref projection and condition shape.")
    parser.add_argument("--checkpoint", required=True, help="Cross V2.1 checkpoint dir.")
    parser.add_argument("--pretrained-model-name-or-path", default=None, help="Optional FLUX path for VAE shape diagnostics.")
    parser.add_argument("--metadata", default=None, help="Optional cross metadata json for one sample shape diagnostic.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--output-json", default=None)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_by_name[args.torch_dtype]

    from controlnet_train.inference.pipeline_cross_v2_1 import (
        _encode_images_to_latents,
        _load_cross_v2_1_control_spec,
        _load_diffusers_model_state_dict,
    )
    from controlnet_train.modules.cross_v2_1_conditioning import (
        build_cross_v2_1_condition,
    )

    checkpoint = Path(args.checkpoint)
    control_spec = _load_cross_v2_1_control_spec(checkpoint)
    state_dict = _load_diffusers_model_state_dict(checkpoint)
    x_weight = state_dict.get("controlnet_x_embedder.weight")
    if x_weight is None:
        raise KeyError(f"Missing controlnet_x_embedder.weight under checkpoint: {checkpoint}")

    projection = summarize_x_embedder_projection(x_weight.float(), control_spec)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "control_spec": {
            "reference_latent_channels": int(control_spec.reference_latent_channels),
            "tissue_channels": int(control_spec.tissue_channels),
            "nuclei_channels": int(control_spec.nuclei_channels),
            "raw_channels": int(control_spec.raw_channels),
            "packed_channels": int(control_spec.packed_channels),
            "packed_reference_latent_channels": int(control_spec.packed_reference_latent_channels),
            "packed_mask_channels": int(control_spec.packed_mask_channels),
            "packed_reference_mask_start": int(control_spec.packed_reference_mask_start),
            "packed_target_mask_start": int(control_spec.packed_target_mask_start),
        },
        "x_embedder_projection_norms": projection,
    }

    if args.pretrained_model_name_or_path and args.metadata:
        sample_report = diagnose_one_sample_shape(
            checkpoint=checkpoint,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            metadata_path=Path(args.metadata),
            sample_index=args.sample_index,
            device=args.device,
            dtype=dtype,
            control_spec=control_spec,
            encode_images_to_latents=_encode_images_to_latents,
            build_cross_v2_1_condition_fn=build_cross_v2_1_condition,
        )
        report["sample_shape_diagnostic"] = sample_report

    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )
    return 0


def summarize_x_embedder_projection(weight: torch.Tensor, control_spec) -> dict[str, Any]:
    spans = {
        "z_ref": (0, int(control_spec.packed_reference_latent_channels)),
        "ref_masks": (
            int(control_spec.packed_reference_mask_start),
            int(control_spec.packed_target_mask_start),
        ),
        "target_masks": (
            int(control_spec.packed_target_mask_start),
            int(control_spec.packed_channels),
        ),
    }
    total_norm = float(torch.linalg.vector_norm(weight).item())
    summary: dict[str, Any] = {
        "weight_shape": list(weight.shape),
        "total_fro_norm": total_norm,
        "groups": {},
    }
    for name, (start, end) in spans.items():
        block = weight[:, start:end]
        norm = float(torch.linalg.vector_norm(block).item())
        per_input_norm = torch.linalg.vector_norm(block, dim=0)
        summary["groups"][name] = {
            "input_span": [start, end],
            "width": int(end - start),
            "fro_norm": norm,
            "fraction_of_total_norm": float(norm / total_norm) if total_norm > 0 else math.nan,
            "mean_column_norm": float(per_input_norm.mean().item()) if per_input_norm.numel() else math.nan,
            "max_column_norm": float(per_input_norm.max().item()) if per_input_norm.numel() else math.nan,
            "zero_column_count": int((per_input_norm == 0).sum().item()),
        }
    z_norm = summary["groups"]["z_ref"]["fro_norm"]
    ref_mask_norm = summary["groups"]["ref_masks"]["fro_norm"]
    target_mask_norm = summary["groups"]["target_masks"]["fro_norm"]
    summary["ratios"] = {
        "z_ref_over_ref_masks": float(z_norm / ref_mask_norm) if ref_mask_norm > 0 else math.inf,
        "z_ref_over_target_masks": float(z_norm / target_mask_norm) if target_mask_norm > 0 else math.inf,
    }
    summary["interpretation"] = (
        "z_ref gate is near zero relative to mask gates"
        if z_norm < 1e-6 or (ref_mask_norm > 0 and z_norm / ref_mask_norm < 0.05)
        else "z_ref gate has non-trivial projection norm"
    )
    return summary


def diagnose_one_sample_shape(
    *,
    checkpoint: Path,
    pretrained_model_name_or_path: str,
    metadata_path: Path,
    sample_index: int,
    device: str,
    dtype: torch.dtype,
    control_spec,
    encode_images_to_latents,
    build_cross_v2_1_condition_fn,
) -> dict[str, Any]:
    from diffusers import AutoencoderKL

    from controlnet_train.cli.eval_controlnet_flux_cross import read_cross_metadata
    from controlnet_train.data.common import load_image_tensor, load_nuclei_mask, load_tissue_mask
    from controlnet_train.inference.pipeline_cross_v2_1 import _torch_load_weights

    records = read_cross_metadata(metadata_path)
    if not records:
        raise ValueError(f"No records found in metadata: {metadata_path}")
    record = records[sample_index % len(records)]
    reference_image = load_image_tensor(record["reference_image"]).unsqueeze(0)
    reference_tissue_mask = load_tissue_mask(record["reference_tissue_mask"]).unsqueeze(0)
    reference_nuclei_mask = load_nuclei_mask(record["reference_nuclei_mask"]).unsqueeze(0)
    target_tissue_mask = load_tissue_mask(record["target_tissue_mask"]).unsqueeze(0)
    target_nuclei_mask = load_nuclei_mask(record["target_nuclei_mask"]).unsqueeze(0)

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    state = _torch_load_weights(checkpoint / "phase5_conditioning.pt")
    modules = build_condition_modules_from_state(state, device=device, dtype=dtype)

    with torch.no_grad():
        z_ref = encode_images_to_latents(vae, reference_image, dtype)
        ref_tissue_feat = modules["tissue_downsampler"](
            modules["hte"](reference_tissue_mask.to(device=device))
        ).to(dtype=dtype)
        ref_nuclei_feat = modules["nuclei_encoder"](reference_nuclei_mask.to(device=device)).to(dtype=dtype)
        tar_tissue_feat = modules["tissue_downsampler"](
            modules["hte"](target_tissue_mask.to(device=device))
        ).to(dtype=dtype)
        tar_nuclei_feat = modules["nuclei_encoder"](target_nuclei_mask.to(device=device)).to(dtype=dtype)
        control_tensor = build_cross_v2_1_condition_fn(
            z_ref=z_ref,
            ref_tissue_feat=ref_tissue_feat,
            ref_nuclei_feat=ref_nuclei_feat,
            tar_tissue_feat=tar_tissue_feat,
            tar_nuclei_feat=tar_nuclei_feat,
        )

    return {
        "sample_id": str(record.get("sample_id", "")),
        "reference_sample_id": str(record.get("reference_sample_id", "")),
        "reference_image_shape_chw": list(reference_image.shape[1:]),
        "z_ref_shape": list(z_ref.shape),
        "ref_tissue_feat_shape": list(ref_tissue_feat.shape),
        "ref_nuclei_feat_shape": list(ref_nuclei_feat.shape),
        "control_tensor_shape": list(control_tensor.shape),
        "expected_raw_channels": int(control_spec.raw_channels),
        "z_ref_stats": tensor_stats(z_ref),
        "control_z_ref_slice_stats": tensor_stats(control_tensor[:, : int(control_spec.reference_latent_channels)]),
    }


def build_condition_modules_from_state(state: dict[str, Any], *, device: str, dtype: torch.dtype) -> dict[str, torch.nn.Module]:
    from controlnet_train.modules import (
        HierarchicalTissueEmbedding,
        NucleiConditionEncoder,
        TissueConditionDownsampler,
    )

    hte_state = state["hte"]
    tissue_state = state["tissue_downsampler"]
    nuclei_state = state["nuclei_encoder"]
    hte_dim = hte_state["parent_embeddings.weight"].shape[1]
    tissue_in = tissue_state["blocks.0.block.0.weight"].shape[1]
    tissue_hidden = tissue_state["blocks.0.block.0.weight"].shape[0]
    tissue_blocks = count_conv_blocks(tissue_state, "blocks")
    tissue_out = tissue_state[f"blocks.{tissue_blocks - 1}.block.0.weight"].shape[0]
    nuclei_embed = nuclei_state["embedding.weight"].shape[1]
    nuclei_out = nuclei_state["downsampler.0.block.0.weight"].shape[0]
    nuclei_blocks = count_conv_blocks(nuclei_state, "downsampler")
    modules = {
        "hte": HierarchicalTissueEmbedding(embedding_dim=hte_dim),
        "tissue_downsampler": TissueConditionDownsampler(
            in_channels=tissue_in,
            hidden_channels=tissue_hidden,
            out_channels=tissue_out,
            num_blocks=tissue_blocks,
        ),
        "nuclei_encoder": NucleiConditionEncoder(
            embedding_dim=nuclei_embed,
            out_channels=nuclei_out,
            num_blocks=nuclei_blocks,
        ),
    }
    for name, module in modules.items():
        module.load_state_dict(state[name])
        module.to(device=device, dtype=dtype)
        module.eval()
    return modules


def count_conv_blocks(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    return len(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith(prefix) and key.endswith("block.0.weight")
        }
    )


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.detach().float()
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "l2_norm": float(torch.linalg.vector_norm(value).item()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
