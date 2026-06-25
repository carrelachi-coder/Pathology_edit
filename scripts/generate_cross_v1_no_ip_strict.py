#!/usr/bin/env python3
"""Strict Cross V1 no-IP inference.

This script is intended to be equivalent to the original Cross V1 inference with
IP-Adapter scale set to 0, while avoiding both IP-Adapter and UNI-2h loading.
It reuses the original ControlNet loader, Cross V1 condition builder, and sampler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image


def _read_metadata_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and "pairs" in payload:
        return list(payload["pairs"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported metadata format: {path}")


def _select_record(records: list[dict[str, Any]], metadata_index: int | None, sample_id: str | None) -> dict[str, Any]:
    if metadata_index is not None:
        for record in records:
            if int(record.get("metadata_index", -1)) == int(metadata_index):
                return record
        if 0 <= metadata_index < len(records):
            return records[metadata_index]
        raise ValueError(f"metadata_index not found: {metadata_index}")
    if sample_id:
        for record in records:
            if str(record.get("sample_id", "")) == sample_id:
                return record
        raise ValueError(f"sample_id not found: {sample_id}")
    if not records:
        raise ValueError("metadata contains no records")
    return records[0]


def _count_conv_blocks(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    return len(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith(prefix) and key.endswith("block.0.weight")
        }
    )


def _load_condition_modules_no_ref(
    *,
    checkpoint_path: Path,
    device: str,
    torch_dtype: torch.dtype,
) -> dict[str, nn.Module]:
    from controlnet_train.inference.pipeline_cross_v1 import _torch_load_weights
    from controlnet_train.modules import (
        HierarchicalTissueEmbedding,
        NucleiConditionEncoder,
        TissueConditionDownsampler,
    )

    state = _torch_load_weights(checkpoint_path / "phase5_conditioning.pt")
    hte_state = state["hte"]
    tissue_state = state["tissue_downsampler"]
    nuclei_state = state["nuclei_encoder"]

    hte_dim = hte_state["parent_embeddings.weight"].shape[1]
    tissue_in = tissue_state["blocks.0.block.0.weight"].shape[1]
    tissue_hidden = tissue_state["blocks.0.block.0.weight"].shape[0]
    tissue_out = tissue_state[
        f"blocks.{_count_conv_blocks(tissue_state, 'blocks') - 1}.block.0.weight"
    ].shape[0]
    nuclei_embed = nuclei_state["embedding.weight"].shape[1]
    nuclei_out = nuclei_state["downsampler.0.block.0.weight"].shape[0]
    nuclei_blocks = _count_conv_blocks(nuclei_state, "downsampler")

    modules: dict[str, nn.Module] = {
        "hte": HierarchicalTissueEmbedding(embedding_dim=hte_dim),
        "tissue_downsampler": TissueConditionDownsampler(
            in_channels=tissue_in,
            hidden_channels=tissue_hidden,
            out_channels=tissue_out,
            num_blocks=_count_conv_blocks(tissue_state, "blocks"),
        ),
        "nuclei_encoder": NucleiConditionEncoder(
            embedding_dim=nuclei_embed,
            out_channels=nuclei_out,
            num_blocks=nuclei_blocks,
        ),
    }

    for name, module in modules.items():
        module.load_state_dict(state[name])
        module.to(device=device, dtype=torch_dtype)
        module.eval()

    return modules


def _pil_to_neg1_tensor(image: Image.Image, image_size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    import numpy as np

    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=dtype)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        ((image.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


@torch.inference_mode()
def _run_pix2pix_transfer(
    *,
    i0_image: Image.Image,
    record: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    device: str,
    torch_dtype: torch.dtype,
    image_size: int,
    base_channels: int,
    num_heads: int,
    cross_attn_scales: str,
    upsample_mode: str,
    region_label_mode: str,
) -> None:
    from controlnet_train.pix2pix_transfer.dataset import (
        NUM_CELL_CLASSES,
        NUM_FINE,
        load_label_mask,
        load_rgb,
        one_hot_mask,
        remap_nuclei_mask,
        tissue_nuclei_region_labels,
    )
    from controlnet_train.pix2pix_transfer.regional_cross_attention import Pix2PixCrossAttnUNet

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = Pix2PixCrossAttnUNet(
        in_ch=3 + NUM_FINE + (NUM_CELL_CLASSES + 1),
        out_ch=3,
        base=base_channels,
        num_heads=num_heads,
        use_region_mask=True,
        residual_output=True,
        cross_attn_scales=cross_attn_scales,
        upsample_mode=upsample_mode,
    ).to(device=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(dtype=torch_dtype)
    model.eval()

    i0 = _pil_to_neg1_tensor(i0_image, image_size, device, torch_dtype)
    reference = load_rgb(record["reference_image"], image_size).unsqueeze(0).to(device=device, dtype=torch_dtype)
    target_tissue = load_label_mask(record["target_tissue_mask"], image_size)
    reference_tissue = load_label_mask(record["reference_tissue_mask"], image_size)
    target_nuclei = remap_nuclei_mask(load_label_mask(record["target_nuclei_mask"], image_size))
    reference_nuclei = remap_nuclei_mask(load_label_mask(record["reference_nuclei_mask"], image_size))

    target_cond = torch.cat(
        [
            i0.squeeze(0).cpu(),
            one_hot_mask(target_tissue, NUM_FINE),
            one_hot_mask(target_nuclei, NUM_CELL_CLASSES + 1),
        ],
        dim=0,
    ).unsqueeze(0).to(device=device, dtype=torch_dtype)
    reference_cond = torch.cat(
        [
            reference.squeeze(0).cpu(),
            one_hot_mask(reference_tissue, NUM_FINE),
            one_hot_mask(reference_nuclei, NUM_CELL_CLASSES + 1),
        ],
        dim=0,
    ).unsqueeze(0).to(device=device, dtype=torch_dtype)
    target_region = tissue_nuclei_region_labels(
        target_tissue,
        target_nuclei,
        label_mode=region_label_mode,
    ).unsqueeze(0).to(device=device)
    reference_region = tissue_nuclei_region_labels(
        reference_tissue,
        reference_nuclei,
        label_mode=region_label_mode,
    ).unsqueeze(0).to(device=device)

    pred = model(
        target_cond,
        reference_cond,
        target_region=target_region,
        reference_region=reference_region,
    )[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _tensor_to_pil(pred).save(output)
    print(f"[pix2pix] Saved epoch transfer: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict Cross V1 no-IP / no-UNI generation")
    parser.add_argument("--checkpoint", required=True, help="Cross V1 checkpoint dir")
    parser.add_argument("--pretrained-model", default="/data/huggingface/FLUX.1-dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", default="")
    parser.add_argument("--metadata-index", type=int, default=None)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--target-tissue-mask", default="")
    parser.add_argument("--target-nuclei-mask", default="")
    parser.add_argument("--reference-tissue-mask", default="")
    parser.add_argument("--reference-nuclei-mask", default="")
    parser.add_argument(
        "--stage1-reference-mask-mode",
        choices=("target", "metadata"),
        default="target",
        help=(
            "Stage-1 source/reference spatial slots. 'target' duplicates target masks "
            "so Stage 1 uses only target layout; 'metadata' keeps original reference masks."
        ),
    )
    parser.add_argument("--prompt", default=None, help="Override prompt; omit to use metadata prompt")
    parser.add_argument("--prompt-mode", choices=("dataset", "empty"), default="dataset")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument(
        "--pix2pix-checkpoint",
        default="",
        help="Optional pix2pix texture-transfer checkpoint, e.g. .../ckpt/epoch0014.pt.",
    )
    parser.add_argument(
        "--pix2pix-output",
        default="",
        help="Output path for pix2pix-transferred image. Defaults to <output stem>_pix2pix.png.",
    )
    parser.add_argument("--pix2pix-base-channels", type=int, default=64)
    parser.add_argument("--pix2pix-num-heads", type=int, default=4)
    parser.add_argument("--pix2pix-cross-attn-scales", default="1/4,1/8,1/16")
    parser.add_argument("--pix2pix-upsample-mode", choices=("bilinear", "nearest"), default="bilinear")
    parser.add_argument(
        "--pix2pix-region-label-mode",
        choices=("tissue", "nuclei", "tissue_nuclei"),
        default="tissue_nuclei",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()

    import sys

    sys.path.insert(0, ".")
    from controlnet_train.data.common import load_nuclei_mask, load_tissue_mask
    from controlnet_train.inference.pipeline_cross_v1 import (
        _load_cross_v1_control_spec,
        _load_flux_controlnet_pipeline,
        _resolve_device,
        _resolve_torch_dtype,
        _sample_with_flux_controlnet,
        _validate_checkpoint_dir,
    )
    from controlnet_train.modules.cross_v1_conditioning import build_cross_v1_condition

    device = _resolve_device(args.device)
    requested_dtype = None
    if args.torch_dtype == "bf16":
        requested_dtype = torch.bfloat16
    elif args.torch_dtype == "fp16":
        requested_dtype = torch.float16
    elif args.torch_dtype == "fp32":
        requested_dtype = torch.float32
    torch_dtype = _resolve_torch_dtype(requested_dtype, device)

    checkpoint = _validate_checkpoint_dir(args.checkpoint)
    control_spec = _load_cross_v1_control_spec(checkpoint)

    record: dict[str, Any] = {}
    if args.metadata:
        record = _select_record(
            _read_metadata_records(args.metadata),
            args.metadata_index,
            args.sample_id or None,
        )

    target_tissue_path = args.target_tissue_mask or record.get("target_tissue_mask")
    target_nuclei_path = args.target_nuclei_mask or record.get("target_nuclei_mask")
    if not all([target_tissue_path, target_nuclei_path]):
        raise ValueError(
            "Need target tissue+nuclei masks, either via --metadata or explicit mask paths."
        )
    if args.stage1_reference_mask_mode == "target":
        reference_tissue_path = target_tissue_path
        reference_nuclei_path = target_nuclei_path
    else:
        reference_tissue_path = args.reference_tissue_mask or record.get("reference_tissue_mask")
        reference_nuclei_path = args.reference_nuclei_mask or record.get("reference_nuclei_mask")
        if not all([reference_tissue_path, reference_nuclei_path]):
            raise ValueError(
                "--stage1-reference-mask-mode metadata requires reference tissue+nuclei masks."
            )

    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_mode == "empty":
        prompt = ""
    else:
        prompt = str(record.get("prompt") or "")

    print("[CrossV1 strict no-IP] Loading ControlNet with original strict loader")
    print("  - No phase5_ip_adapter.pt is loaded")
    print("  - No ReferenceImageEncoder / UNI-2h is constructed")
    print(f"  - checkpoint={checkpoint}")
    print(f"  - prompt_mode={args.prompt_mode} prompt_len={len(prompt)}")
    print(f"  - stage1_reference_mask_mode={args.stage1_reference_mask_mode}")

    pipe, controlnet = _load_flux_controlnet_pipeline(
        pretrained_model_name_or_path=args.pretrained_model,
        checkpoint_path=checkpoint,
        packed_channels=control_spec.packed_channels,
        device=device,
        torch_dtype=torch_dtype,
    )
    modules = _load_condition_modules_no_ref(
        checkpoint_path=checkpoint,
        device=device,
        torch_dtype=torch_dtype,
    )

    target_tissue_mask = load_tissue_mask(target_tissue_path)
    target_nuclei_mask = load_nuclei_mask(target_nuclei_path)
    reference_tissue_mask = load_tissue_mask(reference_tissue_path)
    reference_nuclei_mask = load_nuclei_mask(reference_nuclei_path)

    target_tissue_feat = modules["tissue_downsampler"](
        modules["hte"](target_tissue_mask.unsqueeze(0).to(device=device))
    ).to(dtype=torch_dtype)
    target_nuclei_feat = modules["nuclei_encoder"](
        target_nuclei_mask.unsqueeze(0).to(device=device)
    ).to(dtype=torch_dtype)

    reference_tissue_feat = None
    reference_nuclei_feat = None
    if control_spec.spatial_mode in {"reference_target", "reference_target_delta"}:
        reference_tissue_feat = modules["tissue_downsampler"](
            modules["hte"](reference_tissue_mask.unsqueeze(0).to(device=device))
        ).to(dtype=torch_dtype)
        reference_nuclei_feat = modules["nuclei_encoder"](
            reference_nuclei_mask.unsqueeze(0).to(device=device)
        ).to(dtype=torch_dtype)

    control_tensor = build_cross_v1_condition(
        reference_tissue_feat=reference_tissue_feat,
        reference_nuclei_feat=reference_nuclei_feat,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
        spatial_mode=control_spec.spatial_mode,
    )

    output_size = tuple(int(v) for v in target_tissue_mask.shape[-2:])
    image = _sample_with_flux_controlnet(
        pipe=pipe,
        controlnet=controlnet,
        prompt=prompt,
        control_tensor=control_tensor,
        source_control_tensor=None,
        source_latents=None,
        source_latent_init_strength=0.0,
        mask_chord_scale=0.0,
        mask_chord_change_mask=None,
        mask_chord_gate_dilate_radius=0,
        mask_chord_gate_feather_radius=0,
        mask_chord_gate_outside_scale=0.0,
        output_size=output_size,
        device=device,
        torch_dtype=torch_dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.control_scale,
        joint_attention_kwargs=None,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"[CrossV1 strict no-IP] Saved: {output}")

    if args.pix2pix_checkpoint:
        if not record:
            raise ValueError("--pix2pix-checkpoint requires --metadata so reference image/masks are available.")
        pix2pix_output = (
            Path(args.pix2pix_output)
            if args.pix2pix_output
            else output.with_name(f"{output.stem}_pix2pix{output.suffix}")
        )
        _run_pix2pix_transfer(
            i0_image=image,
            record=record,
            checkpoint_path=args.pix2pix_checkpoint,
            output_path=pix2pix_output,
            device=device,
            torch_dtype=torch_dtype,
            image_size=int(output_size[0]),
            base_channels=args.pix2pix_base_channels,
            num_heads=args.pix2pix_num_heads,
            cross_attn_scales=args.pix2pix_cross_attn_scales,
            upsample_mode=args.pix2pix_upsample_mode,
            region_label_mode=args.pix2pix_region_label_mode,
        )


if __name__ == "__main__":
    main()
