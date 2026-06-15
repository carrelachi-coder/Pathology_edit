"""Fixed-t Cross V1 paired-vs-alternate forward loss-gap grid.

This diagnostic runs no diffusion sampling. For each selected probe it fixes
the target latent, noise, prompt, target/reference labels, and control tensor,
then evaluates paired-reference and alternate-reference IP branches at a grid
of normalized RF timesteps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_cross_v1_directionality_gate import select_records_from_manifest  # noqa: E402
from diagnose_cross_v1_generation_gate import (  # noqa: E402
    ALTERNATE_MODES,
    build_manifest_row,
    parse_indices,
    parse_mode_selection,
    read_records,
    record_sample_id,
    reference_case_id,
    reference_sample_id,
    resolve_prompt,
    select_gate_records,
)


class DeviceShim:
    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)


def stable_int_hash(value: str) -> int:
    result = 0
    for character in value:
        result = (result * 131 + ord(character)) % 1_000_000_007
    return result


def bootstrap_stderr(values: list[float], *, iters: int, seed: int) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return math.nan
    if iters <= 0:
        mean = sum(finite) / len(finite)
        variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
        return math.sqrt(variance) / math.sqrt(len(finite))
    rng = random.Random(int(seed))
    means = []
    n = len(finite)
    for _ in range(int(iters)):
        total = 0.0
        for _ in range(n):
            total += finite[rng.randrange(n)]
        means.append(total / n)
    mean = sum(means) / len(means)
    variance = sum((value - mean) ** 2 for value in means) / (len(means) - 1)
    return math.sqrt(variance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross V1 fixed-t forward loss-gap grid.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Eval-ready Cross V1 checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-manifest", default=os.environ.get("PROBE_SELECTION_MANIFEST"))
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--record-indices", default="")
    parser.add_argument("--selection-seed", type=int, default=20260611)
    parser.add_argument("--noise-seed", type=int, default=20260613)
    parser.add_argument("--t-values", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--alternate-mode", choices=("same_dataset", "different_dataset", "both"), default="both")
    parser.add_argument("--prompt-mode", choices=("dataset", "empty"), default="dataset")
    parser.add_argument("--prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--tumor-label", type=int, default=1)
    parser.add_argument("--min-tumor-fraction", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0)
    parser.add_argument(
        "--deterministic-epsilon",
        action="store_true",
        help="Derive per-(probe,t) epsilon from stable hashes instead of one batch noise tensor.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations for per-t summary stderr.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260611,
        help="Bootstrap seed for per-t summary stderr.",
    )
    parser.add_argument(
        "--dump-noisy-dir",
        default=None,
        help="Optional directory to dump decoded x_t images for visual t-direction checks.",
    )
    parser.add_argument("--max-probes", type=int, default=0, help="Debug cap after selection. 0 disables.")
    parser.add_argument("--selection-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_noisy_dir = Path(args.dump_noisy_dir) if args.dump_noisy_dir else None
    if dump_noisy_dir is not None:
        dump_noisy_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata)
    records = read_records(metadata_path)
    t_values = parse_t_values(args.t_values)
    alternate_modes = parse_mode_selection(args.alternate_mode, ALTERNATE_MODES)

    if args.selection_manifest:
        selected = select_records_from_manifest(
            Path(args.selection_manifest),
            records,
            alternate_modes=alternate_modes,
        )
    else:
        selected = select_gate_records(
            records,
            record_indices=parse_indices(args.record_indices),
            num_samples=args.num_samples,
            seed=args.selection_seed,
            tumor_label=args.tumor_label,
            min_tumor_fraction=args.min_tumor_fraction,
            alternate_modes=alternate_modes,
        )
    if args.max_probes > 0:
        selected = selected[: int(args.max_probes)]

    manifest = [
        build_manifest_row(index, paired, alternates)
        for index, paired, alternates in selected
    ]
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    planned_forward_pairs = len(selected) * len(t_values) * len(alternate_modes)
    print(
        f"selected {len(selected)} probes; fixed-t rows={planned_forward_pairs} "
        f"({len(selected)} probes x {len(t_values)} t x {len(alternate_modes)} alt modes)",
        flush=True,
    )
    if args.selection_only:
        return 0

    from controlnet_train.data.common import (
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v1 import load_cross_v1_bundle
    from controlnet_train.training.flux_phase5_cross_v1 import (
        _build_cross_v1_control_batch,
        _build_ip_adapter_kwargs,
        _prepare_packed_latent_image_ids,
    )
    from diffusers import FluxControlNetPipeline

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bundle = load_cross_v1_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        uni_checkpoint_path=args.uni_checkpoint_path,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=1,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=args.ip_adapter_scale,
    )
    bundle.flux_pipeline.transformer.eval()
    bundle.controlnet.eval()
    for module in bundle.condition_modules.values():
        module.eval()

    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(selected), batch_size):
        batch_items = selected[start : start + batch_size]
        print(
            f"[t-grid] batch {start // batch_size + 1}/{math.ceil(len(selected) / batch_size)} "
            f"probes={start}-{start + len(batch_items) - 1}",
            flush=True,
        )
        rows.extend(
            run_t_grid_batch(
                args=args,
                bundle=bundle,
                batch_items=batch_items,
                t_values=t_values,
                alternate_modes=alternate_modes,
                load_image_tensor=load_image_tensor,
                load_tissue_mask=load_tissue_mask,
                load_nuclei_mask=load_nuclei_mask,
                FluxControlNetPipeline=FluxControlNetPipeline,
                build_cross_v1_control_batch=_build_cross_v1_control_batch,
                build_ip_adapter_kwargs=_build_ip_adapter_kwargs,
                prepare_packed_latent_image_ids=_prepare_packed_latent_image_ids,
                batch_start=start,
                dump_noisy_dir=dump_noisy_dir,
            )
        )

    write_csv(output_dir / "t_grid_metrics.csv", rows)
    summary = summarize_t_grid_rows(rows)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "metadata": str(args.metadata),
            "selection_manifest": str(args.selection_manifest) if args.selection_manifest else None,
            "num_probes": len(selected),
            "t_values": t_values,
            "alternate_modes": alternate_modes,
            "prompt_mode": args.prompt_mode,
            "noise_seed": args.noise_seed,
            "semantics": {
                "loss_gap": "alternate_feature_loss - paired_loss; positive means paired reference is better",
                "t_values": "normalized RF timestep/sigma values used both for noising and transformer timestep",
                "variant": "alternate swaps reference image only while retaining paired reference labels/control",
            },
        }
    )
    (output_dir / "t_grid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


def parse_t_values(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("--t-values must contain at least one value")
    for item in values:
        if not math.isfinite(item) or item <= 0.0 or item >= 1.0:
            raise ValueError(f"t values must be finite and within (0,1), got {item!r}")
    return values


def run_t_grid_batch(
    *,
    args: argparse.Namespace,
    bundle,
    batch_items: list[tuple[int, dict[str, Any], dict[str, dict[str, Any]]]],
    t_values: list[float],
    alternate_modes: list[str],
    load_image_tensor,
    load_tissue_mask,
    load_nuclei_mask,
    FluxControlNetPipeline,
    build_cross_v1_control_batch,
    build_ip_adapter_kwargs,
    prepare_packed_latent_image_ids,
    batch_start: int,
    dump_noisy_dir: Path | None,
) -> list[dict[str, Any]]:
    device = torch.device(bundle.device)
    weight_dtype = bundle.torch_dtype
    paired_records = [item[1] for item in batch_items]
    alternates_by_mode = {
        mode: [item[2][mode] for item in batch_items]
        for mode in alternate_modes
    }
    batch_probe_keys = [
        "|".join(
            [
                str(batch_start + local_index),
                record_sample_id(row),
                reference_sample_id(row),
                reference_case_id(row),
            ]
        )
        for local_index, row in enumerate(paired_records)
    ]
    base_batch = {
        "target_image": torch.stack([load_image_tensor(row["target_image"]) for row in paired_records]),
        "reference_image": torch.stack([load_image_tensor(row["reference_image"]) for row in paired_records]),
        "target_tissue_mask": torch.stack([load_tissue_mask(row["target_tissue_mask"]) for row in paired_records]),
        "target_nuclei_mask": torch.stack([load_nuclei_mask(row["target_nuclei_mask"]) for row in paired_records]),
        "reference_tissue_mask": torch.stack([load_tissue_mask(row["reference_tissue_mask"]) for row in paired_records]),
        "reference_nuclei_mask": torch.stack([load_nuclei_mask(row["reference_nuclei_mask"]) for row in paired_records]),
    }
    prompts = [
        "" if args.prompt_mode == "empty" else resolve_prompt(args, row)
        for row in paired_records
    ]
    bsz = len(paired_records)

    if dump_noisy_dir is not None:
        for local_index, row in enumerate(paired_records):
            sample_dir = dump_noisy_dir / f"{batch_start + local_index:03d}_{safe_name(record_sample_id(row))}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            if not (sample_dir / "target.png").exists():
                tensor_to_pil(load_image_tensor(row["target_image"])).save(sample_dir / "target.png")
            if not (sample_dir / "reference_paired.png").exists():
                tensor_to_pil(load_image_tensor(row["reference_image"])).save(sample_dir / "reference_paired.png")

    with torch.no_grad():
        pixel_latents, noising_latents, control_tensor = build_cross_v1_control_batch(
            batch=base_batch,
            modules=bundle.condition_modules,
            vae=bundle.flux_pipeline.vae,
            weight_dtype=weight_dtype,
            spatial_mode=bundle.control_spec.spatial_mode,
        )
        packed_pixel_latents = FluxControlNetPipeline._pack_latents(
            pixel_latents,
            bsz,
            pixel_latents.shape[1],
            pixel_latents.shape[2],
            pixel_latents.shape[3],
        )
        packed_noising_latents = FluxControlNetPipeline._pack_latents(
            noising_latents,
            bsz,
            noising_latents.shape[1],
            noising_latents.shape[2],
            noising_latents.shape[3],
        )
        control_image = FluxControlNetPipeline._pack_latents(
            control_tensor,
            bsz,
            control_tensor.shape[1],
            control_tensor.shape[2],
            control_tensor.shape[3],
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = bundle.flux_pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=weight_dtype)
        if text_ids.dim() == 3:
            text_ids = text_ids[0]
        text_ids = text_ids.to(device=device, dtype=weight_dtype)
        latent_image_ids = prepare_packed_latent_image_ids(
            packed_height=pixel_latents.shape[2] // 2,
            packed_width=pixel_latents.shape[3] // 2,
            device=device,
            dtype=weight_dtype,
        )
        if latent_image_ids.shape[0] != packed_pixel_latents.shape[1]:
            raise ValueError(
                f"latent image id length mismatch: {latent_image_ids.shape[0]} vs {packed_pixel_latents.shape[1]}"
            )

        accelerator = DeviceShim(device)
        normal_kwargs = build_ip_adapter_kwargs(
            base_batch,
            bundle.condition_modules,
            accelerator,
            weight_dtype,
            bundle.flux_pipeline.transformer,
            regional=bool(bundle.regional_ip_adapter),
            query_token_count=int(packed_pixel_latents.shape[1]),
            strict=bool(bundle.regional_ip_strict),
            regional_token_mode=bundle.regional_ip_token_mode,
            regional_label_mode=bundle.regional_ip_label_mode,
            use_soft_bias=bool(getattr(bundle, "regional_ip_use_soft_bias", False)),
        )
        alt_kwargs_by_mode = {}
        for mode, alternate_records in alternates_by_mode.items():
            alt_batch = dict(base_batch)
            alt_batch["reference_image"] = torch.stack(
                [load_image_tensor(row["reference_image"]) for row in alternate_records]
            )
            # Keep paired labels/control: this is the real_feature arm.
            alt_kwargs_by_mode[mode] = build_ip_adapter_kwargs(
                alt_batch,
                bundle.condition_modules,
                accelerator,
                weight_dtype,
                bundle.flux_pipeline.transformer,
                regional=bool(bundle.regional_ip_adapter),
                query_token_count=int(packed_pixel_latents.shape[1]),
                strict=bool(bundle.regional_ip_strict),
                regional_token_mode=bundle.regional_ip_token_mode,
                regional_label_mode=bundle.regional_ip_label_mode,
                use_soft_bias=bool(getattr(bundle, "regional_ip_use_soft_bias", False)),
            )

        output_rows: list[dict[str, Any]] = []
        controlnet_blocks_repeat = False if getattr(bundle.controlnet, "input_hint_block", None) is None else True
        for t_value in t_values:
            sigma = torch.full(
                (bsz, 1, 1),
                float(t_value),
                device=packed_pixel_latents.device,
                dtype=packed_pixel_latents.dtype,
            )
            if args.deterministic_epsilon:
                noise = torch.stack(
                    [
                        torch.randn(
                            packed_pixel_latents[i : i + 1].shape,
                            generator=torch.Generator(device=device).manual_seed(
                                int(args.noise_seed)
                                + stable_int_hash(f"{batch_probe_keys[i]}|{t_value:g}")
                            ),
                            device=packed_pixel_latents.device,
                            dtype=packed_pixel_latents.dtype,
                        )[0]
                        for i in range(bsz)
                    ],
                    dim=0,
                )
            else:
                generator = torch.Generator(device=device).manual_seed(int(args.noise_seed) + int(batch_start))
                noise = torch.randn(
                    packed_pixel_latents.shape,
                    generator=generator,
                    device=packed_pixel_latents.device,
                    dtype=packed_pixel_latents.dtype,
                )
            noisy_model_input = (1.0 - sigma) * packed_noising_latents + sigma * noise
            if dump_noisy_dir is not None:
                for local_index, row in enumerate(paired_records):
                    sample_dir = dump_noisy_dir / f"{batch_start + local_index:03d}_{safe_name(record_sample_id(row))}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    xt_path = sample_dir / f"x_t_{format_t_value(t_value)}.png"
                    if not xt_path.exists():
                        decode_packed_latent_to_pil(
                            bundle=bundle,
                            packed_latent=noisy_model_input[local_index : local_index + 1],
                            output_size=tuple(int(v) for v in row["target_image"].shape[-2:])
                            if torch.is_tensor(row["target_image"])
                            else tuple(int(v) for v in base_batch["target_image"].shape[-2:]),
                        ).save(xt_path)
            target_velocity = (noisy_model_input - packed_pixel_latents) / sigma.clamp_min(1e-6)
            timestep = torch.full(
                (bsz,),
                float(t_value),
                device=packed_pixel_latents.device,
                dtype=packed_pixel_latents.dtype,
            )
            controlnet_guidance = None
            if bundle.controlnet.config.guidance_embeds:
                controlnet_guidance = torch.full(
                    (bsz,),
                    float(args.guidance_scale),
                    device=device,
                    dtype=packed_pixel_latents.dtype,
                )
            transformer_guidance = None
            if bundle.flux_pipeline.transformer.config.guidance_embeds:
                transformer_guidance = torch.full(
                    (bsz,),
                    float(args.guidance_scale),
                    device=device,
                    dtype=packed_pixel_latents.dtype,
                )
            controlnet_block_samples, controlnet_single_block_samples = bundle.controlnet(
                hidden_states=noisy_model_input,
                controlnet_cond=control_image,
                controlnet_mode=None,
                conditioning_scale=float(args.controlnet_conditioning_scale),
                timestep=timestep,
                guidance=controlnet_guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )
            normal_pred = predict_with_ip(
                bundle=bundle,
                hidden_states=noisy_model_input,
                timestep=timestep,
                guidance=transformer_guidance,
                pooled_prompt_embeds=pooled_prompt_embeds,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                latent_image_ids=latent_image_ids,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                joint_attention_kwargs=normal_kwargs,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )
            normal_loss = per_sample_mse(normal_pred, target_velocity)
            for mode, alt_kwargs in alt_kwargs_by_mode.items():
                alt_pred = predict_with_ip(
                    bundle=bundle,
                    hidden_states=noisy_model_input,
                    timestep=timestep,
                    guidance=transformer_guidance,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    joint_attention_kwargs=alt_kwargs,
                    controlnet_blocks_repeat=controlnet_blocks_repeat,
                )
                alt_loss = per_sample_mse(alt_pred, target_velocity)
                pred_l2 = torch.sqrt(torch.mean((alt_pred.float() - normal_pred.float()) ** 2, dim=(1, 2)))
                alternate_records = alternates_by_mode[mode]
                for local_index, (metadata_index, paired, _) in enumerate(batch_items):
                    gap = float((alt_loss[local_index] - normal_loss[local_index]).detach().cpu().item())
                    output_rows.append(
                        {
                            "metadata_index": metadata_index,
                            "sample_id": record_sample_id(paired),
                            "dataset": paired.get("dataset", ""),
                            "paired_reference_sample_id": reference_sample_id(paired),
                            "alternate_reference_sample_id": reference_sample_id(alternate_records[local_index]),
                            "paired_reference_case_id": reference_case_id(paired),
                            "alternate_reference_case_id": reference_case_id(alternate_records[local_index]),
                            "alternate_mode": mode,
                            "prompt_mode": args.prompt_mode,
                            "t": float(t_value),
                            "paired_loss": float(normal_loss[local_index].detach().cpu().item()),
                            "alternate_feature_loss": float(alt_loss[local_index].detach().cpu().item()),
                            "loss_gap": gap,
                            "paired_win": gap > 0.0,
                            "pred_l2": float(pred_l2[local_index].detach().cpu().item()),
                            "deterministic_epsilon": bool(args.deterministic_epsilon),
                        }
                    )
        return output_rows


def predict_with_ip(
    *,
    bundle,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    guidance: torch.Tensor | None,
    pooled_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    controlnet_block_samples,
    controlnet_single_block_samples,
    joint_attention_kwargs: dict,
    controlnet_blocks_repeat: bool,
) -> torch.Tensor:
    kwargs = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "guidance": guidance,
        "pooled_projections": pooled_prompt_embeds,
        "encoder_hidden_states": prompt_embeds,
        "controlnet_block_samples": controlnet_block_samples,
        "controlnet_single_block_samples": controlnet_single_block_samples,
        "txt_ids": text_ids,
        "img_ids": latent_image_ids,
        "joint_attention_kwargs": dict(joint_attention_kwargs),
        "return_dict": False,
    }
    if controlnet_blocks_repeat:
        kwargs["controlnet_blocks_repeat"] = True
    return bundle.flux_pipeline.transformer(**kwargs)[0]


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).pow(2).flatten(1).mean(dim=1)


def summarize_t_grid_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_t = {}
    for t_value in sorted({float(row["t"]) for row in rows}):
        by_t[f"{t_value:g}"] = summarize_rows([row for row in rows if float(row["t"]) == t_value])
    by_mode = {}
    for mode in sorted({str(row["alternate_mode"]) for row in rows}):
        mode_rows = [row for row in rows if str(row["alternate_mode"]) == mode]
        by_mode[mode] = {
            "overall": summarize_rows(mode_rows),
            "by_t": {
                f"{t_value:g}": summarize_rows([row for row in mode_rows if float(row["t"]) == t_value])
                for t_value in sorted({float(row["t"]) for row in mode_rows})
            },
        }
    return {
        "overall": summarize_rows(rows),
        "by_t": by_t,
        "by_alternate_mode": by_mode,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [float(row["loss_gap"]) for row in rows if math.isfinite(float(row["loss_gap"]))]
    wins = [1.0 if bool(row["paired_win"]) else 0.0 for row in rows]
    t_boot = [float(row["loss_gap"]) for row in rows if math.isfinite(float(row["loss_gap"]))]
    return {
        "n": len(rows),
        "mean_loss_gap": finite_mean(gaps),
        "stderr_loss_gap": sample_stderr(gaps),
        "boot_se_loss_gap": bootstrap_stderr(
            t_boot,
            iters=2000,
            seed=20260611 + stable_int_hash(str(rows[0].get("t", ""))) if rows else 20260611,
        ),
        "mean_minus_2stderr": finite_mean(gaps) - 2.0 * sample_stderr(gaps) if len(gaps) > 1 else math.nan,
        "positive": int(sum(1 for gap in gaps if gap > 0.0)),
        "positive_rate": finite_mean(wins),
        "mean_pred_l2": finite_mean([float(row["pred_l2"]) for row in rows]),
        "mean_paired_loss": finite_mean([float(row["paired_loss"]) for row in rows]),
        "mean_alternate_feature_loss": finite_mean([float(row["alternate_feature_loss"]) for row in rows]),
        "effective_probe_pairs": len({(row["metadata_index"], row["sample_id"]) for row in rows}),
    }


def decode_packed_latent_to_pil(
    *,
    bundle,
    packed_latent: torch.Tensor,
    output_size: tuple[int, int],
) -> Any:
    height, width = output_size
    vae_device = next(bundle.flux_pipeline.vae.parameters()).device
    latents = bundle.flux_pipeline._unpack_latents(
        packed_latent,
        height,
        width,
        bundle.flux_pipeline.vae_scale_factor,
    )
    latents = (latents / bundle.flux_pipeline.vae.config.scaling_factor) + bundle.flux_pipeline.vae.config.shift_factor
    image = bundle.flux_pipeline.vae.decode(
        latents.to(device=vae_device, dtype=torch.float32),
        return_dict=False,
    )[0]
    return bundle.flux_pipeline.image_processor.postprocess(image, output_type="pil")[0]


def tensor_to_pil(image: torch.Tensor) -> Any:
    from PIL import Image
    array = (image.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def format_t_value(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def sample_stderr(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return math.nan
    mean = finite_mean(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(variance) / math.sqrt(len(finite))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
