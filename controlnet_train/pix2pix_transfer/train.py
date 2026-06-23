"""Train supervised I0/reference -> target pix2pix texture transfer."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .dataset import I0ReferenceTextureDataset
from .dataset import load_rgb as load_rgb_neg1
from .losses import Pix2PixTransferLoss
from .regional_cross_attention import Pix2PixCrossAttnUNet, model_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-root", default=None)
    parser.add_argument("--i0-field", default="i0_image")
    parser.add_argument(
        "--i0-cache-dir",
        default=None,
        help="Directory containing cached ControlNet I0 images named by metadata index.",
    )
    parser.add_argument("--val-metadata", default=None)
    parser.add_argument("--val-i0-cache-dir", default=None)
    parser.add_argument(
        "--lazy-generate-i0",
        action="store_true",
        help="Generate missing cached I0 images during training/eval and save them.",
    )
    parser.add_argument("--pretrained-model-name-or-path", default=None)
    parser.add_argument("--checkpoint", default=None, help="Cross V1 ControlNet checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", default=None)
    parser.add_argument("--controlnet-torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--controlnet-num-inference-steps", type=int, default=28)
    parser.add_argument("--controlnet-guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument("--source-latent-init-strength", type=float, default=0.0)
    parser.add_argument("--mask-chord-scale", type=float, default=0.0)
    parser.add_argument("--mask-chord-use-gate", action="store_true")
    parser.add_argument("--mask-chord-gate-dilate-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-feather-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-outside-scale", type=float, default=0.0)
    parser.add_argument("--i0-prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--i0-prompt", default=None)
    parser.add_argument("--i0-generation-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--upsample-mode",
        choices=("bilinear", "nearest"),
        default="bilinear",
        help=(
            "Decoder resize-conv upsampling mode. bilinear is smoother and usually "
            "reduces checkerboard artifacts; nearest is sharper but can look blockier."
        ),
    )
    parser.add_argument(
        "--cross-attn-scales",
        default="1/4,1/8,1/16",
        help=(
            "Comma-separated cross-attention scales. Default uses 1/4, 1/8 and 1/16 "
            "so fine reference texture reaches the high-resolution decoder."
        ),
    )
    parser.add_argument(
        "--region-label-mode",
        choices=("tissue", "nuclei", "tissue_nuclei"),
        default="tissue_nuclei",
    )
    parser.add_argument("--no-region-mask", action="store_true")
    parser.add_argument("--no-residual-output", action="store_true")
    parser.add_argument("--augment-flips", action="store_true")
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-perc", type=float, default=1.0)
    parser.add_argument("--lambda-gram", type=float, default=1.0)
    parser.add_argument("--lambda-contextual", type=float, default=1.0)
    parser.add_argument(
        "--l1-blur-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for low-frequency L1; 0 keeps full-resolution pixel L1.",
    )
    parser.add_argument("--content-layers", default="3,8,15,22")
    parser.add_argument("--gram-layers", default="3,8,15")
    parser.add_argument("--contextual-layers", default="8,15")
    parser.add_argument("--texture-min-pixels", type=int, default=8)
    parser.add_argument("--contextual-max-samples", type=int, default=256)
    parser.add_argument("--contextual-temperature", type=float, default=0.1)
    parser.add_argument("--loss-normalization-decay", type=float, default=0.99)
    parser.add_argument(
        "--loss-normalization-steps",
        type=int,
        default=200,
        help="Calibrate EMA loss scales for this many steps, then freeze them; <=0 never freezes.",
    )
    parser.add_argument(
        "--no-loss-normalization",
        action="store_true",
        help="Disable synchronized EMA normalization of L1/content/Gram/contextual losses.",
    )
    parser.add_argument("--vgg-weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--eval-num-samples", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, int]:
    if not is_distributed():
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        return device, 0, 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank, rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def autocast_context(device: torch.device, mixed_precision: str):
    enabled = mixed_precision != "no" and device.type == "cuda"
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def controlnet_dtype_by_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[str(name)]


def _batch_item(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        return value[index].item() if value.ndim > 0 else value.item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _batch_string(batch: dict[str, Any], key: str, index: int) -> str:
    value = batch.get(key, "")
    item = _batch_item(value, index)
    return "" if item is None else str(item)


def _batch_int(batch: dict[str, Any], key: str, index: int) -> int:
    value = batch.get(key, index)
    item = _batch_item(value, index)
    try:
        return int(item)
    except (TypeError, ValueError):
        return int(index)


def _pil_to_tensor_neg1(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _write_i0_into_batch(batch: dict[str, Any], index: int, i0: torch.Tensor) -> None:
    batch["i0"][index].copy_(i0)
    batch["target_cond"][index, :3].copy_(i0)
    missing = batch.get("i0_missing")
    if torch.is_tensor(missing):
        missing[index] = False


class LazyI0Generator:
    def __init__(self, args: argparse.Namespace, device: torch.device, *, rank: int) -> None:
        self.args = args
        self.device = device
        self.rank = int(rank)
        self.bundle = None
        self.generated = 0
        self.loaded_after_race = 0

    def _load_bundle(self):
        if self.bundle is not None:
            return self.bundle
        if not self.args.pretrained_model_name_or_path:
            raise ValueError("--lazy-generate-i0 requires --pretrained-model-name-or-path")
        if not self.args.checkpoint:
            raise ValueError("--lazy-generate-i0 requires --checkpoint")
        if not self.args.uni_checkpoint_path:
            raise ValueError("--lazy-generate-i0 requires --uni-checkpoint-path")
        from controlnet_train.inference.pipeline_cross_v1 import load_cross_v1_bundle

        self.bundle = load_cross_v1_bundle(
            pretrained_model_name_or_path=self.args.pretrained_model_name_or_path,
            checkpoint_path=self.args.checkpoint,
            uni_checkpoint_path=self.args.uni_checkpoint_path,
            device=str(self.device),
            torch_dtype=controlnet_dtype_by_name(self.args.controlnet_torch_dtype),
            num_inference_steps=self.args.controlnet_num_inference_steps,
            guidance_scale=self.args.controlnet_guidance_scale,
            controlnet_conditioning_scale=self.args.controlnet_conditioning_scale,
            ip_adapter_scale=self.args.ip_scale,
        )
        return self.bundle

    def _resolve_prompt(self, batch: dict[str, Any], index: int) -> str:
        if self.args.i0_prompt:
            return str(self.args.i0_prompt)
        if self.args.i0_prompt_source == "metadata":
            prompt = _batch_string(batch, "prompt", index)
            if prompt:
                return prompt
        if self.args.i0_prompt_source == "dataset":
            dataset_name = _batch_string(batch, "dataset", index)
            if dataset_name:
                from controlnet_train.data.common import default_prompt_for_dataset

                return default_prompt_for_dataset(dataset_name)
        prompt = _batch_string(batch, "prompt", index)
        return prompt or "H&E stained cancer histopathology at 40x magnification"

    def _missing_indices(self, batch: dict[str, Any]) -> list[int]:
        missing = batch.get("i0_missing")
        if missing is None:
            return []
        if torch.is_tensor(missing):
            return [int(i) for i in torch.nonzero(missing.bool(), as_tuple=False).view(-1).tolist()]
        return [i for i, value in enumerate(missing) if bool(value)]

    @torch.no_grad()
    def fill_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        missing_indices = self._missing_indices(batch)
        if not missing_indices:
            return batch

        from controlnet_train.inference.pipeline_cross_v1 import run_cross_v1_bundle

        bundle = self._load_bundle()
        height = int(batch["i0"].shape[-2])
        width = int(batch["i0"].shape[-1])
        for index in missing_indices:
            cache_path = Path(_batch_string(batch, "i0_cache_path", index))
            if cache_path.exists():
                i0 = load_rgb_neg1(cache_path, height)
                self.loaded_after_race += 1
                _write_i0_into_batch(batch, index, i0)
                continue

            reference_image = ((batch["reference_image"][index].float() + 1.0) * 0.5).clamp(0.0, 1.0)
            reference_tissue_mask = batch["reference_tissue_mask"][index, 0].long()
            reference_nuclei_mask = batch["reference_nuclei_mask"][index, 0].long()
            target_tissue_mask = batch["target_tissue_mask"][index, 0].long()
            target_nuclei_mask = batch["target_nuclei_mask"][index, 0].long()
            metadata_index = _batch_int(batch, "metadata_index", index)
            prompt = self._resolve_prompt(batch, index)
            image = run_cross_v1_bundle(
                bundle,
                reference_image=reference_image,
                reference_tissue_mask=reference_tissue_mask,
                reference_nuclei_mask=reference_nuclei_mask,
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                prompt=prompt,
                source_latent_init_strength=self.args.source_latent_init_strength,
                mask_chord_scale=self.args.mask_chord_scale,
                mask_chord_use_gate=self.args.mask_chord_use_gate,
                mask_chord_gate_dilate_radius=self.args.mask_chord_gate_dilate_radius,
                mask_chord_gate_feather_radius=self.args.mask_chord_gate_feather_radius,
                mask_chord_gate_outside_scale=self.args.mask_chord_gate_outside_scale,
                seed=int(self.args.i0_generation_seed) + int(metadata_index),
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(cache_path)
            i0 = _pil_to_tensor_neg1(image, (width, height))
            self.generated += 1
            _write_i0_into_batch(batch, index, i0)
        if self.generated and self.generated % 10 == 0:
            print(f"[rank {self.rank}] lazy generated I0 count={self.generated}")
        return batch


def save_training_sample(
    *,
    output_dir: Path,
    step: int,
    batch: dict[str, Any],
    pred: torch.Tensor,
    max_items: int = 4,
) -> None:
    count = min(int(max_items), int(pred.shape[0]))
    grid = [
        batch["i0"][:count].detach().cpu(),
        batch["reference_image"][:count].detach().cpu(),
        pred[:count].detach().cpu(),
        batch["target_image"][:count].detach().cpu(),
    ]
    rows = ["I0", "Reference", "Prediction", "Target"]
    cell_w, cell_h = _tensor_to_pil(grid[0][0]).size
    label_h = 24
    row_label_w = 88
    canvas = Image.new(
        "RGB",
        (row_label_w + count * cell_w, label_h + len(grid) * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for col in range(count):
        draw.text((row_label_w + col * cell_w + 6, 5), f"sample {col}", fill=(0, 0, 0))
    for row, (label, tensor) in enumerate(zip(rows, grid)):
        draw.text((6, label_h + row * cell_h + 6), label, fill=(0, 0, 0))
        for col in range(count):
            canvas.paste(
                _tensor_to_pil(tensor[col]),
                (row_label_w + col * cell_w, label_h + row * cell_h),
            )
    target_path = output_dir / "samples" / f"step{step:08d}.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target_path)


def _flat_training_sample(
    *,
    batch: dict[str, Any],
    pred: torch.Tensor,
    max_items: int,
) -> torch.Tensor:
    count = min(int(max_items), int(pred.shape[0]))
    return torch.cat(
        [
            batch["i0"][:count].detach().cpu(),
            batch["reference_image"][:count].detach().cpu(),
            pred[:count].detach().cpu(),
            batch["target_image"][:count].detach().cpu(),
        ],
        dim=0,
    )


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        ((image.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _metadata_index_list(value: Any, count: int) -> list[str]:
    if torch.is_tensor(value):
        values = value.detach().cpu().view(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value] * count
    return [str(v) for v in values[:count]]


def _save_eval_panel(
    *,
    output_path: Path,
    i0: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    pred: torch.Tensor,
    metadata_index: Any,
) -> None:
    count = int(pred.shape[0])
    if count <= 0:
        return
    columns = ["I0 ControlNet", "Target GT", "Reference GT", "Model output"]
    images = [i0, target, reference, pred]
    cell_w, cell_h = _tensor_to_pil(images[0][0]).size
    label_h = 26
    row_label_w = 92
    canvas = Image.new(
        "RGB",
        (row_label_w + len(columns) * cell_w, label_h + count * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(columns):
        draw.text((row_label_w + col * cell_w + 6, 6), label, fill=(0, 0, 0))
    row_labels = _metadata_index_list(metadata_index, count)
    for row in range(count):
        draw.text((6, label_h + row * cell_h + 6), f"idx {row_labels[row]}", fill=(0, 0, 0))
        for col, tensor in enumerate(images):
            canvas.paste(
                _tensor_to_pil(tensor[row]),
                (row_label_w + col * cell_w, label_h + row * cell_h),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.no_grad()
def save_eval_panel(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    mixed_precision: str,
    i0_generator: LazyI0Generator | None = None,
) -> None:
    model.eval()
    for batch in loader:
        if i0_generator is not None:
            batch = i0_generator.fill_batch(batch)
        target_cond = batch["target_cond"].to(device, non_blocking=True)
        reference_cond = batch["reference_cond"].to(device, non_blocking=True)
        target_region = batch["target_region"].to(device, non_blocking=True)
        reference_region = batch["reference_region"].to(device, non_blocking=True)
        with autocast_context(device, mixed_precision):
            pred = model(
                target_cond,
                reference_cond,
                target_region=target_region,
                reference_region=reference_region,
            )
        _save_eval_panel(
            output_path=output_dir / "eval" / f"epoch{epoch + 1:04d}.png",
            i0=batch["i0"],
            target=batch["target_image"],
            reference=batch["reference_image"],
            pred=pred.detach().cpu(),
            metadata_index=batch.get("metadata_index", ""),
        )
        return


def select_eval_indices(length: int, count: int, seed: int) -> list[int]:
    if count <= 0 or count >= length:
        return list(range(length))
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def main() -> int:
    args = parse_args()
    device, local_rank, rank, world_size = setup_distributed(args)
    seed_everything(args.seed, rank)
    if args.lazy_generate_i0:
        if not args.i0_cache_dir:
            raise ValueError("--lazy-generate-i0 requires --i0-cache-dir")
        if args.augment_flips:
            raise ValueError(
                "--lazy-generate-i0 cannot be combined with --augment-flips. "
                "First train/fill the cache without flips, then resume with augmentation."
            )

    output_dir = Path(args.output_dir)
    if rank == 0:
        (output_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        (output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (output_dir / "eval").mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2),
            encoding="utf8",
        )

    dataset = I0ReferenceTextureDataset(
        args.metadata,
        image_size=args.image_size,
        i0_field=args.i0_field,
        i0_cache_dir=args.i0_cache_dir,
        allow_missing_i0=args.lazy_generate_i0,
        metadata_root=args.metadata_root,
        max_samples=args.max_samples,
        region_label_mode=args.region_label_mode,
        augment_flips=args.augment_flips,
        split="train",
    )
    sampler = (
        DistributedSampler(dataset, shuffle=True, drop_last=True)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    eval_loader = None
    if rank == 0 and args.val_metadata:
        val_dataset = I0ReferenceTextureDataset(
            args.val_metadata,
            image_size=args.image_size,
            i0_field=args.i0_field,
            i0_cache_dir=args.val_i0_cache_dir or args.i0_cache_dir,
            allow_missing_i0=args.lazy_generate_i0,
            metadata_root=args.metadata_root,
            max_samples=None,
            region_label_mode=args.region_label_mode,
            augment_flips=False,
            split="val",
        )
        eval_indices = select_eval_indices(
            len(val_dataset),
            int(args.eval_num_samples),
            int(args.eval_seed),
        )
        eval_subset = Subset(val_dataset, eval_indices)
        eval_loader = DataLoader(
            eval_subset,
            batch_size=max(1, int(args.eval_batch_size)),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        print(f"fixed eval metadata indices: {eval_indices}")

    in_ch = 3 + 16 + 6
    model = Pix2PixCrossAttnUNet(
        in_ch=in_ch,
        out_ch=3,
        base=args.base_channels,
        num_heads=args.num_heads,
        use_region_mask=not args.no_region_mask,
        residual_output=not args.no_residual_output,
        cross_attn_scales=args.cross_attn_scales,
        upsample_mode=args.upsample_mode,
    ).to(device)
    if rank == 0:
        print(f"model trainable params: {model_parameter_count(model):,}")
        print(f"decoder upsample mode: {args.upsample_mode} + conv")

    criterion = Pix2PixTransferLoss(
        lambda_l1=args.lambda_l1,
        lambda_perc=args.lambda_perc,
        lambda_gram=args.lambda_gram,
        lambda_contextual=args.lambda_contextual,
        vgg_weights=args.vgg_weights,
        content_layers=args.content_layers,
        gram_layers=args.gram_layers,
        contextual_layers=args.contextual_layers,
        texture_min_pixels=args.texture_min_pixels,
        contextual_max_samples=args.contextual_max_samples,
        contextual_temperature=args.contextual_temperature,
        normalize_losses=not args.no_loss_normalization,
        normalization_decay=args.loss_normalization_decay,
        normalization_steps=args.loss_normalization_steps,
        l1_blur_sigma=args.l1_blur_sigma,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.mixed_precision == "fp16" and device.type == "cuda"
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "loss_normalizer" in ckpt:
            criterion.normalizer.load_state_dict(ckpt["loss_normalizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        if rank == 0:
            print(f"resumed from {args.resume} at epoch={start_epoch} step={global_step}")

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    i0_generator = (
        LazyI0Generator(args, device, rank=rank)
        if args.lazy_generate_i0
        else None
    )

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            if i0_generator is not None:
                batch = i0_generator.fill_batch(batch)
            target_cond = batch["target_cond"].to(device, non_blocking=True)
            reference_cond = batch["reference_cond"].to(device, non_blocking=True)
            target = batch["target_image"].to(device, non_blocking=True)
            target_region = batch["target_region"].to(device, non_blocking=True)
            reference_region = batch["reference_region"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.mixed_precision):
                pred = model(
                    target_cond,
                    reference_cond,
                    target_region=target_region,
                    reference_region=reference_region,
                )
                loss, logs = criterion(
                    pred,
                    target,
                    reference=reference_cond[:, :3],
                    target_region=target_region,
                    reference_region=reference_region,
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()


            if rank == 0 and global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"epoch={epoch:04d} step={global_step:08d} "
                    f"loss={logs['total']:.5f} l1={logs['l1']:.5f} "
                    f"perc={logs['perc']:.5f} elapsed={elapsed/60:.1f}m"
                )
                # gamma 打印:放在 rank==0 和 log 同一个块里
                net = unwrap_model(model)
                gammas = [
                    f"{name}={p.item():.5f}"
                    for name, p in net.named_parameters()
                    if name.endswith("gamma")
                ]
                if gammas:
                    print(f"  [gamma] {' | '.join(gammas)}")
                print(
                    "  [texture-loss] "
                    f"gram={logs['gram']:.5f} contextual={logs['contextual']:.5f} "
                    f"norm_l1={logs['norm_l1']:.3f} norm_content={logs['norm_perc']:.3f} "
                    f"norm_gram={logs['norm_gram']:.3f} "
                    f"norm_contextual={logs['norm_contextual']:.3f}"
                )

            if rank == 0 and global_step % args.sample_every == 0:
                cpu_batch = {
                    "i0": batch["i0"],
                    "reference_image": batch["reference_image"],
                    "target_image": batch["target_image"],
                }
                save_training_sample(
                    output_dir=output_dir,
                    step=global_step,
                    batch=cpu_batch,
                    pred=pred,
                )
            global_step += 1

        if world_size > 1:
            dist.barrier()

        if (
            rank == 0
            and eval_loader is not None
            and args.eval_every_epochs > 0
            and (epoch + 1) % args.eval_every_epochs == 0
        ):
            save_eval_panel(
                model=unwrap_model(model),
                loader=eval_loader,
                device=device,
                output_dir=output_dir,
                epoch=epoch,
                mixed_precision=args.mixed_precision,
                i0_generator=i0_generator,
            )
            print(f"saved eval panel epoch {epoch + 1}")

        if rank == 0 and (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss_normalizer": criterion.normalizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": vars(args),
                },
                output_dir / "ckpt" / f"epoch{epoch + 1:04d}.pt",
            )
            print(f"saved checkpoint epoch {epoch + 1}")

        if world_size > 1:
            dist.barrier()

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
