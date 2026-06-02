from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import time
import warnings

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from .config import BaselineConfig
from .data import TissueSegmentationDataset, build_manifest, dataset_balanced_weights, load_manifest, load_mask, remap_mask_to_coarse, coarse_remap_table
from .losses import segmentation_loss
from .metrics import segmentation_metrics
from .model import BaselineSegmenter


def _ddp_env() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _gather_object_to_main(value: object, distributed: bool, dst: int = 0) -> list[object]:
    if not distributed:
        return [value]
    if hasattr(dist, "gather_object"):
        gathered: list[object | None] | None = [None for _ in range(dist.get_world_size())] if dist.get_rank() == dst else None
        dist.gather_object(value, object_gather_list=gathered, dst=dst)
        return list(gathered or [])
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return list(gathered) if dist.get_rank() == dst else []


def _save_model_state(model: torch.nn.Module, path: Path) -> None:
    torch.save(_unwrap_model(model).state_dict(), path)


def _resolve_resume_checkpoint(resume: str | None, output_dir: Path) -> Path | None:
    if not resume:
        return None
    if resume == "latest":
        latest = output_dir / "checkpoint_last.pt"
        if latest.exists():
            return latest
        warnings.warn(
            f"No segmentator checkpoint_last.pt found under {output_dir}; training from scratch.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    checkpoint_path = Path(resume).expanduser()
    candidates = [checkpoint_path] if checkpoint_path.is_absolute() else [checkpoint_path, output_dir / checkpoint_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    warnings.warn(
        f"Segmentator checkpoint not found: {resume}; training from scratch.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def _load_training_state(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported segmentator checkpoint format: {checkpoint_path}")

    if "model" not in checkpoint:
        _unwrap_model(model).load_state_dict(checkpoint, strict=True)
        return {
            "start_epoch": 0,
            "history": [],
            "best_miou": float("-inf"),
            "best_core5_miou": float("-inf"),
            "metrics": {},
            "weights_only": True,
        }

    _unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return {
        "start_epoch": int(checkpoint.get("completed_epochs", checkpoint.get("epoch", 0))),
        "history": list(checkpoint.get("history") or []),
        "best_miou": float(checkpoint.get("best_miou", float("-inf"))),
        "best_core5_miou": float(checkpoint.get("best_core5_miou", float("-inf"))),
        "metrics": dict(checkpoint.get("metrics") or {}),
        "weights_only": False,
    }


def _save_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    output_dir: Path,
    completed_epochs: int,
    history: list[dict[str, float]],
    best_miou: float,
    best_core5_miou: float,
    metrics: dict[str, object],
) -> None:
    torch.save(
        {
            "format": "segmentator_training_checkpoint_v1",
            "completed_epochs": completed_epochs,
            "epoch": completed_epochs,
            "model": _unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_miou": best_miou,
            "best_core5_miou": best_core5_miou,
            "metrics": metrics,
        },
        output_dir / "checkpoint_last.pt",
    )


def _run_mask2former_sanity_check(model: BaselineSegmenter, image_size: int, num_classes: int, device: torch.device) -> dict[str, object]:
    was_training = model.training
    model.train()
    dummy_img = torch.zeros(1, 3, image_size, image_size, device=device)
    dummy_mask = torch.zeros(1, image_size, image_size, dtype=torch.long, device=device)
    try:
        align = model._input_alignment()
        pad_h = (align - dummy_img.shape[-2] % align) % align
        pad_w = (align - dummy_img.shape[-1] % align) % align
        feature_img = F.pad(dummy_img, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else dummy_img
        model.eval()
        with torch.no_grad():
            encoder_feats = model.encoder(feature_img)
            if model.feature_pyramid is None:
                raise RuntimeError("Mask2Former sanity check expected a feature pyramid.")
            pyramid_feats = model.feature_pyramid(encoder_feats)
        encoder_shapes = [list(feat.shape) for feat in encoder_feats]
        pyramid_shapes = [list(feat.shape) for feat in pyramid_feats]
        if len(pyramid_shapes) != 4:
            raise RuntimeError(f"expected 4 Mask2Former feature levels, got {len(pyramid_shapes)}: {pyramid_shapes}")
        if any(shape[1] != 256 for shape in pyramid_shapes):
            raise RuntimeError(f"expected all Mask2Former feature levels to have 256 channels, got {pyramid_shapes}")
        base_h, base_w = encoder_shapes[1][-2:]
        expected_hw = [(base_h * 2, base_w * 2), (base_h, base_w), (base_h // 2, base_w // 2), (base_h // 4, base_w // 4)]
        actual_hw = [(shape[-2], shape[-1]) for shape in pyramid_shapes]
        if actual_hw != expected_hw:
            raise RuntimeError(f"unexpected Mask2Former feature level shapes: expected {expected_hw}, got {actual_hw}")

        model.train()
        losses = model.loss(dummy_img, dummy_mask)
        total = losses.get("total")
        if not torch.is_tensor(total) or not torch.isfinite(total.detach()).all():
            raise RuntimeError(f"non-finite sanity loss: {total}")
        return {
            "input_shape": list(dummy_img.shape),
            "aligned_input_shape": list(feature_img.shape),
            "encoder_shapes": encoder_shapes,
            "pyramid_shapes": pyramid_shapes,
            "pyramid_strides": list(model.feature_pyramid.strides),
        }
    except Exception as exc:
        raise RuntimeError(
            "Mask2Former sanity check failed. This usually means the installed mmseg/mmcv/mmdet "
            "versions do not support the configured 4-input/3-transformer-level pixel decoder, "
            "or MSDeformAttn is unavailable on the selected device."
        ) from exc
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_class_weights(dataset: TissueSegmentationDataset, num_classes: int, mode: str, remap_invalid_to: int) -> tuple[torch.Tensor | None, dict[str, object]]:
    if mode == "none":
        return None, {
            "mode": mode,
            "weights": None,
            "remap_invalid_to": remap_invalid_to,
            "ignore_index": dataset.ignore_index,
            "note": "Skipped pixel histogram scan because class weighting is disabled.",
        }

    counts = torch.zeros(num_classes, dtype=torch.float64)
    ignored_pixels = 0
    invalid_values: dict[int, int] = {}
    table = coarse_remap_table(dataset.mask_remap, num_classes=num_classes, ignore_index=dataset.ignore_index)
    skipped_samples: list[str] = []
    for record in dataset.records:
        try:
            mask = load_mask(record.mask_path).numpy()
        except OSError as exc:
            skipped_samples.append(record.sample_id)
            warnings.warn(
                f"Skipping unreadable segmentator mask {record.sample_id}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        valid_raw = (mask >= 0) & (mask < table.numel())
        invalid_raw = ~valid_raw
        for value, count in zip(*np.unique(mask[invalid_raw], return_counts=True)):
            invalid_values[int(value)] = invalid_values.get(int(value), 0) + int(count)
        remapped = remap_mask_to_coarse(torch.from_numpy(mask), table, ignore_index=dataset.ignore_index).numpy()
        valid = (remapped >= 0) & (remapped < num_classes)
        ignored_pixels += int((~valid).sum())
        bincount = np.bincount(remapped[valid].reshape(-1), minlength=num_classes)
        counts += torch.from_numpy(bincount).double()

    frequencies = counts / counts.sum().clamp_min(1.0)
    metadata: dict[str, object] = {
        "mode": mode,
        "pixel_counts": [int(v) for v in counts.tolist()],
        "frequencies": [float(v) for v in frequencies.tolist()],
        "remap_invalid_to": remap_invalid_to,
        "ignore_index": dataset.ignore_index,
        "ignored_pixels": ignored_pixels,
        "invalid_values": invalid_values,
        "skipped_unreadable_samples": skipped_samples,
    }
    if mode == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(frequencies.clamp_min(1e-8))
        weights = weights / weights.mean().clamp_min(1e-8)
        metadata["weights"] = [float(v) for v in weights.tolist()]
        return weights.float(), metadata
    raise ValueError(f"unsupported class weighting mode: {mode}")


def summarize_mask_label_space(dataset: TissueSegmentationDataset, num_classes: int) -> dict[str, object]:
    table = coarse_remap_table(dataset.mask_remap, num_classes=num_classes, ignore_index=dataset.ignore_index)
    summary: dict[str, object] = {}
    for record in dataset.records:
        dataset_summary = summary.setdefault(
            record.dataset_id,
            {
                "samples": 0,
                "total_pixels": 0,
                "raw_values": {},
                "remapped_values": {},
                "ignored_pixels": 0,
                "skipped_unreadable_samples": [],
            },
        )
        dataset_summary["samples"] += 1
        try:
            mask = load_mask(record.mask_path)
        except OSError as exc:
            dataset_summary["skipped_unreadable_samples"].append(record.sample_id)
            warnings.warn(
                f"Skipping unreadable segmentator mask {record.sample_id}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        dataset_summary["total_pixels"] += int(mask.numel())
        raw_values, raw_counts = torch.unique(mask, return_counts=True)
        raw_hist = dataset_summary["raw_values"]
        for value, count in zip(raw_values.tolist(), raw_counts.tolist()):
            key = str(int(value))
            raw_hist[key] = int(raw_hist.get(key, 0)) + int(count)

        remapped = remap_mask_to_coarse(mask, table, ignore_index=dataset.ignore_index)
        valid = (remapped >= 0) & (remapped < num_classes)
        dataset_summary["ignored_pixels"] += int((~valid).sum().item())
        remapped_values, remapped_counts = torch.unique(remapped[valid], return_counts=True)
        remapped_hist = dataset_summary["remapped_values"]
        for value, count in zip(remapped_values.tolist(), remapped_counts.tolist()):
            key = str(int(value))
            remapped_hist[key] = int(remapped_hist.get(key, 0)) + int(count)

    for dataset_summary in summary.values():
        if isinstance(dataset_summary, dict):
            dataset_summary["raw_values"] = dict(sorted(dataset_summary["raw_values"].items(), key=lambda item: int(item[0])))
            dataset_summary["remapped_values"] = dict(sorted(dataset_summary["remapped_values"].items(), key=lambda item: int(item[0])))
            total_pixels = int(dataset_summary["total_pixels"])
            dataset_summary["ignored_fraction"] = float(dataset_summary["ignored_pixels"] / total_pixels) if total_pixels else 0.0
    return dict(sorted(summary.items()))


def compact_label_space_summary(summary: dict[str, object]) -> dict[str, object]:
    compact: dict[str, object] = {}
    for dataset_id, raw_dataset_summary in summary.items():
        if not isinstance(raw_dataset_summary, dict):
            continue
        raw_values = raw_dataset_summary.get("raw_values", {})
        remapped_values = raw_dataset_summary.get("remapped_values", {})
        if not isinstance(raw_values, dict) or not isinstance(remapped_values, dict):
            continue
        remapped_total = sum(int(value) for value in remapped_values.values())
        compact[dataset_id] = {
            "samples": raw_dataset_summary.get("samples", 0),
            "total_pixels": raw_dataset_summary.get("total_pixels", 0),
            "ignored_pixels": raw_dataset_summary.get("ignored_pixels", 0),
            "ignored_fraction": raw_dataset_summary.get("ignored_fraction", 0.0),
            "raw_values": [int(value) for value in raw_values],
            "remapped_values": remapped_values,
            "remapped_fractions": {
                key: (float(int(value) / remapped_total) if remapped_total else 0.0)
                for key, value in remapped_values.items()
            },
            "skipped_unreadable_samples": raw_dataset_summary.get("skipped_unreadable_samples", []),
        }
    return compact


def _export_val_outputs(
    output_dir: Path,
    sample_ids: list[str],
    preds: list[torch.Tensor],
    probs: list[torch.Tensor],
    entropy: list[torch.Tensor],
    logits: list[torch.Tensor],
    export_tensors: bool = False,
) -> None:
    export_dir = output_dir / "val_outputs"
    export_dir.mkdir(parents=True, exist_ok=True)
    pred_t = torch.cat(preds, dim=0)
    prob_t = torch.cat(probs, dim=0) if export_tensors else None
    entropy_t = torch.cat(entropy, dim=0) if export_tensors else None
    logits_t = torch.cat(logits, dim=0) if export_tensors else None
    for idx, sample_id in enumerate(sample_ids):
        sample_dir = export_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred_t[idx].numpy().astype(np.uint8), mode="L").save(sample_dir / "pred_mask.png")
        if export_tensors:
            torch.save(prob_t[idx], sample_dir / "softmax_probability.pt")
            torch.save(entropy_t[idx], sample_dir / "entropy_uncertainty.pt")
            torch.save(logits_t[idx], sample_dir / "per_class_logits.pt")


def run_stage4_baseline(dataset_root: str | Path, config: BaselineConfig, uni2h_repo: str | Path = "UNI-2h") -> dict[str, float]:
    dataset_root = Path(dataset_root)
    distributed, rank, local_rank, world_size = _ddp_env()
    main_process = rank == 0
    print(
        f"[rank {rank}] segmentator startup distributed={distributed} local_rank={local_rank} world_size={world_size}",
        flush=True,
    )
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed segmentator training requires CUDA devices.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            print(f"[rank {rank}] initializing process group", flush=True)
            dist.init_process_group(backend="nccl")
            print(f"[rank {rank}] process group ready", flush=True)
    set_seed(config.seed)
    if config.disable_cudnn:
        torch.backends.cudnn.enabled = False
    if main_process:
        print(f"[rank {rank}] loading manifest from {config.manifest_path or dataset_root}", flush=True)
    manifest = (
        load_manifest(config.manifest_path, root=dataset_root)
        if config.manifest_path is not None
        else build_manifest(dataset_root, config.train_split, config.val_split, seed=config.seed)
    )

    train_ds = TissueSegmentationDataset(
        list(manifest.train),
        config.image_size,
        augment=True,
        num_classes=config.num_classes,
        remap_invalid_to=config.remap_invalid_to,
        ignore_index=config.ignore_index,
        mask_remap=config.mask_remap,
    )
    val_ds = TissueSegmentationDataset(
        list(manifest.val),
        config.image_size,
        augment=False,
        num_classes=config.num_classes,
        remap_invalid_to=config.remap_invalid_to,
        ignore_index=config.ignore_index,
        mask_remap=config.mask_remap,
    )
    if main_process:
        print(
            f"[rank {rank}] datasets ready train={len(train_ds)} val={len(val_ds)} class_weighting={config.class_weighting}",
            flush=True,
        )
    class_weights, class_weight_metadata = compute_class_weights(
        train_ds,
        config.num_classes,
        config.class_weighting,
        config.remap_invalid_to,
    )
    label_space_summary = summarize_mask_label_space(train_ds, config.num_classes) if main_process else {}
    if main_process:
        print(
            json.dumps(
                {"label_space_summary_compact": compact_label_space_summary(label_space_summary)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if config.decoder == "mask2former":
        class_weight_metadata["effective_invalid_target"] = config.mask2former_ignore_index
        class_weight_metadata["note"] = "Mask2Former masks invalid labels with ignore_index during target assignment; no-object query weight is separate."
    train_sampler = None
    train_shuffle = True
    train_drop_last = False
    if config.balanced_datasets:
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(config.seed + rank)
        train_sampler = WeightedRandomSampler(
            dataset_balanced_weights(train_ds.records),
            num_samples=int(math.ceil((config.samples_per_epoch or len(train_ds)) / world_size)) if distributed else (config.samples_per_epoch or len(train_ds)),
            replacement=True,
            generator=sampler_generator,
        )
        train_shuffle = False
    elif distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
        train_shuffle = False
    val_sampler = (
        DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=config.seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=config.num_workers, drop_last=train_drop_last)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, sampler=val_sampler, num_workers=config.num_workers)

    device = torch.device(f"cuda:{local_rank}" if distributed else ("cuda" if torch.cuda.is_available() else "cpu"))
    class_weights_device = class_weights.to(device) if class_weights is not None else None
    print(f"[rank {rank}] building model on {device}", flush=True)
    model = BaselineSegmenter(
        num_classes=config.num_classes,
        freeze_encoder=config.freeze_encoder,
        local_repo=uni2h_repo,
        decoder=config.decoder,
        mask2former_queries=config.mask2former_queries,
        mask2former_ignore_index=config.mask2former_ignore_index,
    ).to(device)
    sanity_check_passed = False
    mask2former_feature_shapes: dict[str, object] = {}
    if config.decoder == "mask2former":
        print(f"[rank {rank}] running Mask2Former sanity check", flush=True)
        mask2former_feature_shapes = _run_mask2former_sanity_check(model, config.image_size, config.num_classes, device)
        sanity_check_passed = True
        print(
            f"[rank {rank}] Mask2Former feature shapes "
            f"encoder={mask2former_feature_shapes['encoder_shapes']} "
            f"pyramid={mask2former_feature_shapes['pyramid_shapes']}",
            flush=True,
        )
        print(f"[rank {rank}] Mask2Former sanity check passed", flush=True)
    if distributed:
        print(f"[rank {rank}] wrapping model with DDP", flush=True)
        if config.decoder == "mask2former":
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        else:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[rank {rank}] DDP ready", flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay)
    use_amp = config.amp and device.type == "cuda" and config.decoder != "mask2former"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = config.resolve_output_dir()
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    history: list[dict[str, float]] = []
    best_miou = float("-inf")
    best_core5_miou = float("-inf")
    metrics: dict[str, object] = {}
    start_epoch = 0
    resume_path = _resolve_resume_checkpoint(config.resume_from_checkpoint, output_dir)
    if resume_path is not None:
        resume_state = _load_training_state(resume_path, model, optimizer, scaler, device)
        start_epoch = int(resume_state["start_epoch"])
        history = resume_state["history"]
        best_miou = float(resume_state["best_miou"])
        best_core5_miou = float(resume_state["best_core5_miou"])
        metrics = resume_state["metrics"]
        if main_process:
            print(
                json.dumps(
                    {
                        "resume_checkpoint": str(resume_path),
                        "start_epoch": start_epoch,
                        "weights_only": bool(resume_state["weights_only"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if main_process:
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    "image_size": config.image_size,
                    "remap_invalid_to": config.remap_invalid_to,
                    "ignore_index": config.ignore_index,
                    "mask_remap": config.mask_remap,
                    "balanced_datasets": config.balanced_datasets,
                    "samples_per_epoch": config.samples_per_epoch,
                    "batch_size_per_gpu": config.batch_size,
                    "batch_size": config.batch_size,
                    "grad_accum_steps": config.grad_accum_steps,
                    "world_size": world_size,
                    "distributed": distributed,
                    "effective_batch_size": config.batch_size * config.grad_accum_steps * world_size,
                    "epochs": config.epochs,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "amp": config.amp,
                    "amp_enabled_runtime": use_amp,
                    "amp_disabled_reason": "mask2former_msdeformattn_stability" if config.amp and config.decoder == "mask2former" else None,
                    "disable_cudnn": config.disable_cudnn,
                    "freeze_encoder": config.freeze_encoder,
                    "decoder": config.decoder,
                    "mask2former_queries": config.mask2former_queries,
                    "mask2former_ignore_index": config.mask2former_ignore_index,
                    "mask2former_sanity_check_passed": sanity_check_passed,
                    "mask2former_feature_shapes": mask2former_feature_shapes,
                    "effective_invalid_target": config.mask2former_ignore_index if config.decoder == "mask2former" else config.ignore_index,
                    "class_weighting": config.class_weighting,
                    "label_space_summary": label_space_summary,
                    "manifest_path": str(config.manifest_path) if config.manifest_path is not None else None,
                    "resume_from_checkpoint": config.resume_from_checkpoint,
                    "resume_checkpoint": str(resume_path) if resume_path is not None else None,
                    "start_epoch": start_epoch,
                    "export_val_predictions": config.export_val_predictions,
                    "export_val_tensors": config.export_val_tensors,
                    "uni2h_repo": str(uni2h_repo),
                    "device": str(device),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "class_weights.json").write_text(json.dumps(class_weight_metadata, indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{config.epochs} train",
            dynamic_ncols=True,
            disable=not main_process,
        )
        running_loss = 0.0
        for step, batch in enumerate(train_bar, start=1):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if config.decoder == "mask2former":
                    losses = model(image, mask)
                else:
                    outputs = model(image)
                    losses = segmentation_loss(
                        outputs["logits"],
                        mask,
                        config.num_classes,
                        class_weights=class_weights_device,
                        invalid_to=config.ignore_index,
                    )
                loss = losses["total"] / config.grad_accum_steps
            scaler.scale(loss).backward()
            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            display_loss = losses["total"].detach()
            if distributed:
                display_loss = display_loss.clone()
                dist.all_reduce(display_loss, op=dist.ReduceOp.SUM)
                display_loss = display_loss / world_size
            running_loss += float(display_loss.cpu().item())
            train_bar.set_postfix(loss=running_loss / step)

        model.eval()
        preds = []
        targets = []
        probs = []
        entropy = []
        logits = []
        sample_ids = []
        dataset_ids = []
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"epoch {epoch + 1}/{config.epochs} val",
                dynamic_ncols=True,
                disable=not main_process,
            )
            for batch in val_bar:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(image)
                preds.append(outputs["pred"].cpu())
                targets.append(mask.cpu())
                if config.export_val_tensors:
                    probs.append(outputs["probs"].cpu())
                    entropy.append(outputs["entropy"].cpu())
                    logits.append(outputs["logits"].cpu())
                sample_ids.extend(str(v) for v in batch["sample_id"])
                dataset_ids.extend(str(v) for v in batch["dataset_id"])

        local_payload = {
            "sample_ids": sample_ids,
            "dataset_ids": dataset_ids,
            "pred": torch.cat(preds, dim=0) if preds else torch.empty(0, config.image_size, config.image_size, dtype=torch.long),
            "target": torch.cat(targets, dim=0) if targets else torch.empty(0, config.image_size, config.image_size, dtype=torch.long),
        }
        if config.export_val_tensors:
            local_payload["probs"] = torch.cat(probs, dim=0) if probs else torch.empty(0)
            local_payload["entropy"] = torch.cat(entropy, dim=0) if entropy else torch.empty(0)
            local_payload["logits"] = torch.cat(logits, dim=0) if logits else torch.empty(0)

        gathered_payloads = _gather_object_to_main(local_payload, distributed)
        if main_process:
            seen_sample_ids: set[str] = set()
            ordered_sample_ids: list[str] = []
            gathered_preds: list[torch.Tensor] = []
            gathered_targets: list[torch.Tensor] = []
            gathered_probs: list[torch.Tensor] = []
            gathered_entropy: list[torch.Tensor] = []
            gathered_logits: list[torch.Tensor] = []
            ordered_dataset_ids: list[str] = []
            for payload in gathered_payloads:
                payload_sample_ids = payload["sample_ids"]
                payload_dataset_ids = payload["dataset_ids"]
                payload_pred = payload["pred"]
                payload_target = payload["target"]
                payload_probs = payload.get("probs")
                payload_entropy = payload.get("entropy")
                payload_logits = payload.get("logits")
                for idx, sample_id in enumerate(payload_sample_ids):
                    if sample_id in seen_sample_ids:
                        continue
                    seen_sample_ids.add(sample_id)
                    ordered_sample_ids.append(sample_id)
                    ordered_dataset_ids.append(str(payload_dataset_ids[idx]))
                    gathered_preds.append(payload_pred[idx : idx + 1])
                    gathered_targets.append(payload_target[idx : idx + 1])
                    if config.export_val_tensors:
                        gathered_probs.append(payload_probs[idx : idx + 1])
                        gathered_entropy.append(payload_entropy[idx : idx + 1])
                        gathered_logits.append(payload_logits[idx : idx + 1])

            pred = torch.cat(gathered_preds, dim=0)
            target = torch.cat(gathered_targets, dim=0)
            metrics = segmentation_metrics(
                pred,
                target,
                config.num_classes,
                class_names=manifest.classes,
                boundary_width=config.boundary_width,
                ignore_index=config.ignore_index,
            )
            per_dataset_metrics = {}
            for dataset_id in sorted(set(ordered_dataset_ids)):
                indices = [idx for idx, value in enumerate(ordered_dataset_ids) if value == dataset_id]
                if not indices:
                    continue
                index_tensor = torch.tensor(indices, dtype=torch.long)
                ds_metrics = segmentation_metrics(
                    pred.index_select(0, index_tensor),
                    target.index_select(0, index_tensor),
                    config.num_classes,
                    class_names=manifest.classes,
                    boundary_width=config.boundary_width,
                    ignore_index=config.ignore_index,
                )
                per_dataset_metrics[dataset_id] = {
                    "samples": len(indices),
                    "mIoU": ds_metrics["mIoU"],
                    "mDice": ds_metrics["mDice"],
                    "foreground_recall": ds_metrics["foreground_recall"],
                    "boundary_f1": ds_metrics["boundary_f1"],
                    "per_class": ds_metrics["per_class"],
                    "groups": ds_metrics["groups"],
                }
            metrics["per_dataset"] = per_dataset_metrics
            history.append({k: v for k, v in metrics.items() if isinstance(v, float)})
            if float(metrics["mIoU"]) > best_miou:
                best_miou = float(metrics["mIoU"])
                _save_model_state(model, output_dir / "best_mIoU.pt")
            core5 = metrics.get("groups", {}).get("core_5_classes", {}) if isinstance(metrics.get("groups"), dict) else {}
            if isinstance(core5, dict) and float(core5.get("mean_iou", float("-inf"))) > best_core5_miou:
                best_core5_miou = float(core5["mean_iou"])
                _save_model_state(model, output_dir / "best_core5.pt")
            (output_dir / "metrics.json").write_text(json.dumps({"history": history, "final": metrics}, indent=2), encoding="utf-8")
            elapsed = time.time() - epoch_start
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "elapsed_sec": round(elapsed, 2),
                        "mIoU": metrics["mIoU"],
                        "mDice": metrics["mDice"],
                        "foreground_recall": metrics["foreground_recall"],
                        "boundary_f1": metrics["boundary_f1"],
                        "per_dataset_mIoU": {k: v["mIoU"] for k, v in per_dataset_metrics.items()},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if epoch == config.epochs - 1 and config.export_val_predictions:
                _export_val_outputs(
                    output_dir,
                    ordered_sample_ids,
                    [pred],
                    [torch.cat(gathered_probs, dim=0)] if config.export_val_tensors else [],
                    [torch.cat(gathered_entropy, dim=0)] if config.export_val_tensors else [],
                    [torch.cat(gathered_logits, dim=0)] if config.export_val_tensors else [],
                    export_tensors=config.export_val_tensors,
                )
            _save_training_state(
                model,
                optimizer,
                scaler,
                output_dir,
                completed_epochs=epoch + 1,
                history=history,
                best_miou=best_miou,
                best_core5_miou=best_core5_miou,
                metrics=metrics,
            )
        if distributed:
            metric_payload = [metrics if main_process else None]
            dist.broadcast_object_list(metric_payload, src=0)
            metrics = metric_payload[0]

    if main_process:
        _save_model_state(model, output_dir / "stage4_baseline.pt")
        _save_model_state(model, output_dir / f"stage4_{config.decoder}.pt")
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "seed": config.seed,
                    "train_split": config.train_split,
                    "val_split": config.val_split,
                    "manifest_path": str(config.manifest_path) if config.manifest_path is not None else None,
                    "train_samples": [r.sample_id for r in manifest.train],
                    "val_samples": [r.sample_id for r in manifest.val],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(json.dumps({"history": history, "final": metrics}, indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return {k: v for k, v in metrics.items() if isinstance(v, float)}
