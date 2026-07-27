from __future__ import annotations

from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
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

from dataset_config.unified_labels import FINE_TO_PARENT, NUM_COARSE, NUM_FINE

from .config import BaselineConfig, SEGMENTATOR_FINE_CLASSES
from .data import (
    TissueSegmentationDataset,
    build_fine_target,
    build_manifest,
    coarse_remap_table,
    dataset_balanced_weights,
    fine_supervision_for_dataset,
    load_manifest,
    load_mask,
    remap_mask_to_coarse,
)
from .losses import segmentation_loss
from .metrics import fine_segmentation_metrics, group_macro_iou, segmentation_metrics
from .model import BaselineSegmenter
from .stain_augmentation import StainAugmentationConfig


def _ddp_env() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


_TRAINABLE_SCOPE_MODULES = {
    "fine": ("fine_head",),
    "boundary": ("refinement_head",),
    "teacher": ("fine_head", "cell_teacher_adapter", "cell_density_head"),
}


def _freeze_for_trainable_scope(model: torch.nn.Module, scope: str) -> list[torch.nn.Parameter]:
    unwrapped = _unwrap_model(model)
    if scope not in _TRAINABLE_SCOPE_MODULES:
        raise ValueError(f"unsupported frozen trainable scope: {scope}")
    for parameter in unwrapped.parameters():
        parameter.requires_grad = False
    selected_parameters: list[torch.nn.Parameter] = []
    for module_name in _TRAINABLE_SCOPE_MODULES[scope]:
        module = getattr(unwrapped, module_name, None)
        if module is None:
            raise ValueError(f"trainable scope {scope!r} requires model.{module_name}")
        selected_parameters.extend(module.parameters())
    selected_parameters = list(dict.fromkeys(selected_parameters))
    for parameter in selected_parameters:
        parameter.requires_grad = True
    unwrapped.encoder.freeze = True
    unwrapped.encoder.trainable_block_count = 0
    return selected_parameters


def _freeze_shared_for_fine(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return _freeze_for_trainable_scope(model, "fine")


def _set_frozen_scope_training_mode(model: torch.nn.Module, scope: str) -> None:
    model.eval()
    unwrapped = _unwrap_model(model)
    for module_name in _TRAINABLE_SCOPE_MODULES[scope]:
        getattr(unwrapped, module_name).train()


def _fine_dataset_macro(per_dataset_metrics: dict[str, dict[str, object]]) -> float:
    values = []
    for dataset_metrics in per_dataset_metrics.values():
        fine_metrics = dataset_metrics.get("fine")
        if not isinstance(fine_metrics, dict) or not fine_metrics.get("available"):
            continue
        value = float(fine_metrics.get("mIoU", float("nan")))
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _majority_child_miou(target: torch.Tensor, ignore_index: int = 255) -> float:
    valid = (target >= 0) & (target < NUM_FINE) & (target != ignore_index)
    if not valid.any():
        return float("nan")
    counts = torch.bincount(target[valid].long(), minlength=NUM_FINE)
    majority_id = int(counts.argmax().item())
    pred = torch.full_like(target, int(ignore_index))
    pred[valid] = majority_id
    return float(
        fine_segmentation_metrics(
            pred,
            target,
            NUM_FINE,
            class_names=SEGMENTATOR_FINE_CLASSES,
            ignore_index=ignore_index,
        )["mIoU"]
    )


def _checkpoint_selection_score(metrics: dict[str, object], config: BaselineConfig) -> tuple[float, bool]:
    if config.checkpoint_mode == "composite":
        return float(metrics["checkpoint_composite"]), True
    if config.checkpoint_mode not in {"fine_dataset_macro", "boundary_f1_4"}:
        raise ValueError(f"unsupported checkpoint mode: {config.checkpoint_mode}")

    eligible = True
    if config.checkpoint_coarse_miou_floor is not None:
        eligible = eligible and float(metrics["mIoU"]) >= config.checkpoint_coarse_miou_floor
    if config.checkpoint_coarse_boundary_f1_4_floor is not None:
        eligible = eligible and float(metrics["boundary_f1_4"]) >= config.checkpoint_coarse_boundary_f1_4_floor
    fine_dataset_macro = float(metrics.get("fine_dataset_macro_mIoU", float("nan")))
    if config.checkpoint_fine_dataset_macro_floor is not None:
        eligible = eligible and fine_dataset_macro >= config.checkpoint_fine_dataset_macro_floor
    score = (
        fine_dataset_macro
        if config.checkpoint_mode == "fine_dataset_macro"
        else float(metrics.get("boundary_f1_4", float("nan")))
    )
    return (score if eligible and math.isfinite(score) else float("-inf")), eligible


def _loss_diagnostics(losses: dict[str, torch.Tensor]) -> dict[str, float | list[float]]:
    diagnostics: dict[str, float | list[float]] = {}
    for name, value in losses.items():
        if not torch.is_tensor(value):
            continue
        detached = value.detach().float().cpu()
        diagnostics[name] = float(detached) if detached.numel() == 1 else detached.flatten().tolist()
    return diagnostics


def _wait_for_free_gpu_memory_before_unfreeze(
    device: torch.device,
    min_free_gb: float,
    poll_seconds: float,
    *,
    main_process: bool,
) -> None:
    if device.type != "cuda" or min_free_gb <= 0:
        return

    memory_device = device
    if memory_device.index is None:
        memory_device = torch.device("cuda", torch.cuda.current_device())
    required_bytes = int(min_free_gb * 1024**3)
    poll_seconds = max(float(poll_seconds), 1.0)
    while True:
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(memory_device)
        free_gb = free_bytes / 1024**3
        total_gb = total_bytes / 1024**3
        if free_bytes >= required_bytes:
            if main_process:
                print(
                    f"GPU memory gate passed before backbone unfreeze: "
                    f"free={free_gb:.1f} GiB required={min_free_gb:.1f} GiB "
                    f"total={total_gb:.1f} GiB",
                    flush=True,
                )
            return
        if main_process:
            print(
                f"waiting before backbone unfreeze: free GPU memory "
                f"{free_gb:.1f} GiB < required {min_free_gb:.1f} GiB; "
                f"checking again in {poll_seconds:g}s",
                flush=True,
            )
        time.sleep(poll_seconds)


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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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
            "best_boundary_f1_4": float("-inf"),
            "best_fine_miou": float("-inf"),
            "best_composite": float("-inf"),
            "epochs_without_improvement": 0,
            "metrics": {},
            "weights_only": True,
        }

    _unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return {
        "start_epoch": int(checkpoint.get("completed_epochs", checkpoint.get("epoch", 0))),
        "history": list(checkpoint.get("history") or []),
        "best_miou": float(checkpoint.get("best_miou", float("-inf"))),
        "best_core5_miou": float(checkpoint.get("best_core5_miou", float("-inf"))),
        "best_boundary_f1_4": float(checkpoint.get("best_boundary_f1_4", float("-inf"))),
        "best_fine_miou": float(checkpoint.get("best_fine_miou", float("-inf"))),
        "best_composite": float(checkpoint.get("best_composite", float("-inf"))),
        "epochs_without_improvement": int(checkpoint.get("epochs_without_improvement", 0)),
        "metrics": dict(checkpoint.get("metrics") or {}),
        "weights_only": False,
    }


def _load_initialization_weights(checkpoint_path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, object]:
    """Load shared coarse weights while allowing newly added hierarchical fine parameters."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported segmentator checkpoint format: {checkpoint_path}")
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = _unwrap_model(model).load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "fine_head.",
        "fine_supported_mask",
        "fine_class_weights",
        "refinement_head.",
        "cell_density_head.",
        "cell_teacher_adapter.",
        "cell_prior_encoder.",
    )
    disallowed_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)]
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"coarse initialization checkpoint is incompatible: missing={disallowed_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "missing_fine_parameters": list(incompatible.missing_keys),
        "unexpected_parameters": list(incompatible.unexpected_keys),
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
    best_boundary_f1_4: float,
    best_fine_miou: float,
    best_composite: float,
    epochs_without_improvement: int,
    metrics: dict[str, object],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    torch.save(
        {
            "format": "segmentator_training_checkpoint_v2",
            "completed_epochs": completed_epochs,
            "epoch": completed_epochs,
            "model": _unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_miou": best_miou,
            "best_core5_miou": best_core5_miou,
            "best_boundary_f1_4": best_boundary_f1_4,
            "best_fine_miou": best_fine_miou,
            "best_composite": best_composite,
            "epochs_without_improvement": epochs_without_improvement,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
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
        expected_hw = [
            (
                int(math.ceil(feature_img.shape[-2] / stride)),
                int(math.ceil(feature_img.shape[-1] / stride)),
            )
            for stride in model.feature_pyramid.strides
        ]
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
            "or MSDeformAttn is unavailable on the selected device. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_label_space_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    summary = payload.get("label_space_summary", payload)
    if not isinstance(summary, dict):
        raise ValueError(f"label-space summary must be a JSON object: {path}")
    return summary


def compute_class_weights(
    dataset: TissueSegmentationDataset,
    num_classes: int,
    mode: str,
    remap_invalid_to: int,
    label_space_summary: dict[str, object] | None = None,
) -> tuple[torch.Tensor | None, dict[str, object]]:
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
    if label_space_summary is not None:
        for raw_dataset_summary in label_space_summary.values():
            if not isinstance(raw_dataset_summary, dict):
                continue
            for value, count in dict(raw_dataset_summary.get("remapped_values", {})).items():
                class_id = int(value)
                if 0 <= class_id < num_classes:
                    counts[class_id] += int(count)
            ignored_pixels += int(raw_dataset_summary.get("ignored_pixels", 0))
            skipped_samples.extend(str(value) for value in raw_dataset_summary.get("skipped_unreadable_samples", []))
    for record in tqdm(dataset.records if label_space_summary is None else [], desc="label-space scan", dynamic_ncols=True):
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
        weights = weights.clamp(0.5, 4.0)
        metadata["weights"] = [float(v) for v in weights.tolist()]
        return weights.float(), metadata
    raise ValueError(f"unsupported class weighting mode: {mode}")


def compute_fine_class_weights(
    label_space_summary: dict[str, object],
    mode: str,
    min_weight: float = 0.5,
    max_weight: float = 4.0,
) -> tuple[torch.Tensor | None, tuple[int, ...], dict[str, object]]:
    """Derive fine-head support and weights from the cached per-dataset raw histograms."""
    counts = torch.zeros(NUM_FINE, dtype=torch.float64)
    for dataset_id, raw_dataset_summary in label_space_summary.items():
        if not isinstance(raw_dataset_summary, dict):
            continue
        allowed = fine_supervision_for_dataset(str(dataset_id))
        branching = allowed.sum(dim=1) > 1
        for value, count in dict(raw_dataset_summary.get("raw_values", {})).items():
            fine_id = int(value)
            if 0 <= fine_id < NUM_FINE:
                parent_id = FINE_TO_PARENT[fine_id]
                if bool(branching[parent_id]) and bool(allowed[parent_id, fine_id]):
                    counts[fine_id] += int(count)

    observed = counts > 0
    supported_ids = tuple(sorted(set(range(NUM_COARSE)) | set(torch.where(observed)[0].tolist())))
    metadata: dict[str, object] = {
        "mode": mode,
        "pixel_counts": [int(value) for value in counts.tolist()],
        "supported_ids": list(supported_ids),
        "unsupported_branch_ids": [idx for idx in range(NUM_COARSE, NUM_FINE) if not bool(observed[idx])],
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
    }
    if mode == "none":
        metadata["weights"] = None
        return None, supported_ids, metadata
    if mode != "inverse_sqrt":
        raise ValueError(f"unsupported fine class weighting mode: {mode}")
    weights = torch.ones(NUM_FINE, dtype=torch.float64)
    if observed.any():
        frequencies = counts[observed] / counts[observed].sum().clamp_min(1.0)
        observed_weights = 1.0 / torch.sqrt(frequencies.clamp_min(1e-8))
        observed_weights = (observed_weights / observed_weights.mean().clamp_min(1e-8)).clamp(min_weight, max_weight)
        weights[observed] = observed_weights
    metadata["weights"] = [float(value) for value in weights.tolist()]
    return weights.float(), supported_ids, metadata


def rare_class_presence_weights(
    dataset: TissueSegmentationDataset,
    class_ids: tuple[int, ...],
    boost: float,
) -> torch.DoubleTensor:
    table = coarse_remap_table(dataset.mask_remap, num_classes=dataset.num_classes, ignore_index=dataset.ignore_index)
    weights = []
    for record in tqdm(dataset.records, desc="rare-class scan", dynamic_ncols=True):
        try:
            remapped = remap_mask_to_coarse(load_mask(record.mask_path), table, ignore_index=dataset.ignore_index)
            present = any(bool((remapped == class_id).any()) for class_id in class_ids)
        except OSError:
            present = False
        weights.append(boost if present else 1.0)
    return torch.DoubleTensor(weights)


def boundary_aware_sampling_weights(
    dataset: TissueSegmentationDataset,
    *,
    boost: float,
    min_boundary_pixels: int,
    width: int,
) -> tuple[torch.DoubleTensor, dict[str, object]]:
    if boost < 1.0:
        raise ValueError("boundary sampling boost must be at least 1")
    if min_boundary_pixels < 1 or width < 1:
        raise ValueError("boundary sampling pixel threshold and width must be positive")

    table = coarse_remap_table(dataset.mask_remap, num_classes=dataset.num_classes, ignore_index=dataset.ignore_index)
    weights = torch.ones(len(dataset.records), dtype=torch.float64)
    dataset_stats: dict[str, dict[str, float | int]] = {}
    skipped_unreadable: list[str] = []
    for index, record in enumerate(tqdm(dataset.records, desc="boundary-rich scan", dynamic_ncols=True)):
        try:
            mask = remap_mask_to_coarse(load_mask(record.mask_path), table, ignore_index=dataset.ignore_index)
        except OSError:
            skipped_unreadable.append(record.sample_id)
            continue
        valid = mask != dataset.ignore_index
        edge = torch.zeros_like(valid)
        horizontal = valid[:, :-1] & valid[:, 1:] & (mask[:, :-1] != mask[:, 1:])
        vertical = valid[:-1, :] & valid[1:, :] & (mask[:-1, :] != mask[1:, :])
        edge[:, :-1] |= horizontal
        edge[:, 1:] |= horizontal
        edge[:-1, :] |= vertical
        edge[1:, :] |= vertical
        raw_edge_pixels = int((edge & valid).sum().item())
        boundary_pixels = min(raw_edge_pixels * (2 * width + 1), int(valid.sum().item()))
        richness = min(boundary_pixels / min_boundary_pixels, 1.0)
        weights[index] = 1.0 + (boost - 1.0) * richness

        stats = dataset_stats.setdefault(
            record.dataset_id,
            {"samples": 0, "rich_samples": 0, "boundary_pixels": 0},
        )
        stats["samples"] = int(stats["samples"]) + 1
        stats["rich_samples"] = int(stats["rich_samples"]) + int(boundary_pixels >= min_boundary_pixels)
        stats["boundary_pixels"] = int(stats["boundary_pixels"]) + boundary_pixels

    metadata_datasets: dict[str, dict[str, float | int]] = {}
    for dataset_id, stats in sorted(dataset_stats.items()):
        samples = max(int(stats["samples"]), 1)
        metadata_datasets[dataset_id] = {
            **stats,
            "rich_fraction": int(stats["rich_samples"]) / samples,
            "mean_boundary_pixels": int(stats["boundary_pixels"]) / samples,
        }
    return weights, {
        "enabled": True,
        "boost": float(boost),
        "min_boundary_pixels": int(min_boundary_pixels),
        "width": int(width),
        "pixel_estimator": "raw_edge_pixels_times_band_diameter_clipped_to_valid_area",
        "total_records": len(dataset.records),
        "skipped_unreadable_samples": skipped_unreadable,
        "datasets": metadata_datasets,
        "mean_multiplier": float(weights.mean().item()),
        "max_multiplier": float(weights.max().item()),
    }


def fine_supervision_sampling_weights(
    dataset: TissueSegmentationDataset,
    *,
    temperature: float,
    rare_class_boost: float,
    min_valid_pixels: int,
    require_nuclei: bool = False,
) -> tuple[torch.DoubleTensor, dict[str, object]]:
    """Build a dataset-balanced sampler over patches with real fine supervision."""
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("fine sampling temperature must be in [0, 1]")
    if rare_class_boost < 1.0:
        raise ValueError("fine rare-class boost must be at least 1")
    if min_valid_pixels < 1:
        raise ValueError("fine minimum valid pixels must be at least 1")

    sample_counts: list[torch.Tensor | None] = [None] * len(dataset.records)
    dataset_indices: dict[str, list[int]] = {}
    dataset_pixel_counts: dict[str, torch.Tensor] = {}
    dataset_presence_counts: dict[str, torch.Tensor] = {}
    excluded_no_branch: dict[str, int] = {}
    excluded_no_valid: dict[str, int] = {}
    excluded_missing_nuclei: dict[str, int] = {}
    skipped_unreadable: list[str] = []

    for index, record in enumerate(tqdm(dataset.records, desc="fine-supervision scan", dynamic_ncols=True)):
        if require_nuclei and record.nuclei_path is None:
            excluded_missing_nuclei[record.dataset_id] = excluded_missing_nuclei.get(record.dataset_id, 0) + 1
            continue
        allowed = fine_supervision_for_dataset(record.dataset_id)
        if not bool((allowed.sum(dim=1) > 1).any()):
            excluded_no_branch[record.dataset_id] = excluded_no_branch.get(record.dataset_id, 0) + 1
            continue
        try:
            fine_target, _ = build_fine_target(
                load_mask(record.mask_path),
                record.dataset_id,
                ignore_index=dataset.ignore_index,
            )
        except OSError:
            skipped_unreadable.append(record.sample_id)
            continue
        valid = (fine_target >= 0) & (fine_target < NUM_FINE)
        if int(valid.sum().item()) < min_valid_pixels:
            excluded_no_valid[record.dataset_id] = excluded_no_valid.get(record.dataset_id, 0) + 1
            continue
        counts = torch.bincount(fine_target[valid], minlength=NUM_FINE).to(dtype=torch.float64)
        sample_counts[index] = counts
        dataset_indices.setdefault(record.dataset_id, []).append(index)
        dataset_pixel_counts.setdefault(record.dataset_id, torch.zeros(NUM_FINE, dtype=torch.float64)).add_(counts)
        dataset_presence_counts.setdefault(record.dataset_id, torch.zeros(NUM_FINE, dtype=torch.long)).add_(counts > 0)

    eligible_count = sum(len(indices) for indices in dataset_indices.values())
    if eligible_count == 0:
        raise ValueError("fine-supervision sampling found no eligible training patches")

    weights = torch.zeros(len(dataset.records), dtype=torch.float64)
    dataset_metadata: dict[str, object] = {}
    for dataset_id, indices in sorted(dataset_indices.items()):
        pixel_counts = dataset_pixel_counts[dataset_id]
        observed = pixel_counts > 0
        max_count = pixel_counts[observed].max() if observed.any() else torch.tensor(1.0, dtype=torch.float64)
        class_multipliers = torch.ones(NUM_FINE, dtype=torch.float64)
        class_multipliers[observed] = torch.sqrt(max_count / pixel_counts[observed].clamp_min(1.0)).clamp(
            1.0,
            rare_class_boost,
        )
        sample_scores = []
        for index in indices:
            counts = sample_counts[index]
            present = counts > 0
            score = float(class_multipliers[present].max().item()) if present.any() else 1.0
            sample_scores.append(score)
        score_total = max(sum(sample_scores), 1.0)
        dataset_mass = len(indices) ** temperature
        for index, score in zip(indices, sample_scores):
            weights[index] = dataset_mass * score / score_total
        dataset_metadata[dataset_id] = {
            "eligible_samples": len(indices),
            "fine_pixel_counts": [int(value) for value in pixel_counts.tolist()],
            "class_presence_samples": [
                int(value) for value in dataset_presence_counts[dataset_id].tolist()
            ],
            "observed_fine_ids": torch.where(observed)[0].tolist(),
            "class_presence_multipliers": [float(value) for value in class_multipliers.tolist()],
            "target_sampling_mass": float(dataset_mass),
        }

    normalized_dataset_mass = {}
    total_mass = float(weights.sum().item())
    for dataset_id, indices in sorted(dataset_indices.items()):
        normalized_dataset_mass[dataset_id] = float(weights[indices].sum().item() / max(total_mass, 1e-12))
    metadata: dict[str, object] = {
        "enabled": True,
        "temperature": float(temperature),
        "rare_class_boost": float(rare_class_boost),
        "min_valid_pixels": int(min_valid_pixels),
        "require_nuclei": bool(require_nuclei),
        "total_records": len(dataset.records),
        "eligible_records": eligible_count,
        "eligible_fraction": eligible_count / max(len(dataset.records), 1),
        "excluded_no_branch": excluded_no_branch,
        "excluded_no_valid_pixels": excluded_no_valid,
        "excluded_missing_nuclei": excluded_missing_nuclei,
        "skipped_unreadable_samples": skipped_unreadable,
        "datasets": dataset_metadata,
        "normalized_dataset_sampling_mass": normalized_dataset_mass,
    }
    return weights, metadata


def summarize_mask_label_space(dataset: TissueSegmentationDataset, num_classes: int) -> dict[str, object]:
    table = coarse_remap_table(dataset.mask_remap, num_classes=num_classes, ignore_index=dataset.ignore_index)
    summary: dict[str, object] = {}
    total_records = len(dataset.records)
    records_iter = tqdm(
        dataset.records,
        desc="label-space scan",
        total=total_records,
        dynamic_ncols=True,
        mininterval=5.0,
        file=sys.stderr,
    )
    for idx, record in enumerate(records_iter, start=1):
        if idx == 1 or idx % 1000 == 0 or idx == total_records:
            print(
                f"label-space scan progress {idx}/{total_records} "
                f"({idx / max(total_records, 1):.1%}) dataset={record.dataset_id}",
                flush=True,
            )
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
    trainable_scope = config.trainable_scope
    if config.freeze_shared_for_fine:
        if trainable_scope not in {"all", "fine"}:
            raise ValueError("--freeze-shared-for-fine conflicts with the requested trainable scope")
        trainable_scope = "fine"
    if trainable_scope not in {"all", *_TRAINABLE_SCOPE_MODULES}:
        raise ValueError(f"unsupported trainable scope: {trainable_scope}")
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
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(seconds=config.ddp_timeout_seconds),
            )
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
        stain_augmentation=StainAugmentationConfig(
            mode=config.stain_augmentation,
            probability=config.stain_augmentation_prob,
            randstainna_root=config.randstainna_root,
            randstainna_yaml=config.randstainna_yaml,
            randstainna_std_hyper=config.randstainna_std_hyper,
            randstainna_distribution=config.randstainna_distribution,
        ),
        cellvit_mode=config.cellvit_mode,
        cell_density_sigma=config.cell_density_sigma,
        augment_vflip=config.augment_vflip,
        augment_rot90=config.augment_rot90,
        augment_scale_crop=config.augment_scale_crop,
        hierarchical_fine=config.hierarchical_fine,
    )
    val_ds = TissueSegmentationDataset(
        list(manifest.val),
        config.image_size,
        augment=False,
        num_classes=config.num_classes,
        remap_invalid_to=config.remap_invalid_to,
        ignore_index=config.ignore_index,
        mask_remap=config.mask_remap,
        cellvit_mode=config.cellvit_mode,
        cell_density_sigma=config.cell_density_sigma,
        hierarchical_fine=config.hierarchical_fine,
    )
    if main_process:
        print(
            f"[rank {rank}] datasets ready train={len(train_ds)} val={len(val_ds)} class_weighting={config.class_weighting}",
            flush=True,
        )
        if config.stain_augmentation != "none":
            print(
                f"[rank {rank}] stain augmentation mode={config.stain_augmentation} "
                f"prob={config.stain_augmentation_prob} "
                f"randstainna_root={config.randstainna_root}",
                flush=True,
            )
    if config.label_space_summary_path is not None:
        label_space_summary = load_label_space_summary(config.label_space_summary_path)
        if main_process:
            print(f"loaded cached label-space summary from {config.label_space_summary_path}", flush=True)
    else:
        label_space_summary = summarize_mask_label_space(train_ds, config.num_classes) if main_process else {}
    if main_process or not distributed:
        class_weights, class_weight_metadata = compute_class_weights(
            train_ds,
            config.num_classes,
            config.class_weighting,
            config.remap_invalid_to,
            label_space_summary=label_space_summary,
        )
    else:
        class_weights, class_weight_metadata = None, {}
    if distributed:
        coarse_weight_payload = [(class_weights, class_weight_metadata) if main_process else None]
        dist.broadcast_object_list(coarse_weight_payload, src=0)
        class_weights, class_weight_metadata = coarse_weight_payload[0]
    fine_class_weights = None
    fine_supported_ids: tuple[int, ...] | None = None
    fine_class_weight_metadata: dict[str, object] = {"enabled": False}
    if config.hierarchical_fine and main_process:
        fine_weighting_mode = config.fine_class_weighting or config.class_weighting
        fine_class_weights, fine_supported_ids, fine_class_weight_metadata = compute_fine_class_weights(
            label_space_summary,
            fine_weighting_mode,
            min_weight=config.fine_class_weight_min,
            max_weight=config.fine_class_weight_max,
        )
        fine_class_weight_metadata["enabled"] = True
    if distributed and config.hierarchical_fine:
        fine_payload = [(fine_class_weights, fine_supported_ids, fine_class_weight_metadata) if main_process else None]
        dist.broadcast_object_list(fine_payload, src=0)
        fine_class_weights, fine_supported_ids, fine_class_weight_metadata = fine_payload[0]
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
    fine_sampling_weights = None
    fine_sampling_metadata: dict[str, object] = {"enabled": False}
    if config.fine_supervision_sampling and main_process:
        fine_sampling_weights, fine_sampling_metadata = fine_supervision_sampling_weights(
            train_ds,
            temperature=config.dataset_sampling_temperature,
            rare_class_boost=config.fine_sampling_rare_class_boost,
            min_valid_pixels=config.fine_sampling_min_valid_pixels,
            require_nuclei=config.fine_sampling_require_nuclei,
        )
    if distributed and config.fine_supervision_sampling:
        sampling_payload = [
            (fine_sampling_weights, fine_sampling_metadata) if main_process else None
        ]
        dist.broadcast_object_list(sampling_payload, src=0)
        fine_sampling_weights, fine_sampling_metadata = sampling_payload[0]
    if main_process and config.fine_supervision_sampling:
        print(json.dumps({"fine_sampling": fine_sampling_metadata}, ensure_ascii=False), flush=True)
    boundary_sampling_weights = None
    boundary_sampling_metadata: dict[str, object] = {"enabled": False}
    if config.boundary_aware_sampling and main_process:
        boundary_sampling_weights, boundary_sampling_metadata = boundary_aware_sampling_weights(
            train_ds,
            boost=config.boundary_sampling_boost,
            min_boundary_pixels=config.boundary_sampling_min_pixels,
            width=config.boundary_sampling_width,
        )
    if distributed and config.boundary_aware_sampling:
        boundary_sampling_payload = [
            (boundary_sampling_weights, boundary_sampling_metadata) if main_process else None
        ]
        dist.broadcast_object_list(boundary_sampling_payload, src=0)
        boundary_sampling_weights, boundary_sampling_metadata = boundary_sampling_payload[0]
    if main_process and config.boundary_aware_sampling:
        print(json.dumps({"boundary_sampling": boundary_sampling_metadata}, ensure_ascii=False), flush=True)
    train_sampler = None
    train_shuffle = True
    train_drop_last = False
    if config.fine_supervision_sampling:
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(config.seed + rank)
        default_samples = int(fine_sampling_metadata["eligible_records"])
        train_sampler = WeightedRandomSampler(
            fine_sampling_weights,
            num_samples=int(math.ceil((config.samples_per_epoch or default_samples) / world_size))
            if distributed
            else (config.samples_per_epoch or default_samples),
            replacement=True,
            generator=sampler_generator,
        )
        train_shuffle = False
    elif config.balanced_datasets or config.rare_class_sampling or config.boundary_aware_sampling:
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(config.seed + rank)
        sampling_weights = (
            dataset_balanced_weights(train_ds.records, temperature=config.dataset_sampling_temperature)
            if config.balanced_datasets
            else torch.ones(len(train_ds.records), dtype=torch.float64)
        )
        if config.rare_class_sampling:
            sampling_weights = sampling_weights * rare_class_presence_weights(
                train_ds,
                config.rare_class_ids,
                config.rare_class_sample_boost,
            )
        if config.boundary_aware_sampling:
            sampling_weights = sampling_weights * boundary_sampling_weights
        train_sampler = WeightedRandomSampler(
            sampling_weights,
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
        if distributed and not config.rank_zero_validation
        else None
    )
    loader_options = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=train_drop_last,
        **loader_options,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        **loader_options,
    )

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
        mask2former_class_weights=tuple(float(value) for value in class_weights.tolist()) if class_weights is not None else None,
        symmetric_padding=config.symmetric_padding,
        boundary_refinement=config.boundary_refinement,
        refinement_loss_weight=config.refinement_loss_weight,
        refinement_boundary_weight=config.refinement_boundary_weight,
        refinement_boundary_widths=config.refinement_boundary_widths,
        refinement_boundary_ce_weight=config.refinement_boundary_ce_weight,
        refinement_consistency_weight=config.refinement_consistency_weight,
        refinement_gate_width=config.refinement_gate_width,
        refinement_gate_threshold=config.refinement_gate_threshold,
        cellvit_mode=config.cellvit_mode,
        cell_prior_dropout=config.cell_prior_dropout,
        cell_aux_loss_weight=config.cell_aux_loss_weight,
        hierarchical_fine=config.hierarchical_fine,
        fine_loss_weight=config.fine_loss_weight,
        fine_only_loss=config.fine_only_loss,
        refinement_only_loss=config.refinement_only_loss,
        fine_class_weights=tuple(float(value) for value in fine_class_weights.tolist()) if fine_class_weights is not None else None,
        fine_supported_ids=fine_supported_ids,
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
    if trainable_scope != "all":
        scoped_parameters = _freeze_for_trainable_scope(model, trainable_scope)
        selected_backbone_parameters = []
        if main_process:
            print(
                f"froze shared segmentator; trainable_scope={trainable_scope} "
                f"parameter_tensors={len(scoped_parameters)}",
                flush=True,
            )
    else:
        selected_backbone_parameters = model.encoder.set_selected_blocks_trainable(
            config.backbone_unfreeze_blocks,
            trainable=True,
        )
    trainable_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if main_process:
        print(
            json.dumps(
                {
                    "trainable_scope": trainable_scope,
                    "trainable_parameter_count": trainable_parameter_count,
                    "trainable_parameter_names": trainable_parameter_names,
                }
            ),
            flush=True,
        )
    if distributed:
        print(f"[rank {rank}] wrapping model with DDP", flush=True)
        if config.decoder == "mask2former":
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=trainable_scope == "all",
            )
        else:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[rank {rank}] DDP ready", flush=True)
    for parameter in selected_backbone_parameters:
        parameter.requires_grad = False
    if selected_backbone_parameters:
        _unwrap_model(model).encoder.freeze = True
        _unwrap_model(model).encoder.trainable_block_count = 0
    selected_ids = {id(parameter) for parameter in selected_backbone_parameters}
    decoder_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in selected_ids]
    primary_group_name = trainable_scope if trainable_scope != "all" else "decoder"
    parameter_groups: list[dict[str, object]] = [{"params": decoder_parameters, "lr": config.lr, "name": primary_group_name}]
    if selected_backbone_parameters:
        parameter_groups.append({"params": selected_backbone_parameters, "lr": config.backbone_lr, "name": "backbone"})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)

    def lr_multiplier(epoch_index: int) -> float:
        if config.lr_scheduler == "none":
            return 1.0
        if config.warmup_epochs > 0 and epoch_index < config.warmup_epochs:
            return max(0.1, float(epoch_index + 1) / config.warmup_epochs)
        remaining = max(config.epochs - config.warmup_epochs, 1)
        progress = min(max((epoch_index - config.warmup_epochs) / remaining, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
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
    best_boundary_f1_4 = float("-inf")
    best_fine_miou = float("-inf")
    best_composite = float("-inf")
    epochs_without_improvement = 0
    metrics: dict[str, object] = {}
    start_epoch = 0
    if config.resume_from_checkpoint and config.init_from_checkpoint:
        raise ValueError("use only one of resume_from_checkpoint and init_from_checkpoint")
    initialization_metadata: dict[str, object] | None = None
    init_path = _resolve_resume_checkpoint(config.init_from_checkpoint, output_dir)
    if init_path is not None:
        initialization_metadata = _load_initialization_weights(init_path, model, device)
        if main_process:
            print(json.dumps({"initialization": initialization_metadata}, ensure_ascii=False), flush=True)
    resume_path = _resolve_resume_checkpoint(config.resume_from_checkpoint, output_dir)
    if resume_path is not None:
        resume_state = _load_training_state(resume_path, model, optimizer, scaler, device, scheduler=scheduler)
        start_epoch = int(resume_state["start_epoch"])
        history = resume_state["history"]
        best_miou = float(resume_state["best_miou"])
        best_core5_miou = float(resume_state["best_core5_miou"])
        best_boundary_f1_4 = float(resume_state["best_boundary_f1_4"])
        best_fine_miou = float(resume_state["best_fine_miou"])
        best_composite = float(resume_state["best_composite"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
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
                    "dataset_sampling_temperature": config.dataset_sampling_temperature,
                    "rare_class_sampling": config.rare_class_sampling,
                    "rare_class_ids": list(config.rare_class_ids),
                    "rare_class_sample_boost": config.rare_class_sample_boost,
                    "samples_per_epoch": config.samples_per_epoch,
                    "batch_size_per_gpu": config.batch_size,
                    "batch_size": config.batch_size,
                    "grad_accum_steps": config.grad_accum_steps,
                    "num_workers": config.num_workers,
                    "world_size": world_size,
                    "distributed": distributed,
                    "ddp_timeout_seconds": config.ddp_timeout_seconds,
                    "rank_zero_validation": config.rank_zero_validation,
                    "checkpoint_mode": config.checkpoint_mode,
                    "checkpoint_coarse_miou_floor": config.checkpoint_coarse_miou_floor,
                    "checkpoint_coarse_boundary_f1_4_floor": config.checkpoint_coarse_boundary_f1_4_floor,
                    "checkpoint_fine_dataset_macro_floor": config.checkpoint_fine_dataset_macro_floor,
                    "effective_batch_size": config.batch_size * config.grad_accum_steps * world_size,
                    "epochs": config.epochs,
                    "lr": config.lr,
                    "backbone_lr": config.backbone_lr,
                    "weight_decay": config.weight_decay,
                    "warmup_epochs": config.warmup_epochs,
                    "lr_scheduler": config.lr_scheduler,
                    "backbone_unfreeze_epoch": config.backbone_unfreeze_epoch,
                    "backbone_unfreeze_blocks": config.backbone_unfreeze_blocks,
                    "min_free_gpu_memory_gb_before_unfreeze": config.min_free_gpu_memory_gb_before_unfreeze,
                    "gpu_memory_poll_seconds": config.gpu_memory_poll_seconds,
                    "early_stopping_patience": config.early_stopping_patience,
                    "checkpoint_boundary_weight": config.checkpoint_boundary_weight,
                    "checkpoint_fine_weight": config.checkpoint_fine_weight,
                    "amp": config.amp,
                    "amp_enabled_runtime": use_amp,
                    "amp_disabled_reason": "mask2former_msdeformattn_stability" if config.amp and config.decoder == "mask2former" else None,
                    "disable_cudnn": config.disable_cudnn,
                    "freeze_encoder": config.freeze_encoder,
                    "decoder": config.decoder,
                    "mask2former_queries": config.mask2former_queries,
                    "mask2former_ignore_index": config.mask2former_ignore_index,
                    "symmetric_padding": config.symmetric_padding,
                    "boundary_refinement": config.boundary_refinement,
                    "refinement_loss_weight": config.refinement_loss_weight,
                    "refinement_boundary_weight": config.refinement_boundary_weight,
                    "refinement_boundary_widths": list(config.refinement_boundary_widths),
                    "refinement_boundary_ce_weight": config.refinement_boundary_ce_weight,
                    "refinement_consistency_weight": config.refinement_consistency_weight,
                    "refinement_gate_width": config.refinement_gate_width,
                    "refinement_gate_threshold": config.refinement_gate_threshold,
                    "boundary_aware_sampling": config.boundary_aware_sampling,
                    "boundary_sampling_boost": config.boundary_sampling_boost,
                    "boundary_sampling_min_pixels": config.boundary_sampling_min_pixels,
                    "boundary_sampling_width": config.boundary_sampling_width,
                    "boundary_sampling": boundary_sampling_metadata,
                    "cellvit_mode": config.cellvit_mode,
                    "cell_density_sigma": config.cell_density_sigma,
                    "cell_prior_dropout": config.cell_prior_dropout,
                    "cell_aux_loss_weight": config.cell_aux_loss_weight,
                    "hierarchical_fine": config.hierarchical_fine,
                    "fine_loss_weight": config.fine_loss_weight,
                    "fine_class_weighting": config.fine_class_weighting or config.class_weighting,
                    "fine_class_weight_min": config.fine_class_weight_min,
                    "fine_class_weight_max": config.fine_class_weight_max,
                    "fine_supervision_sampling": config.fine_supervision_sampling,
                    "fine_sampling_rare_class_boost": config.fine_sampling_rare_class_boost,
                    "fine_sampling_min_valid_pixels": config.fine_sampling_min_valid_pixels,
                    "fine_sampling_require_nuclei": config.fine_sampling_require_nuclei,
                    "fine_sampling": fine_sampling_metadata,
                    "freeze_shared_for_fine": config.freeze_shared_for_fine,
                    "trainable_scope": trainable_scope,
                    "trainable_parameter_count": trainable_parameter_count,
                    "trainable_parameter_names": trainable_parameter_names,
                    "fine_only_loss": config.fine_only_loss,
                    "refinement_only_loss": config.refinement_only_loss,
                    "fine_supported_ids": list(fine_supported_ids or []),
                    "augment_vflip": config.augment_vflip,
                    "augment_rot90": config.augment_rot90,
                    "augment_scale_crop": config.augment_scale_crop,
                    "metric_sample_limit": config.metric_sample_limit,
                    "mask2former_sanity_check_passed": sanity_check_passed,
                    "mask2former_feature_shapes": mask2former_feature_shapes,
                    "effective_invalid_target": config.mask2former_ignore_index if config.decoder == "mask2former" else config.ignore_index,
                    "class_weighting": config.class_weighting,
                    "label_space_summary_path": str(config.label_space_summary_path) if config.label_space_summary_path is not None else None,
                    "label_space_summary": label_space_summary,
                    "manifest_path": str(config.manifest_path) if config.manifest_path is not None else None,
                    "resume_from_checkpoint": config.resume_from_checkpoint,
                    "resume_checkpoint": str(resume_path) if resume_path is not None else None,
                    "init_from_checkpoint": config.init_from_checkpoint,
                    "initialization": initialization_metadata,
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
        (output_dir / "class_weights.json").write_text(
            json.dumps({"coarse": class_weight_metadata, "fine": fine_class_weight_metadata}, indent=2),
            encoding="utf-8",
        )
        if config.fine_supervision_sampling:
            (output_dir / "fine_sampling.json").write_text(
                json.dumps(fine_sampling_metadata, indent=2),
                encoding="utf-8",
            )
        if config.boundary_aware_sampling:
            (output_dir / "boundary_sampling.json").write_text(
                json.dumps(boundary_sampling_metadata, indent=2),
                encoding="utf-8",
            )
    if distributed:
        dist.barrier()

    encoder_unfrozen = False
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        if (
            not encoder_unfrozen
            and trainable_scope == "all"
            and config.backbone_unfreeze_blocks > 0
            and config.backbone_unfreeze_epoch >= 0
            and epoch + 1 >= config.backbone_unfreeze_epoch
        ):
            _wait_for_free_gpu_memory_before_unfreeze(
                device,
                config.min_free_gpu_memory_gb_before_unfreeze,
                config.gpu_memory_poll_seconds,
                main_process=main_process,
            )
            _unwrap_model(model).encoder.set_selected_blocks_trainable(config.backbone_unfreeze_blocks, True)
            encoder_unfrozen = True
            if main_process:
                print(f"unfroze last {config.backbone_unfreeze_blocks} UNI2-h blocks at epoch {epoch + 1}", flush=True)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        if trainable_scope != "all":
            _set_frozen_scope_training_mode(model, trainable_scope)
        else:
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
            nuclei_density = batch.get("nuclei_density")
            nuclei_available = batch.get("nuclei_available")
            fine_mask = batch.get("fine_mask")
            fine_allowed = batch.get("fine_allowed")
            if torch.is_tensor(nuclei_density):
                nuclei_density = nuclei_density.to(device)
            if torch.is_tensor(nuclei_available):
                nuclei_available = nuclei_available.to(device)
            if torch.is_tensor(fine_mask):
                fine_mask = fine_mask.to(device)
            if torch.is_tensor(fine_allowed):
                fine_allowed = fine_allowed.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if config.decoder == "mask2former" or config.hierarchical_fine:
                    losses = model(
                        image,
                        mask,
                        nuclei_density=nuclei_density,
                        nuclei_available=nuclei_available,
                        fine_target=fine_mask,
                        fine_allowed=fine_allowed,
                    )
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
            local_nonfinite = (~torch.isfinite(loss)).any().to(dtype=torch.int32)
            any_nonfinite = local_nonfinite.clone()
            if distributed:
                dist.all_reduce(any_nonfinite, op=dist.ReduceOp.MAX)
            if bool(any_nonfinite.item()):
                diagnostics = _loss_diagnostics(losses)
                print(
                    json.dumps(
                        {
                            "nonfinite_loss": diagnostics,
                            "local_nonfinite": bool(local_nonfinite.item()),
                            "rank": rank,
                            "epoch": epoch + 1,
                            "step": step,
                            "dataset_ids": [str(value) for value in batch["dataset_id"]],
                            "sample_ids": [str(value) for value in batch["sample_id"]],
                            "fine_valid_pixels": int((fine_mask != config.ignore_index).sum().item())
                            if torch.is_tensor(fine_mask)
                            else None,
                        },
                        allow_nan=True,
                    ),
                    flush=True,
                )
                raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}, step {step}")
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
        validation_model = _unwrap_model(model)
        run_validation = not distributed or not config.rank_zero_validation or main_process
        preds = []
        targets = []
        probs = []
        entropy = []
        logits = []
        sample_ids = []
        dataset_ids = []
        group_ids = []
        fine_preds = []
        fine_targets = []
        if run_validation:
            if main_process and distributed and config.rank_zero_validation:
                print(
                    f"epoch {epoch + 1}: rank 0 validating all {len(val_ds)} samples "
                    "without DDP forward collectives",
                    flush=True,
                )
        with torch.no_grad() if run_validation else torch.enable_grad():
            if not run_validation:
                val_bar = ()
            else:
                val_bar = tqdm(
                    val_loader,
                    desc=f"epoch {epoch + 1}/{config.epochs} val",
                    dynamic_ncols=True,
                    disable=not main_process,
                )
            for batch in val_bar:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                nuclei_density = batch.get("nuclei_density")
                fine_allowed = batch.get("fine_allowed")
                if torch.is_tensor(nuclei_density):
                    nuclei_density = nuclei_density.to(device)
                if torch.is_tensor(fine_allowed):
                    fine_allowed = fine_allowed.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = validation_model(image, nuclei_density=nuclei_density, fine_allowed=fine_allowed)
                preds.append(outputs["pred"].to(device="cpu", dtype=torch.uint8))
                targets.append(mask.to(device="cpu", dtype=torch.uint8))
                if config.hierarchical_fine:
                    fine_preds.append(outputs["fine_pred"].to(device="cpu", dtype=torch.uint8))
                    fine_targets.append(batch["fine_mask"].to(device="cpu", dtype=torch.uint8))
                if config.export_val_tensors:
                    probs.append(outputs["probs"].cpu())
                    entropy.append(outputs["entropy"].cpu())
                    logits.append(outputs["logits"].cpu())
                sample_ids.extend(str(v) for v in batch["sample_id"])
                dataset_ids.extend(str(v) for v in batch["dataset_id"])
                group_ids.extend(str(v) for v in batch["group_id"])

        local_payload = {
            "sample_ids": sample_ids,
            "dataset_ids": dataset_ids,
            "group_ids": group_ids,
            "pred": torch.cat(preds, dim=0) if preds else torch.empty(0, config.image_size, config.image_size, dtype=torch.uint8),
            "target": torch.cat(targets, dim=0) if targets else torch.empty(0, config.image_size, config.image_size, dtype=torch.uint8),
        }
        if config.hierarchical_fine:
            local_payload["fine_pred"] = torch.cat(fine_preds, dim=0) if fine_preds else torch.empty(0, config.image_size, config.image_size, dtype=torch.uint8)
            local_payload["fine_target"] = torch.cat(fine_targets, dim=0) if fine_targets else torch.empty(0, config.image_size, config.image_size, dtype=torch.uint8)
        if config.export_val_tensors:
            local_payload["probs"] = torch.cat(probs, dim=0) if probs else torch.empty(0)
            local_payload["entropy"] = torch.cat(entropy, dim=0) if entropy else torch.empty(0)
            local_payload["logits"] = torch.cat(logits, dim=0) if logits else torch.empty(0)

        if distributed and config.rank_zero_validation:
            gathered_payloads = [local_payload] if main_process else []
        else:
            gathered_payloads = _gather_object_to_main(local_payload, distributed)
        scheduler.step()
        if main_process:
            seen_sample_ids: set[str] = set()
            ordered_sample_ids: list[str] = []
            gathered_preds: list[torch.Tensor] = []
            gathered_targets: list[torch.Tensor] = []
            gathered_fine_preds: list[torch.Tensor] = []
            gathered_fine_targets: list[torch.Tensor] = []
            gathered_probs: list[torch.Tensor] = []
            gathered_entropy: list[torch.Tensor] = []
            gathered_logits: list[torch.Tensor] = []
            ordered_dataset_ids: list[str] = []
            ordered_group_ids: list[str] = []
            for payload in gathered_payloads:
                payload_sample_ids = payload["sample_ids"]
                payload_dataset_ids = payload["dataset_ids"]
                payload_group_ids = payload["group_ids"]
                payload_pred = payload["pred"]
                payload_target = payload["target"]
                payload_fine_pred = payload.get("fine_pred")
                payload_fine_target = payload.get("fine_target")
                payload_probs = payload.get("probs")
                payload_entropy = payload.get("entropy")
                payload_logits = payload.get("logits")
                for idx, sample_id in enumerate(payload_sample_ids):
                    if sample_id in seen_sample_ids:
                        continue
                    seen_sample_ids.add(sample_id)
                    ordered_sample_ids.append(sample_id)
                    ordered_dataset_ids.append(str(payload_dataset_ids[idx]))
                    ordered_group_ids.append(str(payload_group_ids[idx]))
                    gathered_preds.append(payload_pred[idx : idx + 1])
                    gathered_targets.append(payload_target[idx : idx + 1])
                    if config.hierarchical_fine:
                        gathered_fine_preds.append(payload_fine_pred[idx : idx + 1])
                        gathered_fine_targets.append(payload_fine_target[idx : idx + 1])
                    if config.export_val_tensors:
                        gathered_probs.append(payload_probs[idx : idx + 1])
                        gathered_entropy.append(payload_entropy[idx : idx + 1])
                        gathered_logits.append(payload_logits[idx : idx + 1])

            pred = torch.cat(gathered_preds, dim=0)
            target = torch.cat(gathered_targets, dim=0)
            fine_pred = torch.cat(gathered_fine_preds, dim=0) if config.hierarchical_fine else None
            fine_target = torch.cat(gathered_fine_targets, dim=0) if config.hierarchical_fine else None
            metrics = segmentation_metrics(
                pred,
                target,
                config.num_classes,
                class_names=manifest.classes,
                boundary_width=config.boundary_width,
                ignore_index=config.ignore_index,
                metric_sample_limit=config.metric_sample_limit,
            )
            metrics["case_macro"] = group_macro_iou(
                pred,
                target,
                ordered_group_ids,
                config.num_classes,
                ignore_index=config.ignore_index,
            )
            if config.hierarchical_fine:
                fine_metrics = fine_segmentation_metrics(
                    fine_pred,
                    fine_target,
                    NUM_FINE,
                    class_names=SEGMENTATOR_FINE_CLASSES,
                    ignore_index=config.ignore_index,
                )
                metrics["fine"] = fine_metrics
                metrics["fine_mIoU"] = float(fine_metrics["mIoU"])
                metrics["fine_accuracy"] = float(fine_metrics["accuracy"])
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
                    metric_sample_limit=config.metric_sample_limit,
                )
                per_dataset_metrics[dataset_id] = {
                    "samples": len(indices),
                    "mIoU": ds_metrics["mIoU"],
                    "mDice": ds_metrics["mDice"],
                    "foreground_recall": ds_metrics["foreground_recall"],
                    "boundary_f1": ds_metrics["boundary_f1"],
                    "boundary_f1_2": ds_metrics["boundary_f1_2"],
                    "boundary_f1_4": ds_metrics["boundary_f1_4"],
                    "boundary_f1_8": ds_metrics["boundary_f1_8"],
                    "hd95": ds_metrics["hd95"],
                    "fragmentation": ds_metrics["fragmentation"],
                    "case_macro": group_macro_iou(
                        pred.index_select(0, index_tensor),
                        target.index_select(0, index_tensor),
                        [ordered_group_ids[index] for index in indices],
                        config.num_classes,
                        ignore_index=config.ignore_index,
                    ),
                    "per_class": ds_metrics["per_class"],
                    "groups": ds_metrics["groups"],
                }
                if config.hierarchical_fine:
                    ds_fine_metrics = fine_segmentation_metrics(
                        fine_pred.index_select(0, index_tensor),
                        fine_target.index_select(0, index_tensor),
                        NUM_FINE,
                        class_names=SEGMENTATOR_FINE_CLASSES,
                        ignore_index=config.ignore_index,
                    )
                    majority_child_miou = _majority_child_miou(
                        fine_target.index_select(0, index_tensor),
                        ignore_index=config.ignore_index,
                    )
                    ds_fine_metrics["majority_child_mIoU"] = majority_child_miou
                    ds_fine_metrics["gain_over_majority"] = (
                        float(ds_fine_metrics["mIoU"]) - majority_child_miou
                        if math.isfinite(majority_child_miou)
                        else float("nan")
                    )
                    per_dataset_metrics[dataset_id]["fine"] = ds_fine_metrics
            metrics["per_dataset"] = per_dataset_metrics
            if config.hierarchical_fine:
                metrics["fine_dataset_macro_mIoU"] = _fine_dataset_macro(per_dataset_metrics)
                majority_values = [
                    float(dataset_metrics["fine"]["majority_child_mIoU"])
                    for dataset_metrics in per_dataset_metrics.values()
                    if isinstance(dataset_metrics.get("fine"), dict)
                    and math.isfinite(float(dataset_metrics["fine"].get("majority_child_mIoU", float("nan"))))
                ]
                metrics["fine_majority_child_dataset_macro_mIoU"] = (
                    float(np.mean(majority_values)) if majority_values else float("nan")
                )
                metrics["fine_gain_over_majority"] = (
                    float(metrics["fine_dataset_macro_mIoU"])
                    - float(metrics["fine_majority_child_dataset_macro_mIoU"])
                )
            if float(metrics["mIoU"]) > best_miou:
                best_miou = float(metrics["mIoU"])
                _save_model_state(model, output_dir / "best_mIoU.pt")
            core5 = metrics.get("groups", {}).get("core_5_classes", {}) if isinstance(metrics.get("groups"), dict) else {}
            if isinstance(core5, dict) and float(core5.get("mean_iou", float("-inf"))) > best_core5_miou:
                best_core5_miou = float(core5["mean_iou"])
                _save_model_state(model, output_dir / "best_core5.pt")
            boundary_f1_4 = float(metrics.get("boundary_f1_4", float("-inf")))
            if boundary_f1_4 > best_boundary_f1_4:
                best_boundary_f1_4 = boundary_f1_4
                _save_model_state(model, output_dir / "best_boundary_f1_4.pt")
            fine_miou = float(metrics.get("fine_mIoU", 0.0))
            if math.isfinite(fine_miou) and fine_miou > best_fine_miou:
                best_fine_miou = fine_miou
                _save_model_state(model, output_dir / "best_fine_mIoU.pt")
            composite = (
                float(metrics["mIoU"])
                + config.checkpoint_boundary_weight * boundary_f1_4
                + config.checkpoint_fine_weight * (fine_miou if math.isfinite(fine_miou) else 0.0)
            )
            metrics["checkpoint_composite"] = composite
            checkpoint_score, checkpoint_eligible = _checkpoint_selection_score(metrics, config)
            metrics["checkpoint_selection_score"] = checkpoint_score
            metrics["checkpoint_eligible"] = checkpoint_eligible
            if checkpoint_score > best_composite + config.early_stopping_min_delta:
                best_composite = checkpoint_score
                epochs_without_improvement = 0
                _save_model_state(model, output_dir / "best_composite.pt")
            else:
                epochs_without_improvement += 1
            metrics["learning_rates"] = {
                str(group.get("name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            }
            history.append({k: v for k, v in metrics.items() if isinstance(v, float)})
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
                        "boundary_f1_4": metrics["boundary_f1_4"],
                        "fine_mIoU": metrics.get("fine_mIoU"),
                        "fine_dataset_macro_mIoU": metrics.get("fine_dataset_macro_mIoU"),
                        "fine_majority_child_dataset_macro_mIoU": metrics.get(
                            "fine_majority_child_dataset_macro_mIoU"
                        ),
                        "fine_gain_over_majority": metrics.get("fine_gain_over_majority"),
                        "fine_accuracy": metrics.get("fine_accuracy"),
                        "hd95": metrics["hd95"],
                        "checkpoint_composite": composite,
                        "checkpoint_selection_score": checkpoint_score,
                        "checkpoint_eligible": checkpoint_eligible,
                        "epochs_without_improvement": epochs_without_improvement,
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
                best_boundary_f1_4=best_boundary_f1_4,
                best_fine_miou=best_fine_miou,
                best_composite=best_composite,
                epochs_without_improvement=epochs_without_improvement,
                metrics=metrics,
                scheduler=scheduler,
            )
        stop_training = config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience
        if distributed:
            metric_payload = [metrics if main_process else None]
            dist.broadcast_object_list(metric_payload, src=0)
            metrics = metric_payload[0]
            stop_payload = [stop_training if main_process else None]
            dist.broadcast_object_list(stop_payload, src=0)
            stop_training = bool(stop_payload[0])
        if stop_training:
            if main_process:
                print(f"early stopping after epoch {epoch + 1}: no composite improvement for {epochs_without_improvement} epochs", flush=True)
            break

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
                    "test_samples": [r.sample_id for r in manifest.test],
                    "train_groups": sorted({r.group_id for r in manifest.train}),
                    "val_groups": sorted({r.group_id for r in manifest.val}),
                    "test_groups": sorted({r.group_id for r in manifest.test}),
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
