from __future__ import annotations

import json
from pathlib import Path
import random
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .config import BaselineConfig
from .data import TissueSegmentationDataset, build_manifest, dataset_balanced_weights, load_manifest, load_mask, remap_mask_to_coarse, coarse_remap_table
from .losses import segmentation_loss
from .metrics import segmentation_metrics
from .model import BaselineSegmenter


def _run_mask2former_sanity_check(model: BaselineSegmenter, image_size: int, num_classes: int, device: torch.device) -> None:
    was_training = model.training
    model.train()
    dummy_img = torch.zeros(1, 3, image_size, image_size, device=device)
    dummy_mask = torch.zeros(1, image_size, image_size, dtype=torch.long, device=device)
    try:
        losses = model.loss(dummy_img, dummy_mask)
        total = losses.get("total")
        if not torch.is_tensor(total) or not torch.isfinite(total.detach()).all():
            raise RuntimeError(f"non-finite sanity loss: {total}")
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
    if mode == "none":
        metadata["weights"] = None
        return None, metadata
    if mode == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(frequencies.clamp_min(1e-8))
        weights = weights / weights.mean().clamp_min(1e-8)
        metadata["weights"] = [float(v) for v in weights.tolist()]
        return weights.float(), metadata
    raise ValueError(f"unsupported class weighting mode: {mode}")


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
    set_seed(config.seed)
    if config.disable_cudnn:
        torch.backends.cudnn.enabled = False
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
    class_weights, class_weight_metadata = compute_class_weights(
        train_ds,
        config.num_classes,
        config.class_weighting,
        config.remap_invalid_to,
    )
    if config.decoder == "mask2former":
        class_weight_metadata["effective_invalid_target"] = config.mask2former_ignore_index
        class_weight_metadata["note"] = "Mask2Former masks invalid labels with ignore_index during target assignment; no-object query weight is separate."
    train_sampler = None
    train_shuffle = True
    if config.balanced_datasets:
        train_sampler = WeightedRandomSampler(
            dataset_balanced_weights(train_ds.records),
            num_samples=config.samples_per_epoch or len(train_ds),
            replacement=True,
        )
        train_shuffle = False
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_device = class_weights.to(device) if class_weights is not None else None
    model = BaselineSegmenter(
        num_classes=config.num_classes,
        freeze_encoder=config.freeze_encoder,
        local_repo=uni2h_repo,
        decoder=config.decoder,
        mask2former_queries=config.mask2former_queries,
        mask2former_ignore_index=config.mask2former_ignore_index,
    ).to(device)
    sanity_check_passed = False
    if config.decoder == "mask2former":
        _run_mask2former_sanity_check(model, config.image_size, config.num_classes, device)
        sanity_check_passed = True
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay)
    use_amp = config.amp and device.type == "cuda" and config.decoder != "mask2former"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = config.resolve_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "image_size": config.image_size,
                "remap_invalid_to": config.remap_invalid_to,
                "ignore_index": config.ignore_index,
                "mask_remap": config.mask_remap,
                "balanced_datasets": config.balanced_datasets,
                "samples_per_epoch": config.samples_per_epoch,
                "batch_size": config.batch_size,
                "grad_accum_steps": config.grad_accum_steps,
                "effective_batch_size": config.batch_size * config.grad_accum_steps,
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
                "effective_invalid_target": config.mask2former_ignore_index if config.decoder == "mask2former" else config.ignore_index,
                "class_weighting": config.class_weighting,
                "manifest_path": str(config.manifest_path) if config.manifest_path is not None else None,
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

    history: list[dict[str, float]] = []
    best_miou = float("-inf")
    best_core5_miou = float("-inf")
    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{config.epochs} train",
            dynamic_ncols=True,
        )
        running_loss = 0.0
        for step, batch in enumerate(train_bar, start=1):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if config.decoder == "mask2former":
                    losses = model.loss(image, mask)
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
            running_loss += float(losses["total"].detach().cpu().item())
            train_bar.set_postfix(loss=running_loss / step)

        model.eval()
        preds = []
        targets = []
        probs = []
        entropy = []
        logits = []
        sample_ids = []
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"epoch {epoch + 1}/{config.epochs} val",
                dynamic_ncols=True,
            )
            for batch in val_bar:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(image)
                preds.append(outputs["pred"].cpu())
                targets.append(mask.cpu())
                probs.append(outputs["probs"].cpu())
                entropy.append(outputs["entropy"].cpu())
                logits.append(outputs["logits"].cpu())
                sample_ids.extend(str(v) for v in batch["sample_id"])

        pred = torch.cat(preds, dim=0)
        target = torch.cat(targets, dim=0)
        metrics = segmentation_metrics(
            pred,
            target,
            config.num_classes,
            class_names=manifest.classes,
            boundary_width=config.boundary_width,
            ignore_index=config.ignore_index,
        )
        history.append({k: v for k, v in metrics.items() if isinstance(v, float)})
        if float(metrics["mIoU"]) > best_miou:
            best_miou = float(metrics["mIoU"])
            torch.save(model.state_dict(), output_dir / "best_mIoU.pt")
        core5 = metrics.get("groups", {}).get("core_5_classes", {}) if isinstance(metrics.get("groups"), dict) else {}
        if isinstance(core5, dict) and float(core5.get("mean_iou", float("-inf"))) > best_core5_miou:
            best_core5_miou = float(core5["mean_iou"])
            torch.save(model.state_dict(), output_dir / "best_core5.pt")
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
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if epoch == config.epochs - 1 and config.export_val_predictions:
            _export_val_outputs(output_dir, sample_ids, preds, probs, entropy, logits, export_tensors=config.export_val_tensors)

    torch.save(model.state_dict(), output_dir / "stage4_baseline.pt")
    torch.save(model.state_dict(), output_dir / f"stage4_{config.decoder}.pt")
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
    return {k: v for k, v in metrics.items() if isinstance(v, float)}
