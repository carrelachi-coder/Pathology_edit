"""Cross V5-min appearance fidelity losses.

The V5 appearance judge is deliberately low-level:

* color/stain statistics compare region distributions directly in pixel space;
* texture compares masked Gram matrices from shallow/mid VGG-style features.

Semantic encoders such as UNI2-h should not be used here; they belong in
semantic or geometry consistency losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CrossV5AppearanceLossConfig:
    """Weights and filters for V5 region appearance fidelity."""

    color_weight: float = 2.0
    texture_weight: float = 1.0
    mean_weight: float = 1.0
    std_weight: float = 1.0
    covariance_weight: float = 0.0
    min_pixels: int = 32
    exclude_labels: tuple[int, ...] = (0,)
    max_regions_per_sample: int | None = None
    color_space: str = "hed"
    eps: float = 1e-6
    stain_eps: float = 1e-3
    gram_normalize_by_channels: bool = True
    gram_standardize_features: bool = False


@dataclass(frozen=True)
class CrossV5GeometryConsistencyLossConfig:
    """Weights for dense target-geometry consistency.

    Callers are expected to run generated images through frozen differentiable
    tissue / nuclei predictors and pass the dense logits or maps here. This
    helper intentionally avoids argmax, watershed, or instance postprocessing.
    """

    tissue_ce_weight: float = 1.0
    tissue_dice_weight: float = 1.0
    nuclei_ce_weight: float = 1.0
    nuclei_dice_weight: float = 1.0
    nuclei_binary_bce_weight: float = 1.0
    nuclei_binary_dice_weight: float = 1.0
    dense_l1_weight: float = 1.0
    ignore_index: int = 255
    dice_eps: float = 1e-6


def cross_v5_appearance_fidelity_loss(
    *,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    prediction_vgg_features: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    reference_vgg_features: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    config: CrossV5AppearanceLossConfig | None = None,
) -> dict[str, torch.Tensor | int]:
    """Match generated target regions to same-class reference regions.

    ``prediction`` and ``reference`` are compared by class ID, not pixel
    alignment. VGG features are optional so callers can pass features from any
    shallow texture encoder without making this module own the encoder.
    """

    cfg = config or CrossV5AppearanceLossConfig()
    _validate_rgb_pair(prediction, reference)
    image_size = tuple(int(v) for v in prediction.shape[-2:])
    target_mask = _resize_mask_to_size(target_tissue_mask, image_size)
    reference_mask = _resize_mask_to_size(reference_tissue_mask, image_size)
    if target_mask.shape != reference_mask.shape:
        raise ValueError(
            f"target/reference masks must match on batch/spatial dims, got {tuple(target_mask.shape)} vs {tuple(reference_mask.shape)}."
        )
    if target_mask.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"mask batch size {target_mask.shape[0]} does not match image batch size {prediction.shape[0]}."
        )

    pred_color = _convert_color_space(
        prediction,
        color_space=cfg.color_space,
        eps=float(cfg.eps),
        stain_eps=float(cfg.stain_eps),
    )
    ref_color = _convert_color_space(
        reference.detach(),
        color_space=cfg.color_space,
        eps=float(cfg.eps),
        stain_eps=float(cfg.stain_eps),
    )
    pred_features = _normalize_feature_mapping(prediction_vgg_features)
    ref_features = _normalize_feature_mapping(reference_vgg_features)
    if set(pred_features) != set(ref_features):
        raise ValueError(
            f"prediction/reference feature layer keys differ: {sorted(pred_features)} vs {sorted(ref_features)}."
        )

    color_losses: list[torch.Tensor] = []
    texture_losses: list[torch.Tensor] = []
    exclude = set(int(label) for label in cfg.exclude_labels)
    min_pixels = max(1, int(cfg.min_pixels))
    for batch_index in range(prediction.shape[0]):
        labels = _shared_labels(
            target_mask[batch_index],
            reference_mask[batch_index],
            exclude_labels=exclude,
        )
        if cfg.max_regions_per_sample is not None and len(labels) > cfg.max_regions_per_sample:
            labels = _largest_labels(
                labels,
                target_mask[batch_index],
                max_regions=int(cfg.max_regions_per_sample),
            )
        for label in labels:
            target_region = target_mask[batch_index] == label
            reference_region = reference_mask[batch_index] == label
            if int(target_region.sum().item()) < min_pixels:
                continue
            if int(reference_region.sum().item()) < min_pixels:
                continue
            color_losses.append(
                _region_color_stat_loss(
                    pred_color[batch_index],
                    ref_color[batch_index],
                    target_region,
                    reference_region,
                    config=cfg,
                )
            )
            if pred_features:
                texture_losses.append(
                    _region_masked_gram_loss(
                        prediction_features={key: value[batch_index] for key, value in pred_features.items()},
                        reference_features={key: value[batch_index] for key, value in ref_features.items()},
                        target_region=target_region,
                        reference_region=reference_region,
                        eps=float(cfg.eps),
                        normalize_by_channels=bool(cfg.gram_normalize_by_channels),
                        standardize_features=bool(cfg.gram_standardize_features),
                    )
                )

    zero = prediction.new_zeros(())
    color_loss = torch.stack(color_losses).mean() if color_losses else zero
    texture_loss = torch.stack(texture_losses).mean() if texture_losses else zero

    total = zero
    active_weight = 0.0
    if color_losses and cfg.color_weight > 0.0:
        total = total + float(cfg.color_weight) * color_loss
        active_weight += float(cfg.color_weight)
    if texture_losses and cfg.texture_weight > 0.0:
        total = total + float(cfg.texture_weight) * texture_loss
        active_weight += float(cfg.texture_weight)
    total = total / active_weight if active_weight > 0.0 else zero
    return {
        "total": total,
        "color": color_loss,
        "texture": texture_loss,
        "regions": len(color_losses),
        "texture_regions": len(texture_losses),
    }


def cross_v5_geometry_consistency_loss(
    *,
    tissue_logits: torch.Tensor | None = None,
    target_tissue_mask: torch.Tensor | None = None,
    nuclei_logits: torch.Tensor | None = None,
    target_nuclei_mask: torch.Tensor | None = None,
    nuclei_binary_logits: torch.Tensor | None = None,
    target_nuclei_binary: torch.Tensor | None = None,
    dense_predictions: Mapping[str, torch.Tensor] | None = None,
    dense_targets: Mapping[str, torch.Tensor] | None = None,
    config: CrossV5GeometryConsistencyLossConfig | None = None,
) -> dict[str, torch.Tensor | int]:
    """Keep generated-image predictor outputs aligned to target geometry.

    This is the missing fourth V5-min loss family: denoise handles generative
    training, appearance handles reference fidelity, and this loss welds the
    generated structure back to target tissue/nuclei masks.
    """

    cfg = config or CrossV5GeometryConsistencyLossConfig()
    sample = _first_not_none(tissue_logits, nuclei_logits, nuclei_binary_logits)
    if sample is None and dense_predictions:
        sample = next(iter(dense_predictions.values()))
    if sample is None:
        raise ValueError("At least one dense geometry prediction tensor is required.")
    zero = sample.new_zeros(())

    tissue_ce = zero
    tissue_dice = zero
    nuclei_ce = zero
    nuclei_dice = zero
    binary_bce = zero
    binary_dice = zero
    dense_l1 = zero
    active_weight = 0.0
    dense_terms = 0

    if tissue_logits is not None:
        if target_tissue_mask is None:
            raise ValueError("target_tissue_mask is required when tissue_logits are provided.")
        logits, target = _align_logits_and_mask(tissue_logits, target_tissue_mask)
        if cfg.tissue_ce_weight > 0.0:
            tissue_ce = F.cross_entropy(logits.float(), target.long(), ignore_index=int(cfg.ignore_index))
            active_weight += float(cfg.tissue_ce_weight)
        if cfg.tissue_dice_weight > 0.0:
            tissue_dice = _soft_dice_loss(
                logits,
                target,
                ignore_index=int(cfg.ignore_index),
                eps=float(cfg.dice_eps),
            )
            active_weight += float(cfg.tissue_dice_weight)

    if nuclei_logits is not None:
        if target_nuclei_mask is None:
            raise ValueError("target_nuclei_mask is required when nuclei_logits are provided.")
        logits, target = _align_logits_and_mask(nuclei_logits, target_nuclei_mask)
        if cfg.nuclei_ce_weight > 0.0:
            nuclei_ce = F.cross_entropy(logits.float(), target.long(), ignore_index=int(cfg.ignore_index))
            active_weight += float(cfg.nuclei_ce_weight)
        if cfg.nuclei_dice_weight > 0.0:
            nuclei_dice = _soft_dice_loss(
                logits,
                target,
                ignore_index=int(cfg.ignore_index),
                eps=float(cfg.dice_eps),
            )
            active_weight += float(cfg.nuclei_dice_weight)

    if nuclei_binary_logits is not None:
        if target_nuclei_binary is None:
            if target_nuclei_mask is None:
                raise ValueError(
                    "target_nuclei_binary or target_nuclei_mask is required when nuclei_binary_logits are provided."
                )
            target_nuclei_binary = torch.where(
                target_nuclei_mask == int(cfg.ignore_index),
                torch.full_like(target_nuclei_mask, int(cfg.ignore_index), dtype=torch.float32),
                (target_nuclei_mask != 0).to(dtype=torch.float32),
            )
        logits, target = _align_binary_logits_and_target(nuclei_binary_logits, target_nuclei_binary)
        valid_mask = target != float(cfg.ignore_index)
        target_float = target.float().clamp(0.0, 1.0)
        if cfg.nuclei_binary_bce_weight > 0.0:
            binary_bce = _masked_bce_with_logits(logits.float(), target_float, valid_mask)
            active_weight += float(cfg.nuclei_binary_bce_weight)
        if cfg.nuclei_binary_dice_weight > 0.0:
            binary_dice = _binary_soft_dice_loss(
                logits.float(),
                target_float,
                valid_mask,
                eps=float(cfg.dice_eps),
            )
            active_weight += float(cfg.nuclei_binary_dice_weight)

    if dense_predictions is not None or dense_targets is not None:
        if not dense_predictions or not dense_targets:
            raise ValueError("dense_predictions and dense_targets must be provided together.")
        if set(dense_predictions) != set(dense_targets):
            raise ValueError(
                f"dense prediction/target keys differ: {sorted(dense_predictions)} vs {sorted(dense_targets)}."
            )
        dense_losses = []
        for key in sorted(dense_predictions):
            pred = dense_predictions[key]
            target = dense_targets[key].to(device=pred.device, dtype=pred.dtype)
            target = _resize_dense_target_to_prediction(target, pred)
            dense_losses.append(F.l1_loss(pred.float(), target.detach().float()))
        if dense_losses:
            dense_l1 = torch.stack(dense_losses).mean()
            dense_terms = len(dense_losses)
            if cfg.dense_l1_weight > 0.0:
                active_weight += float(cfg.dense_l1_weight)

    total = zero
    if cfg.tissue_ce_weight > 0.0 and tissue_logits is not None:
        total = total + float(cfg.tissue_ce_weight) * tissue_ce
    if cfg.tissue_dice_weight > 0.0 and tissue_logits is not None:
        total = total + float(cfg.tissue_dice_weight) * tissue_dice
    if cfg.nuclei_ce_weight > 0.0 and nuclei_logits is not None:
        total = total + float(cfg.nuclei_ce_weight) * nuclei_ce
    if cfg.nuclei_dice_weight > 0.0 and nuclei_logits is not None:
        total = total + float(cfg.nuclei_dice_weight) * nuclei_dice
    if cfg.nuclei_binary_bce_weight > 0.0 and nuclei_binary_logits is not None:
        total = total + float(cfg.nuclei_binary_bce_weight) * binary_bce
    if cfg.nuclei_binary_dice_weight > 0.0 and nuclei_binary_logits is not None:
        total = total + float(cfg.nuclei_binary_dice_weight) * binary_dice
    if cfg.dense_l1_weight > 0.0 and dense_terms > 0:
        total = total + float(cfg.dense_l1_weight) * dense_l1
    total = total / active_weight if active_weight > 0.0 else zero

    return {
        "total": total,
        "tissue_ce": tissue_ce,
        "tissue_dice": tissue_dice,
        "nuclei_ce": nuclei_ce,
        "nuclei_dice": nuclei_dice,
        "nuclei_binary_bce": binary_bce,
        "nuclei_binary_dice": binary_dice,
        "dense_l1": dense_l1,
        "dense_terms": dense_terms,
    }


def masked_gram_matrix(
    features: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-6,
    normalize_by_channels: bool = True,
    standardize_features: bool = False,
) -> torch.Tensor:
    """Compute a Gram matrix over feature pixels selected by ``mask``."""

    if features.ndim != 3:
        raise ValueError(f"features must have shape (C,H,W), got {tuple(features.shape)}.")
    mask = _resize_single_mask(mask, tuple(int(v) for v in features.shape[-2:])).to(
        device=features.device,
        dtype=features.dtype,
    )
    feature_values = features.float()
    if standardize_features:
        feature_values = _standardize_region_features(feature_values, mask, eps=float(eps))
    weighted = feature_values * mask.unsqueeze(0).float()
    flat = weighted.flatten(1)
    normalizer = mask.sum().clamp_min(float(eps))
    if normalize_by_channels:
        normalizer = normalizer * max(int(features.shape[0]), 1)
    return flat @ flat.t() / normalizer


def rgb_to_hed_concentrations(
    rgb: torch.Tensor,
    *,
    eps: float = 1e-6,
    stain_eps: float = 1e-3,
) -> torch.Tensor:
    """Approximate RGB to HED optical-density concentrations.

    The first two channels correspond to hematoxylin/eosin-like stain
    concentrations. The third residual channel is retained for stable stats.
    Inputs may be in ``[0,1]`` or ``[-1,1]``.
    """

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape (B,3,H,W), got {tuple(rgb.shape)}.")
    x = rgb.float()
    if float(x.detach().min().item()) < -0.01:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    od = -torch.log(x + max(float(stain_eps), float(eps)))
    stain_matrix = x.new_tensor(
        [
            [0.650, 0.704, 0.286],
            [0.072, 0.990, 0.105],
            [0.268, 0.570, 0.776],
        ]
    )
    inv = torch.linalg.pinv(stain_matrix)
    concentrations = torch.einsum("cd,bdhw->bchw", inv, od)
    return concentrations


def _region_color_stat_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target_region: torch.Tensor,
    reference_region: torch.Tensor,
    *,
    config: CrossV5AppearanceLossConfig,
) -> torch.Tensor:
    pred_stats = _region_stats(prediction, target_region, eps=float(config.eps))
    ref_stats = _region_stats(reference, reference_region, eps=float(config.eps))
    total = prediction.new_zeros(())
    active = 0.0
    if config.mean_weight > 0.0:
        total = total + float(config.mean_weight) * F.l1_loss(pred_stats["mean"], ref_stats["mean"])
        active += float(config.mean_weight)
    if config.std_weight > 0.0:
        total = total + float(config.std_weight) * F.l1_loss(pred_stats["std"], ref_stats["std"])
        active += float(config.std_weight)
    if config.covariance_weight > 0.0:
        total = total + float(config.covariance_weight) * F.l1_loss(
            pred_stats["covariance"],
            ref_stats["covariance"],
        )
        active += float(config.covariance_weight)
    return total / active if active > 0.0 else total


def _region_masked_gram_loss(
    *,
    prediction_features: Mapping[str, torch.Tensor],
    reference_features: Mapping[str, torch.Tensor],
    target_region: torch.Tensor,
    reference_region: torch.Tensor,
    eps: float,
    normalize_by_channels: bool,
    standardize_features: bool,
) -> torch.Tensor:
    losses = []
    for key in sorted(prediction_features):
        pred = prediction_features[key]
        ref = reference_features[key].detach()
        if pred.ndim != 3 or ref.ndim != 3:
            raise ValueError(f"feature layer {key!r} must have shape (C,H,W).")
        if pred.shape[0] != ref.shape[0]:
            raise ValueError(
                f"feature layer {key!r} channel mismatch: {tuple(pred.shape)} vs {tuple(ref.shape)}."
            )
        pred_gram = masked_gram_matrix(
            pred,
            target_region,
            eps=eps,
            normalize_by_channels=normalize_by_channels,
            standardize_features=standardize_features,
        )
        ref_gram = masked_gram_matrix(
            ref,
            reference_region,
            eps=eps,
            normalize_by_channels=normalize_by_channels,
            standardize_features=standardize_features,
        )
        losses.append(F.l1_loss(pred_gram, ref_gram))
    if not losses:
        sample = next(iter(prediction_features.values()))
        return sample.new_zeros(())
    return torch.stack(losses).mean()


def _region_stats(image: torch.Tensor, region: torch.Tensor, *, eps: float) -> dict[str, torch.Tensor]:
    values = image[:, region.to(device=image.device, dtype=torch.bool)].float()
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("region must select at least one pixel.")
    mean = values.mean(dim=1)
    centered = values - mean[:, None]
    variance = centered.square().mean(dim=1)
    std = torch.sqrt(variance + float(eps))
    covariance = centered @ centered.t() / max(int(values.shape[1]), 1)
    return {"mean": mean, "std": std, "covariance": covariance}


def _align_logits_and_mask(logits: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 4:
        raise ValueError(f"logits must have shape (B,C,H,W), got {tuple(logits.shape)}.")
    target = _resize_mask_to_size(target_mask, tuple(int(v) for v in logits.shape[-2:])).to(device=logits.device)
    if target.shape[0] != logits.shape[0]:
        raise ValueError(f"target batch size {target.shape[0]} does not match logits batch size {logits.shape[0]}.")
    return logits, target.long()


def _align_binary_logits_and_target(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"binary logits must have shape (B,1,H,W) or (B,H,W), got {tuple(logits.shape)}.")
    target_resized = _resize_binary_target_to_size(target, tuple(int(v) for v in logits.shape[-2:])).to(
        device=logits.device,
        dtype=logits.dtype,
    )
    if target_resized.shape[0] != logits.shape[0]:
        raise ValueError(
            f"binary target batch size {target_resized.shape[0]} does not match logits batch size {logits.shape[0]}."
        )
    return logits[:, 0], target_resized


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, *, ignore_index: int, eps: float) -> torch.Tensor:
    num_classes = int(logits.shape[1])
    valid = target != int(ignore_index)
    safe_target = target.clamp(0, num_classes - 1)
    probs = torch.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(safe_target.long(), num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=probs.dtype)
    valid_f = valid.unsqueeze(1).to(dtype=probs.dtype)
    probs = probs * valid_f
    one_hot = one_hot * valid_f
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    present = one_hot.sum(dim=dims) > 0
    dice = (2.0 * intersection + float(eps)) / (denominator + float(eps))
    if not bool(present.any().item()):
        return logits.new_zeros(())
    return 1.0 - dice[present].mean()


def _masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.detach(), reduction="none")
    valid = valid_mask.to(device=loss.device, dtype=loss.dtype)
    if not bool((valid > 0).any().item()):
        return logits.new_zeros(())
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _binary_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    valid = valid_mask.to(device=logits.device, dtype=logits.dtype)
    probs = torch.sigmoid(logits.float()) * valid
    target = target.detach().float() * valid
    intersection = (probs * target).sum()
    denominator = probs.sum() + target.sum()
    if float(valid.sum().detach().item()) <= 0.0:
        return logits.new_zeros(())
    return 1.0 - (2.0 * intersection + float(eps)) / (denominator + float(eps))


def _resize_dense_target_to_prediction(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    prediction_was_3d = prediction.ndim == 3
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)
    if target.ndim != 4 or prediction.ndim != 4:
        raise ValueError(
            f"dense predictions/targets must have shape (B,C,H,W) or (B,H,W), got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if target.shape[0] != prediction.shape[0]:
        raise ValueError(f"dense target batch size {target.shape[0]} does not match prediction {prediction.shape[0]}.")
    if target.shape[-2:] != prediction.shape[-2:]:
        target = F.interpolate(target.float(), size=prediction.shape[-2:], mode="bilinear", align_corners=False)
    if prediction_was_3d:
        return target[:, 0]
    return target


def _resize_binary_target_to_size(target: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3:
        raise ValueError(f"binary target must have shape (B,H,W), (B,1,H,W), or (H,W), got {tuple(target.shape)}.")
    if tuple(int(v) for v in target.shape[-2:]) == tuple(int(v) for v in image_size):
        return target.float()
    resized = F.interpolate(target.unsqueeze(1).float(), size=image_size, mode="nearest")
    return resized[:, 0]


def _first_not_none(*values: torch.Tensor | None) -> torch.Tensor | None:
    for value in values:
        if value is not None:
            return value
    return None


def _convert_color_space(
    image: torch.Tensor,
    *,
    color_space: str,
    eps: float = 1e-6,
    stain_eps: float = 1e-3,
) -> torch.Tensor:
    normalized = str(color_space or "rgb").strip().lower()
    if normalized in {"rgb", "raw"}:
        return image.float()
    if normalized in {"hed", "he"}:
        return rgb_to_hed_concentrations(image, eps=eps, stain_eps=stain_eps)
    raise ValueError(f"Unsupported color_space {color_space!r}; choose 'hed' or 'rgb'.")


def _standardize_region_features(features: torch.Tensor, mask: torch.Tensor, *, eps: float) -> torch.Tensor:
    region = mask.to(device=features.device, dtype=torch.bool)
    values = features[:, region].float()
    if values.shape[1] == 0:
        return features.float()
    mean = values.mean(dim=1).view(-1, 1, 1)
    std = values.std(dim=1, unbiased=False).clamp_min(float(eps)).view(-1, 1, 1)
    standardized = (features.float() - mean) / std
    return standardized


def _normalize_feature_mapping(
    features: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if features is None:
        return {}
    if isinstance(features, Mapping):
        result = {str(key): value for key, value in features.items()}
    else:
        result = {str(index): value for index, value in enumerate(features)}
    for key, value in result.items():
        if value.ndim != 4:
            raise ValueError(f"feature layer {key!r} must have shape (B,C,H,W), got {tuple(value.shape)}.")
    return result


def _resize_mask_to_size(mask: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), (B,1,H,W), or (H,W), got {tuple(mask.shape)}.")
    if tuple(int(v) for v in mask.shape[-2:]) == tuple(int(v) for v in image_size):
        return mask.long()
    resized = F.interpolate(mask.unsqueeze(1).float(), size=image_size, mode="nearest")
    return resized[:, 0].long()


def _resize_single_mask(mask: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"single mask must have shape (H,W), got {tuple(mask.shape)}.")
    if tuple(int(v) for v in mask.shape[-2:]) == tuple(int(v) for v in image_size):
        return mask
    resized = F.interpolate(mask[None, None].float(), size=image_size, mode="nearest")
    return resized[0, 0]


def _shared_labels(
    target_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    exclude_labels: set[int],
) -> list[int]:
    target_labels = {int(value) for value in torch.unique(target_mask.detach()).cpu().tolist()}
    reference_labels = {int(value) for value in torch.unique(reference_mask.detach()).cpu().tolist()}
    return sorted((target_labels & reference_labels) - exclude_labels)


def _largest_labels(labels: list[int], mask: torch.Tensor, *, max_regions: int) -> list[int]:
    return sorted(
        labels,
        key=lambda label: int((mask == label).sum().item()),
        reverse=True,
    )[: max(0, int(max_regions))]


def _validate_rgb_pair(prediction: torch.Tensor, reference: torch.Tensor) -> None:
    if prediction.ndim != 4 or reference.ndim != 4:
        raise ValueError(
            f"prediction/reference must have shape (B,C,H,W), got {tuple(prediction.shape)} and {tuple(reference.shape)}."
        )
    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction/reference shapes differ: {tuple(prediction.shape)} vs {tuple(reference.shape)}."
        )
    if prediction.shape[1] != 3:
        raise ValueError(f"expected RGB tensors with C=3, got {prediction.shape[1]}.")


__all__ = [
    "CrossV5AppearanceLossConfig",
    "CrossV5GeometryConsistencyLossConfig",
    "cross_v5_appearance_fidelity_loss",
    "cross_v5_geometry_consistency_loss",
    "masked_gram_matrix",
    "rgb_to_hed_concentrations",
]
