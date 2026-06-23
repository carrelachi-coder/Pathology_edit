"""Masked Gatys texture/stain transfer for ControlNet outputs.

The optimizer starts from a generated structure image ``I0`` and updates pixels
directly so masked VGG Gram statistics match the reference image by tissue
label, while a VGG content anchor keeps the original structure close to ``I0``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


VGG19_RELU_LAYER_ALIASES = {
    "conv1_1": 1,
    "relu1_1": 1,
    "conv2_1": 6,
    "relu2_1": 6,
    "conv3_1": 11,
    "relu3_1": 11,
    "conv4_1": 20,
    "relu4_1": 20,
    "conv4_2": 22,
    "relu4_2": 22,
    "conv5_1": 29,
    "relu5_1": 29,
}
DEFAULT_STYLE_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")
DEFAULT_LAYER_WEIGHTS = {
    "conv1_1": 1.0,
    "conv2_1": 1.0,
    "conv3_1": 0.5,
    "conv4_1": 0.25,
    "conv5_1": 0.0,
}
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
NUCLEI_STAIN_LABEL_OFFSET = 256


@dataclass(frozen=True)
class GatysTransferConfig:
    """Configuration for one masked Gatys optimization."""

    steps: int = 300
    optimizer: str = "lbfgs"
    lr: float = 1.0
    adam_lr: float = 0.02
    content_weight: float = 1.0
    use_content_loss: bool = True
    style_weight: float = 1e4
    tv_weight: float = 0.0
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS
    content_layer: str = "conv4_2"
    layer_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_LAYER_WEIGHTS))
    min_region_pixels: int = 4
    log_every: int = 25
    save_every: int = 0
    preserve_background: bool = True
    background_label: int = 0
    optimize_background: bool = False
    missing_region_fallback: str = "pooled"
    save_mask_debug: bool = True
    pre_color_match: str = "none"
    color_match_scope: str = "region"
    color_match_strength: float = 1.0
    macenko_io: float = 240.0
    macenko_beta: float = 0.15
    macenko_alpha: float = 1.0
    device: str = "cuda"
    torch_dtype: str = "fp32"
    vgg_weights: str = "imagenet"
    vgg_weights_path: str | None = None


@dataclass(frozen=True)
class GatysTransferResult:
    image: Image.Image
    history: list[dict[str, float]]
    active_regions: tuple[int, ...]
    output_path: Path | None = None
    metrics_path: Path | None = None
    pre_color_match_path: Path | None = None


class VGG19FeatureExtractor(nn.Module):
    """Frozen VGG19 feature extractor for selected ReLU layer activations."""

    def __init__(
        self,
        features: nn.Module,
        *,
        layers: Sequence[str],
        normalize_mean: Sequence[float] = IMAGENET_RGB_MEAN,
        normalize_std: Sequence[float] = IMAGENET_RGB_STD,
    ) -> None:
        super().__init__()
        self.features = features.eval()
        self.layer_indices = _parse_vgg19_layers(layers)
        if not self.layer_indices:
            raise ValueError("at least one VGG19 layer is required")
        self.names_by_index = {index: name for name, index in self.layer_indices.items()}
        self.max_index = max(self.names_by_index)
        self.register_buffer(
            "normalize_mean",
            torch.tensor(tuple(normalize_mean), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "normalize_std",
            torch.tensor(tuple(normalize_std), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape (B,3,H,W), got {tuple(images.shape)}")
        device, dtype = _module_device_dtype(self)
        x = images.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        mean = self.normalize_mean.to(device=device, dtype=dtype)
        std = self.normalize_std.to(device=device, dtype=dtype).clamp_min(1e-6)
        x = (x - mean) / std
        outputs: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.features):
            x = layer(x)
            name = self.names_by_index.get(index)
            if name is not None:
                outputs[name] = x
            if index >= self.max_index:
                break
        missing = [name for name in self.layer_indices if name not in outputs]
        if missing:
            raise ValueError(f"VGG19 ended before requested layers: {missing}")
        return outputs


class MaskedGatysStyleTransfer:
    """Reusable runner around a frozen feature extractor."""

    def __init__(self, extractor: nn.Module, config: GatysTransferConfig | None = None) -> None:
        self.extractor = extractor.eval()
        self.config = config or GatysTransferConfig()

    def run(
        self,
        *,
        initial_image: torch.Tensor,
        reference_image: torch.Tensor,
        target_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        regions: Sequence[int] | None = None,
        output_dir: str | Path | None = None,
        output_name: str = "masked_gatys.png",
    ) -> GatysTransferResult:
        cfg = self.config
        device, dtype = _module_device_dtype(self.extractor)
        i0 = _prepare_image_tensor(initial_image, device=device, dtype=dtype)
        ref = _prepare_image_tensor(reference_image, device=device, dtype=dtype)
        target_mask = _prepare_mask_tensor(target_mask, device=device)
        reference_mask = _prepare_mask_tensor(reference_mask, device=device)
        if target_mask.shape[-2:] != i0.shape[-2:]:
            target_mask = resize_label_mask(target_mask, i0.shape[-2:])
        if reference_mask.shape[-2:] != ref.shape[-2:]:
            reference_mask = resize_label_mask(reference_mask, ref.shape[-2:])

        output_root = Path(output_dir) if output_dir is not None else None
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)

        pre_color_match_path = None
        if str(cfg.pre_color_match).strip().lower() not in {"", "none", "off", "false"}:
            i0 = color_match_tensor(
                i0,
                ref,
                target_mask=target_mask,
                reference_mask=reference_mask,
                mode=cfg.pre_color_match,
                scope=cfg.color_match_scope,
                strength=cfg.color_match_strength,
                background_label=cfg.background_label,
                fallback=cfg.missing_region_fallback,
                macenko_io=cfg.macenko_io,
                macenko_beta=cfg.macenko_beta,
                macenko_alpha=cfg.macenko_alpha,
            )
            if output_root is not None:
                pre_color_match_path = output_root / "pre_gatys_color_matched.png"
                tensor_to_pil(i0).save(pre_color_match_path)

        fallback_mode = str(cfg.missing_region_fallback).strip().lower()
        if fallback_mode not in {"pooled", "skip"}:
            raise ValueError("--missing-region-fallback must be one of: pooled, skip")

        target_regions = tuple(
            _resolve_target_regions(
                target_mask,
                regions,
                background_label=cfg.background_label,
                include_background=cfg.optimize_background,
            )
        )
        reference_regions = tuple(_mask_labels(reference_mask, background_label=cfg.background_label, include_background=False))
        reference_region_set = set(reference_regions)
        shared_regions = tuple(region for region in target_regions if region in reference_region_set)
        missing_reference_regions = tuple(region for region in target_regions if region not in reference_region_set)
        fallback_regions = missing_reference_regions if fallback_mode == "pooled" else ()
        active_regions = tuple(dict.fromkeys((*shared_regions, *fallback_regions)))
        if not active_regions:
            raise ValueError(
                "no stylizable target labels found. Check the target/reference masks, "
                "or pass --optimize-background if label 0 contains tissue."
            )

        print(
            "[masked-gatys] "
            f"target_labels={_format_labels(target_regions)} "
            f"reference_labels={_format_labels(reference_regions)} "
            f"shared={_format_labels(shared_regions)} "
            f"fallback={_format_labels(fallback_regions)}",
            flush=True,
        )

        use_content_loss = bool(cfg.use_content_loss) and float(cfg.content_weight) > 0.0
        with torch.no_grad():
            target_features = self.extractor(i0)
            ref_features = self.extractor(ref)
            content_target = target_features[cfg.content_layer].detach() if use_content_loss else None
            ref_grams = precompute_reference_grams(
                ref_features,
                reference_mask,
                regions=shared_regions,
                layers=cfg.style_layers,
                min_region_pixels=cfg.min_region_pixels,
            )
            if fallback_regions:
                pooled_ref_grams = precompute_pooled_reference_grams(
                    ref_features,
                    reference_mask,
                    layers=cfg.style_layers,
                    background_label=cfg.background_label,
                    min_region_pixels=cfg.min_region_pixels,
                )
                for region in fallback_regions:
                    for layer_name, gram in pooled_ref_grams.items():
                        ref_grams[(layer_name, int(region))] = gram

        x = i0.detach().clone().requires_grad_(True)
        target_bg = (target_mask == int(cfg.background_label)).unsqueeze(1)
        history: list[dict[str, float]] = []

        optimizer_name = cfg.optimizer.lower()
        if optimizer_name == "lbfgs":
            optimizer: torch.optim.Optimizer = torch.optim.LBFGS(
                [x],
                lr=float(cfg.lr),
                max_iter=1,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam([x], lr=float(cfg.adam_lr))
        else:
            raise ValueError("--optimizer must be one of: lbfgs, adam")

        latest_losses: dict[str, torch.Tensor] = {}

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            x.data.clamp_(0.0, 1.0)
            features = self.extractor(x)
            content_loss = (
                F.mse_loss(features[cfg.content_layer].float(), content_target.float())
                if content_target is not None
                else x.new_zeros(())
            )
            style_loss, style_terms = masked_style_loss(
                features,
                ref_grams,
                target_mask,
                style_layers=cfg.style_layers,
                regions=active_regions,
                layer_weights=cfg.layer_weights,
                min_region_pixels=cfg.min_region_pixels,
            )
            tv = total_variation_loss(x) if cfg.tv_weight > 0.0 else x.new_zeros(())
            total = (
                (float(cfg.content_weight) * content_loss if use_content_loss else x.new_zeros(()))
                + float(cfg.style_weight) * style_loss
                + float(cfg.tv_weight) * tv
            )
            total.backward()
            grad = x.grad.detach() if x.grad is not None else None
            latest_losses.clear()
            latest_losses.update(
                {
                    "total": total.detach(),
                    "content": content_loss.detach(),
                    "style": style_loss.detach(),
                    "tv": tv.detach(),
                    "grad_l2": (
                        torch.linalg.vector_norm(grad.float()).detach()
                        if grad is not None
                        else x.new_zeros(())
                    ),
                    "style_terms": torch.tensor(float(style_terms), device=x.device),
                }
            )
            return total

        for step in range(1, int(cfg.steps) + 1):
            if optimizer_name == "lbfgs":
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
                if cfg.preserve_background and not cfg.optimize_background:
                    x[target_bg.expand_as(x)] = i0[target_bg.expand_as(i0)]
            if step == 1 or step == cfg.steps or (cfg.log_every > 0 and step % cfg.log_every == 0):
                row = {
                    "step": float(step),
                    "total": float(latest_losses["total"].item()),
                    "content": float(latest_losses["content"].item()),
                    "style": float(latest_losses["style"].item()),
                    "tv": float(latest_losses["tv"].item()),
                    "grad_l2": float(latest_losses["grad_l2"].item()),
                    "style_terms": float(latest_losses["style_terms"].item()),
                }
                delta = (x.detach() - i0).abs().float()
                row["delta_mean"] = float(delta.mean().item())
                row["delta_max"] = float(delta.max().item())
                history.append(row)
                print(
                    f"[masked-gatys] step={step}/{cfg.steps} "
                    f"total={row['total']:.6g} content={row['content']:.6g} "
                    f"style={row['style']:.6g} terms={int(row['style_terms'])} "
                    f"grad={row['grad_l2']:.6g} delta={row['delta_mean']:.6g}/{row['delta_max']:.6g}",
                    flush=True,
                )
            if output_root is not None and cfg.save_every > 0 and step % cfg.save_every == 0:
                tensor_to_pil(x.detach()).save(output_root / f"step_{step:04d}.png")

        image = tensor_to_pil(x.detach())
        output_path = None
        metrics_path = None
        if output_root is not None:
            output_path = output_root / output_name
            image.save(output_path)
            if cfg.save_mask_debug:
                _save_active_mask_debug(
                    target_mask,
                    output_root / "masked_gatys_active_mask.png",
                    target_regions=target_regions,
                    shared_regions=shared_regions,
                    fallback_regions=fallback_regions,
                    background_label=cfg.background_label,
                )
            metrics_path = output_root / "masked_gatys_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "config": _jsonable_config(cfg),
                        "pre_color_match_path": str(pre_color_match_path) if pre_color_match_path is not None else None,
                        "active_regions": list(active_regions),
                        "target_regions": list(target_regions),
                        "reference_regions": list(reference_regions),
                        "shared_regions": list(shared_regions),
                        "fallback_regions": list(fallback_regions),
                        "missing_reference_regions": list(missing_reference_regions),
                        "history": history,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf8",
            )
        return GatysTransferResult(
            image=image,
            history=history,
            active_regions=active_regions,
            output_path=output_path,
            metrics_path=metrics_path,
            pre_color_match_path=pre_color_match_path,
        )


def run_masked_gatys_transfer(
    *,
    initial_image_path: str | Path,
    reference_image_path: str | Path,
    target_mask_path: str | Path,
    reference_mask_path: str | Path,
    target_nuclei_mask_path: str | Path | None = None,
    reference_nuclei_mask_path: str | Path | None = None,
    output_dir: str | Path,
    regions: Sequence[int] | None = None,
    config: GatysTransferConfig | None = None,
) -> GatysTransferResult:
    cfg = config or GatysTransferConfig()
    device = _resolve_device(cfg.device)
    extractor = build_vgg19_feature_extractor(
        layers=_gatys_feature_layers(cfg),
        weights=cfg.vgg_weights,
        weights_path=cfg.vgg_weights_path,
        device=device,
        dtype=_resolve_torch_dtype(cfg.torch_dtype),
    )
    runner = MaskedGatysStyleTransfer(extractor, cfg)
    target_mask = load_label_mask(target_mask_path)
    reference_mask = load_label_mask(reference_mask_path)
    output_root = Path(output_dir)
    if target_nuclei_mask_path is not None:
        target_mask = overlay_nuclei_on_tissue_mask(
            target_mask,
            load_label_mask(target_nuclei_mask_path),
        )
        save_label_mask(target_mask, output_root / "target_gatys_composite_mask.png")
    if reference_nuclei_mask_path is not None:
        reference_mask = overlay_nuclei_on_tissue_mask(
            reference_mask,
            load_label_mask(reference_nuclei_mask_path),
        )
        save_label_mask(reference_mask, output_root / "reference_gatys_composite_mask.png")
    return runner.run(
        initial_image=load_rgb_tensor(initial_image_path),
        reference_image=load_rgb_tensor(reference_image_path),
        target_mask=target_mask,
        reference_mask=reference_mask,
        regions=regions,
        output_dir=output_dir,
    )


def _gatys_feature_layers(config: GatysTransferConfig) -> tuple[str, ...]:
    layers = list(config.style_layers)
    if bool(config.use_content_loss) and float(config.content_weight) > 0.0:
        layers.insert(0, config.content_layer)
    return tuple(dict.fromkeys(layers))


def build_vgg19_feature_extractor(
    *,
    layers: Sequence[str],
    weights: str = "imagenet",
    weights_path: str | Path | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> VGG19FeatureExtractor:
    try:
        from torchvision.models import VGG19_Weights, vgg19
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Masked Gatys transfer requires torchvision. Install the phase5 environment "
            "or run with a Python env that provides torchvision."
        ) from exc

    mode = str(weights or "imagenet").strip().lower()
    if weights_path:
        model = vgg19(weights=None)
        _load_vgg19_weights(model, weights_path)
    elif mode in {"imagenet", "default", "pretrained"}:
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    elif mode in {"none", "random", "untrained"}:
        model = vgg19(weights=None)
    else:
        raise ValueError("--vgg-weights must be one of: imagenet, none")
    if device.type in {"cpu", "mps"} and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    extractor = VGG19FeatureExtractor(model.features, layers=layers)
    extractor.to(device=device, dtype=dtype)
    extractor.eval()
    extractor.requires_grad_(False)
    return extractor


def masked_style_loss(
    features: Mapping[str, torch.Tensor],
    reference_grams: Mapping[tuple[str, int], torch.Tensor],
    target_mask: torch.Tensor,
    *,
    style_layers: Sequence[str],
    regions: Sequence[int],
    layer_weights: Mapping[str, float],
    min_region_pixels: int,
) -> tuple[torch.Tensor, int]:
    first = next(iter(features.values()))
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for layer_name in style_layers:
        feat = features.get(layer_name)
        if feat is None:
            continue
        layer_weight = float(layer_weights.get(layer_name, 1.0))
        if layer_weight <= 0.0:
            continue
        small_mask = resize_label_mask(target_mask, feat.shape[-2:]).to(device=feat.device)
        for region in regions:
            key = (layer_name, int(region))
            ref_gram = reference_grams.get(key)
            if ref_gram is None:
                continue
            region_mask = small_mask[0] == int(region)
            if int(region_mask.sum().item()) < int(min_region_pixels):
                continue
            gram = region_gram(feat[0], region_mask)
            losses.append(F.mse_loss(gram.float(), ref_gram.to(device=gram.device).float()))
            weights.append(layer_weight)
    if not losses:
        return first.new_zeros(()), 0
    weighted = [loss * weight for loss, weight in zip(losses, weights)]
    return torch.stack(weighted).sum() / max(sum(weights), 1e-12), len(losses)


def precompute_reference_grams(
    reference_features: Mapping[str, torch.Tensor],
    reference_mask: torch.Tensor,
    *,
    regions: Sequence[int],
    layers: Sequence[str],
    min_region_pixels: int,
) -> dict[tuple[str, int], torch.Tensor]:
    grams: dict[tuple[str, int], torch.Tensor] = {}
    for layer_name in layers:
        feat = reference_features[layer_name]
        small_mask = resize_label_mask(reference_mask, feat.shape[-2:]).to(device=feat.device)
        for region in regions:
            region_mask = small_mask[0] == int(region)
            if int(region_mask.sum().item()) < int(min_region_pixels):
                continue
            grams[(layer_name, int(region))] = region_gram(feat[0].detach(), region_mask).detach()
    return grams


def precompute_pooled_reference_grams(
    reference_features: Mapping[str, torch.Tensor],
    reference_mask: torch.Tensor,
    *,
    layers: Sequence[str],
    background_label: int,
    min_region_pixels: int,
) -> dict[str, torch.Tensor]:
    grams: dict[str, torch.Tensor] = {}
    for layer_name in layers:
        feat = reference_features[layer_name]
        small_mask = resize_label_mask(reference_mask, feat.shape[-2:]).to(device=feat.device)
        pooled_mask = small_mask[0] != int(background_label)
        if int(pooled_mask.sum().item()) < int(min_region_pixels):
            pooled_mask = torch.ones_like(pooled_mask, dtype=torch.bool)
        grams[layer_name] = region_gram(feat[0].detach(), pooled_mask).detach()
    return grams


def overlay_nuclei_on_tissue_mask(
    tissue_mask: torch.Tensor,
    nuclei_mask: torch.Tensor,
) -> torch.Tensor:
    tissue = _mask_to_2d(tissue_mask).long().clone()
    nuclei = _mask_to_2d(nuclei_mask).long()
    if tuple(nuclei.shape) != tuple(tissue.shape):
        nuclei = resize_label_mask(nuclei.unsqueeze(0), tuple(tissue.shape))[0]
    nuclei_pixels = nuclei != 0
    tissue[nuclei_pixels] = nuclei[nuclei_pixels] + NUCLEI_STAIN_LABEL_OFFSET
    return tissue


def save_label_mask(mask: torch.Tensor, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = _mask_to_2d(mask).detach().cpu().numpy()
    if array.size == 0 or int(array.max()) <= 255:
        image_array = array.astype(np.uint8, copy=False)
    else:
        image_array = array.astype(np.uint16, copy=False)
    Image.fromarray(image_array).save(path)
    return path


def color_match_tensor(
    image: torch.Tensor,
    reference: torch.Tensor,
    *,
    target_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
    mode: str = "lab",
    scope: str = "region",
    strength: float = 1.0,
    background_label: int = 0,
    fallback: str = "pooled",
    macenko_io: float = 240.0,
    macenko_beta: float = 0.15,
    macenko_alpha: float = 1.0,
) -> torch.Tensor:
    mode = str(mode or "none").strip().lower()
    if mode in {"", "none", "off", "false"}:
        return image
    if mode not in {"lab", "macenko"}:
        raise ValueError("--pre-gatys-color-match must be one of: none, lab, macenko")

    device = image.device
    dtype = image.dtype
    source_array = np.asarray(tensor_to_pil(image).convert("RGB"), dtype=np.uint8)
    reference_array = np.asarray(tensor_to_pil(reference).convert("RGB"), dtype=np.uint8)
    if mode == "lab":
        from controlnet_train.cli.eval_controlnet_flux_cross_v1 import _match_image_color_to_reference

        matched = _match_image_color_to_reference(
            source=Image.fromarray(source_array, mode="RGB"),
            reference=Image.fromarray(reference_array, mode="RGB"),
            method=mode,
        )
        matched_array = np.asarray(matched.convert("RGB"), dtype=np.float32) / 255.0
    else:
        target_mask_np = _mask_numpy_or_none(target_mask)
        reference_mask_np = _mask_numpy_or_none(reference_mask)
        if str(scope).strip().lower() == "region" and target_mask_np is not None and reference_mask_np is not None:
            matched_uint8 = macenko_stain_transfer_by_mask(
                source_array,
                reference_array,
                target_mask_np,
                reference_mask_np,
                background_label=background_label,
                fallback=fallback,
                io=macenko_io,
                beta=macenko_beta,
                alpha=macenko_alpha,
            )
        else:
            source_select = target_mask_np != int(background_label) if target_mask_np is not None else None
            reference_select = reference_mask_np != int(background_label) if reference_mask_np is not None else None
            matched_uint8 = macenko_stain_transfer(
                source_array,
                reference_array,
                source_mask=source_select,
                reference_mask=reference_select,
                io=macenko_io,
                beta=macenko_beta,
                alpha=macenko_alpha,
            )
        matched_array = matched_uint8.astype(np.float32) / 255.0
    matched_tensor = (
        torch.from_numpy(matched_array)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    alpha = float(np.clip(strength, 0.0, 1.0))
    if alpha < 1.0:
        matched_tensor = image * (1.0 - alpha) + matched_tensor * alpha
    return matched_tensor.clamp(0.0, 1.0)


def macenko_stain_transfer_by_mask(
    source: np.ndarray,
    reference: np.ndarray,
    target_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    background_label: int = 0,
    fallback: str = "pooled",
    io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
    min_region_pixels: int = 10,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    reference = np.asarray(reference, dtype=np.uint8)
    output = np.asarray(source, dtype=np.uint8).copy()
    pooled_source = target_mask != int(background_label)
    pooled_reference = reference_mask != int(background_label)
    he_source = estimate_macenko_stain_matrix(
        source,
        mask=pooled_source,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    he_reference = estimate_macenko_stain_matrix(
        reference,
        mask=pooled_reference,
        io=io,
        beta=beta,
        alpha=alpha,
    )
    conc_source = macenko_concentrations(source, he_source, io=io)
    conc_reference = macenko_concentrations(reference, he_reference, io=io)
    target_labels = [int(label) for label in np.unique(target_mask) if int(label) != int(background_label)]
    reference_labels = {int(label) for label in np.unique(reference_mask) if int(label) != int(background_label)}
    fallback_mode = str(fallback or "pooled").strip().lower()
    for label in sorted(target_labels):
        source_region = target_mask == int(label)
        if int(source_region.sum()) < int(min_region_pixels):
            continue
        if label in reference_labels and int((reference_mask == label).sum()) >= int(min_region_pixels):
            reference_region = reference_mask == label
        elif fallback_mode == "pooled" and int(pooled_reference.sum()) >= int(min_region_pixels):
            reference_region = pooled_reference
        else:
            continue
        transferred = macenko_apply_concentration_match(
            source,
            conc_source,
            conc_reference,
            he_reference,
            source_mask=source_region,
            reference_mask=reference_region,
            io=io,
        )
        output[source_region] = transferred[source_region]
    return output


def macenko_apply_concentration_match(
    source: np.ndarray,
    conc_source: np.ndarray,
    conc_reference: np.ndarray,
    reference_stain_matrix: np.ndarray,
    *,
    source_mask: np.ndarray,
    reference_mask: np.ndarray,
    io: float = 240.0,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    h, w, _ = source.shape
    source_select = _valid_bool_mask(source_mask, (h, w))
    if source_select is None:
        return source.copy()
    reference_select = np.asarray(reference_mask, dtype=bool)
    if not np.any(reference_select):
        return source.copy()
    source_flat_mask = source_select.reshape(-1)
    reference_flat_mask = reference_select.reshape(-1)
    if int(source_flat_mask.sum()) < 1 or int(reference_flat_mask.sum()) < 1:
        return source.copy()
    max_source = np.percentile(conc_source[source_flat_mask], 99, axis=0)
    max_reference = np.percentile(conc_reference[reference_flat_mask], 99, axis=0)
    max_source = np.where(max_source < 1e-6, 1e-6, max_source)
    region_conc = conc_source[source_flat_mask] * (max_reference / max_source)[None, :]
    region_rgb = od_to_rgb(region_conc @ reference_stain_matrix, io=io)
    output = source.copy().reshape(-1, 3)
    output[source_flat_mask] = region_rgb
    return output.reshape(h, w, 3)


def macenko_stain_transfer(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    source_mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    reference = np.asarray(reference, dtype=np.uint8)
    h, w, _ = source.shape
    source_select = _valid_bool_mask(source_mask, (h, w))
    reference_select = _valid_bool_mask(reference_mask, reference.shape[:2])
    he_source = estimate_macenko_stain_matrix(source, mask=source_select, io=io, beta=beta, alpha=alpha)
    he_reference = estimate_macenko_stain_matrix(reference, mask=reference_select, io=io, beta=beta, alpha=alpha)
    conc_source = macenko_concentrations(source, he_source, io=io)
    conc_reference = macenko_concentrations(reference, he_reference, io=io)
    source_flat_mask = source_select.reshape(-1) if source_select is not None else np.ones((h * w,), dtype=bool)
    reference_flat_mask = (
        reference_select.reshape(-1)
        if reference_select is not None
        else np.ones((reference.shape[0] * reference.shape[1],), dtype=bool)
    )
    if int(source_flat_mask.sum()) < 1 or int(reference_flat_mask.sum()) < 1:
        return source.copy()
    max_source = np.percentile(conc_source[source_flat_mask], 99, axis=0)
    max_reference = np.percentile(conc_reference[reference_flat_mask], 99, axis=0)
    max_source = np.where(max_source < 1e-6, 1e-6, max_source)
    conc_matched = conc_source * (max_reference / max_source)[None, :]
    od_new = conc_matched @ he_reference
    rgb_new = od_to_rgb(od_new.reshape(h, w, 3), io=io)
    output = source.copy()
    if source_select is None:
        output = rgb_new
    else:
        output[source_select] = rgb_new[source_select]
    return output


def rgb_to_od(image: np.ndarray, io: float = 240.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return -np.log((image + 1.0) / float(io))


def od_to_rgb(od: np.ndarray, io: float = 240.0) -> np.ndarray:
    rgb = float(io) * np.exp(-np.asarray(od, dtype=np.float64))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def estimate_macenko_stain_matrix(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> np.ndarray:
    od = rgb_to_od(image, io=io).reshape(-1, 3)
    if mask is not None:
        od = od[np.asarray(mask, dtype=bool).reshape(-1)]
    od = od[np.all(np.isfinite(od), axis=1)]
    if od.shape[0] < 3:
        return _default_he_matrix()
    od = np.clip(od, 0.0, None)
    stain_strength = np.linalg.norm(od, axis=1)
    od_hat = od[stain_strength > float(beta)]
    if od_hat.shape[0] < 10:
        relaxed_threshold = max(float(beta) * 0.25, 1e-6)
        od_hat = od[stain_strength > relaxed_threshold]
    if od_hat.shape[0] < 3:
        return _default_he_matrix()
    try:
        cov = np.cov(od_hat.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
            return _default_he_matrix()
        order = np.argsort(eigvals)[::-1][:2]
        v = eigvecs[:, order]
        if v[0, 0] < 0:
            v[:, 0] *= -1
        if v[0, 1] < 0:
            v[:, 1] *= -1
        projection = od_hat @ v
        phi = np.arctan2(projection[:, 1], projection[:, 0])
        min_phi = np.percentile(phi, float(alpha))
        max_phi = np.percentile(phi, 100.0 - float(alpha))
        v1 = v @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = v @ np.array([np.cos(max_phi), np.sin(max_phi)])
        he = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
        he = np.clip(he, 1e-6, None)
        norms = np.linalg.norm(he, axis=1, keepdims=True)
        if np.any(norms < 1e-8) or not np.all(np.isfinite(he)):
            return _default_he_matrix()
        return he / norms
    except np.linalg.LinAlgError:
        return _default_he_matrix()


def macenko_concentrations(image: np.ndarray, stain_matrix: np.ndarray, *, io: float = 240.0) -> np.ndarray:
    od = rgb_to_od(image, io=io).reshape(-1, 3)
    concentrations = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)[0].T
    return np.clip(concentrations, 0.0, None)


def _default_he_matrix() -> np.ndarray:
    he = np.array(
        [
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
        ],
        dtype=np.float64,
    )
    return he / np.linalg.norm(he, axis=1, keepdims=True)


def _valid_bool_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    value = np.asarray(mask, dtype=bool)
    if value.shape != tuple(shape):
        return None
    if not np.any(value):
        return None
    return value


def _mask_numpy_or_none(mask: torch.Tensor | None) -> np.ndarray | None:
    if mask is None:
        return None
    return _mask_to_2d(mask).detach().cpu().numpy()


def region_gram(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"features must have shape (C,H,W), got {tuple(features.shape)}")
    selected = features[:, mask.to(device=features.device, dtype=torch.bool)]
    channels = int(features.shape[0])
    pixels = int(selected.shape[1])
    if pixels <= 0:
        return features.new_zeros((channels, channels))
    return (selected @ selected.transpose(0, 1)) / float(max(1, channels * pixels))


def resize_label_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (H,W), (B,H,W), or (B,1,H,W), got {tuple(mask.shape)}")
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask.long()
    return F.interpolate(mask.unsqueeze(1).float(), size=size, mode="nearest")[:, 0].long()


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    return (
        torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
        + torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    )


def load_rgb_tensor(path: str | Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)


def load_label_mask(path: str | Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path)).astype(np.int64, copy=False)).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def parse_region_labels(value: str | None) -> tuple[int, ...] | None:
    if value is None or not str(value).strip():
        return None
    labels = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            labels.append(int(item))
    return tuple(labels)


def _resolve_regions(
    target_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    regions: Sequence[int] | None,
    background_label: int,
) -> list[int]:
    if regions is not None:
        return [int(label) for label in regions if int(label) != int(background_label)]
    target_labels = {int(v) for v in torch.unique(target_mask).detach().cpu().tolist()}
    reference_labels = {int(v) for v in torch.unique(reference_mask).detach().cpu().tolist()}
    return sorted((target_labels & reference_labels) - {int(background_label)})


def _resolve_target_regions(
    target_mask: torch.Tensor,
    regions: Sequence[int] | None,
    *,
    background_label: int,
    include_background: bool,
) -> list[int]:
    target_labels = set(_mask_labels(target_mask, background_label=background_label, include_background=include_background))
    if regions is not None:
        requested = [int(label) for label in regions]
        if not include_background:
            requested = [label for label in requested if label != int(background_label)]
        return [label for label in requested if label in target_labels]
    return sorted(target_labels)


def _mask_labels(mask: torch.Tensor, *, background_label: int, include_background: bool) -> list[int]:
    labels = {int(v) for v in torch.unique(mask).detach().cpu().tolist()}
    if not include_background:
        labels.discard(int(background_label))
    return sorted(labels)


def _mask_to_2d(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask[0]
    if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
        return mask[0, 0]
    raise ValueError(f"mask must be 2D or singleton-batch mask, got {tuple(mask.shape)}")


def _format_labels(labels: Sequence[int], *, max_items: int = 12) -> str:
    values = [int(label) for label in labels]
    if len(values) <= max_items:
        return str(values)
    shown = ", ".join(str(v) for v in values[:max_items])
    return f"[{shown}, ... +{len(values) - max_items}]"


def _save_active_mask_debug(
    target_mask: torch.Tensor,
    output_path: Path,
    *,
    target_regions: Sequence[int],
    shared_regions: Sequence[int],
    fallback_regions: Sequence[int],
    background_label: int,
) -> None:
    mask = target_mask.detach().cpu()
    if mask.ndim == 3:
        mask = mask[0]
    shared = {int(label) for label in shared_regions}
    fallback = {int(label) for label in fallback_regions}
    target = {int(label) for label in target_regions}
    array = np.zeros((*mask.shape[-2:], 3), dtype=np.uint8)
    for label in sorted(target):
        label_mask = mask == int(label)
        if int(label) == int(background_label) and label not in fallback:
            color = (0, 0, 0)
        elif label in shared:
            color = (30, 190, 95)
        elif label in fallback:
            color = (245, 170, 35)
        else:
            color = (220, 45, 45)
        array[label_mask.numpy()] = color
    Image.fromarray(array, mode="RGB").save(output_path)


def _parse_vgg19_layers(layers: Sequence[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for layer in layers:
        value = str(layer).strip().lower()
        if not value:
            continue
        if value in VGG19_RELU_LAYER_ALIASES:
            parsed[value] = VGG19_RELU_LAYER_ALIASES[value]
        else:
            parsed[value] = int(value)
    return parsed


def _prepare_image_tensor(tensor: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"image tensor must have shape (3,H,W) or (B,3,H,W), got {tuple(tensor.shape)}")
    if tensor.shape[0] != 1:
        raise ValueError(f"masked Gatys transfer expects batch size 1, got {int(tensor.shape[0])}")
    return tensor.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _prepare_mask_tensor(tensor: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"mask tensor must have shape (H,W), (B,H,W), or (B,1,H,W), got {tuple(tensor.shape)}")
    if tensor.shape[0] != 1:
        raise ValueError(f"masked Gatys transfer expects batch size 1, got {int(tensor.shape[0])}")
    return tensor.to(device=device, dtype=torch.long)


def _resolve_device(device: str | torch.device) -> torch.device:
    value = str(device)
    if value == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if value == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(value)


def _resolve_torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported torch dtype: {value}")


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in module.parameters():
        return parameter.device, parameter.dtype
    for buffer in module.buffers():
        return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def _load_vgg19_weights(model: nn.Module, weights_path: str | Path) -> None:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
    if not isinstance(state, dict):
        raise ValueError(f"VGG19 weights must be a state dict, got {type(state).__name__}")
    cleaned = {}
    for key, value in state.items():
        key = str(key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("features."):
            cleaned[key] = value
        elif key.startswith("vgg.features."):
            cleaned[key[len("vgg.") :]] = value
    model.load_state_dict(cleaned or state, strict=False)


def _jsonable_config(config: GatysTransferConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["layer_weights"] = dict(config.layer_weights)
    return payload
