"""Glue interfaces for wiring Cross V5-min into a FLUX training loop.

This file intentionally keeps V5 assembly separate from the existing V3/V4
training loop. It provides the pieces that the loop needs to call:

* frozen predictor bridge that preserves gradients to generated RGB;
* weighted four-family loss assembly;
* explicit AdaLN hook specs for FLUX DiT integration;
* pairing-policy defaults for coverage and appearance-gap sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Protocol, Sequence

import torch
import torch.nn as nn

from controlnet_train.training.cross_v1_losses import unpack_flux_packed_latents
from controlnet_train.training.cross_v5_losses import (
    CrossV5AppearanceLossConfig,
    CrossV5GeometryConsistencyLossConfig,
    cross_v5_appearance_fidelity_loss,
    cross_v5_geometry_consistency_loss,
)
from controlnet_train.modules.cross_v5_conditioning import CrossV5AdaLNOutput, CrossV5TissueBank


class CrossV5AppearanceModulator(Protocol):
    """Structural interface for V5 appearance modulation.

    The V5-min default is ``CrossV5SpatialAdaLNModulator``; the simpler
    per-class ``CrossV5AdaLNModulator`` remains a fallback. Both expose the
    same call shape and keep appearance on the normalization scale/shift path.
    """

    hidden_dim: int
    mlp: nn.Module

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        target_class_ids: torch.Tensor,
        bank: CrossV5TissueBank,
        fallback_prototypes: torch.Tensor | None = None,
        target_structure_tokens: torch.Tensor | None = None,
    ) -> CrossV5AdaLNOutput:
        ...


class V5TextureFeatureExtractor(Protocol):
    """Callable that returns shallow texture features for generated/reference RGB."""

    def __call__(self, images: torch.Tensor) -> Mapping[str, torch.Tensor] | Sequence[torch.Tensor]:
        ...


class V5DenseGeometryPredictor(Protocol):
    """Callable frozen predictor used for differentiable geometry consistency."""

    def __call__(self, images: torch.Tensor) -> Mapping[str, torch.Tensor]:
        ...


@dataclass(frozen=True)
class CrossV5LatentDecodeConfig:
    """Options for differentiable latent-to-RGB reconstruction."""

    prediction_type: str = "velocity"
    packed_latents: bool = False
    latent_channels: int | None = None
    latent_height: int | None = None
    latent_width: int | None = None
    vae_dtype: torch.dtype | None = None
    clamp_rgb: bool = False
    require_grad: bool = True


@dataclass(frozen=True)
class CrossV5LossWeights:
    """Default V5-min loss weights.

    The defaults are intentionally conservative: denoise remains the anchor,
    appearance and geometry start modestly until the decoded-image branches are
    visually verified. Color/texture proportions live in
    ``CrossV5AppearanceLossConfig``; this class only owns family-level weights.
    """

    denoise: float = 1.0
    appearance: float = 0.75
    geometry: float = 0.25
    swap_sensitivity: float = 0.0


@dataclass(frozen=True)
class CrossV5LossIntervals:
    """How often expensive decoded-image losses run.

    FLUX timesteps conventionally decrease as denoising proceeds, so the
    default geometry cutoff treats smaller timestep values as lower-noise
    steps. Set ``geometry_timestep_max=None`` if a scheduler uses a different
    convention and the caller wants pure interval gating.
    """

    appearance: int = 1
    geometry: int = 4
    smoke: int = 500
    appearance_timestep_min: float | None = None
    appearance_timestep_max: float | None = None
    geometry_timestep_min: float | None = None
    geometry_timestep_max: float | None = 350.0


@dataclass(frozen=True)
class CrossV5PairingPolicy:
    """Sampling targets for V5 pair construction."""

    same_wsi_fraction: float = 0.35
    cross_wsi_fraction: float = 0.45
    high_appearance_gap_fraction: float = 0.20
    full_coverage_fraction: float = 0.55
    partial_coverage_fraction: float = 0.35
    low_coverage_fraction: float = 0.10
    class_bank_dropout_prob: float = 0.15
    min_ref_presence_tokens: float = 1.0
    min_bank_token_confidence: float = 0.5

    def normalized_pair_mode_weights(self) -> dict[str, float]:
        return _normalize_weights(
            {
                "same_wsi": self.same_wsi_fraction,
                "cross_wsi": self.cross_wsi_fraction,
                "high_appearance_gap": self.high_appearance_gap_fraction,
            }
        )

    def normalized_coverage_weights(self) -> dict[str, float]:
        return _normalize_weights(
            {
                "full": self.full_coverage_fraction,
                "partial": self.partial_coverage_fraction,
                "low": self.low_coverage_fraction,
            }
        )


@dataclass(frozen=True)
class CrossV5AdaLNHookSpec:
    """Declarative spec for inserting appearance modulation into FLUX blocks."""

    block_indices: tuple[int, ...] = (-1,)
    hook_point: str = "post_norm_hidden"
    detach_bank: bool = False
    require_nonzero_gamma: bool = True


@dataclass(frozen=True)
class CrossV5AdaLNInstallSummary:
    """Result of installing V5 appearance modulation hooks."""

    installed_block_indices: tuple[int, ...]
    hook_point: str


@dataclass
class CrossV5StepContext:
    """Decoded tensors and masks needed by V5 losses for one train step."""

    prediction_rgb: torch.Tensor
    reference_rgb: torch.Tensor
    target_tissue_mask: torch.Tensor
    reference_tissue_mask: torch.Tensor
    target_nuclei_mask: torch.Tensor | None = None
    target_nuclei_binary: torch.Tensor | None = None
    target_dense_geometry: Mapping[str, torch.Tensor] | None = None


@dataclass
class CrossV5LossBundle:
    """Weighted total plus scalar components."""

    total: torch.Tensor
    components: dict[str, torch.Tensor | int]


class CrossV5AdaLNAdapterMixin:
    """Mixin for V5-ready FLUX block wrappers.

    Subclasses call ``_apply_cross_v5_adaln(...)`` at the actual post-norm
    hidden-state point in their forward method. This is a reference integration
    surface; it does not assume diffusers private field names.
    """

    def set_cross_v5_adaln_modulator(
        self,
        modulator: CrossV5AppearanceModulator,
        *,
        hook_point: str,
        detach_bank: bool,
    ) -> None:
        self.cross_v5_adaln_modulator = modulator
        self.cross_v5_hook_point = str(hook_point)
        self.cross_v5_detach_bank = bool(detach_bank)

    def _apply_cross_v5_adaln(
        self,
        hidden_states: torch.Tensor,
        *,
        target_class_ids: torch.Tensor,
        bank: CrossV5TissueBank,
        fallback_prototypes: torch.Tensor | None = None,
        target_structure_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulator = getattr(self, "cross_v5_adaln_modulator", None)
        if modulator is None:
            return hidden_states
        return apply_cross_v5_adaln_to_hidden(
            hidden_states=hidden_states,
            target_class_ids=target_class_ids,
            bank=bank,
            modulator=modulator,
            fallback_prototypes=fallback_prototypes,
            target_structure_tokens=target_structure_tokens,
            detach_bank=bool(getattr(self, "cross_v5_detach_bank", False)),
        )


def freeze_predictor_for_v5_loss(module: nn.Module) -> nn.Module:
    """Freeze predictor parameters while keeping forward differentiable.

    Do not wrap calls to the returned module in ``torch.no_grad()`` during
    generator training. Gradients must flow from predictor outputs back to
    ``prediction_rgb``; only predictor parameters are frozen.
    """

    module.eval()
    module.requires_grad_(False)
    return module


def install_cross_v5_adaln_hooks(
    *,
    transformer: nn.Module,
    modulator: CrossV5AppearanceModulator,
    spec: CrossV5AdaLNHookSpec,
    block_attr: str = "transformer_blocks",
    setter_name: str = "set_cross_v5_adaln_modulator",
) -> CrossV5AdaLNInstallSummary:
    """Install AdaLN modulation on explicit V5-ready FLUX block wrappers.

    Real diffusers FLUX block internals vary by version, so this installer does
    not guess private attributes. A block must expose ``setter_name`` and accept
    ``(modulator, hook_point=..., detach_bank=...)``. This keeps the integration
    honest: if a block is not V5-ready, installation fails loudly instead of
    silently becoming a no-op.
    """

    if spec.require_nonzero_gamma:
        _validate_nonzero_adaln_gamma(modulator)

    blocks = list(getattr(transformer, block_attr, []) or [])
    selected = _normalize_block_indices(spec.block_indices, len(blocks))
    installed: list[int] = []
    for index in selected:
        block = blocks[index]
        setter = getattr(block, setter_name, None)
        if setter is None:
            raise AttributeError(
                f"Block {index} does not expose {setter_name}(...). Wrap FLUX blocks with a V5 AdaLN adapter first."
            )
        setter(modulator, hook_point=spec.hook_point, detach_bank=spec.detach_bank)
        installed.append(index)
    return CrossV5AdaLNInstallSummary(
        installed_block_indices=tuple(installed),
        hook_point=spec.hook_point,
    )


def apply_cross_v5_adaln_to_hidden(
    *,
    hidden_states: torch.Tensor,
    target_class_ids: torch.Tensor,
    bank: CrossV5TissueBank,
    modulator: CrossV5AppearanceModulator,
    fallback_prototypes: torch.Tensor | None = None,
    target_structure_tokens: torch.Tensor | None = None,
    detach_bank: bool = False,
) -> torch.Tensor:
    """Small adapter called by a V5-ready DiT block at its AdaLN hook point."""

    if detach_bank:
        bank = CrossV5TissueBank(
            prototypes=bank.prototypes.detach(),
            local_tokens=bank.local_tokens.detach(),
            class_present=bank.class_present,
            class_mass=bank.class_mass.detach(),
            token_class_ids=bank.token_class_ids,
            token_class_confidence=bank.token_class_confidence.detach(),
        )
        if fallback_prototypes is not None:
            fallback_prototypes = fallback_prototypes.detach()
    output = modulator(
        hidden_states=hidden_states,
        target_class_ids=target_class_ids,
        bank=bank,
        fallback_prototypes=fallback_prototypes,
        target_structure_tokens=target_structure_tokens,
    )
    return output.hidden_states


def run_frozen_predictor_bridge(
    *,
    predictor: Callable[[torch.Tensor], Mapping[str, torch.Tensor]],
    prediction_rgb: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Run a frozen dense predictor without cutting gradients to RGB."""

    if not prediction_rgb.requires_grad:
        raise ValueError(
            "prediction_rgb does not require grad. Decode/bridge path is detached; "
            "do not call the VAE decode or predictor under torch.no_grad()."
        )
    outputs = predictor(prediction_rgb)
    if not isinstance(outputs, Mapping):
        raise TypeError("V5 geometry predictor must return a mapping of dense tensors.")
    return outputs


def reconstruct_cross_v5_x0_latents(
    *,
    noisy_latents: torch.Tensor,
    model_prediction: torch.Tensor,
    sigma: torch.Tensor | float,
    alpha: torch.Tensor | float | None = None,
    prediction_type: str = "velocity",
) -> torch.Tensor:
    """Reconstruct x0 latents without detaching the generator path."""

    if noisy_latents.shape != model_prediction.shape:
        raise ValueError(
            "noisy_latents and model_prediction shapes must match, got "
            f"{tuple(noisy_latents.shape)} vs {tuple(model_prediction.shape)}."
        )
    sigma_t = _broadcast_schedule_value(sigma, noisy_latents)
    mode = str(prediction_type).strip().lower()
    if mode in {"velocity", "flow", "v", "v_prediction"}:
        return noisy_latents - sigma_t * model_prediction
    if mode in {"epsilon", "eps", "noise"}:
        alpha_t = _broadcast_schedule_value(1.0 - sigma_t if alpha is None else alpha, noisy_latents)
        return (noisy_latents - sigma_t * model_prediction) / alpha_t.clamp_min(1e-6)
    raise ValueError(f"Unsupported Cross V5 prediction_type {prediction_type!r}.")


def decode_cross_v5_prediction_rgb(
    *,
    vae: nn.Module,
    noisy_latents: torch.Tensor,
    model_prediction: torch.Tensor,
    sigma: torch.Tensor | float,
    alpha: torch.Tensor | float | None = None,
    config: CrossV5LatentDecodeConfig | None = None,
    unpack_latents: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Differentiably reconstruct predicted RGB for V5 appearance/geometry losses.

    The caller may freeze VAE parameters, but must not wrap this function in
    ``torch.no_grad()``. Gradients from RGB losses need to flow through decode
    to ``model_prediction`` and then to the trainable generator branch.
    """

    cfg = config or CrossV5LatentDecodeConfig()
    x0_latents = reconstruct_cross_v5_x0_latents(
        noisy_latents=noisy_latents,
        model_prediction=model_prediction,
        sigma=sigma,
        alpha=alpha,
        prediction_type=cfg.prediction_type,
    )
    if cfg.packed_latents:
        if unpack_latents is not None:
            x0_latents = unpack_latents(x0_latents)
        else:
            if cfg.latent_channels is None or cfg.latent_height is None or cfg.latent_width is None:
                raise ValueError(
                    "latent_channels, latent_height, and latent_width are required when packed_latents=True "
                    "and no unpack_latents callback is provided."
                )
            x0_latents = unpack_flux_packed_latents(
                x0_latents,
                channels=int(cfg.latent_channels),
                height=int(cfg.latent_height),
                width=int(cfg.latent_width),
            )

    vae_config = getattr(vae, "config", object())
    scaling_factor = float(getattr(vae_config, "scaling_factor", 1.0))
    shift_factor = float(getattr(vae_config, "shift_factor", 0.0))
    decode_latents = (x0_latents / scaling_factor) + shift_factor
    if cfg.vae_dtype is not None:
        decode_latents = decode_latents.to(dtype=cfg.vae_dtype)
    decoded = vae.decode(decode_latents, return_dict=False)[0]
    rgb = (decoded.float() + 1.0) / 2.0
    if cfg.clamp_rgb:
        rgb = rgb.clamp(0.0, 1.0)
    if cfg.require_grad and (noisy_latents.requires_grad or model_prediction.requires_grad) and not rgb.requires_grad:
        raise ValueError(
            "Decoded RGB is detached. Do not call the Cross V5 latent decode bridge under torch.no_grad(), "
            "and do not detach x0 latents before VAE decode."
        )
    return rgb


def assemble_cross_v5_losses(
    *,
    denoise_loss: torch.Tensor,
    context: CrossV5StepContext,
    weights: CrossV5LossWeights,
    appearance_config: CrossV5AppearanceLossConfig | None = None,
    texture_feature_extractor: V5TextureFeatureExtractor | None = None,
    geometry_config: CrossV5GeometryConsistencyLossConfig | None = None,
    geometry_predictions: Mapping[str, torch.Tensor] | None = None,
    swap_sensitivity_loss: torch.Tensor | None = None,
) -> CrossV5LossBundle:
    """Assemble denoise + color + texture + geometry into one train loss."""

    total = float(weights.denoise) * denoise_loss
    components: dict[str, torch.Tensor | int] = {
        "denoise": denoise_loss,
    }

    if weights.appearance > 0.0:
        pred_features = None
        ref_features = None
        if texture_feature_extractor is not None:
            pred_features = texture_feature_extractor(context.prediction_rgb)
            ref_features = texture_feature_extractor(context.reference_rgb.detach())

        appearance_cfg = appearance_config or CrossV5AppearanceLossConfig()
        appearance = cross_v5_appearance_fidelity_loss(
            prediction=context.prediction_rgb,
            reference=context.reference_rgb,
            target_tissue_mask=context.target_tissue_mask,
            reference_tissue_mask=context.reference_tissue_mask,
            prediction_vgg_features=pred_features,
            reference_vgg_features=ref_features,
            config=appearance_cfg,
        )
        appearance_total = appearance["total"]
        if isinstance(appearance_total, torch.Tensor):
            total = total + float(weights.appearance) * appearance_total
        components.update({f"appearance_{key}": value for key, value in appearance.items()})

    if geometry_predictions is not None and weights.geometry > 0.0:
        geometry = cross_v5_geometry_consistency_loss(
            tissue_logits=geometry_predictions.get("tissue_logits"),
            target_tissue_mask=context.target_tissue_mask,
            nuclei_logits=geometry_predictions.get("nuclei_logits"),
            target_nuclei_mask=context.target_nuclei_mask,
            nuclei_binary_logits=geometry_predictions.get("nuclei_binary_logits"),
            target_nuclei_binary=context.target_nuclei_binary,
            dense_predictions=_extract_dense_geometry_predictions(geometry_predictions),
            dense_targets=context.target_dense_geometry,
            config=geometry_config,
        )
        geometry_total = geometry["total"]
        if isinstance(geometry_total, torch.Tensor):
            total = total + float(weights.geometry) * geometry_total
        components.update({f"geometry_{key}": value for key, value in geometry.items()})

    if swap_sensitivity_loss is not None and weights.swap_sensitivity > 0.0:
        total = total + float(weights.swap_sensitivity) * swap_sensitivity_loss
        components["swap_sensitivity"] = swap_sensitivity_loss

    return CrossV5LossBundle(total=total, components=components)


def assemble_cross_v5_step_losses(
    *,
    denoise_loss: torch.Tensor,
    context: CrossV5StepContext,
    weights: CrossV5LossWeights,
    global_step: int,
    timestep: torch.Tensor | None,
    intervals: CrossV5LossIntervals | None = None,
    appearance_config: CrossV5AppearanceLossConfig | None = None,
    texture_feature_extractor: V5TextureFeatureExtractor | None = None,
    geometry_config: CrossV5GeometryConsistencyLossConfig | None = None,
    geometry_predictor: V5DenseGeometryPredictor | None = None,
    geometry_predictions: Mapping[str, torch.Tensor] | None = None,
    swap_sensitivity_loss: torch.Tensor | None = None,
) -> CrossV5LossBundle:
    """Gate expensive V5 branches for one training step, then assemble losses.

    This wrapper is the intended FLUX train-loop entry point. It uses
    ``CrossV5LossIntervals`` to keep appearance/geometry scheduling outside the
    lower-level loss combiner. Geometry predictor execution happens only when
    both the step interval and low-noise timestep cutoff allow it.
    """

    schedule = intervals or CrossV5LossIntervals()
    run_appearance = bool(
        weights.appearance > 0.0
        and should_run_cross_v5_branch(
            global_step=global_step,
            interval=schedule.appearance,
            timestep=timestep,
            timestep_min=schedule.appearance_timestep_min,
            timestep_max=schedule.appearance_timestep_max,
        )
    )
    geometry_source_available = geometry_predictions is not None or geometry_predictor is not None
    run_geometry = bool(
        weights.geometry > 0.0
        and geometry_source_available
        and should_run_cross_v5_branch(
            global_step=global_step,
            interval=schedule.geometry,
            timestep=timestep,
            timestep_min=schedule.geometry_timestep_min,
            timestep_max=schedule.geometry_timestep_max,
        )
    )

    effective_weights = replace(
        weights,
        appearance=weights.appearance if run_appearance else 0.0,
        geometry=weights.geometry if run_geometry else 0.0,
    )
    effective_geometry_predictions = geometry_predictions
    if run_geometry and effective_geometry_predictions is None:
        assert geometry_predictor is not None
        effective_geometry_predictions = run_frozen_predictor_bridge(
            predictor=geometry_predictor,
            prediction_rgb=context.prediction_rgb,
        )
    if not run_geometry:
        effective_geometry_predictions = None

    bundle = assemble_cross_v5_losses(
        denoise_loss=denoise_loss,
        context=context,
        weights=effective_weights,
        appearance_config=appearance_config,
        texture_feature_extractor=texture_feature_extractor if run_appearance else None,
        geometry_config=geometry_config,
        geometry_predictions=effective_geometry_predictions,
        swap_sensitivity_loss=swap_sensitivity_loss,
    )
    bundle.components.update(
        {
            "gate_appearance": int(run_appearance),
            "gate_geometry": int(run_geometry),
            "global_step": int(global_step),
        }
    )
    if timestep is not None:
        bundle.components["timestep_mean"] = timestep.detach().float().mean()
    return bundle


def validate_cross_v5_predictor_grad_bridge(
    *,
    predictor: Callable[[torch.Tensor], Mapping[str, torch.Tensor]],
    prediction_rgb: torch.Tensor,
) -> dict[str, float]:
    """Small differentiability smoke check for generated RGB -> predictor."""

    if not prediction_rgb.requires_grad:
        prediction_rgb = prediction_rgb.detach().clone().requires_grad_(True)
    outputs = run_frozen_predictor_bridge(predictor=predictor, prediction_rgb=prediction_rgb)
    tensors = [value.float().mean() for value in outputs.values() if isinstance(value, torch.Tensor)]
    if not tensors:
        raise ValueError("Predictor returned no tensor outputs.")
    probe = torch.stack(tensors).sum()
    grad = torch.autograd.grad(probe, prediction_rgb, retain_graph=True, allow_unused=False)[0]
    return {
        "probe": float(probe.detach().cpu().item()),
        "rgb_grad_abs_mean": float(grad.detach().float().abs().mean().cpu().item()),
        "rgb_grad_abs_max": float(grad.detach().float().abs().max().cpu().item()),
    }


def _extract_dense_geometry_predictions(predictions: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
    reserved = {"tissue_logits", "nuclei_logits", "nuclei_binary_logits"}
    dense = {key: value for key, value in predictions.items() if key not in reserved}
    return dense or None


def _broadcast_schedule_value(value: torch.Tensor | float, target: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=target.device, dtype=target.dtype)
    while tensor.ndim < target.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


def _normalize_weights(values: Mapping[str, float]) -> dict[str, float]:
    clipped = {key: max(0.0, float(value)) for key, value in values.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        raise ValueError(f"At least one sampling weight must be positive, got {values}.")
    return {key: value / total for key, value in clipped.items()}


def _normalize_block_indices(indices: Sequence[int], total_blocks: int) -> tuple[int, ...]:
    if total_blocks <= 0:
        raise ValueError("No transformer blocks are available for V5 AdaLN hook installation.")
    normalized: list[int] = []
    for raw in indices:
        index = int(raw)
        if index < 0:
            index = total_blocks + index
        if index < 0 or index >= total_blocks:
            raise ValueError(f"Block index {raw} resolves to {index}, outside [0, {total_blocks}).")
        if index not in normalized:
            normalized.append(index)
    return tuple(normalized)


def _validate_nonzero_adaln_gamma(modulator: CrossV5AppearanceModulator) -> None:
    """Catch accidental zero-init of the V5 appearance modulation path."""

    hidden_dim = int(getattr(modulator, "hidden_dim"))
    mlp = getattr(modulator, "mlp")
    final = mlp[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("Cross V5 AdaLN modulator must end with a Linear layer.")
    gamma_weight = final.weight[:hidden_dim]
    gamma_bias = final.bias[:hidden_dim] if final.bias is not None else None
    has_nonzero_weight = bool((gamma_weight.detach().abs() > 0).any().item())
    has_nonzero_bias = bool(gamma_bias is not None and (gamma_bias.detach().abs() > 0).any().item())
    if not has_nonzero_weight and not has_nonzero_bias:
        raise ValueError(
            "Cross V5 appearance gamma path is zero-initialized. "
            "Use normal initialization for AdaLN modulation so reference appearance is active from step 1."
        )


def should_run_cross_v5_branch(
    *,
    global_step: int,
    interval: int,
    timestep: torch.Tensor | None = None,
    timestep_min: float | None = None,
    timestep_max: float | None = None,
) -> bool:
    """Step/timestep gate for expensive decoded-image V5 branches."""

    if int(interval) <= 0:
        return False
    if int(global_step) % int(interval) != 0:
        return False
    if timestep is None or (timestep_min is None and timestep_max is None):
        return True
    value = float(timestep.detach().float().mean().cpu().item())
    if timestep_min is not None and value < float(timestep_min):
        return False
    if timestep_max is not None and value > float(timestep_max):
        return False
    return True


__all__ = [
    "CrossV5AdaLNHookSpec",
    "CrossV5AdaLNAdapterMixin",
    "CrossV5AdaLNInstallSummary",
    "CrossV5AppearanceModulator",
    "CrossV5LatentDecodeConfig",
    "CrossV5LossBundle",
    "CrossV5LossIntervals",
    "CrossV5LossWeights",
    "CrossV5PairingPolicy",
    "CrossV5StepContext",
    "apply_cross_v5_adaln_to_hidden",
    "assemble_cross_v5_losses",
    "assemble_cross_v5_step_losses",
    "decode_cross_v5_prediction_rgb",
    "freeze_predictor_for_v5_loss",
    "install_cross_v5_adaln_hooks",
    "run_frozen_predictor_bridge",
    "reconstruct_cross_v5_x0_latents",
    "should_run_cross_v5_branch",
    "validate_cross_v5_predictor_grad_bridge",
]
