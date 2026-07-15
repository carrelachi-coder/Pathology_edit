"""Inference-only orientation gates derived from target/reference tissue masks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .losses import gaussian_blur_image
from .orientation_supervision import _grayscale, _single_channel_mask, _structure_tensor, _tissue_weight


@dataclass(frozen=True)
class PairConditionedRot90Gate:
    gate: torch.Tensor
    supported_pairs: list[int]
    mean_confidence: list[float]


@dataclass(frozen=True)
class FineTextureSteeringWeights:
    weights: torch.Tensor
    mean_confidence: torch.Tensor
    raw_mean_confidence: torch.Tensor
    active_fraction: torch.Tensor
    fallback_fraction: torch.Tensor
    candidate_fractions: torch.Tensor
    mean_selected_angle_degrees: torch.Tensor


def build_fine_texture_steering_weights(
    i0: torch.Tensor,
    reference: torch.Tensor,
    *,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor | None,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor | None,
    candidate_angles_degrees: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
    smoothing_sigma: float = 8.0,
    min_coherence: float = 0.20,
    min_relative_energy: float = 0.50,
    min_resultant: float = 0.15,
    minimum_strength: float = 0.0,
    minimum_support: float = 0.05,
    temperature: float = 0.08,
    reference_direction_mode: str = "global_mean",
    local_histogram_bins: int = 36,
    local_histogram_concentration: float = 8.0,
    boundary_exclusion_radius: int = 3,
    nuclei_exclusion_radius: int = 5,
) -> FineTextureSteeringWeights:
    """Choose a rotated reference feature bank from local I0 texture direction.

    I0 provides a smoothed local doubled-angle field.  The reference supplies a
    direction evidence per tissue label.  ``global_mean`` preserves the legacy
    single-vector behavior.  ``local_histogram`` retains a multi-modal histogram
    of local reference directions so opposing local textures do not cancel to
    zero before candidate rotations are scored.  Each target location then
    selects the closest candidate rotation while nuclei neighborhoods and
    low-confidence regions fall back to the unrotated reference.
    """

    if i0.shape != reference.shape or i0.ndim != 4 or i0.shape[1] != 3:
        raise ValueError("i0 and reference must have identical shape [B,3,H,W]")
    angles = tuple(float(value) % 180.0 for value in candidate_angles_degrees)
    if not angles or abs(angles[0]) > 1.0e-6:
        raise ValueError("candidate_angles_degrees must start with 0 for the fallback path")
    if len(set(round(value, 6) for value in angles)) != len(angles):
        raise ValueError("candidate_angles_degrees must be unique modulo 180 degrees")
    coherence_floor = min(max(float(min_coherence), 0.0), 0.999)
    relative_energy_floor = max(0.0, float(min_relative_energy))
    resultant_floor = min(max(float(min_resultant), 0.0), 0.999)
    strength_floor = min(max(float(minimum_strength), 0.0), 1.0)
    support_floor = min(max(float(minimum_support), 0.0), 1.0)
    temperature = max(float(temperature), 1.0e-3)
    smoothing_sigma = max(float(smoothing_sigma), 0.0)
    direction_mode = str(reference_direction_mode).strip().lower()
    if direction_mode not in {"global_mean", "local_histogram"}:
        raise ValueError(
            "reference_direction_mode must be 'global_mean' or 'local_histogram'"
        )
    histogram_bins = int(local_histogram_bins)
    if histogram_bins < 8:
        raise ValueError("local_histogram_bins must be at least 8")
    histogram_concentration = max(float(local_histogram_concentration), 1.0e-3)
    spatial_size = tuple(i0.shape[-2:])

    target_labels = _single_channel_mask(
        target_tissue_mask,
        name="target_tissue_mask",
        spatial_size=spatial_size,
    ).long()
    reference_labels = _single_channel_mask(
        reference_tissue_mask,
        name="reference_tissue_mask",
        spatial_size=spatial_size,
    ).long()
    target_base = _tissue_weight(
        target_labels,
        target_nuclei_mask,
        None,
        spatial_size=spatial_size,
        boundary_exclusion_radius=boundary_exclusion_radius,
        nuclei_exclusion_radius=nuclei_exclusion_radius,
        min_trust=0.0,
    ).to(device=i0.device, dtype=torch.float32)
    reference_base = _tissue_weight(
        reference_labels,
        reference_nuclei_mask,
        None,
        spatial_size=spatial_size,
        boundary_exclusion_radius=boundary_exclusion_radius,
        nuclei_exclusion_radius=nuclei_exclusion_radius,
        min_trust=0.0,
    ).to(device=i0.device, dtype=torch.float32)

    with torch.no_grad():
        target_vector, target_energy, target_coherence = _structure_tensor(
            _grayscale(i0.detach()),
            inner_sigma=0.75,
            outer_sigma=2.0,
            integration_sigma=3.0,
        )
        reference_vector, reference_energy, reference_coherence = _structure_tensor(
            _grayscale(reference.detach()),
            inner_sigma=0.75,
            outer_sigma=2.0,
            integration_sigma=3.0,
        )

        def directional_weight(
            energy: torch.Tensor,
            coherence: torch.Tensor,
            base: torch.Tensor,
        ) -> torch.Tensor:
            image_energy = (
                (energy * base).sum(dim=(2, 3), keepdim=True)
                / base.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            ).clamp_min(1.0e-8)
            relative = (energy / image_energy).clamp(0.0, 3.0)
            coherence_weight = (
                (coherence - coherence_floor) / (1.0 - coherence_floor)
            ).clamp(0.0, 1.0)
            valid = coherence.ge(coherence_floor) & relative.ge(relative_energy_floor)
            return base * valid.to(base.dtype) * coherence_weight * relative

        target_weight = directional_weight(target_energy, target_coherence, target_base)
        reference_weight = directional_weight(
            reference_energy,
            reference_coherence,
            reference_base,
        )
        target_denominator = gaussian_blur_image(target_weight, smoothing_sigma).clamp_min(1.0e-8)
        target_mean = (
            gaussian_blur_image(target_vector * target_weight, smoothing_sigma)
            / target_denominator
        )
        target_resultant = torch.linalg.vector_norm(target_mean, dim=1, keepdim=True)
        target_unit = target_mean / target_resultant.clamp_min(1.0e-8)
        target_support = (
            gaussian_blur_image(target_weight, smoothing_sigma)
            / gaussian_blur_image(target_base, smoothing_sigma).clamp_min(1.0e-8)
        ).clamp(0.0, 1.0)

        if direction_mode == "global_mean":
            reference_unit_map = torch.zeros_like(target_unit)
            reference_evidence_map = torch.zeros_like(target_resultant)
            for batch_index in range(i0.shape[0]):
                global_weight = reference_weight[batch_index : batch_index + 1]
                global_mean = (
                    (reference_vector[batch_index : batch_index + 1] * global_weight).sum(
                        dim=(2, 3), keepdim=True
                    )
                    / global_weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-8)
                )
                global_resultant = torch.linalg.vector_norm(global_mean, dim=1, keepdim=True)
                global_unit = global_mean / global_resultant.clamp_min(1.0e-8)
                labels = torch.unique(target_labels[batch_index]).tolist()
                for label in labels:
                    label = int(label)
                    if label == 0:
                        continue
                    target_region = target_labels[batch_index : batch_index + 1].eq(label)
                    reference_region = reference_labels[batch_index : batch_index + 1].eq(label)
                    label_weight = global_weight * reference_region.to(global_weight.dtype)
                    if float(label_weight.sum().item()) > 1.0e-6:
                        label_mean = (
                            (reference_vector[batch_index : batch_index + 1] * label_weight).sum(
                                dim=(2, 3), keepdim=True
                            )
                            / label_weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-8)
                        )
                        label_resultant = torch.linalg.vector_norm(
                            label_mean,
                            dim=1,
                            keepdim=True,
                        )
                        label_unit = label_mean / label_resultant.clamp_min(1.0e-8)
                    else:
                        label_unit = global_unit
                        label_resultant = global_resultant
                    reference_unit_map[batch_index : batch_index + 1] = torch.where(
                        target_region.expand(-1, 2, -1, -1),
                        label_unit.expand(-1, -1, *spatial_size),
                        reference_unit_map[batch_index : batch_index + 1],
                    )
                    reference_evidence_map[batch_index : batch_index + 1] = torch.where(
                        target_region,
                        label_resultant.expand(-1, -1, *spatial_size),
                        reference_evidence_map[batch_index : batch_index + 1],
                    )

            candidate_scores = []
            for angle in angles:
                radians = math.radians(2.0 * angle)
                cosine = math.cos(radians)
                sine = math.sin(radians)
                rotated_reference = torch.stack(
                    (
                        cosine * reference_unit_map[:, 0] + sine * reference_unit_map[:, 1],
                        -sine * reference_unit_map[:, 0] + cosine * reference_unit_map[:, 1],
                    ),
                    dim=1,
                )
                candidate_scores.append(
                    (target_unit * rotated_reference).sum(dim=1, keepdim=True)
                )
            scores = torch.cat(candidate_scores, dim=1)
            reference_confidence = (
                (reference_evidence_map - resultant_floor) / (1.0 - resultant_floor)
            ).clamp(0.0, 1.0)
            reference_supported = reference_evidence_map.ge(resultant_floor)
        else:
            # Preserve the full local orientation distribution within each
            # tissue label.  A log-sum-exp match against histogram modes avoids
            # the cancellation caused by one global doubled-angle vector.
            scores = target_unit.new_zeros(
                (i0.shape[0], len(angles), *spatial_size),
                dtype=torch.float32,
            )
            reference_evidence_map = target_resultant.new_zeros(target_resultant.shape)
            bin_radians = (
                torch.arange(histogram_bins, device=i0.device, dtype=torch.float32)
                + 0.5
            ) * (2.0 * math.pi / float(histogram_bins))
            bin_vectors = torch.stack((torch.cos(bin_radians), torch.sin(bin_radians)), dim=1)
            rotated_bin_vectors = []
            for angle in angles:
                radians = math.radians(2.0 * angle)
                cosine = math.cos(radians)
                sine = math.sin(radians)
                rotated_bin_vectors.append(
                    torch.stack(
                        (
                            cosine * bin_vectors[:, 0] + sine * bin_vectors[:, 1],
                            -sine * bin_vectors[:, 0] + cosine * bin_vectors[:, 1],
                        ),
                        dim=1,
                    )
                )
            rotated_bins = torch.stack(rotated_bin_vectors, dim=0)
            for batch_index in range(i0.shape[0]):
                for label_value in torch.unique(target_labels[batch_index]).tolist():
                    label = int(label_value)
                    if label == 0:
                        continue
                    target_region = target_labels[batch_index, 0].eq(label)
                    reference_region = reference_labels[batch_index, 0].eq(label)
                    if not bool(target_region.any().item()):
                        continue
                    label_weight = reference_weight[batch_index, 0] * reference_region.to(
                        reference_weight.dtype
                    )
                    reference_base_mass = (
                        reference_base[batch_index, 0]
                        * reference_region.to(reference_base.dtype)
                    ).sum().clamp_min(1.0)
                    directional_mass = label_weight.sum()
                    support = (directional_mass / reference_base_mass).clamp(0.0, 1.0)
                    reference_evidence_map[batch_index, 0, target_region] = support
                    if float(directional_mass.item()) <= 1.0e-6:
                        continue

                    reference_angles = torch.remainder(
                        torch.atan2(
                            reference_vector[batch_index, 1],
                            reference_vector[batch_index, 0],
                        ),
                        2.0 * math.pi,
                    )
                    bin_indices = torch.floor(
                        reference_angles * (float(histogram_bins) / (2.0 * math.pi))
                    ).long().clamp(0, histogram_bins - 1)
                    histogram = label_weight.new_zeros(histogram_bins, dtype=torch.float32)
                    histogram.scatter_add_(
                        0,
                        bin_indices[reference_region].reshape(-1),
                        label_weight[reference_region].float().reshape(-1),
                    )
                    histogram = histogram / histogram.sum().clamp_min(1.0e-8)
                    log_histogram = histogram.clamp_min(1.0e-8).log()
                    target_vectors = (
                        target_unit[batch_index]
                        .reshape(2, -1)
                        .transpose(0, 1)[target_region.reshape(-1)]
                        .float()
                    )
                    compatibility = torch.einsum(
                        "nc,abc->nab",
                        target_vectors,
                        rotated_bins,
                    )
                    label_scores = torch.logsumexp(
                        histogram_concentration * compatibility
                        + log_histogram.view(1, 1, -1),
                        dim=2,
                    ) / histogram_concentration
                    scores[batch_index].reshape(len(angles), -1)[
                        :, target_region.reshape(-1)
                    ] = label_scores.transpose(0, 1)
            reference_confidence = reference_evidence_map.clamp(0.0, 1.0)
            reference_supported = reference_evidence_map.ge(support_floor)

        selected = torch.softmax(scores / temperature, dim=1)
        target_confidence = (
            (target_resultant - resultant_floor) / (1.0 - resultant_floor)
        ).clamp(0.0, 1.0)
        raw_confidence = torch.sqrt(
            (target_confidence * reference_confidence * target_support).clamp(0.0, 1.0)
        ) * target_base.gt(0).to(target_support.dtype)
        # Once both images provide usable directional evidence, do not dilute a
        # non-zero candidate almost entirely back into the unrotated reference.
        # Low-evidence regions still use the conservative raw confidence/fallback.
        supported = (
            target_support.ge(support_floor)
            & target_resultant.ge(resultant_floor)
            & reference_supported
            & target_base.gt(0)
        )
        confidence = torch.where(
            supported,
            strength_floor + (1.0 - strength_floor) * raw_confidence,
            raw_confidence,
        ).clamp(0.0, 1.0)
        fallback = torch.zeros_like(selected)
        fallback[:, :1] = 1.0
        weights = selected * confidence + fallback * (1.0 - confidence)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        eligible = target_base.sum().clamp_min(1.0)
        candidate_fractions = (weights * target_base).sum(dim=(0, 2, 3)) / eligible
        angle_tensor = weights.new_tensor(angles).view(1, -1, 1, 1)
        mean_angle = (weights * angle_tensor * target_base).sum() / eligible

    return FineTextureSteeringWeights(
        weights=weights.detach(),
        mean_confidence=(confidence * target_base).sum().div(eligible).detach(),
        raw_mean_confidence=(raw_confidence * target_base).sum().div(eligible).detach(),
        active_fraction=(confidence.gt(0.10).to(target_base.dtype) * target_base).sum().div(
            eligible
        ).detach(),
        fallback_fraction=((1.0 - confidence) * target_base).sum().div(eligible).detach(),
        candidate_fractions=candidate_fractions.detach(),
        mean_selected_angle_degrees=mean_angle.detach(),
    )


def _batched_labels(labels: torch.Tensor, *, name: str) -> torch.Tensor:
    if labels.ndim == 3:
        labels = labels.unsqueeze(1)
    if labels.ndim != 4 or labels.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W], got {tuple(labels.shape)}")
    return labels.long()


def _adjacent_pairs(labels: torch.Tensor) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        valid = first.ne(second) & first.ne(0) & second.ne(0)
        if not bool(valid.any().item()):
            continue
        values = torch.stack([first[valid], second[valid]], dim=1)
        values, _ = values.sort(dim=1)
        for first_label, second_label in torch.unique(values, dim=0).detach().cpu().tolist():
            pairs.add((int(first_label), int(second_label)))
    return pairs


def _pair_boundary(labels: torch.Tensor, first_label: int, second_label: int) -> torch.Tensor:
    first = labels.eq(int(first_label)).view(1, 1, *labels.shape)
    second = labels.eq(int(second_label)).view(1, 1, *labels.shape)
    first_near_second = first & F.max_pool2d(second.float(), 3, stride=1, padding=1).gt(0.0)
    second_near_first = second & F.max_pool2d(first.float(), 3, stride=1, padding=1).gt(0.0)
    return first_near_second | second_near_first


def _pair_orientation(
    labels: torch.Tensor,
    first_label: int,
    boundary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    first = labels.eq(int(first_label)).view(1, 1, *labels.shape).float()
    kernel_x = first.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )
    kernels = torch.stack([kernel_x, kernel_x.transpose(0, 1)], dim=0).unsqueeze(1)
    gradient = F.conv2d(first, kernels, padding=1)
    gx, gy = gradient[:, :1], gradient[:, 1:]
    magnitude_squared = gx.square() + gy.square()
    magnitude = magnitude_squared.sqrt()
    vectors = torch.cat([gx.square() - gy.square(), 2.0 * gx * gy], dim=1)
    vectors = vectors / magnitude_squared.clamp_min(1.0e-6)
    weight = magnitude * boundary.float()
    denominator = weight.sum().clamp_min(1.0e-6)
    mean_vector = (vectors * weight).sum(dim=(0, 2, 3)) / denominator
    coherence = mean_vector.square().sum().sqrt().clamp(0.0, 1.0)
    unit_vector = mean_vector / coherence.clamp_min(1.0e-6)
    return unit_vector, coherence, int(boundary.sum().item())


def build_pair_conditioned_rot90_gate(
    target_tissue_mask: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    *,
    target_nuclei_mask: torch.Tensor | None = None,
    min_boundary_pixels: int = 16,
    boundary_radius: int = 12,
    feather_radius: int = 5,
    coherence_floor: float = 0.25,
    temperature: float = 0.15,
) -> PairConditionedRot90Gate:
    """Select 0/90-degree reference updates independently for each tissue-pair boundary."""

    target = _batched_labels(target_tissue_mask, name="target_tissue_mask")
    reference = _batched_labels(reference_tissue_mask, name="reference_tissue_mask")
    if target.shape[0] != reference.shape[0]:
        raise ValueError("target and reference tissue masks must have the same batch size")
    nuclei = None
    if target_nuclei_mask is not None:
        nuclei = _batched_labels(target_nuclei_mask, name="target_nuclei_mask")
        if tuple(nuclei.shape) != tuple(target.shape):
            raise ValueError("target nuclei and tissue masks must have identical shapes")

    gate = torch.zeros_like(target, dtype=torch.float32)
    supported_counts: list[int] = []
    mean_confidences: list[float] = []
    radius = max(0, int(boundary_radius))
    coherence_floor = min(max(float(coherence_floor), 0.0), 0.999)
    temperature = max(float(temperature), 1.0e-3)

    for batch_index in range(target.shape[0]):
        target_labels = target[batch_index, 0]
        reference_labels = reference[batch_index, 0]
        shared_pairs = _adjacent_pairs(target_labels) & _adjacent_pairs(reference_labels)
        numerator = gate.new_zeros((1, 1, *target_labels.shape))
        support = gate.new_zeros((1, 1, *target_labels.shape))
        confidences: list[float] = []
        supported = 0
        for first_label, second_label in sorted(shared_pairs):
            target_boundary = _pair_boundary(target_labels, first_label, second_label)
            reference_boundary = _pair_boundary(reference_labels, first_label, second_label)
            target_vector, target_coherence, target_count = _pair_orientation(
                target_labels,
                first_label,
                target_boundary,
            )
            reference_vector, reference_coherence, reference_count = _pair_orientation(
                reference_labels,
                first_label,
                reference_boundary,
            )
            if min(target_count, reference_count) < int(min_boundary_pixels):
                continue
            target_confidence = ((target_coherence - coherence_floor) / (1.0 - coherence_floor)).clamp(0.0, 1.0)
            reference_confidence = (
                (reference_coherence - coherence_floor) / (1.0 - coherence_floor)
            ).clamp(0.0, 1.0)
            confidence = target_confidence * reference_confidence
            if float(confidence.item()) <= 0.0:
                continue
            similarity = (target_vector * reference_vector).sum().clamp(-1.0, 1.0)
            rotate_probability = torch.sigmoid(-similarity / temperature)
            pair_band = target_boundary.float()
            if radius > 0:
                pair_band = F.max_pool2d(
                    pair_band,
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                )
            family = target_labels.eq(first_label) | target_labels.eq(second_label)
            pair_band = pair_band * family.view(1, 1, *family.shape)
            if int(feather_radius) > 0:
                pair_band = gaussian_blur_image(pair_band, max(0.5, float(feather_radius) / 2.0))
            weighted_band = pair_band * confidence
            numerator = numerator + weighted_band * rotate_probability
            support = support + weighted_band
            supported += 1
            confidences.append(float(confidence.item()))
        sample_gate = numerator / support.clamp_min(1.0)
        if nuclei is not None:
            sample_gate = sample_gate * nuclei[batch_index : batch_index + 1].eq(0).to(sample_gate.dtype)
        gate[batch_index : batch_index + 1] = sample_gate.clamp(0.0, 1.0)
        supported_counts.append(supported)
        mean_confidences.append(sum(confidences) / max(1, len(confidences)))
    return PairConditionedRot90Gate(
        gate=gate,
        supported_pairs=supported_counts,
        mean_confidence=mean_confidences,
    )
