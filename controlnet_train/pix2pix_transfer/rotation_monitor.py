"""Fixed baseline-vs-current monitoring for rotated-reference continuation."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .identity_losses import family_image_descriptor, laplacian_band_descriptor
from .losses import boundary_band_mask, high_frequency_residual_loss
from .orientation_supervision import multiscale_target_orientation_loss


def _foreground(tissue: torch.Tensor, nuclei: torch.Tensor) -> torch.Tensor:
    if tissue.ndim == 3:
        tissue = tissue.unsqueeze(1)
    if nuclei.ndim == 3:
        nuclei = nuclei.unsqueeze(1)
    return tissue.ne(0) | nuclei.ne(0)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    both_zero = numerator.abs().lt(1.0e-8) & denominator.abs().lt(1.0e-8)
    ratio = numerator / denominator.clamp_min(1.0e-8)
    return torch.where(both_zero, torch.ones_like(ratio), ratio)


@torch.no_grad()
def compute_rotation_monitor_metrics(
    *,
    target_i0: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    baseline_clean: torch.Tensor,
    current_clean: torch.Tensor,
    baseline_rotated: torch.Tensor,
    current_rotated: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    trust_map: torch.Tensor,
) -> dict[str, float]:
    target_foreground = _foreground(target_tissue_mask, target_nuclei_mask)
    reference_foreground = _foreground(reference_tissue_mask, reference_nuclei_mask)
    baseline_descriptor = family_image_descriptor(baseline_clean, target_foreground)
    current_descriptor = family_image_descriptor(current_clean, target_foreground)
    reference_descriptor = family_image_descriptor(reference, reference_foreground)
    baseline_ref_distance = (baseline_descriptor - reference_descriptor).abs().mean()
    current_ref_distance = (current_descriptor - reference_descriptor).abs().mean()

    boundary = boundary_band_mask(target_tissue_mask, radius=1)
    baseline_seam = high_frequency_residual_loss(
        baseline_clean,
        target_i0,
        boundary,
        blur_sigma=1.0,
    )
    current_seam = high_frequency_residual_loss(
        current_clean,
        target_i0,
        boundary,
        blur_sigma=1.0,
    )
    baseline_orientation = multiscale_target_orientation_loss(
        baseline_rotated,
        target,
        target_tissue_mask=target_tissue_mask,
        target_nuclei_mask=target_nuclei_mask,
        trust_map=trust_map,
    ).orientation
    current_orientation = multiscale_target_orientation_loss(
        current_rotated,
        target,
        target_tissue_mask=target_tissue_mask,
        target_nuclei_mask=target_nuclei_mask,
        trust_map=trust_map,
    ).orientation

    nuclei = target_nuclei_mask.ne(0)
    reference_nuclei = reference_nuclei_mask.ne(0)
    if bool(nuclei.any().item()) and bool(reference_nuclei.any().item()):
        baseline_band = laplacian_band_descriptor(baseline_clean, nuclei).mean()
        current_band = laplacian_band_descriptor(current_clean, nuclei).mean()
        nuclei_ratio = _safe_ratio(current_band, baseline_band)
    else:
        nuclei_ratio = target.new_tensor(1.0)

    return {
        "clean_drift_mae": float((current_clean - baseline_clean).abs().mean().item()),
        "baseline_rotation_sensitivity": float(
            (baseline_rotated - baseline_clean).abs().mean().item()
        ),
        "current_rotation_sensitivity": float(
            (current_rotated - current_clean).abs().mean().item()
        ),
        "rotated_model_delta_mae": float(
            (current_rotated - baseline_rotated).abs().mean().item()
        ),
        "baseline_rotated_orientation": float(baseline_orientation.item()),
        "current_rotated_orientation": float(current_orientation.item()),
        "baseline_clean_ref_distance": float(baseline_ref_distance.item()),
        "current_clean_ref_distance": float(current_ref_distance.item()),
        "current_clean_ref_distance_ratio": float(
            _safe_ratio(current_ref_distance, baseline_ref_distance).item()
        ),
        "baseline_boundary_seam": float(baseline_seam.item()),
        "current_boundary_seam": float(current_seam.item()),
        "current_boundary_seam_ratio": float(_safe_ratio(current_seam, baseline_seam).item()),
        "current_nuclei_band_ratio_vs_baseline": float(nuclei_ratio.item()),
    }


def rotation_monitor_reasons(
    metrics: dict[str, float],
    *,
    max_clean_drift: float,
    max_ref_distance_ratio: float,
    max_boundary_seam_ratio: float,
    min_nuclei_band_ratio: float,
    max_nuclei_band_ratio: float,
) -> list[str]:
    checks = (
        (metrics["clean_drift_mae"] > max_clean_drift, "clean_drift"),
        (
            metrics["current_clean_ref_distance_ratio"] > max_ref_distance_ratio,
            "ref_distance",
        ),
        (
            metrics["current_boundary_seam_ratio"] > max_boundary_seam_ratio,
            "boundary_seam",
        ),
        (
            not min_nuclei_band_ratio
            <= metrics["current_nuclei_band_ratio_vs_baseline"]
            <= max_nuclei_band_ratio,
            "nuclei_band",
        ),
    )
    return [name for failed, name in checks if failed]


def _to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        ((image.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def save_rotation_monitor_panel(
    *,
    output_path: str | Path,
    target_i0: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    rotated_reference: torch.Tensor,
    baseline_clean: torch.Tensor,
    current_clean: torch.Tensor,
    baseline_rotated: torch.Tensor,
    current_rotated: torch.Tensor,
    angles_degrees: torch.Tensor,
) -> Path:
    rows = (
        ("I0", target_i0),
        ("Target", target),
        ("Reference", reference),
        ("Rotated ref", rotated_reference),
        ("Epoch25 clean", baseline_clean),
        ("Current clean", current_clean),
        ("Epoch25 rotated", baseline_rotated),
        ("Current rotated", current_rotated),
    )
    count = min(int(target_i0.shape[0]), 5)
    cell_w, cell_h = _to_pil(target_i0[0]).size
    label_h = 24
    row_label_w = 108
    canvas = Image.new(
        "RGB",
        (row_label_w + count * cell_w, label_h + len(rows) * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    angle_values = angles_degrees.detach().cpu().tolist()
    for column in range(count):
        draw.text(
            (row_label_w + column * cell_w + 6, 5),
            f"sample {column} angle={float(angle_values[column]):.1f}",
            fill="black",
        )
    for row_index, (label, images) in enumerate(rows):
        y = label_h + row_index * cell_h
        draw.text((6, y + 6), label, fill="black")
        for column in range(count):
            canvas.paste(_to_pil(images[column]), (row_label_w + column * cell_w, y))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path
