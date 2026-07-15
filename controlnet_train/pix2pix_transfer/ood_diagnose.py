"""OOD diagnostic panel helpers for pix2pix texture transfer."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .identity_losses import (
    family_image_descriptor,
    grayscale_structure_losses,
    laplacian_band_descriptor,
)
from .losses import boundary_band_mask, high_frequency_residual_loss


def _batched(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(0) if value.ndim == 3 else value


def _foreground_mask(tissue: torch.Tensor, nuclei: torch.Tensor) -> torch.Tensor:
    tissue = _batched(tissue)
    nuclei = _batched(nuclei)
    return tissue.ne(0) | nuclei.ne(0)


@torch.no_grad()
def compute_identity_metrics(
    *,
    target_i0: torch.Tensor,
    target: torch.Tensor,
    references: list[torch.Tensor],
    outputs: list[torch.Tensor],
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    reference_tissue_masks: list[torch.Tensor],
    reference_nuclei_masks: list[torch.Tensor],
) -> dict[str, float]:
    if not references or len(references) != len(outputs):
        raise ValueError("identity metrics require matching non-empty references and outputs")
    if len(reference_tissue_masks) != len(references) or len(reference_nuclei_masks) != len(references):
        raise ValueError("reference image and mask counts must match")
    output_batch = torch.stack(outputs).float()
    reference_batch = torch.stack(references).float()
    target_foreground = _foreground_mask(target_tissue_mask, target_nuclei_mask)
    output_descriptors = torch.cat(
        [family_image_descriptor(output_batch[index : index + 1], target_foreground) for index in range(len(outputs))],
        dim=0,
    )
    reference_descriptors = torch.cat(
        [
            family_image_descriptor(
                reference_batch[index : index + 1],
                _foreground_mask(reference_tissue_masks[index], reference_nuclei_masks[index]),
            )
            for index in range(len(references))
        ],
        dim=0,
    )
    distances = (output_descriptors[:, None] - reference_descriptors[None]).abs().mean(dim=2)
    selected = torch.arange(len(outputs), device=distances.device)
    top1 = distances.argmin(dim=1).eq(selected).float().mean()
    own_distance = distances[selected, selected].mean()
    i0 = _batched(target_i0).float()
    i0_descriptor = family_image_descriptor(i0, target_foreground)
    i0_distances = (i0_descriptor - reference_descriptors).abs().mean(dim=1)
    improvement = ((i0_distances - distances[selected, selected]) / i0_distances.clamp_min(1.0e-6)).mean()

    repeated_tissue = _batched(target_tissue_mask).repeat(len(outputs), 1, 1, 1)
    boundary = boundary_band_mask(repeated_tissue, radius=1)
    boundary_seam = high_frequency_residual_loss(
        output_batch,
        _batched(target_i0).repeat(len(outputs), 1, 1, 1),
        boundary,
        blur_sigma=1.0,
    )
    repeated_target = _batched(target).repeat(len(outputs), 1, 1, 1)
    gray_loss, edge_loss = grayscale_structure_losses(output_batch, repeated_target)

    nuclei_ratios = []
    target_nuclei = _batched(target_nuclei_mask)
    if int(target_nuclei.ne(0).sum().item()) > 0:
        for index, reference_nuclei in enumerate(reference_nuclei_masks):
            reference_nuclei = _batched(reference_nuclei)
            if int(reference_nuclei.ne(0).sum().item()) == 0:
                continue
            output_band = laplacian_band_descriptor(output_batch[index : index + 1], target_nuclei).mean()
            reference_band = laplacian_band_descriptor(
                reference_batch[index : index + 1],
                reference_nuclei,
            ).mean()
            nuclei_ratios.append(output_band / reference_band.clamp_min(1.0e-6))
    nuclei_ratio = torch.stack(nuclei_ratios).mean() if nuclei_ratios else output_batch.new_tensor(0.0)
    return {
        "selected_ref_top1": float(top1.item()),
        "own_ref_descriptor_distance": float(own_distance.item()),
        "identity_improvement_vs_i0": float(improvement.item()),
        "tissue_boundary_seam": float(boundary_seam.item()),
        "nuclei_band_energy_ratio": float(nuclei_ratio.item()),
        "gray_structure_error": float((gray_loss + edge_loss).item()),
    }


def save_identity_metrics(metrics: list[dict[str, float]], output_path: str | Path) -> Path:
    if not metrics:
        raise ValueError("No identity metrics were provided")
    keys = sorted(set().union(*(item.keys() for item in metrics)))
    aggregate = {
        key: sum(float(item[key]) for item in metrics if key in item)
        / max(1, sum(key in item for item in metrics))
        for key in keys
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"aggregate": aggregate, "probes": metrics}, indent=2, sort_keys=True),
        encoding="utf8",
    )
    return path


def tensor_to_pil(image: torch.Tensor, *, size: tuple[int, int] | None = None) -> Image.Image:
    array = (
        ((image.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    pil = Image.fromarray(array, mode="RGB")
    if size is not None and pil.size != size:
        pil = pil.resize(size, Image.Resampling.BILINEAR)
    return pil


def save_ood_panel(
    *,
    output_path: str | Path,
    target_i0: torch.Tensor,
    target: torch.Tensor,
    references: list[torch.Tensor],
    outputs: list[torch.Tensor],
    title: str = "",
) -> Path:
    if len(references) != len(outputs):
        raise ValueError("references and outputs must have the same length")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = tensor_to_pil(target_i0).size
    label_h = 24
    row_label_w = 92
    columns = max(2, len(references))
    rows = 4
    canvas = Image.new(
        "RGB",
        (row_label_w + columns * cell_w, label_h + rows * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((6, 5), title, fill=(0, 0, 0))
    for col in range(columns):
        if col < len(references):
            draw.text((row_label_w + col * cell_w + 6, 5), f"ref {col}", fill=(0, 0, 0))
    rows_data = [
        ("Target I0", [target_i0] * columns),
        ("Target GT", [target] * columns),
        ("Reference", references),
        ("Output", outputs),
    ]
    for row, (label, tensors) in enumerate(rows_data):
        draw.text((6, label_h + row * cell_h + 6), label, fill=(0, 0, 0))
        for col in range(columns):
            tensor = tensors[min(col, len(tensors) - 1)]
            canvas.paste(tensor_to_pil(tensor, size=(cell_w, cell_h)), (row_label_w + col * cell_w, label_h + row * cell_h))
    canvas.save(path)
    return path


def save_ood_comparison_panel(
    *,
    output_path: str | Path,
    target_i0: torch.Tensor,
    target: torch.Tensor,
    references: list[torch.Tensor],
    baseline_outputs: list[torch.Tensor],
    gated_outputs: list[torch.Tensor],
    title: str = "",
) -> Path:
    if not references or len(references) != len(baseline_outputs) or len(references) != len(gated_outputs):
        raise ValueError("references, baseline outputs, and gated outputs must have the same non-zero length")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = tensor_to_pil(target_i0).size
    label_h = 24
    row_label_w = 92
    columns = max(2, len(references))
    rows_data = [
        ("Target I0", [target_i0] * columns),
        ("Target GT", [target] * columns),
        ("Reference", references),
        ("Baseline", baseline_outputs),
        ("Pair-gated", gated_outputs),
    ]
    canvas = Image.new(
        "RGB",
        (row_label_w + columns * cell_w, label_h + len(rows_data) * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((6, 5), title, fill=(0, 0, 0))
    for col in range(len(references)):
        draw.text((row_label_w + col * cell_w + 6, 5), f"ref {col}", fill=(0, 0, 0))
    for row, (label, tensors) in enumerate(rows_data):
        draw.text((6, label_h + row * cell_h + 6), label, fill=(0, 0, 0))
        for col in range(columns):
            tensor = tensors[min(col, len(tensors) - 1)]
            canvas.paste(
                tensor_to_pil(tensor, size=(cell_w, cell_h)),
                (row_label_w + col * cell_w, label_h + row * cell_h),
            )
    canvas.save(path)
    return path


def save_ood_summary_grid(panel_paths: list[str | Path], output_path: str | Path) -> Path:
    paths = [Path(path) for path in panel_paths if Path(path).exists()]
    if not paths:
        raise ValueError("No existing panel paths were provided")
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    total_height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, total_height), "white")
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
