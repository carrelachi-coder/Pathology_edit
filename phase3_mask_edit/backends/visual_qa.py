"""Visual QA artifacts for Phase 3 LLM contour projection runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from phase3_mask_edit.backends.llm_contour import ContourProposal, rasterize_contour_proposal
from phase3_mask_edit.backends.llm_preview import (
    add_coordinate_grid_overlay,
    id_mask_to_llm_preview_rgb,
)
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import save_change_region, save_metadata, save_rgb_mask
from phase3_mask_edit.core.validation import ValidationResult
from phase3_mask_edit.generic.tumor_burden import PrimitiveEditResult


def save_visual_qa_bundle(
    *,
    source_mask: np.ndarray,
    proposal: ContourProposal,
    schema: MaskProfileSchema,
    edit_result: PrimitiveEditResult,
    output_dir: str | Path,
    primitive_config: Mapping[str, Any] | None = None,
    preserve_labels: Sequence[str] = (),
    forbidden_labels: Sequence[str] = (),
    validation: ValidationResult | None = None,
    projection_mode: str | None = None,
    comparison_summary: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write a compact visual QA bundle and return artifact paths."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(source_mask)
    raw_template = rasterize_contour_proposal(proposal)
    selected = np.asarray(edit_result.change_region, dtype=bool)
    source_rgb = id_mask_to_llm_preview_rgb(mask)
    legal_domain = _legal_domain_for_proposal(
        mask,
        proposal,
        schema=schema,
        preserve_labels=preserve_labels,
        forbidden_labels=forbidden_labels,
    )

    score_maps = _score_maps(
        mask,
        raw_template=raw_template,
        legal_domain=legal_domain,
        schema=schema,
        proposal=proposal,
        edit_result=edit_result,
        primitive_config=primitive_config or {},
    )

    paths: dict[str, Path] = {}
    paths["source_mask_rgb"] = _save_rgb(source_rgb, out / "source_mask_rgb.png")
    paths["source_mask_rgb_grid"] = _save_rgb(
        add_coordinate_grid_overlay(source_rgb),
        out / "source_mask_rgb_grid.png",
    )
    paths["raw_template_overlay"] = _save_rgb(
        _overlay_mask(source_rgb, raw_template, color=(255, 142, 36), alpha=0.55),
        out / "raw_template_overlay.png",
    )
    paths["legal_domain"] = save_change_region(legal_domain, out / "legal_domain.png")
    paths["template_score_heatmap"] = _save_rgb(
        _heatmap(score_maps["template_score"], legal_domain=legal_domain),
        out / "template_score_heatmap.png",
    )
    paths["spatial_score_heatmap"] = _save_rgb(
        _heatmap(score_maps["spatial_score"], legal_domain=legal_domain),
        out / "spatial_score_heatmap.png",
    )
    paths["noise_heatmap"] = _save_rgb(
        _heatmap(score_maps["noise_score"], legal_domain=legal_domain),
        out / "noise_heatmap.png",
    )
    paths["final_score_heatmap"] = _save_rgb(
        _outline_mask(
            _heatmap(score_maps["final_score"], legal_domain=legal_domain),
            selected,
            color=(255, 255, 255),
        ),
        out / "final_score_heatmap.png",
    )
    paths["selected_change_region"] = save_change_region(
        selected,
        out / "selected_change_region.png",
    )
    paths["target_mask_rgb"] = save_rgb_mask(edit_result.target_mask, out / "target_mask_rgb.png")
    paths["v1_v2_side_by_side"] = _save_rgb(
        _side_by_side_panel(
            source_rgb=source_rgb,
            raw_template=raw_template,
            legal_domain=legal_domain,
            final_score=score_maps["final_score"],
            selected=selected,
            target_rgb=id_mask_to_llm_preview_rgb(edit_result.target_mask),
            validation=validation,
            edit_result=edit_result,
            comparison_summary=comparison_summary,
        ),
        out / "v1_v2_side_by_side.png",
    )

    manifest = _manifest(
        source_mask=mask,
        raw_template=raw_template,
        selected=selected,
        legal_domain=legal_domain,
        edit_result=edit_result,
        validation=validation,
        projection_mode=projection_mode,
        artifact_paths={key: str(value) for key, value in paths.items()},
    )
    paths["visual_qa_manifest"] = save_metadata(manifest, out / "visual_qa_manifest.json")
    return {key: str(value) for key, value in paths.items()}


def _legal_domain_for_proposal(
    mask: np.ndarray,
    proposal: ContourProposal,
    *,
    schema: MaskProfileSchema,
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
) -> np.ndarray:
    labels = tuple(
        dict.fromkeys(label for region in proposal.regions for label in region.source_labels)
    )
    legal = np.zeros(mask.shape, dtype=bool)
    for label in labels:
        legal |= np.isin(mask, schema.resolve_fine_ids(label))
    for label in tuple(dict.fromkeys((*preserve_labels, *forbidden_labels))):
        legal &= ~np.isin(mask, schema.resolve_fine_ids(label))
    legal &= ~np.isin(mask, tuple(schema.skip_fine_ids))
    return legal


def _score_maps(
    mask: np.ndarray,
    *,
    raw_template: np.ndarray,
    legal_domain: np.ndarray,
    schema: MaskProfileSchema,
    proposal: ContourProposal,
    edit_result: PrimitiveEditResult,
    primitive_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    ops = edit_result.ops_log
    terms = ops.get("score_terms", {})
    template_sigma = _float(terms.get("template_sigma"), 3.0)
    noise_sigma = _float(terms.get("noise_sigma"), 18.0)
    noise_amplitude = _float(terms.get("noise_amplitude"), 0.18)
    w_template = _float(terms.get("w_template"), 0.45)
    w_spatial = _float(terms.get("w_spatial"), 0.45)
    w_noise = _float(terms.get("w_noise"), 0.10)

    template = _signed_template_score(raw_template, sigma=template_sigma)
    spatial = _spatial_score(
        mask,
        legal_domain=legal_domain,
        schema=schema,
        primitive=proposal.primitive,
        primitive_config=primitive_config,
    )
    noise = _smooth_noise(
        mask.shape,
        seed=int(ops.get("noise_seed", 0)),
        sigma=noise_sigma,
    )
    template_norm = _normalize(template, legal_domain)
    spatial_norm = _normalize(spatial, legal_domain)
    noise_norm = _normalize(noise, legal_domain)
    final = (
        w_template * template_norm
        + w_spatial * spatial_norm
        + w_noise * noise_amplitude * noise_norm
    )
    final[~legal_domain] = 0.0
    return {
        "template_score": template_norm,
        "spatial_score": spatial_norm,
        "noise_score": noise_norm,
        "final_score": final,
    }


def _spatial_score(
    mask: np.ndarray,
    *,
    legal_domain: np.ndarray,
    schema: MaskProfileSchema,
    primitive: str,
    primitive_config: Mapping[str, Any],
) -> np.ndarray:
    ranges = primitive_config.get("parameter_ranges", {})
    score = np.zeros(mask.shape, dtype=float)
    if primitive == "stromal_immune_infiltration":
        tumor = np.isin(mask, schema.tumor_fine_ids)
        if np.any(tumor):
            decay_px = _float(ranges.get("peritumoral_falloff_radius_px"), 48.0)
            dist = ndimage.distance_transform_edt(~tumor)
            score = np.exp(-dist / max(decay_px, 1.0))
    elif primitive == "necrosis_appearance":
        tumor = np.isin(mask, schema.tumor_fine_ids)
        score = ndimage.distance_transform_edt(tumor)
    else:
        score = ndimage.distance_transform_edt(legal_domain)
    score[~legal_domain] = 0.0
    return score


def _signed_template_score(candidate: np.ndarray, *, sigma: float) -> np.ndarray:
    if not np.any(candidate):
        return np.zeros(candidate.shape, dtype=float)
    inside = ndimage.distance_transform_edt(candidate)
    outside = ndimage.distance_transform_edt(~candidate)
    return ndimage.gaussian_filter((inside - outside).astype(float), sigma=sigma)


def _smooth_noise(shape: tuple[int, int], *, seed: int, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(size=shape), sigma=sigma)


def _normalize(values: np.ndarray, domain: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=float)
    data = np.asarray(values, dtype=float)[domain]
    if data.size == 0:
        return result
    low = float(np.percentile(data, 2))
    high = float(np.percentile(data, 98))
    if high <= low:
        return result
    normalized = np.clip((np.asarray(values, dtype=float) - low) / (high - low), 0.0, 1.0)
    result[domain] = normalized[domain]
    return result


def _heatmap(values: np.ndarray, *, legal_domain: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    rgb = np.zeros(arr.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * arr, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255 * (1.0 - np.abs(arr - 0.5) * 2.0), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255 * (1.0 - arr), 0, 255).astype(np.uint8)
    rgb[~legal_domain] = np.array([50, 50, 50], dtype=np.uint8)
    return rgb


def _overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    out = np.asarray(rgb, dtype=np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _outline_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
) -> np.ndarray:
    outline = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    out = np.asarray(rgb, dtype=np.uint8).copy()
    out[outline] = np.asarray(color, dtype=np.uint8)
    return out


def _side_by_side_panel(
    *,
    source_rgb: np.ndarray,
    raw_template: np.ndarray,
    legal_domain: np.ndarray,
    final_score: np.ndarray,
    selected: np.ndarray,
    target_rgb: np.ndarray,
    validation: ValidationResult | None,
    edit_result: PrimitiveEditResult,
    comparison_summary: Mapping[str, Any] | None,
) -> np.ndarray:
    tiles = [
        ("source", source_rgb),
        ("raw template", _overlay_mask(source_rgb, raw_template, color=(255, 142, 36), alpha=0.55)),
        ("legal domain", _overlay_mask(source_rgb, legal_domain, color=(58, 167, 255), alpha=0.50)),
        ("final score", _outline_mask(_heatmap(final_score, legal_domain=legal_domain), selected, color=(255, 255, 255))),
        ("selected", _overlay_mask(source_rgb, selected, color=(255, 255, 255), alpha=0.70)),
        ("target", target_rgb),
    ]
    tile_h, tile_w = source_rgb.shape[:2]
    label_h = 22
    cols = 3
    rows = 2
    panel = np.full((rows * (tile_h + label_h), cols * tile_w, 3), 255, dtype=np.uint8)
    image = Image.fromarray(panel, mode="RGB")
    draw = ImageDraw.Draw(image)
    for index, (label, tile) in enumerate(tiles):
        row = index // cols
        col = index % cols
        x = col * tile_w
        y = row * (tile_h + label_h)
        image.paste(Image.fromarray(tile.astype(np.uint8), mode="RGB"), (x, y + label_h))
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0))

    status = "validation: n/a" if validation is None else f"validation: {validation.passed}"
    details = [
        status,
        f"selected: {edit_result.selected_pixels}",
        f"warnings: {','.join(edit_result.warnings) or 'none'}",
    ]
    if comparison_summary is not None:
        details.append(f"compare: {comparison_summary.get('projection_mode', 'n/a')}")
    draw.text((4, panel.shape[0] - label_h + 4), " | ".join(details), fill=(0, 0, 0))
    return np.asarray(image, dtype=np.uint8)


def _manifest(
    *,
    source_mask: np.ndarray,
    raw_template: np.ndarray,
    selected: np.ndarray,
    legal_domain: np.ndarray,
    edit_result: PrimitiveEditResult,
    validation: ValidationResult | None,
    projection_mode: str | None,
    artifact_paths: Mapping[str, str],
) -> dict[str, Any]:
    intersection = int(np.count_nonzero(selected & raw_template))
    union = int(np.count_nonzero(selected | raw_template))
    return {
        "projection_mode": projection_mode or edit_result.ops_log.get("projection_mode"),
        "projection_backend": edit_result.ops_log.get("projection_backend"),
        "primitive": edit_result.ops_log.get("primitive"),
        "target_pixels": edit_result.ops_log.get("target_pixels"),
        "selected_pixels": int(np.count_nonzero(selected)),
        "legal_domain_pixels": int(np.count_nonzero(legal_domain)),
        "raw_template_pixels": int(np.count_nonzero(raw_template)),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "selected_raw_template_iou": intersection / union if union else 0.0,
        "ops_log_selected_raw_template_iou": edit_result.ops_log.get(
            "selected_raw_template_iou"
        ),
        "label_safety_visible": None,
        "pathology_placement": None,
        "boundary_naturalness": None,
        "area_plausibility": None,
        "template_respect": None,
        "fragmentation": None,
        "artifact_explainability": None,
        "reviewer_notes": "",
        "validation_passed": None if validation is None else validation.passed,
        "warnings": list(edit_result.warnings),
        "artifact_paths": dict(artifact_paths),
        "source_shape": list(source_mask.shape),
    }


def _save_rgb(rgb: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(p)
    return p


def _float(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)
