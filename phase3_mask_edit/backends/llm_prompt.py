"""Prompt, request, and feedback helpers for LLM contour proposals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phase3_mask_edit.backends.llm_contour import (
    CONTOUR_PROPOSAL_BACKEND,
    CONTOUR_PROPOSAL_SCHEMA_VERSION,
)
from phase3_mask_edit.backends.llm_preview import llm_palette_legend
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.validation import ValidationResult
from phase3_mask_edit.generic.tumor_burden import PrimitiveEditResult


@dataclass(frozen=True)
class ContourProposalRequest:
    """Provider request for one contour proposal attempt."""

    prompt: str
    context: dict[str, Any]
    attempt_index: int
    image_paths: tuple[str, ...] = ()
    repair_feedback: dict[str, Any] | None = None
    provider_metadata: dict[str, Any] | None = None


def build_mask_context(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
    allowed_source_labels: Sequence[str],
    target_label: str,
    grid_spacing_px: int = 64,
    max_regions: int = 8,
    max_points_per_region: int = 64,
) -> dict[str, Any]:
    """Build compact mask context for a contour proposal prompt."""

    arr = np.asarray(mask)
    height, width = arr.shape
    label_areas: dict[str, int] = {}
    for label in sorted(schema.readable_labels):
        ids = schema.resolve_fine_ids(label)
        label_areas[label] = int(np.count_nonzero(np.isin(arr, ids)))

    target_area_hint = _build_target_area_hint(
        label_areas,
        intent=intent,
        primitive_config=primitive_config,
    )
    source_spatial_hints = _build_source_spatial_hints(
        arr,
        schema=schema,
        source_labels=allowed_source_labels,
        grid_spacing_px=grid_spacing_px,
    )
    source_contour_context = _build_source_contour_context(
        arr,
        schema=schema,
        source_labels=allowed_source_labels,
    )
    llm_task_requirements = _build_llm_task_requirements(
        primitive_config,
        target_area_hint=target_area_hint,
    )
    contour_style_hint = _build_contour_style_hint(
        intent=intent,
        primitive_config=primitive_config,
        target_area_hint=target_area_hint,
        source_spatial_hints=source_spatial_hints,
    )

    return {
        "mask_shape": [height, width],
        "reference_profile": intent.reference_profile or schema.reference_profile,
        "primitive": intent.primitive,
        "strength": intent.strength,
        "target_label": target_label,
        "allowed_source_labels": list(allowed_source_labels),
        "label_areas": label_areas,
        "label_palette_rgb": llm_palette_legend(),
        "visual_label_legend": _visual_label_legend(),
        "proposal_policy": {
            "backend": CONTOUR_PROPOSAL_BACKEND,
            "schema_version": CONTOUR_PROPOSAL_SCHEMA_VERSION,
            "max_regions": int(max_regions),
            "max_points_per_region": int(max_points_per_region),
            "template_role": "coarse_template",
            "known_optional_v2_fields": [
                "source_component_ids",
                "adjacency_side",
                "placement_relation",
                "template_role",
                "shape_hints",
            ],
            "executor_contract": (
                "LLM polygons are approximate organic intent templates. "
                "Deterministic code projects them to legal source labels, "
                "controls final area, and writes final pixels."
            ),
        },
        "target_area_hint": target_area_hint,
        "llm_task_requirements": llm_task_requirements,
        "contour_style_hint": contour_style_hint,
        "source_spatial_hints": source_spatial_hints,
        "source_contour_context": source_contour_context,
        "primitive_policy": {
            "name": primitive_config.get("name", intent.primitive),
            "pathology_meaning": primitive_config.get("pathology_meaning"),
            "mask_operation": primitive_config.get("mask_operation", {}),
            "spatial_policy": primitive_config.get("spatial_policy", {}),
            "parameter_ranges": primitive_config.get("parameter_ranges", {}),
            "validation_rules": primitive_config.get("validation_rules", []),
        },
        "preview": {
            "coordinate_system": "original_mask_xy",
            "origin": "top_left",
            "point_format": "[x, y]",
            "x_axis": "horizontal_column_right",
            "y_axis": "vertical_row_down",
            "width": width,
            "height": height,
            "grid_spacing_px": int(grid_spacing_px),
        },
    }


def build_contour_prompt(
    *,
    context: Mapping[str, Any],
    repair_feedback: Mapping[str, Any] | None = None,
) -> str:
    """Build a strict JSON-only contour proposal prompt."""

    height, width = context["mask_shape"]
    skeleton = {
        "schema_version": CONTOUR_PROPOSAL_SCHEMA_VERSION,
        "backend": CONTOUR_PROPOSAL_BACKEND,
        "primitive": context["primitive"],
        "reference_profile": context["reference_profile"],
        "target_label": context["target_label"],
        "template_role": "coarse_template",
        "placement_relation": _default_placement_relation(context),
        "shape_hints": _default_shape_hints(context),
        "coordinate_system": {
            "origin": "top_left",
            "point_format": "[x, y]",
            "x_axis": "horizontal_column_right",
            "y_axis": "vertical_row_down",
            "width": width,
            "height": height,
        },
        "regions": [
            {
                "region_id": "r1",
                "type": "polygon",
                "source_labels": context["allowed_source_labels"],
                "source_component_ids": ["source_1"],
                "adjacency_side": _default_placement_relation(context),
                "template_role": "coarse_template",
                "shape_hints": _default_shape_hints(context),
                "points": [
                    [205, 118],
                    [232, 126],
                    [258, 146],
                    [276, 172],
                    [292, 205],
                    [286, 236],
                    [262, 255],
                    [230, 246],
                    [206, 218],
                    [194, 184],
                    [188, 148],
                ],
                "confidence": 0.75,
            }
        ],
    }

    parts = [
        "Return only one valid JSON object. Do not use markdown fences.",
        "Propose one or more polygon change regions for the pathology mask edit.",
        "Treat every polygon as a coarse organic template for pathology placement, not as the final changed-pixel mask.",
        "Coordinates must be in the original mask coordinate system, not a resized preview.",
        "Use [x, y] points: x is the horizontal column increasing right; y is the vertical row increasing down; origin is top-left.",
        "All points must be inside mask bounds.",
        "Use only the allowed source labels and exact target label from the context.",
        _source_label_visual_instruction(context),
        _task_requirements_instruction(context),
        _organic_shape_instruction(context),
        _contour_style_instruction(context),
        "Use source_contour_context as the primary placement reference. It gives source-label component ids, contours, and adjacent tissue on the other side of the boundary.",
        "Within each source component, contour_adjacency_segments groups contour coordinates by the tissue just across the boundary. Prefer choosing component ids and adjacency sides before drawing points.",
        "For stromal immune infiltration, prefer Stroma contour segments adjacent to Tumor and avoid segments adjacent to Necrosis unless the recipe explicitly asks for necrosis-adjacent change.",
        "For necrosis appearance, prefer Tumor interior components and avoid hugging the outer Tumor boundary unless the intent requires it.",
        "Generate a rough organic template around the intended pathology location; do not optimize vertices to be pixel-perfect source-label coordinates.",
        "The downstream executor will rasterize, project to legal source labels, control final changed area, write the target label, and validate.",
        "If target_area_hint is present, its target_changed_pixels_min/max refer to the desired area after deterministic projection, not raw polygon area.",
        "Draw raw templates broad enough to express placement and shape variety; the projector will trim and refill within the legal domain.",
        "Use source_spatial_hints only as location anchors. Do not trace tile boxes or component bboxes as the final shape.",
        "Optional V2 fields are allowed only when named in proposal_policy. Use template_role='coarse_template' when you include template_role.",
        "",
        "Mask context JSON:",
        json.dumps(context, indent=2, ensure_ascii=False),
        "",
        "Required output shape example:",
        json.dumps(skeleton, indent=2, ensure_ascii=False),
    ]
    if repair_feedback:
        parts.extend(
            [
                "",
                "Previous attempt feedback JSON:",
                json.dumps(repair_feedback, indent=2, ensure_ascii=False),
            ]
        )
    return "\n".join(parts)


def build_repair_feedback(
    *,
    status: str,
    attempt_index: int,
    error: str | None = None,
    validation: ValidationResult | None = None,
    edit_result: PrimitiveEditResult | None = None,
) -> dict[str, Any]:
    """Build compact structured feedback for the next attempt."""

    feedback: dict[str, Any] = {
        "status": status,
        "attempt": int(attempt_index),
    }
    if error:
        feedback["error"] = error
    if validation is not None:
        feedback["failed_checks"] = [
            {
                "name": check.name,
                "detail": check.detail,
            }
            for check in validation.failed_checks
        ]
    if edit_result is not None:
        ops_log = edit_result.ops_log
        feedback["projection"] = {
            "candidate_pixels": ops_log.get("candidate_pixels"),
            "projected_pixels": ops_log.get("projected_pixels"),
            "selected_pixels": edit_result.selected_pixels,
            "projection_retained_fraction": ops_log.get(
                "projection_retained_fraction"
            ),
            "projection_mode": ops_log.get("projection_mode"),
            "projection_backend": ops_log.get("projection_backend"),
            "legal_domain_pixels": ops_log.get("legal_domain_pixels"),
            "target_pixels": ops_log.get("target_pixels"),
            "area_shortfall": ops_log.get("area_shortfall"),
            "template_overlap_with_legal_domain": ops_log.get(
                "template_overlap_with_legal_domain"
            ),
            "top_failed_reason": _top_projection_failed_reason(
                validation=validation,
                edit_result=edit_result,
            ),
        }
        feedback["warnings"] = list(edit_result.warnings)
    return feedback


def _top_projection_failed_reason(
    *,
    validation: ValidationResult | None,
    edit_result: PrimitiveEditResult,
) -> str | None:
    ops_log = edit_result.ops_log
    legal = ops_log.get("legal_domain_pixels")
    target = ops_log.get("target_pixels")
    selected = edit_result.selected_pixels
    if isinstance(legal, int) and isinstance(target, int) and legal < target:
        return "legal_domain_too_small"
    if "organic_projection_area_shortfall" in edit_result.warnings:
        return "projector_area_shortfall_after_cleanup"
    overlap = ops_log.get("template_overlap_with_legal_domain")
    if isinstance(overlap, (int, float)) and float(overlap) < 0.05:
        return "template_overlap_with_legal_domain_too_low"
    if validation is not None and validation.failed_checks:
        return validation.failed_checks[0].name
    if selected == 0:
        return "selected_pixels_empty"
    return None


def save_prompt_text(prompt: str, path: str | Path) -> Path:
    """Save a prompt string to disk."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(prompt, encoding="utf-8")
    return p


def _build_target_area_hint(
    label_areas: Mapping[str, int],
    *,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    ranges = primitive_config.get("parameter_ranges", {})
    if not isinstance(ranges, Mapping):
        return None

    primitive_name = primitive_config.get("name", intent.primitive)
    if primitive_name == "stromal_immune_infiltration":
        bucket = _interval_for_strength(
            ranges.get("immune_area_delta_fraction"),
            intent.strength,
        )
        reference_pixels = int(label_areas.get("Stroma", 0)) + int(
            label_areas.get("Immune infiltrate", 0)
        )
        reference = "Stroma + Immune infiltrate"
    elif primitive_name == "necrosis_appearance":
        bucket = _interval_for_strength(
            ranges.get("target_changed_area_fraction"),
            intent.strength,
        )
        reference_pixels = int(label_areas.get("Tumor", 0))
        reference = "Tumor"
    elif primitive_name == "intratumoral_immune_infiltration":
        bucket = _interval_for_strength(
            ranges.get("target_changed_area_fraction"),
            intent.strength,
        )
        reference_pixels = int(label_areas.get("Tumor", 0))
        reference = "Tumor"
    else:
        bucket = None
        reference_pixels = 0
        reference = None

    if bucket is None or reference is None or reference_pixels <= 0:
        return None

    lower, upper = bucket
    projected_min = int(np.ceil(reference_pixels * lower))
    projected_max = int(np.floor(reference_pixels * upper))
    return {
        "area_semantics": "projection_after_legal_source_label_filtering",
        "reference_area_label": reference,
        "reference_pixels": reference_pixels,
        "strength": intent.strength,
        "target_fraction_min": lower,
        "target_fraction_max": upper,
        "target_changed_pixels_min": projected_min,
        "target_changed_pixels_max": projected_max,
        "raw_polygon_margin_guidance": (
            "Raw polygons are coarse organic templates. They may be broader than "
            "the projected target range because Phase 3 will select final legal "
            "source-label pixels deterministically."
        ),
    }


def _interval_for_strength(value: Any, strength: str) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    interval = value.get(strength)
    if (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(item, (int, float)) for item in interval)
    ):
        return float(interval[0]), float(interval[1])
    return None


def _build_llm_task_requirements(
    primitive_config: Mapping[str, Any],
    *,
    target_area_hint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    name = primitive_config.get("name")
    operation = primitive_config.get("mask_operation", {})
    spatial = primitive_config.get("spatial_pattern", {})
    ranges = primitive_config.get("parameter_ranges", {})
    validation_rules = primitive_config.get("validation_rules", [])

    if name == "stromal_immune_infiltration":
        return {
            "pathology_goal": "Increase stromal tumor-infiltrating lymphocytes around tumor.",
            "mask_edit": "Convert Stroma pixels to Immune infiltrate pixels.",
            "source_label": "Stroma",
            "target_label": "Immune infiltrate",
            "where_to_draw": [
                "Draw outside tumor, on green Stroma.",
                "Prefer peritumoral Stroma near the red Tumor boundary.",
                "For mild strength, prefer patchy stromal infiltrate over one continuous broad band.",
                "Use partial peritumoral band only when patchy regions cannot reach the target area naturally.",
                "Do not draw mainly inside red Tumor or blue Necrosis.",
            ],
            "shape_style": [
                "Organic irregular patch contours.",
                "Patchy separated or weakly connected regions are preferred for mild strength.",
                "Natural uneven border that follows stromal compartment boundaries.",
                "Avoid one large wedge-shaped continuous band unless required for area.",
                "Avoid repeated identical blobs or regular geometric shapes.",
            ],
            "area_requirement": _area_requirement_text(target_area_hint),
            "recipe_constraints": {
                "spatial_pattern": spatial,
                "parameter_ranges": ranges,
                "validation_rules": validation_rules,
            },
        }
    if name == "necrosis_appearance":
        return {
            "pathology_goal": "Add ischemic tumor necrosis.",
            "mask_edit": "Convert Tumor pixels to Necrosis pixels.",
            "source_label": operation.get("source", "Tumor"),
            "target_label": operation.get("target", "Necrosis"),
            "where_to_draw": [
                "Draw inside red Tumor.",
                "Prefer tumor interior, away from non-tumor tissue.",
                "Do not draw mainly on Stroma or Background.",
            ],
            "shape_style": [
                "Irregular blob or fissure-like region.",
                "Natural uneven necrotic boundary.",
                "Avoid regular geometric shapes.",
            ],
            "area_requirement": _area_requirement_text(target_area_hint),
            "recipe_constraints": {
                "spatial_pattern": spatial,
                "parameter_ranges": ranges,
                "validation_rules": validation_rules,
            },
        }
    return {
        "pathology_goal": primitive_config.get("pathology_meaning"),
        "mask_operation": operation,
        "area_requirement": _area_requirement_text(target_area_hint),
        "recipe_constraints": {
            "spatial_pattern": spatial,
            "parameter_ranges": ranges,
            "validation_rules": validation_rules,
        },
    }


def _build_contour_style_hint(
    *,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
    target_area_hint: Mapping[str, Any] | None,
    source_spatial_hints: Mapping[str, Any],
) -> dict[str, Any]:
    primitive_name = primitive_config.get("name", intent.primitive)
    target_pixels = 0
    if target_area_hint:
        target_pixels = int(target_area_hint.get("target_changed_pixels_max") or 0)
    if primitive_name == "stromal_immune_infiltration":
        strength_regions = {
            "mild": [2, 4],
            "moderate": [2, 4],
            "significant": [3, 6],
        }
        recommended_regions = strength_regions.get(intent.strength, [1, 3])
        if target_pixels < 1800:
            recommended_regions = [1, min(recommended_regions[1], 2)]
        elif target_pixels > 7000:
            recommended_regions = [max(recommended_regions[0], 3), max(recommended_regions[1], 5)]
        components = source_spatial_hints.get("Stroma", {}).get("components", [])
        if isinstance(components, list):
            large_components = sum(1 for item in components if int(item.get("area_px", 0)) >= 512)
            if large_components <= 1:
                recommended_regions = [1, min(recommended_regions[1], 3)]
        return {
            "recommended_region_count_range": recommended_regions,
            "points_per_region_range": [24, 48],
            "boundary_style": (
                "Smooth organic amoeba-like boundaries. Use densely spaced points along curves. "
                "Consecutive points should be close together, usually about 10-30 px apart, to produce smooth arcs. "
                "Avoid long straight edges between distant vertices. "
                "The contour should look like a natural tissue boundary, not a geometric shape."
            ),
            "region_variation_requirement": (
                "If multiple regions are used, each must have different size, aspect ratio, "
                "orientation, and boundary irregularity. Do not duplicate the same diamond/blob shape."
            ),
            "connectivity_guidance": (
                "For mild strength, prefer several separated or weakly connected patchy contours, not one broad wedge-shaped band. "
                "For stronger strengths, use more separated contours only if Stroma has enough distinct areas."
            ),
            "anti_pattern": (
                "Avoid one large continuous wedge or crescent that converts most available peritumoral Stroma in a single region."
            ),
        }
    if primitive_name == "necrosis_appearance":
        return {
            "recommended_region_count_range": [1, 2],
            "points_per_region_range": [16, 34],
            "boundary_style": "irregular necrotic blob or fissure-like contour",
            "region_variation_requirement": (
                "If multiple foci are used, make them visibly different; do not duplicate shapes."
            ),
            "connectivity_guidance": "Prefer one connected intratumoral necrosis region unless multiple foci are needed.",
        }
    return {
        "recommended_region_count_range": [1, 3],
        "points_per_region_range": [16, 32],
        "boundary_style": "irregular organic boundary",
        "region_variation_requirement": "Do not duplicate shapes across regions.",
    }


def _area_requirement_text(target_area_hint: Mapping[str, Any] | None) -> str:
    if not target_area_hint:
        return "Follow the strength bucket from the recipe if present."
    return (
        "After Phase 3 deterministically projects the coarse template to legal source-label pixels, "
        f"the changed area should be about {target_area_hint.get('target_changed_pixels_min')} "
        f"to {target_area_hint.get('target_changed_pixels_max')} pixels "
        f"({target_area_hint.get('target_fraction_min'):.2f}-"
        f"{target_area_hint.get('target_fraction_max'):.2f} of "
        f"{target_area_hint.get('reference_area_label')}). "
        "The raw polygon is only a coarse template; deterministic projection controls final area."
    )


def _task_requirements_instruction(context: Mapping[str, Any]) -> str:
    requirements = context.get("llm_task_requirements")
    if not requirements:
        return "Follow the primitive recipe constraints in the context."
    return (
        "Follow llm_task_requirements exactly: respect where_to_draw, "
        "shape_style, and area_requirement. These summarize the recipe for this primitive."
    )


def _contour_style_instruction(context: Mapping[str, Any]) -> str:
    hint = context.get("contour_style_hint")
    if not isinstance(hint, Mapping):
        return "Use enough contour points to avoid simple geometric shapes."
    return (
        "Follow contour_style_hint: use the recommended region count, "
        "points_per_region_range, and boundary_style. If multiple regions are used, "
        "their shapes must be visibly different; never output identical diamonds or repeated blobs."
    )


def _build_source_spatial_hints(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
    grid_spacing_px: int,
) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for label in source_labels:
        ids = schema.resolve_fine_ids(label)
        source_mask = np.isin(mask, ids)
        components = _connected_component_summary(source_mask)
        tiles = _source_grid_tiles(source_mask, grid_spacing_px=grid_spacing_px)
        hints[label] = {
            "total_pixels": int(np.count_nonzero(source_mask)),
            "components": components,
            "high_purity_grid_tiles": tiles,
            "tile_note": (
                "Each tile is [x0, y0, x1, y1] in original coordinates; "
                "source_fraction is the fraction of pixels in that tile with this source label."
            ),
        }
    return hints


def _build_source_contour_context(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
    max_components_per_label: int = 6,
    max_contour_points: int = 96,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for label in source_labels:
        ids = schema.resolve_fine_ids(label)
        source_mask = np.isin(mask, ids)
        components = _component_contour_context(
            mask,
            source_mask,
            schema=schema,
            max_components=max_components_per_label,
            max_contour_points=max_contour_points,
        )
        context[label] = {
            "coordinate_system": "original_mask_xy",
            "contour_point_format": "[x, y]",
            "components": components,
            "usage_note": (
                "Use these source-label contours as geometry references. "
                "Adjacent tissue tells what lies just outside the source boundary."
            ),
        }
    return context


def _component_contour_context(
    mask: np.ndarray,
    source_mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    max_components: int,
    max_contour_points: int,
) -> list[dict[str, Any]]:
    from scipy import ndimage
    from skimage import measure

    labeled, count = ndimage.label(source_mask)
    components: list[dict[str, Any]] = []
    for component_id in range(1, count + 1):
        component = labeled == component_id
        area = int(np.count_nonzero(component))
        if area == 0:
            continue
        ys, xs = np.where(component)
        contours = measure.find_contours(component.astype(float), level=0.5)
        contour_xy: list[list[int]] = []
        if contours:
            contour = max(contours, key=len)
            contour_xy = _simplify_contour_points(
                [[float(point[1]), float(point[0])] for point in contour],
                max_points=max_contour_points,
            )
        components.append(
            {
                "component_id": f"source_{component_id}",
                "area_px": area,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                "contour_simplified": contour_xy,
                "adjacent_tissue": _adjacent_tissue_summary(mask, component, schema=schema),
                "contour_adjacency_segments": _contour_adjacency_segments(
                    mask,
                    _boundary_points_for_adjacency(component, contour_xy),
                    component,
                    schema=schema,
                ),
            }
        )
    components.sort(key=lambda item: int(item["area_px"]), reverse=True)
    return components[:max_components]


def _simplify_contour_points(
    points: Sequence[Sequence[float]],
    *,
    max_points: int,
) -> list[list[int]]:
    if not points:
        return []
    if len(points) <= max_points:
        selected = points
    else:
        indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
        selected = [points[int(index)] for index in indices]
    simplified: list[list[int]] = []
    previous: tuple[int, int] | None = None
    for point in selected:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        current = (x, y)
        if current != previous:
            simplified.append([x, y])
            previous = current
    return simplified


def _adjacent_tissue_summary(
    mask: np.ndarray,
    component: np.ndarray,
    *,
    schema: MaskProfileSchema,
) -> list[dict[str, Any]]:
    from scipy import ndimage

    ring = ndimage.binary_dilation(component, structure=np.ones((3, 3))) & ~component
    labels, counts = np.unique(mask[ring], return_counts=True)
    summary: list[dict[str, Any]] = []
    for label_id, count in zip(labels.tolist(), counts.tolist()):
        label_name = _label_name_for_id(int(label_id), schema)
        summary.append(
            {
                "label": label_name,
                "fine_id": int(label_id),
                "boundary_neighbor_pixels": int(count),
            }
        )
    summary.sort(key=lambda item: int(item["boundary_neighbor_pixels"]), reverse=True)
    return summary[:8]


def _contour_adjacency_segments(
    mask: np.ndarray,
    contour_xy: Sequence[Sequence[int]],
    component: np.ndarray,
    *,
    schema: MaskProfileSchema,
    max_segments_per_label: int = 8,
) -> dict[str, Any]:
    if not contour_xy:
        return {}
    labels_by_point: list[str] = []
    usable_points: list[list[int]] = []
    for point in contour_xy:
        x, y = int(point[0]), int(point[1])
        if _touches_patch_edge(mask.shape, x=x, y=y, radius=2):
            continue
        usable_points.append([x, y])
        labels_by_point.append(
            _dominant_adjacent_label_at_point(mask, component, x=x, y=y, schema=schema)
        )
    if not usable_points:
        return {
            "_meta": {
                "patch_edge_points_ignored": True,
                "reason": "all candidate adjacency points touched the patch edge",
            }
        }

    segments_by_label: dict[str, list[dict[str, Any]]] = {}
    start = 0
    current = labels_by_point[0]
    for index in range(1, len(labels_by_point) + 1):
        is_end = index == len(labels_by_point)
        if is_end or labels_by_point[index] != current:
            if current != "Source":
                segment = _segment_from_points(usable_points, start, index - 1)
                segments_by_label.setdefault(current, []).append(segment)
            if not is_end:
                start = index
                current = labels_by_point[index]

    compact: dict[str, Any] = {}
    for label, segments in segments_by_label.items():
        compact[label] = segments[:max_segments_per_label]
    return compact


def _touches_patch_edge(
    shape: tuple[int, int],
    *,
    x: int,
    y: int,
    radius: int,
) -> bool:
    height, width = shape
    return x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height


def _boundary_points_for_adjacency(
    component: np.ndarray,
    contour_xy: Sequence[Sequence[int]],
    *,
    max_points: int = 192,
) -> list[list[int]]:
    from scipy import ndimage

    eroded = ndimage.binary_erosion(component, structure=np.ones((3, 3)), border_value=0)
    boundary = component & ~eroded
    ys, xs = np.where(boundary)
    if len(xs) == 0:
        return [list(point) for point in contour_xy]
    points = [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
    return [points[int(index)] for index in indices]


def _dominant_adjacent_label_at_point(
    mask: np.ndarray,
    component: np.ndarray,
    *,
    x: int,
    y: int,
    schema: MaskProfileSchema,
) -> str:
    height, width = mask.shape
    x0, x1 = max(x - 2, 0), min(x + 3, width)
    y0, y1 = max(y - 2, 0), min(y + 3, height)
    local_component = component[y0:y1, x0:x1]
    local_mask = mask[y0:y1, x0:x1]
    outside = ~local_component
    if not np.any(outside):
        return "Source"
    labels, counts = np.unique(local_mask[outside], return_counts=True)
    if len(labels) == 0:
        return "Source"
    order = np.argsort(counts)[::-1]
    for idx in order:
        label_id = int(labels[int(idx)])
        label_name = _label_name_for_id(label_id, schema)
        if label_name != "Source":
            return label_name
    return "Source"


def _segment_from_points(
    contour_xy: Sequence[Sequence[int]],
    start: int,
    end: int,
) -> dict[str, Any]:
    points = [list(contour_xy[idx]) for idx in range(start, end + 1)]
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    return {
        "point_index_range": [int(start), int(end)],
        "points": points,
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "point_count": len(points),
    }


def _label_name_for_id(label_id: int, schema: MaskProfileSchema) -> str:
    for label, ids in schema.label_to_fine_ids.items():
        if label_id in ids:
            return label
    if label_id in schema.skip_fine_ids:
        return "Background"
    return f"fine_id_{label_id}"


def _connected_component_summary(source_mask: np.ndarray) -> list[dict[str, Any]]:
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover - scipy is already used by validation.
        return []

    labeled, count = ndimage.label(source_mask)
    components: list[dict[str, Any]] = []
    for component_id in range(1, count + 1):
        ys, xs = np.where(labeled == component_id)
        area = int(len(xs))
        if area <= 0:
            continue
        components.append(
            {
                "component_id": f"c{component_id}",
                "area_px": area,
                "bbox": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ],
                "centroid": [
                    round(float(xs.mean()), 1),
                    round(float(ys.mean()), 1),
                ],
            }
        )
    components.sort(key=lambda item: int(item["area_px"]), reverse=True)
    return components[:8]


def _source_grid_tiles(
    source_mask: np.ndarray,
    *,
    grid_spacing_px: int,
) -> list[dict[str, Any]]:
    height, width = source_mask.shape
    tiles: list[dict[str, Any]] = []
    spacing = max(int(grid_spacing_px), 1)
    for y0 in range(0, height, spacing):
        for x0 in range(0, width, spacing):
            y1_excl = min(y0 + spacing, height)
            x1_excl = min(x0 + spacing, width)
            tile = source_mask[y0:y1_excl, x0:x1_excl]
            tile_pixels = int(tile.size)
            source_pixels = int(np.count_nonzero(tile))
            if source_pixels == 0:
                continue
            source_fraction = source_pixels / tile_pixels
            if source_fraction < 0.10 and source_pixels < 256:
                continue
            tiles.append(
                {
                    "bbox": [x0, y0, x1_excl - 1, y1_excl - 1],
                    "source_pixels": source_pixels,
                    "source_fraction": round(source_fraction, 3),
                }
            )
    tiles.sort(
        key=lambda item: (float(item["source_fraction"]), int(item["source_pixels"])),
        reverse=True,
    )
    return tiles[:16]


def _visual_label_legend() -> dict[str, str]:
    return {
        "Background": "dark gray / nearly black",
        "Tumor": "red",
        "Stroma": "green",
        "Necrosis": "blue",
        "Immune infiltrate": "purple",
        "Normal epithelium": "yellow",
        "Blood vessel": "teal",
        "Other tissue": "gray",
    }


def _source_label_visual_instruction(context: Mapping[str, Any]) -> str:
    allowed = context.get("allowed_source_labels", [])
    target = context.get("target_label", "")
    legend = context.get("visual_label_legend", {})
    if not isinstance(allowed, list) or not allowed:
        return "Place polygons mostly on the allowed source-label colors in the grid image."
    source_descriptions = []
    for label in allowed:
        color = legend.get(label, "its legend color") if isinstance(legend, Mapping) else "its legend color"
        source_descriptions.append(f"{label} ({color})")
    forbidden_examples = []
    for label in ("Tumor", "Necrosis", "Other tissue", "Immune infiltrate"):
        if label not in allowed and isinstance(legend, Mapping) and label in legend:
            forbidden_examples.append(f"{label} ({legend[label]})")
    message = (
        "In the grid image, place each coarse template mainly over allowed source tissue: "
        + ", ".join(source_descriptions)
        + f". These pixels will be converted to {target}. "
        "The template does not need pixel-perfect vertices on source pixels; deterministic projection will keep only legal source-label pixels. "
        "Use source_contour_context as the drawing guide for component choice, adjacency side, and approximate placement."
    )
    if forbidden_examples:
        message += (
            " Avoid drawing mainly on "
            + ", ".join(forbidden_examples)
            + " because those pixels will be removed by projection."
        )
    return message


def _organic_shape_instruction(context: Mapping[str, Any]) -> str:
    primitive = context.get("primitive")
    if primitive == "stromal_immune_infiltration":
        return (
            "Shape style: propose a natural, pathology-like irregular stromal immune coarse template. "
            "For mild strength, prefer multiple patchy organic contours instead of one broad continuous band. "
            "The intended placement should be on the Stroma side of the Tumor-Stroma interface, especially Tumor-adjacent Stroma. "
            "The template may be rough near local boundaries because legal projection will keep final pixels inside Stroma. "
            "Use many boundary points, following contour_style_hint, for each substantial region and follow the green stromal compartment and nearby tissue boundaries. "
            "Avoid rectangles, diamonds, circles, symmetric shapes, repeated duplicate parts, tiny decorative polygons, and one large wedge-shaped band. "
            "If validation feedback says the area is too small, enlarge several existing patches or add a differently shaped patch; do not add an identical copy."
        )
    if primitive == "necrosis_appearance":
        return (
            "Shape style: propose an irregular intratumoral necrosis contour with uneven pathology-like boundaries. "
            "Prefer one organic contour unless multiple necrotic foci are explicitly needed. "
            "Use many boundary points, following contour_style_hint, for each substantial region. "
            "Avoid rectangles, diamonds, circles, symmetric shapes, repeated duplicate parts, and tiny decorative polygons."
        )
    return (
        "Shape style: use organic irregular polygon boundaries with enough points to avoid simple geometric templates. "
        "Avoid rectangles, diamonds, symmetric shapes, and repeated duplicate parts."
    )


def _default_placement_relation(context: Mapping[str, Any]) -> str:
    primitive = context.get("primitive")
    if primitive == "stromal_immune_infiltration":
        return "tumor_adjacent_stroma"
    if primitive == "necrosis_appearance":
        return "tumor_interior"
    return "generic_label_safe"


def _default_shape_hints(context: Mapping[str, Any]) -> list[str]:
    primitive = context.get("primitive")
    if primitive == "stromal_immune_infiltration":
        return ["patchy", "band_like", "irregular_boundary"]
    if primitive == "necrosis_appearance":
        return ["patchy", "irregular_boundary"]
    return ["irregular_boundary"]
