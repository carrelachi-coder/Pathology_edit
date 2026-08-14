"""Specialized Breast cord/nest tissue candidates.

Cord formation is deliberately cell-first: a source-scaled chain of virtual
tumor-cell footprints is laid out in operational Stroma and the ordinary Tumor
support is derived from those footprints at cell scale.  Nest formation is the
inverse tissue-first program: one detached irregular Tumor island is created in
the certified peritumoral band and the existing cell executor fills it later.
No new annotation channel is introduced.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import (
    CandidateMask,
    EditPlan,
    ResolvedAreaContract,
)
from phase3_mask_edit_refine.scene import SceneAnalysis
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

from .scene import JointSceneAnalysis

CORD_PRIMITIVE_ID = "invasive-cord-formation-v1"
NEST_PRIMITIVE_ID = "peritumoral-tumor-nest-formation-v1"
SPECIALIZED_ARCHITECTURE_PRIMITIVES = frozenset(
    {CORD_PRIMITIVE_ID, NEST_PRIMITIVE_ID}
)
ARCHITECTURE_EXECUTOR_VERSION = "breast-invasive-architecture-v1"


def compile_joint_tissue_plan_with_witness(
    plan: EditPlan,
    *,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
    scene: SceneAnalysis,
):
    """Compile area authority without forcing specialized shapes through a
    generic interface-front raster solver.

    The actual specialized candidate and all ordinary tissue gates remain the
    executable witness.  This step freezes only the immutable pixel target.
    """

    if plan.primitive_id not in SPECIALIZED_ARCHITECTURE_PRIMITIVES:
        from phase3_mask_edit_refine.execution import compile_edit_plan_with_witness

        return compile_edit_plan_with_witness(
            plan,
            source_mask=source_mask,
            schema=schema,
            scene=scene,
        )
    source_region = np.isin(source_mask, _source_fine_ids(plan, schema))
    desired = int(plan.area_budget.target_pixels(source_mask, source_region))
    hard_min, hard_max = plan.area_budget.hard_pixel_interval(
        source_mask, source_region
    )
    resolved = min(max(desired, hard_min), hard_max)
    from dataclasses import replace

    compiled = replace(
        plan,
        resolved_area=ResolvedAreaContract(
            desired_pixels=desired,
            hard_min_pixels=hard_min,
            hard_max_pixels=hard_max,
            resolved_pixels=resolved,
            fallback_policy=plan.area_budget.fallback_policy,
            used_fallback=resolved < desired,
            binding_constraint="specialized_architecture_pixel_target",
            solver_version=ARCHITECTURE_EXECUTOR_VERSION,
        ),
    )
    return (
        compiled,
        {
            "compiler": ARCHITECTURE_EXECUTOR_VERSION,
            "resolved_pixels": resolved,
            "geometry_deferred_to_specialized_executor": True,
        },
        (),
        {"compiler": ARCHITECTURE_EXECUTOR_VERSION},
    )


def generate_joint_tissue_candidates(
    source_tissue: np.ndarray,
    *,
    schema: MaskProfileSchema,
    tissue_scene: SceneAnalysis,
    joint_scene: JointSceneAnalysis,
    plan: EditPlan,
    bundle: ActiveKnowledgeBundle,
    seed: int,
    candidate_limit: int | None = None,
    compiled_replay_parts=None,
    compiled_replay_audit=None,
) -> tuple[CandidateMask, ...]:
    """Dispatch specialized architectures and preserve the generic backend."""

    if plan.primitive_id not in SPECIALIZED_ARCHITECTURE_PRIMITIVES:
        from phase3_mask_edit_refine.candidates import generate_candidates

        return generate_candidates(
            source_tissue,
            schema=schema,
            scene=tissue_scene,
            plan=plan,
            bundle=bundle,
            seed=seed,
            candidate_limit=candidate_limit,
            compiled_replay_parts=compiled_replay_parts,
            compiled_replay_audit=compiled_replay_audit,
        )
    limit = max(1, min(8, int(candidate_limit or plan.tool_program.candidate_count)))
    generated = []
    for variant in range(limit):
        if plan.primitive_id == CORD_PRIMITIVE_ID:
            candidate = _cell_seeded_cord_candidate(
                source_tissue,
                schema=schema,
                tissue_scene=tissue_scene,
                joint_scene=joint_scene,
                plan=plan,
                seed=seed,
                variant=variant,
            )
        else:
            candidate = _detached_nest_candidate(
                source_tissue,
                schema=schema,
                tissue_scene=tissue_scene,
                joint_scene=joint_scene,
                plan=plan,
                seed=seed,
                variant=variant,
            )
        if candidate is not None:
            generated.append(candidate)
    return tuple(generated)


def _cell_seeded_cord_candidate(
    source_tissue,
    *,
    schema,
    tissue_scene,
    joint_scene,
    plan,
    seed,
    variant,
):
    planned = plan.candidate_interfaces[variant % len(plan.candidate_interfaces)]
    source_component = tissue_scene.component_masks.get(planned.source_component_id)
    target_component = tissue_scene.component_masks.get(planned.target_component_id)
    anchor = _planned_anchor_mask(tissue_scene, planned)
    if source_component is None or target_component is None or not np.any(anchor):
        return None
    source_ids = _source_fine_ids(plan, schema)
    legal = np.asarray(source_component, dtype=bool) & np.isin(
        source_tissue, source_ids
    )
    legal &= ~_prohibited_union(tissue_scene, legal.shape)
    # A connected cord must remain an internal invasion structure.  If its
    # derived support reaches the raster edge it opens the enclosing Stromal
    # component and becomes a patch-border artifact rather than a cord.
    legal[[0, -1], :] = False
    legal[:, [0, -1]] = False
    diameter = _nominal_neoplastic_diameter(joint_scene)
    target_pixels = _resolved_pixels(plan, source_tissue, legal)
    if target_pixels <= 0:
        return None

    anchor_point = _anchor_point(anchor, legal)
    if anchor_point is None:
        return None
    direction = _outward_direction(anchor_point, target_component)
    maximum_depth = max(1.0, float(planned.allowed_edit_band_px[1]))
    centers = _cord_centers(
        anchor_point=anchor_point,
        direction=direction,
        legal=legal,
        diameter=diameter,
        maximum_depth=maximum_depth,
        target_pixels=target_pixels,
        seed=seed,
        variant=variant,
    )
    if len(centers) < 5:
        return None
    footprints = np.zeros_like(legal)
    nucleus_radius = max(2, int(round(0.42 * diameter)))
    for row, col in centers:
        _paint_disk(footprints, row, col, nucleus_radius)
    footprints &= legal
    support_radius = max(1, int(round(0.30 * diameter)))
    support = ndimage.binary_dilation(footprints, iterations=support_radius) & legal
    support = ndimage.binary_closing(
        support, structure=np.ones((3, 3), dtype=bool), iterations=1
    ) & legal
    support = _component_nearest_point(support, anchor_point)
    if not np.any(support):
        return None
    envelope = (
        ndimage.binary_dilation(
            footprints,
            iterations=support_radius + max(3, int(round(0.45 * diameter))),
        )
        & legal
    )
    support = _grow_connected_to_size(
        support,
        envelope=envelope,
        target_pixels=target_pixels,
        seed=seed + variant * 1009,
    )
    if support is None or not _touches(support, target_component):
        return None
    if _component_count(support) != 1:
        return None
    target = np.asarray(source_tissue).copy()
    target_id = _target_fine_id(plan, schema)
    target[support] = target_id
    return CandidateMask(
        candidate_id=_candidate_id(plan, "cord", variant, support),
        interface_id=planned.interface_id,
        tool_name="cell_seeded_cord",
        target_mask=target,
        change_region=support,
        tool_trace={
            "seed": int(seed),
            "variant": int(variant),
            "interface_id": planned.interface_id,
            "interface_ids": [item.interface_id for item in plan.candidate_interfaces],
            "source_component_id": planned.source_component_id,
            "source_component_ids": [
                item.source_component_id for item in plan.candidate_interfaces
            ],
            "target_component_id": planned.target_component_id,
            "target_component_ids": [
                item.target_component_id for item in plan.candidate_interfaces
            ],
            "target_fine_id": int(target_id),
            "tool_adapter_version": ARCHITECTURE_EXECUTOR_VERSION,
            "execution_order": "cells_then_tumor_mask",
            "tumor_mask_derivation": (
                "cell_footprints_plus_cell_scale_closing"
            ),
            "cell_seed_centers_yx": [[int(y), int(x)] for y, x in centers],
            "nominal_nucleus_diameter_px": float(diameter),
            "nucleus_footprint_radius_px": int(nucleus_radius),
            "support_closing_radius_px": int(support_radius),
            "support_width_policy": "one_to_three_cells_variable",
            "path_policy": "slightly_curved_source_scaled_invasion_path",
            "changed_pixels": int(np.count_nonzero(support)),
        },
    )


def _detached_nest_candidate(
    source_tissue,
    *,
    schema,
    tissue_scene,
    joint_scene,
    plan,
    seed,
    variant,
):
    planned = plan.candidate_interfaces[variant % len(plan.candidate_interfaces)]
    source_component = tissue_scene.component_masks.get(planned.source_component_id)
    interface = tissue_scene.interface_masks.get(planned.interface_id)
    anchor = _planned_anchor_mask(tissue_scene, planned)
    if source_component is None or interface is None or not np.any(anchor):
        return None
    source_ids = _source_fine_ids(plan, schema)
    legal = np.asarray(source_component, dtype=bool) & np.isin(
        source_tissue, source_ids
    )
    legal &= ~_prohibited_union(tissue_scene, legal.shape)
    legal[[0, -1], :] = False
    legal[:, [0, -1]] = False
    diameter = _nominal_neoplastic_diameter(joint_scene)
    target_pixels = _resolved_pixels(plan, source_tissue, legal)
    equivalent_radius = float(np.sqrt(target_pixels / np.pi))
    # A nest is a small island, not a displaced bulk compartment.  Excessive
    # area therefore fails closed instead of inflating the island.
    if equivalent_radius > 4.5 * diameter:
        return None
    tumor = np.isin(source_tissue, tuple(schema.resolve_fine_ids("Tumor")))
    distance_to_tumor = ndimage.distance_transform_edt(~tumor)
    distance_to_interface = ndimage.distance_transform_edt(~interface)
    minimum_gap = max(2.0, 0.65 * diameter)
    maximum_band = max(1.0, float(planned.allowed_edit_band_px[1]))
    center_domain = (
        legal
        & (distance_to_tumor >= minimum_gap + equivalent_radius)
        & (distance_to_interface <= maximum_band - equivalent_radius)
    )
    clearance = ndimage.distance_transform_edt(legal)
    center_domain &= clearance >= max(2.0, 0.9 * equivalent_radius)
    if not np.any(center_domain):
        return None
    anchor_distance = ndimage.distance_transform_edt(~anchor)
    # Stay near the invasive front once the whole island and its parent gap
    # fit.  Maximizing raw stromal clearance pushed nests toward the middle of
    # the patch and turned a peritumoral edit into a remote focus.
    score = -distance_to_interface + 0.02 * clearance - 0.01 * anchor_distance
    coords = np.argwhere(center_domain)
    values = score[center_domain]
    order = np.argsort(-values)
    choice = coords[order[min(variant, len(order) - 1)]]
    center_y, center_x = (int(value) for value in choice)
    nest = _irregular_island(
        legal=legal,
        center_y=center_y,
        center_x=center_x,
        target_pixels=target_pixels,
        phase=0.73 * (seed + 1) + 1.19 * variant,
    )
    if nest is None or _component_count(nest) != 1:
        return None
    if _touches(nest, tumor):
        return None
    observed_gap = float(np.min(distance_to_tumor[nest]))
    observed_interface_distance = float(
        np.min(distance_to_interface[nest])
    )
    if observed_gap + 1e-6 < minimum_gap:
        return None
    target = np.asarray(source_tissue).copy()
    target_id = _target_fine_id(plan, schema)
    target[nest] = target_id
    return CandidateMask(
        candidate_id=_candidate_id(plan, "nest", variant, nest),
        interface_id=planned.interface_id,
        tool_name="peritumoral_tumor_island",
        target_mask=target,
        change_region=nest,
        tool_trace={
            "seed": int(seed),
            "variant": int(variant),
            "interface_id": planned.interface_id,
            "interface_ids": [item.interface_id for item in plan.candidate_interfaces],
            "source_component_id": planned.source_component_id,
            "source_component_ids": [
                item.source_component_id for item in plan.candidate_interfaces
            ],
            "target_component_id": planned.target_component_id,
            "target_component_ids": [
                item.target_component_id for item in plan.candidate_interfaces
            ],
            "target_fine_id": int(target_id),
            "tool_adapter_version": ARCHITECTURE_EXECUTOR_VERSION,
            "execution_order": "tumor_island_then_cells",
            "island_geometry": "single_detached_irregular_harmonic_blob",
            "island_center_yx": [center_y, center_x],
            "minimum_interface_distance_px": observed_interface_distance,
            "minimum_parent_tumor_gap_px": float(observed_gap),
            "nominal_nucleus_diameter_px": float(diameter),
            "changed_pixels": int(np.count_nonzero(nest)),
        },
    )


def _cord_centers(
    *,
    anchor_point,
    direction,
    legal,
    diameter,
    maximum_depth,
    target_pixels,
    seed,
    variant,
):
    rng = np.random.default_rng(seed + 7919 * variant)
    step = max(3.0, 0.72 * diameter)
    estimated_support_per_cell = np.pi * (0.72 * diameter) ** 2 * 0.38
    requested = int(np.clip(np.ceil(target_pixels / estimated_support_per_cell), 7, 34))
    nodes = int(np.ceil(requested / 1.6))
    nodes = min(nodes, max(5, int(maximum_depth // step)))
    if nodes < 5:
        return []
    tangent = np.asarray((-direction[1], direction[0]), dtype=float)
    sign = -1.0 if variant % 2 else 1.0
    centers: list[tuple[int, int]] = []
    for index in range(nodes):
        progress = (index + 0.55) / max(nodes, 1)
        distance = (index + 0.55) * step
        curve = sign * (0.16 + 0.03 * (variant % 3)) * diameter
        curve *= np.sin(np.pi * progress) * (0.45 + progress)
        base = np.asarray(anchor_point, dtype=float) + direction * distance + tangent * curve
        lane_count = 1 + int(index % 3 != 1 and len(centers) + 1 < requested)
        lane_offsets = (0.0,) if lane_count == 1 else (-0.42 * diameter, 0.42 * diameter)
        for lane in lane_offsets:
            jitter = tangent * rng.normal(0.0, 0.08 * diameter)
            proposed = base + tangent * lane + jitter
            snapped = _nearest_legal(proposed, legal, radius=max(2, int(0.45 * diameter)))
            if snapped is not None and snapped not in centers:
                centers.append(snapped)
            if len(centers) >= requested:
                break
        if len(centers) >= requested:
            break
    return centers


def _irregular_island(*, legal, center_y, center_x, target_pixels, phase):
    rows, cols = np.indices(legal.shape)
    dy = rows - center_y
    dx = cols - center_x
    theta = np.arctan2(dy, dx)
    axis_ratio = 1.18
    radial = np.sqrt((dy * axis_ratio) ** 2 + (dx / axis_ratio) ** 2)
    modulation = (
        1.0
        + 0.13 * np.sin(3.0 * theta + phase)
        + 0.07 * np.sin(5.0 * theta - 0.61 * phase)
    )
    priority = radial / np.maximum(modulation, 0.65)
    candidates = np.argwhere(legal)
    if len(candidates) < target_pixels:
        return None
    values = priority[legal]
    order = np.argsort(values, kind="stable")[:target_pixels]
    island = np.zeros_like(legal)
    chosen = candidates[order]
    island[chosen[:, 0], chosen[:, 1]] = True
    component = _component_nearest_point(island, (center_y, center_x))
    if int(np.count_nonzero(component)) != target_pixels:
        return None
    if np.any(ndimage.binary_fill_holes(component) & ~component):
        return None
    return component


def _grow_connected_to_size(mask, *, envelope, target_pixels, seed):
    current = np.asarray(mask, dtype=bool).copy()
    if int(np.count_nonzero(current)) > target_pixels:
        return None
    rng = np.random.default_rng(seed)
    while int(np.count_nonzero(current)) < target_pixels:
        frontier = ndimage.binary_dilation(current) & envelope & ~current
        coords = np.argwhere(frontier)
        if not len(coords):
            return None
        remaining = target_pixels - int(np.count_nonzero(current))
        if len(coords) > remaining:
            order = np.argsort(rng.random(len(coords)))[:remaining]
            coords = coords[order]
        current[coords[:, 0], coords[:, 1]] = True
    return current


def _planned_anchor_mask(scene, planned):
    masks = tuple(
        scene.anchor_masks[item]
        for item in planned.execution_contract.anchor_segment_ids
        if item in scene.anchor_masks
    )
    if not masks:
        return np.zeros(next(iter(scene.interface_masks.values())).shape, dtype=bool)
    return np.logical_or.reduce(masks)


def _anchor_point(anchor, legal):
    candidates = np.argwhere(anchor & ndimage.binary_dilation(legal))
    if not len(candidates):
        candidates = np.argwhere(anchor)
    if not len(candidates):
        return None
    clearance = ndimage.distance_transform_edt(legal)
    values = clearance[candidates[:, 0], candidates[:, 1]]
    point = candidates[int(np.argmax(values))]
    return int(point[0]), int(point[1])


def _outward_direction(anchor_point, target_component):
    target_coords = np.argwhere(target_component)
    center = target_coords.mean(axis=0)
    direction = np.asarray(anchor_point, dtype=float) - center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return np.asarray((0.0, 1.0), dtype=float)
    return direction / norm


def _nearest_legal(point, legal, *, radius):
    row, col = (int(round(value)) for value in point)
    y0, y1 = max(0, row - radius), min(legal.shape[0], row + radius + 1)
    x0, x1 = max(0, col - radius), min(legal.shape[1], col + radius + 1)
    coords = np.argwhere(legal[y0:y1, x0:x1])
    if not len(coords):
        return None
    coords = coords + np.asarray((y0, x0))
    distance = np.sum((coords - np.asarray(point)) ** 2, axis=1)
    chosen = coords[int(np.argmin(distance))]
    return int(chosen[0]), int(chosen[1])


def _paint_disk(canvas, row, col, radius):
    y0, y1 = max(0, row - radius), min(canvas.shape[0], row + radius + 1)
    x0, x1 = max(0, col - radius), min(canvas.shape[1], col + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] |= (yy - row) ** 2 + (xx - col) ** 2 <= radius**2


def _source_fine_ids(plan, schema):
    explicit = tuple(
        int(value)
        for value in plan.tool_program.parameter_ranges.get(
            "editable_source_fine_ids", ()
        )
    )
    if explicit:
        return explicit
    return tuple(
        sorted(
            {
                fine_id
                for label in plan.source_labels
                for fine_id in schema.resolve_fine_ids(label)
            }
        )
    )


def _target_fine_id(plan, schema):
    explicit = tuple(
        int(value)
        for value in plan.tool_program.parameter_ranges.get(
            "editable_target_fine_ids", ()
        )
    )
    return int(explicit[0] if explicit else schema.resolve_fine_ids(plan.target_label)[0])


def _resolved_pixels(plan, source_tissue, source_region):
    if plan.resolved_area is not None:
        return int(plan.resolved_area.resolved_pixels)
    return int(plan.area_budget.target_pixels(source_tissue, source_region))


def _prohibited_union(scene, shape):
    masks: Iterable[np.ndarray] = scene.prohibited_region_masks.values()
    values = tuple(np.asarray(item, dtype=bool) for item in masks)
    return np.logical_or.reduce(values) if values else np.zeros(shape, dtype=bool)


def _nominal_neoplastic_diameter(scene):
    areas = [
        item.area_px
        for item in scene.cells.instances
        if item.class_id == 1
        and item.completeness_status == "complete"
        and not item.touches_border
        and not item.quality_flags
        and item.area_px > 0
    ]
    if areas:
        return max(4.0, 2.0 * float(np.sqrt(np.median(areas) / np.pi)))
    return max(4.0, float(scene.population.nominal_nucleus_diameter_px or 8.0))


def _component_nearest_point(mask, point):
    labeled, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count <= 1:
        return np.asarray(mask, dtype=bool)
    row, col = point
    label = int(labeled[int(row), int(col)])
    if label > 0:
        return labeled == label
    coords = np.argwhere(mask)
    nearest = coords[int(np.argmin(np.sum((coords - np.asarray(point)) ** 2, axis=1)))]
    return labeled == int(labeled[tuple(nearest)])


def _component_count(mask):
    return int(ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))[1])


def _touches(left, right):
    return bool(np.any(ndimage.binary_dilation(left, iterations=1) & right))


def _candidate_id(plan, kind, variant, change):
    digest = hashlib.sha256(np.asarray(change, dtype=np.uint8).tobytes()).hexdigest()
    return f"{kind}-{plan.case_id}-{variant:02d}-{digest[:12]}"


__all__ = [
    "ARCHITECTURE_EXECUTOR_VERSION",
    "CORD_PRIMITIVE_ID",
    "NEST_PRIMITIVE_ID",
    "SPECIALIZED_ARCHITECTURE_PRIMITIVES",
    "compile_joint_tissue_plan_with_witness",
    "generate_joint_tissue_candidates",
]
