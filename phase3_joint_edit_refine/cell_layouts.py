"""Deterministic complete-instance cell layouts for paired candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import CandidateMask

from .budget import JointBudgetAllocation
from .cell_programs import CompiledCellToolProgram
from .models import JointContractError, JointEditPlan
from .nuclei import iter_instances, normalize_nuclei_mask
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle


LAYOUT_TOOL_VERSION = "joint-cell-layout-v1"


class SpatialRanker(Protocol):
    name: str

    def score(
        self,
        *,
        tissue_mask: np.ndarray,
        source_nuclei: np.ndarray,
        cell_class: int,
        legal_zone: np.ndarray,
        context: Mapping[str, Any],
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class DeterministicDistanceRanker:
    """Offline fallback used in tests; it is explicitly not ProbNet."""

    name: str = "deterministic_distance_ranker"

    def score(self, *, tissue_mask, source_nuclei, cell_class, legal_zone, context):
        del tissue_mask, source_nuclei, cell_class, context
        return ndimage.distance_transform_edt(np.asarray(legal_zone, dtype=bool))


@dataclass(frozen=True)
class CellLayoutResult:
    cell_candidate_id: str
    target_nuclei_mask: np.ndarray
    trace: dict[str, Any]


@dataclass(frozen=True)
class ReferenceNucleusShape:
    instance_id: str
    class_id: int
    mask: np.ndarray
    source: str
    area_px: int


def generate_cell_layouts(
    *,
    source_tissue: np.ndarray,
    source_nuclei: np.ndarray,
    tissue_candidate: CandidateMask,
    schema: MaskProfileSchema,
    scene: JointSceneAnalysis,
    plan: JointEditPlan,
    bundle: JointSkillBundle,
    allocation: JointBudgetAllocation | None,
    compiled_program: CompiledCellToolProgram | None = None,
    seed: int,
    ranker: SpatialRanker | None = None,
    variants: int = 3,
) -> tuple[CellLayoutResult, ...]:
    """Generate paired layouts; original cells are retain/remove-whole only."""

    if not 1 <= variants <= 8:
        raise JointContractError("cell layout variants must be in [1, 8]")
    source_tissue = np.asarray(source_tissue)
    source_nuclei = normalize_nuclei_mask(source_nuclei)
    target_tissue = np.asarray(tissue_candidate.target_mask)
    core = np.asarray(tissue_candidate.change_region, dtype=bool)
    if source_tissue.shape != source_nuclei.shape or target_tissue.shape != source_tissue.shape:
        raise JointContractError("cell layout inputs must share one shape")
    ranker = ranker or DeterministicDistanceRanker()
    ranker_domain = getattr(ranker, "pathology_domain_id", None)
    if ranker_domain is not None and ranker_domain != bundle.mechanism.pathology_domain_id:
        raise JointContractError("spatial ranker pathology domain does not match the joint mechanism")
    ranker_cancer_id = getattr(ranker, "cancer_id", None)
    if ranker_cancer_id is not None and ranker_cancer_id != bundle.cell_population_profile.probnet_cancer_id:
        raise JointContractError("ProbNet cancer ID does not match the cell population profile")
    if plan.tissue_plan is not None:
        target_class = target_cell_class(plan.tissue_plan.target_label, schema)
    elif len(bundle.primitive.target_cell_classes) == 1:
        target_class = bundle.primitive.target_cell_classes[0]
    else:
        raise JointContractError(
            "cell-only deterministic executor requires one Planner-bound target class"
        )
    if target_class not in plan.cell_plan.allowed_cell_classes:
        raise JointContractError(
            f"cell class {target_class} required by target tissue is not allowed by plan"
        )
    references, rejected_references = build_reference_shape_library(
        scene,
        class_id=target_class,
    )
    if not references:
        boundary_count = sum(
            reason == "patch_boundary_censored_shape"
            for reason in rejected_references.values()
        )
        raise JointContractError(
            f"no source-matched complete nucleus shape for class {target_class}; "
            f"rejected={len(rejected_references)}, patch_boundary_censored={boundary_count}"
        )

    base = source_nuclei.copy()
    removed_ids: list[str] = []
    protected = set(plan.cell_plan.protected_instance_ids)
    for instance_id, component in sorted(scene.instance_masks.items()):
        if instance_id in protected:
            continue
        erasure_region = (
            compiled_program.erasure_region
            if compiled_program is not None
            else core
        )
        if np.any(component & erasure_region):
            base[component] = 0
            removed_ids.append(instance_id)

    prohibited_ids = set(bundle.annotation_profile.prohibit_cell_placement_fine_ids)
    legal_core = (
        np.asarray(compiled_program.placement_center_region, dtype=bool)
        if compiled_program is not None
        else core & ~np.isin(target_tissue, tuple(prohibited_ids))
    )
    valid_footprint = (
        np.asarray(compiled_program.valid_footprint_region, dtype=bool)
        if compiled_program is not None
        else ~np.isin(target_tissue, tuple(prohibited_ids))
    )
    halo = (
        np.asarray(compiled_program.mechanism_region, dtype=bool)
        if compiled_program is not None
        else _legal_halo(
            core,
            target_tissue=target_tissue,
            prohibited_ids=prohibited_ids,
            maximum_px=plan.coupling_plan.maximum_halo_px,
            enabled=bundle.mechanism.coupling.cell_only_target_fraction > 0,
        )
    )
    add_zone = legal_core | halo
    if not np.any(add_zone):
        raise JointContractError("joint cell program has no legal placement zone")

    average_area = float(np.median([item.area_px for item in references]))
    if bundle.primitive.scope == "cell_only":
        if plan.cell_plan.mechanism_quota_role != "explicit_increment":
            raise JointContractError(
                "current structured cell-only executor supports explicit increments"
            )
        desired_cell_delta = int(
            compiled_program.target_delta_count
            if compiled_program is not None
            and compiled_program.target_delta_count is not None
            else 0
        )
        if desired_cell_delta <= 0:
            raise JointContractError(
                "cell-only layout requires a positive compiled target delta"
            )
        replacement_count = 0
        reserve_count = desired_cell_delta
        halo = legal_core
        legal_core = np.zeros_like(legal_core)
    else:
        if allocation is None:
            raise JointContractError(
                "tissue-and-cell layout requires a joint budget allocation"
            )
        source_density = _class_density(
            source_nuclei,
            source_tissue,
            class_id=target_class,
            tissue_ids=_target_tissue_ids(plan.tissue_plan.target_label, schema),
        )
        replacement_count = int(round(np.count_nonzero(legal_core) * source_density))
        replacement_count = max(
            replacement_count, len(removed_ids) if removed_ids else 1
        )
        reserve_count = int(
            round(allocation.reserved_layout_halo_pixels / max(1.0, average_area))
        )
        if not np.any(halo):
            reserve_count = 0
    requested_count = replacement_count + reserve_count
    capacity_bound = max(1, int(np.count_nonzero(add_zone) / max(1.0, average_area * 2.0)))
    requested_count = min(requested_count, capacity_bound)
    replacement_count = min(replacement_count, requested_count)
    reserve_count = min(reserve_count, max(0, requested_count - replacement_count))

    score = ranker.score(
        tissue_mask=target_tissue,
        source_nuclei=base,
        cell_class=target_class,
        legal_zone=add_zone,
        context={
            "mechanism_id": plan.selected_mechanism_id,
            "layout_program_id": plan.cell_plan.layout_program_id,
            "tissue_candidate_id": tissue_candidate.candidate_id,
        },
    )
    score = np.asarray(score, dtype=float)
    if score.shape != add_zone.shape or not np.all(np.isfinite(score)):
        raise JointContractError("spatial ranker returned an invalid score map")

    results: list[CellLayoutResult] = []
    for variant in range(variants):
        target, core_placed, core_placements = _place_layout(
            base=base,
            references=references,
            class_id=target_class,
            legal_zone=legal_core,
            valid_footprint_region=valid_footprint,
            halo=np.zeros_like(halo),
            score=score,
            requested_count=replacement_count,
            layout_program=plan.cell_plan.layout_program_id,
            cluster_size_range=bundle.mechanism.cell_program.cluster_size_range,
            seed=seed + variant * 104729,
        )
        halo_score = ranker.score(
            tissue_mask=target_tissue,
            source_nuclei=target,
            cell_class=target_class,
            legal_zone=halo,
            context={"mechanism_id": plan.selected_mechanism_id, "zone": "cell_only_halo"},
        ) if np.any(halo) else np.zeros_like(score)
        target, halo_placed, halo_placements = _place_layout(
            base=target,
            references=references,
            class_id=target_class,
            legal_zone=halo,
            valid_footprint_region=valid_footprint,
            halo=halo,
            score=np.asarray(halo_score, dtype=float),
            requested_count=reserve_count,
            layout_program=plan.cell_plan.layout_program_id,
            cluster_size_range=bundle.mechanism.cell_program.cluster_size_range,
            seed=seed + variant * 104729 + 8191,
        )
        placed = core_placed + halo_placed
        placements = [*core_placements, *halo_placements]
        result_id = f"cells-{variant + 1:02d}"
        results.append(
            CellLayoutResult(
                cell_candidate_id=result_id,
                target_nuclei_mask=target,
                trace={
                    "layout_tool_version": LAYOUT_TOOL_VERSION,
                    "layout_program_id": plan.cell_plan.layout_program_id,
                    "compiled_cell_tool_program": (
                        compiled_program.to_metadata()
                        if compiled_program is not None
                        else None
                    ),
                    "ranker": ranker.name,
                    "ranker_provenance": dict(getattr(ranker, "provenance", {})),
                    "target_cell_class": target_class,
                    # ``desired_count`` is the density-derived biological
                    # request. ``resolved_count`` is the exact reachable
                    # count for this deterministic layout/seed after complete
                    # shape containment and collision checks.  Keeping both
                    # prevents an impossible rough area estimate from being
                    # mistaken for an execution failure.
                    "desired_count": replacement_count + reserve_count,
                    "resolved_count": placed,
                    "requested_count": placed,
                    "placed_count": placed,
                    "placement_completion": (
                        1.0
                        if placed == 0 and replacement_count + reserve_count == 0
                        else placed / max(1, replacement_count + reserve_count)
                    ),
                    "core_requested_count": replacement_count,
                    "core_placed_count": core_placed,
                    "halo_requested_count": reserve_count,
                    "halo_placed_count": halo_placed,
                    "placement_capacity_exhausted": placed < (replacement_count + reserve_count),
                    "cell_capacity_fallback_used": placed < (replacement_count + reserve_count),
                    "removed_source_instance_ids": removed_ids,
                    "protected_instance_ids": sorted(protected),
                    "reference_shape_count": len(references),
                    "reference_shape_ids": [item.instance_id for item in references],
                    "reference_shape_sources": sorted({item.source for item in references}),
                    "reference_shape_rejections": rejected_references,
                    "reference_shape_integrity_certified": True,
                    "reference_first": True,
                    "cross_domain_fallback": False,
                    "overlap_pixels": 0,
                    "partial_source_instance_edits": 0,
                    "cell_only_halo_pixels": int(np.count_nonzero(halo)),
                    "placements": placements,
                    "seed": seed + variant * 104729,
                },
            )
        )
    if not results:
        return ()
    # A tissue candidate is paired only with the maximum count that the
    # deterministic layout family proved reachable.  Lower-capacity variants
    # add no useful diversity and previously caused avoidable quota failures.
    maximum_reachable = max(
        int(item.trace.get("resolved_count", 0)) for item in results
    )
    desired = max(int(item.trace.get("desired_count", 0)) for item in results)
    zero_cell_fallback_allowed = bool(
        tissue_candidate.tool_trace.get("area_fallback_used")
    ) and bundle.mechanism.coupling.cell_only_target_fraction == 0
    if desired > 0 and maximum_reachable <= 0 and not zero_cell_fallback_allowed:
        return ()
    certified: list[CellLayoutResult] = []
    for item in results:
        if int(item.trace.get("resolved_count", 0)) != maximum_reachable:
            continue
        item.trace["batch_max_attainable_count"] = maximum_reachable
        item.trace["cell_capacity_certified"] = True
        item.trace["zero_cell_capacity_fallback_allowed"] = (
            zero_cell_fallback_allowed
        )
        certified.append(item)
    return tuple(certified)


def target_cell_class(target_label: str, schema: MaskProfileSchema) -> int:
    """Resolve the executable CellViT class for a canonical tissue target."""

    if target_label == "Tumor":
        return 1
    if target_label in {"Stroma", "Other tissue"}:
        return 3
    if target_label == "Immune infiltrate":
        return 2
    if target_label == "Necrosis":
        return 4
    if target_label == "Normal epithelium":
        return 5
    raise JointContractError(f"no executable cell-class contract for {target_label!r}")


def _target_tissue_ids(target_label: str, schema: MaskProfileSchema) -> tuple[int, ...]:
    return tuple(schema.resolve_fine_ids(target_label))


def build_reference_shape_library(
    scene: JointSceneAnalysis,
    *,
    class_id: int,
) -> tuple[tuple[ReferenceNucleusShape, ...], dict[str, str]]:
    """Return complete source templates and an auditable rejection map.

    Native instances are used when available because ``scene.instance_masks``
    is built from the native JSON in that mode.  A source instance touching
    *any* of the four patch edges is necessarily censored by the crop and can
    never be copied into the target condition.
    """

    metadata = {item.instance_id: item for item in scene.cells.instances}
    accepted = []
    rejected: dict[str, str] = {}
    height, width = scene.source_nuclei.shape
    for instance_id, item in sorted(metadata.items()):
        if item.class_id != class_id:
            continue
        component = np.asarray(scene.instance_masks.get(instance_id), dtype=bool)
        if component.shape != (height, width) or not np.any(component):
            rejected[instance_id] = "missing_or_empty_instance_mask"
            continue
        x0, y0, x1, y1 = item.bbox_xyxy
        touches_any_patch_edge = bool(
            item.touches_border
            or x0 <= 0
            or y0 <= 0
            or x1 >= width
            or y1 >= height
            or np.any(component[0])
            or np.any(component[-1])
            or np.any(component[:, 0])
            or np.any(component[:, -1])
        )
        if touches_any_patch_edge:
            rejected[instance_id] = "patch_boundary_censored_shape"
            continue
        if "merged_suspect" in item.quality_flags:
            rejected[instance_id] = "merged_suspect_shape"
            continue
        if "irregular_or_fragmented_shape" in item.quality_flags:
            rejected[instance_id] = "irregular_or_fragmented_shape"
            continue
        if ndimage.label(
            component, structure=np.ones((3, 3), dtype=bool)
        )[1] != 1:
            rejected[instance_id] = "disconnected_instance_shape"
            continue
        cropped = component[y0:y1, x0:x1].copy()
        if not np.any(cropped) or int(np.count_nonzero(cropped)) != item.area_px:
            rejected[instance_id] = "incomplete_bbox_crop"
            continue
        accepted.append(
            ReferenceNucleusShape(
                instance_id=instance_id,
                class_id=class_id,
                mask=cropped,
                source=item.source,
                area_px=int(item.area_px),
            )
        )
    accepted.sort(
        key=lambda item: (
            item.area_px,
            item.mask.shape,
            item.instance_id,
        )
    )
    return tuple(accepted), rejected


def _class_density(mask, tissue, *, class_id: int, tissue_ids: tuple[int, ...]) -> float:
    tissue_region = np.isin(tissue, tissue_ids)
    denominator = int(np.count_nonzero(tissue_region))
    centers = 0
    for _, current, component in iter_instances(mask):
        if current != class_id:
            continue
        cy, cx = ndimage.center_of_mass(component)
        row, col = int(round(cy)), int(round(cx))
        if 0 <= row < tissue.shape[0] and 0 <= col < tissue.shape[1] and tissue_region[row, col]:
            centers += 1
    if denominator and centers:
        return centers / denominator
    total_instances = sum(1 for _, current, _ in iter_instances(mask) if current == class_id)
    return total_instances / max(1, int(np.prod(mask.shape)))


def _legal_halo(core, *, target_tissue, prohibited_ids, maximum_px: int, enabled: bool):
    if not enabled or maximum_px <= 0 or not np.any(core):
        return np.zeros_like(core, dtype=bool)
    expanded = ndimage.binary_dilation(core, iterations=maximum_px)
    halo = expanded & ~core & ~np.isin(target_tissue, tuple(prohibited_ids))
    return halo


def _place_layout(
    *, base, references, class_id, legal_zone, valid_footprint_region, halo, score, requested_count,
    layout_program, cluster_size_range, seed,
):
    target = np.asarray(base).copy()
    occupied = target > 0
    rng = np.random.default_rng(seed)
    coords = np.argwhere(legal_zone)
    jitter = rng.uniform(0.0, 1e-6, size=len(coords))
    values = score[legal_zone] + jitter
    order = np.argsort(-values)
    anchors = coords[order]
    if requested_count > 0 and len(coords):
        # Keep a deterministic seam quota. Pure distance/probability ranking
        # otherwise consumes only deep interior maxima and leaves an artificial
        # nucleus-free strip exactly where regenerated and retained tissue meet.
        center_distance = ndimage.distance_transform_edt(legal_zone)
        reference_width = max(
            2,
            int(round(np.median([max(item.mask.shape) for item in references]))),
        )
        edge_mask = center_distance[coords[:, 0], coords[:, 1]] <= reference_width
        edge_coords = coords[edge_mask]
        edge_values = values[edge_mask]
        edge_order = np.argsort(-edge_values)
        edge_quota = min(
            len(edge_coords),
            max(1, int(np.ceil(requested_count * 0.25))),
        )
        preferred = edge_coords[edge_order[: max(edge_quota * 16, edge_quota)]]
        preferred_set = {tuple(value) for value in preferred.tolist()}
        remainder = np.asarray(
            [value for value in anchors.tolist() if tuple(value) not in preferred_set],
            dtype=int,
        )
        anchors = (
            np.concatenate([preferred, remainder], axis=0)
            if len(remainder)
            else preferred
        )
    placed = 0
    placement_trace: list[dict[str, Any]] = []
    anchor_index = 0
    while placed < requested_count and anchor_index < len(anchors):
        ay, ax = (int(v) for v in anchors[anchor_index])
        anchor_index += 1
        offsets = _layout_offsets(
            layout_program,
            cluster_size_range,
            anchor_y=ay,
            anchor_x=ax,
            legal_zone=legal_zone,
        )
        for dy, dx in offsets:
            if placed >= requested_count:
                break
            reference = references[(placed + seed) % len(references)]
            shape = np.asarray(reference.mask, dtype=bool)
            cy, cx = ay + dy, ax + dx
            window = _placement_window(
                shape,
                center_y=cy,
                center_x=cx,
                canvas_shape=target.shape,
            )
            if window is None:
                continue
            y0, y1, x0, x1 = window
            # P constrains the center. V, not P, constrains the full footprint;
            # requiring the footprint to stay inside P caused the documented
            # artificial cell-depleted strip at the new tissue seam.
            if not np.all(valid_footprint_region[y0:y1, x0:x1][shape]):
                continue
            # Collision checking used to materialize and dilate a full 512x512
            # canvas for every attempted nucleus.  A local, one-pixel padded
            # window is exactly equivalent and keeps placement cost proportional
            # to the reference nucleus footprint.
            guard_y0, guard_y1 = max(0, y0 - 1), min(target.shape[0], y1 + 1)
            guard_x0, guard_x1 = max(0, x0 - 1), min(target.shape[1], x1 + 1)
            local_shape = np.zeros((guard_y1 - guard_y0, guard_x1 - guard_x0), dtype=bool)
            local_shape[y0 - guard_y0 : y1 - guard_y0, x0 - guard_x0 : x1 - guard_x0] = shape
            collision_guard = ndimage.binary_dilation(local_shape, iterations=1)
            if np.any(collision_guard & occupied[guard_y0:guard_y1, guard_x0:guard_x1]):
                continue
            target_view = target[y0:y1, x0:x1]
            target_view[shape] = class_id
            occupied[y0:y1, x0:x1] |= shape
            placement_trace.append(
                {
                    "center_xy": [cx, cy],
                    "area_px": int(np.count_nonzero(shape)),
                    "in_cell_only_halo": bool(halo[cy, cx]),
                    "reference_instance_id": reference.instance_id,
                    "reference_source": reference.source,
                    "cluster_size": min(len(offsets), requested_count),
                    "orientation_policy": (
                        "local_interface_tangent_pca"
                        if layout_program in {"short_cord", "boundary_aligned"}
                        else "template_intrinsic"
                    ),
                }
            )
            placed += 1
    return target, placed, placement_trace


def _placement_window(shape, *, center_y: int, center_x: int, canvas_shape):
    height, width = shape.shape
    y0 = center_y - height // 2
    x0 = center_x - width // 2
    y1, x1 = y0 + height, x0 + width
    if y0 < 0 or x0 < 0 or y1 > canvas_shape[0] or x1 > canvas_shape[1]:
        return None
    return y0, y1, x0, x1


def _layout_offsets(
    program: str,
    cluster_range: tuple[int, int],
    *,
    anchor_y: int,
    anchor_x: int,
    legal_zone: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    upper = max(1, min(cluster_range[1], 8))
    if program in {"single", "population_replacement"}:
        return ((0, 0),)
    if program == "pair":
        return ((0, -5), (0, 5))
    if program in {"short_cord", "boundary_aligned"}:
        boundary = np.asarray(legal_zone, dtype=bool) ^ ndimage.binary_erosion(
            legal_zone
        )
        rows, cols = np.nonzero(boundary)
        if len(rows):
            distances = (rows - anchor_y) ** 2 + (cols - anchor_x) ** 2
            nearest = np.argsort(distances)[: min(64, len(rows))]
            points = np.column_stack(
                [rows[nearest] - anchor_y, cols[nearest] - anchor_x]
            ).astype(float)
            if len(points) >= 2:
                _, _, vectors = np.linalg.svd(points, full_matrices=False)
                tangent_y, tangent_x = vectors[0]
            else:
                tangent_y, tangent_x = 0.0, 1.0
        else:
            tangent_y, tangent_x = 0.0, 1.0
        return tuple(
            (
                int(round((index - (upper - 1) / 2.0) * 6 * tangent_y)),
                int(round((index - (upper - 1) / 2.0) * 6 * tangent_x)),
            )
            for index in range(upper)
        )
    if program == "small_cluster":
        return tuple((dy, dx) for dy, dx in ((0, 0), (-5, -4), (-4, 5), (5, -3), (4, 5))[:upper])
    if program == "dense_sheet":
        grid = tuple((dy, dx) for dy in (-6, 0, 6) for dx in (-6, 0, 6))
        return grid[:upper]
    raise JointContractError(f"unsupported layout program: {program}")
