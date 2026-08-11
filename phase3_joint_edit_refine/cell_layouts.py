"""Deterministic complete-instance cell layouts for paired candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import CandidateMask

from .budget import JointBudgetAllocation
from .cell_programs import CompiledCellToolProgram
from .executable_contract import ExecutableJointContract
from .models import JointContractError, JointEditPlan
from .nuclei import iter_instances, normalize_nuclei_mask
from .scene import JointSceneAnalysis
from .seam import anchor_coverage_fraction, class_center_mask
from .skills.repository import JointSkillBundle

LAYOUT_TOOL_VERSION = "joint-cell-layout-v2"


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
    executable_contract: ExecutableJointContract,
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
    executable_contract.validate_identity()
    if executable_contract.tissue_candidate_id != tissue_candidate.candidate_id:
        raise JointContractError(
            "cell layout received a contract for another tissue candidate"
        )
    compiled_program = executable_contract.cell_program
    if (
        source_tissue.shape != source_nuclei.shape
        or target_tissue.shape != source_tissue.shape
    ):
        raise JointContractError("cell layout inputs must share one shape")
    ranker = ranker or DeterministicDistanceRanker()
    ranker_domain = getattr(ranker, "pathology_domain_id", None)
    if (
        ranker_domain is not None
        and ranker_domain != bundle.mechanism.pathology_domain_id
    ):
        raise JointContractError(
            "spatial ranker pathology domain does not match the joint mechanism"
        )
    ranker_cancer_id = getattr(ranker, "cancer_id", None)
    if (
        ranker_cancer_id is not None
        and ranker_cancer_id != bundle.cell_population_profile.probnet_cancer_id
    ):
        raise JointContractError(
            "ProbNet cancer ID does not match the cell population profile"
        )
    if plan.tissue_plan is not None:
        target_class = target_cell_class(plan.tissue_plan.target_label, schema)
    elif len(bundle.primitive.target_cell_classes) == 1:
        target_class = bundle.primitive.target_cell_classes[0]
    elif plan.cell_plan.allowed_cell_classes:
        target_class = plan.cell_plan.allowed_cell_classes[0]
    else:
        raise JointContractError(
            "cell-only deterministic executor requires one Planner-bound target class"
        )
    if target_class not in plan.cell_plan.allowed_cell_classes:
        raise JointContractError(
            f"cell class {target_class} required by target tissue is not allowed by plan"
        )
    if plan.cell_plan.baseline_mode == "selective_remove":
        return _build_selective_removal_results(
            source_nuclei=source_nuclei,
            scene=scene,
            compiled_program=compiled_program,
            executable_contract=executable_contract,
            plan=plan,
            seed=seed,
            variants=variants,
        )
    if plan.cell_plan.baseline_mode == "render_owned_clearance":
        results = _build_selective_removal_results(
            source_nuclei=source_nuclei,
            scene=scene,
            compiled_program=compiled_program,
            executable_contract=executable_contract,
            plan=plan,
            seed=seed,
            variants=variants,
        )
        for result in results:
            result.trace.update(
                {
                    "execution_engine": (
                        "deterministic_complete_viable_instance_clearance_v1"
                    ),
                    "execution_program_id": (
                        executable_contract.execution_program_id
                    ),
                    "render_owned_debris_transition": True,
                    "synthetic_dead_nucleus_count": 0,
                    "render_material_policy": (
                        "necrotic-debris-inside-generation-support"
                    ),
                }
            )
        return results

    reference_classes = (
        plan.cell_plan.allowed_cell_classes
        if bundle.primitive.scope == "cell_only"
        else (target_class,)
    )
    references_by_class: dict[int, tuple[ReferenceNucleusShape, ...]] = {}
    rejected_references: dict[str, str] = {}
    for class_id in reference_classes:
        current, rejected = build_reference_shape_library(
            scene,
            class_id=class_id,
        )
        current = _prioritize_local_references(
            current,
            scene=scene,
            interface_ids=plan.cell_plan.interface_ids,
            core_zone=plan.cell_plan.core_zone,
        )
        if current:
            references_by_class[class_id] = current
        rejected_references.update(rejected)
    references = references_by_class.get(target_class, ())
    if bundle.primitive.primitive_id == "cellularity-increase-v1" and set(
        references_by_class
    ) != set(reference_classes):
        missing = sorted(set(reference_classes) - set(references_by_class))
        raise JointContractError(
            "cellularity increase lacks complete component-local shapes for classes: "
            + ", ".join(str(value) for value in missing)
        )
    if not references and len(reference_classes) == 1:
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
        erasure_region = compiled_program.erasure_region
        if np.any(component & erasure_region):
            base[component] = 0
            removed_ids.append(instance_id)

    legal_core = np.asarray(compiled_program.placement_center_region, dtype=bool)
    valid_footprint = np.asarray(compiled_program.valid_footprint_region, dtype=bool)
    halo = np.asarray(compiled_program.mechanism_region, dtype=bool)
    add_zone = legal_core | halo
    if not np.any(add_zone):
        raise JointContractError("joint cell program has no legal placement zone")

    if bundle.primitive.scope == "cell_only" and len(reference_classes) > 1:
        if not references_by_class:
            raise JointContractError(
                "local cellularity edit has no complete compatible source shapes"
            )
        return _build_multiclass_addition_results(
            source_tissue=source_tissue,
            target_tissue=target_tissue,
            base=base,
            scene=scene,
            schema=schema,
            bundle=bundle,
            plan=plan,
            compiled_program=compiled_program,
            executable_contract=executable_contract,
            references_by_class=references_by_class,
            rejected_references=rejected_references,
            ranker=ranker,
            seed=seed,
            variants=variants,
        )

    average_area = float(np.median([item.area_px for item in references]))
    if bundle.primitive.scope == "cell_only":
        if plan.cell_plan.mechanism_quota_role != "explicit_increment":
            raise JointContractError(
                "current structured cell-only executor supports explicit increments"
            )
        desired_cell_delta = int(
            compiled_program.target_delta_count
            if compiled_program.target_delta_count is not None
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
        source_density = _interface_class_density(
            scene,
            interface_ids=plan.cell_plan.interface_ids,
            class_id=target_class,
        )
        if source_density is None:
            source_density = _class_density_from_scene(
                scene,
                source_tissue,
                class_id=target_class,
                tissue_ids=_target_tissue_ids(plan.tissue_plan.target_label, schema),
            )
        replacement_count = round(np.count_nonzero(legal_core) * source_density)
        replacement_count = max(
            replacement_count, len(removed_ids) if removed_ids else 1
        )
        reserve_count = round(
            allocation.reserved_layout_halo_pixels / max(1.0, average_area)
        )
        if not np.any(halo):
            reserve_count = 0
    biological_replacement_count = replacement_count
    biological_reserve_count = reserve_count
    biological_desired_count = replacement_count + reserve_count
    capacity_bound = max(
        1, int(np.count_nonzero(add_zone) / max(1.0, average_area * 2.0))
    )
    requested_count = min(biological_desired_count, capacity_bound)
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

    orientation_mask = np.logical_or.reduce(
        [scene.tissue.anchor_masks[item] for item in plan.cell_plan.anchor_ids]
    )
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
            nominal_nucleus_diameter_px=compiled_program.nominal_nucleus_diameter_px,
            orientation_mask=orientation_mask,
            continuity_region=compiled_program.continuity_region,
            continuity_anchor_mask=compiled_program.continuity_anchor_mask,
            continuity_maximum_empty_run_px=(
                compiled_program.continuity_maximum_empty_run_px
            ),
            minimum_effect_span_px=0,
            minimum_effect_foci=0,
            seed=seed + variant * 104729,
        )
        halo_score = (
            ranker.score(
                tissue_mask=target_tissue,
                source_nuclei=target,
                cell_class=target_class,
                legal_zone=halo,
                context={
                    "mechanism_id": plan.selected_mechanism_id,
                    "zone": "cell_only_halo",
                },
            )
            if np.any(halo)
            else np.zeros_like(score)
        )
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
            nominal_nucleus_diameter_px=compiled_program.nominal_nucleus_diameter_px,
            orientation_mask=orientation_mask,
            continuity_region=np.zeros_like(halo),
            continuity_anchor_mask=np.zeros_like(halo),
            continuity_maximum_empty_run_px=0,
            minimum_effect_span_px=compiled_program.minimum_effect_span_px,
            minimum_effect_foci=compiled_program.minimum_effect_foci,
            seed=seed + variant * 104729 + 8191,
        )
        placed = core_placed + halo_placed
        placements = [*core_placements, *halo_placements]
        target_centers = class_center_mask(target, class_id=target_class)
        continuity_coverage = anchor_coverage_fraction(
            compiled_program.continuity_anchor_mask,
            target_centers,
            maximum_empty_run_px=(
                compiled_program.continuity_maximum_empty_run_px
            ),
        )
        result_id = f"cells-{variant + 1:02d}"
        results.append(
            CellLayoutResult(
                cell_candidate_id=result_id,
                target_nuclei_mask=target,
                trace={
                    "layout_tool_version": LAYOUT_TOOL_VERSION,
                    "execution_engine": "deterministic_research_layout_v1",
                    "execution_program_id": (
                        executable_contract.execution_program_id
                    ),
                    "production_density_calibrated": False,
                    "layout_program_id": plan.cell_plan.layout_program_id,
                    "compiled_cell_tool_program": (compiled_program.to_metadata()),
                    "executable_contract_id": executable_contract.contract_id,
                    "executable_contract_version": (executable_contract.schema_version),
                    "ranker": ranker.name,
                    "ranker_provenance": dict(getattr(ranker, "provenance", {})),
                    "target_cell_class": target_class,
                    # ``desired_count`` is the density-derived biological
                    # request. ``resolved_count`` is the exact reachable
                    # count for this deterministic layout/seed after complete
                    # shape containment and collision checks.  Keeping both
                    # prevents an impossible rough area estimate from being
                    # mistaken for an execution failure.
                    "biological_desired_count": biological_desired_count,
                    "biological_replacement_count": biological_replacement_count,
                    "biological_halo_count": biological_reserve_count,
                    "geometric_capacity_estimate": capacity_bound,
                    "desired_count": biological_desired_count,
                    "resolved_count": placed,
                    "requested_count": requested_count,
                    "attempted_count": requested_count,
                    "placed_count": placed,
                    "placement_completion": (
                        1.0
                        if placed == 0 and biological_desired_count == 0
                        else placed / max(1, biological_desired_count)
                    ),
                    "core_requested_count": replacement_count,
                    "core_placed_count": core_placed,
                    "halo_requested_count": reserve_count,
                    "halo_placed_count": halo_placed,
                    "placement_capacity_exhausted": placed
                    < (replacement_count + reserve_count),
                    "cell_capacity_fallback_used": placed < biological_desired_count,
                    "continuity_mode": compiled_program.continuity_mode,
                    "continuity_width_px": (
                        compiled_program.continuity_width_px
                    ),
                    "continuity_maximum_empty_run_px": (
                        compiled_program.continuity_maximum_empty_run_px
                    ),
                    "continuity_anchor_coverage_fraction": (
                        continuity_coverage
                    ),
                    "continuity_minimum_anchor_coverage_fraction": (
                        compiled_program.continuity_minimum_anchor_coverage_fraction
                    ),
                    "removed_source_instance_ids": removed_ids,
                    "protected_instance_ids": sorted(protected),
                    "reference_shape_count": len(references),
                    "reference_shape_ids": [item.instance_id for item in references],
                    "reference_shape_sources": sorted(
                        {item.source for item in references}
                    ),
                    "reference_shape_rejections": rejected_references,
                    "reference_shape_integrity_certified": True,
                    "reference_shape_locality": _reference_shape_locality(
                        plan.cell_plan.core_zone
                    ),
                    "reference_first": True,
                    "cross_domain_fallback": False,
                    "overlap_pixels": 0,
                    "partial_source_instance_edits": 0,
                    "cell_only_halo_pixels": int(np.count_nonzero(halo)),
                    "placements": placements,
                    "accepted_center_ledger": _accepted_center_ledger(
                        placements
                    ),
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
    zero_cell_fallback_allowed = (
        bool(tissue_candidate.tool_trace.get("area_fallback_used"))
        and bundle.mechanism.coupling.cell_only_target_fraction == 0
    )
    if desired > 0 and maximum_reachable <= 0 and not zero_cell_fallback_allowed:
        return ()
    certified: list[CellLayoutResult] = []
    for item in results:
        if int(item.trace.get("resolved_count", 0)) != maximum_reachable:
            continue
        item.trace["batch_max_attainable_count"] = maximum_reachable
        item.trace["capacity_max_count"] = maximum_reachable
        item.trace["resolved_count"] = maximum_reachable
        item.trace["requested_count"] = maximum_reachable
        item.trace["placed_count"] = maximum_reachable
        item.trace["cell_capacity_fallback_used"] = maximum_reachable < desired
        item.trace["cell_capacity_certified"] = True
        item.trace["zero_cell_capacity_fallback_allowed"] = zero_cell_fallback_allowed
        certified.append(item)
    return tuple(certified)


def _build_selective_removal_results(
    *,
    source_nuclei: np.ndarray,
    scene: JointSceneAnalysis,
    compiled_program: CompiledCellToolProgram,
    executable_contract: ExecutableJointContract,
    plan: JointEditPlan,
    seed: int,
    variants: int,
) -> tuple[CellLayoutResult, ...]:
    target = np.asarray(source_nuclei).copy()
    removed_ids = []
    for instance_id, component in sorted(scene.instance_masks.items()):
        if np.any(np.asarray(component, dtype=bool) & compiled_program.erasure_region):
            target[component] = 0
            removed_ids.append(instance_id)
    resolved = len(removed_ids)
    biological_desired = int(
        compiled_program.biological_target_delta_count
        if compiled_program.biological_target_delta_count is not None
        else resolved
    )
    field_driven = (
        compiled_program.depletion_parameters.get("resolution_mode")
        == "density_field"
    )
    desired = resolved if field_driven else biological_desired
    if resolved <= 0:
        return ()
    metadata = {item.instance_id: item for item in scene.cells.instances}
    removed_by_class: dict[int, int] = {}
    removed_by_band = {"core": 0, "transition": 0, "outer_reference": 0}
    source_by_band = {"core": 0, "transition": 0, "outer_reference": 0}
    radial_source_counts: dict[str, int] = {}
    radial_removed_counts: dict[str, int] = {}
    core = np.asarray(compiled_program.depletion_core_region, dtype=bool)
    transition = np.asarray(
        compiled_program.depletion_transition_region, dtype=bool
    )
    outer = np.asarray(
        compiled_program.depletion_outer_reference_region, dtype=bool
    )
    removed_set = set(removed_ids)
    if compiled_program.depletion_profile_id is not None:
        subbands = int(
            compiled_program.depletion_parameters.get(
                "transition_subband_count", 1
            )
        )
        radial_source_counts = {
            "core": 0,
            **{f"transition_{index + 1}": 0 for index in range(subbands)},
            "outer_reference": 0,
        }
        radial_removed_counts = dict.fromkeys(radial_source_counts, 0)
        anchor_distance = ndimage.distance_transform_edt(
            ~np.asarray(compiled_program.depletion_anchor_mask, dtype=bool)
        )
        core_end = float(
            compiled_program.depletion_parameters.get(
                "core_width_cell_diameters", 1.25
            )
        ) * compiled_program.nominal_nucleus_diameter_px
        transition_width = float(
            compiled_program.depletion_parameters.get(
                "transition_width_cell_diameters", 1.75
            )
        ) * compiled_program.nominal_nucleus_diameter_px
        for item in scene.cells.instances:
            row, col = round(item.centroid_xy[1]), round(item.centroid_xy[0])
            if core[row, col]:
                band = "core"
            elif transition[row, col]:
                band = "transition"
            elif outer[row, col]:
                band = "outer_reference"
            else:
                continue
            source_by_band[band] += 1
            if item.instance_id in removed_set:
                removed_by_band[band] += 1
            if band == "transition":
                normalized = max(
                    0.0,
                    (float(anchor_distance[row, col]) - core_end)
                    / max(1e-6, transition_width),
                )
                radial_band = (
                    f"transition_{min(subbands - 1, int(normalized * subbands)) + 1}"
                )
            else:
                radial_band = band
            radial_source_counts[radial_band] += 1
            if item.instance_id in removed_set:
                radial_removed_counts[radial_band] += 1
    for instance_id in removed_ids:
        class_id = metadata[instance_id].class_id
        removed_by_class[class_id] = removed_by_class.get(class_id, 0) + 1
    trace = {
        "layout_tool_version": LAYOUT_TOOL_VERSION,
        "execution_engine": (
            "deterministic_anchored_density_gradient_removal_v1"
            if compiled_program.depletion_profile_id is not None
            else "deterministic_complete_instance_removal_v1"
        ),
        "execution_program_id": executable_contract.execution_program_id,
        "ranker": "not_applicable_removal_uses_compiled_gradient",
        "ranker_provenance": {
            "role": "no_new_placement",
            "probnet_used": False,
        },
        "layout_program_id": plan.cell_plan.layout_program_id,
        "compiled_cell_tool_program": compiled_program.to_metadata(),
        "executable_contract_id": executable_contract.contract_id,
        "executable_contract_version": executable_contract.schema_version,
        "target_cell_classes": list(plan.cell_plan.allowed_cell_classes),
        "biological_desired_count": biological_desired,
        "count_resolution_mode": (
            "density_field" if field_driven else "explicit_count"
        ),
        "desired_count": desired,
        "resolved_count": resolved,
        "requested_count": resolved,
        "attempted_count": resolved,
        "placed_count": resolved,
        "removed_count": resolved,
        "class_removed_counts": {
            str(key): value for key, value in sorted(removed_by_class.items())
        },
        "depletion_profile_id": compiled_program.depletion_profile_id,
        "depletion_source_counts_by_band": source_by_band,
        "depletion_removed_counts_by_band": removed_by_band,
        "depletion_removal_fractions_by_band": {
            key: (
                removed_by_band[key] / source_by_band[key]
                if source_by_band[key]
                else 0.0
            )
            for key in source_by_band
        },
        "depletion_radial_source_counts": radial_source_counts,
        "depletion_radial_removed_counts": radial_removed_counts,
        "depletion_radial_removal_fractions": {
            key: (
                radial_removed_counts[key] / radial_source_counts[key]
                if radial_source_counts[key]
                else 0.0
            )
            for key in radial_source_counts
        },
        "batch_max_attainable_count": resolved,
        "capacity_max_count": resolved,
        "cell_capacity_certified": True,
        "cell_capacity_fallback_used": (
            False if field_driven else resolved < desired
        ),
        "placement_capacity_exhausted": (
            False if field_driven else resolved < desired
        ),
        "removed_source_instance_ids": removed_ids,
        "protected_instance_ids": list(executable_contract.protected_instance_ids),
        "reference_shape_ids": [],
        "reference_shape_rejections": {},
        "reference_shape_integrity_certified": True,
        "reference_shape_locality": "not_applicable_removal_only",
        "reference_first": False,
        "cross_domain_fallback": False,
        "overlap_pixels": 0,
        "partial_source_instance_edits": 0,
        "placements": [],
        "accepted_center_ledger": [],
        "seed": seed,
    }
    return tuple(
        CellLayoutResult(
            cell_candidate_id=f"remove-cells-{index + 1:02d}",
            target_nuclei_mask=target.copy(),
            trace={**trace, "variant": index + 1},
        )
        for index in range(min(variants, 1))
    )


def _build_multiclass_addition_results(
    *,
    source_tissue: np.ndarray,
    target_tissue: np.ndarray,
    base: np.ndarray,
    scene: JointSceneAnalysis,
    schema: MaskProfileSchema,
    bundle: JointSkillBundle,
    plan: JointEditPlan,
    compiled_program: CompiledCellToolProgram,
    executable_contract: ExecutableJointContract,
    references_by_class: dict[int, tuple[ReferenceNucleusShape, ...]],
    rejected_references: dict[str, str],
    ranker: SpatialRanker,
    seed: int,
    variants: int,
) -> tuple[CellLayoutResult, ...]:
    del source_tissue
    desired = int(compiled_program.target_delta_count or 0)
    if desired <= 0:
        raise JointContractError(
            "multiclass cellularity increase requires a positive target delta"
        )
    placement_zone = np.asarray(compiled_program.placement_center_region, dtype=bool)
    local_counts = {class_id: 0 for class_id in references_by_class}
    for item in scene.cells.instances:
        if item.class_id not in local_counts:
            continue
        x, y = item.centroid_xy
        row, col = round(y), round(x)
        if (
            0 <= row < placement_zone.shape[0]
            and 0 <= col < placement_zone.shape[1]
            and placement_zone[row, col]
        ):
            local_counts[item.class_id] += 1
    if sum(local_counts.values()) == 0:
        local_counts = {class_id: 1 for class_id in references_by_class}
    quotas = _largest_remainder_class_quotas(local_counts, desired)
    results = []
    for variant in range(variants):
        target = np.asarray(base).copy()
        placements = []
        placed_by_class = {}
        for offset, class_id in enumerate(sorted(quotas)):
            requested = quotas[class_id]
            if requested <= 0:
                continue
            compatible_ids = _compatible_host_fine_ids(
                schema=schema,
                bundle=bundle,
                class_id=class_id,
            )
            class_valid = np.asarray(
                compiled_program.valid_footprint_region, dtype=bool
            ) & np.isin(target_tissue, compatible_ids)
            class_zone = placement_zone & class_valid
            if not np.any(class_zone):
                placed_by_class[class_id] = 0
                continue
            score = ranker.score(
                tissue_mask=target_tissue,
                source_nuclei=target,
                cell_class=class_id,
                legal_zone=class_zone,
                context={
                    "mechanism_id": plan.selected_mechanism_id,
                    "zone": plan.cell_plan.core_zone,
                    "population_mode": "local_composition_preserving_add",
                },
            )
            target, placed, current = _place_layout(
                base=target,
                references=references_by_class[class_id],
                class_id=class_id,
                legal_zone=class_zone,
                valid_footprint_region=class_valid,
                halo=np.zeros_like(class_zone),
                score=np.asarray(score, dtype=float),
                requested_count=requested,
                layout_program=plan.cell_plan.layout_program_id,
                cluster_size_range=(1, 1),
                nominal_nucleus_diameter_px=(
                    compiled_program.nominal_nucleus_diameter_px
                ),
                orientation_mask=np.zeros_like(class_zone),
                continuity_region=np.zeros_like(class_zone),
                continuity_anchor_mask=np.zeros_like(class_zone),
                continuity_maximum_empty_run_px=0,
                minimum_effect_span_px=0,
                minimum_effect_foci=0,
                seed=seed + variant * 104729 + offset * 8191,
            )
            placements.extend(current)
            placed_by_class[class_id] = placed
        placed_total = sum(placed_by_class.values())
        results.append(
            CellLayoutResult(
                cell_candidate_id=f"multiclass-cells-{variant + 1:02d}",
                target_nuclei_mask=target,
                trace={
                    "layout_tool_version": LAYOUT_TOOL_VERSION,
                    "execution_engine": "deterministic_local_composition_add_v1",
                    "execution_program_id": (
                        executable_contract.execution_program_id
                    ),
                    "layout_program_id": plan.cell_plan.layout_program_id,
                    "compiled_cell_tool_program": compiled_program.to_metadata(),
                    "executable_contract_id": executable_contract.contract_id,
                    "executable_contract_version": executable_contract.schema_version,
                    "ranker": ranker.name,
                    "ranker_provenance": dict(getattr(ranker, "provenance", {})),
                    "target_cell_classes": sorted(references_by_class),
                    "biological_desired_count": desired,
                    "desired_count": desired,
                    "resolved_count": placed_total,
                    "requested_count": desired,
                    "attempted_count": desired,
                    "placed_count": placed_total,
                    "class_requested_counts": {
                        str(key): value for key, value in sorted(quotas.items())
                    },
                    "class_placed_counts": {
                        str(key): value
                        for key, value in sorted(placed_by_class.items())
                    },
                    "placement_completion": placed_total / max(1, desired),
                    "cell_capacity_fallback_used": placed_total < desired,
                    "placement_capacity_exhausted": placed_total < desired,
                    "removed_source_instance_ids": [],
                    "protected_instance_ids": list(
                        executable_contract.protected_instance_ids
                    ),
                    "reference_shape_ids": sorted(
                        item.instance_id
                        for values in references_by_class.values()
                        for item in values
                    ),
                    "reference_shape_rejections": rejected_references,
                    "reference_shape_integrity_certified": True,
                    "reference_shape_locality": _reference_shape_locality(
                        plan.cell_plan.core_zone
                    ),
                    "reference_first": True,
                    "cross_domain_fallback": False,
                    "overlap_pixels": 0,
                    "partial_source_instance_edits": 0,
                    "placements": placements,
                    "accepted_center_ledger": _accepted_center_ledger(
                        placements
                    ),
                    "seed": seed + variant * 104729,
                },
            )
        )
    maximum = max((int(item.trace["placed_count"]) for item in results), default=0)
    if maximum <= 0:
        return ()
    certified = []
    for item in results:
        if int(item.trace["placed_count"]) != maximum:
            continue
        item.trace["resolved_count"] = maximum
        item.trace["requested_count"] = maximum
        item.trace["placed_count"] = maximum
        item.trace["batch_max_attainable_count"] = maximum
        item.trace["capacity_max_count"] = maximum
        item.trace["cell_capacity_certified"] = True
        item.trace["cell_capacity_fallback_used"] = maximum < desired
        certified.append(item)
    return tuple(certified)


def _largest_remainder_class_quotas(
    counts: dict[int, int], total: int
) -> dict[int, int]:
    denominator = max(1, sum(counts.values()))
    raw = {key: total * value / denominator for key, value in counts.items()}
    quotas = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(raw[key] - quotas[key]), -counts[key], key),
    )
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def _accepted_center_ledger(
    placements: list[dict[str, Any]],
) -> list[dict[str, int]]:
    """Expose the executor's accepted centers as the sole placement authority.

    Re-segmenting a raster after pasting can merge neighboring nuclei and move
    connected-component centroids.  Gates must therefore validate the exact
    coordinates accepted by the placement tool, just as the mature ProbNet
    adapter already does.
    """

    result = []
    for item in placements:
        center = item.get("center_xy")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise JointContractError(
                "deterministic placement is missing its accepted center"
            )
        result.append(
            {
                "row": int(center[1]),
                "col": int(center[0]),
                "class_id": int(item["cell_class"]),
            }
        )
    return result


def _compatible_host_fine_ids(
    *, schema: MaskProfileSchema, bundle: JointSkillBundle, class_id: int
) -> tuple[int, ...]:
    fine_ids = set()
    for (
        label,
        classes,
    ) in bundle.cell_observation_profile.tissue_compatible_classes.items():
        if (
            class_id in classes
            and label in bundle.primitive.host_tissue_labels
            and label in schema.readable_labels
        ):
            fine_ids.update(schema.resolve_fine_ids(label))
    if not fine_ids:
        raise JointContractError(
            f"cell class {class_id} has no annotation-compatible host tissue"
        )
    return tuple(sorted(fine_ids))


def target_cell_class(target_label: str, schema: MaskProfileSchema) -> int:
    """Resolve the executable CellViT class for a canonical tissue target."""

    if target_label == "Tumor":
        return 1
    if target_label in {"Stroma", "Other tissue"}:
        return 3
    if target_label == "Immune infiltrate":
        return 2
    if target_label == "Necrosis":
        # Use the reliably observed inflammatory population as the primary
        # placement/reference class. Sparse dead nuclei remain an additional
        # legal class selected by the mature target-tissue prior.
        return 2
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
        if ndimage.label(component, structure=np.ones((3, 3), dtype=bool))[1] != 1:
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


def _class_density_from_scene(
    scene: JointSceneAnalysis,
    tissue,
    *,
    class_id: int,
    tissue_ids: tuple[int, ...],
) -> float:
    """Measure fallback density without re-segmenting the semantic raster."""

    tissue_region = np.isin(tissue, tissue_ids)
    denominator = int(np.count_nonzero(tissue_region))
    centers = 0
    total_instances = 0
    for item in scene.cells.instances:
        if item.class_id != class_id:
            continue
        total_instances += 1
        col, row = item.centroid_xy
        row, col = round(row), round(col)
        if (
            0 <= row < tissue.shape[0]
            and 0 <= col < tissue.shape[1]
            and tissue_region[row, col]
        ):
            centers += 1
    if denominator and centers:
        return centers / denominator
    return total_instances / max(1, int(np.prod(tissue.shape)))


def _interface_class_density(
    scene: JointSceneAnalysis,
    *,
    interface_ids: tuple[str, ...],
    class_id: int,
) -> float | None:
    zones = [
        item
        for item in scene.population.zones
        if item.interface_id in interface_ids
        and item.side == "target"
        and item.distance_band_px is not None
        and item.distance_band_px[0] == 0.0
    ]
    total_area = sum(item.area_px for item in zones)
    total_count = sum(item.class_counts.get(class_id, 0) for item in zones)
    if total_area <= 0 or total_count < 3:
        return None
    return float(total_count / total_area)


def _prioritize_local_references(
    references: tuple[ReferenceNucleusShape, ...],
    *,
    scene: JointSceneAnalysis,
    interface_ids: tuple[str, ...],
    core_zone: str,
) -> tuple[ReferenceNucleusShape, ...]:
    metadata = {item.instance_id: item for item in scene.cells.instances}
    if core_zone.startswith("pop:component:"):
        component_id = core_zone.removeprefix("pop:component:")
        # Local population primitives must learn morphology from the selected
        # tissue component.  Falling back to a distant same-class nucleus can
        # silently import a different size distribution into the edit.
        return tuple(
            item
            for item in references
            if metadata[item.instance_id].tissue_component_id == component_id
        )
    local = tuple(
        item
        for item in references
        if metadata[item.instance_id].nearest_interface_id in interface_ids
    )
    return local if len(local) >= 5 else references


def _reference_shape_locality(core_zone: str) -> str:
    return (
        "selected_tissue_component"
        if core_zone.startswith("pop:component:")
        else "selected_interface_neighborhood_preferred"
    )


def _legal_halo(core, *, target_tissue, prohibited_ids, maximum_px: int, enabled: bool):
    if not enabled or maximum_px <= 0 or not np.any(core):
        return np.zeros_like(core, dtype=bool)
    expanded = ndimage.binary_dilation(core, iterations=maximum_px)
    halo = expanded & ~core & ~np.isin(target_tissue, tuple(prohibited_ids))
    return halo


def _place_layout(
    *,
    base,
    references,
    class_id,
    legal_zone,
    valid_footprint_region,
    halo,
    score,
    requested_count,
    layout_program,
    cluster_size_range,
    nominal_nucleus_diameter_px,
    orientation_mask,
    continuity_region,
    continuity_anchor_mask,
    continuity_maximum_empty_run_px,
    minimum_effect_span_px,
    minimum_effect_foci,
    seed,
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
        anchors = _continuity_first_anchors(
            coords=coords,
            values=values,
            default_order=anchors,
            continuity_region=np.asarray(continuity_region, dtype=bool),
            continuity_anchor_mask=np.asarray(
                continuity_anchor_mask,
                dtype=bool,
            ),
            maximum_empty_run_px=max(
                1,
                int(continuity_maximum_empty_run_px),
            ),
        )
        anchors = _effect_first_anchors(
            anchors,
            minimum_effect_span_px=max(0, int(minimum_effect_span_px)),
            minimum_effect_foci=max(0, int(minimum_effect_foci)),
        )
    effective_cluster_range = cluster_size_range
    if minimum_effect_foci > 0 and requested_count > 0:
        # Reserve enough independent anchors to satisfy the skill-owned focus
        # count. A legal abundance edit must not collapse into a few maximum-
        # sized clumps simply because the template family permits them.
        maximum_per_focus = max(1, requested_count // minimum_effect_foci)
        effective_cluster_range = (
            min(int(cluster_size_range[0]), maximum_per_focus),
            min(int(cluster_size_range[1]), maximum_per_focus),
        )
    placed = 0
    placement_trace: list[dict[str, Any]] = []
    seam_region = np.asarray(continuity_region, dtype=bool)
    anchor_index = 0
    while placed < requested_count and anchor_index < len(anchors):
        ay, ax = (int(v) for v in anchors[anchor_index])
        anchor_index += 1
        offsets = _layout_offsets(
            layout_program,
            effective_cluster_range,
            anchor_y=ay,
            anchor_x=ax,
            legal_zone=legal_zone,
            orientation_mask=orientation_mask,
            nominal_nucleus_diameter_px=nominal_nucleus_diameter_px,
            seed=seed,
        )
        group_start = len(placement_trace)
        group_id = f"cluster-{anchor_index:04d}"
        for dy, dx in offsets:
            if placed >= requested_count:
                break
            reference = references[(placed + seed) % len(references)]
            shape = np.asarray(reference.mask, dtype=bool)
            cy, cx = ay + dy, ax + dx
            # Every member of a pair/cluster/cord owns its own accepted center.
            # The anchor being legal is not sufficient: template offsets can
            # otherwise leave P while their footprints still happen to fit V.
            if (
                cy < 0
                or cx < 0
                or cy >= legal_zone.shape[0]
                or cx >= legal_zone.shape[1]
                or not legal_zone[cy, cx]
            ):
                continue
            window = _placement_window(
                shape,
                center_y=cy,
                center_x=cx,
                canvas_shape=target.shape,
            )
            if window is None:
                continue
            y0, y1, x0, x1 = window
            if (
                y0 <= 0
                or x0 <= 0
                or y1 >= target.shape[0]
                or x1 >= target.shape[1]
            ):
                continue
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
            local_shape = np.zeros(
                (guard_y1 - guard_y0, guard_x1 - guard_x0), dtype=bool
            )
            local_shape[
                y0 - guard_y0 : y1 - guard_y0, x0 - guard_x0 : x1 - guard_x0
            ] = shape
            collision_guard = ndimage.binary_dilation(local_shape, iterations=1)
            if np.any(collision_guard & occupied[guard_y0:guard_y1, guard_x0:guard_x1]):
                continue
            target_view = target[y0:y1, x0:x1]
            target_view[shape] = class_id
            occupied[y0:y1, x0:x1] |= shape
            placement_trace.append(
                {
                    "center_xy": [cx, cy],
                    "cell_class": class_id,
                    "area_px": int(np.count_nonzero(shape)),
                    "in_cell_only_halo": bool(halo[cy, cx]),
                    "in_interface_seam": bool(seam_region[cy, cx]),
                    "in_interface_continuity_region": bool(
                        seam_region[cy, cx]
                    ),
                    "reference_instance_id": reference.instance_id,
                    "reference_source": reference.source,
                    "cluster_id": group_id,
                    "planned_cluster_size": min(len(offsets), requested_count),
                    "spacing_px": max(
                        2,
                        round(nominal_nucleus_diameter_px * 0.75),
                    ),
                    "orientation_policy": (
                        "local_interface_tangent_pca"
                        if layout_program in {"short_cord", "boundary_aligned"}
                        else "template_intrinsic"
                    ),
                }
            )
            placed += 1
        actual_cluster_size = len(placement_trace) - group_start
        for item in placement_trace[group_start:]:
            item["cluster_size"] = actual_cluster_size
    return target, placed, placement_trace


def _effect_first_anchors(
    anchors: np.ndarray,
    *,
    minimum_effect_span_px: int,
    minimum_effect_foci: int,
) -> np.ndarray:
    """Front-load spatially distinct legal anchors for meaningful cell edits.

    The incoming order remains the ProbNet/ranker preference authority for the
    first focus.  Subsequent required foci use deterministic farthest-point
    sampling, after which all remaining anchors retain their original order.
    This makes the skill's spatial intent executable without allowing the LLM
    to invent nucleus coordinates.
    """

    points = np.asarray(anchors, dtype=int)
    required = min(max(0, int(minimum_effect_foci)), len(points))
    if required <= 1 and minimum_effect_span_px <= 0:
        return points
    if not len(points):
        return points

    chosen_indices = [0]
    available = np.ones(len(points), dtype=bool)
    available[0] = False
    minimum_span_sq = float(max(0, minimum_effect_span_px) ** 2)
    while len(chosen_indices) < max(1, required) and np.any(available):
        chosen = points[np.asarray(chosen_indices, dtype=int)]
        candidates = np.flatnonzero(available)
        deltas = points[candidates, None, :] - chosen[None, :, :]
        minimum_distance_sq = np.min(
            np.sum(deltas.astype(float) ** 2, axis=2), axis=1
        )
        if len(chosen_indices) == 1 and minimum_span_sq > 0:
            clears_span = minimum_distance_sq >= minimum_span_sq
            pool = candidates[clears_span] if np.any(clears_span) else candidates
            pool_distances = (
                minimum_distance_sq[clears_span]
                if np.any(clears_span)
                else minimum_distance_sq
            )
        else:
            pool = candidates
            pool_distances = minimum_distance_sq
        next_index = int(pool[int(np.argmax(pool_distances))])
        chosen_indices.append(next_index)
        available[next_index] = False

    chosen_set = set(chosen_indices)
    remainder = [index for index in range(len(points)) if index not in chosen_set]
    return points[np.asarray([*chosen_indices, *remainder], dtype=int)]


def _continuity_first_anchors(
    *,
    coords: np.ndarray,
    values: np.ndarray,
    default_order: np.ndarray,
    continuity_region: np.ndarray,
    continuity_anchor_mask: np.ndarray,
    maximum_empty_run_px: int,
) -> np.ndarray:
    """Prioritize distributed, anchor-covering centers without a count quota."""

    if not np.any(continuity_region) or not np.any(continuity_anchor_mask):
        return default_order
    in_region = continuity_region[coords[:, 0], coords[:, 1]]
    region_indices = np.flatnonzero(in_region)
    if not len(region_indices):
        return default_order
    region_coords = coords[region_indices]
    region_values = values[region_indices]
    tree = cKDTree(region_coords.astype(float))
    anchor_points = np.argwhere(continuity_anchor_mask)
    sampled = _sample_anchor_points(
        anchor_points,
        spacing_px=max(1, maximum_empty_run_px),
    )
    preferred_indices: list[int] = []
    used: set[int] = set()
    for point in sampled:
        neighbors = tree.query_ball_point(
            np.asarray(point, dtype=float),
            r=max(1, maximum_empty_run_px),
        )
        available = [index for index in neighbors if index not in used]
        if not available:
            continue
        chosen = max(
            available,
            key=lambda index: (
                float(region_values[index]),
                -int(region_coords[index, 0]),
                -int(region_coords[index, 1]),
            ),
        )
        used.add(chosen)
        preferred_indices.append(region_indices[chosen])
    if not preferred_indices:
        return default_order
    preferred = coords[np.asarray(preferred_indices, dtype=int)]
    preferred_set = {tuple(value) for value in preferred.tolist()}
    remainder = np.asarray(
        [
            value
            for value in default_order.tolist()
            if tuple(value) not in preferred_set
        ],
        dtype=int,
    )
    return (
        np.concatenate([preferred, remainder], axis=0)
        if len(remainder)
        else preferred
    )


def _sample_anchor_points(
    points: np.ndarray,
    *,
    spacing_px: int,
) -> np.ndarray:
    """Deterministically retain enough points to cover a curved anchor."""

    if not len(points):
        return points
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    selected = [ordered[0]]
    remaining = ordered[1:]
    while len(remaining):
        distances = np.min(
            np.sum(
                (remaining[:, None, :] - np.asarray(selected)[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        index = int(np.argmax(distances))
        if float(distances[index]) <= float(spacing_px**2):
            break
        selected.append(remaining[index])
        remaining = np.delete(remaining, index, axis=0)
    return np.asarray(selected, dtype=int)


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
    orientation_mask: np.ndarray,
    nominal_nucleus_diameter_px: float,
    seed: int,
) -> tuple[tuple[int, int], ...]:
    lower = max(1, int(cluster_range[0]))
    upper = max(lower, int(cluster_range[1]))
    spacing = max(2, round(float(nominal_nucleus_diameter_px) * 0.75))
    cardinality = lower + (
        (int(anchor_y) * 1009 + int(anchor_x) * 9176 + int(seed)) % (upper - lower + 1)
    )
    if program in {"single", "population_replacement"}:
        return ((0, 0),)
    if program == "pair":
        return ((0, -spacing), (0, spacing))
    if program in {"short_cord", "boundary_aligned"}:
        boundary = np.asarray(orientation_mask, dtype=bool)
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
                round((index - (cardinality - 1) / 2.0) * spacing * tangent_y),
                round((index - (cardinality - 1) / 2.0) * spacing * tangent_x),
            )
            for index in range(cardinality)
        )
    if program == "small_cluster":
        angles = np.linspace(0.0, 2.0 * np.pi, cardinality, endpoint=False)
        return tuple(
            (0, 0)
            if index == 0
            else (
                round(spacing * np.sin(angle)),
                round(spacing * np.cos(angle)),
            )
            for index, angle in enumerate(angles)
        )
    if program == "dense_sheet":
        side = int(np.ceil(np.sqrt(cardinality)))
        origin = (side - 1) / 2.0
        grid = tuple(
            (
                round((row - origin) * spacing),
                round((col - origin) * spacing),
            )
            for row in range(side)
            for col in range(side)
        )
        return grid[:cardinality]
    raise JointContractError(f"unsupported layout program: {program}")
