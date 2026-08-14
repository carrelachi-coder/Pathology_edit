"""Deterministic complete-instance cell layouts for paired candidates."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import CandidateMask

from .budget import JointBudgetAllocation
from .cell_programs import CompiledCellToolProgram
from .executable_contract import ExecutableJointContract
from .models import JointContractError, JointEditPlan
from .nuclei import _semantic_instance_labels, normalize_nuclei_mask
from .reference_shapes import ReferenceNucleusShape
from .scene import JointSceneAnalysis
from .seam import (
    anchor_coverage_fraction,
    class_center_mask,
    compile_continuity_center_quota,
    compile_executable_continuity_count,
)
from .skills.repository import JointSkillBundle
from .spatial_contracts import (
    BREAST_SMALL_CLUSTER_MEMBER_SPACING_DIAMETERS,
    BREAST_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS,
    BREAST_SMALL_CLUSTER_MINIMUM_FOCUS_SIZE,
    BREAST_SMALL_CLUSTER_TARGET_FOCUS_COUNT,
    SCATTER_MINIMUM_CENTER_SEPARATION_DIAMETERS,
    SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS,
    SMALL_CLUSTER_MAXIMUM_FOCUS_SIZE,
    SMALL_CLUSTER_MEMBER_RADIUS_DIAMETERS,
    SMALL_CLUSTER_MINIMUM_FOCUS_SIZE,
    SMALL_CLUSTER_TARGET_FOCUS_COUNT,
    small_cluster_maximum_hotspot_span_px,
)

LAYOUT_TOOL_VERSION = "joint-cell-layout-v16"

_INDEPENDENT_FOCUS_PRIMITIVES = frozenset(
    {
        "peritumoral-neoplastic-scatter-increase-v1",
        "peritumoral-small-cluster-increase-v1",
    }
)


def independent_focus_minimum_center_separation_px(
    primitive_id: str,
    nominal_nucleus_diameter_px: float,
) -> float:
    """Return the raster-focus separation that preflight must certify.

    The postcondition gate reconstructs foci from accepted centers rather than
    trusting planner or executor cluster IDs.  Peritumoral scatter and
    small-cluster programs therefore need a certificate that cannot merge
    separately committed foci under that graph rule.  For the small-cluster
    primitive, treating every witness as an independently legal one-cell focus
    is conservative but valid because its skill-owned cardinality range is
    one to four cells per focus.
    """

    if primitive_id not in _INDEPENDENT_FOCUS_PRIMITIVES:
        return 0.0
    separation_diameters = (
        SCATTER_MINIMUM_CENTER_SEPARATION_DIAMETERS
        if primitive_id == "peritumoral-neoplastic-scatter-increase-v1"
        else SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS
    )
    return separation_diameters * max(
        0.0, float(nominal_nucleus_diameter_px)
    )


def certificate_aligned_cluster_size_range(
    *,
    primitive_id: str,
    mechanism_id: str | None = None,
    configured_range: tuple[int, int],
    packing_certificate: dict[str, Any],
    nominal_nucleus_diameter_px: float,
) -> tuple[int, int]:
    """Bind execution grouping to an independent-focus packing witness."""

    minimum, maximum = (int(item) for item in configured_range)
    required_separation = independent_focus_minimum_center_separation_px(
        primitive_id,
        nominal_nucleus_diameter_px,
    )
    certified_separation = float(
        packing_certificate.get("minimum_center_separation_px", 0.0)
    )
    certificate_proves_independent_foci = bool(
        packing_certificate.get("passed") is True
        and int(packing_certificate.get("requested_count", 0)) > 0
        and required_separation > 0.0
        and certified_separation + 1e-6 >= required_separation
    )
    strict_breast_cluster = (
        mechanism_id == "breast-peritumoral-small-cluster"
    )
    minimum_focus_size = (
        BREAST_SMALL_CLUSTER_MINIMUM_FOCUS_SIZE
        if strict_breast_cluster
        else SMALL_CLUSTER_MINIMUM_FOCUS_SIZE
    )
    target_focus_count = (
        BREAST_SMALL_CLUSTER_TARGET_FOCUS_COUNT
        if strict_breast_cluster
        else SMALL_CLUSTER_TARGET_FOCUS_COUNT
    )
    if (
        primitive_id == "peritumoral-small-cluster-increase-v1"
        and minimum <= minimum_focus_size <= maximum
        and certificate_proves_independent_foci
        and int(packing_certificate.get("requested_count", 0))
        >= target_focus_count * minimum_focus_size
    ):
        # The certificate proves count and complete-shape capacity, while the
        # executor must still prove the stricter localized budding-like
        # topology. Exclude singleton groups and let the executor balance an
        # request as at least three obvious 3--4-cell buds instead of
        # scattering small groups around the full annulus.
        return (
            minimum_focus_size,
            min(SMALL_CLUSTER_MAXIMUM_FOCUS_SIZE, maximum),
        )
    return (minimum, maximum)


def certificate_capacity_reference_ids(
    packing_certificate: dict[str, Any],
) -> tuple[str, ...]:
    """Return the concrete fallback shapes that prove certified capacity."""

    return tuple(
        dict.fromkeys(
            str(item.get("reference_instance_id"))
            for item in packing_certificate.get("placements", ())
            if isinstance(item, dict) and item.get("reference_instance_id")
        )
    )


def certificate_aligned_references(
    references: tuple[ReferenceNucleusShape, ...],
    packing_certificate: dict[str, Any],
) -> tuple[ReferenceNucleusShape, ...]:
    """Keep the complete eligible family and validate fallback bindings.

    The exact packer may safely recover a lower count by retrying with the
    smallest eligible complete shape.  That witness is a guaranteed fallback,
    not permission to clone the smallest contour for the final layout.  The
    executor first tries all eligible same-class shapes and may replay the
    bound witness shape only when a diverse attempt cannot complete.
    """

    if not (
        packing_certificate.get("passed") is True
        and packing_certificate.get("capacity_optimized_shape_fallback_used")
        is True
    ):
        return references
    witness_ids = set(certificate_capacity_reference_ids(packing_certificate))
    available_ids = {item.instance_id for item in references}
    if not witness_ids or not witness_ids.issubset(available_ids):
        raise JointContractError(
            "capacity-optimized packing witnesses are absent from execution references"
        )
    return references


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
            allow_calibrated_fallback=(
                bundle.primitive.scope == "cell_only"
            ),
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
        # Source-cell clearance and target-population synthesis are separate
        # actions. A Stroma-to-Tumor transition may remove many stromal nuclei
        # but must regenerate the neoplastic population at the observed target
        # density; one-for-one replacement caused systematic overpopulation.
        replacement_count = max(replacement_count, 1)
        reserve_count = round(
            allocation.reserved_layout_halo_pixels / max(1.0, average_area)
        )
        if not np.any(halo):
            reserve_count = 0
    source_density_requested_count = replacement_count
    continuity_quota = None
    preferred_seam_count = 0
    if bundle.primitive.scope == "tissue_and_cell":
        continuity_quota = compile_continuity_center_quota(
            nuclei_mask=base,
            target_tissue_mask=target_tissue,
            tissue_change=np.asarray(tissue_candidate.change_region, dtype=bool),
            continuity_region=compiled_program.continuity_region,
            continuity_anchor_mask=compiled_program.continuity_anchor_mask,
            continuity_width_px=compiled_program.continuity_width_px,
            density_ratio_range=compiled_program.continuity_density_ratio_range,
            requires_new_target_cells=(
                compiled_program.continuity_requires_new_target_cells
            ),
            target_class=target_class,
            target_fine_ids=_target_tissue_ids(
                plan.tissue_plan.target_label,
                schema,
            ),
        )
        preferred_seam_count = compile_executable_continuity_count(
            continuity_quota,
            anchor_pixels=int(
                np.count_nonzero(compiled_program.continuity_anchor_mask)
            ),
            maximum_empty_run_px=(
                compiled_program.continuity_maximum_empty_run_px
            ),
            minimum_anchor_coverage_fraction=(
                compiled_program.continuity_minimum_anchor_coverage_fraction
            ),
        )
        if (
            continuity_quota.maximum_count is not None
            and np.all(
                ~legal_core
                | np.asarray(compiled_program.continuity_region, dtype=bool)
            )
        ):
            replacement_count = min(
                replacement_count,
                int(continuity_quota.maximum_count),
            )
    executable_desired_count = replacement_count + reserve_count
    biological_replacement_count = replacement_count
    biological_reserve_count = reserve_count
    biological_desired_count = executable_desired_count
    if (
        bundle.primitive.scope == "cell_only"
        and compiled_program.biological_target_delta_count is not None
    ):
        biological_desired_count = int(
            compiled_program.biological_target_delta_count
        )
        biological_reserve_count = biological_desired_count
    capacity_bound = max(
        1, int(np.count_nonzero(add_zone) / max(1.0, average_area * 2.0))
    )
    packing_certificate = executable_contract.packing_certificate or {}
    references = certificate_aligned_references(
        references,
        packing_certificate,
    )
    certified_requested_count = int(
        packing_certificate.get("requested_count", 0)
        if packing_certificate.get("passed") is True
        else 0
    )
    if bundle.primitive.scope == "cell_only" and certified_requested_count > 0:
        if certified_requested_count != executable_desired_count:
            raise JointContractError(
                "cell-only executable count differs from its packing certificate"
            )
        # The exact certificate has already tested concrete complete shapes,
        # retained nuclei, V and one-pixel collision clearance.  A rough
        # area/(2*median-area) estimate is only a diagnostic and must not cap
        # the immutable executable count after that stronger proof passes.
        requested_count = certified_requested_count
    else:
        requested_count = min(executable_desired_count, capacity_bound)
    references = _calibrated_reference_variants(
        references,
        minimum_count=requested_count,
    )
    replacement_count = min(replacement_count, requested_count)
    reserve_count = min(reserve_count, max(0, requested_count - replacement_count))
    execution_cluster_size_range = certificate_aligned_cluster_size_range(
        primitive_id=bundle.primitive.primitive_id,
        mechanism_id=bundle.mechanism.mechanism_id,
        configured_range=bundle.mechanism.cell_program.cluster_size_range,
        packing_certificate=packing_certificate,
        nominal_nucleus_diameter_px=(
            compiled_program.nominal_nucleus_diameter_px
        ),
    )
    candidate_focus_witness_centers = (
        tuple(
            (int(item["row"]), int(item["col"]))
            for item in packing_certificate.get("placements", ())
            if isinstance(item, dict)
            and int(item.get("class_id", -1)) == target_class
            and "row" in item
            and "col" in item
        )
        if bundle.primitive.primitive_id in _INDEPENDENT_FOCUS_PRIMITIVES
        and packing_certificate.get("passed") is True
        else ()
    )
    certified_focus_witness_centers = (
        candidate_focus_witness_centers
        if _centers_satisfy_minimum_span(
            candidate_focus_witness_centers,
            minimum_span_px=compiled_program.minimum_effect_span_px,
        )
        else ()
    )

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
            cluster_size_range=execution_cluster_size_range,
            nominal_nucleus_diameter_px=compiled_program.nominal_nucleus_diameter_px,
            orientation_mask=orientation_mask,
            continuity_region=compiled_program.continuity_region,
            continuity_anchor_mask=compiled_program.continuity_anchor_mask,
            continuity_maximum_empty_run_px=(
                compiled_program.continuity_maximum_empty_run_px
            ),
            continuity_minimum_anchor_coverage_fraction=(
                compiled_program.continuity_minimum_anchor_coverage_fraction
            ),
            continuity_preferred_count=preferred_seam_count,
            minimum_effect_span_px=0,
            minimum_effect_foci=0,
            enforce_single_scatter_separation=(
                bundle.primitive.primitive_id != "cellularity-increase-v1"
            ),
            enforce_small_cluster_group_separation=(
                bundle.primitive.primitive_id
                == "peritumoral-small-cluster-increase-v1"
            ),
            strict_breast_small_cluster=(
                bundle.mechanism.mechanism_id
                == "breast-peritumoral-small-cluster"
            ),
            enforce_multisite_population=(
                bundle.mechanism.mechanism_id
                == "breast-local-population-modulation"
                and bundle.primitive.primitive_id
                == "neoplastic-cell-abundance-increase-v1"
            ),
            certified_witness_centers=(),
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
            cluster_size_range=execution_cluster_size_range,
            nominal_nucleus_diameter_px=compiled_program.nominal_nucleus_diameter_px,
            orientation_mask=orientation_mask,
            continuity_region=np.zeros_like(halo),
            continuity_anchor_mask=np.zeros_like(halo),
            continuity_maximum_empty_run_px=0,
            continuity_minimum_anchor_coverage_fraction=0.0,
            continuity_preferred_count=0,
            minimum_effect_span_px=compiled_program.minimum_effect_span_px,
            minimum_effect_foci=(
                requested_count
                if bundle.primitive.primitive_id
                == "peritumoral-neoplastic-scatter-increase-v1"
                else compiled_program.minimum_effect_foci
            ),
            enforce_single_scatter_separation=(
                bundle.primitive.primitive_id != "cellularity-increase-v1"
            ),
            enforce_small_cluster_group_separation=(
                bundle.primitive.primitive_id
                == "peritumoral-small-cluster-increase-v1"
            ),
            strict_breast_small_cluster=(
                bundle.mechanism.mechanism_id
                == "breast-peritumoral-small-cluster"
            ),
            enforce_multisite_population=(
                bundle.mechanism.mechanism_id
                == "breast-local-population-modulation"
                and bundle.primitive.primitive_id
                == "neoplastic-cell-abundance-increase-v1"
            ),
            certified_witness_centers=certified_focus_witness_centers,
            certified_fallback_reference_ids=(
                certificate_capacity_reference_ids(packing_certificate)
            ),
            previously_used_reference_digests=(
                item["reference_shape_sha256"]
                for item in core_placements
                if item.get("reference_shape_sha256")
            ),
            seed=seed + variant * 104729 + 8191,
        )
        placed = core_placed + halo_placed
        placements = [*core_placements, *halo_placements]
        scatter_metrics = (
            _scatter_placement_metrics(placements)
            if bundle.primitive.primitive_id
            == "peritumoral-neoplastic-scatter-increase-v1"
            else {}
        )
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
                    "configured_cluster_size_range": list(
                        bundle.mechanism.cell_program.cluster_size_range
                    ),
                    "execution_cluster_size_range": list(
                        execution_cluster_size_range
                    ),
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
                    "source_density_requested_count": (
                        source_density_requested_count
                    ),
                    "biological_halo_count": biological_reserve_count,
                    "geometric_capacity_estimate": capacity_bound,
                    "desired_count": biological_desired_count,
                    "resolved_count": placed,
                    "requested_count": requested_count,
                    "attempted_count": requested_count,
                    "placed_count": placed,
                    "class_requested_counts": {
                        str(target_class): requested_count
                    },
                    "class_placed_counts": {
                        str(target_class): placed
                    },
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
                    "reference_shape_unique_digest_count": len(
                        {_reference_shape_digest(item) for item in references}
                    ),
                    "reference_shape_sampling_policy": (
                        "same_class_source_without_replacement_then_"
                        "calibrated_library_without_replacement_then_"
                        "certified_fallback"
                    ),
                    "reference_shape_areas_by_class": {
                        str(target_class): [
                            int(item.area_px) for item in references
                        ]
                    },
                    "reference_shape_authority": (
                        scene.reference_shape_authority.to_metadata()
                        if scene.reference_shape_authority is not None
                        and any(
                            item.source.startswith(
                                "calibrated_dataset_instance_library"
                            )
                            for item in references
                        )
                        else None
                    ),
                    "reference_shape_rejections": rejected_references,
                    "reference_shape_integrity_certified": True,
                    "reference_shape_locality": _reference_shape_locality(
                        plan.cell_plan.core_zone,
                        references=references,
                    ),
                    "reference_first": True,
                    **scatter_metrics,
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
        authoritative_radial = {
            name: tuple(instance_ids)
            for name, instance_ids in (
                compiled_program.depletion_radial_instance_ids or {}
            ).items()
        }
        authoritative_bands = {
            name: tuple(instance_ids)
            for name, instance_ids in (
                compiled_program.depletion_band_instance_ids or {}
            ).items()
        }
        for band in source_by_band:
            ids = authoritative_bands.get(band, ())
            source_by_band[band] = len(ids)
            removed_by_band[band] = len(set(ids) & removed_set)
        for radial_band in radial_source_counts:
            ids = authoritative_radial.get(radial_band, ())
            radial_source_counts[radial_band] = len(ids)
            radial_removed_counts[radial_band] = len(
                set(ids) & removed_set
            )
        # The executor does not re-segment or re-bin the scene. The immutable
        # compiler authority is the only denominator used in its audit trace.
        for instance_id in removed_set:
            if instance_id not in set(
                compiled_program.depletion_population_instance_ids
            ):
                raise JointContractError(
                    "depletion erasure contains an instance outside compiler authority"
                )
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
        "depletion_instance_authority": {
            "source": "compiled_cell_tool_program",
            "population_instance_count": len(
                compiled_program.depletion_population_instance_ids
            ),
            "population_instance_ids": list(
                compiled_program.depletion_population_instance_ids
            ),
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
    certificate_counts = (
        executable_contract.packing_certificate or {}
    ).get("class_requested_counts", {})
    quotas = (
        {
            int(class_id): int(count)
            for class_id, count in certificate_counts.items()
            if int(class_id) in references_by_class and int(count) > 0
        }
        if isinstance(certificate_counts, Mapping)
        else {}
    )
    if sum(quotas.values()) != desired:
        quotas = _largest_remainder_class_quotas(local_counts, desired)
    effect_class = max(
        quotas,
        key=lambda class_id: (quotas[class_id], -class_id),
    )
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
                continuity_minimum_anchor_coverage_fraction=0.0,
                continuity_preferred_count=0,
                minimum_effect_span_px=(
                    compiled_program.minimum_effect_span_px
                    if class_id == effect_class
                    else 0
                ),
                minimum_effect_foci=(
                    min(requested, compiled_program.minimum_effect_foci)
                    if class_id == effect_class
                    else 0
                ),
                enforce_single_scatter_separation=False,
                enforce_small_cluster_group_separation=False,
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
                    "reference_shape_sources": sorted(
                        {
                            item.source
                            for values in references_by_class.values()
                            for item in values
                        }
                    ),
                    "reference_shape_areas_by_class": {
                        str(class_id): [
                            int(item.area_px) for item in values
                        ]
                        for class_id, values in sorted(
                            references_by_class.items()
                        )
                    },
                    "reference_shape_authority": (
                        scene.reference_shape_authority.to_metadata()
                        if scene.reference_shape_authority is not None
                        and any(
                            item.source.startswith(
                                "calibrated_dataset_instance_library"
                            )
                            for values in references_by_class.values()
                            for item in values
                        )
                        else None
                    ),
                    "reference_shape_rejections": rejected_references,
                    "reference_shape_integrity_certified": True,
                    "reference_shape_locality": _reference_shape_locality(
                        plan.cell_plan.core_zone,
                        references=tuple(
                            item
                            for values in references_by_class.values()
                            for item in values
                        ),
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
    allow_calibrated_fallback: bool = False,
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
        if _semantic_instance_labels(cropped)[1] != 1:
            rejected[instance_id] = "semantic_multi_instance_shape"
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
    calibrated = (
        scene.reference_shape_authority.shapes_by_class.get(class_id, ())
        if allow_calibrated_fallback
        and scene.reference_shape_authority is not None
        else ()
    )
    if calibrated and scene.cells.observation_quality != "native_instance":
        # A semantic watershed does not provide native object identity, but it
        # does provide complete, same-patch, same-class contours.  Prefer those
        # local morphologies once each, then use the digest-bound dataset
        # library when the requested increment exceeds the patch supply.
        return _unique_reference_shapes((*accepted, *calibrated)), rejected
    if not accepted and calibrated:
        return tuple(calibrated), rejected
    return _unique_reference_shapes(tuple(accepted)), rejected


def _reference_shape_digest(reference: ReferenceNucleusShape) -> str:
    shape = np.ascontiguousarray(np.asarray(reference.mask, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(str(shape.shape).encode("ascii"))
    digest.update(shape.tobytes(order="C"))
    return digest.hexdigest()


def _unique_reference_shapes(
    references: tuple[ReferenceNucleusShape, ...],
) -> tuple[ReferenceNucleusShape, ...]:
    """Keep source order while collapsing byte-identical nucleus contours."""

    seen: set[str] = set()
    result = []
    for item in references:
        digest = _reference_shape_digest(item)
        if digest in seen:
            continue
        seen.add(digest)
        result.append(item)
    return tuple(result)


def _calibrated_reference_variants(
    references: tuple[ReferenceNucleusShape, ...],
    *,
    minimum_count: int,
) -> tuple[ReferenceNucleusShape, ...]:
    """Add bounded same-class size variants only when unique contours run out.

    Same-patch complete shapes retain priority.  Dataset-library contours are
    deterministically resized around the patch/source median area, with small
    bounded scale changes, only until the requested morphology supply is met.
    Every variant is rechecked as one connected semantic instance.
    """

    unique = list(_unique_reference_shapes(references))
    requested = max(0, int(minimum_count))
    if len(unique) >= requested or not unique:
        return tuple(unique)
    target_area = float(np.median([item.area_px for item in unique]))
    library = tuple(
        item
        for item in unique
        if item.source == "calibrated_dataset_instance_library"
    )
    if not library:
        return tuple(unique)
    seen = {_reference_shape_digest(item) for item in unique}
    # Mild scale variation preserves class morphology without inventing
    # arbitrary deformations or changing aspect ratio.
    for multiplier in (0.90, 1.10, 0.82, 1.18):
        for item in library:
            source_area = max(1, int(item.area_px))
            desired_area = max(1.0, target_area * multiplier * multiplier)
            scale = float(np.sqrt(desired_area / source_area))
            new_height = max(1, round(item.mask.shape[0] * scale))
            new_width = max(1, round(item.mask.shape[1] * scale))
            # scipy nearest-neighbor zoom avoids a hard dependency on cv2 in
            # the local contract test environment.
            resized = ndimage.zoom(
                np.asarray(item.mask, dtype=np.uint8),
                zoom=(
                    new_height / item.mask.shape[0],
                    new_width / item.mask.shape[1],
                ),
                order=0,
                prefilter=False,
            ).astype(bool)
            if not np.any(resized):
                continue
            rows, cols = np.nonzero(resized)
            resized = np.ascontiguousarray(
                resized[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1],
                dtype=bool,
            )
            if ndimage.label(resized, structure=np.ones((3, 3), dtype=bool))[1] != 1:
                continue
            if _semantic_instance_labels(resized)[1] != 1:
                continue
            variant = ReferenceNucleusShape(
                instance_id=(
                    f"{item.instance_id}:scale-{scale:.4f}:"
                    f"{_reference_shape_digest(item)[:8]}"
                ),
                class_id=item.class_id,
                mask=resized,
                source="calibrated_dataset_instance_library_resized",
                area_px=int(np.count_nonzero(resized)),
                parent_instance_id=item.instance_id,
                scale_factor=scale,
            )
            digest = _reference_shape_digest(variant)
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(variant)
            if len(unique) >= requested:
                return tuple(unique)
    return tuple(unique)


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
    calibrated = tuple(
        item
        for item in references
        if item.source == "calibrated_dataset_instance_library"
    )
    same_patch = tuple(
        item for item in references if item.instance_id in metadata
    )
    if core_zone.startswith("pop:component:"):
        component_id = core_zone.removeprefix("pop:component:")
        # Local population primitives must learn morphology from the selected
        # tissue component.  Falling back to a distant same-class nucleus can
        # silently import a different size distribution into the edit.
        local = tuple(
            item
            for item in same_patch
            if metadata[item.instance_id].tissue_component_id == component_id
        )
        return _unique_reference_shapes((*local, *calibrated))
    local = tuple(
        item
        for item in same_patch
        if metadata[item.instance_id].nearest_interface_id in interface_ids
    )
    if len(local) >= 5:
        # Local same-class contours own normal execution priority, but exact
        # preflight may have needed another complete same-patch contour for a
        # tight capacity witness.  Keep that source family behind the local
        # shapes so the certificate remains executable without promoting the
        # dataset library ahead of native patch morphology.
        local_ids = {item.instance_id for item in local}
        remaining_same_patch = tuple(
            item for item in same_patch if item.instance_id not in local_ids
        )
        return _unique_reference_shapes(
            (*local, *remaining_same_patch, *calibrated)
        )
    return _unique_reference_shapes((*same_patch, *calibrated))


def _reference_shape_locality(
    core_zone: str,
    *,
    references: tuple[ReferenceNucleusShape, ...] = (),
) -> str:
    if references and all(
        item.source.startswith("calibrated_dataset_instance_library")
        for item in references
    ):
        return "calibrated_dataset_instance_library"
    if references and any(
        item.source.startswith("calibrated_dataset_instance_library")
        for item in references
    ):
        return "same_patch_then_calibrated_dataset_instance_library"
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
    continuity_minimum_anchor_coverage_fraction,
    continuity_preferred_count,
    minimum_effect_span_px,
    minimum_effect_foci,
    seed,
    enforce_single_scatter_separation=True,
    enforce_small_cluster_group_separation=True,
    strict_breast_small_cluster=False,
    enforce_multisite_population=False,
    certified_witness_centers=(),
    certified_fallback_reference_ids=(),
    previously_used_reference_digests=(),
):
    target = np.asarray(base).copy()
    occupied = target > 0
    rng = np.random.default_rng(seed)
    references = _reference_sampling_order(references, rng=rng)
    used_reference_digests: set[str] = {
        str(value) for value in previously_used_reference_digests
    }
    # Rank only centers that can hold at least one complete local reference
    # shape before any new placement. Ordering every legal pixel let the
    # effect-span compiler choose distant endpoints that were already occupied;
    # those attempts failed and the executor then collapsed back to a small
    # high-score cluster.
    free = np.asarray(valid_footprint_region, dtype=bool) & ~ndimage.binary_dilation(
        occupied, iterations=1
    )
    initially_fit = np.zeros_like(free, dtype=bool)
    for reference in references:
        initially_fit |= ndimage.binary_erosion(
            free,
            structure=np.asarray(reference.mask, dtype=bool),
            border_value=0,
        )
    executable_centers = np.asarray(legal_zone, dtype=bool) & initially_fit
    coords = np.argwhere(executable_centers)
    jitter = rng.uniform(0.0, 1e-6, size=len(coords))
    values = score[executable_centers] + jitter
    order = np.argsort(-values)
    anchors = coords[order]
    anchor_sampling_policy = "probnet_ranked"
    planned_small_cluster_group_count = 0
    small_cluster_target_focus_count = (
        BREAST_SMALL_CLUSTER_TARGET_FOCUS_COUNT
        if strict_breast_small_cluster
        else SMALL_CLUSTER_TARGET_FOCUS_COUNT
    )
    if (
        layout_program == "small_cluster"
        and requested_count > 0
        and (
            enforce_small_cluster_group_separation
            or enforce_multisite_population
        )
    ):
        planned_small_cluster_group_count = max(
            max(0, int(minimum_effect_foci)),
            (
                int(
                    np.ceil(
                        requested_count
                        / max(1, int(cluster_size_range[1]))
                    )
                )
                if (
                    strict_breast_small_cluster
                    or enforce_multisite_population
                )
                else 0
            ),
            (
                small_cluster_target_focus_count
                if enforce_small_cluster_group_separation
                else 0
            ),
        )
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
            minimum_anchor_coverage_fraction=float(
                continuity_minimum_anchor_coverage_fraction
            ),
            maximum_preferred_count=min(
                requested_count,
                max(1, int(continuity_preferred_count)),
            ),
        )
        if (
            layout_program == "single"
            and enforce_single_scatter_separation
            and certified_witness_centers
        ):
            anchors = _probnet_hard_core_anchor_order(
                coords,
                values=values,
                minimum_center_separation_px=(
                    2.25 * float(nominal_nucleus_diameter_px)
                ),
                minimum_effect_span_px=max(0, int(minimum_effect_span_px)),
                requested_count=requested_count,
                rng=rng,
            )
            selected_scatter_centers = anchors[:requested_count]
            hard_core_ok = _centers_satisfy_minimum_separation(
                selected_scatter_centers,
                minimum_separation_px=(
                    2.25 * float(nominal_nucleus_diameter_px)
                ),
            )
            span_ok = _centers_satisfy_minimum_span(
                selected_scatter_centers,
                minimum_span_px=minimum_effect_span_px,
            )
            if hard_core_ok and span_ok:
                anchor_sampling_policy = "probnet_weighted_hard_core_without_replacement"
            else:
                anchors = _certified_witness_first_anchors(
                    anchors,
                    certified_witness_centers=certified_witness_centers,
                )
                anchor_sampling_policy = "certified_packing_witness_fallback"
        elif (
            layout_program == "small_cluster"
            and enforce_small_cluster_group_separation
        ):
            anchors = _localized_small_cluster_anchor_order(
                coords,
                values=values,
                default_order=anchors,
                nominal_nucleus_diameter_px=nominal_nucleus_diameter_px,
                minimum_effect_span_px=max(0, int(minimum_effect_span_px)),
                required_focus_count=planned_small_cluster_group_count,
                minimum_anchor_separation_diameters=(
                    BREAST_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS
                    if strict_breast_small_cluster
                    else SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS
                ),
                strict_breast_small_cluster=strict_breast_small_cluster,
            )
            anchor_sampling_policy = (
                "probnet_ranked_localized_front_segment"
            )
        else:
            anchors = _effect_first_anchors(
                anchors,
                minimum_effect_span_px=max(0, int(minimum_effect_span_px)),
                minimum_effect_foci=max(
                    max(0, int(minimum_effect_foci)),
                    planned_small_cluster_group_count,
                ),
            )
            anchors = _certified_witness_first_anchors(
                anchors,
                certified_witness_centers=certified_witness_centers,
            )
    effective_cluster_range = cluster_size_range
    if minimum_effect_foci > 0 and requested_count > 0:
        # Reserve enough independent anchors to satisfy the skill-owned focus
        # count. A legal abundance edit must not collapse into a few maximum-
        # sized clumps simply because the template family permits them.
        maximum_per_focus = max(
            1,
            int(np.ceil(requested_count / minimum_effect_foci)),
        )
        effective_cluster_range = (
            min(int(cluster_size_range[0]), maximum_per_focus),
            min(int(cluster_size_range[1]), maximum_per_focus),
        )
    placed = 0
    placement_trace: list[dict[str, Any]] = []
    seam_region = np.asarray(continuity_region, dtype=bool)
    anchor_index = 0
    committed_group_count = 0
    allow_reference_reuse = False
    while placed < requested_count:
        if anchor_index >= len(anchors):
            if allow_reference_reuse:
                break
            # First exhaust every legal center while requiring an unused
            # morphology.  Only then may the executor reuse a contour as the
            # certified capacity fallback.
            allow_reference_reuse = True
            anchor_index = 0
            continue
        ay, ax = (int(v) for v in anchors[anchor_index])
        anchor_index += 1
        remaining_count = requested_count - placed
        group_cluster_range = effective_cluster_range
        if layout_program == "small_cluster" and (
            enforce_small_cluster_group_separation
            or enforce_multisite_population
        ):
            remaining_groups = max(
                1,
                planned_small_cluster_group_count - committed_group_count,
            )
            planned_group_size = int(
                np.ceil(remaining_count / remaining_groups)
            )
            planned_group_size = int(
                np.clip(
                    planned_group_size,
                    effective_cluster_range[0],
                    effective_cluster_range[1],
                )
            )
            group_cluster_range = (
                planned_group_size,
                planned_group_size,
            )
        offsets = _layout_offsets(
            layout_program,
            group_cluster_range,
            anchor_y=ay,
            anchor_x=ax,
            legal_zone=legal_zone,
            orientation_mask=orientation_mask,
            nominal_nucleus_diameter_px=nominal_nucleus_diameter_px,
            seed=seed,
            compact_small_cluster=strict_breast_small_cluster,
        )
        if layout_program in {"pair", "small_cluster", "short_cord"}:
            minimum_group_size = max(1, int(effective_cluster_range[0]))
            maximum_group_size = max(
                minimum_group_size,
                int(effective_cluster_range[1]),
            )
            if remaining_count < minimum_group_size:
                break
            group_target_size = min(
                len(offsets), remaining_count, maximum_group_size
            )
            leftover = remaining_count - group_target_size
            if 0 < leftover < minimum_group_size:
                shrink_by = minimum_group_size - leftover
                if group_target_size - shrink_by >= minimum_group_size:
                    group_target_size -= shrink_by
            if not (
                layout_program == "small_cluster"
                and enforce_small_cluster_group_separation
            ):
                offsets = offsets[:group_target_size]
        else:
            group_target_size = min(len(offsets), remaining_count)
        group_start = len(placement_trace)
        group_footprints: list[tuple[int, int, int, int, np.ndarray]] = []
        group_reference_digests: set[str] = set()
        group_id = f"cluster-{anchor_index:04d}"
        for dy, dx in offsets:
            if (
                placed >= requested_count
                or len(placement_trace) - group_start >= group_target_size
            ):
                break
            cy, cx = ay + dy, ax + dx
            if (
                layout_program == "single"
                and enforce_single_scatter_separation
                and any(
                    (cy - int(item["center_xy"][1])) ** 2
                    + (cx - int(item["center_xy"][0])) ** 2
                    <= (
                        SCATTER_MINIMUM_CENTER_SEPARATION_DIAMETERS
                        * float(nominal_nucleus_diameter_px)
                    )
                    ** 2
                    for item in placement_trace
                )
            ):
                # A single-cell scatter program must remain separated in the
                # final instance graph; ordinary non-overlap is insufficient
                # because two complete nuclei can still form one local focus.
                continue
            if (
                layout_program == "small_cluster"
                and enforce_small_cluster_group_separation
                and any(
                (cy - int(item["center_xy"][1])) ** 2
                + (cx - int(item["center_xy"][0])) ** 2
                <= (
                    SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS
                    * float(nominal_nucleus_diameter_px)
                )
                ** 2
                for item in placement_trace[:group_start]
                )
            ):
                # The gate rebuilds foci from the final raster and accepted
                # centers, deliberately ignoring planner cluster IDs. Keep
                # separately committed groups disconnected under that exact
                # graph rule so two legal clusters cannot merge into an
                # over-cardinality focus after rasterization.
                continue
            if (
                layout_program == "small_cluster"
                and enforce_small_cluster_group_separation
                and any(
                    (cy - int(item["center_xy"][1])) ** 2
                    + (cx - int(item["center_xy"][0])) ** 2
                    > (
                        small_cluster_maximum_hotspot_span_px(
                            nominal_nucleus_diameter_px,
                            minimum_effect_span_px,
                            compact_breast=strict_breast_small_cluster,
                        )
                    )
                    ** 2
                    for item in placement_trace
                )
            ):
                # All foci must remain inside one finite invasive-front
                # neighborhood. A globally dispersed annulus layout belongs
                # to the scatter primitive, even when cluster IDs are valid.
                continue
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
            fit = _first_fitting_reference(
                references=references,
                start_index=placed + seed,
                center_y=cy,
                center_x=cx,
                canvas_shape=target.shape,
                valid_footprint_region=valid_footprint_region,
                occupied=occupied,
                used_reference_digests=(
                    used_reference_digests | group_reference_digests
                ),
                allow_reference_reuse=allow_reference_reuse,
            )
            if fit is None:
                continue
            reference, shape, y0, y1, x0, x1 = fit
            reference_digest = _reference_shape_digest(reference)
            target_view = target[y0:y1, x0:x1]
            target_view[shape] = class_id
            occupied[y0:y1, x0:x1] |= shape
            group_footprints.append((y0, y1, x0, x1, shape))
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
                    "reference_shape_sha256": reference_digest,
                    "reference_parent_instance_id": (
                        reference.parent_instance_id
                    ),
                    "reference_scale_factor": float(reference.scale_factor),
                    "reference_reused": reference_digest
                    in used_reference_digests | group_reference_digests,
                    "reference_reuse_after_exhaustive_fit_search": bool(
                        allow_reference_reuse
                    ),
                    "certified_capacity_fallback_shape": bool(
                        reference.instance_id
                        in set(certified_fallback_reference_ids)
                    ),
                    "cluster_id": group_id,
                    "planned_cluster_size": group_target_size,
                    "spacing_px": max(
                        2,
                        round(nominal_nucleus_diameter_px * 0.75),
                    ),
                    "orientation_policy": (
                        "local_interface_tangent_pca"
                        if layout_program in {"short_cord", "boundary_aligned"}
                        else "template_intrinsic"
                    ),
                    "anchor_sampling_policy": anchor_sampling_policy,
                }
            )
            group_reference_digests.add(reference_digest)
            placed += 1
        actual_cluster_size = len(placement_trace) - group_start
        if (
            layout_program in {"pair", "small_cluster", "short_cord"}
            and actual_cluster_size < int(group_cluster_range[0])
        ):
            # Pair/cluster/cord semantics are atomic.  A partially fitting
            # template must not silently become one or more isolated cells.
            # Every committed footprint was collision-free against the source,
            # so clearing these trial pixels exactly restores the prior state.
            for y0, y1, x0, x1, shape in group_footprints:
                target[y0:y1, x0:x1][shape] = 0
                occupied[y0:y1, x0:x1][shape] = False
            del placement_trace[group_start:]
            placed -= actual_cluster_size
            continue
        used_reference_digests.update(group_reference_digests)
        for item in placement_trace[group_start:]:
            item["cluster_size"] = actual_cluster_size
        if actual_cluster_size:
            committed_group_count += 1
    return target, placed, placement_trace


def _localized_small_cluster_anchor_order(
    anchors: np.ndarray,
    *,
    values: np.ndarray,
    default_order: np.ndarray,
    nominal_nucleus_diameter_px: float,
    minimum_effect_span_px: int,
    required_focus_count: int,
    minimum_anchor_separation_diameters: float = (
        SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS
    ),
    strict_breast_small_cluster: bool = False,
) -> np.ndarray:
    """Front-load a compact multi-focus hotspot on one interface segment.

    ProbNet chooses the hotspot seed. Other focus anchors must be far enough
    apart to remain distinct, jointly satisfy the primitive span floor, and
    all fit under the localized hotspot span ceiling. Returning no anchors is
    intentional: a patch without this capacity must be reselected instead of
    degrading into annular scatter.
    """

    points = np.asarray(anchors, dtype=int)
    scores = np.asarray(values, dtype=float)
    ranked = np.asarray(default_order, dtype=int)
    required = max(2, int(required_focus_count))
    if len(points) < required or len(ranked) != len(points):
        return np.empty((0, 2), dtype=int)

    diameter = max(1.0, float(nominal_nucleus_diameter_px))
    minimum_between = (
        max(0.0, float(minimum_anchor_separation_diameters)) * diameter
    )
    maximum_span = small_cluster_maximum_hotspot_span_px(
        diameter,
        minimum_effect_span_px,
        compact_breast=strict_breast_small_cluster,
    )
    minimum_span = max(0.0, float(minimum_effect_span_px))
    index_by_center = {
        tuple(int(value) for value in point): index
        for index, point in enumerate(points)
    }
    ranked_indices = [
        index_by_center[tuple(int(value) for value in point)]
        for point in ranked
    ]

    best: tuple[float, list[int]] | None = None
    for seed_index in ranked_indices[: min(256, len(ranked_indices))]:
        seed = points[seed_index].astype(float)
        seed_distances = np.linalg.norm(points - seed, axis=1)
        endpoint_candidates = [
            index
            for index in ranked_indices
            if index != seed_index
            and seed_distances[index] > minimum_between
            and minimum_span <= seed_distances[index] <= maximum_span
        ]
        for endpoint_index in endpoint_candidates[:128]:
            selected = [seed_index, endpoint_index]
            while len(selected) < required:
                chosen = points[np.asarray(selected, dtype=int)]
                distances = np.linalg.norm(
                    points[:, None, :] - chosen[None, :, :],
                    axis=2,
                )
                candidates = [
                    index
                    for index in ranked_indices
                    if index not in selected
                    and float(np.min(distances[index])) > minimum_between
                    and float(np.max(distances[index])) <= maximum_span
                ]
                if not candidates:
                    break
                selected.append(candidates[0])
            if len(selected) < required:
                continue
            chosen = points[np.asarray(selected, dtype=int)].astype(float)
            pairwise = np.linalg.norm(
                chosen[:, None, :] - chosen[None, :, :],
                axis=2,
            )
            span = float(np.max(pairwise))
            if span + 1e-6 < minimum_span or span > maximum_span + 1e-6:
                continue
            quality = float(np.sum(scores[np.asarray(selected, dtype=int)]))
            proposal = (quality, selected)
            if best is None or proposal[0] > best[0]:
                best = proposal
        if best is not None and seed_index == best[1][0]:
            # The first feasible highest-ranked hotspot remains the ProbNet
            # authority; lower-ranked seeds cannot drift remotely.
            break
    if best is None:
        return np.empty((0, 2), dtype=int)

    selected = best[1]
    chosen = points[np.asarray(selected, dtype=int)]
    distances = np.linalg.norm(
        points[:, None, :] - chosen[None, :, :],
        axis=2,
    )
    local_remainder = [
        index
        for index in ranked_indices
        if index not in selected
        and float(np.max(distances[index])) <= maximum_span
    ]
    return points[np.asarray([*selected, *local_remainder], dtype=int)]


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
    required = max(
        max(0, int(minimum_effect_foci)),
        2 if minimum_effect_span_px > 0 else 0,
    )
    required = min(required, len(points))
    if required <= 1 and minimum_effect_span_px <= 0:
        return points
    if not len(points):
        return points

    minimum_span_sq = float(max(0, minimum_effect_span_px) ** 2)
    chosen_indices = [0]
    if minimum_effect_span_px > 0 and required >= 2:
        # A two-sweep farthest-point heuristic is not exact for a general 2-D
        # legal center set and can miss a valid hard span by several pixels.
        # The Euclidean diameter belongs to the convex hull; test that small
        # vertex set exactly and retain rank order only for endpoint ordering.
        endpoint_a, endpoint_b, diameter_sq = _exact_diameter_endpoint_pair(
            points
        )
        if diameter_sq >= minimum_span_sq:
            chosen_indices = sorted((endpoint_a, endpoint_b))
    available = np.ones(len(points), dtype=bool)
    available[np.asarray(chosen_indices, dtype=int)] = False
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


def _exact_diameter_endpoint_pair(
    points: np.ndarray,
) -> tuple[int, int, float]:
    """Return a deterministic exact Euclidean-diameter pair for 2-D points."""

    value = np.asarray(points, dtype=float)
    if len(value) < 2:
        return 0, 0, 0.0
    try:
        candidate_indices = np.asarray(
            ConvexHull(value).vertices,
            dtype=int,
        )
    except QhullError:
        # Collinear legal domains are common for narrow interfaces. Their
        # diameter endpoints are the extrema along any varying coordinate.
        ranges = np.ptp(value, axis=0)
        axis = int(np.argmax(ranges))
        candidate_indices = np.asarray(
            [int(np.argmin(value[:, axis])), int(np.argmax(value[:, axis]))],
            dtype=int,
        )
    candidates = value[candidate_indices]
    deltas = candidates[:, None, :] - candidates[None, :, :]
    distances_sq = np.sum(deltas**2, axis=2)
    left, right = np.unravel_index(
        int(np.argmax(distances_sq)),
        distances_sq.shape,
    )
    return (
        int(candidate_indices[left]),
        int(candidate_indices[right]),
        float(distances_sq[left, right]),
    )


def _certified_witness_first_anchors(
    anchors: np.ndarray,
    *,
    certified_witness_centers,
) -> np.ndarray:
    """Front-load exact packing witnesses without inventing new coordinates."""

    points = np.asarray(anchors, dtype=int)
    if not len(points) or not certified_witness_centers:
        return points
    index_by_center = {
        (int(row), int(col)): index
        for index, (row, col) in enumerate(points)
    }
    witness_indices = []
    for center in certified_witness_centers:
        key = tuple(int(value) for value in center)
        index = index_by_center.get(key)
        if index is not None and index not in witness_indices:
            witness_indices.append(index)
    witness_set = set(witness_indices)
    remainder = [
        index for index in range(len(points)) if index not in witness_set
    ]
    return points[np.asarray([*witness_indices, *remainder], dtype=int)]


def _centers_satisfy_minimum_span(
    centers,
    *,
    minimum_span_px: int,
) -> bool:
    """Prove that certificate centers preserve the compiled effect span."""

    minimum = max(0, int(minimum_span_px))
    if minimum <= 0:
        return bool(centers)
    points = np.asarray(centers, dtype=float)
    if len(points) < 2:
        return False
    _left, _right, diameter_sq = _exact_diameter_endpoint_pair(points)
    return diameter_sq + 1e-6 >= float(minimum**2)


def _first_fitting_reference(
    *,
    references: tuple[ReferenceNucleusShape, ...],
    start_index: int,
    center_y: int,
    center_x: int,
    canvas_shape: tuple[int, int],
    valid_footprint_region: np.ndarray,
    occupied: np.ndarray,
    used_reference_digests: set[str] | None = None,
    allow_reference_reuse: bool = True,
) -> tuple[ReferenceNucleusShape, np.ndarray, int, int, int, int] | None:
    """Choose an unused authority shape before reusing a fitting contour."""

    used = used_reference_digests or set()
    for allow_reuse in ((False, True) if allow_reference_reuse else (False,)):
        for offset in range(len(references)):
            # Preserve source-first ordering while unused morphologies remain.
            # Once capacity forces reuse, rotate the fallback family so the
            # same smallest contour does not become the universal clone.
            reference_index = (
                offset
                if not allow_reuse
                else (int(start_index) + offset) % len(references)
            )
            reference = references[reference_index]
            reference_digest = _reference_shape_digest(reference)
            if (reference_digest in used) != allow_reuse:
                continue
            shape = np.asarray(reference.mask, dtype=bool)
            window = _placement_window(
                shape,
                center_y=center_y,
                center_x=center_x,
                canvas_shape=canvas_shape,
            )
            if window is None:
                continue
            y0, y1, x0, x1 = window
            if y0 <= 0 or x0 <= 0 or y1 >= canvas_shape[0] or x1 >= canvas_shape[1]:
                continue
            # P constrains the center. V, not P, constrains the full footprint.
            if not np.all(valid_footprint_region[y0:y1, x0:x1][shape]):
                continue
            guard_y0, guard_y1 = max(0, y0 - 1), min(canvas_shape[0], y1 + 1)
            guard_x0, guard_x1 = max(0, x0 - 1), min(canvas_shape[1], x1 + 1)
            local_shape = np.zeros(
                (guard_y1 - guard_y0, guard_x1 - guard_x0), dtype=bool
            )
            local_shape[
                y0 - guard_y0 : y1 - guard_y0,
                x0 - guard_x0 : x1 - guard_x0,
            ] = shape
            collision_guard = ndimage.binary_dilation(local_shape, iterations=1)
            if np.any(
                collision_guard
                & occupied[guard_y0:guard_y1, guard_x0:guard_x1]
            ):
                continue
            return reference, shape, y0, y1, x0, x1
    return None


def _reference_sampling_order(
    references: tuple[ReferenceNucleusShape, ...],
    *,
    rng: np.random.Generator,
) -> tuple[ReferenceNucleusShape, ...]:
    """Sample source contours first, each morphology without replacement."""

    groups = ([], [], [])
    for item in _unique_reference_shapes(tuple(references)):
        if item.source == "calibrated_dataset_instance_library_resized":
            groups[2].append(item)
        elif item.source == "calibrated_dataset_instance_library":
            groups[1].append(item)
        else:
            groups[0].append(item)
    ordered = []
    for group in groups:
        if not group:
            continue
        order = rng.permutation(len(group))
        ordered.extend(group[int(index)] for index in order)
    return tuple(ordered)


def _probnet_hard_core_anchor_order(
    anchors: np.ndarray,
    *,
    values: np.ndarray,
    minimum_center_separation_px: float,
    minimum_effect_span_px: int,
    requested_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a weighted hard-core scatter prefix without lattice regularity.

    ProbNet remains the spatial authority.  Gumbel ranking samples from its
    relative mass without replacement; the hard-core test only rejects centers
    that would merge two intended single-cell foci.  We retry bounded
    deterministic Gumbel draws until the compiled span/focus contract passes.
    Packing witnesses remain a fallback, not the normal visual layout.
    """

    points = np.asarray(anchors, dtype=int)
    scores = np.asarray(values, dtype=float)
    count = min(max(0, int(requested_count)), len(points))
    if count <= 0 or not len(points):
        return points
    finite = np.asarray(scores, dtype=float)
    finite_values = finite[np.isfinite(finite)]
    if finite_values.size:
        shifted = np.where(
            np.isfinite(finite),
            finite - float(np.min(finite_values)),
            0.0,
        )
        positive = shifted[np.isfinite(shifted) & (shifted > 0)]
        scale = float(np.median(positive)) if positive.size else 1.0
        log_mass = np.log(
            np.clip(
                shifted / max(scale, 1e-12) + 1e-3,
                1e-12,
                None,
            )
        )
    else:
        log_mass = np.zeros(len(points), dtype=float)
    minimum_sq = max(0.0, float(minimum_center_separation_px)) ** 2
    best: list[int] = []
    for _attempt in range(32):
        priority = log_mass + rng.gumbel(size=len(points))
        order = np.argsort(-priority, kind="stable")
        selected: list[int] = []
        for index in order:
            point = points[int(index)]
            if selected:
                chosen = points[np.asarray(selected, dtype=int)]
                distance_sq = np.sum((chosen - point) ** 2, axis=1)
                if np.any(distance_sq <= minimum_sq):
                    continue
            selected.append(int(index))
            if len(selected) >= count:
                break
        if len(selected) > len(best):
            best = selected
        chosen_points = points[np.asarray(selected, dtype=int)]
        if len(selected) >= count and _centers_satisfy_minimum_span(
            chosen_points,
            minimum_span_px=minimum_effect_span_px,
        ):
            best = selected
            break
    selected_set = set(best)
    remainder = [index for index in range(len(points)) if index not in selected_set]
    return points[np.asarray([*best, *remainder], dtype=int)]


def _centers_satisfy_minimum_separation(
    centers: np.ndarray,
    *,
    minimum_separation_px: float,
) -> bool:
    """Verify the exact hard-core rule on a proposed scatter prefix."""

    points = np.asarray(centers, dtype=float)
    if not len(points):
        return False
    if len(points) == 1:
        return True
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :],
        axis=2,
    )
    distances[np.diag_indices_from(distances)] = np.inf
    return bool(
        float(np.min(distances))
        > max(0.0, float(minimum_separation_px))
    )


def _scatter_placement_metrics(
    placements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose auditable spacing dispersion for a final scatter layout."""

    policies = sorted(
        {
            str(item.get("anchor_sampling_policy"))
            for item in placements
            if item.get("anchor_sampling_policy")
        }
    )
    centers = np.asarray(
        [
            [float(item["center_xy"][1]), float(item["center_xy"][0])]
            for item in placements
            if isinstance(item.get("center_xy"), (list, tuple))
            and len(item["center_xy"]) == 2
        ],
        dtype=float,
    )
    nearest: list[float] = []
    if len(centers) >= 2:
        distances = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :],
            axis=2,
        )
        distances[np.diag_indices_from(distances)] = np.inf
        nearest = [float(value) for value in np.min(distances, axis=1)]
    return {
        "scatter_anchor_sampling_policies": policies,
        "scatter_nearest_neighbor_distances_px": nearest,
        "scatter_nearest_neighbor_range_px": (
            float(np.ptp(nearest)) if nearest else 0.0
        ),
        "scatter_nearest_neighbor_cv": (
            float(np.std(nearest) / np.mean(nearest))
            if nearest and float(np.mean(nearest)) > 0.0
            else 0.0
        ),
    }


def _continuity_first_anchors(
    *,
    coords: np.ndarray,
    values: np.ndarray,
    default_order: np.ndarray,
    continuity_region: np.ndarray,
    continuity_anchor_mask: np.ndarray,
    maximum_empty_run_px: int,
    minimum_anchor_coverage_fraction: float,
    maximum_preferred_count: int,
) -> np.ndarray:
    """Prioritize distributed seam centers without overpopulating the seam.

    The seam contract requires sufficient anchor coverage, but that does not
    authorize every sampled anchor neighborhood to receive a new nucleus. The
    density compiler and exact packing certificate own the finite seam quota.
    Extra target-population cells may still use the rest of the changed core.
    """

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
    # Mirror the hard geometric lower edge used by the seam compiler. This is
    # intentionally independent of tissue type and never asks the LLM for a
    # cell count.
    required = max(
        1,
        int(
            np.ceil(
                np.count_nonzero(continuity_anchor_mask)
                * float(
                    np.clip(minimum_anchor_coverage_fraction, 0.0, 1.0)
                )
                / max(1, 2 * int(maximum_empty_run_px) + 1)
            )
        ),
    )
    preferred_indices = preferred_indices[
        : max(required, int(maximum_preferred_count))
    ]
    preferred = coords[np.asarray(preferred_indices, dtype=int)]
    preferred_set = {tuple(value) for value in preferred.tolist()}
    non_seam_remainder = []
    seam_remainder = []
    for value in default_order.tolist():
        key = tuple(value)
        if key in preferred_set:
            continue
        target = (
            seam_remainder
            if continuity_region[int(value[0]), int(value[1])]
            else non_seam_remainder
        )
        target.append(value)
    remainder = np.asarray(
        [*non_seam_remainder, *seam_remainder],
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
    compact_small_cluster: bool = False,
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
        if compact_small_cluster:
            # Budding-like foci are packed as one compact geometric unit.
            # The anchor is the focus centroid rather than a mandatory cell
            # center, so triangular/square templates avoid the former
            # center-plus-wide-ring appearance.  Cardinality remains 3--4 and
            # every pair is close enough for the final-raster focus graph.
            cluster_spacing = max(
                4,
                round(
                    float(nominal_nucleus_diameter_px)
                    * BREAST_SMALL_CLUSTER_MEMBER_SPACING_DIAMETERS
                ),
            )
            half = cluster_spacing / 2.0
            if cardinality <= 1:
                compact = ((0.0, 0.0),)
            elif cardinality == 2:
                compact = ((0.0, -half), (0.0, half))
            elif cardinality == 3:
                height = np.sqrt(3.0) * cluster_spacing / 2.0
                compact = (
                    (-2.0 * height / 3.0, 0.0),
                    (height / 3.0, -half),
                    (height / 3.0, half),
                )
            else:
                compact = (
                    (-half, -half),
                    (-half, half),
                    (half, -half),
                    (half, half),
                )
            angle = (
                2.0
                * np.pi
                * ((int(anchor_y) * 1009 + int(anchor_x) * 9176 + int(seed)) % 8)
                / 8.0
            )
            cosine, sine = np.cos(angle), np.sin(angle)
            rotated = []
            for dy, dx in compact:
                offset = (
                    round(dy * cosine - dx * sine),
                    round(dy * sine + dx * cosine),
                )
                if offset not in rotated:
                    rotated.append(offset)
            return tuple(rotated)
        # A fixed left/right pair fails in a curved, narrow peritumoral
        # annulus even when another tangential neighbor is legal.  Search a
        # deterministic ring and retain legal partner centers.  A four-pixel
        # floor preserves the executor's one-pixel collision clearance for
        # the smallest 3x3 semantic nuclei used by contract fixtures.
        cluster_spacing = max(
            4,
            round(
                float(nominal_nucleus_diameter_px)
                * SMALL_CLUSTER_MEMBER_RADIUS_DIAMETERS
            ),
        )
        phase = (
            (int(anchor_y) * 1009 + int(anchor_x) * 9176 + int(seed)) % 16
        )
        ring = []
        preferred_steps = (
            0,
            4,
            12,
            8,
            2,
            14,
            6,
            10,
            1,
            15,
            3,
            13,
            5,
            11,
            7,
            9,
        )
        for step in preferred_steps:
            angle = 2.0 * np.pi * ((phase + step) % 16) / 16.0
            offset = (
                round(cluster_spacing * np.sin(angle)),
                round(cluster_spacing * np.cos(angle)),
            )
            if offset == (0, 0) or offset in ring:
                continue
            row, col = anchor_y + offset[0], anchor_x + offset[1]
            if (
                0 <= row < legal_zone.shape[0]
                and 0 <= col < legal_zone.shape[1]
                and legal_zone[row, col]
            ):
                ring.append(offset)
        # Return the complete deterministic ring. The atomic group executor
        # tries alternatives until it fills the planned 2--4-cell focus; it
        # must not roll back merely because one of the first directions is
        # blocked by the neighboring focus.
        return ((0, 0), *ring)
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
