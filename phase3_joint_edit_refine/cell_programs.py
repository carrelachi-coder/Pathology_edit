"""Compile Planner cell intent into deterministic erasure/placement contracts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from itertools import pairwise

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import CandidateMask

from .authority import nucleus_instance_has_destructive_authority
from .models import JointCaseContext, JointContractError, JointEditPlan
from .scene import JointSceneAnalysis
from .seam import compile_adaptive_seam, target_cell_class_for_tissue
from .skills.repository import JointSkillBundle

CELL_TOOL_COMPILER_VERSION = "joint-cell-tool-compiler-v18"
DEPLETION_FIELD_AREA_RASTER_TOLERANCE = 0.05
QUALIFIED_RESIDUAL_MINIMUM_AREA_RATIO = 0.05
IMMUNE_RESIDUAL_MINIMUM_AREA_RATIO = 0.10
QUALIFIED_RESIDUAL_MINIMUM_AREA_PX = 4


def _instance_center_is_in_region(item, region: np.ndarray) -> bool:
    """Return whether an instance's audited center belongs to a receiving ROI."""

    x, y = item.centroid_xy
    row, col = round(y), round(x)
    return bool(
        0 <= row < region.shape[0]
        and 0 <= col < region.shape[1]
        and region[row, col]
    )


def _is_biological_instance(
    item,
    *,
    residual_area_floor_px: float | None = None,
) -> bool:
    """Keep native objects and only explicitly qualified residual objects."""

    if item.source == "instance_json_semantic_seeded_residual":
        return False
    if item.source not in {
        "instance_json_semantic_fallback",
        "instance_json_semantic_unseeded",
    }:
        return True
    return bool(
        residual_area_floor_px is not None
        and float(item.area_px) >= float(residual_area_floor_px)
    )


def _qualified_residual_area_floor(
    scene: JointSceneAnalysis,
    cell_classes: tuple[int, ...],
) -> float:
    """Derive a conservative residual-object floor from trusted morphology."""

    allowed = set(cell_classes)
    trusted_areas = [
        float(item.area_px)
        for item in scene.cells.instances
        if item.source == "instance_json_cellvit_seed"
        and item.class_id in allowed
        and item.completeness_status == "complete"
        and not item.touches_border
        and not item.quality_flags
        and item.area_px > 0
    ]
    if not trusted_areas and scene.reference_shape_authority is not None:
        trusted_areas = [
            float(shape.area_px)
            for class_id in allowed
            for shape in scene.reference_shape_authority.shapes_by_class.get(
                class_id, ()
            )
            if shape.area_px > 0
        ]
    if not trusted_areas:
        return float("inf")
    ratio = (
        IMMUNE_RESIDUAL_MINIMUM_AREA_RATIO
        if allowed == {2}
        else QUALIFIED_RESIDUAL_MINIMUM_AREA_RATIO
    )
    return max(
        float(QUALIFIED_RESIDUAL_MINIMUM_AREA_PX),
        ratio * float(np.median(trusted_areas)),
    )


def depletion_field_area_cell_squares(
    *,
    core_region: np.ndarray,
    transition_region: np.ndarray,
    outer_reference_region: np.ndarray,
    nominal_nucleus_diameter_px: float,
) -> float:
    """Measure the exact three-band field in nucleus-diameter squares."""

    field = (
        np.asarray(core_region, dtype=bool)
        | np.asarray(transition_region, dtype=bool)
        | np.asarray(outer_reference_region, dtype=bool)
    )
    return float(np.count_nonzero(field)) / max(
        1.0, float(nominal_nucleus_diameter_px) ** 2
    )


def depletion_field_area_is_sufficient(
    *,
    core_region: np.ndarray,
    transition_region: np.ndarray,
    outer_reference_region: np.ndarray,
    nominal_nucleus_diameter_px: float,
    minimum_field_area_cell_diameter_squares: float,
) -> tuple[bool, float, float]:
    """Apply the shared finite-raster tolerance used by compiler and gate."""

    observed = depletion_field_area_cell_squares(
        core_region=core_region,
        transition_region=transition_region,
        outer_reference_region=outer_reference_region,
        nominal_nucleus_diameter_px=nominal_nucleus_diameter_px,
    )
    effective_minimum = (
        1.0 - DEPLETION_FIELD_AREA_RASTER_TOLERANCE
    ) * float(minimum_field_area_cell_diameter_squares)
    return observed >= effective_minimum, observed, effective_minimum


@dataclass(frozen=True)
class DepletionInstanceAuthority:
    """Exact complete-instance population used by compiler, executor and gate."""

    population_instance_ids: tuple[str, ...]
    band_instance_ids: dict[str, tuple[str, ...]]
    radial_instance_ids: dict[str, tuple[str, ...]]
    effective_core_end_px: float
    effective_transition_width_px: float


@dataclass(frozen=True)
class CompiledCellToolProgram:
    """Masks that separate cell semantics from model execution.

    T_pop is the target-tissue population accounting region, E erases whole
    source instances, P accepts new centers, V must contain every new footprint,
    S is model context/render support, M is the mechanism zone, C is the
    candidate-local continuity region and A is the Planner-selected active
    anchor. Cell abundance is computed from T_pop, never from P. Only E may
    destroy source cells; only P may supply centers; S never enters the edit
    budget.
    """

    program_id: str
    primitive_id: str
    mechanism_id: str
    baseline_mode: str
    mechanism_program_id: str
    quota_role: str
    population_target_region: np.ndarray
    erasure_region: np.ndarray
    placement_center_region: np.ndarray
    valid_footprint_region: np.ndarray
    support_context_region: np.ndarray
    mechanism_region: np.ndarray
    continuity_region: np.ndarray
    continuity_anchor_mask: np.ndarray
    depletion_core_region: np.ndarray
    depletion_transition_region: np.ndarray
    depletion_outer_reference_region: np.ndarray
    depletion_anchor_mask: np.ndarray
    depletion_anchor_type: str
    depletion_profile_id: str | None
    depletion_parameters: dict[str, float | int | str]
    depletion_population_instance_ids: tuple[str, ...]
    depletion_band_instance_ids: dict[str, tuple[str, ...]]
    depletion_radial_instance_ids: dict[str, tuple[str, ...]]
    continuity_mode: str
    continuity_width_px: int
    continuity_maximum_empty_run_px: int
    continuity_minimum_anchor_coverage_fraction: float
    continuity_density_ratio_range: tuple[float, float]
    continuity_requires_new_target_cells: bool
    whole_instance_closure_px: int
    target_classes: tuple[int, ...]
    selected_interface_ids: tuple[str, ...]
    selected_anchor_ids: tuple[str, ...]
    nominal_nucleus_diameter_px: float
    target_delta_count: int | None
    biological_target_delta_count: int | None
    minimum_effect_span_px: int
    minimum_effect_foci: int
    policies: dict[str, str]

    def to_metadata(self) -> dict:
        result = asdict(self)
        for key in (
            "population_target_region",
            "erasure_region",
            "placement_center_region",
            "valid_footprint_region",
            "support_context_region",
            "mechanism_region",
            "continuity_region",
            "continuity_anchor_mask",
            "depletion_core_region",
            "depletion_transition_region",
            "depletion_outer_reference_region",
            "depletion_anchor_mask",
        ):
            value = result.pop(key)
            result[f"{key}_pixels"] = int(np.count_nonzero(value))
            result[f"{key}_sha256"] = _mask_digest(value)
        result["compiler_version"] = CELL_TOOL_COMPILER_VERSION
        return result


class CellToolProgramCompiler:
    """Fail-closed compiler shared by mature and deterministic executors."""

    def compile(
        self,
        *,
        case: JointCaseContext,
        schema: MaskProfileSchema,
        scene: JointSceneAnalysis,
        plan: JointEditPlan,
        bundle: JointSkillBundle,
        tissue_candidate: CandidateMask,
    ) -> CompiledCellToolProgram:
        primitive = bundle.primitive
        cell = plan.cell_plan
        if cell.baseline_mode not in primitive.allowed_baseline_modes:
            raise JointContractError("Planner baseline mode is illegal for primitive")
        if cell.mechanism_quota_role not in primitive.allowed_quota_roles:
            raise JointContractError("Planner quota role is illegal for primitive")
        expected_layout = bundle.mechanism.cell_program.layout_for(
            case.primitive_id
        )
        if (
            cell.mechanism_program_id != expected_layout
            or cell.layout_program_id != expected_layout
        ):
            raise JointContractError(
                "Planner cell program differs from the skill-compiled primitive layout"
            )
        if set(cell.allowed_cell_classes) - set(
            bundle.mechanism.cell_program.allowed_cell_classes
        ):
            raise JointContractError(
                "Planner requested a class outside the mechanism skill"
            )

        target_tissue = np.asarray(tissue_candidate.target_mask)
        tissue_change = np.asarray(tissue_candidate.change_region, dtype=bool)
        valid_erasure_footprint = ~np.isin(
            target_tissue,
            bundle.annotation_profile.prohibit_generation_support_fine_ids,
        )
        automatic_lumen_ids = {
            "glas-gland-v1": ("gland_or_lumen_support",),
            "panda-gleason-v1": ("gland_lumen_map",),
        }.get(bundle.annotation_profile.annotation_profile_id, ())
        protected_structure_ids = tuple(
            dict.fromkeys(
                (
                    *bundle.mechanism.representability.protected_auxiliary_structures,
                    *(
                        structure_id
                        for structure_id in automatic_lumen_ids
                        if structure_id in scene.auxiliary_structure_masks
                    ),
                )
            )
        )
        for structure_id in protected_structure_ids:
            structure = scene.auxiliary_structure_masks.get(structure_id)
            if structure is None:
                raise JointContractError(
                    f"required protected auxiliary {structure_id!r} is unavailable"
                )
            valid_erasure_footprint &= ~np.asarray(structure, dtype=bool)
        effect_classes = set(cell.allowed_cell_classes)
        source_diameter = float(
            scene.population.nominal_nucleus_diameter_px or 8.0
        )
        complete_areas = [
            float(item.area_px)
            for item in scene.cells.instances
            if _is_biological_instance(item)
            if item.completeness_status == "complete"
            and not item.touches_border
            and not item.quality_flags
            and item.area_px > 0
            and (
                not effect_classes or item.class_id in effect_classes
            )
        ]
        native_seed_areas = [
            float(item.area_px)
            for item in scene.cells.instances
            if item.source == "instance_json_cellvit_seed"
            and item.completeness_status == "complete"
            and not item.touches_border
            and not item.quality_flags
            and item.area_px > 0
            and (not effect_classes or item.class_id in effect_classes)
        ]
        glas_source_depletion = bool(
            bundle.annotation_profile.annotation_profile_id == "glas-gland-v1"
            and any(action.startswith("remove") for action in cell.actions)
            and not any(action.startswith("add") for action in cell.actions)
        )
        calibrated_diameter = (
            scene.reference_shape_authority.nominal_diameter_px(
                tuple(sorted(effect_classes))
            )
            if primitive.scope == "cell_only"
            and scene.reference_shape_authority is not None
            and not native_seed_areas
            and not glas_source_depletion
            else None
        )
        # Native seeds own biological scale when present. If a native raster
        # has only provenance-only residuals for the target class, use the
        # digest-bound dataset library instead of those tiny partitions.
        diameter = float(calibrated_diameter or source_diameter)
        scale_areas = native_seed_areas or complete_areas
        effect_diameter = (
            diameter
            if calibrated_diameter is not None
            else (
                max(
                    3.0,
                    2.0
                    * np.sqrt(float(np.median(scale_areas)) / np.pi),
                )
                if scale_areas
                else diameter
            )
        )
        # The same native-derived ruler must own spans, radial fields, support
        # margins and whole-instance closure. Using the raw residual-partition
        # diameter for the latter stages collapsed valid GLaS depletion fields
        # even though preflight had already applied the finite-raster floor.
        diameter = float(effect_diameter)
        minimum_effect_span_cell_diameters = (
            primitive.minimum_effect_span_cell_diameters
        )
        minimum_effect_foci = primitive.minimum_effect_foci
        if case.annotation_profile_id == "panda-gleason-v1":
            panda_local_effect_overrides = {
                "cell-type-abundance-increase-v1": (2.5, 1),
                "cell-type-abundance-decrease-v1": (1.5, 0),
                "cellularity-increase-v1": (3.0, 3),
                "cellularity-decrease-v1": (4.0, 0),
                "neoplastic-cell-abundance-increase-v1": (2.0, 1),
                "neoplastic-cell-abundance-decrease-v1": (1.5, 0),
            }
            if case.primitive_id in panda_local_effect_overrides:
                (
                    minimum_effect_span_cell_diameters,
                    minimum_effect_foci,
                ) = panda_local_effect_overrides[case.primitive_id]
        skill_minimum_effect_span_px = int(
            np.floor(
                minimum_effect_span_cell_diameters * effect_diameter
            )
        )
        if (
            primitive.scope == "cell_only"
            and case.cell_count_extent_budget is not None
            and case.cell_count_extent_budget.maximum_extent_px
            < skill_minimum_effect_span_px
        ):
            raise JointContractError(
                "cell-only extent cannot realize the skill-owned minimum effect span"
            )
        empty = np.zeros_like(tissue_change, dtype=bool)
        depletion_core = empty.copy()
        depletion_transition = empty.copy()
        depletion_outer = empty.copy()
        depletion_anchor = empty.copy()
        panda_cord_parent_band = empty.copy()
        depletion_parameters: dict[str, float | int | str] = {}
        depletion_population_instance_ids: tuple[str, ...] = ()
        depletion_band_instance_ids: dict[str, tuple[str, ...]] = {}
        depletion_radial_instance_ids: dict[str, tuple[str, ...]] = {}
        depletion_profile_id = None
        anchor_zone = (
            np.asarray(scene.population_zone_masks[cell.core_zone], dtype=bool)
            if cell.core_zone in scene.population_zone_masks
            else self._anchor_zone(
                scene=scene,
                interface_ids=cell.interface_ids,
                anchor_ids=cell.anchor_ids,
                maximum_px=bundle.mechanism.cell_program.halo_distance_px[1],
            )
        )
        if primitive.scope == "tissue_and_cell":
            if plan.tissue_plan is None or not np.any(tissue_change):
                raise JointContractError(
                    "tissue primitive requires a nonempty tissue plan"
                )
            center_region = tissue_change.copy()
            population_target_region = tissue_change.copy()
            mechanism_region = anchor_zone & (
                ndimage.binary_dilation(
                    tissue_change,
                    iterations=bundle.mechanism.cell_program.halo_distance_px[1],
                )
            )
            if bundle.mechanism.coupling.cell_only_target_fraction <= 0:
                mechanism_region &= tissue_change
            if (
                bundle.annotation_profile.annotation_profile_id
                == "panda-gleason-v1"
                and case.primitive_id
                == "infiltrative-nest-cord-extension-v1"
            ):
                target_ids = tuple(
                    schema.resolve_fine_ids(plan.tissue_plan.target_label)
                )
                panda_cord_parent_band = (
                    anchor_zone
                    & ndimage.binary_dilation(
                        tissue_change,
                        iterations=max(1, int(np.ceil(diameter))),
                    )
                    & ~tissue_change
                    & np.isin(target_tissue, target_ids)
                )
                center_region |= panda_cord_parent_band
                population_target_region |= panda_cord_parent_band
                mechanism_region |= panda_cord_parent_band
        else:
            if plan.tissue_plan is not None or np.any(tissue_change):
                raise JointContractError("cell-only primitive forbids tissue changes")
            if case.cell_count_extent_budget is None:
                raise JointContractError(
                    "cell-only primitive requires count/extent budget"
                )
            if cell.layout_program_id == "localized_density_gradient":
                depletion = bundle.mechanism.cell_program.cellularity_depletion
                if depletion is None:
                    raise JointContractError(
                        "cellularity decrease has no executable depletion contract"
                    )
                if cell.spatial_anchor_type not in depletion.allowed_anchor_types:
                    raise JointContractError(
                        "Planner depletion anchor type is not skill-authorized"
                    )
                if cell.core_zone not in scene.population_zone_masks:
                    raise JointContractError(
                        "cellularity decrease requires one bound population component"
                    )
                component = np.asarray(
                    scene.population_zone_masks[cell.core_zone], dtype=bool
                )
                (
                    depletion_core,
                    depletion_transition,
                    depletion_outer,
                    depletion_anchor,
                ) = self._compile_depletion_regions(
                    scene=scene,
                    component=component,
                    component_id=cell.core_zone.removeprefix("pop:component:"),
                    spatial_anchor_type=cell.spatial_anchor_type,
                    cell_classes=cell.allowed_cell_classes,
                    interface_ids=cell.interface_ids,
                    anchor_ids=cell.anchor_ids,
                    allowed_neighbor_labels=depletion.allowed_neighbor_labels,
                    diameter_px=diameter,
                    core_width_cell_diameters=(
                        depletion.core_width_cell_diameters
                    ),
                    transition_width_cell_diameters=(
                        depletion.transition_width_cell_diameters
                    ),
                    outer_width_cell_diameters=(
                        depletion.outer_reference_width_cell_diameters
                    ),
                    maximum_extent_px=(
                        case.cell_count_extent_budget.maximum_extent_px
                    ),
                    maximize_outer_reference=(
                        bundle.annotation_profile.annotation_profile_id
                        in {
                            "glas-gland-v1",
                            "ignite-semantic-v1",
                            "orca-semantic-v1",
                            "puma-semantic-v1",
                        }
                    ),
                )
                center_region = depletion_core | depletion_transition
                population_target_region = (
                    center_region | depletion_outer
                )
                mechanism_region = population_target_region.copy()
                depletion_profile_id = depletion.program_id
                depletion_parameters = {
                    "core_width_cell_diameters": (
                        depletion.core_width_cell_diameters
                    ),
                    "transition_width_cell_diameters": (
                        depletion.transition_width_cell_diameters
                    ),
                    "outer_reference_width_cell_diameters": (
                        depletion.outer_reference_width_cell_diameters
                    ),
                    "core_removal_weight": depletion.core_removal_weight,
                    "transition_removal_weight": (
                        depletion.transition_removal_weight
                    ),
                    "resolution_mode": depletion.resolution_mode,
                    "core_target_removal_fraction": (
                        depletion.core_target_removal_fraction
                    ),
                    "transition_start_removal_fraction": (
                        depletion.transition_start_removal_fraction
                    ),
                    "transition_end_removal_fraction": (
                        depletion.transition_end_removal_fraction
                    ),
                    "transition_subband_count": (
                        depletion.transition_subband_count
                    ),
                    "minimum_core_residual_fraction": (
                        depletion.minimum_core_residual_fraction
                    ),
                    "minimum_transition_residual_fraction": (
                        depletion.minimum_transition_residual_fraction
                    ),
                    "minimum_core_removals": depletion.minimum_core_removals,
                    "minimum_transition_removals": (
                        depletion.minimum_transition_removals
                    ),
                    "maximum_new_gap_cell_diameters": (
                        depletion.maximum_new_gap_cell_diameters
                    ),
                    "minimum_outer_reference_instances": (
                        depletion.minimum_outer_reference_instances
                    ),
                    "minimum_field_area_cell_diameter_squares": (
                        depletion.minimum_field_area_cell_diameter_squares
                    ),
                }
            elif cell.core_zone in scene.population_zone_masks:
                center_region = self._bounded_population_zone(
                    scene=scene,
                    zone_id=cell.core_zone,
                    cell_classes=cell.allowed_cell_classes,
                    maximum_extent_px=case.cell_count_extent_budget.maximum_extent_px,
                )
            else:
                center_region = self._interface_zone(
                    scene=scene,
                    interface_ids=cell.interface_ids,
                    anchor_ids=cell.anchor_ids,
                    minimum_px=max(
                        case.cell_count_extent_budget.interface_min_px,
                        bundle.mechanism.cell_program.halo_distance_px[0],
                    ),
                    maximum_px=min(
                        case.cell_count_extent_budget.interface_max_px,
                        case.cell_count_extent_budget.maximum_extent_px,
                        bundle.mechanism.cell_program.halo_distance_px[1],
                    ),
                )
            if cell.layout_program_id != "localized_density_gradient":
                mechanism_region = center_region.copy()
                population_target_region = center_region.copy()

        prohibited = tuple(bundle.annotation_profile.prohibit_cell_placement_fine_ids)
        valid = ~np.isin(target_tissue, prohibited)
        receiving_region = np.ones_like(valid, dtype=bool)
        receiving_auxiliary = (
            bundle.mechanism.representability.receiving_auxiliary_structures
        )
        if receiving_auxiliary:
            for structure_id in receiving_auxiliary:
                structure = scene.auxiliary_structure_masks.get(structure_id)
                if structure is None:
                    raise JointContractError(
                        f"required receiving auxiliary {structure_id!r} is unavailable"
                    )
                receiving_region &= np.asarray(structure, dtype=bool)
            valid &= receiving_region
        if primitive.scope == "cell_only" and cell.core_zone.startswith(
            "pop:component:"
        ):
            selected_component_id = cell.core_zone.removeprefix("pop:component:")
            component_labels = {
                item.component_id: item.label for item in scene.tissue.graph.components
            }
            component_label = component_labels.get(selected_component_id)
            if (
                bundle.annotation_profile.annotation_profile_id
                == "panda-gleason-v1"
                and component_label == "Stroma"
                and cell.baseline_mode == "structured_add"
            ):
                external_stroma = scene.auxiliary_structure_masks.get(
                    "external_cellular_stroma_map"
                )
                if external_stroma is None:
                    raise JointContractError(
                        "PANDA stromal cell edits require external cellular stroma"
                    )
                external_stroma = np.asarray(external_stroma, dtype=bool)
                valid &= external_stroma
            compatible_classes = set(
                bundle.cell_observation_profile.tissue_compatible_classes.get(
                    component_label, ()
                )
            )
            if not set(cell.allowed_cell_classes).issubset(compatible_classes):
                raise JointContractError(
                    "cell program class is incompatible with the selected tissue component"
                )
            # Keep both centers and complete footprints inside the exact bound
            # component. Adjacent legal tissue is not an overflow buffer.
            valid &= np.asarray(scene.population_zone_masks[cell.core_zone], dtype=bool)
        if primitive.scope == "tissue_and_cell" and plan.tissue_plan is not None:
            valid &= np.isin(
                target_tissue,
                schema.resolve_fine_ids(plan.tissue_plan.target_label),
            )
        if primitive.host_tissue_labels:
            host_ids: set[int] = set()
            for label in primitive.host_tissue_labels:
                if label in schema.readable_labels:
                    host_ids.update(schema.resolve_fine_ids(label))
            valid &= np.isin(target_tissue, tuple(sorted(host_ids)))
        if primitive.primitive_id in {
            "peritumoral-neoplastic-scatter-increase-v1",
            "peritumoral-small-cluster-increase-v1",
        }:
            # The peritumoral postcondition defines its outer radius from the
            # source Tumor raster, whereas the interface compiler measures P
            # from a one-pixel boundary segment.  Those two distance fields
            # differ by one raster pixel at some orientations.  Bind V to the
            # biological Tumor-relative annulus as well, so exact footprint
            # containment and the final gate share one geometry authority.
            tumor = np.isin(
                target_tissue,
                tuple(schema.resolve_fine_ids("Tumor")),
            )
            outer_distance = ndimage.distance_transform_edt(~tumor)
            outer_maximum = max(
                1,
                int(bundle.mechanism.cell_program.halo_distance_px[1]),
            )
            valid &= ~tumor & (outer_distance <= outer_maximum)
        if (
            bundle.annotation_profile.annotation_profile_id == "glas-gland-v1"
            and primitive.primitive_id
            in {
                "peritumoral-neoplastic-scatter-increase-v1",
                "peritumoral-small-cluster-increase-v1",
            }
        ):
            native = scene.auxiliary_structure_masks.get(
                "native_gland_instance_map"
            )
            if native is None:
                raise JointContractError(
                    "GLaS periglandular placement requires native_gland_instance_map"
                )
            native_region = np.asarray(native) != 0
            native_outer_distance = ndimage.distance_transform_edt(
                ~native_region
            )
            valid &= (
                ~native_region
                & (native_outer_distance > 0)
                & (native_outer_distance <= outer_maximum)
            )
        if (
            bundle.annotation_profile.annotation_profile_id == "puma-semantic-v1"
            and bundle.mechanism.mechanism_id
            == "melanoma-discohesive-junctional"
            and primitive.primitive_id
            == "peritumoral-neoplastic-scatter-increase-v1"
        ):
            junction = scene.auxiliary_structure_masks.get(
                "epidermis_or_junction_map"
            )
            if junction is None:
                raise JointContractError(
                    "PUMA junctional scatter requires epidermis_or_junction_map"
                )
            junction = np.asarray(junction, dtype=bool)
            junction_distance = ndimage.distance_transform_edt(~junction)
            valid &= (
                ~junction
                & (junction_distance > 0)
                & (junction_distance <= outer_maximum)
            )
        if (
            primitive.scope == "cell_only"
            and cell.layout_program_id != "localized_density_gradient"
            and cell.interface_ids
        ):
            # Interface-local additions own one exact spatial extent: both
            # accepted centers and every pixel of each calibrated reference
            # footprint must remain inside the mechanism zone. Letting V span
            # the entire compatible host allowed complete nuclei whose centers
            # were legal to leak beyond the bounded peritumoral annulus.
            valid &= mechanism_region
        center_region &= valid
        mechanism_region &= valid
        if (
            primitive.scope == "cell_only"
            and cell.layout_program_id != "localized_density_gradient"
        ):
            population_target_region = center_region.copy()
        if not np.any(center_region):
            raise JointContractError(
                "compiled cell program has no legal placement center"
            )

        biological_delta = (
            case.cell_count_extent_budget.target_delta_count
            if case.cell_count_extent_budget is not None
            else None
        )
        resolved_delta = biological_delta
        if cell.baseline_mode == "render_owned_clearance":
            if (
                case.primitive_id
                not in bundle.mechanism.cell_program.render_owned_clearance_primitives
            ):
                raise JointContractError(
                    "mechanism does not authorize render-owned clearance"
                )
            if plan.tissue_plan is None:
                raise JointContractError(
                    "render-owned clearance requires a tissue transition"
                )
            source_classes = {
                target_cell_class_for_tissue(label, schema)
                for label in plan.tissue_plan.source_labels
            }
            protected_ids = set(cell.protected_instance_ids)
            selected = []
            erasure = np.zeros_like(tissue_change)
            for item in scene.cells.instances:
                component = np.asarray(
                    scene.instance_masks[item.instance_id],
                    dtype=bool,
                )
                if (
                    not _is_biological_instance(item)
                    or item.class_id not in source_classes
                    or item.instance_id in protected_ids
                    or item.touches_border
                    or item.completeness_status != "complete"
                    or item.quality_flags
                    or not _instance_center_is_in_region(item, receiving_region)
                    or not np.any(component & tissue_change)
                ):
                    continue
                selected.append(item.instance_id)
                erasure |= component
            if not selected:
                raise JointContractError(
                    "render-owned clearance found no complete viable source instance"
                )
            resolved_delta = len(selected)
        elif cell.baseline_mode == "selective_remove":
            if cell.layout_program_id == "localized_density_gradient":
                depletion = bundle.mechanism.cell_program.cellularity_depletion
                selected, depletion_authority = (
                    self._select_gradient_removal_instances(
                        scene=scene,
                        population_region=population_target_region,
                        core_region=depletion_core,
                        transition_region=depletion_transition,
                        outer_reference_region=depletion_outer,
                        anchor_mask=depletion_anchor,
                        valid_erasure_footprint_region=(
                            valid_erasure_footprint
                        ),
                        cell_classes=cell.allowed_cell_classes,
                        protected_instance_ids=cell.protected_instance_ids,
                        target_count=int(biological_delta or 0),
                        minimum_count=(
                            case.cell_count_extent_budget.min_delta_count
                            if case.cell_count_extent_budget is not None
                            else 0
                        ),
                        maximum_count=(
                            case.cell_count_extent_budget.max_delta_count
                            if case.cell_count_extent_budget is not None
                            else int(biological_delta or 0)
                        ),
                        nominal_nucleus_diameter_px=diameter,
                        contract=depletion,
                        selection_variant=int(
                            case.provenance.get(
                                "depletion_removal_selection_variant", 0
                            )
                        ),
                    )
                )
                depletion_population_instance_ids = (
                    depletion_authority.population_instance_ids
                )
                depletion_band_instance_ids = (
                    depletion_authority.band_instance_ids
                )
                depletion_radial_instance_ids = (
                    depletion_authority.radial_instance_ids
                )
                depletion_parameters.update(
                    {
                        "effective_core_end_px": (
                            depletion_authority.effective_core_end_px
                        ),
                        "effective_transition_width_px": (
                            depletion_authority.effective_transition_width_px
                        ),
                        "instance_authority": "complete_eligible_scene_instances",
                    }
                )
            else:
                selected = self._select_removal_instances(
                    scene=scene,
                    center_region=center_region,
                    cell_classes=cell.allowed_cell_classes,
                    protected_instance_ids=cell.protected_instance_ids,
                    target_count=int(biological_delta or 0),
                    minimum_count=(
                        case.cell_count_extent_budget.min_delta_count
                        if case.cell_count_extent_budget is not None
                        else 0
                    ),
                    preserve_class_composition=False,
                )
            erasure = np.zeros_like(tissue_change)
            for instance_id in selected:
                erasure |= np.asarray(scene.instance_masks[instance_id], dtype=bool)
            resolved_delta = len(selected)
        elif cell.baseline_mode == "regenerate_target_population":
            # E is an exact union of complete, removable source instances. It
            # is not the tissue-change raster T. Conflating the two previously
            # made every incomplete/border nucleus an implicit destructive
            # edit and also encouraged downstream code to use the eroded
            # center domain P as the population denominator.
            protected_ids = set(cell.protected_instance_ids)
            erasure = np.zeros_like(tissue_change)
            for item in scene.cells.instances:
                if (
                    not nucleus_instance_has_destructive_authority(item)
                    or item.instance_id in protected_ids
                    or item.touches_border
                    or item.completeness_status != "complete"
                    or item.quality_flags
                ):
                    continue
                component = np.asarray(
                    scene.instance_masks[item.instance_id], dtype=bool
                )
                # Tissue transitions own every complete removable instance
                # whose footprint intersects T.  Requiring the centroid to
                # lie in the receiving field left boundary-straddling source
                # nuclei partially embedded in the new tissue label; the
                # executable contract then (correctly) rejected them as an
                # incompatible retained population.  Whole-instance closure
                # and candidate feasibility already bound any spill outside T.
                if np.any(component & tissue_change):
                    erasure |= component
        else:
            erasure = np.zeros_like(tissue_change)
        support_radius = max(1, round(1.25 * diameter))
        support = (
            ndimage.binary_dilation(
                erasure | population_target_region,
                iterations=support_radius,
            )
            & valid
        )
        if plan.tissue_plan is not None:
            continuity_target_class = target_cell_class_for_tissue(
                plan.tissue_plan.target_label,
                schema,
            )
        elif len(primitive.target_cell_classes) == 1:
            continuity_target_class = primitive.target_cell_classes[0]
        else:
            continuity_target_class = cell.allowed_cell_classes[0]
        seam = compile_adaptive_seam(
            scene=scene,
            tissue_change=tissue_change,
            interface_ids=cell.interface_ids,
            anchor_ids=cell.anchor_ids,
            target_class=continuity_target_class,
            contract=bundle.mechanism.cell_program.seam_for(
                case.primitive_id
            ),
        )
        if (
            case.annotation_profile_id == "panda-gleason-v1"
            and case.primitive_id == "residual-tumor-fragmentation-v1"
        ):
            seam = replace(
                seam,
                mode="not_applicable",
                anchor_mask=np.zeros_like(tissue_change, dtype=bool),
                continuity_region=np.zeros_like(tissue_change, dtype=bool),
                minimum_anchor_coverage_fraction=0.0,
                requires_new_target_cells=False,
            )
        continuity_mode = seam.mode
        continuity_region = (
            seam.continuity_region | panda_cord_parent_band
        )
        continuity_requires_new_target_cells = seam.requires_new_target_cells
        if primitive.scope == "cell_only":
            # Cell-only mechanisms may be interface-local, but they do not
            # create an artificial tissue seam. Their spatial localization is
            # already enforced by P/V/M and the cell-zone gates.
            continuity_mode = "not_applicable"
            continuity_requires_new_target_cells = False
        elif cell.baseline_mode == "render_owned_clearance":
            continuity_mode = "render_owned_tissue_transition"
            continuity_requires_new_target_cells = False
        if np.any(continuity_region & ~center_region):
            raise JointContractError(
                "compiled continuity seam lies outside placement center region"
            )
        return CompiledCellToolProgram(
            program_id=cell.tool_program_id,
            primitive_id=case.primitive_id,
            mechanism_id=plan.selected_mechanism_id,
            baseline_mode=cell.baseline_mode,
            mechanism_program_id=cell.mechanism_program_id,
            quota_role=cell.mechanism_quota_role,
            population_target_region=population_target_region,
            erasure_region=erasure,
            placement_center_region=center_region,
            valid_footprint_region=valid,
            support_context_region=support,
            mechanism_region=mechanism_region,
            continuity_region=continuity_region,
            continuity_anchor_mask=seam.anchor_mask,
            depletion_core_region=depletion_core,
            depletion_transition_region=depletion_transition,
            depletion_outer_reference_region=depletion_outer,
            depletion_anchor_mask=depletion_anchor,
            depletion_anchor_type=cell.spatial_anchor_type,
            depletion_profile_id=depletion_profile_id,
            depletion_parameters=depletion_parameters,
            depletion_population_instance_ids=(
                depletion_population_instance_ids
            ),
            depletion_band_instance_ids=depletion_band_instance_ids,
            depletion_radial_instance_ids=depletion_radial_instance_ids,
            continuity_mode=continuity_mode,
            continuity_width_px=seam.width_px,
            continuity_maximum_empty_run_px=seam.maximum_empty_run_px,
            continuity_minimum_anchor_coverage_fraction=(
                seam.minimum_anchor_coverage_fraction
            ),
            continuity_density_ratio_range=seam.density_ratio_range,
            continuity_requires_new_target_cells=(
                continuity_requires_new_target_cells
            ),
            whole_instance_closure_px=max(1, round(2.0 * diameter)),
            target_classes=cell.allowed_cell_classes,
            selected_interface_ids=cell.interface_ids,
            selected_anchor_ids=cell.anchor_ids,
            nominal_nucleus_diameter_px=diameter,
            target_delta_count=(resolved_delta),
            biological_target_delta_count=biological_delta,
            minimum_effect_span_px=(
                max(
                    case.cell_count_extent_budget.minimum_effect_span_px,
                    skill_minimum_effect_span_px,
                )
                if case.cell_count_extent_budget is not None
                else skill_minimum_effect_span_px
            ),
            minimum_effect_foci=(
                max(
                    case.cell_count_extent_budget.minimum_effect_foci,
                    minimum_effect_foci,
                )
                if case.cell_count_extent_budget is not None
                else minimum_effect_foci
            ),
            policies={
                "T_pop": (
                    "changed-target-tissue-population-area"
                    if primitive.scope == "tissue_and_cell"
                    else "authorized-cell-only-population-zone"
                ),
                "E": cell.erasure_policy,
                "P": cell.placement_center_policy,
                "V": cell.valid_footprint_policy,
                "S": cell.probnet_context_policy,
                "reference_shapes": (
                    "native-same-patch-first-otherwise-digest-bound-"
                    "dataset-calibrated-complete-shapes"
                ),
                "instance_authority": (
                    "source-native-or-distance-watershed-for-count-density-"
                    "occupancy-and-removal"
                ),
                "shape_containment": (
                    "per-reference-footprint-exactly-certified-against-V"
                ),
                "counts": ("patch-adaptive-target-population-or-explicit-cell-budget"),
                "continuity": (
                    "planner-anchor-x-actual-tissue-change-x-local-cell-scale"
                ),
                "render_owned_material": (
                    "non-nuclear-debris-is-not-represented-as-nucleus-in-mask"
                    if cell.baseline_mode == "render_owned_clearance"
                    else "not_applicable"
                ),
            },
        )

    @staticmethod
    def _compile_depletion_regions(
        *,
        scene: JointSceneAnalysis,
        component: np.ndarray,
        component_id: str,
        spatial_anchor_type: str,
        cell_classes: tuple[int, ...],
        interface_ids: tuple[str, ...],
        anchor_ids: tuple[str, ...],
        allowed_neighbor_labels: tuple[str, ...],
        diameter_px: float,
        core_width_cell_diameters: float,
        transition_width_cell_diameters: float,
        outer_width_cell_diameters: float,
        maximum_extent_px: int,
        maximize_outer_reference: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compile an interface-inward three-band cellularity field."""

        component = np.asarray(component, dtype=bool)
        if spatial_anchor_type == "population_peak":
            eligible = []
            allowed_classes = set(cell_classes)
            for item in scene.cells.instances:
                row = int(round(item.centroid_xy[1]))
                col = int(round(item.centroid_xy[0]))
                if (
                    item.class_id in allowed_classes
                    and item.completeness_status == "complete"
                    and not item.touches_border
                    and not item.quality_flags
                    and 0 <= row < component.shape[0]
                    and 0 <= col < component.shape[1]
                    and component[row, col]
                ):
                    eligible.append((item.instance_id, row, col))
            if not eligible:
                raise JointContractError(
                    "population-peak depletion has no complete target-class nucleus"
                )
            points = np.asarray(
                [(row, col) for _instance_id, row, col in eligible],
                dtype=float,
            )
            radius = max(
                3.0 * float(diameter_px),
                (
                    float(core_width_cell_diameters)
                    + float(transition_width_cell_diameters)
                )
                * float(diameter_px),
            )
            neighborhoods = cKDTree(points).query_ball_point(points, radius)
            peak_index = min(
                range(len(eligible)),
                key=lambda index: (
                    -len(neighborhoods[index]),
                    eligible[index][0],
                ),
            )
            _instance_id, row, col = eligible[peak_index]
            anchor = np.zeros_like(component, dtype=bool)
            anchor[row, col] = True
        else:
            anchor = CellToolProgramCompiler._validated_anchor_mask(
                scene=scene,
                interface_ids=interface_ids,
                anchor_ids=anchor_ids,
            )
            interfaces = {
                item.interface_id: item
                for item in scene.tissue.graph.interfaces
            }
            allowed = set(allowed_neighbor_labels)
            for interface_id in interface_ids:
                interface = interfaces.get(interface_id)
                if interface is None:
                    raise JointContractError(
                        f"Planner selected unknown depletion interface {interface_id}"
                    )
                if interface.source_component_id == component_id:
                    neighbor = interface.target_label
                elif interface.target_component_id == component_id:
                    neighbor = interface.source_label
                else:
                    raise JointContractError(
                        "depletion interface does not touch the selected component"
                    )
                if neighbor not in allowed:
                    raise JointContractError(
                        f"depletion neighbor {neighbor!r} is not skill-authorized"
                    )
        distance = ndimage.distance_transform_edt(~anchor)
        maximum_observed_distance = float(
            np.max(distance[component], initial=0.0)
        )
        core_end, transition_end, outer_end = _depletion_band_edges(
            diameter_px=diameter_px,
            core_width_cell_diameters=core_width_cell_diameters,
            transition_width_cell_diameters=transition_width_cell_diameters,
            outer_width_cell_diameters=outer_width_cell_diameters,
            maximum_extent_px=maximum_extent_px,
            maximum_observed_distance_px=maximum_observed_distance,
            maximize_outer_reference=maximize_outer_reference,
        )
        core = component & (distance <= core_end)
        transition = component & (distance > core_end) & (
            distance <= transition_end
        )
        outer = component & (distance > transition_end)
        if not maximize_outer_reference:
            outer &= distance <= outer_end
        if not np.any(core) or not np.any(transition) or not np.any(outer):
            raise JointContractError(
                "depletion anchor cannot represent core, transition and outer-reference bands"
            )
        return core, transition, outer, anchor

    @staticmethod
    def _select_gradient_removal_instances(
        *,
        scene: JointSceneAnalysis,
        population_region: np.ndarray,
        core_region: np.ndarray,
        transition_region: np.ndarray,
        outer_reference_region: np.ndarray,
        anchor_mask: np.ndarray,
        valid_erasure_footprint_region: np.ndarray,
        cell_classes: tuple[int, ...],
        protected_instance_ids: tuple[str, ...],
        target_count: int,
        minimum_count: int,
        maximum_count: int,
        nominal_nucleus_diameter_px: float,
        contract,
        selection_variant: int = 0,
    ) -> tuple[tuple[str, ...], DepletionInstanceAuthority]:
        """Select complete nuclei with a stronger core than transition thinning."""

        field_area_ok, field_area, effective_minimum_field_area = (
            depletion_field_area_is_sufficient(
                core_region=core_region,
                transition_region=transition_region,
                outer_reference_region=outer_reference_region,
                nominal_nucleus_diameter_px=nominal_nucleus_diameter_px,
                minimum_field_area_cell_diameter_squares=(
                    contract.minimum_field_area_cell_diameter_squares
                ),
            )
        )
        if not field_area_ok:
            raise JointContractError(
                "depletion three-band field is below the skill-owned minimum "
                "area: "
                f"observed={field_area:.6f}, "
                f"minimum={effective_minimum_field_area:.6f}"
            )

        protected = set(protected_instance_ids)
        allowed = set(cell_classes)
        residual_area_floor = _qualified_residual_area_floor(
            scene, cell_classes
        )
        population = []
        by_band: dict[str, list] = {
            "core": [],
            "transition": [],
            "outer_reference": [],
        }
        for item in scene.cells.instances:
            x, y = item.centroid_xy
            row, col = round(y), round(x)
            if not (
                _is_biological_instance(
                    item,
                    residual_area_floor_px=residual_area_floor,
                )
                and item.class_id in allowed
                and 0 <= row < population_region.shape[0]
                and 0 <= col < population_region.shape[1]
                and population_region[row, col]
            ):
                continue
            population.append(item)
            if (
                item.instance_id in protected
                or item.touches_border
                or item.completeness_status != "complete"
                or item.quality_flags
            ):
                continue
            component = np.asarray(
                scene.instance_masks[item.instance_id], dtype=bool
            )
            if np.any(component & ~valid_erasure_footprint_region):
                continue
            overlaps_outer = np.any(component & outer_reference_region)
            if overlaps_outer and not outer_reference_region[row, col]:
                # The outer band is an unchanged local density reference, not
                # merely a center-exclusion band.
                continue
            if core_region[row, col]:
                by_band["core"].append(item)
            elif transition_region[row, col]:
                by_band["transition"].append(item)
            elif outer_reference_region[row, col]:
                by_band["outer_reference"].append(item)
        core_count = len(by_band["core"])
        transition_count = len(by_band["transition"])
        outer_reference_count = len(by_band["outer_reference"])
        if outer_reference_count < contract.minimum_outer_reference_instances:
            raise JointContractError(
                "depletion outer-reference band lacks enough unchanged "
                "complete nuclei"
            )
        core_capacity = max(
            0,
            core_count
            - int(np.ceil(core_count * contract.minimum_core_residual_fraction)),
        )
        transition_capacity = max(
            0,
            transition_count
            - int(
                np.ceil(
                    transition_count
                    * contract.minimum_transition_residual_fraction
                )
            ),
        )
        if (
            core_capacity < contract.minimum_core_removals
            or transition_capacity < contract.minimum_transition_removals
        ):
            raise JointContractError(
                "depletion bands lack enough complete nuclei after residual floors"
            )
        class_counts: dict[int, int] = {}
        for item in population:
            class_counts[item.class_id] = class_counts.get(item.class_id, 0) + 1
        maximum = min(
            max(0, int(target_count)), core_capacity + transition_capacity
        )
        minimum = max(
            int(minimum_count),
            contract.minimum_core_removals
            + contract.minimum_transition_removals,
        )
        anchor_distance = ndimage.distance_transform_edt(~anchor_mask)
        effective_core_end = float(
            np.max(anchor_distance[np.asarray(core_region, dtype=bool)], initial=0.0)
        )
        effective_transition_end = float(
            np.max(
                anchor_distance[np.asarray(transition_region, dtype=bool)],
                initial=effective_core_end,
            )
        )
        effective_transition_width = max(
            1.0, effective_transition_end - effective_core_end
        )
        radial_groups = CellToolProgramCompiler._density_field_radial_groups(
            by_band=by_band,
            anchor_distance=anchor_distance,
            contract=contract,
            effective_core_end_px=effective_core_end,
            effective_transition_width_px=effective_transition_width,
            include_outer_reference=True,
        )
        authority = DepletionInstanceAuthority(
            population_instance_ids=tuple(
                sorted(
                    item.instance_id
                    for band in ("core", "transition", "outer_reference")
                    for item in by_band[band]
                )
            ),
            band_instance_ids={
                band: tuple(sorted(item.instance_id for item in by_band[band]))
                for band in ("core", "transition", "outer_reference")
            },
            radial_instance_ids={
                name: tuple(sorted(item.instance_id for item in items))
                for name, items, _target in radial_groups
            },
            effective_core_end_px=effective_core_end,
            effective_transition_width_px=effective_transition_width,
        )
        if contract.resolution_mode == "density_field":
            selected = CellToolProgramCompiler._select_density_field_instances(
                scene=scene,
                population=population,
                by_band=by_band,
                anchor_distance=anchor_distance,
                cell_classes=cell_classes,
                minimum_count=minimum_count,
                maximum_count=maximum_count,
                contract=contract,
                effective_core_end_px=effective_core_end,
                effective_transition_width_px=effective_transition_width,
                selection_variant=selection_variant,
            )
            return selected, authority
        band_availability = {
            band: {
                class_id: sum(
                    item.class_id == class_id for item in by_band[band]
                )
                for class_id in class_counts
            }
            for band in ("core", "transition")
        }
        for resolved in range(maximum, minimum - 1, -1):
            class_quotas = _largest_remainder_quotas(class_counts, resolved)
            band_quotas = CellToolProgramCompiler._resolve_depletion_band_quotas(
                total=resolved,
                core_count=core_count,
                transition_count=transition_count,
                core_capacity=core_capacity,
                transition_capacity=transition_capacity,
                core_weight=contract.core_removal_weight,
                transition_weight=contract.transition_removal_weight,
                minimum_core=contract.minimum_core_removals,
                minimum_transition=contract.minimum_transition_removals,
            )
            if band_quotas is None:
                continue
            allocation = _allocate_class_band_counts(
                class_quotas=class_quotas,
                core_quota=band_quotas[0],
                availability=band_availability,
            )
            if allocation is None:
                continue
            selected = []
            for band in ("core", "transition"):
                for class_id in sorted(class_quotas):
                    quota = allocation[(class_id, band)]
                    candidates = [
                        item
                        for item in by_band[band]
                        if item.class_id == class_id
                    ]
                    candidates.sort(
                        key=lambda item: (
                            float(
                                anchor_distance[
                                    round(item.centroid_xy[1]),
                                    round(item.centroid_xy[0]),
                                ]
                            ),
                            _stable_instance_jitter(item.instance_id),
                            item.instance_id,
                        )
                    )
                    selected.extend(candidates[:quota])
            if len(selected) == resolved:
                return tuple(item.instance_id for item in selected), authority
        raise JointContractError(
            "no exact class-preserving core/transition depletion allocation is feasible"
        )

    @staticmethod
    def _resolve_depletion_band_quotas(
        *,
        total: int,
        core_count: int,
        transition_count: int,
        core_capacity: int,
        transition_capacity: int,
        core_weight: float,
        transition_weight: float,
        minimum_core: int,
        minimum_transition: int,
    ) -> tuple[int, int] | None:
        denominator = (
            core_weight * core_count + transition_weight * transition_count
        )
        desired_core = (
            total * core_weight * core_count / denominator
            if denominator > 0
            else total
        )
        feasible = []
        for core_quota in range(minimum_core, core_capacity + 1):
            transition_quota = total - core_quota
            if not minimum_transition <= transition_quota <= transition_capacity:
                continue
            core_fraction = core_quota / max(1, core_count)
            transition_fraction = transition_quota / max(1, transition_count)
            if not core_fraction > transition_fraction > 0:
                continue
            feasible.append(
                (
                    abs(core_quota - desired_core),
                    -core_fraction,
                    core_quota,
                    transition_quota,
                )
            )
        if not feasible:
            return None
        _, _, core_quota, transition_quota = min(feasible)
        return core_quota, transition_quota

    @staticmethod
    def _select_density_field_instances(
        *,
        scene: JointSceneAnalysis,
        population: list,
        by_band: dict[str, list],
        anchor_distance: np.ndarray,
        cell_classes: tuple[int, ...],
        minimum_count: int,
        maximum_count: int,
        contract,
        effective_core_end_px: float,
        effective_transition_width_px: float,
        selection_variant: int = 0,
    ) -> tuple[str, ...]:
        """Resolve deletion count from a radial density field, not a count target."""

        del population, cell_classes
        core_end = max(1.0, float(effective_core_end_px))
        transition_width = max(1.0, float(effective_transition_width_px))
        radial_bands = CellToolProgramCompiler._density_field_radial_groups(
            by_band=by_band,
            anchor_distance=anchor_distance,
            contract=contract,
            effective_core_end_px=core_end,
            effective_transition_width_px=transition_width,
            include_outer_reference=False,
        )
        quotas = []
        maximum_removals = []
        for name, items, target_fraction in radial_bands:
            residual_floor = (
                contract.minimum_core_residual_fraction
                if name == "core"
                else contract.minimum_transition_residual_fraction
            )
            maximum_removable = max(
                0, len(items) - int(np.ceil(len(items) * residual_floor))
            )
            maximum_removals.append(maximum_removable)
            quota = min(
                maximum_removable,
                int(np.floor(len(items) * target_fraction + 0.5)),
            )
            quotas.append(quota)
        quotas[0] = max(quotas[0], contract.minimum_core_removals)
        transition_total = sum(quotas[1:])
        if transition_total < contract.minimum_transition_removals:
            for index in range(1, len(radial_bands)):
                capacity = maximum_removals[index] - quotas[index]
                if capacity <= 0:
                    continue
                addition = min(
                    capacity,
                    contract.minimum_transition_removals - transition_total,
                )
                quotas[index] += addition
                transition_total += addition
                if transition_total >= contract.minimum_transition_removals:
                    break
        resolved = sum(quotas)
        if resolved > maximum_count:
            quotas = _cap_density_field_quotas(
                quotas=quotas,
                source_counts=[len(items) for _, items, _ in radial_bands],
                target_fractions=[
                    target_fraction
                    for _, _, target_fraction in radial_bands
                ],
                maximum_count=maximum_count,
                minimum_core=contract.minimum_core_removals,
                minimum_transition=contract.minimum_transition_removals,
            )
            resolved = sum(quotas)
        quotas = _enforce_density_field_gradient_quotas(
            quotas=quotas,
            source_counts=[len(items) for _, items, _ in radial_bands],
            maximum_removals=maximum_removals,
            target_fractions=[
                target_fraction for _, _, target_fraction in radial_bands
            ],
            minimum_count=minimum_count,
            maximum_count=maximum_count,
            minimum_core=contract.minimum_core_removals,
            minimum_transition=contract.minimum_transition_removals,
        )
        resolved = sum(quotas)
        if not minimum_count <= resolved <= maximum_count:
            raise JointContractError(
                "density field-derived removal count is outside its safety bounds"
            )
        radial_target_mismatches = []
        for (name, items, target_fraction), quota in zip(
            radial_bands, quotas
        ):
            source_count = len(items)
            if source_count <= 0:
                continue
            realized_fraction = quota / source_count
            tolerance = max(0.12, 1.0 / source_count)
            if abs(realized_fraction - target_fraction) > tolerance:
                radial_target_mismatches.append(
                    {
                        "band": name,
                        "source_count": source_count,
                        "quota": quota,
                        "realized_fraction": realized_fraction,
                        "target_fraction": target_fraction,
                        "tolerance": tolerance,
                    }
                )
        if radial_target_mismatches:
            raise JointContractError(
                "density field count budget cannot realize the skill-owned "
                "radial removal targets: "
                + "; ".join(
                    f"{item['band']}={item['quota']}/{item['source_count']} "
                    f"target={item['target_fraction']:.4f}"
                    for item in radial_target_mismatches
                )
            )
        if quotas[0] < contract.minimum_core_removals or sum(
            quotas[1:]
        ) < contract.minimum_transition_removals:
            raise JointContractError(
                "density field count cap violates minimum core/transition realization"
            )
        quota_by_band_and_class: list[dict[int, int]] = []
        all_classes: set[int] = set()
        for (_, items, _), quota in zip(radial_bands, quotas):
            class_counts: dict[int, int] = {}
            for item in items:
                class_counts[item.class_id] = class_counts.get(item.class_id, 0) + 1
            all_classes.update(class_counts)
            quota_by_band_and_class.append(
                _largest_remainder_quotas(class_counts, quota)
            )
        selected = []
        for class_id in sorted(all_classes):
            class_bands = [
                [item for item in items if item.class_id == class_id]
                for _name, items, _target in radial_bands
            ]
            class_removal_quotas = [
                class_quotas.get(class_id, 0)
                for class_quotas in quota_by_band_and_class
            ]
            fixed_retained = [
                item
                for item in by_band.get("outer_reference", ())
                if item.class_id == class_id
            ]
            selected.extend(
                _select_density_field_removals_preserving_coverage(
                    class_bands,
                    removal_quotas=class_removal_quotas,
                    fixed_retained=fixed_retained,
                    selection_variant=selection_variant,
                )
            )
        if len(selected) != resolved:
            raise JointContractError(
                "density field could not realize its complete-instance quotas"
            )
        return tuple(item.instance_id for item in selected)

    @staticmethod
    def _density_field_radial_groups(
        *,
        by_band: dict[str, list],
        anchor_distance: np.ndarray,
        contract,
        effective_core_end_px: float,
        effective_transition_width_px: float,
        include_outer_reference: bool,
    ) -> list[tuple[str, list, float]]:
        """Assign nuclei to the one radial ruler frozen in the contract."""

        core_end = max(1.0, float(effective_core_end_px))
        transition_width = max(1.0, float(effective_transition_width_px))
        subband_count = int(contract.transition_subband_count)
        radial_bands: list[tuple[str, list, float]] = [
            (
                "core",
                list(by_band["core"]),
                float(contract.core_target_removal_fraction),
            )
        ]
        transition_groups = [[] for _ in range(subband_count)]
        for item in by_band["transition"]:
            row, col = round(item.centroid_xy[1]), round(item.centroid_xy[0])
            normalized = max(
                0.0,
                (float(anchor_distance[row, col]) - core_end)
                / max(1e-6, transition_width),
            )
            index = min(subband_count - 1, int(normalized * subband_count))
            transition_groups[index].append(item)
        transition_targets = np.linspace(
            contract.transition_start_removal_fraction,
            contract.transition_end_removal_fraction,
            subband_count,
        )
        radial_bands.extend(
            (f"transition_{index + 1}", items, float(transition_targets[index]))
            for index, items in enumerate(transition_groups)
        )
        if include_outer_reference:
            radial_bands.append(
                (
                    "outer_reference",
                    list(by_band.get("outer_reference", ())),
                    0.0,
                )
            )
        return radial_bands

    @staticmethod
    def _bounded_population_zone(
        *,
        scene: JointSceneAnalysis,
        zone_id: str,
        cell_classes: tuple[int, ...],
        maximum_extent_px: int,
    ) -> np.ndarray:
        zone = np.asarray(scene.population_zone_masks[zone_id], dtype=bool)
        if not np.any(zone):
            raise JointContractError("selected population zone is empty")
        centers = []
        residual_area_floor = _qualified_residual_area_floor(
            scene, cell_classes
        )
        for item in scene.cells.instances:
            if (
                not _is_biological_instance(
                    item,
                    residual_area_floor_px=residual_area_floor,
                )
                or item.class_id not in cell_classes
            ):
                continue
            x, y = item.centroid_xy
            row, col = round(y), round(x)
            if 0 <= row < zone.shape[0] and 0 <= col < zone.shape[1] and zone[row, col]:
                centers.append((row, col))
        # ``maximum_extent_px`` is the diameter of the authorized local edit,
        # not a radius that may silently double the changed span.
        radius = max(1, int(maximum_extent_px) // 2)
        if centers:
            values = np.asarray(centers, dtype=float)
            pairwise = np.sqrt(
                np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2)
            )
            neighbor_counts = np.sum(pairwise <= radius, axis=1)
            interior = ndimage.distance_transform_edt(zone)
            index = max(
                range(len(centers)),
                key=lambda current: (
                    int(neighbor_counts[current]),
                    float(
                        interior[
                            round(values[current, 0]),
                            round(values[current, 1]),
                        ]
                    ),
                    -values[current, 0],
                    -values[current, 1],
                ),
            )
            center_y, center_x = values[index]
        else:
            center_y, center_x = ndimage.center_of_mass(zone)
        rows, cols = np.ogrid[: zone.shape[0], : zone.shape[1]]
        local = (rows - float(center_y)) ** 2 + (
            cols - float(center_x)
        ) ** 2 <= radius**2
        bounded = zone & local
        if not np.any(bounded):
            raise JointContractError(
                "population extent compiler produced an empty zone"
            )
        return bounded

    @staticmethod
    def _select_removal_instances(
        *,
        scene: JointSceneAnalysis,
        center_region: np.ndarray,
        cell_classes: tuple[int, ...],
        protected_instance_ids: tuple[str, ...],
        target_count: int,
        minimum_count: int,
        preserve_class_composition: bool,
    ) -> tuple[str, ...]:
        protected = set(protected_instance_ids)
        residual_area_floor = _qualified_residual_area_floor(
            scene, cell_classes
        )
        candidates = []
        for item in scene.cells.instances:
            if (
                not _is_biological_instance(
                    item,
                    residual_area_floor_px=residual_area_floor,
                )
                or item.instance_id in protected
                or item.class_id not in cell_classes
                or item.touches_border
                or item.completeness_status != "complete"
                or item.quality_flags
            ):
                continue
            x, y = item.centroid_xy
            row, col = round(y), round(x)
            if (
                0 <= row < center_region.shape[0]
                and 0 <= col < center_region.shape[1]
                and center_region[row, col]
            ):
                candidates.append(item)
        if len(candidates) < minimum_count:
            raise JointContractError(
                "selected population zone lacks enough complete removable instances"
            )
        resolved = min(max(0, target_count), len(candidates))
        if resolved <= 0:
            raise JointContractError("selective removal resolved to zero instances")
        if preserve_class_composition:
            class_counts: dict[int, int] = {}
            for item in candidates:
                class_counts[item.class_id] = class_counts.get(item.class_id, 0) + 1
            quotas = _largest_remainder_quotas(class_counts, resolved)
            selected = []
            for class_id, quota in sorted(quotas.items()):
                selected.extend(
                    CellToolProgramCompiler._spatially_disperse(
                        candidates=[
                            item for item in candidates if item.class_id == class_id
                        ],
                        count=quota,
                        center_region=center_region,
                    )
                )
            return tuple(item.instance_id for item in selected)
        selected = CellToolProgramCompiler._spatially_disperse(
            candidates=candidates,
            count=resolved,
            center_region=center_region,
        )
        return tuple(item.instance_id for item in selected)

    @staticmethod
    def _spatially_disperse(*, candidates, count, center_region):
        selected = []
        remaining = list(candidates)
        zone_center = np.asarray(ndimage.center_of_mass(center_region), dtype=float)
        while remaining and len(selected) < count:
            if not selected:
                next_item = min(
                    remaining,
                    key=lambda item: (
                        (item.centroid_xy[1] - zone_center[0]) ** 2
                        + (item.centroid_xy[0] - zone_center[1]) ** 2,
                        item.instance_id,
                    ),
                )
            else:
                chosen = np.asarray(
                    [(item.centroid_xy[1], item.centroid_xy[0]) for item in selected],
                    dtype=float,
                )
                next_item = max(
                    remaining,
                    key=lambda item: (
                        float(
                            np.min(
                                np.sum(
                                    (
                                        chosen
                                        - np.asarray(
                                            [
                                                item.centroid_xy[1],
                                                item.centroid_xy[0],
                                            ]
                                        )
                                    )
                                    ** 2,
                                    axis=1,
                                )
                            )
                        ),
                        item.instance_id,
                    ),
                )
            selected.append(next_item)
            remaining.remove(next_item)
        return tuple(selected)


    @staticmethod
    def _interface_zone(
        *,
        scene: JointSceneAnalysis,
        interface_ids: tuple[str, ...],
        anchor_ids: tuple[str, ...],
        minimum_px: int,
        maximum_px: int,
    ) -> np.ndarray:
        if not interface_ids:
            raise JointContractError("cell-only mechanism must bind interface IDs")
        masks = []
        for interface_id in interface_ids:
            try:
                masks.append(scene.tissue.interface_masks[interface_id])
            except KeyError as exc:
                raise JointContractError(
                    f"Planner selected unknown interface {interface_id}"
                ) from exc
        interface = np.logical_or.reduce(masks)
        anchors = CellToolProgramCompiler._validated_anchor_mask(
            scene=scene,
            interface_ids=interface_ids,
            anchor_ids=anchor_ids,
        )
        interface &= ndimage.binary_dilation(
            anchors,
            iterations=max(1, maximum_px),
        )
        distance = ndimage.distance_transform_edt(~interface)
        return (distance >= minimum_px) & (distance <= maximum_px)

    @staticmethod
    def _anchor_zone(
        *,
        scene: JointSceneAnalysis,
        interface_ids: tuple[str, ...],
        anchor_ids: tuple[str, ...],
        maximum_px: int,
    ) -> np.ndarray:
        anchors = CellToolProgramCompiler._validated_anchor_mask(
            scene=scene,
            interface_ids=interface_ids,
            anchor_ids=anchor_ids,
        )
        return ndimage.binary_dilation(
            anchors,
            iterations=max(1, int(maximum_px)),
        )

    @staticmethod
    def _validated_anchor_mask(
        *,
        scene: JointSceneAnalysis,
        interface_ids: tuple[str, ...],
        anchor_ids: tuple[str, ...],
    ) -> np.ndarray:
        if not interface_ids or not anchor_ids:
            raise JointContractError(
                "cell tool program requires interface and anchor IDs"
            )
        known_interfaces = set(interface_ids)
        metadata = {
            item.anchor_segment_id: item for item in scene.tissue.graph.anchor_segments
        }
        unknown = sorted(set(anchor_ids) - set(metadata))
        if unknown:
            raise JointContractError(
                "Planner selected unknown cell anchors: " + ", ".join(unknown)
            )
        detached = sorted(
            anchor_id
            for anchor_id in anchor_ids
            if metadata[anchor_id].interface_id not in known_interfaces
        )
        if detached:
            raise JointContractError(
                "cell anchors do not belong to selected interfaces: "
                + ", ".join(detached)
            )
        return np.logical_or.reduce(
            [scene.tissue.anchor_masks[anchor_id] for anchor_id in anchor_ids]
        )


def _depletion_band_edges(
    *,
    diameter_px: float,
    core_width_cell_diameters: float,
    transition_width_cell_diameters: float,
    outer_width_cell_diameters: float,
    maximum_extent_px: int,
    maximum_observed_distance_px: float,
    maximize_outer_reference: bool = False,
) -> tuple[float, float, float]:
    """Fit all three density bands inside the executable radial extent.

    Skill widths are desired proportions in local cell diameters.  A large
    patch-calibrated nucleus diameter can make their nominal sum exceed the
    instruction's maximum extent.  Clipping cumulative edges independently
    would collapse the outer-reference band to zero.  Instead, preserve the
    skill-owned proportions and scale the complete three-band program as one
    unit.  This changes neither the edit count nor the allowed host region.
    """

    nominal = np.asarray(
        [
            core_width_cell_diameters,
            transition_width_cell_diameters,
            outer_width_cell_diameters,
        ],
        dtype=np.float64,
    ) * max(1.0, float(diameter_px))
    if np.any(nominal <= 0):
        raise JointContractError("depletion band widths must all be positive")
    available = min(
        float(maximum_extent_px),
        float(maximum_observed_distance_px),
    )
    if available < 3.0:
        raise JointContractError(
            "depletion anchor has insufficient radial extent for three bands"
        )
    nominal_total = float(np.sum(nominal))
    if nominal_total <= available:
        widths = nominal
        if maximize_outer_reference:
            widths = widths.copy()
            widths[2] += available - nominal_total
    else:
        # The core+transition field carries the requested biological change;
        # the outer band is an unchanged density reference. Under an extent
        # conflict, reserve one local cell diameter for that reference and
        # distribute the remaining radius across core/transition in their
        # skill-owned ratio. The outer-instance gate still vetoes a reserve
        # that contains too few real nuclei.
        outer_width = min(float(nominal[2]), max(1.0, float(diameter_px)))
        editable_width = available - outer_width
        if editable_width <= 2.0:
            raise JointContractError(
                "depletion extent cannot reserve editable and outer-reference bands"
            )
        editable_nominal = float(nominal[0] + nominal[1])
        widths = np.asarray(
            [
                editable_width * float(nominal[0]) / editable_nominal,
                editable_width * float(nominal[1]) / editable_nominal,
                outer_width,
            ],
            dtype=np.float64,
        )
    core_end = float(widths[0])
    transition_end = float(widths[0] + widths[1])
    outer_end = float(np.sum(widths))
    if not 0.0 < core_end < transition_end < outer_end <= available + 1e-6:
        raise JointContractError(
            "depletion band compiler could not preserve three ordered bands"
        )
    return core_end, transition_end, outer_end


def _cap_density_field_quotas(
    *,
    quotas: list[int],
    source_counts: list[int],
    target_fractions: list[float],
    maximum_count: int,
    minimum_core: int,
    minimum_transition: int,
) -> list[int]:
    """Reduce a radial quota without flattening its biological gradient."""

    result = [max(0, int(value)) for value in quotas]
    maximum = max(0, int(maximum_count))
    while sum(result) > maximum:
        candidates = []
        for index, value in enumerate(result):
            if value <= 0:
                continue
            if index == 0 and value - 1 < int(minimum_core):
                continue
            if index > 0 and sum(result[1:]) - 1 < int(minimum_transition):
                continue
            trial = list(result)
            trial[index] -= 1
            fractions = [
                trial_value / source_count if source_count else 0.0
                for trial_value, source_count in zip(trial, source_counts)
            ]
            observed = [
                current
                for current, source_count in zip(fractions, source_counts)
                if source_count
            ]
            monotonic_violations = sum(
                outer > inner + 0.08
                for inner, outer in pairwise(observed)
            )
            errors = [
                abs(current - target)
                for current, target, source_count in zip(
                    fractions,
                    target_fractions,
                    source_counts,
                )
                if source_count
            ]
            candidates.append(
                (
                    monotonic_violations,
                    max(errors, default=0.0),
                    sum(errors),
                    -index,
                    trial,
                )
            )
        if not candidates:
            raise JointContractError(
                "density field count cap cannot preserve core/transition minima"
            )
        result = min(candidates)[-1]
    return result


def _enforce_density_field_gradient_quotas(
    *,
    quotas: list[int],
    source_counts: list[int],
    maximum_removals: list[int],
    target_fractions: list[float],
    minimum_count: int,
    maximum_count: int,
    minimum_core: int,
    minimum_transition: int,
) -> list[int]:
    """Make whole-instance quotas executable as a stronger-core gradient.

    Rounding independent target fractions can make core and transition
    depletion equal even when the continuous skill profile is ordered. This
    bounded repair changes the fewest whole nuclei while respecting residual,
    total-count and core/transition minima.
    """

    result = [max(0, int(value)) for value in quotas]
    counts = [max(0, int(value)) for value in source_counts]
    capacities = [max(0, int(value)) for value in maximum_removals]

    def aggregate_ordered(values: list[int]) -> bool:
        core_fraction = values[0] / max(1, counts[0])
        transition_source = sum(counts[1:])
        transition_fraction = sum(values[1:]) / max(1, transition_source)
        return core_fraction > transition_fraction > 0

    def radial_ordered(values: list[int]) -> bool:
        observed = [
            (value, count)
            for value, count in zip(values, counts)
            if count > 0
        ]
        for (inner_value, inner_count), (outer_value, outer_count) in pairwise(
            observed
        ):
            inner = inner_value / inner_count
            outer = outer_value / outer_count
            if outer <= inner + 1e-9:
                continue
            if max(0, outer_value - 1) / outer_count > inner + 1e-9:
                return False
        return True

    for _ in range(sum(capacities) + len(capacities) + 1):
        if (
            minimum_count <= sum(result) <= maximum_count
            and result[0] >= minimum_core
            and sum(result[1:]) >= minimum_transition
            and aggregate_ordered(result)
            and radial_ordered(result)
        ):
            return result
        candidates = []
        # Prefer removing an excessive outer quota. This reduces abrupt
        # transition depletion and never weakens the biologically strongest
        # core band.
        if sum(result) > minimum_count and sum(result[1:]) > minimum_transition:
            for index in range(1, len(result)):
                if result[index] <= 0:
                    continue
                trial = list(result)
                trial[index] -= 1
                errors = [
                    abs(value / max(1, count) - target)
                    for value, count, target in zip(
                        trial, counts, target_fractions
                    )
                    if count
                ]
                candidates.append(
                    (
                        0 if aggregate_ordered(trial) else 1,
                        0 if radial_ordered(trial) else 1,
                        max(errors, default=0.0),
                        sum(errors),
                        -index,
                        trial,
                    )
                )
        # If the minimum total prevents a transition reduction, strengthen the
        # core by one complete instance when its residual floor permits it.
        if (
            result[0] < capacities[0]
            and sum(result) < maximum_count
        ):
            trial = list(result)
            trial[0] += 1
            errors = [
                abs(value / max(1, count) - target)
                for value, count, target in zip(
                    trial, counts, target_fractions
                )
                if count
            ]
            candidates.append(
                (
                    0 if aggregate_ordered(trial) else 1,
                    0 if radial_ordered(trial) else 1,
                    max(errors, default=0.0),
                    sum(errors),
                    0,
                    trial,
                )
            )
        # The core may already be at its residual-safe capacity while rounded
        # transition quotas remain one instance below the minimum effect. In
        # that state a bounded transition increment is the only executable
        # repair (for example 7,2,1,1,0 -> 7,2,1,2,0).
        if sum(result) < minimum_count and sum(result) < maximum_count:
            for index in range(1, len(result)):
                if result[index] >= capacities[index]:
                    continue
                trial = list(result)
                trial[index] += 1
                errors = [
                    abs(value / max(1, count) - target)
                    for value, count, target in zip(
                        trial, counts, target_fractions
                    )
                    if count
                ]
                candidates.append(
                    (
                        0 if aggregate_ordered(trial) else 1,
                        0 if radial_ordered(trial) else 1,
                        max(errors, default=0.0),
                        sum(errors),
                        index,
                        trial,
                    )
                )
        if not candidates:
            break
        result = min(candidates)[-1]
    raise JointContractError(
        "density field cannot realize a whole-instance stronger-core gradient: "
        f"initial_quotas={quotas}, final_quotas={result}, "
        f"source_counts={counts}, maximum_removals={capacities}, "
        f"count_range=[{minimum_count},{maximum_count}]"
    )


def _select_density_field_removals_preserving_coverage(
    candidates_by_band: list[list],
    *,
    removal_quotas: list[int],
    fixed_retained: list,
    selection_variant: int = 0,
) -> list:
    """Remove radial quotas while leaving one global spatial coverage net.

    Density decrease should thin a population, not erase one compact cluster.
    Retention quotas remain band-specific, but retained nuclei are chosen
    jointly across all bands and the unchanged outer reference.
    """

    if len(candidates_by_band) != len(removal_quotas):
        raise JointContractError("density-field coverage quotas are misaligned")
    bands = [
        sorted(items, key=lambda item: item.instance_id)
        for items in candidates_by_band
    ]
    retain_needed = []
    for items, removal_quota in zip(bands, removal_quotas):
        quota = max(0, int(removal_quota))
        if quota > len(items):
            raise JointContractError("density-field class quota exceeds band capacity")
        retain_needed.append(len(items) - quota)
    retained_ids = {item.instance_id for item in fixed_retained}

    def point(item) -> np.ndarray:
        return np.asarray(
            (float(item.centroid_xy[1]), float(item.centroid_xy[0])),
            dtype=float,
        )

    retained_points = [point(item) for item in fixed_retained]
    all_items = [item for band in bands for item in band]
    if not retained_points and any(retain_needed):
        values = np.asarray([point(item) for item in all_items], dtype=float)
        centroid = np.mean(values, axis=0)
        eligible = [
            (band_index, item)
            for band_index, band in enumerate(bands)
            if retain_needed[band_index] > 0
            for item in band
        ]
        ordered_first = sorted(
            eligible,
            key=lambda pair: (
                float(np.sum((point(pair[1]) - centroid) ** 2)),
                _stable_instance_jitter(pair[1].instance_id),
                pair[1].instance_id,
            ),
        )
        first_band, first_item = ordered_first[
            int(selection_variant) % len(ordered_first)
        ]
        retained_ids.add(first_item.instance_id)
        retained_points.append(point(first_item))
        retain_needed[first_band] -= 1

    while any(retain_needed):
        choices = [
            (band_index, item)
            for band_index, band in enumerate(bands)
            if retain_needed[band_index] > 0
            for item in band
            if item.instance_id not in retained_ids
        ]
        if not choices or not retained_points:
            raise JointContractError(
                "density-field retention net cannot satisfy band quotas"
            )
        retained_array = np.asarray(retained_points, dtype=float)
        next_band, next_item = max(
            choices,
            key=lambda pair: (
                float(
                    np.min(
                        np.sum(
                            (retained_array - point(pair[1])) ** 2,
                            axis=1,
                        )
                    )
                ),
                -_stable_instance_jitter(pair[1].instance_id),
                pair[1].instance_id,
            ),
        )
        retained_ids.add(next_item.instance_id)
        retained_points.append(point(next_item))
        retain_needed[next_band] -= 1
    fixed_ids = {item.instance_id for item in fixed_retained}

    def maximum_coverage_gap(candidate_retained_ids: set[str]) -> float:
        retained = [
            point(item)
            for item in [*fixed_retained, *all_items]
            if item.instance_id in candidate_retained_ids
        ]
        if not retained:
            return float("inf")
        retained_array = np.asarray(retained, dtype=float)
        return max(
            (
                float(
                    np.sqrt(
                        np.min(
                            np.sum(
                                (retained_array - point(item)) ** 2,
                                axis=1,
                            )
                        )
                    )
                )
                for item in all_items
            ),
            default=0.0,
        )

    # Greedy farthest-point retention is followed by deterministic same-band
    # swaps. The swap cannot alter any radial quota, but it removes avoidable
    # local holes that a one-pass k-center approximation may leave behind.
    retained_ids |= fixed_ids
    current_gap = maximum_coverage_gap(retained_ids)
    while True:
        improvements = []
        for band in bands:
            retained_band = [
                item for item in band if item.instance_id in retained_ids
            ]
            removed_band = [
                item for item in band if item.instance_id not in retained_ids
            ]
            for retained_item in retained_band:
                for removed_item in removed_band:
                    trial = set(retained_ids)
                    trial.remove(retained_item.instance_id)
                    trial.add(removed_item.instance_id)
                    gap = maximum_coverage_gap(trial)
                    if gap + 1e-9 < current_gap:
                        improvements.append(
                            (
                                gap,
                                retained_item.instance_id,
                                removed_item.instance_id,
                                trial,
                            )
                        )
        if not improvements:
            break
        current_gap, _old_id, _new_id, retained_ids = min(
            improvements,
            key=lambda item: item[:3],
        )
    if selection_variant:
        alternatives = []
        for band in bands:
            retained_band = [
                item for item in band if item.instance_id in retained_ids
            ]
            removed_band = [
                item for item in band if item.instance_id not in retained_ids
            ]
            for retained_item in retained_band:
                for removed_item in removed_band:
                    trial = set(retained_ids)
                    trial.remove(retained_item.instance_id)
                    trial.add(removed_item.instance_id)
                    alternatives.append(
                        (
                            maximum_coverage_gap(trial),
                            retained_item.instance_id,
                            removed_item.instance_id,
                            trial,
                        )
                    )
        if alternatives:
            ordered = sorted(alternatives, key=lambda item: item[:3])
            retained_ids = ordered[
                (int(selection_variant) - 1) % len(ordered)
            ][-1]
    return [item for item in all_items if item.instance_id not in retained_ids]


def _largest_remainder_quotas(counts: dict[int, int], total: int) -> dict[int, int]:
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


def _allocate_class_band_counts(
    *,
    class_quotas: dict[int, int],
    core_quota: int,
    availability: dict[str, dict[int, int]],
) -> dict[tuple[int, str], int] | None:
    """Solve the tiny class-by-band transport problem without a solver dependency."""

    classes = sorted(class_quotas)

    def search(index: int, remaining_core: int, allocation: dict):
        if index == len(classes):
            return allocation if remaining_core == 0 else None
        class_id = classes[index]
        total = class_quotas[class_id]
        minimum_core = max(
            0, total - availability["transition"].get(class_id, 0)
        )
        maximum_core = min(
            total,
            availability["core"].get(class_id, 0),
            remaining_core,
        )
        for core_count in range(maximum_core, minimum_core - 1, -1):
            transition_count = total - core_count
            if transition_count > availability["transition"].get(class_id, 0):
                continue
            current = dict(allocation)
            current[(class_id, "core")] = core_count
            current[(class_id, "transition")] = transition_count
            resolved = search(index + 1, remaining_core - core_count, current)
            if resolved is not None:
                return resolved
        return None

    return search(0, core_quota, {})


def _stable_instance_jitter(instance_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(instance_id.encode("utf-8")).digest()[:8], "big"
    )


def _mask_digest(mask: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()
