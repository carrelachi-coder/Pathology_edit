"""Compile Planner cell intent into deterministic erasure/placement contracts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import CandidateMask

from .models import JointCaseContext, JointContractError, JointEditPlan
from .scene import JointSceneAnalysis
from .seam import compile_adaptive_seam, target_cell_class_for_tissue
from .skills.repository import JointSkillBundle

CELL_TOOL_COMPILER_VERSION = "joint-cell-tool-compiler-v6"


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
        if (
            cell.mechanism_program_id
            not in bundle.mechanism.cell_program.layout_programs
        ):
            raise JointContractError(
                "Planner mechanism program is not exposed by skill"
            )
        if set(cell.allowed_cell_classes) - set(
            bundle.mechanism.cell_program.allowed_cell_classes
        ):
            raise JointContractError(
                "Planner requested a class outside the mechanism skill"
            )

        target_tissue = np.asarray(tissue_candidate.target_mask)
        tissue_change = np.asarray(tissue_candidate.change_region, dtype=bool)
        diameter = float(scene.population.nominal_nucleus_diameter_px or 8.0)
        empty = np.zeros_like(tissue_change, dtype=bool)
        depletion_core = empty.copy()
        depletion_transition = empty.copy()
        depletion_outer = empty.copy()
        depletion_anchor = empty.copy()
        depletion_parameters: dict[str, float | int | str] = {}
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
        else:
            if plan.tissue_plan is not None or np.any(tissue_change):
                raise JointContractError("cell-only primitive forbids tissue changes")
            if case.cell_count_extent_budget is None:
                raise JointContractError(
                    "cell-only primitive requires count/extent budget"
                )
            if case.primitive_id == "cellularity-decrease-v1":
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
            if case.primitive_id != "cellularity-decrease-v1":
                mechanism_region = center_region.copy()
                population_target_region = center_region.copy()

        prohibited = tuple(bundle.annotation_profile.prohibit_cell_placement_fine_ids)
        valid = ~np.isin(target_tissue, prohibited)
        if primitive.scope == "cell_only" and cell.core_zone.startswith(
            "pop:component:"
        ):
            selected_component_id = cell.core_zone.removeprefix("pop:component:")
            component_labels = {
                item.component_id: item.label for item in scene.tissue.graph.components
            }
            component_label = component_labels.get(selected_component_id)
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
        center_region &= valid
        mechanism_region &= valid
        if (
            primitive.scope == "cell_only"
            and case.primitive_id != "cellularity-decrease-v1"
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
                    item.class_id not in source_classes
                    or item.instance_id in protected_ids
                    or item.touches_border
                    or item.completeness_status != "complete"
                    or item.quality_flags
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
            if case.primitive_id == "cellularity-decrease-v1":
                depletion = bundle.mechanism.cell_program.cellularity_depletion
                selected = self._select_gradient_removal_instances(
                    scene=scene,
                    population_region=population_target_region,
                    core_region=depletion_core,
                    transition_region=depletion_transition,
                    outer_reference_region=depletion_outer,
                    anchor_mask=depletion_anchor,
                    cell_classes=cell.allowed_cell_classes,
                    protected_instance_ids=cell.protected_instance_ids,
                    target_count=int(biological_delta or 0),
                    minimum_count=(
                        case.cell_count_extent_budget.min_delta_count
                        if case.cell_count_extent_budget is not None
                        else 0
                    ),
                    contract=depletion,
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
                    item.instance_id in protected_ids
                    or item.touches_border
                    or item.completeness_status != "complete"
                    or item.quality_flags
                ):
                    continue
                component = np.asarray(
                    scene.instance_masks[item.instance_id], dtype=bool
                )
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
            contract=bundle.mechanism.cell_program.seam,
        )
        continuity_mode = seam.mode
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
        if np.any(seam.continuity_region & ~center_region):
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
            continuity_region=seam.continuity_region,
            continuity_anchor_mask=seam.anchor_mask,
            depletion_core_region=depletion_core,
            depletion_transition_region=depletion_transition,
            depletion_outer_reference_region=depletion_outer,
            depletion_anchor_mask=depletion_anchor,
            depletion_anchor_type=cell.spatial_anchor_type,
            depletion_profile_id=depletion_profile_id,
            depletion_parameters=depletion_parameters,
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
                "reference_shapes": ("complete-nonborder-nonmerged-same-class-first"),
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
        interface_ids: tuple[str, ...],
        anchor_ids: tuple[str, ...],
        allowed_neighbor_labels: tuple[str, ...],
        diameter_px: float,
        core_width_cell_diameters: float,
        transition_width_cell_diameters: float,
        outer_width_cell_diameters: float,
        maximum_extent_px: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compile an interface-inward three-band cellularity field."""

        anchor = CellToolProgramCompiler._validated_anchor_mask(
            scene=scene,
            interface_ids=interface_ids,
            anchor_ids=anchor_ids,
        )
        interfaces = {
            item.interface_id: item for item in scene.tissue.graph.interfaces
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
        core_end = min(
            float(maximum_extent_px),
            max(1.0, core_width_cell_diameters * diameter_px),
        )
        transition_end = min(
            float(maximum_extent_px),
            core_end + transition_width_cell_diameters * diameter_px,
        )
        outer_end = min(
            float(maximum_extent_px),
            transition_end + outer_width_cell_diameters * diameter_px,
        )
        component = np.asarray(component, dtype=bool)
        core = component & (distance <= core_end)
        transition = component & (distance > core_end) & (
            distance <= transition_end
        )
        outer = component & (distance > transition_end) & (
            distance <= outer_end
        )
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
        cell_classes: tuple[int, ...],
        protected_instance_ids: tuple[str, ...],
        target_count: int,
        minimum_count: int,
        contract,
    ) -> tuple[str, ...]:
        """Select complete nuclei with a stronger core than transition thinning."""

        protected = set(protected_instance_ids)
        allowed = set(cell_classes)
        population = []
        by_band: dict[str, list] = {"core": [], "transition": []}
        for item in scene.cells.instances:
            x, y = item.centroid_xy
            row, col = round(y), round(x)
            if not (
                item.class_id in allowed
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
            if np.any(component & outer_reference_region):
                # The outer band is an unchanged local density reference, not
                # merely a center-exclusion band.
                continue
            if core_region[row, col]:
                by_band["core"].append(item)
            elif transition_region[row, col]:
                by_band["transition"].append(item)
        core_count = len(by_band["core"])
        transition_count = len(by_band["transition"])
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
                return tuple(item.instance_id for item in selected)
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
        for item in scene.cells.instances:
            if item.class_id not in cell_classes:
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
        candidates = []
        for item in scene.cells.instances:
            if (
                item.instance_id in protected
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
