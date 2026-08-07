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

CELL_TOOL_COMPILER_VERSION = "joint-cell-tool-compiler-v4"


@dataclass(frozen=True)
class CompiledCellToolProgram:
    """Masks that separate cell semantics from model execution.

    E erases whole source instances, P accepts new centers, V must contain every
    new footprint, S is model context/render support, M is the mechanism zone,
    C is the candidate-local continuity region and A is the Planner-selected
    active anchor. Only E may destroy source cells; only P may supply centers;
    S never enters the edit budget.
    """

    program_id: str
    primitive_id: str
    mechanism_id: str
    baseline_mode: str
    mechanism_program_id: str
    quota_role: str
    erasure_region: np.ndarray
    placement_center_region: np.ndarray
    valid_footprint_region: np.ndarray
    support_context_region: np.ndarray
    mechanism_region: np.ndarray
    continuity_region: np.ndarray
    continuity_anchor_mask: np.ndarray
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
            "erasure_region",
            "placement_center_region",
            "valid_footprint_region",
            "support_context_region",
            "mechanism_region",
            "continuity_region",
            "continuity_anchor_mask",
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
            if cell.core_zone in scene.population_zone_masks:
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
            mechanism_region = center_region.copy()

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
                preserve_class_composition=(
                    case.primitive_id == "cellularity-decrease-v1"
                ),
            )
            erasure = np.zeros_like(tissue_change)
            for instance_id in selected:
                erasure |= np.asarray(scene.instance_masks[instance_id], dtype=bool)
            resolved_delta = len(selected)
        elif cell.baseline_mode == "regenerate_target_population":
            erasure = tissue_change.copy()
        else:
            erasure = np.zeros_like(tissue_change)
        diameter = float(scene.population.nominal_nucleus_diameter_px or 8.0)
        support_radius = max(1, round(1.25 * diameter))
        support = (
            ndimage.binary_dilation(
                erasure | center_region,
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
            erasure_region=erasure,
            placement_center_region=center_region,
            valid_footprint_region=valid,
            support_context_region=support,
            mechanism_region=mechanism_region,
            continuity_region=seam.continuity_region,
            continuity_anchor_mask=seam.anchor_mask,
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


def _mask_digest(mask: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()
