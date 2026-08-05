"""Compile Planner cell intent into deterministic erasure/placement contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import CandidateMask

from .models import JointCaseContext, JointContractError, JointEditPlan
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle


CELL_TOOL_COMPILER_VERSION = "joint-cell-tool-compiler-v2"


@dataclass(frozen=True)
class CompiledCellToolProgram:
    """The four masks that separate cell semantics from model execution.

    E erases whole source instances, P accepts new centers, V must contain every
    new footprint, and S is model context/render support. Only E may destroy
    source cells; only P may supply centers; S never enters the edit budget.
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
    target_classes: tuple[int, ...]
    selected_interface_ids: tuple[str, ...]
    nominal_nucleus_diameter_px: float
    target_delta_count: int | None
    policies: dict[str, str]

    def to_metadata(self) -> dict:
        result = asdict(self)
        for key in (
            "erasure_region",
            "placement_center_region",
            "valid_footprint_region",
            "support_context_region",
            "mechanism_region",
        ):
            value = result.pop(key)
            result[f"{key}_pixels"] = int(np.count_nonzero(value))
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
        if cell.mechanism_program_id not in bundle.mechanism.cell_program.layout_programs:
            raise JointContractError("Planner mechanism program is not exposed by skill")
        if set(cell.allowed_cell_classes) - set(
            bundle.mechanism.cell_program.allowed_cell_classes
        ):
            raise JointContractError(
                "Planner requested a class outside the mechanism skill"
            )

        target_tissue = np.asarray(tissue_candidate.target_mask)
        tissue_change = np.asarray(tissue_candidate.change_region, dtype=bool)
        if primitive.scope == "tissue_and_cell":
            if plan.tissue_plan is None or not np.any(tissue_change):
                raise JointContractError(
                    "tissue primitive requires a nonempty tissue plan"
                )
            center_region = tissue_change.copy()
            mechanism_region = (
                ndimage.binary_dilation(
                    tissue_change,
                    iterations=bundle.mechanism.cell_program.halo_distance_px[1],
                )
                & ~tissue_change
                if bundle.mechanism.coupling.cell_only_target_fraction > 0
                else np.zeros_like(tissue_change)
            )
        else:
            if plan.tissue_plan is not None or np.any(tissue_change):
                raise JointContractError("cell-only primitive forbids tissue changes")
            if case.cell_count_extent_budget is None:
                raise JointContractError(
                    "cell-only primitive requires count/extent budget"
                )
            center_region = self._interface_zone(
                scene=scene,
                interface_ids=cell.interface_ids,
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

        prohibited = tuple(
            bundle.annotation_profile.prohibit_cell_placement_fine_ids
        )
        valid = ~np.isin(target_tissue, prohibited)
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

        destructive = cell.baseline_mode in {
            "regenerate_target_population",
            "selective_remove",
        }
        erasure = tissue_change.copy() if destructive else np.zeros_like(tissue_change)
        diameter = float(scene.population.nominal_nucleus_diameter_px or 8.0)
        support_radius = max(1, int(round(1.25 * diameter)))
        support = ndimage.binary_dilation(
            erasure | center_region,
            iterations=support_radius,
        ) & valid
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
            target_classes=cell.allowed_cell_classes,
            selected_interface_ids=cell.interface_ids,
            nominal_nucleus_diameter_px=diameter,
            target_delta_count=(
                case.cell_count_extent_budget.target_delta_count
                if case.cell_count_extent_budget is not None
                else None
            ),
            policies={
                "E": cell.erasure_policy,
                "P": cell.placement_center_policy,
                "V": cell.valid_footprint_policy,
                "S": cell.probnet_context_policy,
                "reference_shapes": (
                    "complete-nonborder-nonmerged-same-class-first"
                ),
                "counts": (
                    "patch-adaptive-target-population-or-explicit-cell-budget"
                ),
            },
        )

    @staticmethod
    def _interface_zone(
        *,
        scene: JointSceneAnalysis,
        interface_ids: tuple[str, ...],
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
        distance = ndimage.distance_transform_edt(~interface)
        return (distance >= minimum_px) & (distance <= maximum_px)
