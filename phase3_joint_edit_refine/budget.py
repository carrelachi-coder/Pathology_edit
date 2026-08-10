"""Deterministic allocation of a joint budget to tissue and cell programs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from phase3_mask_edit_refine.models import AreaBudget, EditPlan

from .models import JointAreaBudget, JointContractError
from .skills.repository import JointSkillBundle

SOLVER_VERSION = "joint-budget-broker-v3"


@dataclass(frozen=True)
class JointBudgetAllocation:
    joint_target_pixels: int
    joint_hard_min_pixels: int
    joint_hard_max_pixels: int
    tissue_target_pixels: int
    tissue_floor_pixels: int
    tissue_execution_floor_pixels: int
    reserved_cell_only_pixels: int
    reserved_layout_halo_pixels: int
    reserved_cell_footprint_spill_pixels: int
    reserved_complete_instance_pixels: int
    fallback_policy: str
    solver_version: str = SOLVER_VERSION

    def to_metadata(self) -> dict:
        return self.__dict__.copy()


class JointFeasibilitySolver:
    """Broker the immutable task budget; neither Planner nor ProbNet sets it."""

    def allocate(
        self,
        *,
        shape: tuple[int, int],
        budget: JointAreaBudget,
        bundle: JointSkillBundle,
    ) -> JointBudgetAllocation:
        total = int(np.prod(shape))
        if total <= 0:
            raise JointContractError("cannot allocate a budget for an empty mask")
        joint_target = budget.target_pixels(shape)
        hard_min, hard_max = budget.hard_interval_pixels(shape)
        layout_reserve = round(
            total * bundle.mechanism.coupling.cell_only_target_fraction
        )
        # P is a legal *center* domain. A complete reference nucleus centered
        # on the newly edited side of a seam may legitimately straddle the
        # unchanged side. Those pixels belong to C/J even though no cell-only
        # center was requested. Keep that executable footprint reserve
        # separate from a true mechanism halo so budgeting does not change P.
        footprint_spill_reserve = round(
            total
            * bundle.mechanism.coupling.cell_footprint_spill_reserve_fraction
        )
        reserve = layout_reserve + footprint_spill_reserve
        tissue_floor = budget.tissue_floor_pixels(shape)
        execution_floor = budget.tissue_execution_floor_pixels(shape)
        tissue_target = max(tissue_floor, joint_target - reserve)
        tissue_target = min(tissue_target, joint_target)
        return JointBudgetAllocation(
            joint_target_pixels=joint_target,
            joint_hard_min_pixels=hard_min,
            joint_hard_max_pixels=hard_max,
            tissue_target_pixels=tissue_target,
            tissue_floor_pixels=tissue_floor,
            tissue_execution_floor_pixels=execution_floor,
            reserved_cell_only_pixels=max(0, joint_target - tissue_target),
            reserved_layout_halo_pixels=layout_reserve,
            reserved_cell_footprint_spill_pixels=footprint_spill_reserve,
            reserved_complete_instance_pixels=0,
            fallback_policy=budget.fallback_policy,
        )

    def reserve_complete_instances(
        self,
        allocation: JointBudgetAllocation,
        *,
        reserve_pixels: int,
        allow_capacity_floor_fallback: bool = False,
    ) -> JointBudgetAllocation:
        """Rebalance tissue burden for inevitable whole-instance closure.

        This is a deterministic feedback step after provisional tissue
        candidates exist.  It does not turn closure pixels into a requested
        cell-only mechanism and never lets the tissue target fall below its
        immutable burden floor.
        """

        closure = max(0, int(reserve_pixels))
        total_reserve = (
            allocation.reserved_layout_halo_pixels
            + allocation.reserved_cell_footprint_spill_pixels
            + closure
        )
        minimum_tissue_pixels = (
            allocation.tissue_execution_floor_pixels
            if allow_capacity_floor_fallback
            else allocation.tissue_floor_pixels
        )
        tissue_target = max(
            minimum_tissue_pixels,
            allocation.joint_target_pixels - total_reserve,
        )
        tissue_target = min(tissue_target, allocation.joint_target_pixels)
        return replace(
            allocation,
            tissue_target_pixels=tissue_target,
            reserved_cell_only_pixels=max(
                0, allocation.joint_target_pixels - tissue_target
            ),
            reserved_complete_instance_pixels=closure,
        )

    def reserve_observed_cell_spill(
        self,
        allocation: JointBudgetAllocation,
        *,
        complete_instance_pixels: int,
        footprint_spill_pixels: int,
        allow_capacity_floor_fallback: bool = False,
    ) -> JointBudgetAllocation:
        """Re-broker T from an executed candidate's exact C-only spill.

        ``complete_instance_pixels`` is the part of removed whole source
        instances outside T. ``footprint_spill_pixels`` is the part of newly
        placed complete target nuclei outside T.  They are kept separate in
        provenance because E and placement footprints have different
        semantics, even though both contribute to the union J.
        """

        complete = max(0, int(complete_instance_pixels))
        footprint = max(0, int(footprint_spill_pixels))
        # These are observed/compiled pixels, not heuristic reserves.  Any
        # requested halo contribution that actually changes nuclei is already
        # present in ``complete`` or ``footprint`` and must not be counted a
        # second time.
        total_reserve = complete + footprint
        minimum_tissue_pixels = (
            allocation.tissue_execution_floor_pixels
            if allow_capacity_floor_fallback
            else allocation.tissue_floor_pixels
        )
        tissue_target = max(
            minimum_tissue_pixels,
            allocation.joint_target_pixels - total_reserve,
        )
        tissue_target = min(tissue_target, allocation.joint_target_pixels)
        return replace(
            allocation,
            tissue_target_pixels=tissue_target,
            reserved_cell_only_pixels=max(
                0, allocation.joint_target_pixels - tissue_target
            ),
            reserved_complete_instance_pixels=complete,
            reserved_cell_footprint_spill_pixels=footprint,
        )

    def bind_tissue_plan(
        self,
        plan: EditPlan,
        *,
        shape: tuple[int, int],
        allocation: JointBudgetAllocation,
    ) -> EditPlan:
        total = int(np.prod(shape))
        target_fraction = allocation.tissue_target_pixels / max(1, total)
        floor_fraction = allocation.tissue_execution_floor_pixels / max(1, total)
        # Tissue generation is exact at its brokered target. The joint gate,
        # not the tissue generator, owns the wider 14--24% union contract.
        tissue_budget = AreaBudget(
            target_fraction=target_fraction,
            min_fraction=(min(target_fraction, floor_fraction)),
            max_fraction=target_fraction,
            basis="whole_mask",
            relative_tolerance=0.0,
            fallback_policy=(
                "max_feasible_below_target"
                if target_fraction > floor_fraction
                else "exact"
            ),
        )
        return replace(plan, area_budget=tissue_budget, resolved_area=None)
