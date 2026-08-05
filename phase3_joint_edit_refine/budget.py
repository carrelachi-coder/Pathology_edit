"""Deterministic allocation of a joint budget to tissue and cell programs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from phase3_mask_edit_refine.models import AreaBudget, EditPlan

from .models import JointAreaBudget, JointContractError
from .skills.repository import JointSkillBundle


SOLVER_VERSION = "joint-budget-broker-v1"


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
        reserve_fraction = bundle.mechanism.coupling.cell_only_target_fraction
        reserve = int(round(total * reserve_fraction))
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
            reserved_layout_halo_pixels=reserve,
            reserved_complete_instance_pixels=0,
            fallback_policy=budget.fallback_policy,
        )

    def reserve_complete_instances(
        self,
        allocation: JointBudgetAllocation,
        *,
        reserve_pixels: int,
    ) -> JointBudgetAllocation:
        """Rebalance tissue burden for inevitable whole-instance closure.

        This is a deterministic feedback step after provisional tissue
        candidates exist.  It does not turn closure pixels into a requested
        cell-only mechanism and never lets the tissue target fall below its
        immutable burden floor.
        """

        closure = max(0, int(reserve_pixels))
        total_reserve = allocation.reserved_layout_halo_pixels + closure
        tissue_target = max(
            allocation.tissue_execution_floor_pixels,
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
            min_fraction=(floor_fraction if target_fraction > floor_fraction else target_fraction),
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
