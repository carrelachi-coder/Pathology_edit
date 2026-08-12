"""Exact, read-only feasibility witnesses for tissue interpretations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.agents import validate_edit_plan
from phase3_mask_edit_refine.execution import compile_edit_plan_with_witness
from phase3_mask_edit_refine.models import CaseContext, EditPlan
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

from .feasibility import (
    JointNucleiPreflight,
    augment_tissue_scene_with_nuclei_preflight,
)
from .models import JointContractError
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle
from .tissue_planner import MultiInterfaceResearchTissuePlanner

CANDIDATE_FEASIBILITY_COMPILER_VERSION = (
    "joint-candidate-feasibility-compiler-v1"
)


@dataclass(frozen=True)
class TissueExecutionWitness:
    """One in-memory whole-mask topology witness; no target is persisted."""

    compiled_plan: EditPlan
    attempt: int
    planner_usage: dict[str, Any]
    compiler_audit: dict[str, Any]
    replay_audit: dict[str, Any]
    realized_tissue_pixels: int
    change_region_sha256: str
    prior_errors: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_FEASIBILITY_COMPILER_VERSION,
            "attempt": self.attempt,
            "planner_usage": self.planner_usage,
            "selected_interface_ids": [
                item.interface_id
                for item in self.compiled_plan.candidate_interfaces
            ],
            "realized_tissue_pixels": self.realized_tissue_pixels,
            "change_region_sha256": self.change_region_sha256,
            "compiler": self.compiler_audit,
            "replay": self.replay_audit,
            "prior_errors": list(self.prior_errors),
            "persisted_target_mask": False,
        }


class CandidateFeasibilityCompiler:
    """Use the execution solver itself as the source-only capacity authority."""

    def __init__(self, *, maximum_attempts: int = 3) -> None:
        if maximum_attempts <= 0:
            raise ValueError("candidate feasibility attempts must be positive")
        self.maximum_attempts = maximum_attempts

    def compile_tissue_witness(
        self,
        *,
        tissue_case: CaseContext,
        source_tissue: np.ndarray,
        schema: MaskProfileSchema,
        scene: JointSceneAnalysis,
        tissue_bundle: ActiveKnowledgeBundle,
        joint_bundle: JointSkillBundle,
        nuclei_preflight: JointNucleiPreflight,
    ) -> TissueExecutionWitness:
        tissue_scene = augment_tissue_scene_with_nuclei_preflight(
            scene.tissue,
            nuclei_preflight,
            auxiliary_structure_masks=scene.auxiliary_structure_masks,
            required_auxiliary_structure_ids=(
                joint_bundle.mechanism.representability.protected_auxiliary_structures
            ),
        )
        planner = MultiInterfaceResearchTissuePlanner()
        feedback: dict[str, Any] = {}
        errors: list[str] = []
        for attempt in range(1, self.maximum_attempts + 1):
            try:
                raw_plan, planner_usage = planner.create_joint_tissue_plan(
                    case=tissue_case,
                    scene=tissue_scene,
                    bundle=tissue_bundle,
                    joint_bundle=joint_bundle,
                    image_paths=(),
                    nuclei_preflight=nuclei_preflight,
                    execution_feedback=feedback,
                )
                validate_edit_plan(
                    raw_plan,
                    case=tissue_case,
                    scene=tissue_scene,
                    bundle=tissue_bundle,
                )
                compiled, audit, parts, replay = compile_edit_plan_with_witness(
                    raw_plan,
                    source_mask=source_tissue,
                    schema=schema,
                    scene=tissue_scene,
                )
                realized = np.logical_or.reduce(
                    tuple(
                        np.asarray(part.change_region, dtype=bool)
                        for part in parts
                    )
                )
                return TissueExecutionWitness(
                    compiled_plan=compiled,
                    attempt=attempt,
                    planner_usage=dict(planner_usage),
                    compiler_audit=dict(audit),
                    replay_audit=dict(replay),
                    realized_tissue_pixels=int(np.count_nonzero(realized)),
                    change_region_sha256=hashlib.sha256(
                        np.ascontiguousarray(
                            realized.astype(np.uint8)
                        ).tobytes()
                    ).hexdigest(),
                    prior_errors=tuple(errors),
                )
            except Exception as exc:  # noqa: BLE001 - fail closed and audit any compiler failure
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                feedback = {
                    "retry_index": attempt,
                    "stage": "planning_or_compilation",
                    "errors": [error],
                    "failed_interface_ids": [],
                }
        raise JointContractError(
            "whole-mask topology witness is not executable after bounded "
            "replan: "
            + "; ".join(errors)
        )
