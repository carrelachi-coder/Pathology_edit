"""Exact, read-only feasibility witnesses for tissue interpretations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

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
from .tissue_tools import compile_tissue_tool_program

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
    deterministic_candidate_metrics: dict[str, float]
    allowed_tool_families: tuple[str, ...]
    compiler_certificate_sha256: str
    tool_program_sha256: str

    @property
    def candidate_id(self) -> str:
        return "tissue-plan:" + self.compiler_certificate_sha256[:20]

    def validate_identity(self) -> None:
        payload = {
            "plan": self.compiled_plan.to_metadata(),
            "change_region_sha256": self.change_region_sha256,
            "metrics": self.deterministic_candidate_metrics,
            "tool_program_sha256": self.tool_program_sha256,
        }
        expected = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if expected != self.compiler_certificate_sha256:
            raise JointContractError(
                "tissue candidate compiler certificate SHA is detached"
            )

    def to_metadata(self) -> dict[str, Any]:
        self.validate_identity()
        return {
            "schema_version": CANDIDATE_FEASIBILITY_COMPILER_VERSION,
            "attempt": self.attempt,
            "planner_usage": self.planner_usage,
            "selected_interface_ids": [
                item.interface_id
                for item in self.compiled_plan.candidate_interfaces
            ],
            "selected_anchor_ids": [
                anchor_id
                for item in self.compiled_plan.candidate_interfaces
                for anchor_id in item.execution_contract.anchor_segment_ids
            ],
            "realized_tissue_pixels": self.realized_tissue_pixels,
            "change_region_sha256": self.change_region_sha256,
            "compiler": self.compiler_audit,
            "replay": self.replay_audit,
            "prior_errors": list(self.prior_errors),
            "candidate_id": self.candidate_id,
            "deterministic_candidate_metrics": dict(
                self.deterministic_candidate_metrics
            ),
            "allowed_tool_families": list(self.allowed_tool_families),
            "compiler_certificate_sha256": self.compiler_certificate_sha256,
            "tool_program_sha256": self.tool_program_sha256,
            "veto_reasons": [],
            "persisted_target_mask": False,
        }


@dataclass(frozen=True)
class TissueCandidateVeto:
    candidate_id: str
    preferred_interface_ids: tuple[str, ...]
    preferred_anchor_ids: tuple[str, ...]
    veto_reasons: tuple[str, ...]
    certificate_sha256: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "preferred_interface_ids": list(self.preferred_interface_ids),
            "preferred_anchor_ids": list(self.preferred_anchor_ids),
            "selectable": False,
            "veto_reasons": list(self.veto_reasons),
            "certificate_sha256": self.certificate_sha256,
        }


@dataclass(frozen=True)
class TissueCandidatePortfolio:
    survivors: tuple[TissueExecutionWitness, ...]
    vetoed: tuple[TissueCandidateVeto, ...]

    def __iter__(self) -> Iterator[TissueExecutionWitness]:
        return iter(self.survivors)

    def __len__(self) -> int:
        return len(self.survivors)

    def __getitem__(self, index: int) -> TissueExecutionWitness:
        return self.survivors[index]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_FEASIBILITY_COMPILER_VERSION,
            "surviving_candidates": [
                item.to_metadata() for item in self.survivors
            ],
            "vetoed_candidates": [item.to_metadata() for item in self.vetoed],
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
        return self.compile_tissue_portfolio(
            tissue_case=tissue_case,
            source_tissue=source_tissue,
            schema=schema,
            scene=scene,
            tissue_bundle=tissue_bundle,
            joint_bundle=joint_bundle,
            nuclei_preflight=nuclei_preflight,
            maximum_candidates=1,
        ).survivors[0]

    def compile_tissue_portfolio(
        self,
        *,
        tissue_case: CaseContext,
        source_tissue: np.ndarray,
        schema: MaskProfileSchema,
        scene: JointSceneAnalysis,
        tissue_bundle: ActiveKnowledgeBundle,
        joint_bundle: JointSkillBundle,
        nuclei_preflight: JointNucleiPreflight,
        maximum_candidates: int = 4,
    ) -> TissueCandidatePortfolio:
        """Compile immutable alternatives before any LLM candidate ranking."""

        feasible = tuple(nuclei_preflight.feasible_interface_ids)
        if not feasible:
            raise JointContractError("tissue portfolio has no feasible interface")
        witnesses: list[TissueExecutionWitness] = []
        seen_changes: set[str] = set()
        failures: list[str] = []
        vetoes: list[TissueCandidateVeto] = []
        # Compile the ordinary global solve, then independently prefer every
        # feasible interface/anchor pair.  An interface can expose several
        # legal spatial sectors; treating it as one candidate would collapse
        # the portfolio back to a single pre-selected plan and leave the LLM
        # nothing real to rank.
        preference_orders: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
            ((), ()),
        ]
        for preferred_interface in feasible:
            preflight_item = nuclei_preflight.interface(preferred_interface)
            anchor_ids = (
                preflight_item.cell_feasible_anchor_segment_ids
                if preflight_item is not None
                else ()
            )
            if not anchor_ids:
                preference_orders.append(
                    (
                        tuple(
                            value
                            for value in feasible
                            if value != preferred_interface
                        ),
                        (),
                    )
                )
                continue
            for anchor_id in anchor_ids:
                preference_orders.append(
                    (
                        tuple(
                            value
                            for value in feasible
                            if value != preferred_interface
                        ),
                        (anchor_id,),
                    )
                )
        for deprioritized, preferred_anchor_ids in preference_orders:
            try:
                witness = self._compile_one(
                    tissue_case=tissue_case,
                    source_tissue=source_tissue,
                    schema=schema,
                    scene=scene,
                    tissue_bundle=tissue_bundle,
                    joint_bundle=joint_bundle,
                    nuclei_preflight=nuclei_preflight,
                    initial_failed_interface_ids=deprioritized,
                    preferred_anchor_ids=preferred_anchor_ids,
                )
            except Exception as exc:  # noqa: BLE001 - every veto is audited
                error = f"{type(exc).__name__}: {exc}"
                failures.append(error)
                payload = {
                    "preferred_interface_ids": [
                        value
                        for value in feasible
                        if value not in deprioritized
                    ],
                    "preferred_anchor_ids": list(preferred_anchor_ids),
                    "veto_reasons": [error],
                }
                digest = hashlib.sha256(
                    json.dumps(
                        payload, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                vetoes.append(
                    TissueCandidateVeto(
                        candidate_id="tissue-veto:" + digest[:20],
                        preferred_interface_ids=tuple(
                            payload["preferred_interface_ids"]
                        ),
                        preferred_anchor_ids=tuple(
                            payload["preferred_anchor_ids"]
                        ),
                        veto_reasons=(error,),
                        certificate_sha256=digest,
                    )
                )
                continue
            if witness.change_region_sha256 in seen_changes:
                continue
            seen_changes.add(witness.change_region_sha256)
            witnesses.append(witness)
            if len(witnesses) >= max(1, maximum_candidates):
                break
        if not witnesses:
            raise JointContractError(
                "whole-mask topology portfolio has no executable survivor: "
                + "; ".join(failures)
            )
        return TissueCandidatePortfolio(
            survivors=tuple(witnesses),
            vetoed=tuple(vetoes),
        )

    def _compile_one(
        self,
        *,
        tissue_case: CaseContext,
        source_tissue: np.ndarray,
        schema: MaskProfileSchema,
        scene: JointSceneAnalysis,
        tissue_bundle: ActiveKnowledgeBundle,
        joint_bundle: JointSkillBundle,
        nuclei_preflight: JointNucleiPreflight,
        initial_failed_interface_ids: tuple[str, ...],
        preferred_anchor_ids: tuple[str, ...],
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
        feedback: dict[str, Any] = {
            "retry_index": 0,
            "stage": "candidate_portfolio_diversification",
            "errors": [],
            "failed_interface_ids": list(initial_failed_interface_ids),
            "preferred_anchor_ids": list(preferred_anchor_ids),
        }
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
                metrics = _measured_tissue_candidate_metrics(
                    compiled_plan=compiled,
                    compiler_audit=audit,
                    replay_parts=parts,
                    replay_audit=replay,
                    source_tissue=source_tissue,
                    scene=scene,
                    nuclei_preflight=nuclei_preflight,
                )
                tools = compile_tissue_tool_program(
                    primitive_id=compiled.primitive_id,
                    mechanism_id=joint_bundle.mechanism.mechanism_id,
                    mechanism_allowed_families=(
                        joint_bundle.mechanism.tissue_program.allowed_tools
                    ),
                    primitive_allowed_executors=(
                        tissue_bundle.edit_contract.allowed_tools
                    ),
                )
                certificate_payload = {
                    "plan": compiled.to_metadata(),
                    "change_region_sha256": hashlib.sha256(
                        np.ascontiguousarray(
                            realized.astype(np.uint8)
                        ).tobytes()
                    ).hexdigest(),
                    "metrics": metrics,
                    "tool_program_sha256": tools.program_sha256,
                }
                certificate_sha = hashlib.sha256(
                    json.dumps(
                        certificate_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
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
                    deterministic_candidate_metrics=metrics,
                    allowed_tool_families=tools.allowed_joint_families,
                    compiler_certificate_sha256=certificate_sha,
                    tool_program_sha256=tools.program_sha256,
                )
            except Exception as exc:  # noqa: BLE001 - fail closed and audit any compiler failure
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                feedback = {
                    "retry_index": attempt,
                    "stage": "planning_or_compilation",
                    "errors": [error],
                    "failed_interface_ids": [],
                    "preferred_anchor_ids": list(preferred_anchor_ids),
                }
        raise JointContractError(
            "whole-mask topology witness is not executable after bounded "
            "replan: "
            + "; ".join(errors)
        )


def _measured_tissue_candidate_metrics(
    *,
    compiled_plan: EditPlan,
    compiler_audit: dict[str, Any],
    replay_parts: tuple[Any, ...],
    replay_audit: dict[str, Any],
    source_tissue: np.ndarray,
    scene: JointSceneAnalysis,
    nuclei_preflight: JointNucleiPreflight,
) -> dict[str, float]:
    """Measure candidate metrics from compiled rasters and preflight ledgers."""

    selected_ids = {
        item.interface_id for item in compiled_plan.candidate_interfaces
    }
    capacities = [
        item for item in nuclei_preflight.interfaces if item.interface_id in selected_ids
    ]
    depth_values = [
        float(item.execution_contract.depth_profile.peak_depth_px)
        for item in compiled_plan.candidate_interfaces
    ]
    total_contact = float(sum(item.contact_pixels for item in capacities))
    maximum_depth = max(depth_values, default=0.0)
    protected = np.asarray(
        nuclei_preflight.protected_tissue_change_mask, dtype=bool
    )
    protected_distance = (
        ndimage.distance_transform_edt(~protected)
        if np.any(protected)
        else np.full(
            protected.shape,
            float(np.hypot(*protected.shape)),
            dtype=float,
        )
    )
    selected_anchors = np.zeros_like(protected)
    for planned in compiled_plan.candidate_interfaces:
        for anchor_id in planned.execution_contract.anchor_segment_ids:
            anchor = scene.tissue.anchor_masks.get(anchor_id)
            if anchor is not None:
                selected_anchors |= np.asarray(anchor, dtype=bool)
    min_protected_distance = (
        float(np.min(protected_distance[selected_anchors]))
        if np.any(selected_anchors)
        else 0.0
    )
    change = np.logical_or.reduce(
        tuple(np.asarray(part.change_region, dtype=bool) for part in replay_parts)
    )
    target_label = compiled_plan.target_label
    target_components = [
        item
        for item in scene.tissue.graph.components
        if item.label == target_label
        and np.any(
            change
            & ndimage.binary_dilation(
                scene.tissue.component_masks[item.component_id],
                structure=np.ones((3, 3), dtype=bool),
            )
        )
    ]
    projection_merge_count = max(0, len(target_components) - 1)
    topology = replay_audit.get("whole_mask_topology", {})
    structural_risk_count = float(
        sum(
            1
            for key, value in topology.items()
            if key != "passed"
            and (
                value is False
                or (isinstance(value, (int, float)) and "violation" in key and value > 0)
            )
        )
    )
    compiled_resolved = float(compiler_audit.get("resolved_pixels", 0))
    return {
        "depth_span_ratio": maximum_depth / max(1.0, total_contact),
        "packing_seam_capacity_margin": min(
            (float(item.capacity_margin_count) for item in capacities),
            default=0.0,
        ),
        "maximum_depth_px": maximum_depth,
        "protected_exclusion_count": float(
            sum(
                len(item.protected_fine_ids_within_band)
                + len(item.protected_instance_overlap_ids)
                for item in capacities
            )
        ),
        "anchor_length_depth_ratio": total_contact / max(1.0, maximum_depth),
        "class1_packing_margin": min(
            (float(item.capacity_margin_count) for item in capacities),
            default=0.0,
        ),
        "projection_merge_count": float(projection_merge_count),
        "protected_distance_px": min_protected_distance,
        "certificate_capacity_margin": compiled_resolved
        - float(nuclei_preflight.meaningful_tissue_floor_pixels),
        "structural_risk_count": structural_risk_count,
    }
