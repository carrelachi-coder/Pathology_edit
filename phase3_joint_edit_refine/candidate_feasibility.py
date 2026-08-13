"""Exact, read-only feasibility witnesses for tissue interpretations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.agents import validate_edit_plan
from phase3_mask_edit_refine.candidates import generate_candidates
from phase3_mask_edit_refine.execution import compile_edit_plan_with_witness
from phase3_mask_edit_refine.gates import GateContext, GateRegistry
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
    "joint-candidate-feasibility-compiler-v2"
)
_COMPILER_CAPABILITY_ISSUER = object()
_ISSUED_TISSUE_PORTFOLIOS: dict[int, tuple[tuple[int, str], ...]] = {}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _gate_semantics_sha256(report: Any) -> str:
    metadata = report.to_metadata()
    return _canonical_sha256(
        {
            "passed": bool(metadata["passed"]),
            "checks": metadata["checks"],
        }
    )


def _candidate_raster_sha256(candidate: Any) -> str:
    digest = hashlib.sha256()
    for name in ("target_mask", "change_region"):
        value = np.ascontiguousarray(np.asarray(getattr(candidate, name)))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    digest.update(str(candidate.tool_name).encode("utf-8"))
    return digest.hexdigest()


def _structural_event_risk_count(topology: Mapping[str, Any]) -> float:
    """Count observed topology events, never policy flags or absent events."""

    pairs = (
        ("source_components_before", "source_components_after"),
        ("target_components_before", "target_components_after"),
        ("source_holes_before", "source_holes_after"),
        ("target_holes_before", "target_holes_after"),
    )
    total = 0.0
    for before_key, after_key in pairs:
        before = topology.get(before_key)
        after = topology.get(after_key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            total += abs(float(after) - float(before))
    if topology.get("passed") is False:
        total += 1.0
    return total


def _gate_structural_event_risk_count(report: Any) -> float:
    topology = next(
        (
            item.metrics
            for item in report.checks
            if item.check_id == "edited_label_topology"
        ),
        None,
    )
    if not isinstance(topology, Mapping):
        raise JointContractError(
            "tissue gate report omits measured topology metrics"
        )
    return _structural_event_risk_count(
        {**topology, "passed": bool(report.passed)}
    )


def _narrow_plan_to_executor(plan: EditPlan, *, executor: str) -> EditPlan:
    from dataclasses import replace

    return replace(
        plan,
        tool_program=replace(
            plan.tool_program,
            allowed_tools=(executor,),
            candidate_count=max(1, plan.tool_program.candidate_count),
        ),
    )


@dataclass(frozen=True)
class TissueExecutionWitness:
    """Compiler-issued candidate/tool artifact that passed tissue hard gates."""

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
    selected_tool_family: str
    selected_concrete_executor: str
    execution_candidate: Any
    tissue_gate_report: Any
    authority_binding_sha256: str
    execution_raster_sha256: str
    tissue_gate_report_sha256: str
    _issuer: object | None = None

    @property
    def candidate_id(self) -> str:
        return "tissue-plan:" + self.compiler_certificate_sha256[:20]

    def validate_identity(self) -> None:
        if self._issuer is not _COMPILER_CAPABILITY_ISSUER:
            raise JointContractError(
                "tissue candidate was not issued by the execution compiler"
            )
        payload = {
            "plan": self.compiled_plan.to_metadata(),
            "change_region_sha256": self.change_region_sha256,
            "metrics": self.deterministic_candidate_metrics,
            "tool_program_sha256": self.tool_program_sha256,
            "selected_tool_family": self.selected_tool_family,
            "selected_concrete_executor": self.selected_concrete_executor,
            "authority_binding_sha256": self.authority_binding_sha256,
            "execution_raster_sha256": self.execution_raster_sha256,
            "tissue_gate_report_sha256": self.tissue_gate_report_sha256,
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
        if (
            not bool(self.tissue_gate_report.passed)
            or _candidate_raster_sha256(self.execution_candidate)
            != self.execution_raster_sha256
            or _canonical_sha256(self.tissue_gate_report.to_metadata())
            != self.tissue_gate_report_sha256
        ):
            raise JointContractError(
                "tissue candidate execution artifact is detached from its compiler certificate"
            )

    def validate_reexecution(
        self,
        *,
        candidates: tuple[Any, ...],
        gate_reports: tuple[Any, ...],
    ) -> str:
        """Require runtime execution to reproduce the pre-LLM certified raster.

        The Planner selects an immutable candidate/tool pair, not merely a
        compilable plan. Runtime may deterministically replay that pair, but it
        must reproduce the exact target/change raster and hard-gate report that
        the feasibility compiler certified before the LLM saw the portfolio.
        """

        self.validate_identity()
        for candidate in candidates:
            if _candidate_raster_sha256(candidate) != self.execution_raster_sha256:
                continue
            report = next(
                (
                    item
                    for item in gate_reports
                    if item.candidate_id == candidate.candidate_id
                ),
                None,
            )
            if report is None or not report.passed:
                continue
            if _gate_semantics_sha256(report) != _gate_semantics_sha256(
                self.tissue_gate_report
            ):
                raise JointContractError(
                    "runtime tissue gate report differs from the pre-LLM certificate"
                )
            return str(candidate.candidate_id)
        raise JointContractError(
            "runtime tissue execution did not reproduce the selected certified raster"
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
            "selected_tool_family": self.selected_tool_family,
            "selected_concrete_executor": self.selected_concrete_executor,
            "compiler_certificate_sha256": self.compiler_certificate_sha256,
            "tool_program_sha256": self.tool_program_sha256,
            "authority_binding_sha256": self.authority_binding_sha256,
            "execution_raster_sha256": self.execution_raster_sha256,
            "tissue_gate_report_sha256": self.tissue_gate_report_sha256,
            "hard_gate_passed": bool(self.tissue_gate_report.passed),
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
    authority_binding: dict[str, Any]
    authority_binding_sha256: str
    _issuer: object | None = None

    def __iter__(self) -> Iterator[TissueExecutionWitness]:
        return iter(self.survivors)

    def __len__(self) -> int:
        return len(self.survivors)

    def __getitem__(self, index: int) -> TissueExecutionWitness:
        return self.survivors[index]

    def validate_authority(self, *, expected_binding_sha256: str) -> None:
        issued = _ISSUED_TISSUE_PORTFOLIOS.get(id(self))
        observed = tuple(
            (id(item), item.compiler_certificate_sha256)
            for item in self.survivors
        )
        if (
            self._issuer is not _COMPILER_CAPABILITY_ISSUER
            or issued is None
            or issued != observed
            or _canonical_sha256(self.authority_binding)
            != self.authority_binding_sha256
            or self.authority_binding_sha256 != expected_binding_sha256
        ):
            raise JointContractError(
                "tissue portfolio is not the compiler-issued capability for this case/preflight/budget"
            )
        for item in self.survivors:
            item.validate_identity()

    def to_metadata(self) -> dict[str, Any]:
        if self._issuer is not _COMPILER_CAPABILITY_ISSUER:
            raise JointContractError(
                "tissue portfolio was not issued by the execution compiler"
            )
        return {
            "schema_version": CANDIDATE_FEASIBILITY_COMPILER_VERSION,
            "surviving_candidates": [
                item.to_metadata() for item in self.survivors
            ],
            "vetoed_candidates": [item.to_metadata() for item in self.vetoed],
            "authority_binding_sha256": self.authority_binding_sha256,
        }


class CandidateFeasibilityCompiler:
    """Use the execution solver itself as the source-only capacity authority."""

    def __init__(
        self,
        *,
        maximum_attempts: int = 3,
        gates: GateRegistry | None = None,
    ) -> None:
        if maximum_attempts <= 0:
            raise ValueError("candidate feasibility attempts must be positive")
        self.maximum_attempts = maximum_attempts
        self.gates = gates or GateRegistry()

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
        authority_binding: dict[str, Any],
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
            authority_binding=authority_binding,
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
        authority_binding: dict[str, Any],
        maximum_candidates: int = 4,
        revoked_candidate_ids: tuple[str, ...] = (),
    ) -> TissueCandidatePortfolio:
        """Compile immutable alternatives before any LLM candidate ranking."""

        authority_binding_sha = _canonical_sha256(authority_binding)
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
                    authority_binding_sha256=authority_binding_sha,
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
            if witness.candidate_id in set(revoked_candidate_ids):
                error = "candidate was revoked by deterministic execution feedback"
                vetoes.append(
                    TissueCandidateVeto(
                        candidate_id=witness.candidate_id,
                        preferred_interface_ids=tuple(
                            value
                            for value in feasible
                            if value not in deprioritized
                        ),
                        preferred_anchor_ids=preferred_anchor_ids,
                        veto_reasons=(error,),
                        certificate_sha256=witness.compiler_certificate_sha256,
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
        portfolio = TissueCandidatePortfolio(
            survivors=tuple(witnesses),
            vetoed=tuple(vetoes),
            authority_binding=dict(authority_binding),
            authority_binding_sha256=authority_binding_sha,
            _issuer=_COMPILER_CAPABILITY_ISSUER,
        )
        _ISSUED_TISSUE_PORTFOLIOS[id(portfolio)] = tuple(
            (id(item), item.compiler_certificate_sha256)
            for item in portfolio.survivors
        )
        return portfolio

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
        authority_binding_sha256: str,
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
                executable_artifacts: list[tuple[str, str, Any, Any]] = []
                for family, executor in zip(
                    tools.allowed_joint_families,
                    tools.allowed_concrete_executors,
                ):
                    family_plan = _narrow_plan_to_executor(
                        compiled,
                        executor=executor,
                    )
                    candidates = generate_candidates(
                        source_tissue,
                        schema=schema,
                        scene=tissue_scene,
                        plan=family_plan,
                        bundle=tissue_bundle,
                        seed=tissue_case.seed,
                        compiled_replay_parts=parts,
                        compiled_replay_audit=replay,
                    )
                    passing = []
                    for candidate in candidates:
                        report = self.gates.run(
                            GateContext(
                                case=tissue_case,
                                source_mask=source_tissue,
                                schema=schema,
                                scene=tissue_scene,
                                bundle=tissue_bundle,
                                plan=family_plan,
                                candidate=candidate,
                            )
                        )
                        if report.passed:
                            passing.append((candidate, report))
                    if not passing:
                        continue
                    candidate, report = passing[0]
                    executable_artifacts.append(
                        (family, executor, candidate, report)
                    )
                if not executable_artifacts:
                    raise JointContractError(
                        "compiled tissue plan has no concrete executor that passes all tissue hard gates"
                    )
                # Each witness binds exactly one candidate/tool pair. The
                # portfolio builder invokes this compiler per anchor; exposing
                # multiple families without separate raster certificates would
                # make the LLM choice ambiguous.
                family, executor, candidate, report = executable_artifacts[0]
                metrics = {
                    **metrics,
                    "structural_risk_count": (
                        _gate_structural_event_risk_count(report)
                    ),
                }
                execution_raster_sha = _candidate_raster_sha256(candidate)
                report_sha = _canonical_sha256(report.to_metadata())
                actual_change = np.asarray(candidate.change_region, dtype=bool)
                change_sha = hashlib.sha256(
                    np.ascontiguousarray(actual_change.astype(np.uint8)).tobytes()
                ).hexdigest()
                certificate_payload = {
                    "plan": compiled.to_metadata(),
                    "change_region_sha256": change_sha,
                    "metrics": metrics,
                    "tool_program_sha256": tools.program_sha256,
                    "selected_tool_family": family,
                    "selected_concrete_executor": executor,
                    "authority_binding_sha256": authority_binding_sha256,
                    "execution_raster_sha256": execution_raster_sha,
                    "tissue_gate_report_sha256": report_sha,
                }
                certificate_sha = _canonical_sha256(certificate_payload)
                return TissueExecutionWitness(
                    compiled_plan=compiled,
                    attempt=attempt,
                    planner_usage=dict(planner_usage),
                    compiler_audit=dict(audit),
                    replay_audit=dict(replay),
                    realized_tissue_pixels=int(np.count_nonzero(actual_change)),
                    change_region_sha256=change_sha,
                    prior_errors=tuple(errors),
                    deterministic_candidate_metrics=metrics,
                    allowed_tool_families=(family,),
                    compiler_certificate_sha256=certificate_sha,
                    tool_program_sha256=tools.program_sha256,
                    selected_tool_family=family,
                    selected_concrete_executor=executor,
                    execution_candidate=candidate,
                    tissue_gate_report=report,
                    authority_binding_sha256=authority_binding_sha256,
                    execution_raster_sha256=execution_raster_sha,
                    tissue_gate_report_sha256=report_sha,
                    _issuer=_COMPILER_CAPABILITY_ISSUER,
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
    structural_risk_count = _structural_event_risk_count(topology)
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
