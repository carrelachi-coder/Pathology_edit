"""One-round workflow orchestration for reviewed directional-anchor subdivision.

Track B is deliberately entered only after the ordinary Track A portfolio has
failed with a topology-passing area-underfill receipt.  The transformed scene,
nuclei preflight, authority binding and compiler portfolio are all rebuilt; no
parent-anchor feasibility or witness is reused.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from phase3_mask_edit_refine.research_anchor_subdivision_v1 import (
    CANDIDATE_VERSION,
    DIRECTIONAL_SECTOR_CANDIDATE_VERSION,
    geodesic_span_steps,
    minimum_witness_contact_pixels,
    rebuild_scene_with_directional_sector_partition,
    rebuild_scene_with_bisected_single_anchor,
    subdivision_eligibility_for_directional_front_block,
    subdivision_eligibility_for_underfill_replan,
)

from .candidate_feasibility import (
    CandidateFeasibilityCompiler,
    TissueCandidatePortfolio,
    TissuePortfolioCompilationError,
)
from .feasibility import JointNucleiPreflight, build_joint_nuclei_preflight
from .models import JointContractError
from .portfolio_authority import (
    build_tissue_portfolio_authority_binding,
    canonical_metadata_sha256,
)
from .scene import JointSceneAnalysis

TRACK_B_WORKFLOW_VERSION = "directional-anchor-subdivision-workflow-v2"


@dataclass(frozen=True)
class TrackBPortfolioResult:
    portfolio: TissueCandidatePortfolio
    scene: JointSceneAnalysis
    nuclei_preflight: JointNucleiPreflight
    authority_binding: dict[str, Any]
    track_b_used: bool
    audit: dict[str, Any]


class TrackBPortfolioCompilationError(JointContractError):
    """Track B ran once but did not replace the audited Track A failure."""

    def __init__(
        self,
        *,
        original_error: TissuePortfolioCompilationError,
        track_b_error: TissuePortfolioCompilationError,
        audit: dict[str, Any],
    ) -> None:
        super().__init__(
            "ordinary Track A portfolio failed and the single fresh Track B "
            f"portfolio also failed: {track_b_error}"
        )
        self.original_error = original_error
        self.track_b_error = track_b_error
        self.audit = dict(audit)


def compile_tissue_portfolio_with_track_b(
    *,
    compiler: CandidateFeasibilityCompiler,
    joint_case: Any,
    tissue_case: Any,
    source_tissue: np.ndarray,
    schema: Any,
    scene: JointSceneAnalysis,
    tissue_bundle: Any,
    joint_bundle: Any,
    allocation: Any,
    nuclei_preflight: JointNucleiPreflight,
    maximum_candidates: int = 4,
    revoked_candidate_ids: tuple[str, ...] = (),
) -> TrackBPortfolioResult:
    """Compile Track A, then at most one fresh Track B portfolio if eligible."""

    original_binding = build_tissue_portfolio_authority_binding(
        joint_case=joint_case,
        tissue_case=tissue_case,
        source_tissue=source_tissue,
        joint_bundle=joint_bundle,
        tissue_bundle=tissue_bundle,
        allocation=allocation,
        nuclei_preflight=nuclei_preflight,
    )
    try:
        portfolio = compiler.compile_tissue_portfolio(
            tissue_case=tissue_case,
            source_tissue=source_tissue,
            schema=schema,
            scene=scene,
            tissue_bundle=tissue_bundle,
            joint_bundle=joint_bundle,
            nuclei_preflight=nuclei_preflight,
            authority_binding=original_binding,
            maximum_candidates=maximum_candidates,
            revoked_candidate_ids=revoked_candidate_ids,
        )
    except TissuePortfolioCompilationError as original_error:
        return _compile_fresh_track_b_portfolio(
            original_error=original_error,
            compiler=compiler,
            joint_case=joint_case,
            tissue_case=tissue_case,
            source_tissue=source_tissue,
            schema=schema,
            scene=scene,
            tissue_bundle=tissue_bundle,
            joint_bundle=joint_bundle,
            allocation=allocation,
            nuclei_preflight=nuclei_preflight,
            maximum_candidates=maximum_candidates,
            revoked_candidate_ids=revoked_candidate_ids,
            original_binding=original_binding,
        )
    return TrackBPortfolioResult(
        portfolio=portfolio,
        scene=scene,
        nuclei_preflight=nuclei_preflight,
        authority_binding=original_binding,
        track_b_used=False,
        audit={
            "workflow_version": TRACK_B_WORKFLOW_VERSION,
            "track_b_used": False,
            "reason": "ordinary_track_a_portfolio_succeeded",
            "original_authority_binding_sha256": canonical_metadata_sha256(
                original_binding
            ),
        },
    )


def _compile_fresh_track_b_portfolio(
    *,
    original_error: TissuePortfolioCompilationError,
    compiler: CandidateFeasibilityCompiler,
    joint_case: Any,
    tissue_case: Any,
    source_tissue: np.ndarray,
    schema: Any,
    scene: JointSceneAnalysis,
    tissue_bundle: Any,
    joint_bundle: Any,
    allocation: Any,
    nuclei_preflight: JointNucleiPreflight,
    maximum_candidates: int,
    revoked_candidate_ids: tuple[str, ...],
    original_binding: dict[str, Any],
) -> TrackBPortfolioResult:
    underfills = tuple(
        {
            **item,
            "trigger_feedback": feedback,
        }
        for item in original_error.structured_failures
        for feedback in item.get(
            "feedback_history",
            (item.get("terminal_feedback", {}),),
        )
        if _is_topology_passing_underfill(feedback)
    )
    directional_front_block = (
        not underfills and _is_directional_single_anchor_front_block(original_error)
    )
    if not underfills and not directional_front_block:
        raise original_error

    front = joint_bundle.mechanism.tissue_program.front
    minimum_pixels = minimum_witness_contact_pixels(
        nuclei_preflight.reference_area_p95
    )
    remaining_ids = (
        tuple(nuclei_preflight.feasible_interface_ids)
        if directional_front_block
        else tuple(
            sorted(
                {
                    str(interface_id)
                    for failure in underfills
                    for interface_id in failure["trigger_feedback"].get(
                        "remaining_feasible_interface_ids", ()
                    )
                }
            )
        )
    )
    transformed_tissue = scene.tissue
    transformed_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    for interface_id in remaining_ids:
        report = nuclei_preflight.interface(interface_id)
        feasible_anchor_ids = (
            tuple(report.cell_feasible_anchor_segment_ids)
            if report is not None
            else ()
        )
        if directional_front_block:
            eligible, reasons = subdivision_eligibility_for_directional_front_block(
                transformed_tissue,
                interface_id=interface_id,
                cell_feasible_anchor_ids=feasible_anchor_ids,
                maximum_selected_anchor_fraction=front.maximum_selected_anchor_fraction,
                minimum_unselected_anchor_count=front.minimum_unselected_anchor_count,
                minimum_subanchor_pixels=minimum_pixels,
            )
        else:
            eligible, reasons = subdivision_eligibility_for_underfill_replan(
                transformed_tissue,
                interface_id=interface_id,
                cell_feasible_anchor_ids=feasible_anchor_ids,
                remaining_feasible_interface_ids=remaining_ids,
                prior_topology_passed=True,
                area_underfill_replan=True,
                maximum_selected_anchor_fraction=(
                    front.maximum_selected_anchor_fraction
                ),
                minimum_unselected_anchor_count=(
                    front.minimum_unselected_anchor_count
                ),
                minimum_subanchor_pixels=minimum_pixels,
            )
        if not eligible:
            rejected_records.append(
                {"interface_id": interface_id, "reasons": list(reasons)}
            )
            continue
        parent = next(
            item
            for item in transformed_tissue.graph.interfaces
            if item.interface_id == interface_id
        ).anchor_segment_ids[0]
        parent_sha = _array_sha256(transformed_tissue.anchor_masks[parent])
        if directional_front_block:
            transformed_tissue = rebuild_scene_with_directional_sector_partition(
                transformed_tissue,
                interface_id=interface_id,
                minimum_subanchor_pixels=minimum_pixels,
                maximum_selected_anchor_fraction=(
                    front.maximum_selected_anchor_fraction
                ),
            )
        else:
            transformed_tissue = rebuild_scene_with_bisected_single_anchor(
                transformed_tissue,
                interface_id=interface_id,
                minimum_subanchor_pixels=minimum_pixels,
            )
        rebuilt = next(
            item
            for item in transformed_tissue.graph.interfaces
            if item.interface_id == interface_id
        )
        children = []
        for child_id in rebuilt.anchor_segment_ids:
            child = transformed_tissue.anchor_masks[child_id]
            children.append(
                {
                    "anchor_segment_id": child_id,
                    "contact_pixels": int(np.count_nonzero(child)),
                    "geodesic_span_steps": geodesic_span_steps(child),
                    "mask_sha256": _array_sha256(child),
                }
            )
        transformed_records.append(
            {
                "interface_id": interface_id,
                "retired_parent_anchor_id": parent,
                "retired_parent_mask_sha256": parent_sha,
                "reference_area_p95": float(
                    nuclei_preflight.reference_area_p95
                ),
                "equivalent_diameter_pixels": minimum_pixels,
                "new_subanchors": children,
            }
        )
    if not transformed_records:
        raise original_error

    transformed_scene = replace(scene, tissue=transformed_tissue)
    fresh_preflight = build_joint_nuclei_preflight(
        case=joint_case,
        source_tissue=source_tissue,
        schema=schema,
        scene=transformed_scene,
        tissue_bundle=tissue_bundle,
        joint_bundle=joint_bundle,
        allocation=allocation,
    )
    transformed_ids = {
        item["interface_id"] for item in transformed_records
    }
    retained_transformed_ids = transformed_ids.intersection(
        fresh_preflight.feasible_interface_ids
    )
    if not retained_transformed_ids:
        raise original_error
    fresh_binding = build_tissue_portfolio_authority_binding(
        joint_case=joint_case,
        tissue_case=tissue_case,
        source_tissue=source_tissue,
        joint_bundle=joint_bundle,
        tissue_bundle=tissue_bundle,
        allocation=allocation,
        nuclei_preflight=fresh_preflight,
    )
    original_scene_sha = _scene_sha256(scene)
    transformed_scene_sha = _scene_sha256(transformed_scene)
    fresh_binding = {
        **fresh_binding,
        "track_b_transformation_authority": {
            "workflow_version": TRACK_B_WORKFLOW_VERSION,
            "candidate_version": (
                DIRECTIONAL_SECTOR_CANDIDATE_VERSION
                if directional_front_block
                else CANDIDATE_VERSION
            ),
            "original_scene_sha256": original_scene_sha,
            "transformed_scene_sha256": transformed_scene_sha,
            "transformed_interfaces_sha256": canonical_metadata_sha256(
                transformed_records
            ),
        },
    }
    audit = {
        "workflow_version": TRACK_B_WORKFLOW_VERSION,
        "candidate_version": (
            DIRECTIONAL_SECTOR_CANDIDATE_VERSION
            if directional_front_block
            else CANDIDATE_VERSION
        ),
        "track_b_used": True,
        "rounds": 1,
        "recursive_subdivision": False,
        "trigger_kind": (
            "directional_single_anchor_front_block"
            if directional_front_block
            else "topology_passing_area_underfill"
        ),
        "original_failure": str(original_error),
        "trigger_feedback": list(underfills),
        "original_scene_sha256": original_scene_sha,
        "transformed_scene_sha256": transformed_scene_sha,
        "original_authority_binding_sha256": canonical_metadata_sha256(
            original_binding
        ),
        "transformed_authority_binding_sha256": canonical_metadata_sha256(
            fresh_binding
        ),
        "original_preflight_sha256": canonical_metadata_sha256(
            nuclei_preflight
        ),
        "transformed_preflight_sha256": canonical_metadata_sha256(
            fresh_preflight
        ),
        "transformed_interfaces": transformed_records,
        "eligibility_rejections": rejected_records,
        "fresh_preflight_rejected_interface_ids": sorted(
            transformed_ids - retained_transformed_ids
        ),
    }
    try:
        portfolio = compiler.compile_tissue_portfolio(
            tissue_case=tissue_case,
            source_tissue=source_tissue,
            schema=schema,
            scene=transformed_scene,
            tissue_bundle=tissue_bundle,
            joint_bundle=joint_bundle,
            nuclei_preflight=fresh_preflight,
            authority_binding=fresh_binding,
            maximum_candidates=maximum_candidates,
            revoked_candidate_ids=revoked_candidate_ids,
        )
    except TissuePortfolioCompilationError as track_b_error:
        audit["track_b_failure"] = {
            "message": str(track_b_error),
            "structured_failures": list(
                track_b_error.structured_failures
            ),
        }
        raise TrackBPortfolioCompilationError(
            original_error=original_error,
            track_b_error=track_b_error,
            audit=audit,
        ) from track_b_error
    return TrackBPortfolioResult(
        portfolio=portfolio,
        scene=transformed_scene,
        nuclei_preflight=fresh_preflight,
        authority_binding=fresh_binding,
        track_b_used=True,
        audit=audit,
    )


def _is_topology_passing_underfill(feedback: Any) -> bool:
    return bool(
        isinstance(feedback, dict)
        and feedback.get("stage") == "tissue_area_underfill"
        and feedback.get("topology_passed") is True
    )


def _is_directional_single_anchor_front_block(
    error: TissuePortfolioCompilationError,
) -> bool:
    token = (
        "no directed interface satisfies both the tissue and preflight "
        "nuclei contracts"
    )
    if token not in str(error):
        return False
    return any(
        token in str(message)
        for failure in error.structured_failures
        for message in failure.get("errors", ())
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _scene_sha256(scene: JointSceneAnalysis) -> str:
    return canonical_metadata_sha256(
        {
            "metadata": scene.to_metadata(),
            "anchor_masks": {
                key: _array_sha256(value)
                for key, value in sorted(scene.tissue.anchor_masks.items())
            },
        }
    )
