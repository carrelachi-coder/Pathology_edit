"""Independent joint critic interface and non-visual research ranker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import (
    JointCandidate,
    JointCaseContext,
    JointCriticRanking,
    JointCriticResult,
    JointGateReport,
)
from .planner_inputs import MaskPlannerArtifactRegistry
from .skills.repository import JointSkillBundle


class JointCritic(Protocol):
    name: str
    supports_pathology_vision: bool

    def review(
        self,
        *,
        case: JointCaseContext,
        bundle: JointSkillBundle,
        candidates: Sequence[JointCandidate],
        gate_reports: Sequence[JointGateReport],
        image_paths: Sequence[str | Path],
        artifact_registry: MaskPlannerArtifactRegistry | None = None,
    ) -> JointCriticResult: ...


@dataclass(frozen=True)
class DeterministicJointResearchCritic:
    """Rank gated candidates but require a later independent visual decision."""

    name: str = "deterministic_joint_research_critic"
    supports_pathology_vision: bool = False

    def review(
        self,
        *,
        case,
        bundle,
        candidates,
        gate_reports,
        image_paths,
        artifact_registry=None,
    ):
        del image_paths, artifact_registry
        passed = {item.candidate_id for item in gate_reports if item.passed}
        rankings = []
        for candidate in candidates:
            if candidate.candidate_id not in passed:
                continue
            completion = float(candidate.tool_trace.get("placement_completion", 0.0))
            if bundle.primitive.budget_mode == "count_extent":
                budget = case.cell_count_extent_budget
                desired = budget.target_delta_count if budget else 0
                actual = int(candidate.tool_trace.get("placed_count", 0))
                quota_score = max(
                    0.0,
                    1.0 - abs(actual - desired) / max(1, desired),
                )
                score = quota_score * completion
            else:
                if case.joint_area_budget is None:
                    continue
                target = case.joint_area_budget.target_fraction
                area_error = abs(candidate.ledger.joint_fraction - target)
                score = max(
                    0.0, 1.0 - area_error / max(target, 1e-6)
                ) * completion
            rankings.append(
                JointCriticRanking(
                    candidate_id=candidate.candidate_id,
                    score=score,
                    confidence=0.35,
                    supporting_rule_ids=bundle.active_rule_ids,
                    veto_reasons=(),
                )
            )
        rankings.sort(key=lambda item: (-item.score, item.candidate_id))
        return JointCriticResult(
            rankings=tuple(rankings),
            abstain=True,
            summary=(
                "deterministic metrics ranked condition candidates, but an independent "
                "multimodal pathology critic has not approved them"
            ),
            usage={"provider": self.name, "input_tokens": 0, "output_tokens": 0},
        )
