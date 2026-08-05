"""Complete per-case audit artifact writer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from phase3_mask_edit_refine.models import (
    CandidateMask,
    CaseContext,
    CriticResult,
    EditPlan,
    GateReport,
    SceneGraph,
)
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle
from phase3_mask_edit_refine.visualization import id_mask_to_rgb


class AuditWriter:
    def __init__(self, root: str | Path, *, case_id: str) -> None:
        self.case_dir = Path(root) / _safe_case_id(case_id)
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.paths: dict[str, str] = {}

    def write_inputs(
        self,
        *,
        case: CaseContext,
        source_mask: np.ndarray,
        scene_graph: SceneGraph,
        bundle: ActiveKnowledgeBundle,
    ) -> None:
        self.paths["case_context"] = self._json("case_context.json", case.to_metadata())
        self.paths["scene_graph"] = self._json("scene_graph.json", scene_graph.to_metadata())
        self.paths["active_skills"] = self._json("active_skills.json", bundle.to_metadata())
        self.paths["source_mask_npy"] = self._npy("source_mask.npy", source_mask)
        self.paths["source_mask_png"] = self._png("source_mask.png", id_mask_to_rgb(source_mask))

    def write_plan(self, plan: EditPlan, *, usage: Mapping[str, Any]) -> None:
        self.paths["planner_plan"] = self._json("planner_plan.json", plan.to_metadata())
        self.paths["planner_usage"] = self._json("planner_usage.json", dict(usage))

    def write_candidates(self, candidates: Sequence[CandidateMask]) -> None:
        root = self.case_dir / "candidates"
        root.mkdir(exist_ok=True)
        for candidate in candidates:
            target = root / candidate.candidate_id.replace(":", "_")
            target.mkdir(exist_ok=True)
            np.save(target / "target_mask.npy", candidate.target_mask, allow_pickle=False)
            np.save(target / "change_region.npy", candidate.change_region, allow_pickle=False)
            Image.fromarray(id_mask_to_rgb(candidate.target_mask)).save(target / "target_mask.png")
            Image.fromarray(candidate.change_region.astype(np.uint8) * 255).save(
                target / "change_region.png"
            )
            (target / "tool_trace.json").write_text(
                json.dumps(candidate.to_metadata(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        self.paths["candidates"] = str(root)

    def write_gate_reports(self, reports: Sequence[GateReport]) -> None:
        self.paths["gate_report"] = self._json(
            "gate_report.json", [report.to_metadata() for report in reports]
        )

    def write_critic(self, critic: CriticResult) -> None:
        self.paths["critic_ranking"] = self._json("critic_ranking.json", asdict(critic))

    def write_selection(
        self,
        *,
        status: str,
        selected_candidate_id: str | None,
        abstain_reasons: Sequence[str],
        target_mask: np.ndarray | None,
        source_mask: np.ndarray,
        usage: Mapping[str, Any],
    ) -> None:
        self.paths["selection"] = self._json(
            "selection.json",
            {
                "status": status,
                "selected_candidate_id": selected_candidate_id,
                "abstain_reasons": list(abstain_reasons),
                "usage": dict(usage),
            },
        )
        if target_mask is not None:
            change = np.asarray(source_mask) != np.asarray(target_mask)
            self.paths["final_mask_npy"] = self._npy("final_mask.npy", target_mask)
            self.paths["final_mask_png"] = self._png(
                "final_mask.png", id_mask_to_rgb(target_mask)
            )
            self.paths["final_diff_png"] = self._png(
                "final_diff.png", change.astype(np.uint8) * 255
            )

    def _json(self, name: str, payload: Any) -> str:
        path = self.case_dir / name
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return str(path)

    def _npy(self, name: str, array: np.ndarray) -> str:
        path = self.case_dir / name
        np.save(path, np.asarray(array), allow_pickle=False)
        return str(path)

    def _png(self, name: str, array: np.ndarray) -> str:
        path = self.case_dir / name
        Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)
        return str(path)


def _safe_case_id(case_id: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in case_id)
