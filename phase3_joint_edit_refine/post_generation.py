"""Fail-closed capability audit for images produced from joint conditions."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .models import JointContractError


class TissueConditionEvaluator(Protocol):
    name: str
    def evaluate(self, *, generated_image: str | Path, target_tissue_mask: str | Path) -> dict[str, float]: ...


class CellConditionEvaluator(Protocol):
    name: str
    def evaluate(self, *, generated_image: str | Path, target_nuclei_mask: str | Path, mechanism_id: str) -> dict[str, float]: ...


class RenderMechanismCritic(Protocol):
    name: str
    def evaluate(self, *, source_image: str | Path, generated_image: str | Path, mechanism_id: str, expectations: tuple[str, ...], vetoes: tuple[str, ...]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CallableTissueConditionEvaluator:
    """Bind a frozen Segmentator inference callable to the audit contract."""

    runner: Callable[..., dict[str, float]]
    name: str = "segmentator_callable"

    def evaluate(self, *, generated_image, target_tissue_mask):
        return _numeric_mapping(
            self.runner(
                generated_image=Path(generated_image),
                target_tissue_mask=Path(target_tissue_mask),
            ),
            required=("fidelity",),
        )


@dataclass(frozen=True)
class CallableCellConditionEvaluator:
    """Bind a frozen CellViT inference callable to the audit contract."""

    runner: Callable[..., dict[str, float]]
    name: str = "cellvit_callable"

    def evaluate(self, *, generated_image, target_nuclei_mask, mechanism_id):
        return _numeric_mapping(
            self.runner(
                generated_image=Path(generated_image),
                target_nuclei_mask=Path(target_nuclei_mask),
                mechanism_id=mechanism_id,
            ),
            required=(
                "count_consistency",
                "type_consistency",
                "spatial_consistency",
                "interface_distance_consistency",
            ),
        )


@dataclass(frozen=True)
class CallableRenderMechanismCritic:
    """Bind an independent visual critic without coupling it to the Planner."""

    runner: Callable[..., dict[str, Any]]
    name: str = "visual_mechanism_critic_callable"

    def evaluate(
        self,
        *,
        source_image,
        generated_image,
        mechanism_id,
        expectations,
        vetoes,
    ):
        result = self.runner(
            source_image=Path(source_image),
            generated_image=Path(generated_image),
            mechanism_id=mechanism_id,
            expectations=expectations,
            vetoes=vetoes,
        )
        if not isinstance(result, dict) or not isinstance(
            result.get("approved"), bool
        ):
            raise JointContractError(
                "visual critic must return an approved boolean"
            )
        veto_reasons = result.get("veto_reasons", [])
        if not isinstance(veto_reasons, list) or not all(
            isinstance(item, str) for item in veto_reasons
        ):
            raise JointContractError(
                "visual critic veto_reasons must be a list of strings"
            )
        return result


@dataclass(frozen=True)
class PostGenerationThresholds:
    tissue_fidelity_min: float = 0.90
    cell_count_consistency_min: float = 0.80
    cell_type_consistency_min: float = 0.80
    spatial_consistency_min: float = 0.80
    interface_distance_consistency_min: float = 0.80
    exterior_mean_absolute_drift_max: float = 0.03


@dataclass(frozen=True)
class PostGenerationAuditResult:
    passed: bool
    capability_status: str
    checks: dict[str, Any]
    reasons: tuple[str, ...]


def audit_generated_joint_image(
    *,
    source_image: str | Path,
    generated_image: str | Path,
    target_tissue_mask: str | Path,
    target_nuclei_mask: str | Path,
    generation_support_mask: str | Path,
    mechanism_id: str,
    expectations: tuple[str, ...],
    vetoes: tuple[str, ...],
    tissue_evaluator: TissueConditionEvaluator | None,
    cell_evaluator: CellConditionEvaluator | None,
    visual_critic: RenderMechanismCritic | None,
    thresholds: PostGenerationThresholds | None = None,
) -> PostGenerationAuditResult:
    """Approve render capability only when all three independent evaluators exist."""

    thresholds = thresholds or PostGenerationThresholds()
    missing = [
        name
        for name, value in (
            ("Segmentator-compatible tissue evaluator", tissue_evaluator),
            ("CellViT-compatible cell evaluator", cell_evaluator),
            ("independent visual mechanism critic", visual_critic),
        )
        if value is None
    ]
    if missing:
        return PostGenerationAuditResult(
            passed=False,
            capability_status="render_unsupported",
            checks={},
            reasons=("missing required post-generation evaluator: " + ", ".join(missing),),
        )
    tissue = tissue_evaluator.evaluate(generated_image=generated_image, target_tissue_mask=target_tissue_mask)
    cells = cell_evaluator.evaluate(generated_image=generated_image, target_nuclei_mask=target_nuclei_mask, mechanism_id=mechanism_id)
    visual = visual_critic.evaluate(source_image=source_image, generated_image=generated_image, mechanism_id=mechanism_id, expectations=expectations, vetoes=vetoes)
    exterior_drift = _exterior_drift(source_image, generated_image, generation_support_mask)
    checks = {
        "tissue": tissue,
        "cells": cells,
        "visual": visual,
        "exterior_mean_absolute_drift": exterior_drift,
        "evaluators": {
            "tissue": tissue_evaluator.name,
            "cells": cell_evaluator.name,
            "visual": visual_critic.name,
        },
    }
    reasons = []
    if float(tissue.get("fidelity", -1)) < thresholds.tissue_fidelity_min:
        reasons.append("target tissue condition fidelity failed")
    for key, minimum in (
        ("count_consistency", thresholds.cell_count_consistency_min),
        ("type_consistency", thresholds.cell_type_consistency_min),
        ("spatial_consistency", thresholds.spatial_consistency_min),
        ("interface_distance_consistency", thresholds.interface_distance_consistency_min),
    ):
        if float(cells.get(key, -1)) < minimum:
            reasons.append(f"cell {key} failed")
    if visual.get("approved") is not True or visual.get("veto_reasons"):
        reasons.append("visual mechanism critic did not approve the render")
    if exterior_drift > thresholds.exterior_mean_absolute_drift_max:
        reasons.append("generation drift outside G exceeded threshold")
    return PostGenerationAuditResult(
        passed=not reasons,
        capability_status=("render_supported" if not reasons else "render_unsupported"),
        checks=checks,
        reasons=tuple(reasons),
    )


def audit_joint_generation_handoff(
    *,
    manifest_path: str | Path,
    generated_image: str | Path,
    output_path: str | Path,
    tissue_evaluator: TissueConditionEvaluator | None,
    cell_evaluator: CellConditionEvaluator | None,
    visual_critic: RenderMechanismCritic | None,
    thresholds: PostGenerationThresholds | None = None,
) -> PostGenerationAuditResult:
    """Verify the v2 handoff and atomically audit one generated H&E result."""

    # Lazy import avoids coupling condition construction to the generator.
    from .generator_adapter import build_frozen_generator_inputs

    _, _, manifest = build_frozen_generator_inputs(
        manifest_path,
        output_dir=Path(output_path).parent,
    )
    paths = manifest["paths"]
    source = manifest["source_assets"]
    result = audit_generated_joint_image(
        source_image=source["image"],
        generated_image=generated_image,
        target_tissue_mask=paths["target_tissue_mask"],
        target_nuclei_mask=paths["target_nuclei_mask"],
        generation_support_mask=paths["generation_support"],
        mechanism_id=manifest["mechanism_id"],
        expectations=tuple(manifest.get("render_expectations", [])),
        vetoes=tuple(manifest.get("render_vetoes", [])),
        tissue_evaluator=tissue_evaluator,
        cell_evaluator=cell_evaluator,
        visual_critic=visual_critic,
        thresholds=thresholds,
    )
    payload = {
        "schema_version": "joint-post-generation-audit-v1",
        "case_id": manifest["case_id"],
        "candidate_id": manifest["candidate_id"],
        "executable_contract_id": manifest["executable_contract_id"],
        "generated_image": str(Path(generated_image)),
        **asdict(result),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _exterior_drift(source_path, generated_path, support_path) -> float:
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=float) / 255.0
    generated = np.asarray(Image.open(generated_path).convert("RGB"), dtype=float) / 255.0
    support = np.asarray(Image.open(support_path).convert("L")) > 0
    if source.shape != generated.shape or source.shape[:2] != support.shape:
        raise JointContractError("post-generation audit inputs are not aligned")
    exterior = ~support
    if not np.any(exterior):
        return 0.0
    return float(np.mean(np.abs(source[exterior] - generated[exterior])))


def _numeric_mapping(value: Any, *, required: tuple[str, ...]) -> dict[str, float]:
    if not isinstance(value, dict):
        raise JointContractError("condition evaluator must return a mapping")
    result = {}
    for key in required:
        current = value.get(key)
        if not isinstance(current, (int, float)) or not np.isfinite(current):
            raise JointContractError(
                f"condition evaluator returned invalid metric: {key}"
            )
        result[key] = float(current)
    for key, current in value.items():
        if key not in result and isinstance(current, (int, float)):
            result[key] = float(current)
    return result
