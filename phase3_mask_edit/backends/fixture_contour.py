"""Fixture-driven LLM contour proposal backend.

This module is the Milestone C adapter: it reads a saved contour proposal JSON
as if it came from a multimodal LLM, then runs the normal Phase 3 safety path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from phase3_mask_edit.backends.llm_contour import (
    CONTOUR_PROPOSAL_BACKEND,
    ContourProposal,
    ContourProposalValidationError,
    DEFAULT_PROJECTION_MODE,
    PROJECTION_MODE_HARD_V1,
    PROJECTION_MODE_COMPARE_V1_V2,
    PROJECTION_MODE_ORGANIC_V2,
    execute_contour_proposal_write,
    load_contour_proposal_json,
    rasterize_contour_proposal,
    validate_contour_proposal,
)
from phase3_mask_edit.backends.llm_preview import (
    add_coordinate_grid_overlay,
    id_mask_to_llm_preview_rgb,
)
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    save_change_region,
    save_id_mask,
    save_metadata,
    save_rgb_mask,
)
from phase3_mask_edit.core.validation import ValidationResult, validate_edit_result
from phase3_mask_edit.generic.tumor_burden import PrimitiveEditResult


STATUS_VALIDATED = "validated"
STATUS_VALIDATION_FAILED = "validation_failed"
STATUS_PROPOSAL_REJECTED = "proposal_rejected"
STATUS_EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True)
class FixtureContourExecutionResult:
    """Result of executing one fixture contour proposal."""

    status: str
    source_mask: np.ndarray
    proposal: ContourProposal | None
    edit_result: PrimitiveEditResult | None
    validation: ValidationResult | None
    artifact_paths: dict[str, str]
    error: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "status": self.status,
            "backend": CONTOUR_PROPOSAL_BACKEND,
            "error": self.error,
            "artifact_paths": dict(self.artifact_paths),
        }
        if self.proposal is not None:
            metadata["proposal"] = {
                "primitive": self.proposal.primitive,
                "reference_profile": self.proposal.reference_profile,
                "target_label": self.proposal.target_label,
                "width": self.proposal.width,
                "height": self.proposal.height,
                "regions": [
                    {
                        "region_id": region.region_id,
                        "source_labels": list(region.source_labels),
                        "points": [list(point) for point in region.points],
                        "confidence": region.confidence,
                    }
                    for region in self.proposal.regions
                ],
            }
        if self.edit_result is not None:
            metadata["edit_result"] = {
                "selected_pixels": self.edit_result.selected_pixels,
                "changed_area_fraction": self.edit_result.changed_area_fraction,
                "warnings": list(self.edit_result.warnings),
                "ops_log": self.edit_result.ops_log,
            }
        if self.validation is not None:
            metadata["validation"] = _jsonable_dataclass(self.validation)
        return metadata


def execute_fixture_contour_backend(
    *,
    old_mask: np.ndarray,
    fixture_path: str | Path,
    schema: MaskProfileSchema,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    allowed_source_labels: Sequence[str] | None = None,
    max_regions: int = 8,
    max_points_per_region: int = 64,
    projection_mode: str = DEFAULT_PROJECTION_MODE,
    organic_seed: int = 0,
) -> FixtureContourExecutionResult:
    """Execute a saved contour proposal through the Phase 3 proposal path."""

    source_mask = np.asarray(old_mask)
    proposal: ContourProposal | None = None
    edit_result: PrimitiveEditResult | None = None
    validation: ValidationResult | None = None
    artifact_paths: dict[str, str] = {}
    error: str | None = None

    try:
        payload = load_contour_proposal_json(fixture_path)
        proposal = validate_contour_proposal(
            payload,
            schema=schema,
            mask_shape=tuple(source_mask.shape),
            primitive=intent.primitive,
            reference_profile=intent.reference_profile or schema.reference_profile,
            target_label=intent.target_label,
            allowed_source_labels=(
                tuple(allowed_source_labels)
                if allowed_source_labels is not None
                else tuple(intent.source_labels) or None
            ),
            max_regions=max_regions,
            max_points_per_region=max_points_per_region,
        )
    except ContourProposalValidationError as exc:
        error = str(exc)
        result = FixtureContourExecutionResult(
            status=STATUS_PROPOSAL_REJECTED,
            source_mask=np.array(source_mask, copy=True),
            proposal=None,
            edit_result=None,
            validation=None,
            artifact_paths={},
            error=error,
        )
        if output_dir is not None:
            artifact_paths = save_fixture_contour_artifacts(
                result,
                output_dir,
                fixture_path=fixture_path,
            )
            result = _replace_artifacts(result, artifact_paths)
        return result

    try:
        primary_projection_mode = (
            PROJECTION_MODE_ORGANIC_V2
            if projection_mode == PROJECTION_MODE_COMPARE_V1_V2
            else projection_mode
        )
        edit_result = execute_contour_proposal_write(
            source_mask,
            proposal,
            schema=schema,
            primitive_config=primitive_config,
            preserve_labels=intent.preserve_labels,
            forbidden_labels=intent.forbidden_labels,
            projection_mode=primary_projection_mode,
            organic_seed=organic_seed,
        )
        edit_result.ops_log["requested_projection_mode"] = projection_mode
        edit_result.ops_log["primary_projection_mode"] = primary_projection_mode
        validation = validate_edit_result(
            src_mask=source_mask,
            target_mask=edit_result.target_mask,
            change_region=edit_result.change_region,
            schema=schema,
            primitive_config=primitive_config,
            changed_area_fraction=edit_result.changed_area_fraction,
        )
        status = STATUS_VALIDATED if validation.passed else STATUS_VALIDATION_FAILED
    except Exception as exc:  # pragma: no cover - defensive boundary for CLI use.
        error = str(exc)
        status = STATUS_EXECUTION_ERROR

    result = FixtureContourExecutionResult(
        status=status,
        source_mask=np.array(source_mask, copy=True),
        proposal=proposal,
        edit_result=edit_result,
        validation=validation,
        artifact_paths={},
        error=error,
    )
    if output_dir is not None:
        artifact_paths = save_fixture_contour_artifacts(
            result,
            output_dir,
            fixture_path=fixture_path,
        )
        if (
            projection_mode == PROJECTION_MODE_COMPARE_V1_V2
            and result.proposal is not None
            and result.error is None
        ):
            artifact_paths.update(
                _save_projection_comparison_artifacts(
                    source_mask,
                    result.proposal,
                    schema=schema,
                    primitive_config=primitive_config,
                    preserve_labels=intent.preserve_labels,
                    forbidden_labels=intent.forbidden_labels,
                    output_dir=Path(output_dir),
                    organic_seed=organic_seed,
                    primary_projection_mode=PROJECTION_MODE_ORGANIC_V2,
                )
            )
        result = _replace_artifacts(result, artifact_paths)
    return result


def save_fixture_contour_artifacts(
    result: FixtureContourExecutionResult,
    output_dir: str | Path,
    *,
    fixture_path: str | Path | None = None,
) -> dict[str, str]:
    """Save the Milestone C artifact bundle for a fixture contour run."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["source_mask"] = save_id_mask(result.source_mask, out / "source_mask.png")
    paths["source_mask_rgb"] = save_rgb_mask(
        result.source_mask,
        out / "source_mask_rgb.png",
    )

    preview_rgb = id_mask_to_llm_preview_rgb(result.source_mask)
    paths["source_mask_llm_rgb"] = _save_rgb_array(
        preview_rgb,
        out / "source_mask_llm_rgb.png",
    )
    paths["source_mask_llm_rgb_grid"] = _save_rgb_array(
        add_coordinate_grid_overlay(preview_rgb),
        out / "source_mask_llm_rgb_grid.png",
    )

    if fixture_path is not None:
        payload = load_contour_proposal_json(fixture_path)
        paths["fixture_response"] = save_metadata(payload, out / "fixture_response.json")

    if result.proposal is not None:
        paths["validated_proposal"] = save_metadata(
            result.proposal.raw_payload,
            out / "validated_proposal.json",
        )
        paths["rasterized_region"] = save_change_region(
            rasterize_contour_proposal(result.proposal),
            out / "rasterized_region.png",
        )

    if result.edit_result is not None:
        paths["projected_region"] = save_change_region(
            result.edit_result.change_region,
            out / "projected_region.png",
        )
        paths["change_region"] = save_change_region(
            result.edit_result.change_region,
            out / "change_region.png",
        )
        paths["target_mask"] = save_id_mask(
            result.edit_result.target_mask,
            out / "target_mask.png",
        )
        paths["target_mask_rgb"] = save_rgb_mask(
            result.edit_result.target_mask,
            out / "target_mask_rgb.png",
        )

    if result.validation is not None:
        paths["validation"] = save_metadata(
            _jsonable_dataclass(result.validation),
            out / "validation.json",
        )

    metadata_without_paths = dict(result.to_metadata())
    metadata_without_paths["artifact_paths"] = {}
    paths["summary"] = save_metadata(metadata_without_paths, out / "execution_summary.json")
    return {key: str(path) for key, path in paths.items()}


def _replace_artifacts(
    result: FixtureContourExecutionResult,
    artifact_paths: dict[str, str],
) -> FixtureContourExecutionResult:
    return FixtureContourExecutionResult(
        status=result.status,
        source_mask=result.source_mask,
        proposal=result.proposal,
        edit_result=result.edit_result,
        validation=result.validation,
        artifact_paths=artifact_paths,
        error=result.error,
    )


def _save_rgb_array(rgb: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(p)
    return p


def _save_projection_comparison_artifacts(
    source_mask: np.ndarray,
    proposal: ContourProposal,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    preserve_labels: Sequence[str],
    forbidden_labels: Sequence[str],
    output_dir: Path,
    organic_seed: int,
    primary_projection_mode: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    comparison_results: dict[str, Any] = {
        "projection_mode": PROJECTION_MODE_COMPARE_V1_V2,
        "primary_projection_mode": primary_projection_mode,
        "debug_projection_modes": [PROJECTION_MODE_HARD_V1],
        "repair_loop_consumes": "primary_result_only",
        "debug_results_are_metadata_only": True,
        "results": {},
    }
    for mode in (PROJECTION_MODE_HARD_V1, PROJECTION_MODE_ORGANIC_V2):
        edit = execute_contour_proposal_write(
            source_mask,
            proposal,
            schema=schema,
            primitive_config=primitive_config,
            preserve_labels=preserve_labels,
            forbidden_labels=forbidden_labels,
            projection_mode=mode,
            organic_seed=organic_seed,
        )
        validation = validate_edit_result(
            src_mask=source_mask,
            target_mask=edit.target_mask,
            change_region=edit.change_region,
            schema=schema,
            primitive_config=primitive_config,
            changed_area_fraction=edit.changed_area_fraction,
        )
        branch_role = "primary" if mode == primary_projection_mode else "debug"
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        branch_summary = {
            "projection_mode": mode,
            "branch_role": branch_role,
            "repair_loop_eligible": branch_role == "primary",
            "edit_result": {
                "selected_pixels": edit.selected_pixels,
                "changed_area_fraction": edit.changed_area_fraction,
                "warnings": list(edit.warnings),
                "ops_log": edit.ops_log,
            },
            "validation": _jsonable_dataclass(validation),
        }
        comparison_results["results"][mode] = {
            "branch_role": branch_role,
            "repair_loop_eligible": branch_role == "primary",
            "selected_pixels": edit.selected_pixels,
            "changed_area_fraction": edit.changed_area_fraction,
            "validation_passed": validation.passed,
            "warnings": list(edit.warnings),
            "projection_retained_fraction": edit.ops_log.get(
                "projection_retained_fraction"
            ),
            "area_shortfall": edit.ops_log.get("area_shortfall"),
        }
        paths[f"{mode}_change_region"] = str(
            save_change_region(edit.change_region, mode_dir / "change_region.png")
        )
        paths[f"{mode}_target_mask"] = str(
            save_id_mask(edit.target_mask, mode_dir / "target_mask.png")
        )
        paths[f"{mode}_target_mask_rgb"] = str(
            save_rgb_mask(edit.target_mask, mode_dir / "target_mask_rgb.png")
        )
        paths[f"{mode}_summary"] = str(
            save_metadata(
                branch_summary,
                mode_dir / "summary.json",
            )
        )
    paths["projection_comparison_summary"] = str(
        save_metadata(comparison_results, output_dir / "projection_comparison_summary.json")
    )
    return paths


def _jsonable_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value
