"""D1 LLM contour agent orchestration with deterministic providers."""

from __future__ import annotations

import base64
import copy
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image

from phase3_mask_edit.backends.fixture_contour import (
    STATUS_EXECUTION_ERROR,
    STATUS_PROPOSAL_REJECTED,
    STATUS_VALIDATED,
    STATUS_VALIDATION_FAILED,
)
from phase3_mask_edit.backends.llm_contour import (
    CONTOUR_PROPOSAL_BACKEND,
    ContourProposal,
    ContourProposalValidationError,
    DEFAULT_PROJECTION_MODE,
    execute_contour_proposal_write,
    load_contour_proposal_json,
    rasterize_contour_proposal,
    validate_contour_proposal,
)
from phase3_mask_edit.backends.llm_preview import (
    add_coordinate_grid_overlay,
    id_mask_to_llm_preview_rgb,
)
from phase3_mask_edit.backends.llm_prompt import (
    ContourProposalRequest,
    build_contour_prompt,
    build_mask_context,
    build_repair_feedback,
    save_prompt_text,
)
from phase3_mask_edit.backends.visual_qa import save_visual_qa_bundle
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


STATUS_PROVIDER_ERROR = "provider_error"
STATUS_PROPOSAL_FAILED = "proposal_failed"


class ContourProposalProvider(Protocol):
    """Provider protocol for contour proposal payloads."""

    name: str

    def propose(self, request: ContourProposalRequest) -> Mapping[str, Any]:
        """Return a raw contour proposal mapping for one attempt."""


class ContourProviderError(RuntimeError):
    """Raised when a contour provider cannot return a usable payload."""


@dataclass(frozen=True)
class FixtureContourProvider:
    """Provider that always returns one saved contour proposal JSON."""

    fixture_path: str | Path
    name: str = "fixture"

    def propose(self, request: ContourProposalRequest) -> Mapping[str, Any]:
        return load_contour_proposal_json(self.fixture_path)


@dataclass(frozen=True)
class OpenAICompatibleTextContourProvider:
    """Text-only provider for OpenAI-compatible chat-completions APIs."""

    model: str
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    name: str = "openai_compatible_text"

    def propose(self, request: ContourProposalRequest) -> Mapping[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ContourProviderError(
                f"Missing API key environment variable: {self.api_key_env}"
            )

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a pathology mask contour proposal agent. "
                        "Return only strict JSON that matches the requested schema."
                    ),
                },
                {"role": "user", "content": request.prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        response_payload = _post_chat_completion(
            payload,
            api_base_url=self.api_base_url,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
        )
        content = _response_content(response_payload)
        return _parse_json_object_content(content)


@dataclass(frozen=True)
class OpenAICompatibleMultimodalContourProvider:
    """Grid-preview provider for OpenAI-compatible vision chat APIs."""

    model: str
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    image_detail: str = "high"
    name: str = "openai_compatible_multimodal"

    def propose(self, request: ContourProposalRequest) -> Mapping[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ContourProviderError(
                f"Missing API key environment variable: {self.api_key_env}"
            )
        if not request.image_paths:
            raise ContourProviderError("Multimodal contour provider requires one image path.")

        grid_preview_url = _image_path_to_data_url(request.image_paths[0])
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a pathology mask contour proposal agent. "
                        "Use the provided grid-overlay mask preview for spatial coordinates. "
                        "Return only strict JSON that matches the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                request.prompt
                                + "\n\nThe attached image is the grid-overlay tissue "
                                "mask preview. Use it as the primary spatial reference. "
                                "Return coordinates in the original mask coordinate system."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": grid_preview_url,
                                "detail": self.image_detail,
                            },
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }
        response_payload = _post_chat_completion(
            payload,
            api_base_url=self.api_base_url,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
        )
        content = _response_content(response_payload)
        return _parse_json_object_content(content)


@dataclass(frozen=True)
class FakeSequenceContourProvider:
    """Provider that returns preset proposal payloads in attempt order."""

    payloads: tuple[Mapping[str, Any], ...]
    name: str = "fake_sequence"

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path]) -> "FakeSequenceContourProvider":
        return cls(tuple(load_contour_proposal_json(path) for path in paths))

    def propose(self, request: ContourProposalRequest) -> Mapping[str, Any]:
        if not self.payloads:
            raise RuntimeError("fake sequence provider has no payloads.")
        index = min(max(request.attempt_index - 1, 0), len(self.payloads) - 1)
        return self.payloads[index]


@dataclass(frozen=True)
class LLMContourAttempt:
    """One contour agent attempt."""

    attempt_index: int
    status: str
    raw_response: dict[str, Any] | None = None
    proposal: ContourProposal | None = None
    edit_result: PrimitiveEditResult | None = None
    validation: ValidationResult | None = None
    repair_feedback: dict[str, Any] | None = None
    artifact_paths: dict[str, str] | None = None
    error: str | None = None


@dataclass(frozen=True)
class LLMContourAgentResult:
    """Result of an LLM contour agent run."""

    status: str
    source_mask: np.ndarray
    attempts: tuple[LLMContourAttempt, ...]
    final_attempt: LLMContourAttempt | None
    context: dict[str, Any]
    artifact_paths: dict[str, str]
    projection_mode: str = DEFAULT_PROJECTION_MODE
    error: str | None = None

    @property
    def edit_result(self) -> PrimitiveEditResult | None:
        return self.final_attempt.edit_result if self.final_attempt else None

    @property
    def validation(self) -> ValidationResult | None:
        return self.final_attempt.validation if self.final_attempt else None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": CONTOUR_PROPOSAL_BACKEND,
            "projection_mode": self.projection_mode,
            "error": self.error,
            "context": self.context,
            "artifact_paths": dict(self.artifact_paths),
            "attempts": [_attempt_metadata(attempt) for attempt in self.attempts],
        }


def execute_llm_contour_agent(
    *,
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
    provider: ContourProposalProvider,
    output_dir: str | Path | None = None,
    allowed_source_labels: Sequence[str] | None = None,
    max_attempts: int = 4,
    max_regions: int = 8,
    max_points_per_region: int = 64,
    grid_spacing_px: int = 64,
    projection_mode: str = DEFAULT_PROJECTION_MODE,
    organic_seed: int = 0,
) -> LLMContourAgentResult:
    """Run a provider-backed contour proposal repair loop."""

    source_mask = np.asarray(old_mask)
    resolved_sources = tuple(
        allowed_source_labels
        if allowed_source_labels is not None
        else tuple(intent.source_labels)
    )
    if not resolved_sources:
        operation = primitive_config.get("mask_operation", {})
        resolved_sources = tuple(_default_source_labels(primitive_config, operation))
    target_label = intent.target_label or _string_or_none(
        primitive_config.get("mask_operation", {}).get("target")
    )
    if not target_label and primitive_config.get("name") == "tumor_burden_increase":
        target_label = "Tumor"
    if not target_label:
        target_label = _backfill_prompt_target_label(
            primitive_config,
            schema,
            source_mask,
        )
    if not target_label:
        raise ValueError("LLM contour agent requires a target label.")

    out = Path(output_dir) if output_dir is not None else None
    artifact_paths: dict[str, str] = {}
    preview_paths: tuple[str, ...] = ()
    if out is not None:
        artifact_paths, preview_paths = _save_run_inputs(
            source_mask,
            out,
            grid_spacing_px=grid_spacing_px,
        )

    context = build_mask_context(
        source_mask,
        schema=schema,
        intent=intent,
        primitive_config=primitive_config,
        allowed_source_labels=resolved_sources,
        target_label=target_label,
        grid_spacing_px=grid_spacing_px,
        max_regions=max_regions,
        max_points_per_region=max_points_per_region,
    )
    if out is not None:
        artifact_paths["mask_context"] = str(
            save_metadata(context, out / "mask_context.json")
        )

    attempts: list[LLMContourAttempt] = []
    repair_feedback: dict[str, Any] | None = None
    next_repair_profile: str | None = None
    final_attempt: LLMContourAttempt | None = None
    context_mode = "full"
    compact_context_enabled = False

    for attempt_index in range(1, max_attempts + 1):
        prompt = build_contour_prompt(
            context=context,
            repair_feedback=repair_feedback,
        )
        request = ContourProposalRequest(
            prompt=prompt,
            context=dict(context),
            attempt_index=attempt_index,
            image_paths=preview_paths,
            repair_feedback=repair_feedback,
            provider_metadata={
                **_provider_request_metadata(provider, preview_paths),
                "context_mode": context_mode,
                "compact_context_enabled": compact_context_enabled,
            },
        )
        attempt = _execute_one_attempt(
            request=request,
            provider=provider,
            source_mask=source_mask,
            schema=schema,
            intent=intent,
            primitive_config=primitive_config,
            allowed_source_labels=resolved_sources,
            target_label=target_label,
            max_regions=max_regions,
            max_points_per_region=max_points_per_region,
            projection_mode=projection_mode,
            organic_seed=organic_seed,
        )
        if out is not None:
            attempt_paths = _save_attempt_artifacts(
                attempt,
                out / f"attempt_{attempt_index:03d}",
                request=request,
                source_mask=source_mask,
                schema=schema,
                intent=intent,
                primitive_config=primitive_config,
                projection_mode=projection_mode,
            )
            attempt = _replace_attempt_paths(attempt, attempt_paths)
        attempts.append(attempt)

        if attempt.status == STATUS_VALIDATED:
            final_attempt = attempt
            break
        repair_feedback = attempt.repair_feedback
        if repair_feedback is not None:
            next_repair_profile = str(repair_feedback.get("repair_profile") or "") or None
            repair_feedback = _with_task_specific_repair_feedback(
                repair_feedback,
                primitive_name=str(primitive_config.get("name", "")),
                repair_profile=next_repair_profile,
                target_label=target_label,
                allowed_source_labels=resolved_sources,
            )
        if (
            attempt.status == STATUS_PROVIDER_ERROR
            and _is_message_length_exceeded(attempt.error)
            and not compact_context_enabled
        ):
            context = _compact_mask_context_for_retry(context)
            context_mode = "compact"
            compact_context_enabled = True
            repair_feedback = _with_compact_context_feedback(repair_feedback)
            if repair_feedback is not None:
                attempt = LLMContourAttempt(
                    attempt_index=attempt.attempt_index,
                    status=attempt.status,
                    raw_response=attempt.raw_response,
                    proposal=attempt.proposal,
                    edit_result=attempt.edit_result,
                    validation=attempt.validation,
                    repair_feedback=repair_feedback,
                    artifact_paths=attempt.artifact_paths,
                    error=attempt.error,
                )
                attempts[-1] = attempt
            if out is not None:
                artifact_paths["mask_context_compact"] = str(
                    save_metadata(context, out / "mask_context_compact.json")
                )

    if final_attempt is None:
        final_attempt = attempts[-1] if attempts else None
    status = (
        STATUS_VALIDATED
        if final_attempt is not None and final_attempt.status == STATUS_VALIDATED
        else STATUS_PROPOSAL_FAILED
    )
    error = None if status == STATUS_VALIDATED else (
        final_attempt.error if final_attempt is not None else "no attempts executed"
    )

    result = LLMContourAgentResult(
        status=status,
        source_mask=np.array(source_mask, copy=True),
        attempts=tuple(attempts),
        final_attempt=final_attempt,
        context=context,
        artifact_paths={},
        projection_mode=projection_mode,
        error=error,
    )
    if out is not None:
        if status == STATUS_VALIDATED and final_attempt is not None and final_attempt.edit_result is not None:
            artifact_paths["final_target_mask"] = str(
                save_id_mask(final_attempt.edit_result.target_mask, out / "final_target_mask.png")
            )
            artifact_paths["final_target_mask_rgb"] = str(
                save_rgb_mask(final_attempt.edit_result.target_mask, out / "final_target_mask_rgb.png")
            )
            artifact_paths["final_change_region"] = str(
                save_change_region(final_attempt.edit_result.change_region, out / "final_change_region.png")
            )
        result = _replace_result_paths(result, artifact_paths)
        save_metadata(result.to_metadata(), out / "execution_summary.json")
    return result


def _execute_one_attempt(
    *,
    request: ContourProposalRequest,
    provider: ContourProposalProvider,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
    allowed_source_labels: Sequence[str],
    target_label: str,
    max_regions: int,
    max_points_per_region: int,
    projection_mode: str,
    organic_seed: int,
) -> LLMContourAttempt:
    proposal: ContourProposal | None = None
    edit_result: PrimitiveEditResult | None = None
    validation: ValidationResult | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None

    try:
        raw = provider.propose(request)
        raw_response = dict(raw)
    except Exception as exc:  # pragma: no cover - defensive provider boundary.
        error = str(exc)
        feedback = build_repair_feedback(
            status=STATUS_PROVIDER_ERROR,
            attempt_index=request.attempt_index,
            error=error,
        )
        return LLMContourAttempt(
            attempt_index=request.attempt_index,
            status=STATUS_PROVIDER_ERROR,
            raw_response=None,
            repair_feedback=feedback,
            error=error,
        )

    try:
        proposal = validate_contour_proposal(
            raw_response,
            schema=schema,
            mask_shape=tuple(source_mask.shape),
            primitive=intent.primitive,
            reference_profile=intent.reference_profile or schema.reference_profile,
            target_label=target_label,
            allowed_source_labels=tuple(allowed_source_labels),
            max_regions=max_regions,
            max_points_per_region=max_points_per_region,
        )
    except ContourProposalValidationError as exc:
        error = str(exc)
        feedback = build_repair_feedback(
            status=STATUS_PROPOSAL_REJECTED,
            attempt_index=request.attempt_index,
            error=error,
        )
        return LLMContourAttempt(
            attempt_index=request.attempt_index,
            status=STATUS_PROPOSAL_REJECTED,
            raw_response=raw_response,
            repair_feedback=feedback,
            error=error,
        )

    try:
        edit_result = execute_contour_proposal_write(
            source_mask,
            proposal,
            schema=schema,
            primitive_config=primitive_config,
            preserve_labels=intent.preserve_labels,
            forbidden_labels=intent.forbidden_labels,
            projection_mode=projection_mode,
            organic_seed=organic_seed,
            strength=intent.strength,
        )
        validation = validate_edit_result(
            src_mask=source_mask,
            target_mask=edit_result.target_mask,
            change_region=edit_result.change_region,
            schema=schema,
            primitive_config=primitive_config,
            changed_area_fraction=edit_result.changed_area_fraction,
            strength=intent.strength,
            execution_log=edit_result.ops_log,
        )
        status = STATUS_VALIDATED if validation.passed else STATUS_VALIDATION_FAILED
        feedback = None
        if status != STATUS_VALIDATED:
            feedback = build_repair_feedback(
                status=status,
                attempt_index=request.attempt_index,
                validation=validation,
                edit_result=edit_result,
            )
        return LLMContourAttempt(
            attempt_index=request.attempt_index,
            status=status,
            raw_response=raw_response,
            proposal=proposal,
            edit_result=edit_result,
            validation=validation,
            repair_feedback=feedback,
        )
    except Exception as exc:  # pragma: no cover - defensive execution boundary.
        error = str(exc)
        feedback = build_repair_feedback(
            status=STATUS_EXECUTION_ERROR,
            attempt_index=request.attempt_index,
            error=error,
        )
        return LLMContourAttempt(
            attempt_index=request.attempt_index,
            status=STATUS_EXECUTION_ERROR,
            raw_response=raw_response,
            proposal=proposal,
            repair_feedback=feedback,
            error=error,
        )


def _save_run_inputs(
    source_mask: np.ndarray,
    out: Path,
    *,
    grid_spacing_px: int,
) -> tuple[dict[str, str], tuple[str, ...]]:
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    paths["source_mask"] = str(save_id_mask(source_mask, out / "source_mask.png"))
    paths["source_mask_rgb"] = str(save_rgb_mask(source_mask, out / "source_mask_rgb.png"))
    preview_rgb = id_mask_to_llm_preview_rgb(source_mask)
    llm_rgb = out / "source_mask_llm_rgb.png"
    Image.fromarray(preview_rgb.astype(np.uint8), mode="RGB").save(llm_rgb)
    paths["source_mask_llm_rgb"] = str(llm_rgb)
    grid_rgb = out / "source_mask_llm_rgb_grid.png"
    Image.fromarray(
        add_coordinate_grid_overlay(preview_rgb, grid_spacing_px=grid_spacing_px),
        mode="RGB",
    ).save(grid_rgb)
    paths["source_mask_llm_rgb_grid"] = str(grid_rgb)
    return paths, (str(grid_rgb), str(llm_rgb))


def _save_attempt_artifacts(
    attempt: LLMContourAttempt,
    out: Path,
    *,
    request: ContourProposalRequest,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
    projection_mode: str,
) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    paths["prompt"] = str(save_prompt_text(request.prompt, out / "prompt.txt"))
    request_metadata = {
        "attempt_index": request.attempt_index,
        "image_paths": list(request.image_paths),
        "provider_metadata": request.provider_metadata,
        "repair_feedback": request.repair_feedback,
    }
    paths["llm_request"] = str(save_metadata(request_metadata, out / "llm_request.json"))
    if attempt.raw_response is not None:
        paths["llm_response"] = str(save_metadata(attempt.raw_response, out / "llm_response.json"))
        (out / "llm_response_raw.txt").write_text(
            json.dumps(attempt.raw_response, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths["llm_response_raw"] = str(out / "llm_response_raw.txt")
    if attempt.proposal is not None:
        paths["validated_proposal"] = str(
            save_metadata(attempt.proposal.raw_payload, out / "validated_proposal.json")
        )
        paths["rasterized_region"] = str(
            save_change_region(rasterize_contour_proposal(attempt.proposal), out / "rasterized_region.png")
        )
    if attempt.edit_result is not None:
        paths["projected_region"] = str(
            save_change_region(attempt.edit_result.change_region, out / "projected_region.png")
        )
        paths["change_region"] = str(
            save_change_region(attempt.edit_result.change_region, out / "change_region.png")
        )
        paths["target_mask"] = str(
            save_id_mask(attempt.edit_result.target_mask, out / "target_mask.png")
        )
        paths["target_mask_rgb"] = str(
            save_rgb_mask(attempt.edit_result.target_mask, out / "target_mask_rgb.png")
        )
        if attempt.proposal is not None:
            paths.update(
                {
                    f"visual_qa_{key}": value
                    for key, value in save_visual_qa_bundle(
                        source_mask=source_mask,
                        proposal=attempt.proposal,
                        schema=schema,
                        edit_result=attempt.edit_result,
                        validation=attempt.validation,
                        output_dir=out / "visual_qa",
                        primitive_config=primitive_config,
                        preserve_labels=intent.preserve_labels,
                        forbidden_labels=intent.forbidden_labels,
                        projection_mode=projection_mode,
                    ).items()
                }
            )
    if attempt.validation is not None:
        paths["validation"] = str(
            save_metadata(_jsonable_dataclass(attempt.validation), out / "validation.json")
        )
    if attempt.repair_feedback is not None:
        paths["repair_feedback"] = str(
            save_metadata(attempt.repair_feedback, out / "repair_feedback.json")
        )
    paths["attempt_summary"] = str(
        save_metadata(_attempt_metadata(attempt), out / "attempt_summary.json")
    )
    return paths


def _attempt_metadata(attempt: LLMContourAttempt) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "attempt_index": attempt.attempt_index,
        "status": attempt.status,
        "error": attempt.error,
        "artifact_paths": dict(attempt.artifact_paths or {}),
    }
    if attempt.proposal is not None:
        metadata["proposal"] = {
            "primitive": attempt.proposal.primitive,
            "reference_profile": attempt.proposal.reference_profile,
            "target_label": attempt.proposal.target_label,
            "regions": [
                {
                    "region_id": region.region_id,
                    "source_labels": list(region.source_labels),
                    "points": [list(point) for point in region.points],
                    "confidence": region.confidence,
                }
                for region in attempt.proposal.regions
            ],
        }
    if attempt.edit_result is not None:
        metadata["edit_result"] = {
            "selected_pixels": attempt.edit_result.selected_pixels,
            "changed_area_fraction": attempt.edit_result.changed_area_fraction,
            "warnings": list(attempt.edit_result.warnings),
            "ops_log": attempt.edit_result.ops_log,
        }
    if attempt.validation is not None:
        metadata["validation"] = _jsonable_dataclass(attempt.validation)
    if attempt.repair_feedback is not None:
        metadata["repair_feedback"] = attempt.repair_feedback
    return metadata


def _replace_attempt_paths(
    attempt: LLMContourAttempt,
    artifact_paths: dict[str, str],
) -> LLMContourAttempt:
    return LLMContourAttempt(
        attempt_index=attempt.attempt_index,
        status=attempt.status,
        raw_response=attempt.raw_response,
        proposal=attempt.proposal,
        edit_result=attempt.edit_result,
        validation=attempt.validation,
        repair_feedback=attempt.repair_feedback,
        artifact_paths=artifact_paths,
        error=attempt.error,
    )


def _replace_result_paths(
    result: LLMContourAgentResult,
    artifact_paths: dict[str, str],
) -> LLMContourAgentResult:
    return LLMContourAgentResult(
        status=result.status,
        source_mask=result.source_mask,
        attempts=result.attempts,
        final_attempt=result.final_attempt,
        context=result.context,
        artifact_paths=artifact_paths,
        projection_mode=result.projection_mode,
        error=result.error,
    )


def _jsonable_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _labels_from_operation(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return []


def _default_source_labels(
    primitive_config: Mapping[str, Any],
    operation: Any,
) -> list[str]:
    if not isinstance(operation, Mapping):
        operation = {}
    if primitive_config.get("name") == "tumor_burden_increase":
        return _labels_from_operation(operation.get("target_priority"))
    labels = _labels_from_operation(operation.get("source"))
    if labels:
        return labels
    labels.extend(_labels_from_operation(operation.get("primary_sources")))
    labels.extend(_labels_from_operation(operation.get("secondary_sources")))
    return list(dict.fromkeys(labels))


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _backfill_prompt_target_label(
    primitive_config: Mapping[str, Any],
    schema: MaskProfileSchema,
    mask: np.ndarray | None = None,
) -> str | None:
    operation = primitive_config.get("mask_operation", {})
    priority = operation.get("backfill_priority", ()) if isinstance(operation, Mapping) else ()
    if isinstance(priority, list):
        for label in priority:
            if not isinstance(label, str) or label not in schema.writable_labels:
                continue
            if mask is not None and label in schema.readable_labels:
                if not np.any(np.isin(mask, schema.resolve_fine_ids(label))):
                    continue
                return label
            return label
    return None


def _is_message_length_exceeded(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return (
        "message_length_exceeds_limit" in lowered
        or "message you submitted was too long" in lowered
    )


def _with_compact_context_feedback(
    repair_feedback: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if repair_feedback is None:
        return None
    updated = dict(repair_feedback)
    updated["repair_instruction"] = (
        "Previous request exceeded the provider message length limit. "
        "The next attempt uses compact mask context with fewer source components, "
        "tiles, contour points, and adjacency segments."
    )
    updated["context_mode_next_attempt"] = "compact"
    return updated


def _with_task_specific_repair_feedback(
    repair_feedback: dict[str, Any] | None,
    *,
    primitive_name: str,
    repair_profile: str | None,
    target_label: str,
    allowed_source_labels: Sequence[str],
) -> dict[str, Any] | None:
    if repair_feedback is None:
        return None
    updated = dict(repair_feedback)
    instruction_parts: list[str] = []
    if primitive_name in {"stromal_desmoplasia", "stroma_decrease", "stromal_reduction"}:
        instruction_parts.append(
            f"Focus on {target_label} with broad enough coverage to satisfy the stroma area guard."
        )
        if repair_profile == "stroma_immune_balance":
            instruction_parts.append(
                "Reduce immune conversion inside the proposed region and keep the write centered on Stroma."
            )
    elif primitive_name == "necrosis_appearance":
        instruction_parts.append(
            "Push the next proposal deeper into Tumor and make the changed area larger."
        )
        if repair_profile == "necrosis_growth":
            instruction_parts.append("Anchor the coarse template on tumor interior components, not the outer edge.")
    elif primitive_name == "tumor_burden_increase":
        instruction_parts.append(
            "Expand the editable region so the deterministic projection can reach the target area."
        )
    elif primitive_name == "tumor_burden_decrease":
        instruction_parts.append(
            "Increase the regression footprint and avoid tiny isolated polygons."
        )
    elif "fine_label_transition" in primitive_name or repair_profile == "fine_transition_area":
        instruction_parts.append(
            "Select a larger connected source component group so the final relative change is within range."
        )
    if instruction_parts:
        updated["repair_instruction"] = " ".join(instruction_parts)
    if allowed_source_labels:
        updated["allowed_source_labels_next_attempt"] = list(allowed_source_labels)
    updated["repair_profile_next_attempt"] = repair_profile
    return updated


def _compact_mask_context_for_retry(context: Mapping[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(dict(context))
    compression = {
        "enabled": True,
        "reason": "provider_message_length_exceeds_limit",
        "source_spatial_components_per_label": 3,
        "source_spatial_tiles_per_label": 3,
        "source_contour_components_per_label": 2,
        "contour_points_per_component": 16,
        "adjacency_segments_per_label": 1,
        "adjacency_points_per_segment": 4,
    }
    compact["context_compression"] = compression
    _compact_source_spatial_hints(
        compact.get("source_spatial_hints"),
        max_components=int(compression["source_spatial_components_per_label"]),
        max_tiles=int(compression["source_spatial_tiles_per_label"]),
    )
    _compact_source_contour_context(
        compact.get("source_contour_context"),
        max_components=int(compression["source_contour_components_per_label"]),
        max_contour_points=int(compression["contour_points_per_component"]),
        max_segments_per_label=int(compression["adjacency_segments_per_label"]),
        max_points_per_segment=int(compression["adjacency_points_per_segment"]),
    )
    return compact


def _compact_source_spatial_hints(
    source_spatial_hints: Any,
    *,
    max_components: int,
    max_tiles: int,
) -> None:
    if not isinstance(source_spatial_hints, dict):
        return
    for hint in source_spatial_hints.values():
        if not isinstance(hint, dict):
            continue
        components = hint.get("components")
        if isinstance(components, list):
            hint["components"] = components[:max_components]
            hint["components_truncated_to"] = int(max_components)
        tiles = hint.get("high_purity_grid_tiles")
        if isinstance(tiles, list):
            hint["high_purity_grid_tiles"] = tiles[:max_tiles]
            hint["high_purity_grid_tiles_truncated_to"] = int(max_tiles)


def _compact_source_contour_context(
    source_contour_context: Any,
    *,
    max_components: int,
    max_contour_points: int,
    max_segments_per_label: int,
    max_points_per_segment: int,
) -> None:
    if not isinstance(source_contour_context, dict):
        return
    for label_context in source_contour_context.values():
        if not isinstance(label_context, dict):
            continue
        components = label_context.get("components")
        if not isinstance(components, list):
            continue
        label_context["components"] = components[:max_components]
        label_context["components_truncated_to"] = int(max_components)
        for component in label_context["components"]:
            if not isinstance(component, dict):
                continue
            contour = component.get("contour_simplified")
            if isinstance(contour, list) and len(contour) > max_contour_points:
                component["contour_simplified"] = _evenly_spaced_items(
                    contour, max_contour_points
                )
                component["contour_points_truncated_to"] = int(max_contour_points)
            segments = component.get("contour_adjacency_segments")
            if isinstance(segments, dict):
                component["contour_adjacency_segments"] = (
                    _compact_adjacency_segments(
                        segments,
                        max_segments_per_label=max_segments_per_label,
                        max_points_per_segment=max_points_per_segment,
                    )
                )


def _compact_adjacency_segments(
    segments: Mapping[str, Any],
    *,
    max_segments_per_label: int,
    max_points_per_segment: int,
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for label, label_segments in segments.items():
        if label == "_meta":
            compact[label] = label_segments
            continue
        if not isinstance(label_segments, list):
            continue
        compact_segments: list[dict[str, Any]] = []
        for segment in label_segments[:max_segments_per_label]:
            if not isinstance(segment, dict):
                continue
            compact_segment = dict(segment)
            points = compact_segment.get("points")
            if isinstance(points, list) and len(points) > max_points_per_segment:
                compact_segment["points"] = _evenly_spaced_items(
                    points, max_points_per_segment
                )
                compact_segment["points_truncated_to"] = int(max_points_per_segment)
            compact_segments.append(compact_segment)
        if compact_segments:
            compact[label] = compact_segments
    return compact


def _evenly_spaced_items(items: list[Any], max_items: int) -> list[Any]:
    if len(items) <= max_items:
        return list(items)
    if max_items <= 0:
        return []
    indices = np.linspace(0, len(items) - 1, num=max_items, dtype=int)
    return [items[int(index)] for index in indices]


def _provider_request_metadata(
    provider: ContourProposalProvider,
    image_paths: Sequence[str],
) -> dict[str, Any]:
    provider_name = getattr(provider, "name", provider.__class__.__name__)
    uses_multimodal = isinstance(provider, OpenAICompatibleMultimodalContourProvider)
    return {
        "provider_name": provider_name,
        "provider_class": provider.__class__.__name__,
        "request_mode": "multimodal" if uses_multimodal else "text",
        "image_paths_available": len(tuple(image_paths)),
        "image_parts_expected": 1 if uses_multimodal else 0,
        "image_policy": (
            "send first image path only as data:image/png;base64 image_url"
            if uses_multimodal
            else "do not send images"
        ),
    }


def _post_chat_completion(
    payload: Mapping[str, Any],
    *,
    api_base_url: str,
    api_key: str,
    timeout_sec: float,
) -> dict[str, Any]:
    endpoint = api_base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            response_data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ContourProviderError(
            f"API request failed with HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ContourProviderError(f"API request failed: {exc}") from exc

    try:
        decoded = json.loads(response_data)
    except json.JSONDecodeError as exc:
        raise ContourProviderError("API response was not valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise ContourProviderError("API response root must be a JSON object.")
    return decoded


def _response_content(response_payload: Mapping[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ContourProviderError("API response missing choices.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ContourProviderError("API response choice must be a mapping.")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ContourProviderError("API response choice missing message.")
    content = message.get("content")
    if not isinstance(content, str):
        raise ContourProviderError("API response message content must be a string.")
    return content


def _parse_json_object_content(content: str) -> Mapping[str, Any]:
    """Parse provider content, tolerating common JSON markdown fences."""

    candidates = [content, _strip_markdown_json_fence(content)]
    extracted = _extract_json_object(content)
    if extracted is not None:
        candidates.append(extracted)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, Mapping):
            raise ContourProviderError(
                "API response content root must be a JSON object."
            )
        return parsed
    raise ContourProviderError("API response content was not valid JSON.") from last_error


def _strip_markdown_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_object(content: str) -> str | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return content[start : end + 1].strip()


def _image_path_to_data_url(path: str | Path) -> str:
    p = Path(path)
    try:
        encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    except OSError as exc:
        raise ContourProviderError(f"Could not read image path: {p}") from exc
    return f"data:image/png;base64,{encoded}"
