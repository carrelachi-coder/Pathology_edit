"""Metrics for comparing measured mask edits with structured benchmark GT."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from phase3_mask_edit.benchmark.intents import (
    labels_from_operation,
    legal_pixel_mask,
    primitive_config_by_name,
    safe_schema_label_mask,
    strength_denominator_pixels,
)
from phase3_mask_edit.benchmark.models import BenchmarkIntent
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.labels import MaskProfileSchema


def evaluate_mask_edit(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    intent: BenchmarkIntent,
    *,
    mode: str = "gt",
    location_iou_threshold: float = 0.05,
    centroid_tolerance_fraction: float = 0.35,
) -> dict[str, Any]:
    schema = MaskProfileSchema.from_reference_profile(intent.profile)
    recipe = load_recipe(default_recipe_path_for_profile(intent.profile))
    primitive_config = primitive_config_by_name(recipe, intent.primitive)
    change_region = source_mask != target_mask
    changed_pixels = int(np.count_nonzero(change_region))
    denominator = strength_denominator_pixels(source_mask, primitive_config, schema)
    area_fraction = changed_pixels / denominator if denominator > 0 else 0.0
    class_delta = measured_class_delta(source_mask, target_mask, schema)
    transition = expected_transition_summary(
        source_mask,
        target_mask,
        intent,
        schema=schema,
        primitive_config=primitive_config,
    )
    class_ok = transition["on_target_transition_pixels"] > 0
    direction_ok = _direction_ok(
        class_delta,
        intent,
        source_mask=source_mask,
        target_mask=target_mask,
        schema=schema,
        primitive_config=primitive_config,
    )
    strength_ok = _strength_ok(area_fraction, intent.expected_area_bucket)
    measured_location = measured_location_summary(change_region, intent.region_hint)
    location_ok = _location_ok(
        measured_location,
        location_iou_threshold=location_iou_threshold,
        centroid_tolerance_fraction=centroid_tolerance_fraction,
    )
    result = {
        "measured_class_delta": class_delta,
        "measured_area_fraction": float(area_fraction),
        "measured_location": measured_location,
        "changed_pixels": changed_pixels,
        "strength_denominator_pixels": int(denominator),
        "direction_hit": bool(direction_ok),
        **transition,
        "spatial_containment_ratio": float(measured_location["containment_ratio"]),
        "magnitude_bucket_pass": bool(strength_ok),
        "class_ok": bool(class_ok),
        "direction_ok": bool(direction_ok),
        "strength_ok": bool(strength_ok),
        "location_ok": bool(location_ok),
        "all_ok": bool(class_ok and direction_ok and strength_ok and location_ok),
    }
    result.update(mode_aware_score_fields(result, mode=mode))
    return result


def mode_aware_score_fields(
    metrics: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Return headline and diagnostic score fields under the mode-specific policy."""

    has_core_fields = any(
        _has_value(metrics.get(key))
        for key in ("class_ok", "direction_ok", "location_ok")
    )
    class_ok = _as_bool(metrics.get("class_ok"))
    direction_ok = _as_bool(metrics.get("direction_ok"))
    location_ok = _as_bool(metrics.get("location_ok"))
    has_magnitude_fields = any(
        _has_value(metrics.get(key))
        for key in (
            "intended_magnitude_bucket_agreement",
            "magnitude_bucket_pass",
            "strength_ok",
        )
    )
    magnitude_agreement = (
        _as_bool(
            metrics.get(
                "intended_magnitude_bucket_agreement",
                metrics.get("magnitude_bucket_pass", metrics.get("strength_ok")),
            )
        )
        if has_magnitude_fields
        else _as_bool(
            metrics.get("strict_all_ok")
            if _has_value(metrics.get("strict_all_ok"))
            else metrics.get("all_ok")
        )
    )
    semantic_core_ok = (
        class_ok and direction_ok and location_ok
        if has_core_fields
        else _as_bool(
            metrics.get("semantic_core_ok")
            if _has_value(metrics.get("semantic_core_ok"))
            else metrics.get("all_ok")
        )
    )
    strict_all_ok = semantic_core_ok and magnitude_agreement
    prompt_mode = str(mode or "").lower() == "prompt"
    return {
        "intended_magnitude_bucket_agreement": magnitude_agreement,
        "semantic_core_ok": semantic_core_ok,
        "strict_all_ok": strict_all_ok,
        "primary_ok": semantic_core_ok if prompt_mode else strict_all_ok,
        "strength_evaluation_policy": (
            "ordinal_secondary_hidden_bucket_diagnostic"
            if prompt_mode
            else "strict_intended_bucket"
        ),
    }


def measured_class_delta(
    source_mask: np.ndarray, target_mask: np.ndarray, schema: MaskProfileSchema
) -> dict[str, int]:
    delta: dict[str, int] = {}
    for label in sorted(schema.readable_labels):
        before = int(
            np.count_nonzero(safe_schema_label_mask(source_mask, schema, label))
        )
        after = int(
            np.count_nonzero(safe_schema_label_mask(target_mask, schema, label))
        )
        if before != after:
            delta[label] = after - before
    return delta


def measured_location_summary(
    change_region: np.ndarray, region_hint: Mapping[str, Any]
) -> dict[str, Any]:
    changed = np.asarray(change_region, dtype=bool)
    hint_mask = _hint_mask(changed.shape, region_hint)
    changed_pixels = int(np.count_nonzero(changed))
    overlap = int(np.count_nonzero(changed & hint_mask)) if hint_mask is not None else 0
    union = (
        int(np.count_nonzero(changed | hint_mask))
        if hint_mask is not None
        else changed_pixels
    )
    rows, cols = np.nonzero(changed)
    if changed_pixels:
        centroid = [float(np.mean(cols)), float(np.mean(rows))]
    else:
        centroid = None
    expected_centroid = (
        region_hint.get("centroid_xy") if isinstance(region_hint, Mapping) else None
    )
    distance = None
    normalized_distance = None
    if (
        centroid is not None
        and isinstance(expected_centroid, list)
        and len(expected_centroid) == 2
    ):
        dx = centroid[0] - float(expected_centroid[0])
        dy = centroid[1] - float(expected_centroid[1])
        distance = float(np.hypot(dx, dy))
        normalized_distance = distance / max(changed.shape)
    return {
        "changed_pixels": changed_pixels,
        "overlap_pixels": overlap,
        "containment_ratio": float(overlap / changed_pixels) if changed_pixels else 0.0,
        "iou": float(overlap / union) if union else 0.0,
        "centroid_xy": centroid,
        "expected_centroid_xy": expected_centroid,
        "centroid_distance_px": distance,
        "centroid_distance_fraction": normalized_distance,
    }


def expected_transition_summary(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    intent: BenchmarkIntent,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure whether changed pixels follow the intended source-to-target mapping."""

    changed = np.asarray(source_mask) != np.asarray(target_mask)
    changed_pixels = int(np.count_nonzero(changed))
    source_allowed, target_expected = _expected_source_target_masks(
        source_mask,
        target_mask,
        intent,
        schema=schema,
        primitive_config=primitive_config,
    )
    on_target = changed & source_allowed & target_expected
    on_target_pixels = int(np.count_nonzero(on_target))
    off_target_pixels = max(0, changed_pixels - on_target_pixels)
    return {
        "on_target_transition_pixels": on_target_pixels,
        "on_target_transition_ratio": float(on_target_pixels / changed_pixels)
        if changed_pixels
        else 0.0,
        "off_target_change_pixels": off_target_pixels,
        "off_target_change_ratio": float(off_target_pixels / changed_pixels)
        if changed_pixels
        else 0.0,
    }


def row_for_eval(
    *,
    sample_id: str,
    mode: str,
    status: str,
    parsed_semantic_diff: Mapping[str, Any] | None,
    planned_primitive: str | None,
    metrics: Mapping[str, Any] | None,
    error: str = "",
    output_dir: str = "",
    organ: str = "",
    profile: str = "",
    primitive: str = "",
    strength: str = "",
    source_dataset: str = "",
    wsi_id: str = "",
    patient_id: str = "",
    ordinal_group_id: str = "",
    contour_agentic: Mapping[str, Any] | None = None,
    semantic_agentic: Mapping[str, Any] | None = None,
    failure_stage: str = "",
) -> dict[str, Any]:
    metrics = metrics or {}
    contour_agentic = contour_agentic or {}
    semantic_agentic = semantic_agentic or {}
    return {
        "sample_id": sample_id,
        "organ": organ,
        "profile": profile,
        "primitive": primitive,
        "strength": strength,
        "mode": mode,
        "status": status,
        "parsed_semantic_diff": json.dumps(
            parsed_semantic_diff or {}, ensure_ascii=False, sort_keys=True
        ),
        "planned_primitive": planned_primitive or "",
        "measured_class_delta": json.dumps(
            metrics.get("measured_class_delta", {}), ensure_ascii=False, sort_keys=True
        ),
        "measured_area_fraction": metrics.get("measured_area_fraction", ""),
        "measured_location": json.dumps(
            metrics.get("measured_location", {}), ensure_ascii=False, sort_keys=True
        ),
        "changed_pixels": metrics.get("changed_pixels", 0),
        "strength_denominator_pixels": metrics.get("strength_denominator_pixels", 0),
        "direction_hit": metrics.get("direction_hit", False),
        "on_target_transition_pixels": metrics.get("on_target_transition_pixels", 0),
        "on_target_transition_ratio": metrics.get("on_target_transition_ratio", 0.0),
        "off_target_change_pixels": metrics.get("off_target_change_pixels", 0),
        "off_target_change_ratio": metrics.get("off_target_change_ratio", 0.0),
        "spatial_containment_ratio": metrics.get("spatial_containment_ratio", 0.0),
        "magnitude_bucket_pass": metrics.get("magnitude_bucket_pass", False),
        "intended_magnitude_bucket_agreement": metrics.get(
            "intended_magnitude_bucket_agreement",
            metrics.get("magnitude_bucket_pass", False),
        ),
        "class_ok": metrics.get("class_ok", False),
        "direction_ok": metrics.get("direction_ok", False),
        "strength_ok": metrics.get("strength_ok", False),
        "location_ok": metrics.get("location_ok", False),
        "semantic_core_ok": metrics.get("semantic_core_ok", False),
        "strict_all_ok": metrics.get("strict_all_ok", metrics.get("all_ok", False)),
        "primary_ok": metrics.get("primary_ok", metrics.get("all_ok", False)),
        "strength_evaluation_policy": metrics.get("strength_evaluation_policy", ""),
        "all_ok": metrics.get("all_ok", False),
        "attempt_count": contour_agentic.get("attempt_count", 0),
        "first_attempt_status": contour_agentic.get("first_attempt_status", ""),
        "final_attempt_status": contour_agentic.get("final_attempt_status", ""),
        "replanned": contour_agentic.get("replanned", False),
        "repair_success": contour_agentic.get("repair_success", False),
        "terminal_failure_reason": contour_agentic.get("terminal_failure_reason", ""),
        "cumulative_success_at_k": contour_agentic.get("cumulative_success_at_k", {}),
        "semantic_attempt_count": semantic_agentic.get("semantic_attempt_count", 0),
        "semantic_first_attempt_status": semantic_agentic.get(
            "semantic_first_attempt_status", ""
        ),
        "semantic_final_attempt_status": semantic_agentic.get(
            "semantic_final_attempt_status", ""
        ),
        "semantic_replanned": semantic_agentic.get("semantic_replanned", False),
        "semantic_repair_success": semantic_agentic.get(
            "semantic_repair_success", False
        ),
        "semantic_terminal_failure_reason": semantic_agentic.get(
            "semantic_terminal_failure_reason", ""
        ),
        "failure_stage": failure_stage,
        "source_dataset": source_dataset,
        "wsi_id": wsi_id,
        "patient_id": patient_id,
        "ordinal_group_id": ordinal_group_id,
        "error": error,
        "output_dir": output_dir,
    }


def _direction_ok(
    delta: Mapping[str, int],
    intent: BenchmarkIntent,
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
) -> bool:
    direction = intent.expected_direction
    target = intent.target_label
    source_labels = set(intent.source_labels)
    if direction == "increase":
        if target and delta.get(target, 0) > 0:
            return True
        return any(
            value > 0 for label, value in delta.items() if label not in source_labels
        )
    if direction == "decrease":
        return any(delta.get(label, 0) < 0 for label in source_labels) or any(
            value < 0 for value in delta.values()
        )
    if direction == "transition":
        operation = primitive_config.get("mask_operation", {})
        operation = operation if isinstance(operation, Mapping) else {}
        source_ids = _integer_ids(operation.get("source_fine_ids"))
        target_ids = _integer_ids(operation.get("target_fine_id"))
        if not source_ids:
            source_ids = _source_fine_ids_for_transition(intent, schema)
        source_before = int(np.count_nonzero(np.isin(source_mask, source_ids)))
        source_after = int(np.count_nonzero(np.isin(target_mask, source_ids)))
        target_before = (
            int(np.count_nonzero(np.isin(source_mask, target_ids))) if target_ids else 0
        )
        target_after = (
            int(np.count_nonzero(np.isin(target_mask, target_ids))) if target_ids else 0
        )
        return source_after < source_before and (
            not target_ids or target_after > target_before
        )
    return bool(delta)


def _strength_ok(area_fraction: float, bucket: tuple[float, float] | None) -> bool:
    if bucket is None:
        return area_fraction > 0
    lower, upper = bucket
    tolerance = max(0.02, (upper - lower) * 0.25)
    return (lower - tolerance) <= area_fraction <= (upper + tolerance)


def _location_ok(
    summary: Mapping[str, Any],
    *,
    location_iou_threshold: float,
    centroid_tolerance_fraction: float,
) -> bool:
    if int(summary.get("changed_pixels") or 0) <= 0:
        return False
    if float(summary.get("iou") or 0.0) >= location_iou_threshold:
        return True
    distance_fraction = summary.get("centroid_distance_fraction")
    if distance_fraction is None:
        return True
    return float(distance_fraction) <= centroid_tolerance_fraction


def _hint_mask(
    shape: tuple[int, int], region_hint: Mapping[str, Any]
) -> np.ndarray | None:
    bbox = region_hint.get("bbox_xyxy") if isinstance(region_hint, Mapping) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    h, w = shape
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    mask = np.zeros(shape, dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _expected_source_target_masks(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    intent: BenchmarkIntent,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    operation = primitive_config.get("mask_operation", {})
    operation = operation if isinstance(operation, Mapping) else {}
    source_ids = _integer_ids(operation.get("source_fine_ids"))
    target_ids = _integer_ids(operation.get("target_fine_id"))
    if source_ids and target_ids:
        return np.isin(source_mask, source_ids), np.isin(target_mask, target_ids)

    source_allowed = np.zeros(source_mask.shape, dtype=bool)
    for label in intent.source_labels:
        source_allowed |= safe_schema_label_mask(source_mask, schema, label)
    if not np.any(source_allowed):
        source_allowed = legal_pixel_mask(source_mask, primitive_config, schema)

    target_labels = list(intent.target_label and [intent.target_label] or [])
    target_labels.extend(labels_from_operation(operation.get("target")))
    target_labels.extend(labels_from_operation(operation.get("backfill_priority")))
    target_expected = np.zeros(target_mask.shape, dtype=bool)
    for label in dict.fromkeys(target_labels):
        target_expected |= safe_schema_label_mask(target_mask, schema, label)
    if not np.any(target_expected) and target_ids:
        target_expected = np.isin(target_mask, target_ids)
    if intent.target_label is None and not target_ids:
        target_expected = source_mask != target_mask
    return source_allowed, target_expected


def _integer_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value if isinstance(item, (int, np.integer)))
    return ()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _has_value(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "null", "none"}


def _source_fine_ids_for_transition(
    intent: BenchmarkIntent, schema: MaskProfileSchema
) -> tuple[int, ...]:
    del schema
    mapping = {
        "gleason_upgrade_3to4": (8,),
        "gleason_upgrade_4to5": (9,),
        "gleason_downgrade_4to3": (9,),
        "benign_to_gleason3": (5,),
        "benign_atrophy": (5,),
        "normal_to_adenomatous": (5,),
        "adenoma_to_carcinoma": (11,),
        "grade_upgrade": (12,),
        "treatment_dedifferentiation": (13,),
    }
    return mapping.get(intent.primitive, ())
