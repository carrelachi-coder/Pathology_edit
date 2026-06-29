"""Metrics for comparing measured mask edits with structured benchmark GT."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from phase3_mask_edit.benchmark.intents import (
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
    class_ok = _class_ok(source_mask, target_mask, intent, schema)
    direction_ok = _direction_ok(class_delta, intent)
    strength_ok = _strength_ok(area_fraction, intent.expected_area_bucket)
    measured_location = measured_location_summary(change_region, intent.region_hint)
    location_ok = _location_ok(
        measured_location,
        location_iou_threshold=location_iou_threshold,
        centroid_tolerance_fraction=centroid_tolerance_fraction,
    )
    return {
        "measured_class_delta": class_delta,
        "measured_area_fraction": float(area_fraction),
        "measured_location": measured_location,
        "class_ok": bool(class_ok),
        "direction_ok": bool(direction_ok),
        "strength_ok": bool(strength_ok),
        "location_ok": bool(location_ok),
        "all_ok": bool(class_ok and direction_ok and strength_ok and location_ok),
    }


def measured_class_delta(source_mask: np.ndarray, target_mask: np.ndarray, schema: MaskProfileSchema) -> dict[str, int]:
    delta: dict[str, int] = {}
    for label in sorted(schema.readable_labels):
        before = int(np.count_nonzero(safe_schema_label_mask(source_mask, schema, label)))
        after = int(np.count_nonzero(safe_schema_label_mask(target_mask, schema, label)))
        if before != after:
            delta[label] = after - before
    return delta


def measured_location_summary(change_region: np.ndarray, region_hint: Mapping[str, Any]) -> dict[str, Any]:
    changed = np.asarray(change_region, dtype=bool)
    hint_mask = _hint_mask(changed.shape, region_hint)
    changed_pixels = int(np.count_nonzero(changed))
    overlap = int(np.count_nonzero(changed & hint_mask)) if hint_mask is not None else 0
    union = int(np.count_nonzero(changed | hint_mask)) if hint_mask is not None else changed_pixels
    rows, cols = np.nonzero(changed)
    if changed_pixels:
        centroid = [float(np.mean(cols)), float(np.mean(rows))]
    else:
        centroid = None
    expected_centroid = region_hint.get("centroid_xy") if isinstance(region_hint, Mapping) else None
    distance = None
    normalized_distance = None
    if centroid is not None and isinstance(expected_centroid, list) and len(expected_centroid) == 2:
        dx = centroid[0] - float(expected_centroid[0])
        dy = centroid[1] - float(expected_centroid[1])
        distance = float(np.hypot(dx, dy))
        normalized_distance = distance / max(changed.shape)
    return {
        "changed_pixels": changed_pixels,
        "overlap_pixels": overlap,
        "iou": float(overlap / union) if union else 0.0,
        "centroid_xy": centroid,
        "expected_centroid_xy": expected_centroid,
        "centroid_distance_px": distance,
        "centroid_distance_fraction": normalized_distance,
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
) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "sample_id": sample_id,
        "organ": organ,
        "profile": profile,
        "primitive": primitive,
        "strength": strength,
        "mode": mode,
        "status": status,
        "parsed_semantic_diff": json.dumps(parsed_semantic_diff or {}, ensure_ascii=False, sort_keys=True),
        "planned_primitive": planned_primitive or "",
        "measured_class_delta": json.dumps(metrics.get("measured_class_delta", {}), ensure_ascii=False, sort_keys=True),
        "measured_area_fraction": metrics.get("measured_area_fraction", ""),
        "measured_location": json.dumps(metrics.get("measured_location", {}), ensure_ascii=False, sort_keys=True),
        "class_ok": metrics.get("class_ok", False),
        "direction_ok": metrics.get("direction_ok", False),
        "strength_ok": metrics.get("strength_ok", False),
        "location_ok": metrics.get("location_ok", False),
        "all_ok": metrics.get("all_ok", False),
        "error": error,
        "output_dir": output_dir,
    }


def _class_ok(source_mask: np.ndarray, target_mask: np.ndarray, intent: BenchmarkIntent, schema: MaskProfileSchema) -> bool:
    if intent.expected_direction == "transition":
        source_before = sum(
            int(np.count_nonzero(source_mask == fine_id))
            for fine_id in _source_fine_ids_for_transition(intent, schema)
        )
        source_after = sum(
            int(np.count_nonzero(target_mask == fine_id))
            for fine_id in _source_fine_ids_for_transition(intent, schema)
        )
        return source_after < source_before
    labels = set(intent.source_labels)
    if intent.target_label:
        labels.add(intent.target_label)
    if not labels:
        return bool(np.any(source_mask != target_mask))
    return any(
        np.any(safe_schema_label_mask(source_mask, schema, label) != safe_schema_label_mask(target_mask, schema, label))
        for label in labels
    )


def _direction_ok(delta: Mapping[str, int], intent: BenchmarkIntent) -> bool:
    direction = intent.expected_direction
    target = intent.target_label
    source_labels = set(intent.source_labels)
    if direction == "increase":
        if target and delta.get(target, 0) > 0:
            return True
        return any(value > 0 for label, value in delta.items() if label not in source_labels)
    if direction == "decrease":
        return any(delta.get(label, 0) < 0 for label in source_labels) or any(value < 0 for value in delta.values())
    if direction == "transition":
        return any(value != 0 for value in delta.values())
    return bool(delta)


def _strength_ok(area_fraction: float, bucket: tuple[float, float] | None) -> bool:
    if bucket is None:
        return area_fraction > 0
    lower, upper = bucket
    tolerance = max(0.02, (upper - lower) * 0.25)
    return (lower - tolerance) <= area_fraction <= (upper + tolerance)


def _location_ok(summary: Mapping[str, Any], *, location_iou_threshold: float, centroid_tolerance_fraction: float) -> bool:
    if int(summary.get("changed_pixels") or 0) <= 0:
        return False
    if float(summary.get("iou") or 0.0) >= location_iou_threshold:
        return True
    distance_fraction = summary.get("centroid_distance_fraction")
    if distance_fraction is None:
        return True
    return float(distance_fraction) <= centroid_tolerance_fraction


def _hint_mask(shape: tuple[int, int], region_hint: Mapping[str, Any]) -> np.ndarray | None:
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


def _source_fine_ids_for_transition(intent: BenchmarkIntent, schema: MaskProfileSchema) -> tuple[int, ...]:
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
