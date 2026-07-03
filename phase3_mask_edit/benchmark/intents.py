"""Build structured GT intents for mask-edit semantic fidelity benchmarks."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy import ndimage

from phase3_mask_edit.benchmark.models import BenchmarkIntent
from phase3_mask_edit.core.applicability import assess_edit_applicability
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.intent import IntentValidationError, validate_intent_against_recipe
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import load_id_mask
from phase3_mask_edit.specialized.catalog import specialized_primitive_names


DEFAULT_PROFILES: dict[str, str] = {
    "breast": "BCSS",
    "prostate": "PANDA",
    "colorectal": "GlaS",
    "lung": "IGNITE",
    "melanoma": "PUMA",
    "oral": "ORCA",
}

DEFAULT_STRENGTHS = ("mild", "moderate", "significant", "xlarge_deid")
SPECIALIZED_NAMES = frozenset(specialized_primitive_names())


@dataclass(frozen=True)
class ProfileSource:
    organ: str
    profile: str
    mask_globs: tuple[str, ...]
    image_globs: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildConfig:
    data_root: Path
    output_dir: Path
    profiles: tuple[ProfileSource, ...]
    patches_per_combo: int = 20
    strengths: tuple[str, ...] = DEFAULT_STRENGTHS
    allowed_primitives: tuple[str, ...] = ()
    excluded_primitives: tuple[str, ...] = ()
    seed: int = 13
    max_masks_per_profile: int | None = None
    early_stop_when_full: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuildConfig":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        base_dir = config_path.parent
        data_root = _resolve_path(payload.get("data_root", "."), base_dir=base_dir)
        output_dir = _resolve_path(payload.get("output_dir", "benchmark_out"), base_dir=base_dir)
        profiles = _profile_sources(payload.get("profiles"), data_root=data_root)
        return cls(
            data_root=data_root,
            output_dir=output_dir,
            profiles=tuple(profiles),
            patches_per_combo=int(payload.get("patches_per_combo", 20)),
            strengths=tuple(payload.get("strengths") or DEFAULT_STRENGTHS),
            allowed_primitives=tuple(payload.get("allowed_primitives") or ()),
            excluded_primitives=tuple(payload.get("excluded_primitives") or ()),
            seed=int(payload.get("seed", 13)),
            max_masks_per_profile=(
                int(payload["max_masks_per_profile"])
                if payload.get("max_masks_per_profile") is not None
                else None
            ),
            early_stop_when_full=bool(payload.get("early_stop_when_full", True)),
        )


def build_benchmark_intents(config: BuildConfig) -> tuple[list[BenchmarkIntent], dict[str, Any]]:
    rng = random.Random(config.seed)
    grouped: dict[tuple[str, str, str], list[BenchmarkIntent]] = defaultdict(list)
    scan_summary: dict[str, Any] = {"profiles": {}, "shortfalls": []}
    allowed = set(config.allowed_primitives)
    excluded = set(config.excluded_primitives)

    for source in config.profiles:
        schema = MaskProfileSchema.from_reference_profile(source.profile)
        recipe = load_recipe(default_recipe_path_for_profile(source.profile))
        primitive_configs = list(_iter_primitive_configs(recipe, allowed=allowed, excluded=excluded))
        masks = list(_discover_masks(source.mask_globs, data_root=config.data_root))
        if config.max_masks_per_profile is not None:
            masks = masks[: config.max_masks_per_profile]
        profile_summary = {
            "mask_count": len(masks),
            "candidate_counts": defaultdict(int),
            "load_errors": [],
        }
        wanted_keys = {
            (source.organ, str(primitive_config.get("name")), strength)
            for primitive_config in primitive_configs
            for strength in _primitive_strengths(primitive_config)
            if strength in config.strengths
            if _primitive_possible_for_schema(
                primitive_config,
                strength=strength,
                profile=schema.reference_profile,
                recipe=recipe,
                schema=schema,
            )
        }
        for mask_path in masks:
            if config.early_stop_when_full and wanted_keys and _all_quotas_full(
                grouped,
                wanted_keys=wanted_keys,
                quota=config.patches_per_combo,
            ):
                profile_summary["stopped_early_after_masks"] = profile_summary.get(
                    "scanned_masks", 0
                )
                break
            profile_summary["scanned_masks"] = int(profile_summary.get("scanned_masks", 0)) + 1
            try:
                mask = load_id_mask(mask_path)
            except Exception as exc:
                profile_summary["load_errors"].append({"mask_path": str(mask_path), "error": str(exc)})
                continue
            context = _fast_context_from_mask(mask, schema)
            image_path = _match_image_path(mask_path, source.image_globs, data_root=config.data_root)
            for primitive_config in primitive_configs:
                primitive_name = str(primitive_config.get("name"))
                strengths = tuple(strength for strength in _primitive_strengths(primitive_config) if strength in config.strengths)
                for strength in strengths:
                    intent = _defaulted_intent(
                        primitive_name=primitive_name,
                        strength=strength,
                        profile=schema.reference_profile,
                        primitive_config=primitive_config,
                        schema=schema,
                    )
                    decision = assess_edit_applicability(intent, recipe, schema, context)
                    if decision.status == "rejected":
                        continue
                    feasibility = estimate_capacity(mask, intent, primitive_config, schema)
                    if feasibility.get("status") != "executable":
                        continue
                    region_hint = recommend_region_hint(mask, schema, intent, primitive_config)
                    if not region_hint:
                        continue
                    sample_seed = _stable_seed(config.seed, mask_path, primitive_name, strength, len(grouped))
                    metadata = {
                        "capacity": feasibility,
                        "applicability": decision.status,
                        "present_labels": sorted(context.present_labels),
                    }
                    anchor_labels = anchor_labels_for_primitive(primitive_config, schema)
                    if anchor_labels:
                        metadata["anchor_labels"] = list(anchor_labels)
                    gt = BenchmarkIntent(
                        sample_id=_sample_id(source.profile, primitive_name, strength, sample_seed),
                        organ=source.organ,
                        profile=schema.reference_profile,
                        image_path=str(image_path) if image_path is not None else None,
                        mask_path=str(mask_path),
                        primitive=primitive_name,
                        strength=strength,
                        region_hint=region_hint,
                        source_labels=tuple(intent.source_labels),
                        target_label=intent.target_label,
                        expected_direction=expected_direction_for_primitive(primitive_config),
                        expected_area_bucket=strength_interval(primitive_config, strength),
                        seed=sample_seed,
                        specialized=primitive_name in SPECIALIZED_NAMES,
                        metadata=metadata,
                    )
                    key = (source.organ, primitive_name, strength)
                    grouped[key].append(gt)
                    profile_summary["candidate_counts"]["|".join(key)] += 1
        profile_summary["candidate_counts"] = dict(profile_summary["candidate_counts"])
        scan_summary["profiles"][source.profile] = profile_summary

    selected: list[BenchmarkIntent] = []
    for key in sorted(grouped):
        candidates = list(grouped[key])
        rng.shuffle(candidates)
        quota = config.patches_per_combo
        selected.extend(candidates[:quota])
        if len(candidates) < quota:
            scan_summary["shortfalls"].append(
                {
                    "organ": key[0],
                    "primitive": key[1],
                    "strength": key[2],
                    "available": len(candidates),
                    "requested": quota,
                }
            )
    selected.sort(key=lambda item: (item.organ, item.primitive, item.strength, item.sample_id))
    return selected, _json_safe_summary(scan_summary)


def estimate_capacity(
    mask: np.ndarray,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
    schema: MaskProfileSchema,
) -> dict[str, Any]:
    interval = strength_interval(primitive_config, intent.strength)
    denominator = strength_denominator_pixels(mask, primitive_config, schema)
    legal_pixels = recommendation_legal_pixels(mask, primitive_config, schema)
    feasible_pixels = _feasible_legal_pixels(mask, primitive_config, schema)
    failed: list[str] = []
    if denominator <= 0:
        failed.append("capacity failed: no denominator pixels in current mask.")
    if legal_pixels <= 0:
        failed.append("capacity failed: no legal source pixels in current mask.")
    failed.extend(recommendation_dependency_failures(mask, primitive_config, schema))
    if interval is not None and denominator > 0:
        lower, upper = interval
        lower_pixels = int(np.ceil(denominator * lower))
        upper_pixels = int(np.floor(denominator * upper))
        lower_pixels = max(lower_pixels, _minimum_pixels_for_strength(primitive_config, intent.strength))
        if feasible_pixels < lower_pixels:
            failed.append(
                f"capacity failed: feasible_pixels={feasible_pixels} below {intent.strength} minimum {lower_pixels}."
            )
    else:
        lower_pixels = 1 if denominator > 0 else 0
        upper_pixels = legal_pixels
    target_pixels = (
        max(1, int(round((lower_pixels + max(upper_pixels, lower_pixels)) / 2)))
        if legal_pixels > 0
        else 0
    )
    achievable_pixels = min(target_pixels, feasible_pixels)
    fraction = achievable_pixels / denominator if denominator > 0 else None
    return {
        "status": "executable" if not failed else "capacity_failed",
        "validation_passed": not failed,
        "validation_failed_checks": failed,
        "changed_area_fraction": fraction,
        "strength_fraction": fraction,
        "strength_range": list(interval) if interval is not None else None,
        "selected_pixels": int(achievable_pixels),
        "legal_pixels": int(legal_pixels),
        "feasible_pixels": int(feasible_pixels),
        "denominator_pixels": int(denominator),
        "notes": ["static_mask_capacity_estimate_only"],
    }


def recommend_region_hint(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    intent: EditIntent,
    primitive_config: Mapping[str, Any],
) -> dict[str, Any]:
    legal = legal_pixel_mask(mask, primitive_config, schema)
    if not np.any(legal):
        return {}
    labels = region_labels_from_mask(mask, legal, schema)
    rows, cols = np.nonzero(legal)
    height, width = mask.shape
    centroid_y = float(np.mean(rows))
    centroid_x = float(np.mean(cols))
    quadrant = _quadrant(centroid_x, centroid_y, width=width, height=height)
    relation = _region_relation(centroid_x, centroid_y, width=width, height=height)
    bbox = [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]
    hint = {
        "type": "auto_recommended_mask_region",
        "location": quadrant,
        "relation": relation,
        "bbox_xyxy": bbox,
        "centroid_xy": [round(centroid_x, 2), round(centroid_y, 2)],
        "area_pixels": int(np.count_nonzero(legal)),
        "source_labels": labels,
        "description": f"{relation} {quadrant} editable region",
        "planner_note": (
            "Inject into EditIntent.region_hint; prompt text should describe this location."
        ),
    }
    anchor_labels = anchor_labels_for_primitive(primitive_config, schema)
    if anchor_labels:
        hint["anchor_labels"] = list(anchor_labels)
        hint["description"] = f"{relation} {quadrant} editable region adjacent to {'/'.join(anchor_labels)}"
    return hint


def inject_region_hint(intent: EditIntent, region_hint: Mapping[str, Any]) -> EditIntent:
    payload = intent.to_metadata()
    merged = dict(payload.get("region_hint") or {})
    merged.update(dict(region_hint))
    payload["region_hint"] = merged
    return EditIntent.from_mapping(payload)


def primitive_config_by_name(recipe: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for primitive in recipe.get("primitives", []):
        if isinstance(primitive, Mapping) and primitive.get("name") == name:
            return primitive
    raise KeyError(f"Unknown primitive: {name}")


def strength_interval(primitive_config: Mapping[str, Any], strength: str) -> tuple[float, float] | None:
    ranges = primitive_config.get("parameter_ranges", {})
    if not isinstance(ranges, Mapping):
        return None
    for key in (
        "target_changed_area_fraction",
        "target_area_delta_fraction",
        "target_area_decrease_fraction",
        "necrosis_area_decrease_fraction",
        "immune_area_delta_fraction",
        "immune_area_decrease_fraction",
        "stroma_area_delta_fraction",
        "stroma_area_decrease_fraction",
        "source_area_transition_fraction",
    ):
        value = ranges.get(key)
        if not isinstance(value, Mapping):
            continue
        interval = value.get(strength)
        if (
            isinstance(interval, list)
            and len(interval) == 2
            and all(isinstance(item, (int, float)) for item in interval)
        ):
            return float(interval[0]), float(interval[1])
    return None


def strength_denominator_pixels(mask: np.ndarray, primitive_config: Mapping[str, Any], schema: MaskProfileSchema) -> int:
    name = primitive_config.get("name")
    if name == "tumor_burden_increase":
        return int(mask.size)
    if name == "tumor_burden_decrease":
        return int(mask.size)
    if name == "tumor_burden_decrease_tumor_relative":
        return int(np.count_nonzero(np.isin(mask, schema.tumor_fine_ids)))
    if name in {"necrosis_appearance", "intratumoral_immune_infiltration"}:
        return int(np.count_nonzero(np.isin(mask, schema.tumor_fine_ids)))
    if name == "necrosis_resolution":
        return int(np.count_nonzero(safe_schema_label_mask(mask, schema, "Necrosis")))
    if name == "immune_infiltration_decrease":
        return int(np.count_nonzero(safe_schema_label_mask(mask, schema, "Immune infiltrate")))
    if name == "stromal_immune_infiltration":
        stroma = safe_schema_label_mask(mask, schema, "Stroma")
        immune = safe_schema_label_mask(mask, schema, "Immune infiltrate")
        return int(np.count_nonzero(stroma | immune))
    if name in {"stromal_desmoplasia", "stroma_decrease", "stromal_reduction"}:
        return int(np.count_nonzero(safe_schema_label_mask(mask, schema, "Stroma")))
    operation = primitive_config.get("mask_operation", {})
    if isinstance(operation, Mapping):
        source_ids = operation.get("source_fine_ids")
        if isinstance(source_ids, int):
            return int(np.count_nonzero(mask == source_ids))
        if isinstance(source_ids, (list, tuple)):
            return int(np.count_nonzero(np.isin(mask, tuple(source_ids))))
    return int(mask.size)


def recommendation_legal_pixels(mask: np.ndarray, primitive_config: Mapping[str, Any], schema: MaskProfileSchema) -> int:
    return int(np.count_nonzero(legal_pixel_mask(mask, primitive_config, schema)))


def legal_pixel_mask(mask: np.ndarray, primitive_config: Mapping[str, Any], schema: MaskProfileSchema) -> np.ndarray:
    name = primitive_config.get("name")
    operation = primitive_config.get("mask_operation", {})
    operation = operation if isinstance(operation, Mapping) else {}
    if name in {"necrosis_appearance", "intratumoral_immune_infiltration"}:
        target = operation.get("target")
        target_mask = safe_schema_label_mask(mask, schema, target) if isinstance(target, str) else np.zeros(mask.shape, dtype=bool)
        return np.isin(mask, schema.tumor_fine_ids) & ~target_mask
    if name == "tumor_burden_increase":
        if not np.any(np.isin(mask, schema.tumor_fine_ids)):
            return np.zeros(mask.shape, dtype=bool)
        legal = np.zeros(mask.shape, dtype=bool)
        for label in labels_from_operation(operation.get("target_priority")):
            legal |= safe_schema_label_mask(mask, schema, label)
        legal &= ~safe_schema_label_mask(mask, schema, "Necrosis")
        return legal
    if name in {"tumor_burden_decrease", "necrosis_resolution", "immune_infiltration_decrease", "stroma_decrease", "stromal_reduction"}:
        source = operation.get("source")
        if isinstance(source, str):
            return safe_schema_label_mask(mask, schema, source)
    if name == "stromal_immune_infiltration":
        return safe_schema_label_mask(mask, schema, "Stroma")
    if name == "stromal_desmoplasia":
        legal = np.zeros(mask.shape, dtype=bool)
        for label in [
            *labels_from_operation(operation.get("primary_sources")),
            *labels_from_operation(operation.get("secondary_sources")),
        ]:
            legal |= safe_schema_label_mask(mask, schema, label)
        return _desmoplasia_policy_legal_mask(mask, primitive_config, schema, legal)
    source_ids = operation.get("source_fine_ids")
    if isinstance(source_ids, int):
        return mask == source_ids
    if isinstance(source_ids, (list, tuple)):
        return np.isin(mask, tuple(source_ids))
    source = operation.get("source")
    if isinstance(source, str):
        return safe_schema_label_mask(mask, schema, source)
    return ~np.isin(mask, tuple(schema.skip_fine_ids))


def _feasible_legal_pixels(mask: np.ndarray, primitive_config: Mapping[str, Any], schema: MaskProfileSchema) -> int:
    legal = legal_pixel_mask(mask, primitive_config, schema)
    if primitive_config.get("name") != "stromal_desmoplasia":
        return int(np.count_nonzero(legal))
    ranges = primitive_config.get("parameter_ranges", {})
    spatial_pattern = primitive_config.get("spatial_pattern", {})
    constraints = spatial_pattern.get("immune_to_stroma_constraints", {}) if isinstance(spatial_pattern, Mapping) else {}
    constraints = constraints if isinstance(constraints, Mapping) else {}
    max_immune_fraction = float(constraints.get("max_fraction_of_total_desmoplasia_delta", 0.30))
    immune = safe_schema_label_mask(mask, schema, "Immune infiltrate") & legal
    immune_pixels = int(np.count_nonzero(immune))
    non_immune_pixels = int(np.count_nonzero(legal & ~immune))
    if max_immune_fraction <= 0:
        return non_immune_pixels
    if max_immune_fraction >= 1:
        return non_immune_pixels + immune_pixels
    feasible_total_by_ratio = int(np.floor(non_immune_pixels / (1.0 - max_immune_fraction)))
    return max(0, min(non_immune_pixels + immune_pixels, feasible_total_by_ratio))


def _minimum_pixels_for_strength(primitive_config: Mapping[str, Any], strength: str) -> int:
    if primitive_config.get("name") != "stromal_desmoplasia":
        return 0
    ranges = primitive_config.get("parameter_ranges", {})
    floor = ranges.get("min_stroma_area_delta_pixels", {}) if isinstance(ranges, Mapping) else {}
    if isinstance(floor, Mapping):
        value = floor.get(strength, 0)
    else:
        value = floor
    return int(value) if isinstance(value, (int, float)) and value > 0 else 0


def _desmoplasia_policy_legal_mask(
    mask: np.ndarray,
    primitive_config: Mapping[str, Any],
    schema: MaskProfileSchema,
    base_legal: np.ndarray,
) -> np.ndarray:
    ranges = primitive_config.get("parameter_ranges", {})
    tumor = np.isin(mask, schema.tumor_fine_ids)
    stroma = safe_schema_label_mask(mask, schema, "Stroma")
    immune = safe_schema_label_mask(mask, schema, "Immune infiltrate")
    max_distance = float(ranges.get("max_distance_from_tumor_px", 64.0)) if isinstance(ranges, Mapping) else 64.0
    dist_to_tumor = ndimage.distance_transform_edt(~tumor)
    legal = np.asarray(base_legal, dtype=bool) & (dist_to_tumor <= max_distance) & ~tumor
    spatial_pattern = primitive_config.get("spatial_pattern", {})
    constraints = spatial_pattern.get("immune_to_stroma_constraints", {}) if isinstance(spatial_pattern, Mapping) else {}
    constraints = constraints if isinstance(constraints, Mapping) else {}
    if bool(constraints.get("require_direct_stroma_adjacency", True)) and np.any(immune):
        stroma_neighbors = ndimage.binary_dilation(stroma, structure=np.ones((3, 3), dtype=bool))
        legal &= (~immune) | stroma_neighbors
    return legal


def recommendation_dependency_failures(mask: np.ndarray, primitive_config: Mapping[str, Any], schema: MaskProfileSchema) -> list[str]:
    failures: list[str] = []
    for kind, labels in recommendation_required_context_labels(primitive_config, schema):
        if not labels:
            failures.append(f"capacity failed: no configured {kind} labels.")
            continue
        if schema_labels_pixel_count(mask, schema, labels) > 0:
            continue
        failures.append(f"capacity failed: no {kind} pixels in current mask ({', '.join(labels)}).")
    return failures


def recommendation_required_context_labels(primitive_config: Mapping[str, Any], schema: MaskProfileSchema) -> tuple[tuple[str, tuple[str, ...]], ...]:
    operation = primitive_config.get("mask_operation", {})
    operation = operation if isinstance(operation, Mapping) else {}
    required_context = primitive_config.get("required_context", ())
    required_context = required_context if isinstance(required_context, list) else ()
    required: list[tuple[str, tuple[str, ...]]] = []
    name = primitive_config.get("name")
    if name == "tumor_burden_increase":
        labels = tuple(filter_schema_labels(labels_from_operation(operation.get("target_priority")), schema))
        required.append(("target tissue", labels))
    if "valid_backfill_tissue" in required_context:
        labels = tuple(filter_schema_labels(labels_from_operation(operation.get("backfill_priority")), schema))
        required.append(("backfill tissue", labels))
    return tuple(required)


def expected_direction_for_primitive(primitive_config: Mapping[str, Any]) -> str:
    name = str(primitive_config.get("name"))
    if name in {"tumor_burden_increase", "necrosis_appearance", "stromal_immune_infiltration", "intratumoral_immune_infiltration", "stromal_desmoplasia"}:
        return "increase"
    if name in {"tumor_burden_decrease", "necrosis_resolution", "immune_infiltration_decrease", "stroma_decrease", "stromal_reduction"}:
        return "decrease"
    operation = primitive_config.get("mask_operation", {})
    if isinstance(operation, Mapping) and operation.get("type") == "fine_label_transition":
        return "transition"
    return "change"


def source_target_labels_for_primitive(primitive_config: Mapping[str, Any], schema: MaskProfileSchema) -> tuple[tuple[str, ...], str | None]:
    operation = primitive_config.get("mask_operation", {})
    operation = operation if isinstance(operation, Mapping) else {}
    if primitive_config.get("name") == "tumor_burden_increase":
        source = tuple(filter_schema_labels(labels_from_operation(operation.get("target_priority")), schema))
        target = "Tumor" if "Tumor" in schema.readable_labels else None
        return source, target
    if primitive_config.get("name") == "stromal_desmoplasia":
        source = tuple(
            filter_schema_labels(
                [
                    *labels_from_operation(operation.get("primary_sources")),
                    *labels_from_operation(operation.get("secondary_sources")),
                ],
                schema,
            )
        )
        target = operation.get("target") if isinstance(operation.get("target"), str) else None
        return source, target if target in schema.readable_labels else None
    source = tuple(filter_schema_labels(labels_from_operation(operation.get("source")), schema))
    if not source:
        source = tuple(filter_schema_labels(labels_from_operation(operation.get("primary_sources")), schema))
    if not source:
        source = tuple(filter_schema_labels(labels_from_operation(operation.get("target_priority")), schema))
    target = operation.get("target") if isinstance(operation.get("target"), str) else None
    if target not in schema.readable_labels:
        target = "Tumor" if primitive_config.get("name") == "tumor_burden_increase" and "Tumor" in schema.readable_labels else None
    return source, target


def anchor_labels_for_primitive(primitive_config: Mapping[str, Any], schema: MaskProfileSchema) -> tuple[str, ...]:
    operation = primitive_config.get("mask_operation", {})
    operation = operation if isinstance(operation, Mapping) else {}
    if primitive_config.get("name") == "tumor_burden_increase":
        anchors = tuple(filter_schema_labels(labels_from_operation(operation.get("source")), schema))
        return anchors or (("Tumor",) if "Tumor" in schema.readable_labels else ())
    return ()


def safe_schema_label_mask(mask: np.ndarray, schema: MaskProfileSchema, label: str | None) -> np.ndarray:
    if not isinstance(label, str) or label not in schema.readable_labels:
        return np.zeros(mask.shape, dtype=bool)
    return np.isin(mask, schema.resolve_fine_ids(label))


def schema_labels_pixel_count(mask: np.ndarray, schema: MaskProfileSchema, labels: Sequence[str]) -> int:
    return sum(int(np.count_nonzero(safe_schema_label_mask(mask, schema, label))) for label in labels)


def labels_from_operation(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def filter_schema_labels(labels: Sequence[str], schema: MaskProfileSchema) -> list[str]:
    return [label for label in labels if label in schema.readable_labels]


def region_labels_from_mask(mask: np.ndarray, region: np.ndarray, schema: MaskProfileSchema) -> list[str]:
    labels: list[str] = []
    for label in sorted(schema.readable_labels):
        if np.any(region & safe_schema_label_mask(mask, schema, label)):
            labels.append(label)
    return labels


def _iter_primitive_configs(recipe: Mapping[str, Any], *, allowed: set[str], excluded: set[str]) -> Iterable[Mapping[str, Any]]:
    composite_names = {
        str(item.get("name"))
        for item in recipe.get("composite_recipes", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    for primitive in recipe.get("primitives", []):
        if not isinstance(primitive, Mapping) or not isinstance(primitive.get("name"), str):
            continue
        name = str(primitive["name"])
        if name in composite_names or name in excluded:
            continue
        if allowed and name not in allowed:
            continue
        yield primitive


def _all_quotas_full(
    grouped: Mapping[tuple[str, str, str], list[BenchmarkIntent]],
    *,
    wanted_keys: set[tuple[str, str, str]],
    quota: int,
) -> bool:
    return all(len(grouped.get(key, ())) >= quota for key in wanted_keys)


def _defaulted_intent(*, primitive_name: str, strength: str, profile: str, primitive_config: Mapping[str, Any], schema: MaskProfileSchema) -> EditIntent:
    source, target = source_target_labels_for_primitive(primitive_config, schema)
    if not source:
        required = primitive_config.get("required_tissue_labels", ())
        if isinstance(required, list):
            source = tuple(filter_schema_labels([str(item) for item in required], schema))
    if not target:
        try:
            target = schema.choose_default_backfill_label(exclude_labels=source)
        except Exception:
            target = None
    return EditIntent(
        primitive=primitive_name,
        strength=strength,
        reference_profile=profile,
        source_labels=source,
        target_label=target,
    )


def _primitive_possible_for_schema(
    primitive_config: Mapping[str, Any],
    *,
    strength: str,
    profile: str,
    recipe: Mapping[str, Any],
    schema: MaskProfileSchema,
) -> bool:
    required = primitive_config.get("required_tissue_labels", ())
    if isinstance(required, list):
        for label in required:
            if not isinstance(label, str):
                continue
            if label not in schema.readable_labels or label not in schema.writable_labels:
                return False
    operation = primitive_config.get("mask_operation", {})
    operation = operation if isinstance(operation, Mapping) else {}
    for key in ("source", "target"):
        label = operation.get(key)
        if isinstance(label, str) and label not in schema.readable_labels:
            return False
    for key in ("target_priority", "backfill_priority", "primary_sources", "secondary_sources"):
        labels = labels_from_operation(operation.get(key))
        if labels and not any(label in schema.readable_labels for label in labels):
            return False
    try:
        validate_intent_against_recipe(
            _defaulted_intent(
                primitive_name=str(primitive_config["name"]),
                strength=strength,
                profile=profile,
                primitive_config=primitive_config,
                schema=schema,
            ),
            recipe,
        )
    except (IntentValidationError, Exception):
        return False
    return True


def _primitive_strengths(primitive_config: Mapping[str, Any]) -> tuple[str, ...]:
    ranges = primitive_config.get("parameter_ranges", {})
    if not isinstance(ranges, Mapping):
        return ()
    strengths: list[str] = []
    for key in DEFAULT_STRENGTHS:
        if key in ranges or any(isinstance(value, Mapping) and key in value for value in ranges.values()):
            strengths.append(key)
    return tuple(strengths)


def _discover_masks(mask_globs: Sequence[str], *, data_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in mask_globs:
        root_pattern = str(data_root / pattern) if not Path(pattern).is_absolute() else pattern
        paths.update(_absolute_glob(root_pattern))
    return sorted(path for path in paths if path.is_file())


def _absolute_glob(pattern: str) -> Iterable[Path]:
    import glob

    return (Path(path) for path in glob.glob(pattern, recursive=True))


def _match_image_path(mask_path: Path, image_globs: Sequence[str], *, data_root: Path) -> Path | None:
    if not image_globs:
        return None
    stem = mask_path.stem
    for pattern in image_globs:
        root_pattern = str(data_root / pattern) if not Path(pattern).is_absolute() else pattern
        for candidate in _absolute_glob(root_pattern):
            if candidate.stem == stem:
                return candidate
    return None


def _profile_sources(payload: Any, *, data_root: Path) -> list[ProfileSource]:
    if not payload:
        return [
            ProfileSource(
                organ=organ,
                profile=profile,
                mask_globs=(f"**/{profile}/**/*mask*.png", f"**/{profile.lower()}/**/*mask*.png"),
                image_globs=(f"**/{profile}/**/*.png", f"**/{profile.lower()}/**/*.png"),
            )
            for organ, profile in DEFAULT_PROFILES.items()
        ]
    sources: list[ProfileSource] = []
    if isinstance(payload, Mapping):
        iterable = payload.items()
        for organ, value in iterable:
            if isinstance(value, str):
                sources.append(ProfileSource(str(organ), value, (f"**/{value}/**/*mask*.png",), ()))
            elif isinstance(value, Mapping):
                sources.append(
                    ProfileSource(
                        organ=str(value.get("organ") or organ),
                        profile=str(value.get("profile") or organ),
                        mask_globs=tuple(value.get("mask_globs") or value.get("masks") or ()),
                        image_globs=tuple(value.get("image_globs") or value.get("images") or ()),
                    )
                )
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            sources.append(
                ProfileSource(
                    organ=str(item.get("organ") or item.get("profile")),
                    profile=str(item["profile"]),
                    mask_globs=tuple(item.get("mask_globs") or item.get("masks") or ()),
                    image_globs=tuple(item.get("image_globs") or item.get("images") or ()),
                )
            )
    for source in sources:
        if not source.mask_globs:
            raise ValueError(f"Profile {source.profile} has no mask_globs in config rooted at {data_root}.")
    return sources


def _fast_context_from_mask(mask: np.ndarray, schema: MaskProfileSchema) -> MaskEditContext:
    """Build the subset of MaskEditContext needed for benchmark feasibility.

    The full context computes Python connected components for every present
    label, which is useful for executor planning but too slow when scanning
    tens of thousands of benchmark candidate masks. Applicability gates used
    here only need present labels, normalized mask, and risk flags.
    """

    normalized = np.asarray(mask)
    total = int(normalized.size) or 1
    label_counts: dict[str, int] = {}
    fine_fractions: dict[int, float] = {}
    for fine_id, count in zip(*np.unique(normalized, return_counts=True)):
        fine_id_int = int(fine_id)
        count_int = int(count)
        fine_fractions[fine_id_int] = count_int / total
        if fine_id_int in schema.skip_fine_ids:
            continue
        for label, fine_ids in schema.label_to_fine_ids.items():
            if fine_id_int in fine_ids:
                label_counts[label] = label_counts.get(label, 0) + count_int
                break
    return MaskEditContext(
        reference_profile=schema.reference_profile,
        mask_shape=tuple(int(dim) for dim in normalized.shape),
        present_labels=frozenset(label_counts),
        label_area_fractions={label: count / total for label, count in label_counts.items()},
        fine_id_area_fractions=fine_fractions,
        adjacency={},
        component_counts={},
        normalized_mask=normalized,
        risk_flags=(),
        semantic_warnings=dict(schema.semantic_warnings),
    )


def _resolve_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _quadrant(x: float, y: float, *, width: int, height: int) -> str:
    vertical = "upper" if y < height / 3 else "lower" if y > 2 * height / 3 else "central"
    horizontal = "left" if x < width / 3 else "right" if x > 2 * width / 3 else "central"
    if vertical == "central" and horizontal == "central":
        return "center"
    if vertical == "central":
        return horizontal
    if horizontal == "central":
        return vertical
    return f"{vertical}_{horizontal}"


def _region_relation(x: float, y: float, *, width: int, height: int) -> str:
    margin = min(width, height) * 0.2
    if x < margin or y < margin or x > width - margin or y > height - margin:
        return "peripheral"
    return "central"


def _sample_id(profile: str, primitive: str, strength: str, seed: int) -> str:
    return f"{profile}_{primitive}_{strength}_{seed:08x}"


def _stable_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.sha1("|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _json_safe_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(summary, ensure_ascii=False))
