"""Explicit authority contracts for GLaS gland and nucleus observations.

This module separates three concepts that were previously conflated:

* a dataset-native gland instance annotation;
* a deterministic connected-component proxy derived from the transformed
  semantic tissue mask; and
* a hybrid nucleus partition that combines trusted CellViT seeds with
  semantic residual coverage.

The helpers are deliberately independent of H&E and of Planner decisions.
They only classify and validate provenance already present in masks, instance
records, and immutable digests.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

GLAND_AUTHORITY_DATASET_NATIVE = "dataset_native_instance"
GLAND_AUTHORITY_SEMANTIC_PROXY = "semantic_connected_component_proxy"
GLAND_AUTHORITY_UNRESOLVED = "unresolved"

NUCLEUS_AUTHORITY_NATIVE = "native_instance"
NUCLEUS_AUTHORITY_HYBRID = "hybrid_cellvit_seeded_semantic_partition"
NUCLEUS_AUTHORITY_SEMANTIC_RASTER = "semantic_raster_partition"
NUCLEUS_AUTHORITY_SEMANTIC_WATERSHED = "semantic_distance_watershed"

_ALLOWED_GLAND_AUTHORITY_KINDS = frozenset(
    {
        GLAND_AUTHORITY_DATASET_NATIVE,
        GLAND_AUTHORITY_SEMANTIC_PROXY,
    }
)
_NATIVE_NUCLEUS_SOURCES = frozenset(
    {
        "instance_json",
        "instance_json_cellvit_seed",
        "dataset_native_instance",
    }
)
_SEMANTIC_RESIDUAL_SOURCES = frozenset(
    {
        "instance_json_semantic_fallback",
        "instance_json_semantic_unseeded",
        "instance_json_semantic_seeded_residual",
    }
)
_SEMANTIC_WATERSHED_SOURCES = frozenset(
    {
        "semantic_component",
        "semantic_distance_watershed",
    }
)


def semantic_gland_component_authority_metadata(
    *,
    source_tissue_mask_sha256: str,
    output_sha256: str,
) -> dict[str, Any]:
    """Return the narrow claim owned by a semantic gland-component proxy."""

    _require_digest(source_tissue_mask_sha256, "source_tissue_mask_sha256")
    _require_digest(output_sha256, "output_sha256")
    return {
        "authority_kind": GLAND_AUTHORITY_SEMANTIC_PROXY,
        "authority_scope": "gland_support_component_exterior_topology",
        "authority_claims": [
            "digest_bound_semantic_gland_support_components",
            "component_exterior_distance_field",
        ],
        "authority_excludes": [
            "dataset_native_gland_instance_identity",
            "original_gland_instance_annotation_digest",
        ],
        "derived_from_tissue_mask_sha256": source_tissue_mask_sha256,
        "derived_gland_component_map_sha256": output_sha256,
    }


def bind_gland_instance_authority_provenance(
    provenance: Mapping[str, Any],
    *,
    structure_id: str = "native_gland_instance_map",
) -> dict[str, Any]:
    """Bind one typed gland authority without overwriting nucleus provenance.

    ``original_instance_mask_digest`` is retained as the legacy nucleus-instance
    authority used throughout the joint pipeline. Gland-instance identity owns
    the dedicated ``original_gland_instance_mask_digest`` field. Older manifests
    that used the generic field for a gland raster remain readable only when that
    field is absent from, or already equal to, the gland authority digest.
    """

    result = dict(provenance)
    raw_records = result.get("auxiliary_structure_provenance")
    raw_digests = result.get("auxiliary_structure_sha256")
    if not isinstance(raw_records, Mapping) or not isinstance(raw_digests, Mapping):
        return result
    if structure_id not in raw_records or structure_id not in raw_digests:
        return result

    records = {str(key): dict(value) for key, value in raw_records.items()}
    record = records.get(structure_id, {})
    output_digest = str(raw_digests.get(structure_id) or "")
    source_digest = str(result.get("source_tissue_mask_sha256") or "")
    generic_original_digest = str(
        result.get("original_instance_mask_digest") or ""
    )
    dedicated_gland_digest = str(
        result.get("original_gland_instance_mask_digest") or ""
    )

    kind = _infer_gland_authority_kind(record)
    if kind == GLAND_AUTHORITY_UNRESOLVED and (
        dedicated_gland_digest == output_digest
        or generic_original_digest == output_digest
    ):
        # Backward-compatible explicit native assertion. Generated semantic
        # proxies are recognized above from their producer/observation scope
        # and therefore can never enter this branch.
        kind = GLAND_AUTHORITY_DATASET_NATIVE
    record["authority_kind"] = kind
    record["authority_digest_field"] = (
        "original_gland_instance_mask_digest"
        if kind == GLAND_AUTHORITY_DATASET_NATIVE
        else (
            "derived_gland_component_map_sha256"
            if kind == GLAND_AUTHORITY_SEMANTIC_PROXY
            else "unresolved"
        )
    )
    records[structure_id] = record

    result["auxiliary_structure_provenance"] = records
    result["gland_instance_authority_kind"] = kind
    result["gland_instance_authority_structure_id"] = structure_id
    result["gland_instance_authority_sha256"] = output_digest

    repairs = list(result.get("authority_provenance_repairs") or ())
    if kind == GLAND_AUTHORITY_SEMANTIC_PROXY:
        result["derived_gland_component_map_sha256"] = output_digest
        result["derived_gland_component_source_tissue_mask_sha256"] = source_digest
        result["original_gland_instance_mask_available_for_execution"] = False
        result["gland_instance_authority_available_for_execution"] = True
        if dedicated_gland_digest == output_digest:
            result.pop("original_gland_instance_mask_digest", None)
            repairs.append(
                "removed_proxy_digest_from_original_gland_instance_field"
            )
        if generic_original_digest == output_digest:
            result.pop("original_instance_mask_digest", None)
            result["original_instance_mask_available_for_execution"] = False
            repairs.append("removed_proxy_digest_from_original_instance_field")
    elif kind == GLAND_AUTHORITY_DATASET_NATIVE:
        result["original_gland_instance_mask_digest"] = output_digest
        result["original_gland_instance_mask_available_for_execution"] = True
        result["gland_instance_authority_available_for_execution"] = True
        # Preserve the historical alias only when it does not already own a
        # different nucleus-instance JSON/raster digest.
        if not generic_original_digest or generic_original_digest == output_digest:
            result["original_instance_mask_digest"] = output_digest
            result["original_instance_mask_available_for_execution"] = True
        else:
            repairs.append("preserved_distinct_nucleus_instance_digest")
        result.pop("derived_gland_component_map_sha256", None)
        result.pop("derived_gland_component_source_tissue_mask_sha256", None)
    else:
        result["original_gland_instance_mask_available_for_execution"] = False
        result["gland_instance_authority_available_for_execution"] = False

    if repairs:
        result["authority_provenance_repairs"] = list(dict.fromkeys(repairs))
    return result


def gland_instance_authority_status(
    provenance: Mapping[str, Any],
    *,
    structure_id: str = "native_gland_instance_map",
) -> dict[str, Any]:
    """Validate typed gland authority and its dedicated digest namespace."""

    records = provenance.get("auxiliary_structure_provenance")
    digests = provenance.get("auxiliary_structure_sha256")
    record = records.get(structure_id) if isinstance(records, Mapping) else None
    output_digest = (
        str(digests.get(structure_id) or "")
        if isinstance(digests, Mapping)
        else ""
    )
    kind = str(provenance.get("gland_instance_authority_kind") or "")
    record_kind = (
        str(record.get("authority_kind") or "")
        if isinstance(record, Mapping)
        else ""
    )
    source_digest = str(provenance.get("source_tissue_mask_sha256") or "")
    record_source_digest = (
        str(record.get("source_tissue_mask_sha256") or "")
        if isinstance(record, Mapping)
        else ""
    )
    record_output_digest = (
        str(record.get("output_sha256") or "")
        if isinstance(record, Mapping)
        else ""
    )

    violations: list[str] = []
    if kind not in _ALLOWED_GLAND_AUTHORITY_KINDS:
        violations.append("unsupported_or_missing_gland_authority_kind")
    if not isinstance(record, Mapping):
        violations.append("gland_authority_record_missing")
    if record_kind != kind:
        violations.append("gland_authority_kind_mismatch")
    if not output_digest or record_output_digest != output_digest:
        violations.append("gland_authority_output_digest_mismatch")
    if provenance.get("gland_instance_authority_sha256") != output_digest:
        violations.append("top_level_gland_authority_digest_mismatch")
    if not source_digest or record_source_digest != source_digest:
        violations.append("gland_authority_source_digest_mismatch")

    if kind == GLAND_AUTHORITY_SEMANTIC_PROXY:
        if provenance.get("derived_gland_component_map_sha256") != output_digest:
            violations.append("semantic_proxy_digest_not_bound")
        if (
            provenance.get("derived_gland_component_source_tissue_mask_sha256")
            != source_digest
        ):
            violations.append("semantic_proxy_source_not_bound")
        if provenance.get("original_instance_mask_digest") == output_digest:
            violations.append("semantic_proxy_masquerades_as_original_instance")
        if provenance.get("original_gland_instance_mask_digest") == output_digest:
            violations.append(
                "semantic_proxy_masquerades_as_original_gland_instance"
            )
        if (
            provenance.get("original_gland_instance_mask_available_for_execution")
            is True
        ):
            violations.append("semantic_proxy_claims_native_gland_availability")
        if provenance.get("gland_instance_authority_available_for_execution") is not True:
            violations.append("semantic_proxy_not_execution_available")
    elif kind == GLAND_AUTHORITY_DATASET_NATIVE:
        if provenance.get("original_gland_instance_mask_digest") != output_digest:
            violations.append("dataset_native_gland_instance_digest_not_bound")
        if (
            provenance.get("original_gland_instance_mask_available_for_execution")
            is not True
        ):
            violations.append("dataset_native_gland_instance_not_execution_available")
        if provenance.get("gland_instance_authority_available_for_execution") is not True:
            violations.append("dataset_native_gland_authority_not_execution_available")

    return {
        "valid": not violations,
        "authority_kind": kind or GLAND_AUTHORITY_UNRESOLVED,
        "structure_id": structure_id,
        "output_sha256": output_digest or None,
        "source_tissue_mask_sha256": source_digest or None,
        "digest_field": (
            "original_gland_instance_mask_digest"
            if kind == GLAND_AUTHORITY_DATASET_NATIVE
            else (
                "derived_gland_component_map_sha256"
                if kind == GLAND_AUTHORITY_SEMANTIC_PROXY
                else None
            )
        ),
        "claim_scope": (
            "dataset_native_gland_instance_identity"
            if kind == GLAND_AUTHORITY_DATASET_NATIVE
            else (
                "semantic_gland_component_exterior_proxy"
                if kind == GLAND_AUTHORITY_SEMANTIC_PROXY
                else "unresolved"
            )
        ),
        "violations": violations,
    }


def summarize_nucleus_instance_authority(
    instances: Iterable[Any],
) -> dict[str, Any]:
    """Summarize native seeds and semantic residual coverage independently."""

    total_count = 0
    total_pixels = 0
    native_count = 0
    native_pixels = 0
    residual_count = 0
    residual_pixels = 0
    watershed_count = 0
    source_counts: Counter[str] = Counter()
    native_by_class: Counter[int] = Counter()
    native_complete_by_class: Counter[int] = Counter()
    residual_by_class: Counter[int] = Counter()

    for raw in instances:
        record = _normalize_nucleus_record(raw)
        total_count += 1
        total_pixels += record["area_px"]
        source = record["source"]
        class_id = record["class_id"]
        source_counts[source] += 1
        if source in _NATIVE_NUCLEUS_SOURCES:
            native_count += 1
            native_pixels += record["area_px"]
            native_by_class[class_id] += 1
            if record["complete_reference_eligible"]:
                native_complete_by_class[class_id] += 1
        elif source in _SEMANTIC_RESIDUAL_SOURCES:
            residual_count += 1
            residual_pixels += record["area_px"]
            residual_by_class[class_id] += 1
        else:
            watershed_count += 1

    if native_count and residual_count:
        quality = NUCLEUS_AUTHORITY_HYBRID
    elif native_count:
        quality = NUCLEUS_AUTHORITY_NATIVE
    elif residual_count:
        quality = NUCLEUS_AUTHORITY_SEMANTIC_RASTER
    else:
        quality = NUCLEUS_AUTHORITY_SEMANTIC_WATERSHED

    return {
        "observation_quality": quality,
        "total_instance_count": total_count,
        "total_instance_pixels": total_pixels,
        "native_seed_instance_count": native_count,
        "native_seed_pixels": native_pixels,
        "semantic_residual_instance_count": residual_count,
        "semantic_residual_pixels": residual_pixels,
        "semantic_watershed_instance_count": watershed_count,
        "native_seed_instance_fraction": native_count / max(1, total_count),
        "native_seed_pixel_fraction": native_pixels / max(1, total_pixels),
        "native_seed_count_by_class": _string_key_counts(native_by_class),
        "native_complete_reference_count_by_class": _string_key_counts(
            native_complete_by_class
        ),
        "semantic_residual_count_by_class": _string_key_counts(residual_by_class),
        "source_counts": dict(sorted(source_counts.items())),
    }



def count_authoritative_complete_references(
    instances: Iterable[Any],
    *,
    allowed_cell_classes: Iterable[int],
    allow_semantic_instance_fallback: bool,
    library_reference_counts: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Count morphology authorities without promoting semantic residuals.

    A hybrid raster may use its trusted native seeds plus a calibrated,
    digest-bound library. Semantic residual partitions remain coverage and
    population-accounting objects only. A pure distance-watershed scene keeps
    the historical semantic fallback behavior when the mechanism explicitly
    allows it.
    """

    normalized = tuple(_normalize_nucleus_record(item) for item in instances)
    summary = summarize_nucleus_instance_authority(instances)
    quality = summary["observation_quality"]
    allowed = {int(value) for value in allowed_cell_classes}
    same_patch_by_class: Counter[int] = Counter()

    for item in normalized:
        if item["class_id"] not in allowed or not item["complete_reference_eligible"]:
            continue
        source = item["source"]
        if quality in {NUCLEUS_AUTHORITY_NATIVE, NUCLEUS_AUTHORITY_HYBRID}:
            accepted = source in _NATIVE_NUCLEUS_SOURCES
        elif quality == NUCLEUS_AUTHORITY_SEMANTIC_WATERSHED:
            accepted = bool(
                allow_semantic_instance_fallback
                and source in _SEMANTIC_WATERSHED_SOURCES
            )
        else:
            # Raster residual partitions are not complete-morphology authority,
            # even when semantic fallback is allowed for population accounting.
            accepted = False
        if accepted:
            same_patch_by_class[item["class_id"]] += 1

    library_by_class = Counter(
        {
            int(class_id): max(0, int(count))
            for class_id, count in (library_reference_counts or {}).items()
            if int(class_id) in allowed and int(count) > 0
        }
    )
    total_by_class = Counter(same_patch_by_class)
    total_by_class.update(library_by_class)
    return {
        "observation_quality": quality,
        "allowed_cell_classes": sorted(allowed),
        "allow_semantic_instance_fallback": bool(
            allow_semantic_instance_fallback
        ),
        "same_patch_reference_count": int(sum(same_patch_by_class.values())),
        "same_patch_reference_count_by_class": _string_key_counts(
            same_patch_by_class
        ),
        "library_reference_count": int(sum(library_by_class.values())),
        "library_reference_count_by_class": _string_key_counts(
            library_by_class
        ),
        "total_reference_count": int(sum(total_by_class.values())),
        "total_reference_count_by_class": _string_key_counts(total_by_class),
        "semantic_residual_role": "coverage_and_accounting_only",
    }


def validate_mechanism_nucleus_authority(
    instances: Iterable[Any],
    *,
    allow_semantic_instance_fallback: bool,
    required_cell_classes: Iterable[int],
    actions: Iterable[str],
    minimum_native_per_class: int = 1,
) -> dict[str, Any]:
    """Decide whether one mechanism may consume the observed instance graph.

    A hybrid graph is acceptable to a no-fallback *add-only* mechanism when
    every required class has at least one complete native reference.  Semantic
    residuals remain accounting coverage only.  A no-fallback removal program
    cannot use a hybrid partition because residual instance identity would
    become destructive authority.
    """

    summary = summarize_nucleus_instance_authority(instances)
    quality = summary["observation_quality"]
    required = tuple(sorted({int(value) for value in required_cell_classes}))
    actions_set = {str(value) for value in actions}
    counts = summary["native_complete_reference_count_by_class"]
    missing_classes = [
        class_id
        for class_id in required
        if int(counts.get(str(class_id), 0)) < int(minimum_native_per_class)
    ]
    reasons: list[str] = []

    if allow_semantic_instance_fallback:
        passed = True
    elif quality == NUCLEUS_AUTHORITY_NATIVE:
        passed = not missing_classes
    elif quality == NUCLEUS_AUTHORITY_HYBRID:
        if "remove_whole" in actions_set:
            reasons.append("hybrid_partition_cannot_authorize_removal")
        if missing_classes:
            reasons.append("required_native_reference_class_missing")
        passed = not reasons
    else:
        reasons.append("native_nucleus_instance_authority_missing")
        if missing_classes:
            reasons.append("required_native_reference_class_missing")
        passed = False

    return {
        **summary,
        "passed": bool(passed),
        "allow_semantic_instance_fallback": bool(
            allow_semantic_instance_fallback
        ),
        "required_cell_classes": list(required),
        "minimum_native_per_class": int(minimum_native_per_class),
        "missing_native_reference_classes": missing_classes,
        "actions": sorted(actions_set),
        "semantic_residual_role": "coverage_and_accounting_only",
        "reasons": list(dict.fromkeys(reasons)),
    }


def validate_nucleus_authority_floor(
    summary: Mapping[str, Any],
    *,
    minimum_native_seed_count: int,
    minimum_native_seed_instance_fraction: float,
    minimum_native_seed_pixel_fraction: float,
    minimum_native_references_by_class: Mapping[int, int] | None = None,
) -> tuple[str, ...]:
    """Return fail-closed authority-floor violations for evaluation runners."""

    reasons: list[str] = []
    if int(summary.get("native_seed_instance_count", 0)) < int(
        minimum_native_seed_count
    ):
        reasons.append("native_seed_count_below_threshold")
    if float(summary.get("native_seed_instance_fraction", 0.0)) < float(
        minimum_native_seed_instance_fraction
    ):
        reasons.append("native_seed_instance_fraction_below_threshold")
    if float(summary.get("native_seed_pixel_fraction", 0.0)) < float(
        minimum_native_seed_pixel_fraction
    ):
        reasons.append("native_seed_pixel_fraction_below_threshold")

    counts = summary.get("native_complete_reference_count_by_class", {})
    counts = counts if isinstance(counts, Mapping) else {}
    for class_id, minimum in sorted(
        (minimum_native_references_by_class or {}).items()
    ):
        if int(counts.get(str(int(class_id)), 0)) < int(minimum):
            reasons.append(
                f"native_complete_class_{int(class_id)}_reference_count_below_threshold"
            )
    return tuple(reasons)


def _infer_gland_authority_kind(record: Mapping[str, Any]) -> str:
    explicit = str(record.get("authority_kind") or "")
    if explicit in _ALLOWED_GLAND_AUTHORITY_KINDS:
        return explicit
    observation_scope = str(record.get("observation_scope") or "")
    producer_id = str(record.get("producer_id") or "")
    if observation_scope == "semantic_fine_mask_topology_only" or producer_id.startswith(
        "joint-semantic-topology-auxiliary"
    ):
        return GLAND_AUTHORITY_SEMANTIC_PROXY
    if observation_scope in {
        "dataset_native_annotation",
        "dataset_native_gland_instance_annotation",
        "native_annotation",
        "native_instance",
    }:
        # This classifier is called only for the named gland-instance
        # auxiliary, so legacy ``native_instance`` here denotes gland identity,
        # not the independently digested nucleus-instance source.
        return GLAND_AUTHORITY_DATASET_NATIVE
    return GLAND_AUTHORITY_UNRESOLVED


def _normalize_nucleus_record(raw: Any) -> dict[str, Any]:
    if isinstance(raw, tuple) and len(raw) == 3:
        instance_id, class_id, component = raw
        region = np.asarray(component, dtype=bool)
        source = _source_from_instance_id(str(instance_id))
        touches_border = bool(
            np.any(region[0])
            or np.any(region[-1])
            or np.any(region[:, 0])
            or np.any(region[:, -1])
        )
        return {
            "instance_id": str(instance_id),
            "class_id": int(class_id),
            "source": source,
            "area_px": int(np.count_nonzero(region)),
            "complete_reference_eligible": bool(
                np.any(region) and not touches_border
            ),
        }

    instance_id = str(getattr(raw, "instance_id", ""))
    class_id = int(getattr(raw, "class_id", 0))
    source = str(getattr(raw, "source", "") or _source_from_instance_id(instance_id))
    area_px = max(0, int(getattr(raw, "area_px", 0)))
    touches_border = bool(getattr(raw, "touches_border", False))
    completeness = str(getattr(raw, "completeness_status", "complete"))
    quality_flags = tuple(getattr(raw, "quality_flags", ()) or ())
    return {
        "instance_id": instance_id,
        "class_id": class_id,
        "source": source,
        "area_px": area_px,
        "complete_reference_eligible": bool(
            class_id > 0
            and area_px > 0
            and not touches_border
            and completeness == "complete"
            and not quality_flags
        ),
    }


def _source_from_instance_id(instance_id: str) -> str:
    if instance_id.startswith("native-raster-cellvit-"):
        return "instance_json_cellvit_seed"
    if instance_id.startswith("native-raster-semantic-unseeded-"):
        return "instance_json_semantic_unseeded"
    if instance_id.startswith("native-raster-semantic-residual-"):
        return "instance_json_semantic_seeded_residual"
    if instance_id.startswith("native-raster-semantic-fallback-"):
        return "instance_json_semantic_fallback"
    if instance_id.startswith("native-"):
        return "instance_json"
    return "semantic_distance_watershed"


def _string_key_counts(values: Counter[int]) -> dict[str, int]:
    return {
        str(int(key)): int(value)
        for key, value in sorted(values.items())
        if int(value) > 0
    }


def _require_digest(value: str, name: str) -> None:
    if len(str(value)) != 64 or any(
        character not in "0123456789abcdef" for character in str(value).lower()
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
