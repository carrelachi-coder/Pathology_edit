"""Post-execution validation for Phase 3 mask edit primitives.

Checks the target mask and change region against recipe validation_rules,
schema constraints and global defaults.  Returns a structured verdict
rather than raising — primitives decide whether to raise or warn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema


class MaskValidationError(ValueError):
    """Raised when a validation check fails with no acceptable fallback."""


@dataclass(frozen=True)
class ValidationResult:
    """Structured verdict from post-execution validation."""

    passed: bool
    primitive: str
    checks: tuple[ValidationCheck, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def failed_checks(self) -> tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)


@dataclass(frozen=True)
class ValidationCheck:
    """One atomic validation assertion."""

    name: str
    passed: bool
    detail: str = ""


# ── top-level entry point ──────────────────────────────────────────

def validate_edit_result(
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    changed_area_fraction: float,
    strength: str = "mild",
) -> ValidationResult:
    """Run all applicable validation checks on a primitive edit output.

    Applies global checks (change area range, label legality, background
    leakage) plus primitive-specific guards declared in the recipe's
    ``validation_rules`` list.
    """

    primitive_name = primitive_config.get("name", "")
    checks: list[ValidationCheck] = []
    warnings: list[str] = []

    # ── global checks ──────────────────────────────────────────
    checks.append(_check_change_area_nonempty(change_region, primitive_name))
    checks.append(
        _check_change_area_range(
            changed_area_fraction,
            primitive_config,
            src_mask=src_mask,
            change_region=change_region,
            schema=schema,
            strength=strength,
        )
    )
    checks.append(_check_label_legality(target_mask, schema))
    checks.append(_check_no_background_leakage(src_mask, target_mask, change_region, schema))

    # ── required labels present in src mask ─────────────────────
    required_labels = primitive_config.get("required_tissue_labels", [])
    if isinstance(required_labels, list):
        checks.append(
            _check_required_labels_present(src_mask, schema, required_labels, primitive_name)
        )

    # ── primitive-specific guards ───────────────────────────────
    rules = primitive_config.get("validation_rules", [])
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, str):
                check = _dispatch_primitive_guard(
                    rule, src_mask, target_mask, change_region,
                    schema, primitive_config, primitive_name,
                )
                if check is not None:
                    checks.append(check)

    # ── aggregate ───────────────────────────────────────────────
    passed = all(c.passed for c in checks)
    failed = [c for c in checks if not c.passed]
    for c in failed:
        warnings.append(f"{c.name}: {c.detail}")

    return ValidationResult(
        passed=passed,
        primitive=primitive_name,
        checks=tuple(checks),
        warnings=tuple(warnings),
    )


# ── global checks ──────────────────────────────────────────────────

def _check_change_area_nonempty(
    change_region: np.ndarray, primitive_name: str
) -> ValidationCheck:
    pixels = int(np.count_nonzero(change_region))
    if pixels > 0:
        return ValidationCheck("change_area_nonempty", True, f"{pixels} pixels changed.")
    return ValidationCheck("change_area_nonempty", False, "no pixels changed.")


def _check_change_area_range(
    changed_area_fraction: float,
    primitive_config: Mapping[str, Any],
    src_mask: np.ndarray | None = None,
    change_region: np.ndarray | None = None,
    schema: MaskProfileSchema | None = None,
    strength: str = "mild",
) -> ValidationCheck:
    ranges = primitive_config.get("parameter_ranges", {})
    defaults = primitive_config.get("_defaults", {})

    if (
        _is_fine_label_transition(primitive_config)
        and src_mask is not None
        and change_region is not None
    ):
        return _check_fine_transition_source_relative_change_area(
            src_mask, change_region, ranges, primitive_config
        )
    if (
        primitive_config.get("name") == "necrosis_appearance"
        and src_mask is not None
        and change_region is not None
        and schema is not None
    ):
        return _check_necrosis_tumor_relative_change_area(
            src_mask, change_region, schema, ranges
        )
    if (
        primitive_config.get("name") == "necrosis_resolution"
        and src_mask is not None
        and change_region is not None
        and schema is not None
    ):
        return _check_necrosis_resolution_relative_change_area(
            src_mask, change_region, schema, ranges
        )
    if (
        primitive_config.get("name") == "stromal_immune_infiltration"
        and src_mask is not None
        and change_region is not None
        and schema is not None
    ):
        return _check_stromal_immune_compartment_relative_change_area(
            src_mask, change_region, schema, ranges
        )
    if (
        primitive_config.get("name") == "intratumoral_immune_infiltration"
        and src_mask is not None
        and change_region is not None
        and schema is not None
    ):
        return _check_intratumoral_immune_tumor_relative_change_area(
            src_mask, change_region, schema, ranges
        )
    if (
        primitive_config.get("name") == "stromal_desmoplasia"
        and src_mask is not None
        and change_region is not None
        and schema is not None
    ):
        return _check_stromal_desmoplasia_stroma_relative_change_area(
            src_mask, change_region, schema, ranges, strength=strength
        )
    if (
        primitive_config.get("name") in {"stroma_decrease", "stromal_reduction"}
        and src_mask is not None
        and change_region is not None
        and schema is not None
    ):
        return _check_stroma_decrease_stroma_relative_change_area(
            src_mask, change_region, schema, ranges
        )

    min_fraction = _resolve_min_changed_area(ranges, defaults)
    max_fraction = _resolve_max_changed_area(ranges, defaults)

    if min_fraction is None:
        min_fraction = 0.08
    if max_fraction is None:
        max_fraction = 0.70

    if min_fraction <= changed_area_fraction <= max_fraction:
        return ValidationCheck(
            "change_area_within_range", True,
            f"changed_area_fraction={changed_area_fraction:.4f} in "
            f"[{min_fraction:.2f}, {max_fraction:.2f}]",
        )
    return ValidationCheck(
        "change_area_within_range", False,
        f"changed_area_fraction={changed_area_fraction:.4f} outside "
        f"[{min_fraction:.2f}, {max_fraction:.2f}]",
    )


def _check_fine_transition_source_relative_change_area(
    src_mask: np.ndarray,
    change_region: np.ndarray,
    ranges: Mapping[str, Any],
    primitive_config: Mapping[str, Any],
) -> ValidationCheck:
    source_ids = _fine_transition_source_ids(primitive_config)
    source_pixels = int(np.count_nonzero(np.isin(src_mask, source_ids)))
    changed_pixels = int(np.count_nonzero(change_region))
    if source_pixels == 0:
        return ValidationCheck(
            "fine_transition_source_relative_change_area",
            False,
            f"source fine IDs {list(source_ids)} absent.",
        )

    fraction = changed_pixels / source_pixels
    min_fraction, max_fraction = _resolve_fine_transition_fraction_range(ranges)
    if min_fraction <= fraction <= max_fraction:
        return ValidationCheck(
            "fine_transition_source_relative_change_area",
            True,
            f"source_relative_fraction={fraction:.4f} in "
            f"[{min_fraction:.2f}, {max_fraction:.2f}]",
        )
    return ValidationCheck(
        "fine_transition_source_relative_change_area",
        False,
        f"source_relative_fraction={fraction:.4f} outside "
        f"[{min_fraction:.2f}, {max_fraction:.2f}]",
    )


def _resolve_fine_transition_fraction_range(ranges: Mapping[str, Any]) -> tuple[float, float]:
    transition_ranges = ranges.get("source_area_transition_fraction", {})
    if isinstance(transition_ranges, Mapping):
        lows: list[float] = []
        highs: list[float] = []
        for interval in transition_ranges.values():
            if (
                isinstance(interval, list)
                and len(interval) == 2
                and all(isinstance(item, (int, float)) for item in interval)
            ):
                lows.append(float(interval[0]))
                highs.append(float(interval[1]))
        if lows and highs:
            return min(lows), max(highs)
    return 0.08, 0.70


def _is_fine_label_transition(primitive_config: Mapping[str, Any]) -> bool:
    mask_operation = primitive_config.get("mask_operation", {})
    return isinstance(mask_operation, Mapping) and mask_operation.get("type") == "fine_label_transition"


def _check_necrosis_tumor_relative_change_area(
    src_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    ranges: Mapping[str, Any],
) -> ValidationCheck:
    tumor_pixels = int(np.count_nonzero(np.isin(src_mask, schema.tumor_fine_ids)))
    if tumor_pixels == 0:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "no tumor pixels for tumor-relative necrosis change area.",
        )

    changed_tumor_fraction = int(np.count_nonzero(change_region)) / tumor_pixels
    min_fraction = _min_interval_lower_bound(
        ranges.get("target_changed_area_fraction", {})
    )
    max_fraction = float(ranges.get("max_necrosis_fraction_of_tumor", 0.60))
    if min_fraction is None:
        min_fraction = 0.0

    if min_fraction <= changed_tumor_fraction <= max_fraction:
        return ValidationCheck(
            "change_area_within_range",
            True,
            f"changed_tumor_fraction={changed_tumor_fraction:.4f} in "
            f"[{min_fraction:.2f}, {max_fraction:.2f}]",
        )
    return ValidationCheck(
        "change_area_within_range",
        False,
        f"changed_tumor_fraction={changed_tumor_fraction:.4f} outside "
        f"[{min_fraction:.2f}, {max_fraction:.2f}]",
    )


def _check_stromal_immune_compartment_relative_change_area(
    src_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    ranges: Mapping[str, Any],
) -> ValidationCheck:
    if "Stroma" not in schema.readable_labels:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "Stroma label not in schema for stromal-immune change area.",
        )
    if "Immune infiltrate" not in schema.readable_labels:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "Immune label not in schema for stromal-immune change area.",
        )

    stroma_ids = schema.resolve_fine_ids("Stroma")
    immune_ids = schema.resolve_fine_ids("Immune infiltrate")
    reference_pixels = int(
        np.count_nonzero(np.isin(src_mask, stroma_ids + immune_ids))
    )
    if reference_pixels == 0:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "no stroma/immune pixels for compartment-relative change area.",
        )

    changed_fraction = int(np.count_nonzero(change_region)) / reference_pixels
    min_fraction = _min_interval_lower_bound(
        ranges.get("immune_area_delta_fraction", {})
    )
    max_fraction = float(ranges.get("max_changed_area_fraction", 0.40))
    if min_fraction is None:
        min_fraction = 0.0

    if min_fraction <= changed_fraction <= max_fraction:
        return ValidationCheck(
            "change_area_within_range",
            True,
            f"changed_stroma_immune_fraction={changed_fraction:.4f} in "
            f"[{min_fraction:.2f}, {max_fraction:.2f}]",
        )
    return ValidationCheck(
        "change_area_within_range",
        False,
        f"changed_stroma_immune_fraction={changed_fraction:.4f} outside "
        f"[{min_fraction:.2f}, {max_fraction:.2f}]",
    )


def _check_intratumoral_immune_tumor_relative_change_area(
    src_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    ranges: Mapping[str, Any],
) -> ValidationCheck:
    tumor_pixels = int(np.count_nonzero(np.isin(src_mask, schema.tumor_fine_ids)))
    if tumor_pixels == 0:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "no tumor pixels for tumor-relative intratumoral immune change area.",
        )

    changed_tumor_fraction = int(np.count_nonzero(change_region)) / tumor_pixels
    min_fraction = _min_interval_lower_bound(
        ranges.get("target_changed_area_fraction", {})
    )
    max_fraction = float(ranges.get("max_changed_area_fraction", 0.30))
    if min_fraction is None:
        min_fraction = 0.0

    if min_fraction <= changed_tumor_fraction <= max_fraction:
        return ValidationCheck(
            "change_area_within_range",
            True,
            f"changed_tumor_fraction={changed_tumor_fraction:.4f} in "
            f"[{min_fraction:.2f}, {max_fraction:.2f}]",
        )
    return ValidationCheck(
        "change_area_within_range",
        False,
        f"changed_tumor_fraction={changed_tumor_fraction:.4f} outside "
        f"[{min_fraction:.2f}, {max_fraction:.2f}]",
    )


def _check_necrosis_resolution_relative_change_area(
    src_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    ranges: Mapping[str, Any],
) -> ValidationCheck:
    if "Necrosis" not in schema.readable_labels:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "Necrosis label not in schema for necrosis-relative resolution change area.",
        )
    necrosis_pixels = int(
        np.count_nonzero(np.isin(src_mask, schema.resolve_fine_ids("Necrosis")))
    )
    if necrosis_pixels == 0:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "no necrosis pixels for necrosis-relative resolution change area.",
        )

    changed_necrosis_fraction = int(np.count_nonzero(change_region)) / necrosis_pixels
    min_fraction = _min_interval_lower_bound(
        ranges.get("necrosis_area_decrease_fraction", {})
    )
    max_fraction = _max_interval_upper_bound(
        ranges.get("necrosis_area_decrease_fraction", {})
    )
    if min_fraction is None:
        min_fraction = 0.0
    if max_fraction is None:
        max_fraction = 1.0

    if min_fraction <= changed_necrosis_fraction <= max_fraction:
        return ValidationCheck(
            "change_area_within_range",
            True,
            f"changed_necrosis_fraction={changed_necrosis_fraction:.4f} in "
            f"[{min_fraction:.2f}, {max_fraction:.2f}]",
        )
    return ValidationCheck(
        "change_area_within_range",
        False,
        f"changed_necrosis_fraction={changed_necrosis_fraction:.4f} outside "
        f"[{min_fraction:.2f}, {max_fraction:.2f}]",
    )


def _check_stromal_desmoplasia_stroma_relative_change_area(
    src_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    ranges: Mapping[str, Any],
    *,
    strength: str = "mild",
) -> ValidationCheck:
    if "Stroma" not in schema.readable_labels:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "Stroma label not in schema for desmoplasia change area.",
        )
    stroma_pixels = int(
        np.count_nonzero(np.isin(src_mask, schema.resolve_fine_ids("Stroma")))
    )
    if stroma_pixels == 0:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "no stroma pixels for stroma-relative desmoplasia change area.",
        )

    changed_stroma_fraction = int(np.count_nonzero(change_region)) / stroma_pixels
    changed_pixels = int(np.count_nonzero(change_region))
    min_fraction = _min_interval_lower_bound(
        ranges.get("stroma_area_delta_fraction", {})
    )
    max_fraction = _max_interval_upper_bound(
        ranges.get("stroma_area_delta_fraction", {})
    )
    if min_fraction is None:
        min_fraction = 0.0
    if max_fraction is None:
        max_fraction = 0.70
    min_pixels = _pixel_floor_for_strength(
        ranges.get("min_stroma_area_delta_pixels", {}),
        strength=strength,
    )
    effective_min_pixels = max(int(np.ceil(stroma_pixels * min_fraction)), min_pixels)
    effective_min_fraction = effective_min_pixels / stroma_pixels
    effective_max_fraction = max(max_fraction, effective_min_fraction)

    if effective_min_fraction <= changed_stroma_fraction <= effective_max_fraction:
        return ValidationCheck(
            "change_area_within_range",
            True,
            f"changed_stroma_fraction={changed_stroma_fraction:.4f} in "
            f"[{effective_min_fraction:.2f}, {effective_max_fraction:.2f}] "
            f"and changed_pixels={changed_pixels} >= {effective_min_pixels}",
        )
    return ValidationCheck(
        "change_area_within_range",
        False,
        f"changed_stroma_fraction={changed_stroma_fraction:.4f} outside "
        f"[{effective_min_fraction:.2f}, {effective_max_fraction:.2f}] "
        f"or changed_pixels={changed_pixels} < {effective_min_pixels}",
    )


def _check_stroma_decrease_stroma_relative_change_area(
    src_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    ranges: Mapping[str, Any],
) -> ValidationCheck:
    if "Stroma" not in schema.readable_labels:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "Stroma label not in schema for stroma decrease change area.",
        )
    stroma_pixels = int(
        np.count_nonzero(np.isin(src_mask, schema.resolve_fine_ids("Stroma")))
    )
    if stroma_pixels == 0:
        return ValidationCheck(
            "change_area_within_range",
            False,
            "no stroma pixels for stroma-relative decrease change area.",
        )

    changed_stroma_fraction = int(np.count_nonzero(change_region)) / stroma_pixels
    min_fraction = _min_interval_lower_bound(
        ranges.get("stroma_area_decrease_fraction", {})
    )
    max_fraction = _max_interval_upper_bound(
        ranges.get("stroma_area_decrease_fraction", {})
    )
    if min_fraction is None:
        min_fraction = 0.0
    if max_fraction is None:
        max_fraction = 0.70

    if min_fraction <= changed_stroma_fraction <= max_fraction:
        return ValidationCheck(
            "change_area_within_range",
            True,
            f"changed_stroma_fraction={changed_stroma_fraction:.4f} in "
            f"[{min_fraction:.2f}, {max_fraction:.2f}]",
        )
    return ValidationCheck(
        "change_area_within_range",
        False,
        f"changed_stroma_fraction={changed_stroma_fraction:.4f} outside "
        f"[{min_fraction:.2f}, {max_fraction:.2f}]",
    )


def _check_label_legality(
    target_mask: np.ndarray, schema: MaskProfileSchema
) -> ValidationCheck:
    known_ids = set(schema.label_to_fine_ids.values())
    known_flat: set[int] = set()
    for ids_tuple in known_ids:
        for id_val in ids_tuple:
            known_flat.add(int(id_val))
    known_flat |= set(schema.skip_fine_ids)

    mask_ids = set(np.unique(target_mask).astype(int).tolist())
    unknown = mask_ids - known_flat

    if not unknown:
        return ValidationCheck("label_legality", True, "all labels in schema.")
    return ValidationCheck(
        "label_legality", False,
        f"unknown fine ids in target mask: {sorted(unknown)}",
    )


def _check_no_background_leakage(
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
) -> ValidationCheck:
    background_ids = tuple(schema.skip_fine_ids)
    src_bg = np.isin(src_mask, background_ids)
    tgt_bg_in_change = np.isin(target_mask, background_ids) & change_region
    leaked_pixels = int(np.count_nonzero(tgt_bg_in_change & ~src_bg))

    if leaked_pixels == 0:
        return ValidationCheck("no_background_leakage", True, "no background leakage.")
    return ValidationCheck(
        "no_background_leakage", False,
        f"{leaked_pixels} pixels changed to Background in change region.",
    )


def _check_required_labels_present(
    src_mask: np.ndarray,
    schema: MaskProfileSchema,
    required_labels: list[str],
    primitive_name: str,
) -> ValidationCheck:
    missing: list[str] = []
    for label in required_labels:
        fine_ids = schema.label_to_fine_ids.get(label)
        if fine_ids is None:
            missing.append(label)
            continue
        if not np.any(np.isin(src_mask, fine_ids)):
            missing.append(label)

    if not missing:
        return ValidationCheck(
            "required_labels_present", True,
            f"all required labels present in src mask.",
        )
    return ValidationCheck(
        "required_labels_present", False,
        f"missing required labels in src mask: {missing}",
    )


# ── primitive-specific guard dispatch ──────────────────────────────

_PRIMITIVE_GUARDS: dict[str, _GuardFn] = {}


def _dispatch_primitive_guard(
    rule: str,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck | None:
    guard = _PRIMITIVE_GUARDS.get(rule)
    if guard is None:
        return ValidationCheck(rule, True, "no guard implemented; skipped.")
    return guard(
        src_mask=src_mask,
        target_mask=target_mask,
        change_region=change_region,
        schema=schema,
        primitive_config=primitive_config,
        primitive_name=primitive_name,
    )


# ── guard implementations ──────────────────────────────────────────

# Type alias for guard function signatures
_GuardFn = _dispatch_primitive_guard  # reuse the callable type from dispatch


def _guard_tumor_area_must_increase(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    tgt_tumor = np.isin(target_mask, schema.tumor_fine_ids)
    src_count = int(np.count_nonzero(src_tumor))
    tgt_count = int(np.count_nonzero(tgt_tumor))
    if tgt_count > src_count:
        return ValidationCheck(
            "tumor_area_must_increase", True,
            f"tumor {src_count} -> {tgt_count} pixels.",
        )
    return ValidationCheck(
        "tumor_area_must_increase", False,
        f"tumor did not increase: {src_count} -> {tgt_count} pixels.",
    )


_PRIMITIVE_GUARDS["tumor_area_must_increase"] = _guard_tumor_area_must_increase


def _guard_fine_transition_source_must_decrease(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    del change_region, schema, primitive_name
    source_ids = _fine_transition_source_ids(primitive_config)
    src_count = int(np.count_nonzero(np.isin(src_mask, source_ids)))
    tgt_count = int(np.count_nonzero(np.isin(target_mask, source_ids)))
    if tgt_count < src_count:
        return ValidationCheck(
            "fine_transition_source_must_decrease",
            True,
            f"source fine IDs {list(source_ids)} {src_count} -> {tgt_count} pixels.",
        )
    return ValidationCheck(
        "fine_transition_source_must_decrease",
        False,
        f"source fine IDs {list(source_ids)} did not decrease: {src_count} -> {tgt_count}.",
    )


_PRIMITIVE_GUARDS["fine_transition_source_must_decrease"] = (
    _guard_fine_transition_source_must_decrease
)


def _guard_fine_transition_target_must_increase(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    del change_region, schema, primitive_name
    target_id = _fine_transition_target_id(primitive_config)
    src_count = int(np.count_nonzero(src_mask == target_id))
    tgt_count = int(np.count_nonzero(target_mask == target_id))
    if tgt_count > src_count:
        return ValidationCheck(
            "fine_transition_target_must_increase",
            True,
            f"target fine ID {target_id} {src_count} -> {tgt_count} pixels.",
        )
    return ValidationCheck(
        "fine_transition_target_must_increase",
        False,
        f"target fine ID {target_id} did not increase: {src_count} -> {tgt_count}.",
    )


_PRIMITIVE_GUARDS["fine_transition_target_must_increase"] = (
    _guard_fine_transition_target_must_increase
)


def _guard_change_region_must_match_source_fine_ids(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    del target_mask, schema, primitive_name
    source_ids = _fine_transition_source_ids(primitive_config)
    outside = int(np.count_nonzero(change_region & ~np.isin(src_mask, source_ids)))
    if outside == 0:
        return ValidationCheck(
            "change_region_must_match_source_fine_ids",
            True,
            f"all changed pixels came from source fine IDs {list(source_ids)}.",
        )
    return ValidationCheck(
        "change_region_must_match_source_fine_ids",
        False,
        f"{outside} changed pixels were outside source fine IDs {list(source_ids)}.",
    )


_PRIMITIVE_GUARDS["change_region_must_match_source_fine_ids"] = (
    _guard_change_region_must_match_source_fine_ids
)


def _guard_tumor_area_must_decrease(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    tgt_tumor = np.isin(target_mask, schema.tumor_fine_ids)
    src_count = int(np.count_nonzero(src_tumor))
    tgt_count = int(np.count_nonzero(tgt_tumor))
    if tgt_count < src_count:
        return ValidationCheck(
            "tumor_area_must_decrease", True,
            f"tumor {src_count} -> {tgt_count} pixels.",
        )
    return ValidationCheck(
        "tumor_area_must_decrease", False,
        f"tumor did not decrease: {src_count} -> {tgt_count} pixels.",
    )


_PRIMITIVE_GUARDS["tumor_area_must_decrease"] = _guard_tumor_area_must_decrease


def _guard_new_tumor_must_touch_or_neighbor_original_tumor(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    new_tumor = np.isin(target_mask, schema.tumor_fine_ids) & change_region & ~src_tumor
    if not np.any(new_tumor):
        return ValidationCheck(
            "new_tumor_must_touch_or_neighbor_original_tumor", True,
            "no new tumor pixels to check.",
        )

    from scipy import ndimage
    dilated_src = ndimage.binary_dilation(src_tumor, structure=np.ones((3, 3)))
    touching = np.any(new_tumor & dilated_src)
    if touching:
        return ValidationCheck(
            "new_tumor_must_touch_or_neighbor_original_tumor", True,
            "new tumor touches or neighbors original tumor.",
        )
    return ValidationCheck(
        "new_tumor_must_touch_or_neighbor_original_tumor", False,
        "new tumor pixels do not touch or neighbor original tumor.",
    )


_PRIMITIVE_GUARDS["new_tumor_must_touch_or_neighbor_original_tumor"] = (
    _guard_new_tumor_must_touch_or_neighbor_original_tumor
)


def _guard_tumor_area_change_must_remain_small(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    max_delta_frac = primitive_config.get("parameter_ranges", {}).get(
        "max_abs_tumor_area_delta_fraction", 0.02
    )
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    tgt_tumor = np.isin(target_mask, schema.tumor_fine_ids)
    delta_frac = abs(int(np.count_nonzero(tgt_tumor)) - int(np.count_nonzero(src_tumor))) / int(src_mask.size)
    if delta_frac <= float(max_delta_frac):
        return ValidationCheck(
            "tumor_area_change_must_remain_small", True,
            f"delta_fraction={delta_frac:.4f} <= {max_delta_frac}",
        )
    return ValidationCheck(
        "tumor_area_change_must_remain_small", False,
        f"delta_fraction={delta_frac:.4f} > {max_delta_frac}",
    )


_PRIMITIVE_GUARDS["tumor_area_change_must_remain_small"] = (
    _guard_tumor_area_change_must_remain_small
)


def _guard_tumor_must_not_fragment_or_disappear(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    tgt_tumor = np.isin(target_mask, schema.tumor_fine_ids)

    from scipy import ndimage
    src_components = int(ndimage.label(src_tumor)[1])
    tgt_components = int(ndimage.label(tgt_tumor)[1])

    tgt_count = int(np.count_nonzero(tgt_tumor))
    if tgt_count == 0:
        return ValidationCheck(
            "tumor_must_not_fragment_or_disappear", False,
            "tumor disappeared entirely.",
        )
    if tgt_components > src_components + 2:
        return ValidationCheck(
            "tumor_must_not_fragment_or_disappear", False,
            f"tumor fragmented: {src_components} -> {tgt_components} components.",
        )
    return ValidationCheck(
        "tumor_must_not_fragment_or_disappear", True,
        f"tumor components: {src_components} -> {tgt_components}.",
    )


_PRIMITIVE_GUARDS["tumor_must_not_fragment_or_disappear"] = (
    _guard_tumor_must_not_fragment_or_disappear
)


def _guard_released_region_must_not_be_background(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    background_ids = tuple(schema.skip_fine_ids)
    bg_in_change = np.isin(target_mask, background_ids) & change_region
    leaked = int(np.count_nonzero(bg_in_change))
    if leaked == 0:
        return ValidationCheck(
            "released_region_must_not_be_background", True,
            "no Background in change region.",
        )
    return ValidationCheck(
        "released_region_must_not_be_background", False,
        f"{leaked} pixels became Background in change region.",
    )


_PRIMITIVE_GUARDS["released_region_must_not_be_background"] = (
    _guard_released_region_must_not_be_background
)


def _guard_necrosis_area_must_increase(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Necrosis" not in schema.readable_labels:
        return ValidationCheck(
            "necrosis_area_must_increase", True,
            "Necrosis label not in schema; skipped.",
        )
    nec_ids = schema.resolve_fine_ids("Necrosis")
    src_nec = int(np.count_nonzero(np.isin(src_mask, nec_ids)))
    tgt_nec = int(np.count_nonzero(np.isin(target_mask, nec_ids)))
    if tgt_nec > src_nec:
        return ValidationCheck(
            "necrosis_area_must_increase", True,
            f"necrosis {src_nec} -> {tgt_nec} pixels.",
        )
    return ValidationCheck(
        "necrosis_area_must_increase", False,
        f"necrosis did not increase: {src_nec} -> {tgt_nec} pixels.",
    )


_PRIMITIVE_GUARDS["necrosis_area_must_increase"] = _guard_necrosis_area_must_increase


def _guard_necrosis_area_must_decrease(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Necrosis" not in schema.readable_labels:
        return ValidationCheck(
            "necrosis_area_must_decrease", True,
            "Necrosis label not in schema; skipped.",
        )
    nec_ids = schema.resolve_fine_ids("Necrosis")
    src_nec = int(np.count_nonzero(np.isin(src_mask, nec_ids)))
    tgt_nec = int(np.count_nonzero(np.isin(target_mask, nec_ids)))
    if tgt_nec < src_nec:
        return ValidationCheck(
            "necrosis_area_must_decrease", True,
            f"necrosis {src_nec} -> {tgt_nec} pixels.",
        )
    return ValidationCheck(
        "necrosis_area_must_decrease", False,
        f"necrosis did not decrease: {src_nec} -> {tgt_nec} pixels.",
    )


_PRIMITIVE_GUARDS["necrosis_area_must_decrease"] = _guard_necrosis_area_must_decrease


def _guard_resolved_necrosis_must_be_original_necrosis(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Necrosis" not in schema.readable_labels:
        return ValidationCheck(
            "resolved_necrosis_must_be_original_necrosis", True,
            "Necrosis label not in schema; skipped.",
        )
    nec_ids = schema.resolve_fine_ids("Necrosis")
    source_necrosis = np.isin(src_mask, nec_ids)
    outside = int(np.count_nonzero(change_region & ~source_necrosis))
    if outside == 0:
        return ValidationCheck(
            "resolved_necrosis_must_be_original_necrosis",
            True,
            "all changed pixels came from original Necrosis.",
        )
    return ValidationCheck(
        "resolved_necrosis_must_be_original_necrosis",
        False,
        f"{outside} changed pixels were not original Necrosis.",
    )


_PRIMITIVE_GUARDS["resolved_necrosis_must_be_original_necrosis"] = (
    _guard_resolved_necrosis_must_be_original_necrosis
)


def _guard_new_necrosis_must_be_inside_original_tumor(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Necrosis" not in schema.readable_labels:
        return ValidationCheck(
            "new_necrosis_must_be_inside_original_tumor", True,
            "Necrosis label not in schema; skipped.",
        )
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    nec_ids = schema.resolve_fine_ids("Necrosis")
    new_nec = np.isin(target_mask, nec_ids) & change_region & ~np.isin(src_mask, nec_ids)
    outside_tumor = new_nec & ~src_tumor
    leaked = int(np.count_nonzero(outside_tumor))
    if leaked == 0:
        return ValidationCheck(
            "new_necrosis_must_be_inside_original_tumor", True,
            "new necrosis inside original tumor.",
        )
    return ValidationCheck(
        "new_necrosis_must_be_inside_original_tumor", False,
        f"{leaked} new necrosis pixels outside original tumor.",
    )


_PRIMITIVE_GUARDS["new_necrosis_must_be_inside_original_tumor"] = (
    _guard_new_necrosis_must_be_inside_original_tumor
)


def _guard_immune_area_must_increase(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Immune infiltrate" not in schema.readable_labels:
        return ValidationCheck(
            "immune_area_must_increase", True,
            "Immune label not in schema; skipped.",
        )
    imm_ids = schema.resolve_fine_ids("Immune infiltrate")
    src_imm = int(np.count_nonzero(np.isin(src_mask, imm_ids)))
    tgt_imm = int(np.count_nonzero(np.isin(target_mask, imm_ids)))
    if tgt_imm > src_imm:
        return ValidationCheck(
            "immune_area_must_increase", True,
            f"immune {src_imm} -> {tgt_imm} pixels.",
        )
    return ValidationCheck(
        "immune_area_must_increase", False,
        f"immune did not increase: {src_imm} -> {tgt_imm} pixels.",
    )


_PRIMITIVE_GUARDS["immune_area_must_increase"] = _guard_immune_area_must_increase


def _guard_immune_area_must_decrease(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Immune infiltrate" not in schema.readable_labels:
        return ValidationCheck(
            "immune_area_must_decrease", True,
            "Immune label not in schema; skipped.",
        )
    imm_ids = schema.resolve_fine_ids("Immune infiltrate")
    src_imm = int(np.count_nonzero(np.isin(src_mask, imm_ids)))
    tgt_imm = int(np.count_nonzero(np.isin(target_mask, imm_ids)))
    if tgt_imm < src_imm:
        return ValidationCheck(
            "immune_area_must_decrease", True,
            f"immune {src_imm} -> {tgt_imm} pixels.",
        )
    return ValidationCheck(
        "immune_area_must_decrease", False,
        f"immune did not decrease: {src_imm} -> {tgt_imm} pixels.",
    )


_PRIMITIVE_GUARDS["immune_area_must_decrease"] = _guard_immune_area_must_decrease


def _guard_tumor_area_must_remain_stable(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    tgt_tumor = np.isin(target_mask, schema.tumor_fine_ids)
    src_count = int(np.count_nonzero(src_tumor))
    tgt_count = int(np.count_nonzero(tgt_tumor))
    delta_frac = abs(tgt_count - src_count) / int(src_mask.size)
    tolerance = 0.02
    if delta_frac <= tolerance:
        return ValidationCheck(
            "tumor_area_must_remain_stable", True,
            f"tumor delta_fraction={delta_frac:.4f} <= {tolerance}",
        )
    return ValidationCheck(
        "tumor_area_must_remain_stable", False,
        f"tumor changed significantly: delta_fraction={delta_frac:.4f} > {tolerance}",
    )


_PRIMITIVE_GUARDS["tumor_area_must_remain_stable"] = _guard_tumor_area_must_remain_stable


def _guard_new_immune_must_be_mainly_outside_tumor(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Immune infiltrate" not in schema.readable_labels:
        return ValidationCheck(
            "new_immune_must_be_mainly_outside_tumor", True,
            "Immune label not in schema; skipped.",
        )
    imm_ids = schema.resolve_fine_ids("Immune infiltrate")
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    new_immune = np.isin(target_mask, imm_ids) & change_region & ~np.isin(src_mask, imm_ids)
    new_immune_count = int(np.count_nonzero(new_immune))
    if new_immune_count == 0:
        return ValidationCheck(
            "new_immune_must_be_mainly_outside_tumor", True,
            "no new immune pixels to check.",
        )
    inside_tumor = int(np.count_nonzero(new_immune & src_tumor))
    inside_fraction = inside_tumor / new_immune_count
    if inside_fraction <= 0.15:
        return ValidationCheck(
            "new_immune_must_be_mainly_outside_tumor", True,
            f"only {inside_fraction:.1%} new immune inside tumor.",
        )
    return ValidationCheck(
        "new_immune_must_be_mainly_outside_tumor", False,
        f"{inside_fraction:.1%} new immune pixels inside tumor (>15%).",
    )


_PRIMITIVE_GUARDS["new_immune_must_be_mainly_outside_tumor"] = (
    _guard_new_immune_must_be_mainly_outside_tumor
)


def _guard_new_immune_must_be_inside_original_tumor(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Immune infiltrate" not in schema.readable_labels:
        return ValidationCheck(
            "new_immune_must_be_inside_original_tumor", True,
            "Immune label not in schema; skipped.",
        )
    imm_ids = schema.resolve_fine_ids("Immune infiltrate")
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    new_immune = np.isin(target_mask, imm_ids) & change_region & ~np.isin(src_mask, imm_ids)
    outside_tumor = int(np.count_nonzero(new_immune & ~src_tumor))
    if outside_tumor == 0:
        return ValidationCheck(
            "new_immune_must_be_inside_original_tumor", True,
            "new immune inside original tumor.",
        )
    return ValidationCheck(
        "new_immune_must_be_inside_original_tumor", False,
        f"{outside_tumor} new immune pixels outside original tumor.",
    )


_PRIMITIVE_GUARDS["new_immune_must_be_inside_original_tumor"] = (
    _guard_new_immune_must_be_inside_original_tumor
)


def _guard_no_background_holes(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    background_ids = tuple(schema.skip_fine_ids)
    src_bg = np.isin(src_mask, background_ids)
    tgt_bg = np.isin(target_mask, background_ids)
    new_bg_in_change = tgt_bg & change_region & ~src_bg
    leaked = int(np.count_nonzero(new_bg_in_change))
    if leaked == 0:
        return ValidationCheck("no_background_holes", True, "no background holes created.")
    return ValidationCheck(
        "no_background_holes", False,
        f"{leaked} new background holes in change region.",
    )


_PRIMITIVE_GUARDS["no_background_holes"] = _guard_no_background_holes


def _guard_backfill_must_be_legal_tissue(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    background_ids = tuple(schema.skip_fine_ids)
    backfill_in_change = target_mask[change_region]
    bg_count = int(np.count_nonzero(np.isin(backfill_in_change, background_ids)))
    if bg_count == 0:
        return ValidationCheck(
            "backfill_must_be_legal_tissue", True,
            "no Background used as backfill.",
        )
    return ValidationCheck(
        "backfill_must_be_legal_tissue", False,
        f"{bg_count} pixels backfilled with Background.",
    )


_PRIMITIVE_GUARDS["backfill_must_be_legal_tissue"] = _guard_backfill_must_be_legal_tissue


def _guard_stroma_area_or_generation_region_must_increase(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Stroma" not in schema.readable_labels:
        return ValidationCheck(
            "stroma_area_or_generation_region_must_increase", True,
            "Stroma label not in schema; skipped.",
        )
    str_ids = schema.resolve_fine_ids("Stroma")
    src_str = int(np.count_nonzero(np.isin(src_mask, str_ids)))
    tgt_str = int(np.count_nonzero(np.isin(target_mask, str_ids)))
    if tgt_str > src_str:
        return ValidationCheck(
            "stroma_area_or_generation_region_must_increase", True,
            f"stroma {src_str} -> {tgt_str} pixels.",
        )
    return ValidationCheck(
        "stroma_area_or_generation_region_must_increase", False,
        f"stroma did not increase: {src_str} -> {tgt_str} pixels.",
    )


_PRIMITIVE_GUARDS["stroma_area_or_generation_region_must_increase"] = (
    _guard_stroma_area_or_generation_region_must_increase
)


def _guard_stroma_area_must_decrease(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Stroma" not in schema.readable_labels:
        return ValidationCheck(
            "stroma_area_must_decrease",
            True,
            "Stroma label not in schema; skipped.",
        )
    str_ids = schema.resolve_fine_ids("Stroma")
    src_str = int(np.count_nonzero(np.isin(src_mask, str_ids)))
    tgt_str = int(np.count_nonzero(np.isin(target_mask, str_ids)))
    if tgt_str < src_str:
        return ValidationCheck(
            "stroma_area_must_decrease", True, f"stroma {src_str} -> {tgt_str} pixels."
        )
    return ValidationCheck(
        "stroma_area_must_decrease",
        False,
        f"stroma did not decrease: {src_str} -> {tgt_str} pixels.",
    )


_PRIMITIVE_GUARDS["stroma_area_must_decrease"] = _guard_stroma_area_must_decrease


def _guard_change_region_must_be_outside_tumor(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    src_tumor = np.isin(src_mask, schema.tumor_fine_ids)
    overlap = int(np.count_nonzero(change_region & src_tumor))
    if overlap == 0:
        return ValidationCheck(
            "change_region_must_be_outside_tumor", True,
            "change region outside original tumor.",
        )
    return ValidationCheck(
        "change_region_must_be_outside_tumor", False,
        f"{overlap} change-region pixels overlap original tumor.",
    )


_PRIMITIVE_GUARDS["change_region_must_be_outside_tumor"] = (
    _guard_change_region_must_be_outside_tumor
)


# ── helpers ────────────────────────────────────────────────────────

def _guard_immune_to_stroma_fraction_within_limit(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Immune infiltrate" not in schema.readable_labels:
        return ValidationCheck(
            "immune_to_stroma_fraction_within_limit",
            True,
            "Immune label not in schema; skipped.",
        )
    total_changed = int(np.count_nonzero(change_region))
    if total_changed == 0:
        return ValidationCheck(
            "immune_to_stroma_fraction_within_limit",
            True,
            "no changed pixels.",
        )

    immune_ids = schema.resolve_fine_ids("Immune infiltrate")
    stroma_ids = schema.resolve_fine_ids("Stroma")
    consumed_immune = (
        np.isin(src_mask, immune_ids)
        & np.isin(target_mask, stroma_ids)
        & change_region
    )
    consumed_count = int(np.count_nonzero(consumed_immune))
    max_fraction = _desmoplasia_max_immune_fraction(primitive_config)
    fraction = consumed_count / total_changed
    if fraction <= max_fraction:
        return ValidationCheck(
            "immune_to_stroma_fraction_within_limit",
            True,
            f"immune_to_stroma_fraction={fraction:.4f} <= {max_fraction:.2f}.",
        )
    return ValidationCheck(
        "immune_to_stroma_fraction_within_limit",
        False,
        f"immune_to_stroma_fraction={fraction:.4f} > {max_fraction:.2f}.",
    )


_PRIMITIVE_GUARDS["immune_to_stroma_fraction_within_limit"] = (
    _guard_immune_to_stroma_fraction_within_limit
)


def _guard_consumed_immune_must_touch_stroma(
    *,
    src_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive_name: str,
) -> ValidationCheck:
    if "Immune infiltrate" not in schema.readable_labels:
        return ValidationCheck(
            "consumed_immune_must_touch_stroma",
            True,
            "Immune label not in schema; skipped.",
        )
    immune_ids = schema.resolve_fine_ids("Immune infiltrate")
    stroma_ids = schema.resolve_fine_ids("Stroma")
    consumed_immune = (
        np.isin(src_mask, immune_ids)
        & np.isin(target_mask, stroma_ids)
        & change_region
    )
    consumed_count = int(np.count_nonzero(consumed_immune))
    if consumed_count == 0:
        return ValidationCheck(
            "consumed_immune_must_touch_stroma",
            True,
            "no immune pixels consumed.",
        )

    original_stroma = np.isin(src_mask, stroma_ids)
    stroma_neighbors = ndimage.binary_dilation(
        original_stroma,
        structure=np.ones((3, 3), dtype=bool),
    )
    non_touching = int(np.count_nonzero(consumed_immune & ~stroma_neighbors))
    if non_touching == 0:
        return ValidationCheck(
            "consumed_immune_must_touch_stroma",
            True,
            f"{consumed_count} consumed immune pixels touch original stroma.",
        )
    return ValidationCheck(
        "consumed_immune_must_touch_stroma",
        False,
        f"{non_touching} consumed immune pixels do not touch original stroma.",
    )


_PRIMITIVE_GUARDS["consumed_immune_must_touch_stroma"] = (
    _guard_consumed_immune_must_touch_stroma
)


def _resolve_min_changed_area(
    ranges: Mapping[str, Any], defaults: Mapping[str, Any]
) -> float | None:
    value = defaults.get("min_changed_area_fraction")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.08


def _desmoplasia_max_immune_fraction(primitive_config: Mapping[str, Any]) -> float:
    spatial_pattern = primitive_config.get("spatial_pattern", {})
    constraints = (
        spatial_pattern.get("immune_to_stroma_constraints", {})
        if isinstance(spatial_pattern, Mapping)
        else {}
    )
    if not isinstance(constraints, Mapping):
        return 0.30
    value = constraints.get("max_fraction_of_total_desmoplasia_delta", 0.30)
    if not isinstance(value, (int, float)):
        return 0.30
    return float(value)


def _fine_transition_source_ids(primitive_config: Mapping[str, Any]) -> tuple[int, ...]:
    mask_operation = primitive_config.get("mask_operation", {})
    if not isinstance(mask_operation, Mapping):
        return ()
    value = mask_operation.get("source_fine_ids")
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
        return tuple(value)
    return ()


def _fine_transition_target_id(primitive_config: Mapping[str, Any]) -> int:
    mask_operation = primitive_config.get("mask_operation", {})
    if not isinstance(mask_operation, Mapping):
        return -1
    value = mask_operation.get("target_fine_id")
    return int(value) if isinstance(value, int) else -1


def _min_interval_lower_bound(value: Any) -> float | None:
    lower_bounds: list[float] = []
    if isinstance(value, Mapping):
        for nested in value.values():
            lower = _min_interval_lower_bound(nested)
            if lower is not None:
                lower_bounds.append(lower)
    elif (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        lower_bounds.append(float(value[0]))

    return min(lower_bounds) if lower_bounds else None


def _max_interval_upper_bound(value: Any) -> float | None:
    upper_bounds: list[float] = []
    if isinstance(value, Mapping):
        for nested in value.values():
            upper = _max_interval_upper_bound(nested)
            if upper is not None:
                upper_bounds.append(upper)
    elif (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        upper_bounds.append(float(value[1]))

    return max(upper_bounds) if upper_bounds else None


def _min_pixel_floor(value: Any) -> int:
    floors: list[int] = []
    if isinstance(value, Mapping):
        for nested in value.values():
            floor = _min_pixel_floor(nested)
            if floor > 0:
                floors.append(floor)
    elif isinstance(value, (int, float)) and int(value) > 0:
        floors.append(int(value))
    return min(floors) if floors else 0


def _pixel_floor_for_strength(value: Any, *, strength: str) -> int:
    if isinstance(value, Mapping):
        raw = value.get(strength)
    else:
        raw = value
    if isinstance(raw, (int, float)) and int(raw) > 0:
        return int(raw)
    return 0


def _resolve_max_changed_area(
    ranges: Mapping[str, Any], defaults: Mapping[str, Any]
) -> float | None:
    value = ranges.get("max_changed_area_fraction")
    if isinstance(value, (int, float)):
        return float(value)
    return defaults.get("max_changed_area_fraction", 0.70)
