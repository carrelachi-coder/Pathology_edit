"""Shared helpers for Phase 3 real-mask visualization and smoke tests."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema, MaskProfileSchemaError
from phase3_mask_edit.core.mask_io import (
    id_to_rgb,
    load_id_mask,
    load_rgb_mask,
    save_change_region,
    save_id_mask,
    save_metadata,
    save_rgb_mask,
)
from phase3_mask_edit.generic.executor import execute_edit


RECIPE_PATH = Path("phase3_mask_edit/recipes/generic.yaml")
DEFAULT_DATA_ROOT = Path("edit_datasets")


def primitive_config(recipe: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return a primitive config by name."""

    for primitive in recipe.get("primitives", []):
        if isinstance(primitive, Mapping) and primitive.get("name") == name:
            return primitive
    raise KeyError(f"Primitive not found in recipe: {name}")


def iter_profile_mask_paths(data_root: Path, profile: str) -> list[Path]:
    """Find candidate tissue mask paths for a profile.

    The preferred layout is the project `edit_datasets/<PROFILE>/metadata.jsonl`
    plus `tissue_masks/`.  Fallback globbing keeps the smoke scripts portable
    across ad-hoc copied validation folders.
    """

    profile_root = data_root / profile
    seen: set[Path] = set()
    paths: list[Path] = []

    meta_path = profile_root / "metadata.jsonl"
    if meta_path.exists():
        for record in _iter_jsonl(meta_path):
            for key in ("conditioning_image", "mask", "mask_path", "tissue_mask"):
                value = record.get(key)
                if not isinstance(value, str) or not value:
                    continue
                candidates = _metadata_path_candidates(profile_root, value)
                for candidate in candidates:
                    if candidate.exists() and candidate not in seen:
                        paths.append(candidate)
                        seen.add(candidate)

    for folder_name in ("tissue_masks", "conditioning", "masks"):
        folder = profile_root / folder_name
        if not folder.exists():
            continue
        for candidate in sorted(folder.rglob("*.png")):
            if candidate not in seen:
                paths.append(candidate)
                seen.add(candidate)

    return paths


def load_mask_auto(path: Path) -> np.ndarray:
    """Load either a grayscale id mask or a unified RGB visualization mask."""

    img = Image.open(path)
    if img.mode in {"RGB", "RGBA"}:
        rgb = np.asarray(img.convert("RGB"))
        channel_equal = (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        )
        if not channel_equal:
            return load_rgb_mask(path)
    return load_id_mask(path)


def select_samples(
    *,
    data_root: Path,
    profile: str,
    primitive: str,
    limit: int,
    include_rejected: bool = False,
) -> list[dict[str, Any]]:
    """Select real-mask samples for a primitive.

    Returns records with `mask_path`, `old_mask`, `schema`, and `context`.
    Rejected records are included only when requested, which is useful for the
    unified smoke runner's failure histogram.
    """

    try:
        schema = MaskProfileSchema.from_reference_profile(profile)
    except MaskProfileSchemaError as exc:
        return [{
            "profile": profile,
            "mask_path": None,
            "load_error": f"schema_error:{exc}",
        }] if include_rejected else []

    paths = iter_profile_mask_paths(data_root, profile)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for mask_path in paths:
        try:
            old_mask = load_mask_auto(mask_path)
            context = MaskEditContext.from_mask(old_mask, schema)
        except Exception as exc:  # noqa: BLE001 - smoke output should explain failures.
            if include_rejected:
                rejected.append({
                    "profile": profile,
                    "mask_path": str(mask_path),
                    "load_error": f"{type(exc).__name__}:{exc}",
                })
            continue

        score = sample_priority_score(primitive, schema, context)
        record = {
            "profile": profile,
            "mask_path": mask_path,
            "old_mask": old_mask,
            "schema": schema,
            "context": context,
            "sample_score": score,
        }
        if score > 0:
            selected.append(record)
        elif include_rejected:
            rejected.append(record)

    selected.sort(key=lambda item: item["sample_score"], reverse=True)
    selected = selected[:limit]
    if include_rejected and len(selected) < limit:
        selected.extend(rejected[: limit - len(selected)])
    return selected


def sample_priority_score(
    primitive: str,
    schema: MaskProfileSchema,
    context: MaskEditContext,
) -> float:
    """Heuristic score for selecting useful real-mask preview samples."""

    present = context.present_labels
    areas = context.label_area_fractions

    if primitive == "necrosis_appearance":
        if "Necrosis" not in schema.writable_labels or "Tumor" not in present:
            return 0.0
        tumor = areas.get("Tumor", 0.0)
        score = tumor
        if "Necrosis" in present:
            score += 0.15
        if "Blood vessel" in present:
            score += 0.08
        return score

    if primitive == "stromal_immune_infiltration":
        if "Immune infiltrate" not in schema.writable_labels or "Stroma" not in present:
            return 0.0
        score = areas.get("Stroma", 0.0)
        if "Tumor" in present:
            tumor_fraction = areas.get("Tumor", 0.0)
            score += 0.15 if tumor_fraction >= 0.05 else 0.08
        if "Immune infiltrate" in present:
            score += 0.05
        return score

    if primitive in {"tumor_burden_increase", "tumor_burden_decrease"}:
        if "Tumor" not in present:
            return 0.0
        return areas.get("Tumor", 0.0)

    return 1.0


def run_primitive_case(
    *,
    old_mask: np.ndarray,
    schema: MaskProfileSchema,
    context: MaskEditContext,
    recipe: Mapping[str, Any],
    primitive: str,
    strength: str,
    seed: int,
) -> dict[str, Any]:
    """Execute one primitive case through the unified executor."""

    intent = EditIntent.from_mapping({
        "primitive": primitive,
        "reference_profile": schema.reference_profile,
        "strength": strength,
        "seed": seed,
    })
    result = execute_edit(old_mask, intent, recipe, schema, context)
    edit_result = result.edit_result
    validation = result.validation

    metadata: dict[str, Any] = {
        "primitive": primitive,
        "profile": schema.reference_profile,
        "strength": strength,
        "seed": seed,
        "status": result.status,
        "applicability": {
            "status": result.applicability.status,
            "reasons": list(result.applicability.reasons),
            "warnings": list(result.applicability.warnings),
            "fallback_actions": list(result.applicability.fallback_actions),
        },
        "validation": None,
        "selected_pixels": 0,
        "changed_area_fraction": 0.0,
        "ops_log": {},
        "failure_reason": None,
    }

    if validation is not None:
        metadata["validation"] = {
            "passed": validation.passed,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                }
                for check in validation.checks
            ],
            "warnings": list(validation.warnings),
        }

    if edit_result is None:
        if result.applicability.reasons:
            metadata["failure_reason"] = ";".join(result.applicability.reasons)
        else:
            metadata["failure_reason"] = result.status
        metadata["target_mask"] = old_mask.copy()
        metadata["change_region"] = np.zeros(old_mask.shape, dtype=bool)
        return metadata

    metadata.update({
        "selected_pixels": int(edit_result.selected_pixels),
        "changed_area_fraction": float(edit_result.changed_area_fraction),
        "ops_log": edit_result.ops_log,
        "target_mask": edit_result.target_mask,
        "change_region": edit_result.change_region,
    })
    metadata.update(_primitive_summary_fields(primitive, edit_result.ops_log))
    return metadata


def save_case_artifacts(
    *,
    output_dir: Path,
    case_name: str,
    old_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    """Save standard mask-preview artifacts for one case."""

    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "src_mask": save_id_mask(old_mask, case_dir / "src_mask.png"),
        "target_mask": save_id_mask(target_mask, case_dir / "target_mask.png"),
        "src_rgb": save_rgb_mask(old_mask, case_dir / "src_rgb.png"),
        "target_rgb": save_rgb_mask(target_mask, case_dir / "target_rgb.png"),
        "change_region": save_change_region(change_region, case_dir / "change_region.png"),
    }
    panel = make_comparison_panel(old_mask, target_mask, change_region, case_name)
    panel_path = case_dir / "panel.png"
    Image.fromarray(panel, mode="RGB").save(panel_path)
    paths["panel"] = panel_path

    json_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"target_mask", "change_region"}
    }
    json_metadata["artifacts"] = {key: str(value) for key, value in paths.items()}
    paths["metadata"] = save_metadata(json_metadata, case_dir / "metadata.json")
    return {key: str(value) for key, value in paths.items()}


def make_comparison_panel(
    old_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    title: str,
) -> np.ndarray:
    """Create a source/target/change side-by-side RGB preview panel."""

    src_rgb = id_to_rgb(old_mask)
    target_rgb = id_to_rgb(target_mask)
    change_rgb = src_rgb.copy()
    change_rgb[np.asarray(change_region, dtype=bool)] = np.array([255, 255, 0], dtype=np.uint8)

    h, w = src_rgb.shape[:2]
    header_h = 26
    footer_h = 26
    gap = 4
    panel = np.full(
        (h + header_h + footer_h, w * 3 + gap * 2, 3),
        245,
        dtype=np.uint8,
    )
    y0 = header_h
    panel[y0:y0 + h, 0:w] = src_rgb
    panel[y0:y0 + h, w + gap:w * 2 + gap] = target_rgb
    panel[y0:y0 + h, w * 2 + gap * 2:w * 3 + gap * 2] = change_rgb

    img = Image.fromarray(panel, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.text((5, 6), "Source", fill=(0, 0, 0))
    draw.text((w + gap + 5, 6), "Target", fill=(0, 0, 0))
    draw.text((w * 2 + gap * 2 + 5, 6), "Change", fill=(0, 0, 0))
    draw.text((5, h + header_h + 6), title[:180], fill=(0, 0, 0))
    return np.asarray(img)


def write_summary_files(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Write aggregate JSON, CSV, and failures summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_rows = [_summary_json_row(row) for row in rows]
    save_metadata(
        {"cases": json_rows, "aggregate": aggregate_rows(json_rows)},
        output_dir / "summary.json",
    )

    csv_rows = [_summary_csv_row(row) for row in rows]
    if csv_rows:
        fieldnames = sorted({key for row in csv_rows for key in row})
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    failures = [
        row for row in json_rows
        if row.get("status") in {"rejected", "execution_failed"}
        or row.get("failure_reason")
    ]
    save_metadata({"failures": failures}, output_dir / "failures.json")


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build simple aggregate counters for smoke summaries."""

    by_status = Counter(str(row.get("status", "unknown")) for row in rows)
    by_profile = Counter(str(row.get("profile", "unknown")) for row in rows)
    by_primitive = Counter(str(row.get("primitive", "unknown")) for row in rows)
    by_failure = Counter(
        str(row.get("failure_reason"))
        for row in rows
        if row.get("failure_reason")
    )
    changed = [
        float(row["changed_area_fraction"])
        for row in rows
        if row.get("changed_area_fraction") is not None
    ]
    return {
        "total_cases": len(rows),
        "by_status": dict(by_status),
        "by_profile": dict(by_profile),
        "by_primitive": dict(by_primitive),
        "failure_reasons": dict(by_failure),
        "changed_area_fraction": _numeric_distribution(changed),
    }


def parse_profiles(values: Iterable[str] | None) -> list[str]:
    if not values:
        return ["BCSS", "IGNITE", "PUMA"]
    return [value.strip() for value in values if value.strip()]


def parse_strengths(values: Iterable[str] | None) -> list[str]:
    if not values:
        return ["mild", "moderate", "significant"]
    return [value.strip() for value in values if value.strip()]


def safe_case_name(*parts: object) -> str:
    raw = "_".join(str(part) for part in parts if part is not None and str(part))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def load_default_recipe(path: Path = RECIPE_PATH) -> Mapping[str, Any]:
    return load_recipe(path)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _metadata_path_candidates(profile_root: Path, value: str) -> list[Path]:
    normalized = value.replace("\\", "/")
    p = Path(normalized)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(profile_root / normalized)
    candidates.append(profile_root / "tissue_masks" / p.name)
    candidates.append(profile_root / "conditioning" / p.name)
    return candidates


def _primitive_summary_fields(primitive: str, ops_log: Mapping[str, Any]) -> dict[str, Any]:
    spatial = ops_log.get("spatial", {}) if isinstance(ops_log, Mapping) else {}
    fields: dict[str, Any] = {}

    if primitive == "necrosis_appearance":
        for key in (
            "changed_tumor_fraction",
            "target_area_reference",
            "selected_components",
            "max_components",
            "used_existing_necrosis_neighborhood",
            "used_blood_vessel_distance",
            "retry_applied",
        ):
            if key in ops_log:
                fields[key] = ops_log[key]
            elif isinstance(spatial, Mapping) and key in spatial:
                fields[key] = spatial[key]

    if primitive == "stromal_immune_infiltration":
        for key in (
            "changed_stroma_immune_fraction",
            "target_area_reference",
            "selected_components",
            "tumor_mode",
            "tumor_fraction",
            "active_weights",
            "hard_distance_limit_px",
            "used_soft_peritumoral_priority",
            "used_existing_immune_neighborhood",
        ):
            if key in ops_log:
                fields[key] = ops_log[key]
            elif isinstance(spatial, Mapping) and key in spatial:
                fields[key] = spatial[key]

    return fields


def _summary_json_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in row.items()
        if key not in {"old_mask", "schema", "context", "target_mask", "change_region"}
    }


def _summary_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"old_mask", "schema", "context", "target_mask", "change_region", "ops_log"}:
            continue
        if isinstance(value, Path):
            compact[key] = str(value)
        elif isinstance(value, (dict, list, tuple)):
            compact[key] = json.dumps(value, ensure_ascii=False)
        else:
            compact[key] = value
    return compact


def _numeric_distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }
