"""Stage-gated review artifacts for the online pathology edit workflow."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from dataset_config.unified_labels import CELL_CLASSES, CELL_COLOR_MAP
from phase3_mask_edit.backends.llm_preview import id_mask_to_llm_preview_rgb
from phase3_mask_edit.benchmark.intents import (
    primitive_config_by_name,
    strength_denominator_pixels,
)
from phase3_mask_edit.core.config import (
    default_recipe_path_for_profile,
    load_recipe,
)
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    load_id_mask,
    save_change_region,
    save_id_mask,
)


STAGE_NAMES = ("mask", "nuclei", "image")
STOP_AFTER_ALIASES = {
    "mask": "mask",
    "tissue": "mask",
    "nuclei": "nuclei",
    "cell": "nuclei",
    "image": "image",
    "generation": "image",
}


def normalize_stop_after(value: str) -> str:
    """Return the public stage name while accepting legacy CLI values."""

    try:
        return STOP_AFTER_ALIASES[str(value).strip().lower()]
    except KeyError as exc:
        allowed = ", ".join(STAGE_NAMES)
        raise ValueError(f"Unknown stop-after stage {value!r}; expected {allowed}.") from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_mask_stage_decisions(
    manifest_path: str | Path,
    *,
    approved_case_ids: tuple[str, ...] = (),
    revision_required_case_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Persist human mask decisions while verifying the approved asset hashes."""

    path = Path(manifest_path)
    manifest = _read_json_if_exists(path)
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Mask stage manifest must contain an entries list.")

    approved = set(approved_case_ids)
    revision_required = set(revision_required_case_ids)
    overlap = approved & revision_required
    if overlap:
        raise ValueError(
            "Cases cannot be both approved and revision-required: "
            + ", ".join(sorted(overlap))
        )
    known = {str(entry.get("case_id") or "") for entry in entries}
    unknown = (approved | revision_required) - known
    if unknown:
        raise ValueError("Unknown case ids: " + ", ".join(sorted(unknown)))

    decided_at = datetime.now(timezone.utc).isoformat()
    for entry in entries:
        case_id = str(entry.get("case_id") or "")
        if case_id in approved:
            lock_path = Path(str(entry.get("lock_path") or ""))
            lock = _read_json_if_exists(lock_path)
            target_path = Path(str(entry.get("target_tissue_mask_path") or ""))
            expected_hash = str(entry.get("target_tissue_sha256") or "")
            locked_hash = str(
                (lock.get("asset_sha256") or {}).get("target_tissue") or ""
            )
            current_hash = sha256_file(target_path)
            if not expected_hash or current_hash != expected_hash:
                raise ValueError(f"Target hash mismatch for approved case {case_id}.")
            if locked_hash != expected_hash:
                raise ValueError(f"Lock hash mismatch for approved case {case_id}.")
            lock["approval"] = {
                "status": "approved",
                "approved_target_sha256": expected_hash,
                "decided_at": decided_at,
                "decision_source": "human_review",
            }
            _write_json(lock, lock_path)
            entry["approval"] = "approved"
            entry["approved_target_sha256"] = expected_hash
            entry["approval_decided_at"] = decided_at
        elif case_id in revision_required:
            lock_path = Path(str(entry.get("lock_path") or ""))
            lock = _read_json_if_exists(lock_path)
            lock["approval"] = {
                "status": "revision_required",
                "approved_target_sha256": None,
                "decided_at": decided_at,
                "decision_source": "human_review",
            }
            _write_json(lock, lock_path)
            entry["approval"] = "revision_required"
            entry["approved_target_sha256"] = None
            entry["approval_decided_at"] = decided_at

    approved_count = sum(
        str(entry.get("approval") or "") == "approved" for entry in entries
    )
    revision_count = sum(
        str(entry.get("approval") or "") == "revision_required"
        for entry in entries
    )
    manifest["approval"] = {
        "status": "approved" if approved_count == len(entries) else "partial",
        "required_entry_count": len(entries),
        "approved_entry_count": approved_count,
        "revision_required_entry_count": revision_count,
    }
    _write_json(manifest, path)
    return manifest


def record_nuclei_stage_decisions(
    manifest_path: str | Path,
    *,
    approved_case_ids: tuple[str, ...] = (),
    revision_required_case_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Persist human nuclei decisions and lock every image-stage input."""

    path = Path(manifest_path)
    manifest = _read_json_if_exists(path)
    if str(manifest.get("stage") or "") != "nuclei":
        raise ValueError("Nuclei stage manifest must have stage='nuclei'.")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Nuclei stage manifest must contain an entries list.")

    approved = set(approved_case_ids)
    revision_required = set(revision_required_case_ids)
    overlap = approved & revision_required
    if overlap:
        raise ValueError(
            "Cases cannot be both approved and revision-required: "
            + ", ".join(sorted(overlap))
        )
    known = {str(entry.get("case_id") or "") for entry in entries}
    unknown = (approved | revision_required) - known
    if unknown:
        raise ValueError("Unknown case ids: " + ", ".join(sorted(unknown)))

    decided_at = datetime.now(timezone.utc).isoformat()
    for entry in entries:
        case_id = str(entry.get("case_id") or "")
        lock_path = Path(str(entry.get("lock_path") or ""))
        lock = _read_json_if_exists(lock_path)
        if case_id in approved:
            expected_nuclei_hash = str(
                entry.get("target_nuclei_sha256") or ""
            )
            locked_assets = dict(lock.get("asset_sha256") or {})
            target_nuclei_path = Path(
                str(
                    entry.get("target_nuclei_mask_path")
                    or lock.get("target_nuclei_mask_path")
                    or ""
                )
            )
            if (
                not expected_nuclei_hash
                or sha256_file(target_nuclei_path) != expected_nuclei_hash
            ):
                raise ValueError(
                    f"Target nuclei hash mismatch for approved case {case_id}."
                )
            if (
                str(locked_assets.get("target_nuclei") or "")
                != expected_nuclei_hash
            ):
                raise ValueError(
                    f"Nuclei lock hash mismatch for approved case {case_id}."
                )

            source_run_dir = Path(
                str(entry.get("run_dir") or lock_path.parent.parent)
            )
            required_assets = {
                "target_tissue": Path(
                    str(lock.get("target_tissue_mask_path") or "")
                ),
                "target_nuclei": target_nuclei_path,
                "new_nuclei": Path(
                    str(lock.get("new_nuclei_mask_path") or "")
                ),
                "semantic_change_region": Path(
                    str(
                        lock.get("semantic_change_region_path")
                        or lock.get("change_region_path")
                        or ""
                    )
                ),
                "generation_change_region": Path(
                    str(
                        lock.get("generation_change_region_path")
                        or source_run_dir / "change_region.png"
                    )
                ),
                "probnet_diagnostics": Path(
                    str(lock.get("diagnostics_path") or "")
                ),
                "cell_fill_log": Path(
                    str(
                        lock.get("cell_fill_log_path")
                        or source_run_dir / "cell_fill_log.json"
                    )
                ),
                "erased_image": Path(
                    str(
                        lock.get("erased_image_path")
                        or source_run_dir / "erased_image.png"
                    )
                ),
            }
            optional_assets = {
                "retained_nuclei": source_run_dir
                / "retained_nuclei_mask.png",
                "target_combined": source_run_dir
                / "target_combined_mask.png",
            }
            for name, asset_path in required_assets.items():
                if not asset_path.is_file():
                    raise ValueError(
                        f"Missing {name} asset for approved case {case_id}: "
                        f"{asset_path}"
                    )
                current_hash = sha256_file(asset_path)
                locked_hash = str(locked_assets.get(name) or "")
                if locked_hash and current_hash != locked_hash:
                    raise ValueError(
                        f"{name} hash mismatch for approved case {case_id}."
                    )
                locked_assets[name] = current_hash
                lock[f"{name}_path"] = str(asset_path)
            for name, asset_path in optional_assets.items():
                if asset_path.is_file():
                    locked_assets[name] = sha256_file(asset_path)
                    lock[f"{name}_path"] = str(asset_path)

            expected_tissue_hash = str(
                lock.get("parent_target_tissue_sha256")
                or locked_assets.get("target_tissue")
                or ""
            )
            if (
                not expected_tissue_hash
                or locked_assets["target_tissue"] != expected_tissue_hash
            ):
                raise ValueError(
                    f"Parent tissue hash mismatch for approved case {case_id}."
                )
            lock["asset_sha256"] = locked_assets
            lock["approval"] = {
                "status": "approved",
                "approved_target_nuclei_sha256": expected_nuclei_hash,
                "approved_target_tissue_sha256": expected_tissue_hash,
                "decided_at": decided_at,
                "decision_source": "human_review",
            }
            _write_json(lock, lock_path)
            entry["approval"] = "approved"
            entry["approved_target_nuclei_sha256"] = expected_nuclei_hash
            entry["approved_target_tissue_sha256"] = expected_tissue_hash
            entry["approval_decided_at"] = decided_at
        elif case_id in revision_required:
            lock["approval"] = {
                "status": "revision_required",
                "approved_target_nuclei_sha256": None,
                "decided_at": decided_at,
                "decision_source": "human_review",
            }
            _write_json(lock, lock_path)
            entry["approval"] = "revision_required"
            entry["approved_target_nuclei_sha256"] = None
            entry["approval_decided_at"] = decided_at

    approved_count = sum(
        str(entry.get("approval") or "") == "approved" for entry in entries
    )
    revision_count = sum(
        str(entry.get("approval") or "") == "revision_required"
        for entry in entries
    )
    manifest["approval"] = {
        "status": "approved" if approved_count == len(entries) else "partial",
        "required_entry_count": len(entries),
        "approved_entry_count": approved_count,
        "revision_required_entry_count": revision_count,
    }
    _write_json(manifest, path)
    return manifest


def build_mask_stage_review(
    *,
    run_dir: str | Path,
    case: Mapping[str, Any],
    variant: Mapping[str, Any],
    state: Mapping[str, Any],
    tissue_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit and render one mask-only result, then write its immutable lock."""

    output_dir = Path(run_dir)
    review_dir = output_dir / "stage_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    source_image_path = Path(str(state["reference_image"]))
    source_mask_path = Path(str(state["reference_tissue_mask"]))
    target_mask_path = Path(str(state["target_tissue_mask"]))
    change_region_path = Path(
        str(state.get("semantic_change_region") or state["change_region"])
    )
    phase3_info = _phase3_execution_info(
        output_dir / "phase3_mask_edit",
        tissue_info=tissue_info,
    )

    source_mask = load_id_mask(source_mask_path)
    target_mask = load_id_mask(target_mask_path)
    target_mask, change_region, canonicalization = (
        canonicalize_mask_stage_artifacts(
            source_mask=source_mask,
            target_mask=target_mask,
            target_mask_path=target_mask_path,
            change_region_path=change_region_path,
            review_dir=review_dir,
        )
    )
    audit = audit_target_mask(
        source_mask=source_mask,
        target_mask=target_mask,
        profile=str(case.get("profile") or case.get("dataset") or ""),
        case=case,
        phase3_info=phase3_info,
    )
    audit["target_mask_canonicalization"] = canonicalization
    audit_path = review_dir / "mask_audit.json"
    _write_json(audit, audit_path)

    response_files = _structured_response_files(output_dir)
    response_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in response_files
    }
    instruction = str(case.get("instruction") or "")
    organic_seed = int(case.get("organic_seed", 0))
    lock = {
        "schema_version": 1,
        "stage": "mask",
        "approval": {
            "status": "pending",
            "approved_target_sha256": None,
        },
        "case_id": str(case.get("case_id") or ""),
        "condition_id": str(case.get("condition_id") or ""),
        "sample_id": str(case.get("sample_id") or ""),
        "dataset": str(case.get("dataset") or ""),
        "profile": str(case.get("profile") or case.get("dataset") or ""),
        "variant_id": str(
            variant.get("variant_id") or variant.get("edit_mode") or ""
        ),
        "instruction": instruction,
        "instruction_sha256": sha256_text(instruction),
        "organic_seed": organic_seed,
        "projection_mode": str(tissue_info.get("projection_mode") or ""),
        "api_model": str(case.get("api_model") or "gpt-4.1-mini"),
        "frozen_target_mask_consumed": False,
        "source_image_path": str(source_image_path),
        "source_tissue_mask_path": str(source_mask_path),
        "target_tissue_mask_path": str(target_mask_path),
        "change_region_path": str(change_region_path),
        "asset_sha256": {
            "source_image": sha256_file(source_image_path),
            "source_tissue": sha256_file(source_mask_path),
            "target_tissue": sha256_file(target_mask_path),
            "change_region": sha256_file(change_region_path),
        },
        "structured_response_sha256": response_hashes,
        "audit_path": str(audit_path),
        "audit_passed": bool(audit["passed"]),
        "target_mask_canonicalization": canonicalization,
    }
    lock_path = review_dir / "mask_stage_lock.json"
    _write_json(lock, lock_path)

    panel_path = review_dir / "mask_review.png"
    render_mask_review_panel(
        source_image_path=source_image_path,
        source_mask=source_mask,
        target_mask=target_mask,
        change_region=change_region,
        case=case,
        lock=lock,
        audit=audit,
        output_path=panel_path,
    )
    return {
        "stage": "mask",
        "status": "awaiting_human_approval",
        "audit_passed": bool(audit["passed"]),
        "lock_path": str(lock_path),
        "audit_path": str(audit_path),
        "panel_path": str(panel_path),
        "target_tissue_mask_path": str(target_mask_path),
        "target_tissue_sha256": lock["asset_sha256"]["target_tissue"],
        "approval": "pending",
    }


def canonicalize_target_mask_changed_islands(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Revert tiny satellite changes while preserving a substantive edit."""

    source = np.asarray(source_mask)
    target = np.asarray(target_mask)
    if source.shape != target.shape:
        raise ValueError("Source and target tissue masks must have identical shapes.")
    change = source != target
    changed_pixels = int(np.count_nonzero(change))
    threshold = _isolated_changed_component_threshold(changed_pixels)
    labeled, component_count = ndimage.label(
        change, structure=np.ones((3, 3), dtype=np.uint8)
    )
    sizes = (
        np.bincount(labeled.ravel())[1:].astype(int)
        if component_count
        else np.asarray([], dtype=int)
    )
    retained_ids = [
        index + 1 for index, area in enumerate(sizes) if int(area) >= threshold
    ]
    removed_ids = (
        [index + 1 for index, area in enumerate(sizes) if int(area) < threshold]
        if retained_ids
        else []
    )
    cleaned = target.copy()
    removed_pixels = 0
    removed_sizes: list[int] = []
    for component_id in removed_ids:
        component = labeled == component_id
        area = int(np.count_nonzero(component))
        cleaned[component] = source[component]
        removed_pixels += area
        removed_sizes.append(area)
    return cleaned, {
        "schema_version": 1,
        "policy_id": "tiny_changed_island_prune_v1",
        "applied": bool(removed_ids),
        "connectivity": 8,
        "threshold_px": int(threshold),
        "pre_changed_pixels": changed_pixels,
        "post_changed_pixels": int(changed_pixels - removed_pixels),
        "removed_component_count": len(removed_ids),
        "removed_component_sizes_px": sorted(removed_sizes),
        "removed_pixels": int(removed_pixels),
        "preserved_all_components_without_substantive_anchor": bool(
            component_count and not retained_ids
        ),
    }


def canonicalize_mask_stage_artifacts(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    target_mask_path: str | Path,
    change_region_path: str | Path,
    review_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the product mask canonicalizer and update locked stage inputs."""

    target_path = Path(target_mask_path)
    change_path = Path(change_region_path)
    review_path = Path(review_dir)
    cleaned, metadata = canonicalize_target_mask_changed_islands(
        source_mask=source_mask,
        target_mask=target_mask,
    )
    metadata["pre_target_sha256"] = sha256_file(target_path)
    if metadata["applied"]:
        review_path.mkdir(parents=True, exist_ok=True)
        backup_path = review_path / "target_mask_pre_tiny_island_prune.png"
        if not backup_path.exists():
            shutil.copyfile(target_path, backup_path)
        save_id_mask(cleaned, target_path)
        save_change_region(np.asarray(source_mask) != cleaned, change_path)
        metadata["pre_target_path"] = str(backup_path)
        metadata["pre_target_backup_sha256"] = sha256_file(backup_path)
    metadata["post_target_sha256"] = sha256_file(target_path)
    metadata["change_region_sha256"] = sha256_file(change_path)
    return cleaned, np.asarray(source_mask) != cleaned, metadata


def build_nuclei_stage_review(
    *,
    run_dir: str | Path,
    case: Mapping[str, Any],
    state: Mapping[str, Any],
    cell_info: Mapping[str, Any],
    approved_mask_stage: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit and render nuclei generated from one hash-locked tissue mask."""

    output_dir = Path(run_dir)
    review_dir = output_dir / "stage_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    target_tissue_path = Path(str(state["target_tissue_mask"]))
    target_nuclei_path = Path(str(state["target_nuclei_mask"]))
    new_nuclei_path = Path(
        str(cell_info.get("new_nuclei_mask") or output_dir / "new_nuclei_mask.png")
    )
    change_region_path = Path(
        str(state.get("semantic_change_region") or state["change_region"])
    )
    diagnostics_path = (
        output_dir / "probnet_cell_fill" / "target_nuclei.diagnostics.json"
    )
    diagnostics = _read_probnet_diagnostics(diagnostics_path)
    target_tissue = load_id_mask(target_tissue_path)
    target_nuclei = _load_grayscale(target_nuclei_path)
    new_nuclei = _load_grayscale(new_nuclei_path)
    change_region = _load_grayscale(change_region_path) > 0

    audit = audit_nuclei_condition(
        target_tissue=target_tissue,
        target_nuclei=target_nuclei,
        new_nuclei=new_nuclei,
        change_region=change_region,
        diagnostics=diagnostics,
        expected_target_sha256=str(
            approved_mask_stage.get("approved_target_sha256")
            or approved_mask_stage.get("target_tissue_sha256")
            or ""
        ),
        target_tissue_path=target_tissue_path,
    )
    audit_path = review_dir / "nuclei_audit.json"
    _write_json(audit, audit_path)

    lock = {
        "schema_version": 1,
        "stage": "nuclei",
        "approval": {"status": "pending"},
        "case_id": str(case.get("case_id") or ""),
        "condition_id": str(case.get("condition_id") or ""),
        "dataset": str(case.get("dataset") or ""),
        "profile": str(case.get("profile") or case.get("dataset") or ""),
        "approved_mask_manifest": str(
            approved_mask_stage.get("approved_mask_manifest") or ""
        ),
        "parent_mask_lock_path": str(
            approved_mask_stage.get("original_lock_path")
            or approved_mask_stage.get("lock_path")
            or ""
        ),
        "parent_target_tissue_sha256": audit["target_tissue_sha256"],
        "asset_sha256": {
            "target_tissue": audit["target_tissue_sha256"],
            "target_nuclei": sha256_file(target_nuclei_path),
            "new_nuclei": sha256_file(new_nuclei_path),
            "change_region": sha256_file(change_region_path),
            "probnet_diagnostics": sha256_file(diagnostics_path),
        },
        "target_tissue_mask_path": str(target_tissue_path),
        "target_nuclei_mask_path": str(target_nuclei_path),
        "new_nuclei_mask_path": str(new_nuclei_path),
        "change_region_path": str(change_region_path),
        "diagnostics_path": str(diagnostics_path),
        "audit_path": str(audit_path),
        "audit_passed": bool(audit["passed"]),
    }
    extra_lock_assets = {
        "generation_change_region": output_dir / "change_region.png",
        "cell_fill_log": output_dir / "cell_fill_log.json",
        "erased_image": output_dir / "erased_image.png",
        "retained_nuclei": output_dir / "retained_nuclei_mask.png",
        "target_combined": output_dir / "target_combined_mask.png",
    }
    lock["semantic_change_region_path"] = str(change_region_path)
    lock["asset_sha256"]["semantic_change_region"] = lock[
        "asset_sha256"
    ]["change_region"]
    for name, asset_path in extra_lock_assets.items():
        if asset_path.is_file():
            lock["asset_sha256"][name] = sha256_file(asset_path)
            lock[f"{name}_path"] = str(asset_path)
    lock_path = review_dir / "nuclei_stage_lock.json"
    _write_json(lock, lock_path)

    panel_path = review_dir / "nuclei_review.png"
    render_nuclei_review_panel(
        target_tissue=target_tissue,
        target_nuclei=target_nuclei,
        new_nuclei=new_nuclei,
        change_region=change_region,
        diagnostics=diagnostics,
        case=case,
        audit=audit,
        probnet_heatmap_path=output_dir
        / "probnet_cell_fill"
        / "vis"
        / "probnet_heatmap.png",
        accepted_centers_path=output_dir
        / "probnet_cell_fill"
        / "vis"
        / "accepted_centers_overlay.png",
        output_path=panel_path,
    )
    return {
        "stage": "nuclei",
        "status": "awaiting_human_approval",
        "audit_passed": bool(audit["passed"]),
        "lock_path": str(lock_path),
        "audit_path": str(audit_path),
        "panel_path": str(panel_path),
        "target_nuclei_mask_path": str(target_nuclei_path),
        "target_nuclei_sha256": lock["asset_sha256"]["target_nuclei"],
        "parent_target_tissue_sha256": audit["target_tissue_sha256"],
        "approval": "pending",
    }


def audit_nuclei_condition(
    *,
    target_tissue: np.ndarray,
    target_nuclei: np.ndarray,
    new_nuclei: np.ndarray,
    change_region: np.ndarray,
    diagnostics: Mapping[str, Any],
    expected_target_sha256: str,
    target_tissue_path: str | Path,
) -> dict[str, Any]:
    """Validate provenance plus the product ProbNet count/type/spatial audit."""

    tissue = np.asarray(target_tissue)
    nuclei = np.asarray(target_nuclei)
    new = np.asarray(new_nuclei)
    change = np.asarray(change_region, dtype=bool)
    shapes_match = tissue.shape == nuclei.shape == new.shape == change.shape
    legal_ids = {0, *CELL_CLASSES}
    observed_ids = sorted(int(value) for value in np.unique(nuclei))
    new_observed_ids = sorted(int(value) for value in np.unique(new))
    current_target_hash = sha256_file(target_tissue_path)

    tissue_diagnostics = diagnostics.get("tissues")
    tissue_diagnostics = (
        tissue_diagnostics if isinstance(tissue_diagnostics, Mapping) else {}
    )
    exact_count_rows: list[dict[str, Any]] = []
    all_exact_counts = bool(tissue_diagnostics)
    all_exact_types = bool(tissue_diagnostics)
    any_type_quota = False
    sampling_audit = diagnostics.get("sampling_audit")
    sampling_audit = (
        sampling_audit if isinstance(sampling_audit, Mapping) else {}
    )
    sampling_tissues = sampling_audit.get("tissues")
    sampling_tissues = (
        sampling_tissues if isinstance(sampling_tissues, Mapping) else {}
    )
    for tissue_id, raw in sorted(tissue_diagnostics.items()):
        item = raw if isinstance(raw, Mapping) else {}
        target_count = int(item.get("target_count") or 0)
        placed = int(item.get("placed") or 0)
        target_by_type = _normalized_count_map(item.get("target_by_type"))
        placed_by_type = _normalized_count_map(item.get("placed_by_type"))
        count_exact = target_count == placed
        type_quota_applicable = bool(target_by_type)
        any_type_quota |= type_quota_applicable
        type_exact = (
            target_by_type == placed_by_type
            if type_quota_applicable
            else None
        )
        sampling_item = sampling_tissues.get(str(tissue_id))
        sampling_item = (
            sampling_item if isinstance(sampling_item, Mapping) else {}
        )
        all_exact_counts &= count_exact
        if type_exact is False:
            all_exact_types = False
        exact_count_rows.append(
            {
                "tissue_id": str(tissue_id),
                "target_count": target_count,
                "placed": placed,
                "target_by_type": target_by_type,
                "placed_by_type": placed_by_type,
                "count_exact": count_exact,
                "type_exact": type_exact,
                "type_quota_applicable": type_quota_applicable,
                "posterior_type_applicable": bool(
                    sampling_item.get("type_applicable", False)
                ),
                "posterior_type_passed": bool(
                    sampling_item.get("type_passed", True)
                ),
                "spatial_applicable": bool(
                    sampling_item.get("spatial_applicable", False)
                ),
                "spatial_passed": bool(
                    sampling_item.get("spatial_passed", True)
                ),
                "density_mode": _nested_value(
                    diagnostics,
                    "patch_adaptive_priors",
                    "tissues",
                    str(tissue_id),
                    "density_mode",
                ),
                "accepted_center_probability": item.get(
                    "accepted_center_probability"
                ),
            }
        )

    new_component_counts = _component_counts_by_type(new)
    diagnostic_placed = int(diagnostics.get("placed") or 0)
    new_component_total = sum(new_component_counts.values())
    accepted_centers = _component_centers(new)
    boundary_metrics = _boundary_distance_metrics(change, accepted_centers)
    new_pixels = new > 0
    sampling_audit_applicable = bool(sampling_audit)
    sampling_audit_passed = bool(
        not sampling_audit_applicable or sampling_audit.get("passed", False)
    )
    checks = {
        "approved_target_tissue_hash_preserved": bool(expected_target_sha256)
        and current_target_hash == expected_target_sha256,
        "all_stage_arrays_share_shape": shapes_match,
        "target_nuclei_ids_are_legal": set(observed_ids).issubset(legal_ids),
        "new_nuclei_ids_are_legal": set(new_observed_ids).issubset(legal_ids),
        "new_nuclei_stay_on_biological_tissue": not bool(
            np.any(new_pixels & (tissue == 0))
        ),
        "probnet_reported_tissues": bool(tissue_diagnostics),
        "exact_target_count_per_tissue": all_exact_counts,
        "exact_type_quota_per_tissue": all_exact_types,
        "probnet_sampling_alignment_passed": sampling_audit_passed,
        "new_component_count_matches_probnet": (
            new_component_total == diagnostic_placed
        ),
    }
    return {
        "schema_version": 2,
        "passed": all(checks.values()),
        "checks": checks,
        "target_tissue_sha256": current_target_hash,
        "expected_target_tissue_sha256": expected_target_sha256,
        "target_nuclei_ids": observed_ids,
        "new_nuclei_ids": new_observed_ids,
        "target_nucleus_components_by_type": _component_counts_by_type(nuclei),
        "new_nucleus_components_by_type": new_component_counts,
        "new_nucleus_component_count": new_component_total,
        "probnet_reported_placed": diagnostic_placed,
        "type_quota_applicable": bool(any_type_quota),
        "sampling_audit_applicable": sampling_audit_applicable,
        "sampling_audit": sampling_audit,
        "per_tissue": exact_count_rows,
        "accepted_center_boundary_distance": boundary_metrics,
        "shape_sources": diagnostics.get("placed_by_shape_source") or {},
        "reference_pool": diagnostics.get("reference_pool") or {},
        "count_policy": _nested_value(
            diagnostics, "patch_adaptive_priors", "count_policy"
        ),
        "type_policy": _nested_value(
            diagnostics, "patch_adaptive_priors", "type_policy"
        ),
        "candidate_queue_policy": sorted(
            {
                str(item.get("candidate_queue_policy"))
                for item in tissue_diagnostics.values()
                if isinstance(item, Mapping)
                and item.get("candidate_queue_policy")
            }
        ),
        "retry_tail_policy": sorted(
            {
                str(item.get("retry_tail_policy"))
                for item in tissue_diagnostics.values()
                if isinstance(item, Mapping) and item.get("retry_tail_policy")
            }
        ),
    }


def render_nuclei_review_panel(
    *,
    target_tissue: np.ndarray,
    target_nuclei: np.ndarray,
    new_nuclei: np.ndarray,
    change_region: np.ndarray,
    diagnostics: Mapping[str, Any],
    case: Mapping[str, Any],
    audit: Mapping[str, Any],
    probnet_heatmap_path: str | Path,
    accepted_centers_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Render one large six-view nuclei-condition review panel."""

    tissue_rgb = Image.fromarray(
        id_mask_to_llm_preview_rgb(target_tissue), "RGB"
    )
    width, height = tissue_rgb.size
    nuclei_overlay = Image.fromarray(
        _nuclei_overlay_rgb(target_tissue, target_nuclei), "RGB"
    )
    changed_boundary = _change_overlay(nuclei_overlay, change_region)
    zoom = _changed_region_zoom(
        np.asarray(nuclei_overlay),
        change_region,
        output_size=(width, height),
    )
    probnet_heatmap = _load_rgb_or_placeholder(
        probnet_heatmap_path, size=(width, height), label="ProbNet heatmap missing"
    )
    accepted_centers = _load_rgb_or_placeholder(
        accepted_centers_path,
        size=(width, height),
        label="Accepted centers missing",
    )
    columns = [
        ("Target tissue mask", tissue_rgb),
        ("Target tissue + nuclei", nuclei_overlay),
        ("Changed-region zoom", zoom),
        ("Changed-region boundary", changed_boundary),
        ("Raw ProbNet P(nucleus)", probnet_heatmap),
        ("ProbNet + accepted centers", accepted_centers),
    ]

    header_height = 335
    caption_height = 34
    panel = Image.new(
        "RGB",
        (
            width * 3,
            header_height + 2 * (caption_height + height),
        ),
        (248, 248, 248),
    )
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    status = "AUTO CHECK PASS" if audit["passed"] else "AUTO CHECK NEEDS REVIEW"
    target_total = sum(
        int(row.get("target_count") or 0) for row in audit.get("per_tissue", [])
    )
    placed_total = sum(
        int(row.get("placed") or 0) for row in audit.get("per_tissue", [])
    )
    quota = _sum_count_maps(
        row.get("target_by_type") or {} for row in audit.get("per_tissue", [])
    )
    prior_rows = _nuclei_prior_rows(diagnostics)
    boundary = audit.get("accepted_center_boundary_distance") or {}
    lines = [
        (
            f"{case.get('review_index') or case.get('case_id', '')} | "
            f"{case.get('dataset', '')} | {case.get('primitive', '')} | {status}"
        ),
        str(case.get("instruction") or ""),
        (
            f"target/actual={target_total}/{placed_total} | "
            f"new components={audit.get('new_nucleus_component_count', 0)} | "
            f"type quota={_format_type_counts(quota)}"
        ),
        (
            "shape source="
            f"{audit.get('shape_sources', {})} | "
            f"boundary distance median={boundary.get('median_px')} px | "
            f"within4={boundary.get('within_4px_fraction')}"
        ),
        "density/support: " + ("; ".join(prior_rows) if prior_rows else "n/a"),
        (
            f"count={audit.get('count_policy')} | "
            f"queue={audit.get('candidate_queue_policy')} | "
            f"tail={audit.get('retry_tail_policy')}"
        ),
        (
            "parent target sha256="
            f"{audit.get('target_tissue_sha256', '')[:20]}... | "
            "image generation not started"
        ),
    ]
    y = 10
    for index, line in enumerate(lines):
        fill = (155, 20, 20) if index == 0 and not audit["passed"] else (20, 20, 20)
        for wrapped in _wrap_text(str(line), max_chars=205):
            draw.text((12, y), wrapped, fill=fill, font=font)
            y += 18
    legend_y = min(header_height - 32, y + 4)
    legend_x = 12
    for raw_id, name in CELL_CLASSES.items():
        color = tuple(int(value) for value in CELL_COLOR_MAP[raw_id])
        draw.rectangle(
            (legend_x, legend_y, legend_x + 18, legend_y + 14),
            fill=color,
            outline=(30, 30, 30),
        )
        draw.text(
            (legend_x + 23, legend_y + 2),
            f"{raw_id} {name}",
            fill=(20, 20, 20),
            font=font,
        )
        legend_x += 190

    for index, (caption, image) in enumerate(columns):
        row = index // 3
        column = index % 3
        x = column * width
        y0 = header_height + row * (caption_height + height)
        draw.rectangle(
            (x, y0, x + width - 1, y0 + caption_height - 1),
            fill=(230, 232, 235),
        )
        draw.text((x + 10, y0 + 10), caption, fill=(20, 20, 20), font=font)
        panel.paste(image, (x, y0 + caption_height))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    return output


def audit_target_mask(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    profile: str,
    case: Mapping[str, Any],
    phase3_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run generic semantic, component, geometry, and provenance checks."""

    source = np.asarray(source_mask)
    target = np.asarray(target_mask)
    if source.shape != target.shape:
        raise ValueError("Source and target tissue masks must have identical shapes.")
    schema = MaskProfileSchema.from_reference_profile(profile)
    change = source != target
    changed_pixels = int(np.count_nonzero(change))
    image_pixels = int(change.size)
    components, component_count = ndimage.label(
        change, structure=np.ones((3, 3), dtype=np.uint8)
    )
    component_sizes = (
        np.bincount(components.ravel())[1:].astype(int).tolist()
        if component_count
        else []
    )

    source_labels = tuple(str(item) for item in case.get("source_labels", ()))
    target_label = str(case.get("target_label") or "")
    raw_target_labels = case.get("target_labels")
    target_labels = (
        tuple(str(item) for item in raw_target_labels)
        if isinstance(raw_target_labels, (list, tuple))
        else ((target_label,) if target_label else ())
    )
    allowed_source_ids = _resolve_ids(schema, source_labels)
    allowed_target_ids = _resolve_ids(schema, target_labels)
    changed_source_ids = sorted(int(value) for value in np.unique(source[change]))
    changed_target_ids = sorted(int(value) for value in np.unique(target[change]))

    transition_source_valid = bool(changed_pixels) and (
        not allowed_source_ids
        or set(changed_source_ids).issubset(allowed_source_ids)
    )
    transition_target_valid = bool(changed_pixels) and (
        not allowed_target_ids
        or set(changed_target_ids).issubset(allowed_target_ids)
    )
    untouched_labels_preserved = _untouched_labels_preserved(
        source=source,
        change=change,
        allowed_source_ids=allowed_source_ids,
    )

    expected_bucket = case.get("expected_area_bucket")
    legal_region = (
        np.isin(source, sorted(allowed_source_ids))
        if allowed_source_ids
        else np.ones(source.shape, dtype=bool)
    )
    legal_pixels = int(np.count_nonzero(legal_region))
    changed_fraction_legal = (
        float(changed_pixels / legal_pixels) if legal_pixels else 0.0
    )
    grade_transition = _is_grade_transition(str(case.get("primitive") or ""))
    grade_whole_components = (
        _whole_source_components_changed(
            source=source,
            change=change,
            source_ids=allowed_source_ids,
        )
        if grade_transition
        else True
    )
    strength_denominator, denominator_policy = _strength_denominator(
        source=source,
        schema=schema,
        profile=profile,
        primitive=str(case.get("primitive") or ""),
        fallback_pixels=legal_pixels,
    )
    changed_fraction_strength = (
        float(changed_pixels / strength_denominator)
        if strength_denominator
        else 0.0
    )
    changed_fraction_image = (
        float(changed_pixels / image_pixels) if image_pixels else 0.0
    )
    minimum_changed_fraction_image = case.get(
        "minimum_changed_area_fraction_image"
    )
    image_area_floor_valid = (
        minimum_changed_fraction_image is None
        or changed_fraction_image >= float(minimum_changed_fraction_image)
    )
    area_valid = _bucket_contains(expected_bucket, changed_fraction_strength)
    area_exception = False
    if not area_valid and grade_transition and grade_whole_components:
        area_exception = _whole_component_area_overshoot(
            source=source,
            change=change,
            source_ids=allowed_source_ids,
            expected_bucket=expected_bucket,
            legal_pixels=legal_pixels,
        )
        area_valid = area_exception
    geometry = _geometry_findings(change, components, component_sizes)

    projection_mode = str(
        (phase3_info or {}).get("projection_mode")
        or case.get("projection_mode")
        or ""
    )
    projection_valid = projection_mode == "organic_v2"
    checks = {
        "non_empty_change": changed_pixels > 0,
        "projection_is_organic_v2": projection_valid,
        "changed_source_labels_match_instruction": transition_source_valid,
        "changed_target_labels_match_instruction": transition_target_valid,
        "unrequested_source_labels_preserved": untouched_labels_preserved,
        "changed_area_matches_strength_bucket": area_valid,
        "changed_area_meets_image_fraction_floor": image_area_floor_valid,
        "grade_transition_uses_whole_source_components": grade_whole_components,
        "no_exact_rectangle_or_diamond_components": not geometry[
            "geometric_components"
        ],
        "no_abnormal_isolated_components": not geometry[
            "abnormal_isolated_components"
        ],
    }
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "changed_pixels": changed_pixels,
        "changed_area_fraction_image": changed_fraction_image,
        "minimum_changed_area_fraction_image": minimum_changed_fraction_image,
        "changed_area_fraction_legal_source": changed_fraction_legal,
        "changed_area_fraction_strength_denominator": changed_fraction_strength,
        "strength_denominator_pixels": int(strength_denominator),
        "strength_denominator_policy": denominator_policy,
        "expected_area_bucket": expected_bucket,
        "changed_area_bucket_exception": (
            "template_selected_whole_component_overshoot"
            if area_exception
            else None
        ),
        "component_count": int(component_count),
        "component_sizes_px": sorted(component_sizes, reverse=True),
        "changed_source_ids": changed_source_ids,
        "changed_target_ids": changed_target_ids,
        "expected_source_ids": sorted(allowed_source_ids),
        "expected_target_ids": sorted(allowed_target_ids),
        "geometry": geometry,
        "projection_mode": projection_mode,
        "frozen_target_mask_consumed": False,
    }


def render_mask_review_panel(
    *,
    source_image_path: str | Path,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    change_region: np.ndarray,
    case: Mapping[str, Any],
    lock: Mapping[str, Any],
    audit: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Render one large five-view mask review panel."""

    source_image = Image.open(source_image_path).convert("RGB")
    width, height = source_image.size
    source_rgb = Image.fromarray(id_mask_to_llm_preview_rgb(source_mask), "RGB")
    target_rgb = Image.fromarray(id_mask_to_llm_preview_rgb(target_mask), "RGB")
    if source_rgb.size != source_image.size:
        source_rgb = source_rgb.resize(source_image.size, Image.Resampling.NEAREST)
        target_rgb = target_rgb.resize(source_image.size, Image.Resampling.NEAREST)
    change = np.asarray(change_region, dtype=bool)
    if change.shape[::-1] != source_image.size:
        change = np.asarray(
            Image.fromarray(change.astype(np.uint8) * 255).resize(
                source_image.size, Image.Resampling.NEAREST
            )
        ) > 0

    overlay = _change_overlay(source_image, change)
    diff = _transition_rgb(source_mask, target_mask, change)
    diff_image = Image.fromarray(diff, "RGB")
    if diff_image.size != source_image.size:
        diff_image = diff_image.resize(source_image.size, Image.Resampling.NEAREST)

    columns = [
        ("Source H&E", source_image),
        ("Source tissue mask", source_rgb),
        ("Target tissue mask", target_rgb),
        ("Changed-region contour", overlay),
        ("Source -> target diff", diff_image),
    ]
    header_height = 205
    caption_height = 34
    panel = Image.new(
        "RGB",
        (width * len(columns), header_height + caption_height + height),
        (248, 248, 248),
    )
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    case_number = str(case.get("review_index") or case.get("case_id") or "")
    title = (
        f"{case_number} | {case.get('dataset', '')} | "
        f"{case.get('primitive', '')} | {case.get('strength', '')}"
    )
    instruction = str(case.get("instruction") or "")
    changed_pct = 100.0 * float(audit["changed_area_fraction_image"])
    status = "AUTO CHECK PASS" if audit["passed"] else "AUTO CHECK NEEDS REVIEW"
    detail = (
        f"{status} | changed={changed_pct:.2f}% | "
        f"components={audit['component_count']} | "
        f"from={audit['changed_source_ids']} -> to={audit['changed_target_ids']}"
    )
    provenance = (
        f"organic_v2 | model={lock['api_model']} | seed={lock['organic_seed']} | "
        f"target sha256={lock['asset_sha256']['target_tissue'][:16]}... | "
        "frozen target consumed=false"
    )
    failed = [
        name for name, passed in audit["checks"].items() if not bool(passed)
    ]
    failed_text = "failed checks: " + (", ".join(failed) if failed else "none")
    lines = [title, instruction, detail, provenance, failed_text]
    y = 10
    for index, line in enumerate(lines):
        fill = (150, 20, 20) if index == 4 and failed else (20, 20, 20)
        for wrapped in _wrap_text(line, max_chars=190):
            draw.text((12, y), wrapped, fill=fill, font=font)
            y += 18
    for index, (caption, image) in enumerate(columns):
        x = index * width
        draw.rectangle(
            (x, header_height, x + width - 1, header_height + caption_height - 1),
            fill=(230, 232, 235),
        )
        draw.text((x + 10, header_height + 10), caption, fill=(20, 20, 20), font=font)
        panel.paste(image, (x, header_height + caption_height))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    return output


def _resolve_ids(schema: MaskProfileSchema, labels: tuple[str, ...]) -> set[int]:
    ids: set[int] = set()
    for label in labels:
        if label in schema.readable_labels:
            ids.update(int(value) for value in schema.resolve_fine_ids(label))
    return ids


def _untouched_labels_preserved(
    *,
    source: np.ndarray,
    change: np.ndarray,
    allowed_source_ids: set[int],
) -> bool:
    if not allowed_source_ids:
        return True
    return not bool(np.any(change & ~np.isin(source, sorted(allowed_source_ids))))


def _bucket_contains(bucket: Any, value: float) -> bool:
    if not isinstance(bucket, (list, tuple)) or len(bucket) != 2:
        return True
    lower, upper = float(bucket[0]), float(bucket[1])
    tolerance = max(0.02, 0.20 * max(abs(lower), abs(upper)))
    return lower - tolerance <= value <= upper + tolerance


def _strength_denominator(
    *,
    source: np.ndarray,
    schema: MaskProfileSchema,
    profile: str,
    primitive: str,
    fallback_pixels: int,
) -> tuple[int, str]:
    """Use the same primitive-specific magnitude denominator as the product."""

    try:
        recipe = load_recipe(default_recipe_path_for_profile(profile))
        primitive_config = primitive_config_by_name(recipe, primitive)
    except (KeyError, TypeError, ValueError):
        return int(fallback_pixels), "instruction_source_labels_fallback"
    return (
        int(strength_denominator_pixels(source, primitive_config, schema)),
        f"product_recipe:{primitive}",
    )


def _is_grade_transition(primitive: str) -> bool:
    normalized = primitive.lower()
    return "grade" in normalized or "gleason" in normalized


def _whole_source_components_changed(
    *,
    source: np.ndarray,
    change: np.ndarray,
    source_ids: set[int],
) -> bool:
    if not source_ids:
        return False
    for source_id in source_ids:
        labeled, count = ndimage.label(
            source == source_id, structure=np.ones((3, 3), dtype=np.uint8)
        )
        for component_id in range(1, count + 1):
            component = labeled == component_id
            overlap = int(np.count_nonzero(component & change))
            if overlap and overlap != int(np.count_nonzero(component)):
                return False
    return True


def _geometry_findings(
    change: np.ndarray,
    labeled: np.ndarray,
    component_sizes: list[int],
) -> dict[str, Any]:
    changed_pixels = int(np.count_nonzero(change))
    isolated_threshold = _isolated_changed_component_threshold(changed_pixels)
    geometric: list[dict[str, Any]] = []
    isolated: list[dict[str, Any]] = []
    for component_id, area in enumerate(component_sizes, start=1):
        component = labeled == component_id
        ys, xs = np.where(component)
        if not len(xs):
            continue
        bbox_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        rectangularity = float(area / max(1, bbox_area))
        rectangle_like = bool(area >= 64 and rectangularity >= 0.97)
        diamond_iou = _ideal_diamond_iou(component, xs, ys)
        solidity = _component_solidity(component)
        diamond_like = bool(
            area >= 64 and diamond_iou >= 0.80 and solidity >= 0.98
        )
        if rectangle_like or diamond_like:
            geometric.append(
                {
                    "component_id": component_id,
                    "area_px": int(area),
                    "rectangularity": rectangularity,
                    "ideal_diamond_iou": diamond_iou,
                    "solidity": solidity,
                    "shape": "rectangle" if rectangle_like else "diamond",
                }
            )
        if area < isolated_threshold:
            isolated.append(
                {
                    "component_id": component_id,
                    "area_px": int(area),
                    "threshold_px": isolated_threshold,
                }
            )
    return {
        "geometric_components": geometric,
        "abnormal_isolated_components": isolated,
        "isolated_component_threshold_px": isolated_threshold,
    }


def _isolated_changed_component_threshold(changed_pixels: int) -> int:
    return max(16, int(math.ceil(0.002 * max(int(changed_pixels), 1))))


def _ideal_diamond_iou(component: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    center_x = 0.5 * float(xs.min() + xs.max())
    center_y = 0.5 * float(ys.min() + ys.max())
    radius_x = max(1.0, 0.5 * float(xs.max() - xs.min() + 1))
    radius_y = max(1.0, 0.5 * float(ys.max() - ys.min() + 1))
    grid_y, grid_x = np.indices(component.shape)
    diamond = (
        np.abs(grid_x - center_x) / radius_x
        + np.abs(grid_y - center_y) / radius_y
    ) <= 1.0
    union = int(np.count_nonzero(component | diamond))
    return (
        float(np.count_nonzero(component & diamond) / union)
        if union
        else 0.0
    )


def _component_solidity(component: np.ndarray) -> float:
    boundary = component & ~ndimage.binary_erosion(component)
    ys, xs = np.where(boundary)
    if len(xs) < 3:
        return 0.0
    points = np.column_stack((xs, ys))
    try:
        hull_area = float(ConvexHull(points).volume)
    except QhullError:
        return 0.0
    return float(np.count_nonzero(component) / max(hull_area, 1.0))


def _whole_component_area_overshoot(
    *,
    source: np.ndarray,
    change: np.ndarray,
    source_ids: set[int],
    expected_bucket: Any,
    legal_pixels: int,
) -> bool:
    if legal_pixels <= 0:
        return False
    bounds = _bucket_bounds(expected_bucket)
    if bounds is None:
        return False
    _, upper = bounds
    changed_fraction = float(np.count_nonzero(change) / legal_pixels)
    if changed_fraction <= upper:
        return False
    upper_pixels = float(upper * legal_pixels)
    for source_id in source_ids:
        labeled, count = ndimage.label(
            source == source_id,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        for component_id in range(1, count + 1):
            component = labeled == component_id
            if np.any(component & change) and np.count_nonzero(component) > upper_pixels:
                return True
    return False


def _bucket_bounds(value: Any) -> tuple[float, float] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return float(value[0]), float(value[1])
    return None


def _change_overlay(source: Image.Image, change: np.ndarray) -> Image.Image:
    from scipy import ndimage as scipy_ndimage

    rgb = np.asarray(source, dtype=np.uint8).copy()
    boundary = change & ~scipy_ndimage.binary_erosion(change)
    halo = scipy_ndimage.binary_dilation(boundary, iterations=2)
    rgb[halo] = np.array([255, 220, 0], dtype=np.uint8)
    rgb[boundary] = np.array([255, 40, 40], dtype=np.uint8)
    return Image.fromarray(rgb, "RGB")


def _read_probnet_diagnostics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ProbNet diagnostics do not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError(f"ProbNet diagnostics must contain at least one gamma: {path}")
    return payload[0]


def _load_grayscale(path: str | Path) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Required stage asset does not exist: {source}")
    return np.asarray(Image.open(source).convert("L"), dtype=np.uint8)


def _normalized_count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): int(count)
        for key, count in value.items()
        if int(count) != 0
    }


def _component_counts_by_type(mask: np.ndarray) -> dict[str, int]:
    source = np.asarray(mask)
    counts: dict[str, int] = {}
    structure = np.ones((3, 3), dtype=np.uint8)
    for raw_id in CELL_CLASSES:
        _, count = ndimage.label(source == raw_id, structure=structure)
        if count:
            counts[str(raw_id)] = int(count)
    return counts


def _component_centers(mask: np.ndarray) -> list[tuple[float, float, int]]:
    source = np.asarray(mask)
    centers: list[tuple[float, float, int]] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for raw_id in CELL_CLASSES:
        labels, count = ndimage.label(source == raw_id, structure=structure)
        if not count:
            continue
        for center_y, center_x in ndimage.center_of_mass(
            source == raw_id, labels, range(1, count + 1)
        ):
            centers.append((float(center_y), float(center_x), int(raw_id)))
    return centers


def _boundary_distance_metrics(
    change_region: np.ndarray,
    centers: list[tuple[float, float, int]],
) -> dict[str, Any]:
    change = np.asarray(change_region, dtype=bool)
    if not centers or not np.any(change):
        return {
            "count": len(centers),
            "median_px": None,
            "mean_px": None,
            "within_4px_fraction": None,
        }
    boundary = change & ~ndimage.binary_erosion(change)
    distance = ndimage.distance_transform_edt(~boundary)
    values = []
    for center_y, center_x, _ in centers:
        y = int(np.clip(round(center_y), 0, change.shape[0] - 1))
        x = int(np.clip(round(center_x), 0, change.shape[1] - 1))
        values.append(float(distance[y, x]))
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median_px": round(float(np.median(array)), 3),
        "mean_px": round(float(np.mean(array)), 3),
        "within_4px_fraction": round(float(np.mean(array <= 4.0)), 4),
    }


def _nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nuclei_overlay_rgb(
    target_tissue: np.ndarray,
    target_nuclei: np.ndarray,
) -> np.ndarray:
    tissue_rgb = id_mask_to_llm_preview_rgb(target_tissue).astype(np.float32)
    rgb = np.round(0.72 * tissue_rgb + 0.28 * 255.0).astype(np.uint8)
    nuclei = np.asarray(target_nuclei)
    for raw_id, color in CELL_COLOR_MAP.items():
        rgb[nuclei == raw_id] = np.asarray(color, dtype=np.uint8)
    return rgb


def _changed_region_zoom(
    rgb: np.ndarray,
    change_region: np.ndarray,
    *,
    output_size: tuple[int, int],
) -> Image.Image:
    change = np.asarray(change_region, dtype=bool)
    if not np.any(change):
        return Image.fromarray(rgb, "RGB")
    ys, xs = np.where(change)
    pad = max(16, int(round(0.10 * max(xs.max() - xs.min(), ys.max() - ys.min(), 1))))
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(rgb.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(rgb.shape[0], int(ys.max()) + pad + 1)
    crop = Image.fromarray(rgb[y0:y1, x0:x1], "RGB")
    return crop.resize(output_size, Image.Resampling.NEAREST)


def _load_rgb_or_placeholder(
    path: str | Path,
    *,
    size: tuple[int, int],
    label: str,
) -> Image.Image:
    source = Path(path)
    if source.exists():
        return Image.open(source).convert("RGB").resize(
            size, Image.Resampling.NEAREST
        )
    image = Image.new("RGB", size, (235, 235, 235))
    ImageDraw.Draw(image).text((12, 12), label, fill=(130, 20, 20))
    return image


def _sum_count_maps(values: Any) -> dict[str, int]:
    total: dict[str, int] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key, count in value.items():
            total[str(key)] = total.get(str(key), 0) + int(count)
    return {key: count for key, count in total.items() if count}


def _format_type_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(
        f"{key}:{counts.get(str(key), 0)}" for key in CELL_CLASSES
    )


def _nuclei_prior_rows(diagnostics: Mapping[str, Any]) -> list[str]:
    tissues = _nested_value(diagnostics, "patch_adaptive_priors", "tissues")
    if not isinstance(tissues, Mapping):
        return []
    rows = []
    for tissue_id, raw in sorted(tissues.items()):
        item = raw if isinstance(raw, Mapping) else {}
        rows.append(
            f"t{tissue_id} {item.get('density_mode', 'unknown')} "
            f"area={item.get('reference_area_px', 0)} "
            f"cells={item.get('local_centroid_count', 0)} "
            f"confidence={_round_optional(item.get('effective_local_confidence'))}"
        )
    return rows


def _round_optional(value: Any) -> Any:
    return round(float(value), 3) if isinstance(value, (int, float)) else None


def _transition_rgb(
    source: np.ndarray,
    target: np.ndarray,
    change: np.ndarray,
) -> np.ndarray:
    source_rgb = id_mask_to_llm_preview_rgb(source)
    target_rgb = id_mask_to_llm_preview_rgb(target)
    background = np.full(source_rgb.shape, 244, dtype=np.uint8)
    if np.any(change):
        blended = np.round(
            0.35 * source_rgb.astype(np.float32)
            + 0.65 * target_rgb.astype(np.float32)
        ).astype(np.uint8)
        background[change] = blended[change]
        boundary = change & ~ndimage.binary_erosion(change)
        background[boundary] = np.array([255, 255, 255], dtype=np.uint8)
    return background


def _structured_response_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    names = {
        "api_response.json",
        "semantic_diff_normalized.json",
        "llm_response.json",
        "validated_proposal.json",
    }
    return sorted(path for path in root.rglob("*.json") if path.name in names)


def _phase3_execution_info(
    root: Path,
    *,
    tissue_info: Mapping[str, Any],
) -> dict[str, Any]:
    summaries = [
        _read_json_if_exists(path)
        for path in sorted(root.rglob("execution_summary.json"))
    ]
    projection_modes = {
        str(summary.get("projection_mode") or "")
        for summary in summaries
        if summary.get("projection_mode")
    }
    tissue_projection = str(tissue_info.get("projection_mode") or "")
    if tissue_projection:
        projection_modes.add(tissue_projection)
    return {
        "projection_mode": (
            next(iter(projection_modes)) if len(projection_modes) == 1 else ""
        ),
        "execution_summary_count": len(summaries),
        "execution_summary_projection_modes": sorted(projection_modes),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
