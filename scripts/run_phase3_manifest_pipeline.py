#!/usr/bin/env python3
"""Batch runner for Phase 3 UI mask-edit debug manifests.

The runner intentionally reuses the non-interactive backend functions from
``scripts.phase3_end_to_end_ui`` so batch runs follow the same organic_v2 contour
path as the Gradio UI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase3_mask_edit.audit.staged_review import (  # noqa: E402
    build_mask_stage_review,
    build_nuclei_stage_review,
    normalize_stop_after,
    sha256_file,
    sha256_text,
)

REPO_DEFAULT_MANIFEST = (
    REPO_ROOT
    / "docs"
    / "superpowers"
    / "debug_sets"
    / "phase3_ui_mask_edit_debug_manifest.json"
)
REMOTE_DEFAULT_MANIFEST = Path(
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/dubug/phase3_ui_mask_edit_debug_manifest.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "phase3_manifest_pipeline"
POSIX_ROOT_ANCHORS = ("/data/", "/home/", "/mnt/", "/workspace/", "/scratch/")
PATCH_DIR_BY_DATASET = {
    "BCSS": "BCSS_PATCHES",
    "GlaS": "GlaS_PATCHES",
    "IGNITE": "IGNITE_PATCHES",
    "ORCA": "ORCA_PATCHES",
    "PANDA": "PANDA_PATCHES",
    "PUMA": "PUMA_PATCHES",
}
FIELD_SUBDIRS = {
    "source_image": ("images",),
    "source_tissue_mask": ("tissue_masks", "masks"),
    "source_nuclei_mask": ("nuclei_masks", "cell_masks", "cells"),
}


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    args.stop_after = normalize_stop_after(args.stop_after)
    manifest = _load_json(args.manifest)
    runtime = _mapping(manifest.get("runtime"))
    runtime["defaults"] = _mapping(manifest.get("defaults"))
    dataset_roots = _dataset_roots(manifest, args)
    variants = _selected_variants(runtime, args.variants)
    cases = _selected_cases(manifest, args)
    approved_mask_manifest = (
        _load_json(args.approved_mask_manifest)
        if args.approved_mask_manifest is not None
        else None
    )
    approved_mask_entries = _approved_mask_entries(approved_mask_manifest)
    if approved_mask_entries and args.stop_after == "mask":
        raise SystemExit(
            "--approved-mask-manifest is for nuclei/image continuation; "
            "the approved mask stage is already complete."
        )
    approved_nuclei_manifest = (
        _load_json(args.approved_nuclei_manifest)
        if args.approved_nuclei_manifest is not None
        else None
    )
    approved_nuclei_entries = _approved_nuclei_entries(
        approved_nuclei_manifest
    )
    if approved_nuclei_entries:
        if args.stop_after != "image":
            raise SystemExit(
                "--approved-nuclei-manifest is only for image continuation; "
                "the approved nuclei stage is already complete."
            )
        if not approved_mask_entries:
            raise SystemExit(
                "--approved-nuclei-manifest also requires "
                "--approved-mask-manifest."
            )

    batch_dir = args.output_root / (args.run_id or time.strftime("%Y%m%d_%H%M%S"))
    batch_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        {"case": case, "variant": variant}
        for case in cases
        for variant in variants
    ]
    batch_plan = {
        "manifest": str(args.manifest),
        "approved_mask_manifest": (
            str(args.approved_mask_manifest)
            if args.approved_mask_manifest is not None
            else None
        ),
        "approved_nuclei_manifest": (
            str(args.approved_nuclei_manifest)
            if args.approved_nuclei_manifest is not None
            else None
        ),
        "output_root": str(batch_dir),
        "stop_after": args.stop_after,
        "case_count": len(cases),
        "variant_count": len(variants),
        "run_count": len(runs),
        "dataset_roots": {key: _path_text(value) for key, value in dataset_roots.items()},
        "runs": [
            {
                "case_id": item["case"].get("case_id"),
                "dataset": item["case"].get("dataset"),
                "variant_id": item["variant"].get("variant_id"),
                "edit_mode": item["variant"].get("edit_mode"),
            }
            for item in runs
        ],
    }
    _write_json(batch_plan, batch_dir / "batch_plan.json")
    if args.plan_only:
        print(json.dumps(batch_plan, indent=2, ensure_ascii=False))
        return 0

    ui = _load_ui_backend()
    summary: dict[str, Any] = {
        **batch_plan,
        "stop_after": args.stop_after,
        "results": [],
    }
    failed = 0
    for item in runs:
        case = item["case"]
        variant = item["variant"]
        case_id = str(case.get("case_id", "case"))
        variant_id = str(variant.get("variant_id") or variant.get("edit_mode") or "variant")
        run_dir = batch_dir / case_id / variant_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_record: dict[str, Any] = {
            "case_id": case_id,
            "dataset": case.get("dataset"),
            "profile": case.get("profile"),
            "variant_id": variant_id,
            "edit_mode": variant.get("edit_mode"),
            "output_dir": _path_text(run_dir),
        }
        try:
            paths = _resolve_case_paths(case, dataset_roots, require_exists=True)
            _write_json({"case": case, "variant": variant, "paths": _stringify(paths)}, run_dir / "run_config.json")
            state = _prepare_state(
                ui,
                case,
                paths,
                run_dir,
                runtime=runtime,
                args=args,
            )
            if approved_mask_entries:
                approved_entry = approved_mask_entries.get(case_id)
                if approved_entry is None:
                    raise KeyError(
                        f"Approved mask manifest has no entry for {case_id}."
                    )
                state, tissue_info, mask_stage = _resume_approved_mask_stage(
                    ui=ui,
                    state=state,
                    case=case,
                    variant=variant,
                    approved_entry=approved_entry,
                    approved_manifest_path=args.approved_mask_manifest,
                    run_dir=run_dir,
                )
                result_record["tissue"] = tissue_info
                result_record["mask_stage"] = mask_stage
            else:
                state, tissue_info = _run_tissue_stage(
                    ui, state, case, variant, runtime, args
                )
                result_record["tissue"] = tissue_info
                result_record["mask_stage"] = build_mask_stage_review(
                    run_dir=run_dir,
                    case=case,
                    variant=variant,
                    state=state,
                    tissue_info=tissue_info,
                )

            if args.stop_after in {"nuclei", "image"}:
                if approved_nuclei_entries:
                    approved_entry = approved_nuclei_entries.get(case_id)
                    if approved_entry is None:
                        raise KeyError(
                            "Approved nuclei manifest has no entry for "
                            f"{case_id}."
                        )
                    state, cell_info, nuclei_stage = (
                        _resume_approved_nuclei_stage(
                            state=state,
                            case=case,
                            approved_entry=approved_entry,
                            approved_manifest_path=args.approved_nuclei_manifest,
                            run_dir=run_dir,
                        )
                    )
                    result_record["cell"] = cell_info
                    result_record["nuclei_stage"] = nuclei_stage
                else:
                    state, cell_info = _run_cell_stage(
                        ui, state, case, runtime, args
                    )
                    result_record["cell"] = cell_info
                    result_record["nuclei_stage"] = build_nuclei_stage_review(
                        run_dir=run_dir,
                        case=case,
                        state=state,
                        cell_info=cell_info,
                        approved_mask_stage=result_record["mask_stage"],
                    )

            if args.stop_after == "image":
                state, generation_info = _run_generation_stage(ui, state, case, runtime, args)
                result_record["generation"] = generation_info

            result_record["status"] = "completed"
        except Exception as exc:  # noqa: BLE001 - batch runner should record every failing case.
            failed += 1
            result_record["status"] = "failed"
            result_record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json(result_record["error"], run_dir / "error.json")
            if args.fail_fast:
                summary["results"].append(result_record)
                _write_json(summary, batch_dir / "batch_summary.json")
                raise
        summary["results"].append(result_record)
        _write_json(summary, batch_dir / "batch_summary.json")
        print(
            f"[{result_record['status']}] {case_id} / {variant_id} -> {_path_text(run_dir)}",
            flush=True,
        )

    summary["completed"] = sum(1 for item in summary["results"] if item.get("status") == "completed")
    summary["failed"] = failed
    _write_json(summary, batch_dir / "batch_summary.json")
    if args.stop_after == "mask":
        _write_mask_stage_manifest(summary, batch_dir / "mask_stage_manifest.json")
    elif args.stop_after == "nuclei":
        _write_nuclei_stage_manifest(
            summary, batch_dir / "nuclei_stage_manifest.json"
        )
    print(json.dumps({"output_root": _path_text(batch_dir), "completed": summary["completed"], "failed": failed}, indent=2))
    return 1 if failed else 0


def _default_manifest_path() -> Path:
    env_path = os.environ.get("PHASE3_MASK_EDIT_MANIFEST")
    if env_path:
        return Path(env_path)
    if REMOTE_DEFAULT_MANIFEST.exists():
        return REMOTE_DEFAULT_MANIFEST
    return REPO_DEFAULT_MANIFEST


def _load_ui_backend():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import scripts.phase3_end_to_end_ui as ui

    return ui


def _prepare_state(
    ui,
    case: dict[str, Any],
    paths: dict[str, Path],
    output_dir: Path,
    *,
    runtime: dict[str, Any] | None = None,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    verification_cfg = _mapping((runtime or {}).get("verification"))

    def verification_option(name: str, default: Any) -> Any:
        explicit = getattr(args, name, None) if args is not None else None
        return _option(explicit, verification_cfg.get(name), default)

    product_release = str(
        verification_option(
            "product_release", ui.DEFAULT_ONLINE_PRODUCT_RELEASE
        )
    )
    segmentator_release_default = _segmentator_release_from_product_release(
        product_release,
        fallback=ui.DEFAULT_SEGMENTATOR_RELEASE,
    )

    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    image_path = _copy_input(paths["source_image"], inputs_dir / _input_name(paths["source_image"], "source_image.png"))
    tissue_path = _copy_input(paths["source_tissue_mask"], inputs_dir / "source_tissue_mask.png")
    nuclei_path = _copy_input(paths["source_nuclei_mask"], inputs_dir / "source_cell_mask.png")

    image = ui._load_rgb_image(image_path)
    tissue = ui.load_id_mask(tissue_path)
    nuclei = ui._load_uint8_mask(nuclei_path)
    ui._validate_same_size(image, tissue, "source_tissue_mask")
    ui._validate_same_size(image, nuclei, "source_cell_mask")
    stage_paths = ui._save_pre_generation_artifacts(
        output_dir=output_dir,
        reference_image=image,
        reference_tissue=tissue,
        target_tissue=tissue,
        semantic_change_region=ui.np.zeros(tissue.shape, dtype=bool),
        change_region=ui.np.zeros(tissue.shape, dtype=bool),
    )
    return {
        "profile": case.get("profile") or case.get("dataset"),
        "output_dir": _path_text(output_dir),
        "reference_image": _path_text(image_path),
        "reference_tissue_mask": _path_text(tissue_path),
        "reference_nuclei_mask": _path_text(nuclei_path),
        "source_mask_rgb": stage_paths["source_mask_rgb"],
        "target_mask_rgb": stage_paths["source_mask_rgb"],
        "manifest_case_id": case.get("case_id"),
        "verification_runtime": {
            "product_release": product_release,
            "segmentator_env": str(
                verification_option(
                    "segmentator_env", ui.DEFAULT_SEGMENTATOR_ENV
                )
            ),
            "segmentator_release": str(
                verification_option(
                    "segmentator_release", segmentator_release_default
                )
            ),
            "segmentator_python": str(
                verification_option(
                    "segmentator_python", ui.DEFAULT_SEGMENTATOR_PYTHON
                )
            ),
            "segmentator_device": str(
                verification_option(
                    "segmentator_device", ui.DEFAULT_SEGMENTATOR_DEVICE
                )
            ),
            "cellvit_script": str(
                verification_option(
                    "cellvit_script", ui.DEFAULT_CELLVIT_SCRIPT
                )
            ),
            "cellvit_model": str(
                verification_option(
                    "cellvit_model", ui.DEFAULT_CELLVIT_MODEL
                )
            ),
            "cellvit_root": str(
                verification_option("cellvit_root", ui.DEFAULT_CELLVIT_ROOT)
            ),
            "cellvit_python": str(
                verification_option(
                    "cellvit_python", ui.DEFAULT_CELLVIT_PYTHON
                )
            ),
            "cellvit_device": str(
                verification_option(
                    "cellvit_device", ui.DEFAULT_CELLVIT_DEVICE
                )
            ),
        },
    }


def _segmentator_release_from_product_release(
    product_release: str | Path,
    *,
    fallback: str | Path,
) -> Path:
    """Resolve the evaluator model from the same frozen product contract."""

    release_path = Path(product_release)
    if not release_path.is_file():
        return Path(fallback)
    payload = json.loads(release_path.read_text(encoding="utf-8"))
    raw = _mapping(payload.get("verification")).get("segmentator_release")
    if not raw:
        return Path(fallback)
    resolved = Path(str(raw))
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _approved_mask_entries(
    manifest: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    if str(manifest.get("stage") or "") != "mask":
        raise ValueError("Approved mask manifest must have stage='mask'.")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Approved mask manifest must contain non-empty entries.")
    indexed: dict[str, dict[str, Any]] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("Approved mask manifest entries must be objects.")
        case_id = str(raw.get("case_id") or "")
        if not case_id:
            raise ValueError("Approved mask entry is missing case_id.")
        if case_id in indexed:
            raise ValueError(f"Duplicate approved mask case_id: {case_id}")
        if str(raw.get("approval") or "") != "approved":
            raise ValueError(f"Mask entry is not approved: {case_id}")
        if not raw.get("approved_target_sha256"):
            raise ValueError(f"Approved mask entry has no target hash: {case_id}")
        indexed[case_id] = raw
    return indexed


def _approved_nuclei_entries(
    manifest: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    if str(manifest.get("stage") or "") != "nuclei":
        raise ValueError("Approved nuclei manifest must have stage='nuclei'.")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Approved nuclei manifest must contain non-empty entries."
        )
    indexed: dict[str, dict[str, Any]] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("Approved nuclei manifest entries must be objects.")
        case_id = str(raw.get("case_id") or "")
        if not case_id:
            raise ValueError("Approved nuclei entry is missing case_id.")
        if case_id in indexed:
            raise ValueError(f"Duplicate approved nuclei case_id: {case_id}")
        if str(raw.get("approval") or "") != "approved":
            raise ValueError(f"Nuclei entry is not approved: {case_id}")
        if not raw.get("approved_target_nuclei_sha256"):
            raise ValueError(
                f"Approved nuclei entry has no target hash: {case_id}"
            )
        indexed[case_id] = raw
    return indexed


def _resume_approved_mask_stage(
    *,
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    variant: dict[str, Any],
    approved_entry: dict[str, Any],
    approved_manifest_path: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reuse an approved mask asset after validating every provenance hash."""

    case_id = str(case.get("case_id") or "")
    lock_path = Path(str(approved_entry.get("lock_path") or ""))
    lock = _load_json(lock_path)
    expected_hash = str(approved_entry["approved_target_sha256"])
    locked_hash = str(
        (lock.get("asset_sha256") or {}).get("target_tissue") or ""
    )
    lock_approval = _mapping(lock.get("approval"))
    if str(lock_approval.get("status") or "") != "approved":
        raise ValueError(f"Mask lock is not approved for {case_id}: {lock_path}")
    if str(lock_approval.get("approved_target_sha256") or "") != expected_hash:
        raise ValueError(f"Approved target hash differs from lock for {case_id}.")
    if locked_hash != expected_hash:
        raise ValueError(f"Locked target hash differs from manifest for {case_id}.")

    expected_fields = {
        "case_id": case_id,
        "dataset": str(case.get("dataset") or ""),
        "profile": str(case.get("profile") or case.get("dataset") or ""),
        "variant_id": str(
            variant.get("variant_id") or variant.get("edit_mode") or ""
        ),
    }
    for field, expected in expected_fields.items():
        locked = str(lock.get(field) or "")
        if locked and locked != expected:
            raise ValueError(
                f"Approved mask {field} mismatch for {case_id}: "
                f"lock={locked!r}, current={expected!r}."
            )
    instruction = str(case.get("instruction") or "")
    if str(lock.get("instruction_sha256") or "") != sha256_text(instruction):
        raise ValueError(f"Instruction hash mismatch for approved mask {case_id}.")

    asset_hashes = _mapping(lock.get("asset_sha256"))
    source_assets = {
        "source_image": Path(str(state["reference_image"])),
        "source_tissue": Path(str(state["reference_tissue_mask"])),
    }
    for name, source in source_assets.items():
        expected = str(asset_hashes.get(name) or "")
        if not expected or sha256_file(source) != expected:
            raise ValueError(
                f"Source asset hash mismatch for approved mask {case_id}: {name}."
            )

    approved_target_source = Path(
        str(
            approved_entry.get("target_tissue_mask_path")
            or lock.get("target_tissue_mask_path")
            or ""
        )
    )
    if sha256_file(approved_target_source) != expected_hash:
        raise ValueError(f"Approved target asset changed on disk for {case_id}.")
    approved_change_source = Path(str(lock.get("change_region_path") or ""))
    expected_change_hash = str(asset_hashes.get("change_region") or "")
    if (
        not expected_change_hash
        or sha256_file(approved_change_source) != expected_change_hash
    ):
        raise ValueError(f"Approved change-region asset changed for {case_id}.")

    target_path = _copy_input(
        approved_target_source,
        run_dir / "approved_target_mask.png",
    )
    change_path = _copy_input(
        approved_change_source,
        run_dir / "approved_semantic_change_region.png",
    )
    if sha256_file(target_path) != expected_hash:
        raise ValueError(f"Approved target copy changed bytes for {case_id}.")
    if sha256_file(change_path) != expected_change_hash:
        raise ValueError(f"Approved change-region copy changed bytes for {case_id}.")

    reference_tissue = ui.load_id_mask(state["reference_tissue_mask"])
    target_tissue = ui.load_id_mask(target_path)
    change_region = ui.load_change_region(change_path)
    if reference_tissue.shape != target_tissue.shape:
        raise ValueError(f"Approved target shape mismatch for {case_id}.")
    if not ui.np.array_equal(
        reference_tissue != target_tissue,
        ui.np.asarray(change_region, dtype=bool),
    ):
        raise ValueError(
            f"Approved change region does not equal source/target diff for {case_id}."
        )

    provenance = {
        "schema_version": 1,
        "status": "approved_mask_reused",
        "approved_mask_manifest": str(approved_manifest_path),
        "approved_entry_case_id": case_id,
        "original_lock_path": str(lock_path),
        "approved_target_source": str(approved_target_source),
        "approved_change_region_source": str(approved_change_source),
        "approved_target_sha256": expected_hash,
        "approved_change_region_sha256": expected_change_hash,
        "tissue_stage_rerun": False,
    }
    provenance_path = run_dir / "approved_mask_provenance.json"
    _write_json(provenance, provenance_path)
    state.update(
        {
            "target_tissue_mask": str(target_path),
            "semantic_change_region": str(change_path),
            "change_region": str(change_path),
            "approved_mask_provenance": str(provenance_path),
        }
    )
    tissue_info = {
        "status": "approved_mask_reused",
        "projection_mode": "organic_v2",
        "target_tissue_mask": str(target_path),
        "change_region": str(change_path),
        "approved_target_sha256": expected_hash,
        "tissue_stage_rerun": False,
    }
    mask_stage = {
        "stage": "mask",
        "status": "approved_mask_reused",
        "approval": "approved",
        "audit_passed": bool(approved_entry.get("audit_passed", True)),
        "approved_mask_manifest": str(approved_manifest_path),
        "original_lock_path": str(lock_path),
        "lock_path": str(lock_path),
        "target_tissue_mask_path": str(target_path),
        "target_tissue_sha256": expected_hash,
        "approved_target_sha256": expected_hash,
        "change_region_path": str(change_path),
        "change_region_sha256": expected_change_hash,
        "tissue_stage_rerun": False,
    }
    return state, tissue_info, mask_stage


def _resume_approved_nuclei_stage(
    *,
    state: dict[str, Any],
    case: dict[str, Any],
    approved_entry: dict[str, Any],
    approved_manifest_path: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reuse approved nuclei and every downstream input without rerunning ProbNet."""

    case_id = str(case.get("case_id") or "")
    lock_path = Path(str(approved_entry.get("lock_path") or ""))
    lock = _load_json(lock_path)
    approval = _mapping(lock.get("approval"))
    if str(approval.get("status") or "") != "approved":
        raise ValueError(f"Nuclei lock is not approved for {case_id}: {lock_path}")

    expected_fields = {
        "case_id": case_id,
        "dataset": str(case.get("dataset") or ""),
        "profile": str(
            case.get("profile") or case.get("dataset") or ""
        ),
    }
    for field, expected in expected_fields.items():
        locked = str(lock.get(field) or "")
        if locked and locked != expected:
            raise ValueError(
                f"Approved nuclei {field} mismatch for {case_id}: "
                f"lock={locked!r}, current={expected!r}."
            )

    asset_hashes = _mapping(lock.get("asset_sha256"))
    expected_nuclei_hash = str(
        approved_entry.get("approved_target_nuclei_sha256") or ""
    )
    if (
        str(approval.get("approved_target_nuclei_sha256") or "")
        != expected_nuclei_hash
        or str(asset_hashes.get("target_nuclei") or "")
        != expected_nuclei_hash
    ):
        raise ValueError(
            f"Approved target nuclei hash differs from lock for {case_id}."
        )

    current_tissue_hash = sha256_file(state["target_tissue_mask"])
    expected_tissue_hash = str(
        approved_entry.get("approved_target_tissue_sha256")
        or approval.get("approved_target_tissue_sha256")
        or lock.get("parent_target_tissue_sha256")
        or ""
    )
    if (
        not expected_tissue_hash
        or current_tissue_hash != expected_tissue_hash
        or str(asset_hashes.get("target_tissue") or "")
        != expected_tissue_hash
    ):
        raise ValueError(
            f"Approved parent tissue hash differs for {case_id}."
        )

    source_run_dir = Path(
        str(approved_entry.get("run_dir") or lock_path.parent.parent)
    )
    source_paths = {
        "target_nuclei": Path(
            str(
                approved_entry.get("target_nuclei_mask_path")
                or lock.get("target_nuclei_mask_path")
                or ""
            )
        ),
        "new_nuclei": _locked_asset_path(
            lock, "new_nuclei", source_run_dir / "new_nuclei_mask.png"
        ),
        "semantic_change_region": _locked_asset_path(
            lock,
            "semantic_change_region",
            Path(str(state["semantic_change_region"])),
        ),
        "generation_change_region": _locked_asset_path(
            lock,
            "generation_change_region",
            source_run_dir / "change_region.png",
        ),
        "probnet_diagnostics": _locked_asset_path(
            lock,
            "probnet_diagnostics",
            source_run_dir
            / "probnet_cell_fill"
            / "target_nuclei.diagnostics.json",
        ),
        "cell_fill_log": _locked_asset_path(
            lock, "cell_fill_log", source_run_dir / "cell_fill_log.json"
        ),
        "erased_image": _locked_asset_path(
            lock, "erased_image", source_run_dir / "erased_image.png"
        ),
    }
    optional_paths = {
        "retained_nuclei": _locked_asset_path(
            lock,
            "retained_nuclei",
            source_run_dir / "retained_nuclei_mask.png",
        ),
        "target_combined": _locked_asset_path(
            lock,
            "target_combined",
            source_run_dir / "target_combined_mask.png",
        ),
    }
    for name, source in source_paths.items():
        if not source.is_file():
            raise ValueError(
                f"Missing approved {name} asset for {case_id}: {source}"
            )
        expected_hash = str(asset_hashes.get(name) or "")
        if not expected_hash or sha256_file(source) != expected_hash:
            raise ValueError(
                f"Approved {name} asset changed for {case_id}."
            )
    if sha256_file(source_paths["target_nuclei"]) != expected_nuclei_hash:
        raise ValueError(
            f"Approved target nuclei asset changed for {case_id}."
        )
    if (
        sha256_file(state["semantic_change_region"])
        != str(asset_hashes["semantic_change_region"])
    ):
        raise ValueError(
            f"Approved semantic change region differs for {case_id}."
        )

    destinations = {
        "target_nuclei": run_dir / "target_nuclei_mask.png",
        "new_nuclei": run_dir / "new_nuclei_mask.png",
        "semantic_change_region": run_dir
        / "approved_semantic_change_region.png",
        "generation_change_region": run_dir / "change_region.png",
        "probnet_diagnostics": run_dir
        / "probnet_cell_fill"
        / "target_nuclei.diagnostics.json",
        "cell_fill_log": run_dir / "cell_fill_log.json",
        "erased_image": run_dir / "erased_image.png",
    }
    copied: dict[str, Path] = {}
    for name, destination in destinations.items():
        copied[name] = _copy_input(source_paths[name], destination)
        if sha256_file(copied[name]) != str(asset_hashes[name]):
            raise ValueError(
                f"Approved {name} copy changed bytes for {case_id}."
            )
    for name, source in optional_paths.items():
        expected_hash = str(asset_hashes.get(name) or "")
        if source.is_file() and expected_hash:
            destination = run_dir / (
                "retained_nuclei_mask.png"
                if name == "retained_nuclei"
                else "target_combined_mask.png"
            )
            copied[name] = _copy_input(source, destination)
            if sha256_file(copied[name]) != expected_hash:
                raise ValueError(
                    f"Approved {name} copy changed bytes for {case_id}."
                )

    cell_fill = _load_json(copied["cell_fill_log"])
    provenance = {
        "schema_version": 1,
        "status": "approved_nuclei_reused",
        "approved_nuclei_manifest": str(approved_manifest_path),
        "approved_entry_case_id": case_id,
        "original_lock_path": str(lock_path),
        "approved_target_tissue_sha256": expected_tissue_hash,
        "approved_target_nuclei_sha256": expected_nuclei_hash,
        "asset_sha256": {
            name: str(asset_hashes[name]) for name in destinations
        },
        "tissue_stage_rerun": False,
        "nuclei_stage_rerun": False,
    }
    provenance_path = run_dir / "approved_nuclei_provenance.json"
    _write_json(provenance, provenance_path)
    state.update(
        {
            "target_nuclei_mask": str(copied["target_nuclei"]),
            "cell_fill": cell_fill,
            "cell_fill_log": str(copied["cell_fill_log"]),
            "semantic_change_region": str(
                copied["semantic_change_region"]
            ),
            "change_region": str(copied["generation_change_region"]),
            "gland_structure_policy": dict(
                cell_fill.get("gland_structure_policy") or {}
            ),
            "approved_nuclei_provenance": str(provenance_path),
        }
    )
    if "target_combined" in copied:
        state["target_combined_mask"] = str(copied["target_combined"])
    cell_info = {
        **cell_fill,
        "status": "approved_nuclei_reused",
        "target_nuclei_mask": str(copied["target_nuclei"]),
        "new_nuclei_mask": str(copied["new_nuclei"]),
        "nuclei_stage_rerun": False,
    }
    nuclei_stage = {
        "stage": "nuclei",
        "status": "approved_nuclei_reused",
        "approval": "approved",
        "audit_passed": bool(approved_entry.get("audit_passed", True)),
        "approved_nuclei_manifest": str(approved_manifest_path),
        "original_lock_path": str(lock_path),
        "lock_path": str(lock_path),
        "target_nuclei_mask_path": str(copied["target_nuclei"]),
        "target_nuclei_sha256": expected_nuclei_hash,
        "approved_target_nuclei_sha256": expected_nuclei_hash,
        "approved_target_tissue_sha256": expected_tissue_hash,
        "tissue_stage_rerun": False,
        "nuclei_stage_rerun": False,
    }
    return state, cell_info, nuclei_stage


def _locked_asset_path(
    lock: dict[str, Any],
    name: str,
    fallback: Path,
) -> Path:
    return Path(str(lock.get(f"{name}_path") or fallback))


def _run_tissue_stage(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    variant: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _is_no_op_case(case) and not args.execute_no_op_cases:
        return _run_no_op_parser_check(ui, state, case, variant, runtime, args)

    parser_cfg = _mapping(runtime.get("parser"))
    contour_cfg = _mapping(runtime.get("contour"))
    edit_mode = str(variant.get("edit_mode") or "prompt")
    prompt_parser = _option(args.prompt_parser, parser_cfg.get("prompt_parser"), "api")
    instruction_parser = _option(args.instruction_parser, parser_cfg.get("instruction_parser"), "api")
    api_base_url = _option(args.api_base_url, parser_cfg.get("api_base_url"), ui.DEFAULT_API_BASE_URL)
    api_key_env = _option(args.api_key_env, parser_cfg.get("api_key_env"), ui.DEFAULT_API_KEY_ENV)
    api_model = _option(args.api_model, parser_cfg.get("api_model"), ui.DEFAULT_API_MODEL)
    contour_api_base_url = _option(
        args.contour_api_base_url,
        contour_cfg.get("api_base_url"),
        api_base_url,
    )
    contour_api_key_env = _option(
        args.contour_api_key_env,
        contour_cfg.get("api_key_env"),
        api_key_env,
    )
    contour_api_model = _option(
        args.contour_api_model,
        contour_cfg.get("api_model"),
        api_model,
    )
    organic_seed = (
        args.organic_seed
        if args.organic_seed is not None
        else int(case.get("organic_seed", runtime.get("defaults", {}).get("organic_seed", 0)))
    )
    state, tissue_log, _, _ = ui.run_tissue_stage(
        state,
        edit_mode,
        str(case.get("old_prompt", "")),
        str(case.get("new_prompt", "")),
        str(case.get("instruction", "")),
        instruction_parser,
        None,
        "",
        None,
        None,
        prompt_parser,
        api_base_url,
        api_key_env,
        api_model,
        api_model,
        _option(args.qwen_model_path, parser_cfg.get("qwen_model_path"), ""),
        _option(args.qwen_device, parser_cfg.get("qwen_device"), ui.DEFAULT_QWEN_DEVICE),
        bool(args.no_few_shot or parser_cfg.get("no_few_shot", False)),
        "",
        "",
        _option(args.contour_provider, contour_cfg.get("provider"), "api-multimodal"),
        contour_api_base_url,
        contour_api_key_env,
        contour_api_model,
        _option(args.contour_api_image_detail, contour_cfg.get("api_image_detail"), "high"),
        args.fixture_file,
        int(_option(args.max_attempts, contour_cfg.get("max_attempts"), 4)),
        int(_option(args.max_regions, contour_cfg.get("max_regions"), 8)),
        int(_option(args.max_points_per_region, contour_cfg.get("max_points_per_region"), 64)),
        organic_seed,
        bool(args.continue_on_phase3_failure),
    )
    return state, _loads_maybe(tissue_log)


def _run_no_op_parser_check(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    variant: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parser_cfg = _mapping(runtime.get("parser"))
    edit_mode = str(variant.get("edit_mode") or "prompt")
    output_dir = Path(state["output_dir"])
    reference_tissue = ui.load_id_mask(state["reference_tissue_mask"])

    if edit_mode == "instruction":
        semantic_diff, parser_info = _resolve_no_op_instruction_diff(
            ui=ui,
            instruction=str(case.get("instruction", "")),
            parser=_option(args.instruction_parser, parser_cfg.get("instruction_parser"), "api"),
            api_base_url=_option(args.api_base_url, parser_cfg.get("api_base_url"), ui.DEFAULT_API_BASE_URL),
            api_key_env=_option(args.api_key_env, parser_cfg.get("api_key_env"), ui.DEFAULT_API_KEY_ENV),
            api_model=_option(args.api_model, parser_cfg.get("api_model"), ui.DEFAULT_API_MODEL),
            output_dir=output_dir,
        )
    else:
        semantic_diff, parser_info = ui._resolve_prompt_semantic_diff(
            old_prompt=str(case.get("old_prompt", "")),
            new_prompt=str(case.get("new_prompt", "")),
            parser=_option(args.prompt_parser, parser_cfg.get("prompt_parser"), "api"),
            api_base_url=_option(args.api_base_url, parser_cfg.get("api_base_url"), ui.DEFAULT_API_BASE_URL),
            api_key_env=_option(args.api_key_env, parser_cfg.get("api_key_env"), ui.DEFAULT_API_KEY_ENV),
            api_model=_option(args.api_model, parser_cfg.get("api_model"), ui.DEFAULT_API_MODEL),
            qwen_model_path=_option(args.qwen_model_path, parser_cfg.get("qwen_model_path"), ""),
            qwen_device=_option(args.qwen_device, parser_cfg.get("qwen_device"), ui.DEFAULT_QWEN_DEVICE),
            no_few_shot=bool(args.no_few_shot or parser_cfg.get("no_few_shot", False)),
            output_dir=output_dir,
        )

    plan = ui.plan_edit_intents(
        semantic_diff,
        reference_profile=state["profile"],
        old_mask=reference_tissue,
        old_prompt=str(case.get("old_prompt", "")),
        new_prompt=str(case.get("new_prompt", "")),
    )
    planned_primitives = [intent.primitive for intent in plan.intents]
    if planned_primitives:
        raise RuntimeError(
            "No-op case produced executable intents: "
            + ", ".join(planned_primitives)
        )

    return _write_no_op_tissue_stage(
        ui,
        state,
        case,
        variant,
        parser_info=parser_info,
        semantic_diff=semantic_diff,
        plan=plan.to_metadata(),
    )


def _resolve_no_op_instruction_diff(
    *,
    ui,
    instruction: str,
    parser: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return ui._resolve_instruction_semantic_diff(
            instruction=instruction,
            parser=parser,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
            api_model=api_model,
            output_dir=output_dir,
        )
    except Exception as exc:
        if parser == "rule-based" and "Could not infer a supported edit" in str(exc):
            from phase3_mask_edit.parser.semantic_diff import DEFAULT_SEMANTIC_DIFF

            semantic_diff = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
            return semantic_diff, {
                "mode": "instruction_rule_based",
                "status": "no_supported_edit_inferred",
            }
        raise


def _write_no_op_tissue_stage(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    variant: dict[str, Any],
    *,
    parser_info: dict[str, Any] | None = None,
    semantic_diff: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = Path(state["output_dir"])
    reference_image = ui._load_rgb_image(state["reference_image"])
    reference_tissue = ui.load_id_mask(state["reference_tissue_mask"])
    target_path = ui.save_id_mask(reference_tissue, output_dir / "target_mask.png")
    change_region = ui.np.zeros(reference_tissue.shape, dtype=bool)
    stage_paths = ui._save_pre_generation_artifacts(
        output_dir=output_dir,
        reference_image=reference_image,
        reference_tissue=reference_tissue,
        target_tissue=reference_tissue,
        semantic_change_region=change_region,
        change_region=change_region,
    )
    phase3_info = {
        "mode": variant.get("edit_mode"),
        "status": "no_op_manifest_case",
        "primitive": "no_op",
        "expected_primitives": list(case.get("expected_primitives", [])),
        "projection_mode": "no_op",
        "parser": parser_info or {"mode": "not_run"},
        "semantic_diff": semantic_diff,
        "plan": plan,
    }
    ui.save_metadata(phase3_info, output_dir / "phase3_mask_edit" / "execution_summary.json")
    state.update(
        {
            "target_tissue_mask": _path_text(target_path),
            "target_mask_rgb": stage_paths["target_mask_rgb"],
            "change_region": stage_paths["change_region"],
            "phase3": phase3_info,
        }
    )
    info = {
        "status": "tissue_done",
        "edit_mode": variant.get("edit_mode"),
        "primitive": "no_op",
        "changed_area_fraction": 0.0,
        "target_tissue_mask": _path_text(target_path),
        "change_region": stage_paths["change_region"],
    }
    return state, info


def _run_cell_stage(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell_cfg = _mapping(runtime.get("cell"))
    model_paths = _mapping(runtime.get("model_paths"))
    profile = str(case.get("profile") or case.get("dataset") or state.get("profile", "BCSS"))
    dataset = str(case.get("dataset") or profile)
    cell_fill_mode = _option(args.cell_fill_mode, cell_cfg.get("cell_fill_mode"), "preserve")
    probnet_ckpt = _resolve_model_path(
        model_paths,
        "probnet_ckpt",
        args.probnet_ckpt,
        profile,
        dataset,
    )
    nuclei_library = _resolve_model_path(
        model_paths,
        "nuclei_library_template",
        args.nuclei_library_template,
        profile,
        dataset,
    )
    density_scale_json = _resolve_model_path(
        model_paths,
        "density_scale_json_template",
        args.density_scale_json_template,
        profile,
        dataset,
    )
    _validate_probnet_inputs_if_needed(
        cell_fill_mode=cell_fill_mode,
        probnet_ckpt=probnet_ckpt,
        nuclei_library=nuclei_library,
        density_scale_json=density_scale_json,
    )
    state, cell_log, *_ = ui.run_cell_stage(
        state,
        cell_fill_mode,
        _option(args.crossing_cell_policy, cell_cfg.get("crossing_cell_policy"), "delete"),
        probnet_ckpt,
        nuclei_library,
        density_scale_json,
        _option(args.probnet_device, cell_cfg.get("probnet_device"), "auto"),
        str(_option(args.probnet_gamma_values, cell_cfg.get("probnet_gamma_values"), "3")),
    )
    return state, _loads_maybe(cell_log)


def _run_generation_stage(
    ui,
    state: dict[str, Any],
    case: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation_cfg = _mapping(runtime.get("generation"))
    model_paths = _mapping(runtime.get("model_paths"))
    profile = str(case.get("profile") or case.get("dataset") or state.get("profile", "BCSS"))
    state, generation_log, _, _ = ui.run_generation_stage(
        state,
        _option(args.generation_mode, generation_cfg.get("generation_mode"), "dry-run"),
        _option(args.cross_backend, generation_cfg.get("cross_backend"), "cross-v1"),
        float(_option(args.route_threshold, generation_cfg.get("route_threshold"), 0.30)),
        _format_profile_path(
            _option(
                args.pretrained_model_name_or_path,
                model_paths.get("pretrained_model_name_or_path"),
                "",
            ),
            profile,
        ),
        _format_profile_path(_option(args.inpaint_checkpoint, model_paths.get("inpaint_checkpoint"), ""), profile),
        _format_profile_path(_option(args.cross_v1_checkpoint, model_paths.get("cross_v1_checkpoint"), ""), profile),
        _format_profile_path(_option(args.pix2pix_checkpoint, model_paths.get("pix2pix_checkpoint"), ""), profile),
        _option(args.device, generation_cfg.get("device"), "cuda"),
    )
    return state, _loads_maybe(generation_log)


def _resolve_case_paths(
    case: dict[str, Any],
    dataset_roots: dict[str, Path],
    *,
    require_exists: bool = False,
) -> dict[str, Path]:
    return {
        "source_image": _resolve_case_path(
            case,
            "source_image",
            dataset_roots,
            require_exists=require_exists,
        ),
        "source_tissue_mask": _resolve_case_path(
            case,
            "source_tissue_mask",
            dataset_roots,
            require_exists=require_exists,
        ),
        "source_nuclei_mask": _resolve_case_path(
            case,
            "source_nuclei_mask",
            dataset_roots,
            require_exists=require_exists,
        ),
    }


def _resolve_case_path(
    case: dict[str, Any],
    field: str,
    dataset_roots: dict[str, Path],
    *,
    require_exists: bool = False,
) -> Path:
    candidates = _case_path_candidates(case, field, dataset_roots)
    if not require_exists:
        return candidates[0]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(_missing_input_message(case, field, candidates))


def _case_path_candidates(
    case: dict[str, Any],
    field: str,
    dataset_roots: dict[str, Path],
) -> list[Path]:
    dataset = str(case.get("dataset", ""))
    root = dataset_roots.get(dataset)
    raw = str(case.get(field) or "")
    rel = str(case.get(f"{field}_relative") or "")
    basename = _case_path_basename(raw, rel)
    candidates: list[Path] = []

    if rel:
        if root is None:
            raise KeyError(f"No data root configured for dataset {dataset!r}")
        candidates.extend(
            _join_relative_candidates_for_dataset(root, dataset, rel)
        )

    if raw:
        if _looks_windows_absolute(raw):
            derived = _derive_relative_from_path(raw, dataset)
            if derived and root is not None:
                candidates.extend(
                    _join_relative_candidates_for_dataset(root, dataset, derived)
                )
            candidates.append(Path(PureWindowsPath(raw)))
        elif _looks_posix_absolute(raw):
            derived = _derive_relative_from_path(raw, dataset)
            if derived and root is not None:
                candidates.extend(
                    _join_relative_candidates_for_dataset(root, dataset, derived)
                )
            candidates.append(Path(PurePosixPath(raw)))
        else:
            path = Path(raw)
            if path.is_absolute():
                candidates.append(path)
            elif root is not None:
                candidates.extend(
                    _join_relative_candidates_for_dataset(root, dataset, raw)
                )
            else:
                raise KeyError(
                    f"No data root configured for relative {field} in dataset {dataset!r}"
                )
    elif not rel:
        raise KeyError(f"{case.get('case_id', '<case>')} missing {field}")

    if root is not None and basename:
        patch_dir = PATCH_DIR_BY_DATASET.get(dataset)
        for subdir in FIELD_SUBDIRS.get(field, ()):
            candidates.append(_join_manifest_path(root, f"{subdir}/{basename}"))
            if patch_dir:
                candidates.append(_join_manifest_path(root, f"{patch_dir}/{subdir}/{basename}"))
                if _path_name_matches(root, dataset):
                    candidates.append(
                        _join_manifest_path(
                            root.parent,
                            f"{patch_dir}/{subdir}/{basename}",
                        )
                    )

    return _dedupe_paths(candidates)


def _join_relative_candidates_for_dataset(
    root: Path,
    dataset: str,
    relative: str,
) -> list[Path]:
    candidates = [_join_manifest_path(root, relative)]
    patch_dir = PATCH_DIR_BY_DATASET.get(dataset)
    if patch_dir and _relative_starts_with(relative, patch_dir) and _path_name_matches(root, dataset):
        candidates.append(_join_manifest_path(root.parent, relative))
    return candidates


def _relative_starts_with(relative: str, dirname: str) -> bool:
    first = PurePosixPath(relative.replace("\\", "/")).parts[:1]
    return bool(first) and first[0].lower() == dirname.lower()


def _path_name_matches(path: Path, name: str) -> bool:
    return PurePosixPath(str(path).replace("\\", "/")).name.lower() == name.lower()


def _case_path_basename(raw: str, rel: str) -> str:
    source = rel or raw
    if not source:
        return ""
    if _looks_windows_absolute(source):
        return PureWindowsPath(source).name
    return PurePosixPath(source.replace("\\", "/")).name


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = _path_text(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _missing_input_message(case: dict[str, Any], field: str, candidates: list[Path]) -> str:
    case_id = case.get("case_id", "<case>")
    dataset = case.get("dataset", "<dataset>")
    tried = "\n".join(f"  - {_path_text(path)}" for path in candidates)
    return (
        f"Input file does not exist for case_id={case_id}, dataset={dataset}, "
        f"field={field}.\n"
        f"Tried:\n{tried}\n"
        "If this case belongs to a different dataset, edit its `dataset` and "
        "`source_*_relative` fields in the manifest. For your current layout, "
        "runtime.data_roots entries should usually be `/data/wqx/flowedit/data`, "
        "the directory that directly contains BCSS_PATCHES, GlaS_PATCHES, etc."
    )


def _derive_relative_from_path(raw: str, dataset: str) -> str | None:
    patch_dir = PATCH_DIR_BY_DATASET.get(dataset)
    if not patch_dir:
        return None
    for cls in (PureWindowsPath, PurePosixPath):
        parts = cls(raw).parts
        lower_parts = [part.lower() for part in parts]
        try:
            index = lower_parts.index(patch_dir.lower())
        except ValueError:
            continue
        return "/".join(parts[index:])
    return None


def _dataset_roots(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    runtime = _mapping(manifest.get("runtime"))
    roots = {
        str(key): _manifest_path(str(value))
        for key, value in _mapping(runtime.get("data_roots")).items()
        if value
    }
    if args.data_root is not None:
        for dataset in manifest.get("datasets", []):
            roots[str(dataset)] = _manifest_path(args.data_root)
    for key, value in _parse_key_value(args.dataset_root, "--dataset-root").items():
        roots[key] = _manifest_path(value)
    return roots


def _selected_variants(runtime: dict[str, Any], variants_csv: str | None) -> list[dict[str, Any]]:
    raw_variants = runtime.get("edit_variants") or ["prompt", "instruction"]
    variants: list[dict[str, Any]] = []
    for raw in raw_variants:
        if isinstance(raw, str):
            variants.append({"variant_id": raw, "edit_mode": raw})
        elif isinstance(raw, dict):
            edit_mode = raw.get("edit_mode") or raw.get("variant_id")
            if edit_mode:
                variants.append({"variant_id": raw.get("variant_id", edit_mode), **raw, "edit_mode": edit_mode})
    if variants_csv:
        wanted = {part.strip() for part in variants_csv.split(",") if part.strip()}
        variants = [
            variant
            for variant in variants
            if variant.get("variant_id") in wanted or variant.get("edit_mode") in wanted
        ]
    if not variants:
        raise SystemExit("No edit variants selected.")
    return variants


def _selected_cases(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = list(manifest.get("cases", []))
    if args.dataset:
        wanted = set(args.dataset)
        cases = [case for case in cases if case.get("dataset") in wanted]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.get("case_id") in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No manifest cases selected.")
    return cases


def _copy_input(source: Path, target: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _input_name(path: Path, fallback: str) -> str:
    suffix = path.suffix.lower()
    return f"source_image{suffix}" if suffix else fallback


def _is_no_op_case(case: dict[str, Any]) -> bool:
    return str(case.get("primitive", "")).lower() == "no_op" or case.get("expected_primitives") == []


def _option(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return ""


def _format_profile_path(value: Any, profile: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    return text.format(profile=profile, profile_lower=profile.lower())


def _resolve_model_path(
    model_paths: dict[str, Any],
    key: str,
    override: Any,
    profile: str,
    dataset: str,
) -> str:
    if override:
        return _normalize_model_path_key(
            key,
            _format_profile_path(override, profile),
            profile,
        )

    by_dataset_keys = []
    if key.endswith("_template"):
        by_dataset_keys.append(f"{key.removesuffix('_template')}_by_dataset")
    by_dataset_keys.append(f"{key}_by_dataset")

    for mapping_key in by_dataset_keys:
        by_dataset = _mapping(model_paths.get(mapping_key))
        for lookup_key in _profile_lookup_keys(profile, dataset):
            value = by_dataset.get(lookup_key)
            if value:
                return _normalize_model_path_key(
                    key,
                    _format_profile_path(value, profile),
                    profile,
                )

    return _normalize_model_path_key(
        key,
        _format_profile_path(model_paths.get(key, ""), profile),
        profile,
    )


def _normalize_model_path_key(key: str, value: str, profile: str) -> str:
    if not value:
        return ""
    path = value.rstrip("/\\")
    if key == "nuclei_library_template" and "{" not in value:
        if _has_statistics_json(path):
            return path
        dataset_path = f"{path}/{profile}"
        if _has_statistics_json(dataset_path):
            return dataset_path
        return path
    if key == "density_scale_json_template" and not path.lower().endswith(".json"):
        return f"{path}/density_scale_{profile.lower()}.json"
    return value


def _validate_probnet_inputs_if_needed(
    *,
    cell_fill_mode: str,
    probnet_ckpt: str,
    nuclei_library: str,
    density_scale_json: str,
) -> None:
    if cell_fill_mode != "probnet":
        return
    missing: list[str] = []
    if probnet_ckpt and not Path(probnet_ckpt).exists():
        missing.append(f"probnet_ckpt missing: {probnet_ckpt}")
    if nuclei_library and not _has_statistics_json(nuclei_library):
        missing.append(f"nuclei_library missing statistics.json: {nuclei_library}")
    if density_scale_json and not Path(density_scale_json).exists():
        missing.append(f"density_scale_json missing: {density_scale_json}")
    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "ProbNet input validation failed before running generate.py:\n"
            f"{detail}\n"
            "Set runtime.model_paths.nuclei_library_template to the exact "
            "directory containing statistics.json, or use "
            "nuclei_library_by_dataset for per-dataset library directories."
        )


def _has_statistics_json(path: str | Path) -> bool:
    return (Path(path) / "statistics.json").exists()


def _profile_lookup_keys(profile: str, dataset: str) -> tuple[str, ...]:
    values = [
        dataset,
        profile,
        dataset.upper(),
        profile.upper(),
        dataset.lower(),
        profile.lower(),
    ]
    return tuple(dict.fromkeys(value for value in values if value))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _loads_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return payload if isinstance(payload, dict) else {"value": payload}
    return {}


def _stringify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stringify(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    if isinstance(value, Path):
        return _path_text(value)
    return value


def _looks_windows_absolute(value: str) -> bool:
    return len(value) >= 3 and value[1:3] in {":\\", ":/"} and value[0].isalpha()


def _looks_posix_absolute(value: str) -> bool:
    return value.startswith("/") and not _looks_windows_absolute(value)


def _manifest_path(value: str | Path) -> Path:
    text = str(value)
    if _looks_posix_absolute(text):
        return Path(PurePosixPath(text))
    return Path(text)


def _join_manifest_path(root: Path, relative: str) -> Path:
    if _looks_posix_absolute(str(root)):
        parts = PurePosixPath(relative).parts
        return Path(PurePosixPath(str(root), *parts))
    return root / Path(relative)


def _path_text(path: Path) -> str:
    text = str(path)
    if text.startswith("\\") and len(text) > 1:
        candidate = "/" + text.lstrip("\\").replace("\\", "/")
        if any(candidate.startswith(anchor) for anchor in POSIX_ROOT_ANCHORS):
            return candidate
    return path.as_posix()


def _parse_key_value(items: list[str] | None, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{label} expects DATASET=/path, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"{label} expects DATASET=/path, got {item!r}")
        parsed[key] = value
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_stringify(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_mask_stage_manifest(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    entries = []
    for result in summary.get("results", []):
        mask_stage = result.get("mask_stage")
        if not isinstance(mask_stage, dict):
            continue
        entries.append(
            {
                "case_id": result.get("case_id"),
                "condition_id": _condition_id_from_case_id(result.get("case_id")),
                "dataset": result.get("dataset"),
                "variant_id": result.get("variant_id"),
                "run_dir": result.get("output_dir"),
                **mask_stage,
            }
        )
    payload = {
        "schema_version": 1,
        "stage": "mask",
        "approval": {
            "status": "pending",
            "required_entry_count": len(entries),
            "approved_entry_count": 0,
        },
        "frozen_target_mask_consumed": False,
        "entry_count": len(entries),
        "all_automatic_checks_passed": bool(entries)
        and all(bool(item.get("audit_passed")) for item in entries),
        "entries": entries,
    }
    _write_json(payload, output_path)


def _write_nuclei_stage_manifest(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    entries = []
    for result in summary.get("results", []):
        nuclei_stage = result.get("nuclei_stage")
        if not isinstance(nuclei_stage, dict):
            continue
        entries.append(
            {
                "case_id": result.get("case_id"),
                "condition_id": _condition_id_from_case_id(
                    result.get("case_id")
                ),
                "dataset": result.get("dataset"),
                "variant_id": result.get("variant_id"),
                "run_dir": result.get("output_dir"),
                **nuclei_stage,
            }
        )
    payload = {
        "schema_version": 1,
        "stage": "nuclei",
        "approval": {
            "status": "pending",
            "required_entry_count": len(entries),
            "approved_entry_count": 0,
        },
        "entry_count": len(entries),
        "all_automatic_checks_passed": bool(entries)
        and all(bool(item.get("audit_passed")) for item in entries),
        "image_generation_started": False,
        "entries": entries,
    }
    _write_json(payload, output_path)


def _condition_id_from_case_id(case_id: Any) -> str:
    text = str(case_id or "")
    if "_" not in text:
        return text
    return text.split("_", 1)[1]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 mask-edit manifest cases through the UI organic_v2 backend.",
    )
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument(
        "--approved-mask-manifest",
        type=Path,
        help=(
            "Resume from a fully approved mask-stage manifest. Every source, "
            "instruction, target, and change-region hash is verified and the "
            "tissue stage is not rerun."
        ),
    )
    parser.add_argument(
        "--approved-nuclei-manifest",
        type=Path,
        help=(
            "Resume image generation from a fully approved nuclei-stage "
            "manifest. Every downstream input hash is verified and neither "
            "the tissue nor nuclei stage is rerun."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", help="Optional output subdirectory name; defaults to a timestamp.")
    parser.add_argument("--plan-only", action="store_true", help="Write and print the expanded batch plan only.")
    parser.add_argument("--variants", help="Comma-separated variant ids/edit modes, e.g. prompt,instruction.")
    parser.add_argument("--case-id", action="append", help="Run only this case id; may be repeated.")
    parser.add_argument("--dataset", action="append", help="Run only this dataset; may be repeated.")
    parser.add_argument("--limit", type=int, help="Limit cases after filters.")
    parser.add_argument("--data-root", type=Path, help="Override all dataset roots as DATA_ROOT/<dataset>.")
    parser.add_argument("--dataset-root", action="append", help="Override one dataset root, e.g. GlaS=/data/.../GlaS.")
    parser.add_argument(
        "--stop-after",
        choices=("mask", "nuclei", "image", "tissue", "cell", "generation"),
        default="generation",
        help=(
            "Stop after the public mask/nuclei/image stage. Legacy "
            "tissue/cell/generation values remain accepted."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--continue-on-phase3-failure", action="store_true")
    parser.add_argument("--execute-no-op-cases", action="store_true", help="Send no-op cases through parser/contour instead of writing zero-change outputs.")

    parser.add_argument("--prompt-parser", choices=("api", "qwen-local"))
    parser.add_argument("--instruction-parser", choices=("api", "rule-based"))
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--api-model")
    parser.add_argument("--qwen-model-path")
    parser.add_argument("--qwen-device")
    parser.add_argument("--no-few-shot", action="store_true")

    parser.add_argument("--contour-provider", choices=("api-text", "api-multimodal", "fixture"))
    parser.add_argument("--contour-api-base-url")
    parser.add_argument("--contour-api-key-env")
    parser.add_argument("--contour-api-model")
    parser.add_argument("--contour-api-image-detail", choices=("low", "high", "auto"))
    parser.add_argument("--fixture-file", type=Path)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-regions", type=int)
    parser.add_argument("--max-points-per-region", type=int)
    parser.add_argument("--organic-seed", type=int)

    parser.add_argument("--cell-fill-mode", choices=("preserve", "blank", "probnet"))
    parser.add_argument("--crossing-cell-policy", choices=("delete", "keep", "majority"))
    parser.add_argument("--probnet-ckpt")
    parser.add_argument("--nuclei-library-template")
    parser.add_argument("--density-scale-json-template")
    parser.add_argument("--probnet-device")
    parser.add_argument("--probnet-gamma-values")

    parser.add_argument("--generation-mode", choices=("agentic", "dry-run", "auto", "inpaint", "cross-v1"))
    parser.add_argument("--cross-backend", choices=("cross-v1",))
    parser.add_argument("--route-threshold", type=float)
    parser.add_argument("--device")
    parser.add_argument("--pretrained-model-name-or-path")
    parser.add_argument("--inpaint-checkpoint")
    parser.add_argument("--cross-v1-checkpoint")
    parser.add_argument("--pix2pix-checkpoint")

    parser.add_argument("--segmentator-env")
    parser.add_argument("--product-release")
    parser.add_argument("--segmentator-release")
    parser.add_argument("--segmentator-python")
    parser.add_argument("--segmentator-device")
    parser.add_argument("--cellvit-script")
    parser.add_argument("--cellvit-model")
    parser.add_argument("--cellvit-root")
    parser.add_argument("--cellvit-python")
    parser.add_argument("--cellvit-device")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
