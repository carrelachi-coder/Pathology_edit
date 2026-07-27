#!/usr/bin/env python3
"""Generate one frozen ProbNet target-nuclei mask per utility cohort row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_config import get_config
from inpaint_cells.generate import (
    build_parser as build_probnet_parser,
    build_reference_pool,
    compute_patch_adaptive_priors,
    generate_for_gamma,
    load_checkpoint_model,
    load_density_scale,
    predict_prob,
    widen_locally_thin_mask,
)
from inpaint_cells.nuclei_library.library import NucleiLibrary
from inpaint_cells.utils.mask_utils import (
    NUCLEI_RAW_TO_INDEX,
    load_nuclei_mask,
    load_tissue_mask,
    save_nuclei_mask,
)
from phase3_mask_edit.benchmark.pathokid import sha256_file


DEFAULT_CHECKPOINT = Path(
    "/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/"
    "best_epoch29_c29607f1b609accb.pt"
)
DEFAULT_LIBRARY = Path(
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/nuclei_library/BCSS"
)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help="Write updated rows here instead of mutating --cohort-manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Store generated nuclei under OUTPUT_ROOT/targets/SAMPLE_ID.",
    )
    parser.add_argument(
        "--base-cohort-manifest",
        type=Path,
        help=(
            "Optional lower-strength cohort. When provided, retain its target nuclei "
            "and run ProbNet only in each row's incremental_change_region."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--density-scale-json", type=Path)
    parser.add_argument("--minimum-mask-width", type=int, default=33)
    parser.add_argument("--local-density-direct-min-area", type=int, default=20000)
    parser.add_argument("--local-density-direct-min-count", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-count", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _probnet_args(args: argparse.Namespace) -> argparse.Namespace:
    # Parse the canonical ProbNet CLI so inference hyperparameter defaults stay
    # exactly aligned with the production single-sample entry point.
    command = [
            "--dataset",
            "BCSS",
            "--ckpt",
            str(args.checkpoint),
            "--library",
            str(args.library),
            "--input-tissue",
            "unused.png",
            "--edit-region",
            "unused.png",
            "--output",
            "unused.png",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--gamma-values",
            "1.5",
        ]
    if args.density_scale_json is not None:
        command.extend(["--density-scale-json", str(args.density_scale_json)])
    parsed = build_probnet_parser().parse_args(command)
    parsed.component_aware_sampling = True
    parsed.component_quota_policy = "area_largest_remainder"
    parsed.backfill_failed_placements = True
    parsed.max_nucleus_overlap_fraction = 0.0
    parsed.local_density_direct_min_area = int(args.local_density_direct_min_area)
    parsed.local_density_direct_min_count = int(args.local_density_direct_min_count)
    parsed.minimum_mask_width = int(args.minimum_mask_width)
    return parsed


def _retain_complete_reference_cells(
    reference_nuclei: np.ndarray, change_region: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Delete a whole source cell only when its centroid is inside the edit."""

    source = np.asarray(reference_nuclei, dtype=np.uint8)
    changed = np.asarray(change_region, dtype=bool)
    retained = np.zeros_like(source, dtype=np.uint8)
    stats = {
        "policy": "keep_whole_if_centroid_outside_generation_region",
        "source_components": 0,
        "kept_components": 0,
        "deleted_components": 0,
        "crossing_components": 0,
        "inside_change_components": 0,
        "outside_change_components": 0,
    }
    for raw_type in np.unique(source):
        if int(raw_type) == 0:
            continue
        labeled, count = ndimage.label(
            source == raw_type,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        stats["source_components"] += int(count)
        for component_id in range(1, count + 1):
            component = labeled == component_id
            touches_change = bool(np.any(component & changed))
            touches_unchanged = bool(np.any(component & ~changed))
            center_y, center_x = ndimage.center_of_mass(component)
            center_row = int(np.clip(round(center_y), 0, source.shape[0] - 1))
            center_col = int(np.clip(round(center_x), 0, source.shape[1] - 1))
            center_inside = bool(changed[center_row, center_col])
            if touches_change and touches_unchanged:
                stats["crossing_components"] += 1
            elif touches_change:
                stats["inside_change_components"] += 1
            else:
                stats["outside_change_components"] += 1
            if center_inside:
                stats["deleted_components"] += 1
            else:
                retained[component] = source[component]
                stats["kept_components"] += 1
    return retained, stats


def _remap_raw_nuclei(mask: np.ndarray) -> np.ndarray:
    remapped = np.zeros(mask.shape, dtype=np.int64)
    for raw_id, index in NUCLEI_RAW_TO_INDEX.items():
        remapped[mask == raw_id] = index
    return remapped


def _save_binary_mask(mask: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise OSError(f"failed to write mask: {path}")
    return str(path)


def _generate_one(
    row: dict,
    *,
    base_row: dict | None,
    output_root: Path | None,
    model,
    library,
    config,
    checkpoint: Path,
    checkpoint_sha256: str,
    density_scales: dict[int, float],
    device: torch.device,
    probnet_args: argparse.Namespace,
    seed: int,
) -> dict:
    target_tissue = load_tissue_mask(row["target_tissue_mask"])
    reference_tissue = load_tissue_mask(row["reference_tissue_mask"])
    reference_raw = load_nuclei_mask(row["reference_nuclei_mask"], remap=False)
    sample_dir = (
        output_root / "targets" / str(row["sample_id"])
        if output_root is not None
        else Path(row["target_tissue_mask"]).parent
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    staged = base_row is not None
    if staged:
        base_nuclei_path = base_row.get("target_nuclei_mask")
        if not base_nuclei_path or not Path(base_nuclei_path).is_file():
            raise FileNotFoundError(
                f"paired base target nuclei missing: {base_nuclei_path}"
            )
        base_raw = load_nuclei_mask(base_nuclei_path, remap=False)
        base_tissue = load_tissue_mask(base_row["target_tissue_mask"])
        semantic_generation_region = base_tissue != target_tissue
    else:
        base_raw = reference_raw
        base_tissue = reference_tissue
        semantic_raw = cv2.imread(str(row["change_region"]), cv2.IMREAD_GRAYSCALE)
        if semantic_raw is None:
            raise FileNotFoundError(row["change_region"])
        semantic_generation_region = semantic_raw > 128
    allowed_region = (reference_tissue > 0) | (base_tissue > 0) | (target_tissue > 0)
    generation_region = widen_locally_thin_mask(
        semantic_generation_region,
        allowed_region,
        probnet_args.minimum_mask_width,
    )

    if staged:
        significant_raw = cv2.imread(str(row["change_region"]), cv2.IMREAD_GRAYSCALE)
        if significant_raw is None:
            raise FileNotFoundError(row["change_region"])
        direct_significant = widen_locally_thin_mask(
            significant_raw > 128,
            allowed_region,
            probnet_args.minimum_mask_width,
        )
        base_generation_path = base_row.get("generation_change_region")
        if not base_generation_path:
            raise ValueError("staged generation requires base generation_change_region")
        base_generation_raw = cv2.imread(
            str(base_generation_path), cv2.IMREAD_GRAYSCALE
        )
        if base_generation_raw is None:
            raise FileNotFoundError(base_generation_path)
        full_generation_region = (
            direct_significant | (base_generation_raw > 128) | generation_region
        ) & allowed_region
    else:
        full_generation_region = generation_region

    if not (
        target_tissue.shape
        == reference_tissue.shape
        == reference_raw.shape
        == base_raw.shape
        == generation_region.shape
    ):
        raise ValueError(
            "target tissue, reference/base nuclei, and edit region must align"
        )
    retained_raw, integrity = _retain_complete_reference_cells(
        base_raw, generation_region
    )
    input_nuclei = _remap_raw_nuclei(retained_raw)
    reference_pool = build_reference_pool(reference_raw, probnet_args)
    calibrated_scales, type_proportions, prior_audit = compute_patch_adaptive_priors(
        reference_nuclei_raw=reference_raw,
        reference_tissue=reference_tissue,
        density_exclusion_region=full_generation_region,
        target_tissue=target_tissue,
        generation_region=generation_region,
        library=library,
        global_density_scale=probnet_args.density_scale,
        local_density_direct_min_area=probnet_args.local_density_direct_min_area,
        local_density_direct_min_count=probnet_args.local_density_direct_min_count,
    )
    for tissue_id, override in density_scales.items():
        calibrated_scales[tissue_id] = calibrated_scales.get(tissue_id, 1.0) * override

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    probability = predict_prob(
        model,
        target_tissue,
        input_nuclei,
        generation_region,
        config.cancer_type_index,
        device,
    )
    target_nuclei, diagnostics = generate_for_gamma(
        probability,
        target_tissue,
        input_nuclei,
        generation_region,
        library,
        reference_pool,
        1.5,
        probnet_args,
        calibrated_scales,
        clear_edit_mask=False,
        type_proportions_by_tissue=type_proportions,
    )
    generation_path = _save_binary_mask(
        full_generation_region,
        sample_dir / "generation_change_region.png",
    )
    inpaint_path = _save_binary_mask(
        full_generation_region,
        sample_dir / "inpaint_change_region.png",
    )
    staged_generation_path = _save_binary_mask(
        generation_region,
        sample_dir / "staged_generation_change_region.png",
    )
    output_path = sample_dir / "target_nuclei_mask.png"
    save_nuclei_mask(target_nuclei, str(output_path))
    diagnostics_path = sample_dir / "target_nuclei_mask.diagnostics.json"
    diagnostics_payload = {
        "status": "complete",
        "sample_id": row["sample_id"],
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "generation_mode": "staged_incremental" if staged else "direct_full_change",
        "base_sample_id": base_row.get("sample_id") if base_row is not None else None,
        "checkpoint_role": "P(nucleus)_spatial_placement_only",
        "probnet_change_region": staged_generation_path,
        "generation_change_region": generation_path,
        "inpaint_change_region": inpaint_path,
        "patch_adaptive_priors": prior_audit,
        "source_cell_integrity": integrity,
        "probnet": diagnostics,
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    updated = dict(row)
    updated["target_nuclei_mask"] = str(output_path)
    updated["target_nuclei_metadata"] = str(diagnostics_path)
    updated["generation_seed"] = seed
    updated["nuclei_checkpoint"] = str(checkpoint)
    updated["nuclei_checkpoint_sha256"] = checkpoint_sha256
    updated["nuclei_generation_mode"] = (
        "staged_incremental" if staged else "direct_full_change"
    )
    updated["generation_change_region"] = generation_path
    updated["inpaint_change_region"] = inpaint_path
    updated["staged_generation_change_region"] = staged_generation_path
    updated["nuclei_checkpoint_role"] = "P(nucleus)_spatial_placement_only"
    if base_row is not None:
        updated["moderate_target_nuclei_mask"] = base_row["target_nuclei_mask"]
        updated["moderate_target_nuclei_metadata"] = base_row.get(
            "target_nuclei_metadata"
        )
    hashes = dict(updated.get("sha256") or {})
    hashes["target_nuclei_mask"] = sha256_file(output_path)
    updated["sha256"] = hashes
    Path(sample_dir / "cohort_row.json").write_text(
        json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return updated


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    all_rows = _read_jsonl(args.cohort_manifest)
    if args.limit is None and len(all_rows) != args.expected_count:
        raise ValueError(
            f"frozen cohort requires {args.expected_count} rows, found {len(all_rows)}"
        )
    selected_indices = list(range(len(all_rows)))
    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]
    rows = [all_rows[index] for index in selected_indices]
    base_rows: dict[str, dict] = {}
    if args.base_cohort_manifest is not None:
        for base_row in _read_jsonl(args.base_cohort_manifest):
            base_id = str(base_row.get("sample_id") or "")
            if not base_id or base_id in base_rows:
                raise ValueError("base cohort sample IDs must be non-empty and unique")
            base_rows[base_id] = base_row
    for path in (args.checkpoint, args.library, args.density_scale_json):
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
    probnet_args = _probnet_args(args)
    config = get_config("BCSS")
    probnet_args.skip_tissue_ids = set(probnet_args.skip_tissue_ids) | set(
        config.skip_tissues
    )
    density_scales = load_density_scale(args.density_scale_json)
    device = torch.device(args.device)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    print(f"loading ProbNet checkpoint: {args.checkpoint}", flush=True)
    model = load_checkpoint_model(args.checkpoint, device, probnet_args.base_ch)
    print(f"loading nuclei library: {args.library}", flush=True)
    library = NucleiLibrary(args.library, dataset=config.name)

    output_manifest = args.output_manifest or args.cohort_manifest
    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)
    updated_rows = list(all_rows)
    if args.resume and output_manifest.is_file() and output_manifest != args.cohort_manifest:
        existing_rows = _read_jsonl(output_manifest)
        if len(existing_rows) != len(all_rows):
            raise ValueError(
                "resume output manifest row count differs from source cohort"
            )
        if [row.get("sample_id") for row in existing_rows] != [
            row.get("sample_id") for row in all_rows
        ]:
            raise ValueError(
                "resume output manifest sample order differs from source cohort"
            )
        updated_rows = existing_rows
    completed = 0
    failures: list[dict] = []
    for index, row in enumerate(rows, start=1):
        existing = updated_rows[selected_indices[index - 1]].get(
            "target_nuclei_mask"
        )
        if args.resume and existing and Path(existing).is_file():
            completed += 1
            continue
        try:
            pair_id = str(row.get("pair_id") or row.get("moderate_sample_id") or "")
            base_row = None
            if args.base_cohort_manifest is not None:
                if pair_id not in base_rows:
                    raise KeyError(f"paired base row not found: {pair_id}")
                base_row = base_rows[pair_id]
            updated_rows[selected_indices[index - 1]] = _generate_one(
                row,
                base_row=base_row,
                output_root=args.output_root,
                model=model,
                library=library,
                config=config,
                checkpoint=args.checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                density_scales=density_scales,
                device=device,
                probnet_args=probnet_args,
                seed=args.seed,
            )
            completed += 1
        except Exception as exc:
            failures.append(
                {
                    "sample_id": row.get("sample_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if index % 10 == 0 or index == len(rows):
            print(
                f"nuclei={completed} failed={len(failures)} "
                f"processed={index}/{len(rows)}",
                flush=True,
            )
            _write_jsonl(output_manifest, updated_rows)
    _write_jsonl(output_manifest, updated_rows)
    failure_path = output_manifest.parent / "nuclei_failures.jsonl"
    _write_jsonl(failure_path, failures)
    summary = {
        "status": "complete" if not failures and completed == len(rows) else "incomplete",
        "requested_count": len(rows),
        "completed_count": completed,
        "failure_count": len(failures),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "library": str(args.library),
        "density_scale_json": (
            str(args.density_scale_json) if args.density_scale_json else None
        ),
        "checkpoint_role": "P(nucleus)_spatial_placement_only",
        "minimum_mask_width_px": args.minimum_mask_width,
        "component_quota_policy": "area_largest_remainder",
        "max_nucleus_overlap_fraction": 0.0,
        "retained_source_nucleus_policy": "bitwise_no_overwrite",
        "placement_retry_policy": "candidate_pool_exhaustion_until_component_quota",
        "retry_candidate_pool": "max(32, 8 * component quota)",
        "base_cohort_manifest": (
            str(args.base_cohort_manifest) if args.base_cohort_manifest else None
        ),
        "source_cohort_manifest": str(args.cohort_manifest),
        "output_manifest": str(output_manifest),
        "output_root": str(args.output_root) if args.output_root else None,
    }
    (output_manifest.parent / "nuclei_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
