#!/usr/bin/env python3
"""Select and build the 120-case frozen ProbNet geometry-only endpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_config import get_config
from inpaint_cells.data.prob_dataset import _choose_crop_origin
from inpaint_cells.generate import (
    compute_patch_adaptive_priors,
    load_checkpoint_model,
    predict_prob,
)
from inpaint_cells.nuclei_library.library import NucleiLibrary
from inpaint_cells.sampling_policy import widen_locally_thin_mask
from inpaint_cells.utils.mask_utils import save_nuclei_mask
from phase3_mask_edit.benchmark.probnet_compact import (
    CanonicalInstance,
    load_semantic_instances,
    reference_pool_from_instances,
    stable_seed,
)
from phase3_mask_edit.benchmark.probnet_spatial import (
    build_statistical_plan,
    erase_complete_instances,
    frozen_sampler_args,
    generate_fixed_plan_layout,
    instances_with_centres_in_region,
    raw_to_internal,
)


PROTOCOL = "probnet_spatial_p2_geometry_v4_strict_20260724"
DATASETS = ("BCSS", "GlaS", "IGNITE", "ORCA", "PANDA", "PUMA")
ABSOLUTE_COUNT_BINS = ("0", "1-5", "6-20", ">20")
ENDPOINT_STRATA = (
    "dataset-low",
    "dataset-mid-low",
    "dataset-mid-high",
    "dataset-high",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_digest(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join(map(str, parts)).encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n"
            )
    temporary.replace(path)


def save_gray(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), values.astype(np.uint8)):
        raise RuntimeError(f"Failed to write {path}")


def count_bin(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    return ">20"


def crop_instance(
    instance: CanonicalInstance,
    *,
    y: int,
    x: int,
    size: int,
) -> CanonicalInstance | None:
    center_x, center_y = instance.centroid_xy
    if not (y <= center_y < y + size and x <= center_x < x + size):
        return None
    local = instance.mask[y : y + size, x : x + size]
    if not np.any(local):
        return None
    yy, xx = np.where(local)
    return CanonicalInstance(
        instance_id=instance.instance_id,
        raw_type=instance.raw_type,
        mask=np.ascontiguousarray(local, dtype=bool),
        centroid_xy=(float(center_x - x), float(center_y - y)),
        area_pixels=int(local.sum()),
        touches_border=bool(
            instance.touches_border
            or np.any(yy == 0)
            or np.any(xx == 0)
            or np.any(yy == size - 1)
            or np.any(xx == size - 1)
        ),
    )


def load_cropped_inputs(
    row: Mapping[str, Any], *, img_size: int
) -> dict[str, Any]:
    tissue_full = cv2.imread(str(row["source_tissue"]), cv2.IMREAD_GRAYSCALE)
    nuclei_full = cv2.imread(str(row["source_nuclei"]), cv2.IMREAD_GRAYSCALE)
    edit_full = cv2.imread(str(row["edit_mask"]), cv2.IMREAD_GRAYSCALE)
    if any(value is None for value in (tissue_full, nuclei_full, edit_full)):
        raise RuntimeError("Failed to read tissue, nuclei, or edit mask")
    edit_full = edit_full > 128
    if "crop_origin_yx" in row:
        y, x = map(int, row["crop_origin_yx"])
    else:
        y, x = _choose_crop_origin(
            tissue_full.shape[0],
            tissue_full.shape[1],
            img_size,
            edit_full.astype(np.float32),
            "mask",
            deterministic=True,
        )
    crop = np.s_[y : y + img_size, x : x + img_size]
    tissue = tissue_full[crop].astype(np.int64)
    nuclei_raw = nuclei_full[crop]
    source_semantic_region = edit_full[crop]
    semantic_region = source_semantic_region & (tissue != 0)
    if not np.any(semantic_region):
        raise RuntimeError(
            "Edit region is empty after biological-foreground clipping"
        )
    generation_region = widen_locally_thin_mask(
        semantic_region, tissue != 0, minimum_width=33
    )
    return {
        "crop_origin_yx": [int(y), int(x)],
        "tissue": tissue,
        "nuclei_raw": nuclei_raw,
        "source_semantic_region": source_semantic_region,
        "semantic_region": semantic_region,
        "generation_region": generation_region,
    }


def build_plan_for_crop(
    *,
    dataset: str,
    crop: Mapping[str, Any],
    library: NucleiLibrary,
    seed: int,
) -> tuple[dict[str, Any], list[Any], dict[int, np.ndarray], dict[str, Any]]:
    config = get_config(dataset)
    args = frozen_sampler_args(skip_tissue_ids=config.skip_tissues)
    density_scales, type_proportions, prior_audit = compute_patch_adaptive_priors(
        reference_nuclei_raw=np.asarray(crop["nuclei_raw"]),
        reference_tissue=np.asarray(crop["tissue"]),
        density_exclusion_region=np.asarray(crop["generation_region"]),
        target_tissue=np.asarray(crop["tissue"]),
        generation_region=np.asarray(crop["generation_region"]),
        library=library,
        global_density_scale=1.0,
        local_density_direct_min_area=args.local_density_direct_min_area,
        local_density_direct_min_count=args.local_density_direct_min_count,
        dataset_name=dataset,
    )
    plan, slots, component_labels = build_statistical_plan(
        tissue_map=np.asarray(crop["tissue"]),
        generation_region=np.asarray(crop["generation_region"]),
        density_scales=density_scales,
        type_proportions_by_tissue=type_proportions,
        library=library,
        args=args,
        seed=seed,
    )
    prior_audit["generation_support"] = {
        "source_semantic_pixels": int(
            np.count_nonzero(crop["source_semantic_region"])
        ),
        "semantic_foreground_pixels": int(
            np.count_nonzero(crop["semantic_region"])
        ),
        "generation_pixels": int(
            np.count_nonzero(crop["generation_region"])
        ),
        "minimum_width_px": 33,
        "source_nucleus_erasure_policy": (
            "complete_component_if_centroid_in_support"
        ),
    }
    return plan, slots, component_labels, prior_audit


def select_endpoint(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.source_manifest)
    if len(rows) != args.expected_source_samples:
        raise RuntimeError(
            f"Expected {args.expected_source_samples} source rows, found {len(rows)}"
        )
    eligible: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    active_dataset: str | None = None
    library: NucleiLibrary | None = None
    ordered_rows = [
        row
        for dataset in DATASETS
        for row in rows
        if str(row["dataset"]) == dataset
    ]
    if len(ordered_rows) != len(rows):
        raise RuntimeError("Source manifest contains an unsupported dataset")
    for index, row in enumerate(ordered_rows, start=1):
        dataset = str(row["dataset"])
        try:
            if dataset != active_dataset:
                library = NucleiLibrary(
                    str(args.library_root / dataset), dataset=dataset
                )
                active_dataset = dataset
            assert library is not None
            crop = load_cropped_inputs(row, img_size=args.img_size)
            plan_seed = stable_seed(
                args.seed, row["source_sample_id"], "statistical_plan"
            )
            plan, slots, _, prior_audit = build_plan_for_crop(
                dataset=dataset,
                crop=crop,
                library=library,
                seed=plan_seed,
            )
            planned_count = len(slots)
            eligible.append(
                {
                    **row,
                    "crop_origin_yx": crop["crop_origin_yx"],
                    "crop_size": args.img_size,
                    "planned_count": planned_count,
                    "planned_count_bin": count_bin(planned_count),
                    "count_bin": count_bin(planned_count),
                    "planned_count_source": plan["count_source"],
                    "planned_type_source": plan["type_source"],
                    "statistical_plan_sha256": sha256_json(plan),
                    "prior_audit_sha256": sha256_json(prior_audit),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "source_sample_id": str(row.get("source_sample_id")),
                    "dataset": dataset,
                    "error": str(exc),
                }
            )
        if index % 250 == 0 or index == len(rows):
            print(
                f"planned={len(eligible)}/{len(rows)} failures={len(failures)}",
                flush=True,
            )

    chosen: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    available_counts = {
        dataset: {
            bin_name: sum(
                1
                for row in eligible
                if row["dataset"] == dataset
                and row["planned_count_bin"] == bin_name
            )
            for bin_name in ABSOLUTE_COUNT_BINS
        }
        for dataset in DATASETS
    }
    eligibility_audit = {
        "protocol": PROTOCOL,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "source_rows": len(rows),
        "eligible_source_rows": len(eligible),
        "planning_failures": failures,
        "count_bin_basis": "frozen_statistical_planned_count",
        "available_counts": available_counts,
        "required_per_dataset_stratum": args.per_bin,
    }
    eligibility_path = args.output_manifest.with_suffix(".eligibility.json")
    write_json(eligibility_path, eligibility_audit)
    print(
        json.dumps(
            {
                "eligibility_audit": str(eligibility_path.resolve()),
                "eligible_source_rows": len(eligible),
                "planning_failure_count": len(failures),
                "available_counts": available_counts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    endpoint_stratum_ranges: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        dataset_rows = [row for row in eligible if row["dataset"] == dataset]
        ranked_rows = sorted(
            dataset_rows,
            key=lambda row: (
                int(row["planned_count"]),
                stable_digest(
                    args.seed,
                    dataset,
                    "load-rank",
                    row["source_sample_id"],
                ),
            ),
        )
        if len(ranked_rows) < args.per_bin * len(ENDPOINT_STRATA):
            raise RuntimeError(
                f"Insufficient {dataset} rows for four load strata: "
                f"{len(ranked_rows)}/{args.per_bin * len(ENDPOINT_STRATA)}"
            )
        candidates_by_stratum: dict[str, list[dict[str, Any]]] = {
            stratum: [] for stratum in ENDPOINT_STRATA
        }
        for rank, row in enumerate(ranked_rows):
            stratum_index = min(
                len(ENDPOINT_STRATA) - 1,
                rank * len(ENDPOINT_STRATA) // len(ranked_rows),
            )
            candidates_by_stratum[ENDPOINT_STRATA[stratum_index]].append(row)
        selected_by_stratum: dict[str, list[dict[str, Any]]] = {}
        endpoint_stratum_ranges[dataset] = {}
        for stratum in ENDPOINT_STRATA:
            candidates = sorted(
                candidates_by_stratum[stratum],
                key=lambda row: stable_digest(
                    args.seed,
                    dataset,
                    stratum,
                    "endpoint-select",
                    row["source_sample_id"],
                ),
            )
            if len(candidates) < args.per_bin:
                raise RuntimeError(
                    f"Insufficient {dataset}/{stratum}: "
                    f"{len(candidates)}/{args.per_bin}"
                )
            selected_by_stratum[stratum] = candidates[: args.per_bin]
            pool_counts = [
                int(row["planned_count"])
                for row in candidates_by_stratum[stratum]
            ]
            selected_counts = [
                int(row["planned_count"])
                for row in selected_by_stratum[stratum]
            ]
            endpoint_stratum_ranges[dataset][stratum] = {
                "eligible_rows": len(candidates_by_stratum[stratum]),
                "pool_min_planned_count": min(pool_counts),
                "pool_max_planned_count": max(pool_counts),
                "selected_min_planned_count": min(selected_counts),
                "selected_max_planned_count": max(selected_counts),
                "selected_absolute_count_bins": dict(
                    Counter(
                        str(row["planned_count_bin"])
                        for row in selected_by_stratum[stratum]
                    )
                ),
            }
        for stratum in ENDPOINT_STRATA:
            for row in selected_by_stratum[stratum]:
                safe_stratum = stratum.replace("-", "_")
                output = {
                    **row,
                    "endpoint_count_stratum": stratum,
                    "end_to_end_id": (
                        f"{dataset}__{safe_stratum}"
                        f"__{counts[(dataset, stratum)]:02d}__{row['sample_id']}"
                    ),
                    "selection_seed": args.seed,
                    "selection_role": (
                        "model_independent_statistical_planned_count_stratum"
                    ),
                }
                chosen.append(output)
                counts[(dataset, stratum)] += 1
    write_jsonl(args.output_manifest, chosen)
    summary = {
        "protocol": PROTOCOL,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "eligible_source_rows": len(eligible),
        "planning_failures": failures,
        "selection_manifest": str(args.output_manifest.resolve()),
        "selection_manifest_sha256": sha256_file(args.output_manifest),
        "selected": len(chosen),
        "seed": args.seed,
        "selection_uses_probnet": False,
        "selection_uses_generator": False,
        "selection_uses_segmentator": False,
        "selection_uses_cellvit": False,
        "selection_uses_pathokid": False,
        "count_bin_basis": "frozen_statistical_planned_count",
        "endpoint_stratum_basis": {
            "method": (
                "within each dataset, rank eligible rows by frozen-policy "
                "planned count, split the ranked rows into four disjoint "
                "equal-size load strata, and select five rows per stratum "
                "with a stable hash"
            ),
            "ordered_strata": list(ENDPOINT_STRATA),
        },
        "endpoint_stratum_ranges": endpoint_stratum_ranges,
        "available_absolute_count_bins": available_counts,
        "counts": {
            dataset: {
                stratum: counts[(dataset, stratum)]
                for stratum in ENDPOINT_STRATA
            }
            for dataset in DATASETS
        },
    }
    write_json(args.output_manifest.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_layouts(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.manifest)
    if len(rows) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} endpoint rows, found {len(rows)}"
        )
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"Checkpoint hash mismatch: {checkpoint_hash} != "
            f"{args.expected_checkpoint_sha256}"
        )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty {args.output_root}")
    device = torch.device(args.device)
    model = load_checkpoint_model(str(args.checkpoint), device, args.base_ch)
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    active_dataset: str | None = None
    library: NucleiLibrary | None = None
    for index, row in enumerate(rows, start=1):
        dataset = str(row["dataset"])
        sample_id = str(row["end_to_end_id"])
        try:
            if dataset != active_dataset:
                library = NucleiLibrary(
                    str(args.library_root / dataset), dataset=dataset
                )
                active_dataset = dataset
            assert library is not None
            config = get_config(dataset)
            sampler_args = frozen_sampler_args(
                skip_tissue_ids=config.skip_tissues
            )
            crop = load_cropped_inputs(row, img_size=args.img_size)
            full_instances, _ = load_semantic_instances(
                row["source_nuclei"], min_area=1
            )
            crop_y, crop_x = crop["crop_origin_yx"]
            cropped_instances = [
                cropped
                for instance in full_instances
                if (
                    cropped := crop_instance(
                        instance,
                        y=int(crop_y),
                        x=int(crop_x),
                        size=args.img_size,
                    )
                )
                is not None
            ]
            hidden_instances = instances_with_centres_in_region(
                cropped_instances,
                np.asarray(crop["generation_region"]),
                np.asarray(crop["tissue"]),
                config.skip_tissues,
            )
            reference_pool = reference_pool_from_instances(
                cropped_instances,
                min_area=8,
                exclude_border=True,
            )
            input_nuclei = erase_complete_instances(
                raw_to_internal(np.asarray(crop["nuclei_raw"])),
                hidden_instances,
            )
            plan_seed = stable_seed(
                args.seed, row["source_sample_id"], "statistical_plan"
            )
            plan, slots, component_labels, prior_audit = build_plan_for_crop(
                dataset=dataset,
                crop=crop,
                library=library,
                seed=plan_seed,
            )
            if len(slots) != int(row["planned_count"]):
                raise RuntimeError(
                    f"Planned count drift: {len(slots)} != "
                    f"{row['planned_count']}"
                )
            probability = predict_prob(
                model,
                np.asarray(crop["tissue"]),
                input_nuclei,
                np.asarray(crop["generation_region"]),
                config.cancer_type_index,
                device,
            )
            layout_seed = stable_seed(
                args.seed,
                sample_id,
                checkpoint_hash,
                "probnet_spatial_layout",
            )
            generated, records, diagnostics = generate_fixed_plan_layout(
                probability=probability,
                tissue_map=np.asarray(crop["tissue"]),
                input_nuclei=input_nuclei,
                generation_region=np.asarray(crop["generation_region"]),
                plan=plan,
                slots=slots,
                component_labels_by_tissue=component_labels,
                library=library,
                reference_pool=reference_pool,
                sampler="probnet",
                gamma=1.5,
                args=sampler_args,
                seed=layout_seed,
            )
            staged = args.output_root / "staged"
            reference_tissue = staged / "reference_tissue" / f"{sample_id}.png"
            reference_nuclei = staged / "reference_nuclei" / f"{sample_id}.png"
            change_region = staged / "edit_masks" / f"{sample_id}.png"
            generated_nuclei = staged / "generated_nuclei" / f"{sample_id}.png"
            save_gray(reference_tissue, np.asarray(crop["tissue"]))
            save_gray(reference_nuclei, np.asarray(crop["nuclei_raw"]))
            save_gray(
                change_region,
                np.asarray(crop["generation_region"], dtype=np.uint8) * 255,
            )
            generated_nuclei.parent.mkdir(parents=True, exist_ok=True)
            save_nuclei_mask(generated, str(generated_nuclei))
            layout_path = args.output_root / "layouts" / f"{sample_id}.json"
            diagnostic_path = (
                args.output_root / "diagnostics" / f"{sample_id}.json"
            )
            plan_path = args.output_root / "plans" / f"{sample_id}.json"
            write_json(
                layout_path,
                {
                    "instances": records,
                    "seed": layout_seed,
                    "sampler": "probnet",
                },
            )
            write_json(
                diagnostic_path,
                {
                    **diagnostics,
                    "patch_adaptive_priors": prior_audit,
                },
            )
            write_json(plan_path, plan)
            output_rows.append(
                {
                    **row,
                    "protocol": PROTOCOL,
                    "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_hash,
                    "crop_origin_yx": crop["crop_origin_yx"],
                    "crop_size": args.img_size,
                    "reference_tissue": str(reference_tissue.resolve()),
                    "reference_nuclei": str(reference_nuclei.resolve()),
                    "target_tissue": str(reference_tissue.resolve()),
                    "edit_mask": str(change_region.resolve()),
                    "generated_nuclei": str(generated_nuclei.resolve()),
                    "layout_records": str(layout_path.resolve()),
                    "layout_diagnostics": str(diagnostic_path.resolve()),
                    "statistical_plan": str(plan_path.resolve()),
                    "layout_seed": layout_seed,
                    "requested_count": len(slots),
                    "placed_count": int(diagnostics["placed"]),
                    "unfilled_count": int(diagnostics["unfilled"]),
                    "placement_completion": float(
                        diagnostics["placement_completion"]
                    ),
                    "exact_type_quota": bool(
                        diagnostics["exact_type_quota"]
                    ),
                    "exact_component_quota": bool(
                        diagnostics["exact_component_quota"]
                    ),
                    "overlap_pixels": int(diagnostics["overlap_pixels"]),
                    "outside_tissue_pixels": int(
                        diagnostics["outside_tissue_pixels"]
                    ),
                    "retained_input_nuclei_unchanged": bool(
                        diagnostics["retained_input_nuclei_unchanged"]
                    ),
                    "target_instances": [
                        {
                            "instance_id": instance.instance_id,
                            "raw_type": int(instance.raw_type),
                            "centroid_xy": [
                                float(instance.centroid_xy[0]),
                                float(instance.centroid_xy[1]),
                            ],
                            "area_pixels": int(instance.area_pixels),
                        }
                        for instance in hidden_instances
                    ],
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"end_to_end_id": sample_id, "dataset": dataset, "error": str(exc)}
            )
        if index % 10 == 0 or index == len(rows):
            print(
                f"layouts={len(output_rows)}/{len(rows)} failures={len(failures)}",
                flush=True,
            )
    manifest = args.output_root / "layout_manifest.jsonl"
    write_jsonl(manifest, output_rows)

    def summarize(rows_to_summarize: list[dict[str, Any]]) -> dict[str, Any]:
        requested = int(
            sum(int(row["requested_count"]) for row in rows_to_summarize)
        )
        placed = int(sum(int(row["placed_count"]) for row in rows_to_summarize))
        return {
            "samples": len(rows_to_summarize),
            "requested": requested,
            "placed": placed,
            "unfilled": requested - placed,
            "placement_completion": (
                placed / requested if requested else 1.0
            ),
            "exact_type_quota_rate": (
                sum(bool(row["exact_type_quota"]) for row in rows_to_summarize)
                / len(rows_to_summarize)
                if rows_to_summarize
                else 1.0
            ),
            "exact_component_quota_rate": (
                sum(
                    bool(row["exact_component_quota"])
                    for row in rows_to_summarize
                )
                / len(rows_to_summarize)
                if rows_to_summarize
                else 1.0
            ),
            "overlap_pixels": int(
                sum(int(row["overlap_pixels"]) for row in rows_to_summarize)
            ),
            "outside_tissue_pixels": int(
                sum(
                    int(row["outside_tissue_pixels"])
                    for row in rows_to_summarize
                )
            ),
            "retained_preservation_rate": (
                sum(
                    bool(row["retained_input_nuclei_unchanged"])
                    for row in rows_to_summarize
                )
                / len(rows_to_summarize)
                if rows_to_summarize
                else 1.0
            ),
        }

    overall = summarize(output_rows)
    by_dataset = {
        dataset: summarize(
            [row for row in output_rows if str(row["dataset"]) == dataset]
        )
        for dataset in DATASETS
    }
    by_load_stratum = {
        stratum: summarize(
            [
                row
                for row in output_rows
                if str(row["endpoint_count_stratum"]) == stratum
            ]
        )
        for stratum in ENDPOINT_STRATA
    }
    safety_gate_passed = bool(
        overall["placement_completion"] >= 0.98
        and overall["overlap_pixels"] == 0
        and overall["outside_tissue_pixels"] == 0
        and overall["retained_preservation_rate"] == 1.0
    )
    validation = {
        "protocol": PROTOCOL,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_role": "spatial_placement_probability_only",
        "count_type_role": "frozen_statistical_policy",
        "evaluation_scope": "geometry_only_no_h_e_or_cellvit",
        "gamma": 1.5,
        "expected": args.expected_samples,
        "completed": len(output_rows),
        "failures": failures,
        "requested_total": int(
            sum(row["requested_count"] for row in output_rows)
        ),
        "placed_total": int(sum(row["placed_count"] for row in output_rows)),
        "overall": overall,
        "by_dataset": by_dataset,
        "by_load_stratum": by_load_stratum,
        "geometry_safety_gate_passed": safety_gate_passed,
        "layout_manifest": str(manifest.resolve()),
        "layout_manifest_sha256": sha256_file(manifest),
        "complete": len(output_rows) == args.expected_samples and not failures,
    }
    write_json(args.output_root / "validation.json", validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0 if validation["complete"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--source-manifest", type=Path, required=True)
    select.add_argument("--library-root", type=Path, required=True)
    select.add_argument("--output-manifest", type=Path, required=True)
    select.add_argument("--expected-source-samples", type=int, default=9950)
    select.add_argument("--per-bin", type=int, default=5)
    select.add_argument("--img-size", type=int, default=256)
    select.add_argument("--seed", type=int, default=20260724)
    select.set_defaults(func=select_endpoint)

    layouts = subparsers.add_parser("layouts")
    layouts.add_argument("--manifest", type=Path, required=True)
    layouts.add_argument("--checkpoint", type=Path, required=True)
    layouts.add_argument("--expected-checkpoint-sha256", required=True)
    layouts.add_argument("--library-root", type=Path, required=True)
    layouts.add_argument("--output-root", type=Path, required=True)
    layouts.add_argument("--expected-samples", type=int, default=120)
    layouts.add_argument("--img-size", type=int, default=256)
    layouts.add_argument("--base-ch", type=int, default=64)
    layouts.add_argument("--seed", type=int, default=20260724)
    layouts.add_argument("--device", default="cuda:0")
    layouts.set_defaults(func=build_layouts)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
