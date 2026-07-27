#!/usr/bin/env python3
"""Run and report the frozen ProbNet spatial-sampler ablation (Benchmark P1)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_config import get_config
from inpaint_cells.generate import load_checkpoint_model, predict_prob
from inpaint_cells.nuclei_library.library import NucleiLibrary
from inpaint_cells.sampling_policy import widen_locally_thin_mask
from phase3_mask_edit.benchmark.probnet_compact import (
    load_semantic_instances,
    read_gray,
    reference_pool_from_instances,
    stable_seed,
)
from phase3_mask_edit.benchmark.probnet_spatial import (
    SAMPLERS,
    build_oracle_plan,
    erase_complete_instances,
    frozen_sampler_args,
    generate_fixed_plan_layout,
    instances_with_centres_in_region,
    raw_to_internal,
    spatial_metrics,
)


PROTOCOL = "probnet_spatial_p1_v2_strict_geometry_20260724"
PRIMARY_BASELINE = "uniform"
SECONDARY_BASELINE = "poisson_only"
MATERIAL_RELATIVE_IMPROVEMENT = 0.05
PLACEMENT_COMPLETION_NONINFERIORITY_MARGIN = 0.005
METRICS = (
    "nnd_w1_um",
    "ripley_k_normalized_l1",
    "boundary_distance_w1_um",
    "component_occupancy_l1_per_target",
    "point_f1_4um",
    "class_aware_point_f1_4um",
)
ERROR_METRICS = set(METRICS[:4])
CORE_SPATIAL_STRUCTURE_METRICS = {
    "nnd_w1_um",
    "ripley_k_normalized_l1",
}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(row), ensure_ascii=False, allow_nan=True) + "\n"
        )
        handle.flush()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def cache_probability(path: Path, probability: np.ndarray) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, probability=probability.astype(np.float16))
    temporary.replace(path)
    return probability.astype(np.float32)


def load_probability(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return payload["probability"].astype(np.float32)


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _case_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row.get("case_id") or row["sample_id"])


def _prepare_provenance(args: argparse.Namespace, checkpoint_hash: str) -> None:
    provenance = args.output_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    config = "\n".join(
        [
            f"protocol: {PROTOCOL}",
            "checkpoint_role: spatial_placement_probability_only",
            f"checkpoint: {args.checkpoint}",
            f"checkpoint_sha256: {checkpoint_hash}",
            "gamma: 1.5",
            "minimum_mask_width: 33",
            "component_quota_policy: area_largest_remainder",
            "require_full_tissue_containment: true",
            "max_nucleus_overlap_fraction: 0.0",
            "retry_candidate_multiplier: 12",
            "retry_candidate_floor: 64",
            "dense_retry_quota_threshold: 20",
            "dense_retry_occupancy_threshold: 0.12",
            "dense_retry_candidate_multiplier: 24",
            "dense_retry_candidate_floor: 128",
            "placement_shape_trials: 4",
            "placement_transform_trials: 12",
            "dense_placement_shape_trials: 6",
            "dense_placement_transform_trials: 24",
            f"primary_baseline: {PRIMARY_BASELINE}",
            f"material_relative_improvement: {MATERIAL_RELATIVE_IMPROVEMENT}",
            "replicates: 5",
            "samplers:",
            *[f"  - {sampler}" for sampler in SAMPLERS],
            "",
        ]
    )
    (provenance / "frozen_config.yaml").write_text(config, encoding="utf-8")
    write_json(
        provenance / "checkpoint_hashes.json",
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "sha256": checkpoint_hash,
            "expected_sha256": args.expected_checkpoint_sha256,
            "verified": checkpoint_hash == args.expected_checkpoint_sha256,
        },
    )
    write_json(
        provenance / "split_and_endpoint_hashes.json",
        {
            "p1_case_manifest": str(args.manifest.resolve()),
            "p1_case_manifest_sha256": sha256_file(args.manifest),
            "selection_uses_probnet": False,
            "source_split": "grouped held-out test",
        },
    )


def run_shard(args: argparse.Namespace) -> int:
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Require shards >= 1 and 0 <= shard-index < shards")
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"Checkpoint hash mismatch: {checkpoint_hash} != "
            f"{args.expected_checkpoint_sha256}"
        )
    if args.shard_index == 0:
        _prepare_provenance(args, checkpoint_hash)
    rows = read_jsonl(args.manifest)
    if args.expected_cases > 0 and len(rows) != args.expected_cases:
        raise RuntimeError(
            f"Expected {args.expected_cases} P1 cases, found {len(rows)}"
        )
    selected = [
        row
        for index, row in enumerate(rows)
        if index % args.shards == args.shard_index
    ]
    output_path = (
        args.output_root
        / "p1_spatial_ablation"
        / "shards"
        / f"shard_{args.shard_index:02d}.jsonl"
    )
    existing = read_jsonl(output_path)
    completed = {
        (
            str(row["sample_id"]),
            str(row["mask_mode"]),
            int(row["replicate"]),
            str(row["sampler"]),
        )
        for row in existing
    }
    failure_path = (
        args.output_root
        / "p1_spatial_ablation"
        / "shards"
        / f"shard_{args.shard_index:02d}_failures.jsonl"
    )
    prior_failures = read_jsonl(failure_path)
    failures: list[dict[str, Any]] = []

    device = torch.device(args.device)
    model = load_checkpoint_model(str(args.checkpoint), device, args.base_ch)
    active_dataset: str | None = None
    library: NucleiLibrary | None = None
    processed_cases = 0
    for local_index, row in enumerate(selected, start=1):
        dataset = str(row["dataset"])
        sample_id = str(row["sample_id"])
        mask_mode = str(row["mask_mode"])
        expected_keys = {
            (sample_id, mask_mode, replicate, sampler)
            for replicate in range(args.replicates)
            for sampler in SAMPLERS
        }
        if expected_keys <= completed:
            continue
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
            tissue = read_gray(row["source_tissue"]).astype(np.int64)
            target_raw = read_gray(row["source_nuclei"])
            source_semantic_region = read_gray(row["mask_path"]) > 0
            semantic_region = source_semantic_region & (tissue != 0)
            if not np.any(semantic_region):
                raise RuntimeError(
                    "Reconstruction region is empty after biological-foreground clipping"
                )
            generation_region = widen_locally_thin_mask(
                semantic_region,
                tissue != 0,
                minimum_width=33,
            )
            target_instances, _ = load_semantic_instances(
                row["source_nuclei"], min_area=1
            )
            hidden_instances = instances_with_centres_in_region(
                target_instances,
                generation_region,
                tissue,
                config.skip_tissues,
            )
            if not hidden_instances:
                raise RuntimeError("No eligible hidden nuclei after support widening")
            reference_instances, _ = load_semantic_instances(
                row["reference_nuclei"], min_area=1
            )
            reference_pool = reference_pool_from_instances(
                reference_instances,
                min_area=8,
                exclude_border=True,
            )
            input_nuclei = erase_complete_instances(
                raw_to_internal(target_raw), hidden_instances
            )
            cache_path = (
                args.output_root
                / "p1_spatial_ablation"
                / "probability_cache"
                / dataset
                / sample_id
                / f"{mask_mode}.npz"
            )
            if cache_path.is_file():
                probability = load_probability(cache_path)
            else:
                probability = predict_prob(
                    model,
                    tissue,
                    input_nuclei,
                    generation_region,
                    config.cancer_type_index,
                    device,
                )
                probability = cache_probability(cache_path, probability)
            plan_seed = stable_seed(
                PROTOCOL, sample_id, mask_mode, "fixed_oracle_plan"
            )
            plan, slots, component_labels = build_oracle_plan(
                tissue_map=tissue,
                generation_region=generation_region,
                hidden_instances=hidden_instances,
                library=library,
                args=sampler_args,
                seed=plan_seed,
            )
            if len(slots) != len(hidden_instances):
                raise RuntimeError(
                    f"Oracle plan count mismatch: {len(slots)} != "
                    f"{len(hidden_instances)}"
                )
            plan_hash = sha256_json(plan)
            for replicate in range(args.replicates):
                replicate_seed = stable_seed(
                    PROTOCOL, sample_id, mask_mode, replicate
                )
                for sampler in SAMPLERS:
                    key = (sample_id, mask_mode, replicate, sampler)
                    if key in completed:
                        continue
                    _, records, diagnostics = generate_fixed_plan_layout(
                        probability=probability,
                        tissue_map=tissue,
                        input_nuclei=input_nuclei,
                        generation_region=generation_region,
                        plan=plan,
                        slots=slots,
                        component_labels_by_tissue=component_labels,
                        library=library,
                        reference_pool=reference_pool,
                        sampler=sampler,
                        gamma=1.5,
                        args=sampler_args,
                        seed=replicate_seed,
                    )
                    metrics = spatial_metrics(
                        hidden_instances=hidden_instances,
                        placement_records=records,
                        generation_region=generation_region,
                        tissue_map=tissue,
                        component_labels_by_tissue=component_labels,
                    )
                    accepted = [
                        record
                        for record in records
                        if record["placement_status"] == "accepted"
                    ]
                    output = {
                        "protocol": PROTOCOL,
                        "sample_id": sample_id,
                        "source_sample_id": row.get("source_sample_id"),
                        "dataset": dataset,
                        "organ": row.get("organ"),
                        "case_id": row.get("case_id"),
                        "mask_mode": mask_mode,
                        "replicate": replicate,
                        "seed": replicate_seed,
                        "sampler": sampler,
                        "checkpoint_sha256": checkpoint_hash,
                        "probability_cache": str(cache_path.resolve()),
                        "plan_sha256": plan_hash,
                        "source_semantic_region_pixels": int(
                            source_semantic_region.sum()
                        ),
                        "semantic_region_pixels": int(semantic_region.sum()),
                        "semantic_pixels_clipped_to_background": int(
                            np.count_nonzero(
                                source_semantic_region & (tissue == 0)
                            )
                        ),
                        "generation_region_pixels": int(generation_region.sum()),
                        "requested_count": len(slots),
                        **metrics,
                        **diagnostics,
                        "accepted_centres_xy": [
                            record["center_xy"] for record in accepted
                        ],
                        "accepted_types": [
                            int(record["cell_type"]) for record in accepted
                        ],
                    }
                    append_jsonl(output_path, output)
                    completed.add(key)
            processed_cases += 1
        except Exception as exc:  # noqa: BLE001
            failure = {
                "sample_id": sample_id,
                "dataset": dataset,
                "mask_mode": mask_mode,
                "error": str(exc),
            }
            append_jsonl(failure_path, failure)
            failures.append(failure)
        if local_index % 5 == 0 or local_index == len(selected):
            print(
                f"shard={args.shard_index}/{args.shards} "
                f"cases={local_index}/{len(selected)} "
                f"rows={len(completed)} failures={len(failures)} "
                f"prior_failure_attempts={len(prior_failures)}",
                flush=True,
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summary = {
        "protocol": PROTOCOL,
        "shard_index": args.shard_index,
        "shards": args.shards,
        "selected_cases": len(selected),
        "processed_cases_this_invocation": processed_cases,
        "completed_rows": len(completed),
        "expected_rows": len(selected) * args.replicates * len(SAMPLERS),
        "failures": failures,
        "prior_failure_attempts": len(prior_failures),
        "complete": (
            len(completed) == len(selected) * args.replicates * len(SAMPLERS)
            and not failures
        ),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["complete"] else 2


def _aggregate_seed_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["sample_id"]), str(row["mask_mode"]), str(row["sampler"]))
        ].append(row)
    result: list[dict[str, Any]] = []
    for (sample_id, mask_mode, sampler), values in sorted(grouped.items()):
        first = values[0]
        output = {
            "sample_id": sample_id,
            "mask_mode": mask_mode,
            "sampler": sampler,
            "dataset": first["dataset"],
            "case_id": first.get("case_id"),
            "requested_count": int(first["requested_count"]),
            "replicates": len(values),
            "placement_completion": _mean(
                float(value["placement_completion"]) for value in values
            ),
            "exact_type_quota_rate": _mean(
                float(bool(value["exact_type_quota"])) for value in values
            ),
            "exact_component_quota_rate": _mean(
                float(bool(value["exact_component_quota"])) for value in values
            ),
            "retained_preservation_rate": _mean(
                float(bool(value["retained_input_nuclei_unchanged"]))
                for value in values
            ),
            "outside_tissue_pixels": _mean(
                float(value["outside_tissue_pixels"]) for value in values
            ),
            "overlap_pixels": _mean(
                float(value["overlap_pixels"]) for value in values
            ),
        }
        for metric in METRICS:
            output[metric] = _mean(float(value[metric]) for value in values)
        result.append(output)
    return result


def _sampler_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    dataset_summary = {
        dataset: {
            metric: _mean(float(row[metric]) for row in values)
            for metric in (
                *METRICS,
                "placement_completion",
                "exact_type_quota_rate",
                "exact_component_quota_rate",
                "retained_preservation_rate",
                "outside_tissue_pixels",
                "overlap_pixels",
            )
        }
        for dataset, values in sorted(by_dataset.items())
    }
    return {
        "equal_dataset_macro": {
            metric: _mean(
                float(values[metric]) for values in dataset_summary.values()
            )
            for metric in next(iter(dataset_summary.values()))
        },
        "by_dataset": dataset_summary,
        "cases": len(rows),
    }


def _paired_bootstrap(
    case_rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    lookup = {
        (str(row["sample_id"]), str(row["mask_mode"]), str(row["sampler"])): row
        for row in case_rows
    }
    paired: list[dict[str, Any]] = []
    for key, probnet in lookup.items():
        sample_id, mask_mode, sampler = key
        if sampler != "probnet":
            continue
        comparator = lookup.get((sample_id, mask_mode, baseline))
        if comparator is None:
            continue
        item = {
            "sample_id": sample_id,
            "mask_mode": mask_mode,
            "dataset": probnet["dataset"],
            "case_id": probnet.get("case_id"),
        }
        for metric in METRICS:
            item[metric] = float(probnet[metric]) - float(comparator[metric])
        paired.append(item)

    by_dataset_cluster: dict[
        str, dict[str, list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in paired:
        by_dataset_cluster[str(row["dataset"])][
            str(row.get("case_id") or row["sample_id"])
        ].append(row)

    def macro(values: Sequence[Mapping[str, Any]], metric: str) -> float:
        dataset_values: dict[str, list[float]] = defaultdict(list)
        for value in values:
            dataset_values[str(value["dataset"])].append(float(value[metric]))
        return _mean(_mean(items) for items in dataset_values.values())

    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(repeats):
        sampled_rows: list[Mapping[str, Any]] = []
        for clusters in by_dataset_cluster.values():
            cluster_ids = list(clusters)
            sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            sampled_rows.extend(
                row for cluster_id in sampled for row in clusters[str(cluster_id)]
            )
        for metric in METRICS:
            draws[metric].append(macro(sampled_rows, metric))
    return {
        metric: {
            "baseline": baseline,
            "delta_probnet_minus_baseline": macro(paired, metric),
            "ci95_low": float(np.nanpercentile(values, 2.5)),
            "ci95_high": float(np.nanpercentile(values, 97.5)),
            "direction": (
                "negative_favors_probnet"
                if metric in ERROR_METRICS
                else "positive_favors_probnet"
            ),
        }
        for metric, values in draws.items()
    }


def report(args: argparse.Namespace) -> int:
    shard_root = args.output_root / "p1_spatial_ablation" / "shards"
    shard_paths = sorted(shard_root.glob("shard_[0-9][0-9].jsonl"))
    if len(shard_paths) != args.shards:
        raise RuntimeError(
            f"Expected {args.shards} shard files, found {len(shard_paths)}"
        )
    seed_rows = [
        row for path in shard_paths for row in read_jsonl(path)
    ]
    expected = args.expected_cases * args.replicates * len(SAMPLERS)
    if len(seed_rows) != expected:
        raise RuntimeError(f"Expected {expected} P1 rows, found {len(seed_rows)}")
    keys = {
        (
            row["sample_id"],
            row["mask_mode"],
            row["replicate"],
            row["sampler"],
        )
        for row in seed_rows
    }
    if len(keys) != len(seed_rows):
        raise RuntimeError("Duplicate P1 case/seed/sampler rows")
    case_rows = _aggregate_seed_rows(seed_rows)
    summary_by_sampler = {
        sampler: _sampler_summary(
            [row for row in case_rows if row["sampler"] == sampler]
        )
        for sampler in SAMPLERS
    }
    paired_uniform = _paired_bootstrap(
        case_rows,
        baseline=PRIMARY_BASELINE,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    paired_poisson = _paired_bootstrap(
        case_rows,
        baseline=SECONDARY_BASELINE,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    safety = summary_by_sampler["probnet"]["equal_dataset_macro"]
    baseline_safety = summary_by_sampler[PRIMARY_BASELINE][
        "equal_dataset_macro"
    ]
    relative_improvement = {
        metric: (
            (
                float(baseline_safety[metric]) - float(safety[metric])
            )
            / float(baseline_safety[metric])
            if (
                metric in ERROR_METRICS
                and np.isfinite(float(baseline_safety[metric]))
                and float(baseline_safety[metric]) > 0
            )
            else float("nan")
        )
        for metric in ERROR_METRICS
    }
    significant_primary_improvements = sorted(
        metric
        for metric, values in paired_uniform.items()
        if metric in ERROR_METRICS and values["ci95_high"] < 0
    )
    material_primary_improvements = sorted(
        metric
        for metric in significant_primary_improvements
        if relative_improvement[metric] >= MATERIAL_RELATIVE_IMPROVEMENT
    )
    primary_improvement = bool(significant_primary_improvements)
    primary_regression = any(
        values["ci95_low"] > 0
        for metric, values in paired_uniform.items()
        if metric in ERROR_METRICS
    )
    core_structure_improvement = any(
        metric in CORE_SPATIAL_STRUCTURE_METRICS
        for metric in material_primary_improvements
    )
    learned_improvement = bool(
        len(material_primary_improvements) >= 2
        and core_structure_improvement
        and not primary_regression
    )
    no_material_safety_regression = bool(
        safety["placement_completion"] >= 0.98
        and safety["placement_completion"]
        >= (
            baseline_safety["placement_completion"]
            - PLACEMENT_COMPLETION_NONINFERIORITY_MARGIN
        )
        and safety["retained_preservation_rate"] == 1.0
        and safety["overlap_pixels"] == 0.0
        and safety["outside_tissue_pixels"] == 0.0
    )
    output_root = args.output_root / "p1_spatial_ablation"
    merged_path = output_root / "per_seed_rows.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in seed_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    case_path = output_root / "per_case_sampler_rows.jsonl"
    with case_path.open("w", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    try:
        import pandas as pd

        pd.DataFrame(seed_rows).drop(
            columns=["accepted_centres_xy", "accepted_types"],
            errors="ignore",
        ).to_parquet(output_root / "per_seed_rows.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        parquet_status = f"not_written: {exc}"
    else:
        parquet_status = "written"
    summary = {
        "protocol": PROTOCOL,
        "checkpoint_role": "spatial_placement_probability_only",
        "primary_baseline": PRIMARY_BASELINE,
        "secondary_baseline": SECONDARY_BASELINE,
        "count_type_role": "oracle_hidden_count_and_exact_hidden_type_quotas",
        "candidate_shape_retry_contract": "shared_within_case_and_seed",
        "seed_rows": len(seed_rows),
        "case_sampler_rows": len(case_rows),
        "samplers": summary_by_sampler,
        "paired_probnet_vs_uniform": paired_uniform,
        "paired_probnet_vs_poisson_only": paired_poisson,
        "relative_error_reduction_vs_uniform": relative_improvement,
        "significant_primary_improvements_vs_uniform": (
            significant_primary_improvements
        ),
        "material_primary_improvements_vs_uniform": (
            material_primary_improvements
        ),
        "material_relative_improvement_threshold": (
            MATERIAL_RELATIVE_IMPROVEMENT
        ),
        "core_spatial_structure_improvement_detected": (
            core_structure_improvement
        ),
        "at_least_one_primary_improvement_detected": primary_improvement,
        "primary_spatial_regression_detected": primary_regression,
        "learned_spatial_improvement_detected": learned_improvement,
        "no_material_safety_regression": no_material_safety_regression,
        "conservative_safety_gate": {
            "placement_completion_at_least_0_98": (
                safety["placement_completion"] >= 0.98
            ),
            "placement_completion_noninferior_to_uniform": (
                safety["placement_completion"]
                >= (
                    baseline_safety["placement_completion"]
                    - PLACEMENT_COMPLETION_NONINFERIORITY_MARGIN
                )
            ),
            "placement_completion_noninferiority_margin": (
                PLACEMENT_COMPLETION_NONINFERIORITY_MARGIN
            ),
            "retained_preservation_exact": (
                safety["retained_preservation_rate"] == 1.0
            ),
            "overlap_pixels_zero": safety["overlap_pixels"] == 0.0,
            "full_shape_tissue_containment_exact": (
                safety["outside_tissue_pixels"] == 0.0
            ),
            "probnet_outside_tissue_pixels": safety[
                "outside_tissue_pixels"
            ],
            "uniform_outside_tissue_pixels": baseline_safety[
                "outside_tissue_pixels"
            ],
        },
        "claim_gate_passed": learned_improvement and no_material_safety_regression,
        "clear_learned_spatial_structure_claim_passed": (
            learned_improvement and no_material_safety_regression
        ),
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_unit": "case/WSI cluster within dataset; equal-dataset macro",
        "per_seed_parquet": parquet_status,
    }
    write_json(output_root / "paired_spatial_summary.json", summary)
    validation = {
        "protocol": PROTOCOL,
        "expected_rows": expected,
        "observed_rows": len(seed_rows),
        "unique_rows": len(keys),
        "expected_replicates": args.replicates,
        "expected_samplers": list(SAMPLERS),
        "complete": len(seed_rows) == expected and len(keys) == expected,
    }
    write_json(output_root / "validation.json", validation)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if validation["complete"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-shard")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--expected-checkpoint-sha256", required=True)
    run.add_argument("--library-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--expected-cases", type=int, default=1200)
    run.add_argument("--replicates", type=int, default=5)
    run.add_argument("--shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--base-ch", type=int, default=64)
    run.set_defaults(func=run_shard)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--output-root", type=Path, required=True)
    report_parser.add_argument("--expected-cases", type=int, default=1200)
    report_parser.add_argument("--replicates", type=int, default=5)
    report_parser.add_argument("--shards", type=int, default=3)
    report_parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    report_parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    report_parser.set_defaults(func=report)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
