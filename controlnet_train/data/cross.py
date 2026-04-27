"""Cross-reconstruction metadata builder and dataset."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from dataset_config import get_config

from .common import (
    LayeredSample,
    load_image_tensor,
    load_layered_dataset_samples,
    load_mask_array,
    load_nuclei_mask,
    load_tissue_mask,
    remap_nuclei_ids_array,
    split_records_by_case,
    write_json,
)

_COVERAGE_BUCKETS = ("full", "partial", "low")
_DEFAULT_FULL_COVERAGE_WEIGHT = 0.6
_DEFAULT_PARTIAL_COVERAGE_WEIGHT = 0.3
_DEFAULT_LOW_COVERAGE_WEIGHT = 0.1


def build_cross_metadata(
    dataset_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    num_ref_per_target: int = 2,
    val_ratio: float = 0.1,
    seed: int = 42,
    top_k: int = 8,
    full_coverage_weight: float = _DEFAULT_FULL_COVERAGE_WEIGHT,
    partial_coverage_weight: float = _DEFAULT_PARTIAL_COVERAGE_WEIGHT,
    low_coverage_weight: float = _DEFAULT_LOW_COVERAGE_WEIGHT,
    skip_invalid_samples: bool = True,
    progress_every: int = 0,
) -> tuple[Path, Path]:
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    coverage_weights = _validate_coverage_weights(
        full=full_coverage_weight,
        partial=partial_coverage_weight,
        low=low_coverage_weight,
    )

    all_pairs: list[dict] = []
    skipped_samples: list[dict] = []
    pair_counts_by_difficulty = {bucket: 0 for bucket in _COVERAGE_BUCKETS}
    for dataset_name, dataset_root in dataset_roots.items():
        samples = load_layered_dataset_samples(dataset_name, dataset_root)
        _log_progress(
            progress_every,
            f"[cross-meta] {dataset_name}: loaded {len(samples)} samples from {dataset_root}",
        )
        valid_samples: list[LayeredSample] = []
        summaries: dict[str, dict] = {}
        for sample_index, sample in enumerate(samples, start=1):
            try:
                summaries[sample.sample_id] = _summarize_sample(sample)
            except (OSError, ValueError) as exc:
                if not skip_invalid_samples:
                    raise
                skipped_samples.append(_skipped_sample_record(sample, exc))
                _log_progress(
                    progress_every,
                    (
                        f"[cross-meta] {dataset_name}: skipped invalid sample "
                        f"{sample.sample_id}: {type(exc).__name__}: {exc}"
                    ),
                )
                continue
            valid_samples.append(sample)
            if progress_every > 0 and sample_index % progress_every == 0:
                _log_progress(
                    progress_every,
                    (
                        f"[cross-meta] {dataset_name}: summarized {sample_index}/{len(samples)} "
                        f"samples, valid={len(valid_samples)}, skipped={len(skipped_samples)}"
                    ),
                )
        _log_progress(
            progress_every,
            (
                f"[cross-meta] {dataset_name}: summary complete, valid={len(valid_samples)}/"
                f"{len(samples)}, skipped_total={len(skipped_samples)}"
            ),
        )

        grouped: dict[str, list[LayeredSample]] = {}
        for sample in valid_samples:
            grouped.setdefault(sample.case_id, []).append(sample)

        targets_seen = 0
        for case_id, case_samples in grouped.items():
            if len(case_samples) < 2:
                continue

            for target in case_samples:
                targets_seen += 1
                target_summary = summaries[target.sample_id]
                candidates_by_bucket: dict[str, list[dict]] = {bucket: [] for bucket in _COVERAGE_BUCKETS}
                for reference in case_samples:
                    if reference.sample_id == target.sample_id:
                        continue
                    reference_summary = summaries[reference.sample_id]
                    coverage = _reference_coverage(
                        target_summary=target_summary,
                        reference_summary=reference_summary,
                    )
                    score = _score_reference(
                        target_summary=target_summary,
                        reference_summary=reference_summary,
                        target=target,
                        reference=reference,
                    )
                    candidates_by_bucket[coverage["pair_difficulty"]].append(
                        {
                            "score": score,
                            "sample": reference,
                            "coverage": coverage,
                        }
                    )

                if not any(candidates_by_bucket.values()):
                    continue

                chosen = _choose_references(
                    candidates_by_bucket=candidates_by_bucket,
                    num_ref_per_target=num_ref_per_target,
                    top_k=top_k,
                    coverage_weights=coverage_weights,
                    rng=rng,
                )
                for candidate in chosen:
                    reference = candidate["sample"]
                    all_pairs.append(_build_pair_record(target, reference, case_id, candidate["coverage"]))
                    pair_counts_by_difficulty[candidate["coverage"]["pair_difficulty"]] += 1
                if progress_every > 0 and targets_seen % progress_every == 0:
                    _log_progress(
                        progress_every,
                        (
                            f"[cross-meta] {dataset_name}: paired {targets_seen} targets, "
                            f"pairs={len(all_pairs)}, "
                            f"full={pair_counts_by_difficulty['full']}, "
                            f"partial={pair_counts_by_difficulty['partial']}, "
                            f"low={pair_counts_by_difficulty['low']}"
                        ),
                    )
        _log_progress(
            progress_every,
            (
                f"[cross-meta] {dataset_name}: pairing complete, targets={targets_seen}, "
                f"pairs_total={len(all_pairs)}"
            ),
        )

    train_pairs, val_pairs = split_records_by_case(
        all_pairs,
        case_id_getter=lambda record: f"{record['dataset']}::{record['case_id']}",
        val_ratio=val_ratio,
        seed=seed,
    )

    train_path = write_json(output_dir / "metadata_cross_train.json", {"pairs": train_pairs})
    val_path = write_json(output_dir / "metadata_cross_val.json", {"pairs": val_pairs})
    if skipped_samples:
        skipped_path = write_json(
            output_dir / "skipped_cross_samples.json",
            {
                "skipped_count": len(skipped_samples),
                "samples": skipped_samples,
            },
        )
        _log_progress(
            progress_every,
            f"[cross-meta] wrote skipped sample report: {skipped_path} ({len(skipped_samples)} samples)",
        )
    _log_progress(
        progress_every,
        (
            f"[cross-meta] done: train={len(train_pairs)}, val={len(val_pairs)}, "
            f"full={pair_counts_by_difficulty['full']}, "
            f"partial={pair_counts_by_difficulty['partial']}, "
            f"low={pair_counts_by_difficulty['low']}, skipped={len(skipped_samples)}"
        ),
    )
    return train_path, val_path


class CrossReconstructionDataset(torch.utils.data.Dataset):
    """Load normalized cross-reconstruction metadata and paired conditions."""

    def __init__(self, metadata_path: str | Path) -> None:
        metadata_path = Path(metadata_path)
        payload = json.loads(metadata_path.read_text(encoding="utf8"))
        self.records = payload["pairs"] if isinstance(payload, dict) else payload

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "dataset": record["dataset"],
            "sample_id": record["sample_id"],
            "reference_sample_id": record["reference_sample_id"],
            "case_id": record["case_id"],
            "target_image": load_image_tensor(record["target_image"]),
            "target_tissue_mask": load_tissue_mask(record["target_tissue_mask"]),
            "target_nuclei_mask": load_nuclei_mask(record["target_nuclei_mask"], remap=True),
            "reference_image": load_image_tensor(record["reference_image"]),
            "reference_tissue_mask": load_tissue_mask(record["reference_tissue_mask"]),
            "reference_nuclei_mask": load_nuclei_mask(record["reference_nuclei_mask"], remap=True),
            "prompt": record["prompt"],
            "distance": int(record["distance"]),
            "pair_difficulty": record.get("pair_difficulty", "full"),
            "tissue_coverage_ratio": float(record.get("tissue_coverage_ratio", 1.0)),
            "area_coverage_ratio": float(record.get("area_coverage_ratio", 1.0)),
            "missing_target_tissue_ids": record.get("missing_target_tissue_ids", []),
        }


def _summarize_sample(sample: LayeredSample) -> dict:
    config = get_config(sample.dataset_name)
    tissue = load_mask_array(sample.tissue_mask_path)
    nuclei = remap_nuclei_ids_array(load_mask_array(sample.nuclei_mask_path))
    image = load_image_tensor(sample.image_path).permute(1, 2, 0).numpy()

    tissue_ids = {int(value) for value in np.unique(tissue) if int(value) not in config.skip_tissues}
    tissue_area_by_id = {
        tissue_id: int(np.count_nonzero(tissue == tissue_id))
        for tissue_id in tissue_ids
    }
    nuclei_hist = np.bincount(nuclei.reshape(-1), minlength=6).astype(np.float32)
    if nuclei_hist.sum() > 0:
        nuclei_hist /= nuclei_hist.sum()
    stain_mean = image.mean(axis=(0, 1))

    return {
        "tissue_ids": tissue_ids,
        "tissue_area_by_id": tissue_area_by_id,
        "nuclei_hist": nuclei_hist,
        "stain_mean": stain_mean,
    }


def _validate_coverage_weights(*, full: float, partial: float, low: float) -> dict[str, float]:
    weights = {
        "full": float(full),
        "partial": float(partial),
        "low": float(low),
    }
    negative = {name: value for name, value in weights.items() if value < 0.0}
    if negative:
        raise ValueError(f"coverage weights must be non-negative, got {negative}.")
    if sum(weights.values()) <= 0.0:
        raise ValueError("At least one coverage weight must be positive.")
    return weights


def _reference_coverage(*, target_summary: dict, reference_summary: dict) -> dict:
    target_ids = set(target_summary["tissue_ids"])
    reference_ids = set(reference_summary["tissue_ids"])
    covered_ids = target_ids & reference_ids
    missing_ids = sorted(target_ids - reference_ids)

    if not target_ids or not missing_ids:
        pair_difficulty = "full"
    elif covered_ids:
        pair_difficulty = "partial"
    else:
        pair_difficulty = "low"

    target_area_by_id = target_summary["tissue_area_by_id"]
    target_area = sum(target_area_by_id.values())
    covered_area = sum(target_area_by_id[tissue_id] for tissue_id in covered_ids)

    return {
        "pair_difficulty": pair_difficulty,
        "tissue_coverage_ratio": len(covered_ids) / max(len(target_ids), 1),
        "area_coverage_ratio": covered_area / max(target_area, 1),
        "missing_target_tissue_ids": missing_ids,
        "covered_target_tissue_ids": sorted(covered_ids),
    }


def _choose_references(
    *,
    candidates_by_bucket: Mapping[str, list[dict]],
    num_ref_per_target: int,
    top_k: int,
    coverage_weights: Mapping[str, float],
    rng: random.Random,
) -> list[dict]:
    for candidates in candidates_by_bucket.values():
        candidates.sort(key=lambda item: item["score"], reverse=True)

    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    target_count = min(num_ref_per_target, sum(len(candidates) for candidates in candidates_by_bucket.values()))

    while len(chosen) < target_count:
        available_buckets = [
            bucket
            for bucket in _COVERAGE_BUCKETS
            if coverage_weights[bucket] > 0.0
            and _available_candidates(candidates_by_bucket[bucket], chosen_ids)
        ]
        if not available_buckets:
            break

        bucket = _weighted_bucket_choice(available_buckets, coverage_weights, rng)
        candidate = _sample_candidate(
            candidates=candidates_by_bucket[bucket],
            chosen_ids=chosen_ids,
            top_k=max(top_k, num_ref_per_target),
            rng=rng,
        )
        if candidate is None:
            break
        chosen.append(candidate)
        chosen_ids.add(candidate["sample"].sample_id)

    return chosen


def _available_candidates(candidates: list[dict], chosen_ids: set[str]) -> list[dict]:
    return [candidate for candidate in candidates if candidate["sample"].sample_id not in chosen_ids]


def _weighted_bucket_choice(
    buckets: list[str],
    coverage_weights: Mapping[str, float],
    rng: random.Random,
) -> str:
    total_weight = sum(coverage_weights[bucket] for bucket in buckets)
    cursor = rng.random() * total_weight
    for bucket in buckets:
        cursor -= coverage_weights[bucket]
        if cursor <= 0.0:
            return bucket
    return buckets[-1]


def _sample_candidate(
    *,
    candidates: list[dict],
    chosen_ids: set[str],
    top_k: int,
    rng: random.Random,
) -> dict | None:
    available = _available_candidates(candidates, chosen_ids)
    if not available:
        return None
    pool = available[: max(top_k, 1)]
    return rng.choice(pool)


def _build_pair_record(
    target: LayeredSample,
    reference: LayeredSample,
    case_id: str,
    coverage: Mapping[str, object],
) -> dict:
    return {
        "dataset": target.dataset_name,
        "sample_id": target.sample_id,
        "reference_sample_id": reference.sample_id,
        "case_id": case_id,
        "target_image": str(target.image_path),
        "target_tissue_mask": str(target.tissue_mask_path),
        "target_nuclei_mask": str(target.nuclei_mask_path),
        "reference_image": str(reference.image_path),
        "reference_tissue_mask": str(reference.tissue_mask_path),
        "reference_nuclei_mask": str(reference.nuclei_mask_path),
        "prompt": target.prompt,
        "distance": _manhattan_distance(target, reference),
        "pair_difficulty": coverage["pair_difficulty"],
        "tissue_coverage_ratio": coverage["tissue_coverage_ratio"],
        "area_coverage_ratio": coverage["area_coverage_ratio"],
        "missing_target_tissue_ids": coverage["missing_target_tissue_ids"],
        "covered_target_tissue_ids": coverage["covered_target_tissue_ids"],
    }


def _skipped_sample_record(sample: LayeredSample, exc: Exception) -> dict:
    return {
        "dataset": sample.dataset_name,
        "sample_id": sample.sample_id,
        "case_id": sample.case_id,
        "image": str(sample.image_path),
        "tissue_mask": str(sample.tissue_mask_path),
        "nuclei_mask": str(sample.nuclei_mask_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _log_progress(progress_every: int, message: str) -> None:
    if progress_every > 0:
        print(message, file=sys.stderr, flush=True)


def _score_reference(
    *,
    target_summary: dict,
    reference_summary: dict,
    target: LayeredSample,
    reference: LayeredSample,
) -> float:
    tissue_coverage_score = len(target_summary["tissue_ids"] & reference_summary["tissue_ids"]) / max(
        len(target_summary["tissue_ids"]),
        1,
    )
    nuclei_hist_similarity = 1.0 - float(
        np.abs(target_summary["nuclei_hist"] - reference_summary["nuclei_hist"]).sum() / 2.0
    )
    stain_distance = float(np.linalg.norm(target_summary["stain_mean"] - reference_summary["stain_mean"]))
    stain_similarity = 1.0 / (1.0 + stain_distance)
    distance_penalty = _manhattan_distance(target, reference) / 4096.0
    return tissue_coverage_score * 10.0 + stain_similarity * 2.0 + nuclei_hist_similarity - distance_penalty


def _manhattan_distance(target: LayeredSample, reference: LayeredSample) -> int:
    if target.patch_y is None or target.patch_x is None or reference.patch_y is None or reference.patch_x is None:
        return 0
    return abs(target.patch_x - reference.patch_x) + abs(target.patch_y - reference.patch_y)
