"""Cross-reconstruction metadata builder and dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, Sequence

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


def build_cross_metadata(
    dataset_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    num_ref_per_target: int = 2,
    val_ratio: float = 0.1,
    seed: int = 42,
    top_k: int = 8,
) -> tuple[Path, Path]:
    rng = random.Random(seed)
    output_dir = Path(output_dir)

    all_pairs: list[dict] = []
    for dataset_name, dataset_root in dataset_roots.items():
        samples = load_layered_dataset_samples(dataset_name, dataset_root)
        summaries = {sample.sample_id: _summarize_sample(sample) for sample in samples}

        grouped: dict[str, list[LayeredSample]] = {}
        for sample in samples:
            grouped.setdefault(sample.case_id, []).append(sample)

        for case_id, case_samples in grouped.items():
            if len(case_samples) < 2:
                continue

            for target in case_samples:
                target_summary = summaries[target.sample_id]
                candidates: list[tuple[float, LayeredSample]] = []
                for reference in case_samples:
                    if reference.sample_id == target.sample_id:
                        continue
                    reference_summary = summaries[reference.sample_id]
                    if not target_summary["tissue_ids"].issubset(reference_summary["tissue_ids"]):
                        continue
                    score = _score_reference(
                        target_summary=target_summary,
                        reference_summary=reference_summary,
                        target=target,
                        reference=reference,
                    )
                    candidates.append((score, reference))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: item[0], reverse=True)
                candidate_pool = [reference for _, reference in candidates[: max(top_k, num_ref_per_target)]]
                chosen = rng.sample(candidate_pool, k=min(num_ref_per_target, len(candidate_pool)))
                for reference in chosen:
                    all_pairs.append(
                        {
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
                        }
                    )

    train_pairs, val_pairs = split_records_by_case(
        all_pairs,
        case_id_getter=lambda record: f"{record['dataset']}::{record['case_id']}",
        val_ratio=val_ratio,
        seed=seed,
    )

    train_path = write_json(output_dir / "metadata_cross_train.json", {"pairs": train_pairs})
    val_path = write_json(output_dir / "metadata_cross_val.json", {"pairs": val_pairs})
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
        }


def _summarize_sample(sample: LayeredSample) -> dict:
    config = get_config(sample.dataset_name)
    tissue = load_mask_array(sample.tissue_mask_path)
    nuclei = remap_nuclei_ids_array(load_mask_array(sample.nuclei_mask_path))
    image = load_image_tensor(sample.image_path).permute(1, 2, 0).numpy()

    tissue_ids = {int(value) for value in np.unique(tissue) if int(value) not in config.skip_tissues}
    nuclei_hist = np.bincount(nuclei.reshape(-1), minlength=6).astype(np.float32)
    if nuclei_hist.sum() > 0:
        nuclei_hist /= nuclei_hist.sum()
    stain_mean = image.mean(axis=(0, 1))

    return {
        "tissue_ids": tissue_ids,
        "nuclei_hist": nuclei_hist,
        "stain_mean": stain_mean,
    }


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
