#!/usr/bin/env python3
"""Render ``warp_preview`` from a sample in ``metadata_cross_{train,val}.json``.

This helper is meant for the training machine where the absolute paths inside
the cross metadata are available. It scans records, chooses a pair whose
target joint classes are covered by the reference, and writes the preview panel.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.modules.warp_preview import (  # noqa: E402
    _load_label,
    _load_rgb,
    _resize_label,
    _resize_rgb,
    compute_patch_warp,
    compute_warp,
    visualize,
)


DEFAULT_METADATA = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_meta/metadata_cross_train.json"
REQUIRED_FIELDS = (
    "reference_image",
    "target_image",
    "reference_tissue_mask",
    "reference_nuclei_mask",
    "target_tissue_mask",
    "target_nuclei_mask",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run warp_preview on one cross metadata pair.")
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument(
        "--out",
        default="phase5_runs/warp_preview/warp_preview_metadata_sample.png",
        help="Output preview PNG path.",
    )
    parser.add_argument("--index", type=int, default=None, help="Use a specific metadata record index.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when scanning records.")
    parser.add_argument("--max-scan", type=int, default=512, help="Maximum records to score when auto-selecting.")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="Stop scanning when target joint-class coverage reaches this value.",
    )
    parser.add_argument("--corr-size", type=int, default=192)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--smooth", type=int, default=3, help="match-field median filter size for hard copy")
    parser.add_argument("--gate", choices=["tissue", "tissue_nucbin", "joint"], default="tissue")
    parser.add_argument("--baseline-gate", choices=["tissue", "tissue_nucbin", "joint"], default="joint")
    parser.add_argument("--baseline-max-ref", type=int, default=2048)
    parser.add_argument("--no-soft", action="store_true", help="Skip expensive soft baseline and reuse mean panel.")
    parser.add_argument("--patch-size", type=int, default=21)
    parser.add_argument("--patch-stride", type=int, default=6)
    parser.add_argument("--patch-topk", type=int, default=1)
    parser.add_argument("--patch-tau", type=float, default=0.05)
    parser.add_argument("--patch-smooth", type=int, default=3)
    parser.add_argument("--density-sigma", type=float, default=3.0)
    parser.add_argument(
        "--selected-json",
        default=None,
        help="Optional JSON path for the selected record and coverage stats.",
    )
    return parser


def read_cross_metadata(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if isinstance(payload, dict):
        records = payload.get("pairs")
        if not isinstance(records, list):
            raise ValueError("cross metadata dict must contain a 'pairs' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported cross metadata payload type: {type(payload)!r}")


def resolve_record_paths(record: dict[str, Any], *, metadata_dir: Path) -> dict[str, Path]:
    missing = [field for field in REQUIRED_FIELDS if not record.get(field)]
    if missing:
        raise ValueError(f"record is missing required field(s): {missing}")
    paths: dict[str, Path] = {}
    for field in REQUIRED_FIELDS:
        path = Path(str(record[field]))
        paths[field] = path if path.is_absolute() else metadata_dir / path
    return paths


def all_paths_exist(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def joint_coverage(
    *,
    ref_tissue: np.ndarray,
    ref_nuclei: np.ndarray,
    target_tissue: np.ndarray,
    target_nuclei: np.ndarray,
    corr_size: int,
) -> dict[str, float]:
    ref_t = _resize_label(ref_tissue, corr_size)
    ref_n = _resize_label(ref_nuclei, corr_size)
    tar_t = _resize_label(target_tissue, corr_size)
    tar_n = _resize_label(target_nuclei, corr_size)

    nuc_classes = int(max(ref_n.max(initial=0), tar_n.max(initial=0))) + 1
    ref_joint = ref_t * nuc_classes + ref_n
    tar_joint = tar_t * nuc_classes + tar_n
    ref_classes = set(np.unique(ref_joint).tolist())
    tar_flat = tar_joint.reshape(-1)
    covered = np.fromiter((value in ref_classes for value in tar_flat), dtype=bool, count=tar_flat.size)
    return {
        "joint_coverage": float(covered.mean()),
        "reference_joint_classes": float(len(ref_classes)),
        "target_joint_classes": float(len(np.unique(tar_joint))),
        "target_pixels": float(tar_flat.size),
    }


def score_record(record: dict[str, Any], *, metadata_dir: Path, corr_size: int) -> tuple[dict[str, Path], dict[str, float]]:
    paths = resolve_record_paths(record, metadata_dir=metadata_dir)
    if not all_paths_exist(paths):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(f"missing path(s): {missing}")
    stats = joint_coverage(
        ref_tissue=_load_label(str(paths["reference_tissue_mask"])),
        ref_nuclei=_load_label(str(paths["reference_nuclei_mask"])),
        target_tissue=_load_label(str(paths["target_tissue_mask"])),
        target_nuclei=_load_label(str(paths["target_nuclei_mask"])),
        corr_size=corr_size,
    )
    return paths, stats


def select_record(
    records: list[dict[str, Any]],
    *,
    metadata_dir: Path,
    index: int | None,
    seed: int,
    max_scan: int,
    min_coverage: float,
    corr_size: int,
) -> tuple[int, dict[str, Any], dict[str, Path], dict[str, float]]:
    if index is not None:
        record = records[index]
        paths, stats = score_record(record, metadata_dir=metadata_dir, corr_size=corr_size)
        return index, record, paths, stats

    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    best: tuple[int, dict[str, Any], dict[str, Path], dict[str, float]] | None = None
    checked = 0
    for record_index in order:
        if checked >= max_scan:
            break
        record = records[record_index]
        try:
            paths, stats = score_record(record, metadata_dir=metadata_dir, corr_size=corr_size)
        except (FileNotFoundError, ValueError):
            continue
        checked += 1
        candidate = (record_index, record, paths, stats)
        if best is None or stats["joint_coverage"] > best[3]["joint_coverage"]:
            best = candidate
        if stats["joint_coverage"] >= min_coverage:
            return candidate
    if best is None:
        raise RuntimeError(f"no usable metadata record found after scanning up to {max_scan} existing records")
    return best


def render_preview(
    *,
    paths: dict[str, Path],
    out_path: Path,
    corr_size: int,
    tau: float,
    smooth: int,
    seed: int,
    gate: str,
    baseline_gate: str,
    baseline_max_ref: int,
    no_soft: bool,
    patch_size: int,
    patch_stride: int,
    patch_topk: int,
    patch_tau: float,
    patch_smooth: int,
    density_sigma: float,
) -> None:
    ref_rgb = _load_rgb(str(paths["reference_image"]))
    tar_rgb = _load_rgb(str(paths["target_image"]))
    ref_t = _load_label(str(paths["reference_tissue_mask"]))
    ref_n = _load_label(str(paths["reference_nuclei_mask"]))
    tar_t = _load_label(str(paths["target_tissue_mask"]))
    tar_n = _load_label(str(paths["target_nuclei_mask"]))

    common = dict(corr_size=corr_size, tau=tau, smooth=smooth, gate=baseline_gate, seed=seed, max_ref=baseline_max_ref)
    warped_mean, _ = compute_warp(ref_rgb, ref_t, ref_n, tar_t, tar_n, mode="mean", **common)
    if no_soft:
        warped_soft = warped_mean.copy()
    else:
        warped_soft, _ = compute_warp(ref_rgb, ref_t, ref_n, tar_t, tar_n, mode="soft", **common)
    warped_hard, validity = compute_warp(ref_rgb, ref_t, ref_n, tar_t, tar_n, mode="hard", **common)
    warped_patch, patch_conf, _ = compute_patch_warp(
        ref_rgb,
        ref_t,
        ref_n,
        tar_t,
        tar_n,
        corr_size=corr_size,
        gate=gate,
        patch_size=patch_size,
        patch_stride=patch_stride,
        patch_topk=patch_topk,
        tau=patch_tau,
        smooth=patch_smooth,
        density_sigma=density_sigma,
        seed=seed,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualize(
        _resize_rgb(ref_rgb, corr_size),
        _resize_label(ref_t, corr_size),
        _resize_label(ref_n, corr_size),
        _resize_rgb(tar_rgb, corr_size),
        _resize_label(tar_t, corr_size),
        _resize_label(tar_n, corr_size),
        warped_mean,
        warped_soft,
        warped_hard,
        warped_patch,
        validity,
        patch_conf,
        str(out_path),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata_path = Path(args.metadata)
    records = read_cross_metadata(metadata_path)
    index, record, paths, stats = select_record(
        records,
        metadata_dir=metadata_path.parent,
        index=args.index,
        seed=args.seed,
        max_scan=args.max_scan,
        min_coverage=args.min_coverage,
        corr_size=args.corr_size,
    )

    out_path = Path(args.out)
    render_preview(
        paths=paths,
        out_path=out_path,
        corr_size=args.corr_size,
        tau=args.tau,
        smooth=args.smooth,
        seed=args.seed,
        gate=args.gate,
        baseline_gate=args.baseline_gate,
        baseline_max_ref=args.baseline_max_ref,
        no_soft=args.no_soft,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_topk=args.patch_topk,
        patch_tau=args.patch_tau,
        patch_smooth=args.patch_smooth,
        density_sigma=args.density_sigma,
    )

    selected = {
        "metadata": str(metadata_path),
        "index": index,
        "sample_id": record.get("sample_id") or Path(str(record["target_tissue_mask"])).stem,
        "reference_sample_id": record.get("reference_sample_id") or Path(str(record["reference_image"])).stem,
        "dataset": record.get("dataset"),
        "paths": {field: str(path) for field, path in paths.items()},
        **stats,
        "smooth": args.smooth,
        "gate": args.gate,
        "baseline_gate": args.baseline_gate,
        "patch_size": args.patch_size,
        "patch_stride": args.patch_stride,
        "patch_topk": args.patch_topk,
        "patch_smooth": args.patch_smooth,
        "out": str(out_path),
    }
    selected_json = Path(args.selected_json) if args.selected_json else out_path.with_suffix(".selected.json")
    selected_json.parent.mkdir(parents=True, exist_ok=True)
    selected_json.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf8")

    print(f"selected index={index} sample_id={selected['sample_id']} ref={selected['reference_sample_id']}")
    print(f"joint_coverage={stats['joint_coverage']:.4f}")
    print(f"saved preview: {out_path}")
    print(f"saved selected record: {selected_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
