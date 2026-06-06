#!/usr/bin/env python3
"""Run object-aware warp on one pair from metadata_cross_{train,val}.json.

This helper is meant for the training machine where absolute paths in the
metadata are available. It selects one usable pair and writes:
  - preview panel
  - *_warped_rgb.png
  - *_validity.png
  - *_object_mask.png
  - *_background.png
  - *_objects.png
  - *.selected.json
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from controlnet_train.modules.object_aware_warp import (
        _load_label,
        _load_rgb,
        _resize_label,
        object_aware_warp,
        visualize_object_warp,
        _save_gray,
        _save_rgb,
    )
except ImportError:
    from object_aware_warp import (
        _load_label,
        _load_rgb,
        _resize_label,
        object_aware_warp,
        visualize_object_warp,
        _save_gray,
        _save_rgb,
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
    p = argparse.ArgumentParser(description="Run object-aware warp on one cross metadata pair.")
    p.add_argument("--metadata", default=DEFAULT_METADATA)
    p.add_argument("--out", default="phase5_runs/warp_preview/object_aware_warp_metadata_sample.png")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-scan", type=int, default=512)
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.add_argument("--selected-json", default=None)
    p.add_argument("--save-prefix", default=None)

    p.add_argument("--size", type=int, default=256)
    p.add_argument("--nuclei-mode", choices=["auto", "instance", "connected"], default="auto")
    p.add_argument("--min-nucleus-area", type=int, default=5)
    p.add_argument("--nucleus-crop-margin", type=int, default=8)
    p.add_argument("--nucleus-topk", type=int, default=5)
    p.add_argument("--nucleus-tau", type=float, default=0.2)
    p.add_argument("--nucleus-context-scale", type=float, default=1.6)
    p.add_argument("--nucleus-feather-sigma", type=float, default=1.0)

    p.add_argument("--bg-patch-size", type=int, default=31)
    p.add_argument("--bg-stride", type=int, default=8)
    p.add_argument("--bg-min-purity", type=float, default=0.75)
    p.add_argument("--bg-max-nuclei-frac", type=float, default=0.25)
    p.add_argument("--bg-topk", type=int, default=5)
    p.add_argument("--bg-tau", type=float, default=0.2)
    p.add_argument("--density-sigma", type=float, default=3.0)
    return p


def read_cross_metadata(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if isinstance(payload, dict):
        records = payload.get("pairs")
        if not isinstance(records, list):
            raise ValueError("cross metadata dict must contain a 'pairs' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported metadata payload type: {type(payload)!r}")


def resolve_paths(record: dict[str, Any], metadata_dir: Path) -> dict[str, Path]:
    missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    paths: dict[str, Path] = {}
    for field in REQUIRED_FIELDS:
        p = Path(str(record[field]))
        paths[field] = p if p.is_absolute() else metadata_dir / p
    return paths


def paths_exist(paths: dict[str, Path]) -> bool:
    return all(p.exists() for p in paths.values())


def joint_coverage(paths: dict[str, Path], *, size: int) -> dict[str, float]:
    ref_t = _resize_label(_load_label(paths["reference_tissue_mask"]), size)
    ref_n = _resize_label(_load_label(paths["reference_nuclei_mask"]), size)
    tar_t = _resize_label(_load_label(paths["target_tissue_mask"]), size)
    tar_n = _resize_label(_load_label(paths["target_nuclei_mask"]), size)

    nuc_classes = int(max(ref_n.max(initial=0), tar_n.max(initial=0))) + 1
    ref_joint = ref_t * nuc_classes + ref_n
    tar_joint = tar_t * nuc_classes + tar_n
    ref_classes = set(np.unique(ref_joint).tolist())
    tar_flat = tar_joint.reshape(-1)
    covered = np.fromiter((v in ref_classes for v in tar_flat), dtype=bool, count=tar_flat.size)
    return {
        "joint_coverage": float(covered.mean()),
        "reference_joint_classes": float(len(ref_classes)),
        "target_joint_classes": float(len(np.unique(tar_joint))),
    }


def select_record(
    records: list[dict[str, Any]],
    *,
    metadata_dir: Path,
    index: int | None,
    seed: int,
    max_scan: int,
    min_coverage: float,
    size: int,
) -> tuple[int, dict[str, Any], dict[str, Path], dict[str, float]]:
    if index is not None:
        record = records[index]
        paths = resolve_paths(record, metadata_dir)
        if not paths_exist(paths):
            raise FileNotFoundError({k: str(v) for k, v in paths.items() if not v.exists()})
        stats = joint_coverage(paths, size=size)
        return index, record, paths, stats

    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    best = None
    checked = 0
    for i in order:
        if checked >= max_scan:
            break
        record = records[i]
        try:
            paths = resolve_paths(record, metadata_dir)
            if not paths_exist(paths):
                continue
            stats = joint_coverage(paths, size=size)
        except Exception:
            continue
        checked += 1
        cand = (i, record, paths, stats)
        if best is None or stats["joint_coverage"] > best[3]["joint_coverage"]:
            best = cand
        if stats["joint_coverage"] >= min_coverage:
            return cand
    if best is None:
        raise RuntimeError(f"no usable record found after scanning {max_scan} records")
    return best


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = Path(args.metadata)
    records = read_cross_metadata(metadata)
    index, record, paths, stats = select_record(
        records,
        metadata_dir=metadata.parent,
        index=args.index,
        seed=args.seed,
        max_scan=args.max_scan,
        min_coverage=args.min_coverage,
        size=args.size,
    )

    ref_rgb = _load_rgb(paths["reference_image"])
    tar_rgb = _load_rgb(paths["target_image"])
    ref_t = _load_label(paths["reference_tissue_mask"])
    ref_n = _load_label(paths["reference_nuclei_mask"])
    tar_t = _load_label(paths["target_tissue_mask"])
    tar_n = _load_label(paths["target_nuclei_mask"])

    result = object_aware_warp(
        ref_rgb=ref_rgb,
        ref_tissue_mask=ref_t,
        ref_nuclei_mask=ref_n,
        tar_tissue_mask=tar_t,
        tar_nuclei_mask=tar_n,
        size=args.size,
        nuclei_mode=args.nuclei_mode,
        seed=args.seed,
        min_nucleus_area=args.min_nucleus_area,
        nucleus_crop_margin=args.nucleus_crop_margin,
        nucleus_topk=args.nucleus_topk,
        nucleus_tau=args.nucleus_tau,
        nucleus_context_scale=args.nucleus_context_scale,
        nucleus_feather_sigma=args.nucleus_feather_sigma,
        bg_patch_size=args.bg_patch_size,
        bg_stride=args.bg_stride,
        bg_min_purity=args.bg_min_purity,
        bg_max_nuclei_frac=args.bg_max_nuclei_frac,
        bg_topk=args.bg_topk,
        bg_tau=args.bg_tau,
        density_sigma=args.density_sigma,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_object_warp(
        ref_rgb=np.asarray(__import__("PIL").Image.open(paths["reference_image"]).convert("RGB").resize((args.size, args.size))) / 255.0,
        ref_tissue=_resize_label(ref_t, args.size),
        ref_nuclei=_resize_label(ref_n, args.size),
        tar_rgb=np.asarray(__import__("PIL").Image.open(paths["target_image"]).convert("RGB").resize((args.size, args.size))) / 255.0,
        tar_tissue=_resize_label(tar_t, args.size),
        tar_nuclei=_resize_label(tar_n, args.size),
        result=result,
        out_path=out_path,
    )

    prefix = Path(args.save_prefix) if args.save_prefix else out_path.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _save_rgb(prefix.with_name(prefix.name + "_warped_rgb.png"), result["warped_rgb"])
    _save_gray(prefix.with_name(prefix.name + "_validity.png"), result["warped_validity"])
    _save_gray(prefix.with_name(prefix.name + "_object_mask.png"), result["warped_object_mask"])
    _save_rgb(prefix.with_name(prefix.name + "_background.png"), result["background_canvas"])
    _save_rgb(prefix.with_name(prefix.name + "_objects.png"), result["object_canvas"])

    selected = {
        "metadata": str(metadata),
        "index": index,
        "sample_id": record.get("sample_id") or paths["target_image"].stem,
        "reference_sample_id": record.get("reference_sample_id") or paths["reference_image"].stem,
        "paths": {k: str(v) for k, v in paths.items()},
        **stats,
        "out": str(out_path),
        "save_prefix": str(prefix),
        "num_nucleus_matches": len(result["nucleus_matches"]),
        "num_tissue_patches": len(result["tissue_patch_table"]),
        "nucleus_matches_preview": result["nucleus_matches"][:100],
    }
    selected_json = Path(args.selected_json) if args.selected_json else out_path.with_suffix(".selected.json")
    selected_json.parent.mkdir(parents=True, exist_ok=True)
    selected_json.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf8")

    print(f"selected index={index} sample_id={selected['sample_id']} ref={selected['reference_sample_id']}")
    print(f"joint_coverage={stats['joint_coverage']:.4f}")
    print(f"saved preview: {out_path}")
    print(f"saved selected record: {selected_json}")
    print(f"saved condition candidates with prefix: {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
