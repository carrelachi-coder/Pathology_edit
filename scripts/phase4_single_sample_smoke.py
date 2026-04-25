#!/usr/bin/env python3
"""Single-sample Phase 4 ProbNet smoke test.

This script starts from a complete layered sample:
  tissue mask + nuclei mask -> erase part of the nuclei layer -> ProbNet generate.
"""

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import get_config
from inpaint_cells.data.prepare_dataset import (
    generate_erasure_region,
    generate_full_image_erasure,
    generate_large_region_erasure,
    generate_local_erasure,
    generate_negative_erasure,
)
from inpaint_cells.generate import (
    draw_edit_contour,
    load_checkpoint_model,
    load_density_scale,
    run_single,
)
from inpaint_cells.nuclei_library.library import NucleiLibrary
from inpaint_cells.utils.mask_utils import load_nuclei_mask, load_tissue_mask, overlay


NUCLEI_RAW_IDS = (101, 102, 103, 104, 105)
ERASURE_FUNCTIONS = {
    "auto": generate_erasure_region,
    "local": generate_local_erasure,
    "large_region": generate_large_region_erasure,
    "full_image": generate_full_image_erasure,
    "negative": generate_negative_erasure,
}


@dataclass(frozen=True)
class ProbeInputs:
    tissue_path: Path
    gt_nuclei_path: Path
    erased_nuclei_path: Path
    edit_region_path: Path
    metadata_path: Path


def _read_gray(path: Path, label: str) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Cannot load {label}: {path}")
    return arr


def _copy_or_write_mask(source: Path, target: Path, values: np.ndarray) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    else:
        cv2.imwrite(str(target), values.astype(np.uint8))


def _type_pixel_counts(mask: np.ndarray, edit_mask: np.ndarray) -> dict[str, int]:
    counts = {}
    region = mask[edit_mask]
    for raw_id in NUCLEI_RAW_IDS:
        n = int(np.count_nonzero(region == raw_id))
        if n > 0:
            counts[str(raw_id)] = n
    return counts


def expand_mask_to_full_nuclei_components(candidate_mask: np.ndarray, nuclei_map: np.ndarray) -> np.ndarray:
    """Expand a candidate changed mask so intersecting nuclei components are erased whole."""
    if candidate_mask.shape != nuclei_map.shape:
        raise ValueError(f"Shape mismatch: candidate={candidate_mask.shape}, nuclei={nuclei_map.shape}")

    expanded = candidate_mask.astype(bool).copy()
    for raw_id in NUCLEI_RAW_IDS:
        binary = (nuclei_map == raw_id).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        for label_id in range(1, num_labels):
            component = labels == label_id
            if np.any(component & candidate_mask):
                expanded |= component
    return expanded


def compare_nuclei_in_edit_region(
    gt_nuclei: np.ndarray,
    pred_nuclei: np.ndarray,
    edit_mask: np.ndarray,
) -> dict:
    gt_pixels = int(np.count_nonzero((gt_nuclei > 0) & edit_mask))
    pred_pixels = int(np.count_nonzero((pred_nuclei > 0) & edit_mask))
    return {
        "edit_region_pixels": int(np.count_nonzero(edit_mask)),
        "gt_nuclei_pixels": gt_pixels,
        "pred_nuclei_pixels": pred_pixels,
        "pixel_density_ratio": float(pred_pixels / gt_pixels) if gt_pixels > 0 else None,
        "gt_type_pixel_counts": _type_pixel_counts(gt_nuclei, edit_mask),
        "pred_type_pixel_counts": _type_pixel_counts(pred_nuclei, edit_mask),
    }


def _labeled_row(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if not panels:
        raise ValueError("At least one panel is required.")

    h, _w = panels[0][1].shape[:2]
    row = np.concatenate([panel for _label, panel in panels], axis=1)
    labeled = np.zeros((h + 34, row.shape[1], 3), dtype=np.uint8)
    labeled[:34] = 35
    labeled[34:] = row

    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_w = panels[0][1].shape[1]
    for idx, (label, _panel) in enumerate(panels):
        cv2.putText(labeled, label, (idx * panel_w + 6, 23), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def write_gt_pred_comparison(
    tissue_path: Path,
    gt_nuclei_path: Path,
    erased_nuclei_path: Path,
    pred_nuclei_path: Path,
    edit_region_path: Path,
    output_path: Path,
) -> Path:
    tissue = load_tissue_mask(str(tissue_path))
    gt_nuclei = load_nuclei_mask(str(gt_nuclei_path), remap=True)
    erased_nuclei = load_nuclei_mask(str(erased_nuclei_path), remap=True)
    pred_nuclei = load_nuclei_mask(str(pred_nuclei_path), remap=True)
    edit_mask = _read_gray(edit_region_path, "edit region") > 0

    panels = [
        ("GT", draw_edit_contour(overlay(tissue, gt_nuclei), edit_mask)),
        ("input", draw_edit_contour(overlay(tissue, erased_nuclei), edit_mask)),
        ("pred", draw_edit_contour(overlay(tissue, pred_nuclei), edit_mask)),
    ]
    comparison = _labeled_row(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    return output_path


def make_erasure_mask(
    tissue_map: np.ndarray,
    nuclei_map: np.ndarray,
    skip_tissues: set[int],
    erasure_mode: str,
    seed: int,
    max_attempts: int = 20,
    require_cells: bool = True,
    min_erased_nuclei_pixels: int = 0,
    min_erased_nuclei_fraction: float = 0.0,
    min_edit_region_pixels: int = 0,
) -> tuple[np.ndarray, str]:
    if tissue_map.shape != nuclei_map.shape:
        raise ValueError(f"Shape mismatch: tissue={tissue_map.shape}, nuclei={nuclei_map.shape}")
    if erasure_mode not in ERASURE_FUNCTIONS:
        raise ValueError(f"Unsupported erasure mode: {erasure_mode}")
    if min_erased_nuclei_pixels < 0 or min_edit_region_pixels < 0:
        raise ValueError("Minimum pixel thresholds must be non-negative.")
    if min_erased_nuclei_fraction < 0 or min_erased_nuclei_fraction > 1:
        raise ValueError("--min-erased-nuclei-fraction must be in [0, 1].")

    cell_mask = np.isin(nuclei_map, NUCLEI_RAW_IDS)
    total_cell_pixels = int(np.count_nonzero(cell_mask))
    if require_cells and not np.any(cell_mask):
        raise ValueError("Input nuclei mask has no raw nuclei IDs in 101-105.")
    required_erased_pixels = max(
        int(min_erased_nuclei_pixels),
        int(np.ceil(total_cell_pixels * min_erased_nuclei_fraction)),
    )

    fn = ERASURE_FUNCTIONS[erasure_mode]
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        result = fn(tissue_map, cell_mask, skip_tissues, rng)
        if result is None:
            continue
        edit_mask, mode_name = result
        edit_mask = expand_mask_to_full_nuclei_components(edit_mask.astype(bool), nuclei_map)
        if not np.any(edit_mask):
            continue
        if np.count_nonzero(edit_mask) < min_edit_region_pixels:
            continue
        erased_nuclei_pixels = int(np.count_nonzero(edit_mask & cell_mask))
        if require_cells and not np.any(edit_mask & cell_mask):
            continue
        if erased_nuclei_pixels < required_erased_pixels:
            continue
        return edit_mask, mode_name

    raise RuntimeError(
        f"Could not create a valid erasure mask after {max_attempts} attempts "
        f"(mode={erasure_mode}, require_cells={require_cells}, "
        f"min_erased_nuclei_pixels={required_erased_pixels}, "
        f"min_edit_region_pixels={min_edit_region_pixels})."
    )


def build_single_sample_probe_inputs(
    dataset: str,
    tissue_path: Path,
    nuclei_path: Path,
    output_dir: Path,
    erasure_mode: str = "local",
    seed: int = 42,
    max_attempts: int = 20,
    min_erased_nuclei_pixels: int = 0,
    min_erased_nuclei_fraction: float = 0.0,
    min_edit_region_pixels: int = 0,
) -> ProbeInputs:
    config = get_config(dataset)
    tissue_map = _read_gray(tissue_path, "tissue mask").astype(np.int64)
    nuclei_map = _read_gray(nuclei_path, "nuclei mask").astype(np.int64)

    edit_mask, actual_mode = make_erasure_mask(
        tissue_map=tissue_map,
        nuclei_map=nuclei_map,
        skip_tissues=set(config.skip_tissues),
        erasure_mode=erasure_mode,
        seed=seed,
        max_attempts=max_attempts,
        require_cells=(erasure_mode != "negative"),
        min_erased_nuclei_pixels=0 if erasure_mode == "negative" else min_erased_nuclei_pixels,
        min_erased_nuclei_fraction=0.0 if erasure_mode == "negative" else min_erased_nuclei_fraction,
        min_edit_region_pixels=min_edit_region_pixels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_tissue = output_dir / "input_tissue.png"
    out_gt_nuclei = output_dir / "gt_nuclei.png"
    out_erased_nuclei = output_dir / "input_nuclei_erased.png"
    out_edit_region = output_dir / "edit_region.png"
    out_metadata = output_dir / "metadata.json"

    erased = nuclei_map.copy()
    erased[edit_mask] = 0

    _copy_or_write_mask(tissue_path, out_tissue, tissue_map)
    _copy_or_write_mask(nuclei_path, out_gt_nuclei, nuclei_map)
    cv2.imwrite(str(out_erased_nuclei), erased.astype(np.uint8))
    cv2.imwrite(str(out_edit_region), edit_mask.astype(np.uint8) * 255)

    metadata = {
        "dataset": config.name,
        "cancer_type": config.cancer_type,
        "cancer_type_index": config.cancer_type_index,
        "source_tissue": str(tissue_path),
        "source_nuclei": str(nuclei_path),
        "erasure_mode": actual_mode,
        "requested_erasure_mode": erasure_mode,
        "seed": seed,
        "edit_region_pixels": int(np.count_nonzero(edit_mask)),
        "gt_nuclei_pixels_in_edit": int(np.count_nonzero((nuclei_map > 0) & edit_mask)),
        "gt_nuclei_pixels_total": int(np.count_nonzero(nuclei_map > 0)),
        "gt_nuclei_fraction_in_edit": float(
            np.count_nonzero((nuclei_map > 0) & edit_mask) / max(np.count_nonzero(nuclei_map > 0), 1)
        ),
        "min_erased_nuclei_pixels": min_erased_nuclei_pixels,
        "min_erased_nuclei_fraction": min_erased_nuclei_fraction,
        "min_edit_region_pixels": min_edit_region_pixels,
    }
    out_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf8")

    return ProbeInputs(
        tissue_path=out_tissue,
        gt_nuclei_path=out_gt_nuclei,
        erased_nuclei_path=out_erased_nuclei,
        edit_region_path=out_edit_region,
        metadata_path=out_metadata,
    )


def _load_profile(dataset: str, profile_json: Path | None, profile_dir: Path | None) -> dict:
    if profile_json is None:
        return {}
    with profile_json.open("r", encoding="utf8") as f:
        profiles = json.load(f)
    profile = profiles.get(dataset, profiles.get(dataset.upper(), profiles.get("DEFAULT", {}))).copy()
    if profile.get("density_scale_json") and profile_dir is not None:
        profile["density_scale_json"] = str((profile_dir / profile["density_scale_json"]).resolve())
    return profile


def _generation_args(args, probe: ProbeInputs, output_path: Path, vis_dir: Path, profile: dict) -> SimpleNamespace:
    skip_ids = set(args.skip_tissue_ids or [])
    return SimpleNamespace(
        dataset=args.dataset,
        ckpt=str(args.ckpt),
        library=str(args.library),
        base_ch=args.base_ch,
        device=args.device,
        seed=args.seed,
        test_dir=None,
        input_tissue=str(probe.tissue_path),
        input_nuclei=str(probe.erased_nuclei_path),
        edit_region=str(probe.edit_region_path),
        output=str(output_path),
        output_dir=str(args.output_dir),
        n=1,
        vis_dir=str(vis_dir),
        gamma_values=args.gamma_values or profile.get("gamma_values", "1.0,2.0,3.0"),
        prob_count_weight=args.prob_count_weight
        if args.prob_count_weight is not None
        else float(profile.get("prob_count_weight", 0.7)),
        density_scale=args.density_scale
        if args.density_scale is not None
        else float(profile.get("density_scale", 1.0)),
        density_scale_json=args.density_scale_json or profile.get("density_scale_json"),
        expected_nucleus_area=args.expected_nucleus_area,
        min_count=args.min_count,
        max_density_per_10k=args.max_density_per_10k
        if args.max_density_per_10k is not None
        else float(profile.get("max_density_per_10k", 900.0)),
        max_count_factor=args.max_count_factor
        if args.max_count_factor is not None
        else float(profile.get("max_count_factor", 2.5)),
        min_region_area=args.min_region_area,
        type_prob_floor=args.type_prob_floor,
        min_distance_mode=args.min_distance_mode,
        min_distance=args.min_distance,
        min_distance_scale=args.min_distance_scale
        if args.min_distance_scale is not None
        else float(profile.get("min_distance_scale", 0.75)),
        min_distance_min=args.min_distance_min,
        min_distance_max=args.min_distance_max,
        min_distance_floor=args.min_distance_floor,
        shrink_distance_for_oversample=args.shrink_distance_for_oversample,
        oversample_base=args.oversample_base,
        oversample_gamma_scale=args.oversample_gamma_scale,
        oversample_min=args.oversample_min,
        oversample_max=args.oversample_max,
        poisson_attempts=args.poisson_attempts,
        skip_tissue_ids=skip_ids,
        no_augment_instances=args.no_augment_instances,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a single-sample Phase 4 ProbNet smoke test.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. BCSS/PANDA/GlaS/IGNITE/PUMA/ORCA")
    parser.add_argument("--input-tissue", required=True, type=Path, help="Complete tissue mask PNG")
    parser.add_argument("--input-nuclei", required=True, type=Path, help="Complete nuclei mask PNG")
    parser.add_argument("--ckpt", required=True, type=Path, help="ProbNet checkpoint, usually best.pt")
    parser.add_argument("--library", required=True, type=Path, help="Dataset-specific nuclei library directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--erasure-mode", choices=sorted(ERASURE_FUNCTIONS), default="local")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-erasure-attempts", type=int, default=100)
    parser.add_argument(
        "--min-erased-nuclei-fraction",
        type=float,
        default=0.25,
        help="Minimum fraction of nuclei pixels to erase for visual smoke tests.",
    )
    parser.add_argument(
        "--min-erased-nuclei-pixels",
        type=int,
        default=0,
        help="Minimum raw nuclei pixels to erase after connected-component expansion.",
    )
    parser.add_argument(
        "--min-edit-region-pixels",
        type=int,
        default=0,
        help="Minimum changed-mask area after connected-component expansion.",
    )
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    parser.add_argument("--profile-json", type=Path, default=Path("inpaint_cells/configs/generation_profiles.json"))
    parser.add_argument("--profile-dir", type=Path, default=Path("inpaint_cells/configs"))
    parser.add_argument("--gamma-values", default=None)
    parser.add_argument("--prob-count-weight", type=float, default=None)
    parser.add_argument("--density-scale", type=float, default=None)
    parser.add_argument("--density-scale-json", default=None)
    parser.add_argument("--expected-nucleus-area", type=float, default=80.0)
    parser.add_argument("--min-count", type=float, default=0.0)
    parser.add_argument("--max-density-per-10k", type=float, default=None)
    parser.add_argument("--max-count-factor", type=float, default=None)
    parser.add_argument("--min-region-area", type=int, default=50)
    parser.add_argument("--type-prob-floor", type=float, default=0.03)
    parser.add_argument("--min-distance-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--min-distance", type=float, default=8.0)
    parser.add_argument("--min-distance-scale", type=float, default=None)
    parser.add_argument("--min-distance-min", type=float, default=4.0)
    parser.add_argument("--min-distance-max", type=float, default=18.0)
    parser.add_argument("--min-distance-floor", type=float, default=3.0)
    parser.add_argument("--shrink-distance-for-oversample", action="store_true", default=True)
    parser.add_argument("--no-shrink-distance-for-oversample", dest="shrink_distance_for_oversample", action="store_false")
    parser.add_argument("--oversample-base", type=float, default=3.0)
    parser.add_argument("--oversample-gamma-scale", type=float, default=0.35)
    parser.add_argument("--oversample-min", type=float, default=1.5)
    parser.add_argument("--oversample-max", type=float, default=8.0)
    parser.add_argument("--poisson-attempts", type=int, default=30)
    parser.add_argument("--skip-tissue-ids", type=int, nargs="*", default=[])
    parser.add_argument("--no-augment-instances", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    probe_dir = args.output_dir / "probe_inputs"
    generated_dir = args.output_dir / "generated"
    vis_dir = args.output_dir / "vis"
    generated_dir.mkdir(parents=True, exist_ok=True)

    probe = build_single_sample_probe_inputs(
        dataset=args.dataset,
        tissue_path=args.input_tissue,
        nuclei_path=args.input_nuclei,
        output_dir=probe_dir,
        erasure_mode=args.erasure_mode,
        seed=args.seed,
        max_attempts=args.max_erasure_attempts,
        min_erased_nuclei_pixels=args.min_erased_nuclei_pixels,
        min_erased_nuclei_fraction=args.min_erased_nuclei_fraction,
        min_edit_region_pixels=args.min_edit_region_pixels,
    )

    profile = _load_profile(args.dataset, args.profile_json, args.profile_dir)
    gen_output = generated_dir / "pred_nuclei.png"
    gen_args = _generation_args(args, probe, gen_output, vis_dir, profile)

    config = get_config(args.dataset)
    config_skip = set(config.skip_tissues)
    gen_args.skip_tissue_ids = set(gen_args.skip_tissue_ids) | config_skip
    density_scales = load_density_scale(gen_args.density_scale_json)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Dataset: {config.name} ({config.cancer_type}), cancer_id={config.cancer_type_index}")
    print(f"Device: {device}")
    print(f"Probe inputs: {probe_dir}")
    print("Loading ProbNet...")
    model = load_checkpoint_model(str(args.ckpt), device, args.base_ch)
    print("Loading nuclei instance library...")
    library = NucleiLibrary(str(args.library), dataset=config.name)
    run_single(gen_args, model, library, config, density_scales, device)

    gt = _read_gray(probe.gt_nuclei_path, "GT nuclei")
    pred = _read_gray(gen_output, "predicted nuclei")
    edit_mask = _read_gray(probe.edit_region_path, "edit region") > 0
    metrics = compare_nuclei_in_edit_region(gt, pred, edit_mask)
    gt_pred_vis = write_gt_pred_comparison(
        tissue_path=probe.tissue_path,
        gt_nuclei_path=probe.gt_nuclei_path,
        erased_nuclei_path=probe.erased_nuclei_path,
        pred_nuclei_path=gen_output,
        edit_region_path=probe.edit_region_path,
        output_path=vis_dir / "gt_pred_comparison.png",
    )

    summary = {
        "probe_inputs": {k: str(v) for k, v in asdict(probe).items()},
        "generated_nuclei": str(gen_output),
        "comparison_vis": str(vis_dir / "gamma_comparison.png"),
        "gt_pred_comparison_vis": str(gt_pred_vis),
        "metrics": metrics,
    }
    summary_path = args.output_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
