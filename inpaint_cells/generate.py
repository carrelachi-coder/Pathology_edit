#!/usr/bin/env python3
"""
Unified ProbNet-centered nuclei mask generation.

This is the Phase 4 inference entry point. ProbNet decides:
  - occupancy / density field: P(nucleus) = 1 - P(background)
  - spatial placement weights through weighted Poisson sampling
  - nucleus type through P(type | center)

The nuclei library is still used for realistic instance shapes and as a
conservative density/area fallback. It no longer decides type distribution or
places cells by rule-only statistics.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import get_config
from inpaint_cells.models.prob_unet import ProbUNet
from inpaint_cells.nuclei_library.library import NucleiLibrary, place_nucleus_layered
from inpaint_cells.utils.mask_utils import (
    NUM_NUCLEI,
    NUCLEI_CLASSES,
    load_tissue_mask,
    load_nuclei_mask,
    overlay,
    save_nuclei_mask,
)


def parse_float_list(value):
    """Parse '1,2,3' or repeated-looking strings into a list of floats."""
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def safe_name_float(value):
    return str(value).replace(".", "p").replace("-", "m")


def load_density_scale(path):
    """Load optional tissue-specific semantic density scale JSON."""
    if not path:
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def load_checkpoint_model(ckpt_path, device, base_ch):
    model = ProbUNet(out_ch=NUM_NUCLEI, base_ch=base_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_prob(model, tissue_map, input_nuclei, edit_mask, cancer_id, device):
    tissue_t = torch.from_numpy(tissue_map.astype(np.int64))[None].to(device)
    nuclei_t = torch.from_numpy(input_nuclei.astype(np.int64))[None].to(device)
    mask_t = torch.from_numpy(edit_mask.astype(np.float32))[None, None].to(device)
    cancer_t = torch.tensor([cancer_id], dtype=torch.int64, device=device)

    with torch.no_grad():
        logits = model(tissue_t, nuclei_t, mask_t, cancer_t)
        prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return prob


def weighted_mean_area(library, tissue_id, fallback):
    stats = library.stats.get(str(tissue_id), {})
    type_stats = stats.get("nuclei_types", {})
    weighted = []
    weights = []
    for info in type_stats.values():
        mean_area = float(info.get("mean_area", 0.0))
        frac = float(info.get("fraction", 0.0))
        if mean_area > 0 and frac > 0:
            weighted.append(mean_area)
            weights.append(frac)
    if weighted and sum(weights) > 0:
        return float(np.average(weighted, weights=weights))
    return float(fallback)


def adaptive_min_distance(expected_area, args, oversample_factor):
    if args.min_distance_mode == "fixed":
        base = args.min_distance
    else:
        diameter = np.sqrt(max(expected_area, 1.0) / np.pi) * 2.0
        base = diameter * args.min_distance_scale
        base = float(np.clip(base, args.min_distance_min, args.min_distance_max))

    if args.shrink_distance_for_oversample:
        base = base / np.sqrt(max(oversample_factor, 1.0))
    return max(base, args.min_distance_floor)


def poisson_candidates(region_mask, min_distance, max_attempts=30):
    """Poisson disk candidates, intentionally local to keep this entry configurable."""
    h, w = region_mask.shape
    valid_ys, valid_xs = np.where(region_mask)
    if len(valid_ys) == 0:
        return []

    cell_size = max(min_distance / np.sqrt(2.0), 1e-3)
    grid_h = int(np.ceil(h / cell_size))
    grid_w = int(np.ceil(w / cell_size))
    grid = -np.ones((grid_h, grid_w), dtype=np.int64)

    points = []
    active = []
    idx = random.randint(0, len(valid_ys) - 1)
    start = (int(valid_ys[idx]), int(valid_xs[idx]))
    points.append(start)
    active.append(0)
    grid[int(start[0] / cell_size), int(start[1] / cell_size)] = 0

    while active:
        active_idx = random.randint(0, len(active) - 1)
        point_idx = active[active_idx]
        py, px = points[point_idx]
        found = False

        for _ in range(max_attempts):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(min_distance, 2 * min_distance)
            ny = int(py + dist * np.sin(angle))
            nx = int(px + dist * np.cos(angle))

            if ny < 0 or ny >= h or nx < 0 or nx >= w or not region_mask[ny, nx]:
                continue

            ngy, ngx = int(ny / cell_size), int(nx / cell_size)
            too_close = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    gy, gx = ngy + dy, ngx + dx
                    if 0 <= gy < grid_h and 0 <= gx < grid_w and grid[gy, gx] >= 0:
                        ey, ex = points[grid[gy, gx]]
                        if (ny - ey) ** 2 + (nx - ex) ** 2 < min_distance ** 2:
                            too_close = True
                            break
                if too_close:
                    break

            if not too_close:
                new_idx = len(points)
                points.append((ny, nx))
                active.append(new_idx)
                grid[ngy, ngx] = new_idx
                found = True
                break

        if not found:
            active.pop(active_idx)

    return points


def compute_target_count(nuc_prob, tissue_region, tissue_id, library, expected_area, args, scale):
    region_area = int(tissue_region.sum())
    prob_count = float(nuc_prob[tissue_region].sum() / max(expected_area, 1.0))

    library_density = float(library.get_density(tissue_id))
    library_count = library_density * region_area / 10000.0

    if library_count <= 0:
        blended = prob_count
    else:
        blended = args.prob_count_weight * prob_count + (1.0 - args.prob_count_weight) * library_count

    scaled = blended * scale
    max_by_density = args.max_density_per_10k * region_area / 10000.0
    max_allowed = max_by_density
    if library_count > 0 and args.max_count_factor > 0:
        max_allowed = min(max_allowed, library_count * args.max_count_factor)

    clipped = float(np.clip(scaled, args.min_count, max_allowed))
    return int(round(clipped)), {
        "region_area": region_area,
        "prob_count": prob_count,
        "library_density_per_10k": library_density,
        "library_count": library_count,
        "semantic_scale": scale,
        "blended_count": blended,
        "clipped_count": clipped,
    }


def choose_weighted_centers(candidates, nuc_prob, target_count, gamma):
    if target_count <= 0 or not candidates:
        return []
    n = min(target_count, len(candidates))
    ys = np.array([p[0] for p in candidates], dtype=np.int64)
    xs = np.array([p[1] for p in candidates], dtype=np.int64)
    scores = np.power(np.clip(nuc_prob[ys, xs], 0.0, 1.0), gamma)
    scores = scores + 1e-8
    probs = scores / scores.sum()
    chosen = np.random.choice(len(candidates), size=n, replace=False, p=probs)
    return [candidates[int(i)] for i in chosen]


def sample_type_at_center(prob, cy, cx, args):
    type_probs = prob[1:, cy, cx].astype(np.float64)
    total = type_probs.sum()
    if total < args.type_prob_floor:
        return None
    type_probs = type_probs / total
    idx = int(np.random.choice(len(type_probs), p=type_probs))
    return NUCLEI_CLASSES[idx]


def generate_for_gamma(prob, tissue, input_nuclei, edit_mask, library, gamma, args, density_scales):
    nuc_prob = 1.0 - prob[0]
    output = input_nuclei.copy()
    output[edit_mask] = 0

    diagnostics = {
        "gamma": gamma,
        "placed": 0,
        "tissues": {},
    }

    for tissue_id in np.unique(tissue[edit_mask]):
        tissue_id = int(tissue_id)
        if tissue_id in args.skip_tissue_ids:
            continue

        tissue_region = edit_mask & (tissue == tissue_id)
        if tissue_region.sum() < args.min_region_area:
            continue

        expected_area = weighted_mean_area(library, tissue_id, args.expected_nucleus_area)
        scale = density_scales.get(tissue_id, args.density_scale)
        target_count, count_info = compute_target_count(
            nuc_prob, tissue_region, tissue_id, library, expected_area, args, scale
        )

        oversample_factor = args.oversample_base * (1.0 + args.oversample_gamma_scale * max(gamma - 1.0, 0.0))
        oversample_factor = float(np.clip(oversample_factor, args.oversample_min, args.oversample_max))
        min_distance = adaptive_min_distance(expected_area, args, oversample_factor)
        candidates = poisson_candidates(tissue_region, min_distance, args.poisson_attempts)
        centers = choose_weighted_centers(candidates, nuc_prob, target_count, gamma)

        placed = 0
        for cy, cx in centers:
            nuc_type = sample_type_at_center(prob, cy, cx, args)
            if nuc_type is None:
                continue
            instance = library.sample_instance(tissue_id, nuc_type)
            if instance is None:
                instance = library.sample_instance(tissue_id)
            if instance is None:
                continue
            if place_nucleus_layered(output, cy, cx, instance, augment=not args.no_augment_instances):
                placed += 1

        diagnostics["placed"] += placed
        diagnostics["tissues"][str(tissue_id)] = {
            **count_info,
            "expected_nucleus_area": expected_area,
            "oversample_factor": oversample_factor,
            "min_distance": min_distance,
            "num_candidates": len(candidates),
            "target_count": target_count,
            "selected_centers": len(centers),
            "placed": placed,
        }

    return output, diagnostics


def heatmap_rgb(values, mask=None):
    values = np.clip(values, 0.0, 1.0)
    img = (values * 255).astype(np.uint8)
    colored = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if mask is not None:
        dim = np.zeros_like(colored)
        dim[:] = [30, 30, 30]
        colored = np.where(mask[..., None], colored, dim)
    return colored


def draw_edit_contour(rgb, edit_mask):
    out = rgb.copy()
    contours, _ = cv2.findContours((edit_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def make_comparison(tissue, input_nuclei, outputs_by_gamma, nuc_prob, edit_mask):
    panels = [
        draw_edit_contour(overlay(tissue, input_nuclei), edit_mask),
        draw_edit_contour(heatmap_rgb(nuc_prob, edit_mask), edit_mask),
    ]
    for gamma, nuclei in outputs_by_gamma:
        panels.append(draw_edit_contour(overlay(tissue, nuclei), edit_mask))

    h, w = tissue.shape
    row = np.concatenate(panels, axis=1)
    labeled = np.zeros((h + 34, row.shape[1], 3), dtype=np.uint8)
    labeled[:34] = 35
    labeled[34:] = row

    labels = ["input", "P(nucleus)"] + [f"gamma={gamma:g}" for gamma, _ in outputs_by_gamma]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        cv2.putText(labeled, label, (i * w + 6, 23), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def run_single(args, model, library, config, density_scales, device):
    tissue = load_tissue_mask(args.input_tissue)
    edit_mask = cv2.imread(args.edit_region, cv2.IMREAD_GRAYSCALE)
    if edit_mask is None:
        raise FileNotFoundError(f"Cannot load edit region mask: {args.edit_region}")
    edit_mask = edit_mask > 128

    if args.input_nuclei:
        input_nuclei = load_nuclei_mask(args.input_nuclei, remap=True)
    else:
        input_nuclei = np.zeros_like(tissue, dtype=np.int64)
    input_nuclei = input_nuclei.copy()
    input_nuclei[edit_mask] = 0

    prob = predict_prob(model, tissue, input_nuclei, edit_mask, config.cancer_type_index, device)
    outputs = []
    diagnostics = []

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gamma_values = parse_float_list(args.gamma_values)
    for idx, gamma in enumerate(gamma_values):
        nuclei, diag = generate_for_gamma(prob, tissue, input_nuclei, edit_mask, library, gamma, args, density_scales)
        diagnostics.append(diag)

        if idx == 0:
            save_path = output_path
        else:
            save_path = output_path.with_name(f"{output_path.stem}_gamma_{safe_name_float(gamma)}{output_path.suffix}")
        save_nuclei_mask(nuclei, str(save_path))
        outputs.append((gamma, nuclei))
        print(f"gamma={gamma:g}: placed {diag['placed']} nuclei -> {save_path}")

    if args.vis_dir:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        comparison = make_comparison(tissue, input_nuclei, outputs, 1.0 - prob[0], edit_mask)
        cv2.imwrite(str(vis_dir / "gamma_comparison.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        with open(vis_dir / "diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)


def discover_batch_samples(data_dir):
    root = Path(data_dir)
    val_dir = root / "val" if (root / "val").is_dir() else root
    gt_tissue = val_dir / "gt_tissue"
    gt_nuclei = val_dir / "gt_nuclei"
    masks = val_dir / "masks"
    if gt_tissue.is_dir() and masks.is_dir():
        samples = []
        for tissue_path in sorted(gt_tissue.glob("*.png")):
            name = tissue_path.name
            nuclei_path = gt_nuclei / name
            mask_path = masks / name
            if nuclei_path.exists() and mask_path.exists():
                samples.append((tissue_path.stem, tissue_path, nuclei_path, mask_path))
        return samples

    samples = []
    for tissue_path in sorted(val_dir.glob("*/tissue_mask.png")):
        sample_dir = tissue_path.parent
        nuclei_path = sample_dir / "nuclei_mask.png"
        mask_path = sample_dir / "edit_mask.png"
        if nuclei_path.exists() and mask_path.exists():
            samples.append((sample_dir.name, tissue_path, nuclei_path, mask_path))
    return samples


def run_batch(args, model, library, config, density_scales, device):
    samples = discover_batch_samples(args.test_dir)
    if args.n > 0:
        samples = samples[:args.n]
    if not samples:
        raise RuntimeError(f"No layered validation samples found in {args.test_dir}")

    output_dir = Path(args.output_dir)
    nuclei_dir = output_dir / "nuclei"
    vis_dir = output_dir / "vis"
    nuclei_dir.mkdir(parents=True, exist_ok=True)
    if args.vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_diag = {}
    gamma_values = parse_float_list(args.gamma_values)
    for idx, (name, tissue_path, nuclei_path, mask_path) in enumerate(samples):
        tissue = load_tissue_mask(str(tissue_path))
        gt_nuclei = load_nuclei_mask(str(nuclei_path), remap=True)
        edit_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 128
        input_nuclei = gt_nuclei.copy()
        input_nuclei[edit_mask] = 0

        prob = predict_prob(model, tissue, input_nuclei, edit_mask, config.cancer_type_index, device)
        outputs = []
        sample_diag = []
        for gamma in gamma_values:
            nuclei, diag = generate_for_gamma(prob, tissue, input_nuclei, edit_mask, library, gamma, args, density_scales)
            suffix = "" if len(gamma_values) == 1 else f"_gamma_{safe_name_float(gamma)}"
            out_path = nuclei_dir / f"{name}{suffix}_nuclei.png"
            save_nuclei_mask(nuclei, str(out_path))
            outputs.append((gamma, nuclei))
            sample_diag.append(diag)

        if args.vis_dir:
            comparison = make_comparison(tissue, input_nuclei, outputs, 1.0 - prob[0], edit_mask)
            cv2.imwrite(str(vis_dir / f"{idx:03d}_{name}.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        all_diag[name] = sample_diag
        print(f"[{idx + 1}/{len(samples)}] {name}: " + ", ".join(f"gamma={d['gamma']:g} placed={d['placed']}" for d in sample_diag))

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(all_diag, f, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="ProbNet-centered Phase 4 nuclei generation")
    parser.add_argument("--dataset", required=True, help="Dataset name: BCSS, PANDA, GlaS, IGNITE, PUMA, ORCA")
    parser.add_argument("--ckpt", required=True, help="ProbNet checkpoint")
    parser.add_argument("--library", required=True, help="Nuclei instance library directory")
    parser.add_argument("--base-ch", type=int, default=64, help="ProbUNet base channels used during training")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-dir", help="Layered validation dataset directory for batch inference")
    mode.add_argument("--input-tissue", help="Single edited tissue mask PNG")

    parser.add_argument("--input-nuclei", default=None, help="Optional existing nuclei mask for single inference")
    parser.add_argument("--edit-region", default=None, help="Single edit region mask PNG")
    parser.add_argument("--output", default="nuclei_mask.png", help="Single output nuclei mask path")
    parser.add_argument("--output-dir", default="phase4_probnet_generate", help="Batch output directory")
    parser.add_argument("--n", type=int, default=10, help="Batch sample limit; <=0 means all")
    parser.add_argument("--vis-dir", default=None, help="Write gamma comparison PNGs and diagnostics")

    parser.add_argument("--gamma-values", default="1.0,2.0,3.0",
                        help="Comma-separated gamma values for weighted center sampling")
    parser.add_argument("--prob-count-weight", type=float, default=0.7,
                        help="Blend weight for ProbNet count vs library density count")
    parser.add_argument("--density-scale", type=float, default=1.0,
                        help="Global semantic density multiplier")
    parser.add_argument("--density-scale-json", default=None,
                        help="Optional JSON mapping tissue_id -> semantic density multiplier")
    parser.add_argument("--expected-nucleus-area", type=float, default=80.0,
                        help="Fallback expected nucleus area in pixels")
    parser.add_argument("--min-count", type=float, default=0.0)
    parser.add_argument("--max-density-per-10k", type=float, default=900.0,
                        help="Absolute count clip: max nuclei per 10k px")
    parser.add_argument("--max-count-factor", type=float, default=2.5,
                        help="If library density exists, cap count at this multiple of library count")
    parser.add_argument("--min-region-area", type=int, default=50)
    parser.add_argument("--type-prob-floor", type=float, default=0.03)

    parser.add_argument("--min-distance-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--min-distance", type=float, default=8.0,
                        help="Fixed Poisson distance when --min-distance-mode=fixed")
    parser.add_argument("--min-distance-scale", type=float, default=0.75,
                        help="Adaptive distance = nucleus_diameter * scale before oversample shrinking")
    parser.add_argument("--min-distance-min", type=float, default=4.0)
    parser.add_argument("--min-distance-max", type=float, default=18.0)
    parser.add_argument("--min-distance-floor", type=float, default=3.0)
    parser.add_argument("--shrink-distance-for-oversample", action="store_true", default=True)
    parser.add_argument("--no-shrink-distance-for-oversample", dest="shrink_distance_for_oversample",
                        action="store_false")
    parser.add_argument("--oversample-base", type=float, default=3.0)
    parser.add_argument("--oversample-gamma-scale", type=float, default=0.35)
    parser.add_argument("--oversample-min", type=float, default=1.5)
    parser.add_argument("--oversample-max", type=float, default=8.0)
    parser.add_argument("--poisson-attempts", type=int, default=30)
    parser.add_argument("--skip-tissue-ids", type=int, nargs="*", default=[],
                        help="Additional tissue IDs to skip")
    parser.add_argument("--no-augment-instances", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = get_config(args.dataset)
    args.skip_tissue_ids = set(args.skip_tissue_ids) | set(config.skip_tissues)
    density_scales = load_density_scale(args.density_scale_json)

    if args.input_tissue and not args.edit_region:
        raise ValueError("--edit-region is required with --input-tissue")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Dataset: {config.name} ({config.cancer_type}), cancer_id={config.cancer_type_index}")
    print(f"Device: {device}")
    print(f"Gamma values: {parse_float_list(args.gamma_values)}")
    print("Loading ProbNet...")
    model = load_checkpoint_model(args.ckpt, device, args.base_ch)
    print("Loading nuclei instance library...")
    library = NucleiLibrary(args.library, dataset=config.name)

    if args.test_dir:
        run_batch(args, model, library, config, density_scales, device)
    else:
        run_single(args, model, library, config, density_scales, device)


if __name__ == "__main__":
    main()
