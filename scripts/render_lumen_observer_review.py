#!/usr/bin/env python3
"""Select and render GLaS/PANDA reviews for the three-layer lumen observer."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.lumen_observer import observe_luminal_spaces
from phase3_joint_edit_refine.nuclei import load_nuclei_mask
from phase3_mask_edit_refine.evidence import load_id_mask


PROFILE = {
    "glas": {
        "dataset_names": {"glas"},
        "architecture_ids": (5, 11, 12, 13),
        "stroma_ids": (2,),
        "lumen_encoding": "within_architecture",
    },
    "panda": {
        "dataset_names": {"panda"},
        "architecture_ids": (8, 9, 10),
        "stroma_ids": (2,),
        "lumen_encoding": "stroma",
    },
}


def _targets(metadata_path: Path, dataset: str) -> list[dict[str, Any]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    accepted = PROFILE[dataset]["dataset_names"]
    by_image: dict[str, dict[str, Any]] = {}
    for row in pairs:
        if str(row.get("dataset", "")).casefold() not in accepted:
            continue
        image = str(row["target_image"])
        if image not in by_image:
            by_image[image] = {
                "sample_id": str(row["sample_id"]),
                "case_id": str(row["case_id"]),
                "source_image": image,
                "source_tissue_mask": str(row["target_tissue_mask"]),
                "source_nuclei_mask": str(row["target_nuclei_mask"]),
            }
    return list(by_image.values())


def _coarse_metric(row: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    tissue = load_id_mask(row["source_tissue_mask"])
    profile = PROFILE[dataset]
    architecture = np.isin(tissue, profile["architecture_ids"])
    stroma = np.isin(tissue, profile["stroma_ids"])
    if not np.any(architecture) or not np.any(stroma):
        return None
    contact = int(
        np.count_nonzero(
            ndimage.binary_dilation(architecture, iterations=1) & stroma
        )
    )
    if contact < 12:
        return None
    labeled, count = ndimage.label(stroma, structure=np.ones((3, 3), dtype=bool))
    adjacent_components = 0
    edge_adjacent_components = 0
    for index in range(1, int(count) + 1):
        component = labeled == index
        if int(np.count_nonzero(component)) < 16:
            continue
        if not np.any(ndimage.binary_dilation(component) & architecture):
            continue
        adjacent_components += 1
        if (
            np.any(component[0])
            or np.any(component[-1])
            or np.any(component[:, 0])
            or np.any(component[:, -1])
        ):
            edge_adjacent_components += 1
    if adjacent_components == 0:
        return None
    return {
        **row,
        "coarse_contact_px": contact,
        "coarse_adjacent_stroma_components": adjacent_components,
        "coarse_edge_adjacent_components": edge_adjacent_components,
        "coarse_score": float(
            contact
            + 36 * adjacent_components
            + 45 * min(edge_adjacent_components, 2)
        ),
    }


def _screen(
    rows: list[dict[str, Any]], dataset: str, *, screen_limit: int
) -> list[dict[str, Any]]:
    coarse = []
    for index, row in enumerate(rows, start=1):
        measured = _coarse_metric(row, dataset)
        if measured is not None:
            coarse.append(measured)
        if index % 250 == 0:
            print(json.dumps({"stage": "coarse", "completed": index, "total": len(rows)}), flush=True)
    coarse.sort(key=lambda item: (-item["coarse_score"], item["sample_id"]))
    # Avoid allowing a single slide/field to dominate the expensive observer.
    retained = []
    per_case: defaultdict[str, int] = defaultdict(int)
    for item in coarse:
        case_id = str(item["case_id"])
        if per_case[case_id] >= 8:
            continue
        retained.append(item)
        per_case[case_id] += 1
        if len(retained) >= screen_limit:
            break
    return retained


def _observe(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    image = np.asarray(Image.open(row["source_image"]).convert("RGB"), dtype=np.uint8)
    tissue = load_id_mask(row["source_tissue_mask"])
    nuclei = load_nuclei_mask(row["source_nuclei_mask"])
    profile = PROFILE[dataset]
    observation = observe_luminal_spaces(
        image,
        tissue,
        nuclei,
        stroma_fine_ids=profile["stroma_ids"],
        architecture_fine_ids=profile["architecture_ids"],
        lumen_encoding=profile["lumen_encoding"],
    )
    metadata = observation.to_metadata()
    confirmed_regions = [
        item for item in observation.regions if item.classification in {"confirmed_lumen", "open_edge_lumen"}
    ]
    uncertain_regions = [
        item for item in observation.regions if item.classification == "uncertain_low_cell_space"
    ]
    open_count = sum(item.classification == "open_edge_lumen" for item in confirmed_regions)
    score = (
        240.0 * len(confirmed_regions)
        + 90.0 * len(uncertain_regions)
        + 180.0 * open_count
        + 0.02 * metadata["confirmed_lumen_pixels"]
        + 0.005 * metadata["uncertain_low_cell_space_pixels"]
    )
    return {
        **row,
        "observation": observation,
        "observation_metadata": metadata,
        "confirmed_region_count": len(confirmed_regions),
        "uncertain_region_count": len(uncertain_regions),
        "open_edge_lumen_count": open_count,
        "observer_score": score,
    }


def _select(observed: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    observed.sort(key=lambda item: (-item["observer_score"], item["sample_id"]))
    selected: list[dict[str, Any]] = []
    used_cases: set[str] = set()

    def add_group(candidates: list[dict[str, Any]], quota: int) -> None:
        added = 0
        for item in candidates:
            if len(selected) >= count or added >= quota:
                return
            case_id = str(item["case_id"])
            if item in selected or case_id in used_cases:
                continue
            selected.append(item)
            used_cases.add(case_id)
            added += 1

    # A reliability board must expose both successes and hard negatives; it is
    # not a gallery containing only confirmed detections.
    open_positive = [item for item in observed if item["open_edge_lumen_count"] > 0]
    closed_positive = [
        item
        for item in observed
        if item["confirmed_region_count"] > item["open_edge_lumen_count"]
    ]
    uncertain_only = [
        item
        for item in observed
        if item["confirmed_region_count"] == 0 and item["uncertain_region_count"] > 0
    ]
    hard_negative = [
        item
        for item in observed
        if item["confirmed_region_count"] == 0 and item["uncertain_region_count"] == 0
    ]
    hard_negative.sort(key=lambda item: (-item["coarse_score"], item["sample_id"]))
    add_group(open_positive, max(2, count // 3))
    add_group(closed_positive, max(2, count // 3))
    add_group(uncertain_only, max(1, count // 5))
    add_group(hard_negative, max(1, count // 5))
    for item in observed:
        if len(selected) >= count:
            break
        if item in selected:
            continue
        selected.append(item)
    return selected[:count]


def _blend(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = (1.0 - alpha) * result[mask] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def _panel(item: dict[str, Any], dataset: str) -> Image.Image:
    image = np.asarray(Image.open(item["source_image"]).convert("RGB"), dtype=np.uint8)
    tissue = load_id_mask(item["source_tissue_mask"])
    nuclei = load_nuclei_mask(item["source_nuclei_mask"])
    observation = item["observation"]
    architecture = np.isin(tissue, PROFILE[dataset]["architecture_ids"])
    stroma = np.isin(tissue, PROFILE[dataset]["stroma_ids"])
    structure_view = _blend(image, stroma, (90, 185, 105), 0.28)
    structure_view = _blend(structure_view, architecture, (220, 70, 90), 0.38)
    structure_view = _blend(structure_view, nuclei > 0, (65, 75, 230), 0.72)
    seed_view = _blend(image, observation.low_cell_seed, (255, 80, 220), 0.62)
    count = observation.local_nucleus_count
    count_norm = np.clip(count / max(float(np.quantile(count[stroma], 0.90)) if np.any(stroma) else 1.0, 1e-3), 0.0, 1.0)
    density_view = image.astype(np.float32) * 0.48
    density_view[..., 0] += 210.0 * count_norm
    density_view[..., 2] += 180.0 * (1.0 - count_norm) * stroma
    density_view = np.clip(density_view, 0, 255).astype(np.uint8)
    final_view = _blend(image, observation.uncertain_low_cell_space, (255, 195, 35), 0.60)
    final_view = _blend(final_view, observation.confirmed_lumen, (20, 220, 235), 0.65)
    tiles = [image, structure_view, seed_view, density_view, final_view]
    labels = ["H&E", "mask+nuclei", "low-cell/RGB seed", "local nucleus density", "final: cyan lumen / yellow uncertain"]
    height, width = image.shape[:2]
    header = 46
    canvas = Image.new("RGB", (width * len(tiles), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (tile, label) in enumerate(zip(tiles, labels, strict=True)):
        canvas.paste(Image.fromarray(tile), (index * width, header))
        draw.text((index * width + 5, 5), label, fill="black", font=font)
    draw.text(
        (5, 23),
        f"{item['sample_id']} | confirmed={item['confirmed_region_count']} open-edge={item['open_edge_lumen_count']} uncertain={item['uncertain_region_count']} d={observation.nominal_cell_diameter_px:.1f}px",
        fill="black",
        font=font,
    )
    return canvas


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _targets(args.metadata, args.dataset)
    screened = _screen(rows, args.dataset, screen_limit=args.screen_limit)
    observed = []
    for index, row in enumerate(screened, start=1):
        observed.append(_observe(row, args.dataset))
        if index % 20 == 0 or index == len(screened):
            print(json.dumps({"stage": "observer", "completed": index, "total": len(screened)}), flush=True)
    selected = _select(observed, args.count)
    if len(selected) < args.count:
        raise RuntimeError(f"only {len(selected)} diverse samples available")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panels = []
    serializable = []
    for index, item in enumerate(selected, start=1):
        sample_dir = output / f"{index:02d}_{item['sample_id']}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        observation = item["observation"]
        Image.fromarray(observation.confirmed_lumen.astype(np.uint8) * 255).save(sample_dir / "confirmed_lumen.png")
        Image.fromarray(observation.uncertain_low_cell_space.astype(np.uint8) * 255).save(sample_dir / "uncertain_low_cell_space.png")
        panel = _panel(item, args.dataset)
        panel.save(sample_dir / "review.png")
        panels.append(panel)
        serializable.append(
            {
                key: value
                for key, value in item.items()
                if key != "observation"
            }
        )
    board_width = max(panel.width for panel in panels)
    board_height = sum(panel.height for panel in panels)
    board = Image.new("RGB", (board_width, board_height), "white")
    y = 0
    for panel in panels:
        board.paste(panel, (0, y))
        y += panel.height
    board.save(output / f"{args.dataset}_lumen_review_10.png")
    manifest = {
        "schema_version": "three-layer-lumen-review-v5",
        "dataset": args.dataset,
        "metadata": str(args.metadata.resolve()),
        "target_count": len(rows),
        "screened_count": len(screened),
        "selected": serializable,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(PROFILE), required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--screen-limit", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"status": "complete", "selected": len(manifest["selected"]), "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
