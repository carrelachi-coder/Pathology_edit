"""Deterministic Planner panels and Critic contact sheets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from dataset_config import FULL_COLOR_MAP
from phase3_mask_edit_refine.models import CandidateMask, GateReport
from phase3_mask_edit_refine.scene import SceneAnalysis


def save_planner_panels(
    *,
    image_path: str | Path,
    mask: np.ndarray,
    scene: SceneAnalysis,
    output_dir: str | Path,
) -> tuple[str, ...]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image = _load_rgb(image_path, expected_shape=mask.shape)
    semantic = _blend(image, id_mask_to_rgb(mask), alpha=0.42)
    component = np.array(image, copy=True)
    draw_image = Image.fromarray(component)
    draw = ImageDraw.Draw(draw_image)
    for item in scene.graph.anchor_segments:
        anchor = scene.anchor_masks[item.anchor_segment_id]
        rows, cols = np.where(anchor)
        color = _indexed_color(item.display_index - 1)
        for row, col in zip(rows.tolist(), cols.tolist()):
            if 0 <= col < draw_image.width and 0 <= row < draw_image.height:
                draw.point((col, row), fill=color)
        x0, y0, _, _ = item.bbox_xyxy
        draw.text((x0, y0), f"A{item.display_index}", fill=color)
    original_path = output / "planner_01_he.png"
    semantic_path = output / "planner_02_semantic_overlay.png"
    interface_path = output / "planner_03_interface_overlay.png"
    Image.fromarray(image).save(original_path)
    Image.fromarray(semantic).save(semantic_path)
    draw_image.save(interface_path)
    return tuple(str(path) for path in (original_path, semantic_path, interface_path))


def save_mask_planner_panels(
    *,
    mask: np.ndarray,
    scene: SceneAnalysis,
    output_dir: str | Path,
) -> tuple[str, ...]:
    """Write execution-planning panels without exposing the source H&E.

    These panels are the only raster inputs permitted for interface/anchor
    planning.  The tissue raster and deterministic scene graph are annotation
    authorities; the Planner must not infer an unannotated structure from raw
    histology.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    semantic, component, interface = build_mask_planner_panels(
        mask=mask,
        scene=scene,
    )
    semantic_path = output / "planner_01_tissue_mask.png"
    component_path = output / "planner_02_component_map.png"
    interface_path = output / "planner_03_interface_anchor_map.png"
    Image.fromarray(semantic).save(semantic_path)
    Image.fromarray(component).save(component_path)
    Image.fromarray(interface).save(interface_path)
    return tuple(
        str(path) for path in (semantic_path, component_path, interface_path)
    )


def build_mask_planner_panels(
    *,
    mask: np.ndarray,
    scene: SceneAnalysis,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render canonical mask-only Planner panels without writing files."""

    semantic = id_mask_to_rgb(mask)
    component = np.full_like(semantic, 24, dtype=np.uint8)
    for index, item in enumerate(scene.graph.components):
        region = scene.component_masks[item.component_id]
        color = np.asarray(_indexed_color(index), dtype=np.uint8)
        component[region] = np.clip(0.65 * semantic[region] + 0.35 * color, 0, 255)
    interface = np.array(semantic, copy=True)
    draw_image = Image.fromarray(interface)
    draw = ImageDraw.Draw(draw_image)
    for item in scene.graph.anchor_segments:
        anchor = scene.anchor_masks[item.anchor_segment_id]
        rows, cols = np.where(anchor)
        color = _indexed_color(item.display_index - 1)
        for row, col in zip(rows.tolist(), cols.tolist()):
            if 0 <= col < draw_image.width and 0 <= row < draw_image.height:
                draw.point((col, row), fill=color)
        x0, y0, _, _ = item.bbox_xyxy
        draw.text((x0, y0), f"A{item.display_index}", fill=color)
    return semantic, component, np.asarray(draw_image)


def save_critic_contact_sheet(
    *,
    image_path: str | Path,
    source_mask: np.ndarray,
    candidates: Sequence[CandidateMask],
    gate_reports: Sequence[GateReport],
    scene: SceneAnalysis,
    output_path: str | Path,
    columns: int = 4,
) -> str:
    report_by_id = {report.candidate_id: report for report in gate_reports}
    passed = [
        candidate
        for candidate in candidates
        if report_by_id.get(candidate.candidate_id) is not None
        and report_by_id[candidate.candidate_id].passed
    ]
    if not passed:
        raise ValueError("critic contact sheet requires at least one gate-passing candidate")
    if np.asarray(source_mask).shape != passed[0].change_region.shape:
        raise ValueError("source mask and candidate dimensions differ")
    base = _load_rgb(image_path, expected_shape=passed[0].change_region.shape)
    tile_width, tile_height = base.shape[1], base.shape[0]
    header_height = 40
    rows = int(np.ceil(len(passed) / columns))
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + header_height)),
        (255, 255, 255),
    )
    for index, candidate in enumerate(passed):
        report = report_by_id[candidate.candidate_id]
        change = np.asarray(candidate.change_region, dtype=bool)
        target_view = _blend(base, id_mask_to_rgb(candidate.target_mask), alpha=0.42)
        changed_boundary = change & ~ndimage.binary_erosion(
            change, structure=np.ones((3, 3), dtype=bool)
        )
        target_view[changed_boundary] = np.array([255, 255, 0], dtype=np.uint8)
        diff_view = np.array(base, copy=True).astype(float)
        diff_view[change] = 0.30 * diff_view[change] + 0.70 * np.array([255, 0, 255])
        interface_ids = candidate.tool_trace.get("interface_ids", [candidate.interface_id])
        for interface_id in interface_ids:
            interface = scene.interface_masks.get(interface_id)
            if interface is not None:
                diff_view[interface] = np.array([0, 255, 255])
        half_width = tile_width // 2
        target_image = Image.fromarray(target_view).resize(
            (half_width, tile_height), Image.Resampling.BILINEAR
        )
        diff_image = Image.fromarray(np.clip(diff_view, 0, 255).astype(np.uint8)).resize(
            (tile_width - half_width, tile_height), Image.Resampling.BILINEAR
        )
        tile = Image.new("RGB", (tile_width, tile_height), (255, 255, 255))
        tile.paste(target_image, (0, 0))
        tile.paste(diff_image, (half_width, 0))
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + header_height)
        draw = ImageDraw.Draw(canvas)
        metrics = {check.check_id: check.metrics for check in report.checks}
        depth = metrics.get("depth_span_ratio", {}).get("p95_depth_px")
        retention = metrics.get("source_component_retention", {}).get("components", {})
        max_consumed = max(
            (float(item.get("changed_fraction", 0.0)) for item in retention.values()),
            default=0.0,
        )
        draw.text(
            (x + 4, y + 4),
            f"{candidate.candidate_id} | {candidate.tool_name}",
            fill=(0, 0, 0),
        )
        draw.text(
            (x + 4, y + 20),
            f"area={int(change.sum())}  p95depth={float(depth or 0):.1f}px  max-consumed={max_consumed:.1%}",
            fill=(0, 0, 0),
        )
        canvas.paste(tile, (x, y + header_height))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return str(target)


def id_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for fine_id in np.unique(arr):
        color = FULL_COLOR_MAP.get(int(fine_id), [128, 128, 128])
        rgb[arr == fine_id] = np.asarray(color, dtype=np.uint8)
    return rgb


def _load_rgb(path: str | Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if image.size != (expected_shape[1], expected_shape[0]):
        raise ValueError(
            f"image/mask dimensions differ: image={image.size}, "
            f"mask={(expected_shape[1], expected_shape[0])}"
        )
    return np.asarray(image)


def _blend(left: np.ndarray, right: np.ndarray, *, alpha: float) -> np.ndarray:
    return np.clip((1.0 - alpha) * left.astype(float) + alpha * right.astype(float), 0, 255).astype(
        np.uint8
    )


def _indexed_color(index: int) -> tuple[int, int, int]:
    palette = (
        (255, 255, 0),
        (0, 255, 255),
        (255, 128, 0),
        (128, 255, 0),
        (255, 0, 255),
        (0, 128, 255),
    )
    return palette[index % len(palette)]
