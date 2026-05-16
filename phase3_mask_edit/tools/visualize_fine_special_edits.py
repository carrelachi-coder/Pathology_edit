"""Visualize dataset-specialized fine-ID transition primitives on real masks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset_config.unified_labels import FINE_LABELS, UNIFIED_COLOR_MAP
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import id_to_rgb, load_id_mask, save_metadata


DEFAULT_CASES: tuple[dict[str, str], ...] = (
    {
        "profile": "PANDA",
        "primitive": "gleason_upgrade_3to4",
        "mask": r"D:\WQX\datasets\PANDA\PANDA_PATCHES\tissue_masks\00e6511435645e50673991768a713c66_y15872_x8448_py2560_px1536.png",
    },
    {
        "profile": "PANDA",
        "primitive": "gleason_upgrade_4to5",
        "mask": r"D:\WQX\datasets\PANDA\PANDA_PATCHES\tissue_masks\00c52cb4db1c7a5811a8f070a910c038_y5888_x6912_py2304_px1280.png",
    },
    {
        "profile": "PANDA",
        "primitive": "gleason_downgrade_4to3",
        "mask": r"D:\WQX\datasets\PANDA\PANDA_PATCHES\tissue_masks\0550b23f29085f41b10d165a46ad4371_y512_x5888_py2048_px0.png",
    },
    {
        "profile": "PANDA",
        "primitive": "benign_to_gleason3",
        "mask": r"D:\WQX\datasets\PANDA\PANDA_PATCHES\tissue_masks\01227af585a47c16680f927d21a7f06a_y9472_x768_py1536_px512.png",
    },
    {
        "profile": "PANDA",
        "primitive": "benign_atrophy",
        "mask": r"D:\WQX\datasets\PANDA\PANDA_PATCHES\tissue_masks\01227af585a47c16680f927d21a7f06a_y9472_x768_py1536_px512.png",
    },
    {
        "profile": "GLAS",
        "primitive": "normal_to_adenomatous",
        "mask": r"D:\WQX\datasets\GLAS\GlaS_PATCHES\tissue_masks\testA_10_py512_px512.png",
    },
    {
        "profile": "GLAS",
        "primitive": "adenoma_to_carcinoma",
        "mask": r"D:\WQX\datasets\GLAS\GlaS_PATCHES\tissue_masks\testA_19_py0_px256.png",
    },
    {
        "profile": "GLAS",
        "primitive": "grade_upgrade",
        "mask": r"D:\WQX\datasets\GLAS\GlaS_PATCHES\tissue_masks\train_78_py256_px1024.png",
    },
    {
        "profile": "GLAS",
        "primitive": "treatment_dedifferentiation",
        "mask": r"D:\WQX\datasets\GLAS\GlaS_PATCHES\tissue_masks\train_13_py0_px256.png",
    },
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create visual QA panels for fine-ID special primitives."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/phase3_fine_special_edits/real_mask_visualization"),
    )
    parser.add_argument("--strength", default="moderate")
    parser.add_argument("--fraction", type=float, default=0.35)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    panel_paths = []
    for case in DEFAULT_CASES:
        summary, panel = _run_case(
            profile=case["profile"],
            primitive=case["primitive"],
            mask_path=Path(case["mask"]),
            output_root=args.output,
            strength=args.strength,
            target_change_fraction=args.fraction,
        )
        summaries.append(summary)
        panel_paths.append(panel)

    contact_sheet = _make_contact_sheet(panel_paths, args.output / "contact_sheet.png")
    save_metadata(
        {
            "cases": summaries,
            "contact_sheet": str(contact_sheet),
            "target_change_fraction": args.fraction,
            "strength": args.strength,
        },
        args.output / "summary.json",
    )
    print(json.dumps({"output": str(args.output), "contact_sheet": str(contact_sheet)}, indent=2))
    return 0


def _run_case(
    *,
    profile: str,
    primitive: str,
    mask_path: Path,
    output_root: Path,
    strength: str,
    target_change_fraction: float,
) -> tuple[dict[str, Any], Path]:
    recipe = load_recipe(default_recipe_path_for_profile(profile))
    schema = MaskProfileSchema.from_reference_profile(profile)
    mask = load_id_mask(mask_path)
    del recipe, target_change_fraction
    case_dir = output_root / f"{profile.lower()}_{primitive}"
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "profile": profile,
        "primitive": primitive,
        "mask": str(mask_path),
        "status": "retired",
        "failure_reason": (
            "legacy_non_llm_deterministic_executor_retired; use LLM contour "
            "organic_v2 for active mask editing"
        ),
    }
    save_metadata(summary, case_dir / "summary.json")
    panel = _make_failure_panel(summary, case_dir / "panel.png")
    return summary, panel


def _change_overlay(rgb: np.ndarray, change: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    highlight = np.array([0, 220, 255], dtype=np.float32)
    out[change] = (0.35 * out[change].astype(np.float32) + 0.65 * highlight).astype(np.uint8)
    return out


def _transition_only_rgb(src: np.ndarray, target: np.ndarray, change: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*src.shape, 3), dtype=np.uint8)
    rgb[:] = np.array([32, 32, 32], dtype=np.uint8)
    rgb[change] = id_to_rgb(target)[change]
    border = change & (src != target)
    rgb[border] = np.maximum(rgb[border], np.array([80, 80, 80], dtype=np.uint8))
    return rgb


def _make_panel(
    *,
    profile: str,
    primitive: str,
    mask_name: str,
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    diff_rgb: np.ndarray,
    ops_log: dict[str, Any],
    output_path: Path,
) -> Path:
    tiles = [
        ("Before", before_rgb),
        ("After", after_rgb),
        ("Changed", overlay_rgb),
        ("Transition", diff_rgb),
    ]
    tile_w = 256
    tile_h = 256
    top_h = 78
    label_h = 24
    stats_h = 92
    gutter = 10
    width = len(tiles) * tile_w + (len(tiles) + 1) * gutter
    height = top_h + label_h + tile_h + stats_h + gutter
    panel = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    title = f"{profile}  {primitive}"
    subtitle = _shorten(mask_name, 92)
    draw.text((gutter, 10), title, fill=(20, 20, 20), font=font)
    draw.text((gutter, 30), subtitle, fill=(70, 70, 70), font=font)
    draw.text((gutter, 50), _ops_line(ops_log), fill=(70, 70, 70), font=font)

    x = gutter
    y = top_h
    for label, arr in tiles:
        draw.text((x, y), label, fill=(25, 25, 25), font=font)
        image = Image.fromarray(arr).resize((tile_w, tile_h), Image.Resampling.NEAREST)
        panel.paste(image, (x, y + label_h))
        x += tile_w + gutter

    stats_y = top_h + label_h + tile_h + 8
    for line in _legend_lines(ops_log):
        draw.text((gutter, stats_y), line, fill=(45, 45, 45), font=font)
        stats_y += 16

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return output_path


def _make_failure_panel(summary: dict[str, Any], output_path: Path) -> Path:
    panel = Image.new("RGB", (640, 160), (250, 245, 240))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((16, 16), f"{summary['profile']} {summary['primitive']}", fill=(20, 20, 20), font=font)
    draw.text((16, 40), f"status: {summary['status']}", fill=(120, 40, 30), font=font)
    reasons = summary.get("applicability", {}).get("reasons", [])
    draw.text((16, 64), f"reasons: {reasons}", fill=(80, 80, 80), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return output_path


def _make_contact_sheet(panel_paths: list[Path], output_path: Path) -> Path:
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    if not panels:
        raise ValueError("no panels to combine")
    columns = 2
    gutter = 16
    w = max(panel.width for panel in panels)
    h = max(panel.height for panel in panels)
    rows = int(np.ceil(len(panels) / columns))
    sheet = Image.new(
        "RGB",
        (columns * w + (columns + 1) * gutter, rows * h + (rows + 1) * gutter),
        (235, 235, 232),
    )
    for index, panel in enumerate(panels):
        row = index // columns
        col = index % columns
        x = gutter + col * (w + gutter)
        y = gutter + row * (h + gutter)
        sheet.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _ops_line(ops_log: dict[str, Any]) -> str:
    source_ids = ops_log.get("source_fine_ids", [])
    target_id = ops_log.get("target_fine_id")
    return (
        f"{_id_names(source_ids)} -> {_id_names([target_id])}; "
        f"selected {ops_log.get('selected_pixels')} / {ops_log.get('candidate_pixels')} px"
    )


def _legend_lines(ops_log: dict[str, Any]) -> list[str]:
    source_ids = ops_log.get("source_fine_ids", [])
    target_id = ops_log.get("target_fine_id")
    return [
        f"Execution: {ops_log.get('execution_strategy')} / {ops_log.get('operation_type')}",
        f"Source IDs: {_id_names(source_ids)}",
        f"Target ID: {_id_names([target_id])}",
        f"Semantics: {ops_log.get('target_change_fraction_semantics')}",
        f"Requested source-relative fraction: {ops_log.get('requested_source_relative_fraction')}",
    ]


def _id_names(ids: Any) -> str:
    clean = [int(value) for value in ids if value is not None]
    return ", ".join(f"{value}:{FINE_LABELS.get(value, 'unknown')}" for value in clean)


def _counts(mask: np.ndarray, ids: Any) -> dict[str, int]:
    return {
        str(int(value)): int(np.count_nonzero(mask == int(value)))
        for value in ids
        if value is not None
    }


def _shorten(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
