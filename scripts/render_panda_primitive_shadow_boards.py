#!/usr/bin/env python3
"""Render five-case PANDA source/mask/delta boards from a frozen replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.nuclei import load_nuclei_mask
from phase3_joint_edit_refine.visualization import NUCLEI_RGB
from phase3_mask_edit_refine.evidence import load_id_mask
from phase3_mask_edit_refine.visualization import id_mask_to_rgb

SCHEMA_VERSION = "panda-primitive-mask-review-boards-v1"


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _mask_composite(tissue: np.ndarray, nuclei: np.ndarray) -> np.ndarray:
    tissue_rgb = id_mask_to_rgb(tissue).astype(float)
    view = np.clip(0.55 * tissue_rgb + 0.45 * 255, 0, 255).astype(np.uint8)
    for class_id, color in NUCLEI_RGB.items():
        view[nuclei == int(class_id)] = np.asarray(color, dtype=np.uint8)
    return view


def _delta_view(
    source_tissue: np.ndarray,
    source_nuclei: np.ndarray,
    target_tissue: np.ndarray,
    target_nuclei: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    view = _mask_composite(source_tissue, np.zeros_like(source_nuclei))
    retained = (source_nuclei == target_nuclei) & (source_nuclei > 0)
    added = (target_nuclei != source_nuclei) & (target_nuclei > 0)
    removed = (target_nuclei != source_nuclei) & (source_nuclei > 0)
    tissue_changed = source_tissue != target_tissue
    view[retained] = np.asarray([120, 120, 120], dtype=np.uint8)
    view[removed] = np.asarray([255, 0, 210], dtype=np.uint8)
    view[added] = np.asarray([0, 255, 80], dtype=np.uint8)
    view[tissue_changed] = np.asarray([0, 220, 255], dtype=np.uint8)
    return view, added | removed | tissue_changed


def _delta_zoom(view: np.ndarray, changed: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(changed)
    if not len(rows):
        return view
    padding = 28
    y0 = max(int(rows.min()) - padding, 0)
    y1 = min(int(rows.max()) + padding + 1, view.shape[0])
    x0 = max(int(cols.min()) - padding, 0)
    x1 = min(int(cols.max()) + padding + 1, view.shape[1])
    return np.asarray(
        Image.fromarray(view[y0:y1, x0:x1]).resize(
            (view.shape[1], view.shape[0]), Image.Resampling.NEAREST
        )
    )


def _resolve_case(case: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(str(case["joint_run_summary"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, list) or len(summary) != 1:
        raise ValueError(f"invalid joint summary: {summary_path}")
    artifacts = summary[0].get("artifact_paths") or {}
    context_path = Path(str(artifacts.get("case_context.json") or ""))
    candidates_path = Path(str(artifacts.get("candidates.json") or ""))
    if not context_path.is_file() or not candidates_path.is_file():
        raise FileNotFoundError(
            f"review artifacts are incomplete for {case['case_id']}"
        )
    context = json.loads(context_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    selected = next(
        (
            item
            for item in candidates
            if item.get("candidate_id") == case["selected_candidate_id"]
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"selected candidate is absent for {case['case_id']}")
    return {"case": case, "context": context, "candidate": selected}


def _render_board(
    evaluation: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    output_path: Path,
    tile_size: int,
) -> list[dict[str, Any]]:
    if len(cases) != 5:
        raise ValueError("each frozen PANDA evaluation must contain five cases")
    header = 58
    labels = (
        "SOURCE H&E",
        "SOURCE MASK",
        "TARGET MASK",
        "DELTA",
        "DELTA ZOOM",
    )
    canvas = Image.new(
        "RGB",
        (len(labels) * tile_size, len(cases) * (tile_size + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    small = _font(13)
    records = []
    for row_index, raw in enumerate(cases):
        record = _resolve_case(raw)
        context = record["context"]
        candidate = record["candidate"]
        source_image = np.asarray(
            Image.open(context["source_image_uri"]).convert("RGB"),
            dtype=np.uint8,
        )
        source_tissue = load_id_mask(context["source_tissue_mask_uri"])
        source_nuclei = load_nuclei_mask(context["source_nuclei_mask_uri"])
        target_tissue = load_id_mask(candidate["target_tissue_mask"])
        target_nuclei = load_nuclei_mask(candidate["target_nuclei_mask"])
        expected_shape = source_tissue.shape
        if any(
            value.shape != expected_shape
            for value in (source_nuclei, target_tissue, target_nuclei)
        ):
            raise ValueError(f"mask shape mismatch for {raw['case_id']}")
        source_view = _mask_composite(source_tissue, source_nuclei)
        target_view = _mask_composite(target_tissue, target_nuclei)
        delta, changed = _delta_view(
            source_tissue, source_nuclei, target_tissue, target_nuclei
        )
        panels = (
            source_image,
            source_view,
            target_view,
            delta,
            _delta_zoom(delta, changed),
        )
        y = row_index * (tile_size + header)
        draw.text(
            (8, y + 5),
            f"{row_index + 1}. {raw['source_sample_id']} | "
            f"candidate={raw['selected_candidate_id']}",
            fill="black",
            font=font,
        )
        draw.text(
            (8, y + 31),
            f"tissue_changed={np.count_nonzero(source_tissue != target_tissue)} "
            f"nuclei_changed={np.count_nonzero(source_nuclei != target_nuclei)}",
            fill=(45, 45, 45),
            font=small,
        )
        for column, (label, panel) in enumerate(zip(labels, panels, strict=True)):
            resized = Image.fromarray(panel).resize(
                (tile_size, tile_size),
                (
                    Image.Resampling.BILINEAR
                    if column == 0
                    else Image.Resampling.NEAREST
                ),
            )
            x = column * tile_size
            canvas.paste(resized, (x, y + header))
            ImageDraw.Draw(canvas).text(
                (x + 7, y + header + 6),
                label,
                fill="white",
                font=small,
                stroke_width=2,
                stroke_fill="black",
            )
        records.append(
            {
                "case_id": raw["case_id"],
                "source_sample_id": raw["source_sample_id"],
                "selected_candidate_id": raw["selected_candidate_id"],
                "source_image": context["source_image_uri"],
                "source_tissue_mask": context["source_tissue_mask_uri"],
                "source_nuclei_mask": context["source_nuclei_mask_uri"],
                "target_tissue_mask": candidate["target_tissue_mask"],
                "target_nuclei_mask": candidate["target_nuclei_mask"],
                "joint_change_mask": candidate["joint_change_mask"],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()
    replay = json.loads(args.replay_manifest.read_text(encoding="utf-8"))
    if replay.get("freeze_status") != (
        "frozen_complete_authority_candidate_execution_gates_passed"
    ):
        raise ValueError("PANDA replay is not complete and frozen")
    evaluations = list(replay.get("frozen_evaluations") or ())
    if len(evaluations) != int(replay.get("evaluation_count", -1)):
        raise ValueError("PANDA replay evaluation count is inconsistent")
    output = args.output_dir.resolve()
    boards = []
    for evaluation in evaluations:
        index = int(evaluation["evaluation_index"])
        name = (
            f"{index:02d}_{evaluation['mechanism_id']}_"
            f"{evaluation['primitive_id']}.png"
        )
        path = output / name
        records = _render_board(
            evaluation,
            list(evaluation["frozen_cases"]),
            output_path=path,
            tile_size=args.tile_size,
        )
        boards.append(
            {
                "evaluation_index": index,
                "mechanism_id": evaluation["mechanism_id"],
                "primitive_id": evaluation["primitive_id"],
                "board": str(path),
                "cases": records,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "replay_manifest": str(args.replay_manifest.resolve()),
        "evaluation_count": len(boards),
        "case_count": sum(len(item["cases"]) for item in boards),
        "all_cases_full_gate_passed": True,
        "boards": boards,
    }
    manifest_path = output / "panda_mask_review_boards.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "evaluation_count": manifest["evaluation_count"],
                "case_count": manifest["case_count"],
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
