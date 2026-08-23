#!/usr/bin/env python3
"""Build G2-600-style review panels for Online Generator results."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_config.unified_labels import (
    CELL_CLASSES,
    CELL_COLOR_MAP,
    FINE_LABELS,
    UNIFIED_COLOR_MAP,
)
from scripts.build_g2_600_review_panels import (
    _draw_legend,
    _fit,
    _font,
    _load_mask,
    _load_rgb,
    tissue_nuclei_overlay,
)

TILE_SIZE = 512
HEADER_HEIGHT = 140
LABEL_HEIGHT = 36
PANEL_TITLES = (
    "Source H&E",
    "Source tissue + nuclei",
    "Target tissue + nuclei",
    "Selected generated image",
    "Generated + semantic/generation boundaries",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = _read_json(args.manifest)
    records = resolve_records(manifest)
    if args.expected_count is not None and len(records) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} records, found {len(records)}."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "panels").mkdir(exist_ok=True)
    preflight = preflight_records(records)
    _write_json(args.output / "panel_preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError(
            "Panel preflight failed: "
            + "; ".join(preflight["failure_reasons"][:10])
        )

    panel_records = []
    for index, record in enumerate(records, start=1):
        panel_path = (
            args.output / "panels" / f"{index:03d}_{record['case_id']}.jpg"
        )
        render_panel(record, panel_path, jpeg_quality=args.jpeg_quality)
        panel_records.append(
            {
                **record,
                "panel_index": index,
                "panel_path": str(panel_path.resolve()),
                "panel_sha256": _sha256(panel_path),
            }
        )
    panel_manifest = {
        "schema_version": "online-generation-review-panels-v1",
        "case_count": len(panel_records),
        "panel_layout": list(PANEL_TITLES),
        "boundary_legend": {
            "semantic_change": "yellow",
            "generation_support": "cyan",
        },
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "records": panel_records,
    }
    _write_json(args.output / "panel_manifest.json", panel_manifest)
    write_html_index(panel_records, args.output / "index.html")
    print(
        json.dumps(
            {
                "status": "completed",
                "case_count": len(panel_records),
                "output": str(args.output.resolve()),
                "index": str((args.output / "index.html").resolve()),
                "manifest": str(
                    (args.output / "panel_manifest.json").resolve()
                ),
            },
            indent=2,
        )
    )
    return 0


def resolve_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for raw in manifest.get("records", ()):
        generation_dir = Path(str(raw["agentic_generation_dir"]))
        workflow_path = generation_dir / "agentic_workflow.json"
        pipeline_summary_path = generation_dir / "pipeline_summary.json"
        report_path = generation_dir / "generation_report.json"
        workflow = _read_json(workflow_path)
        pipeline_summary = _read_json(pipeline_summary_path)
        selected = workflow.get("selected_attempt") or {}
        verification = selected.get("verification") or {}
        change_regions = pipeline_summary.get("change_regions", {})
        generation_policy = change_regions.get(
            "generation_context_policy", {}
        )
        selected_image = generation_dir / "generated_image.png"
        semantic_region = generation_dir / "semantic_change_region.png"
        generation_region = generation_dir / "generation_change_region.png"
        records.append(
            {
                "case_id": str(raw["case_id"]),
                "category": str(raw.get("category") or ""),
                "primitive_id": str(raw.get("primitive_id") or ""),
                "instruction": str(raw.get("instruction") or ""),
                "source_image": str(Path(str(raw["source_image"])).resolve()),
                "source_tissue_mask": str(
                    Path(str(raw["source_tissue_mask"])).resolve()
                ),
                "source_nuclei_mask": str(
                    Path(str(raw["source_nuclei_mask"])).resolve()
                ),
                "target_tissue_mask": str(
                    Path(str(raw["target_tissue_mask"])).resolve()
                ),
                "target_nuclei_mask": str(
                    Path(str(raw["target_nuclei_mask"])).resolve()
                ),
                "semantic_change_region": str(semantic_region.resolve()),
                "generation_change_region": str(generation_region.resolve()),
                "selected_image": str(selected_image.resolve()),
                "agentic_workflow": str(workflow_path.resolve()),
                "pipeline_summary": str(pipeline_summary_path.resolve()),
                "generation_report": str(report_path.resolve()),
                "workflow_status": str(workflow.get("status") or ""),
                "semantic_matches_tissue_difference": bool(
                    change_regions.get("semantic_matches_tissue_difference")
                ),
                "semantic_matches_joint_difference": bool(
                    change_regions.get("semantic_matches_joint_difference")
                ),
                "selected_model": str(
                    selected.get("requested_mode")
                    or workflow.get("image_generation_provenance", {}).get(
                        "selected_mode", ""
                    )
                ),
                "quality_score": verification.get("quality_score"),
                "evidence_coverage": verification.get("evidence_coverage"),
                "failed_checks": list(
                    verification.get("failed_checks") or ()
                ),
                "reason_codes": list(
                    verification.get("reason_codes") or ()
                ),
                "metrics": dict(verification.get("metrics") or {}),
                "generation_context_policy": dict(generation_policy),
                "route": dict(pipeline_summary.get("route") or {}),
                "code_commit": str(raw.get("code_commit") or ""),
            }
        )
        for key in (
            "source_image",
            "source_tissue_mask",
            "source_nuclei_mask",
            "target_tissue_mask",
            "target_nuclei_mask",
            "semantic_change_region",
            "generation_change_region",
            "selected_image",
            "agentic_workflow",
            "pipeline_summary",
            "generation_report",
        ):
            path = Path(str(records[-1][key]))
            records[-1][f"{key}_sha256"] = (
                _sha256(path) if path.is_file() else ""
            )
    return records


def preflight_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    seen: set[str] = set()
    observed: list[dict[str, Any]] = []
    required_paths = (
        "source_image",
        "source_tissue_mask",
        "source_nuclei_mask",
        "target_tissue_mask",
        "target_nuclei_mask",
        "semantic_change_region",
        "generation_change_region",
        "selected_image",
        "agentic_workflow",
        "pipeline_summary",
        "generation_report",
    )
    for record in records:
        case_id = str(record["case_id"])
        if case_id in seen:
            failures.append(f"duplicate case id: {case_id}")
            continue
        seen.add(case_id)
        for key in required_paths:
            path = Path(str(record[key]))
            if not path.is_file():
                failures.append(f"{case_id}: missing {key}: {path}")
            elif _sha256(path) != str(record.get(f"{key}_sha256") or ""):
                failures.append(f"{case_id}: hash mismatch for {key}")
        if failures and any(item.startswith(f"{case_id}:") for item in failures):
            continue
        source = _load_mask(record["source_tissue_mask"])
        target = _load_mask(record["target_tissue_mask"])
        source_nuclei = _load_mask(record["source_nuclei_mask"])
        target_nuclei = _load_mask(record["target_nuclei_mask"])
        semantic = _load_mask(record["semantic_change_region"]) > 0
        generation = _load_mask(record["generation_change_region"]) > 0
        shapes = {
            source.shape,
            target.shape,
            source_nuclei.shape,
            target_nuclei.shape,
            semantic.shape,
            generation.shape,
        }
        if len(shapes) != 1:
            failures.append(f"{case_id}: mask shape mismatch")
            continue
        exact_semantic = (source != target) | (source_nuclei != target_nuclei)
        if not np.array_equal(semantic, exact_semantic):
            failures.append(f"{case_id}: semantic region is not exact joint diff")
        if np.any(semantic & ~generation):
            failures.append(f"{case_id}: generation region misses semantic pixels")
        observed.append(
            {
                "case_id": case_id,
                "semantic_pixels": int(np.count_nonzero(semantic)),
                "generation_pixels": int(np.count_nonzero(generation)),
                "generation_to_semantic_ratio": float(
                    np.count_nonzero(generation)
                    / max(1, np.count_nonzero(semantic))
                ),
            }
        )
    return {
        "schema_version": "online-generation-review-preflight-v2",
        "passed": not failures,
        "case_count": len(records),
        "failure_count": len(failures),
        "failure_reasons": failures,
        "observed_regions": observed,
    }


def render_panel(
    record: Mapping[str, Any],
    output_path: Path,
    *,
    jpeg_quality: int,
) -> None:
    source_image = _load_rgb(record["source_image"])
    generated_image = _load_rgb(record["selected_image"])
    source_tissue = _load_mask(record["source_tissue_mask"])
    target_tissue = _load_mask(record["target_tissue_mask"])
    source_nuclei = _load_mask(record["source_nuclei_mask"])
    target_nuclei = _load_mask(record["target_nuclei_mask"])
    semantic = _load_mask(record["semantic_change_region"]) > 0
    generation = _load_mask(record["generation_change_region"]) > 0

    tiles = (
        source_image,
        tissue_nuclei_overlay(source_image, source_tissue, source_nuclei),
        tissue_nuclei_overlay(source_image, target_tissue, target_nuclei),
        generated_image,
        draw_semantic_generation_boundaries(
            generated_image,
            semantic,
            generation,
        ),
    )
    canvas = Image.new(
        "RGB",
        (TILE_SIZE * len(tiles), HEADER_HEIGHT + TILE_SIZE + LABEL_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    score = record.get("quality_score")
    score_text = "n/a" if score is None else f"{float(score):.3f}"
    title = (
        f"{record['case_id']} | {record['category']} | "
        f"{record['primitive_id']} | semantic={semantic.mean():.1%} | "
        f"generation={generation.mean():.1%} | "
        f"selected={record['selected_model']} | "
        f"{record['workflow_status']} | quality={score_text}"
    )
    draw.text((12, 8), title, fill=(18, 18, 18), font=_font(20))
    tissue_ids = sorted(
        set(np.unique(source_tissue)).union(np.unique(target_tissue)) - {255}
    )
    _draw_legend(
        draw,
        y=43,
        items=[
            (
                FINE_LABELS.get(int(item), f"Tissue {item}"),
                UNIFIED_COLOR_MAP.get(int(item), [0, 0, 0]),
            )
            for item in tissue_ids
        ],
        font=_font(17),
        prefix="Tissue:",
    )
    cell_ids = sorted(
        set(np.unique(source_nuclei)).union(np.unique(target_nuclei))
        & set(CELL_CLASSES)
    )
    _draw_legend(
        draw,
        y=78,
        items=[(CELL_CLASSES[item], CELL_COLOR_MAP[item]) for item in cell_ids],
        font=_font(17),
        prefix="Nuclei:",
    )
    _draw_legend(
        draw,
        y=113,
        items=[
            ("semantic change", (255, 220, 0)),
            ("generation support", (0, 235, 255)),
        ],
        font=_font(17),
        prefix="Boundaries:",
    )
    draw.text(
        (12, 145),
        str(record["instruction"])[:245],
        fill=(45, 45, 45),
        font=_font(15),
    )
    for index, (label, tile) in enumerate(zip(PANEL_TITLES, tiles)):
        x = index * TILE_SIZE
        canvas.paste(_fit(tile, TILE_SIZE), (x, HEADER_HEIGHT))
        draw.rectangle(
            (x, HEADER_HEIGHT, x + TILE_SIZE - 1, HEADER_HEIGHT + TILE_SIZE - 1),
            outline=(70, 70, 70),
            width=1,
        )
        draw.text(
            (x + 8, HEADER_HEIGHT + TILE_SIZE + 7),
            label,
            fill=(15, 15, 15),
            font=_font(16),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(
        output_path,
        quality=jpeg_quality,
        subsampling=0,
        optimize=True,
    )


def draw_semantic_generation_boundaries(
    image: Image.Image,
    semantic: np.ndarray,
    generation: np.ndarray,
) -> Image.Image:
    if semantic.shape != generation.shape:
        raise ValueError("semantic and generation masks must align")
    fitted = image.resize(
        (semantic.shape[1], semantic.shape[0]), Image.Resampling.BILINEAR
    )
    result = np.asarray(fitted).copy()
    semantic_line = _boundary_line(semantic, maximum=5, minimum=5)
    generation_line = _boundary_line(generation, maximum=7, minimum=5)
    result[generation_line] = np.array([0, 235, 255], dtype=np.uint8)
    result[semantic_line] = np.array([255, 220, 0], dtype=np.uint8)
    return Image.fromarray(result)


def _boundary_line(
    mask: np.ndarray,
    *,
    maximum: int,
    minimum: int,
) -> np.ndarray:
    binary = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    outer = np.asarray(binary.filter(ImageFilter.MaxFilter(maximum))) > 0
    inner = np.asarray(binary.filter(ImageFilter.MinFilter(minimum))) > 0
    return outer & ~inner


def write_html_index(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    cards = []
    for record in records:
        relative = Path(str(record["panel_path"])).relative_to(path.parent)
        score = record.get("quality_score")
        score_text = "n/a" if score is None else f"{float(score):.3f}"
        cards.append(
            "<article><h2>"
            + html.escape(
                f"{record['panel_index']:03d}. {record['case_id']} | "
                f"{record['category']} | {record['selected_model']} | "
                f"quality={score_text}"
            )
            + "</h2><a href='"
            + html.escape(str(relative))
            + "'><img loading='lazy' src='"
            + html.escape(str(relative))
            + "'></a></article>"
        )
    document = """<!doctype html><html><head><meta charset="utf-8">
<title>Online generation review panels</title><style>
body{margin:0;background:#eceff1;color:#111;font:14px system-ui,sans-serif}
header{position:sticky;top:0;background:#fff;padding:12px 18px;border-bottom:1px solid #bbb;z-index:2}
main{padding:16px}article{background:#fff;margin:0 0 18px;padding:10px;border:1px solid #bbb}
h1,h2{margin:0 0 8px}h2{font-size:16px}img{display:block;width:100%;height:auto}
</style></head><body><header><h1>Online Generator review panels</h1>
<div>Yellow: semantic tissue change. Cyan: generator-only support boundary.</div></header>
<main>""" + "\n".join(cards) + "</main></body></html>"
    path.write_text(document, encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
