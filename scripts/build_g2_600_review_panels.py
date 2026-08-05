#!/usr/bin/env python3
"""Build one-row, hash-verified human review panels for a G2 cohort."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_config.unified_labels import (  # noqa: E402
    CELL_CLASSES,
    CELL_COLOR_MAP,
    FINE_LABELS,
    UNIFIED_COLOR_MAP,
)


TILE_SIZE = 512
HEADER_HEIGHT = 142
LABEL_HEIGHT = 34
PANEL_TITLES = (
    "Source H&E",
    "Source tissue + nuclei",
    "Target tissue + nuclei",
    "Selected generated image",
    "Generated image + changed-region boundary",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approved-mask-manifest", type=Path, required=True)
    parser.add_argument("--approved-nuclei-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=600)
    parser.add_argument("--min-change-fraction-image", type=float, default=0.05)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = resolve_records(
        manifest=_read_json(args.manifest),
        approved_masks=_read_json(args.approved_mask_manifest),
        approved_nuclei=_read_json(args.approved_nuclei_manifest),
        evaluator_summary=_read_json(args.evaluator_summary),
    )
    if len(records) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} panel records, found {len(records)}."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "panels").mkdir(exist_ok=True)
    preflight = preflight_records(
        records,
        minimum_change_fraction=args.min_change_fraction_image,
    )
    _write_json(args.output / "panel_preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError(
            "Panel preflight failed: "
            + "; ".join(preflight["failure_reasons"][:10])
        )

    manifest_rows = []
    for index, record in enumerate(records, start=1):
        panel_path = args.output / "panels" / f"{index:03d}_{record['case_id']}.jpg"
        render_panel(record, panel_path, jpeg_quality=args.jpeg_quality)
        manifest_rows.append(
            {
                **record,
                "panel_index": index,
                "panel_path": str(panel_path),
                "panel_sha256": _sha256(panel_path),
            }
        )
    _write_json(
        args.output / "panel_manifest.json",
        {
            "schema_version": 1,
            "case_count": len(manifest_rows),
            "minimum_changed_area_fraction_image": (
                args.min_change_fraction_image
            ),
            "panel_layout": list(PANEL_TITLES),
            "records": manifest_rows,
        },
    )
    write_html_index(manifest_rows, args.output / "index.html")
    print(
        json.dumps(
            {
                "status": "completed",
                "case_count": len(manifest_rows),
                "output": str(args.output),
                "index": str(args.output / "index.html"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def resolve_records(
    *,
    manifest: Mapping[str, Any],
    approved_masks: Mapping[str, Any],
    approved_nuclei: Mapping[str, Any],
    evaluator_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    mask_by_id = _index_entries(approved_masks, "approved mask")
    nuclei_by_id = _index_entries(approved_nuclei, "approved nuclei")
    eval_by_id = {
        str(item["case_id"]): item
        for item in evaluator_summary.get("results", ())
    }
    records = []
    for case in manifest.get("cases", ()):
        case_id = str(case["case_id"])
        if case_id not in mask_by_id or case_id not in nuclei_by_id:
            raise KeyError(f"Missing approved stage artifact for {case_id}.")
        if case_id not in eval_by_id:
            raise KeyError(f"Missing evaluator result for {case_id}.")
        evaluation = eval_by_id[case_id]
        replay_path = Path(str(evaluation["evaluator_replay"]))
        replay = _read_json(replay_path)
        selected_image = Path(str(replay["generated_image"]))
        records.append(
            {
                "case_id": case_id,
                "condition_id": str(case.get("condition_id") or ""),
                "dataset": str(case.get("dataset") or ""),
                "organ": str(case.get("organ") or ""),
                "primitive": str(
                    case.get("g2_primitive") or case.get("primitive") or ""
                ),
                "instruction": str(case.get("instruction") or ""),
                "source_image": str(case["source_image"]),
                "source_tissue_mask": str(case["source_tissue_mask"]),
                "source_nuclei_mask": str(case["source_nuclei_mask"]),
                "source_tissue_sha256": str(case["source_mask_sha256"]),
                "target_tissue_mask": str(
                    mask_by_id[case_id]["target_tissue_mask_path"]
                ),
                "target_tissue_sha256": str(
                    mask_by_id[case_id]["target_tissue_sha256"]
                ),
                "target_nuclei_mask": str(
                    nuclei_by_id[case_id]["target_nuclei_mask_path"]
                ),
                "target_nuclei_sha256": str(
                    nuclei_by_id[case_id]["target_nuclei_sha256"]
                ),
                "selected_image": str(selected_image),
                "selected_image_sha256": str(
                    evaluation["selected_image_sha256"]
                ),
                "selected_model": str(evaluation.get("selected_model") or ""),
                "quality_status": str(evaluation.get("status") or ""),
                "quality_score": evaluation.get("quality_score"),
            }
        )
    return records


def preflight_records(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_change_fraction: float,
) -> dict[str, Any]:
    failures = []
    fractions = []
    seen = set()
    for record in records:
        case_id = str(record["case_id"])
        if case_id in seen:
            failures.append(f"duplicate case id: {case_id}")
            continue
        seen.add(case_id)
        for key, hash_key in (
            ("source_tissue_mask", "source_tissue_sha256"),
            ("target_tissue_mask", "target_tissue_sha256"),
            ("target_nuclei_mask", "target_nuclei_sha256"),
            ("selected_image", "selected_image_sha256"),
        ):
            path = Path(str(record[key]))
            if not path.is_file():
                failures.append(f"{case_id}: missing {key}: {path}")
            elif _sha256(path) != str(record[hash_key]):
                failures.append(f"{case_id}: hash mismatch for {key}")
        source = _load_mask(record["source_tissue_mask"])
        target = _load_mask(record["target_tissue_mask"])
        if source.shape != target.shape:
            failures.append(f"{case_id}: source/target shape mismatch")
            continue
        fraction = float(np.mean(source != target))
        fractions.append(fraction)
        if fraction + 1e-12 < minimum_change_fraction:
            failures.append(
                f"{case_id}: changed fraction {fraction:.6f} "
                f"< {minimum_change_fraction:.6f}"
            )
    return {
        "schema_version": 1,
        "passed": not failures,
        "case_count": len(records),
        "minimum_changed_area_fraction_image": minimum_change_fraction,
        "observed_minimum_changed_area_fraction_image": (
            min(fractions) if fractions else None
        ),
        "observed_median_changed_area_fraction_image": (
            float(np.median(fractions)) if fractions else None
        ),
        "failure_count": len(failures),
        "failure_reasons": failures,
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
    changed = source_tissue != target_tissue

    tiles = (
        source_image,
        tissue_nuclei_overlay(source_image, source_tissue, source_nuclei),
        tissue_nuclei_overlay(source_image, target_tissue, target_nuclei),
        generated_image,
        draw_changed_boundary(generated_image, changed),
    )
    width = TILE_SIZE * len(tiles)
    canvas = Image.new("RGB", (width, HEADER_HEIGHT + TILE_SIZE + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(22)
    small = _font(18)
    title = (
        f"{record['case_id']} | {record['dataset']} / {record['organ']} | "
        f"edit: {record['primitive']} | changed={changed.mean():.1%} | "
        f"selected={record['selected_model']} | {record['quality_status']}"
    )
    draw.text((12, 8), title, fill=(18, 18, 18), font=font)
    tissue_ids = sorted(
        set(np.unique(source_tissue)).union(np.unique(target_tissue))
        - {255}
    )
    _draw_legend(
        draw,
        y=45,
        items=[
            (FINE_LABELS.get(int(item), f"Tissue {item}"), UNIFIED_COLOR_MAP.get(int(item), [0, 0, 0]))
            for item in tissue_ids
        ],
        font=small,
        prefix="Tissue:",
    )
    cell_ids = sorted(
        set(np.unique(source_nuclei)).union(np.unique(target_nuclei))
        & set(CELL_CLASSES)
    )
    _draw_legend(
        draw,
        y=83,
        items=[
            (CELL_CLASSES[item], CELL_COLOR_MAP[item])
            for item in cell_ids
        ],
        font=small,
        prefix="Nuclei:",
    )
    draw.text(
        (12, 118),
        record["instruction"][:245],
        fill=(45, 45, 45),
        font=_font(15),
    )
    for index, (title_text, tile) in enumerate(zip(PANEL_TITLES, tiles)):
        x = index * TILE_SIZE
        canvas.paste(_fit(tile, TILE_SIZE), (x, HEADER_HEIGHT))
        draw.rectangle(
            (x, HEADER_HEIGHT, x + TILE_SIZE - 1, HEADER_HEIGHT + TILE_SIZE - 1),
            outline=(70, 70, 70),
            width=1,
        )
        draw.text(
            (x + 8, HEADER_HEIGHT + TILE_SIZE + 6),
            title_text,
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


def tissue_nuclei_overlay(
    image: Image.Image,
    tissue_mask: np.ndarray,
    nuclei_mask: np.ndarray,
) -> Image.Image:
    rgb = np.asarray(_fit(image, tissue_mask.shape[1])).copy()
    if rgb.shape[:2] != tissue_mask.shape:
        rgb = np.asarray(
            image.resize(
                (tissue_mask.shape[1], tissue_mask.shape[0]),
                Image.Resampling.BILINEAR,
            )
        ).copy()
    colors = np.zeros_like(rgb)
    for label, color in UNIFIED_COLOR_MAP.items():
        colors[tissue_mask == label] = color
    tissue_valid = tissue_mask != 255
    rgb[tissue_valid] = (
        0.42 * rgb[tissue_valid] + 0.58 * colors[tissue_valid]
    ).astype(np.uint8)
    for label, color in CELL_COLOR_MAP.items():
        rgb[nuclei_mask == label] = color
    return Image.fromarray(rgb)


def draw_changed_boundary(image: Image.Image, changed: np.ndarray) -> Image.Image:
    fitted = image.resize(
        (changed.shape[1], changed.shape[0]), Image.Resampling.BILINEAR
    )
    result = np.asarray(fitted).copy()
    mask = Image.fromarray((changed.astype(np.uint8) * 255), mode="L")
    halo = np.asarray(mask.filter(ImageFilter.MaxFilter(9))) > 0
    core = np.asarray(mask.filter(ImageFilter.MinFilter(5))) > 0
    outer = halo & ~core
    line = (np.asarray(mask.filter(ImageFilter.MaxFilter(5))) > 0) & ~(
        np.asarray(mask.filter(ImageFilter.MinFilter(5))) > 0
    )
    result[outer] = (0.35 * result[outer]).astype(np.uint8)
    result[line] = np.array([255, 220, 0], dtype=np.uint8)
    return Image.fromarray(result)


def write_html_index(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    cards = []
    for record in records:
        relative = Path(record["panel_path"]).relative_to(path.parent)
        cards.append(
            "<article><h2>"
            + html.escape(
                f"{record['panel_index']:03d}. {record['case_id']} | "
                f"{record['primitive']}"
            )
            + "</h2><a href='"
            + html.escape(str(relative))
            + "'><img loading='lazy' src='"
            + html.escape(str(relative))
            + "'></a></article>"
        )
    document = """<!doctype html><html><head><meta charset="utf-8">
<title>G2-600 review panels</title><style>
body{margin:0;background:#eceff1;color:#111;font:14px system-ui,sans-serif}
header{position:sticky;top:0;background:#fff;padding:12px 18px;border-bottom:1px solid #bbb;z-index:2}
main{padding:16px}article{background:#fff;margin:0 0 18px;padding:10px;border:1px solid #bbb}
h1,h2{margin:0 0 8px}h2{font-size:16px}img{display:block;width:100%;height:auto}
</style></head><body><header><h1>G2-600 one-row review panels</h1>
<div>Click any panel to inspect its original full-resolution JPEG.</div></header>
<main>""" + "\n".join(cards) + "</main></body></html>"
    path.write_text(document, encoding="utf-8")


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    items: Sequence[tuple[str, Sequence[int]]],
    font: ImageFont.ImageFont,
    prefix: str,
) -> None:
    x = 12
    draw.text((x, y), prefix, fill=(20, 20, 20), font=font)
    x += int(draw.textlength(prefix, font=font)) + 14
    for label, color in items:
        draw.rectangle((x, y + 2, x + 18, y + 20), fill=tuple(color), outline=(0, 0, 0))
        x += 24
        draw.text((x, y), label, fill=(20, 20, 20), font=font)
        x += int(draw.textlength(label, font=font)) + 18


def _index_entries(payload: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    entries = {
        str(item["case_id"]): item for item in payload.get("entries", ())
    }
    if len(entries) != len(payload.get("entries", ())):
        raise RuntimeError(f"Duplicate case ids in {label} manifest.")
    return entries


def _fit(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def _load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_mask(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    return array[..., 0] if array.ndim == 3 else array


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


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
