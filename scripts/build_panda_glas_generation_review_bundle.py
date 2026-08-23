#!/usr/bin/env python3
"""Build a self-contained five-column PANDA/GLaS Online generation review."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from collections import Counter, defaultdict
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
HEADER_HEIGHT = 142
LABEL_HEIGHT = 34
PANEL_SIZE = (2560, 688)
PANEL_TITLES = (
    "Source H&E",
    "Source tissue + nuclei",
    "Target tissue + nuclei",
    "Selected generated image",
    "Generated + semantic/generation boundaries",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-overrides", type=Path)
    parser.add_argument("--expected-count", type=int, default=105)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def copy_asset(
    source: str | Path, destination: Path, *, bundle_root: Path
) -> dict[str, Any]:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return {
        "path": str(destination),
        "relative_path": str(destination.relative_to(bundle_root)),
        "absolute_path": str(destination),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
    }


def attempt_map(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for attempt in workflow.get("attempts") or ():
        artifact = attempt.get("artifact") or {}
        image_path = artifact.get("image_path")
        if image_path and Path(image_path).is_file():
            result[str(attempt.get("requested_mode"))] = attempt
    return result


def select_attempt(
    workflow: dict[str, Any], override: str | None
) -> tuple[dict[str, Any], str]:
    attempts = attempt_map(workflow)
    if override:
        if override not in attempts:
            raise RuntimeError(f"Requested visual override is unavailable: {override}")
        return attempts[override], "visual_review_override"
    selected = workflow.get("selected_attempt") or {}
    mode = str(selected.get("requested_mode") or "")
    if mode not in attempts:
        raise RuntimeError("Agentic workflow has no persisted selected attempt")
    return attempts[mode], "agentic_selected_attempt"


def boundary(mask: np.ndarray, maximum: int, minimum: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    outer = np.asarray(image.filter(ImageFilter.MaxFilter(maximum))) > 0
    inner = np.asarray(image.filter(ImageFilter.MinFilter(minimum))) > 0
    return outer & ~inner


def boundary_overlay(
    image: Image.Image, semantic: np.ndarray, generation: np.ndarray
) -> Image.Image:
    result = np.asarray(
        image.convert("RGB").resize(
            (semantic.shape[1], semantic.shape[0]), Image.Resampling.BILINEAR
        )
    ).copy()
    result[boundary(generation, 7, 5)] = [0, 235, 255]
    result[boundary(semantic, 5, 5)] = [255, 220, 0]
    return Image.fromarray(result)


def image_metrics(
    source: Image.Image,
    generated: Image.Image,
    semantic: np.ndarray,
    generation: np.ndarray,
) -> dict[str, Any]:
    source_array = np.asarray(source.convert("RGB"), dtype=np.int16)
    generated_array = np.asarray(generated.convert("RGB"), dtype=np.int16)
    difference = np.mean(np.abs(generated_array - source_array), axis=2)
    inner_boundary = boundary(generation, 5, 1) & generation
    context = generation & ~semantic

    def mean(region: np.ndarray) -> float:
        return float(difference[region].mean()) if np.any(region) else 0.0

    return {
        "outside_generation_support_source_exact": bool(
            np.array_equal(source_array[~generation], generated_array[~generation])
        ),
        "outside_generation_support_changed_pixels": int(
            np.count_nonzero(np.any(source_array != generated_array, axis=2) & ~generation)
        ),
        "mean_rgb_delta_semantic": mean(semantic),
        "mean_rgb_delta_generator_context": mean(context),
        "mean_rgb_delta_inner_support_boundary": mean(inner_boundary),
    }


def resolve_records(
    source_manifest: Path, overrides: dict[str, Any]
) -> list[dict[str, Any]]:
    payload = read_json(source_manifest)
    records: list[dict[str, Any]] = []
    for raw in payload.get("records") or ():
        override_value = overrides.get(raw["case_id"])
        if isinstance(override_value, dict):
            override_mode = override_value.get("selected_mode")
            override_reason = override_value.get("reason")
            generation_dir = Path(
                override_value.get(
                    "generation_dir", raw["agentic_generation_dir"]
                )
            )
        else:
            override_mode = override_value
            override_reason = None
            generation_dir = Path(raw["agentic_generation_dir"])
        workflow = read_json(generation_dir / "agentic_workflow.json")
        selected, selection_source = select_attempt(workflow, override_mode)
        attempts = attempt_map(workflow)
        selected_artifact = selected["artifact"]
        selected_verification = selected.get("verification") or {}
        record = {
            **raw,
            "agentic_generation_dir": str(generation_dir),
            "workflow_status": workflow.get("status"),
            "selected_mode": selected.get("requested_mode"),
            "selected_image": selected_artifact["image_path"],
            "selected_attempt_index": selected.get("attempt_index"),
            "selected_quality_score": selected_verification.get("quality_score"),
            "selected_failed_checks": selected_verification.get("failed_checks") or [],
            "selection_source": selection_source,
            "selection_reason": override_reason,
            "attempts": {
                mode: {
                    "image": attempt["artifact"]["image_path"],
                    "attempt_index": attempt.get("attempt_index"),
                    "quality_score": (attempt.get("verification") or {}).get(
                        "quality_score"
                    ),
                    "failed_checks": (attempt.get("verification") or {}).get(
                        "failed_checks"
                    )
                    or [],
                }
                for mode, attempt in attempts.items()
            },
        }
        records.append(record)
    records.sort(
        key=lambda item: (
            0 if item["dataset"] == "PANDA" else 1,
            int(item["primitive_ordinal"]),
            str(item["case_id"]),
        )
    )
    return records


def materialize_assets(record: dict[str, Any], root: Path) -> dict[str, Any]:
    case_root = (
        root
        / "assets"
        / slug(record["dataset"])
        / f"{int(record['primitive_ordinal']):02d}_{slug(record['category'])}_{slug(record['primitive_id'])}"
        / record["case_id"]
    )
    sources = {
        "source": record["source_image"],
        "source_tissue": record["source_tissue_mask"],
        "source_nuclei": record["source_nuclei_mask"],
        "target_tissue": record["target_tissue_mask"],
        "target_nuclei": record["target_nuclei_mask"],
        "semantic": record["semantic_change_region"],
        "generation_support": record["generation_change_region"],
        "selected": record["selected_image"],
    }
    for mode, attempt in record["attempts"].items():
        sources[f"attempt_{slug(mode)}"] = attempt["image"]
    return {
        name: copy_asset(
            path,
            case_root / f"{name}.png",
            bundle_root=root,
        )
        for name, path in sources.items()
    }


def render_panel(record: dict[str, Any], output_path: Path) -> None:
    assets = record["assets"]
    source = _load_rgb(assets["source"]["path"])
    generated = _load_rgb(assets["selected"]["path"])
    source_tissue = _load_mask(assets["source_tissue"]["path"])
    target_tissue = _load_mask(assets["target_tissue"]["path"])
    source_nuclei = _load_mask(assets["source_nuclei"]["path"])
    target_nuclei = _load_mask(assets["target_nuclei"]["path"])
    semantic = _load_mask(assets["semantic"]["path"]) > 0
    generation = _load_mask(assets["generation_support"]["path"]) > 0
    tiles = (
        source,
        tissue_nuclei_overlay(source, source_tissue, source_nuclei),
        tissue_nuclei_overlay(source, target_tissue, target_nuclei),
        generated,
        boundary_overlay(generated, semantic, generation),
    )
    canvas = Image.new("RGB", PANEL_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    score = record.get("selected_quality_score")
    score_text = "n/a" if score is None else f"{float(score):.3f}"
    title = (
        f"{record['case_id']} | {record['dataset']} | {record['category']} | "
        f"{record['primitive_id']} | selected={record['selected_mode']} | "
        f"{record['workflow_status']} ({score_text})"
    )
    draw.text((12, 8), title, fill=(18, 18, 18), font=_font(19))
    tissue_ids = sorted(
        (set(np.unique(source_tissue)) | set(np.unique(target_tissue))) - {255}
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
        font=_font(16),
        prefix="Tissue:",
    )
    cell_ids = sorted(
        (set(np.unique(source_nuclei)) | set(np.unique(target_nuclei)))
        & set(CELL_CLASSES)
    )
    _draw_legend(
        draw,
        y=78,
        items=[(CELL_CLASSES[item], CELL_COLOR_MAP[item]) for item in cell_ids],
        font=_font(16),
        prefix="Nuclei:",
    )
    draw.text(
        (12, 113),
        (
            f"semantic={semantic.mean():.1%} | generation support={generation.mean():.1%} | "
            "yellow=semantic boundary | cyan=generation-support boundary | "
            f"selection={record['selection_source']}"
        ),
        fill=(45, 45, 45),
        font=_font(15),
    )
    for index, (label, tile) in enumerate(zip(PANEL_TITLES, tiles)):
        x = index * TILE_SIZE
        canvas.paste(_fit(tile, TILE_SIZE), (x, HEADER_HEIGHT))
        draw.rectangle(
            (x, HEADER_HEIGHT, x + 511, HEADER_HEIGHT + 511),
            outline=(70, 70, 70),
            width=1,
        )
        draw.text(
            (x + 8, HEADER_HEIGHT + TILE_SIZE + 6),
            label,
            fill=(15, 15, 15),
            font=_font(15),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, subsampling=0, optimize=True)


def render_attempt_board(records: list[dict[str, Any]], output_path: Path) -> None:
    row_height = 540
    canvas = Image.new("RGB", (2048, 42 + row_height * len(records)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 8),
        f"{records[0]['dataset']} | {records[0]['category']} | {records[0]['primitive_id']} | 5-case Inpaint/Cross visual review",
        fill=(18, 18, 18),
        font=_font(20),
    )
    for row_index, record in enumerate(records):
        assets = record["assets"]
        source = _load_rgb(assets["source"]["path"])
        semantic = _load_mask(assets["semantic"]["path"]) > 0
        generation = _load_mask(assets["generation_support"]["path"]) > 0
        attempt_assets = {
            key.removeprefix("attempt_"): value
            for key, value in assets.items()
            if key.startswith("attempt_")
        }
        missing = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (235, 235, 235))
        missing_draw = ImageDraw.Draw(missing)
        missing_draw.text(
            (145, 238),
            "Not run by agent",
            fill=(70, 70, 70),
            font=_font(20),
        )
        inpaint = next(
            (
                _load_rgb(value["path"])
                for key, value in attempt_assets.items()
                if key == "inpaint"
            ),
            missing,
        )
        cross = next(
            (
                _load_rgb(value["path"])
                for key, value in attempt_assets.items()
                if key.startswith("cross")
            ),
            missing,
        )
        selected = _load_rgb(assets["selected"]["path"])
        tiles = (source, inpaint, cross, boundary_overlay(selected, semantic, generation))
        labels = ("Source", "Inpaint", "Cross-v1", f"Selected + boundaries ({record['selected_mode']})")
        y = 42 + row_index * row_height
        for column, (tile, label) in enumerate(zip(tiles, labels)):
            x = column * TILE_SIZE
            canvas.paste(_fit(tile, TILE_SIZE), (x, y))
            draw.text((x + 7, y + 514), label, fill=(10, 10, 10), font=_font(15))
        draw.text(
            (1550, y + 5),
            record["case_id"],
            fill=(10, 10, 10),
            font=_font(14),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=91, subsampling=0, optimize=True)


def build_html(records: list[dict[str, Any]]) -> str:
    cards = []
    for record in records:
        panel = Path(record["panel"]["path"])
        relative = panel.relative_to(Path(record["bundle_root"]))
        cards.append(
            f"<article data-dataset='{html.escape(record['dataset'])}'><h2>{record['panel_index']:03d}. {html.escape(record['case_id'])}</h2>"
            f"<p>{html.escape(record['dataset'])} · {html.escape(record['category'])} · {html.escape(record['primitive_id'])} · selected={html.escape(record['selected_mode'])}</p>"
            f"<a href='{html.escape(str(relative))}'><img loading='lazy' src='{html.escape(str(relative))}'></a></article>"
        )
    return """<!doctype html><html><head><meta charset='utf-8'><title>PANDA + GLaS Online generation review</title>
<style>body{margin:0;background:#eceff1;font:14px system-ui;color:#111}header{position:sticky;top:0;background:white;padding:12px 18px;border-bottom:1px solid #aaa;z-index:2}main{padding:16px}article{background:white;border:1px solid #bbb;margin:0 0 18px;padding:10px}h1,h2,p{margin:0 0 8px}h2{font-size:16px}img{display:block;width:100%;height:auto}a{color:#075c66}</style></head><body><header><h1>PANDA + GLaS · Online Inpaint/Cross generation review</h1><p>Five columns: source H&amp;E · source masks · target masks · selected generation · semantic/generation boundaries</p><p><a href='panel_manifest.json'>panel manifest</a> · <a href='final_integrity_audit.json'>integrity audit</a> · <a href='attempt_review_boards/'>Inpaint/Cross boards</a></p></header><main>""" + "\n".join(cards) + "</main></body></html>"


def main() -> int:
    args = parse_args()
    overrides = read_json(args.selection_overrides) if args.selection_overrides else {}
    records = resolve_records(args.manifest.resolve(), overrides)
    if len(records) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} cases, found {len(records)}")
    output = args.output.resolve()
    (output / "panels").mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records, start=1):
        record["assets"] = materialize_assets(record, output)
        source = _load_rgb(record["assets"]["source"]["path"])
        selected = _load_rgb(record["assets"]["selected"]["path"])
        semantic = _load_mask(record["assets"]["semantic"]["path"]) > 0
        generation = _load_mask(record["assets"]["generation_support"]["path"]) > 0
        if np.any(semantic & ~generation):
            failures.append(f"{record['case_id']}: support misses semantic pixels")
        record["image_metrics"] = image_metrics(source, selected, semantic, generation)
        if not record["image_metrics"]["outside_generation_support_source_exact"]:
            failures.append(f"{record['case_id']}: selected image changed outside support")
        record["panel_index"] = index
        record["bundle_root"] = str(output)
        panel_path = output / "panels" / f"{index:03d}_{record['case_id']}.jpg"
        render_panel(record, panel_path)
        if Image.open(panel_path).size != PANEL_SIZE:
            failures.append(f"{record['case_id']}: wrong panel dimensions")
        record["panel"] = {
            "path": str(panel_path),
            "relative_path": str(panel_path.relative_to(output)),
            "absolute_path": str(panel_path),
            "sha256": sha256(panel_path),
            "width": PANEL_SIZE[0],
            "height": PANEL_SIZE[1],
        }
        grouped[(record["dataset"], int(record["primitive_ordinal"]), record["category"], record["primitive_id"])].append(record)
    board_records = []
    for board_index, (key, group) in enumerate(sorted(grouped.items()), start=1):
        if len(group) != 5 and args.expected_count == 105:
            failures.append(f"{key}: expected five cases, found {len(group)}")
        source_hashes = [item["assets"]["source"]["sha256"] for item in group]
        selected_hashes = [item["assets"]["selected"]["sha256"] for item in group]
        if len(source_hashes) != len(set(source_hashes)):
            failures.append(f"{key}: duplicate source panels")
        if len(selected_hashes) != len(set(selected_hashes)):
            failures.append(f"{key}: duplicate selected-generation panels")
        board_path = output / "attempt_review_boards" / f"{board_index:02d}_{slug('_'.join(map(str, key)))}.jpg"
        render_attempt_board(group, board_path)
        board_records.append(
            {
                "key": list(key),
                "path": str(board_path),
                "relative_path": str(board_path.relative_to(output)),
                "absolute_path": str(board_path),
                "sha256": sha256(board_path),
            }
        )
    manifest = {
        "schema_version": "panda-glas-online-generation-review-bundle-v1",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "case_count": len(records),
        "panel_dimensions": list(PANEL_SIZE),
        "panel_layout": list(PANEL_TITLES),
        "dataset_counts": dict(Counter(record["dataset"] for record in records)),
        "selected_mode_counts": dict(Counter(record["selected_mode"] for record in records)),
        "records": records,
        "attempt_review_boards": board_records,
    }
    write_json(output / "panel_manifest.json", manifest)
    (output / "index.html").write_text(build_html(records), encoding="utf-8")
    integrity = {
        "schema_version": "panda-glas-online-generation-review-integrity-v1",
        "passed": not failures,
        "case_count": len(records),
        "panel_count": len(list((output / "panels").glob("*.jpg"))),
        "attempt_review_board_count": len(board_records),
        "all_panels_2560x688": not any("wrong panel" in item for item in failures),
        "all_selected_images_source_exact_outside_generation_support": not any(
            "outside support" in item for item in failures
        ),
        "failure_reasons": failures,
        "panel_manifest_sha256": sha256(output / "panel_manifest.json"),
        "index_sha256": sha256(output / "index.html"),
    }
    write_json(output / "final_integrity_audit.json", integrity)
    print(json.dumps({"status": "completed" if not failures else "failed", "output": str(output), "cases": len(records), "failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
