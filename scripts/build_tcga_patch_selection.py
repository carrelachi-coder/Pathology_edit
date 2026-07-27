#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024
PngImagePlugin.MAX_TEXT_MEMORY = 1024 * 1024 * 1024

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from segmentator.patch_selection import (
    ORGAN_RULES,
    finalize_row_scores,
    select_candidate_pool,
    select_case_disjoint_sets,
)


CASE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
PALETTE = {
    0: [30, 30, 30],
    1: [220, 40, 40],
    2: [45, 170, 75],
    3: [145, 70, 190],
    4: [45, 110, 225],
    5: [245, 145, 35],
    6: [35, 200, 205],
    7: [205, 190, 45],
    255: [255, 255, 255],
}
LABELS = {
    0: "Background",
    1: "Tumor",
    2: "Stroma",
    3: "Necrosis",
    4: "Immune infiltrate",
    5: "Normal epithelium",
    6: "Blood vessel",
    7: "Other tissue",
    255: "Ignore",
}


def _read_shard_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    by_filename: dict[str, dict[str, object]] = {}
    for path in paths:
        for row in csv.DictReader(path.open()):
            if row.get("filename"):
                by_filename[row["filename"]] = dict(row)
    return [by_filename[name] for name in sorted(by_filename)]


def _training_cases(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    cases: set[str] = set()
    for split in ("train", "val"):
        for record in payload.get(split, []):
            matches = CASE_RE.findall(" ".join(str(value) for value in record.values()))
            cases.update(match.upper() for match in matches)
    return cases


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path_without_suffix: Path, rows: list[Mapping[str, object]]) -> None:
    materialized = [dict(row) for row in rows]
    _write_csv(path_without_suffix.with_suffix(".csv"), materialized)
    path_without_suffix.with_suffix(".json").write_text(
        json.dumps(materialized, indent=2, sort_keys=True) + "\n"
    )


def _ensure_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_ino != destination.stat().st_ino or source.stat().st_dev != destination.stat().st_dev:
            raise FileExistsError(f"unexpected existing destination: {destination}")
        return
    os.link(source, destination)


def _materialize_subset(root: Path, rows: list[dict[str, object]], mask_dir: Path) -> None:
    for row in rows:
        filename = str(row["filename"])
        stem = Path(filename).stem
        _ensure_link(Path(str(row["image_path"])), root / "images" / filename)
        _ensure_link(Path(str(row["text_path"])), root / "txts" / f"{stem}.txt")
        _ensure_link(mask_dir / filename, root / "model_masks" / filename)
    _write_manifest(root / "manifest", rows)


def _annotation_package(
    root: Path,
    complex_rows: list[dict[str, object]],
    random_rows: list[dict[str, object]],
    *,
    seed: int,
) -> None:
    all_rows = [dict(row, stratum="complex") for row in complex_rows] + [
        dict(row, stratum="random") for row in random_rows
    ]
    for row in all_rows:
        filename = str(row["filename"])
        image_path = Path(str(row["image_path"]))
        _ensure_link(image_path, root / "images" / filename)
        with Image.open(image_path) as image:
            shape = image.size
        label_path = root / "labels_primary" / filename
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not label_path.exists():
            Image.new("L", shape, color=255).save(label_path, compress_level=1)

    rng = random.Random(seed)
    double_rows: list[dict[str, object]] = []
    for organ in ORGAN_RULES:
        organ_complex = [row for row in all_rows if row["organ"] == organ and row["stratum"] == "complex"]
        organ_random = [row for row in all_rows if row["organ"] == organ and row["stratum"] == "random"]
        double_rows.extend(rng.sample(organ_complex, min(25, len(organ_complex))))
        double_rows.extend(rng.sample(organ_random, min(10, len(organ_random))))
    for row in double_rows:
        filename = str(row["filename"])
        source = root / "labels_primary" / filename
        destination = root / "labels_secondary" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            Image.open(source).save(destination, compress_level=1)

    _write_manifest(root / "manifest", all_rows)
    _write_manifest(root / "double_annotation_manifest", double_rows)
    (root / "palette.json").write_text(
        json.dumps(
            {str(label_id): {"name": LABELS[label_id], "rgb": PALETTE[label_id]} for label_id in LABELS},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (root / "README_zh.txt").write_text(
        "统一8类语义分割标注。标签PNG必须与原图同尺寸，允许值为0-7或255。\n"
        "255表示Ignore/无法判定。标注员不得查看model_masks或TXT描述。\n"
        "labels_secondary仅用于双人独立标注，不得参考labels_primary。\n"
    )


def _colorize(mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label_id, color in PALETTE.items():
        if label_id != 255:
            output[mask == label_id] = np.asarray(color, dtype=np.uint8)
    return output


def _contact_sheet(path: Path, rows: list[dict[str, object]], mask_dir: Path, title: str) -> None:
    if not rows:
        return
    cells = []
    for row in rows[:16]:
        with Image.open(str(row["image_path"])) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_dir / str(row["filename"])) as mask_image:
            mask = np.asarray(mask_image, dtype=np.uint8)
        overlay = (rgb.astype(np.float32) * 0.60 + _colorize(mask).astype(np.float32) * 0.40).astype(np.uint8)
        cell = Image.fromarray(overlay).resize((256, 256))
        draw = ImageDraw.Draw(cell)
        draw.rectangle((0, 0, 256, 28), fill=(0, 0, 0))
        draw.text((5, 5), f"score={float(row['selection_score']):.3f} other={float(row['other_fraction']):.1%}", fill="white")
        cells.append(cell)
    sheet = Image.new("RGB", (1024, 1024), color="white")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % 4) * 256, (index // 4) * 256))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score and select low-Other TCGA patches.")
    parser.add_argument("--metrics-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--organ-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--source-exclusions", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-target", type=int, default=10000)
    parser.add_argument("--candidate-organ-floor", type=int, default=500)
    parser.add_argument("--complex-per-organ", type=int, default=250)
    parser.add_argument("--random-per-organ", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    excluded_filenames: set[str] = set()
    if args.source_exclusions and args.source_exclusions.exists():
        exclusion_payload = json.loads(args.source_exclusions.read_text())
        failures = exclusion_payload.get("failures", exclusion_payload) if isinstance(exclusion_payload, dict) else exclusion_payload
        for failure in failures:
            excluded_filenames.add(str(failure["filename"] if isinstance(failure, dict) else failure))
    manifest_rows = {
        row["filename"]: row
        for row in csv.DictReader(args.organ_manifest.open())
        if row["filename"] not in excluded_filenames
    }
    rows = _read_shard_rows(args.metrics_csv)
    if len(rows) != len(manifest_rows):
        missing = sorted(set(manifest_rows) - {str(row["filename"]) for row in rows})
        raise RuntimeError(f"metrics coverage mismatch rows={len(rows)} manifest={len(manifest_rows)} missing={missing[:10]}")

    training_cases = _training_cases(args.training_manifest)
    for row in rows:
        row.update(manifest_rows[str(row["filename"])])
        row["training_overlap"] = str(row["case_id"]).upper() in training_cases
    scored = finalize_row_scores(rows)

    case_caps = {"breast": 1, "lung": 1, "prostate": 2, "colorectal": 2, "head_neck": 2, "skin": 5}
    random_case_caps = {organ: 1 for organ in ORGAN_RULES}
    random_case_caps["skin"] = 3
    selected = select_case_disjoint_sets(
        scored,
        complex_per_organ=args.complex_per_organ,
        random_per_organ=args.random_per_organ,
        seed=args.seed,
        case_caps=case_caps,
        random_case_caps=random_case_caps,
    )
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if any(value for deficit in selected.deficits.values() for value in deficit.values()):
        (output_root / "selection_deficits.json").write_text(
            json.dumps(selected.deficits, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(f"selection deficits without threshold relaxation: {selected.deficits}")

    candidates = select_candidate_pool(
        scored,
        target=args.candidate_target,
        organ_floor=args.candidate_organ_floor,
        case_cap=10,
        required_rows=selected.complex_rows,
    )
    if len(candidates) != args.candidate_target:
        raise RuntimeError(f"candidate deficit expected={args.candidate_target} actual={len(candidates)}")
    candidate_counts = Counter(str(row["organ"]) for row in candidates)
    candidate_floor_deficits = {
        organ: max(0, args.candidate_organ_floor - candidate_counts.get(organ, 0)) for organ in ORGAN_RULES
    }
    if any(candidate_floor_deficits.values()):
        (output_root / "candidate_floor_deficits.json").write_text(
            json.dumps(candidate_floor_deficits, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(f"candidate organ-floor deficits: {candidate_floor_deficits}")

    _write_csv(output_root / "metrics" / "all_scored.csv", scored)
    _materialize_subset(output_root / "candidate_10000", candidates, args.mask_dir)
    _materialize_subset(output_root / "complex_1500", selected.complex_rows, args.mask_dir)
    _materialize_subset(output_root / "random_eval_600", selected.random_rows, args.mask_dir)
    _annotation_package(output_root / "annotation_png", selected.complex_rows, selected.random_rows, seed=args.seed)

    for organ in ORGAN_RULES:
        eligible = [
            row
            for row in scored
            if row["organ"] == organ
            and bool(row["quality_pass"])
            and bool(row["organ_constraints_pass"])
            and not bool(row["training_overlap"])
        ]
        ranked = sorted(eligible, key=lambda row: (-float(row["selection_score"]), str(row["filename"])))
        _contact_sheet(output_root / "qa" / f"{organ}_top16.png", ranked[:16], args.mask_dir, f"{organ} top")
        _contact_sheet(output_root / "qa" / f"{organ}_bottom16.png", list(reversed(ranked[-16:])), args.mask_dir, f"{organ} bottom")

    complex_cases = {str(row["case_id"]) for row in selected.complex_rows}
    random_cases = {str(row["case_id"]) for row in selected.random_rows}
    summary = {
        "metrics_rows": len(scored),
        "source_exclusions": sorted(excluded_filenames),
        "training_cases": len(training_cases),
        "training_overlap_rows": sum(bool(row["training_overlap"]) for row in scored),
        "quality_pass_by_organ": dict(Counter(str(row["organ"]) for row in scored if bool(row["quality_pass"]))),
        "eligible_by_organ": dict(
            Counter(
                str(row["organ"])
                for row in scored
                if bool(row["quality_pass"])
                and bool(row["organ_constraints_pass"])
                and not bool(row["training_overlap"])
            )
        ),
        "candidate_by_organ": dict(Counter(str(row["organ"]) for row in candidates)),
        "complex_by_organ": dict(Counter(str(row["organ"]) for row in selected.complex_rows)),
        "random_by_organ": dict(Counter(str(row["organ"]) for row in selected.random_rows)),
        "complex_random_case_overlap": len(complex_cases & random_cases),
        "deficits": selected.deficits,
        "other_caps": {organ: rule.other_cap for organ, rule in ORGAN_RULES.items()},
    }
    (output_root / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
