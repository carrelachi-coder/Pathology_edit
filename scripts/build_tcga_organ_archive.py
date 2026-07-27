#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from segmentator.patch_selection import PROJECT_TO_ORGAN, organ_from_project, parse_tcga_patch_name


GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
EXPECTED_COUNTS = {
    "breast": 69045,
    "prostate": 15174,
    "colorectal": 24806,
    "lung": 44445,
    "skin": 5365,
    "head_neck": 37202,
}


def _fetch_tcga_case_projects() -> dict[str, str]:
    filters = {"op": "=", "content": {"field": "project.program.name", "value": "TCGA"}}
    query = urllib.parse.urlencode(
        {
            "filters": json.dumps(filters, separators=(",", ":")),
            "fields": "submitter_id,project.project_id",
            "size": "12000",
        }
    )
    with urllib.request.urlopen(f"{GDC_CASES_URL}?{query}", timeout=120) as response:
        payload = json.load(response)
    return {
        str(hit["submitter_id"]).upper(): str(hit["project"]["project_id"]).upper()
        for hit in payload["data"]["hits"]
        if str(hit.get("project", {}).get("project_id", "")).upper() in PROJECT_TO_ORGAN
    }


def _load_or_fetch_case_projects(cache_path: Path) -> dict[str, str]:
    if cache_path.exists():
        return {str(k).upper(): str(v).upper() for k, v in json.loads(cache_path.read_text()).items()}
    mapping = _fetch_tcga_case_projects()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    return mapping


def _ensure_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_ino != destination.stat().st_ino or source.stat().st_dev != destination.stat().st_dev:
            raise FileExistsError(f"destination is not the expected hard link: {destination}")
        return
    os.link(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Group paired TCGA PNG/TXT patches into six organ folders.")
    parser.add_argument("--source-dir", type=Path, default=Path("/data1/zhao/wqx/patches_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data1/zhao/wqx/patches_all_by_organ"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--case-project-cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--no-links", action="store_true")
    parser.add_argument("--allow-count-drift", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest or output_dir / "organ_manifest.csv"
    cache_path = args.case_project_cache or output_dir / "tcga_case_projects.json"

    png_paths = sorted(source_dir.glob("*.png"))
    txt_stems = {path.stem for path in source_dir.glob("*.txt")}
    png_stems = {path.stem for path in png_paths}
    missing_txt = sorted(png_stems - txt_stems)
    extra_txt = sorted(txt_stems - png_stems)
    if missing_txt or extra_txt:
        raise RuntimeError(f"PNG/TXT stem mismatch: missing_txt={len(missing_txt)} extra_txt={len(extra_txt)}")

    case_projects = _load_or_fetch_case_projects(cache_path)
    rows: list[dict[str, object]] = []
    counts = {organ: 0 for organ in EXPECTED_COUNTS}
    for image_path in png_paths:
        parsed = parse_tcga_patch_name(image_path.name)
        project_id = case_projects.get(parsed.case_id)
        if project_id is None:
            raise KeyError(f"GDC project missing for {parsed.case_id}")
        organ = organ_from_project(project_id)
        counts[organ] += 1
        rows.append(
            {
                "filename": image_path.name,
                "stem": image_path.stem,
                "case_id": parsed.case_id,
                "wsi": parsed.wsi,
                "x": parsed.x,
                "y": parsed.y,
                "project_id": project_id,
                "organ": organ,
                "image_path": str(image_path),
                "text_path": str(source_dir / f"{image_path.stem}.txt"),
                "organ_image_path": str(output_dir / organ / "images" / image_path.name),
                "organ_text_path": str(output_dir / organ / "txts" / f"{image_path.stem}.txt"),
            }
        )

    if not args.allow_count_drift and counts != EXPECTED_COUNTS:
        raise RuntimeError(f"unexpected organ counts: expected={EXPECTED_COUNTS} actual={counts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_links:
        tasks = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for row in rows:
                tasks.append(executor.submit(_ensure_hardlink, Path(str(row["image_path"])), Path(str(row["organ_image_path"]))))
                tasks.append(executor.submit(_ensure_hardlink, Path(str(row["text_path"])), Path(str(row["organ_text_path"]))))
            for index, task in enumerate(tasks, start=1):
                task.result()
                if index % 20000 == 0 or index == len(tasks):
                    print(f"hard links {index}/{len(tasks)}", flush=True)

    fields = list(rows[0]) if rows else []
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "total": len(rows),
        "counts": counts,
        "missing_txt": len(missing_txt),
        "extra_txt": len(extra_txt),
        "hard_links_created": not args.no_links,
    }
    (output_dir / "archive_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
