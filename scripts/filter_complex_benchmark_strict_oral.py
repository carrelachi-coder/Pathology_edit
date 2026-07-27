#!/usr/bin/env python3
"""Filter non-oral TCGA-HNSC cases from the complex benchmark cohort.

The script keeps only unambiguous oral-cavity ``primary_site`` values in the
existing ``head_neck`` compatibility group.  In apply mode it backs up every
modified manifest, archives removed annotation-package assets, and removes the
corresponding generated sample directories.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen


STRICT_ORAL_PRIMARY_SITES = {
    "Floor of mouth",
    "Gum",
    "Lip",
    "Other and unspecified parts of mouth",
    "Other and unspecified parts of tongue",
}

METADATA_FILES = (
    "directions.csv",
    "directions.json",
    "pairs.csv",
    "pairs.json",
    "manual_review.csv",
    "summary.json",
    "validation.json",
    "manifest_hashes.json",
    "paired_directions_auto_1500.json",
    "captions_missing_zh.jsonl",
    "translations_zh_qwen_turbo_final.jsonl",
    "annotation_package/caption_manifest.csv",
    "annotation_package/caption_summary.json",
    "annotation_package/double_annotation_manifest.csv",
    "annotation_package/pair_review.csv",
    "annotation_package/patch_annotation_manifest.csv",
    "annotation_package/summary.json",
)

ANNOTATION_ASSET_DIRS = (
    "captions_en",
    "captions_zh",
    "cellvit_masks_auto",
    "images",
    "labels_primary",
    "labels_secondary",
    "tissue_masks_auto",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_gdc_primary_sites(case_ids: Iterable[str]) -> dict[str, str]:
    case_ids = sorted(set(case_ids))
    filters = {
        "op": "in",
        "content": {"field": "cases.submitter_id", "value": case_ids},
    }
    query = urlencode(
        {
            "filters": json.dumps(filters, separators=(",", ":")),
            "fields": "submitter_id,primary_site",
            "format": "JSON",
            "size": max(100, len(case_ids)),
        }
    )
    with urlopen(f"https://api.gdc.cancer.gov/cases?{query}", timeout=60) as response:
        payload = json.load(response)
    result = {
        hit["submitter_id"]: hit.get("primary_site", "")
        for hit in payload["data"]["hits"]
    }
    missing = sorted(set(case_ids) - set(result))
    if missing:
        raise RuntimeError(f"GDC did not return primary_site for: {missing}")
    return result


def _filter_rows(
    path: Path,
    key: str,
    excluded: set[str],
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, rows = _read_csv(path)
    return fieldnames, [row for row in rows if row[key] not in excluded]


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _build_caption_summary(
    caption_rows: list[dict[str, str]], package_root: Path
) -> dict[str, Any]:
    return {
        "captions": len(caption_rows),
        "captions_by_organ": _counter_dict(row["organ"] for row in caption_rows),
        "chinese_caption_files": len(list((package_root / "captions_zh").glob("*.txt"))),
        "english_caption_files": len(list((package_root / "captions_en").glob("*.txt"))),
        "translation_sources": _counter_dict(
            row["translation_source"] for row in caption_rows
        ),
    }


def _backup_metadata(dataset_root: Path, backup_root: Path) -> None:
    for relative in METADATA_FILES:
        source = dataset_root / relative
        if not source.exists():
            continue
        target = backup_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256(source) != _sha256(target):
                raise FileExistsError(f"backup already exists with different content: {target}")
            continue
        shutil.copy2(source, target)


def _archive_annotation_assets(
    package_root: Path,
    archive_root: Path,
    annotation_ids: set[str],
    pair_ids: set[str],
) -> int:
    moved = 0
    for directory in ANNOTATION_ASSET_DIRS:
        source_dir = package_root / directory
        for annotation_id in sorted(annotation_ids):
            for source in source_dir.glob(f"{annotation_id}.*"):
                target = archive_root / directory / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    source.unlink()
                else:
                    shutil.move(str(source), str(target))
                moved += 1
    for pair_id in sorted(pair_ids):
        for source in (package_root / "pair_previews").glob(f"{pair_id}.*"):
            target = archive_root / "pair_previews" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                source.unlink()
            else:
                shutil.move(str(source), str(target))
            moved += 1
    return moved


def _remove_generated_samples(generated_roots: Iterable[Path], sample_ids: set[str]) -> dict[str, int]:
    removed: dict[str, int] = {}
    for root in generated_roots:
        count = 0
        if not root.is_dir():
            removed[str(root)] = count
            continue
        for model_root in root.iterdir():
            if not model_root.is_dir() or model_root.name.startswith("_"):
                continue
            organ_root = model_root / "head_neck"
            for sample_id in sample_ids:
                sample_root = organ_root / sample_id
                if sample_root.is_dir():
                    shutil.rmtree(sample_root)
                    count += 1
        removed[str(root)] = count
    return removed


def _filter_jsonl(path: Path, excluded_pair_ids: set[str]) -> int:
    if not path.exists():
        return 0
    retained = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("pair_id") in excluded_pair_ids:
            removed += 1
        else:
            retained.append(json.dumps(payload, ensure_ascii=False))
    _atomic_write_text(path, "\n".join(retained) + ("\n" if retained else ""))
    return removed


def filter_dataset(
    dataset_root: Path,
    generated_roots: list[Path],
    apply: bool,
    filter_name: str,
) -> dict[str, Any]:
    directions_fields, direction_rows = _read_csv(dataset_root / "directions.csv")
    pairs_fields, pair_rows = _read_csv(dataset_root / "pairs.csv")
    head_neck_cases = {
        row["case_id"] for row in pair_rows if row["organ"] == "head_neck"
    }
    primary_sites = _query_gdc_primary_sites(head_neck_cases)
    excluded_cases = {
        case_id
        for case_id, site in primary_sites.items()
        if site not in STRICT_ORAL_PRIMARY_SITES
    }
    excluded_pair_ids = {
        row["pair_id"]
        for row in pair_rows
        if row["organ"] == "head_neck" and row["case_id"] in excluded_cases
    }
    excluded_sample_ids = {
        row["sample_id"]
        for row in direction_rows
        if row["pair_id"] in excluded_pair_ids
    }
    excluded_annotation_ids = {
        f"{pair_id}-{side}" for pair_id in excluded_pair_ids for side in ("a", "b")
    }
    retained_pairs = [row for row in pair_rows if row["pair_id"] not in excluded_pair_ids]
    retained_directions = [
        row for row in direction_rows if row["sample_id"] not in excluded_sample_ids
    ]
    report: dict[str, Any] = {
        "apply": apply,
        "filter_name": filter_name,
        "strict_oral_primary_sites": sorted(STRICT_ORAL_PRIMARY_SITES),
        "head_neck_cases_before": len(head_neck_cases),
        "excluded_cases": len(excluded_cases),
        "excluded_pairs": len(excluded_pair_ids),
        "excluded_directions": len(excluded_sample_ids),
        "retained_pairs": len(retained_pairs),
        "retained_directions": len(retained_directions),
        "retained_head_neck_pairs": sum(
            row["organ"] == "head_neck" for row in retained_pairs
        ),
        "retained_head_neck_directions": sum(
            row["organ"] == "head_neck" for row in retained_directions
        ),
        "primary_site_counts": _counter_dict(primary_sites.values()),
    }
    if not apply:
        return report

    audit_root = dataset_root / "cohort_filters" / filter_name
    backup_root = audit_root / "metadata_backup"
    _backup_metadata(dataset_root, backup_root)
    audit_root.mkdir(parents=True, exist_ok=True)

    with (audit_root / "case_primary_sites.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "primary_site", "decision"],
            lineterminator="\n",
        )
        writer.writeheader()
        for case_id in sorted(primary_sites):
            writer.writerow(
                {
                    "case_id": case_id,
                    "primary_site": primary_sites[case_id],
                    "decision": "exclude" if case_id in excluded_cases else "keep",
                }
            )
    for filename, values in (
        ("excluded_cases.txt", excluded_cases),
        ("excluded_pairs.txt", excluded_pair_ids),
        ("excluded_sample_ids.txt", excluded_sample_ids),
        ("excluded_annotation_ids.txt", excluded_annotation_ids),
    ):
        _atomic_write_text(audit_root / filename, "\n".join(sorted(values)) + "\n")

    _write_csv(dataset_root / "directions.csv", directions_fields, retained_directions)
    _write_csv(dataset_root / "pairs.csv", pairs_fields, retained_pairs)
    for relative, key, excluded in (
        ("manual_review.csv", "pair_id", excluded_pair_ids),
        ("annotation_package/caption_manifest.csv", "pair_id", excluded_pair_ids),
        ("annotation_package/double_annotation_manifest.csv", "pair_id", excluded_pair_ids),
        ("annotation_package/pair_review.csv", "pair_id", excluded_pair_ids),
        ("annotation_package/patch_annotation_manifest.csv", "pair_id", excluded_pair_ids),
    ):
        fields, rows = _filter_rows(dataset_root / relative, key, excluded)
        _write_csv(dataset_root / relative, fields, rows)

    directions_json_path = dataset_root / "directions.json"
    directions_json = json.loads(directions_json_path.read_text())
    _write_json(
        directions_json_path,
        [row for row in directions_json if row["sample_id"] not in excluded_sample_ids],
    )
    pairs_json_path = dataset_root / "pairs.json"
    pairs_json = json.loads(pairs_json_path.read_text())
    _write_json(
        pairs_json_path,
        [row for row in pairs_json if row["pair_id"] not in excluded_pair_ids],
    )

    jsonl_removed = {}
    for relative in ("captions_missing_zh.jsonl", "translations_zh_qwen_turbo_final.jsonl"):
        jsonl_removed[relative] = _filter_jsonl(
            dataset_root / relative, excluded_pair_ids
        )

    package_root = dataset_root / "annotation_package"
    archived_assets = _archive_annotation_assets(
        package_root,
        audit_root / "removed_annotation_assets",
        excluded_annotation_ids,
        excluded_pair_ids,
    )

    _, caption_rows = _read_csv(package_root / "caption_manifest.csv")
    _, patch_rows = _read_csv(package_root / "patch_annotation_manifest.csv")
    _, pair_review_rows = _read_csv(package_root / "pair_review.csv")
    _, double_rows = _read_csv(package_root / "double_annotation_manifest.csv")
    caption_summary = _build_caption_summary(caption_rows, package_root)
    package_summary = json.loads((package_root / "summary.json").read_text())
    package_summary.update(
        {
            "captions": caption_summary,
            "double_annotation_patches": len(double_rows),
            "double_annotations_by_organ": _counter_dict(
                row["organ"] for row in double_rows
            ),
            "pairs": len(pair_review_rows),
            "patches": len(patch_rows),
            "patches_by_organ": _counter_dict(row["organ"] for row in patch_rows),
            "unique_stems": len({row["stem"] for row in patch_rows}),
        }
    )
    _write_json(package_root / "caption_summary.json", caption_summary)
    _write_json(package_root / "summary.json", package_summary)

    summary_path = dataset_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    pairs_per_wsi = Counter(row["wsi"] for row in retained_pairs)
    summary.update(
        {
            "annotation_package": package_summary,
            "cohort_filter": {
                "name": filter_name,
                "clinical_scope": "strict Oral within the head_neck compatibility label",
                "excluded_cases": len(excluded_cases),
                "excluded_pairs": len(excluded_pair_ids),
                "excluded_directions": len(excluded_sample_ids),
                "primary_site_source": "NCI GDC cases API",
                "audit_path": str(audit_root),
            },
            "selected_directions": len(retained_directions),
            "selected_pairs": len(retained_pairs),
            "selected_pairs_by_organ": _counter_dict(
                row["organ"] for row in retained_pairs
            ),
            "selected_pairs_per_wsi_histogram": {
                str(key): value
                for key, value in sorted(Counter(pairs_per_wsi.values()).items())
            },
            "selected_wsis": len(pairs_per_wsi),
        }
    )
    _write_json(summary_path, summary)

    canonical_path = dataset_root / "paired_directions_auto_1500.json"
    canonical = json.loads(canonical_path.read_text())
    records = [
        row for row in canonical["records"] if row["sample_id"] not in excluded_sample_ids
    ]
    canonical["records"] = records
    canonical["provenance"]["cohort_filter"] = {
        "name": filter_name,
        "strict_oral_primary_sites": sorted(STRICT_ORAL_PRIMARY_SITES),
        "excluded_cases_path": str(audit_root / "excluded_cases.txt"),
        "excluded_pairs_path": str(audit_root / "excluded_pairs.txt"),
    }
    canonical["provenance"]["patch_manifest_sha256"] = _sha256(
        package_root / "patch_annotation_manifest.csv"
    )
    canonical["provenance"]["pair_review_sha256"] = _sha256(
        package_root / "pair_review.csv"
    )
    canonical["summary"] = {
        "directions": len(records),
        "pairs": len({row["pair_id"] for row in records}),
        "unique_target_images": len({row["target_image"] for row in records}),
        "unique_wsis": len({row["wsi_id"] for row in records}),
        "organ_counts": _counter_dict(row["organ"] for row in records),
    }
    _write_json(canonical_path, canonical)
    strict_canonical_path = dataset_root / "paired_directions_strict_oral_1454.json"
    _write_json(strict_canonical_path, canonical)

    validation_path = dataset_root / "validation.json"
    validation = json.loads(validation_path.read_text())
    validation.update(
        {
            "annotation_patches": len(patch_rows),
            "caption_entries": len(caption_rows),
            "caption_unique_annotation_ids": len(
                {row["annotation_id"] for row in caption_rows}
            ),
            "directions": len(retained_directions),
            "double_annotation_patches": len(double_rows),
            "double_annotations_by_organ": _counter_dict(
                row["organ"] for row in double_rows
            ),
            "organ_pairs": _counter_dict(row["organ"] for row in retained_pairs),
            "pair_previews": len(list((package_root / "pair_previews").glob("*"))),
            "pairs": len(retained_pairs),
            "unique_patches": len({row["stem"] for row in patch_rows}),
            "unique_wsi": len({row["wsi"] for row in retained_pairs}),
        }
    )
    _write_json(validation_path, validation)

    hashes_path = dataset_root / "manifest_hashes.json"
    hashes = json.loads(hashes_path.read_text())
    hashes = {relative: _sha256(dataset_root / relative) for relative in hashes}
    _write_json(hashes_path, hashes)

    generated_removed = _remove_generated_samples(generated_roots, excluded_sample_ids)
    report.update(
        {
            "audit_root": str(audit_root),
            "archived_annotation_assets": archived_assets,
            "jsonl_rows_removed": jsonl_removed,
            "generated_sample_directories_removed": generated_removed,
            "canonical_manifest": str(canonical_path),
            "strict_canonical_manifest": str(strict_canonical_path),
        }
    )
    _write_json(audit_root / "filter_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--generated-root", type=Path, action="append", default=[],
        help="Generation root whose model/organ/sample directories should be cleaned.",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--filter-name", default=f"strict_oral_{date.today():%Y%m%d}"
    )
    args = parser.parse_args()
    report = filter_dataset(
        args.dataset_root.resolve(),
        [path.resolve() for path in args.generated_root],
        args.apply,
        args.filter_name,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
