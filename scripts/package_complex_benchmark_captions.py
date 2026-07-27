#!/usr/bin/env python3
"""Export and install bilingual captions for a paired benchmark annotation package."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ORGAN_ZH = {
    "breast": "乳腺",
    "colorectal": "结直肠",
    "head_neck": "头颈部",
    "lung": "肺",
    "prostate": "前列腺",
    "skin": "皮肤",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _read_caption_text(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = payload.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
        if text:
            return text, encoding
    raise UnicodeDecodeError("utf-8", payload, 0, len(payload), f"cannot decode caption: {path}")


def _caption_source_rows(dataset_root: Path) -> list[dict[str, object]]:
    pairs = _read_csv(dataset_root / "pairs.csv")
    rows: list[dict[str, object]] = []
    seen_annotations: set[str] = set()
    seen_stems: set[str] = set()
    for pair in pairs:
        for side in ("a", "b"):
            annotation_id = f"{pair['pair_id']}-{side}"
            stem = pair[f"{side}_stem"]
            text_path = Path(pair[f"{side}_text_path"])
            if annotation_id in seen_annotations or stem in seen_stems:
                raise RuntimeError(f"duplicate caption row: annotation_id={annotation_id} stem={stem}")
            if not text_path.is_file():
                raise FileNotFoundError(f"missing English caption: {text_path}")
            caption_en, source_text_encoding = _read_caption_text(text_path)
            organ = pair["organ"]
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "pair_id": pair["pair_id"],
                    "side": side,
                    "organ": organ,
                    "organ_zh": ORGAN_ZH.get(organ, organ),
                    "stem": stem,
                    "source_text_path": str(text_path),
                    "source_text_encoding": source_text_encoding,
                    "caption_en": caption_en,
                }
            )
            seen_annotations.add(annotation_id)
            seen_stems.add(stem)
    return rows


def _translation_stem(row: Mapping[str, object]) -> str:
    recorded = str(
        row.get("stem")
        or row.get("base_name")
        or row.get("txt_name")
        or row.get("filename")
        or ""
    ).strip()
    lowered = recorded.lower()
    for suffix in (".txt", ".png"):
        if lowered.endswith(suffix):
            return recorded[: -len(suffix)]
    return recorded


def _load_translations(paths: Sequence[Path]) -> dict[str, dict[str, str]]:
    translations: dict[str, dict[str, str]] = {}
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        else:
            rows = _read_csv(path)
        for row in rows:
            stem = _translation_stem(row)
            caption_zh = str(row.get("caption_zh", "")).strip()
            if not stem or not caption_zh:
                continue
            previous = translations.get(stem)
            if previous and previous["caption_zh"] != caption_zh:
                raise RuntimeError(f"conflicting Chinese translations for stem={stem}")
            translations[stem] = {
                "caption_zh": caption_zh,
                "translation_source": str(
                    row.get("translation_source") or f"existing:{path.name}"
                ),
            }
    return translations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_hashes(dataset_root: Path) -> None:
    hash_path = dataset_root / "manifest_hashes.json"
    hashes = json.loads(hash_path.read_text()) if hash_path.exists() else {}
    relative_paths = {Path(path) for path in hashes}
    relative_paths.update(
        {
            Path("annotation_package/caption_manifest.csv"),
            Path("annotation_package/caption_summary.json"),
            Path("annotation_package/pair_review.csv"),
            Path("annotation_package/patch_annotation_manifest.csv"),
        }
    )
    for relative_path in sorted(relative_paths):
        path = dataset_root / relative_path
        if path.exists():
            hashes[str(relative_path)] = _sha256(path)
    _write_json(hash_path, hashes)


def install_caption_package(
    dataset_root: Path,
    source_rows: Sequence[Mapping[str, object]],
    translations: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    package_root = dataset_root / "annotation_package"
    package_manifest_path = package_root / "patch_annotation_manifest.csv"
    patch_rows = _read_csv(package_manifest_path)
    patch_by_annotation = {row["annotation_id"]: row for row in patch_rows}
    if set(patch_by_annotation) != {str(row["annotation_id"]) for row in source_rows}:
        raise RuntimeError("caption sources do not match patch_annotation_manifest.csv")

    missing = sorted(str(row["stem"]) for row in source_rows if str(row["stem"]) not in translations)
    if missing:
        raise RuntimeError(f"missing Chinese translations: count={len(missing)} examples={missing[:5]}")

    caption_rows: list[dict[str, object]] = []
    for source_row in source_rows:
        annotation_id = str(source_row["annotation_id"])
        translation = translations[str(source_row["stem"])]
        caption_en = str(source_row["caption_en"]).strip()
        caption_zh = str(translation["caption_zh"]).strip()
        en_path = package_root / "captions_en" / f"{annotation_id}.txt"
        zh_path = package_root / "captions_zh" / f"{annotation_id}.txt"
        en_path.parent.mkdir(parents=True, exist_ok=True)
        zh_path.parent.mkdir(parents=True, exist_ok=True)
        en_path.write_text(caption_en + "\n", encoding="utf-8")
        zh_path.write_text(caption_zh + "\n", encoding="utf-8")
        caption_row = {
            **source_row,
            "caption_en_path": str(en_path),
            "caption_en_relpath": str(en_path.relative_to(package_root)),
            "caption_zh_path": str(zh_path),
            "caption_zh_relpath": str(zh_path.relative_to(package_root)),
            "translation_source": translation["translation_source"],
            "caption_zh": caption_zh,
        }
        caption_rows.append(caption_row)
        patch_by_annotation[annotation_id].update(
            {
                "source_text_path": source_row["source_text_path"],
                "caption_en_path": str(en_path),
                "caption_en_relpath": str(en_path.relative_to(package_root)),
                "caption_zh_path": str(zh_path),
                "caption_zh_relpath": str(zh_path.relative_to(package_root)),
                "caption_translation_source": translation["translation_source"],
            }
        )

    _write_csv(package_root / "caption_manifest.csv", caption_rows)
    _write_csv(package_manifest_path, [patch_by_annotation[row["annotation_id"]] for row in patch_rows])
    caption_by_annotation = {str(row["annotation_id"]): row for row in caption_rows}
    pair_review_path = package_root / "pair_review.csv"
    pair_review_rows = _read_csv(pair_review_path)
    for pair_row in pair_review_rows:
        for side in ("a", "b"):
            caption_row = caption_by_annotation[str(pair_row[f"{side}_annotation_id"])]
            pair_row[f"{side}_caption_en_relpath"] = caption_row["caption_en_relpath"]
            pair_row[f"{side}_caption_zh_relpath"] = caption_row["caption_zh_relpath"]
    _write_csv(pair_review_path, pair_review_rows)
    source_counts = Counter(str(row["translation_source"]) for row in caption_rows)
    summary = {
        "captions": len(caption_rows),
        "english_caption_files": len(list((package_root / "captions_en").glob("*.txt"))),
        "chinese_caption_files": len(list((package_root / "captions_zh").glob("*.txt"))),
        "captions_by_organ": dict(sorted(Counter(str(row["organ"]) for row in caption_rows).items())),
        "translation_sources": dict(sorted(source_counts.items())),
    }
    _write_json(package_root / "caption_summary.json", summary)

    readme_path = package_root / "README_zh.txt"
    readme = readme_path.read_text(encoding="utf-8")
    marker = "captions_en 和 captions_zh"
    if marker not in readme:
        readme += (
            "captions_en 和 captions_zh 分别保存英文原注释和中文翻译，文件名与 annotation_id 一致。\n"
            "caption_manifest.csv 同时提供绝对路径、相对路径、双语文本和翻译来源。\n"
        )
        readme_path.write_text(readme, encoding="utf-8")

    package_summary_path = package_root / "summary.json"
    package_summary = json.loads(package_summary_path.read_text())
    package_summary["captions"] = summary
    _write_json(package_summary_path, package_summary)
    dataset_summary_path = dataset_root / "summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text())
    dataset_summary.setdefault("annotation_package", {})["captions"] = summary
    _write_json(dataset_summary_path, dataset_summary)
    _update_hashes(dataset_root)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--translation-file", type=Path, action="append", default=[])
    parser.add_argument("--export-missing-jsonl", type=Path, default=None)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    source_rows = _caption_source_rows(args.dataset_root)
    translations = _load_translations(args.translation_file)
    missing_rows = [row for row in source_rows if str(row["stem"]) not in translations]
    if args.export_missing_jsonl:
        _write_jsonl(args.export_missing_jsonl, missing_rows)
    result: dict[str, object] = {
        "source_captions": len(source_rows),
        "existing_translations": len(source_rows) - len(missing_rows),
        "missing_translations": len(missing_rows),
    }
    if args.install:
        result["installed"] = install_caption_package(
            args.dataset_root,
            source_rows,
            translations,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
