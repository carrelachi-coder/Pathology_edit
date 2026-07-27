#!/usr/bin/env python3
"""Validate bilingual captions installed in the complex benchmark doctor package."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


CJK_RE = re.compile(r"[\u3400-\u9fff]")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_source_text(path: Path) -> str:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return payload.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("caption", payload, 0, len(payload), "unsupported encoding")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_caption_package(dataset_root: Path) -> dict[str, object]:
    package_root = dataset_root / "annotation_package"
    caption_rows = _read_csv(package_root / "caption_manifest.csv")
    patch_rows = _read_csv(package_root / "patch_annotation_manifest.csv")
    pair_rows = _read_csv(package_root / "pair_review.csv")
    caption_by_id = {row["annotation_id"]: row for row in caption_rows}
    if len(caption_by_id) != len(caption_rows):
        raise ValueError("caption manifest contains duplicate annotation IDs")
    patch_by_id = {row["annotation_id"]: row for row in patch_rows}
    if set(caption_by_id) != set(patch_by_id):
        raise ValueError("caption and patch annotation ID sets differ")

    minimum_cjk = None
    source_counts = Counter()
    for annotation_id, caption_row in caption_by_id.items():
        patch_row = patch_by_id[annotation_id]
        en_path = package_root / caption_row["caption_en_relpath"]
        zh_path = package_root / caption_row["caption_zh_relpath"]
        if not en_path.is_file() or not zh_path.is_file():
            raise FileNotFoundError(f"caption files missing for {annotation_id}")
        caption_en = en_path.read_text(encoding="utf-8").strip()
        caption_zh = zh_path.read_text(encoding="utf-8").strip()
        if caption_en != caption_row["caption_en"] or caption_zh != caption_row["caption_zh"]:
            raise ValueError(f"caption file and manifest text differ for {annotation_id}")
        if caption_en != _read_source_text(Path(caption_row["source_text_path"])):
            raise ValueError(f"packaged English differs from source for {annotation_id}")
        if patch_row["caption_en_relpath"] != caption_row["caption_en_relpath"]:
            raise ValueError(f"English caption link differs for {annotation_id}")
        if patch_row["caption_zh_relpath"] != caption_row["caption_zh_relpath"]:
            raise ValueError(f"Chinese caption link differs for {annotation_id}")
        cjk_count = len(CJK_RE.findall(caption_zh))
        minimum_cjk = cjk_count if minimum_cjk is None else min(minimum_cjk, cjk_count)
        if cjk_count < 5:
            raise ValueError(f"Chinese caption contains too little CJK text for {annotation_id}")
        source_counts[caption_row["translation_source"]] += 1

    pair_links = 0
    for pair_row in pair_rows:
        for side in ("a", "b"):
            annotation_id = pair_row[f"{side}_annotation_id"]
            caption_row = caption_by_id[annotation_id]
            for language in ("en", "zh"):
                key = f"{side}_caption_{language}_relpath"
                if pair_row[key] != caption_row[f"caption_{language}_relpath"]:
                    raise ValueError(f"pair caption link differs for {pair_row['pair_id']} side {side}")
                pair_links += 1

    hashes = json.loads((dataset_root / "manifest_hashes.json").read_text())
    for relative_path, expected_hash in hashes.items():
        if _sha256(dataset_root / relative_path) != expected_hash:
            raise ValueError(f"manifest hash mismatch: {relative_path}")
    if any("google" in source.lower() for source in source_counts):
        raise ValueError("temporary Google translations were included")

    result = {
        "bilingual_caption_status": "pass",
        "caption_entries": len(caption_rows),
        "caption_unique_annotation_ids": len(caption_by_id),
        "caption_english_files": len(list((package_root / "captions_en").glob("*.txt"))),
        "caption_chinese_files": len(list((package_root / "captions_zh").glob("*.txt"))),
        "caption_minimum_cjk_characters": minimum_cjk or 0,
        "caption_files_match_manifest": True,
        "caption_english_matches_original_sources": True,
        "caption_patch_manifest_links_valid": True,
        "caption_pair_review_links_valid": True,
        "caption_pair_links_verified": pair_links,
        "caption_translation_sources": dict(sorted(source_counts.items())),
        "manifest_hashes_verified_present": len(hashes),
    }
    validation_path = dataset_root / "validation.json"
    validation = json.loads(validation_path.read_text())
    validation.update(result)
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_caption_package(args.dataset_root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
