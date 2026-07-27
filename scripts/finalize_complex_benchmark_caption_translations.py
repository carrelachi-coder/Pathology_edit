#!/usr/bin/env python3
"""Merge caption translation QA retries and apply auditable terminology fixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable, Mapping


CJK_RE = re.compile(r"[\u3400-\u9fff]")
LOWERCASE_WORD_RE = re.compile(r"\b[a-z]{4,}\b")
ALLOWED_LOWERCASE_TERMS = {"hackberry"}
TEXT_REPLACEMENTS = (
    ("发育不良", "异型增生"),
    ("特殊类型未定（NST）", "非特殊类型（NST）"),
    ("特殊类型未定 (NST)", "非特殊类型（NST）"),
    ("沿肺泡壁呈铺路石样（lepidic）生长", "沿肺泡壁呈附壁型生长"),
    (" obliteration ", "闭塞"),
    ("的 presence ", "的存在"),
    ("但 findings ", "但上述发现"),
    (" findings ", "上述发现"),
    ("（lepidic）", ""),
    ("（lepidic growth）", ""),
    (" irregular ", "不规则"),
    (" indicative of ", "提示"),
    (" poorly formed ", "形成不良"),
    (" collectively ", "共同"),
    (" noted ", "可见"),
    (" structures", "结构"),
    (" intervening ", "间隔"),
    (" alteration", "改变"),
    (" obscured", "模糊"),
    (" discern ", "判断"),
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finalize_rows(
    base_rows: Iterable[Mapping[str, object]],
    override_rows: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    overrides = {str(row["stem"]): dict(row) for row in override_rows}
    finalized = []
    stats = {"qa_overrides": 0, "terminology_corrected": 0}
    seen = set()
    for base_row in base_rows:
        stem = str(base_row["stem"])
        if stem in seen:
            raise ValueError(f"duplicate base stem: {stem}")
        seen.add(stem)
        row = dict(overrides.get(stem, base_row))
        if stem in overrides:
            stats["qa_overrides"] += 1
            row["translation_source"] = f"{row['translation_source']}_qa_retry"
        caption = str(row["caption_zh"]).strip()
        original_caption = caption
        for source, target in TEXT_REPLACEMENTS:
            caption = caption.replace(source, target)
        caption = re.sub(r"\s+([，。；：])", r"\1", caption)
        row["caption_zh"] = caption
        if caption != original_caption:
            stats["terminology_corrected"] += 1
            row["translation_source"] = f"{row['translation_source']}+terminology_qa"
        if len(CJK_RE.findall(caption)) < 5:
            raise ValueError(f"caption contains too little Chinese: {stem}")
        disallowed = set(LOWERCASE_WORD_RE.findall(caption)) - ALLOWED_LOWERCASE_TERMS
        if disallowed:
            raise ValueError(f"untranslated lowercase words for {stem}: {sorted(disallowed)}")
        if "发育不良" in caption:
            raise ValueError(f"unfixed dysplasia term: {stem}")
        finalized.append(row)
    extra = set(overrides) - seen
    if extra:
        raise ValueError(f"override stems are absent from base rows: {sorted(extra)[:5]}")
    stats["rows"] = len(finalized)
    return finalized, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--override-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    finalized, stats = finalize_rows(
        _read_jsonl(args.base_jsonl),
        _read_jsonl(args.override_jsonl),
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in finalized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json")
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
