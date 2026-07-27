#!/usr/bin/env python3
"""Resume-safe English-to-Chinese translation for exported benchmark captions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request


GOOGLE_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _translate_google(text: str, *, retries: int) -> str:
    query = urllib.parse.urlencode(
        {
            "client": "gtx",
            "sl": "en",
            "tl": "zh-CN",
            "dt": "t",
            "q": text,
        }
    )
    request = urllib.request.Request(
        f"{GOOGLE_TRANSLATE_URL}?{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            translated = "".join(part[0] for part in payload[0] if part and part[0]).strip()
            if len(CJK_RE.findall(translated)) < 5:
                raise ValueError(f"translation contains too little Chinese: {translated[:100]}")
            return translated
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"translation failed after {retries} attempts: {exc}") from exc
            time.sleep(min(30.0, 1.5 * (2**attempt)) + random.random())
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    source_rows = _read_jsonl(args.input_jsonl)
    completed_rows = _read_jsonl(args.output_jsonl)
    completed = {str(row["stem"]): row for row in completed_rows}
    todo = [row for row in source_rows if str(row["stem"]) not in completed]
    print(f"source={len(source_rows)} completed={len(completed)} todo={len(todo)}", flush=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("a", encoding="utf-8") as output_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_translate_google, str(row["caption_en"]), retries=args.retries): row
                for row in todo
            }
            failures = []
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    caption_zh = future.result()
                except Exception as exc:
                    failures.append({"stem": row["stem"], "error": str(exc)})
                    continue
                translated = {
                    **row,
                    "caption_zh": caption_zh,
                    "translation_source": "google_translate_en_zh_2026-07-13",
                }
                output_handle.write(json.dumps(translated, ensure_ascii=False) + "\n")
                output_handle.flush()
                completed[str(row["stem"])] = translated
                if index % args.log_every == 0 or index == len(todo):
                    print(
                        f"processed={index}/{len(todo)} total_completed={len(completed)} failures={len(failures)}",
                        flush=True,
                    )
    summary = {
        "source": len(source_rows),
        "completed": len(completed),
        "failures": failures,
    }
    summary_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures or len(completed) != len(source_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
