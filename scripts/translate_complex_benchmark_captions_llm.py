#!/usr/bin/env python3
"""Resume-safe pathology caption translation through an OpenAI-compatible LLM API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Mapping, Sequence


CJK_RE = re.compile(r"[\u3400-\u9fff]")
SYSTEM_PROMPT = """You are a senior bilingual pathology editor. Translate English pathology image captions into concise, professional Simplified Chinese for review by pathologists.

Rules:
1. Translate faithfully. Never add, remove, strengthen, or reinterpret a diagnosis or observation.
2. Preserve uncertainty exactly: possible, suggestive of, consistent with, may indicate, and definitive statements must remain distinct.
3. Preserve negation, specimen organ, tumor type, grade, tissue structure, cell morphology, and ancillary-test terminology.
4. Use standard pathology Chinese. Preferred terms include:
   dysplasia=异型增生; atypia=异型性; pleomorphism=多形性; hyperchromatic nuclei=深染核;
   nuclear-to-cytoplasmic ratio=核质比; prominent nucleoli=核仁明显; stroma=间质;
   desmoplastic stroma=促结缔组织增生性间质; poorly differentiated=低分化;
   necrosis=坏死; inflammatory infiltrate=炎性细胞浸润; glandular structure=腺体结构.
5. Do not explain the translation. Return only valid JSON in the requested schema.
6. Translate every ordinary English prose word. You may retain only standard pathology abbreviations,
   gene/protein names, and established eponyms in parentheses; never leave partial English clauses or add translator notes.
"""


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _chat_completions_endpoint(api_base_url: str) -> str:
    base = api_base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_json_payload(content: str) -> dict[str, object]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


def _request_batch(
    rows: Sequence[Mapping[str, object]],
    *,
    endpoint: str,
    api_key: str,
    model: str,
    retries: int,
) -> tuple[list[str], dict[str, int]]:
    items = [{"id": str(index), "english": str(row["caption_en"])} for index, row in enumerate(rows)]
    user_prompt = (
        "Translate every item independently. Return exactly this JSON schema: "
        '{"translations":[{"id":"0","zh":"..."}]}. '
        "Every input id must appear exactly once.\n\nINPUT:\n"
        + json.dumps(items, ensure_ascii=False)
    )
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PathologyEditBenchmark/1.0",
        },
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.load(response)
            content = response_payload["choices"][0]["message"]["content"]
            parsed = _extract_json_payload(str(content))
            translations = parsed.get("translations")
            if not isinstance(translations, list):
                raise ValueError("response is missing translations list")
            by_id = {}
            for item in translations:
                if not isinstance(item, dict):
                    raise ValueError("translation item must be an object")
                item_id = str(item.get("id", ""))
                translated = str(item.get("zh", "")).strip()
                if item_id in by_id:
                    raise ValueError(f"duplicate translation id={item_id}")
                if len(CJK_RE.findall(translated)) < 5:
                    raise ValueError(f"translation id={item_id} contains too little Chinese")
                by_id[item_id] = translated
            expected_ids = {str(index) for index in range(len(rows))}
            if set(by_id) != expected_ids:
                raise ValueError(f"translation ids mismatch expected={expected_ids} actual={set(by_id)}")
            usage = response_payload.get("usage") or {}
            return (
                [by_id[str(index)] for index in range(len(rows))],
                {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                },
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"batch failed after {retries} attempts: {exc}") from exc
            time.sleep(min(30.0, 1.5 * (2**attempt)) + random.random())
    raise AssertionError("unreachable")


def _translate_with_split(
    rows: Sequence[Mapping[str, object]],
    *,
    endpoint: str,
    api_key: str,
    model: str,
    retries: int,
) -> tuple[list[tuple[Mapping[str, object], str]], dict[str, int]]:
    try:
        translations, usage = _request_batch(
            rows,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            retries=retries,
        )
        return list(zip(rows, translations)), usage
    except RuntimeError:
        if len(rows) == 1:
            raise
        midpoint = len(rows) // 2
        left, left_usage = _translate_with_split(
            rows[:midpoint], endpoint=endpoint, api_key=api_key, model=model, retries=retries
        )
        right, right_usage = _translate_with_split(
            rows[midpoint:], endpoint=endpoint, api_key=api_key, model=model, retries=retries
        )
        return left + right, {
            "prompt_tokens": left_usage["prompt_tokens"] + right_usage["prompt_tokens"],
            "completion_tokens": left_usage["completion_tokens"] + right_usage["completion_tokens"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--model", default="qwen-turbo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if not args.api_key_stdin:
        parser.error("--api-key-stdin is required so credentials never enter argv or files")
    api_key = sys.stdin.readline().strip()
    if not api_key:
        parser.error("API key was not provided on stdin")

    source_rows = _read_jsonl(args.input_jsonl)
    completed_rows = _read_jsonl(args.output_jsonl)
    completed = {str(row["stem"]): row for row in completed_rows}
    todo = [row for row in source_rows if str(row["stem"]) not in completed]
    if args.max_items > 0:
        todo = todo[: args.max_items]
    endpoint = _chat_completions_endpoint(args.api_base_url)
    print(
        f"model={args.model} source={len(source_rows)} completed={len(completed)} "
        f"todo_this_run={len(todo)} batch_size={args.batch_size} workers={args.workers}",
        flush=True,
    )
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    failures = []
    batches = [todo[offset : offset + args.batch_size] for offset in range(0, len(todo), args.batch_size)]
    processed = 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("a", encoding="utf-8") as output_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _translate_with_split,
                    batch,
                    endpoint=endpoint,
                    api_key=api_key,
                    model=args.model,
                    retries=args.retries,
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                processed += len(batch)
                try:
                    translated_rows, batch_usage = future.result()
                except Exception as exc:
                    failures.extend({"stem": row["stem"], "error": str(exc)} for row in batch)
                else:
                    usage["prompt_tokens"] += batch_usage["prompt_tokens"]
                    usage["completion_tokens"] += batch_usage["completion_tokens"]
                    for row, caption_zh in translated_rows:
                        output_row = {
                            **row,
                            "caption_zh": caption_zh,
                            "translation_source": f"{args.model}_cursorai_art_2026-07-13",
                        }
                        output_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                        completed[str(row["stem"])] = output_row
                    output_handle.flush()
                if processed % args.log_every < len(batch) or processed == len(todo):
                    print(
                        f"processed={processed}/{len(todo)} total_completed={len(completed)} "
                        f"failures={len(failures)} usage={usage}",
                        flush=True,
                    )
    summary = {
        "model": args.model,
        "source": len(source_rows),
        "completed": len(completed),
        "processed_this_run": len(todo),
        "failures": failures,
        "usage": usage,
    }
    summary_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    expected = min(len(source_rows), len(completed_rows) + len(todo))
    return 1 if failures or len(completed) != expected else 0


if __name__ == "__main__":
    raise SystemExit(main())
