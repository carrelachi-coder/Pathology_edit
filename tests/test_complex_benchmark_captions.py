from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.package_complex_benchmark_captions import (
    _caption_source_rows,
    _load_translations,
    _read_caption_text,
    install_caption_package,
)
from scripts.finalize_complex_benchmark_caption_translations import finalize_rows
from scripts.validate_complex_benchmark_caption_package import validate_caption_package
from scripts.translate_complex_benchmark_captions import CJK_RE
from scripts.translate_complex_benchmark_captions_llm import (
    _chat_completions_endpoint,
    _extract_json_payload,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class CaptionPackageTests(unittest.TestCase):
    def test_finalizer_merges_retry_and_repairs_pathology_terms(self):
        base = [
            {
                "stem": "a",
                "caption_en": "Dysplasia with poorly formed glands.",
                "caption_zh": "可见发育不良伴 poorly formed 腺体。",
                "translation_source": "qwen",
            }
        ]
        retry = [
            {
                **base[0],
                "caption_zh": "可见发育不良，提示 poorly formed 腺体。",
            }
        ]

        finalized, stats = finalize_rows(base, retry)

        self.assertEqual(finalized[0]["caption_zh"], "可见异型增生，提示形成不良腺体。")
        self.assertEqual(stats["qa_overrides"], 1)
        self.assertEqual(stats["terminology_corrected"], 1)

    def test_llm_endpoint_accepts_root_or_v1_base_url(self):
        self.assertEqual(
            _chat_completions_endpoint("https://api.example.test/"),
            "https://api.example.test/v1/chat/completions",
        )
        self.assertEqual(
            _chat_completions_endpoint("https://api.example.test/v1"),
            "https://api.example.test/v1/chat/completions",
        )

    def test_llm_json_parser_accepts_fenced_json(self):
        payload = _extract_json_payload(
            '```json\n{"translations":[{"id":"0","zh":"可见肿瘤细胞。"}]}\n```'
        )

        self.assertEqual(payload["translations"][0]["zh"], "可见肿瘤细胞。")

    def test_caption_reader_falls_back_to_cp1252_without_dropping_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "caption.txt"
            path.write_bytes("Gland–stroma interface.".encode("cp1252"))

            text, encoding = _read_caption_text(path)

            self.assertEqual(text, "Gland–stroma interface.")
            self.assertEqual(encoding, "cp1252")

    def test_exports_sources_and_installs_portable_bilingual_captions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package_root = root / "annotation_package"
            text_a = root / "a.txt"
            text_b = root / "b.txt"
            text_a.write_text("Tumor cells are present.\n")
            text_b.write_text("The stroma is fibrotic.\n")
            write_csv(
                root / "pairs.csv",
                [
                    {
                        "pair_id": "breast-0001",
                        "organ": "breast",
                        "a_stem": "a",
                        "a_text_path": str(text_a),
                        "b_stem": "b",
                        "b_text_path": str(text_b),
                    }
                ],
            )
            write_csv(
                package_root / "patch_annotation_manifest.csv",
                [
                    {"annotation_id": "breast-0001-a", "stem": "a"},
                    {"annotation_id": "breast-0001-b", "stem": "b"},
                ],
            )
            write_csv(
                package_root / "pair_review.csv",
                [
                    {
                        "pair_id": "breast-0001",
                        "a_annotation_id": "breast-0001-a",
                        "b_annotation_id": "breast-0001-b",
                    }
                ],
            )
            (package_root / "README_zh.txt").write_text("说明。\n")
            (package_root / "summary.json").write_text("{}\n")
            (root / "summary.json").write_text('{"annotation_package": {}}\n')
            (root / "validation.json").write_text('{"status": "pass"}\n')

            source_rows = _caption_source_rows(root)
            summary = install_caption_package(
                root,
                source_rows,
                {
                    "a": {"caption_zh": "可见肿瘤细胞。", "translation_source": "test"},
                    "b": {"caption_zh": "间质纤维化。", "translation_source": "test"},
                },
            )

            self.assertEqual(summary["captions"], 2)
            self.assertEqual((package_root / "captions_en/breast-0001-a.txt").read_text().strip(), "Tumor cells are present.")
            self.assertEqual((package_root / "captions_zh/breast-0001-b.txt").read_text().strip(), "间质纤维化。")
            manifest = list(
                csv.DictReader(
                    (package_root / "caption_manifest.csv").open(encoding="utf-8-sig")
                )
            )
            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest[0]["caption_en_relpath"], "captions_en/breast-0001-a.txt")
            self.assertEqual(manifest[0]["caption_zh_relpath"], "captions_zh/breast-0001-a.txt")
            pair_review = list(
                csv.DictReader((package_root / "pair_review.csv").open(encoding="utf-8-sig"))
            )
            self.assertEqual(
                pair_review[0]["b_caption_zh_relpath"],
                "captions_zh/breast-0001-b.txt",
            )
            validation = validate_caption_package(root)
            self.assertEqual(validation["bilingual_caption_status"], "pass")
            self.assertEqual(validation["caption_pair_links_verified"], 4)

    def test_translation_loader_supports_old_csv_and_new_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            old_csv = root / "old.csv"
            dotted_stem = "TCGA-TEST-01Z-00-DX1.UUID_100_200"
            write_csv(old_csv, [{"base_name": dotted_stem, "caption_zh": "旧译文"}])
            new_jsonl = root / "new.jsonl"
            new_jsonl.write_text(
                json.dumps(
                    {
                        "stem": "b",
                        "caption_zh": "新译文",
                        "translation_source": "new_source",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            translations = _load_translations([old_csv, new_jsonl])

            self.assertEqual(translations[dotted_stem]["caption_zh"], "旧译文")
            self.assertEqual(translations["b"]["translation_source"], "new_source")
            self.assertGreaterEqual(len(CJK_RE.findall(translations["b"]["caption_zh"])), 3)


if __name__ == "__main__":
    unittest.main()
