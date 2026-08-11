import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phase3_joint_edit_refine.g2_he_review import (
    HE_REVIEW_SCHEMA_VERSION,
    _bind_codex_semantic_intent,
    _load_codex_semantic_review,
)
from phase3_joint_edit_refine.g2_qualification import QUALIFICATION_SCHEMA_VERSION
from phase3_joint_edit_refine.g2_v2_manifest import freeze_g2_v2_manifest


class G2V2ManifestTests(unittest.TestCase):
    def test_freeze_binds_source_qualification_and_visual_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_paths = []
            for name in ("image.png", "tissue.png", "nuclei.png"):
                path = root / name
                path.write_bytes(name.encode("ascii"))
                source_paths.append(path)
            legacy = root / "legacy.json"
            legacy.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "case-1",
                                "sample_id": "sample-1",
                                "source_image": str(source_paths[0]),
                                "source_tissue_mask": str(source_paths[1]),
                                "source_nuclei_mask": str(source_paths[2]),
                                "organic_seed": 7,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            qualification = root / "qualification.jsonl"
            q = {
                "schema_version": QUALIFICATION_SCHEMA_VERSION,
                "source_index": 0,
                "case_id": "case-1",
                "sample_id": "sample-1",
                "organ": "breast",
                "dataset": "BCSS",
                "pathology_domain_id": "breast-invasive-carcinoma-v1",
                "annotation_profile_id": "bcss-semantic-v1",
                "cell_observation_profile_id": "cellvit-five-class-v1",
                "cell_population_profile_id": "breast-cellvit-source-first-v1",
                "instruction": "increase tumor burden",
                "legacy_primitive": "tumor_burden_increase",
                "source_assets": {
                    "image": str(source_paths[0]),
                    "tissue_mask": str(source_paths[1]),
                    "nuclei_mask": str(source_paths[2]),
                    "image_sha256": _sha(source_paths[0]),
                    "tissue_mask_sha256": _sha(source_paths[1]),
                    "nuclei_mask_sha256": _sha(source_paths[2]),
                },
            }
            qualification.write_text(json.dumps(q) + "\n", encoding="utf-8")
            decision = root / "decision.jsonl"
            semantic_review = (
                Path(__file__).resolve().parents[1]
                / "phase3_joint_edit_refine"
                / "resources"
                / "g2_v2_codex_semantic_review_20260811.json"
            )
            semantic = _bind_codex_semantic_intent(
                case_id="case-1",
                instruction="increase tumor burden",
                primitive_id="tumor-burden-increase-v1",
                qualification_digest=_sha(qualification),
                semantic_review_digest=_sha(semantic_review),
                semantic_templates=_load_codex_semantic_review(semantic_review),
            )
            semantic_digest = hashlib.sha256(
                json.dumps(
                    semantic,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            d = {
                "schema_version": HE_REVIEW_SCHEMA_VERSION,
                "case_id": "case-1",
                "source_index": 0,
                "organ": "breast",
                "dataset": "BCSS",
                "pathology_domain_id": "breast-invasive-carcinoma-v1",
                "annotation_profile_id": "bcss-semantic-v1",
                "decision_status": "supported_mechanism",
                "selected_joint_primitive": "tumor-burden-increase-v1",
                "selected_mechanism_id": "breast-cohesive-nst-front",
                "recommended_instruction": "increase tumor burden",
                "prebound_semantic_intent": semantic,
                "prebound_semantic_intent_sha256": semantic_digest,
                "reason_code": "supported",
                "visual_observations": ["cohesive front"],
                "execution_allowed": True,
                "review_basis": {
                    "qualification_sha256": _sha(qualification),
                    "reviewer": "current_codex_session",
                    "source_image_sha256": _sha(source_paths[0]),
                    "source_tissue_mask_sha256": _sha(source_paths[1]),
                    "source_nuclei_mask_sha256": _sha(source_paths[2]),
                },
            }
            decision.write_text(json.dumps(d) + "\n", encoding="utf-8")

            result = freeze_g2_v2_manifest(
                legacy,
                qualification,
                decision,
                output_dir=root / "frozen",
                expected_cases=1,
            )

            manifest = Path(result["manifest"])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["execution_case_count"], 1)
            self.assertEqual(payload["cases"][0]["mechanism_id"], "breast-cohesive-nst-front")
            self.assertEqual(payload["cases"][0]["budget_contract"]["mode"], "joint_area")
            self.assertEqual(
                payload["cases"][0]["prebound_semantic_intent"]["parser"],
                "current_codex_session_semantic_parser_v2",
            )
            self.assertEqual(result["manifest_sha256"], _sha(manifest))
            self.assertFalse(result["target_mask_created"])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
