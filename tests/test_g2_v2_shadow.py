import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phase3_joint_edit_refine.g2_v2_manifest import (
    G2_V2_MANIFEST_SCHEMA,
    PRIMITIVE_ONTOLOGY_VERSION,
)
from phase3_joint_edit_refine.g2_v2_shadow import build_g2_v2_shadow


class G2V2ShadowTests(unittest.TestCase):
    def test_shadow_is_stratified_and_materializes_source_bound_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = {}
            for name in ("image.png", "tissue.png", "nuclei.png"):
                path = root / name
                path.write_bytes(name.encode("ascii"))
                assets[name] = path
            cases = []
            index = 0
            for organ, dataset, domain, annotation, population, mechanism in (
                ("breast", "BCSS", "breast-invasive-carcinoma-v1", "bcss-semantic-v1", "breast-cellvit-source-first-v1", "breast-annotation-anchored-boundary-growth"),
                ("lung", "IGNITE", "lung-carcinoma-v1", "ignite-semantic-v1", "lung-cellvit-source-first-v1", "lung-solid-squamous-growth"),
            ):
                primitive = "cohesive-boundary-expansion-v1"
                for primitive in (primitive,) * 2:
                    cases.append(
                        _case(
                            index,
                            organ=organ,
                            dataset=dataset,
                            domain=domain,
                            annotation=annotation,
                            population=population,
                            mechanism=mechanism,
                            primitive=primitive,
                            assets=assets,
                            execution_allowed=True,
                        )
                    )
                    index += 1
            cases.append(
                _case(
                    index,
                    organ="lung",
                    dataset="IGNITE",
                    domain="lung-carcinoma-v1",
                    annotation="ignite-semantic-v1",
                    population="lung-cellvit-source-first-v1",
                    mechanism=None,
                    primitive=None,
                    assets=assets,
                    execution_allowed=False,
                )
            )
            manifest = root / "frozen.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": G2_V2_MANIFEST_SCHEMA,
                        "primitive_ontology_version": PRIMITIVE_ONTOLOGY_VERSION,
                        "case_count": len(cases),
                        "cases": cases,
                    }
                ),
                encoding="utf-8",
            )

            result = build_g2_v2_shadow(
                manifest,
                output_dir=root / "shadow",
                per_organ=2,
                abstain_controls=1,
            )

            self.assertEqual(result["selected_executable_count"], 4)
            self.assertEqual(result["selected_abstain_control_count"], 1)
            runnable = json.loads(Path(result["runnable_manifest"]).read_text())
            self.assertEqual({item["provenance"]["provider"] for item in runnable}, {"BCSS", "IGNITE"})
            self.assertTrue(all(item["cell_count_extent_budget"] is None for item in runnable))
            self.assertTrue(all(item["provenance"]["g2_v2_manifest_sha256"] == _sha(manifest) for item in runnable))

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["cases"][0]["review_basis"]["reviewer"] = (
                "offline_heuristic"
            )
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "visual_review_not_current_codex"
            ):
                build_g2_v2_shadow(
                    manifest,
                    output_dir=root / "invalid-shadow",
                    per_organ=2,
                    abstain_controls=1,
                )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["cases"][0]["review_basis"]["reviewer"] = (
                "current_codex_session"
            )
            payload["cases"][0]["mechanism_id"] = (
                "colorectal-gland-forming-front"
            )
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "mechanism_not_in_execution_scope"
            ):
                build_g2_v2_shadow(
                    manifest,
                    output_dir=root / "closed-mechanism-shadow",
                    per_organ=2,
                    abstain_controls=1,
                )


def _case(index, *, organ, dataset, domain, annotation, population, mechanism, primitive, assets, execution_allowed):
    instruction = (
        "increase tumor burden"
        if primitive == "tumor-burden-increase-v1"
        else "expand the tumor boundary locally"
    )
    semantic = (
        {
            "schema_version": "joint-semantic-intent-v3",
            "instruction": instruction,
            "instruction_mode": "direct_edit",
            "scenario": "direct_edit",
            "clinical_direction": "unspecified",
            "treatment_context": "none",
            "scenario_target": "tumor",
            "explicit_edit_scope": "tissue_burden",
            "primitive_id": primitive,
            "subject": "tumor-burden",
            "direction": (
                "increase"
                if primitive in {
                    "tumor-burden-increase-v1",
                    "cohesive-boundary-expansion-v1",
                }
                else "decrease"
            ),
            "explicit_cell_class": None,
            "explicit_location": None,
            "user_constraints": [],
            "uncertainties": [],
            "parser": "current_codex_session_semantic_parser_v1",
            "primitive_hypotheses": [
                {
                    "primitive_id": primitive,
                    "semantic_fit": "explicit",
                    "priority": 0,
                    "rationale": "fixture",
                }
            ],
            "parser_metadata": {
                "reviewer": "current_codex_session",
                "llm_api_used": False,
                "execution_runner_may_not_reparse": True,
            },
        }
        if execution_allowed
        else None
    )
    semantic_digest = (
        hashlib.sha256(
            json.dumps(
                semantic,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if semantic is not None
        else None
    )
    return {
        "source_index": index,
        "case_id": f"case-{index}",
        "organ": organ,
        "dataset": dataset,
        "pathology_domain_id": domain,
        "annotation_profile_id": annotation,
        "cell_observation_profile_id": "cellvit-five-class-v1",
        "cell_population_profile_id": population,
        "source_image_uri": str(assets["image.png"]),
        "source_tissue_mask_uri": str(assets["tissue.png"]),
        "source_nuclei_mask_uri": str(assets["nuclei.png"]),
        "source_nuclei_instances_uri": None,
        "source_digests": {
            "image_sha256": _sha(assets["image.png"]),
            "tissue_mask_sha256": _sha(assets["tissue.png"]),
            "nuclei_mask_sha256": _sha(assets["nuclei.png"]),
            "nuclei_instances_sha256": None,
            "auxiliary_structure_sha256": {},
        },
        "decision_status": "supported_mechanism" if execution_allowed else "abstain",
        "execution_allowed": execution_allowed,
        "instruction": instruction,
        "primitive_id": primitive,
        "mechanism_id": mechanism,
        "prebound_semantic_intent": semantic,
        "prebound_semantic_intent_sha256": semantic_digest,
        "decision_reason_code": "fixture",
        "visual_observations": ["fixture review"],
        "review_basis": {
            "reviewer": "current_codex_session",
            "llm_api_used": False,
        },
        "joint_area_budget": ({"target_fraction": 0.19, "min_fraction": 0.14, "max_fraction": 0.24, "tissue_min_fraction": 0.14} if execution_allowed else None),
        "seed": 7,
        "pixel_size_um": None,
        "source_manifest_metadata": {},
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
