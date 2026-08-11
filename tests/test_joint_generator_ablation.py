from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_joint_edit_refine.generator_ablation import (
    audit_generator_paired_ablation,
    prepare_generator_paired_ablation,
)


class _TissueEvaluator:
    name = "fixture-segmentator"

    def evaluate(self, *, generated_image, target_tissue_mask):
        arm = Path(generated_image).stem
        target = Path(target_tissue_mask).stem
        source_arm = arm == "legacy_tissue_legacy_nuclei"
        source_target = target == "source_tissue"
        return {"fidelity": 0.95 if source_arm == source_target else 0.20}


class _CellEvaluator:
    name = "fixture-cellvit"

    def evaluate(self, *, generated_image, target_nuclei_mask, mechanism_id):
        del mechanism_id
        arm = Path(generated_image).stem
        target = Path(target_nuclei_mask).stem
        source_arm = arm != "joint_tissue_joint_nuclei"
        source_target = target == "source_nuclei"
        value = 0.95 if source_arm == source_target else 0.20
        return {
            "count_consistency": value,
            "type_consistency": value,
            "spatial_consistency": value,
            "interface_distance_consistency": value,
        }


class _VisualCritic:
    name = "fixture-visual-critic"

    def evaluate(self, **_kwargs):
        return {"approved": True, "veto_reasons": []}


def test_paired_ablation_is_digest_bound_and_separates_condition_axes(tmp_path: Path):
    manifest = _write_handoff(tmp_path)
    plan = prepare_generator_paired_ablation(
        manifest,
        output_root=tmp_path / "ablation",
        generator_snapshot="fixture-generator-checkpoint",
    )
    arms = {item["arm_id"]: item for item in plan["arms"]}
    assert (
        arms["legacy_tissue_legacy_nuclei"]["target_tissue_mask"]
        != arms["joint_tissue_legacy_nuclei"]["target_tissue_mask"]
    )
    assert (
        arms["joint_tissue_legacy_nuclei"]["target_nuclei_mask"]
        != arms["joint_tissue_joint_nuclei"]["target_nuclei_mask"]
    )
    assert len({item["generation_support"] for item in arms.values()}) == 1

    source = np.asarray(Image.open(tmp_path / "source_image.png"))
    outputs = {}
    for arm_id in arms:
        path = tmp_path / f"{arm_id}.png"
        Image.fromarray(source).save(path)
        outputs[arm_id] = path
    audit = audit_generator_paired_ablation(
        plan["plan_path"],
        generated_images=outputs,
        tissue_evaluator=_TissueEvaluator(),
        cell_evaluator=_CellEvaluator(),
        visual_critic=_VisualCritic(),
    )
    assert audit["passed"]
    assert audit["capability_status"] == "render_supported"
    assert audit["checks"]["tissue_condition_response"]
    assert audit["checks"]["cell_condition_response"]


def test_paired_ablation_fails_closed_without_evaluators(tmp_path: Path):
    manifest = _write_handoff(tmp_path)
    plan = prepare_generator_paired_ablation(
        manifest,
        output_root=tmp_path / "ablation",
        generator_snapshot="fixture-generator-checkpoint",
    )
    audit = audit_generator_paired_ablation(
        plan["plan_path"],
        generated_images={},
        tissue_evaluator=None,
        cell_evaluator=None,
        visual_critic=None,
    )
    assert not audit["passed"]
    assert audit["capability_status"] == "render_unsupported"
    assert "missing evaluators" in audit["reasons"][0]


def _write_handoff(root: Path) -> Path:
    source_image = np.full((16, 16, 3), 128, dtype=np.uint8)
    source_tissue = np.zeros((16, 16), dtype=np.uint8)
    source_tissue[:, :8] = 1
    target_tissue = source_tissue.copy()
    target_tissue[:, 8:10] = 1
    source_nuclei = np.zeros((16, 16), dtype=np.uint8)
    source_nuclei[3:5, 3:5] = 1
    target_nuclei = source_nuclei.copy()
    target_nuclei[9:11, 9:11] = 1
    assets = {
        "source_image": source_image,
        "source_tissue": source_tissue,
        "source_nuclei": source_nuclei,
        "target_tissue_mask": target_tissue,
        "target_nuclei_mask": target_nuclei,
    }
    paths: dict[str, str] = {}
    digests: dict[str, str] = {}
    for name, array in assets.items():
        path = root / f"{name}.png"
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    tissue_change = source_tissue != target_tissue
    cell_change = source_nuclei != target_nuclei
    joint = tissue_change | cell_change
    support = joint.copy()
    support[1:15, 1:15] |= joint[1:15, 1:15]
    artifact_arrays = {
        "tissue_change": tissue_change,
        "cell_change": cell_change,
        "joint_change": joint,
        "generation_support": support,
        "contract_T_population": tissue_change,
        "contract_E_erasure": np.zeros_like(joint),
        "contract_P_placement_centers": cell_change,
        "contract_V_valid_footprints": cell_change,
        "contract_S_support_context": support,
        "contract_M_mechanism_region": joint,
        "contract_C_continuity_region": tissue_change,
        "contract_A_selected_anchor": tissue_change,
    }
    for name, array in artifact_arrays.items():
        path = root / f"{name}.png"
        Image.fromarray(array.astype(np.uint8) * 255).save(path)
        paths[name] = str(path)
    paths["target_tissue_mask"] = paths["target_tissue_mask"]
    paths["target_nuclei_mask"] = paths["target_nuclei_mask"]
    contract_id = "fixture-contract"
    contract_path = root / "executable_contract.json"
    contract_path.write_text(
        json.dumps({"contract_id": contract_id}), encoding="utf-8"
    )
    paths["executable_contract"] = str(contract_path)
    for name, path in paths.items():
        if name.startswith("source_"):
            continue
        digests[name + "_sha256"] = _sha256(Path(path))
    binding = {
        "schema_version": "joint-result-binding-v2",
        "contract_id": contract_id,
        "candidate_id": "fixture-candidate",
        "target_tissue_sha256": digests["target_tissue_mask_sha256"],
        "target_nuclei_sha256": digests["target_nuclei_mask_sha256"],
        "tissue_change_sha256": digests["tissue_change_sha256"],
        "cell_change_sha256": digests["cell_change_sha256"],
        "joint_change_sha256": digests["joint_change_sha256"],
        "generation_support_sha256": digests["generation_support_sha256"],
        "contract_T_population_sha256": digests[
            "contract_T_population_sha256"
        ],
    }
    binding["binding_id"] = _canonical_digest(binding)
    manifest = {
        "schema_version": "joint-generation-handoff-v3",
        "case_id": "fixture-case",
        "candidate_id": "fixture-candidate",
        "executable_contract_id": contract_id,
        "primitive_id": "architecture-progression-v1",
        "mechanism_id": "prostate-gleason-architecture-progression",
        "ledger": {
            "joint_fraction": float(np.mean(joint)),
            "generation_support_fraction": float(np.mean(support)),
        },
        "render_expectations": ["target architecture"],
        "render_vetoes": ["filled lumen"],
        "source_assets": {
            "image": paths["source_image"],
            "tissue": paths["source_tissue"],
            "nuclei": paths["source_nuclei"],
        },
        "paths": {key: value for key, value in paths.items() if not key.startswith("source_")},
        "digests": digests,
        "execution_contract": {
            "executable_contract": {"contract_id": contract_id}
        },
        "result_binding": binding,
    }
    destination = root / "handoff.json"
    destination.write_text(json.dumps(manifest), encoding="utf-8")
    return destination


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
