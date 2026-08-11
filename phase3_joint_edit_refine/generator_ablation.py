"""Digest-bound three-arm ablation for the frozen H&E generator.

The ablation changes exactly one condition axis at a time:

1. source tissue + source nuclei (reconstruction control),
2. target tissue + source nuclei (tissue response),
3. target tissue + target nuclei (joint response).

All arms share the source image, generation support, prompt, route and the
frozen pipeline's fixed seed.  Passing the mask-side contract never promotes a
mechanism: generator capability is opened only by the paired audit below.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from controlnet_train.inference.pipeline import EditPipelineInputs, run_edit_pipeline

from .generator_adapter import build_frozen_generator_inputs
from .models import JointContractError
from .post_generation import (
    CellConditionEvaluator,
    RenderMechanismCritic,
    TissueConditionEvaluator,
)

PAIRED_ABLATION_SCHEMA = "joint-generator-paired-ablation-v1"
PAIRED_ABLATION_AUDIT_SCHEMA = "joint-generator-paired-ablation-audit-v1"
FROZEN_PIPELINE_SEED = 42


@dataclass(frozen=True)
class PairedAblationThresholds:
    condition_fidelity_min: float = 0.80
    target_contrast_min: float = 0.03
    exterior_mean_absolute_drift_max: float = 0.03


def prepare_generator_paired_ablation(
    manifest_path: str | Path,
    *,
    output_root: str | Path,
    generator_snapshot: str,
) -> dict[str, Any]:
    """Freeze the three condition arms before loading any generator model."""

    if not generator_snapshot.strip():
        raise JointContractError("paired ablation requires a generator snapshot")
    manifest_path = Path(manifest_path)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    inputs, route, manifest = build_frozen_generator_inputs(
        manifest_path,
        output_dir=root / "validation-only",
    )
    paths = manifest["paths"]
    source = manifest["source_assets"]
    common = {
        "reference_image": str(source["image"]),
        "reference_tissue_mask": str(source["tissue"]),
        "reference_nuclei_mask": str(source["nuclei"]),
        "generation_support": str(paths["generation_support"]),
        "prompt": inputs.prompt,
        "dataset": inputs.dataset,
        "force_mode": route.mode,
    }
    arms = [
        _arm(
            "legacy_tissue_legacy_nuclei",
            tissue=source["tissue"],
            nuclei=source["nuclei"],
            common=common,
            output_dir=root / "01_legacy_tissue_legacy_nuclei",
        ),
        _arm(
            "joint_tissue_legacy_nuclei",
            tissue=paths["target_tissue_mask"],
            nuclei=source["nuclei"],
            common=common,
            output_dir=root / "02_joint_tissue_legacy_nuclei",
        ),
        _arm(
            "joint_tissue_joint_nuclei",
            tissue=paths["target_tissue_mask"],
            nuclei=paths["target_nuclei_mask"],
            common=common,
            output_dir=root / "03_joint_tissue_joint_nuclei",
        ),
    ]
    payload = {
        "schema_version": PAIRED_ABLATION_SCHEMA,
        "case_id": manifest["case_id"],
        "candidate_id": manifest["candidate_id"],
        "primitive_id": manifest["primitive_id"],
        "mechanism_id": manifest["mechanism_id"],
        "generator_snapshot": generator_snapshot,
        "seed": FROZEN_PIPELINE_SEED,
        "seed_authority": "frozen_phase5_pipeline_fixed_seed",
        "render_expectations": list(manifest.get("render_expectations", ())),
        "render_vetoes": list(manifest.get("render_vetoes", ())),
        "source_handoff_manifest": str(manifest_path.resolve()),
        "source_handoff_manifest_sha256": _sha256(manifest_path),
        "route": asdict(route),
        "arms": arms,
        "capability_policy": (
            "all three arms and the independent paired audit must pass; "
            "mask-side success alone cannot promote render capability"
        ),
    }
    payload["ablation_id"] = _canonical_digest(payload)
    destination = root / "paired_ablation_plan.json"
    _write_json(destination, payload)
    return {**payload, "plan_path": str(destination.resolve())}


def run_generator_paired_ablation(
    plan_path: str | Path,
    *,
    inpaint_bundle: object,
    cross_bundle: object,
    inpaint_runner,
    cross_runner,
    tissue_evaluator: TissueConditionEvaluator,
    cell_evaluator: CellConditionEvaluator,
    visual_critic: RenderMechanismCritic,
    thresholds: PairedAblationThresholds | None = None,
) -> dict[str, Any]:
    """Run all arms and immediately perform the fail-closed paired audit."""

    plan = _load_plan(plan_path)
    outputs: dict[str, str] = {}
    for arm in plan["arms"]:
        inputs = EditPipelineInputs(
            reference_image=arm["reference_image"],
            reference_tissue_mask=arm["reference_tissue_mask"],
            reference_nuclei_mask=arm["reference_nuclei_mask"],
            target_tissue_mask=arm["target_tissue_mask"],
            target_nuclei_mask=arm["target_nuclei_mask"],
            generation_change_region=arm["generation_support"],
            output_dir=arm["output_dir"],
            prompt=arm.get("prompt"),
            dataset=arm.get("dataset"),
            force_mode=arm["force_mode"],
            save_debug_artifacts=True,
        )
        result = run_edit_pipeline(
            inputs=inputs,
            inpaint_bundle=inpaint_bundle,
            cross_bundle=cross_bundle,
            inpaint_runner=inpaint_runner,
            cross_runner=cross_runner,
        )
        outputs[arm["arm_id"]] = str((result.output_dir / "final.png").resolve())
    return audit_generator_paired_ablation(
        plan_path,
        generated_images=outputs,
        tissue_evaluator=tissue_evaluator,
        cell_evaluator=cell_evaluator,
        visual_critic=visual_critic,
        thresholds=thresholds,
    )


def audit_generator_paired_ablation(
    plan_path: str | Path,
    *,
    generated_images: dict[str, str | Path],
    tissue_evaluator: TissueConditionEvaluator | None,
    cell_evaluator: CellConditionEvaluator | None,
    visual_critic: RenderMechanismCritic | None,
    thresholds: PairedAblationThresholds | None = None,
) -> dict[str, Any]:
    """Measure response to tissue and nuclei conditions independently."""

    plan_path = Path(plan_path)
    plan = _load_plan(plan_path)
    thresholds = thresholds or PairedAblationThresholds()
    missing_evaluators = [
        name
        for name, evaluator in (
            ("tissue", tissue_evaluator),
            ("cell", cell_evaluator),
            ("visual", visual_critic),
        )
        if evaluator is None
    ]
    expected_arm_ids = {item["arm_id"] for item in plan["arms"]}
    missing_outputs = sorted(
        arm_id
        for arm_id in expected_arm_ids
        if not Path(str(generated_images.get(arm_id, ""))).is_file()
    )
    if missing_evaluators or missing_outputs:
        payload = {
            "schema_version": PAIRED_ABLATION_AUDIT_SCHEMA,
            "ablation_id": plan["ablation_id"],
            "passed": False,
            "capability_status": "render_unsupported",
            "reasons": [
                *(
                    ["missing evaluators: " + ", ".join(missing_evaluators)]
                    if missing_evaluators
                    else []
                ),
                *(
                    ["missing generated arms: " + ", ".join(missing_outputs)]
                    if missing_outputs
                    else []
                ),
            ],
            "checks": {},
        }
        return _write_audit(plan_path, payload)

    assert tissue_evaluator is not None
    assert cell_evaluator is not None
    assert visual_critic is not None
    arms = {item["arm_id"]: item for item in plan["arms"]}
    source_tissue = arms["legacy_tissue_legacy_nuclei"][
        "reference_tissue_mask"
    ]
    source_nuclei = arms["legacy_tissue_legacy_nuclei"][
        "reference_nuclei_mask"
    ]
    target_tissue = arms["joint_tissue_joint_nuclei"]["target_tissue_mask"]
    target_nuclei = arms["joint_tissue_joint_nuclei"]["target_nuclei_mask"]
    measurements: dict[str, Any] = {}
    for arm_id in sorted(expected_arm_ids):
        image = Path(generated_images[arm_id])
        measurements[arm_id] = {
            "generated_image": str(image.resolve()),
            "generated_image_sha256": _sha256(image),
            "tissue_to_source": tissue_evaluator.evaluate(
                generated_image=image,
                target_tissue_mask=source_tissue,
            ),
            "tissue_to_target": tissue_evaluator.evaluate(
                generated_image=image,
                target_tissue_mask=target_tissue,
            ),
            "cells_to_source": cell_evaluator.evaluate(
                generated_image=image,
                target_nuclei_mask=source_nuclei,
                mechanism_id=plan["mechanism_id"],
            ),
            "cells_to_target": cell_evaluator.evaluate(
                generated_image=image,
                target_nuclei_mask=target_nuclei,
                mechanism_id=plan["mechanism_id"],
            ),
            "exterior_mean_absolute_drift": _exterior_drift(
                arms[arm_id]["reference_image"],
                image,
                arms[arm_id]["generation_support"],
            ),
        }
    legacy = measurements["legacy_tissue_legacy_nuclei"]
    tissue_only = measurements["joint_tissue_legacy_nuclei"]
    joint = measurements["joint_tissue_joint_nuclei"]
    tissue_changed = _sha256(Path(source_tissue)) != _sha256(Path(target_tissue))
    cells_changed = _sha256(Path(source_nuclei)) != _sha256(Path(target_nuclei))
    tissue_contrast = (
        float(tissue_only["tissue_to_target"]["fidelity"])
        - float(legacy["tissue_to_target"]["fidelity"])
    )
    cell_target_score_joint = _cell_score(joint["cells_to_target"])
    cell_target_score_tissue_only = _cell_score(tissue_only["cells_to_target"])
    cell_contrast = cell_target_score_joint - cell_target_score_tissue_only
    checks = {
        "source_reconstruction": (
            float(legacy["tissue_to_source"]["fidelity"])
            >= thresholds.condition_fidelity_min
            and _cell_score(legacy["cells_to_source"])
            >= thresholds.condition_fidelity_min
        ),
        "tissue_condition_response": (
            not tissue_changed
            or (
                float(tissue_only["tissue_to_target"]["fidelity"])
                >= thresholds.condition_fidelity_min
                and tissue_contrast >= thresholds.target_contrast_min
            )
        ),
        "joint_tissue_fidelity": (
            float(joint["tissue_to_target"]["fidelity"])
            >= thresholds.condition_fidelity_min
        ),
        "cell_condition_response": (
            cells_changed
            and cell_target_score_joint >= thresholds.condition_fidelity_min
            and cell_contrast >= thresholds.target_contrast_min
        ),
        "exterior_preservation": all(
            float(item["exterior_mean_absolute_drift"])
            <= thresholds.exterior_mean_absolute_drift_max
            for item in measurements.values()
        ),
    }
    visual = visual_critic.evaluate(
        source_image=arms["joint_tissue_joint_nuclei"]["reference_image"],
        generated_image=generated_images["joint_tissue_joint_nuclei"],
        mechanism_id=plan["mechanism_id"],
        expectations=tuple(plan.get("render_expectations", ())),
        vetoes=tuple(plan.get("render_vetoes", ())),
    )
    checks["independent_visual_critic"] = bool(
        visual.get("approved") is True and not visual.get("veto_reasons")
    )
    reasons = [name for name, passed in checks.items() if not passed]
    payload = {
        "schema_version": PAIRED_ABLATION_AUDIT_SCHEMA,
        "ablation_id": plan["ablation_id"],
        "case_id": plan["case_id"],
        "candidate_id": plan["candidate_id"],
        "primitive_id": plan["primitive_id"],
        "mechanism_id": plan["mechanism_id"],
        "generator_snapshot": plan["generator_snapshot"],
        "seed": plan["seed"],
        "passed": not reasons,
        "capability_status": (
            "render_supported" if not reasons else "render_unsupported"
        ),
        "reasons": reasons,
        "checks": checks,
        "metrics": {
            "tissue_condition_changed": tissue_changed,
            "cell_condition_changed": cells_changed,
            "tissue_target_contrast": tissue_contrast,
            "cell_target_contrast": cell_contrast,
            "thresholds": asdict(thresholds),
            "arms": measurements,
            "visual": visual,
            "evaluators": {
                "tissue": tissue_evaluator.name,
                "cell": cell_evaluator.name,
                "visual": visual_critic.name,
            },
        },
    }
    return _write_audit(plan_path, payload)


def _arm(
    arm_id: str,
    *,
    tissue: str | Path,
    nuclei: str | Path,
    common: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    values = {
        "arm_id": arm_id,
        **common,
        "target_tissue_mask": str(tissue),
        "target_nuclei_mask": str(nuclei),
        "output_dir": str(output_dir.resolve()),
    }
    values["condition_sha256"] = _canonical_digest(
        {
            "target_tissue_mask_sha256": _sha256(Path(tissue)),
            "target_nuclei_mask_sha256": _sha256(Path(nuclei)),
            "generation_support_sha256": _sha256(
                Path(values["generation_support"])
            ),
        }
    )
    return values


def _load_plan(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointContractError(f"could not load paired ablation plan: {exc}") from exc
    if payload.get("schema_version") != PAIRED_ABLATION_SCHEMA:
        raise JointContractError("unsupported paired ablation schema")
    observed_id = payload.get("ablation_id")
    canonical = dict(payload)
    canonical.pop("ablation_id", None)
    if observed_id != _canonical_digest(canonical):
        raise JointContractError("paired ablation plan digest drift")
    if payload.get("seed") != FROZEN_PIPELINE_SEED:
        raise JointContractError("paired ablation seed differs from frozen pipeline")
    handoff = Path(str(payload.get("source_handoff_manifest") or ""))
    if (
        not handoff.is_file()
        or _sha256(handoff) != payload.get("source_handoff_manifest_sha256")
    ):
        raise JointContractError("paired ablation source handoff has digest drift")
    for arm in payload.get("arms", ()):
        expected = _canonical_digest(
            {
                "target_tissue_mask_sha256": _sha256(
                    Path(arm["target_tissue_mask"])
                ),
                "target_nuclei_mask_sha256": _sha256(
                    Path(arm["target_nuclei_mask"])
                ),
                "generation_support_sha256": _sha256(
                    Path(arm["generation_support"])
                ),
            }
        )
        if expected != arm.get("condition_sha256"):
            raise JointContractError(
                f"paired ablation condition drift: {arm.get('arm_id')}"
            )
    return payload


def _cell_score(metrics: dict[str, Any]) -> float:
    keys = (
        "count_consistency",
        "type_consistency",
        "spatial_consistency",
        "interface_distance_consistency",
    )
    values = [float(metrics[key]) for key in keys]
    if not all(np.isfinite(value) for value in values):
        raise JointContractError("paired cell evaluator returned non-finite values")
    return float(np.mean(values))


def _exterior_drift(source_path, generated_path, support_path) -> float:
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=float) / 255.0
    generated = np.asarray(
        Image.open(generated_path).convert("RGB"), dtype=float
    ) / 255.0
    support = np.asarray(Image.open(support_path).convert("L")) > 0
    if source.shape != generated.shape or source.shape[:2] != support.shape:
        raise JointContractError("paired ablation images and support are not aligned")
    exterior = ~support
    return (
        float(np.mean(np.abs(source[exterior] - generated[exterior])))
        if np.any(exterior)
        else 0.0
    )


def _write_audit(plan_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    destination = plan_path.parent / "paired_ablation_audit.json"
    _write_json(destination, payload)
    return {**payload, "audit_path": str(destination.resolve())}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
