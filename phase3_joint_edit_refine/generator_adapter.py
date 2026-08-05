"""Read-only bridge from an approved joint handoff to the frozen H&E pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from controlnet_train.inference.pipeline import EditPipelineInputs, run_edit_pipeline

from .models import JointContractError


@dataclass(frozen=True)
class JointGeneratorRoutingConfig:
    inpaint_max_joint_fraction: float = 0.12
    cross_min_joint_fraction: float = 0.30


@dataclass(frozen=True)
class JointGeneratorRoute:
    mode: str
    joint_fraction: float
    generation_support_fraction: float
    reason: str


def route_joint_handoff(manifest: dict[str, Any], *, config: JointGeneratorRoutingConfig | None = None) -> JointGeneratorRoute:
    config = config or JointGeneratorRoutingConfig()
    ledger = manifest.get("ledger")
    if not isinstance(ledger, dict):
        raise JointContractError("joint handoff has no ledger")
    joint = float(ledger.get("joint_fraction", 0.0))
    support = float(ledger.get("generation_support_fraction", 0.0))
    if joint <= 0:
        raise JointContractError("zero-joint-change handoff is a noop and cannot be generated")
    if joint <= config.inpaint_max_joint_fraction:
        mode, reason = "inpaint", "small joint support favors local preservation"
    elif joint >= config.cross_min_joint_fraction:
        mode, reason = "cross", "large joint structural change requires cross generation"
    else:
        mode, reason = "inpaint", "gray-zone joint edit starts with preservation-oriented inpaint"
    return JointGeneratorRoute(mode, joint, support, reason)


def build_frozen_generator_inputs(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    prompt: str | None = None,
    dataset: str | None = None,
    routing_config: JointGeneratorRoutingConfig | None = None,
) -> tuple[EditPipelineInputs, JointGeneratorRoute, dict[str, Any]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "joint-generation-handoff-v1":
        raise JointContractError("unsupported joint generation handoff schema")
    paths = manifest.get("paths")
    digests = manifest.get("digests")
    source = manifest.get("source_assets")
    if not all(isinstance(item, dict) for item in (paths, digests, source)):
        raise JointContractError("joint handoff paths/digests/source assets are incomplete")
    for name in ("target_tissue_mask", "target_nuclei_mask", "joint_change", "generation_support"):
        path = Path(paths.get(name, ""))
        if not path.is_file() or _sha256(path) != digests.get(name + "_sha256"):
            raise JointContractError(f"joint handoff artifact is missing or has digest drift: {name}")
    # The current frozen pipeline already accepts target tissue+nuclei and a
    # separate erase/regeneration support mask. We force only the route because
    # its legacy router measures tissue diff rather than J.
    route = route_joint_handoff(manifest, config=routing_config)
    inputs = EditPipelineInputs(
        reference_image=source["image"],
        reference_tissue_mask=source["tissue"],
        reference_nuclei_mask=source["nuclei"],
        target_tissue_mask=paths["target_tissue_mask"],
        target_nuclei_mask=paths["target_nuclei_mask"],
        generation_change_region=paths["generation_support"],
        output_dir=output_dir,
        prompt=prompt,
        dataset=dataset,
        force_mode=route.mode,
        save_debug_artifacts=True,
    )
    return inputs, route, manifest


def run_frozen_joint_generator(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    inpaint_bundle: object,
    cross_bundle: object,
    inpaint_runner,
    cross_runner,
    prompt: str | None = None,
    dataset: str | None = None,
    routing_config: JointGeneratorRoutingConfig | None = None,
):
    inputs, route, manifest = build_frozen_generator_inputs(
        manifest_path,
        output_dir=output_dir,
        prompt=prompt,
        dataset=dataset,
        routing_config=routing_config,
    )
    result = run_edit_pipeline(
        inputs=inputs,
        inpaint_bundle=inpaint_bundle,
        cross_bundle=cross_bundle,
        inpaint_runner=inpaint_runner,
        cross_runner=cross_runner,
    )
    audit = {
        "schema_version": "joint-frozen-generator-route-v1",
        "case_id": manifest["case_id"],
        "candidate_id": manifest["candidate_id"],
        "route": asdict(route),
        "legacy_pipeline_selected_mode": result.selected_mode,
        "semantic_change_ratio_reported_by_legacy": result.change_ratio,
        "joint_change_is_authoritative_for_routing": True,
    }
    path = Path(output_dir) / "joint_generation_route.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return result, audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
