"""Read-only bridge from an approved joint handoff to the frozen H&E pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from controlnet_train.inference.pipeline import (
    EditPipelineInputs,
    resolve_prompt,
    run_edit_pipeline,
)
from controlnet_train.inference.router import (
    AgenticRouteFeatures,
    AgenticRoutingDecision,
)

from .models import JointContractError


@dataclass(frozen=True)
class JointGeneratorRoutingConfig:
    inpaint_max_generation_support_fraction: float = 0.12
    force_cross_min_generation_support_fraction: float = 0.50
    cell_only_decrease_cross_first: bool = False
    cell_only_increase_inpaint_first: bool = True
    generic_immune_decrease_cross_first: bool = True


@dataclass(frozen=True)
class JointGeneratorRoute:
    mode: str
    joint_fraction: float
    generation_support_fraction: float
    force_cross: bool
    reason: str


def route_joint_handoff(manifest: dict[str, Any], *, config: JointGeneratorRoutingConfig | None = None) -> JointGeneratorRoute:
    config = config or JointGeneratorRoutingConfig()
    if not (
        0.0
        <= config.inpaint_max_generation_support_fraction
        <= config.force_cross_min_generation_support_fraction
        <= 1.0
    ):
        raise JointContractError(
            "generation-support routing thresholds must satisfy 0 <= inpaint <= cross <= 1"
        )
    ledger = manifest.get("ledger")
    if not isinstance(ledger, dict):
        raise JointContractError("joint handoff has no ledger")
    joint = float(ledger.get("joint_fraction", 0.0))
    support = float(ledger.get("generation_support_fraction", 0.0))
    if joint <= 0:
        raise JointContractError("zero-joint-change handoff is a noop and cannot be generated")
    if not joint <= support <= 1.0:
        raise JointContractError(
            "generation support fraction must contain the joint change and be at most one"
        )
    tissue = float(ledger.get("tissue_fraction", 0.0))
    cell = float(ledger.get("cell_fraction", 0.0))
    cell_only = tissue == 0.0 and cell > 0.0
    primitive_id = str(manifest.get("primitive_id", ""))
    cell_only_decrease = cell_only and "decrease" in primitive_id
    cell_only_increase = cell_only and "increase" in primitive_id
    if support >= config.force_cross_min_generation_support_fraction:
        mode, force_cross, reason = (
            "cross",
            True,
            "large generation support requires cross generation",
        )
    elif (
        primitive_id == "generic-immune-infiltrate-decrease-v1"
        and config.generic_immune_decrease_cross_first
    ):
        mode, force_cross, reason = (
            "cross",
            False,
            "generic immune-infiltrate decrease starts with Cross-v1 "
            "to avoid localized inpaint blur and false cell-like texture; "
            "inpaint remains the agentic fallback",
        )
    elif cell_only_decrease and config.cell_only_decrease_cross_first:
        mode, force_cross, reason = (
            "cross",
            False,
            "cell-only decrease starts with Cross-v1; "
            "inpaint remains the agentic fallback",
        )
    elif cell_only_decrease:
        mode, force_cross, reason = (
            "inpaint",
            False,
            "cell-only decrease uses expanded-context inpaint to clear the "
            "full stained nucleus footprint; Cross-v1 remains fallback",
        )
    elif cell_only_increase and config.cell_only_increase_inpaint_first:
        mode, force_cross, reason = (
            "inpaint",
            False,
            "cell-only increase starts with preservation-oriented inpaint; "
            "Cross-v1 remains the agentic fallback",
        )
    elif support <= config.inpaint_max_generation_support_fraction:
        mode, force_cross, reason = (
            "inpaint",
            False,
            "small generation support favors local preservation",
        )
    else:
        mode, force_cross, reason = (
            "inpaint",
            False,
            "gray-zone generation support starts with preservation-oriented inpaint",
        )
    return JointGeneratorRoute(mode, joint, support, force_cross, reason)


def build_agentic_joint_route(
    manifest: dict[str, Any],
    *,
    joint_change_mask: np.ndarray,
    generation_support_mask: np.ndarray,
    reference_tissue_mask: np.ndarray,
    config: JointGeneratorRoutingConfig | None = None,
) -> AgenticRoutingDecision:
    """Translate an approved joint handoff into the online agent route.

    The production agent historically routes only on tissue-label deltas. A
    nuclei-only joint edit therefore looks like a no-op unless the approved
    generation support is made authoritative. Large generation supports are
    Cross-only; smaller supports retain the standard alternate backend used by
    the online evaluator/recovery loop.
    """

    route = route_joint_handoff(manifest, config=config)
    joint_change = np.asarray(joint_change_mask, dtype=bool)
    generation_support = np.asarray(generation_support_mask, dtype=bool)
    tissue = np.asarray(reference_tissue_mask)
    if (
        joint_change.ndim != 2
        or generation_support.shape != joint_change.shape
        or tissue.shape != joint_change.shape
    ):
        raise JointContractError(
            "joint change, generation support, and reference tissue mask must be aligned 2D arrays"
        )
    if np.any(joint_change & ~generation_support):
        raise JointContractError("generation support must contain the joint change")
    measured_support_fraction = float(np.mean(generation_support))
    fraction_tolerance = 1.0 / generation_support.size
    if not np.isclose(
        measured_support_fraction,
        route.generation_support_fraction,
        rtol=0.0,
        atol=fraction_tolerance,
    ):
        raise JointContractError(
            "generation support mask fraction does not match the approved handoff ledger"
        )
    changed_pixels = int(np.count_nonzero(generation_support))
    if changed_pixels <= 0:
        raise JointContractError("approved joint handoff has an empty generation support mask")
    components, component_count = ndimage.label(generation_support)
    sizes = np.bincount(components.ravel())[1:]
    largest_component = int(sizes.max()) if sizes.size else 0
    ys, xs = np.where(generation_support)
    bbox_pixels = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    tissue_pixels = int(np.count_nonzero(tissue))
    features = AgenticRouteFeatures(
        change_ratio_image=changed_pixels / int(generation_support.size),
        change_ratio_tissue=(
            changed_pixels / tissue_pixels if tissue_pixels else 0.0
        ),
        component_count=int(component_count),
        largest_component_fraction=largest_component / changed_pixels,
        bbox_fraction=bbox_pixels / int(generation_support.size),
        transition_count=0,
        changed_tissue_ids_from=(),
        changed_tissue_ids_to=(),
    )
    if route.force_cross:
        candidates = ("cross",)
    elif route.mode == "inpaint":
        candidates = ("inpaint", "cross")
    else:
        candidates = ("cross", "inpaint")
    thresholds = config or JointGeneratorRoutingConfig()
    confidence = (
        0.55
        if thresholds.inpaint_max_generation_support_fraction
        < route.generation_support_fraction
        < thresholds.force_cross_min_generation_support_fraction
        else 0.90
    )
    return AgenticRoutingDecision(
        primary_mode=route.mode,
        candidate_modes=candidates,
        confidence=confidence,
        reason=f"approved joint handoff: {route.reason}",
        features=features,
    )


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
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        "joint-generation-handoff-v2",
        "joint-generation-handoff-v3",
    }:
        raise JointContractError("unsupported joint generation handoff schema")
    paths = manifest.get("paths")
    digests = manifest.get("digests")
    source = manifest.get("source_assets")
    if not all(isinstance(item, dict) for item in (paths, digests, source)):
        raise JointContractError("joint handoff paths/digests/source assets are incomplete")
    required_artifacts = (
        "target_tissue_mask",
        "target_nuclei_mask",
        "tissue_change",
        "cell_change",
        "joint_change",
        "generation_support",
        "contract_E_erasure",
        "contract_P_placement_centers",
        "contract_V_valid_footprints",
        "contract_S_support_context",
        "contract_M_mechanism_region",
        "contract_C_continuity_region",
        "contract_A_selected_anchor",
        "executable_contract",
    )
    if schema_version == "joint-generation-handoff-v3":
        required_artifacts += ("contract_T_population",)
    for name in required_artifacts:
        path = Path(paths.get(name, ""))
        if not path.is_file() or _sha256(path) != digests.get(name + "_sha256"):
            raise JointContractError(f"joint handoff artifact is missing or has digest drift: {name}")
    _validate_result_binding(manifest)
    contract = manifest.get("execution_contract", {}).get("executable_contract")
    if not isinstance(contract, dict):
        raise JointContractError("joint handoff lacks an executable contract")
    if contract.get("contract_id") != manifest.get("executable_contract_id"):
        raise JointContractError("joint handoff executable contract ID drift")
    # The current frozen pipeline already accepts target tissue+nuclei and a
    # separate erase/regeneration support mask. We force only the route because
    # its legacy router measures tissue diff rather than the actual generator
    # support G.
    route = route_joint_handoff(manifest, config=routing_config)
    compiled_prompt = compile_joint_render_prompt(
        manifest,
        prompt=prompt,
        dataset=dataset,
    )
    inputs = EditPipelineInputs(
        reference_image=source["image"],
        reference_tissue_mask=source["tissue"],
        reference_nuclei_mask=source["nuclei"],
        target_tissue_mask=paths["target_tissue_mask"],
        target_nuclei_mask=paths["target_nuclei_mask"],
        generation_change_region=paths["generation_support"],
        output_dir=output_dir,
        prompt=compiled_prompt,
        dataset=dataset,
        force_mode=route.mode,
        save_debug_artifacts=True,
    )
    return inputs, route, manifest


def compile_joint_render_prompt(
    manifest: dict[str, Any],
    *,
    prompt: str | None,
    dataset: str | None,
) -> str:
    """Compile reviewed render requirements into the frozen model prompt.

    Spatial authority still comes exclusively from generation support.  The
    prompt describes what the frozen observation model must render inside that
    support and what it must not invent; it never expands the editable region.
    """

    required = manifest.get("render_expectations")
    vetoes = manifest.get("render_vetoes")
    if not isinstance(required, list) or not required:
        raise JointContractError("joint handoff has no primitive-specific render expectations")
    if not isinstance(vetoes, list) or not vetoes:
        raise JointContractError("joint handoff has no primitive-specific render vetoes")
    if not all(isinstance(item, str) and item.strip() for item in required + vetoes):
        raise JointContractError("joint handoff render requirements are malformed")
    base = resolve_prompt(prompt, dataset)
    render_clause = "; ".join(item.strip() for item in required)
    veto_clause = "; ".join(item.strip() for item in vetoes)
    return (
        f"{base}. Within the provided editable generation support, render: "
        f"{render_clause}. Do not render: {veto_clause}. Preserve the image "
        "outside the provided generation support."
    )


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
    tissue_evaluator=None,
    cell_evaluator=None,
    visual_critic=None,
    post_generation_thresholds=None,
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
    from .post_generation import audit_joint_generation_handoff

    post_generation = audit_joint_generation_handoff(
        manifest_path=manifest_path,
        generated_image=result.output_dir / "final.png",
        output_path=result.output_dir / "joint_post_generation_audit.json",
        tissue_evaluator=tissue_evaluator,
        cell_evaluator=cell_evaluator,
        visual_critic=visual_critic,
        thresholds=post_generation_thresholds,
    )
    audit = {
        "schema_version": "joint-frozen-generator-route-v2",
        "case_id": manifest["case_id"],
        "candidate_id": manifest["candidate_id"],
        "route": asdict(route),
        "legacy_pipeline_selected_mode": result.selected_mode,
        "semantic_change_ratio_reported_by_legacy": result.change_ratio,
        "generation_support_is_authoritative_for_routing": True,
        "large_generation_support_is_cross_only": route.force_cross,
        "compiled_render_prompt": result.prompt,
        "render_expectations": list(manifest.get("render_expectations", ())),
        "render_vetoes": list(manifest.get("render_vetoes", ())),
        "post_generation_capability_status": post_generation.capability_status,
        "post_generation_passed": post_generation.passed,
        "post_generation_audit": str(
            result.output_dir / "joint_post_generation_audit.json"
        ),
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


def _validate_result_binding(manifest: dict[str, Any]) -> None:
    binding = manifest.get("result_binding")
    if not isinstance(binding, dict):
        raise JointContractError("joint handoff lacks a final result binding")
    binding_version = binding.get("schema_version")
    if binding_version not in {
        "joint-result-binding-v1",
        "joint-result-binding-v2",
    }:
        raise JointContractError("unsupported joint result binding schema")
    expected = dict(binding)
    observed_id = expected.pop("binding_id", None)
    calculated = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if observed_id != calculated:
        raise JointContractError("joint handoff result binding digest drift")
    if binding.get("contract_id") != manifest.get("executable_contract_id"):
        raise JointContractError("joint result binding contract ID drift")
    if binding.get("candidate_id") != manifest.get("candidate_id"):
        raise JointContractError("joint result binding candidate ID drift")
    digests = manifest.get("digests", {})
    names = {
        "target_tissue_sha256": "target_tissue_mask_sha256",
        "target_nuclei_sha256": "target_nuclei_mask_sha256",
        "tissue_change_sha256": "tissue_change_sha256",
        "cell_change_sha256": "cell_change_sha256",
        "joint_change_sha256": "joint_change_sha256",
        "generation_support_sha256": "generation_support_sha256",
    }
    if binding_version == "joint-result-binding-v2":
        names["contract_T_population_sha256"] = (
            "contract_T_population_sha256"
        )
    if any(binding.get(left) != digests.get(right) for left, right in names.items()):
        raise JointContractError("joint result binding artifact digest drift")
