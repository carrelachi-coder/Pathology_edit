"""Materialize one frozen-generator input from a fully validated mask program.

A step handoff is relative to its preceding state. Using only the final step's
G with the original H&E loses earlier edits. This adapter preserves the original
reference, final masks, and union of every validated step's support. It does not
split requests, relax mask gates, or assert image-level/pathology validity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from controlnet_train.inference.pipeline import EditPipelineInputs, resolve_prompt
from .generator_adapter import (
    JointGeneratorRoute, JointGeneratorRoutingConfig,
    build_frozen_generator_inputs, route_joint_handoff,
)
from .models import JointContractError


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    value = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.asarray(Image.open(path))
    if value.ndim != 2:
        raise JointContractError("program masks must be aligned two-dimensional label maps")
    return value


def _verified_document(path: str | Path, key: str, expected: str) -> dict:
    value = _read(path)
    payload = dict(value)
    recorded = payload.pop(key, None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode()).hexdigest()
    if digest != expected or recorded != expected:
        raise JointContractError(f"program document digest drift: {key}")
    return value


def build_frozen_program_generator_inputs(
    program_result_path: str | Path, *, output_dir: str | Path,
    dataset: str | None = None, prompt: str | None = None,
    backend: str = "auto", routing_config: JointGeneratorRoutingConfig | None = None,
) -> tuple[EditPipelineInputs, JointGeneratorRoute, dict[str, Any]]:
    """Verify the complete state chain before writing cumulative inputs.

    ``backend`` is an explicit caller selection, recorded separately from the
    automatic recommendation. It cannot bypass a support-forced Cross route.
    Counts are recorded per step; recycled instance IDs are never summed into
    a purported net cell count. Area fractions are measured against the source.
    """
    if backend not in {"auto", "inpaint", "cross"}:
        raise JointContractError("backend must be auto, inpaint or cross")
    program_result_path = Path(program_result_path).resolve()
    result = _read(program_result_path)
    evaluation = result.get("evaluation", {})
    steps = result.get("steps", [])
    if (result.get("schema_version") != "joint-edit-program-run-v1"
            or result.get("status") != "validated" or evaluation.get("passed") is not True
            or not steps or evaluation.get("completed_steps") != len(steps)
            or evaluation.get("required_steps") != len(steps)):
        raise JointContractError("generation requires a fully validated program")
    artifacts = result["artifact_paths"]
    program = _verified_document(artifacts["final_program"], "program_sha256", result["edit_program_sha256"])
    request = _verified_document(artifacts["semantic_request"], "request_sha256", result["semantic_request_sha256"])
    if (program.get("status") != "validated" or program.get("request_sha256") != request["request_sha256"]
            or len(program.get("steps", [])) != len(steps)
            or len({s["step_id"] for s in steps}) != len(steps)):
        raise JointContractError("program structure does not match its execution")

    verified = []
    cumulative_support = None
    original_image_digest = None
    previous = None
    for step, declared in zip(steps, program["steps"]):
        if (step.get("status") != "validated" or declared.get("status") != "validated"
                or any(step.get(k) != declared.get(k) for k in ("step_id", "intent_id"))
                or step.get("primitive_id") != declared.get("selected_primitive_id")
                or step.get("mechanism_id") != declared.get("selected_mechanism_id")):
            raise JointContractError("program step identity/status drift")
        paths = step["workflow_artifact_paths"]
        handoff_path = paths["handoff_manifest"]
        inputs, _, handoff = build_frozen_generator_inputs(
            handoff_path, output_dir=output_dir, dataset=dataset, prompt=prompt,
            routing_config=routing_config,
        )
        if (handoff["candidate_id"] != step["selected_candidate_id"]
                or handoff["primitive_id"] != step["primitive_id"]
                or handoff["mechanism_id"] != step["mechanism_id"]):
            raise JointContractError("program step handoff identity drift")
        reports = _read(paths["joint_gate_reports.json"])
        selected = [r for r in reports if r["candidate_id"] == step["selected_candidate_id"]]
        if (not selected or any(r.get("passed") is not True for r in selected)
                or any(c.get("passed") is not True for r in selected for c in r.get("checks", [])
                       if c.get("severity") == "hard")):
            raise JointContractError("program step does not have passing selected gates")
        context = _read(paths["case_context.json"])
        image_digest = _sha(inputs.reference_image)
        if image_digest != context.get("provenance", {}).get("source_image_sha256"):
            raise JointContractError("program source image digest drift")
        if original_image_digest is not None and image_digest != original_image_digest:
            raise JointContractError("program steps use different source images")
        original_image_digest = image_digest
        source_tissue, source_nuclei = _mask(inputs.reference_tissue_mask), _mask(inputs.reference_nuclei_mask)
        target_tissue, target_nuclei = _mask(inputs.target_tissue_mask), _mask(inputs.target_nuclei_mask)
        support = _mask(inputs.generation_change_region) > 0
        shape = source_tissue.shape
        if (any(x.shape != shape for x in (source_nuclei, target_tissue, target_nuclei, support))
                or Image.open(inputs.reference_image).size != (shape[1], shape[0])):
            raise JointContractError("program source, target and support shapes differ")
        for kind in ("tissue", "nuclei"):
            if (_sha(getattr(inputs, f"reference_{kind}_mask")) != step[f"input_{kind}_sha256"]
                    or _sha(getattr(inputs, f"target_{kind}_mask")) != step[f"output_{kind}_sha256"]):
                raise JointContractError("program state digest drift")
            if previous and step[f"input_{kind}_sha256"] != previous[f"output_{kind}_sha256"]:
                raise JointContractError("program state chain is discontinuous")
        tissue_change, cell_change = source_tissue != target_tissue, source_nuclei != target_nuclei
        if np.any((tissue_change | cell_change) & ~support):
            raise JointContractError("step generation support omits changed pixels")
        for name, actual in (("tissue_change", tissue_change), ("cell_change", cell_change),
                             ("joint_change", tissue_change | cell_change)):
            if not np.array_equal(_mask(handoff["paths"][name]) > 0, actual):
                raise JointContractError(f"step change raster disagrees with its state: {name}")
        if not np.isclose(support.mean(), handoff["ledger"]["generation_support_fraction"], atol=0, rtol=1e-12):
            raise JointContractError("step support fraction disagrees with its raster")
        cumulative_support = support.copy() if cumulative_support is None else cumulative_support | support
        verified.append((inputs, handoff, {
            "step_id": step["step_id"], "primitive_id": step["primitive_id"],
            "handoff": str(Path(handoff_path).resolve()), "handoff_sha256": _sha(handoff_path),
            "gate_report_sha256": _sha(paths["joint_gate_reports.json"]),
            "case_context_sha256": _sha(paths["case_context.json"]),
            "area_budget": context.get("joint_area_budget"),
            "cell_count_budget": context.get("cell_count_extent_budget"),
            "realized_ledger": handoff["ledger"],
        }))
        previous = step
    first, last = verified[0][0], verified[-1][0]
    tissue_change = _mask(first.reference_tissue_mask) != _mask(last.target_tissue_mask)
    cell_change = _mask(first.reference_nuclei_mask) != _mask(last.target_nuclei_mask)
    joint_change = tissue_change | cell_change
    if not np.any(joint_change):
        raise JointContractError("program final state is a no-op relative to its source")
    if np.any(joint_change & ~cumulative_support):
        raise JointContractError("cumulative support omits final changes")
    ledger = {"tissue_fraction": float(tissue_change.mean()), "cell_fraction": float(cell_change.mean()),
              "joint_fraction": float(joint_change.mean()), "generation_support_fraction": float(cumulative_support.mean())}
    route = route_joint_handoff({"ledger": ledger, "primitive_id": verified[0][1]["primitive_id"]
                                if len({x[1]["primitive_id"] for x in verified}) == 1 else "validated-edit-program"}, config=routing_config)
    automatic_route = asdict(route)
    if backend == "inpaint" and route.force_cross:
        raise JointContractError("explicit inpaint cannot override support-forced Cross")
    if backend != "auto":
        route = replace(route, mode=backend, reason=f"caller explicitly selected {backend}; automatic recommendation: {automatic_route['mode']}")
    out = Path(output_dir).resolve()
    manifest_path = out / "program_generation_manifest.json"
    if manifest_path.exists():
        raise JointContractError("program generation export already exists; use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    mask_paths = {}
    for name, value in (("generation_support", cumulative_support), ("tissue_change", tissue_change),
                        ("cell_change", cell_change), ("joint_change", joint_change)):
        path = out / (name + ".png")
        Image.fromarray(value.astype(np.uint8) * 255).save(path)
        mask_paths[name] = str(path)
    # Do not concatenate intermediate 'increase'/'decrease' instructions: later
    # steps may deliberately reverse earlier ones. Final masks own the outcome.
    compiled_prompt = first.prompt if len(verified) == 1 else (
        f"{resolve_prompt(prompt, dataset)}. Render the final tissue and nucleus condition "
        "masks within the supplied generation support. Preserve the image outside that support."
    )
    inputs = EditPipelineInputs(reference_image=first.reference_image,
        reference_tissue_mask=first.reference_tissue_mask, reference_nuclei_mask=first.reference_nuclei_mask,
        target_tissue_mask=last.target_tissue_mask, target_nuclei_mask=last.target_nuclei_mask,
        generation_change_region=mask_paths["generation_support"], output_dir=out / "generated",
        prompt=compiled_prompt, dataset=dataset, force_mode=route.mode, save_debug_artifacts=True)
    payload = json.loads(json.dumps(asdict(inputs), default=str))
    input_assets = {k: str(v) for k,v in payload.items() if k.startswith(("reference_", "target_"))}
    input_assets.update(mask_paths)
    manifest = {"schema_version": "joint-program-generation-handoff-v1",
        "program_result": str(program_result_path), "program_result_sha256": _sha(program_result_path),
        "semantic_request_sha256": request["request_sha256"], "edit_program_sha256": program["program_sha256"],
        "steps": [x[2] for x in verified], "generator_inputs": payload,
        "artifact_digests": {k: _sha(v) for k,v in input_assets.items()},
        "cumulative_ledger": ledger, "support_policy": "union_of_all_validated_step_supports",
        "area_reference": "original_source_to_final_target", "count_policy": "per_step_events_only_no_net_instance_count_claim",
        "automatic_route": automatic_route, "selected_route": asdict(route),
        "route_selection": "automatic" if backend == "auto" else "explicit_caller_selection",
        "claim_scope": "validated_mask_program_only", "independent_image_checks": "not_run"}
    (out / "generator_inputs.json").write_text(json.dumps(payload, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return inputs, route, manifest
