"""Read-only subprocess adapter for the mature online ProbNet cell pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from inpaint_cells.instance_authority import write_instance_authority

from .cell_layouts import CellLayoutResult, build_reference_shape_library
from .executable_contract import ExecutableJointContract
from .instance_authority import build_scene_instance_authority
from .models import JointContractError
from .nuclei import load_nuclei_mask, to_raw_nuclei_mask
from .scene import JointSceneAnalysis
from .seam import (
    compile_continuity_center_quota,
    compile_executable_continuity_count,
    target_cell_class_for_tissue,
)

MATURE_EXECUTION_VERSION = "online-probnet-mature-v20"


@dataclass(frozen=True)
class MatureProbNetConfig:
    """Runtime assets are explicit; annotation profile never implies dataset."""

    dataset_name: str
    checkpoint: str
    instance_library: str
    device: str = "auto"
    base_channels: int = 64
    python_executable: str = sys.executable
    # Touching semantic nuclei are already separated by the same watershed as
    # the joint scene graph.  A second median-ratio filter would remove valid
    # pleomorphic nuclei and make the sampler's size ruler disagree with the
    # gate.  Complete-instance size calibration still applies the scene's
    # explicit merged-suspect rule.
    reference_shape_max_area_ratio: float = 0.0


class MatureProbNetCellExecutor:
    """Execute the unchanged mature CLI under a compiled E/P/V/S contract."""

    name = MATURE_EXECUTION_VERSION

    def __init__(
        self,
        config: MatureProbNetConfig,
        *,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.config = config
        self._runner = runner

    def validate_assets(self) -> None:
        for label, value in (
            ("ProbNet checkpoint", self.config.checkpoint),
            ("nucleus instance library", self.config.instance_library),
        ):
            path = Path(value)
            if label.endswith("library"):
                valid = path.is_dir()
            else:
                valid = path.is_file()
            if not valid:
                raise JointContractError(f"{label} is unavailable: {path}")

    def supports(self, contract: ExecutableJointContract) -> bool:
        contract.validate_identity()
        if self.config.dataset_name.lower() != contract.population_dataset_name.lower():
            raise JointContractError(
                "mature ProbNet dataset does not match the cell population profile"
            )
        return (
            contract.cell_program.baseline_mode
            == "regenerate_target_population"
        )

    def build_command(
        self,
        *,
        seed: int,
        target_tissue_path: Path,
        source_tissue_path: Path,
        source_nuclei_path: Path,
        reference_nuclei_shapes_path: Path,
        generation_region_path: Path,
        population_region_path: Path,
        placement_region_path: Path,
        erasure_region_path: Path,
        required_placement_region_path: Path | None,
        packing_witness_path: Path | None,
        minimum_required_placements: int,
        maximum_required_placements: int | None,
        required_nucleus_class: int | None,
        output_path: Path,
        prohibited_tissue_ids: tuple[int, ...],
        allowed_new_cell_classes: tuple[int, ...],
        source_instance_authority_path: Path | None = None,
    ) -> list[str]:
        mature_device = (
            "cuda"
            if str(self.config.device).lower().startswith("cuda")
            else self.config.device
        )
        command = [
            self.config.python_executable,
            "-m",
            "inpaint_cells.generate",
            "--dataset",
            self.config.dataset_name,
            "--ckpt",
            self.config.checkpoint,
            "--library",
            self.config.instance_library,
            "--base-ch",
            str(self.config.base_channels),
            "--device",
            mature_device,
            "--seed",
            str(seed),
            "--input-tissue",
            str(target_tissue_path),
            "--reference-tissue",
            str(source_tissue_path),
            "--input-nuclei",
            str(source_nuclei_path),
            "--reference-nuclei-shapes",
            str(reference_nuclei_shapes_path),
            "--edit-region",
            str(generation_region_path),
            "--population-region",
            str(population_region_path),
            "--placement-region",
            str(placement_region_path),
            "--deletion-region",
            str(erasure_region_path),
            "--trust-complete-deletion-region",
            "--output",
            str(output_path),
            "--no-widen-edit-region",
            "--require-sampling-audit",
            "--require-exact-target-count",
            "--require-full-tissue-containment",
            "--reference-shape-max-area-ratio",
            str(self.config.reference_shape_max_area_ratio),
            "--allowed-nucleus-types",
            *[
                str(100 + int(value))
                for value in allowed_new_cell_classes
            ],
        ]
        if source_instance_authority_path is not None:
            command.extend(
                ["--source-instance-authority", str(source_instance_authority_path)]
            )
        if prohibited_tissue_ids:
            command.extend(
                ["--skip-tissue-ids", *[str(value) for value in prohibited_tissue_ids]]
            )
        if required_placement_region_path is not None:
            command.extend(
                [
                    "--required-placement-region",
                    str(required_placement_region_path),
                    "--minimum-required-placements",
                    str(max(0, int(minimum_required_placements))),
                ]
            )
        if packing_witness_path is not None:
            command.extend(
                ["--packing-witness", str(packing_witness_path)]
            )
        if maximum_required_placements is not None:
            command.extend(
                [
                    "--maximum-required-placements",
                    str(int(maximum_required_placements)),
                ]
            )
        if required_nucleus_class is not None:
            command.extend(
                [
                    "--required-nucleus-type",
                    str(100 + int(required_nucleus_class)),
                ]
            )
        return command

    def execute(
        self,
        *,
        contract: ExecutableJointContract,
        source_tissue: np.ndarray,
        target_tissue: np.ndarray,
        source_nuclei: np.ndarray,
        scene: JointSceneAnalysis,
        output_dir: str | Path,
        prohibited_tissue_ids: tuple[int, ...],
        seed: int,
        variants: int,
    ) -> tuple[CellLayoutResult, ...]:
        if not self.supports(contract):
            raise JointContractError(
                "mature ProbNet baseline only realizes target-population regeneration; "
                "structured mechanism layouts require the deterministic layout executor"
            )
        self.validate_assets()
        program = contract.cell_program
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target_tissue_path = directory / "target_tissue.png"
        source_tissue_path = directory / "source_tissue.png"
        source_nuclei_path = directory / "source_nuclei.png"
        reference_nuclei_shapes_path = directory / "reference_nuclei_shapes.png"
        source_instance_authority_path = directory / "source_instance_authority.json"
        generation_path = directory / "generation_region.png"
        population_path = directory / "population_target_region.png"
        placement_path = directory / "placement_region.png"
        erasure_path = directory / "erasure_region.png"
        required_placement_path = directory / "required_placement_region.png"
        packing_witness_path = directory / "packing_witness.json"
        _save_mask(target_tissue_path, target_tissue)
        _save_mask(source_tissue_path, source_tissue)
        raw_source_nuclei = to_raw_nuclei_mask(source_nuclei)
        _save_mask(source_nuclei_path, raw_source_nuclei)
        authority = build_scene_instance_authority(scene, source_nuclei)
        write_instance_authority(source_instance_authority_path, authority)
        # ``--edit-region`` is the mature CLI model-support domain and must
        # contain T_pop, every legal center in P, and every complete-instance
        # deletion pixel in E. T_pop is an abundance denominator, not a center
        # mask; P remains the only authority for accepted new centers.
        mature_generation_region = (
            np.asarray(program.population_target_region, dtype=bool)
            | np.asarray(program.placement_center_region, dtype=bool)
            | np.asarray(program.erasure_region, dtype=bool)
        )
        if np.any(mature_generation_region & ~program.support_context_region):
            raise JointContractError(
                "mature ProbNet generation region exceeds executable support"
            )
        _save_binary(generation_path, mature_generation_region)
        _save_binary(population_path, program.population_target_region)
        _save_binary(placement_path, program.placement_center_region)
        _save_binary(erasure_path, program.erasure_region)
        packing_witness = _compile_packing_witness(
            contract=contract,
            scene=scene,
        )
        packing_witness_path.write_text(
            json.dumps(packing_witness, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        minimum_required_placements = int(
            bool(program.continuity_requires_new_target_cells)
            and np.any(program.continuity_region)
        )
        maximum_required_placements = None
        required_nucleus_class = None
        continuity_quota = None
        recomputed_continuity_count = 0
        if minimum_required_placements:
            required_nucleus_class = target_cell_class_for_tissue(
                contract.target_label,
                None,
            )
            retained_for_quota = np.asarray(source_nuclei).copy()
            retained_for_quota[
                np.asarray(program.erasure_region, dtype=bool)
            ] = 0
            continuity_quota = compile_continuity_center_quota(
                nuclei_mask=retained_for_quota,
                target_tissue_mask=target_tissue,
                tissue_change=(
                    np.asarray(source_tissue) != np.asarray(target_tissue)
                ),
                continuity_region=program.continuity_region,
                continuity_anchor_mask=program.continuity_anchor_mask,
                continuity_width_px=program.continuity_width_px,
                density_ratio_range=program.continuity_density_ratio_range,
                requires_new_target_cells=(
                    program.continuity_requires_new_target_cells
                ),
                target_class=required_nucleus_class,
                target_fine_ids=contract.target_host_fine_ids,
            )
            minimum_required_placements = compile_executable_continuity_count(
                continuity_quota,
                anchor_pixels=int(
                    np.count_nonzero(program.continuity_anchor_mask)
                ),
                maximum_empty_run_px=(
                    program.continuity_maximum_empty_run_px
                ),
                minimum_anchor_coverage_fraction=(
                    program.continuity_minimum_anchor_coverage_fraction
                ),
            )
            # Feasibility has already compiled the continuous density/seam
            # estimates into an exact complete-footprint witness. That
            # immutable integer is the execution authority. Recomputing and
            # using a different seam quota here can make the remainder quota
            # impossible even though this candidate was certified.
            recomputed_continuity_count = minimum_required_placements
            minimum_required_placements = int(
                packing_witness.get("required_seam_count", 0)
            )
            maximum_required_placements = minimum_required_placements
            _save_binary(required_placement_path, program.continuity_region)

        eligible: set[str] = set()
        reference_supported_classes: set[int] = set()
        rejected: dict[str, str] = {}
        for class_id in program.target_classes:
            references, current_rejected = build_reference_shape_library(
                scene, class_id=class_id
            )
            eligible.update(item.instance_id for item in references)
            if references:
                reference_supported_classes.add(int(class_id))
            rejected.update(current_rejected)
        # Protection is a pixel-mutation contract, not a ban on read-only
        # shape reuse. Complete, non-border protected nuclei are often the
        # best same-patch references for sparse target populations. Copying a
        # footprint does not alter the source instance.
        if not eligible:
            raise JointContractError(
                "mature ProbNet execution has no complete non-border reference shape"
            )
        executable_new_cell_classes = tuple(
            item
            for item in contract.allowed_new_cell_classes
            if item in reference_supported_classes
        )
        if required_nucleus_class is not None and (
            required_nucleus_class not in executable_new_cell_classes
        ):
            raise JointContractError(
                "the compiled seam class has no complete same-patch reference shape"
            )
        if not executable_new_cell_classes:
            raise JointContractError(
                "no allowed new cell class has a complete same-patch reference shape"
            )
        reference_nuclei = np.asarray(source_nuclei).copy()
        for instance in scene.cells.instances:
            if instance.instance_id not in eligible:
                reference_nuclei[
                    np.asarray(scene.instance_masks[instance.instance_id], dtype=bool)
                ] = 0
        _save_mask(
            reference_nuclei_shapes_path,
            to_raw_nuclei_mask(reference_nuclei),
        )

        checkpoint_digest = _sha256(Path(self.config.checkpoint))
        instance_library_digest = _tree_sha256(
            Path(self.config.instance_library)
        )
        results = []
        rejected_variants = []
        for variant in range(variants):
            current_seed = int(seed + variant * 104729)
            output_path = directory / f"nuclei_{variant + 1:02d}.png"
            command = self.build_command(
                seed=current_seed,
                target_tissue_path=target_tissue_path,
                source_tissue_path=source_tissue_path,
                source_nuclei_path=source_nuclei_path,
                reference_nuclei_shapes_path=reference_nuclei_shapes_path,
                source_instance_authority_path=source_instance_authority_path,
                generation_region_path=generation_path,
                population_region_path=population_path,
                placement_region_path=placement_path,
                erasure_region_path=erasure_path,
                required_placement_region_path=(
                    required_placement_path
                    if minimum_required_placements
                    else None
                ),
                packing_witness_path=packing_witness_path,
                minimum_required_placements=minimum_required_placements,
                maximum_required_placements=maximum_required_placements,
                required_nucleus_class=required_nucleus_class,
                output_path=output_path,
                prohibited_tissue_ids=prohibited_tissue_ids,
                allowed_new_cell_classes=executable_new_cell_classes,
            )
            completed = self._runner(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                sampling_rejection = _probnet_sampling_audit_rejection(
                    completed=completed,
                    output_path=output_path,
                )
                if sampling_rejection is not None:
                    rejected_variants.append(
                        {
                            "variant": variant + 1,
                            "seed": current_seed,
                            **sampling_rejection,
                        }
                    )
                    continue
                raise JointContractError(
                    "mature ProbNet cell execution failed: "
                    + (completed.stderr or completed.stdout)[-2000:]
                )
            diagnostic_path = output_path.with_suffix(".diagnostics.json")
            if not output_path.is_file() or not diagnostic_path.is_file():
                raise JointContractError(
                    "mature ProbNet did not produce mask and diagnostics atomically"
                )
            diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            if not isinstance(diagnostics, list) or not diagnostics:
                raise JointContractError("mature ProbNet diagnostics are malformed")
            selected = diagnostics[0]
            audit = selected.get("sampling_audit") or {}
            if audit.get("passed") is not True:
                rejected_variants.append(
                    {
                        "variant": variant + 1,
                        "seed": current_seed,
                        "reasons": _sampling_audit_reason_codes(audit),
                        "sampling_audit": audit,
                    }
                )
                continue
            target = load_nuclei_mask(output_path)
            accepted_center_ledger = _accepted_center_ledger(selected)
            accepted_instance_area_ledger = _accepted_instance_area_ledger(
                selected
            )
            placed_by_shape_source = {
                "same_patch_complete_instance": 0,
                "calibrated_instance_library": 0,
                "unknown": 0,
            }
            placed_by_target_class: dict[str, int] = {}
            for item in accepted_instance_area_ledger:
                raw_source = str(item.get("shape_source") or "unknown")
                if raw_source in {
                    "same_patch",
                    "same_patch_complete_instance",
                    "reference_patch",
                }:
                    source_key = "same_patch_complete_instance"
                elif raw_source in {
                    "library",
                    "calibrated_library",
                    "instance_library",
                }:
                    source_key = "calibrated_instance_library"
                else:
                    source_key = "unknown"
                placed_by_shape_source[source_key] += 1
                class_key = str(int(item["class_id"]))
                placed_by_target_class[class_key] = (
                    placed_by_target_class.get(class_key, 0) + 1
                )
            contract_errors = contract.validate_candidate(
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
                target_tissue=target_tissue,
                target_nuclei=target,
                tissue_change=(
                    np.asarray(source_tissue) != np.asarray(target_tissue)
                ),
                cell_change=(
                    np.asarray(source_nuclei) != np.asarray(target)
                ),
                scene=scene,
                new_cell_center_ledger=accepted_center_ledger,
            )
            if contract_errors:
                # Variants are independent paired candidates. One stochastic
                # layout that crosses S/P/V must be rejected at candidate
                # granularity; it must not erase sibling variants that already
                # passed the immutable contract. Sampling-audit failures are
                # likewise variant-local because each seed has an independent
                # spatial draw. Only a genuine subprocess/artifact failure is
                # systemic and fails the whole executor above.
                rejected_variants.append(
                    {
                        "variant": variant + 1,
                        "seed": current_seed,
                        "reasons": list(contract_errors),
                    }
                )
                continue
            desired = sum(
                int(item.get("target_count", 0))
                for item in (selected.get("tissues") or {}).values()
                if isinstance(item, dict)
            )
            placed = int(selected.get("placed", desired))
            modifier_certificate = _mechanism_modifier_certificate(
                mechanism_program_id=program.mechanism_program_id,
                accepted_center_ledger=accepted_center_ledger,
                mechanism_region=program.mechanism_region,
                continuity_region=program.continuity_region,
                required_nucleus_class=required_nucleus_class,
                minimum_required_placements=minimum_required_placements,
                sampling_audit_passed=audit.get("passed") is True,
            )
            packing_witness_used = any(
                bool(
                    (
                        (item.get("exact_count_backfill") or {}).get(
                            "packing_witness_fallback"
                        )
                        or {}
                    ).get("used", False)
                )
                for item in (selected.get("tissues") or {}).values()
                if isinstance(item, dict)
            )
            results.append(
                CellLayoutResult(
                    cell_candidate_id=f"mature-cells-{variant + 1:02d}",
                    target_nuclei_mask=target,
                    trace={
                        "layout_tool_version": MATURE_EXECUTION_VERSION,
                        "execution_engine": MATURE_EXECUTION_VERSION,
                        "execution_program_id": contract.execution_program_id,
                        "production_density_calibrated": True,
                        "mature_probnet_contract": True,
                        "ranker": "frozen_probnet_context_stabilized_spatial_sampler",
                        "ranker_provenance": {
                            "checkpoint_sha256": checkpoint_digest,
                            "instance_library_sha256": instance_library_digest,
                            "dataset_name": self.config.dataset_name,
                            "role": "mature_count_type_spatial_shape_pipeline",
                        },
                        "compiled_cell_tool_program": program.to_metadata(),
                        "executable_contract_id": contract.contract_id,
                        "executable_contract_version": contract.schema_version,
                        "executable_contract": contract.to_metadata(),
                        "desired_count": desired,
                        "biological_desired_count": desired,
                        "geometric_capacity_estimate": desired,
                        "resolved_count": placed,
                        "requested_count": desired,
                        "attempted_count": desired,
                        "placed_count": placed,
                        "batch_max_attainable_count": placed,
                        "capacity_max_count": placed,
                        "cell_capacity_certified": placed == desired,
                        "cell_capacity_fallback_used": packing_witness_used,
                        "packing_witness_fallback": {
                            "used": packing_witness_used,
                            "version": packing_witness.get("version"),
                            "contract_id": packing_witness.get("contract_id"),
                            "policy": (
                                "probnet_ranked_contract_owned_complete_source_shapes"
                            ),
                        },
                        "placement_capacity_exhausted": placed < desired,
                        "reference_shape_ids": sorted(eligible),
                        "reference_shape_rejections": rejected,
                        "contract_allowed_new_cell_classes": list(
                            contract.allowed_new_cell_classes
                        ),
                        "reference_supported_new_cell_classes": list(
                            executable_new_cell_classes
                        ),
                        "reference_shape_integrity_certified": True,
                        "reference_first": True,
                        "shape_sampling": selected.get("shape_sampling", {}),
                        "patch_adaptive_priors": selected.get(
                            "patch_adaptive_priors", {}
                        ),
                        "source_instance_authority": {
                            "schema_version": authority["schema_version"],
                            "authority_sha256": authority["authority_sha256"],
                            "observation_quality": authority[
                                "observation_quality"
                            ],
                            "instance_count": len(authority["instances"]),
                        },
                        "spatial_prior": selected.get("spatial_prior", {}),
                        "sampling_audit": audit,
                        "mechanism_modifier_certified": bool(
                            modifier_certificate["passed"]
                        ),
                        "mechanism_modifier_certificate": modifier_certificate,
                        "sampling_feedback": selected.get("sampling_feedback", {}),
                        "accepted_center_ledger": [
                            {"row": row, "col": col, "class_id": class_id}
                            for row, col, class_id in accepted_center_ledger
                        ],
                        "placements": _architecture_placement_trace(
                            contract=contract,
                            accepted_center_ledger=accepted_center_ledger,
                        ),
                        "accepted_instance_area_ledger": (
                            accepted_instance_area_ledger
                        ),
                        "placed_by_shape_source_counts": (
                            placed_by_shape_source
                        ),
                        "placed_by_target_class_counts": (
                            placed_by_target_class
                        ),
                        "mature_generation_region_policy": (
                            "population_T_pop_union_placement_P_union_erasure_E"
                        ),
                        "population_quota_region_policy": (
                            "target_tissue_population_area_T_pop_not_placement_P"
                        ),
                        "population_target_region_pixels": int(
                            np.count_nonzero(program.population_target_region)
                        ),
                        "placement_center_region_pixels": int(
                            np.count_nonzero(program.placement_center_region)
                        ),
                        "required_placement_region_pixels": int(
                            np.count_nonzero(program.continuity_region)
                            if minimum_required_placements
                            else 0
                        ),
                        "minimum_required_placements": (
                            minimum_required_placements
                        ),
                        "maximum_required_placements": (
                            maximum_required_placements
                        ),
                        "required_nucleus_class": required_nucleus_class,
                        "compiled_continuity_quota": (
                            {
                                "minimum_count": continuity_quota.minimum_count,
                                "maximum_count": continuity_quota.maximum_count,
                                "target_count": continuity_quota.target_count,
                                "selected_count": minimum_required_placements,
                                "recomputed_count_before_contract_binding": (
                                    recomputed_continuity_count
                                ),
                                "packing_certificate_count": int(
                                    packing_witness.get(
                                        "required_seam_count", 0
                                    )
                                ),
                                "selection_policy": (
                                    "packing_certificate_is_immutable_"
                                    "execution_authority_v2"
                                ),
                                "expected_count": continuity_quota.expected_count,
                                "outer_count": continuity_quota.outer_count,
                                "outer_pixels": continuity_quota.outer_pixels,
                                "inner_pixels": continuity_quota.inner_pixels,
                            }
                            if continuity_quota is not None
                            else None
                        ),
                        "continuity_placement_policy": (
                            "typed_seam_quota_then_full_P_population_remainder_v3"
                            if minimum_required_placements
                            else "not_required"
                        ),
                        "mature_generation_region_pixels": int(
                            np.count_nonzero(mature_generation_region)
                        ),
                        "overlap_pixels": 0,
                        "partial_source_instance_edits": 0,
                        "removed_source_instance_ids": list(
                            contract.erase_instance_ids
                        ),
                        "protected_instance_ids": list(
                            contract.protected_instance_ids
                        ),
                        "cross_domain_fallback": False,
                        "seed": current_seed,
                        "command_sha256": hashlib.sha256(
                            "\0".join(command).encode("utf-8")
                        ).hexdigest(),
                    },
                )
            )
        if not results:
            reasons = sorted(
                {
                    reason
                    for item in rejected_variants
                    for reason in item["reasons"]
                }
            )
            raise JointContractError(
                "all mature ProbNet variants failed sampling audit or violated "
                "the executable contract: "
                + "; ".join(reasons)
            )
        for result in results:
            result.trace["rejected_variant_audit"] = list(rejected_variants)
        return tuple(results)


_SAMPLING_AUDIT_FAILURE_SENTINEL = (
    "ProbNet count/type/spatial sampling audit failed"
)


def _sampling_audit_reason_codes(audit: dict) -> list[str]:
    reasons = [str(item) for item in (audit.get("failure_reasons") or ())]
    if not reasons and audit.get("primary_failure_reason"):
        reasons = [str(audit["primary_failure_reason"])]
    if not reasons:
        reasons = ["sampling_audit_failed"]
    return [f"sampling_audit:{item}" for item in sorted(set(reasons))]


def _probnet_sampling_audit_rejection(
    *,
    completed: subprocess.CompletedProcess,
    output_path: Path,
) -> dict | None:
    """Recognize a complete but audit-rejected stochastic variant.

    The mature CLI intentionally exits nonzero when ``--require-sampling-audit``
    rejects a spatial draw, after it has atomically written the mask and its
    diagnostics. That is candidate evidence, not a process failure. We only
    downgrade the exact audited condition; missing/malformed artifacts and all
    unrelated subprocess errors remain fail-closed in the caller.
    """

    process_text = (completed.stderr or completed.stdout or "")
    if _SAMPLING_AUDIT_FAILURE_SENTINEL not in process_text:
        return None
    diagnostic_path = output_path.with_suffix(".diagnostics.json")
    if not output_path.is_file() or not diagnostic_path.is_file():
        return None
    try:
        diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(diagnostics, list) or not diagnostics:
        return None
    selected = diagnostics[0]
    if not isinstance(selected, dict):
        return None
    audit = selected.get("sampling_audit") or {}
    if not isinstance(audit, dict) or audit.get("passed") is not False:
        return None
    return {
        "reasons": _sampling_audit_reason_codes(audit),
        "sampling_audit": audit,
        "subprocess_return_code": int(completed.returncode),
    }


def _compile_packing_witness(
    *,
    contract: ExecutableJointContract,
    scene: JointSceneAnalysis,
) -> dict:
    """Materialize the contract-owned footprint witness for mature fallback.

    Coordinates and instance IDs come from the exact pre-ProbNet packing
    solver.  Shape pixels are copied only from complete, non-border source
    instances already authorized by that immutable contract.
    """

    certificate = contract.packing_certificate
    if not certificate:
        raise JointContractError(
            "mature execution requires a bound packing certificate"
        )
    metadata = {item.instance_id: item for item in scene.cells.instances}
    placements = []
    for item in certificate.get("placements") or []:
        instance_id = str(item["reference_instance_id"])
        nucleus = metadata.get(instance_id)
        component = scene.instance_masks.get(instance_id)
        if nucleus is None or component is None:
            raise JointContractError(
                f"packing witness references unknown instance {instance_id}"
            )
        class_id = int(item["class_id"])
        if nucleus.class_id != class_id:
            raise JointContractError(
                "packing witness class differs from source reference class"
            )
        x0, y0, x1, y1 = nucleus.bbox_xyxy
        shape = np.asarray(component, dtype=bool)[y0:y1, x0:x1]
        if not np.any(shape) or int(np.count_nonzero(shape)) != int(
            item["area_px"]
        ):
            raise JointContractError(
                "packing witness source footprint is incomplete or mutated"
            )
        offsets = np.argwhere(shape) - np.asarray(
            [shape.shape[0] // 2, shape.shape[1] // 2]
        )
        placements.append(
            {
                "row": int(item["row"]),
                "col": int(item["col"]),
                "nucleus_type": 100 + class_id,
                "reference_instance_id": instance_id,
                "required_seam": bool(item.get("required_seam", False)),
                "offsets_yx": offsets.astype(int).tolist(),
            }
        )
    if len(placements) != int(certificate["requested_count"]):
        raise JointContractError(
            "packing witness does not realize its certified count"
        )
    return {
        "version": "compiled-packing-witness-v4",
        "contract_id": contract.contract_id,
        "certificate_version": certificate.get("version"),
        "requested_count": int(certificate["requested_count"]),
        "required_seam_count": int(
            certificate.get("required_seam_count", 0)
        ),
        "class_reference_median_area_px": (
            _mature_nucleus_area_medians(certificate)
        ),
        "local_median_area_ratio_interval": list(
            certificate.get("local_median_area_ratio_interval")
            or [0.60, 1.67]
        ),
        "placements": placements,
    }


def _architecture_placement_trace(
    *,
    contract: ExecutableJointContract,
    accepted_center_ledger: tuple[tuple[int, int, int], ...],
) -> list[dict]:
    """Expose mature replay as the same cord/nest group audited in research mode."""

    if contract.primitive_id not in {
        "invasive-cord-formation-v1",
        "peritumoral-tumor-nest-formation-v1",
    }:
        return []
    expected = {
        (int(item["row"]), int(item["col"]), int(item["class_id"]))
        for item in (contract.packing_certificate or {}).get("placements", ())
    }
    realized = {
        (int(row), int(col), int(class_id))
        for row, col, class_id in accepted_center_ledger
    }
    if not expected or realized != expected:
        return []
    group_size = len(accepted_center_ledger)
    orientation = (
        "cell_seeded_invasion_path"
        if contract.primitive_id == "invasive-cord-formation-v1"
        else "detached_island_population"
    )
    return [
        {
            "center_xy": [int(col), int(row)],
            "cell_class": int(class_id),
            "cluster_id": "certified-architecture-0001",
            "planned_cluster_size": int(group_size),
            "cluster_size": int(group_size),
            "orientation_policy": orientation,
            "packing_witness_replayed": True,
            "execution_engine": MATURE_EXECUTION_VERSION,
        }
        for row, col, class_id in accepted_center_ledger
    ]


def _mature_nucleus_area_medians(certificate: dict) -> dict[str, float]:
    """Map CellViT class IDs onto the mature sampler's 100-series schema."""

    return {
        str(100 + int(class_id)): float(value)
        for class_id, value in (
            certificate.get("class_reference_median_area_px") or {}
        ).items()
    }


def _mechanism_modifier_certificate(
    *,
    mechanism_program_id: str,
    accepted_center_ledger: tuple[tuple[int, int, int], ...],
    mechanism_region: np.ndarray,
    continuity_region: np.ndarray,
    required_nucleus_class: int | None,
    minimum_required_placements: int,
    sampling_audit_passed: bool,
) -> dict:
    """Prove what the mature sampler actually realized for typed programs.

    ProbNet remains the ranker and mature density sampler.  This certificate
    prevents a mechanism label from being accepted merely because it appeared
    in a Planner response. Boundary programs bind the typed seam quota; the
    remaining population-replacement centers may occupy the larger legal P/T
    domain. Dense-sheet programs still require every center in their mechanism
    region.
    """

    mechanism = np.asarray(mechanism_region, dtype=bool)
    continuity = np.asarray(continuity_region, dtype=bool)
    inside_mechanism = 0
    typed_continuity = 0
    outside = []
    for row, col, class_id in accepted_center_ledger:
        if 0 <= row < mechanism.shape[0] and 0 <= col < mechanism.shape[1]:
            if mechanism[row, col]:
                inside_mechanism += 1
            else:
                outside.append([int(row), int(col), int(class_id)])
            if (
                continuity[row, col]
                and (
                    required_nucleus_class is None
                    or int(class_id) == int(required_nucleus_class)
                )
            ):
                typed_continuity += 1
        else:
            outside.append([int(row), int(col), int(class_id)])
    if mechanism_program_id == "boundary_aligned":
        passed = bool(
            sampling_audit_passed
            and typed_continuity >= minimum_required_placements
            and minimum_required_placements > 0
        )
        policy = (
            "typed_seam_quota_in_continuity_band_with_population_remainder_in_P"
        )
    elif mechanism_program_id == "dense_sheet":
        passed = bool(
            sampling_audit_passed
            and accepted_center_ledger
            and not outside
        )
        policy = "all_new_centers_in_compiled_dense_population_region"
    else:
        passed = False
        policy = "baseline_or_non_mature_modifier"
    return {
        "schema_version": "mature-mechanism-modifier-certificate-v2",
        "program_id": mechanism_program_id,
        "policy": policy,
        "passed": passed,
        "accepted_center_count": len(accepted_center_ledger),
        "inside_mechanism_count": inside_mechanism,
        "outside_mechanism_centers": outside,
        "typed_continuity_count": typed_continuity,
        "minimum_typed_continuity_count": minimum_required_placements,
        "required_nucleus_class": required_nucleus_class,
        "sampling_audit_passed": bool(sampling_audit_passed),
    }


def _save_mask(path: Path, mask: np.ndarray) -> None:
    array = np.asarray(mask)
    if array.ndim != 2 or array.min(initial=0) < 0 or array.max(initial=0) > 255:
        raise JointContractError("mature ProbNet PNG masks require 2-D IDs in [0,255]")
    Image.fromarray(array.astype(np.uint8)).save(path)


def _save_binary(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def _accepted_center_ledger(
    diagnostics: dict,
) -> tuple[tuple[int, int, int], ...]:
    result = []
    for tissue in (diagnostics.get("tissues") or {}).values():
        if not isinstance(tissue, dict):
            continue
        for item in tissue.get("accepted_centers") or ():
            if not isinstance(item, dict):
                continue
            raw_class = int(item.get("nucleus_type", 0))
            class_id = raw_class - 100 if raw_class >= 100 else raw_class
            result.append((int(item["row"]), int(item["col"]), class_id))
    return tuple(result)


def _accepted_instance_area_ledger(diagnostics: dict) -> list[dict]:
    """Return the realized, transformed footprint of every accepted nucleus."""

    result = []
    for tissue in (diagnostics.get("tissues") or {}).values():
        if not isinstance(tissue, dict):
            continue
        for item in tissue.get("accepted_centers") or ():
            if not isinstance(item, dict):
                continue
            raw_class = int(item.get("nucleus_type", 0))
            class_id = raw_class - 100 if raw_class >= 100 else raw_class
            area_px = int(item.get("area_px", 0))
            if class_id <= 0 or area_px <= 0:
                continue
            result.append(
                {
                    "row": int(item["row"]),
                    "col": int(item["col"]),
                    "class_id": int(class_id),
                    "area_px": area_px,
                    "shape_source": str(item.get("shape_source", "unknown")),
                }
            )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
