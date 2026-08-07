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

from .cell_layouts import CellLayoutResult, build_reference_shape_library
from .executable_contract import ExecutableJointContract
from .models import JointContractError
from .nuclei import load_nuclei_mask, to_raw_nuclei_mask
from .scene import JointSceneAnalysis
from .seam import (
    compile_continuity_center_quota,
    target_cell_class_for_tissue,
)

MATURE_EXECUTION_VERSION = "online-probnet-mature-v7"


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
        minimum_required_placements: int,
        maximum_required_placements: int | None,
        required_nucleus_class: int | None,
        output_path: Path,
        prohibited_tissue_ids: tuple[int, ...],
        allowed_new_cell_classes: tuple[int, ...],
    ) -> list[str]:
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
            self.config.device,
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
        generation_path = directory / "generation_region.png"
        population_path = directory / "population_target_region.png"
        placement_path = directory / "placement_region.png"
        erasure_path = directory / "erasure_region.png"
        required_placement_path = directory / "required_placement_region.png"
        _save_mask(target_tissue_path, target_tissue)
        _save_mask(source_tissue_path, source_tissue)
        _save_mask(source_nuclei_path, to_raw_nuclei_mask(source_nuclei))
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
        minimum_required_placements = int(
            bool(program.continuity_requires_new_target_cells)
            and np.any(program.continuity_region)
        )
        maximum_required_placements = None
        required_nucleus_class = None
        continuity_quota = None
        if minimum_required_placements:
            required_nucleus_class = target_cell_class_for_tissue(
                contract.target_label,
                None,
            )
            continuity_quota = compile_continuity_center_quota(
                nuclei_mask=source_nuclei,
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
            # Balance the case-local expectation with the densest legal seam
            # realization.  Reserving only the expectation can crowd the
            # exterior band and select undersized shapes; taking the upper
            # bound can over-concentrate ProbNet mass inside the seam.  Their
            # midpoint is deterministic and remains inside the gate-compiled
            # interval.
            minimum_required_placements = (
                int(
                    np.ceil(
                        (
                            continuity_quota.target_count
                            + continuity_quota.maximum_count
                        )
                        / 2.0
                    )
                )
                if continuity_quota.maximum_count is not None
                else continuity_quota.target_count
            )
            if continuity_quota.maximum_count is not None:
                maximum_required_placements = minimum_required_placements
            _save_binary(required_placement_path, program.continuity_region)

        eligible: set[str] = set()
        rejected: dict[str, str] = {}
        for class_id in program.target_classes:
            references, current_rejected = build_reference_shape_library(
                scene, class_id=class_id
            )
            eligible.update(item.instance_id for item in references)
            rejected.update(current_rejected)
        protected = set(contract.protected_instance_ids)
        eligible.difference_update(protected)
        for instance_id in protected:
            rejected.setdefault(instance_id, "executable_contract_protected_instance")
        if not eligible:
            raise JointContractError(
                "mature ProbNet execution has no complete non-border reference shape"
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
                generation_region_path=generation_path,
                population_region_path=population_path,
                placement_region_path=placement_path,
                erasure_region_path=erasure_path,
                required_placement_region_path=(
                    required_placement_path
                    if minimum_required_placements
                    else None
                ),
                minimum_required_placements=minimum_required_placements,
                maximum_required_placements=maximum_required_placements,
                required_nucleus_class=required_nucleus_class,
                output_path=output_path,
                prohibited_tissue_ids=prohibited_tissue_ids,
                allowed_new_cell_classes=contract.allowed_new_cell_classes,
            )
            completed = self._runner(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
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
                raise JointContractError("mature ProbNet sampling audit did not pass")
            target = load_nuclei_mask(output_path)
            accepted_center_ledger = _accepted_center_ledger(selected)
            accepted_instance_area_ledger = _accepted_instance_area_ledger(
                selected
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
                # passed the immutable contract. Systemic command/audit errors
                # still fail the whole executor above.
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
            results.append(
                CellLayoutResult(
                    cell_candidate_id=f"mature-cells-{variant + 1:02d}",
                    target_nuclei_mask=target,
                    trace={
                        "layout_tool_version": MATURE_EXECUTION_VERSION,
                        "execution_engine": MATURE_EXECUTION_VERSION,
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
                        "cell_capacity_fallback_used": False,
                        "placement_capacity_exhausted": placed < desired,
                        "reference_shape_ids": sorted(eligible),
                        "reference_shape_rejections": rejected,
                        "reference_shape_integrity_certified": True,
                        "reference_first": True,
                        "shape_sampling": selected.get("shape_sampling", {}),
                        "patch_adaptive_priors": selected.get(
                            "patch_adaptive_priors", {}
                        ),
                        "spatial_prior": selected.get("spatial_prior", {}),
                        "sampling_audit": audit,
                        "sampling_feedback": selected.get("sampling_feedback", {}),
                        "accepted_center_ledger": [
                            {"row": row, "col": col, "class_id": class_id}
                            for row, col, class_id in accepted_center_ledger
                        ],
                        "accepted_instance_area_ledger": (
                            accepted_instance_area_ledger
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
                                "selection_policy": (
                                    "midpoint_expected_and_gate_upper_bound_"
                                    "to_balance_seam_and_exterior_capacity_v1"
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
                            "compiled_seam_quota_then_exterior_T_pop_remainder_v2"
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
                "all mature ProbNet variants violated the executable contract: "
                + "; ".join(reasons)
            )
        for result in results:
            result.trace["rejected_variant_audit"] = list(rejected_variants)
        return tuple(results)


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
