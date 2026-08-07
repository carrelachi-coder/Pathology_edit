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

MATURE_EXECUTION_VERSION = "online-probnet-mature-v3"


@dataclass(frozen=True)
class MatureProbNetConfig:
    """Runtime assets are explicit; annotation profile never implies dataset."""

    dataset_name: str
    checkpoint: str
    instance_library: str
    device: str = "auto"
    base_channels: int = 64
    python_executable: str = sys.executable
    reference_shape_max_area_ratio: float = 3.0


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
        generation_region_path: Path,
        placement_region_path: Path,
        erasure_region_path: Path,
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
            str(source_nuclei_path),
            "--edit-region",
            str(generation_region_path),
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
        generation_path = directory / "generation_region.png"
        placement_path = directory / "placement_region.png"
        erasure_path = directory / "erasure_region.png"
        _save_mask(target_tissue_path, target_tissue)
        _save_mask(source_tissue_path, source_tissue)
        _save_mask(source_nuclei_path, to_raw_nuclei_mask(source_nuclei))
        # ``--edit-region`` is the mature CLI generation domain and must
        # contain every complete-instance deletion pixel. P contains legal
        # centers but not necessarily the full footprint of a source nucleus
        # crossing T, so the CLI receives P union E. The executable contract
        # still audits new centers against P/mechanism and all changed pixels
        # against S after generation.
        mature_generation_region = (
            np.asarray(program.placement_center_region, dtype=bool)
            | np.asarray(program.erasure_region, dtype=bool)
        )
        if np.any(mature_generation_region & ~program.support_context_region):
            raise JointContractError(
                "mature ProbNet generation region exceeds executable support"
            )
        _save_binary(generation_path, mature_generation_region)
        _save_binary(placement_path, program.placement_center_region)
        _save_binary(erasure_path, program.erasure_region)

        eligible: set[str] = set()
        rejected: dict[str, str] = {}
        for class_id in program.target_classes:
            references, current_rejected = build_reference_shape_library(
                scene, class_id=class_id
            )
            eligible.update(item.instance_id for item in references)
            rejected.update(current_rejected)
        if not eligible:
            raise JointContractError(
                "mature ProbNet execution has no complete non-border reference shape"
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
                generation_region_path=generation_path,
                placement_region_path=placement_path,
                erasure_region_path=erasure_path,
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
                        "mature_generation_region_policy": (
                            "placement_centers_union_complete_instance_erasure"
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
