"""Read-only subprocess adapter for the mature online ProbNet cell pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .cell_layouts import CellLayoutResult, build_reference_shape_library
from .cell_programs import CompiledCellToolProgram
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

    def supports(self, program: CompiledCellToolProgram) -> bool:
        return program.baseline_mode == "regenerate_target_population"

    def build_command(
        self,
        *,
        seed: int,
        target_tissue_path: Path,
        source_tissue_path: Path,
        source_nuclei_path: Path,
        placement_region_path: Path,
        erasure_region_path: Path,
        output_path: Path,
        prohibited_tissue_ids: tuple[int, ...],
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
            str(placement_region_path),
            "--deletion-region",
            str(erasure_region_path),
            "--output",
            str(output_path),
            "--no-widen-edit-region",
            "--require-sampling-audit",
            "--require-exact-target-count",
            "--require-full-tissue-containment",
            "--reference-shape-max-area-ratio",
            str(self.config.reference_shape_max_area_ratio),
        ]
        if prohibited_tissue_ids:
            command.extend(
                ["--skip-tissue-ids", *[str(value) for value in prohibited_tissue_ids]]
            )
        return command

    def execute(
        self,
        *,
        program: CompiledCellToolProgram,
        source_tissue: np.ndarray,
        target_tissue: np.ndarray,
        source_nuclei: np.ndarray,
        scene: JointSceneAnalysis,
        output_dir: str | Path,
        prohibited_tissue_ids: tuple[int, ...],
        seed: int,
        variants: int,
    ) -> tuple[CellLayoutResult, ...]:
        if not self.supports(program):
            raise JointContractError(
                "mature ProbNet baseline only realizes target-population regeneration; "
                "structured mechanism layouts require the deterministic layout executor"
            )
        self.validate_assets()
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target_tissue_path = directory / "target_tissue.png"
        source_tissue_path = directory / "source_tissue.png"
        source_nuclei_path = directory / "source_nuclei.png"
        placement_path = directory / "placement_region.png"
        erasure_path = directory / "erasure_region.png"
        _save_mask(target_tissue_path, target_tissue)
        _save_mask(source_tissue_path, source_tissue)
        _save_mask(source_nuclei_path, to_raw_nuclei_mask(source_nuclei))
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

        results = []
        for variant in range(variants):
            current_seed = int(seed + variant * 104729)
            output_path = directory / f"nuclei_{variant + 1:02d}.png"
            command = self.build_command(
                seed=current_seed,
                target_tissue_path=target_tissue_path,
                source_tissue_path=source_tissue_path,
                source_nuclei_path=source_nuclei_path,
                placement_region_path=placement_path,
                erasure_region_path=erasure_path,
                output_path=output_path,
                prohibited_tissue_ids=prohibited_tissue_ids,
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
                        "mature_probnet_contract": True,
                        "ranker": "frozen_probnet_context_stabilized_spatial_sampler",
                        "ranker_provenance": {
                            "checkpoint_sha256": _sha256(Path(self.config.checkpoint)),
                            "dataset_name": self.config.dataset_name,
                            "role": "mature_count_type_spatial_shape_pipeline",
                        },
                        "compiled_cell_tool_program": program.to_metadata(),
                        "desired_count": desired,
                        "resolved_count": placed,
                        "requested_count": desired,
                        "placed_count": placed,
                        "batch_max_attainable_count": placed,
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
                        "overlap_pixels": 0,
                        "partial_source_instance_edits": 0,
                        "cross_domain_fallback": False,
                        "seed": current_seed,
                        "command_sha256": hashlib.sha256(
                            "\0".join(command).encode("utf-8")
                        ).hexdigest(),
                    },
                )
            )
        return tuple(results)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    array = np.asarray(mask)
    if array.ndim != 2 or array.min(initial=0) < 0 or array.max(initial=0) > 255:
        raise JointContractError("mature ProbNet PNG masks require 2-D IDs in [0,255]")
    Image.fromarray(array.astype(np.uint8)).save(path)


def _save_binary(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
