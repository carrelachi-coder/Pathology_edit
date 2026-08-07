"""Atomic research artifacts for joint condition review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .executable_contract import ExecutableJointContract
from .models import JointCandidate, JointCaseContext
from .nuclei import to_raw_nuclei_mask


class JointAuditWriter:
    def __init__(self, root: str | Path, *, case_id: str) -> None:
        self.case_dir = Path(root) / str(case_id)
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.paths: dict[str, str] = {"case_dir": str(self.case_dir)}

    def write_json(self, name: str, payload: Any) -> str:
        path = self.case_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        self.paths[name] = str(path)
        return str(path)

    def write_inputs(self, *, case: JointCaseContext, scene_metadata: dict, skill_metadata: dict) -> None:
        self.write_json("case_context.json", case.to_metadata())
        self.write_json("joint_scene_graph.json", scene_metadata)
        self.write_json("active_joint_skills.json", skill_metadata)

    def write_candidates(self, candidates: tuple[JointCandidate, ...]) -> None:
        directory = self.case_dir / "candidates"
        directory.mkdir(exist_ok=True)
        manifest = []
        for candidate in candidates:
            target_tissue = directory / f"{candidate.candidate_id}.tissue.png"
            target_nuclei = directory / f"{candidate.candidate_id}.nuclei.png"
            diff = directory / f"{candidate.candidate_id}.joint.png"
            _save_id_mask(target_tissue, candidate.target_tissue_mask)
            _save_id_mask(target_nuclei, to_raw_nuclei_mask(candidate.target_nuclei_mask))
            _save_binary(diff, candidate.joint_change)
            manifest.append(
                {
                    **candidate.to_metadata(),
                    "target_tissue_mask": str(target_tissue),
                    "target_nuclei_mask": str(target_nuclei),
                    "joint_change_mask": str(diff),
                }
            )
        self.write_json("candidates.json", manifest)
        self.paths["candidates_dir"] = str(directory)

    def write_tissue_execution_review(
        self,
        *,
        pass_index: int,
        source_image_path: str,
        source_tissue: np.ndarray,
        source_nuclei: np.ndarray,
        tissue_scene,
        tissue_plan,
        execution_batch,
    ) -> str:
        """Persist Planner anchors and every exploratory tissue result.

        This artifact is deliberately written before the workflow is allowed to
        fail closed.  A tissue or cell-capacity rejection must remain visually
        inspectable instead of leaving only a JSON error string.
        """

        directory = self.case_dir / f"tissue_candidates_pass_{pass_index}"
        directory.mkdir(parents=True, exist_ok=True)
        report_by_id = {
            item.candidate_id: item
            for item in execution_batch.tissue_gate_reports
        }
        cell_by_id = {
            item.candidate_id: item
            for item in execution_batch.cell_feasibility_reports
        }
        contract_by_id = {
            item.tissue_candidate_id: item
            for item in execution_batch.executable_contracts
        }
        records = []
        for candidate in execution_batch.all_candidates:
            mask_path = directory / f"{candidate.candidate_id}.tissue.png"
            diff_path = directory / f"{candidate.candidate_id}.change.png"
            _save_id_mask(mask_path, candidate.target_mask)
            _save_binary(diff_path, candidate.change_region)
            tissue_report = report_by_id[candidate.candidate_id]
            cell_report = cell_by_id.get(candidate.candidate_id)
            records.append(
                {
                    **candidate.to_metadata(),
                    "target_tissue_mask": str(mask_path),
                    "tissue_change_mask": str(diff_path),
                    "tissue_gate_passed": tissue_report.passed,
                    "failed_tissue_checks": [
                        item.check_id
                        for item in tissue_report.checks
                        if item.severity == "hard" and not item.passed
                    ],
                    "cell_feasibility": (
                        cell_report.to_metadata() if cell_report is not None else None
                    ),
                    "executable_contract_id": (
                        contract_by_id[candidate.candidate_id].contract_id
                        if candidate.candidate_id in contract_by_id
                        else None
                    ),
                }
            )
        self.write_json(f"tissue_candidates_pass_{pass_index}.json", records)

        best = min(
            execution_batch.all_candidates,
            key=lambda candidate: _tissue_review_score(
                candidate.candidate_id,
                report_by_id=report_by_id,
                cell_by_id=cell_by_id,
                contract_by_id=contract_by_id,
            ),
        )
        tissue_report = report_by_id[best.candidate_id]
        cell_report = cell_by_id.get(best.candidate_id)
        contract = contract_by_id.get(best.candidate_id)
        image = Image.open(source_image_path).convert("RGB").resize(
            (source_tissue.shape[1], source_tissue.shape[0])
        )
        base = np.asarray(image, dtype=np.uint8)
        empty = np.zeros_like(source_tissue, dtype=bool)
        source_panel = _overlay(base, source_tissue, source_nuclei, empty)

        planner_panel = np.array(base, copy=True)
        selected_interfaces = np.zeros_like(source_tissue, dtype=bool)
        selected_anchors = np.zeros_like(source_tissue, dtype=bool)
        for planned in tissue_plan.candidate_interfaces:
            current = tissue_scene.interface_masks.get(planned.interface_id)
            if current is not None:
                selected_interfaces |= np.asarray(current, dtype=bool)
            for anchor_id in planned.execution_contract.anchor_segment_ids:
                anchor = tissue_scene.anchor_masks.get(anchor_id)
                if anchor is not None:
                    selected_anchors |= np.asarray(anchor, dtype=bool)
        planner_panel[selected_interfaces] = [0, 235, 255]
        planner_panel[selected_anchors] = [255, 225, 0]

        target_panel = _overlay(
            base,
            best.target_mask,
            source_nuclei,
            best.change_region,
        )
        target_panel[np.asarray(best.change_region, dtype=bool)] = np.clip(
            0.55 * target_panel[np.asarray(best.change_region, dtype=bool)]
            + 0.45 * np.asarray([255, 0, 210]),
            0,
            255,
        ).astype(np.uint8)

        execution_panel = np.array(base, copy=True)
        change = np.asarray(best.change_region, dtype=bool)
        execution_panel[change] = np.clip(
            0.35 * execution_panel[change] + 0.65 * np.asarray([255, 0, 210]),
            0,
            255,
        ).astype(np.uint8)
        if contract is not None:
            erasure = np.asarray(
                contract.cell_program.erasure_region, dtype=bool
            )
            support = np.asarray(
                contract.cell_program.support_context_region, dtype=bool
            )
            execution_panel[support & ~change] = np.clip(
                0.45 * execution_panel[support & ~change]
                + 0.55 * np.asarray([60, 130, 255]),
                0,
                255,
            ).astype(np.uint8)
            execution_panel[erasure] = [35, 235, 80]
        elif cell_report is not None:
            for instance_id in cell_report.removable_instance_ids:
                component = getattr(tissue_scene, "instance_masks", {}).get(
                    instance_id
                )
                if component is not None:
                    execution_panel[np.asarray(component, dtype=bool)] = [35, 235, 80]

        panels = [source_panel, planner_panel, target_panel, execution_panel]
        labels = [
            "1 SOURCE H&E + tissue/nuclei",
            "2 PLANNER cyan=interface yellow=anchor",
            f"3 TOOL best={best.candidate_id}",
            "4 CONTRACT magenta=T green=whole nuclei blue=G",
        ]
        tile_h, tile_w = source_tissue.shape
        header = 28
        footer = 54
        canvas = Image.new("RGB", (tile_w * 4, tile_h + header + footer), "black")
        draw = ImageDraw.Draw(canvas)
        for index, (panel, label) in enumerate(zip(panels, labels)):
            x = index * tile_w
            canvas.paste(Image.fromarray(panel), (x, header))
            draw.text((x + 5, 7), label, fill="white")
        failed_tissue = [
            item.check_id
            for item in tissue_report.checks
            if item.severity == "hard" and not item.passed
        ]
        if contract is not None:
            stage = "EXECUTABLE CONTRACT PRODUCED"
            detail = f"contract={contract.contract_id[:16]}..."
        elif not tissue_report.passed:
            stage = "STOP: TISSUE GATE"
            detail = ", ".join(failed_tissue) or "unknown tissue failure"
        else:
            stage = "STOP: CELL FEASIBILITY"
            detail = ", ".join(cell_report.reasons if cell_report else ()) or "no cell report"
        draw.text((6, tile_h + header + 7), stage, fill=(255, 220, 70))
        draw.text((6, tile_h + header + 27), detail[:220], fill="white")
        path = self.case_dir / f"tissue_execution_review_pass_{pass_index}.png"
        canvas.save(path)
        self.paths[f"tissue_execution_review_pass_{pass_index}"] = str(path)
        return str(path)

    def write_executable_contract(
        self, contract: ExecutableJointContract
    ) -> dict[str, str]:
        """Persist the immutable contract and its exact execution masks."""

        directory = self.case_dir / "executable_contracts" / contract.contract_id
        directory.mkdir(parents=True, exist_ok=True)
        masks = {
            "E_erasure": contract.cell_program.erasure_region,
            "P_placement_centers": contract.cell_program.placement_center_region,
            "V_valid_footprints": contract.cell_program.valid_footprint_region,
            "S_support_context": contract.cell_program.support_context_region,
            "M_mechanism_region": contract.cell_program.mechanism_region,
            "C_continuity_region": contract.cell_program.continuity_region,
            "A_selected_anchor": contract.cell_program.continuity_anchor_mask,
        }
        result = {}
        for name, mask in masks.items():
            path = directory / f"{name}.png"
            _save_binary(path, mask)
            result[name] = str(path)
        manifest = directory / "contract.json"
        manifest.write_text(
            json.dumps(
                contract.to_metadata(),
                indent=2,
                sort_keys=True,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        result["manifest"] = str(manifest)
        self.paths[f"executable_contract_{contract.contract_id}"] = str(manifest)
        self.paths["executable_contracts_dir"] = str(directory.parent)
        return result

    def write_review_board(
        self,
        *,
        source_image_path: str,
        source_tissue: np.ndarray,
        source_nuclei: np.ndarray,
        candidates: tuple[JointCandidate, ...],
    ) -> str:
        image = Image.open(source_image_path).convert("RGB").resize((source_tissue.shape[1], source_tissue.shape[0]))
        base = np.asarray(image, dtype=np.uint8)
        rows = []
        source_panel = _overlay(base, source_tissue, source_nuclei, np.zeros_like(source_tissue, dtype=bool))
        for candidate in candidates:
            target_panel = _overlay(base, candidate.target_tissue_mask, candidate.target_nuclei_mask, candidate.joint_change)
            row = np.concatenate([source_panel, target_panel], axis=1)
            labeled = Image.new("RGB", (row.shape[1], row.shape[0] + 22), "black")
            labeled.paste(Image.fromarray(row), (0, 22))
            ImageDraw.Draw(labeled).text((6, 4), candidate.candidate_id, fill="white")
            rows.append(np.asarray(labeled))
        if not rows:
            rows = [source_panel]
        board = np.concatenate(rows, axis=0)
        path = self.case_dir / "joint_condition_review.png"
        Image.fromarray(board).save(path)
        self.paths["joint_condition_review"] = str(path)
        return str(path)

    def write_joint_execution_review(
        self,
        *,
        source_image_path: str,
        source_tissue: np.ndarray,
        source_nuclei: np.ndarray,
        candidates: tuple[JointCandidate, ...],
        gate_reports,
    ) -> str | None:
        """Write one reader-facing view of the best paired candidate."""

        if not candidates:
            return None
        reports = {item.candidate_id: item for item in gate_reports}
        candidate = min(
            candidates,
            key=lambda item: (
                0 if reports[item.candidate_id].passed else 1,
                sum(
                    check.severity == "hard" and not check.passed
                    for check in reports[item.candidate_id].checks
                ),
                item.candidate_id,
            ),
        )
        report = reports[candidate.candidate_id]
        image = Image.open(source_image_path).convert("RGB").resize(
            (source_tissue.shape[1], source_tissue.shape[0])
        )
        base = np.asarray(image, dtype=np.uint8)
        empty = np.zeros_like(source_tissue, dtype=bool)
        source_panel = _overlay(base, source_tissue, source_nuclei, empty)
        target_panel = _overlay(
            base,
            candidate.target_tissue_mask,
            candidate.target_nuclei_mask,
            candidate.joint_change,
        )

        diff_panel = np.array(base, copy=True)
        tissue_change = np.asarray(candidate.tissue_change, dtype=bool)
        cell_change = np.asarray(candidate.cell_change, dtype=bool)
        diff_panel[tissue_change] = np.clip(
            0.30 * diff_panel[tissue_change] + 0.70 * np.asarray([255, 0, 210]),
            0,
            255,
        ).astype(np.uint8)
        cell_only = cell_change & ~tissue_change
        diff_panel[cell_only] = [30, 235, 75]
        diff_panel[cell_change & tissue_change] = [255, 225, 0]

        support_panel = np.array(base, copy=True)
        support = np.asarray(candidate.generation_support, dtype=bool)
        support_panel[support] = np.clip(
            0.30 * support_panel[support] + 0.70 * np.asarray([45, 125, 255]),
            0,
            255,
        ).astype(np.uint8)
        removed = (np.asarray(source_nuclei) > 0) & (
            np.asarray(candidate.target_nuclei_mask) == 0
        )
        added = (np.asarray(source_nuclei) == 0) & (
            np.asarray(candidate.target_nuclei_mask) > 0
        )
        support_panel[removed] = [35, 235, 80]
        support_panel[added] = [255, 145, 20]

        labels = [
            "1 SOURCE joint condition",
            "2 TARGET joint condition",
            "3 DIFF magenta=T green=C-only yellow=overlap",
            "4 GENERATION blue=G green=removed orange=added",
        ]
        panels = [source_panel, target_panel, diff_panel, support_panel]
        tile_h, tile_w = source_tissue.shape
        header = 28
        footer = 54
        canvas = Image.new("RGB", (tile_w * 4, tile_h + header + footer), "black")
        draw = ImageDraw.Draw(canvas)
        for index, (panel, label) in enumerate(zip(panels, labels)):
            x = index * tile_w
            canvas.paste(Image.fromarray(panel), (x, header))
            draw.text((x + 5, 7), label, fill="white")
        failed = [
            item.check_id
            for item in report.checks
            if item.severity == "hard" and not item.passed
        ]
        status = "PASS: READY FOR INDEPENDENT VISUAL CRITIC" if report.passed else "STOP: JOINT GATE"
        draw.text((6, tile_h + header + 7), status, fill=(255, 220, 70))
        draw.text(
            (6, tile_h + header + 27),
            (
                f"candidate={candidate.candidate_id} | "
                + (", ".join(failed) if failed else "all deterministic hard gates passed")
            )[:260],
            fill="white",
        )
        path = self.case_dir / "joint_execution_review.png"
        canvas.save(path)
        self.paths["joint_execution_review"] = str(path)
        return str(path)

    def write_source_joint_overlay(
        self,
        *,
        source_image_path: str,
        source_tissue: np.ndarray,
        source_nuclei: np.ndarray,
    ) -> str:
        image = Image.open(source_image_path).convert("RGB").resize((source_tissue.shape[1], source_tissue.shape[0]))
        panel = _overlay(
            np.asarray(image, dtype=np.uint8),
            source_tissue,
            source_nuclei,
            np.zeros_like(source_tissue, dtype=bool),
        )
        path = self.case_dir / "source_joint_overlay.png"
        Image.fromarray(panel).save(path)
        self.paths["source_joint_overlay"] = str(path)
        return str(path)

    def write_reference_shape_review(
        self,
        *,
        source_image_path: str,
        instance_masks: dict[str, np.ndarray],
        eligible_ids: tuple[str, ...] | list[str],
        rejected: dict[str, str],
    ) -> str:
        """Visual proof that patch-censored source shapes were not reusable."""

        canvas = Image.open(source_image_path).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        eligible = set(eligible_ids)
        for instance_id in sorted(eligible.union(rejected)):
            component = instance_masks.get(instance_id)
            if component is None:
                continue
            ys, xs = np.where(np.asarray(component, dtype=bool))
            if not ys.size:
                continue
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            if instance_id in eligible:
                color, width = (30, 220, 70), 1
            else:
                color, width = (255, 45, 45), 2
            draw.rectangle(box, outline=color, width=width)
        draw.rectangle((0, 0, min(canvas.width - 1, 360), 22), fill=(0, 0, 0))
        draw.text(
            (5, 5),
            f"green=eligible {len(eligible)} | red=rejected {len(rejected)}",
            fill=(255, 255, 255),
        )
        path = self.case_dir / "reference_shape_review.png"
        canvas.save(path)
        self.paths["reference_shape_review"] = str(path)
        self.write_json(
            "reference_shape_audit.json",
            {
                "eligible_instance_ids": sorted(eligible),
                "rejected_instances": dict(sorted(rejected.items())),
                "four_patch_edges_are_censoring_boundaries": True,
            },
        )
        return str(path)


def _save_id_mask(path: Path, mask: np.ndarray) -> None:
    values = np.asarray(mask)
    maximum = int(values.max(initial=0))
    mode = "I;16" if maximum > 255 else "L"
    dtype = np.uint16 if maximum > 255 else np.uint8
    Image.fromarray(values.astype(dtype), mode=mode).save(path)


def _save_binary(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def _overlay(image: np.ndarray, tissue: np.ndarray, nuclei: np.ndarray, change: np.ndarray) -> np.ndarray:
    result = image.astype(float)
    palette = np.asarray([[0,0,0],[220,40,40],[70,170,80],[80,100,220],[210,150,40],[160,70,190],[50,180,180],[190,190,80]], dtype=float)
    ids = np.asarray(tissue, dtype=int)
    color = palette[ids % len(palette)]
    result = 0.72 * result + 0.28 * color
    nuclei_mask = np.asarray(nuclei) > 0
    result[nuclei_mask] = [35, 20, 90]
    boundary = np.asarray(change, dtype=bool) ^ ndimage_binary_erosion(change)
    result[boundary] = [255, 255, 0]
    return np.clip(result, 0, 255).astype(np.uint8)


def ndimage_binary_erosion(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return ndimage.binary_erosion(np.asarray(mask, dtype=bool))


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _tissue_review_score(
    candidate_id: str,
    *,
    report_by_id: dict,
    cell_by_id: dict,
    contract_by_id: dict,
) -> tuple[int, int, int, str]:
    report = report_by_id[candidate_id]
    hard_failures = sum(
        item.severity == "hard" and not item.passed for item in report.checks
    )
    cell = cell_by_id.get(candidate_id)
    cell_failures = len(cell.reasons) if cell is not None else 999
    return (
        0 if candidate_id in contract_by_id else 1,
        hard_failures,
        cell_failures,
        candidate_id,
    )
