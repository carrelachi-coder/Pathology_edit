"""Atomic research artifacts for joint condition review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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
