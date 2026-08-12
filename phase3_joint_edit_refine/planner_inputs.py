"""Digest-bound raster authority for mask-graph LLM stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from phase3_mask_edit_refine.evidence import sha256_file

from .models import JointContractError

MASK_PLANNER_ARTIFACT_KINDS = {
    "planner_01_tissue_mask.png": "source_tissue_semantic_panel",
    "planner_02_component_map.png": "source_tissue_component_panel",
    "planner_03_interface_anchor_map.png": "source_interface_anchor_panel",
    "planner_mask_tissue_nuclei.png": "source_tissue_nuclei_panel",
    "joint_condition_mask_review.png": "candidate_mask_condition_board",
}
MASK_PLANNER_REGISTRY_SCHEMA = "mask-planner-artifact-registry-v2"


def _case_binding_payload(case: Any) -> dict[str, Any]:
    provenance = getattr(case, "provenance", None)
    if not isinstance(provenance, Mapping):
        raise JointContractError("mask Planner case provenance is unavailable")
    tissue_digest = provenance.get("source_tissue_mask_sha256") or provenance.get(
        "source_mask_sha256"
    )
    nuclei_digest = provenance.get("source_nuclei_mask_sha256")
    required = {
        "case_id": getattr(case, "case_id", None),
        "source_image_sha256": provenance.get("source_image_sha256"),
        "source_tissue_mask_sha256": tissue_digest,
        "source_nuclei_mask_sha256": nuclei_digest,
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        raise JointContractError(
            "mask Planner artifact binding lacks case/input identity: "
            + ", ".join(missing)
        )
    return required


def mask_planner_case_binding_sha256(case: Any) -> str:
    payload = json.dumps(
        _case_binding_payload(case), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MaskPlannerArtifactRecord:
    artifact_kind: str
    canonical_path: str
    artifact_sha256: str
    case_binding_sha256: str
    case_id: str
    source_image_sha256: str
    source_tissue_mask_sha256: str
    source_nuclei_mask_sha256: str
    producer_id: str
    producer_version: str


class MaskPlannerArtifactRegistry:
    """In-memory capability registry bound to one case and audit directory.

    A path is authoritative only after the pipeline writer has registered the
    just-written artifact. Validation re-hashes the file, rejects symlinks and
    path traversal, and checks the current case/input binding. A matching
    basename has no authority by itself.
    """

    def __init__(self, *, case: Any, pipeline_owned_root: str | Path) -> None:
        root = Path(pipeline_owned_root).absolute()
        if root.is_symlink():
            raise JointContractError("mask Planner artifact root cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        self.pipeline_owned_root = root.resolve(strict=True)
        self.case_binding_sha256 = mask_planner_case_binding_sha256(case)
        payload = _case_binding_payload(case)
        self.case_id = str(payload["case_id"])
        self.source_image_sha256 = str(payload["source_image_sha256"])
        self.source_tissue_mask_sha256 = str(
            payload["source_tissue_mask_sha256"]
        )
        self.source_nuclei_mask_sha256 = str(
            payload["source_nuclei_mask_sha256"]
        )
        self._records: dict[str, MaskPlannerArtifactRecord] = {}

    def register(
        self,
        path: str | Path,
        *,
        artifact_kind: str,
        producer_id: str,
        producer_version: str,
    ) -> str:
        candidate = self._canonical_pipeline_file(path)
        expected_kind = MASK_PLANNER_ARTIFACT_KINDS.get(candidate.name)
        if expected_kind != artifact_kind:
            raise JointContractError(
                "mask Planner artifact kind does not match its pipeline slot"
            )
        record = MaskPlannerArtifactRecord(
            artifact_kind=artifact_kind,
            canonical_path=str(candidate),
            artifact_sha256=sha256_file(candidate),
            case_binding_sha256=self.case_binding_sha256,
            case_id=self.case_id,
            source_image_sha256=self.source_image_sha256,
            source_tissue_mask_sha256=self.source_tissue_mask_sha256,
            source_nuclei_mask_sha256=self.source_nuclei_mask_sha256,
            producer_id=str(producer_id),
            producer_version=str(producer_version),
        )
        if not record.producer_id or not record.producer_version:
            raise JointContractError("mask Planner artifact producer is incomplete")
        self._records[str(candidate)] = record
        return str(candidate)

    def validate(
        self, image_paths: Sequence[str | Path], *, case: Any
    ) -> tuple[str, ...]:
        expected_binding = mask_planner_case_binding_sha256(case)
        if expected_binding != self.case_binding_sha256:
            raise JointContractError(
                "mask Planner artifact registry belongs to another case/input binding"
            )
        validated: list[str] = []
        for path in image_paths:
            candidate = self._canonical_pipeline_file(path)
            record = self._records.get(str(candidate))
            if record is None:
                raise JointContractError(
                    "mask-graph LLM input is not registered in the current case audit"
                )
            if (
                record.case_binding_sha256 != expected_binding
                or record.case_id != str(getattr(case, "case_id", ""))
                or record.source_image_sha256
                != str(
                    _case_binding_payload(case)["source_image_sha256"]
                )
                or record.source_tissue_mask_sha256
                != str(
                    _case_binding_payload(case)[
                        "source_tissue_mask_sha256"
                    ]
                )
                or record.source_nuclei_mask_sha256
                != str(
                    _case_binding_payload(case)[
                        "source_nuclei_mask_sha256"
                    ]
                )
                or record.artifact_kind
                != MASK_PLANNER_ARTIFACT_KINDS.get(candidate.name)
                or record.artifact_sha256 != sha256_file(candidate)
            ):
                raise JointContractError(
                    "mask-graph LLM artifact digest/provenance validation failed"
                )
            validated.append(str(candidate))
        return tuple(validated)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MASK_PLANNER_REGISTRY_SCHEMA,
            "case_id": self.case_id,
            "case_binding_sha256": self.case_binding_sha256,
            "pipeline_owned_root": str(self.pipeline_owned_root),
            "artifacts": [
                asdict(self._records[key]) for key in sorted(self._records)
            ],
        }

    def _canonical_pipeline_file(self, path: str | Path) -> Path:
        raw = Path(path)
        if ".." in raw.parts:
            raise JointContractError("mask Planner artifact path traversal is forbidden")
        supplied = raw.absolute()
        if supplied.is_symlink():
            raise JointContractError("mask Planner artifacts cannot be symlinks")
        try:
            canonical = supplied.resolve(strict=True)
        except FileNotFoundError as exc:
            raise JointContractError(
                "mask Planner artifact does not exist"
            ) from exc
        # macOS maps /var to /private/var. That root alias is canonicalized by
        # the OS rather than supplied by the caller. Explicit symlink path
        # components inside the pipeline-owned root are still prohibited.
        try:
            relative = canonical.relative_to(self.pipeline_owned_root)
        except ValueError as exc:
            raise JointContractError(
                "mask Planner artifact is outside the pipeline-owned case directory"
            ) from exc
        cursor = self.pipeline_owned_root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise JointContractError(
                    "mask Planner artifact traverses a symlink"
                )
        if not canonical.is_file():
            raise JointContractError(
                "mask Planner artifact is outside the pipeline-owned case directory"
            )
        return canonical


def validate_mask_planner_image_paths(
    image_paths: Sequence[str | Path],
    *,
    case: Any,
    artifact_registry: MaskPlannerArtifactRegistry | None,
) -> tuple[str, ...]:
    """Validate raster capabilities for a direct Planner/Critic caller."""

    if not image_paths:
        return ()
    if artifact_registry is None:
        raise JointContractError(
            "mask-graph LLM raster inputs require the current case artifact registry"
        )
    return artifact_registry.validate(image_paths, case=case)
