"""Sealed, content-reconstructable raster authority for mask-graph LLM stages."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.evidence import load_id_mask, sha256_file
from phase3_mask_edit_refine.scene import build_scene_analysis
from phase3_mask_edit_refine.visualization import build_mask_planner_panels

from .audit import build_mask_planner_overlay, build_mask_review_board
from .models import JointContractError
from .nuclei import load_nuclei_mask

MASK_PLANNER_ARTIFACT_KINDS = {
    "planner_01_tissue_mask.png": "source_tissue_semantic_panel",
    "planner_02_component_map.png": "source_tissue_component_panel",
    "planner_03_interface_anchor_map.png": "source_interface_anchor_panel",
    "planner_mask_tissue_nuclei.png": "source_tissue_nuclei_panel",
    "joint_condition_mask_review.png": "candidate_mask_condition_board",
}
SOURCE_MASK_ARTIFACT_KINDS = frozenset(
    value
    for value in MASK_PLANNER_ARTIFACT_KINDS.values()
    if value != "candidate_mask_condition_board"
)
MASK_PLANNER_REGISTRY_SCHEMA = "mask-planner-artifact-registry-v3"
_SOURCE_PRODUCER_ID = "sealed-deterministic-mask-panel-writer-v3"
_CANDIDATE_PRODUCER_ID = "sealed-deterministic-candidate-board-writer-v3"
_PIPELINE_ISSUER = object()


def _case_binding_payload(case: Any) -> dict[str, Any]:
    provenance = getattr(case, "provenance", None)
    if not isinstance(provenance, Mapping):
        raise JointContractError("mask Planner case provenance is unavailable")
    tissue_digest = provenance.get("source_tissue_mask_sha256") or provenance.get(
        "source_mask_sha256"
    )
    required = {
        "case_id": getattr(case, "case_id", None),
        "source_image_sha256": provenance.get("source_image_sha256"),
        "source_tissue_mask_sha256": tissue_digest,
        "source_nuclei_mask_sha256": provenance.get("source_nuclei_mask_sha256"),
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


def _rgb_sha256(value: np.ndarray) -> str:
    rgb = np.ascontiguousarray(np.asarray(value, dtype=np.uint8))
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def candidate_portfolio_sha256(candidates: Sequence[Any]) -> str:
    """Bind a Critic board to exact candidate IDs and final mask rasters."""

    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(str(candidate.candidate_id).encode("utf-8"))
        for field in (
            "target_tissue_mask",
            "target_nuclei_mask",
            "tissue_change",
            "cell_change",
            "joint_change",
        ):
            value = np.ascontiguousarray(np.asarray(getattr(candidate, field)))
            digest.update(field.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(value.shape).encode("ascii"))
            digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MaskPlannerArtifactRecord:
    artifact_kind: str
    canonical_path: str
    artifact_sha256: str
    expected_rgb_sha256: str
    case_binding_sha256: str
    case_id: str
    source_image_sha256: str
    source_tissue_mask_sha256: str
    source_nuclei_mask_sha256: str
    producer_id: str
    producer_version: str
    authority_payload_sha256: str
    candidate_portfolio_sha256: str | None = None


class MaskPlannerArtifactRegistry:
    """Pipeline-issued raster capability bound to reconstructable mask content.

    Construction and generic registration are deliberately unavailable. The
    trusted factory validates the supplied arrays against the case source mask
    files, renders the canonical panels itself, and retains immutable expected
    pixels. Candidate boards are likewise rendered from the exact candidate
    tuple and carry its portfolio digest. A caller cannot turn an arbitrary RGB
    file into an authority by choosing a filename, root, or producer string.
    """

    def __init__(
        self,
        *,
        case: Any,
        pipeline_owned_root: str | Path,
        source_tissue: np.ndarray,
        source_nuclei: np.ndarray,
        schema: MaskProfileSchema,
        pixel_size_um: float | None,
        _issuer: object | None = None,
    ) -> None:
        if _issuer is not _PIPELINE_ISSUER:
            raise JointContractError(
                "mask Planner registries must be issued by the deterministic panel factory"
            )
        root = Path(pipeline_owned_root).absolute()
        if root.is_symlink():
            raise JointContractError("mask Planner artifact root cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        self.pipeline_owned_root = root.resolve(strict=True)
        self.case_binding_sha256 = mask_planner_case_binding_sha256(case)
        payload = _case_binding_payload(case)
        self.case_id = str(payload["case_id"])
        self.source_image_sha256 = str(payload["source_image_sha256"])
        self.source_tissue_mask_sha256 = str(payload["source_tissue_mask_sha256"])
        self.source_nuclei_mask_sha256 = str(payload["source_nuclei_mask_sha256"])
        self._source_tissue = np.array(source_tissue, copy=True)
        self._source_nuclei = np.array(source_nuclei, copy=True)
        self._source_tissue.setflags(write=False)
        self._source_nuclei.setflags(write=False)
        # Reconstruct the component/interface/anchor authority from the exact
        # source tissue pixels and annotation schema inside the sealed writer.
        # A direct caller cannot inject a fabricated SceneAnalysis and have it
        # rendered as if it belonged to this case.
        self._scene = build_scene_analysis(
            self._source_tissue,
            schema=schema,
            pixel_size_um=pixel_size_um,
        )
        self._records: dict[str, MaskPlannerArtifactRecord] = {}
        self._expected_rgb: dict[str, np.ndarray] = {}
        self._verify_source_authority(case)
        self.source_image_paths = self._write_source_panels()

    @classmethod
    def issue(
        cls,
        *,
        case: Any,
        pipeline_owned_root: str | Path,
        source_tissue: np.ndarray,
        source_nuclei: np.ndarray,
        schema: MaskProfileSchema,
        pixel_size_um: float | None,
    ) -> MaskPlannerArtifactRegistry:
        return cls(
            case=case,
            pipeline_owned_root=pipeline_owned_root,
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            schema=schema,
            pixel_size_um=pixel_size_um,
            _issuer=_PIPELINE_ISSUER,
        )

    def register(self, *_args: Any, **_kwargs: Any) -> str:
        """Reject the former self-authorizing public registration API."""

        raise JointContractError(
            "generic mask Planner artifact registration is forbidden; use the sealed writer"
        )

    def write_candidate_board(self, *, candidates: Sequence[Any]) -> str:
        exact = tuple(candidates)
        if not exact:
            raise JointContractError("Critic board requires at least one candidate")
        portfolio_sha = candidate_portfolio_sha256(exact)
        board = build_mask_review_board(
            source_tissue=self._source_tissue,
            source_nuclei=self._source_nuclei,
            candidates=exact,
        )
        return self._write_authoritative_image(
            self.pipeline_owned_root / "joint_condition_mask_review.png",
            artifact_kind="candidate_mask_condition_board",
            rgb=board,
            producer_id=_CANDIDATE_PRODUCER_ID,
            candidate_portfolio_digest=portfolio_sha,
        )

    def validate(
        self,
        image_paths: Sequence[str | Path],
        *,
        case: Any,
        allowed_artifact_kinds: frozenset[str],
        candidate_portfolio: Sequence[Any] | None = None,
    ) -> tuple[str, ...]:
        expected_binding = mask_planner_case_binding_sha256(case)
        if expected_binding != self.case_binding_sha256:
            raise JointContractError(
                "mask Planner artifact registry belongs to another case/input binding"
            )
        expected_portfolio_sha = (
            candidate_portfolio_sha256(tuple(candidate_portfolio))
            if candidate_portfolio is not None
            else None
        )
        validated: list[str] = []
        case_payload = _case_binding_payload(case)
        for path in image_paths:
            candidate = self._canonical_pipeline_file(path)
            record = self._records.get(str(candidate))
            if record is None:
                raise JointContractError(
                    "mask-graph LLM input was not issued by the sealed panel writer"
                )
            if record.artifact_kind not in allowed_artifact_kinds:
                raise JointContractError(
                    "mask-graph LLM artifact kind is forbidden for this stage"
                )
            expected_rgb = self._expected_rgb.get(str(candidate))
            try:
                with Image.open(candidate) as opened:
                    actual_rgb = np.asarray(opened.convert("RGB"))
            except Exception as exc:
                raise JointContractError(
                    "mask-graph LLM artifact cannot be decoded"
                ) from exc
            authority_payload = self._authority_payload(
                artifact_kind=record.artifact_kind,
                expected_rgb_sha256=record.expected_rgb_sha256,
                candidate_portfolio_digest=record.candidate_portfolio_sha256,
            )
            if (
                expected_rgb is None
                or actual_rgb.shape != expected_rgb.shape
                or not np.array_equal(actual_rgb, expected_rgb)
                or record.expected_rgb_sha256 != _rgb_sha256(actual_rgb)
                or record.authority_payload_sha256 != authority_payload
                or record.case_binding_sha256 != expected_binding
                or record.case_id != str(getattr(case, "case_id", ""))
                or record.source_image_sha256
                != str(case_payload["source_image_sha256"])
                or record.source_tissue_mask_sha256
                != str(case_payload["source_tissue_mask_sha256"])
                or record.source_nuclei_mask_sha256
                != str(case_payload["source_nuclei_mask_sha256"])
                or record.artifact_kind
                != MASK_PLANNER_ARTIFACT_KINDS.get(candidate.name)
                or record.artifact_sha256 != sha256_file(candidate)
                or record.candidate_portfolio_sha256 != expected_portfolio_sha
            ):
                raise JointContractError(
                    "mask-graph LLM artifact content/provenance validation failed"
                )
            validated.append(str(candidate))
        return tuple(validated)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MASK_PLANNER_REGISTRY_SCHEMA,
            "case_id": self.case_id,
            "case_binding_sha256": self.case_binding_sha256,
            "pipeline_owned_root": str(self.pipeline_owned_root),
            "generic_registration_enabled": False,
            "artifacts": [
                asdict(self._records[key]) for key in sorted(self._records)
            ],
        }

    def _verify_source_authority(self, case: Any) -> None:
        if sha256_file(case.source_tissue_mask_uri) != self.source_tissue_mask_sha256:
            raise JointContractError("source tissue file digest is detached from case provenance")
        if sha256_file(case.source_nuclei_mask_uri) != self.source_nuclei_mask_sha256:
            raise JointContractError("source nuclei file digest is detached from case provenance")
        disk_tissue = load_id_mask(case.source_tissue_mask_uri)
        disk_nuclei = load_nuclei_mask(case.source_nuclei_mask_uri)
        if not np.array_equal(disk_tissue, self._source_tissue):
            raise JointContractError("panel writer tissue pixels differ from the case authority")
        if not np.array_equal(disk_nuclei, self._source_nuclei):
            raise JointContractError("panel writer nuclei pixels differ from the case authority")

    def _write_source_panels(self) -> tuple[str, ...]:
        panels = build_mask_planner_panels(
            mask=self._source_tissue,
            scene=self._scene,
        )
        directory = self.pipeline_owned_root / "planner_panels"
        paths = []
        for name, rgb in zip(
            (
                "planner_01_tissue_mask.png",
                "planner_02_component_map.png",
                "planner_03_interface_anchor_map.png",
            ),
            panels,
        ):
            paths.append(
                self._write_authoritative_image(
                    directory / name,
                    artifact_kind=MASK_PLANNER_ARTIFACT_KINDS[name],
                    rgb=rgb,
                    producer_id=_SOURCE_PRODUCER_ID,
                )
            )
        overlay = build_mask_planner_overlay(
            source_tissue=self._source_tissue,
            source_nuclei=self._source_nuclei,
        )
        paths.append(
            self._write_authoritative_image(
                self.pipeline_owned_root / "planner_mask_tissue_nuclei.png",
                artifact_kind="source_tissue_nuclei_panel",
                rgb=overlay,
                producer_id=_SOURCE_PRODUCER_ID,
            )
        )
        return tuple(paths)

    def _write_authoritative_image(
        self,
        path: Path,
        *,
        artifact_kind: str,
        rgb: np.ndarray,
        producer_id: str,
        candidate_portfolio_digest: str | None = None,
    ) -> str:
        parent = self._prepare_authoritative_parent(path.parent)
        target = parent / path.name
        if target.is_symlink():
            raise JointContractError(
                "mask Planner writer refuses an existing target symlink"
            )
        if target.exists():
            target_stat = target.lstat()
            if not stat.S_ISREG(target_stat.st_mode):
                raise JointContractError(
                    "mask Planner writer refuses a non-regular target"
                )
            if target_stat.st_nlink != 1:
                raise JointContractError(
                    "mask Planner writer refuses a multiply-linked target"
                )
        value = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                Image.fromarray(value).save(stream, format="PNG")
                stream.flush()
                os.fsync(stream.fileno())
            temporary_stat = temporary.lstat()
            if (
                not stat.S_ISREG(temporary_stat.st_mode)
                or temporary_stat.st_nlink != 1
            ):
                raise JointContractError(
                    "mask Planner temporary artifact lost exclusive-file authority"
                )
            with Image.open(temporary) as opened:
                encoded_rgb = np.asarray(opened.convert("RGB"))
            if (
                encoded_rgb.shape != value.shape
                or not np.array_equal(encoded_rgb, value)
            ):
                raise JointContractError(
                    "mask Planner temporary artifact failed encode verification"
                )
            # Atomic replacement changes the directory entry itself; it never
            # follows a target symlink or mutates a multiply-linked inode.
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        canonical = self._canonical_pipeline_file(target)
        expected_sha = _rgb_sha256(value)
        record = MaskPlannerArtifactRecord(
            artifact_kind=artifact_kind,
            canonical_path=str(canonical),
            artifact_sha256=sha256_file(canonical),
            expected_rgb_sha256=expected_sha,
            case_binding_sha256=self.case_binding_sha256,
            case_id=self.case_id,
            source_image_sha256=self.source_image_sha256,
            source_tissue_mask_sha256=self.source_tissue_mask_sha256,
            source_nuclei_mask_sha256=self.source_nuclei_mask_sha256,
            producer_id=producer_id,
            producer_version="v3",
            authority_payload_sha256=self._authority_payload(
                artifact_kind=artifact_kind,
                expected_rgb_sha256=expected_sha,
                candidate_portfolio_digest=candidate_portfolio_digest,
            ),
            candidate_portfolio_sha256=candidate_portfolio_digest,
        )
        immutable = np.array(value, copy=True)
        immutable.setflags(write=False)
        self._records[str(canonical)] = record
        self._expected_rgb[str(canonical)] = immutable
        return str(canonical)

    def _prepare_authoritative_parent(self, parent: Path) -> Path:
        """Create a writer directory without following child symlinks."""

        requested = parent.absolute()
        try:
            relative = requested.relative_to(self.pipeline_owned_root)
        except ValueError as exc:
            raise JointContractError(
                "mask Planner writer target is outside the pipeline-owned root"
            ) from exc
        cursor = self.pipeline_owned_root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise JointContractError(
                    "mask Planner writer parent chain contains a symlink"
                )
            if cursor.exists():
                current_stat = cursor.lstat()
                if not stat.S_ISDIR(current_stat.st_mode):
                    raise JointContractError(
                        "mask Planner writer parent is not a directory"
                    )
            else:
                cursor.mkdir()
        return cursor.resolve(strict=True)

    def _authority_payload(
        self,
        *,
        artifact_kind: str,
        expected_rgb_sha256: str,
        candidate_portfolio_digest: str | None,
    ) -> str:
        payload = {
            "case_binding_sha256": self.case_binding_sha256,
            "artifact_kind": artifact_kind,
            "expected_rgb_sha256": expected_rgb_sha256,
            "candidate_portfolio_sha256": candidate_portfolio_digest,
            "writer_schema": MASK_PLANNER_REGISTRY_SCHEMA,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

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
            raise JointContractError("mask Planner artifact does not exist") from exc
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
                raise JointContractError("mask Planner artifact traverses a symlink")
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
    candidate_portfolio: Sequence[Any] | None = None,
) -> tuple[str, ...]:
    """Validate a direct Planner/Critic raster capability for its exact stage."""

    if not image_paths:
        return ()
    if artifact_registry is None:
        raise JointContractError(
            "mask-graph LLM raster inputs require the current case artifact registry"
        )
    allowed_kinds = (
        frozenset({"candidate_mask_condition_board"})
        if candidate_portfolio is not None
        else SOURCE_MASK_ARTIFACT_KINDS
    )
    return artifact_registry.validate(
        image_paths,
        case=case,
        allowed_artifact_kinds=allowed_kinds,
        candidate_portfolio=candidate_portfolio,
    )
