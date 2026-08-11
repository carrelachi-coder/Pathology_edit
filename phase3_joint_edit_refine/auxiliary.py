"""Deterministic auxiliary structure producers for executable joint edits.

The producers in this module do not infer histology from H&E.  They recover
only topology that is already observable in the versioned fine tissue mask:
an internal non-pattern space must be completely enclosed by one gland/pattern
component.  The resulting masks are protection maps, not new tissue labels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from phase3_mask_edit_refine.evidence import sha256_file

from .models import JointCaseContext, JointContractError

AUXILIARY_PRODUCER_VERSION = "joint-semantic-topology-auxiliary-v1"


@dataclass(frozen=True)
class ProducedAuxiliary:
    structure_id: str
    path: str
    sha256: str
    provenance: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "path": self.path,
            "sha256": self.sha256,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class _AuxiliarySpecification:
    structure_id: str
    producer_kind: str
    fine_ids: tuple[int, ...]


def materialize_profile_auxiliaries(
    case: JointCaseContext,
    *,
    source_tissue: np.ndarray,
    output_dir: str | Path,
) -> tuple[JointCaseContext, tuple[ProducedAuxiliary, ...]]:
    """Produce missing profile-owned topology maps and bind their provenance.

    User-supplied maps remain authoritative and are never overwritten.  A
    generated map may be empty: that means the producer observed no enclosed
    structure, not that the required producer was skipped.
    """

    specifications = _profile_specifications(
        case.annotation_profile_id,
        source_tissue=source_tissue,
    )
    if case.annotation_profile_id == "glas-gland-v1":
        present = sorted(
            int(value)
            for value in np.unique(source_tissue)
            if int(value) in {5, 11, 12, 13}
        )
        case = replace(
            case,
            provenance={
                **case.provenance,
                "gland_fine_label_signature": present,
            },
        )
    missing = [
        item
        for item in specifications
        if item.structure_id not in case.auxiliary_structure_uris
    ]
    if not missing:
        _validate_bound_auxiliary_provenance(case)
        return case, ()

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    uris = dict(case.auxiliary_structure_uris)
    digests = dict(case.provenance.get("auxiliary_structure_sha256", {}))
    provenance_by_id = dict(
        case.provenance.get("auxiliary_structure_provenance", {})
    )
    produced = []
    source_digest = case.provenance["source_tissue_mask_sha256"]
    for specification in missing:
        structure_id = specification.structure_id
        if specification.producer_kind == "enclosed_pattern_spaces":
            mask, details = _enclosed_pattern_spaces(
                source_tissue,
                pattern_fine_ids=specification.fine_ids,
            )
            protection_semantics = "enclosed_internal_space"
        elif specification.producer_kind == "explicit_profile_structure":
            mask = np.isin(source_tissue, specification.fine_ids)
            details = _explicit_structure_units(
                source_tissue,
                fine_ids=specification.fine_ids,
            )
            protection_semantics = "explicit_profile_structure"
        else:
            raise JointContractError(
                f"unknown auxiliary producer kind: {specification.producer_kind}"
            )
        path = root / f"{structure_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        digest = sha256_file(path)
        current_provenance = {
            "producer_id": AUXILIARY_PRODUCER_VERSION,
            "producer_version": AUXILIARY_PRODUCER_VERSION,
            "observation_scope": "semantic_fine_mask_topology_only",
            "protection_semantics": protection_semantics,
            "source_tissue_mask_sha256": source_digest,
            "output_sha256": digest,
            "pattern_fine_ids": list(specification.fine_ids),
            **details,
        }
        uris[structure_id] = str(path)
        digests[structure_id] = digest
        provenance_by_id[structure_id] = current_provenance
        produced.append(
            ProducedAuxiliary(
                structure_id=structure_id,
                path=str(path),
                sha256=digest,
                provenance=current_provenance,
            )
        )
    effective_provenance = {
        **case.provenance,
        "auxiliary_structure_sha256": digests,
        "auxiliary_structure_provenance": provenance_by_id,
        "available_auxiliary_structures": sorted(uris),
    }
    effective = replace(
        case,
        auxiliary_structure_uris=uris,
        provenance=effective_provenance,
    )
    _validate_bound_auxiliary_provenance(effective)
    return effective, tuple(produced)


def _profile_specifications(
    annotation_profile_id: str,
    *,
    source_tissue: np.ndarray,
) -> tuple[_AuxiliarySpecification, ...]:
    if annotation_profile_id == "glas-gland-v1":
        return (
            _AuxiliarySpecification(
                "gland_or_lumen_support",
                "enclosed_pattern_spaces",
                (5, 11, 12, 13),
            ),
        )
    if annotation_profile_id == "panda-gleason-v1":
        return (
            _AuxiliarySpecification(
                "native_pattern_and_lumen_map",
                "enclosed_pattern_spaces",
                (8, 9, 10),
            ),
            _AuxiliarySpecification(
                "native_pattern_map",
                "explicit_profile_structure",
                (8, 9, 10),
            ),
            _AuxiliarySpecification(
                "gland_lumen_map",
                "enclosed_pattern_spaces",
                (8, 9, 10),
            ),
        )
    if annotation_profile_id == "puma-semantic-v1" and np.any(
        np.asarray(source_tissue) == 5
    ):
        # PUMA fine ID 5 is explicitly mapped to epidermis by the versioned
        # annotation profile.  This is a protection/relationship map, not an
        # H&E-derived guess at a junctional component.  If epidermis is absent,
        # no map is produced and epidermis-dependent mechanisms fail closed.
        return (
            _AuxiliarySpecification(
                "epidermis_or_junction_map",
                "explicit_profile_structure",
                (5,),
            ),
        )
    return ()


def _enclosed_pattern_spaces(
    tissue: np.ndarray,
    *,
    pattern_fine_ids: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return only holes enclosed by one same-fine-label component.

    Processing each fine label separately prevents a mixed-pattern ring from
    being called one native lumen.  Border-connected complement and arbitrary
    stromal islands between separate glands are excluded by construction.
    """

    values = np.asarray(tissue)
    if values.ndim != 2:
        raise JointContractError("auxiliary producers require one 2-D tissue mask")
    protected = np.zeros_like(values, dtype=bool)
    component_count = 0
    space_count = 0
    spaces_by_fine_id: dict[str, int] = {}
    structure_units: list[dict[str, Any]] = []
    hierarchy_relations: list[dict[str, str]] = []
    for fine_id in pattern_fine_ids:
        labeled, count = ndimage.label(
            values == int(fine_id),
            structure=np.ones((3, 3), dtype=bool),
        )
        component_count += int(count)
        current_spaces = 0
        for component_index in range(1, count + 1):
            component = labeled == component_index
            if not np.any(component):
                continue
            rows, cols = np.nonzero(component)
            unit_id = f"fine:{int(fine_id)}:unit:{int(component_index):04d}"
            child_ids: list[str] = []
            filled = ndimage.binary_fill_holes(component)
            holes = filled & ~component
            if not np.any(holes):
                structure_units.append(
                    _structure_unit_record(
                        unit_id=unit_id,
                        fine_id=int(fine_id),
                        component=component,
                        rows=rows,
                        cols=cols,
                        enclosed_space_ids=(),
                    )
                )
                continue
            hole_labels, hole_count = ndimage.label(
                holes,
                structure=np.ones((3, 3), dtype=bool),
            )
            for hole_index in range(1, hole_count + 1):
                hole = hole_labels == hole_index
                # Tiny one-pixel raster pinholes are annotation noise rather
                # than auditable native spaces.
                if int(np.count_nonzero(hole)) < 4:
                    continue
                boundary = ndimage.binary_dilation(hole) & ~hole
                if not np.any(boundary) or not np.all(component[boundary]):
                    continue
                protected |= hole
                current_spaces += 1
                space_id = f"{unit_id}:space:{current_spaces:03d}"
                child_ids.append(space_id)
                hierarchy_relations.append(
                    {
                        "source_id": space_id,
                        "relation": "enclosed_space_of",
                        "target_id": unit_id,
                    }
                )
            structure_units.append(
                _structure_unit_record(
                    unit_id=unit_id,
                    fine_id=int(fine_id),
                    component=component,
                    rows=rows,
                    cols=cols,
                    enclosed_space_ids=tuple(child_ids),
                )
            )
        if current_spaces:
            spaces_by_fine_id[str(fine_id)] = current_spaces
            space_count += current_spaces
    return protected, {
        "observed_pattern_component_count": component_count,
        "enclosed_space_count": space_count,
        "enclosed_space_pixels": int(np.count_nonzero(protected)),
        "enclosed_spaces_by_fine_id": spaces_by_fine_id,
        "empty_map_is_valid_observation": True,
        "structural_hierarchy_schema": "semantic-structure-units-v1",
        "structure_units": structure_units,
        "hierarchy_relations": hierarchy_relations,
    }


def _structure_unit_record(
    *,
    unit_id: str,
    fine_id: int,
    component: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    enclosed_space_ids: tuple[str, ...],
) -> dict[str, Any]:
    packed = np.packbits(np.asarray(component, dtype=np.uint8), axis=None)
    return {
        "unit_id": unit_id,
        "unit_type": "gland_or_native_pattern_component",
        "fine_id": int(fine_id),
        "area_px": int(np.count_nonzero(component)),
        "bbox_xyxy": [
            int(cols.min()),
            int(rows.min()),
            int(cols.max()) + 1,
            int(rows.max()) + 1,
        ],
        "component_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        "enclosed_space_ids": list(enclosed_space_ids),
        "parent_relation": "member_of_tissue_component_resolved_in_scene",
    }


def _explicit_structure_units(
    tissue: np.ndarray,
    *,
    fine_ids: tuple[int, ...],
) -> dict[str, Any]:
    units = []
    for fine_id in fine_ids:
        labeled, count = ndimage.label(
            np.asarray(tissue) == int(fine_id),
            structure=np.ones((3, 3), dtype=bool),
        )
        for component_index in range(1, count + 1):
            component = labeled == component_index
            rows, cols = np.nonzero(component)
            if not len(rows):
                continue
            units.append(
                _structure_unit_record(
                    unit_id=(
                        f"fine:{int(fine_id)}:unit:{int(component_index):04d}"
                    ),
                    fine_id=int(fine_id),
                    component=component,
                    rows=rows,
                    cols=cols,
                    enclosed_space_ids=(),
                )
            )
    return {
        "observed_structure_pixels": int(
            np.count_nonzero(np.isin(tissue, fine_ids))
        ),
        "profile_fine_ids": list(fine_ids),
        "empty_map_is_valid_observation": False,
        "structural_hierarchy_schema": "explicit-profile-structure-v1",
        "structure_units": units,
        "hierarchy_relations": [],
    }


def _validate_bound_auxiliary_provenance(case: JointCaseContext) -> None:
    if not case.auxiliary_structure_uris:
        return
    digests = case.provenance.get("auxiliary_structure_sha256")
    records = case.provenance.get("auxiliary_structure_provenance")
    if not isinstance(digests, dict) or not isinstance(records, dict):
        raise JointContractError(
            "auxiliary structures require digest and producer provenance maps"
        )
    if set(case.auxiliary_structure_uris) != set(digests) or set(digests) != set(records):
        raise JointContractError(
            "auxiliary URI, digest and producer provenance IDs differ"
        )
    source_digest = case.provenance.get("source_tissue_mask_sha256")
    for structure_id in sorted(case.auxiliary_structure_uris):
        record = records.get(structure_id)
        if not isinstance(record, dict):
            raise JointContractError(
                f"auxiliary structure {structure_id!r} has no producer record"
            )
        if record.get("output_sha256") != digests[structure_id]:
            raise JointContractError(
                f"auxiliary structure {structure_id!r} output digest is unbound"
            )
        if record.get("source_tissue_mask_sha256") != source_digest:
            raise JointContractError(
                f"auxiliary structure {structure_id!r} source digest is unbound"
            )
        if not record.get("producer_id") or not record.get("producer_version"):
            raise JointContractError(
                f"auxiliary structure {structure_id!r} producer identity is missing"
            )
