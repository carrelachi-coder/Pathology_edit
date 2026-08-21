"""Joint change accounting with no tissue/cell double counting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .models import ChangeLedger, JointCandidate, JointContractError
from .nuclei import iter_instances, normalize_nuclei_mask


@dataclass(frozen=True)
class LedgerAnalysis:
    tissue_change: np.ndarray
    cell_change: np.ndarray
    joint_change: np.ndarray
    generation_support: np.ndarray
    ledger: ChangeLedger
    whole_instance_changes: bool
    partial_source_instance_ids: tuple[str, ...]


def analyze_joint_change(
    *,
    source_tissue: np.ndarray,
    target_tissue: np.ndarray,
    source_nuclei: np.ndarray,
    target_nuclei: np.ndarray,
    generation_halo_px: int,
    generation_allowed_region: np.ndarray | None = None,
    generation_support_contract: np.ndarray | None = None,
    source_instance_masks: dict[str, np.ndarray] | None = None,
    source_instance_classes: dict[str, int] | None = None,
    erased_source_instance_ids: tuple[str, ...] | None = None,
) -> LedgerAnalysis:
    source_tissue = np.asarray(source_tissue)
    target_tissue = np.asarray(target_tissue)
    source_nuclei = normalize_nuclei_mask(source_nuclei)
    target_nuclei = normalize_nuclei_mask(target_nuclei)
    shape = source_tissue.shape
    if any(
        np.asarray(value).shape != shape
        for value in (target_tissue, source_nuclei, target_nuclei)
    ):
        raise JointContractError("joint ledger inputs must share one shape")
    if not 0 <= int(generation_halo_px) <= 128:
        raise JointContractError("generation halo must be in [0, 128]")

    tissue_change = source_tissue != target_tissue
    removed = np.zeros(shape, dtype=bool)
    retained = np.zeros(shape, dtype=bool)
    removed_ids: list[str] = []
    retained_ids: list[str] = []
    partial_ids: list[str] = []
    if source_instance_masks is not None:
        if source_instance_classes is None or set(source_instance_masks) != set(source_instance_classes):
            raise JointContractError("native source instance masks/classes are inconsistent")
        source_instances = tuple(
            (instance_id, int(source_instance_classes[instance_id]), np.asarray(component, dtype=bool))
            for instance_id, component in sorted(source_instance_masks.items())
        )
    else:
        source_instances = tuple(iter_instances(source_nuclei))
    certified_erased = (
        set(erased_source_instance_ids)
        if erased_source_instance_ids is not None
        else None
    )
    observed_source_ids = {item[0] for item in source_instances}
    if certified_erased is not None and not certified_erased.issubset(
        observed_source_ids
    ):
        raise JointContractError(
            "certified erasure ledger contains an unknown source instance"
        )
    for instance_id, class_id, component in source_instances:
        if certified_erased is not None and instance_id in certified_erased:
            removed |= component
            removed_ids.append(instance_id)
            continue
        # Native instance authority supplies identity and class, while the
        # persisted semantic raster remains the pixel baseline.  A CellViT
        # class can legitimately differ from that raster; unchanged native
        # instances must therefore be compared with their source pixels, not
        # painted wholesale as if every footprint pixel equalled ``class_id``.
        same = target_nuclei[component] == source_nuclei[component]
        if np.all(same):
            retained |= component
            retained_ids.append(instance_id)
            continue
        if np.any(same):
            partial_ids.append(instance_id)
        removed |= component
        removed_ids.append(instance_id)

    # Cell generation owns only pixels whose target class differs from the
    # source.  Do not promote an entire semantic target component to "added":
    # a newly placed nucleus can become 8-connected to a retained same-class
    # nucleus (especially in dense semantic masks), and the target connected
    # component may then span far beyond the executable generation support.
    added = (target_nuclei != source_nuclei) & (target_nuclei != 0)
    added_ids: list[str] = []
    for instance_id, class_id, component in iter_instances(target_nuclei):
        if np.any(component & added):
            added_ids.append("target-" + instance_id)

    cell_change = removed | added
    joint_change = tissue_change | cell_change
    if generation_support_contract is not None:
        generation_support = np.asarray(generation_support_contract, dtype=bool)
        if generation_support.shape != shape:
            raise JointContractError(
                "generation support contract must match mask shape"
            )
        outside_support = joint_change & ~generation_support
        if np.any(outside_support):
            rows, cols = np.nonzero(outside_support)
            bbox = [
                int(cols.min()),
                int(rows.min()),
                int(cols.max()) + 1,
                int(rows.max()) + 1,
            ]
            raise JointContractError(
                "joint change extends outside executable generation support: "
                f"pixels={len(rows)}, bbox_xyxy={bbox}"
            )
    elif generation_halo_px and np.any(joint_change):
        structure = ndimage.generate_binary_structure(2, 1)
        generation_support = ndimage.binary_dilation(
            joint_change,
            structure=structure,
            iterations=int(generation_halo_px),
        )
    else:
        generation_support = joint_change.copy()
    if generation_allowed_region is not None:
        allowed = np.asarray(generation_allowed_region, dtype=bool)
        if allowed.shape != shape:
            raise JointContractError("generation allowed region must match mask shape")
        if generation_support_contract is not None:
            if np.any(generation_support & ~allowed):
                raise JointContractError(
                    "executable generation support enters a prohibited source region"
                )
        else:
            generation_support = joint_change | (generation_support & allowed)
    ledger = ChangeLedger(
        tissue_pixels=int(tissue_change.sum()),
        removed_nucleus_pixels=int(removed.sum()),
        added_nucleus_pixels=int(added.sum()),
        cell_pixels=int(cell_change.sum()),
        cell_only_pixels=int(np.count_nonzero(cell_change & ~tissue_change)),
        joint_pixels=int(joint_change.sum()),
        generation_support_pixels=int(generation_support.sum()),
        total_pixels=int(np.prod(shape)),
        removed_instance_ids=tuple(removed_ids),
        added_instance_ids=tuple(added_ids),
        retained_instance_ids=tuple(retained_ids),
    )
    return LedgerAnalysis(
        tissue_change=tissue_change,
        cell_change=cell_change,
        joint_change=joint_change,
        generation_support=generation_support,
        ledger=ledger,
        whole_instance_changes=not partial_ids,
        partial_source_instance_ids=tuple(partial_ids),
    )


def build_joint_candidate(
    *,
    candidate_id: str,
    tissue_candidate_id: str,
    cell_candidate_id: str,
    mechanism_id: str,
    source_tissue: np.ndarray,
    target_tissue: np.ndarray,
    source_nuclei: np.ndarray,
    target_nuclei: np.ndarray,
    generation_halo_px: int,
    generation_allowed_region: np.ndarray | None = None,
    generation_support_contract: np.ndarray | None = None,
    source_instance_masks: dict[str, np.ndarray] | None = None,
    source_instance_classes: dict[str, int] | None = None,
    erased_source_instance_ids: tuple[str, ...] | None = None,
    tool_trace: dict,
) -> JointCandidate:
    analysis = analyze_joint_change(
        source_tissue=source_tissue,
        target_tissue=target_tissue,
        source_nuclei=source_nuclei,
        target_nuclei=target_nuclei,
        generation_halo_px=generation_halo_px,
        generation_allowed_region=generation_allowed_region,
        generation_support_contract=generation_support_contract,
        source_instance_masks=source_instance_masks,
        source_instance_classes=source_instance_classes,
        erased_source_instance_ids=erased_source_instance_ids,
    )
    return JointCandidate(
        candidate_id=candidate_id,
        tissue_candidate_id=tissue_candidate_id,
        cell_candidate_id=cell_candidate_id,
        mechanism_id=mechanism_id,
        target_tissue_mask=np.asarray(target_tissue).copy(),
        target_nuclei_mask=normalize_nuclei_mask(target_nuclei),
        tissue_change=analysis.tissue_change,
        cell_change=analysis.cell_change,
        joint_change=analysis.joint_change,
        generation_support=analysis.generation_support,
        ledger=analysis.ledger,
        tool_trace={
            **tool_trace,
            "whole_instance_changes": analysis.whole_instance_changes,
            "partial_source_instance_ids": list(analysis.partial_source_instance_ids),
        },
    )
