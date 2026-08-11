from __future__ import annotations

import hashlib

import numpy as np

from phase3_joint_edit_refine.auxiliary import _profile_specifications
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_joint_edit_refine.specialized_execution import (
    compile_structural_void_execution,
    execute_architecture_progression,
    validate_architecture_postcondition,
)
from phase3_mask_edit.core.labels import MaskProfileSchema


def test_structural_void_compiler_requires_separated_complete_footprints():
    shape = (128, 128)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    tissue = np.full(shape, 2, dtype=np.uint8)
    tumor = (rows - 64) ** 2 + (cols - 28) ** 2 <= 18**2
    tissue[tumor] = 1
    nuclei = np.zeros(shape, dtype=np.uint8)
    for row, col in ((55, 22), (64, 28), (73, 34)):
        nuclei[row - 2 : row + 3, col - 2 : col + 3] = 1
    airspace = np.zeros(shape, dtype=bool)
    airspace[20:108, 56:118] = True
    alveolar = np.zeros(shape, dtype=bool)
    alveolar[:, 54:56] = True

    contract = compile_structural_void_execution(
        source_tissue=tissue,
        source_nuclei=nuclei,
        source_tissue_sha256=hashlib.sha256(tissue.tobytes()).hexdigest(),
        primary_tumor_region=tumor,
        receiving_void_region=airspace,
        protected_structure_region=alveolar,
        target_delta_count=4,
        maximum_primary_separation_px=96,
    )

    assert contract.estimated_capacity >= 4
    assert not np.any(contract.placement_center_region & alveolar)
    assert not np.any(contract.valid_footprint_region & alveolar)
    assert contract.minimum_primary_separation_px >= 5


def test_architecture_adapter_changes_whole_pattern_components_and_protects_lumen():
    tissue = np.full((128, 128), 2, dtype=np.uint8)
    rows, cols = np.ogrid[:128, :128]
    left_outer = (rows - 42) ** 2 + (cols - 42) ** 2 <= 20**2
    left_lumen = (rows - 42) ** 2 + (cols - 42) ** 2 < 7**2
    right_outer = (rows - 86) ** 2 + (cols - 86) ** 2 <= 16**2
    right_lumen = (rows - 86) ** 2 + (cols - 86) ** 2 < 6**2
    tissue[left_outer & ~left_lumen] = 8
    tissue[right_outer & ~right_lumen] = 8
    lumen = left_lumen | right_lumen

    candidate = execute_architecture_progression(
        tissue,
        schema=MaskProfileSchema.from_reference_profile("PANDA"),
        transition_id="gleason_upgrade_3to4",
        target_tissue_pixels=900,
        gland_lumen_map=lumen,
    )
    report = validate_architecture_postcondition(
        source_tissue=tissue,
        candidate=candidate,
        transition_id="gleason_upgrade_3to4",
        gland_lumen_map=lumen,
    )

    assert report["passed"]
    assert not np.any(candidate.change_region & lumen)
    assert np.all(candidate.target_mask[candidate.change_region] == 9)


def test_auxiliary_roles_do_not_treat_receiving_void_as_protected():
    repository = JointSkillRepository()
    stas = repository.mechanisms["lung-stas-airspace-spread"]
    architecture = repository.mechanisms[
        "prostate-gleason-architecture-progression"
    ]

    assert stas.representability.receiving_auxiliary_structures == (
        "airspace_void_map",
    )
    assert stas.representability.protected_auxiliary_structures == (
        "alveolar_structure_map",
    )
    assert architecture.representability.receiving_auxiliary_structures == (
        "native_pattern_map",
    )
    assert architecture.representability.protected_auxiliary_structures == (
        "gland_lumen_map",
    )
    panda = _profile_specifications(
        "panda-gleason-v1",
        source_tissue=np.zeros((8, 8), dtype=np.uint8),
    )
    assert {item.structure_id for item in panda} >= {
        "native_pattern_map",
        "gland_lumen_map",
    }
