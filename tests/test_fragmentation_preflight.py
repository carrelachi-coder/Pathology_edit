from phase3_joint_edit_refine.executable_contract import (
    _protected_population_may_cross_tissue_change,
)
from phase3_joint_edit_refine.feasibility import (
    _protected_instance_is_tissue_exclusion,
)


def test_fragmentation_freezes_protected_cell_but_not_underlying_tissue():
    assert not _protected_instance_is_tissue_exclusion(
        tissue_geometry_mode="residual_fragmentation",
        class_id=1,
        required_clearance_classes={1},
        target_compatible_classes={2},
    )


def test_nonfragmentation_keeps_incompatible_protected_cell_as_tissue_exclusion():
    assert _protected_instance_is_tissue_exclusion(
        tissue_geometry_mode="natural_external_retreat",
        class_id=1,
        required_clearance_classes={1},
        target_compatible_classes={2},
    )


def test_fragmentation_contract_retains_existing_protected_population():
    assert _protected_population_may_cross_tissue_change(
        tissue_geometry_mode="residual_fragmentation"
    )
    assert not _protected_population_may_cross_tissue_change(
        tissue_geometry_mode="natural_external_retreat"
    )
