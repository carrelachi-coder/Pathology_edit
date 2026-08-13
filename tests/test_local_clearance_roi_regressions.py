from types import SimpleNamespace

import numpy as np
import pytest

from phase3_joint_edit_refine.feasibility import (
    augment_tissue_scene_with_nuclei_preflight,
)
from phase3_joint_edit_refine.models import JointContractError
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.scene import build_scene_analysis


def _scene_and_preflight():
    tissue = np.full((24, 24), 2, dtype=np.uint8)
    tissue[6:18, 6:18] = 1
    scene = build_scene_analysis(
        tissue,
        schema=MaskProfileSchema.from_reference_profile("BCSS"),
    )
    preflight = SimpleNamespace(
        protected_tissue_change_mask=np.zeros_like(tissue, dtype=bool)
    )
    return scene, preflight


def test_receiving_roi_prohibits_its_complement_before_tissue_execution():
    scene, preflight = _scene_and_preflight()
    roi = np.zeros((24, 24), dtype=bool)
    roi[8:16, 8:16] = True

    augmented = augment_tissue_scene_with_nuclei_preflight(
        scene,
        preflight,
        auxiliary_structure_masks={"local_clearance_roi": roi},
        required_auxiliary_structure_ids=("local_clearance_roi",),
        receiving_auxiliary_structure_ids=("local_clearance_roi",),
    )

    prohibited = augmented.prohibited_region_masks[
        "joint:auxiliary:local_clearance_roi"
    ]
    assert np.array_equal(prohibited, ~roi)


def test_protected_auxiliary_remains_directly_prohibited():
    scene, preflight = _scene_and_preflight()
    lumen = np.zeros((24, 24), dtype=bool)
    lumen[10:14, 10:14] = True

    augmented = augment_tissue_scene_with_nuclei_preflight(
        scene,
        preflight,
        auxiliary_structure_masks={"lumen": lumen},
        required_auxiliary_structure_ids=("lumen",),
    )

    assert np.array_equal(
        augmented.prohibited_region_masks["joint:auxiliary:lumen"], lumen
    )


def test_receiving_auxiliary_must_be_present():
    scene, preflight = _scene_and_preflight()
    with pytest.raises(JointContractError, match="is unavailable"):
        augment_tissue_scene_with_nuclei_preflight(
            scene,
            preflight,
            auxiliary_structure_masks={},
            receiving_auxiliary_structure_ids=("roi",),
        )
