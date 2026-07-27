import numpy as np
import pytest

pytest.importorskip("cv2")
from inpaint_cells.data.prob_dataset import _choose_crop_origin


def test_validation_crop_is_centered_on_edit_region_and_repeatable():
    edit_mask = np.zeros((512, 512), dtype=np.float32)
    edit_mask[300:380, 100:180] = 1

    first = _choose_crop_origin(512, 512, 256, edit_mask, 'mask', deterministic=True)
    second = _choose_crop_origin(512, 512, 256, edit_mask, 'mask', deterministic=True)

    assert first == second == (211, 11)


def test_validation_crop_without_mask_uses_image_center():
    edit_mask = np.zeros((512, 768), dtype=np.float32)

    assert _choose_crop_origin(512, 768, 256, edit_mask, 'mask', deterministic=True) == (128, 256)
