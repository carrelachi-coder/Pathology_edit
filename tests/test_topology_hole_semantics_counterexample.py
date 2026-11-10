"""Document a local/whole-mask topology disagreement without changing policy."""

import numpy as np
from scipy import ndimage

from phase3_mask_edit_refine.execution import _hole_count
from phase3_mask_edit_refine.topology import _topology_rejection_reason


def test_local_guard_can_accept_while_whole_mask_hole_count_changes():
    source = np.array(
        [
            [1, 0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1],
        ],
        dtype=bool,
    )
    target = np.zeros_like(source)
    target[1, 4] = True
    unselected_target = np.zeros_like(source)
    row, col = 2, 3

    assert _topology_rejection_reason(
        row,
        col,
        source_component_state=source,
        target_state=target,
        unselected_target=unselected_target,
    ) is None

    source_after = source.copy()
    source_after[row, col] = False
    assert (_hole_count(source), _hole_count(source_after)) == (2, 1)

    before_holes = ndimage.binary_fill_holes(source) & ~source
    after_holes = ndimage.binary_fill_holes(source_after) & ~source_after
    assert (ndimage.label(before_holes)[1], ndimage.label(after_holes)[1]) == (3, 3)
    assert (ndimage.label(source, structure=np.ones((3, 3), bool))[1],
            ndimage.label(source_after, structure=np.ones((3, 3), bool))[1]) == (1, 1)
    assert (_hole_count(target), _hole_count(target | (source & ~source_after))) == (0, 0)
