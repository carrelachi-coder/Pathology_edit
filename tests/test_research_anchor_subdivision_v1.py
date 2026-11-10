import numpy as np
import pytest

from phase3_joint_edit_refine.tissue_planner import (
    _directional_sector_selection_limit,
)
from phase3_mask_edit_refine.research_anchor_subdivision_v1 import (
    bisect_connected_interface_anchor,
)


def test_bisection_is_deterministic_exhaustive_and_disjoint():
    anchor = np.zeros((12, 20), dtype=bool)
    anchor[3, 2:17] = True
    anchor[3:10, 16] = True

    first = bisect_connected_interface_anchor(anchor)
    second = bisect_connected_interface_anchor(anchor)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert np.array_equal(first[0] | first[1], anchor)
    assert not np.any(first[0] & first[1])
    assert all(np.count_nonzero(part) >= 2 for part in first)


def test_two_subanchors_retain_one_unedited_witness_without_changing_rule():
    anchor = np.zeros((5, 12), dtype=bool)
    anchor[2, 1:11] = True
    parts = bisect_connected_interface_anchor(anchor)
    ids = tuple(f"subanchor:{index}" for index in range(len(parts)))

    assert _directional_sector_selection_limit(
        interface_anchor_ids=ids,
        allowed_anchor_ids=ids,
        maximum_selected_anchor_fraction=0.75,
        minimum_unselected_anchor_count=1,
    ) == 1


def test_small_or_disconnected_anchor_fails_closed():
    too_small = np.zeros((3, 3), dtype=bool)
    too_small[1, 1:3] = True
    with pytest.raises(ValueError, match="too small"):
        bisect_connected_interface_anchor(too_small)

    disconnected = np.zeros((5, 8), dtype=bool)
    disconnected[1, 1:4] = True
    disconnected[3, 4:7] = True
    with pytest.raises(ValueError, match="8-connected"):
        bisect_connected_interface_anchor(disconnected)
