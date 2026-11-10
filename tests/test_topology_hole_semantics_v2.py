import numpy as np

from phase3_mask_edit_refine import execution, gates
from phase3_mask_edit_refine.topology import _topology_rejection_reason
from phase3_mask_edit_refine.topology_semantics import (
    WHOLE_MASK_HOLE_SEMANTICS_VERSION,
    hole_count_4,
    hole_intersection_count_4,
)


def _accepted_counterexample():
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
    return source, target, (2, 3)


def test_candidate_matches_local_guard_on_accepted_counterexample():
    source, target, (row, col) = _accepted_counterexample()
    assert _topology_rejection_reason(
        row,
        col,
        source_component_state=source,
        target_state=target,
        unselected_target=np.zeros_like(source),
    ) is None
    source_after = source.copy()
    source_after[row, col] = False
    assert (hole_count_4(source), hole_count_4(source_after)) == (3, 3)


def test_diagonal_holes_remain_distinct_and_intersections_are_counted():
    mask = np.ones((5, 5), dtype=bool)
    mask[1, 1] = False
    mask[2, 2] = False
    region = np.zeros_like(mask)
    region[1, 1] = True
    assert hole_count_4(mask) == 2
    assert hole_intersection_count_4(mask, region) == 1


def test_execution_and_gate_use_one_versioned_shared_helper():
    assert WHOLE_MASK_HOLE_SEMANTICS_VERSION == "digital-topology-8fg-4bg-v2"
    assert execution._hole_count is hole_count_4
    assert gates._hole_count is hole_count_4
    assert execution._hole_intersection_count is hole_intersection_count_4
    assert gates._hole_intersection_count is hole_intersection_count_4
    assert execution.EXECUTION_SOLVER_VERSION == "mask-edit-refine-topology-solver-v19"
