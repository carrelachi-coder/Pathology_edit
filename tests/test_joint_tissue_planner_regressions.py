from phase3_joint_edit_refine.tissue_planner import (
    _component_turnover_profile_mode,
)
from phase3_joint_edit_refine.skills.repository import (
    _instruction_has_explicit_local_selection,
)


def test_partial_component_turnover_keeps_mechanism_profile() -> None:
    assert (
        _component_turnover_profile_mode(
            preferred_mode="multi_lobe",
            allow_source_component_resolution=True,
            requested_allocation_px=400,
            source_component_area_px=10_000,
        )
        == "multi_lobe"
    )


def test_complete_component_resolution_uses_uniform_front() -> None:
    assert (
        _component_turnover_profile_mode(
            preferred_mode="multi_lobe",
            allow_source_component_resolution=True,
            requested_allocation_px=10_000,
            source_component_area_px=10_000,
        )
        == "uniform_front"
    )


def test_selected_region_is_an_explicit_local_selection() -> None:
    assert _instruction_has_explicit_local_selection(
        "Clear invasive tumor in this selected region."
    )


def test_unbounded_clearance_is_not_an_explicit_local_selection() -> None:
    assert not _instruction_has_explicit_local_selection(
        "Clear invasive tumor."
    )
