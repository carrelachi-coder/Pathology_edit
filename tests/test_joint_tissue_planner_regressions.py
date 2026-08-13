from types import SimpleNamespace

import numpy as np

from phase3_mask_edit_refine.gates import _check_boundary_naturalness
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


def _annular_boundary_context(*, geometry_mode: str):
    rows, cols = np.ogrid[:256, :256]
    radius_squared = (rows - 128) ** 2 + (cols - 128) ** 2
    change = (radius_squared >= 76**2) & (radius_squared <= 80**2)
    return SimpleNamespace(
        candidate=SimpleNamespace(change_region=change),
        plan=SimpleNamespace(
            tool_program=SimpleNamespace(
                parameter_ranges={
                    "tissue_geometry_mode": geometry_mode,
                    "max_boundary_compactness": 40.0,
                }
            )
        ),
    )


def test_component_turnover_does_not_treat_annulus_compactness_as_roughness() -> None:
    result = _check_boundary_naturalness(
        _annular_boundary_context(geometry_mode="component_boundary_turnover")
    )

    assert result.passed
    assert not result.metrics["change_band_compactness_applicable"]
    assert result.metrics["maximum_component_compactness"] > 40.0


def test_interface_front_uses_front_specific_geometry_gates() -> None:
    result = _check_boundary_naturalness(
        _annular_boundary_context(geometry_mode="interface_front")
    )

    assert result.passed
    assert not result.metrics["change_band_compactness_applicable"]
    assert result.metrics["boundary_attached_geometry"]
