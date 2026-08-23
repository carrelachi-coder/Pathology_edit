"""Focused regressions for approved joint-handoff generation routing."""

from __future__ import annotations

import numpy as np

from phase3_joint_edit_refine.generator_adapter import (
    JointGeneratorRoutingConfig,
    build_agentic_joint_route,
    route_joint_handoff,
)


def _manifest(support_fraction: float) -> dict[str, object]:
    return {
        "ledger": {
            "joint_fraction": 0.04,
            "generation_support_fraction": support_fraction,
        }
    }


def test_default_route_forces_cross_at_exactly_half_image_support() -> None:
    route = route_joint_handoff(_manifest(0.50))

    assert route.mode == "cross"
    assert route.force_cross is True

    change = np.zeros((10, 10), dtype=bool)
    change.flat[:4] = True
    support = np.zeros((10, 10), dtype=bool)
    support.flat[:50] = True
    decision = build_agentic_joint_route(
        _manifest(0.50),
        joint_change_mask=change,
        generation_support_mask=support,
        reference_tissue_mask=np.ones((10, 10), dtype=np.uint8),
    )

    assert decision.primary_mode == "cross"
    assert decision.candidate_modes == ("cross",)


def test_support_just_below_half_is_not_cross_only() -> None:
    route = route_joint_handoff(_manifest(0.49))

    assert route.mode == "inpaint"
    assert route.force_cross is False


def test_explicit_threshold_override_remains_supported() -> None:
    route = route_joint_handoff(
        _manifest(0.30),
        config=JointGeneratorRoutingConfig(
            force_cross_min_generation_support_fraction=0.30
        ),
    )

    assert route.mode == "cross"
    assert route.force_cross is True


def test_cell_only_decrease_defaults_to_inpaint_with_cross_fallback() -> None:
    manifest = {
        "primitive_id": "cell-type-abundance-decrease-v1",
        "ledger": {
            "tissue_fraction": 0.0,
            "cell_fraction": 0.04,
            "joint_fraction": 0.04,
            "generation_support_fraction": 0.10,
        },
    }

    route = route_joint_handoff(manifest)

    assert route.mode == "inpaint"
    assert route.force_cross is False
    assert "full stained nucleus footprint" in route.reason
