"""Fail-closed raster authority for mask-graph LLM stages."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .models import JointContractError


MASK_PLANNER_ARTIFACT_NAMES = frozenset(
    {
        "planner_01_tissue_mask.png",
        "planner_02_component_map.png",
        "planner_03_interface_anchor_map.png",
        "planner_mask_tissue_nuclei.png",
        "joint_condition_mask_review.png",
    }
)


def validate_mask_planner_image_paths(
    image_paths: Sequence[str | Path],
) -> tuple[str, ...]:
    """Reject any raster that is not a pipeline-owned mask planning panel.

    Prompt wording is not a security boundary.  This validator makes the
    observation contract executable: raw H&E, H&E overlays, arbitrary crops,
    and reader-facing review boards cannot be attached to a Planner or mask
    critic request even when a caller bypasses the normal workflow.
    """

    normalized = tuple(str(Path(item)) for item in image_paths)
    unauthorized = sorted(
        {
            Path(item).name
            for item in normalized
            if Path(item).name not in MASK_PLANNER_ARTIFACT_NAMES
        }
    )
    if unauthorized:
        raise JointContractError(
            "mask-graph LLM input contains an unauthorized raster artifact: "
            + ", ".join(unauthorized)
        )
    return normalized
