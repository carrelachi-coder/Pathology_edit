"""One source-nucleus instance authority shared by every joint stage."""

from __future__ import annotations

import numpy as np

from inpaint_cells.instance_authority import (
    binary_mask_sha256,
    build_instance_authority,
)

from .nuclei import to_raw_nuclei_mask
from .scene import JointSceneAnalysis


def build_scene_instance_authority(
    scene: JointSceneAnalysis,
    source_nuclei: np.ndarray,
) -> dict:
    """Return the sole count/shape/density authority for one source patch."""

    return build_instance_authority(
        shape=np.asarray(source_nuclei).shape,
        source_nuclei_raw=to_raw_nuclei_mask(source_nuclei),
        observation_quality=scene.cells.observation_quality,
        instances=(
            {
                "instance_id": item.instance_id,
                "raw_class_id": 100 + int(item.class_id),
                "row": float(item.centroid_xy[1]),
                "col": float(item.centroid_xy[0]),
                "tissue_fine_id": int(item.tissue_fine_id),
                "completeness_status": item.completeness_status,
                "source": item.source,
                "area_px": item.area_px,
                "bbox_xyxy": item.bbox_xyxy,
                "footprint_sha256": binary_mask_sha256(
                    scene.instance_masks[item.instance_id]
                ),
            }
            for item in scene.cells.instances
        ),
    )


def authority_trace(authority: dict) -> dict:
    return {
        "schema_version": authority["schema_version"],
        "authority_sha256": authority["authority_sha256"],
        "observation_quality": authority["observation_quality"],
        "instance_count": len(authority["instances"]),
    }
