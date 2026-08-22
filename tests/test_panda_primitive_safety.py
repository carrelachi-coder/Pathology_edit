from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from phase3_joint_edit_refine.auxiliary import materialize_profile_auxiliaries
from phase3_joint_edit_refine.gates import (
    audit_directional_extension_raster,
)
from phase3_joint_edit_refine.models import JointCaseContext
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_mask_edit_refine.candidates import (
    compile_directional_tapered_projection_field,
)


def test_panda_growth_catalog_has_no_independent_generic_burden_primitive() -> None:
    repository = JointSkillRepository()
    for mechanism_id in (
        "prostate-pattern-3-growth",
        "prostate-pattern-4-growth",
        "prostate-pattern-5-growth",
    ):
        mechanism = repository.mechanisms[mechanism_id]
        assert "tumor-burden-increase-v1" not in mechanism.supported_primitives
        assert "cohesive-boundary-expansion-v1" in mechanism.supported_primitives


def test_pattern5_organic_cord_is_long_rounded_and_cell_scale() -> None:
    shape = (180, 180)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    parent = (rows - 90) ** 2 + (cols - 52) ** 2 <= 30**2
    legal = ~parent
    anchor = parent & ndimage.binary_dilation(legal)
    anchor &= cols >= 75

    projection, priority = compile_directional_tapered_projection_field(
        legal,
        anchor_mask=anchor,
        parent_mask=parent,
        maximum_depth_px=90,
        maximum_width_px=24,
        tip_width_px=12,
        shape_mode="organic_rounded_cord",
    )
    finite = np.flatnonzero(np.isfinite(priority))
    selected_ids = finite[np.argsort(priority.flat[finite])[:900]]
    selected = np.zeros(shape, dtype=bool)
    selected.flat[selected_ids] = True

    audit = audit_directional_extension_raster(
        change=selected,
        parent=parent,
        other_tumor=np.zeros(shape, dtype=bool),
        selected_anchor=anchor,
        nominal_nucleus_diameter_px=12,
        minimum_directionality_ratio=1.0,
        minimum_skeleton_length_width_ratio=5.0,
    )

    assert projection.sum() > selected.sum() >= 900
    assert audit["passed"], audit
    assert audit["longitudinal_span_px"] >= 5 * 12
    assert 0.45 <= audit["tip_to_neck_width_ratio"] <= 0.90
    assert audit["maximum_width_px"] >= 12


def test_panda_auxiliary_separates_lumen_wall_and_external_stroma(tmp_path) -> None:
    shape = (128, 128)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    radius = np.sqrt((rows - 64) ** 2 + (cols - 64) ** 2)
    tissue = np.full(shape, 2, dtype=np.uint8)
    tissue[(radius >= 25) & (radius <= 38)] = 9
    image = np.full((*shape, 3), (210, 155, 180), dtype=np.uint8)
    image[radius < 25] = (245, 225, 232)
    image[(radius >= 25) & (radius <= 38)] = (180, 90, 135)
    image[:20, :20] = 255
    nuclei = np.zeros(shape, dtype=np.uint8)
    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        y = int(round(64 + 31 * np.sin(angle)))
        x = int(round(64 + 31 * np.cos(angle)))
        nuclei[(rows - y) ** 2 + (cols - x) ** 2 <= 3**2] = 1
    for y, x in ((105, 105), (108, 92), (94, 108)):
        nuclei[(rows - y) ** 2 + (cols - x) ** 2 <= 3**2] = 3

    case = JointCaseContext(
        case_id="panda-auxiliary-safety",
        instruction="protect gland units",
        source_image_uri="unused-image.png",
        source_tissue_mask_uri="unused-tissue.png",
        source_nuclei_mask_uri="unused-nuclei.png",
        pathology_domain_id="prostate-adenocarcinoma-v1",
        annotation_profile_id="panda-gleason-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id="prostate-cell-population-v1",
        primitive_id="cohesive-boundary-expansion-v1",
        joint_area_budget=None,
        seed=17,
        provenance={
            "source_image_sha256": "image-digest",
            "source_tissue_mask_sha256": "tissue-digest",
            "source_nuclei_mask_sha256": "nuclei-digest",
        },
    )
    effective, _ = materialize_profile_auxiliaries(
        case,
        source_tissue=tissue,
        source_image=image,
        source_nuclei=nuclei,
        output_dir=tmp_path,
    )
    lumen = np.asarray(
        Image.open(effective.auxiliary_structure_uris["gland_lumen_map"])
    ) > 0
    wall = np.asarray(
        Image.open(effective.auxiliary_structure_uris["gland_unit_wall_map"])
    ) > 0
    external = np.asarray(
        Image.open(
            effective.auxiliary_structure_uris["external_cellular_stroma_map"]
        )
    ) > 0

    assert lumen[64, 64]
    assert wall[64, 64] and wall[64, 94]
    assert external[105, 105]
    assert not external[64, 64]
    assert not external[5, 5]
