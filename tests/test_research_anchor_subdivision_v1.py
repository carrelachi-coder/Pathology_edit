import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from phase3_joint_edit_refine.feasibility import build_joint_nuclei_preflight
from phase3_joint_edit_refine.models import JointAreaBudget, JointCaseContext
from phase3_joint_edit_refine.scene import build_joint_scene_analysis
from phase3_joint_edit_refine.tissue_planner import (
    _directional_sector_selection_limit,
)
from phase3_joint_edit_refine.workflow import JointPathologyEditWorkflow
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.research_anchor_subdivision_v1 import (
    bisect_connected_interface_anchor,
    minimum_witness_contact_pixels,
    rebuild_scene_with_bisected_single_anchor,
    subdivision_eligibility_for_underfill_replan,
)
from phase3_mask_edit_refine.scene import build_scene_analysis


def test_minimum_witness_uses_one_reference_equivalent_diameter():
    assert minimum_witness_contact_pixels(9.0) == 4
    assert minimum_witness_contact_pixels(641.6) == 29
    with pytest.raises(ValueError, match="positive and finite"):
        minimum_witness_contact_pixels(0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        minimum_witness_contact_pixels(float("nan"))


def test_reference_scaled_minimum_rejects_short_arcs_before_bisection():
    minimum = minimum_witness_contact_pixels(641.6)
    short = np.zeros((3, 52), dtype=bool)
    short[1, 1:51] = True
    with pytest.raises(ValueError, match="too small"):
        bisect_connected_interface_anchor(
            short,
            minimum_subanchor_pixels=minimum,
        )

    long = np.zeros((3, 87), dtype=bool)
    long[1, 1:86] = True
    parts = bisect_connected_interface_anchor(
        long,
        minimum_subanchor_pixels=minimum,
    )
    assert all(np.count_nonzero(part) >= minimum for part in parts)

    thick_but_short = np.ones((20, 20), dtype=bool)
    with pytest.raises(ValueError, match="insufficient span"):
        bisect_connected_interface_anchor(
            thick_but_short,
            minimum_subanchor_pixels=minimum,
        )


def test_bisection_is_deterministic_exhaustive_and_disjoint():
    anchor = np.zeros((12, 20), dtype=bool)
    anchor[3, 2:17] = True
    anchor[3:10, 16] = True

    first = bisect_connected_interface_anchor(anchor, minimum_subanchor_pixels=2)
    second = bisect_connected_interface_anchor(anchor, minimum_subanchor_pixels=2)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert np.array_equal(first[0] | first[1], anchor)
    assert not np.any(first[0] & first[1])
    assert all(np.count_nonzero(part) >= 2 for part in first)


def test_two_subanchors_retain_one_unedited_witness_without_changing_rule():
    anchor = np.zeros((5, 12), dtype=bool)
    anchor[2, 1:11] = True
    parts = bisect_connected_interface_anchor(anchor, minimum_subanchor_pixels=2)
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
        bisect_connected_interface_anchor(too_small, minimum_subanchor_pixels=2)
    with pytest.raises(ValueError, match="must be positive"):
        bisect_connected_interface_anchor(too_small, minimum_subanchor_pixels=0)

    disconnected = np.zeros((5, 8), dtype=bool)
    disconnected[1, 1:4] = True
    disconnected[3, 4:7] = True
    with pytest.raises(ValueError, match="8-connected"):
        bisect_connected_interface_anchor(disconnected, minimum_subanchor_pixels=2)


def test_scene_rebuild_gives_subanchors_new_identities_without_mutating_source():
    tissue = np.full((32, 32), 2, dtype=np.uint8)
    tissue[10:22, 10:22] = 1
    source = build_scene_analysis(
        tissue,
        schema=MaskProfileSchema.from_reference_profile("BCSS"),
    )
    interface = next(
        item
        for item in source.graph.interfaces
        if item.source_label == "Stroma" and item.target_label == "Tumor"
    )
    assert len(interface.anchor_segment_ids) == 1
    old_anchor_id = interface.anchor_segment_ids[0]

    rebuilt = rebuild_scene_with_bisected_single_anchor(
        source,
        interface_id=interface.interface_id,
        minimum_subanchor_pixels=2,
    )
    rebuilt_interface = next(
        item
        for item in rebuilt.graph.interfaces
        if item.interface_id == interface.interface_id
    )

    assert source.graph.interfaces != rebuilt.graph.interfaces
    assert source.graph.interfaces[source.graph.interfaces.index(interface)].anchor_segment_ids == (
        old_anchor_id,
    )
    assert len(rebuilt_interface.anchor_segment_ids) == 2
    assert old_anchor_id not in rebuilt.anchor_masks
    assert all(anchor_id.startswith(f"{old_anchor_id}:sub:") for anchor_id in rebuilt_interface.anchor_segment_ids)
    assert np.array_equal(
        np.logical_or.reduce(
            [rebuilt.anchor_masks[item] for item in rebuilt_interface.anchor_segment_ids]
        ),
        source.anchor_masks[old_anchor_id],
    )

    eligible, reasons = subdivision_eligibility_for_underfill_replan(
        source,
        interface_id=interface.interface_id,
        cell_feasible_anchor_ids=(old_anchor_id,),
        remaining_feasible_interface_ids=(interface.interface_id,),
        prior_topology_passed=True,
        area_underfill_replan=True,
        maximum_selected_anchor_fraction=0.75,
        minimum_unselected_anchor_count=1,
        minimum_subanchor_pixels=2,
    )
    assert eligible
    assert reasons == ()


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"area_underfill_replan": False}, "not_area_underfill_replan"),
        ({"prior_topology_passed": False}, "prior_topology_not_passed"),
        ({"remaining_feasible_interface_ids": ()}, "interface_not_remaining_feasible"),
        ({"cell_feasible_anchor_ids": ()}, "parent_anchor_not_cell_feasible"),
        ({"maximum_selected_anchor_fraction": float("nan")}, "invalid_maximum_selected_anchor_fraction"),
        ({"maximum_selected_anchor_fraction": 1.1}, "invalid_maximum_selected_anchor_fraction"),
        ({"minimum_unselected_anchor_count": -1}, "invalid_minimum_unselected_anchor_count"),
        ({"minimum_subanchor_pixels": 0}, "invalid_minimum_subanchor_pixels"),
    ],
)
def test_subdivision_eligibility_fails_closed(overrides, expected_reason):
    tissue = np.full((32, 32), 2, dtype=np.uint8)
    tissue[10:22, 10:22] = 1
    scene = build_scene_analysis(
        tissue,
        schema=MaskProfileSchema.from_reference_profile("BCSS"),
    )
    interface = next(
        item
        for item in scene.graph.interfaces
        if item.source_label == "Stroma" and item.target_label == "Tumor"
    )
    anchor_id = interface.anchor_segment_ids[0]
    arguments = {
        "interface_id": interface.interface_id,
        "cell_feasible_anchor_ids": (anchor_id,),
        "remaining_feasible_interface_ids": (interface.interface_id,),
        "prior_topology_passed": True,
        "area_underfill_replan": True,
        "maximum_selected_anchor_fraction": 0.75,
        "minimum_unselected_anchor_count": 1,
        "minimum_subanchor_pixels": 2,
    }
    arguments.update(overrides or {})
    eligible, reasons = subdivision_eligibility_for_underfill_replan(
        scene,
        **arguments,
    )
    assert not eligible
    assert expected_reason in reasons


def test_rebuilt_subanchors_receive_fresh_nuclei_preflight(tmp_path):
    size = 96
    center = size // 2
    rows, cols = np.ogrid[:size, :size]
    tumor = (rows - center) ** 2 + (cols - center) ** 2 <= 12**2
    tissue = np.full((size, size), 2, dtype=np.uint8)
    tissue[tumor] = 1
    nuclei = np.zeros_like(tissue)
    instances = {}
    instance_index = 0
    for row in range(6, size - 6, 8):
        for col in range(6, size - 6, 8):
            class_id = 1 if tumor[row, col] else 3
            nuclei[row - 1 : row + 2, col - 1 : col + 2] = class_id
            instances[str(instance_index)] = {
                "type": class_id,
                "centroid": [row, col],
                "contour": [
                    [col - 1, row - 1],
                    [col + 1, row - 1],
                    [col + 1, row + 1],
                    [col - 1, row + 1],
                ],
            }
            instance_index += 1

    tissue_path = tmp_path / "tissue.npy"
    nuclei_path = tmp_path / "nuclei.png"
    image_path = tmp_path / "image.png"
    instances_path = tmp_path / "instances.json"
    np.save(tissue_path, tissue, allow_pickle=False)
    Image.fromarray(nuclei).save(nuclei_path)
    image = np.full((size, size, 3), (222, 183, 202), dtype=np.uint8)
    image[tumor] = (172, 88, 130)
    Image.fromarray(image).save(image_path)
    instances_path.write_text(json.dumps({"nuc": instances}), encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    case = JointCaseContext(
        case_id="synthetic-anchor-subdivision",
        instruction="expand the tumor boundary locally",
        source_image_uri=str(image_path),
        source_tissue_mask_uri=str(tissue_path),
        source_nuclei_mask_uri=str(nuclei_path),
        source_nuclei_instances_uri=str(instances_path),
        auxiliary_structure_uris={},
        pathology_domain_id="breast-invasive-carcinoma-v1",
        annotation_profile_id="bcss-semantic-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id="breast-cellvit-source-first-v1",
        primitive_id="cohesive-boundary-expansion-v1",
        joint_area_budget=JointAreaBudget(
            target_fraction=0.10,
            min_fraction=0.08,
            max_fraction=0.12,
            tissue_min_fraction=0.08,
        ),
        seed=17,
        provenance={
            "source_image_sha256": digest(image_path),
            "source_tissue_mask_sha256": digest(tissue_path),
            "source_nuclei_mask_sha256": digest(nuclei_path),
            "source_nuclei_instances_sha256": digest(instances_path),
            "original_label_map_digest": digest(tissue_path),
            "original_instance_mask_digest": digest(instances_path),
            "preprocessing_revision": "synthetic-bcss-v1",
            "joint_mechanism_id": "breast-annotation-anchored-boundary-growth",
            "joint_primitive_id": "cohesive-boundary-expansion-v1",
            "available_auxiliary_structures": [],
        },
        pixel_size_um=0.465,
    )
    schema = MaskProfileSchema.from_reference_profile("BCSS")
    joint_scene = build_joint_scene_analysis(
        tissue,
        nuclei,
        schema=schema,
        pixel_size_um=case.pixel_size_um,
        nuclei_instances_path=str(instances_path),
    )
    interface = next(
        item
        for item in joint_scene.tissue.graph.interfaces
        if item.source_label == "Stroma" and item.target_label == "Tumor"
    )
    assert len(interface.anchor_segment_ids) == 1
    rebuilt_tissue = rebuild_scene_with_bisected_single_anchor(
        joint_scene.tissue,
        interface_id=interface.interface_id,
        minimum_subanchor_pixels=minimum_witness_contact_pixels(9.0),
    )
    rebuilt_scene = replace(joint_scene, tissue=rebuilt_tissue)

    workflow = JointPathologyEditWorkflow(
        tissue_planner=object(),
        joint_planner=object(),
        critic=object(),
    )
    prepared, rejected = workflow._prepare_interpretations(
        case=case,
        source_tissue=tissue,
        schema=schema,
        scene=rebuilt_scene,
    )
    assert prepared, rejected
    item = next(iter(prepared.values()))
    report = item.nuclei_preflight.interface(interface.interface_id)
    assert report is not None
    new_ids = next(
        current.anchor_segment_ids
        for current in rebuilt_tissue.graph.interfaces
        if current.interface_id == interface.interface_id
    )
    assert len(new_ids) == 2
    assert set(report.cell_feasible_anchor_segment_ids).issubset(set(new_ids))
    assert report.anchor_continuity_reports
    assert {
        current["anchor_segment_id"]
        for current in report.anchor_continuity_reports
    } == set(new_ids)
    assert report.cell_feasible_anchor_segment_ids
    assert _directional_sector_selection_limit(
        interface_anchor_ids=new_ids,
        allowed_anchor_ids=report.cell_feasible_anchor_segment_ids,
        maximum_selected_anchor_fraction=0.75,
        minimum_unselected_anchor_count=1,
    ) == 1
    portfolio = item.tissue_feasibility_portfolio
    assert portfolio.survivors
    selected_anchor_ids = {
        anchor_id
        for survivor in portfolio.survivors
        for candidate_interface in survivor.compiled_plan.candidate_interfaces
        for anchor_id in candidate_interface.execution_contract.anchor_segment_ids
    }
    assert selected_anchor_ids.intersection(new_ids)
    assert not interface.anchor_segment_ids[0] in selected_anchor_ids
    assert all(
        not anchor_id.endswith(":anchor:0001")
        for anchor_id in report.cell_feasible_anchor_segment_ids
    )

    no_target_nuclei = np.zeros_like(nuclei)
    negative_joint_scene = build_joint_scene_analysis(
        tissue,
        no_target_nuclei,
        schema=schema,
        pixel_size_um=case.pixel_size_um,
    )
    negative_interface = next(
        current
        for current in negative_joint_scene.tissue.graph.interfaces
        if current.interface_id == interface.interface_id
    )
    assert len(negative_interface.anchor_segment_ids) == 1
    negative_scene = replace(
        negative_joint_scene,
        tissue=rebuild_scene_with_bisected_single_anchor(
            negative_joint_scene.tissue,
            interface_id=negative_interface.interface_id,
            minimum_subanchor_pixels=minimum_witness_contact_pixels(9.0),
        ),
    )
    negative_preflight = build_joint_nuclei_preflight(
        case=item.case,
        source_tissue=tissue,
        schema=schema,
        scene=negative_scene,
        tissue_bundle=item.tissue_bundle,
        joint_bundle=item.bundle,
        allocation=item.allocation,
    )
    negative_report = negative_preflight.interface(interface.interface_id)
    assert negative_report is not None
    assert not negative_report.feasible
    assert "no_complete_same_class_reference_shape" in negative_report.reasons
    assert interface.interface_id not in negative_preflight.feasible_interface_ids
    assert set(negative_report.cell_feasible_anchor_segment_ids).issubset(
        set(new_ids)
    )
    assert interface.anchor_segment_ids[0] not in (
        negative_report.cell_feasible_anchor_segment_ids
    )
