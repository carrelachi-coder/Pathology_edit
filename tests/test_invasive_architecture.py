from __future__ import annotations

import numpy as np
from scipy import ndimage

import phase3_joint_edit_refine.invasive_architecture as invasive_architecture
from phase3_joint_edit_refine.invasive_architecture import (
    CORD_PRIMITIVE_ID,
    NEST_PRIMITIVE_ID,
    generate_joint_tissue_candidates,
)
from phase3_joint_edit_refine.scene import build_joint_scene_analysis
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import (
    AreaBudget,
    DepthProfile,
    EditPlan,
    InterfaceExecutionContract,
    PlannedInterface,
    ResolvedAreaContract,
    ToolProgram,
)


def _fixture():
    schema = MaskProfileSchema.from_reference_profile("BCSS")
    tissue = np.full((160, 160), 2, dtype=np.uint8)
    tissue[45:115, 15:65] = 1
    nuclei = np.zeros_like(tissue)
    for row, col in ((55, 35), (70, 40), (85, 35), (100, 42)):
        nuclei[row - 3 : row + 4, col - 3 : col + 4] = 1
    scene = build_joint_scene_analysis(
        tissue,
        nuclei,
        schema=schema,
        pixel_size_um=None,
    )
    interface = next(
        item
        for item in scene.tissue.graph.interfaces
        if item.source_label == "Stroma" and item.target_label == "Tumor"
    )
    return schema, tissue, scene, interface


def _plan(interface, *, primitive_id, tool_name, geometry_mode, pixels):
    anchor_id = interface.anchor_segment_ids[0]
    return EditPlan(
        schema_version="mask-edit-plan-v3",
        case_id="architecture-fixture",
        normalized_intent=primitive_id,
        primitive_id=primitive_id,
        source_labels=("Stroma",),
        target_label="Tumor",
        area_budget=AreaBudget(0.01, 0.005, 0.02),
        candidate_interfaces=(
            PlannedInterface(
                interface_id=interface.interface_id,
                source_component_id=interface.source_component_id,
                target_component_id=interface.target_component_id,
                anchor_segment=anchor_id,
                allowed_edit_band_px=(0.0, 90.0),
                execution_contract=InterfaceExecutionContract(
                    anchor_segment_ids=(anchor_id,),
                    area_allocation_fraction=1.0,
                    depth_profile=DepthProfile(
                        mode="tapered_lobe",
                        peak_depth_px=90.0,
                        edge_depth_px=2.0,
                        taper_fraction=0.25,
                        lobe_count=1,
                        noise_amplitude_px=1.0,
                        noise_correlation_px=8.0,
                    ),
                    min_anchor_coverage_fraction=0.01,
                    max_off_anchor_contact_fraction=0.03,
                    allocation_tolerance_fraction=0.02,
                ),
                prohibited_region_ids=(),
                supporting_rule_ids=(),
                expected_morphology=geometry_mode,
                confidence=0.9,
            ),
        ),
        tool_program=ToolProgram(
            allowed_tools=(tool_name,),
            parameter_ranges={
                "tissue_geometry_mode": geometry_mode,
                "editable_source_fine_ids": [2],
                "editable_target_fine_ids": [1],
            },
            candidate_count=4,
        ),
        hard_invariants=(),
        uncertainties=(),
        planner_confidence=0.9,
        resolved_area=ResolvedAreaContract(
            desired_pixels=pixels,
            hard_min_pixels=pixels,
            hard_max_pixels=pixels,
            resolved_pixels=pixels,
            fallback_policy="exact",
            used_fallback=False,
            binding_constraint="test",
            solver_version="test-v1",
        ),
    )


def test_cord_is_cells_then_connected_cell_scale_tumor_support():
    schema, tissue, scene, interface = _fixture()
    plan = _plan(
        interface,
        primitive_id=CORD_PRIMITIVE_ID,
        tool_name="cell_seeded_cord",
        geometry_mode="cell_seeded_invasive_cord",
        pixels=650,
    )
    candidates = generate_joint_tissue_candidates(
        tissue,
        schema=schema,
        tissue_scene=scene.tissue,
        joint_scene=scene,
        plan=plan,
        bundle=None,
        seed=3,
        candidate_limit=4,
    )

    assert candidates
    candidate = candidates[0]
    assert candidate.tool_name == "cell_seeded_cord"
    assert candidate.tool_trace["execution_order"] == "cells_then_tumor_mask"
    assert len(candidate.tool_trace["cell_seed_centers_yx"]) >= 5
    assert int(candidate.change_region.sum()) == 650
    assert ndimage.label(candidate.change_region, np.ones((3, 3)))[1] == 1
    source_tumor = tissue == 1
    assert np.any(ndimage.binary_dilation(candidate.change_region) & source_tumor)


def test_nest_is_one_irregular_detached_tumor_island_then_cells():
    schema, tissue, scene, interface = _fixture()
    plan = _plan(
        interface,
        primitive_id=NEST_PRIMITIVE_ID,
        tool_name="peritumoral_tumor_island",
        geometry_mode="peritumoral_detached_tumor_island",
        pixels=450,
    )
    candidates = generate_joint_tissue_candidates(
        tissue,
        schema=schema,
        tissue_scene=scene.tissue,
        joint_scene=scene,
        plan=plan,
        bundle=None,
        seed=3,
        candidate_limit=4,
    )

    assert candidates
    candidate = candidates[0]
    assert candidate.tool_name == "peritumoral_tumor_island"
    assert candidate.tool_trace["execution_order"] == "tumor_island_then_cells"
    assert int(candidate.change_region.sum()) == 450
    assert ndimage.label(candidate.change_region, np.ones((3, 3)))[1] == 1
    source_tumor = tissue == 1
    assert not np.any(ndimage.binary_dilation(candidate.change_region) & source_tumor)
    before = ndimage.label(source_tumor, np.ones((3, 3)))[1]
    after = ndimage.label(candidate.target_mask == 1, np.ones((3, 3)))[1]
    assert after == before + 1
    assert candidate.tool_trace["minimum_parent_tumor_gap_px"] >= 2.0


def test_cord_reduces_virtual_support_radius_after_budget_rebalance():
    schema, tissue, scene, interface = _fixture()
    plan = _plan(
        interface,
        primitive_id=CORD_PRIMITIVE_ID,
        tool_name="cell_seeded_cord",
        geometry_mode="cell_seeded_invasive_cord",
        pixels=300,
    )

    candidates = generate_joint_tissue_candidates(
        tissue,
        schema=schema,
        tissue_scene=scene.tissue,
        joint_scene=scene,
        plan=plan,
        bundle=None,
        seed=3,
        candidate_limit=1,
    )

    assert candidates
    candidate = candidates[0]
    default_support_radius = max(
        1,
        int(round(0.30 * candidate.tool_trace["nominal_nucleus_diameter_px"])),
    )
    default_nucleus_radius = max(
        2,
        int(round(0.42 * candidate.tool_trace["nominal_nucleus_diameter_px"])),
    )
    assert (
        candidate.tool_trace["support_closing_radius_px"] < default_support_radius
        or candidate.tool_trace["nucleus_footprint_radius_px"]
        < default_nucleus_radius
    )
    assert int(candidate.change_region.sum()) == 300
    assert all(
        candidate.change_region[row, col]
        for row, col in candidate.tool_trace["cell_seed_centers_yx"]
    )


def test_nest_searches_past_one_invalid_exact_area_raster(monkeypatch):
    schema, tissue, scene, interface = _fixture()
    plan = _plan(
        interface,
        primitive_id=NEST_PRIMITIVE_ID,
        tool_name="peritumoral_tumor_island",
        geometry_mode="peritumoral_detached_tumor_island",
        pixels=450,
    )
    original = invasive_architecture._irregular_island
    attempts = 0

    def reject_first(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return None
        return original(**kwargs)

    monkeypatch.setattr(
        invasive_architecture,
        "_irregular_island",
        reject_first,
    )
    candidates = generate_joint_tissue_candidates(
        tissue,
        schema=schema,
        tissue_scene=scene.tissue,
        joint_scene=scene,
        plan=plan,
        bundle=None,
        seed=3,
        candidate_limit=1,
    )

    assert attempts >= 2
    assert candidates
