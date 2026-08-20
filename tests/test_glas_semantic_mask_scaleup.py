"""Regression tests for semantic-mask GLaS execution and visible budgets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phase3_joint_edit_refine.cell_layouts import build_reference_shape_library
from phase3_joint_edit_refine.models import CellCountExtentBudget, JointCaseContext
from phase3_joint_edit_refine.scene import build_joint_scene_analysis
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_joint_edit_refine.workflow import _apply_glas_visible_cell_budget
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.skills.catalog_manifest import (
    load_verified_official_catalog_manifest,
)
from scripts.run_glas_primitive_mask_review import (
    MASK_REVIEW_CELL_BUDGETS,
    build_parser,
)


def _glas_case() -> JointCaseContext:
    return JointCaseContext(
        case_id="budget-only",
        instruction="Increase local cellularity.",
        source_image_uri="image.png",
        source_tissue_mask_uri="tissue.png",
        source_nuclei_mask_uri="nuclei.png",
        pathology_domain_id="colorectal-adenocarcinoma-v1",
        annotation_profile_id="glas-gland-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id="colorectal-cellvit-source-first-v1",
        primitive_id="cellularity-increase-v1",
        joint_area_budget=None,
        cell_count_extent_budget=None,
        seed=1,
        provenance={"fixture": True},
    )


def test_official_mask_catalog_manifest_is_synchronized():
    payload, digest = load_verified_official_catalog_manifest()
    assert payload["package_count"] > 0
    assert len(digest) == 64


def test_runner_defaults_to_semantic_nuclei_mask_without_cellvit_paths():
    args = build_parser().parse_args(
        [
            "--cross-meta-eval", "cross-meta.json",
            "--output-dir", "output",
            "--probnet-checkpoint", "probnet.pt",
            "--nuclei-instance-library", "library",
        ]
    )
    assert args.nucleus_instance_source == "semantic-mask"
    assert args.cellvit_model is None
    assert args.cellvit_root is None
    assert args.cellvit_python is None


def test_visible_review_budgets_are_larger_than_commit_7c6c69f():
    local = MASK_REVIEW_CELL_BUDGETS["cellularity-increase-v1"]
    scatter = MASK_REVIEW_CELL_BUDGETS[
        "peritumoral-neoplastic-scatter-increase-v1"
    ]
    cluster = MASK_REVIEW_CELL_BUDGETS[
        "peritumoral-small-cluster-increase-v1"
    ]
    assert (local.target_delta_count, local.min_delta_count) == (20, 12)
    assert local.max_delta_count == 28
    assert local.maximum_extent_px == 384
    assert (scatter.target_delta_count, scatter.min_delta_count) == (10, 4)
    assert scatter.maximum_extent_px == 144
    assert scatter.minimum_effect_foci == 4
    assert (cluster.target_delta_count, cluster.min_delta_count) == (12, 6)
    assert cluster.maximum_extent_px == 160
    assert cluster.minimum_effect_foci == 3


def test_derived_glas_budget_is_scaled_but_explicit_non_glas_policy_is_not():
    base = CellCountExtentBudget(6, 4, 8, 64, 2, 48, 20, 2)
    scaled, metadata = _apply_glas_visible_cell_budget(
        _glas_case(),
        primitive_id="peritumoral-neoplastic-scatter-increase-v1",
        budget=base,
        metadata={"policy_id": "legacy"},
    )
    assert (scaled.target_delta_count, scaled.min_delta_count) == (10, 4)
    assert scaled.maximum_extent_px >= 144
    assert metadata["policy_id"] == "glas-visible-cell-effect-budget-v3-feasible"


def test_gland_mechanisms_allow_semantic_nucleus_instance_fallback_and_64px_halo():
    repository = JointSkillRepository()
    budding = repository.mechanisms["colorectal-tumor-budding-front"]
    local = repository.mechanisms["colorectal-local-population-modulation"]
    assert budding.representability.allow_semantic_instance_fallback is True
    assert budding.cell_program.halo_distance_px == (4, 64)
    assert local.representability.allow_semantic_instance_fallback is True
    assert local.cell_program.halo_distance_px == (0, 64)


def test_semantic_nuclei_mask_supplies_reusable_class1_shapes_without_instance_json():
    tissue = np.full((96, 96), 2, dtype=np.uint8)
    tissue[20:76, 20:76] = 12
    nuclei = np.zeros_like(tissue)
    for row, col in ((28, 28), (28, 45), (45, 28), (45, 45), (62, 62)):
        nuclei[row - 2 : row + 3, col - 2 : col + 3] = 1
    scene = build_joint_scene_analysis(
        tissue,
        nuclei,
        schema=MaskProfileSchema.from_reference_profile("GLaS"),
        pixel_size_um=0.25,
        nuclei_instances_path=None,
    )
    assert scene.cells.observation_quality == "semantic_distance_watershed"
    references, rejected = build_reference_shape_library(scene, class_id=1)
    assert references, rejected
    assert all(item.source == "semantic_distance_watershed" for item in references)
