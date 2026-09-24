"""Semantic selection must fit the CLI without changing certified evidence."""

import json

from phase3_joint_edit_refine.agents import (
    _compact_semantic_option_metadata,
    _compact_semantic_scene_metadata,
)


def test_semantic_mask_graph_summary_preserves_observed_counts():
    interfaces = [
        {"source_label": "Tumor", "target_label": "Stroma"}
        for _ in range(3000)
    ]
    scene = {
        "tissue": {
            "width": 512, "height": 512,
            "labels_present": {"Tumor": 100000, "Stroma": 162144},
            "components": [{"label": "Tumor"}, {"label": "Stroma"}],
            "interfaces": interfaces, "warnings": [],
        },
        "cells": {
            "class_counts": {"1": 20}, "observation_quality": "native_instance",
            "instances": [{}] * 20, "border_censored_instance_ids": ["n1"],
            "merged_suspect_instance_ids": [], "warnings": [],
        },
        "population": {
            "zones": [{"zone_kind": "interface_band", "density": "x" * 100}]
            * 6000,
            "median_nucleus_area_px": 30,
            "nominal_nucleus_diameter_px": 6, "warnings": [],
        },
        "nucleus_instance_authority": {"source": "native_instance"},
        "structural_hierarchy": {
            "schema_version": "v2", "levels": ["tissue", "cell"],
            "observation_policy": "mask_only", "execution_semantics": {},
            "structure_units": [], "relations": [{}] * 6000,
        },
        "reference_shape_authority": None,
    }
    assert len(json.dumps(scene)) > 1_000_000
    compact = _compact_semantic_scene_metadata(scene)
    assert len(json.dumps(compact)) < 10_000
    assert compact["tissue"]["interface_counts_by_label_pair"] == {
        "Tumor -> Stroma": 3000
    }
    assert compact["population"]["zone_counts_by_kind"] == {
        "interface_band": 6000
    }
    assert compact["cells"]["instance_count"] == 20
    assert len(compact["full_mask_graph_sha256"]) == 64


def test_semantic_option_keeps_certificate_counts_and_measured_metrics():
    option = {
        "option_id": "primitive::mechanism",
        "deterministic_candidate_metrics": {"feasible_interface_count": 2},
        "feasibility": {
            "aggregate_tissue_capacity_pixels": 10000,
            "whole_mask_topology_portfolio": {
                "schema_version": "v1", "authority_binding_sha256": "a" * 64,
                "surviving_candidates": [
                    {"hard_gate_passed": True, "realized_tissue_pixels": 100,
                     "compiler": {"trace": "x" * 100000}},
                    {"hard_gate_passed": False, "realized_tissue_pixels": 200,
                     "compiler": {"trace": "x" * 100000}},
                ],
                "vetoed_candidates": [{}],
            },
        },
    }
    compact = _compact_semantic_option_metadata(option)
    assert compact["deterministic_candidate_metrics"] == {
        "feasible_interface_count": 2
    }
    assert compact["feasibility"]["aggregate_tissue_capacity_pixels"] == 10000
    portfolio = compact["feasibility"]["whole_mask_topology_portfolio"]
    assert portfolio["surviving_candidate_count"] == 2
    assert portfolio["hard_gate_passing_candidate_count"] == 1
    assert portfolio["realized_tissue_pixels_range"] == [100, 200]
    assert len(json.dumps(compact)) < 2_000
