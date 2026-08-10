from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

import inpaint_cells.generate as generate_module
from inpaint_cells.generate import (
    COMPONENT_SHAPE_POLICY_NAME,
    DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT,
    DEFAULT_PROBNET_ODDS_GAMMA,
    SAMPLING_AUDIT_POLICY_NAME,
    SAMPLING_FEEDBACK_POLICY_NAME,
    PlacementQuotaError,
    _merge_required_center_stage_diagnostics,
    _require_complete_target_count,
    allocate_type_counts,
    allocate_weight_proportional_counts,
    balanced_type_at_center,
    balanced_type_order_at_center,
    blend_context_stabilized_probability,
    build_buffer_retained_by_type_overrides,
    calibrated_local_type_distribution,
    choose_weighted_centers,
    compute_patch_adaptive_priors,
    confidence_adaptive_type_prior_weights,
    constrain_type_quota_to_shape_support,
    count_retained_centers_by_type,
    exact_backfill_candidate_budget,
    fuse_density_head_with_tissue_prior,
    generate_two_stage_for_gamma,
    initialize_component_sampling_diagnostics,
    make_accepted_centers_overlay,
    next_sampling_feedback_parameters,
    place_candidate_with_retries,
    place_candidate_with_type_fallback,
    predict_context_stabilized_spatial_probability,
    probability_concentration_diagnostics,
    probability_mass_region_centers,
    probnet_sampling_alignment_audit,
    realize_compiled_packing_witness,
    same_tissue_quota_reassignment_centers,
    sample_type_at_center,
    select_low_variance_type,
    shape_sampling_diagnostics,
    spatial_context_halo_radius,
    supported_joint_nucleus_probability,
    supported_nucleus_shape_types,
)
from inpaint_cells.nuclei_library.library import ReferenceNucleiInstancePool
from inpaint_cells.sampling_policy import (
    retry_pool_target,
    valid_biological_tissue_mask,
)
from phase3_joint_edit_refine.mature_probnet_adapter import (
    _mechanism_modifier_certificate,
)


def test_compiled_packing_witness_is_probnet_ranked_and_complete():
    shape = (24, 24)
    base = np.zeros(shape, dtype=np.uint8)
    centers = np.ones(shape, dtype=bool)
    valid = np.ones(shape, dtype=bool)
    probability = np.zeros(shape, dtype=np.float32)
    probability[5, 5] = 0.3
    probability[12, 12] = 0.9
    probability[18, 18] = 0.6
    witness = {
        "version": "compiled-packing-witness-v4",
        "contract_id": "fixture",
        "requested_count": 3,
        "placements": [
            {
                "row": row,
                "col": col,
                "nucleus_type": 101,
                "reference_instance_id": f"ref-{index}",
                "offsets_yx": [[0, 0]],
            }
            for index, (row, col) in enumerate(
                ((5, 5), (12, 12), (18, 18))
            )
        ],
    }

    realized = realize_compiled_packing_witness(
        base_output=base,
        packing_witness=witness,
        center_region=centers,
        valid_tissue_mask=valid,
        tissue_id=2,
        target_count=2,
        nucleus_probability=probability,
        minimum_separation_px=1,
    )

    assert realized is not None
    output, ledger = realized
    assert int(np.count_nonzero(output)) == 2
    assert [(item["row"], item["col"]) for item in ledger] == [
        (12, 12),
        (18, 18),
    ]
    assert all(item["shape_source"] == "compiled_reference_witness" for item in ledger)


def test_compiled_packing_witness_rejects_diagonal_instance_contact():
    shape = (12, 12)
    witness = {
        "version": "compiled-packing-witness-v4",
        "contract_id": "diagonal-fixture",
        "requested_count": 2,
        "placements": [
            {
                "row": row,
                "col": col,
                "nucleus_type": 101,
                "reference_instance_id": f"ref-{index}",
                "offsets_yx": [[0, 0]],
            }
            for index, (row, col) in enumerate(((5, 5), (6, 6)))
        ],
    }

    realized = realize_compiled_packing_witness(
        base_output=np.zeros(shape, dtype=np.uint8),
        packing_witness=witness,
        center_region=np.ones(shape, dtype=bool),
        valid_tissue_mask=np.ones(shape, dtype=bool),
        tissue_id=2,
        target_count=2,
        nucleus_probability=np.ones(shape, dtype=np.float32),
        minimum_separation_px=1,
    )

    assert realized is None


def test_reference_pool_can_be_scoped_by_original_instance_center():
    nuclei = np.zeros((16, 16), dtype=np.uint16)
    nuclei[2:5, 2:5] = 101
    nuclei[10:13, 10:13] = 102
    pool = ReferenceNucleiInstancePool.from_mask(
        nuclei,
        min_area=1,
        exclude_border=False,
    )
    first_region = np.zeros_like(nuclei, dtype=bool)
    first_region[:8, :8] = True

    scoped = pool.subset_by_center_region(first_region)

    assert scoped.counts()[101] == 1
    assert scoped.counts()[102] == 0
    assert scoped.instances[101][0]["center_y"] == 3.0
    assert scoped.instances[101][0]["center_x"] == 3.0


def test_default_probnet_odds_gamma_preserves_sharp_spatial_structure():
    assert DEFAULT_PROBNET_ODDS_GAMMA == 3.0
    assert DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT == 2.0 / 3.0
    assert SAMPLING_AUDIT_POLICY_NAME.endswith("spatial_v3")
    assert SAMPLING_FEEDBACK_POLICY_NAME == "reason_directed_gamma_then_seed_v1"


def test_probability_concentration_diagnostic_has_signed_direction():
    probability = np.linspace(0.05, 0.95, 400, dtype=np.float64)
    mass = probability**3

    over = probability_concentration_diagnostics(
        probability,
        mass,
        probability[-12:],
    )
    under = probability_concentration_diagnostics(
        probability,
        mass,
        probability[:12],
    )

    assert over["direction"] == "overconcentrated"
    assert over["z_score"] > over["z_threshold"]
    assert under["direction"] == "underfollow"
    assert under["z_score"] < -under["z_threshold"]


def test_sampling_feedback_adjusts_gamma_once_then_changes_only_seed():
    retry = next_sampling_feedback_parameters(
        initial_gamma=3.0,
        current_gamma=3.0,
        base_seed=42,
        attempt_index=1,
        previous_failure_reasons=["PROBNET_OVERCONCENTRATED"],
        gamma_already_adjusted=False,
    )
    final = next_sampling_feedback_parameters(
        initial_gamma=3.0,
        current_gamma=retry["sampling_gamma"],
        base_seed=42,
        attempt_index=2,
        previous_failure_reasons=["PROBNET_OVERCONCENTRATED"],
        gamma_already_adjusted=True,
    )

    assert retry["action"] == "decrease_gamma"
    assert retry["sampling_gamma"] == 2.25
    assert retry["seed"] == 43
    assert final["action"] == "resample_seed"
    assert final["sampling_gamma"] == 2.25
    assert final["seed"] == 44


def test_sampling_feedback_keeps_gamma_when_tissues_disagree_on_direction():
    retry = next_sampling_feedback_parameters(
        initial_gamma=3.0,
        current_gamma=3.0,
        base_seed=42,
        attempt_index=1,
        previous_failure_reasons=[
            "PROBNET_OVERCONCENTRATED",
            "PROBNET_UNDERFOLLOW",
        ],
        gamma_already_adjusted=False,
    )

    assert retry["action"] == "resample_seed"
    assert retry["sampling_gamma"] == 3.0
    assert retry["gamma_adjusted"] is False


def _sampling_audit_fixture(centers, *, target_count=9, probability=None):
    shape = (40, 40)
    tissue = np.ones(shape, dtype=np.uint8)
    region = np.ones(shape, dtype=bool)
    input_nuclei = np.zeros(shape, dtype=np.uint8)
    output_nuclei = np.zeros(shape, dtype=np.uint8)
    for row, col in centers:
        output_nuclei[int(row), int(col)] = 1
    type_prob = np.zeros((5, *shape), dtype=np.float32)
    type_prob[0] = (
        np.asarray(probability, dtype=np.float32)
        if probability is not None
        else 0.60
    )
    diagnostics = {
        "tissues": {
            "1": {
                "target_count": target_count,
                "placed_by_type": {"101": len(centers)},
                "posterior_expected_by_type": {"101": float(target_count)},
                "type_shape_support": {"supported_types": [101]},
            }
        }
    }
    return probnet_sampling_alignment_audit(
        input_nuclei=input_nuclei,
        output_nuclei=output_nuclei,
        tissue=tissue,
        generation_region=region,
        placement_type_prob=type_prob,
        gamma=3.0,
        generation_diagnostics=diagnostics,
    )


def test_probnet_sampling_audit_accepts_distributed_centers_for_flat_prior():
    centers = [(8, 8), (8, 20), (8, 32), (20, 8), (20, 20), (20, 32), (32, 8), (32, 20), (32, 32)]

    audit = _sampling_audit_fixture(centers)

    assert audit["passed"] is True
    assert audit["organ_specific_constraints"] is False
    assert audit["tissues"]["1"]["probability_bin_tv"] is None
    assert audit["tissues"]["1"]["probability_mass_coverage_ratio"] < 1.10


def test_probnet_sampling_audit_rejects_boundary_collapse_for_flat_prior():
    centers = [(3 + 4 * index, 1) for index in range(9)]

    audit = _sampling_audit_fixture(centers)

    assert audit["passed"] is False
    assert audit["tissues"]["1"]["spatial_passed"] is False
    assert audit["tissues"]["1"]["probability_mass_coverage_ratio"] > 1.10
    assert "PROBNET_COVERAGE_GAP" in audit["failure_reasons"]
    assert audit["primary_failure_reason"] == "PROBNET_COVERAGE_GAP"


def test_sampling_audit_conditions_exact_seam_on_required_cell_class():
    shape = (40, 40)
    tissue = np.ones(shape, dtype=np.uint8)
    region = np.ones(shape, dtype=bool)
    required = np.zeros(shape, dtype=bool)
    required[:, :20] = True
    required_centers = [(10, 5), (10, 15), (30, 5), (30, 15)]
    remainder_centers = [(10, 25), (10, 35), (30, 25), (30, 35)]
    input_nuclei = np.zeros(shape, dtype=np.uint8)
    output_nuclei = np.zeros(shape, dtype=np.uint8)
    for row, col in required_centers + remainder_centers:
        output_nuclei[row, col] = 1
    type_prob = np.zeros((5, *shape), dtype=np.float32)
    type_prob[0] = 0.20
    type_prob[1] = 0.01
    type_prob[1, :10, :10] = 0.95
    accepted = [
        {
            "row": row,
            "col": col,
            "nucleus_type": 101,
            "tissue_id": 1,
        }
        for row, col in required_centers
    ] + [
        {
            "row": row,
            "col": col,
            "nucleus_type": 102,
            "tissue_id": 1,
        }
        for row, col in remainder_centers
    ]
    diagnostics = {
        "tissues": {
            "1": {
                "target_count": 8,
                "placed": 8,
                "placed_by_type": {"101": 4, "102": 4},
                "posterior_expected_by_type": {"101": 4.0, "102": 4.0},
                "type_shape_support": {"supported_types": [101, 102]},
                "accepted_centers": accepted,
            }
        }
    }

    audit = probnet_sampling_alignment_audit(
        input_nuclei=input_nuclei,
        output_nuclei=output_nuclei,
        tissue=tissue,
        generation_region=region,
        placement_type_prob=type_prob,
        gamma=3.0,
        generation_diagnostics=diagnostics,
        exact_required_region=required,
        exact_required_count=4,
        exact_required_nucleus_type=101,
    )

    assert audit["passed"] is True
    assert (
        audit["tissues"]["1"]["spatial_conditioning_policy"]
        == "compiled_typed_seam_and_exterior_observed_strata_v3"
    )


def test_sampling_audit_conditions_minimum_seam_on_realized_count():
    shape = (40, 40)
    tissue = np.ones(shape, dtype=np.uint8)
    region = np.ones(shape, dtype=bool)
    required = np.zeros(shape, dtype=bool)
    required[:, :20] = True
    required_centers = [(7, 5), (7, 15), (20, 10), (33, 5), (33, 15)]
    remainder_centers = [(8, 30), (20, 30), (32, 30)]
    input_nuclei = np.zeros(shape, dtype=np.uint8)
    output_nuclei = np.zeros(shape, dtype=np.uint8)
    for row, col in required_centers + remainder_centers:
        output_nuclei[row, col] = 1
    type_prob = np.zeros((5, *shape), dtype=np.float32)
    type_prob[0] = 0.60
    diagnostics = {
        "tissues": {
            "1": {
                "target_count": 8,
                "placed": 8,
                "placed_by_type": {"101": 8},
                "posterior_expected_by_type": {"101": 8.0},
                "type_shape_support": {"supported_types": [101]},
                "accepted_centers": [
                    {
                        "row": row,
                        "col": col,
                        "nucleus_type": 101,
                        "tissue_id": 1,
                    }
                    for row, col in required_centers + remainder_centers
                ],
            }
        }
    }

    audit = probnet_sampling_alignment_audit(
        input_nuclei=input_nuclei,
        output_nuclei=output_nuclei,
        tissue=tissue,
        generation_region=region,
        placement_type_prob=type_prob,
        gamma=3.0,
        generation_diagnostics=diagnostics,
        exact_required_region=required,
        # The executable contract only requires at least four, while five
        # centers were realized in the seam.
        exact_required_count=4,
        exact_required_nucleus_type=101,
    )

    assert audit["passed"] is True
    assert (
        audit["tissues"]["1"]["spatial_conditioning_policy"]
        == "compiled_typed_seam_and_exterior_observed_strata_v3"
    )
    assert [
        item["target_count"]
        for item in audit["tissues"]["1"]["spatial_strata"]
    ] == [5, 3]


def test_sampling_audit_uses_accepted_center_ledger_for_tissue_attribution():
    shape = (40, 40)
    tissue = np.ones(shape, dtype=np.uint8)
    tissue[:, 20:] = 2
    region = np.ones(shape, dtype=bool)
    input_nuclei = np.zeros(shape, dtype=np.uint8)
    output_nuclei = np.zeros(shape, dtype=np.uint8)
    output_nuclei[18:23, 17:22] = 1
    type_prob = np.zeros((5, *shape), dtype=np.float32)
    type_prob[0] = 0.60
    diagnostics = {
        "tissues": {
            "2": {
                "target_count": 1,
                "placed": 1,
                "placed_by_type": {"101": 1},
                "posterior_expected_by_type": {"101": 1.0},
                "type_shape_support": {"supported_types": [101]},
                "accepted_centers": [
                    {
                        "row": 20,
                        "col": 21,
                        "nucleus_type": 101,
                        "tissue_id": 2,
                    }
                ],
            }
        }
    }

    audit = probnet_sampling_alignment_audit(
        input_nuclei=input_nuclei,
        output_nuclei=output_nuclei,
        tissue=tissue,
        generation_region=region,
        placement_type_prob=type_prob,
        gamma=3.0,
        generation_diagnostics=diagnostics,
    )

    assert audit["passed"] is True
    assert audit["global_instance_count_passed"] is True
    assert audit["tissues"]["2"]["observed_new_instance_count"] == 1
    assert audit["tissues"]["2"]["mask_component_attributed_count"] == 0
    assert (
        audit["tissues"]["2"]["center_record_policy"]
        == "accepted_placement_center_ledger"
    )


def test_sampling_audit_counts_edge_instance_when_shape_centroid_is_outside_region():
    shape = (24, 24)
    tissue = np.full(shape, 2, dtype=np.uint8)
    region = np.zeros(shape, dtype=bool)
    region[8:16, 8:16] = True
    input_nuclei = np.zeros(shape, dtype=np.uint8)
    output_nuclei = np.zeros(shape, dtype=np.uint8)
    # The placement center (10, 8) is legal, and one instance pixel overlaps
    # the generation region. Its long asymmetric tail puts the component's
    # geometric centroid outside the region.
    output_nuclei[10, 7:9] = 1
    output_nuclei[11:14, 4:8] = 1
    type_prob = np.zeros((5, *shape), dtype=np.float32)
    type_prob[0] = 0.60
    diagnostics = {
        "tissues": {
            "2": {
                "target_count": 1,
                "placed": 1,
                "placed_by_type": {"101": 1},
                "posterior_expected_by_type": {"101": 1.0},
                "type_shape_support": {"supported_types": [101]},
                "accepted_centers": [
                    {
                        "row": 10,
                        "col": 8,
                        "nucleus_type": 101,
                        "tissue_id": 2,
                    }
                ],
            }
        }
    }

    audit = probnet_sampling_alignment_audit(
        input_nuclei=input_nuclei,
        output_nuclei=output_nuclei,
        tissue=tissue,
        generation_region=region,
        placement_type_prob=type_prob,
        gamma=3.0,
        generation_diagnostics=diagnostics,
    )

    assert audit["passed"] is True
    assert audit["new_instance_count"] == 1
    assert audit["global_instance_count_passed"] is True
    assert audit["tissues"]["2"]["mask_component_attributed_count"] == 1


def test_probnet_sampling_audit_accepts_edge_centers_for_sharp_edge_prior():
    region = np.ones((40, 40), dtype=bool)
    boundary_distance = generate_module.ndimage.distance_transform_edt(
        np.pad(region, 1)
    )[1:-1, 1:-1]
    probability = 0.05 + 0.90 * np.exp(-(boundary_distance - 1.0) / 2.0)
    centers = [
        (1, 5),
        (1, 15),
        (1, 25),
        (1, 35),
        (8, 1),
        (18, 1),
        (28, 1),
        (38, 1),
        (38, 25),
    ]

    audit = _sampling_audit_fixture(centers, probability=probability)

    assert audit["passed"] is True
    assert audit["policy"] == SAMPLING_AUDIT_POLICY_NAME
    assert audit["failure_reasons"] == []
    assert audit["tissues"]["1"]["boundary_quantile_error"] < 0.25
    assert audit["tissues"]["1"]["probability_bin_tv_role"] == (
        "diagnostic_score_term_not_hard_gate_under_constrained_without_"
        "replacement_sampling"
    )


def _retry_args():
    return SimpleNamespace(
        retry_candidate_multiplier=12.0,
        retry_candidate_floor=64,
        dense_retry_quota_threshold=20,
        dense_retry_occupancy_threshold=0.12,
        dense_retry_candidate_multiplier=24.0,
        dense_retry_candidate_floor=128,
    )


def test_exact_target_count_rejects_silent_generation_shortfall():
    try:
        _require_complete_target_count(
            tissue_id=2,
            target_count=14,
            placed=13,
        )
    except PlacementQuotaError as exc:
        assert "target=14" in str(exc)
        assert "placed=13" in str(exc)
        assert "shortfall=1" in str(exc)
    else:
        raise AssertionError("Expected a target-count shortfall to fail")


def test_exact_target_count_accepts_complete_quota():
    assert (
        _require_complete_target_count(
            tissue_id=2,
            target_count=14,
            placed=14,
        )
        == 0
    )


def test_buffer_retained_ledger_does_not_merge_adjacent_core_cells():
    tissue = np.ones((12, 12), dtype=np.uint8)
    generation_mask = np.ones((12, 12), dtype=bool)
    input_nuclei = np.zeros((12, 12), dtype=np.uint8)
    input_nuclei[1:3, 1:3] = 1
    core_tissues = {
        "1": {
            "placed": 2,
            "placed_by_type": {"101": 2},
        }
    }

    overrides = build_buffer_retained_by_type_overrides(
        tissue,
        input_nuclei,
        generation_mask,
        core_tissues,
    )

    assert overrides == {"1": {101: 3}}


def test_density_head_and_tissue_prior_are_normalized_then_equally_weighted():
    fused, audit = fuse_density_head_with_tissue_prior(
        [6.0222, 1.5911, 3.9192, 0.0153, 0.0463],
        {
            101: 0.2019,
            102: 0.2485,
            103: 0.5289,
            104: 0.0035,
            105: 0.0171,
        },
        density_weight=0.5,
    )

    quota = allocate_type_counts(fused, 14)

    assert quota == {101: 5, 102: 3, 103: 6}
    assert audit["density_head_weight"] == 0.5
    assert abs(sum(fused.values()) - 1.0) < 1e-9


def test_density_head_can_fully_determine_type_quota():
    fused, audit = fuse_density_head_with_tissue_prior(
        [6.0222, 1.5911, 3.9192, 0.0153, 0.0463],
        {
            101: 0.2019,
            102: 0.2485,
            103: 0.5289,
            104: 0.0035,
            105: 0.0171,
        },
        density_weight=1.0,
    )

    assert allocate_type_counts(fused, 14) == {101: 7, 102: 2, 103: 5}
    assert audit["density_head_weight"] == 1.0


def test_conflicting_uncertain_density_head_defers_to_target_tissue_prior():
    fused, audit = fuse_density_head_with_tissue_prior(
        [0.50, 0.27, 0.20, 0.003, 0.027],
        {
            101: 0.123,
            102: 0.638,
            103: 0.217,
            104: 0.0002,
            105: 0.0218,
        },
        density_weight=1.0,
        adaptive=True,
    )

    quota = allocate_type_counts(fused, 13)

    assert audit["adaptive_weighting"] is True
    assert 0.0 < audit["density_head_weight"] < 0.35
    assert quota[102] >= 6
    assert quota[101] <= 3
    assert sum(quota.values()) == 13


def test_confident_agreeing_density_head_keeps_high_effective_weight():
    fused, audit = fuse_density_head_with_tissue_prior(
        [0.96, 0.01, 0.01, 0.01, 0.01],
        {101: 0.95, 102: 0.01, 103: 0.01, 104: 0.02, 105: 0.01},
        density_weight=1.0,
        adaptive=True,
    )

    assert audit["density_head_weight"] > 0.8
    assert allocate_type_counts(fused, 20)[101] >= 18


def test_unsupported_shape_quota_is_redistributed_without_changing_count():
    feasible, audit = constrain_type_quota_to_shape_support(
        {101: 7, 102: 23, 103: 19, 105: 1},
        {101: 0.18, 102: 0.29, 103: 0.53, 105: 0.001},
        {101, 102, 103},
    )

    assert feasible == {101: 7, 102: 23, 103: 20}
    assert sum(feasible.values()) == 50
    assert audit["unsupported_requested_by_type"] == {"105": 1}
    assert audit["redistributed_count"] == 1


def test_forced_tissue_library_support_uses_stored_exact_tissue_shapes():
    class FakeLibrary:
        instances: ClassVar[dict] = {
            2: [
                {"type": 101},
                {"type": 102},
            ],
            7: [{"type": 105}],
        }

    class FakeReferencePool:
        def counts(self):
            return {101: 0, 103: 4}

    supported = supported_nucleus_shape_types(
        FakeLibrary(),
        FakeReferencePool(),
        2,
        force_tissue_library=True,
    )

    assert supported == {101, 102}


def test_generation_count_subtracts_retained_buffer_nuclei_by_centroid():
    nuclei = np.zeros((20, 20), dtype=np.uint8)
    nuclei[3:6, 3:6] = 1
    nuclei[10:13, 10:13] = 3
    region = np.zeros((20, 20), dtype=bool)
    region[1:9, 1:9] = True

    counts = count_retained_centers_by_type(nuclei, region)

    assert counts[101] == 1
    assert counts[103] == 0


def test_count_density_uses_full_pre_edit_source_tissue_not_deletion_mask():
    class FakeLibrary:
        def get_density(self, tissue_id):
            return 2.0

        def get_type_distribution(self, tissue_id):
            return {101: 1.0}

    source_tissue = np.full((100, 200), 2, dtype=np.uint8)
    target_tissue = np.full((100, 200), 2, dtype=np.uint8)
    source_nuclei = np.zeros_like(source_tissue)
    for index in range(20):
        row = 2 + (index // 10) * 40
        col = 2 + (index % 10) * 18
        source_nuclei[row : row + 3, col : col + 3] = 101
    deletion = np.ones_like(source_tissue, dtype=bool)
    generation = np.ones_like(source_tissue, dtype=bool)

    scales, _, audit = compute_patch_adaptive_priors(
        reference_nuclei_raw=source_nuclei,
        reference_tissue=source_tissue,
        density_exclusion_region=deletion,
        target_tissue=target_tissue,
        generation_region=generation,
        library=FakeLibrary(),
        dataset_name="BCSS",
    )

    tissue_audit = audit["tissues"]["2"]
    assert tissue_audit["density_reference_image"] == "pre_edit_source_patch"
    assert tissue_audit["density_reference_tissue_ids"] == [2]
    assert tissue_audit["density_reference_deletion_exclusion_applied"] is False
    assert tissue_audit["local_centroid_count"] == 20
    assert tissue_audit["reference_area_px"] == 20000
    assert tissue_audit["target_density_per_10k_px"] == 10.0
    assert scales[2] == 5.0


def test_absent_glas_grade_uses_target_prior_with_pre_edit_gland_calibration():
    class FakeLibrary:
        def get_density(self, tissue_id):
            return {11: 2.0, 12: 4.0, 13: 5.0}.get(int(tissue_id), 2.0)

        def get_type_distribution(self, tissue_id):
            return {101: 1.0}

    source_tissue = np.full((100, 200), 12, dtype=np.uint8)
    target_tissue = np.full((100, 200), 13, dtype=np.uint8)
    source_nuclei = np.zeros_like(source_tissue)
    for index in range(20):
        row = 2 + (index // 10) * 40
        col = 2 + (index % 10) * 18
        source_nuclei[row : row + 3, col : col + 3] = 101
    deletion = np.ones_like(source_tissue, dtype=bool)
    generation = np.ones_like(source_tissue, dtype=bool)

    scales, _, audit = compute_patch_adaptive_priors(
        reference_nuclei_raw=source_nuclei,
        reference_tissue=source_tissue,
        density_exclusion_region=deletion,
        target_tissue=target_tissue,
        generation_region=generation,
        library=FakeLibrary(),
        dataset_name="GlaS",
    )

    tissue_audit = audit["tissues"]["13"]
    calibration = tissue_audit[
        "dataset_prior_calibration_from_pre_edit_source"
    ]
    assert tissue_audit["reference_area_px"] == 0
    assert tissue_audit["density_mode"] == (
        "target_dataset_prior_times_pre_edit_family_calibration"
    )
    assert calibration["source_tissue_ids"] == [5, 11, 12, 13]
    assert calibration["scale"] == 2.5
    assert tissue_audit["target_density_per_10k_px"] == 12.5
    assert scales[13] == 2.5


def test_center_type_assignment_samples_local_probnet_posterior():
    probability = np.zeros((5, 1, 1), dtype=np.float64)
    probability[:, 0, 0] = [0.8, 0.0, 0.2, 0.0, 0.0]
    args = SimpleNamespace(type_prob_floor=0.03)

    np.random.seed(19)
    sampled = [
        sample_type_at_center(probability, 0, 0, args)
        for _ in range(1000)
    ]

    assert set(sampled) == {101, 103}
    assert sampled.count(101) > 700
    assert sampled.count(103) > 100


def test_center_type_assignment_only_renormalizes_for_shape_support():
    probability = np.zeros((5, 1, 1), dtype=np.float64)
    probability[:, 0, 0] = [0.8, 0.1, 0.1, 0.0, 0.0]
    args = SimpleNamespace(type_prob_floor=0.03)

    np.random.seed(23)
    sampled = {
        sample_type_at_center(
            probability,
            0,
            0,
            args,
            supported_types={102, 103},
        )
        for _ in range(100)
    }

    assert sampled == {102, 103}


def test_local_type_distribution_log_pools_probnet_with_tissue_prior():
    distribution = calibrated_local_type_distribution(
        [0.507, 0.306, 0.181, 0.0003, 0.0057],
        tissue_type_prior={
            101: 0.1229,
            102: 0.6383,
            103: 0.2167,
            104: 0.0002,
            105: 0.0219,
        },
        prior_weight=0.5,
    )

    assert int(np.argmax(distribution)) == 1
    assert distribution[1] > 0.49
    assert distribution[0] < 0.28


def test_local_type_prior_calibration_respects_shape_support():
    distribution = calibrated_local_type_distribution(
        [0.6, 0.3, 0.1, 0.0, 0.0],
        supported_types={102, 103},
        tissue_type_prior={101: 0.9, 102: 0.08, 103: 0.02},
        prior_weight=0.5,
    )

    assert distribution[0] == 0.0
    assert distribution[1] > distribution[2]
    np.testing.assert_allclose(distribution.sum(), 1.0)


def test_cumulative_posterior_balancing_avoids_small_sample_type_flip():
    distribution = calibrated_local_type_distribution(
        [0.507, 0.306, 0.181, 0.0003, 0.0057],
        tissue_type_prior={
            101: 0.1229,
            102: 0.6383,
            103: 0.2167,
            104: 0.0002,
            105: 0.0219,
        },
        prior_weight=DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT,
    )
    placed = {nucleus_type: 0 for nucleus_type in generate_module.NUCLEI_CLASSES}
    expected = np.zeros(len(generate_module.NUCLEI_CLASSES), dtype=np.float64)

    for _ in range(13):
        nucleus_type = select_low_variance_type(
            distribution,
            placed_by_type=placed,
            expected_type_mass=expected,
        )
        expected += distribution
        placed[nucleus_type] += 1

    assert placed[102] > placed[101]
    for index, nucleus_type in enumerate(generate_module.NUCLEI_CLASSES):
        assert abs(placed[nucleus_type] - expected[index]) <= 1.0


def test_type_prior_weight_fades_out_with_source_support_confidence():
    weights = confidence_adaptive_type_prior_weights(
        {
            "tissues": {
                "1": {"effective_local_confidence": 1.0},
                "2": {"effective_local_confidence": 0.25},
                "4": {"effective_local_confidence": 0.0},
            }
        },
        maximum_weight=2.0 / 3.0,
    )

    assert weights[1] == 0.0
    assert weights[2] == 0.5
    assert weights[4] == 2.0 / 3.0


def test_balanced_center_type_only_commits_supported_posterior():
    probability = np.zeros((5, 1, 1), dtype=np.float64)
    probability[:, 0, 0] = [0.7, 0.2, 0.1, 0.0, 0.0]
    args = SimpleNamespace(
        local_type_prior_weight=DEFAULT_LOCAL_TYPE_PRIOR_WEIGHT,
        local_type_prior_floor=1e-4,
    )
    placed = {nucleus_type: 0 for nucleus_type in generate_module.NUCLEI_CLASSES}
    expected = np.zeros(len(generate_module.NUCLEI_CLASSES), dtype=np.float64)

    nucleus_type, posterior = balanced_type_at_center(
        probability,
        0,
        0,
        args,
        placed_by_type=placed,
        expected_type_mass=expected,
        supported_types={102, 103},
        tissue_type_prior={101: 0.9, 102: 0.08, 103: 0.02},
    )

    assert nucleus_type in {102, 103}
    assert posterior[0] == 0.0
    np.testing.assert_allclose(posterior.sum(), 1.0)


def test_balanced_type_order_preserves_primary_choice_and_legal_fallbacks():
    probability = np.zeros((5, 1, 1), dtype=np.float64)
    probability[:, 0, 0] = [0.8, 0.15, 0.05, 0.0, 0.0]
    args = SimpleNamespace(
        local_type_prior_weight=0.0,
        local_type_prior_floor=1e-4,
    )
    placed = {nucleus_type: 0 for nucleus_type in generate_module.NUCLEI_CLASSES}
    expected = np.zeros(len(generate_module.NUCLEI_CLASSES), dtype=np.float64)

    primary, posterior = balanced_type_at_center(
        probability,
        0,
        0,
        args,
        placed_by_type=placed,
        expected_type_mass=expected,
        supported_types={101, 102, 103},
    )
    order, ordered_posterior = balanced_type_order_at_center(
        probability,
        0,
        0,
        args,
        placed_by_type=placed,
        expected_type_mass=expected,
        supported_types={101, 102, 103},
    )

    assert order[0] == primary
    assert set(order) == {101, 102, 103}
    np.testing.assert_allclose(ordered_posterior, posterior)


def test_exact_backfill_type_fallback_keeps_geometry_strict(monkeypatch):
    calls = []

    def fake_place_candidate_with_retries(*, nucleus_type, **kwargs):
        del kwargs
        calls.append(nucleus_type)
        if nucleus_type == 102:
            return False, None, 7, None
        return True, "reference", 3, (4, 5)

    monkeypatch.setattr(
        generate_module,
        "place_candidate_with_retries",
        fake_place_candidate_with_retries,
    )
    audit = {}
    placed, accepted_type, source, trials, center = (
        place_candidate_with_type_fallback(
            nucleus_types=(102, 103),
            placement_audit=audit,
            output=np.zeros((8, 8), dtype=np.uint8),
        )
    )

    assert placed is True
    assert accepted_type == 103
    assert source == "reference"
    assert trials == 10
    assert center == (4, 5)
    assert calls == [102, 103]
    assert audit["type_fallback_rank"] == 1
    assert audit["preferred_nucleus_type"] == 102
    assert audit["accepted_nucleus_type"] == 103


def test_position_mass_only_sums_realizable_probnet_type_channels():
    probability = np.zeros((5, 1, 2), dtype=np.float64)
    probability[0, 0] = [0.8, 0.1]
    probability[1, 0] = [0.1, 0.7]
    probability[2, 0] = [0.1, 0.2]

    neoplastic_only = supported_joint_nucleus_probability(
        probability,
        {101},
    )
    stromal_mix = supported_joint_nucleus_probability(
        probability,
        {102, 103},
    )

    np.testing.assert_allclose(neoplastic_only, [[0.8, 0.1]])
    np.testing.assert_allclose(stromal_mix, [[0.2, 0.9]])


def test_retry_pool_expands_for_dense_components():
    ordinary = retry_pool_target(
        quota=5,
        component_area=10000,
        expected_nucleus_area=80,
        args=_retry_args(),
    )
    dense = retry_pool_target(
        quota=25,
        component_area=10000,
        expected_nucleus_area=80,
        args=_retry_args(),
    )

    assert ordinary == (64, False, 0.04)
    assert dense == (600, True, 0.2)


def test_valid_biological_tissue_excludes_background_and_skipped_labels():
    tissue = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)

    allowed = valid_biological_tissue_mask(tissue, {2, 4})

    np.testing.assert_array_equal(
        allowed,
        np.array([[False, True, False], [True, False, False]]),
    )


def test_probnet_mass_queue_is_seeded_and_coverage_contract_is_executable():
    candidates = [(0, value) for value in range(7)]
    probability = np.array(
        [[0.2, 0.9, 0.4, 0.7, 0.3, 0.8, 0.5]],
        dtype=np.float32,
    )

    np.random.seed(31)
    first = choose_weighted_centers(
        candidates,
        probability,
        target_count=len(candidates),
        gamma=1.5,
    )
    np.random.seed(31)
    second = choose_weighted_centers(
        candidates,
        probability,
        target_count=len(candidates),
        gamma=1.5,
        coverage_count=3,
        coverage_radius=2.0,
    )
    np.random.seed(31)
    repeated = choose_weighted_centers(
        candidates,
        probability,
        target_count=len(candidates),
        gamma=1.5,
        coverage_count=3,
        coverage_radius=2.0,
    )

    assert second == repeated
    assert set(first) == set(candidates)
    assert set(second) == set(candidates)
    assert first != sorted(
        candidates,
        key=lambda point: -float(probability[point]),
    )
    assert all(
        abs(second[index][1] - second[other][1]) >= 2
        for index in range(3)
        for other in range(index)
    )


def test_exact_backfill_queue_covers_every_supported_pixel_reproducibly():
    region = np.array(
        [[True, False, True], [True, True, False]],
        dtype=bool,
    )
    probability = np.array(
        [[0.7, 0.99, 0.7], [0.2, 0.9, 0.1]],
        dtype=np.float32,
    )

    np.random.seed(37)
    ranked = probability_mass_region_centers(region, probability, gamma=1.5)
    np.random.seed(37)
    repeated = probability_mass_region_centers(
        region,
        probability,
        gamma=1.5,
    )

    assert ranked.tolist() == repeated.tolist()
    assert {tuple(point) for point in ranked.tolist()} == {
        (0, 0),
        (0, 2),
        (1, 0),
        (1, 1),
    }


def test_exact_backfill_budget_is_quota_scaled_and_bounded():
    args = SimpleNamespace(
        exact_backfill_candidates_per_missing=128,
        exact_backfill_candidate_floor=512,
        exact_backfill_candidate_ceiling=4096,
    )

    small, small_audit = exact_backfill_candidate_budget(1, 10000, args)
    medium, medium_audit = exact_backfill_candidate_budget(20, 10000, args)
    large, large_audit = exact_backfill_candidate_budget(100, 10000, args)

    assert small == 512
    assert medium == 2560
    assert large == 4096
    assert small_audit["truncated"] is True
    assert medium_audit["shortfall"] == 20
    assert large_audit["policy"].endswith("next_deterministic_seed")


def test_failed_reference_shape_is_quarantined_before_library_fallback(
    monkeypatch,
):
    reference = {
        "type": 101,
        "mask": np.ones((3, 3), dtype=bool),
        "marker": "reference",
    }
    library = {
        "type": 101,
        "mask": np.ones((1, 1), dtype=bool),
        "marker": "library",
    }

    class FakeSampler:
        def __init__(self):
            self.remaining = [reference]
            self.released = []

        def sample_instance(self, tissue_id, nuc_type, allow_cross_tissue=True):
            del tissue_id, nuc_type, allow_cross_tissue
            if self.remaining:
                return self.remaining.pop(), "reference"
            return library, "library"

        def release_failed_instance(self, instance, source):
            assert source == "reference"
            self.remaining.append(instance)
            self.released.append(instance)

    monkeypatch.setattr(
        generate_module,
        "retry_transform_specs",
        lambda args, trial_count: [
            {
                "offset_yx": (0, 0),
                "rotation_quarters": 0,
                "flip_horizontal": False,
                "flip_vertical": False,
                "scale": 1.0,
            }
        ],
    )
    attempted_markers = []

    def fake_place(*args, **kwargs):
        del kwargs
        instance = args[3]
        attempted_markers.append(instance["marker"])
        return instance["marker"] == "library"

    monkeypatch.setattr(generate_module, "place_nucleus_layered", fake_place)
    sampler = FakeSampler()
    args = SimpleNamespace(
        dense_placement_shape_trials=2,
        placement_shape_trials=2,
        dense_placement_transform_trials=1,
        placement_transform_trials=1,
        no_augment_instances=True,
        max_nucleus_overlap_fraction=0.0,
        require_full_tissue_containment=True,
        nucleus_spacing_margin_px=1,
    )

    placed, source, trials, center = place_candidate_with_retries(
        output=np.zeros((7, 7), dtype=np.uint8),
        candidate_y=3,
        candidate_x=3,
        nucleus_type=101,
        tissue_id=1,
        shape_sampler=sampler,
        center_region=np.ones((7, 7), dtype=bool),
        valid_tissue_mask=np.ones((7, 7), dtype=bool),
        dense_retry=False,
        force_tissue_library=False,
        args=args,
    )

    assert placed is True
    assert source == "library"
    assert trials == 2
    assert center == (3, 3)
    assert attempted_markers == ["reference", "library"]
    assert sampler.remaining == [reference]
    assert sampler.released == [reference]


def test_quota_reassignment_uses_free_centers_across_same_tissue():
    tissue_region = np.zeros((5, 8), dtype=bool)
    tissue_region[1:4, 1:3] = True
    tissue_region[1:4, 5:7] = True
    output = np.zeros_like(tissue_region, dtype=np.uint8)
    output[2, 1] = 1
    probability = np.full(tissue_region.shape, 0.1, dtype=np.float32)
    probability[1:4, 5:7] = 0.9

    np.random.seed(41)
    centers = same_tissue_quota_reassignment_centers(
        tissue_region,
        output,
        probability,
        gamma=3.0,
        separation=1,
    )

    assert centers.shape[0] == 6
    assert {tuple(point) for point in centers.tolist()} == {
        (1, 5),
        (1, 6),
        (2, 5),
        (2, 6),
        (3, 5),
        (3, 6),
    }


def test_zero_quota_component_is_registered_for_same_tissue_reassignment():
    diagnostics = initialize_component_sampling_diagnostics(
        [(1, 90), (4, 10)],
        {1: 2},
        {1: 0.75, 4: 0.25},
    )

    assert diagnostics["1"]["quota"] == 2
    assert diagnostics["4"]["quota"] == 0
    assert diagnostics["4"]["attempted_centers"] == 0
    assert diagnostics["4"]["placed"] == 0
    assert diagnostics["4"]["probnet_mass_fraction"] == 0.25


def test_component_shape_policy_is_reported_when_stage_places_no_new_shape():
    class FakeSampler:
        def diagnostics(self):
            return {
                "policy": (
                    "same_class_reference_without_replacement_then_library"
                ),
                "selected_by_source": {"reference": 0, "library": 0},
            }

    diagnostics = shape_sampling_diagnostics(
        FakeSampler(),
        {},
        {"reference": 0, "library": 0},
        component_policy_active=True,
    )

    assert diagnostics["policy"] == COMPONENT_SHAPE_POLICY_NAME
    assert diagnostics["component_local"] == {}
    assert diagnostics["selected_by_source"] == {
        "reference": 0,
        "library": 0,
    }


def test_component_count_is_allocated_by_integrated_probnet_mass():
    counts = allocate_weight_proportional_counts(
        [(1, 90.0), (2, 10.0)],
        target_count=10,
    )

    assert counts == {1: 9, 2: 1}


def test_context_stabilized_probability_uses_geometric_blend():
    context = np.array([[0.81, 0.36]], dtype=np.float32)
    halo = np.array([[0.16, 0.64]], dtype=np.float32)

    blended = blend_context_stabilized_probability(
        context,
        halo,
        halo_weight=0.25,
    )

    expected = np.exp(0.75 * np.log(context) + 0.25 * np.log(halo))
    np.testing.assert_allclose(blended, expected, rtol=1e-6)
    assert blended[0, 0] < context[0, 0]
    assert blended[0, 1] > context[0, 1]


def test_spatial_context_halo_is_derived_from_nucleus_area():
    radius = spatial_context_halo_radius(
        80.0,
        diameter_scale=1.25,
        minimum=4,
        maximum=24,
    )

    assert radius == 13


def test_spatial_forward_moves_artificial_edit_boundary_outward(monkeypatch):
    captured = {}

    def fake_predict_fields(
        model,
        tissue_map,
        input_nuclei,
        edit_mask,
        cancer_id,
        device,
    ):
        captured["input_nuclei"] = input_nuclei.copy()
        captured["edit_mask"] = edit_mask.copy()
        prob = np.full((6, *edit_mask.shape), 0.1, dtype=np.float32)
        prob[0] = 0.5
        return prob, None

    monkeypatch.setattr(generate_module, "predict_fields", fake_predict_fields)
    tissue = np.ones((7, 7), dtype=np.uint8)
    nuclei = np.zeros((7, 7), dtype=np.uint8)
    nuclei[2, 3] = 1
    edit_mask = np.zeros((7, 7), dtype=bool)
    edit_mask[3, 3] = True
    context_prob = np.full((6, 7, 7), 0.1, dtype=np.float32)
    context_prob[0] = 0.5
    args = SimpleNamespace(
        spatial_context_halo_weight=1.0,
        expected_nucleus_area=1.0,
        spatial_context_halo_diameter_scale=1.0,
        spatial_context_halo_min_px=1,
        spatial_context_halo_max_px=1,
    )

    _, _, audit = predict_context_stabilized_spatial_probability(
        object(),
        tissue,
        nuclei,
        edit_mask,
        0,
        "cpu",
        context_prob,
        args,
    )

    expected_support = np.zeros((7, 7), dtype=bool)
    expected_support[3, 3] = True
    expected_support[2, 3] = True
    expected_support[4, 3] = True
    expected_support[3, 2] = True
    expected_support[3, 4] = True
    np.testing.assert_array_equal(captured["edit_mask"], expected_support)
    assert captured["input_nuclei"][2, 3] == 0
    assert audit["edit_mask_area"] == 1
    assert audit["prediction_support_area"] == 5
    assert not audit["second_forward_skipped"]


def test_probability_mass_sampling_favors_but_does_not_monopolize_high_score():
    candidates = [(0, 0), (0, 1)]
    probability = np.array([[0.65, 0.55]], dtype=np.float32)

    np.random.seed(41)
    first_choices = [
        choose_weighted_centers(
            candidates,
            probability,
            target_count=1,
            gamma=1.5,
        )[0]
        for _ in range(1000)
    ]

    high_count = first_choices.count((0, 0))
    assert 580 < high_count < 720


def test_accepted_centers_overlay_marks_only_new_instances():
    probability = np.full((32, 32), 0.5, dtype=np.float32)
    input_nuclei = np.zeros((32, 32), dtype=np.uint8)
    input_nuclei[4:7, 4:7] = 1
    output_nuclei = input_nuclei.copy()
    output_nuclei[20:23, 20:23] = 2
    edit_mask = np.ones((32, 32), dtype=bool)

    rendered = make_accepted_centers_overlay(
        probability,
        input_nuclei,
        output_nuclei,
        edit_mask,
    )

    assert rendered.shape == (32, 32, 3)
    assert tuple(rendered[21, 21]) != tuple(rendered[5, 5])


def test_required_seam_stage_merges_into_one_exact_population_ledger():
    required = {
        "placed": 1,
        "placed_by_shape_source": {"reference": 1, "library": 0},
        "tissues": {
            "1": {
                "target_count": 1,
                "placed": 1,
                "placed_by_type": {"101": 1},
                "target_by_type": None,
                "posterior_expected_by_type": {"101": 0.8},
                "accepted_centers": [
                    {"row": 10, "col": 11, "nucleus_type": 101}
                ],
            }
        },
    }
    remainder = {
        "placed": 12,
        "placed_by_shape_source": {"reference": 7, "library": 5},
        "shape_sampling": {},
        "tissues": {
            "1": {
                "target_count": 12,
                "placed": 12,
                "placed_by_type": {"101": 12},
                "target_by_type": None,
                "posterior_expected_by_type": {"101": 11.2},
                "accepted_centers": [
                    {"row": 20 + index, "col": 21, "nucleus_type": 101}
                    for index in range(12)
                ],
            }
        },
    }

    _merge_required_center_stage_diagnostics(
        remainder,
        required,
        required_tissue_id=1,
        required_center_pixels=817,
        minimum_required_centers=1,
    )

    assert remainder["placed"] == 13
    assert remainder["tissues"]["1"]["target_count"] == 13
    assert remainder["tissues"]["1"]["placed"] == 13
    assert len(remainder["tissues"]["1"]["accepted_centers"]) == 13
    assert remainder["placed_by_shape_source"] == {
        "reference": 8,
        "library": 5,
    }
    assert (
        remainder["regeneration_stages"]["policy"]
        == "typed_seam_quota_then_full_P_population_remainder_v3"
    )


def test_boundary_modifier_requires_typed_seam_quota_not_all_population_centers():
    mechanism = np.zeros((32, 32), dtype=bool)
    mechanism[8:12, 8:24] = True
    continuity = mechanism.copy()
    ledger = tuple(
        [(9, 9 + index, 101) for index in range(3)]
        + [(20, 4 + index, 101) for index in range(7)]
    )

    certificate = _mechanism_modifier_certificate(
        mechanism_program_id="boundary_aligned",
        accepted_center_ledger=ledger,
        mechanism_region=mechanism,
        continuity_region=continuity,
        required_nucleus_class=101,
        minimum_required_placements=3,
        sampling_audit_passed=True,
    )

    assert certificate["passed"] is True
    assert certificate["typed_continuity_count"] == 3
    assert certificate["inside_mechanism_count"] == 3
    assert len(certificate["outside_mechanism_centers"]) == 7
    assert (
        certificate["policy"]
        == "typed_seam_quota_in_continuity_band_with_population_remainder_in_P"
    )


def test_dense_sheet_modifier_still_rejects_centers_outside_mechanism_region():
    mechanism = np.zeros((20, 20), dtype=bool)
    mechanism[3:17, 3:17] = True

    certificate = _mechanism_modifier_certificate(
        mechanism_program_id="dense_sheet",
        accepted_center_ledger=((8, 8, 101), (18, 18, 101)),
        mechanism_region=mechanism,
        continuity_region=np.zeros_like(mechanism),
        required_nucleus_class=101,
        minimum_required_placements=0,
        sampling_audit_passed=True,
    )

    assert certificate["passed"] is False
    assert certificate["inside_mechanism_count"] == 1
    assert len(certificate["outside_mechanism_centers"]) == 1


def test_required_seam_stage_materializes_zero_remainder_ledger():
    required = {
        "placed": 3,
        "placed_by_shape_source": {"reference": 3, "library": 0},
        "tissues": {
            "2": {
                "target_count": 3,
                "placed": 3,
                "placed_by_type": {"103": 3},
                "target_by_type": {},
                "posterior_expected_by_type": {"103": 3.0},
                "accepted_centers": [
                    {"row": 5 + index * 4, "col": 7, "nucleus_type": 103}
                    for index in range(3)
                ],
                "type_shape_support": {"supported_types": [103]},
            }
        },
    }
    remainder = {
        "placed": 0,
        "placed_by_shape_source": {"reference": 0, "library": 0},
        "shape_sampling": {},
        "tissues": {},
    }

    _merge_required_center_stage_diagnostics(
        remainder,
        required,
        required_tissue_id=2,
        required_center_pixels=100,
        minimum_required_centers=3,
    )

    info = remainder["tissues"]["2"]
    assert info["zero_remainder_materialized"] is True
    assert info["target_count"] == 3
    assert info["placed"] == 3
    assert len(info["accepted_centers"]) == 3


def test_two_stage_execution_uses_compiled_total_for_remainder(monkeypatch):
    calls = []

    def fake_generate_for_gamma(*args, **kwargs):
        override = kwargs.get("new_target_count_overrides") or {}
        count = int(override.get(2, 0))
        calls.append(count)
        return np.asarray(args[2]).copy(), {
            "placed": count,
            "placed_by_shape_source": {"reference": count, "library": 0},
            "tissues": {
                "2": {
                    "target_count": count,
                    "placed": count,
                    "placed_by_type": {"103": count},
                    "target_by_type": {"103": count},
                    "posterior_expected_by_type": {"103": float(count)},
                    "accepted_centers": [
                        {"row": 2, "col": 2, "nucleus_type": 103}
                    ]
                    * count,
                }
            },
        }

    monkeypatch.setattr(
        generate_module,
        "generate_for_gamma",
        fake_generate_for_gamma,
    )
    tissue = np.full((8, 8), 2, dtype=np.int64)
    mask = np.ones_like(tissue, dtype=bool)
    required = np.zeros_like(mask)
    required[2:4, 2:4] = True
    _, diagnostics = generate_two_stage_for_gamma(
        np.zeros((6, 8, 8), dtype=np.float32),
        tissue,
        np.zeros_like(tissue),
        mask,
        mask,
        object(),
        object(),
        1.0,
        SimpleNamespace(skip_tissue_ids=[]),
        {},
        population_mask=mask,
        required_center_mask=required,
        minimum_required_centers=15,
        maximum_required_centers=20,
        required_nucleus_type=3,
        packing_witness={
            "requested_count": 25,
            "required_seam_count": 15,
        },
    )
    assert calls == [15, 10]
    assert diagnostics["tissues"]["2"]["target_count"] == 25
    assert diagnostics["tissues"]["2"]["placed"] == 25


def test_two_stage_rejects_runtime_seam_quota_that_differs_from_contract():
    tissue = np.full((8, 8), 2, dtype=np.int64)
    mask = np.ones_like(tissue, dtype=bool)
    required = np.zeros_like(mask)
    required[2:4, 2:4] = True

    with pytest.raises(PlacementQuotaError, match="immutable packing certificate"):
        generate_two_stage_for_gamma(
            np.zeros((6, 8, 8), dtype=np.float32),
            tissue,
            np.zeros_like(tissue),
            mask,
            mask,
            object(),
            object(),
            1.0,
            SimpleNamespace(skip_tissue_ids=[]),
            {},
            population_mask=mask,
            required_center_mask=required,
            minimum_required_centers=8,
            maximum_required_centers=8,
            packing_witness={
                "requested_count": 25,
                "required_seam_count": 15,
            },
        )
