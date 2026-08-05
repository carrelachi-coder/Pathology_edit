import numpy as np

from phase3_mask_edit.audit import (
    ConservativeP1Policy,
    OnlineAuditPolicy,
    OnlineSemanticAuditor,
    SemanticPrediction,
    apply_conservative_p1,
    dataset_native_metric_class_ids,
    source_evaluator_quality,
    source_relative_tissue_metrics,
    to_coarse_mask,
)


def _probabilities(mask: np.ndarray, confidence: float = 0.96) -> np.ndarray:
    class_count = 8
    remainder = (1.0 - confidence) / (class_count - 1)
    probabilities = np.full(
        (class_count, *mask.shape), remainder, dtype=np.float32
    )
    for class_id in range(class_count):
        probabilities[class_id][mask == class_id] = confidence
    return probabilities


def _edit_pair() -> tuple[np.ndarray, np.ndarray]:
    source = np.ones((24, 24), dtype=np.uint8)
    target = source.copy()
    target[9:15, 9:15] = 2
    return source, target


def test_p1_relabels_only_a_low_confidence_island_with_strong_context():
    source, target = _edit_pair()
    prediction = target.copy()
    prediction[2, 2] = 2
    probabilities = _probabilities(prediction)
    probabilities[:, 2, 2] = 0.01
    probabilities[1, 2, 2] = 0.43
    probabilities[2, 2, 2] = 0.51

    result = apply_conservative_p1(
        predicted_mask=prediction,
        probabilities=probabilities,
        source_mask=source,
        target_mask=target,
        source_prediction=source,
        policy=ConservativeP1Policy(protected_boundary_radius_pixels=1),
    )

    assert result.raw_mask[2, 2] == 2
    assert result.audited_mask[2, 2] == 1
    assert len(result.operations) == 1
    assert result.operations[0].operation == "enclosed_hole_fill"
    assert result.operations[0].region == "U_far"


def test_p1_does_not_modify_the_semantic_boundary_band():
    source, target = _edit_pair()
    prediction = target.copy()
    prediction[9, 10] = 3
    probabilities = _probabilities(prediction)
    probabilities[:, 9, 10] = 0.01
    probabilities[2, 9, 10] = 0.43
    probabilities[3, 9, 10] = 0.51

    result = apply_conservative_p1(
        predicted_mask=prediction,
        probabilities=probabilities,
        source_mask=source,
        target_mask=target,
        source_prediction=source,
        policy=ConservativeP1Policy(protected_boundary_radius_pixels=1),
    )

    assert result.audited_mask[9, 10] == 3
    assert not result.operations


def test_p1_preserves_a_stable_source_component_outside_the_edit():
    source, target = _edit_pair()
    source_prediction = source.copy()
    source_prediction[2, 2] = 2
    prediction = target.copy()
    prediction[2, 2] = 2
    probabilities = _probabilities(prediction)
    probabilities[:, 2, 2] = 0.01
    probabilities[1, 2, 2] = 0.43
    probabilities[2, 2, 2] = 0.51

    result = apply_conservative_p1(
        predicted_mask=prediction,
        probabilities=probabilities,
        source_mask=source,
        target_mask=target,
        source_prediction=source_prediction,
        policy=ConservativeP1Policy(protected_boundary_radius_pixels=1),
    )

    assert result.audited_mask[2, 2] == 2
    assert not result.operations


def test_online_auditor_shadow_never_changes_the_decision_mask():
    source, target = _edit_pair()
    generated = target.copy()
    generated[2, 2] = 2
    generated_probabilities = _probabilities(generated)
    generated_probabilities[:, 2, 2] = 0.01
    generated_probabilities[1, 2, 2] = 0.43
    generated_probabilities[2, 2, 2] = 0.51

    auditor = OnlineSemanticAuditor(
        OnlineAuditPolicy(
            postprocess_mode="shadow",
            p1=ConservativeP1Policy(protected_boundary_radius_pixels=1),
        )
    )
    result = auditor.audit(
        source_mask=source,
        target_mask=target,
        source_prediction=SemanticPrediction(source, _probabilities(source)),
        generated_prediction=SemanticPrediction(
            generated, generated_probabilities
        ),
    )

    assert result.decision_input == "raw"
    np.testing.assert_array_equal(result.decision_mask, generated)
    assert result.p1_result is not None
    assert result.p1_result.audited_mask[2, 2] == 1
    assert result.to_metadata()["audit_scope"] == "online_agent_product"


def test_dataset_native_label_contract_excludes_unannotated_and_unsupported_ids():
    assert dataset_native_metric_class_ids("PANDA", level="fine") == (
        2,
        5,
        8,
        9,
        10,
    )
    assert 3 not in dataset_native_metric_class_ids("PANDA", level="fine")
    assert 14 not in dataset_native_metric_class_ids("BCSS", level="fine")


def test_source_evaluator_quality_can_be_calibrated_on_the_edit_region():
    source = np.ones((12, 12), dtype=np.uint8)
    prediction = np.full_like(source, 2)
    region = np.zeros_like(source, dtype=bool)
    region[4:8, 4:8] = True
    prediction[region] = 1

    quality = source_evaluator_quality(
        source_mask=source,
        source_prediction=prediction,
        source_probabilities=_probabilities(prediction),
        class_ids=(1, 2),
        region=region,
    )

    assert quality["metrics"]["source_region_accuracy"] == 1.0
    assert quality["metrics"]["source_class_recall_min"] == 1.0


def test_source_relative_metrics_capture_soft_semantic_direction_without_argmax_change():
    source, target = _edit_pair()
    changed = source != target
    source_prediction = np.full_like(source, 3)
    generated_prediction = np.full_like(source, 3)
    source_probabilities = np.full((8, *source.shape), 0.05, dtype=np.float32)
    source_probabilities[3] = 0.65
    generated_probabilities = np.array(source_probabilities, copy=True)
    source_probabilities[1, changed] = 0.20
    source_probabilities[2, changed] = 0.10
    source_probabilities[3, changed] = 0.45
    generated_probabilities[1, changed] = 0.10
    generated_probabilities[2, changed] = 0.20
    generated_probabilities[3, changed] = 0.45

    metrics = source_relative_tissue_metrics(
        source_mask=source,
        target_mask=target,
        source_prediction=source_prediction,
        generated_prediction=generated_prediction,
        source_probabilities=source_probabilities,
        generated_probabilities=generated_probabilities,
        class_ids=(1, 2, 3),
    )

    changed_metrics = metrics["changed_region"]
    assert changed_metrics["target_probability_gain"] > 0
    assert changed_metrics["source_probability_suppression"] > 0
    assert changed_metrics["soft_margin_gain"] > 0
    assert changed_metrics["target_probability_gain_fraction"] == 1.0
    assert changed_metrics["source_probability_suppression_fraction"] == 1.0
    assert changed_metrics["margin_gain_fraction"] == 1.0


def test_source_relative_metrics_keep_target_only_direction_without_scoring_other():
    source = np.full((24, 24), 7, dtype=np.uint8)
    target = source.copy()
    target[9:15, 9:15] = 1
    changed = source != target
    source_prediction = np.full_like(source, 7)
    generated_prediction = np.full_like(source, 7)
    source_probabilities = _probabilities(source_prediction)
    generated_probabilities = np.array(source_probabilities, copy=True)
    source_probabilities[:, changed] = 0.10 / 6
    generated_probabilities[:, changed] = 0.10 / 6
    source_probabilities[1, changed] = 0.10
    generated_probabilities[1, changed] = 0.30
    source_probabilities[7, changed] = 0.80
    generated_probabilities[7, changed] = 0.60

    metrics = source_relative_tissue_metrics(
        source_mask=source,
        target_mask=target,
        source_prediction=source_prediction,
        generated_prediction=generated_prediction,
        source_probabilities=source_probabilities,
        generated_probabilities=generated_probabilities,
        class_ids=(1,),
    )

    changed_metrics = metrics["changed_region"]
    assert changed_metrics["target_direction_support_pixels"] == 36
    assert changed_metrics["source_direction_support_pixels"] == 0
    assert changed_metrics["margin_direction_support_pixels"] == 0
    assert changed_metrics["target_probability_gain_fraction"] == 1.0
    assert changed_metrics["source_probability_suppression_fraction"] is None
    assert changed_metrics["margin_gain_fraction"] is None


def test_source_relative_metrics_keep_source_only_direction_without_scoring_other():
    source = np.ones((24, 24), dtype=np.uint8)
    target = source.copy()
    target[9:15, 9:15] = 7
    changed = source != target
    source_prediction = np.ones_like(source)
    generated_prediction = np.ones_like(source)
    source_probabilities = _probabilities(source_prediction)
    generated_probabilities = np.array(source_probabilities, copy=True)
    source_probabilities[:, changed] = 0.10 / 6
    generated_probabilities[:, changed] = 0.10 / 6
    source_probabilities[1, changed] = 0.80
    generated_probabilities[1, changed] = 0.55
    source_probabilities[7, changed] = 0.10
    generated_probabilities[7, changed] = 0.35

    metrics = source_relative_tissue_metrics(
        source_mask=source,
        target_mask=target,
        source_prediction=source_prediction,
        generated_prediction=generated_prediction,
        source_probabilities=source_probabilities,
        generated_probabilities=generated_probabilities,
        class_ids=(1,),
    )

    changed_metrics = metrics["changed_region"]
    assert changed_metrics["target_direction_support_pixels"] == 0
    assert changed_metrics["source_direction_support_pixels"] == 36
    assert changed_metrics["margin_direction_support_pixels"] == 0
    assert changed_metrics["target_probability_gain_fraction"] is None
    assert changed_metrics["source_probability_suppression_fraction"] == 1.0
    assert changed_metrics["margin_gain_fraction"] is None


def test_fine_only_edit_uses_the_fine_region_for_coarse_preservation_audit():
    source_fine = np.full((20, 20), 9, dtype=np.uint8)
    target_fine = source_fine.copy()
    target_fine[6:14, 6:14] = 10
    source_coarse = to_coarse_mask(source_fine)
    target_coarse = to_coarse_mask(target_fine)
    semantic = source_fine != target_fine
    coarse_probabilities = _probabilities(source_coarse)
    fine_probabilities_source = np.zeros((16, 20, 20), dtype=np.float32)
    fine_probabilities_target = np.zeros((16, 20, 20), dtype=np.float32)
    for probabilities, mask in (
        (fine_probabilities_source, source_fine),
        (fine_probabilities_target, target_fine),
    ):
        probabilities[:] = 0.04 / 15
        for class_id in range(16):
            probabilities[class_id][mask == class_id] = 0.96

    auditor = OnlineSemanticAuditor(OnlineAuditPolicy(postprocess_mode="off"))
    result = auditor.audit(
        source_mask=source_coarse,
        target_mask=target_coarse,
        source_prediction=SemanticPrediction(
            source_coarse, coarse_probabilities
        ),
        generated_prediction=SemanticPrediction(
            target_coarse, coarse_probabilities
        ),
        semantic_change_region=semantic,
        source_fine_mask=source_fine,
        target_fine_mask=target_fine,
        source_fine_prediction=SemanticPrediction(
            source_fine, fine_probabilities_source
        ),
        generated_fine_prediction=SemanticPrediction(
            target_fine, fine_probabilities_target
        ),
        fine_class_ids=(8, 9, 10),
    )

    assert result.raw_metrics["changed_region"]["accuracy"] == 1.0
    assert result.fine_metrics["changed_region"]["accuracy"] == 1.0
    assert (
        result.fine_metrics["changed_region"]["macro_iou_detail"][
            "macro_policy"
        ]
        == "target_or_prediction_present"
    )
    assert (
        result.fine_metrics["source_evaluator_calibration"]["accuracy"]
        == 1.0
    )
    assert (
        result.fine_metrics["source_evaluator_calibration"]["macro_miou"]
        == 1.0
    )


def test_preservation_excludes_the_full_generation_context():
    source = np.ones((20, 20), dtype=np.uint8)
    target = source.copy()
    semantic = np.zeros_like(source, dtype=bool)
    semantic[8:12, 8:12] = True
    target[semantic] = 2
    generation = np.zeros_like(source, dtype=bool)
    generation[4:16, 4:16] = True
    generated = target.copy()
    generated[generation & ~semantic] = 2

    without_context = source_relative_tissue_metrics(
        source_mask=source,
        target_mask=target,
        source_prediction=source,
        generated_prediction=generated,
        source_probabilities=_probabilities(source),
        generated_probabilities=_probabilities(generated),
        class_ids=(1, 2),
        semantic_change_region=semantic,
    )
    with_context = source_relative_tissue_metrics(
        source_mask=source,
        target_mask=target,
        source_prediction=source,
        generated_prediction=generated,
        source_probabilities=_probabilities(source),
        generated_probabilities=_probabilities(generated),
        class_ids=(1, 2),
        semantic_change_region=semantic,
        preservation_exclusion_region=generation,
    )

    assert without_context["preservation"]["prediction_relative_drift_U_far"] > 0
    assert with_context["preservation"]["prediction_relative_drift_U_far"] == 0
    assert with_context["changed_region"] == without_context["changed_region"]
    assert with_context["boundary"] == without_context["boundary"]
    assert with_context["region_pixels"]["R"] == int(np.count_nonzero(semantic))
    assert with_context["region_pixels"]["P_exclude"] == int(
        np.count_nonzero(generation)
    )


def test_cross_global_appearance_shift_is_calibrated_before_drift_scoring():
    source = np.ones((48, 48), dtype=np.uint8)
    target = source.copy()
    target[20:28, 20:28] = 2
    source_probabilities = np.zeros((8, 48, 48), dtype=np.float32)
    source_probabilities[1] = 0.80
    source_probabilities[3] = 0.20
    source_probabilities[1, :, :10] = 0.55
    source_probabilities[3, :, :10] = 0.45
    generated_probabilities = np.array(source_probabilities, copy=True)
    generated_probabilities[1] -= 0.15
    generated_probabilities[3] += 0.15
    changed = source != target
    generated_probabilities[:, changed] = 0.0
    generated_probabilities[2, changed] = 0.90
    generated_probabilities[1, changed] = 0.10
    source_prediction = np.argmax(source_probabilities, axis=0).astype(np.uint8)
    generated_prediction = np.argmax(
        generated_probabilities, axis=0
    ).astype(np.uint8)

    metrics = source_relative_tissue_metrics(
        source_mask=source,
        target_mask=target,
        source_prediction=source_prediction,
        generated_prediction=generated_prediction,
        source_probabilities=source_probabilities,
        generated_probabilities=generated_probabilities,
        class_ids=(1, 2, 3),
    )

    preservation = metrics["preservation"]
    assert preservation["prediction_relative_drift_U_far"] > 0.15
    assert preservation["appearance_calibration_applicable"]
    assert preservation["appearance_calibrated_prediction_drift_U_far"] == 0.0
    assert preservation["global_appearance_probability_shift_l1"] > 0.0


def test_localized_semantic_replacement_survives_appearance_calibration():
    source = np.ones((48, 48), dtype=np.uint8)
    target = source.copy()
    target[20:28, 20:28] = 2
    source_probabilities = np.zeros((8, 48, 48), dtype=np.float32)
    source_probabilities[1] = 0.90
    source_probabilities[3] = 0.10
    generated_probabilities = np.array(source_probabilities, copy=True)
    generated_probabilities[:, 2:16, 2:16] = 0.0
    generated_probabilities[3, 2:16, 2:16] = 0.90
    generated_probabilities[1, 2:16, 2:16] = 0.10
    changed = source != target
    generated_probabilities[:, changed] = 0.0
    generated_probabilities[2, changed] = 0.90
    generated_probabilities[1, changed] = 0.10
    source_prediction = np.argmax(source_probabilities, axis=0).astype(np.uint8)
    generated_prediction = np.argmax(
        generated_probabilities, axis=0
    ).astype(np.uint8)

    metrics = source_relative_tissue_metrics(
        source_mask=source,
        target_mask=target,
        source_prediction=source_prediction,
        generated_prediction=generated_prediction,
        source_probabilities=source_probabilities,
        generated_probabilities=generated_probabilities,
        class_ids=(1, 2, 3),
    )

    preservation = metrics["preservation"]
    assert preservation["appearance_calibration_applicable"]
    assert preservation["appearance_calibrated_prediction_drift_U_far"] > 0.08
