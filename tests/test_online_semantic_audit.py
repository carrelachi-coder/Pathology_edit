import numpy as np

from phase3_mask_edit.audit import (
    ConservativeP1Policy,
    OnlineAuditPolicy,
    OnlineSemanticAuditor,
    SemanticPrediction,
    apply_conservative_p1,
    dataset_native_metric_class_ids,
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
