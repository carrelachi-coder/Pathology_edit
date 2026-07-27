import json
from pathlib import Path

import numpy as np

from phase3_mask_edit.benchmark.conditional_fidelity import (
    Detections,
    cell_distribution_metrics,
    detections_from_conic,
    jensen_shannon_divergence,
    load_label_mask,
    rescale_detections,
    spatial_matching_metrics,
    tissue_fidelity_metrics,
)
from phase3_mask_edit.cli.run_conditional_fidelity_benchmark import scalar_metrics


def test_tissue_fidelity_is_perfect_for_identical_masks():
    target = np.asarray([[0, 1], [2, 2]])

    metrics = tissue_fidelity_metrics(target, target, class_ids=(0, 1, 2))

    assert metrics["macro_dice"] == 1.0
    assert metrics["macro_miou"] == 1.0
    assert metrics["class_presence_recall"] == 1.0
    assert metrics["tissue_area_distribution_jsd"] == 0.0


def test_tissue_macro_penalizes_a_hallucinated_class():
    target = np.asarray([[0, 1], [1, 1]])
    predicted = np.asarray([[2, 1], [1, 1]])

    metrics = tissue_fidelity_metrics(target, predicted, class_ids=(0, 1, 2))

    assert metrics["macro_dice"] < 1.0
    assert metrics["per_class"]["2"]["iou"] == 0.0


def test_jsd_is_symmetric_and_bounded():
    left = np.asarray([8, 2, 0])
    right = np.asarray([2, 8, 0])

    value = jensen_shannon_divergence(left, right)

    assert 0 < value < 1
    assert np.isclose(value, jensen_shannon_divergence(right, left))


def test_conic_loader_uses_instance_centroids_and_majority_type(tmp_path: Path):
    conic = np.zeros((5, 5, 2), dtype=np.int32)
    conic[1:3, 1:3, 0] = 1
    conic[1:3, 1:3, 1] = 3
    path = tmp_path / "conic.npy"
    np.save(path, conic)

    detections = detections_from_conic(path, mpp=0.5)

    np.testing.assert_allclose(detections.xy, [[1.5, 1.5]])
    np.testing.assert_array_equal(detections.class_ids, [3])


def test_cell_distribution_and_spatial_matching():
    target = Detections(
        xy=np.asarray([[10, 10], [30, 30]]),
        class_ids=np.asarray([1, 2]),
        mpp=0.5,
        image_size=(256, 256),
    )
    predicted = Detections(
        xy=np.asarray([[12, 10], [31, 30], [100, 100]]),
        class_ids=np.asarray([1, 2, 2]),
        mpp=0.5,
        image_size=(256, 256),
    )

    distribution = cell_distribution_metrics(target, predicted, class_ids=(1, 2))
    spatial = spatial_matching_metrics(
        target, predicted, max_distance_um=3.0, class_aware=True
    )

    assert distribution["total_count_abs_error"] == 1
    assert spatial["true_positive"] == 2
    assert spatial["false_positive"] == 1
    assert spatial["false_negative"] == 0


def test_conic_detections_are_rescaled_to_the_512_evaluation_grid():
    detections = Detections(
        xy=np.asarray([[1.5, 2.5]]),
        class_ids=np.asarray([3]),
        mpp=0.5,
        image_size=(256, 256),
    )

    scaled = rescale_detections(detections, image_size=(512, 512), mpp=0.25)

    np.testing.assert_allclose(scaled.xy, [[3.0, 5.0]])
    assert scaled.image_size == (512, 512)
    assert scaled.mpp == 0.25


def test_label_loader_rejects_non_512_evaluation_masks(tmp_path: Path):
    from PIL import Image
    import pytest

    path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(path)

    with pytest.raises(ValueError, match="expected label mask size"):
        load_label_mask(path, expected_size=(512, 512))


def test_conditional_fidelity_config_freezes_pathdiff_conic_policy():
    import yaml

    config = yaml.safe_load(Path("benchmark_configs/conditional_fidelity.yaml").read_text())

    assert config["models"]["pathdiff_conic"]["strict_spatial_evaluator"] == "conic"
    assert config["models"]["pixcell_controlnet"]["strict_spatial_evaluator"] == "cellvit"
    assert config["models"]["unipath_7b"]["strict_spatial_evaluator"] is None
    assert config["models"]["cross_v1_project"]["condition_structure_metrics"] is True
    assert config["models"]["mupad_text"]["condition_structure_metrics"] is False
    assert config["evaluation_frame"] == {
        "image_size": [512, 512],
        "mpp": 0.25,
        "fov_um": 128.0,
        "input_policy": "mpp_normalized_generated_patch_only",
    }


def test_scalar_metrics_flattens_per_class_values():
    flattened = scalar_metrics(
        "tissue",
        {
            "macro_dice": 0.75,
            "per_class": {"1": {"area_fraction_abs_error": 0.125}},
            "policy": "ignored text",
        },
    )

    assert flattened == {
        "tissue.macro_dice": 0.75,
        "tissue.per_class.1.area_fraction_abs_error": 0.125,
    }
