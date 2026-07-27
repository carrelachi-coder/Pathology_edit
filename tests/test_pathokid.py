from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from phase3_mask_edit.benchmark.pathokid import (
    BootstrapDraws,
    cluster_bootstrap_kid,
    l2_normalize,
    polynomial_kernel,
    stable_transform_repr,
    subset_kid,
    unbiased_kid,
    paired_bootstrap_delta,
    real_vs_real_kid_curve,
)
from phase3_mask_edit.cli.run_pathokid_benchmark import validate_image_frame


def test_l2_normalize_produces_unit_rows():
    features = np.asarray([[3.0, 4.0], [5.0, 12.0]])

    normalized = l2_normalize(features)

    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_unbiased_kid_is_symmetric_and_permutation_invariant():
    real = l2_normalize(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    generated = l2_normalize(
        np.asarray([[1.0, 2.0], [2.0, 1.0], [1.0, -1.0]])
    )

    expected = unbiased_kid(real, generated)

    assert np.isclose(expected, unbiased_kid(generated, real))
    assert np.isclose(expected, unbiased_kid(real[::-1], generated[[1, 2, 0]]))


def test_polynomial_kernel_uses_standard_feature_dimension_scale():
    left = np.asarray([[1.0, 0.0]])
    right = np.asarray([[0.5, np.sqrt(0.75)]])

    kernel = polynomial_kernel(left, right)

    np.testing.assert_allclose(kernel, [[1.25**3]])


def test_subset_kid_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(7)
    real = l2_normalize(rng.normal(size=(12, 8)))
    generated = l2_normalize(rng.normal(size=(12, 8)))

    first = subset_kid(real, generated, subset_size=8, repeats=5, seed=41)
    second = subset_kid(real, generated, subset_size=8, repeats=5, seed=41)

    np.testing.assert_array_equal(first, second)


def test_cluster_bootstrap_matches_explicit_repeated_rows():
    real = l2_normalize(
        np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    )
    generated = l2_normalize(
        np.asarray([[0.9, 0.1], [0.7, 0.3], [0.1, 0.9], [0.3, 0.7]])
    )
    draws = BootstrapDraws(
        group_names=("a", "b"),
        counts=np.asarray([[2, 0], [0, 2], [1, 1]]),
        seed=3,
    )

    actual = cluster_bootstrap_kid(real, generated, ["a", "a", "b", "b"], draws)
    explicit_indices = ([0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 2, 3])
    expected = np.asarray(
        [unbiased_kid(real[index], generated[index]) for index in explicit_indices]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_pathokid_config_freezes_two_feature_extractors():
    config = yaml.safe_load(Path("benchmark_configs/pathokid.yaml").read_text())

    assert set(config["feature_extractors"]) == {"uni2h", "conch"}
    assert config["evaluation"]["bootstrap_repeats"] == 100
    assert config["evaluation"]["seed"] == 20260715
    assert config["evaluation_frame"]["image_size"] == [512, 512]


def test_pathokid_rejects_non_512_source_images(tmp_path: Path):
    valid = tmp_path / "valid.png"
    invalid = tmp_path / "invalid.png"
    Image.new("RGB", (512, 512), "white").save(valid)
    Image.new("RGB", (256, 256), "white").save(invalid)

    validate_image_frame([valid], expected_size=(512, 512))
    with pytest.raises(ValueError, match="requires normalized"):
        validate_image_frame([invalid], expected_size=(512, 512))


def test_transform_repr_removes_process_specific_addresses():
    class Transform:
        def __repr__(self):
            return "Compose(<function rgb at 0x7ffee1234abc>)"

    assert stable_transform_repr(Transform()) == (
        "Compose(<function rgb at 0xADDR>)"
    )


def test_real_vs_real_curve_uses_disjoint_then_bootstrap_modes():
    rng = np.random.default_rng(17)
    features = l2_normalize(rng.normal(size=(20, 8)))
    strata = ["a"] * 10 + ["b"] * 10

    curve = real_vs_real_kid_curve(
        features, strata, sample_sizes=(6, 12), repeats=4, seed=3
    )

    assert curve[6]["sampling_mode"] == "stratified_disjoint"
    assert np.all(curve[6]["source_overlap_count"] == 0)
    assert curve[12]["sampling_mode"] == "stratified_independent_bootstrap"
    assert len(curve[12]["values"]) == 4


def test_paired_bootstrap_delta_reports_probability_left_is_lower():
    result = paired_bootstrap_delta(
        np.asarray([1.0, 2.0, 4.0]), np.asarray([2.0, 3.0, 3.0])
    )

    assert np.isclose(result["probability_left_better"], 2 / 3)
    assert result["ci95_low"] < result["ci95_high"]
