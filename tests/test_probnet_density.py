import numpy as np
import torch

from inpaint_cells.data.density_targets import (
    build_center_density_targets,
    expand_edit_mask_to_complete_instances,
    extract_class_centers,
    select_instances_by_centroid,
)
from inpaint_cells.losses.focal_dice import CenterDensityLoss
from inpaint_cells.models.prob_unet import ProbUNet, freeze_non_density_parameters
from inpaint_cells.sampling_policy import widen_locally_thin_mask


def test_complete_instance_erasure_expands_intersecting_components_only():
    nuclei = np.zeros((16, 16), dtype=np.uint8)
    nuclei[3:7, 3:7] = 1
    nuclei[10:13, 10:13] = 1
    edit = np.zeros_like(nuclei, dtype=bool)
    edit[5:9, 5:9] = True

    expanded = expand_edit_mask_to_complete_instances(nuclei, edit)

    assert expanded[3:7, 3:7].all()
    assert not expanded[10:13, 10:13].any()


def test_centroid_erasure_keeps_a_boundary_crossing_nucleus_whole():
    nuclei = np.zeros((20, 20), dtype=np.uint8)
    nuclei[5:10, 4:12] = 1
    nuclei[12:16, 12:16] = 2
    edit = np.zeros_like(nuclei, dtype=bool)
    edit[:, 10:] = True

    selected = select_instances_by_centroid(nuclei, edit)

    assert not selected[5:10, 4:12].any()
    assert selected[12:16, 12:16].all()


def test_frozen_support_policy_widens_a_thin_branch():
    semantic = np.zeros((80, 80), dtype=bool)
    semantic[10:70, 40] = True

    widened = widen_locally_thin_mask(
        semantic,
        np.ones_like(semantic),
        minimum_width=33,
    )

    assert np.all(semantic <= widened)
    assert np.count_nonzero(widened[40]) >= 33


def test_density_target_preserves_one_unit_per_center_at_boundaries():
    nuclei = np.zeros((20, 20), dtype=np.uint8)
    nuclei[0:3, 0:3] = 1
    nuclei[7:10, 7:10] = 1
    nuclei[15:19, 16:20] = 2
    tissue = np.zeros_like(nuclei, dtype=np.int64)
    tissue[:, 10:] = 3
    edit = np.ones_like(nuclei, dtype=bool)

    density, counts = build_center_density_targets(
        nuclei,
        tissue,
        edit,
        sigma=2.0,
    )

    np.testing.assert_allclose(density.sum(axis=(1, 2)), [2.0, 1.0, 0, 0, 0], atol=1e-6)
    assert counts.sum() == 3
    assert counts[0, 0] == 2
    assert counts[3, 1] == 1


def test_density_target_uses_centers_inside_changed_region():
    nuclei = np.zeros((16, 16), dtype=np.uint8)
    nuclei[2:5, 2:5] = 1
    nuclei[10:14, 10:14] = 2
    tissue = np.zeros_like(nuclei, dtype=np.int64)
    edit = np.zeros_like(nuclei, dtype=bool)
    edit[:8, :8] = True

    density, counts = build_center_density_targets(nuclei, tissue, edit)

    assert abs(float(density[0].sum()) - 1.0) < 1e-6
    assert density[1].sum() == 0
    assert counts.sum() == 1


def test_erasure_expansion_does_not_change_density_condition_region():
    nuclei = np.zeros((12, 12), dtype=np.uint8)
    nuclei[3:7, 3:7] = 1
    tissue = np.zeros_like(nuclei, dtype=np.int64)
    edit = np.zeros_like(nuclei, dtype=bool)
    edit[6, 6] = True

    erasure = expand_edit_mask_to_complete_instances(nuclei, edit)
    density, counts = build_center_density_targets(nuclei, tissue, edit)

    assert erasure[3:7, 3:7].all()
    assert edit.sum() == 1
    assert density.sum() == 0
    assert counts.sum() == 0


def test_source_centers_prevent_crop_border_fragments_becoming_instances():
    source_nuclei = np.zeros((20, 20), dtype=np.uint8)
    source_nuclei[7:11, 0:4] = 1
    source_nuclei[7:11, 8:12] = 2
    source_centers = extract_class_centers(source_nuclei)

    crop_x = 2
    cropped_nuclei = source_nuclei[:, crop_x:18]
    cropped_tissue = np.zeros_like(cropped_nuclei, dtype=np.int64)
    cropped_edit = np.ones_like(cropped_nuclei, dtype=bool)
    adjusted_centers = [
        (class_id, center_y, center_x - crop_x)
        for class_id, center_y, center_x in source_centers
    ]

    density, counts = build_center_density_targets(
        cropped_nuclei,
        cropped_tissue,
        cropped_edit,
        centers=adjusted_centers,
    )

    assert density[0].sum() == 0
    assert abs(float(density[1].sum()) - 1.0) < 1e-6
    assert counts.sum() == 1


def test_probunet_density_head_is_optional_and_nonnegative():
    torch.manual_seed(3)
    tissue = torch.zeros((1, 32, 32), dtype=torch.long)
    nuclei = torch.zeros((1, 32, 32), dtype=torch.long)
    mask = torch.ones((1, 1, 32, 32))
    cancer = torch.zeros((1,), dtype=torch.long)

    legacy_compatible = ProbUNet(base_ch=8)
    assert legacy_compatible(tissue, nuclei, mask, cancer).shape == (1, 6, 32, 32)
    logits, no_density = legacy_compatible(
        tissue, nuclei, mask, cancer, return_density=True
    )
    assert logits.shape == (1, 6, 32, 32)
    assert no_density is None

    model = ProbUNet(base_ch=8, with_density_head=True)
    logits, density = model(tissue, nuclei, mask, cancer, return_density=True)
    assert logits.shape == (1, 6, 32, 32)
    assert density.shape == (1, 5, 32, 32)
    assert torch.all(density >= 0)


def test_probunet_uses_per_class_density_initialization_bias():
    biases = [-8.0, -9.0, -10.0, -11.0, -12.0]
    model = ProbUNet(
        base_ch=8,
        with_density_head=True,
        density_init_bias=biases,
    )

    torch.testing.assert_close(
        model.density_head[-1].bias.detach(),
        torch.tensor(biases),
    )


def test_center_density_loss_downweights_empty_groups_after_separate_averaging():
    target = torch.zeros((1, 5, 8, 8))
    target[0, 0, 4, 4] = 1.0
    prediction = target.clone()
    prediction[:, 1:] = 0.01
    tissue = torch.zeros((1, 8, 8), dtype=torch.long)
    mask = torch.ones((1, 1, 8, 8))

    _, equal_parts = CenterDensityLoss(empty_group_weight=1.0)(
        prediction, target, tissue, mask
    )
    _, balanced_parts = CenterDensityLoss(empty_group_weight=0.1)(
        prediction, target, tissue, mask
    )

    torch.testing.assert_close(equal_parts['density'], balanced_parts['density'])
    assert balanced_parts['density_empty'].item() == 0.0
    assert balanced_parts['count'] < equal_parts['count']


def test_density_shape_loss_is_invariant_to_mass_and_count_gradient_restores_mass():
    target = torch.zeros((1, 5, 8, 8))
    target[0, 0, 4, 4] = 1.0
    prediction = (target * 0.4).clone().requires_grad_(True)
    tissue = torch.zeros((1, 8, 8), dtype=torch.long)
    mask = torch.ones((1, 1, 8, 8))

    _, parts = CenterDensityLoss(empty_group_weight=0.1)(
        prediction, target, tissue, mask
    )
    (parts['count'] + parts['total_count']).backward()

    assert parts['density'].item() == 0.0
    assert parts['count'].item() > 0.0
    assert parts['total_count'].item() > 0.0
    assert prediction.grad[0, 0, 4, 4].item() < 0.0


def test_center_density_loss_is_finite_and_backpropagates():
    raw_prediction = torch.randn((2, 5, 16, 16), requires_grad=True)
    prediction = torch.nn.functional.softplus(raw_prediction)
    target = torch.zeros_like(prediction)
    target[:, 0, 8, 8] = 1.0
    tissue = torch.zeros((2, 16, 16), dtype=torch.long)
    mask = torch.ones((2, 1, 16, 16))

    total, parts = CenterDensityLoss()(prediction, target, tissue, mask)
    total.backward()

    assert torch.isfinite(total)
    assert torch.isfinite(parts['density'])
    assert torch.isfinite(parts['count'])
    assert raw_prediction.grad is not None
    assert torch.isfinite(raw_prediction.grad).all()


def test_center_density_count_loss_is_zero_for_exact_target():
    target = torch.zeros((1, 5, 12, 12))
    target[0, 2, 6, 6] = 1.0
    tissue = torch.zeros((1, 12, 12), dtype=torch.long)
    mask = torch.ones((1, 1, 12, 12))

    _, parts = CenterDensityLoss()(target.clone(), target, tissue, mask)

    assert parts['density'].item() == 0.0
    assert parts['count'].item() == 0.0


def test_center_density_has_separate_empty_sample_false_positive_term():
    target = torch.zeros((2, 5, 12, 12))
    target[1, 0, 6, 6] = 1.0
    prediction = target.clone()
    prediction[0, 0, 6, 6] = 2.0
    tissue = torch.zeros((2, 12, 12), dtype=torch.long)
    mask = torch.ones((2, 1, 12, 12))

    _, parts = CenterDensityLoss()(prediction, target, tissue, mask)

    assert parts['empty_sample'].item() > 0.0
    assert parts['high_count'].item() == 0.0


def test_high_count_weight_increases_patch_count_emphasis():
    target = torch.zeros((2, 5, 8, 8))
    target[0, 0, 4, 4] = 1.0
    target[1, 0].flatten()[:21] = 1.0
    prediction = target.clone()
    prediction[1] = 0.0
    tissue = torch.zeros((2, 8, 8), dtype=torch.long)
    mask = torch.ones((2, 1, 8, 8))

    _, unweighted = CenterDensityLoss(high_count_weight=1.0)(
        prediction, target, tissue, mask
    )
    _, weighted = CenterDensityLoss(high_count_weight=3.0)(
        prediction, target, tissue, mask
    )

    assert weighted['total_count'] > unweighted['total_count']
    assert weighted['high_count'].item() > 0.0


def test_density_head_only_freezes_every_other_parameter():
    model = ProbUNet(base_ch=8, with_density_head=True)

    names = freeze_non_density_parameters(model)

    assert names
    assert all(name.startswith('density_head.') for name in names)
    assert all(
        parameter.requires_grad == name.startswith('density_head.')
        for name, parameter in model.named_parameters()
    )
