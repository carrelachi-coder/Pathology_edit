import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from segmentator.config import BaselineConfig, SEGMENTATOR_CLASSES, SampleRecord
from segmentator.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TissueSegmentationDataset,
    build_fine_target,
    dataset_balanced_weights,
    fine_supervision_for_dataset,
    nuclei_mask_to_density,
)
from segmentator.losses import (
    boundary_band_cross_entropy,
    masked_segmentation_loss,
    multi_scale_soft_boundary_loss,
    outside_boundary_consistency_loss,
    segmentation_loss,
    soft_boundary_loss,
)
from segmentator.metrics import fine_segmentation_metrics, fragmentation_metrics, group_macro_iou, segmentation_metrics
from segmentator.model import BoundaryRefinementHead, CellDensityHead, CellPriorEncoder, FineCellTeacherAdapter, HierarchicalFineHead, SimpleFeaturePyramid, UPerLikeDecoder, Uni2hFeatureEncoder
from segmentator.training import (
    _checkpoint_selection_score,
    _fine_dataset_macro,
    _freeze_for_trainable_scope,
    _freeze_shared_for_fine,
    _majority_child_miou,
    _wait_for_free_gpu_memory_before_unfreeze,
    compute_class_weights,
    compute_fine_class_weights,
    boundary_aware_sampling_weights,
    fine_supervision_sampling_weights,
)
from segmentator.cli import build_parser
from segmentator.stain_augmentation import StainAugmentationConfig, build_stain_augmenter
from scripts.build_segmentator_multidataset_manifest import _assert_group_disjoint, _group_id, _split_records


class SegmentatorDataTests(unittest.TestCase):
    def test_segmentator_classes_follow_unified_coarse_label_ids(self):
        self.assertEqual(SEGMENTATOR_CLASSES[3], "necrosis")
        self.assertEqual(SEGMENTATOR_CLASSES[4], "immune_infiltrate")
        self.assertEqual(SEGMENTATOR_CLASSES[5], "normal_epithelium")

    def test_dataset_applies_imagenet_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            mask_path = root / "sample_mask.png"
            Image.new("RGB", (4, 4), (255, 255, 255)).save(image_path)
            Image.new("L", (4, 4), 1).save(mask_path)

            dataset = TissueSegmentationDataset(
                [SampleRecord(image_path=image_path, mask_path=mask_path, sample_id="sample")],
                image_size=4,
            )

            item = dataset[0]
            image = item["image"]

            self.assertIsInstance(image, torch.Tensor)
            self.assertTrue(torch.allclose(image[:, 0, 0], torch.tensor([(1.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])))
            self.assertFalse(math.isclose(float(image.max()), 1.0))

    def test_stain_augmentation_disabled_does_not_require_optional_dependencies(self):
        augmenter = build_stain_augmenter(
            StainAugmentationConfig(mode="randstainna", probability=0.0)
        )

        self.assertIsNone(augmenter)

    def test_segmentator_cli_accepts_randstainna_arguments(self):
        args = build_parser().parse_args(
            [
                "--dataset-root",
                "data",
                "--stain-augmentation",
                "randstainna",
                "--stain-augmentation-prob",
                "0.7",
                "--randstainna-root",
                "third_party/RandStainNA",
                "--randstainna-std-hyper",
                "-0.3",
                "--randstainna-distribution",
                "normal",
            ]
        )

        self.assertEqual(args.stain_augmentation, "randstainna")
        self.assertEqual(args.stain_augmentation_prob, 0.7)
        self.assertEqual(args.randstainna_std_hyper, -0.3)
        self.assertEqual(args.randstainna_distribution, "normal")

    def test_segmentator_cli_accepts_gpu_memory_gate_arguments(self):
        args = build_parser().parse_args(
            [
                "--dataset-root",
                "data",
                "--min-free-gpu-memory-gb-before-unfreeze",
                "24",
                "--gpu-memory-poll-seconds",
                "30",
            ]
        )

        self.assertEqual(args.min_free_gpu_memory_gb_before_unfreeze, 24.0)
        self.assertEqual(args.gpu_memory_poll_seconds, 30.0)

    def test_segmentator_cli_accepts_rank_zero_validation_arguments(self):
        args = build_parser().parse_args(
            [
                "--dataset-root",
                "data",
                "--ddp-timeout-seconds",
                "7200",
                "--rank-zero-validation",
            ]
        )

        self.assertEqual(args.ddp_timeout_seconds, 7200.0)
        self.assertTrue(args.rank_zero_validation)

    def test_segmentator_cli_accepts_boundary_v2_arguments(self):
        args = build_parser().parse_args(
            [
                "--dataset-root",
                "data",
                "--boundary-refinement",
                "--boundary-aware-sampling",
                "--refinement-boundary-widths",
                "2",
                "4",
                "8",
                "--refinement-boundary-ce-weight",
                "1.0",
                "--refinement-consistency-weight",
                "2.0",
                "--refinement-gate-width",
                "4",
                "--refinement-gate-threshold",
                "0.15",
            ]
        )

        self.assertTrue(args.boundary_aware_sampling)
        self.assertEqual(args.refinement_boundary_widths, [2, 4, 8])
        self.assertEqual(args.refinement_boundary_ce_weight, 1.0)
        self.assertEqual(args.refinement_consistency_weight, 2.0)
        self.assertEqual(args.refinement_gate_width, 4)
        self.assertEqual(args.refinement_gate_threshold, 0.15)

    def test_segmentator_cli_accepts_coarse_preserving_fine_probe_arguments(self):
        args = build_parser().parse_args(
            [
                "--dataset-root",
                "data",
                "--hierarchical-fine",
                "--fine-class-weighting",
                "inverse_sqrt",
                "--fine-class-weight-min",
                "0.75",
                "--fine-class-weight-max",
                "2.0",
                "--fine-supervision-sampling",
                "--fine-sampling-rare-class-boost",
                "4.0",
                "--fine-sampling-min-valid-pixels",
                "1",
                "--freeze-shared-for-fine",
                "--fine-only-loss",
                "--checkpoint-mode",
                "fine_dataset_macro",
            ]
        )

        self.assertEqual(args.fine_class_weighting, "inverse_sqrt")
        self.assertEqual(args.fine_class_weight_min, 0.75)
        self.assertEqual(args.fine_class_weight_max, 2.0)
        self.assertTrue(args.fine_supervision_sampling)
        self.assertEqual(args.fine_sampling_rare_class_boost, 4.0)
        self.assertEqual(args.fine_sampling_min_valid_pixels, 1)
        self.assertTrue(args.freeze_shared_for_fine)
        self.assertTrue(args.fine_only_loss)
        self.assertEqual(args.checkpoint_mode, "fine_dataset_macro")

    def test_fine_class_weights_use_independent_clamp(self):
        summary = {
            "panda": {"raw_values": {"8": 1000000, "9": 1000, "10": 1}},
        }

        weights, supported_ids, metadata = compute_fine_class_weights(
            summary,
            "inverse_sqrt",
            min_weight=0.75,
            max_weight=2.0,
        )

        self.assertTrue({8, 9, 10}.issubset(supported_ids))
        self.assertGreaterEqual(float(weights[[8, 9, 10]].min()), 0.75)
        self.assertLessEqual(float(weights[[8, 9, 10]].max()), 2.0)
        self.assertEqual(metadata["min_weight"], 0.75)
        self.assertEqual(metadata["max_weight"], 2.0)

    def test_fine_sampling_excludes_coarse_only_and_caps_rare_class_boost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            masks = {
                "bcss_generic": ("bcss", torch.ones((8, 8), dtype=torch.uint8)),
                "bcss_angio": ("bcss", torch.tensor([[15] + [1] * 7] + [[1] * 8] * 7, dtype=torch.uint8)),
                "glas_tumor": ("glas", torch.full((8, 8), 11, dtype=torch.uint8)),
                "puma_coarse": ("puma", torch.ones((8, 8), dtype=torch.uint8)),
            }
            for sample_id, (dataset_id, mask) in masks.items():
                image_path = root / f"{sample_id}.png"
                mask_path = root / f"{sample_id}_mask.png"
                Image.new("RGB", (8, 8), (255, 255, 255)).save(image_path)
                Image.fromarray(mask.numpy(), mode="L").save(mask_path)
                records.append(
                    SampleRecord(
                        image_path=image_path,
                        mask_path=mask_path,
                        sample_id=sample_id,
                        dataset_id=dataset_id,
                    )
                )
            dataset = TissueSegmentationDataset(records, image_size=8, hierarchical_fine=True)

            weights, metadata = fine_supervision_sampling_weights(
                dataset,
                temperature=0.0,
                rare_class_boost=4.0,
                min_valid_pixels=1,
            )

        self.assertEqual(metadata["eligible_records"], 3)
        self.assertEqual(metadata["excluded_no_branch"], {"puma": 1})
        self.assertEqual(float(weights[3]), 0.0)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))
        self.assertEqual(metadata["datasets"]["bcss"]["class_presence_multipliers"][15], 4.0)
        self.assertEqual(metadata["datasets"]["bcss"]["class_presence_samples"][15], 1)

    def test_boundary_sampling_upweights_boundary_rich_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            masks = {
                "flat": torch.ones((16, 16), dtype=torch.uint8),
                "split": torch.cat(
                    [
                        torch.ones((16, 8), dtype=torch.uint8),
                        torch.full((16, 8), 2, dtype=torch.uint8),
                    ],
                    dim=1,
                ),
            }
            for sample_id, mask in masks.items():
                image_path = root / f"{sample_id}.png"
                mask_path = root / f"{sample_id}_mask.png"
                Image.new("RGB", (16, 16), (255, 255, 255)).save(image_path)
                Image.fromarray(mask.numpy(), mode="L").save(mask_path)
                records.append(
                    SampleRecord(
                        image_path=image_path,
                        mask_path=mask_path,
                        sample_id=sample_id,
                        dataset_id="test",
                    )
                )
            dataset = TissueSegmentationDataset(records, image_size=16)

            weights, metadata = boundary_aware_sampling_weights(
                dataset,
                boost=3.0,
                min_boundary_pixels=32,
                width=1,
            )

        self.assertEqual(float(weights[0]), 1.0)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertEqual(metadata["datasets"]["test"]["rich_samples"], 1)

    def test_fine_dataset_macro_excludes_coarse_only_datasets(self):
        value = _fine_dataset_macro(
            {
                "bcss": {"fine": {"available": True, "mIoU": 0.2}},
                "panda": {"fine": {"available": True, "mIoU": 0.4}},
                "orca": {"fine": {"available": False, "mIoU": float("nan")}},
            }
        )

        self.assertAlmostEqual(value, 0.3)

    def test_majority_child_baseline_uses_only_valid_fine_pixels(self):
        target = torch.tensor([[[8, 8, 9, 255]]], dtype=torch.long)

        value = _majority_child_miou(target)

        self.assertAlmostEqual(value, (2.0 / 3.0) / 2.0)

    def test_fine_checkpoint_selection_enforces_coarse_floor(self):
        config = BaselineConfig(
            hierarchical_fine=True,
            checkpoint_mode="fine_dataset_macro",
            checkpoint_coarse_miou_floor=0.611,
            checkpoint_coarse_boundary_f1_4_floor=0.516,
        )
        metrics = {
            "mIoU": 0.60,
            "boundary_f1_4": 0.53,
            "fine_dataset_macro_mIoU": 0.4,
        }

        score, eligible = _checkpoint_selection_score(metrics, config)

        self.assertFalse(eligible)
        self.assertEqual(score, float("-inf"))

    def test_gpu_memory_gate_waits_until_threshold_is_met(self):
        gib = 1024**3
        with (
            patch("segmentator.training.torch.cuda.empty_cache") as empty_cache,
            patch(
                "segmentator.training.torch.cuda.mem_get_info",
                side_effect=[(10 * gib, 80 * gib), (30 * gib, 80 * gib)],
            ) as mem_get_info,
            patch("segmentator.training.time.sleep") as sleep,
        ):
            _wait_for_free_gpu_memory_before_unfreeze(
                torch.device("cuda:0"),
                24.0,
                30.0,
                main_process=False,
            )

        self.assertEqual(empty_cache.call_count, 2)
        self.assertEqual(mem_get_info.call_count, 2)
        sleep.assert_called_once_with(30.0)

    def test_gpu_memory_gate_resolves_indexless_cuda_device(self):
        gib = 1024**3
        with (
            patch("segmentator.training.torch.cuda.empty_cache"),
            patch("segmentator.training.torch.cuda.current_device", return_value=2),
            patch("segmentator.training.torch.cuda.mem_get_info", return_value=(30 * gib, 80 * gib)) as mem_get_info,
        ):
            _wait_for_free_gpu_memory_before_unfreeze(
                torch.device("cuda"),
                24.0,
                30.0,
                main_process=False,
            )

        mem_get_info.assert_called_once_with(torch.device("cuda:2"))

    def test_dataset_remaps_fine_labels_to_coarse_and_ignores_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            mask_path = root / "sample_mask.png"
            Image.new("RGB", (3, 1), (255, 255, 255)).save(image_path)
            Image.fromarray(torch.tensor([[8, 14, 99]], dtype=torch.uint8).numpy(), mode="L").save(mask_path)

            dataset = TissueSegmentationDataset(
                [SampleRecord(image_path=image_path, mask_path=mask_path, sample_id="sample", dataset_id="panda")],
                image_size=3,
                ignore_index=255,
            )

            mask = dataset[0]["mask"]

            self.assertEqual(mask.tolist(), [[1, 1, 255]] * 3)

    def test_fine_target_only_supervises_dataset_specific_branching_parents(self):
        panda_mask = torch.tensor([[0, 2, 5, 8, 9, 10]])
        panda_target, panda_allowed = build_fine_target(panda_mask, "panda")
        puma_target, puma_allowed = build_fine_target(torch.tensor([[0, 1, 2, 3]]), "puma")

        self.assertEqual(panda_target.tolist(), [[255, 255, 255, 8, 9, 10]])
        self.assertEqual(torch.where(panda_allowed[1])[0].tolist(), [8, 9, 10])
        self.assertTrue(torch.all(puma_target == 255))
        self.assertEqual(torch.where(puma_allowed[1])[0].tolist(), [1])

    def test_bcss_fine_branch_preserves_generic_tumor_and_subtypes(self):
        allowed = fine_supervision_for_dataset("bcss")
        target, _ = build_fine_target(torch.tensor([[1, 14, 15, 2]]), "bcss")

        self.assertEqual(torch.where(allowed[1])[0].tolist(), [1, 14, 15])
        self.assertEqual(target.tolist(), [[1, 14, 15, 255]])

    def test_dataset_skips_unreadable_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_image_path = root / "bad.png"
            bad_mask_path = root / "bad_mask.png"
            good_image_path = root / "good.png"
            good_mask_path = root / "good_mask.png"
            bad_image_path.write_bytes(b"not-a-real-png")
            Image.new("L", (4, 4), 1).save(bad_mask_path)
            Image.new("RGB", (4, 4), (255, 255, 255)).save(good_image_path)
            Image.new("L", (4, 4), 2).save(good_mask_path)
            dataset = TissueSegmentationDataset(
                [
                    SampleRecord(bad_image_path, bad_mask_path, "bad"),
                    SampleRecord(good_image_path, good_mask_path, "good"),
                ],
                image_size=4,
            )

            with self.assertWarns(RuntimeWarning):
                item = dataset[0]

            self.assertEqual(item["sample_id"], "good")
            self.assertEqual(tuple(item["image"].shape), (3, 4, 4))

    def test_class_weights_skip_unreadable_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            bad_mask_path = root / "bad_mask.png"
            good_mask_path = root / "good_mask.png"
            Image.new("RGB", (2, 2), (255, 255, 255)).save(image_path)
            bad_mask_path.write_bytes(b"not-a-real-png")
            Image.new("L", (2, 2), 1).save(good_mask_path)
            dataset = TissueSegmentationDataset(
                [
                    SampleRecord(image_path, bad_mask_path, "bad"),
                    SampleRecord(image_path, good_mask_path, "good"),
                ],
                image_size=2,
                num_classes=3,
                mask_remap="coarse",
            )

            with self.assertWarns(RuntimeWarning):
                _, metadata = compute_class_weights(dataset, num_classes=3, mode="inverse_sqrt", remap_invalid_to=255)

            self.assertEqual(metadata["pixel_counts"], [0, 4, 0])
            self.assertEqual(metadata["skipped_unreadable_samples"], ["bad"])

    def test_losses_and_metrics_ignore_partial_label_pixels(self):
        logits = torch.zeros(1, 2, 1, 2)
        target = torch.tensor([[[1, 255]]])

        losses = segmentation_loss(logits, target, num_classes=2, invalid_to=255)
        metrics = segmentation_metrics(torch.tensor([[[1, 0]]]), target, num_classes=2, ignore_index=255)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(metrics["per_class"]["class_1"]["support_pixels"], 1)

    def test_dataset_balanced_weights_equalize_dataset_sampling(self):
        root = Path("unused")
        records = [
            SampleRecord(root / "a.png", root / "a.png", "a", dataset_id="big"),
            SampleRecord(root / "b.png", root / "b.png", "b", dataset_id="big"),
            SampleRecord(root / "c.png", root / "c.png", "c", dataset_id="small"),
        ]

        weights = dataset_balanced_weights(records)

        self.assertEqual(weights.tolist(), [0.5, 0.5, 1.0])

    def test_dataset_temperature_interpolates_between_balanced_and_natural_sampling(self):
        root = Path("unused")
        records = [
            SampleRecord(root / "a.png", root / "a.png", "a", dataset_id="big"),
            SampleRecord(root / "b.png", root / "b.png", "b", dataset_id="big"),
            SampleRecord(root / "c.png", root / "c.png", "c", dataset_id="small"),
        ]

        weights = dataset_balanced_weights(records, temperature=0.5)

        self.assertAlmostEqual(weights[0].item(), 2**-0.5)
        self.assertEqual(weights[2].item(), 1.0)

    def test_cellvit_density_maps_are_smooth_bounded_and_six_channel(self):
        mask = torch.zeros(17, 17, dtype=torch.long)
        mask[8, 8] = 101
        mask[4, 4] = 102

        density = nuclei_mask_to_density(mask, sigma=2.0)

        self.assertEqual(tuple(density.shape), (6, 17, 17))
        self.assertGreater(float(density[0, 8, 8]), float(density[0, 0, 0]))
        self.assertGreaterEqual(float(density.min()), 0.0)
        self.assertLessEqual(float(density.max()), 1.0)

    def test_cellvit_density_uses_component_centers_instead_of_nucleus_area(self):
        mask = torch.zeros(17, 17, dtype=torch.long)
        mask[5:12, 5:12] = 101

        density = nuclei_mask_to_density(mask, sigma=1.0)

        self.assertAlmostEqual(float(density[0, 8, 8]), 1.0, places=5)
        self.assertLess(float(density[0, 5, 5]), float(density[0, 8, 8]))

    def test_group_split_has_no_cross_partition_leakage(self):
        records = [
            {"group_id": f"case_{case}", "sample_id": f"{case}_{patch}"}
            for case in range(12)
            for patch in range(3)
        ]

        train, val, test = _split_records(records, 0.2, 0.2, __import__("random").Random(7))

        _assert_group_disjoint(train, val, test)
        self.assertEqual(len(train) + len(val) + len(test), len(records))

    def test_dataset_group_ids_collapse_patient_and_wsi_subregions(self):
        self.assertEqual(_group_id("ignite", "patient5_he_roi6_py256_px0.png"), "patient5")
        self.assertEqual(
            _group_id("orca", "TCGA-AB-1234-01Z-00-DX1.UUID_1_py256_px0.png"),
            "TCGA-AB-1234-01Z-00-DX1.UUID",
        )


class _FakeUniBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.requested_n = None

    def get_intermediate_layers(self, x, *, n, reshape):
        self.requested_n = n
        self.requested_reshape = reshape
        return [torch.zeros(x.shape[0], 1536, 2, 2) for _ in n]


class SegmentatorModelTests(unittest.TestCase):
    def test_freeze_shared_for_fine_leaves_only_fine_head_trainable(self):
        model = nn.Module()
        model.encoder = nn.Linear(4, 4)
        model.decoder = nn.Linear(4, 3)
        model.fine_head = nn.Linear(4, 2)

        fine_parameters = _freeze_shared_for_fine(model)

        self.assertTrue(all(parameter.requires_grad for parameter in fine_parameters))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.decoder.parameters()))
        self.assertTrue(model.encoder.freeze)
        self.assertEqual(model.encoder.trainable_block_count, 0)

    def test_teacher_scope_trains_only_fine_adapter_and_density_modules(self):
        model = nn.Module()
        model.encoder = nn.Linear(4, 4)
        model.decoder = nn.Linear(4, 3)
        model.fine_head = nn.Linear(4, 2)
        model.cell_teacher_adapter = nn.Linear(4, 4)
        model.cell_density_head = nn.Linear(4, 6)

        selected = _freeze_for_trainable_scope(model, "teacher")

        self.assertTrue(selected)
        self.assertTrue(all(parameter.requires_grad for parameter in selected))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.decoder.parameters()))

    def test_cell_teacher_adapter_is_identity_at_initialization(self):
        adapter = FineCellTeacherAdapter(channels=32, hidden_channels=16)
        features = torch.randn(2, 32, 8, 8)

        with torch.no_grad():
            adapted = adapter(features)

        self.assertTrue(torch.equal(adapted, features))

    def test_hierarchical_fine_head_outputs_full_unified_label_space(self):
        head = HierarchicalFineHead(in_channels=32, num_subtypes=16)

        logits = head(torch.randn(2, 32, 8, 8), (32, 32))

        self.assertEqual(tuple(logits.shape), (2, 16, 32, 32))

    def test_masked_fine_loss_is_zero_for_coarse_only_batch(self):
        logits = torch.full(
            (2, 16, 4, 4),
            torch.finfo(torch.float32).min,
            requires_grad=True,
        )
        target = torch.full((2, 4, 4), 255, dtype=torch.long)

        losses = masked_segmentation_loss(logits, target, num_classes=16)
        losses["total"].backward()

        self.assertEqual(float(losses["total"]), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_fine_metrics_ignore_non_branching_pixels(self):
        target = torch.tensor([[[255, 8, 9, 10]]])
        pred = torch.tensor([[[0, 8, 9, 9]]])

        metrics = fine_segmentation_metrics(pred, target, num_classes=16)

        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["valid_pixels"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_uni2h_encoder_uses_spaced_intermediate_layers(self):
        fake_backbone = _FakeUniBackbone()
        with patch("segmentator.model._load_uni2h_model", return_value=fake_backbone):
            encoder = Uni2hFeatureEncoder(local_repo="unused")

        features = encoder(torch.zeros(1, 3, 28, 28))

        self.assertEqual(fake_backbone.requested_n, [5, 11, 17, 23])
        self.assertTrue(fake_backbone.requested_reshape)
        self.assertEqual(len(features), 4)

    def test_uni2h_encoder_can_request_single_intermediate_layer(self):
        fake_backbone = _FakeUniBackbone()
        with patch("segmentator.model._load_uni2h_model", return_value=fake_backbone):
            encoder = Uni2hFeatureEncoder(local_repo="unused", intermediate_layers=(23,))

        features = encoder(torch.zeros(1, 3, 28, 28))

        self.assertEqual(fake_backbone.requested_n, [23])
        self.assertEqual(len(features), 1)

    def test_uper_decoder_normalizes_channel_last_features(self):
        decoder = UPerLikeDecoder((4, 4, 4, 4), num_classes=3)
        decoder.eval()
        feats = [
            torch.randn(2, 4, 8, 8) * 0.1,
            torch.randn(2, 4, 4, 4) * 1.0,
            torch.randn(2, 4, 2, 2) * 10.0,
            torch.randn(2, 4, 1, 1) * 100.0,
        ]

        with torch.no_grad():
            logits = decoder(feats)

        self.assertEqual(tuple(logits.shape), (2, 3, 8, 8))

    def test_simple_feature_pyramid_builds_patch14_compatible_features_from_distinct_depths(self):
        pyramid = SimpleFeaturePyramid(in_channels=32, out_channels=32)
        pyramid.eval()
        feats = [torch.full((2, 32, 16, 16), float(idx + 1)) for idx in range(4)]

        with torch.no_grad():
            outputs = pyramid(feats)

        self.assertEqual(pyramid.strides, (7, 14, 28, 56))
        self.assertEqual([tuple(x.shape[-2:]) for x in outputs], [(32, 32), (16, 16), (8, 8), (4, 4)])
        self.assertEqual([x.shape[1] for x in outputs], [32, 32, 32, 32])

    def test_simple_feature_pyramid_rejects_single_feature_map(self):
        pyramid = SimpleFeaturePyramid(in_channels=32, out_channels=32)

        with self.assertRaisesRegex(RuntimeError, "requires 4 feature maps"):
            pyramid([torch.randn(2, 32, 16, 16)])

    def test_boundary_refinement_and_cell_heads_preserve_expected_shapes(self):
        refinement = BoundaryRefinementHead(num_classes=8)
        prior = CellPriorEncoder(out_channels=256)
        teacher = CellDensityHead(in_channels=256)
        image = torch.randn(2, 3, 32, 32)
        logits = torch.randn(2, 8, 32, 32)
        density = torch.rand(2, 6, 32, 32)
        features = torch.randn(2, 256, 8, 8)

        refined, gate = refinement(image, logits, return_gate=True)
        self.assertEqual(tuple(refined.shape), tuple(logits.shape))
        self.assertEqual(tuple(gate.shape), (2, 1, 32, 32))
        self.assertTrue(torch.equal(refined, logits))
        self.assertGreaterEqual(float(gate.min()), 0.0)
        self.assertLessEqual(float(gate.max()), 1.0)
        self.assertEqual(tuple(prior(density, (8, 8)).shape), tuple(features.shape))
        self.assertEqual(tuple(teacher(features, (32, 32)).shape), tuple(density.shape))

    def test_boundary_v2_losses_are_finite_and_identity_consistent(self):
        logits = torch.randn(1, 3, 16, 16)
        refined = logits.clone().requires_grad_(True)
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        target[:, :, 8:] = 1
        gate = torch.zeros(1, 1, 16, 16)

        boundary = multi_scale_soft_boundary_loss(logits, target, num_classes=3, widths=(2, 4))
        boundary_ce = boundary_band_cross_entropy(logits, target, num_classes=3, width=2)
        consistency = outside_boundary_consistency_loss(refined, logits, gate, target)

        self.assertTrue(torch.isfinite(boundary))
        self.assertTrue(torch.isfinite(boundary_ce))
        self.assertAlmostEqual(float(consistency), 0.0, places=6)

    def test_boundary_loss_and_fragmentation_metrics_detect_small_islands(self):
        logits = torch.zeros(1, 3, 8, 8)
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        target[:, 2:6, 2:6] = 1
        prediction = target.clone()
        prediction[:, 0, 0] = 2

        loss = soft_boundary_loss(logits, target, num_classes=3)
        fragmentation = fragmentation_metrics(prediction, num_classes=3)
        macro = group_macro_iou(prediction, target, ["case-a"], num_classes=3)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(fragmentation["overall"]["components_lt_16"], 0)
        self.assertEqual(macro["groups"], 1)


if __name__ == "__main__":
    unittest.main()
