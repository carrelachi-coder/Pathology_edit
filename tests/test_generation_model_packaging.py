from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.package_generation_models import (
    PROBNET_RELEASE_FILENAME,
    _sha256,
    package_cross_pix2pix,
    package_probnet,
)


class GenerationModelPackagingTests(unittest.TestCase):
    def test_probnet_release_pins_epoch29_and_excludes_density_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "probnet.pt"
            torch.save(
                {
                    "model": {"weight": torch.ones(1)},
                    "epoch": 28,
                    "global_step": 33785,
                    "val_loss": 3.9,
                    "val_metrics": {},
                },
                source,
            )
            args = SimpleNamespace(
                probnet_checkpoint=source,
                output_root=root / "release",
                overwrite=False,
                hf_namespace="test-user",
                git_commit="deadbeef",
            )

            with patch(
                "scripts.package_generation_models.FROZEN_PROBNET_SHA256",
                _sha256(source),
            ):
                output = package_probnet(args)

            self.assertEqual(
                (output / PROBNET_RELEASE_FILENAME).read_bytes(),
                source.read_bytes(),
            )
            self.assertEqual((output / "best.pt").read_bytes(), source.read_bytes())
            self.assertFalse((output / "configs").exists())
            manifest = json.loads((output / "manifest.json").read_text())
            metadata = manifest["model_metadata"]
            self.assertEqual(metadata["epoch_human"], 29)
            self.assertEqual(
                metadata["runtime_role"],
                "per_pixel_placement_and_local_type_evidence_"
                "without_total_count",
            )
            self.assertEqual(
                metadata["candidate_queue_policy"],
                "probnet_odds_mass_without_replacement",
            )
            self.assertEqual(metadata["gamma"], 3.0)
            self.assertEqual(metadata["candidate_diversity_weight"], 0.0)
            self.assertEqual(
                metadata["retry_tail_policy"],
                "same_probnet_mass_permutation_then_component_pixel_backfill_"
                "then_same_tissue_quota_reassignment",
            )
            self.assertEqual(
                metadata["component_quota_reassignment_policy"],
                "unplaceable_component_quota_to_same_tissue_probnet_mass_tail",
            )
            self.assertEqual(
                metadata["failed_reference_shape_policy"],
                "quarantine_within_current_candidate_try_alternative_"
                "reference_or_library_then_restore",
            )
            self.assertEqual(
                metadata["exact_backfill_candidate_policy"],
                "quota_scaled_budget_exhaustion_advances_to_next_"
                "deterministic_seed_without_relaxing_count_type_or_shape",
            )
            self.assertEqual(metadata["exact_backfill_candidates_per_missing"], 128)
            self.assertEqual(metadata["exact_backfill_candidate_floor"], 512)
            self.assertEqual(metadata["exact_backfill_candidate_ceiling"], 4096)
            self.assertEqual(
                metadata["instance_connectivity_policy"],
                "largest_8_connected_component_after_transform",
            )
            self.assertEqual(
                metadata["sampling_audit_policy"],
                "probnet_patch_relative_count_type_spatial_v3",
            )
            self.assertEqual(metadata["sampling_audit_max_attempts"], 3)
            self.assertEqual(
                metadata["sampling_feedback_policy"],
                "reason_directed_gamma_then_seed_v1",
            )
            self.assertEqual(metadata["sampling_feedback_gamma_min"], 1.5)
            self.assertEqual(metadata["sampling_feedback_gamma_max"], 5.0)
            self.assertEqual(
                metadata["nuclei_generation_region_policy"],
                "semantic_region_for_generic_edits_whole_connected_gland_"
                "for_glas_structure_edits",
            )
            self.assertEqual(
                manifest["loading"]["environment_variables"][
                    "PATHOLOGY_PROBNET_CHECKPOINT"
                ],
                f"/models/pathology-probnet/{PROBNET_RELEASE_FILENAME}",
            )

    def test_cross_pix2pix_release_is_inference_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cross = root / "cross"
            cross.mkdir()
            (cross / "config.json").write_text("{}\n", encoding="utf-8")
            (cross / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            torch.save(
                {
                    "hte": {"weight": torch.ones(1)},
                    "tissue_downsampler": {"weight": torch.ones(1)},
                    "nuclei_encoder": {"weight": torch.ones(1)},
                    "ref_encoder_proj_mlp": {"weight": torch.ones(100)},
                },
                cross / "phase5_conditioning.pt",
            )
            pix2pix = root / "source_pix2pix.pt"
            torch.save(
                {
                    "model": {"weight": torch.ones(2)},
                    "optimizer": {"state": torch.ones(4)},
                    "discriminator": {"weight": torch.ones(3)},
                    "d_optimizer": {"state": torch.ones(4)},
                    "epoch": 26,
                    "global_step": 214895,
                    "args": {
                        "base_channels": 64,
                        "cross4_steering_reference_mode": "local_histogram",
                        "metadata": "/private/training/metadata.json",
                    },
                },
                pix2pix,
            )
            args = SimpleNamespace(
                cross_v1_checkpoint=cross,
                pix2pix_checkpoint=pix2pix,
                output_root=root / "release",
                overwrite=False,
                hf_namespace="test-user",
                git_commit="deadbeef",
            )

            output = package_cross_pix2pix(args)

            slim = torch.load(
                output / "pix2pix" / "pix2pix_epoch26_step214895.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(set(slim), {
                "format_version",
                "model",
                "args",
                "epoch",
                "global_step",
                "source_checkpoint_sha256",
                "trust_gate",
            })
            self.assertNotIn("metadata", slim["args"])
            self.assertEqual(slim["trust_gate"], "nuclei_reference_support_v2")
            conditioning = torch.load(
                output / "cross_v1" / "phase5_conditioning.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(set(conditioning), {"hte", "tissue_downsampler", "nuclei_encoder"})
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["private"])
            self.assertEqual(manifest["repo_id"], "test-user/pathology-cross-v1-pix2pix")
            self.assertEqual(
                manifest["loading"]["environment_variables"]["PATHOLOGY_PIX2PIX_CHECKPOINT"],
                "/models/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt",
            )
            self.assertFalse(manifest["dependencies"]["uni_required_for_inference"])


if __name__ == "__main__":
    unittest.main()
