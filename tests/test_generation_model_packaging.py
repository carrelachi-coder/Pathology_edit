from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.package_generation_models import package_cross_pix2pix


class GenerationModelPackagingTests(unittest.TestCase):
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
