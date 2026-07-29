import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from controlnet_train.pix2pix_transfer.dataset import NUM_CELL_CLASSES, NUM_FINE
from controlnet_train.pix2pix_transfer.identity_adapter import FamilyFeatureFiLM
from controlnet_train.pix2pix_transfer.inference import (
    Pix2PixPostprocessConfig,
    load_pix2pix_postprocessor,
)
from controlnet_train.pix2pix_transfer.regional_cross_attention import (
    Pix2PixCrossAttnUNet,
)
from scripts.generate_cross_v1_no_ip_strict import _run_pix2pix_transfer


class Pix2PixPostprocessInferenceTests(unittest.TestCase):
    def test_strict_cross_uses_checkpoint_nuclei_trust_policy(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return Image.new("RGB", (8, 8)), {
                "epoch": 26,
                "global_step": 214895,
                "use_wsi_identity": True,
                "trust_gate": "nuclei_reference_support_v2",
            }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "pix2pix.pt"
            output = Path(tmp) / "output.png"
            record = {
                "reference_image": "reference.png",
                "target_tissue_mask": "target_tissue.png",
                "target_nuclei_mask": "target_nuclei.png",
                "reference_tissue_mask": "reference_tissue.png",
                "reference_nuclei_mask": "reference_nuclei.png",
            }
            with (
                patch(
                    "controlnet_train.pix2pix_transfer.inference."
                    "load_pix2pix_postprocessor",
                    return_value=object(),
                ),
                patch(
                    "controlnet_train.pix2pix_transfer.inference."
                    "run_pix2pix_postprocess",
                    side_effect=fake_run,
                ),
            ):
                _run_pix2pix_transfer(
                    i0_image=Image.new("RGB", (8, 8)),
                    record=record,
                    checkpoint_path=checkpoint,
                    output_path=output,
                    device="cpu",
                    torch_dtype=torch.float32,
                    image_size=8,
                )

        self.assertNotIn("enable_highres_nuclei_trust", captured)

    def test_identity_adapter_preserves_bfloat16_feature_dtype(self):
        adapter = FamilyFeatureFiLM(8, gamma_max=0.3, gamma_init=0.1).to(
            dtype=torch.bfloat16
        )
        target = torch.randn(1, 8, 8, 8, dtype=torch.bfloat16)
        reference = torch.randn(1, 8, 8, 8, dtype=torch.bfloat16)
        mask = torch.ones(1, 16, 16, dtype=torch.long)

        output, _ = adapter(
            target,
            reference,
            target_mask=mask,
            reference_mask=mask,
            min_pixels=1,
        )

        self.assertEqual(output.dtype, torch.bfloat16)

    def test_epoch25_settings_restore_identity_but_ignore_retired_trust_gate(self):
        args = {
            "base_channels": 8,
            "num_heads": 4,
            "cross_attn_scales": "1/4,1/8,1/16",
            "upsample_mode": "bilinear",
            "region_label_mode": "tissue_nuclei",
            "wsi_identity_adapter": True,
            "identity_gamma_max": 0.3,
            "identity_gamma_init": 0.1,
            "identity_min_tissue_pixels": 16,
            "identity_min_nuclei_pixels": 4,
            "ref_trust_gate": True,
            "ref_fallback_scale": 0.05,
            "ref_soft_context_scale": 0.2,
            "ref_soft_context_radius": 5,
            "matched_tissue_trust_floor": 0.65,
            "matched_nuclei_trust_floor": 0.7,
        }
        config = Pix2PixPostprocessConfig.from_checkpoint_args(args)
        self.assertTrue(config.use_wsi_identity)
        self.assertFalse(hasattr(config, "ref_trust_gate"))
        self.assertFalse(hasattr(config, "ref_fallback_scale"))
        self.assertTrue(config.highres_nuclei_trust_enabled)
        self.assertAlmostEqual(config.highres_nuclei_unmatched_scale, 0.20)
        self.assertAlmostEqual(config.highres_nuclei_matched_floor, 0.60)

        model = Pix2PixCrossAttnUNet(
            in_ch=3 + NUM_FINE + (NUM_CELL_CLASSES + 1),
            base=8,
            num_heads=4,
            use_wsi_identity=True,
            identity_min_tissue_pixels=16,
            identity_min_nuclei_pixels=4,
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "pix2pix_model.pt"
            torch.save(
                {
                    "args": args,
                    "epoch": 24,
                    "global_step": 211895,
                    "model": model.state_dict(),
                },
                checkpoint_path,
            )
            bundle = load_pix2pix_postprocessor(
                checkpoint_path,
                device="cpu",
                torch_dtype=torch.float32,
            )

        self.assertIsNotNone(bundle.model.identity_adapter)
        self.assertEqual(bundle.epoch, 24)
        self.assertEqual(bundle.global_step, 211895)


if __name__ == "__main__":
    unittest.main()
