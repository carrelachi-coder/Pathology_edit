import importlib.util
import math
import unittest
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_cross_v2_1_z_ref_activation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "diagnose_cross_v2_1_z_ref_activation",
    _MODULE_PATH,
)
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)


class CrossV21ZRefActivationDiagnosticTests(unittest.TestCase):
    def test_split_x_embedder_keeps_bias_only_in_no_z_path(self):
        x_embedder = torch.nn.Linear(4, 2)
        with torch.no_grad():
            x_embedder.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0, 10.0, 20.0],
                        [-1.0, 0.5, -10.0, 5.0],
                    ]
                )
            )
            x_embedder.bias.copy_(torch.tensor([0.25, -0.75]))
        control = torch.tensor([[[3.0, 4.0, 0.1, -0.2]]])

        full, no_z, z_only = diagnostic.split_controlnet_x_embedder_embedding(
            x_embedder,
            control,
            z_width=2,
        )

        expected_z = torch.nn.functional.linear(control[..., :2], x_embedder.weight[:, :2])
        expected_no_z = torch.nn.functional.linear(
            control[..., 2:],
            x_embedder.weight[:, 2:],
            x_embedder.bias,
        )
        self.assertTrue(torch.allclose(z_only, expected_z))
        self.assertTrue(torch.allclose(no_z, expected_no_z))
        self.assertTrue(torch.allclose(full, x_embedder(control)))
        self.assertTrue(torch.allclose(full - no_z, z_only))

    def test_contribution_metrics_reports_delta_ratio(self):
        full = torch.tensor([3.0, 4.0])
        baseline = torch.tensor([0.0, 4.0])

        metrics = diagnostic.contribution_metrics(full, baseline)

        self.assertAlmostEqual(metrics["full_norm"], 5.0)
        self.assertAlmostEqual(metrics["delta_norm"], 3.0)
        self.assertAlmostEqual(metrics["delta_over_full_norm"], 0.6)
        self.assertAlmostEqual(metrics["cosine_full_baseline"], 0.8)

    def test_pack_latents_places_raw_z_channels_in_packed_prefix(self):
        latents = torch.zeros(1, 3, 2, 2)
        latents[:, 0] = 1.0
        latents[:, 1] = 2.0
        latents[:, 2] = 9.0

        packed = diagnostic.pack_latents(
            latents,
            batch_size=1,
            num_channels_latents=3,
            height=2,
            width=2,
        )

        self.assertEqual(tuple(packed.shape), (1, 1, 12))
        self.assertTrue(torch.equal(packed[..., :8], torch.tensor([[[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]]])))
        self.assertTrue(torch.equal(packed[..., 8:], torch.tensor([[[9.0, 9.0, 9.0, 9.0]]])))

    def test_tensor_list_metrics_handles_empty_outputs(self):
        metrics = diagnostic.tensor_list_contribution_metrics([], [])

        self.assertEqual(metrics["tensor_count"], 0)
        self.assertTrue(math.isnan(metrics["delta_over_full_norm"]))


if __name__ == "__main__":
    unittest.main()
