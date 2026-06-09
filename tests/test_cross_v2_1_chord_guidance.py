import unittest

import torch

from controlnet_train.cli.eval_controlnet_flux_cross_v2_1 import parse_args
from controlnet_train.inference import pipeline_cross_v2_1 as pipeline


class _Config:
    guidance_embeds = False


class _FakeControlNet:
    config = _Config()

    def __init__(self):
        self.calls = 0
        self.seen_hidden_shape = None

    def __call__(self, *, hidden_states, controlnet_cond, **kwargs):
        self.calls += 1
        self.seen_hidden_shape = tuple(hidden_states.shape)
        return [controlnet_cond], []


class _FakeTransformer:
    config = _Config()

    def __init__(self):
        self.calls = 0

    def __call__(self, *, hidden_states, controlnet_block_samples, **kwargs):
        self.calls += 1
        return (controlnet_block_samples[0],)


class _FakePipe:
    def __init__(self):
        self.transformer = _FakeTransformer()


class CrossV21ChordGuidanceTests(unittest.TestCase):
    def test_source_init_timesteps_uses_tail_by_strength(self):
        timesteps = torch.tensor([1000, 800, 600, 400, 200, 0])

        selected = pipeline._source_init_timesteps(timesteps, 0.5)

        self.assertTrue(torch.equal(selected, torch.tensor([400, 200, 0])))

    def test_source_init_timesteps_keeps_at_least_one_step(self):
        timesteps = torch.tensor([1000, 800, 600, 400])

        selected = pipeline._source_init_timesteps(timesteps, 0.01)

        self.assertTrue(torch.equal(selected, torch.tensor([400])))

    def test_prepare_source_noised_latents_matches_flow_training_mix(self):
        source = torch.full((1, 2, 3), 0.8)
        noise = torch.full((1, 2, 3), -0.2)
        sigma = torch.tensor([[[0.25]]])

        mixed = pipeline._prepare_source_noised_latents(
            source_latents=source,
            noise_latents=noise,
            sigma=sigma,
        )

        self.assertTrue(torch.allclose(mixed, torch.full((1, 2, 3), 0.55)))

    def test_build_mask_change_map_marks_tissue_or_nuclei_changes(self):
        reference_tissue = torch.tensor([[1, 1], [2, 2]])
        target_tissue = torch.tensor([[1, 3], [2, 2]])
        reference_nuclei = torch.tensor([[0, 0], [1, 1]])
        target_nuclei = torch.tensor([[0, 0], [1, 2]])

        change = pipeline._build_mask_change_map(
            reference_tissue_mask=reference_tissue,
            reference_nuclei_mask=reference_nuclei,
            target_tissue_mask=target_tissue,
            target_nuclei_mask=target_nuclei,
        )

        expected = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
        self.assertTrue(torch.equal(change, expected))

    def test_build_packed_change_gate_expands_channels_before_packing(self):
        gate = pipeline._build_packed_change_gate(
            change_mask=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            latent_height=2,
            latent_width=2,
            packed_channels=8,
            device="cpu",
            dtype=torch.float32,
        )

        expected = torch.tensor([[[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]])
        self.assertEqual(tuple(gate.shape), (1, 1, 8))
        self.assertTrue(torch.equal(gate, expected))

    def test_velocity_pair_batches_source_and_target_forward(self):
        pipe = _FakePipe()
        controlnet = _FakeControlNet()
        hidden_states = torch.zeros(1, 2, 4)
        source_cond = torch.full((1, 2, 4), 3.0)
        target_cond = torch.full((1, 2, 4), 9.0)

        source_pred, target_pred = pipeline._predict_flux_controlnet_velocity_pair(
            pipe=pipe,
            controlnet=controlnet,
            hidden_states=hidden_states,
            source_controlnet_cond=source_cond,
            target_controlnet_cond=target_cond,
            conditioning_scale=1.0,
            timestep=torch.tensor(500.0),
            guidance_scale=3.5,
            pooled_projections=torch.zeros(1, 1),
            encoder_hidden_states=torch.zeros(1, 1, 1),
            txt_ids=torch.zeros(1, 3),
            img_ids=torch.zeros(2, 3),
            controlnet_blocks_repeat=False,
        )

        self.assertEqual(controlnet.calls, 1)
        self.assertEqual(pipe.transformer.calls, 1)
        self.assertEqual(tuple(controlnet.seen_hidden_shape), (2, 2, 4))
        self.assertTrue(torch.equal(source_pred, torch.full((1, 2, 4), 3.0)))
        self.assertTrue(torch.equal(target_pred, torch.full((1, 2, 4), 9.0)))


class CrossV21ChordCliTests(unittest.TestCase):
    def test_eval_parser_accepts_chord_guidance_options(self):
        args = parse_args(
            [
                "--pretrained-model-name-or-path",
                "flux",
                "--checkpoint",
                "ckpt",
                "--metadata",
                "metadata.json",
                "--output-dir",
                "outputs",
                "--source-latent-init-strength",
                "0.35",
                "--mask-chord-scale",
                "1.2",
                "--mask-chord-use-gate",
                "--mask-chord-gate-dilate-radius",
                "1",
                "--mask-chord-gate-feather-radius",
                "2",
                "--mask-chord-gate-outside-scale",
                "0.1",
            ]
        )

        self.assertAlmostEqual(args.source_latent_init_strength, 0.35)
        self.assertAlmostEqual(args.mask_chord_scale, 1.2)
        self.assertTrue(args.mask_chord_use_gate)
        self.assertEqual(args.mask_chord_gate_dilate_radius, 1)
        self.assertEqual(args.mask_chord_gate_feather_radius, 2)
        self.assertAlmostEqual(args.mask_chord_gate_outside_scale, 0.1)


if __name__ == "__main__":
    unittest.main()
