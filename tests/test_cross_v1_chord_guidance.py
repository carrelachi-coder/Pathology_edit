import unittest

import torch

from controlnet_train.cli.eval_controlnet_flux_cross_v1 import parse_args
from controlnet_train.inference import pipeline_cross_v1 as pipeline


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
        self.seen_ip_hidden_shape = None

    def __call__(self, *, hidden_states, controlnet_block_samples, joint_attention_kwargs=None, **kwargs):
        self.calls += 1
        if joint_attention_kwargs is not None:
            ip_hidden_states = joint_attention_kwargs.get("ip_hidden_states")
            self.seen_ip_hidden_shape = tuple(ip_hidden_states[0].shape)
        return (controlnet_block_samples[0],)


class _FakePipe:
    def __init__(self):
        self.transformer = _FakeTransformer()


class CrossV1ChordGuidanceTests(unittest.TestCase):
    def test_repeat_joint_attention_kwargs_duplicates_ip_hidden_states(self):
        hidden = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)

        repeated = pipeline._repeat_joint_attention_kwargs(
            {"ip_hidden_states": [hidden]},
            repeats=2,
        )

        self.assertEqual(tuple(repeated["ip_hidden_states"][0].shape), (2, 2, 3))
        self.assertTrue(torch.equal(repeated["ip_hidden_states"][0][0], hidden[0]))
        self.assertTrue(torch.equal(repeated["ip_hidden_states"][0][1], hidden[0]))

    def test_velocity_pair_batches_source_target_and_repeats_ip_tokens(self):
        pipe = _FakePipe()
        controlnet = _FakeControlNet()
        hidden_states = torch.zeros(1, 2, 4)
        source_cond = torch.full((1, 2, 4), 3.0)
        target_cond = torch.full((1, 2, 4), 9.0)
        ip_hidden = torch.zeros(1, 3, 5)

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
            joint_attention_kwargs={"ip_hidden_states": [ip_hidden]},
        )

        self.assertEqual(controlnet.calls, 1)
        self.assertEqual(pipe.transformer.calls, 1)
        self.assertEqual(tuple(controlnet.seen_hidden_shape), (2, 2, 4))
        self.assertEqual(pipe.transformer.seen_ip_hidden_shape, (2, 3, 5))
        self.assertTrue(torch.equal(source_pred, torch.full((1, 2, 4), 3.0)))
        self.assertTrue(torch.equal(target_pred, torch.full((1, 2, 4), 9.0)))


class CrossV1ChordCliTests(unittest.TestCase):
    def test_eval_parser_accepts_chord_guidance_options(self):
        args = parse_args(
            [
                "--pretrained-model-name-or-path",
                "flux",
                "--checkpoint",
                "ckpt",
                "--uni-checkpoint-path",
                "uni.bin",
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
