import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None

_MODULE_PATH = Path(__file__).resolve().parents[1] / "controlnet_train" / "cli" / "run_cross_v1_then_masked_gatys.py"
_SPEC = importlib.util.spec_from_file_location("run_cross_v1_then_masked_gatys", _MODULE_PATH)
pipeline_cli = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pipeline_cli
_SPEC.loader.exec_module(pipeline_cli)


@unittest.skipIf(torch is None, "torch is required for pipeline tests")
class CrossThenGatysPipelineTests(unittest.TestCase):
    def test_parser_accepts_core_inputs(self):
        args = pipeline_cli.parse_args(
            [
                "--pretrained-model-name-or-path",
                "flux",
                "--checkpoint",
                "checkpoint-66000",
                "--uni-checkpoint-path",
                "uni.bin",
                "--reference-image",
                "ref.png",
                "--reference-tissue-mask",
                "ref_t.png",
                "--reference-nuclei-mask",
                "ref_n.png",
                "--target-tissue-mask",
                "t_t.png",
                "--target-nuclei-mask",
                "t_n.png",
                "--output-dir",
                "out",
                "--steps",
                "5",
                "--no-content-loss",
                "--vgg-weights",
                "none",
                "--optimize-background",
            ]
        )

        self.assertEqual(args.checkpoint, "checkpoint-66000")
        self.assertEqual(args.steps, 5)
        self.assertEqual(args.torch_dtype, "bf16")
        self.assertEqual(args.gatys_torch_dtype, "fp32")
        self.assertTrue(args.no_content_loss)
        self.assertTrue(args.optimize_background)

    def test_metadata_reader_and_sampler_work(self):
        payload = {
            "pairs": [
                {"sample_id": "a"},
                {"sample_id": "b"},
                {"sample_id": "c"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text(json.dumps(payload), encoding="utf8")

            records = pipeline_cli.read_cross_metadata(path)
            sampled = pipeline_cli.select_eval_records(records, num_samples=2, seed=7)

        self.assertEqual(len(records), 3)
        self.assertEqual(len(sampled), 2)

    def test_batch_helper_writes_manifest_from_injected_records(self):
        class FakeBundle:
            checkpoint_path = Path("checkpoint-66000")
            pretrained_model_name_or_path = Path("flux")
            num_inference_steps = 28
            guidance_scale = 3.5
            controlnet_conditioning_scale = 1.0
            ip_adapter_scale = 1.0

        record = {
            "dataset": "BCSS",
            "sample_id": "sample_a",
            "reference_sample_id": "sample_b",
            "prompt": "demo",
        }

        def fake_generate_i0(*args, **kwargs):
            return Image.fromarray(np.full((4, 4, 3), 120, dtype=np.uint8), mode="RGB")

        class FakeGatysRunner:
            def run(self, *, initial_image, reference_image, target_mask, reference_mask, regions, output_dir, output_name):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                final = output_dir / output_name
                Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8), mode="RGB").save(final)
                return type(
                    "R",
                    (),
                    {
                        "image": Image.open(final),
                        "history": [{"step": 1.0, "total": 1.0}],
                        "output_path": final,
                        "metrics_path": output_dir / "masked_gatys_metrics.json",
                    },
                )()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["reference_image.png", "reference_tissue_mask.png", "reference_nuclei_mask.png", "target_tissue_mask.png", "target_nuclei_mask.png"]:
                if "image" in name:
                    Image.fromarray(np.full((4, 4, 3), 50, dtype=np.uint8), mode="RGB").save(root / name)
                else:
                    Image.fromarray(np.ones((4, 4), dtype=np.uint8), mode="L").save(root / name)
            record.update(
                {
                    "reference_image": str(root / "reference_image.png"),
                    "reference_tissue_mask": str(root / "reference_tissue_mask.png"),
                    "reference_nuclei_mask": str(root / "reference_nuclei_mask.png"),
                    "target_tissue_mask": str(root / "target_tissue_mask.png"),
                    "target_nuclei_mask": str(root / "target_nuclei_mask.png"),
                }
            )
            out = root / "out"
            result = pipeline_cli.run_cross_v1_then_masked_gatys_batch(
                records=[record],
                bundle=FakeBundle(),
                output_dir=out,
                prompt_override=None,
                generate_i0=fake_generate_i0,
                gatys_runner=FakeGatysRunner(),
                gatys_config=pipeline_cli.GatysTransferConfig(
                    steps=1,
                    optimizer="adam",
                    style_layers=("conv1_1",),
                    layer_weights={"conv1_1": 1.0},
                    device="cpu",
                    vgg_weights="none",
                ),
                regions=(1,),
            )
            manifest = json.loads((out / "batch_manifest.json").read_text(encoding="utf8"))

        self.assertEqual(len(result), 1)
        self.assertEqual(manifest[0]["sample_id"], "sample_a")

    def test_two_stage_pipeline_uses_injected_runners(self):
        class FakeBundle:
            checkpoint_path = Path("checkpoint-66000")
            pretrained_model_name_or_path = Path("flux")
            num_inference_steps = 28
            guidance_scale = 3.5
            controlnet_conditioning_scale = 1.0
            ip_adapter_scale = 1.0

        def fake_generate_i0(*args, **kwargs):
            image = Image.fromarray(np.full((4, 4, 3), 120, dtype=np.uint8), mode="RGB")
            return image

        class FakeGatysRunner:
            def run(self, *, initial_image, reference_image, target_mask, reference_mask, regions, output_dir, output_name):
                self.seen = {
                    "initial_shape": tuple(initial_image.shape),
                    "reference_shape": tuple(reference_image.shape),
                    "target_shape": tuple(target_mask.shape),
                    "regions": regions,
                }
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                final = output_dir / output_name
                Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8), mode="RGB").save(final)
                return type(
                    "R",
                    (),
                    {
                        "image": Image.open(final),
                        "history": [{"step": 1.0, "total": 1.0}],
                        "output_path": final,
                        "metrics_path": output_dir / "masked_gatys_metrics.json",
                    },
                )()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["reference_image.png", "reference_tissue_mask.png", "reference_nuclei_mask.png", "target_tissue_mask.png", "target_nuclei_mask.png"]:
                if "image" in name:
                    Image.fromarray(np.full((4, 4, 3), 50, dtype=np.uint8), mode="RGB").save(root / name)
                else:
                    Image.fromarray(np.ones((4, 4), dtype=np.uint8), mode="L").save(root / name)
            fake_runner = FakeGatysRunner()
            result = pipeline_cli.run_cross_v1_then_masked_gatys(
                bundle=FakeBundle(),
                reference_image_path=root / "reference_image.png",
                reference_tissue_mask_path=root / "reference_tissue_mask.png",
                reference_nuclei_mask_path=root / "reference_nuclei_mask.png",
                target_tissue_mask_path=root / "target_tissue_mask.png",
                target_nuclei_mask_path=root / "target_nuclei_mask.png",
                output_dir=root / "out",
                prompt="test prompt",
                generate_i0=fake_generate_i0,
                gatys_runner=fake_runner,
                i0_output_name="i0.png",
                gatys_output_name="final.png",
                regions=(1,),
            )

            self.assertTrue(result.i0_path.exists())
            self.assertTrue(result.final_path.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertEqual(fake_runner.seen["initial_shape"], (1, 3, 4, 4))
            self.assertEqual(fake_runner.seen["regions"], (1,))


if __name__ == "__main__":
    unittest.main()
