import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn

    from controlnet_train.inference import pipeline
    from controlnet_train.modules import ChangeMaskEncoder
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    nn = None
    pipeline = None
    ChangeMaskEncoder = None


if nn is not None:

    class FakeFluxControlNetModel(nn.Module):
        loaded_from_config = False

        def __init__(self):
            super().__init__()
            self.controlnet_x_embedder = nn.Linear(64, 4)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("loader should patch x_embedder before loading weights")

        @classmethod
        def load_config(cls, _checkpoint_path):
            return {"fake": True}

        @classmethod
        def from_config(cls, _config):
            cls.loaded_from_config = True
            return cls()
else:

    class FakeFluxControlNetModel:
        loaded_from_config = False


class FakeFluxControlNetPipeline:
    def __init__(self, controlnet):
        self.controlnet = controlnet
        self.device = None
        self.progress_disabled = None

    @classmethod
    def from_pretrained(cls, _pretrained_model_name_or_path, *, controlnet, torch_dtype):
        controlnet._pipeline_dtype = torch_dtype
        return cls(controlnet)

    def to(self, device):
        self.device = device

    def set_progress_bar_config(self, *, disable):
        self.progress_disabled = disable


class Phase5InpaintInferenceLoaderTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is required for the inference loader test")
    def test_load_flux_controlnet_pipeline_patches_embedder_before_loading_weights(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "config.json").write_text("{}", encoding="utf8")
            torch.save(
                {
                    "controlnet_x_embedder.weight": torch.full((4, 400), 2.0),
                    "controlnet_x_embedder.bias": torch.full((4,), 3.0),
                },
                checkpoint / "diffusion_pytorch_model.bin",
            )

            fake_diffusers = types.SimpleNamespace(
                FluxControlNetModel=FakeFluxControlNetModel,
                FluxControlNetPipeline=FakeFluxControlNetPipeline,
            )

            with patch.dict(sys.modules, {"diffusers": fake_diffusers}):
                pipe, controlnet = pipeline._load_flux_controlnet_pipeline(
                    pretrained_model_name_or_path="fake-flux",
                    checkpoint_path=checkpoint,
                    packed_channels=400,
                    device="cpu",
                    torch_dtype=torch.float32,
                )

        self.assertTrue(FakeFluxControlNetModel.loaded_from_config)
        self.assertEqual(controlnet.controlnet_x_embedder.in_features, 400)
        self.assertTrue(torch.all(controlnet.controlnet_x_embedder.weight == 2.0))
        self.assertTrue(torch.all(controlnet.controlnet_x_embedder.bias == 3.0))
        self.assertIs(pipe.controlnet, controlnet)
        self.assertEqual(pipe.device, "cpu")
        self.assertTrue(pipe.progress_disabled)

    @unittest.skipIf(torch is None, "torch is required for the change mask encoder test")
    def test_change_mask_encoder_casts_input_to_module_dtype(self):
        encoder = ChangeMaskEncoder(out_channels=4).to(dtype=torch.float64)
        change_mask = torch.ones(1, 1, 8, 8, dtype=torch.float32)

        output = encoder(change_mask)

        self.assertEqual(output.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
