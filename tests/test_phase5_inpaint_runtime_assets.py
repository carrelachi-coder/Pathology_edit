import unittest
from pathlib import Path

import yaml


class Phase5InpaintRuntimeAssetTests(unittest.TestCase):
    def test_environment_yaml_contains_phase5_training_dependencies(self):
        env_path = Path("envs/phase5_controlnet_inpaint.yaml")
        payload = yaml.safe_load(env_path.read_text(encoding="utf8"))

        self.assertEqual(payload["name"], "pathology-phase5-inpaint")
        deps = payload["dependencies"]
        dep_text = "\n".join(str(item) for item in deps)

        self.assertIn("python=3.11", dep_text)
        self.assertIn("pytorch", dep_text)
        self.assertIn("torchvision", dep_text)

        pip_deps = next(item["pip"] for item in deps if isinstance(item, dict) and "pip" in item)
        pip_text = "\n".join(pip_deps)
        self.assertIn("accelerate", pip_text)
        self.assertIn("diffusers", pip_text)
        self.assertIn("transformers", pip_text)
        self.assertIn("bitsandbytes", pip_text)

    def test_training_bash_references_dataset_build_and_inpaint_training(self):
        script_path = Path("scripts/train_phase5_inpaint.sh")
        script = script_path.read_text(encoding="utf8")

        self.assertIn("build_inpaint_dataset.py", script)
        self.assertIn("train_controlnet_flux_inpaint.py", script)
        self.assertIn("--dataset-root", script)
        self.assertIn("--train-metadata", script)
        self.assertIn("accelerate launch", script)


if __name__ == "__main__":
    unittest.main()
