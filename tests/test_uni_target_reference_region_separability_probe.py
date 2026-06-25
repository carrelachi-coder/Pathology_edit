import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


_PROBE = None
_IMPORT_ERROR = None
if torch is not None:
    try:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "probe_uni_target_reference_region_separability.py"
        )
        spec = importlib.util.spec_from_file_location(
            "probe_uni_target_reference_region_separability",
            script_path,
        )
        _PROBE = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = _PROBE
        spec.loader.exec_module(_PROBE)
    except ModuleNotFoundError as exc:
        _IMPORT_ERROR = exc


@unittest.skipIf(torch is None, "torch is required for UNI target/reference probe tests")
class UniTargetReferenceRegionSeparabilityProbeTests(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"probe dependencies unavailable: {_IMPORT_ERROR}")

    def test_build_pair_outputs_reports_same_label_closer_than_cross_and_unpaired(self):
        paired = _pair_entry("target_0", "ref_0")
        unpaired = _pair_entry("other_target", "ref_1")
        descriptors = {
            (paired.target_key, 1): _descriptor(
                key=paired.target_key,
                role="target",
                sample_id="target_0",
                label_id=1,
                mean=[1.00, 0.00, 0.00],
                std=[0.10, 0.10, 0.10],
            ),
            (paired.reference_key, 1): _descriptor(
                key=paired.reference_key,
                role="reference",
                sample_id="ref_0",
                label_id=1,
                mean=[0.98, 0.02, 0.00],
                std=[0.11, 0.10, 0.09],
            ),
            (paired.reference_key, 2): _descriptor(
                key=paired.reference_key,
                role="reference",
                sample_id="ref_0",
                label_id=2,
                mean=[0.00, 1.00, 0.00],
                std=[0.50, 0.45, 0.40],
            ),
            (unpaired.reference_key, 1): _descriptor(
                key=unpaired.reference_key,
                role="reference",
                sample_id="ref_1",
                label_id=1,
                mean=[0.20, 0.80, 0.00],
                std=[0.35, 0.35, 0.35],
            ),
        }

        pair_rows, summary = _PROBE.build_pair_outputs(
            pair_entries=[paired, unpaired],
            descriptors=descriptors,
            label_ids=[1, 2],
            rng=__import__("random").Random(7),
            unpaired_references_per_target=4,
            max_unpaired_pairs=20,
            mean_weight=1.0,
            std_weight=0.5,
            pooled_cosine_weight=0.25,
        )

        self.assertEqual(summary["counts"]["paired_same_label_pairs"], 1)
        self.assertEqual(summary["counts"]["paired_cross_label_pairs"], 1)
        self.assertEqual(summary["counts"]["unpaired_same_label_pairs"], 1)
        self.assertEqual(
            summary["comparisons"]["region_loss_style_distance"][
                "paired_cross_greater_than_paired_same_probability"
            ],
            1.0,
        )
        self.assertEqual(
            summary["comparisons"]["region_loss_style_distance"][
                "unpaired_same_greater_than_paired_same_probability"
            ],
            1.0,
        )
        self.assertIn("decode/UNI path is viable", summary["target_reference_uni_verdict"]["reading"])
        self.assertTrue(all("region_loss_style_distance" in row for row in pair_rows))

    def test_build_pair_entries_reads_cross_metadata_schema(self):
        records = [
            {
                "dataset": "BCSS",
                "sample_id": "target_0",
                "reference_sample_id": "ref_0",
                "target_image": "images/target_0.png",
                "target_tissue_mask": "tissue_masks/target_0.png",
                "reference_image": "images/ref_0.png",
                "reference_tissue_mask": "tissue_masks/ref_0.png",
            }
        ]

        entries = _PROBE.build_pair_entries(records, base_dir=Path("/data/root"))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].sample_id, "target_0")
        self.assertEqual(entries[0].reference_sample_id, "ref_0")
        self.assertEqual(entries[0].target_image_path, Path("/data/root/images/target_0.png"))
        self.assertEqual(entries[0].reference_tissue_mask_path, Path("/data/root/tissue_masks/ref_0.png"))

    def test_target_and_reference_descriptor_keys_do_not_collide_for_same_paths(self):
        same_image = Path("/tmp/shared.png")
        same_mask = Path("/tmp/shared_mask.png")
        pair = _PROBE.PairEntry(
            index=0,
            dataset="BCSS",
            sample_id="target_as_shared",
            reference_sample_id="reference_as_shared",
            target_image_path=same_image,
            target_tissue_mask_path=same_mask,
            reference_image_path=same_image,
            reference_tissue_mask_path=same_mask,
        )

        image_entries = _PROBE.build_image_entries([pair])

        self.assertNotEqual(pair.target_key, pair.reference_key)
        self.assertEqual(len(image_entries), 2)
        self.assertEqual({entry.role for entry in image_entries}, {"target", "reference"})

    def test_collect_descriptors_applies_vae_roundtrip_to_targets_only(self):
        target = _PROBE.ImageEntry(
            key="target::/tmp/t.png::/tmp/t_mask.png",
            role="target",
            dataset="BCSS",
            sample_id="target_0",
            image_path=Path("/tmp/t.png"),
            tissue_mask_path=Path("/tmp/t_mask.png"),
        )
        reference = _PROBE.ImageEntry(
            key="reference::/tmp/r.png::/tmp/r_mask.png",
            role="reference",
            dataset="BCSS",
            sample_id="ref_0",
            image_path=Path("/tmp/r.png"),
            tissue_mask_path=Path("/tmp/r_mask.png"),
        )

        descriptors, skipped = _PROBE.collect_descriptors(
            [target, reference],
            label_ids=[1],
            label_lookup={"tumor": 1},
            label_mode="fine",
            fine_to_parent={1: 1},
            encoder=_MeanEncoder(),
            target_vae_roundtrip=True,
            vae=_OffsetVae(),
            vae_dtype=torch.float32,
            load_image_tensor=lambda _path: torch.full((3, 4, 4), 0.25),
            load_tissue_mask=lambda _path: torch.ones((4, 4), dtype=torch.long),
            resize_mask_to_token_labels=lambda masks, tokens: torch.ones(
                (masks.shape[0], tokens), dtype=torch.long
            ),
            samples_per_label=1,
            min_region_tokens=1,
            min_region_fraction=0.0,
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            progress_every=0,
        )

        self.assertEqual(skipped, [])
        self.assertGreater(float(descriptors[(target.key, 1)].mean[0]), 0.70)
        self.assertLess(float(descriptors[(reference.key, 1)].mean[0]), 0.30)


def _pair_entry(target: str, reference: str):
    return _PROBE.PairEntry(
        index=0,
        dataset="BCSS",
        sample_id=target,
        reference_sample_id=reference,
        target_image_path=Path(f"/tmp/{target}.png"),
        target_tissue_mask_path=Path(f"/tmp/{target}_mask.png"),
        reference_image_path=Path(f"/tmp/{reference}.png"),
        reference_tissue_mask_path=Path(f"/tmp/{reference}_mask.png"),
    )


def _descriptor(
    *,
    key: str,
    role: str,
    sample_id: str,
    label_id: int,
    mean: list[float],
    std: list[float],
):
    return _PROBE.DescriptorItem(
        label_name="tumor" if label_id == 1 else "stroma",
        label_id=label_id,
        image=_PROBE.ImageEntry(
            key=key,
            role=role,
            dataset="BCSS",
            sample_id=sample_id,
            image_path=Path(f"/tmp/{sample_id}.png"),
            tissue_mask_path=Path(f"/tmp/{sample_id}_mask.png"),
        ),
        token_count=8,
        token_fraction=0.25,
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32),
    )


class _MeanEncoder:
    def extract_uni_features(self, images):
        values = images.float().mean(dim=(1, 2, 3))
        return values[:, None, None].repeat(1, 4, 2)


class _OffsetVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = type("Config", (), {"scaling_factor": 1.0, "shift_factor": 0.0})()

    def encode(self, images):
        return type("EncodeOutput", (), {"latent_dist": _OffsetPosterior(images)})()

    def decode(self, latents, return_dict=False):
        return (latents,)


class _OffsetPosterior:
    def __init__(self, images):
        self.images = images

    def mode(self):
        return self.images + 1.0
