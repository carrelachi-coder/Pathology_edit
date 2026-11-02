"""Counterexamples for manuscript prerequisites; fixtures are never paper results."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from phase3_mask_edit.benchmark.experiment_identity import asset_identity, validate_resume, sha256_file
from phase3_mask_edit.benchmark.conditional_fidelity import detections_from_cellvit_json
from scripts import analyze_two_primitive_trajectory as trajectory
from scripts import predict_segmentator_masks_batch as segment
from scripts import run_phase3_inpaint_pipeline as generation


class ManuscriptCounterexamples(unittest.TestCase):
    def test_missing_detection_and_unproven_zero_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'cells.json'
            for status in ('empty_missing_raw_json', 'missing', 'failed', None):
                p.write_text(json.dumps({'cells': [], 'benchmark_provenance': {'status': status}}))
                with self.assertRaises(ValueError):
                    detections_from_cellvit_json(p, mpp=.25, image_size=(8, 8))
            p.write_text(json.dumps({'cells': [], 'benchmark_provenance': {'status': 'completed'}}))
            self.assertEqual(len(detections_from_cellvit_json(p, mpp=.25, image_size=(8, 8)).xy), 0)

    def test_cellvit_producer_missing_raw_is_failure_without_zero_mask(self):
        import batch_cellvit_ui_equiv as producer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); images = root / 'images'; images.mkdir()
            Image.new('RGB', (8, 8)).save(images / 'a.png')
            model = root / 'model'; model.write_bytes(b'fixture')
            argv = ['batch', '--images-dir', str(images), '--output-dir', str(root/'masks'),
                '--json-output-dir', str(root/'json'), '--raw-outdir', str(root/'raw'),
                '--model', str(model), '--cellvit-root', str(root)]
            with patch('sys.argv', argv), patch.object(producer, 'write_openslide_tiff', side_effect=lambda source, dest: dest.write_bytes(b'fixture')), patch.object(producer.subprocess, 'run'):
                self.assertEqual(producer.main(), 1)
            self.assertFalse((root/'masks/a.png').exists())
            self.assertEqual(json.loads((root/'json/a.json').read_text())['benchmark_provenance']['status'], 'missing')

    def test_source_mutation_after_handoff_is_rejected(self):
        from tests.test_joint_generator_ablation import _write_handoff
        from phase3_joint_edit_refine.generator_adapter import build_frozen_generator_inputs
        from phase3_joint_edit_refine.models import JointContractError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = _write_handoff(root)
            build_frozen_generator_inputs(manifest, output_dir=root/'out')
            source = Path(json.loads(manifest.read_text())['source_assets']['image'])
            source.write_bytes(b'replaced source at same path')
            with self.assertRaisesRegex(JointContractError, 'source asset.*drift'):
                build_frozen_generator_inputs(manifest, output_dir=root/'out')

    def test_inpaint_effective_parameters_and_cache_key_include_sampling(self):
        from scripts import run_phase3_inpaint_pipeline as gen
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('config.json', 'phase5_conditioning.pt'):
                (root / name).write_bytes(b'fixture')
            gen._INPAINT_BUNDLE_CACHE.clear()
            with patch.object(gen, 'load_inpaint_bundle', side_effect=lambda **kw: SimpleNamespace(**kw)) as loader:
                first = gen._cached_inpaint_bundle(pretrained_model_name_or_path='fixture', checkpoint_path=root, device='cpu', seed=19, num_inference_steps=7, guidance_scale=2.)
                same = gen._cached_inpaint_bundle(pretrained_model_name_or_path='fixture', checkpoint_path=root, device='cpu', seed=19, num_inference_steps=7, guidance_scale=2.)
                changed = gen._cached_inpaint_bundle(pretrained_model_name_or_path='fixture', checkpoint_path=root, device='cpu', seed=20, num_inference_steps=9, guidance_scale=4.)
            self.assertIs(first, same)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual((changed.seed, changed.num_inference_steps, changed.guidance_scale), (20, 9, 4.))
            gen._INPAINT_BUNDLE_CACHE.clear()

    def test_generic_semantic_region_cannot_omit_joint_changes(self):
        from scripts import run_agentic_edit_workflow as agentic
        from phase3_mask_edit.core.mask_io import save_id_mask
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (8, 8)).save(root / 'image.png')
            save_id_mask(np.ones((8, 8), dtype=np.int64), root / 'source.png')
            save_id_mask(np.full((8, 8), 2, dtype=np.int64), root / 'target.png')
            Image.new('L', (8, 8)).save(root / 'nuclei.png')
            Image.fromarray(np.pad(np.ones((4, 8), dtype=np.uint8)*255, ((0, 4), (0, 0)))).save(root/'region.png')
            args = agentic.build_parser().parse_args(['--profile', 'BCSS', '--reference-image', str(root/'image.png'),
                '--reference-tissue-mask', str(root/'source.png'), '--reference-nuclei-mask', str(root/'nuclei.png'),
                '--target-tissue-mask', str(root/'target.png'), '--target-nuclei-mask', str(root/'nuclei.png'),
                '--semantic-change-region', str(root/'region.png'), '--output', str(root/'out')])
            with self.assertRaisesRegex(ValueError, 'omits 32 actual joint'):
                agentic._load_and_validate_inputs(args)

    def test_vae_sampling_uses_explicit_generator(self):
        import torch
        from controlnet_train.inference.pipeline import _encode_images_to_latents
        class VAE:
            config = SimpleNamespace(shift_factor=0., scaling_factor=1.)
            def parameters(self):
                return iter([torch.zeros(1)])
            def encode(self, images):
                return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator: torch.rand(1, generator=generator)))
        def draw(seed):
            return _encode_images_to_latents(VAE(), torch.zeros(1, 3, 8, 8), torch.float32,
                generator=torch.Generator().manual_seed(seed))
        self.assertTrue(torch.equal(draw(17), draw(17)))
        self.assertFalse(torch.equal(draw(17), draw(18)))

    def test_same_postprocess_name_does_not_hide_different_eligibility(self):
        from controlnet_train.pix2pix_transfer.inference import apply_cross_low_stain_protection
        results = []
        for region in (np.ones((64, 64), dtype=bool), np.pad(np.ones((32, 64), dtype=bool), ((0, 32), (0, 0)))):
            _, _, info = apply_cross_low_stain_protection(cross_image=Image.new('RGB', (64, 64), 'white'),
                pix2pix_image=Image.new('RGB', (64, 64), 'red'), target_nuclei_mask=np.zeros((64, 64)), protection_region=region)
            results.append(info)
        self.assertEqual(results[0]['configuration'], results[1]['configuration'])
        self.assertNotEqual(results[0]['eligibility_region_sha256'], results[1]['eligibility_region_sha256'])
        self.assertIn('applied_mask_sha256', results[0])

    def test_formal_entry_rejects_unfrozen_protocol_before_generation(self):
        from scripts.run_manuscript_global_cohort import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); protocol = root / 'protocol.json'
            protocol.write_text(json.dumps({'status': 'draft', 'formal_launch_permitted': False}))
            with self.assertRaisesRegex(ValueError, 'remains blocked'):
                main(['--protocol', str(protocol), '--cohort-manifest', str(root/'not-created'), '--output-root', str(root/'out')])
            self.assertFalse((root/'out').exists())

    def test_segmentation_flushes_a_before_existing_b_c(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'in'; out = root / 'out'
            source.mkdir(); out.mkdir()
            for name in ('a', 'b', 'c'):
                Image.new('RGB', (8, 8)).save(source / f'{name}.png')
            for name in ('b', 'c'):
                Image.new('L', (8, 8)).save(out / f'{name}.png')
            seen = []
            def predict(model, paths, output, device):
                seen.extend(p.name for p in paths)
                for p in paths:
                    Image.new('L', (8, 8)).save(output / f'{p.stem}.png')
            with patch('segmentator.inference.load_checkpoint', return_value=SimpleNamespace(to=lambda x: object())), patch.object(segment, 'predict_batch', side_effect=predict):
                code = segment.main(['--checkpoint', 'fixture', '--input-dir', str(source), '--output-dir', str(out), '--batch-size', '2', '--skip-existing', '--device', 'cpu'])
            self.assertEqual(code, 0)
            self.assertEqual(seen, ['a.png'])
            report = json.loads((out / 'prediction_summary.json').read_text())
            self.assertEqual((report['written'], report['skipped'], report['missing_outputs']), (1, 2, []))

    def test_resume_rejects_same_path_changed_inputs_and_corrupt_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; output = root / 'output'; record = root / 'identity.json'
            source.write_bytes(b'old'); output.write_bytes(b'result')
            identity = {'input': asset_identity(source), 'endpoint': 'precomposite'}
            record.write_text(json.dumps({'identity': identity, 'outputs': {output.name: sha256_file(output)}}))
            self.assertTrue(validate_resume(record, identity, [output]))
            source.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'identity drift'):
                validate_resume(record, {'input': asset_identity(source), 'endpoint': 'precomposite'}, [output])
            output.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'output content'):
                validate_resume(record, identity, [output])

    def test_sham_returns_real_precomposite_even_if_empty_support_composite_copies_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(generation_mode='cross-v1', cross_backend='cross-v1', route_threshold=.35,
                pretrained_model_name_or_path='fixture', cross_v1_checkpoint='fixture', pix2pix_checkpoint='fixture',
                reference_image='source', reference_tissue_mask='tissue', reference_nuclei_mask='nuclei',
                profile='BCSS', prompt='source description', device='cpu', endpoint_stage='postprocessed_precomposite')
            with patch.object(generation, 'validate_production_controlnet_checkpoint', return_value={}), patch.object(generation, 'validate_production_pix2pix_checkpoint', return_value={}), patch.object(generation, '_run_cross_v1_no_ip_generation', return_value=(Image.new('RGB', (8, 8), 'red'), Image.new('RGB', (8, 8), 'blue'), {})) as runner:
                endpoint, info = generation._run_generation_stage(args=args, output_dir=root,
                    reference_image=np.zeros((8, 8, 3), dtype=np.uint8), change_region=np.zeros((8, 8), dtype=bool),
                    target_tissue_path=root / 'tissue', target_nuclei_path=root / 'nuclei')
            runner.assert_called_once()
            self.assertTrue(info['real_generation_completed'])
            self.assertEqual(Image.open(endpoint).getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(Image.open(root / 'generated_image.png').getpixel((0, 0)), (0, 0, 0))

    def test_patient_clusters_are_complete_and_permutations_refit(self):
        rows = [{'wsi_id': wsi, 'patient_id': patient} for patient in ('p1', 'p2', 'p3') for wsi in (patient+'a', patient+'b')]
        groups = trajectory.independent_groups(rows, 'patient_id')
        rng = np.random.default_rng(710)
        reference = rng.normal(size=(6, 5))
        features = {p: {s: {'reference': reference.copy(), 'inpaint': reference+rng.normal(size=(6, 5)), 'cross': reference+rng.normal(size=(6, 5))} for s in trajectory.STRENGTHS} for p in trajectory.PRIMITIVES}
        original = trajectory.cross_fitted_axes
        seen = []
        def capture(candidate, units):
            for indices in (np.flatnonzero(groups == p) for p in np.unique(groups)):
                changed = np.any(candidate['u1']['moderate']['cross'][indices] != features['u1']['moderate']['cross'][indices], axis=1)
                self.assertEqual(len(set(changed)), 1)
            seen.append(tuple(np.any(candidate["u1"]["moderate"]["cross"] != features["u1"]["moderate"]["cross"], axis=1)))
            return original(candidate, units)
        with patch.object(trajectory, 'cross_fitted_axes', side_effect=capture):
            value = trajectory.operation_label_permutation_pvalue(features, groups, strength='moderate', backend='cross', repeats=8, seed=1)
        self.assertEqual(len(seen), 9)  # observed and every one of 2^3 raw-label swaps
        def enumerated_stat(swaps):
            flip = np.isin(groups, np.unique(groups)[np.asarray(swaps, dtype=bool)])[:, None]
            candidate = {p: {strength: {backend: np.where(flip, features[other][strength][backend], values)
                for backend, values in features[p][strength].items()} for strength in trajectory.STRENGTHS}
                for p, other in (("u1", "u2"), ("u2", "u1"))}
            def unit(x):
                return x / np.linalg.norm(x)
            axes = {p: np.zeros((6, 5)) for p in trajectory.PRIMITIVES}
            displacements = {}
            for p in trajectory.PRIMITIVES:
                cell = candidate[p]['moderate']
                delta = cell['cross'] - cell['reference']
                displacements[p] = delta / np.linalg.norm(delta, axis=1)[:, None]
                for patient in np.unique(groups):
                    train = groups != patient
                    local = (cell['inpaint'] - cell['reference'])[train].mean(axis=0)
                    global_ = delta[train].mean(axis=0)
                    axes[p][~train] = unit(unit(local) + unit(global_))
            margin = .5 * np.sum((displacements['u1']-displacements['u2']) * (axes['u1']-axes['u2']), axis=1)
            return np.mean([margin[groups == p].mean() for p in np.unique(groups)])
        observed = enumerated_stat((False, False, False))
        expected = sum(enumerated_stat(swap) >= observed - 1e-12 for swap in itertools.product((False, True), repeat=3)) / 8
        self.assertAlmostEqual(value, expected)
        self.assertEqual(len(set(seen)), 8)
        self.assertTrue(0 <= value <= 1)
        rows[0].pop('patient_id')
        with self.assertRaisesRegex(ValueError, 'missing verified'):
            trajectory.independent_groups(rows, 'patient_id')


if __name__ == '__main__':
    unittest.main()
