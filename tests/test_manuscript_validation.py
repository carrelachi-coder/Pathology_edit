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
            seen.append(True)
            return original(candidate, units)
        with patch.object(trajectory, 'cross_fitted_axes', side_effect=capture):
            value = trajectory.operation_label_permutation_pvalue(features, groups, strength='moderate', backend='cross', repeats=8, seed=1)
        self.assertEqual(len(seen), 9)  # observed and every one of 2^3 raw-label swaps
        self.assertTrue(0 <= value <= 1)
        rows[0].pop('patient_id')
        with self.assertRaisesRegex(ValueError, 'missing verified'):
            trajectory.independent_groups(rows, 'patient_id')


if __name__ == '__main__':
    unittest.main()
