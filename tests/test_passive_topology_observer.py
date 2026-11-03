"""Synthetic noninterference only. No BCSS source replay, model or GPU call."""
import errno
import json
import random
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from phase3_mask_edit_refine import passive_observer as po
from phase3_mask_edit_refine.observer_comparison import compare_observer_runs, _load, rejection_pixel_sets
from phase3_mask_edit_refine.execution import compile_edit_plan_with_witness, TopologySafeAreaUnderfillError
from phase3_mask_edit_refine.models import AreaBudget, RefineContractError
from phase3_mask_edit_refine.scene import build_scene_analysis
from phase3_mask_edit_refine.skills import SkillRepository
from phase3_mask_edit_refine.topology import topology_safe_priority_grow
from scripts.review_passive_topology_observer import review, StripObservation
from tests.test_mask_edit_refine import _manual_plan, _case


def _fixture():
    y, x = np.ogrid[:64, :64]
    source = np.full((64, 64), 2, dtype=np.uint8)
    source[(y-32)**2+(x-32)**2 <= 15**2] = 12
    schema = SkillRepository().annotation_schema('glas-gland-v1')
    scene = build_scene_analysis(source, schema=schema, pixel_size_um=.465)
    interface = max(scene.interfaces_for(source_labels=('Stroma',), target_label='Tumor'), key=lambda v: v.contact_pixels)
    case = _case(primitive='tumor-burden-increase-v1', area=.03)
    plan = _manual_plan(case=case, interface=interface, source_label='Stroma', target_label='Tumor',
                        area=.03, supporting_rules=(), band_max=32.)
    return source, schema, scene, plan


def _compile(fixture, observer=None):
    source, schema, scene, plan = fixture
    with po.observation_session(observer) if observer else nullcontext():
        try:
            result = compile_edit_plan_with_witness(plan, source_mask=source, schema=schema, scene=scene)
            # Normalize results with the strict passive encoder, not by final area alone.
            encoder = po.PassiveObserver.__new__(po.PassiveObserver)
            encoder.mode = 'hash_only'; encoder._arrays = 0; encoder.bytes_written = 0; encoder.max_bytes = 10**9
            return ('success', encoder._encode(result))
        except RefineContractError as exc:
            return ('failure', type(exc).__name__, str(exc), getattr(exc, 'feedback', None))


def _inject(monkeypatch, observer, fault):
    def broken(*args, **kwargs):
        if fault == 'enospc':
            raise OSError(errno.ENOSPC, 'injected full disk')
        if fault == 'copy':
            raise MemoryError('injected mask-copy failure')
        raise RuntimeError('injected observer failure')
    target = {'write': '_write_line', 'enospc': '_write_line', 'copy': '_copy_array',
              'queue': 'queue', 'throws': 'record', 'serialize': 'serialize'}[fault]
    if target == 'queue':
        monkeypatch.setattr(observer._queue, 'put_nowait', broken)
    elif target == 'serialize':
        monkeypatch.setattr(po, '_json', broken)
    else:
        monkeypatch.setattr(observer, target, broken)


def test_static_business_ast_unchanged():
    report = review(Path(__file__).resolve().parents[1])
    assert report['passed']
    assert sum(len(row['passive_calls']) for row in report['files']) >= 25


def test_static_review_rejects_eager_guard_or_rng_callback():
    import ast
    for code in ['observe("bad", value=guard())', 'observe("bad", value=rng.random())',
                 'observe("bad", value=[x for x in lazy])']:
        with pytest.raises(AssertionError):
            StripObservation().visit(ast.parse(code))


def test_real_synthetic_executor_off_hash_full_exact_semantics(tmp_path):
    fixture = _fixture()
    source_before = fixture[0].copy()
    rng_before = (random.getstate(), np.random.get_state())
    baseline = _compile(fixture)
    b = po.PassiveObserver(tmp_path/'B', mode='hash_only')
    c = po.PassiveObserver(tmp_path/'C', mode='full_capture')
    assert _compile(fixture, b) == baseline == _compile(fixture, c)
    assert baseline[0] == 'success'
    assert b.status()['valid'] and c.status()['valid']
    assert compare_observer_runs(tmp_path/'B', tmp_path/'C')['status'] == 'semantic_match'
    assert random.getstate() == rng_before[0]
    now = np.random.get_state()
    assert now[0] == rng_before[1][0] and np.array_equal(now[1], rng_before[1][1]) and now[2:] == rng_before[1][2:]
    assert np.array_equal(fixture[0], source_before)
    events, _ = _load(tmp_path/'C')
    names = [e['event'] for e in events]
    assert {'envelopes.ready', 'ownership.ready', 'probe.complete', 'grow.end', 'guard.accept'} <= set(names)


@pytest.mark.parametrize('fault', ['write', 'enospc', 'serialize', 'copy', 'queue', 'throws'])
def test_faults_do_not_change_real_executor_mask_or_decision(tmp_path, monkeypatch, fault):
    fixture = _fixture()
    expected = _compile(fixture)
    obs = po.PassiveObserver(tmp_path/fault, mode='full_capture')
    with monkeypatch.context() as m:
        _inject(m, obs, fault)
        actual = _compile(fixture, obs)
    assert actual == expected
    assert not obs.status()['valid'] and not obs.status()['interpretation_permitted']
    assert obs.status()['faults']


def test_nonfinite_values_are_explicit_not_zero(tmp_path):
    obs = po.PassiveObserver(tmp_path/'numbers', mode='hash_only')
    with po.observation_session(obs):
        po.observe('numbers', values=[float('nan'), float('inf'), -float('inf'), None, 0])
    events, _ = _load(tmp_path/'numbers')
    text = json.dumps(events, allow_nan=False)
    assert all(s in text for s in ['nan', '+inf', '-inf', 'null'])
    assert obs.status()['valid']


def test_nonfinite_serialization_failure_is_isolated(tmp_path, monkeypatch):
    obs = po.PassiveObserver(tmp_path/'nonfinite-fault', mode='hash_only')
    original = po._json
    def reject_nonfinite(value):
        if 'nonfinite' in str(value):
            raise ValueError('injected serializer refusal of nonfinite')
        return original(value)
    with monkeypatch.context() as m:
        m.setattr(po, '_json', reject_nonfinite)
        with po.observation_session(obs):
            po.observe('numbers', value=float('inf'))
            decision = 'unchanged-business-result'
    assert decision == 'unchanged-business-result' and not obs.status()['valid']


def test_detached_array_data_and_flags(tmp_path):
    obs = po.PassiveObserver(tmp_path/'detached', mode='full_capture')
    source = np.arange(12, dtype=np.uint16).reshape(3, 4)
    before = source.copy()
    with po.observation_session(obs):
        po.observe('array', mask=source[:, ::2])
        assert source.flags.writeable
        source[:] = 0
    saved = np.load(tmp_path/'detached/array_00000001.npy', allow_pickle=False)
    assert np.array_equal(saved, before[:, ::2])
    saved[:] = 9
    assert not source.any()


def _planner_failure_run(observer=None):
    """Real bounded retry handler with synthetic planner/executor dependencies."""
    from phase3_joint_edit_refine import candidate_feasibility as cf
    source, schema, scene, plan = _fixture()
    seen = []
    def planner(**kwargs):
        seen.append((tuple(kwargs['execution_feedback'].get('failed_interface_ids', [])),
                     tuple(kwargs['execution_feedback']['preferred_anchor_ids'])))
        return plan, {'synthetic': True}
    def compiler(*args, **kwargs):
        raise RefineContractError('synthetic interface contract rejection')
    with patch.object(cf, 'augment_tissue_scene_with_nuclei_preflight', return_value=scene), \
         patch.object(cf.MultiInterfaceResearchTissuePlanner, 'create_joint_tissue_plan', side_effect=planner), \
         patch.object(cf, 'validate_edit_plan'), \
         patch.object(cf, 'compile_joint_tissue_plan_with_witness', side_effect=compiler):
        with po.observation_session(observer) if observer else nullcontext():
            try:
                cf.CandidateFeasibilityCompiler()._compile_one(
                    tissue_case=_case(primitive='tumor-burden-increase-v1', area=.03),
                    source_tissue=source, schema=schema, scene=SimpleNamespace(tissue=scene, auxiliary_structure_masks={}),
                    tissue_bundle=None, joint_bundle=SimpleNamespace(mechanism=SimpleNamespace(representability=SimpleNamespace(protected_auxiliary_structures=(), receiving_auxiliary_structures=()))),
                    nuclei_preflight=None, initial_failed_interface_ids=(), preferred_anchor_ids=('fixed-anchor',),
                    authority_binding_sha256='synthetic')
            except cf.JointContractError as exc:
                return seen, type(exc).__name__, str(exc)
    raise AssertionError('expected frozen rejection')


@pytest.mark.parametrize('fault', ['write', 'enospc', 'serialize', 'copy', 'queue', 'throws'])
def test_recorder_fault_cannot_change_anchor_or_bounded_replan_sequence(tmp_path, monkeypatch, fault):
    expected = _planner_failure_run()
    obs = po.PassiveObserver(tmp_path/fault, mode='hash_only')
    # Copy fault is encountered through the first source mask carried by a plan/execution event.
    if fault == 'copy':
        original = obs.record
        def add_array(event, values):
            original(event, dict(values, synthetic_array=np.zeros((1, 1), dtype=np.uint8)))
        monkeypatch.setattr(obs, 'record', add_array)
    with monkeypatch.context() as m:
        _inject(m, obs, fault)
        actual = _planner_failure_run(obs)
    assert actual == expected and len(actual[0]) == 3
    assert all(anchors == ('fixed-anchor',) for _, anchors in actual[0])
    assert not obs.status()['valid']


def test_bounded_replan_normal_hash_full_exact(tmp_path):
    expected = _planner_failure_run()
    b = po.PassiveObserver(tmp_path/'B', mode='hash_only')
    c = po.PassiveObserver(tmp_path/'C', mode='full_capture')
    assert _planner_failure_run(b) == _planner_failure_run(c) == expected
    assert compare_observer_runs(tmp_path/'B', tmp_path/'C')['status'] == 'semantic_match'


@pytest.mark.parametrize('changed', ['input', 'contract', 'plan_order', 'anchor_order', 'allocation', 'mask', 'topology', 'exception', 'rng'])
def test_strict_comparator_rejects_each_semantic_mismatch(tmp_path, changed):
    for mode, name in [('hash_only', 'B'), ('full_capture', 'C')]:
        obs = po.PassiveObserver(tmp_path/name, mode=mode)
        values = {'input': 'source-hash', 'contract': 'contract-hash', 'plan_order': [1, 2],
                  'anchor_order': ['a', 'b'], 'allocation': [4, 6], 'mask': np.eye(3, dtype=np.uint8),
                  'topology': {'passed': False}, 'exception': ValueError('same text'), 'rng': [42, 19]}
        if name == 'C':
            replacement = {'input':'other', 'contract':'other', 'plan_order':[2,1], 'anchor_order':['b','a'],
                           'allocation':[6,4], 'mask':np.zeros((3,3), dtype=np.uint8), 'topology':{'passed':True},
                           'exception':TypeError('different text'), 'rng':[19,42]}
            values[changed] = replacement[changed]
        with po.observation_session(obs):
            po.observe('comparison_fixture', **values)
    assert compare_observer_runs(tmp_path/'B', tmp_path/'C')['status'] == 'mismatch_hold'


def test_output_collision_does_not_write_into_old_directory(tmp_path):
    old = tmp_path/'old'; old.mkdir(); (old/'keep').write_text('original')
    before = sorted(p.name for p in old.iterdir())
    obs = po.PassiveObserver(old, mode='full_capture')
    with po.observation_session(obs):
        po.observe('must_not_write', mask=np.zeros((2, 2)))
    assert not obs.status()['valid'] and before == sorted(p.name for p in old.iterdir())
    assert (old/'keep').read_text() == 'original'


def test_limits_invalidate_evidence_without_changing_business(tmp_path):
    for kwargs in [{'max_events':0}, {'max_bytes':1}]:
        obs = po.PassiveObserver(tmp_path/str(len(list(tmp_path.iterdir()))), mode='full_capture', **kwargs)
        assert _compile(_fixture(), obs) == _compile(_fixture())
        assert not obs.status()['valid']


def test_existing_local_generator_state_not_advanced(tmp_path):
    rng = np.random.default_rng(91); before = json.dumps(rng.bit_generator.state)
    obs = po.PassiveObserver(tmp_path/'rng', mode='hash_only')
    with po.observation_session(obs):
        po.observe('grow.end', rng=rng)
    assert before == json.dumps(rng.bit_generator.state)


def test_repeated_rejections_keep_counts_separate_from_unique_pixels():
    def event(seq, kind, **values):
        return {'sequence':seq, 'event':kind, 'values':{'mapping':list(map(list, values.items()))}}
    events = [event(0, 'grow.begin'), event(1, 'guard.reject', row=2, col=3, reason='source_retention'),
              event(2, 'guard.reject', row=2, col=3, reason='source_retention'),
              event(3, 'guard.local_result', row=2, col=3, reason='source_hole_change')]
    result = rejection_pixel_sets(events)[0]
    assert result['event_counts']['source_retention'] == 2
    assert len(result['rejections']['source_retention']) == 1
    assert result['rejections']['source_retention'] & result['rejections']['source_hole_change'] == {(2,3)}


@pytest.mark.parametrize('tamper', ['truncate_stream', 'delete_raster', 'change_raster', 'remove_manifest_entry'])
def test_missing_or_corrupt_evidence_cannot_be_accepted(tmp_path, tamper):
    for mode, name in [('hash_only', 'B'), ('full_capture', 'C')]:
        obs = po.PassiveObserver(tmp_path/name, mode=mode)
        with po.observation_session(obs):
            po.observe('mask', value=np.eye(3, dtype=np.uint8))
    assert compare_observer_runs(tmp_path/'B', tmp_path/'C')['status'] == 'semantic_match'
    if tamper == 'truncate_stream':
        (tmp_path/'C/semantic.jsonl').write_text('')
    elif tamper == 'delete_raster':
        (tmp_path/'C/array_00000001.npy').unlink()
    elif tamper == 'change_raster':
        np.save(tmp_path/'C/array_00000001.npy', np.zeros((3,3), dtype=np.uint8))
    else:
        path = tmp_path/'C/observer_status.json'; data = json.loads(path.read_text())
        data['artifacts'] = []; path.write_text(json.dumps(data))
    assert compare_observer_runs(tmp_path/'B', tmp_path/'C')['status'] == 'mismatch_hold'


def test_no_backend_initialization_rng_creation_or_lazy_iterator_consumption(tmp_path, monkeypatch):
    rng = np.random.default_rng(73)
    lazy_calls = []
    def lazy():
        lazy_calls.append('consumed')
        yield 1
    iterator = lazy()
    obs = po.PassiveObserver(tmp_path/'lazy', mode='hash_only')
    with monkeypatch.context() as m:
        m.setattr(np.random, 'default_rng', lambda *a, **k: pytest.fail('observer created RNG'))
        m.setattr(np.random, 'seed', lambda *a, **k: pytest.fail('observer reseeded RNG'))
        m.setattr(random, 'seed', lambda *a, **k: pytest.fail('observer reseeded Python RNG'))
        with po.observation_session(obs):
            po.observe('grow.end', rng=rng)
            po.observe('not_serializable', lazy=iterator)
    assert not lazy_calls and not obs.status()['valid']


def test_disk_failure_during_raster_write_and_close_never_escapes(tmp_path, monkeypatch):
    fixture = _fixture(); expected = _compile(fixture)
    obs = po.PassiveObserver(tmp_path/'raster-full', mode='full_capture')
    def disk_full(*args):
        raise OSError(errno.ENOSPC, 'injected raster ENOSPC')
    monkeypatch.setattr(obs, '_write_raster', disk_full)
    assert _compile(fixture, obs) == expected
    assert not obs.status()['valid']


def test_structured_underfill_exception_preserved_without_string_callback(tmp_path):
    # Observer must not call __str__ again, or modify exception args/feedback.
    class StatefulError(TopologySafeAreaUnderfillError):
        def __str__(self):
            raise AssertionError('observer re-rendered business exception')
    error = StatefulError('saved error text', feedback={'realized': 22, 'topology': {'passed': True}})
    obs = po.PassiveObserver(tmp_path/'exception', mode='hash_only')
    with po.observation_session(obs):
        po.observe('compiler.exception', exception=error, rendered_error='StatefulError: saved error text')
    assert obs.status()['valid'] and error.args == ('saved error text',)
    assert error.feedback == {'realized':22, 'topology':{'passed':True}}
    assert 'saved error text' in (tmp_path/'exception/semantic.jsonl').read_text()


def test_actual_underfill_failure_preserved_off_hash_full(tmp_path):
    source, schema, scene, plan = _fixture()
    plan = replace(plan, area_budget=AreaBudget(.50, .49, .60, fallback_policy='max_feasible_below_target'))
    fixture = source, schema, scene, plan
    expected = _compile(fixture)
    assert expected[0:2] == ('failure', 'TopologySafeAreaUnderfillError')
    b = po.PassiveObserver(tmp_path/'B', mode='hash_only')
    c = po.PassiveObserver(tmp_path/'C', mode='full_capture')
    assert _compile(fixture, b) == _compile(fixture, c) == expected
    assert compare_observer_runs(tmp_path/'B', tmp_path/'C')['status'] == 'semantic_match'
    events, _ = _load(tmp_path/'C')
    probes = [dict(e['values']['mapping']) for e in events if e['event'] == 'probe.complete']
    assert len(probes) >= 2
    assert all({'selected', 'combined_change', 'source_region', 'target_region', 'topology'} <= set(p) for p in probes)
    assert probes[0]['total'] != probes[-1]['total']


def test_close_failure_is_out_of_business_handler(tmp_path):
    obs = po.PassiveObserver(tmp_path/'close-fault', mode='hash_only')
    original = obs._stream
    class BrokenClose:
        def write(self, value):
            return original.write(value)
        def close(self):
            original.close()
            raise OSError('injected close failure')
    obs._stream = BrokenClose()
    assert _compile(_fixture(), obs) == _compile(_fixture())
    assert not obs.status()['valid'] and obs.status()['faults'][0]['stage'] == 'close_stream'


def test_loaded_cpu_rng_state_can_be_read_without_touching_cuda(tmp_path, monkeypatch):
    import sys
    reads = []
    class ExistingCPUGenerator:
        def get_state(self):
            reads.append('existing_cpu_state')
            return SimpleNamespace(numpy=lambda: np.arange(4, dtype=np.uint8))
    class ForbiddenCUDA:
        def __getattribute__(self, name):
            raise AssertionError('observer touched CUDA')
    already_loaded = SimpleNamespace(default_generator=ExistingCPUGenerator(), cuda=ForbiddenCUDA())
    obs = po.PassiveObserver(tmp_path/'cpu-state', mode='hash_only')
    with monkeypatch.context() as m:
        m.setitem(sys.modules, 'torch', already_loaded)
        with po.observation_session(obs):
            po.observe('grow.end')
    assert reads == ['existing_cpu_state'] and obs.status()['valid']


def test_directed_interface_rejection_keeps_pre_plan_filter_evidence(tmp_path):
    from phase3_joint_edit_refine import tissue_planner as tp
    from phase3_joint_edit_refine.skills.schema import TissueFrontContract
    source, schema, scene, plan = _fixture()
    front = TissueFrontContract('fixture', .1, .2, 1, 0., 32, 1., 100., False, 1., 0)
    case = _case(primitive='tumor-burden-increase-v1', area=.03)
    bundle = SimpleNamespace(edit_contract=SimpleNamespace(source_label_options=('Stroma',), target_label='Tumor', allowed_tools=()))
    joint = SimpleNamespace(mechanism=SimpleNamespace(mechanism_id='synthetic', tissue_program=SimpleNamespace(
        primitive_label_contracts={case.primitive_id:{'source_labels':('Stroma',), 'target_labels':('Tumor',)}},
        allowed_tools=(), front=front)))
    preflight = SimpleNamespace(reference_area_p95=16, feasible_interface_ids=())
    def run(observer=None):
        with patch.object(tp, 'compile_tissue_tool_program', return_value=SimpleNamespace(allowed_concrete_executors=())):
            with po.observation_session(observer) if observer else nullcontext():
                try:
                    tp.MultiInterfaceResearchTissuePlanner().create_joint_tissue_plan(
                        case=case, scene=scene, bundle=bundle, joint_bundle=joint,
                        image_paths=(), nuclei_preflight=preflight)
                except RefineContractError as exc:
                    return type(exc).__name__, str(exc)
    expected = run()
    assert expected == ('RefineContractError', 'no directed interface satisfies both the tissue and preflight nuclei contracts')
    b = po.PassiveObserver(tmp_path/'B', mode='hash_only')
    c = po.PassiveObserver(tmp_path/'C', mode='full_capture')
    assert run(b) == run(c) == expected
    assert compare_observer_runs(tmp_path/'B', tmp_path/'C')['status'] == 'semantic_match'
    events, _ = _load(tmp_path/'C')
    assert [e['event'] for e in events] == ['planner.label_filter', 'planner.preflight_filter', 'planner.front_filter']
    assert dict(events[-1]['values']['mapping'])['legal']['items'] == []
