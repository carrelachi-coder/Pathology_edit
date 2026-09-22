"""Cumulative support and state-chain regressions, without loading a generator."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from phase3_joint_edit_refine import program_generator_adapter as adapter
from phase3_joint_edit_refine.models import JointContractError


def _write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def _seal(value, key):
    value[key] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         separators=(',', ':')).encode()).hexdigest()
    return value


def _png(path, data):
    Image.fromarray(data.astype(np.uint8)).save(path)
    return str(path)


@pytest.fixture
def program(tmp_path, monkeypatch):
    image = _png(tmp_path/'image.png', np.full((8,8,3), 127))
    tissue = np.ones((8,8), dtype=np.uint8)
    nuclei = np.zeros_like(tissue)
    tissue_path = _png(tmp_path/'source_tissue.png', tissue)
    nuclei_path = _png(tmp_path/'source_nuclei.png', nuclei)
    steps, declarations, manifests, sources = [], [], {}, []
    for i, y in enumerate((1,5), 1):
        source_t, source_n = tissue_path, nuclei_path
        sources.append((source_t, source_n))
        tissue = tissue.copy(); tissue[y,1] = 2
        tissue_path = _png(tmp_path/f't{i}.png', tissue)
        nuclei_path = _png(tmp_path/f'n{i}.png', nuclei)
        g = np.zeros_like(tissue); g[y:y+2,1:3] = 255
        support = _png(tmp_path/f'g{i}.png', g)
        tc = adapter._mask(source_t) != tissue
        hp = tmp_path/f'handoff{i}.json'
        m = dict(candidate_id=f'candidate-{i}', primitive_id='cohesive-boundary-expansion-v1',
                 mechanism_id='growth', paths={
                     'tissue_change':_png(tmp_path/f'tc{i}.png', tc*255),
                     'cell_change':_png(tmp_path/f'nc{i}.png', np.zeros_like(tc)),
                     'joint_change':_png(tmp_path/f'jc{i}.png', tc*255)},
                 ledger=dict(generation_support_fraction=float((g>0).mean())))
        _write(hp,m)
        inp = SimpleNamespace(reference_image=image, reference_tissue_mask=source_t,
                              reference_nuclei_mask=source_n, target_tissue_mask=tissue_path,
                              target_nuclei_mask=nuclei_path, generation_change_region=support,
                              prompt='fixture render prompt')
        manifests[str(hp)] = (inp,None,m)
        gates = _write(tmp_path/f'gates{i}.json',[dict(candidate_id=f'candidate-{i}',passed=True,
                         checks=[dict(check_id='fixture', severity='hard',passed=True)])])
        ctx = _write(tmp_path/f'context{i}.json',dict(provenance=dict(source_image_sha256=adapter._sha(image)),
                         joint_area_budget=dict(target_fraction=.06)))
        steps.append(dict(step_id=f'step-{i:03d}',intent_id=f'intent-{i}',status='validated',
            primitive_id=m['primitive_id'],mechanism_id='growth',selected_candidate_id=m['candidate_id'],
            input_tissue_sha256=adapter._sha(source_t),input_nuclei_sha256=adapter._sha(source_n),
            output_tissue_sha256=adapter._sha(tissue_path),output_nuclei_sha256=adapter._sha(nuclei_path),
            workflow_artifact_paths={'handoff_manifest':str(hp),'joint_gate_reports.json':gates,'case_context.json':ctx}))
        declarations.append(dict(step_id=f'step-{i:03d}',intent_id=f'intent-{i}',status='validated',
                            selected_primitive_id=m['primitive_id'],selected_mechanism_id='growth'))
    request=_seal({'instruction':'expand then expand'},'request_sha256')
    declaration=_seal(dict(status='validated',request_sha256=request['request_sha256'],steps=declarations),'program_sha256')
    result=dict(schema_version='joint-edit-program-run-v1',status='validated',
        evaluation=dict(passed=True,completed_steps=2,required_steps=2),steps=steps,
        edit_program_sha256=declaration['program_sha256'],semantic_request_sha256=request['request_sha256'],
        artifact_paths=dict(final_program=_write(tmp_path/'program.json',declaration),
                            semantic_request=_write(tmp_path/'request.json',request)))
    path=tmp_path/'program_result.json';_write(path,result)
    calls=[]
    def validate(path, **kwargs):
        calls.append(path)
        return manifests[path]
    monkeypatch.setattr(adapter,'build_frozen_generator_inputs',validate)
    return SimpleNamespace(path=path,result=result,manifests=manifests,calls=calls,out=tmp_path/'export')


def build(program, **kwargs):
    return adapter.build_frozen_program_generator_inputs(program.path,output_dir=program.out,dataset='BCSS',**kwargs)


def test_union_preserves_earlier_edit_and_original_reference(program):
    inputs, route, manifest=build(program,backend='cross')
    g=adapter._mask(inputs.generation_change_region)>0
    assert g[1,1] and g[5,1] and g.sum()==8
    assert adapter._mask(inputs.reference_tissue_mask)[1,1]==1
    assert adapter._mask(inputs.target_tissue_mask)[1,1]==2
    assert adapter._mask(inputs.target_tissue_mask)[5,1]==2
    assert len(program.calls)==2
    assert manifest['cumulative_ledger']['joint_fraction']==2/64
    assert manifest['automatic_route']['mode']=='inpaint'
    assert route.mode=='cross' and manifest['route_selection']=='explicit_caller_selection'
    assert manifest['independent_image_checks']=='not_run'
    assert 'net_instance_count' not in manifest['cumulative_ledger']
    assert 'final tissue and nucleus' in inputs.prompt


@pytest.mark.parametrize('status',['failed','partially_validated','review_required'])
def test_partial_program_is_not_exported(program,status):
    program.result['status']=status;_write(program.path,program.result)
    with pytest.raises(JointContractError,match='fully validated'):build(program)
    assert not program.out.exists()


def test_final_step_only_support_cannot_hide_first_step(program):
    # The final step mask is intentionally disjoint from the first one.
    inputs,_,_=build(program)
    final=program.manifests[program.calls[-1]][0]
    assert not np.array_equal(adapter._mask(inputs.generation_change_region),adapter._mask(final.generation_change_region))


def test_source_digest_drift_is_rejected(program):
    inp=next(iter(program.manifests.values()))[0]
    _png(Path(inp.reference_tissue_mask),np.zeros((8,8)))
    with pytest.raises(JointContractError,match='state digest'):build(program)


def test_discontinuous_chain_is_rejected(program):
    step=program.result['steps'][1];inp,_,_=program.manifests[step['workflow_artifact_paths']['handoff_manifest']]
    alternate=Path(inp.reference_tissue_mask).with_name('disconnected.png')
    a=adapter._mask(inp.reference_tissue_mask).copy();a[7,7]=3;_png(alternate,a)
    inp.reference_tissue_mask=str(alternate);step['input_tissue_sha256']=adapter._sha(alternate)
    _write(program.path,program.result)
    with pytest.raises(JointContractError,match='discontinuous'):build(program)


def test_selected_gate_failure_is_rejected(program):
    p=Path(program.result['steps'][1]['workflow_artifact_paths']['joint_gate_reports.json'])
    d=json.loads(p.read_text());d[0]['checks'][0]['passed']=False;_write(p,d)
    with pytest.raises(JointContractError,match='passing selected gates'):build(program)


def test_declared_program_digest_drift_is_rejected(program):
    p=Path(program.result['artifact_paths']['final_program']);d=json.loads(p.read_text());d['steps'].reverse();_write(p,d)
    with pytest.raises(JointContractError,match='document digest'):build(program)


def test_support_omission_is_rejected(program):
    inp,_,m=list(program.manifests.values())[1]
    _png(Path(inp.generation_change_region),np.zeros((8,8)))
    m['ledger']['generation_support_fraction']=0
    with pytest.raises(JointContractError,match='omits changed'):build(program)


def test_forced_cross_cannot_be_overridden(program):
    for inp,_,m in program.manifests.values():
        _png(Path(inp.generation_change_region),np.full((8,8),255))
        m['ledger']['generation_support_fraction']=1.
    with pytest.raises(JointContractError,match='cannot override'):build(program,backend='inpaint')
    assert not program.out.exists()


def test_existing_export_is_not_overwritten(program):
    build(program)
    with pytest.raises(JointContractError,match='already exists'):build(program)


def test_single_step_keeps_primitive_prompt_and_support(program):
    result=program.result;result['steps']=result['steps'][:1]
    result['evaluation'].update(completed_steps=1,required_steps=1)
    p=Path(result['artifact_paths']['final_program']);d=json.loads(p.read_text());d.pop('program_sha256');d['steps']=d['steps'][:1]
    _seal(d,'program_sha256');_write(p,d);result['edit_program_sha256']=d['program_sha256'];_write(program.path,result)
    inputs,_,manifest=build(program)
    original=next(iter(program.manifests.values()))[0]
    assert inputs.prompt==original.prompt
    np.testing.assert_array_equal(adapter._mask(inputs.generation_change_region),adapter._mask(original.generation_change_region))
    assert manifest['route_selection']=='automatic'


def test_source_image_change_is_rejected(program):
    image=next(iter(program.manifests.values()))[0].reference_image
    _png(Path(image),np.zeros((8,8,3)))
    with pytest.raises(JointContractError,match='image digest'):build(program)
