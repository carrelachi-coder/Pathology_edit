import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from phase3_mask_edit.benchmark.experiment_identity import sha256_file, validate_resume
from phase3_mask_edit.benchmark.manuscript_contract import preflight, verified_asset, validate_target_pair


def test_frozen_flags_do_not_bypass_missing_assets(tmp_path):
    protocol = {'cohort_selection': {'formal_source_list_frozen': False}}
    with pytest.raises(ValueError, match='source list'):
        preflight(protocol, tmp_path/'manifest', tmp_path)
    protocol['cohort_selection']['formal_source_list_frozen'] = True
    with pytest.raises(ValueError, match='unresolved'):
        preflight(protocol, tmp_path/'manifest', tmp_path)
    for asset in ('unresolved', None, {}, {'status':'frozen','path':str(tmp_path/'missing'),'sha256':'a'*64}):
        with pytest.raises(ValueError):
            verified_asset(asset)


def test_partial_attempt_cannot_be_reused_or_overwritten(tmp_path):
    stage = tmp_path/'stage1_no_ip.png'; stage.write_bytes(b'original intermediate')
    with pytest.raises(ValueError, match='lack experiment identity'):
        validate_resume(tmp_path/'identity.json', {'source':'changed'}, [tmp_path/'generated.png'])
    assert stage.read_bytes() == b'original intermediate'
    (tmp_path/'identity.json').write_text(json.dumps({'identity':{'source':'same'},'status':'failed'}))
    with pytest.raises(ValueError, match='incomplete'):
        validate_resume(tmp_path/'identity.json', {'source':'same'}, [tmp_path/'generated.png'])


def test_segmentation_zero_byte_output_is_not_valid(tmp_path):
    from scripts.predict_segmentator_masks_batch import prediction_valid
    source=tmp_path/'source.png'; Image.new('RGB',(8,8)).save(source)
    out=tmp_path/'out.png'; out.write_bytes(b'')
    assert not prediction_valid(source,out,{},8)


def test_effect_and_interval_use_equal_cluster_weights():
    from scripts.analyze_two_primitive_trajectory import cluster_summary
    result = cluster_summary(np.array([1.]*8+[-1.,-1.]), np.array(['a']*8+['b','c']), repeats=100, seed=42)
    assert result['mean'] == pytest.approx(-1/3)
    assert result['mean'] != pytest.approx(.6)


def test_changed_cell_json_cannot_be_reused(tmp_path):
    import batch_cellvit_ui_equiv as producer
    images=tmp_path/'images'; images.mkdir(); Image.new('RGB',(8,8)).save(images/'a.png')
    masks=tmp_path/'masks'; masks.mkdir(); Image.new('L',(8,8),101).save(masks/'a.png')
    output=tmp_path/'json'; output.mkdir(); model=tmp_path/'model'; model.write_bytes(b'weights')
    canonical=output/'a.json'; canonical.write_text(json.dumps({'cells':[{'type':1}]}))
    (output/'a.completion.json').write_text(json.dumps({'json_sha256':sha256_file(canonical)}))
    canonical.write_text(json.dumps({'cells':[]}))
    argv=['batch','--images-dir',str(images),'--output-dir',str(masks),'--json-output-dir',str(output),'--raw-outdir',str(tmp_path/'raw'),'--model',str(model),'--skip-existing']
    with patch('sys.argv',argv), patch.object(producer.subprocess,'run') as runner:
        with pytest.raises(ValueError,match='independent completion'):
            producer.main()
        runner.assert_not_called()


def test_duplicate_and_reversed_target_strengths_rejected(tmp_path):
    def asset(p): return {'status':'frozen','path':str(p),'sha256':sha256_file(p)}
    roi=tmp_path/'roi.png'; Image.new('L',(8,8),1).save(roi)
    nuclei=tmp_path/'nuclei.png'; Image.new('L',(8,8)).save(nuclei)
    handoff=tmp_path/'handoff.json'; handoff.write_text(json.dumps({'execution_contract':{'joint_area_budget':{'floor':.1}}}))
    tissue=tmp_path/'target.png'; Image.new('L',(8,8),1).save(tissue)
    row={'joint_handoff_manifest':str(handoff),'target_tissue_mask':str(tissue),'target_nuclei_mask':str(nuclei)}
    states={f'u1_{s}':dict(row) for s in ('moderate','significant')}
    frozen={s:{'handoff':asset(handoff),'execution_budget':{'joint_area_budget':{'floor':.1},'cell_count_extent_budget':None},'target_sha256':[sha256_file(tissue),sha256_file(nuclei)]} for s in states}
    op={'code':'u1','target_parent':1,'direction':1,'minimum_target_gap':.1,'strengths':{s:{'dose_interval':[0.,1.]} for s in ('moderate','significant')}}
    with pytest.raises(ValueError, match='duplicate or unordered'):
        validate_target_pair(states,op,frozen,{'u1_roi':asset(roi)})


def test_frozen_but_inconsistent_protocol_never_calls_generator(tmp_path):
    from scripts.run_manuscript_global_cohort import main
    from scripts import run_embedding_utility_generation as utility
    manifest=tmp_path/'manifest'; manifest.write_text(json.dumps({'prompt':'different','generation_seed':42})+'\n')
    protocol={'status':'frozen','formal_launch_permitted':True,'blockers':[],
        'generation_manifest_sha256':sha256_file(manifest),'endpoint_stage':'postprocessed_precomposite',
        'cluster_unit':'patient_id','sampling':{'prompt':'frozen prompt','seed':42},'cohort_selection':{'formal_source_list_frozen':False}}
    p=tmp_path/'protocol'; p.write_text(json.dumps(protocol))
    argv=['--protocol',str(p),'--cohort-manifest',str(manifest),'--output-root',str(tmp_path/'out')]
    with patch.object(utility,'main') as generate:
        with pytest.raises(ValueError,match='source list'):
            main(argv)
        with patch('phase3_mask_edit.benchmark.manuscript_contract.preflight', return_value={}):
            with pytest.raises(ValueError,match='prompt differs'):
                main(argv)
        generate.assert_not_called()


def test_dirty_runtime_rejected_even_with_matching_head_and_assets(tmp_path):
    from phase3_mask_edit.benchmark.manuscript_contract import ASSETS
    manifest=tmp_path/'manifest'; manifest.write_text('{}')
    asset={'status':'frozen','path':str(manifest),'sha256':sha256_file(manifest)}
    protocol={'cohort_selection':{'formal_source_list_frozen':True},'formal_assets':{key:asset for key in ASSETS},'implementation_commit':'matching'}
    with patch('subprocess.check_output',side_effect=['matching\n',' M scripts/runtime.py\n']):
        with pytest.raises(ValueError,match='dirty'):
            preflight(protocol,manifest,tmp_path)


def test_target_doses_must_increase_from_source_then_between_strengths(tmp_path):
    def asset(p): return {'status':'frozen','path':str(p),'sha256':sha256_file(p)}
    roi=tmp_path/'roi.png'; Image.new('L',(8,8),1).save(roi)
    source=tmp_path/'source.png'; Image.new('L',(8,8),2).save(source)
    nuclei=tmp_path/'nuclei.png'; Image.new('L',(8,8)).save(nuclei)
    handoff=tmp_path/'handoff'; handoff.write_text(json.dumps({'execution_contract':{'joint_area_budget':{'floor':.1}}}))
    states={'sham':{'reference_tissue_mask':str(source)}}; frozen={}
    op={'code':'u1','target_parent':1,'direction':1,'minimum_target_gap':.1,'strengths':{s:{'dose_interval':[0.,1.]} for s in ('moderate','significant')}}
    for strength,count in [('moderate',16),('significant',48)]:
        target=tmp_path/(strength+'.png'); values=np.full(64,2,dtype=np.uint8); values[:count]=1; Image.fromarray(values.reshape(8,8)).save(target)
        state='u1_'+strength; states[state]={'joint_handoff_manifest':str(handoff),'target_tissue_mask':str(target),'target_nuclei_mask':str(nuclei)}
        frozen[state]={'handoff':asset(handoff),'execution_budget':{'joint_area_budget':{'floor':.1},'cell_count_extent_budget':None},'target_sha256':[sha256_file(target),sha256_file(nuclei)]}
    validate_target_pair(states,op,frozen,{'u1_roi':asset(roi)})
    Image.new('L',(8,8),1).save(source)
    with pytest.raises(ValueError,match='direction from source'):
        validate_target_pair(states,op,frozen,{'u1_roi':asset(roi)})


def test_dependency_drift_rejected_after_clean_checkout_check(tmp_path):
    import sys, importlib.metadata
    from phase3_mask_edit.benchmark.manuscript_contract import ASSETS
    manifest=tmp_path/'manifest'; manifest.write_text('{}')
    asset={'status':'frozen','path':str(manifest),'sha256':sha256_file(manifest)}
    packages={k:importlib.metadata.version(k) for k in ('torch','numpy','scipy','Pillow','torchvision')}
    packages.update(diffusers='frozen-unavailable',transformers='frozen-unavailable',safetensors='frozen-unavailable')
    packages['torch']='wrong-version'
    protocol={'cohort_selection':{'formal_source_list_frozen':True},'formal_assets':{k:asset for k in ASSETS},
        'implementation_commit':'matching','runtime_environments':{'generation_and_cellvit':{'python_path':sys.executable,'packages':packages}}}
    with patch('subprocess.check_output',side_effect=['matching\n','']):
        with pytest.raises(ValueError,match='dependencies differ'):
            preflight(protocol,manifest,tmp_path)
