import json
import numpy as np
import pytest
from PIL import Image
from scripts.prepare_manuscript_joint_assets import prepare


def test_freeze_keeps_technical_patient_outside_formal_and_preserves_budgets(tmp_path):
    rows=[]
    for i in range(3):
        name=f'TCGA-AA-{i:04d}'
        image=tmp_path/(name+'.png');Image.new('RGB',(8,8)).save(image)
        tissue=tmp_path/(name+'_tissue.png');array=np.full((8,8),2,np.uint8);array[:2]=1;array[2:4]=4;Image.fromarray(array).save(tissue)
        nuclei=tmp_path/(name+'_nuclei.png');Image.new('L',(8,8)).save(nuclei)
        rows.append(dict(sample_id=name,reference_id=name,patient_id=name,wsi_id=name,reference_image=str(image),reference_tissue_mask=str(tissue),reference_nuclei_mask=str(nuclei)))
    rules=prepare(rows,tmp_path/'freeze',formal_count=1)
    assert rules['technical_source']['patient_id']!=rules['formal_sources'][0]['patient_id']
    cases=json.loads((tmp_path/'freeze/technical_joint_cases.json').read_text())
    assert len(cases)==4
    assert {c['joint_area_budget']['target_fraction'] for c in cases}=={.19,.32}
    assert {c['joint_area_budget']['capacity_floor_policy'] for c in cases}=={'strict'}
    assert not rules['formal_generation_launched']
    assert not rules['formal_source_list_frozen']
    with pytest.raises(FileExistsError):prepare(rows,tmp_path/'freeze',formal_count=1)


def test_overlap_index_matches_exact_filenames_and_patient_groups():
    from scripts.audit_manuscript_training_overlap import index_strings
    names,patients=index_strings({'train':[{'reference_image':r'images\TCGA-AA-0001_roi.png'}, {'image':'/data/TCGA-BB-0002_other.png'}]})
    assert names=={'TCGA-AA-0001_roi.png','TCGA-BB-0002_other.png'}
    assert patients=={'TCGA-AA-0001','TCGA-BB-0002'}


def test_capacity_screen_rejects_changed_frozen_source(tmp_path):
    from scripts.probe_manuscript_joint_capacity import verify_source_inputs
    from phase3_mask_edit.benchmark.experiment_identity import asset_identity
    image=tmp_path/'image.png';Image.new('RGB',(8,8)).save(image)
    source={'inputs':{field:asset_identity(image) for field in
        ('reference_image','reference_tissue_mask','reference_nuclei_mask')}}
    verify_source_inputs(source)
    Image.new('RGB',(8,8),'red').save(image)
    with pytest.raises(ValueError,match='frozen source content changed'):
        verify_source_inputs(source)


def test_screen_timeout_retains_partial_artifacts_and_is_not_success(tmp_path):
    import sys
    from scripts.run_manuscript_source_qualification import run_one
    partial=tmp_path/'partial.json'
    code='import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text("partial"); time.sleep(10)'
    receipt=run_one([sys.executable,'-c',code,str(partial)],
        log_path=tmp_path/'screen.log',seconds=.5)
    assert receipt['timed_out'] is True
    assert receipt['controller_exit_code']==124
    assert partial.read_text()=='partial'
