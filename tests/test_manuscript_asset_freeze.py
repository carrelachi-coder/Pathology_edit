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
    with pytest.raises(FileExistsError):prepare(rows,tmp_path/'freeze',formal_count=1)


def test_overlap_index_matches_exact_filenames_and_patient_groups():
    from scripts.audit_manuscript_training_overlap import index_strings
    names,patients=index_strings({'train':[{'reference_image':r'images\TCGA-AA-0001_roi.png'}, {'image':'/data/TCGA-BB-0002_other.png'}]})
    assert names=={'TCGA-AA-0001_roi.png','TCGA-BB-0002_other.png'}
    assert patients=={'TCGA-AA-0001','TCGA-BB-0002'}
