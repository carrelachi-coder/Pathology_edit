"""A cohort amendment retains old failures and cannot re-enter technical patients."""
import json
import numpy as np
from PIL import Image
import pytest
from scripts.freeze_manuscript_supplemental_pool import freeze


@pytest.fixture
def sources(tmp_path):
    root=tmp_path/'BCSS_PATCHES'
    for folder in ('images','tissue_masks','nuclei_masks'):(root/folder).mkdir(parents=True)
    rows=[]
    for patient,suffix in [('0001','a'),('0001','b'),('0001','c'),('0002','a'),('0003','a'),('0004','a')]:
        name=f'TCGA-AA-{patient}_{suffix}.png'
        tissue=np.full((512,512),2,np.uint8);tissue[:32]=1;tissue[32:64]=4
        if patient=='0003':tissue[:]=2
        if patient=='0004':tissue[:]=1;tissue[:128]=2;tissue[128:132]=4
        Image.new('RGB',(512,512),'white').save(root/'images'/name)
        Image.fromarray(tissue).save(root/'tissue_masks'/name)
        Image.new('L',(512,512)).save(root/'nuclei_masks'/name)
        rows.append(dict(dataset_id='bcss',dataset_root=str(root),group_id=f'TCGA-AA-{patient}',
            image=name,mask=name,nuclei=name,images_dir='images',masks_dir='tissue_masks',nuclei_dir='nuclei_masks'))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'train':rows,'val':[],'test':[]}))
    initial=tmp_path/'initial.json';initial.write_text(json.dumps({'candidate_sources':[
        {'inputs':{'reference_image':{'path':str(root/'images/TCGA-AA-0001_a.png')}}}]}))
    excluded=tmp_path/'excluded.json';excluded.write_text(json.dumps({'excluded_patient_ids':['TCGA-AA-0002']}))
    return manifest,initial,excluded,root,tmp_path/'new_census'


def test_append_order_cap_exclusion_and_old_denominator_are_preserved(sources):
    manifest,initial,excluded,root,output=sources;before=initial.read_bytes()
    result=freeze(*sources,per_patient=1)
    assert initial.read_bytes()==before
    assert result['candidate_count']==result['patient_count']==1
    assert result['candidate_sources'][0]['inputs']['reference_image']['path'].endswith('TCGA-AA-0001_b.png')
    assert result['raw_unique_images']==6
    assert result['screen_counts']=={'already_in_initial_census':1,'candidate_for_expensive_qualification':1,
        'unexamined_after_per_patient_cap':1,'excluded_technical_patient':1,
        'missing_source_tumor_stroma_or_immune':1,'stroma_below_27_percent_tissue_floor':1}
    assert result['joint_budgets']==[.19,.32]
    assert result['formal_generation_launched'] is False
    with pytest.raises(FileExistsError):freeze(*sources,per_patient=1)


def test_supplement_cannot_silently_change_dataset_or_patient_identity(sources):
    manifest,_,_,_,output=sources
    data=json.loads(manifest.read_text());data['train'][0]['dataset_root']='/another/export'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='same canonical BCSS export'):freeze(*sources)
    assert not (output/'supplemental_candidate_pool.json').exists()


def test_all_remaining_appends_later_eligible_sources_without_changing_budget(sources):
    result=freeze(*sources,per_patient=None)
    assert result['candidate_count']==2
    assert result['per_patient_cap'] is None
    assert 'unexamined_after_per_patient_cap' not in result['screen_counts']
    assert [s['inputs']['reference_image']['path'].split('_')[-1] for s in result['candidate_sources']]==['b.png','c.png']
    assert result['joint_budgets']==[.19,.32]
