"""Synthetic joint assets exercise the real ledger, handoff, and entry guards."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import binary_dilation
from phase3_mask_edit.benchmark.experiment_identity import sha256_file, digest_payload, asset_identity
from phase3_mask_edit.benchmark.manuscript_contract import preflight, target_dose
from phase3_joint_edit_refine.ledger import analyze_joint_change
from tests.test_joint_generator_ablation import _write_handoff


def save(path, array):
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)
    return str(path)


def frozen(path):
    return {'status':'frozen','path':str(path),'sha256':sha256_file(path)}


@pytest.fixture
def cohort(tmp_path):
    source = np.full((64,64),2,np.uint8); source[:,1:16]=15
    source[:,0]=0; source[:,-1]=0
    nuclei=np.zeros_like(source); nuclei[8:12,8:12]=101; nuclei[45:48,6:9]=101
    footprint=np.zeros_like(source,dtype=bool); footprint[6:18,6:18]=True
    retained=nuclei>0; retained[8:12,8:12]=False
    image=tmp_path/'TCGA-X0-0001_source.png'; save(image,np.full((64,64,3),128,np.uint8))
    tissue=tmp_path/'source.png'; save(tissue,source)
    cells=tmp_path/'cells.png'; save(cells,nuclei)
    eligibility=tmp_path/'eligibility.png'; save(eligibility,np.full_like(source,255))
    common={'reference_id':'source','patient_id':'TCGA-X0-0001','wsi_id':'slide-1','profile':'BCSS','prompt':'frozen source prompt','generation_seed':42,
        'reference_image':str(image),'reference_tissue_mask':str(tissue),'reference_nuclei_mask':str(cells),'postprocess_eligibility_region':str(eligibility)}
    rows=[dict(common,state='sham',sample_id='sham',target_tissue_mask=str(tissue),target_nuclei_mask=str(cells))]
    protocol={'source_count':1,'operations':{}}
    targets={}; union=np.zeros_like(source,dtype=bool); raw_union=union.copy()
    for code,parent,primitive,mechanism in [('u1',1,'cohesive-boundary-expansion-v1','breast-annotation-anchored-boundary-growth'),('u2',4,'generic-immune-infiltrate-increase-v1','breast-generic-immune-compartment-turnover')]:
        protocol['operations'][code]={'code':code,'target_parent':parent,'primitive_id':primitive,'mechanism_id':mechanism,'direction':1,'minimum_target_gap':.05,'strengths':{s:{'dose_interval':[0.,1.]} for s in ('moderate','significant')}}
        for strength,width in [('moderate',8),('significant',16)]:
            state=code+'_'+strength; folder=tmp_path/state; folder.mkdir()
            manifest_path=_write_handoff(folder); h=json.loads(manifest_path.read_text())
            target=source.copy()
            if code=='u1': target[:,16:16+width]=1
            else: target[:,40:40+width]=4
            target_cells=nuclei.copy(); target_cells[8:12,8:12]=0; target_cells[target_cells==101]=1
            analysis=analyze_joint_change(source_tissue=source,target_tissue=target,source_nuclei=nuclei,target_nuclei=target_cells,generation_halo_px=0,
                source_instance_masks={'deleted':footprint,'retained':retained},source_instance_classes={'deleted':1,'retained':1},erased_source_instance_ids=('deleted',))
            union |= analysis.joint_change
            # Compare against normalized raw differences, so the counterexample
            # proves complete-footprint loss independently of encoding changes.
            raw_union |= (source!=target) | ((nuclei % 100)!=target_cells)
            for name,path in h['paths'].items():
                if name=='executable_contract': continue
                array={'target_tissue_mask':target,'target_nuclei_mask':target_cells,'tissue_change':analysis.tissue_change,'cell_change':analysis.cell_change,'joint_change':analysis.joint_change,'generation_support':analysis.generation_support}.get(name,np.zeros_like(source))
                if array.dtype==bool: array=array.astype(np.uint8)*255
                save(Path(path),array)
            h['source_assets']={'image':str(image),'tissue':str(tissue),'nuclei':str(cells)}
            h['primitive_id']=primitive;h['mechanism_id']=mechanism
            h['ledger']={'joint_fraction':float(analysis.joint_change.mean()),'generation_support_fraction':float(analysis.generation_support.mean())}
            for name,path in h['paths'].items(): h['digests'][name+'_sha256']=sha256_file(path)
            for key,path in [('source_image_sha256',image),('source_tissue_mask_sha256',tissue),('source_nuclei_mask_sha256',cells)]:h['digests'][key]=sha256_file(path)
            binding=h['result_binding']
            for key in list(binding):
                if key in h['digests']: binding[key]=h['digests'][key]
            binding['target_tissue_sha256']=h['digests']['target_tissue_mask_sha256']; binding['target_nuclei_sha256']=h['digests']['target_nuclei_mask_sha256']
            binding.pop('binding_id');binding['binding_id']=digest_payload(binding)
            h['execution_contract']['joint_area_budget']={'floor':.1}
            manifest_path.write_text(json.dumps(h))
            rows.append(dict(common,state=state,sample_id=state,target_tissue_mask=h['paths']['target_tissue_mask'],target_nuclei_mask=h['paths']['target_nuclei_mask'],generation_change_region=h['paths']['generation_support'],joint_handoff_manifest=str(manifest_path)))
            targets[state]={'handoff':frozen(manifest_path),'execution_budget':{'joint_area_budget':{'floor':.1},'cell_count_extent_budget':None},'target_sha256':[sha256_file(h['paths'][key]) for key in ('target_tissue_mask','target_nuclei_mask')]}
    rois={}
    for name,array in [('u1_roi',(source>0)&(source!=255)),('u2_roi',source==2),('protected_roi',(source>0)&(source!=255)&~binary_dilation(union,iterations=8))]:
        path=tmp_path/(name+'.png');save(path,array.astype(np.uint8)*255);rois[name]=frozen(path)
    assets={'approved_joint_targets':{'source':targets},'protected_roi_files_and_hashes':{'source':rois},'patient_wsi_mapping_audit':{'source':{'patient_id':'TCGA-X0-0001','wsi_id':'slide-1'}},'pilot_exclusion_audit':{'excluded_reference_ids':['pilot']},'training_overlap_audits':{'source':{'status':'completed','generator':{},'segmentator':{},'cellvit':{}}}}
    # A clean fixture repository supplies the actual git guard; ML packages and
    # generation are explicit CPU stubs, not claims of real model execution.
    repo=tmp_path/'repo';repo.mkdir();subprocess.run(['git','init','-q',str(repo)],check=True)
    (repo/'fixture').write_text('synthetic runtime')
    subprocess.run(['git','-C',str(repo),'add','fixture'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
    protocol.update(status='frozen',formal_launch_permitted=True,blockers=[],implementation_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),cohort_selection={'formal_source_list_frozen':True},endpoint_stage='postprocessed_precomposite',cluster_unit='patient_id',sampling={'prompt':common['prompt'],'seed':42,'dtype':'bf16','steps':28,'guidance':3.5,'conditioning_scale':1.},runtime_environments={'generation_and_cellvit':{'python_path':sys.executable,'packages':{k:'fixture' for k in ('torch','numpy','scipy','Pillow','diffusers','transformers','safetensors','torchvision')}}})
    weights=tmp_path/'weights';weights.write_bytes(b'CPU model stub');protocol['model_assets']={k:asset_identity(weights) for k in ('base_model','cross','pix2pix')}
    return tmp_path,repo,rows,protocol,assets,raw_union


def run_entry(fixture):
    from scripts.run_manuscript_global_cohort import main
    from scripts import run_embedding_utility_generation as generation
    root,repo,rows,protocol,assets,_=fixture
    manifest=root/'manifest.jsonl';manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    protocol['generation_manifest_sha256']=sha256_file(manifest)
    protocol['formal_assets']={'generation_manifest':frozen(manifest)}
    for key,value in assets.items():
        path=root/(key+'.json');path.write_text(json.dumps(value));protocol['formal_assets'][key]=frozen(path)
    p=root/'protocol.json';p.write_text(json.dumps(protocol))
    def validate(p,m,r):
        with patch('importlib.metadata.version',return_value='fixture'):
            return preflight(p,m,repo)
    with patch('phase3_mask_edit.benchmark.manuscript_contract.preflight',side_effect=validate),patch.object(generation,'main',return_value=0) as generate:
        try: main(['--protocol',str(p),'--cohort-manifest',str(manifest),'--output-root',str(root/'out')])
        except ValueError:
            generate.assert_not_called();assert not (root/'out').exists();raise
        generate.assert_called_once()
    return json.loads((root/'out/cohort_roi_validation.json').read_text())


def test_complete_five_states_pass_real_handoff_and_roi_preflight(cohort):
    report=run_entry(cohort)
    assert report['source']['u1_roi']['denominator_pixels']==64*62
    assert report['source']['u2_roi']['denominator_pixels']==64*47
    assert len(report['source']['protected_roi']['binary_mask_sha256'])==64


@pytest.mark.parametrize('name',['u1_roi','u2_roi'])
def test_hash_correct_cropped_source_roi_stops_before_generation(cohort,name):
    root,_,_,_,assets,_=cohort
    path=root/(name+'.png');roi=np.asarray(Image.open(path)).copy();roi[:,:48]=0;save(path,roi)
    assets['protected_roi_files_and_hashes']['source'][name]=frozen(path)
    with pytest.raises(ValueError,match='measurement ROI differs'):
        run_entry(cohort)


def test_normalized_raw_difference_protection_is_rejected(cohort):
    root,_,rows,_,assets,raw=cohort
    source=np.asarray(Image.open(rows[0]['reference_tissue_mask']))
    path=root/'protected_roi.png';approved=np.asarray(Image.open(path))>0
    wrong=(source>0)&(source!=255)&~binary_dilation(raw,iterations=8)
    assert np.any(wrong!=approved)
    save(path,wrong.astype(np.uint8)*255);assets['protected_roi_files_and_hashes']['source']['protected_roi']=frozen(path)
    with pytest.raises(ValueError,match='protected ROI differs'):
        run_entry(cohort)


def test_ignore_outside_source_roi_allowed_but_inside_is_invalid(cohort):
    _,_,rows,protocol,assets,_=cohort
    row=rows[1];op=protocol['operations']['u1'];roi=assets['protected_roi_files_and_hashes']['source']['u1_roi']
    # Historical/readout assets may retain ignore outside the fixed ROI.
    source_path=Path(row['reference_tissue_mask']); source=np.asarray(Image.open(source_path)).copy(); source[:,-1]=255;save(source_path,source)
    target=np.asarray(Image.open(row['target_tissue_mask'])).copy();target[:,-1]=255;save(Path(row['target_tissue_mask']),target)
    assert target_dose(row,op,roi)>0
    target=np.asarray(Image.open(row['target_tissue_mask'])).copy();target[30,30]=255;save(Path(row['target_tissue_mask']),target)
    with pytest.raises(ValueError,match='ignore'):
        target_dose(row,op,roi)


def test_formal_generation_rejects_ignore_before_model_call(cohort):
    _,_,rows,_,_,_=cohort
    source=Path(rows[0]['reference_tissue_mask']); values=np.asarray(Image.open(source)).copy();values[:,-1]=255;save(source,values)
    with pytest.raises(ValueError,match='formal generation tissue conditions'):
        run_entry(cohort)
