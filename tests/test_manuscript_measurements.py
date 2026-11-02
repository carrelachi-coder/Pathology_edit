"""Readout counterexamples distinguish total cells, typed density, and absence."""
from dataclasses import asdict
import json
import numpy as np
from PIL import Image
import pytest
from phase3_mask_edit.benchmark.experiment_identity import sha256_file
from phase3_mask_edit.benchmark.manuscript_measurements import (
    inflammatory_density,tissue_readout,verify_postprocess,
)


@pytest.fixture
def detection(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(8,8),'white').save(image)
    mask=tmp_path/'mask.png';Image.new('L',(8,8)).save(mask)
    canonical=tmp_path/'cells.json';model='a'*64
    payload={'cells':[{'type':1,'centroid':[1,1]},
        {'type':2,'contour':[[0,0],[2,0],[2,2],[0,2]]},
        {'type':2,'centroid':[6,1]}],
        'benchmark_provenance':{'status':'completed','image_sha256':sha256_file(image),
            'mask_sha256':sha256_file(mask),'model_sha256':model,'mpp':.25}}
    def write():
        canonical.write_text(json.dumps(payload))
        canonical.with_suffix('.completion.json').write_text(json.dumps({
            'json_sha256':sha256_file(canonical),'mask_sha256':sha256_file(mask),
            'input_sha256':sha256_file(image),'model_sha256':model}))
    write();roi=np.zeros((8,8),bool);roi[:,:4]=True
    def measure():return inflammatory_density(canonical,mask,image,roi,model_sha256=model,mpp=.25)
    return payload,write,measure,canonical


def test_type_two_density_uses_fixed_roi_not_total_detected_cells(detection):
    _,_,measure,_=detection;r=measure()
    assert r['total_detected_cells']==3
    assert r['type2_raw_candidates']==2
    assert r['count']==1
    assert r['denominator_pixels']==32
    assert r['value']==pytest.approx(1/(32*(.25/1000)**2))


def test_empty_completed_detection_is_valid_zero(detection):
    payload,write,measure,_=detection;payload['cells']=[];write();r=measure()
    assert r['status']=='valid_zero_with_completion'
    assert r['value']==0 and r['count']==0


def test_missing_or_changed_detection_cannot_be_measured_zero(detection):
    _,_,measure,path=detection
    path.write_text('{}')
    with pytest.raises(ValueError,match='independent completion'):measure()
    path.with_suffix('.completion.json').unlink()
    with pytest.raises(FileNotFoundError):measure()


def test_unlocalizable_type_two_is_uncertain_not_zero(detection):
    payload,write,measure,_=detection;payload['cells']=[{'type':2}];write();r=measure()
    assert r['status']=='evaluator_uncertain'
    assert r['value'] is None and r['count'] is None


def test_offset_corrected_polygon_area_centroid(detection):
    payload,write,measure,_=detection
    payload['wsi_metadata']={'patch_size':10}
    payload['cells']=[{'type':2,'offset_global':[1,1],
        'contour':[[3,3],[4,3],[5,3],[5,5],[3,5]]}]
    write();assert measure()['count']==1


def test_no_patch_size_uses_the_same_offset_orientation_as_the_rasterizer(detection):
    payload,write,measure,_=detection
    payload['cells']=[{'type':2,'offset_global':[1,5],
        'contour':[[1,5],[3,5],[3,7],[1,7]]}]
    write();assert measure()['count']==1


def test_batch_conversion_cannot_import_a_different_live_checkout(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    import batch_cellvit_ui_equiv as producer
    monkeypatch.setitem(sys.modules,'scripts.run_cellvit_single_patch',SimpleNamespace(
        rasterize_cells_json=lambda *_:np.full((8,8),255,np.uint8)))
    image=tmp_path/'image.png';Image.new('RGB',(8,8)).save(image)
    canonical=tmp_path/'cells.json';canonical.write_text(json.dumps({'cells':[
        {'type':2,'contour':[[1,1],[3,1],[3,3],[1,3]]}]}))
    result=producer.rasterize_cells_json(canonical,image)
    assert set(np.unique(result))=={0,102}


def test_tissue_roi_stays_fixed_and_ignore_is_not_zero():
    mask=np.zeros((8,8),np.uint8);mask[0,:]=1;roi=np.zeros_like(mask,bool);roi[:,:4]=True
    assert tissue_readout(mask,roi,1)['value']==4/32
    mask[0,0]=255;r=tissue_readout(mask,roi,1)
    assert r['status']=='evaluator_uncertain' and r['value'] is None


@pytest.fixture
def postprocess(tmp_path):
    from controlnet_train.pix2pix_transfer.inference import (
        apply_cross_low_stain_protection,CrossLowStainProtectionConfig,
    )
    config=CrossLowStainProtectionConfig();ones=np.ones((32,32),bool);zeros=np.zeros_like(ones,np.uint8)
    _,alpha,proof=apply_cross_low_stain_protection(cross_image=Image.new('RGB',(32,32),'white'),
        pix2pix_image=Image.new('RGB',(32,32),(200,200,200)),target_nuclei_mask=zeros,
        protection_region=ones,config=config)
    p=tmp_path/'alpha.png';alpha.save(p);proof['mask']=str(p)
    eligibility=tmp_path/'eligibility.png';Image.fromarray(ones.astype(np.uint8)*255).save(eligibility)
    nuclei=tmp_path/'nuclei.png';Image.fromarray(zeros).save(nuclei)
    return {'cross_v1':{'pix2pix_v2':{'cross_rgb_od_low_stain_protection':proof}}}, {'postprocess_eligibility_region':str(eligibility),'target_nuclei_mask':str(nuclei)},asdict(config)


def test_actual_postprocess_rasters_match_frozen_configuration(postprocess):
    info,row,config=postprocess;r=verify_postprocess(info,row,expected_config=config)
    assert r['alpha_nonzero_pixels']>0


def test_same_policy_name_cannot_hide_changed_actual_region(postprocess):
    info,row,config=postprocess
    Image.new('L',(32,32)).save(row['postprocess_eligibility_region'])
    with pytest.raises(ValueError,match='mask frames'):verify_postprocess(info,row,expected_config=config)


def test_actual_alpha_or_nucleus_exclusion_drift_is_rejected(postprocess):
    info,row,config=postprocess
    mask=np.asarray(Image.open(row['target_nuclei_mask'])).copy();mask[16,16]=101
    Image.fromarray(mask).save(row['target_nuclei_mask'])
    with pytest.raises(ValueError,match='raster identity'):verify_postprocess(info,row,expected_config=config)


@pytest.fixture
def generation(postprocess,tmp_path):
    from phase3_mask_edit.benchmark.experiment_identity import asset_identity
    from scripts.measure_manuscript_joint_technical import verified_generation
    protection,row,config=postprocess
    image=tmp_path/'source.png';Image.new('RGB',(32,32),'white').save(image)
    tissue=tmp_path/'tissue.png';Image.new('L',(32,32),2).save(tissue)
    support=tmp_path/'support.png';Image.new('L',(32,32)).save(support)
    row.update(sample_id='synthetic',reference_image=str(image),reference_tissue_mask=str(tissue),
        reference_nuclei_mask=row['target_nuclei_mask'],target_tissue_mask=str(tissue),
        generation_change_region=str(support),generation_seed=42)
    directory=tmp_path/'cross-v1/synthetic';directory.mkdir(parents=True)
    endpoint=directory/'generated_image_precomposite.png';Image.new('RGB',(32,32),(200,200,200)).save(endpoint)
    composite=directory/'generated_image.png';Image.new('RGB',(32,32),'white').save(composite)
    info_path=directory/'generation_info.json'
    info={**protection,'real_generation_completed':True,'generation_mode':'cross-v1',
        'endpoint_stage':'postprocessed_precomposite','endpoint_image':str(endpoint),'prompt':'frozen prompt'}
    info['cross_v1'].update(num_inference_steps=28,guidance_scale=3.5,controlnet_conditioning_scale=1.,seed=42)
    model=tmp_path/'dummy_weights';model.write_bytes(b'explicit bookkeeping fixture, no model call')
    model_asset=asset_identity(model)
    protocol={'generation_manifest_sha256':'a'*64,'model_assets':{key:model_asset for key in ('base_model','cross','pix2pix')},
        'sampling':{'prompt':'frozen prompt','steps':28,'guidance':3.5,'conditioning_scale':1.,'seed':42,'dtype':'bf16'},
        'postprocess':{'configuration':config}}
    identity={'row':row,'code':'synthetic-code','cohort_manifest':{'sha256':'a'*64},
        'inputs':{key:asset_identity(row[key]) for key in ('reference_image','reference_tissue_mask','reference_nuclei_mask','target_tissue_mask','target_nuclei_mask')},
        'support':asset_identity(support),'postprocess_eligibility':asset_identity(row['postprocess_eligibility_region']),
        'models':{str(model):model_asset},'parameters':{'backend':'cross-v1','endpoint_stage':'postprocessed_precomposite','dtype':'bf16',
            'num_inference_steps':'28','guidance_scale':'3.5','controlnet_conditioning_scale':'1.0'}}
    record={'identity':identity,'status':'completed','outputs':{}}
    def write():
        info_path.write_text(json.dumps(info))
        record['outputs']={p.name:sha256_file(p) for p in (info_path,endpoint,composite)}
        (directory/'experiment_identity.json').write_text(json.dumps(record))
    write()
    def verify():return verified_generation(row,tmp_path,protocol,'synthetic-code')
    return info,record,write,verify,endpoint,row


def test_exact_precomposite_record_passes_consumer_bookkeeping(generation):
    _,_,_,verify,endpoint,_=generation
    actual,_=verify();assert actual==endpoint


def test_composite_cannot_replace_global_sham_endpoint_even_after_rehash(generation):
    info,_,write,verify,endpoint,_=generation
    info['endpoint_stage']='source_composite';info['endpoint_image']=str(endpoint.with_name('generated_image.png'));write()
    with pytest.raises(ValueError,match='precomposite endpoint'):verify()


def test_failed_generation_cannot_be_measured_from_leftover_png(generation):
    _,record,write,verify,_,_=generation;record['status']='failed';write()
    with pytest.raises(ValueError,match='incomplete'):verify()


def test_effective_seed_drift_is_rejected_after_metadata_rehash(generation):
    info,_,write,verify,_,_=generation;info['cross_v1']['seed']=43;write()
    with pytest.raises(ValueError,match='effective generation parameter'):verify()


def test_endpoint_frame_drift_cannot_resize_the_measurement_denominator(generation):
    _,_,write,verify,endpoint,_=generation;Image.new('RGB',(64,64)).save(endpoint);write()
    with pytest.raises(ValueError,match='endpoint frame'):verify()


def test_changed_input_cannot_reuse_old_generated_measurements(generation):
    _,_,_,verify,_,row=generation;Image.new('L',(32,32),1).save(row['reference_tissue_mask'])
    with pytest.raises(ValueError,match='input content drift'):verify()
