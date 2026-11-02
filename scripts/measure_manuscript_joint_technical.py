#!/usr/bin/env python3
"""Verify real five-state generation and measure six images per technical source."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from phase3_joint_edit_refine.generator_adapter import build_frozen_generator_inputs
from phase3_mask_edit.benchmark.experiment_identity import asset_identity,sha256_file,digest_payload
from phase3_mask_edit.benchmark.manuscript_contract import (
    preflight,validate_launch_scope,verified_asset,validate_source_roi,target_dose,
)
from phase3_mask_edit.benchmark.manuscript_measurements import (
    CELL_POINT_POLICY,tissue_readout,inflammatory_density,verify_tissue_prediction,verify_postprocess,
)
from scripts.freeze_manuscript_joint_technical import segmentator_environment
from scripts.run_manuscript_global_cohort import validate_cohort
from scripts.run_manuscript_source_qualification import run_one


def generation_code_identity(repo):
    return digest_payload({str(p.relative_to(repo)):sha256_file(p)
        for folder in ('controlnet_train','scripts','phase3_mask_edit/benchmark')
        for p in sorted((repo/folder).rglob('*.py'))})


def verified_generation(row, root, protocol, code_identity):
    directory=Path(root)/'cross-v1'/row['sample_id']
    saved=json.loads((directory/'experiment_identity.json').read_text());identity=saved['identity']
    if (saved.get('status')!='completed' or identity.get('row')!=row
            or identity.get('code')!=code_identity
            or identity.get('cohort_manifest',{}).get('sha256')!=protocol['generation_manifest_sha256']):
        raise ValueError('generated state is incomplete or has mismatched experiment identity')
    for name in ('reference_image','reference_tissue_mask','reference_nuclei_mask','target_tissue_mask','target_nuclei_mask'):
        if identity['inputs'][name]['sha256']!=sha256_file(row[name]):raise ValueError('generated input content drift')
    for key,field in (('support','generation_change_region'),('postprocess_eligibility','postprocess_eligibility_region')):
        if identity[key]['sha256']!=sha256_file(row[field]):raise ValueError('generation support or postprocess eligibility drift')
    for name in ('base_model','cross','pix2pix'):
        expected=protocol['model_assets'][name]
        if not any(a['sha256']==expected['sha256'] and Path(a['path']).resolve()==Path(expected['path']).resolve()
                   for a in identity['models'].values()):raise ValueError('generation model identity differs')
    sampling=protocol['sampling'];parameters=identity['parameters']
    expected_params={'backend':'cross-v1','endpoint_stage':'postprocessed_precomposite',
        'dtype':sampling['dtype'],'num_inference_steps':sampling['steps'],
        'guidance_scale':sampling['guidance'],'controlnet_conditioning_scale':sampling['conditioning_scale']}
    if any(parameters.get(key)!=str(value) for key,value in expected_params.items()):
        raise ValueError('generation parameters differ from frozen protocol')
    endpoint=directory/'generated_image_precomposite.png';info_path=directory/'generation_info.json'
    for path in (endpoint,info_path,directory/'generated_image.png'):
        if saved.get('outputs',{}).get(path.name)!=sha256_file(path):raise ValueError('generated output content drift')
    info=json.loads(info_path.read_text())
    if (info.get('real_generation_completed') is not True or info.get('endpoint_stage')!='postprocessed_precomposite'
            or Path(info.get('endpoint_image','')).resolve()!=endpoint.resolve()
            or info.get('prompt')!=sampling['prompt'] or info.get('generation_mode')!='cross-v1'):
        raise ValueError('state is not a real global precomposite endpoint')
    effective=info['cross_v1']
    for key,value in (('num_inference_steps',sampling['steps']),('guidance_scale',sampling['guidance']),
                      ('controlnet_conditioning_scale',sampling['conditioning_scale']),('seed',sampling['seed'])):
        if effective.get(key)!=value:raise ValueError('effective generation parameter differs')
    with Image.open(endpoint) as actual,Image.open(row['reference_image']) as source:
        if actual.size!=source.size:raise ValueError('generated endpoint frame differs from the fixed source')
    return endpoint,verify_postprocess(info,row,expected_config=protocol['postprocess']['configuration'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('protocol','cohort-manifest','generation-output','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    protocol=json.loads(args.protocol.read_text());validate_launch_scope(protocol,technical_validation=True)
    assets=preflight(protocol,args.cohort_manifest,repo,technical_validation=True)
    rows=[json.loads(line) for line in args.cohort_manifest.read_text().splitlines() if line.strip()]
    cohort=validate_cohort(rows,source_count=protocol['source_count'],handoff_validator=build_frozen_generator_inputs,
        protocol=protocol,assets=assets,technical_validation=True)
    if protocol['measurement']['cell_point_policy']!=CELL_POINT_POLICY:raise ValueError('unfrozen cell point rule')
    scope=json.loads((args.generation_output/'cohort_scope_validation.json').read_text())
    if (scope.get('scope')!='joint_gpu_technical_acceptance' or scope.get('formal_launch_permitted') is not False
            or scope.get('protocol_sha256')!=sha256_file(args.protocol)
            or scope.get('generation_manifest_sha256')!=sha256_file(args.cohort_manifest)):
        raise ValueError('generation scope/protocol identity differs')
    summary=json.loads((args.generation_output/'cross-v1/generation_summary_shard00-of-01.json').read_text())
    if (summary.get('status')!='complete' or summary.get('requested_count')!=len(rows)
            or summary.get('completed_count')!=len(rows) or summary.get('failure_count')!=0):
        raise ValueError('generation has no complete terminal record')
    args.output.mkdir(parents=True,exist_ok=False);records=[];controller=[]
    report={'scope':'joint_gpu_technical_acceptance','formal_launch_permitted':False,
        'requested_generated':len(rows),'requested_measurements':len(cohort)*6,
        'chain_integrity_accepted':False,'status':'started','records':records,
        'protocol':asset_identity(args.protocol),'generation_scope':asset_identity(args.generation_output/'cohort_scope_validation.json'),
        'code_commit':protocol['implementation_commit'],'controllers':controller,
        'interpretation':'technical identity/completeness only; order, ties and reversals are observations, not acceptance gates'}
    path=args.output/'joint_technical_report.json'
    def save():path.write_text(json.dumps(report,indent=2,allow_nan=False))
    def execute(command,name,*,allow_failure=False):
        result=run_one(command,log_path=args.output/(name+'.log'),seconds=1200)
        controller.append({'stage':name,**result});save()
        if result['controller_exit_code']!=0 and not allow_failure:raise RuntimeError(name+' failed: '+str(result['controller_exit_code']))
        return result
    save()
    try:
        seg=protocol['runtime_environments']['segmentator']
        if segmentator_environment(seg['python_path'])!=seg:raise ValueError('segmentation environment drift')
        package=protocol['cellvit_implementation_asset'];release=protocol['evaluator_release_asset']
        if asset_identity(package['path'])['sha256']!=package['sha256']:raise ValueError('CellViT implementation content drift')
        release_path=verified_asset(release)
        for name in ('segmentator','cellvit'):
            model=protocol['model_assets'][name]
            if sha256_file(model['path'])!=model['sha256']:raise ValueError('evaluator checkpoint content drift')
        images=args.output/'measurement_images';images.mkdir();rois={};source_predictions={}
        code=generation_code_identity(repo)
        for reference,states in cohort.items():
            source=states['sham'];frozen_rois=assets['protected_roi_files_and_hashes'][reference]
            rois[reference]={op:validate_source_roi(source['reference_tissue_mask'],protocol['operations'][op],frozen_rois[op+'_roi']) for op in ('u1','u2')}
            rois[reference]['protected']=np.asarray(Image.open(verified_asset(frozen_rois['protected_roi'])))>0
            endpoints={'original':(Path(source['reference_image']),None,source)}
            for state,row in states.items():
                endpoint,protection=verified_generation(row,args.generation_output,protocol,code)
                endpoints[state]=(endpoint,protection,row)
            for state,(endpoint,protection,row) in endpoints.items():
                image=images/(reference+'__'+state+'.png');shutil.copyfile(endpoint,image)
                records.append({'reference_id':reference,'patient_id':source['patient_id'],'state':state,
                    'image':asset_identity(image),'postprocess':protection,'tissue_status':'missing','cell_status':'missing'})
            save()
        for record in records:
            reference=record['reference_id'];state=record['state'];image=Path(record['image']['path'])
            folder=args.output/'segmentator'/image.stem
            record['tissue_status']='failure';save()
            execute([seg['python_path'],str(repo/'scripts/predict_segmentator_mask.py'),'--release',str(release_path),
                '--input',str(image),'--output-dir',str(folder),'--profile','BCSS','--device',args.device],image.stem+'_tissue')
            predicted=verify_tissue_prediction(folder,image,checkpoint_sha256=protocol['model_assets']['segmentator']['sha256'],
                release_sha256=release['sha256'],code_commit=protocol['implementation_commit'])
            if state=='original':source_predictions[reference]=predicted
            record['tissue_mask']=asset_identity(folder/'coarse_mask.png');record['tissue_status']='completed'
            source=cohort[reference]['sham'];row=cohort[reference].get(state,source)
            record['tissue']={}
            for op in ('u1','u2'):
                measure=tissue_readout(predicted,rois[reference][op],protocol['operations'][op]['target_parent'])
                target=target_dose(row,protocol['operations'][op],assets['protected_roi_files_and_hashes'][reference][op+'_roi'])
                measure.update(target_value=target,signed_target_error=measure['value']-target if measure['value'] is not None else None)
                measure['absolute_target_error']=abs(measure['signed_target_error']) if measure['signed_target_error'] is not None else None
                record['tissue'][op]=measure
            protected=rois[reference]['protected'];n=int(protected.sum())
            with Image.open(image) as im:rgb=np.asarray(im.convert('RGB'),dtype=float)
            with Image.open(source['reference_image']) as im:original=np.asarray(im.convert('RGB'),dtype=float)
            record['preservation']={'status':'completed' if n else 'not_applicable','denominator_pixels':n,
                'coarse_agreement_vs_original_prediction':float(np.mean(predicted[protected]==source_predictions[reference][protected])) if n else None,
                'rgb_mean_absolute_drift_over_255':float(np.abs(rgb-original)[protected].mean()/255.) if n else None}
            save()
        if not args.device.startswith('cuda:'):raise ValueError('real CellViT technical validation requires an explicit CUDA device')
        mpp=protocol['measurement']['nominal_mpp'];cell_model=protocol['model_assets']['cellvit']
        execute([sys.executable,str(repo/'batch_cellvit_ui_equiv.py'),'--images-dir',str(images),
            '--output-dir',str(args.output/'cell_masks'),'--json-output-dir',str(args.output/'cell_json'),
            '--raw-outdir',str(args.output/'cell_raw'),'--model',cell_model['path'],
            '--cellvit-root',str(Path(package['path']).parent),'--gpu',args.device.split(':')[-1],
            '--batch-size','3','--mpp',str(mpp),'--magnification','40','--resolution','0.25'],'cellvit',allow_failure=True)
        if asset_identity(package['path'])['sha256']!=package['sha256']:raise ValueError('CellViT implementation changed during inference')
        for record in records:
            image=Path(record['image']['path']);canonical=args.output/'cell_json'/(image.stem+'.json')
            try:
                cell=inflammatory_density(canonical,args.output/'cell_masks'/(image.stem+'.png'),image,
                    rois[record['reference_id']]['u2'],model_sha256=cell_model['sha256'],mpp=mpp)
                record['cell_status']=cell['status'];record['inflammatory_density']=cell
                record['cell_json']=asset_identity(canonical)
            except FileNotFoundError as exc:record.update(cell_status='missing',cell_error=str(exc))
            except ValueError as exc:record.update(cell_status='failure',cell_error=str(exc))
            save()
        orders=[]
        valid={'valid_nonzero','valid_zero_with_completion'}
        terminal=valid|{'evaluator_uncertain'}
        complete=len(records)==len(cohort)*6 and all(r['tissue_status']=='completed' and r['cell_status'] in terminal
            and all(m['status'] in terminal for m in r['tissue'].values()) for r in records) and all(c['controller_exit_code']==0 for c in controller)
        for reference in cohort:
            states={r['state']:r for r in records if r['reference_id']==reference}
            for op in ('u1','u2'):
                counts=[states[s]['tissue'][op]['count'] for s in ('sham',op+'_moderate',op+'_significant')]
                orders.append({'reference_id':reference,'operation':op,'counts_sham_moderate_significant':counts,
                    'strict_order':counts[0]<counts[1]<counts[2] if all(x is not None for x in counts) else None,
                    'adjacent_tie':any(a==b for a,b in zip(counts,counts[1:])) if all(x is not None for x in counts) else None,
                    'reversal':any(b<a for a,b in zip(counts,counts[1:])) if all(x is not None for x in counts) else None,
                    'sham_minus_original':states['sham']['tissue'][op]['value']-states['original']['tissue'][op]['value']
                        if all(states[s]['tissue'][op]['value'] is not None for s in ('sham','original')) else None})
            cell_counts=[states[s].get('inflammatory_density',{}).get('count') for s in ('sham','u2_moderate','u2_significant')]
            orders.append({'reference_id':reference,'operation':'secondary_inflammatory_cell_density',
                'counts_sham_moderate_significant':cell_counts,
                'strict_order':cell_counts[0]<cell_counts[1]<cell_counts[2] if all(x is not None for x in cell_counts) else None,
                'adjacent_tie':any(a==b for a,b in zip(cell_counts,cell_counts[1:])) if all(x is not None for x in cell_counts) else None,
                'reversal':any(b<a for a,b in zip(cell_counts,cell_counts[1:])) if all(x is not None for x in cell_counts) else None})
        report.update(status='complete' if complete else 'incomplete',chain_integrity_accepted=complete,
            measured_images=len(records),orders=orders,physical_scale_basis=protocol['measurement']['scale_basis'],
            analysis_valid_images=sum(r['cell_status'] in valid and all(m['status'] in valid for m in r['tissue'].values()) for r in records),
            cell_status_counts=dict(Counter(r['cell_status'] for r in records)),
            tissue_status_counts={op:dict(Counter(r['tissue'][op]['status'] for r in records)) for op in ('u1','u2')},
            preservation_status_counts=dict(Counter(r['preservation']['status'] for r in records)),
            acceptance_policy='completed inference and correctly recorded terminal statuses; evaluator uncertainty remains null and is excluded from valid analysis, not converted to zero or used to select a different case')
        save();return 0 if complete else 2
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc));save();raise


if __name__=='__main__':raise SystemExit(main())
