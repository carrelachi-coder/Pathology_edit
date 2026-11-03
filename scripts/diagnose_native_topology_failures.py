#!/usr/bin/env python3
"""Read saved source conditions and error logs. Never run a planner or executor."""
import argparse,ast,csv,hashlib,json,re
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def literals_after(text, start):
    if start>=len(text) or text[start]!='{':raise ValueError('missing dictionary')
    depth=0;quote=None;escaped=False
    for i in range(start,len(text)):
        c=text[i]
        if quote:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c==quote:quote=None
        elif c in ['"',"'"]:quote=c
        elif c=='{':depth+=1
        elif c=='}':
            depth-=1
            if depth==0:
                tree=ast.parse(text[start:i+1],mode='eval')
                class NormalizeInf(ast.NodeTransformer):
                    def visit_Name(self,node):
                        if node.id=='inf':return ast.copy_location(ast.Constant(value=None),node)
                        return node
                return ast.literal_eval(NormalizeInf().visit(tree))
    raise ValueError('unterminated dictionary')


def floor_trials(errors):
    states={};parse_errors=[]
    for error in errors:
        for m in re.finditer(r'minimum=(\d+), realized=(\d+), topology=',error):
            try:
                topology=literals_after(error,m.end())
                state={'minimum_pixels':int(m[1]),'realized_pixels':int(m[2]),'topology':topology}
                fingerprint=json.dumps(state,sort_keys=True)
                states.setdefault(fingerprint,state)
            except (SyntaxError,ValueError) as exc:parse_errors.append(type(exc).__name__+': '+str(exc))
    return list(states.values()),parse_errors


def components(mask):
    labels,n=ndimage.label(mask,structure=np.ones((3,3),bool));sizes=sorted(np.bincount(labels.ravel())[1:].tolist(),reverse=True)
    return {'pixels':int(mask.sum()),'components_8connected':int(n),'largest_component_pixels':max(sizes,default=0),'largest_component_fraction':max(sizes,default=0)/max(1,int(mask.sum())),'component_sizes_pixels':sizes}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['sources','saved-topology','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    sources={s['reference_id']:s for s in json.loads(args.sources.read_text())['candidate_sources']}
    receipts=json.loads(args.saved_topology.read_text());rows=[];source_stats=[]
    for ordinal,receipt in enumerate(receipts):
        source=sources[receipt['reference_id']];arrays={}
        for key in ['reference_image','reference_tissue_mask','reference_nuclei_mask']:
            asset=source['inputs'][key];assert sha(asset['path'])==asset['sha256']
            with Image.open(asset['path']) as im:arrays[key]=np.array(im.convert('RGB') if key=='reference_image' else im)
        tissue=arrays['reference_tissue_mask'];nuclei=arrays['reference_nuclei_mask'];assert tissue.shape==nuclei.shape==(512,512)
        geometry={str(v):components(tissue==v) for v in [0,1,2,4]}
        contact={str(v):int(np.count_nonzero((tissue==2)&ndimage.binary_dilation(tissue==v,structure=np.ones((3,3),bool)))) for v in [1,4]}
        source_stats.append({'source_alias':f'S{ordinal:02d}','reference_id':source['reference_id'],'input_sha256':{k:v['sha256'] for k,v in source['inputs'].items()},'mask_geometry':geometry,'stroma_side_8neighbor_contact_pixels':contact,'nucleus_foreground_pixels':int(np.count_nonzero(nuclei)),'image_path_for_private_plot':source['inputs']['reference_tissue_mask']['path']})
        for op,check in enumerate(receipt['result']['checks']):
            metrics=check['metrics'];witness=metrics['whole_mask_topology_witness'];errors=witness.get('errors',[])
            trials,parse_errors=floor_trials(errors)
            valid=[t for t in trials if t['topology'].get('passed') is True]
            best=max(valid,key=lambda t:t['realized_pixels']) if valid else None
            interfaces=metrics['interface_capacity'];feasible=[i for i in interfaces if i['interface_id'] in metrics['feasible_interface_ids']]
            anchor_reports=[r for i in feasible for r in i.get('anchor_continuity_reports',[])]
            permitted={a for i in feasible for a in i.get('cell_feasible_anchor_segment_ids',[])}
            coverage=[float(r['potential_anchor_coverage_fraction']) for r in anchor_reports if r.get('feasible') and 'potential_anchor_coverage_fraction' in r]
            leaf='topology_safe_area_underfill' if any('TopologySafeAreaUnderfillError' in e for e in errors) else 'no_jointly_feasible_directed_interface' if any('no directed interface satisfies both the tissue and preflight nuclei contracts' in e for e in errors) else 'unresolved_saved_error'
            floor=metrics['meaningful_tissue_floor_pixels'];aggregate=metrics['aggregate_executable_tissue_capacity_pixels'];realized=best['realized_pixels'] if best else None
            row={'source_alias':f'S{ordinal:02d}','reference_id':source['reference_id'],'operation':'U1' if op==0 else 'U2','primitive_id':check['primitive_id'],'terminal_failure':leaf,'hard_tissue_floor_pixels':floor,'aggregate_preflight_capacity_pixels':aggregate,'best_logged_topology_passing_floor_trial_pixels':realized,'deficit_to_floor_pixels':floor-realized if realized is not None else None,'preflight_vs_logged_trial_gap_pixels':aggregate-realized if realized is not None else None,'gap_is_not_a_stepwise_loss_decomposition':True,'feasible_interfaces':len(feasible),'all_preflight_interfaces':len(interfaces),'preflight_cell_feasible_anchor_count':len(permitted),'anchor_preflight_report_count':len(anchor_reports),'anchor_coverage_range_for_feasible_reports':[min(coverage),max(coverage)] if coverage else None,'unique_logged_floor_states':len(trials),'logged_trial_count_is_not_execution_count':True,'parse_errors':parse_errors,'best_logged_topology_state':best['topology'] if best else None,'logged_floor_states':trials,'source_stroma_pixels':geometry['2']['pixels'],'source_stroma_components_8connected':geometry['2']['components_8connected'],'source_largest_stroma_component_pixels':geometry['2']['largest_component_pixels'],'source_largest_stroma_component_fraction':geometry['2']['largest_component_fraction'],'target_seed_pixels':geometry[str(1 if op==0 else 4)]['pixels'],'target_seed_components_8connected':geometry[str(1 if op==0 else 4)]['components_8connected'],'target_seed_largest_component_pixels':geometry[str(1 if op==0 else 4)]['largest_component_pixels'],'source_mask_only_not_failed_target':True,'evidence_gaps':{'failed_target_or_change_mask': 'not_persisted','selected_anchor_masks_per_attempt':'not_persisted','per_attempt_selected_interface_plan':'not_persisted','per_interface_actual_contribution':'structured_exception_feedback_not_retained_by_qualification_wrapper','legal_envelope_and_unique_ownership_counts':'not_persisted','per_guard_rejected_pixel_counters':'not_persisted','exact_stepwise_area_loss_pixels':None},'raw_error_sha256':hashlib.sha256(json.dumps(errors).encode()).hexdigest()}
            rows.append(row)
            patterns=set()
            for trial in trials:
                t=trial['topology']
                if t.get('passed'):continue
                if t.get('source_holes_before')!=t.get('source_holes_after') and not t.get('source_hole_change_allowed_by_selected_target_merge'):
                    patterns.add('source_hole_count_changed_without_recorded_allowance')
                for kind in ['source_components','target_components','target_holes']:
                    if t.get(kind+'_before')!=t.get(kind+'_after'):patterns.add(kind+'_count_changed')
                if t.get('changed_component_area_passed') is False:patterns.add('changed_component_area_check_failed')
            row['observed_invalid_topology_patterns']=sorted(patterns)
    report={'scope':'read-only diagnosis of18 saved source-topology failures; no search, execution or generation','inputs':{'sources_sha256':sha(args.sources),'saved_topology_sha256':sha(args.saved_topology),'analysis_script_sha256':sha(__file__)},'source_count':len(source_stats),'operation_count':len(rows),'source_geometry':source_stats,'diagnoses':rows,'limits':['Only source masks and saved error metadata were analyzed. No failed target mask was reconstructed.','A maximum over unique logged topology-passing floor trials is not a global maximum, an executed attempt count or a final joint J.','Aggregate preflight capacity and logged trial area have different scopes; their difference cannot be attributed additively to individual pipeline stages.','Missing telemetry stays unavailable; zero is not imputed.','No source, anchor or execution path was retried; no thresholds, mechanisms or qualification outcomes changed.'],'batch_compilation_run':False,'formal_he_run':False,'qualification_counts_changed':False}
    (args.output/'diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    fields=['source_alias','operation','terminal_failure','hard_tissue_floor_pixels','aggregate_preflight_capacity_pixels','best_logged_topology_passing_floor_trial_pixels','deficit_to_floor_pixels','source_stroma_components_8connected','source_largest_stroma_component_pixels','target_seed_pixels','target_seed_components_8connected','feasible_interfaces','preflight_cell_feasible_anchor_count']
    with (args.output/'diagnosis_matrix.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({'sources':len(source_stats),'operations':len(rows),'parse_errors':sum(len(r['parse_errors']) for r in rows),'stroma_component_counts':[s['mask_geometry']['2']['components_8connected'] for s in source_stats],'largest_stroma_sizes':[s['mask_geometry']['2']['largest_component_pixels'] for s in source_stats],'best_logged_pixels':[(r['source_alias'],r['operation'],r['best_logged_topology_passing_floor_trial_pixels']) for r in rows]}))
if __name__=='__main__':main()
