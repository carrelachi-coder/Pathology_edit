"""Post-hoc raster measurements; never use gate passed as a quantitative value."""
import json,hashlib,re
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from scipy import ndimage
ROOT=Path('/data1/lyw/pathology_edit_eval/r2_de_terra_six_v4_20260924')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(p):return np.array(Image.open(p))
def measure(row,handoff):
 m=json.loads(Path(handoff).read_text());r=row['record'];p=m['paths'];a=m['source_assets']
 for key,pk in [('image','source_image_sha256'),('tissue','source_tissue_mask_sha256'),('nuclei','source_nuclei_mask_sha256')]:
  assert sha(a[key])==r['provenance'][pk],f'source_hash_mismatch:{key}'
 for key in ['target_tissue_mask','target_nuclei_mask','generation_support']:
  assert sha(p[key])==m['digests'][key+'_sha256'],f'target_hash_mismatch:{key}'
 t0,t1,n0,n1=map(arr,[a['tissue'],p['target_tissue_mask'],a['nuclei'],p['target_nuclei_mask']]);assert t0.shape==t1.shape==n0.shape==n1.shape
 dt=t0!=t1;dn=n0!=n1;raster_j=dt|dn;declared_cell=arr(p['cell_change'])>0;j=dt|declared_cell;g=arr(p['generation_support'])>0;size=t0.size
 profile=json.loads((ROOT/'code/phase3_joint_edit_refine/skills/catalog/annotation-profile'/r['annotation_profile_id']/'references/joint_contract.json').read_text())
 from phase3_mask_edit.core.labels import MaskProfileSchema
 schema=MaskProfileSchema.from_reference_profile(row['dataset'])
 key=m['mechanism_id']+'::'+m['primitive_id'];tc=m['execution_contract']['executable_contract']['tissue_label_contract']
 source_map=profile.get('mechanism_editable_source_fine_ids',{});target_map=profile.get('mechanism_editable_target_fine_ids',{})
 allowed_source=set(source_map.get(key,source_map.get(m['mechanism_id'],[])));allowed_target=set(target_map.get(key,target_map.get(m['mechanism_id'],[])))
 if m['primitive_scope']!='cell_only':
  if not allowed_source:
   for label in tc.get('source_labels',[]):allowed_source.update(schema.resolve_fine_ids(label))
  if not allowed_target and tc.get('target_label'):allowed_target.update(schema.resolve_fine_ids(tc['target_label']))
 if r['annotation_profile_id']=='bcss-semantic-v1':allowed_source.difference_update({14,15});allowed_target.difference_update({14,15})
 protected=np.isin(t0,sorted(set(profile.get('protected_fine_ids',[]))-allowed_source));forbidden=np.isin(t1,profile.get('prohibit_cell_placement_fine_ids',[]))
 vals=sorted(set(np.unique(t0))|set(np.unique(t1)));delta={str(int(v)):int(np.count_nonzero(t1==v)-np.count_nonzero(t0==v)) for v in vals}
 from phase3_mask_edit.core.labels import MaskProfileSchema
 schema=MaskProfileSchema.from_reference_profile(row['dataset']);tumor0=np.isin(t0,schema.tumor_fine_ids);tumor1=np.isin(t1,schema.tumor_fine_ids)
 cc0,k0=ndimage.label(tumor0,np.ones((3,3)));cc1,k1=ndimage.label(tumor1,np.ones((3,3)))
 changecc,kc=ndimage.label(dt,np.ones((3,3)));componentareas=np.bincount(changecc.ravel())[1:]
 budget=m['execution_contract']['joint_area_budget'];cb=m['execution_contract']['cell_count_extent_budget'];ledger=m['ledger'];trace=m['provenance']
 out={'measurement_version':'r2de-raster-v2-authorized-protection','authorized_source_fine_ids':sorted(allowed_source),'authorized_target_fine_ids':sorted(allowed_target),'unauthorized_tissue_transition_pixels':int((dt&(~np.isin(t0,sorted(allowed_source))|~np.isin(t1,sorted(allowed_target)))).sum()),'case_id':r['case_id'],'dataset':row['dataset'],'level':row['level'],'primitive_id':m['primitive_id'],'expected_primitive_id':row['primitive_id'],'primitive_match':m['primitive_id']==row['primitive_id'],'mechanism_id':m['mechanism_id'],'source_group':row['source_group'],'sample_id':row['sample_id'],'source_pixels':size,'tissue_change_pixels':int(dt.sum()),'cell_change_pixels':int(dn.sum()),'raster_label_delta_union_pixels':int(raster_j.sum()),'operational_cell_change_pixels':int(declared_cell.sum()),'nuclear_label_changes_outside_declared_cell_region':int((dn&~declared_cell).sum()),'joint_change_pixels':int(j.sum()),'joint_fraction':float(j.mean()),'G_fraction':float(g.mean()),'change_outside_G_pixels':int((j&~g).sum()),'protected_tissue_pixels':int(protected.sum()),'protected_tissue_changed_pixels':int((dt&protected).sum()),'protected_tissue_change_rate':float((dt&protected).sum()/protected.sum()) if protected.any() else None,'cell_only_tissue_changed_pixels':int(dt.sum()) if m['primitive_scope']=='cell_only' else None,'new_nuclear_pixels_in_prohibited_tissue':int(((n1>0)&(n0!=n1)&forbidden).sum()),'source_tumor_fraction':float(tumor0.mean()),'target_tumor_fraction':float(tumor1.mean()),'tumor_delta_pp':float((tumor1.mean()-tumor0.mean())*100),'source_tumor_components':k0,'target_tumor_components':k1,'tissue_change_components':kc,'dominant_tissue_change_fraction':float(componentareas.max()/dt.sum()) if dt.any() else None,'changed_original_tumor_components':len(set(cc0[dt&tumor0].tolist())-{0}),'fine_label_delta_pixels':delta,'tissue_change_mask_exact':bool(np.array_equal(dt,arr(p['tissue_change'])>0)),'cell_change_mask_exact':bool(np.array_equal(dn,arr(p['cell_change'])>0)),'joint_change_mask_exact':bool(np.array_equal(j,arr(p['joint_change'])>0)),'ledger_joint_pixels_matches_raster':int(ledger['joint_pixels'])==int(j.sum()),'active_rule_ids':m['active_rule_ids'],'skill_versions':m['execution_contract']['executable_contract']['skill_versions'],'handoff':str(handoff)}
 if budget:
  target=budget['target_fraction'];out.update(joint_target_fraction=target,joint_error_pp=100*(j.mean()-target),joint_abs_relative_error=abs(j.mean()-target)/max(target,1e-9),joint_within_declared_range=budget['min_fraction']<=j.mean()<=budget['max_fraction'],joint_budget=budget,original_joint_budget=r['joint_area_budget'])
 # Instance event identities are not independently segmented ground truth. Keep separate from raster metrics.
 added=ledger.get('added_instance_ids',[]);removed=ledger.get('removed_instance_ids',[])
 def cls(i):
  match=re.search(r'nuc-c(\d+)-',i);return str(int(match.group(1))) if match else 'unknown'
 addc=Counter(cls(x) for x in added);remc=Counter(cls(x) for x in removed)
 out.update(added_instance_events=len(added),removed_instance_events=len(removed),net_instance_events=len(added)-len(removed),added_events_by_class=dict(addc),removed_events_by_class=dict(remc),unique_event_ids=len(set(added))==len(added) and len(set(removed))==len(removed),instance_count_measurement_basis='frozen instance event IDs; raster area/preservation independently recomputed; not independent instance segmentation')
 if cb:
  intent=r['prebound_semantic_request']['intents'][0];classid={'neoplastic':'1','inflammatory':'2','connective':'3','dead':'4','epithelial':'5'}.get(intent.get('cell_class'))
  actual=(addc[classid]-remc[classid]) if classid else len(added)-len(removed)
  requested=abs(cb['target_delta_count']);direction=-1 if intent['operation']=='decrease' else 1;target=direction*requested
  out.update(cell_target_delta=target,cell_actual_delta_events=actual,cell_count_error_events=actual-target,cell_direction_correct=actual*direction>0,cell_count_budget=cb)
 intent=r['prebound_semantic_request']['intents'][0]
 if intent['target']=='tumor_extent':out['requested_direction_correct']=bool((tumor1.sum()-tumor0.sum())*(1 if intent['operation']=='increase' else -1)>0)
 out['independent_raster_contract_checks_passed']=bool(out['primitive_match'] and out['unauthorized_tissue_transition_pixels']==0 and out['change_outside_G_pixels']==0 and out['protected_tissue_changed_pixels']==0 and out['new_nuclear_pixels_in_prohibited_tissue']==0 and out['nuclear_label_changes_outside_declared_cell_region']==0 and (out['cell_only_tissue_changed_pixels'] in (None,0)) and all(out[k] for k in ['tissue_change_mask_exact','joint_change_mask_exact','ledger_joint_pixels_matches_raster']))
 return json.loads(json.dumps(out,default=lambda value:value.item() if isinstance(value,np.generic) else str(value)))
