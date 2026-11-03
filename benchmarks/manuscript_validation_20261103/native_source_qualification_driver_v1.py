"""Frozen two-stage source-only batch driver; never calls a generation model."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image
P=Path('/data1/zhao/wqx/benchmarks/runs/manuscript_final_joint_assets_20261102_v1')
BASE=P/'native_crop_recovery_20261103_v1';ROOT=BASE/'qualification_v1'
REPO=Path('/home/lyw/wqx-DL/flow-edit/manuscript-execution-20ace976')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)
def classify(receipt,source,*,topology=False):
 if receipt is None:return 'not_run'
 if receipt.get('timed_out'):return 'timeout'
 if receipt.get('controller_exit_code')!=0 or 'result' not in receipt:return 'execution_error'
 result=receipt['result'];checks=result.get('checks',[])
 if len(checks)!=2 or result['reference_id']!=source['reference_id']:return 'execution_error'
 if any(c.get('error') for c in checks):return 'execution_error'
 if [c.get('primitive_id') for c in checks]!=['cohesive-boundary-expansion-v1','generic-immune-infiltrate-increase-v1']:return 'execution_error'
 if any(type(c.get('passed_necessary_capacity')) is not bool for c in checks):return 'execution_error'
 if topology:
  if not all(c.get('passed_necessary_capacity') is True for c in checks):return 'capacity_recheck_mismatch'
  if any('passed_topology_witness' not in c for c in checks):return 'execution_error'
  return 'passed' if all(c['passed_topology_witness'] is True for c in checks) else 'failed_condition'
 return 'passed' if all(c.get('passed_necessary_capacity') is True for c in checks) else 'failed_condition'
def main():
 ROOT.mkdir(exist_ok=False);protocol=BASE/'qualification_protocol_v1.json';plan=json.loads(protocol.read_text());sources_path=BASE/'server_candidate_sources_v1.json';templates=P/'technical_joint_cases_v5.json';state={'scope':'native BCSS necessary-capacity and source-topology only','started_utc':datetime.now(timezone.utc).isoformat(),'formal_launch_permitted':False,'batch_four_target_compilation_permitted':False,'status':'preflight','protocol_sha256':sha(protocol),'driver_sha256':sha(__file__)}
 def save(**fields):state.update(fields);write(ROOT/'state.json',state)
 try:
  assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==plan['execution_code_commit']
  assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
  assert sha(sources_path)==plan['source_census_sha256'] and sha(templates)==plan['case_templates_sha256']
  assert sha(__file__)==plan['driver_sha256']
  sources=json.loads(sources_path.read_text())['candidate_sources'];assert len(sources)==216
  key=lambda s:(s['patient_id'],s['source_roi_id'],s['source_crop_xy'][1],s['source_crop_xy'][0])
  assert sources==sorted(sources,key=key)
  assert len({s['reference_id'] for s in sources})==216
  cases=json.loads(templates.read_text())
  for index in [1,3]:
   budget=cases[index]['joint_area_budget'];assert budget==plan['significant_joint_budget']
  excluded=set(json.loads((P/'technical_patient_exclusion_audit_v2.json').read_text())['excluded_patient_ids'])
  assert sha(P/'technical_patient_exclusion_audit_v2.json')==plan['technical_exclusion_sha256']
  integrity=[]
  for source in sources:
   assert source['patient_id'] not in excluded
   arrays=[]
   for field in ['reference_image','reference_tissue_mask','reference_nuclei_mask']:
    asset=source['inputs'][field];path=Path(asset['path']);assert path.is_relative_to(BASE/'local_evidence_v1')
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==asset['sha256']
    with Image.open(path) as im:array=np.asarray(im.convert('RGB') if field=='reference_image' else im).copy()
    header=json.dumps({'shape':list(array.shape),'dtype':'uint8'},sort_keys=True).encode()
    actual=hashlib.sha256(header+b'\0'+np.ascontiguousarray(array).tobytes()).hexdigest();assert actual==source['decoded_sha256'][len(arrays)]
    arrays.append(array)
   rgb,tissue,nuclei=arrays
   assert rgb.shape==(512,512,3) and tissue.shape==nuclei.shape==(512,512)
   assert np.isin(tissue,range(16)).all() and np.isin(nuclei,[0,1,2,3,4,5,101,102,103,104,105]).all()
   assert all(np.any(tissue==v) for v in [1,2,4]) and int(np.count_nonzero(tissue==2))>=70779
   integrity.append({'reference_id':source['reference_id'],'status':'verified','files':3})
  write(ROOT/'data_integrity.json',{'requested_sources':216,'verified_sources':len(integrity),'verified_files':len(integrity)*3,'records':integrity})
  os.chdir(REPO)
  os.environ.update(plan['environment'])
  controller=REPO/'scripts/run_manuscript_source_qualification.py'
  def execute(pool,output,mode):
   command=[sys.executable,str(controller),'--source-freeze',str(pool),'--cases',str(templates),'--output',str(output),'--workers',str(plan['workers']),'--seconds-per-source',str(plan['seconds_per_source'][mode])]
   if mode=='capacity':command.append('--capacity-only')
   save(status='running_'+mode,stage_source_count=len(json.loads(pool.read_text())['candidate_sources']))
   with (ROOT/(mode+'_controller.log')).open('x') as log:result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
   progress=output/'controller_progress.json';rows=json.loads(progress.read_text()) if progress.exists() else []
   for r in rows:
    if 'result_asset' in r:assert sha(r['result_asset']['path'])==r['result_asset']['sha256']
   with (ROOT/(mode+'_final_progress.json')).open('x') as f:json.dump(rows,f,indent=2)
   return result.returncode,rows
  code,capacity=execute(sources_path,ROOT/'capacity','capacity');by_capacity={r['reference_id']:r for r in capacity}
  assert len(by_capacity)==len(capacity) and all(r['reference_id']==sources[r['index']]['reference_id'] for r in capacity)
  status={s['reference_id']:classify(by_capacity.get(s['reference_id']),s) for s in sources}
  write(ROOT/'capacity_classified.json',[{'reference_id':s['reference_id'],'patient_id':s['patient_id'],'status':status[s['reference_id']]} for s in sources])
  if code!=0 or len(capacity)!=216:raise RuntimeError('capacity controller incomplete; no topology launch')
  selected=[s for s in sources if status[s['reference_id']]=='passed']
  subset=ROOT/'capacity_passing_sources.json';write(subset,{'candidate_sources':selected,'parent_census_sha256':sha(sources_path),'capacity_progress_sha256':sha(ROOT/'capacity_final_progress.json'),'scope':'necessary-capacity pass subset in unchanged source order; not approved targets'})
  if selected:
   code,topology=execute(subset,ROOT/'topology','topology')
   assert all(r['reference_id']==selected[r['index']]['reference_id'] for r in topology)
  else:code,topology=0,[];write(ROOT/'topology_final_progress.json',[])
  by_topology={r['reference_id']:r for r in topology};ledger=[]
  for s in sources:
   cap=status[s['reference_id']];top=classify(by_topology.get(s['reference_id']),s,topology=True) if cap=='passed' else 'not_run_capacity_not_passed'
   ledger.append({'reference_id':s['reference_id'],'patient_id':s['patient_id'],'capacity_status':cap,'topology_status':top,'four_targets_status':'not_run','mask_review_status':'not_run','formal_he_status':'not_run'})
  write(ROOT/'source_status_ledger.json',ledger)
  if code!=0 or len(topology)!=len(selected):raise RuntimeError('topology controller incomplete; retain partials')
  passing={r['patient_id'] for r in ledger if r['topology_status']=='passed'}
  unknown_status={'timeout','execution_error','not_run','capacity_recheck_mismatch'}
  unknown={r['patient_id'] for r in ledger if r['capacity_status'] in unknown_status or r['topology_status'] in unknown_status}
  old=json.loads((BASE/'private_old_patient_sets.json').read_text());old_pass=set(old['passing_patients']);old_unknown=set(old['unknown_patients'])
  counts=lambda field:{v:sum(r[field]==v for r in ledger) for v in sorted({r[field] for r in ledger})}
  first={}
  for source in sources:
   if source['patient_id'] in passing and any(r['reference_id']==source['reference_id'] and r['topology_status']=='passed' for r in ledger):first.setdefault(source['patient_id'],source['reference_id'])
  write(ROOT/'private_first_new_topology_source_per_patient.json',first)
  report={'scope':'source-only native recovery capacity/topology qualification; no four-target or clinical claim','status':'complete','requested_sources':216,'requested_patients':67,'verified_input_files':648,'capacity_status_counts':counts('capacity_status'),'topology_status_counts':counts('topology_status'),'capacity_passing_sources':len(selected),'topology_requested_sources':len(selected),'topology_completed_sources':len(topology),'new_topology_passing_sources':sum(r['topology_status']=='passed' for r in ledger),'new_topology_passing_patients':len(passing),'new_topology_patients_outside_old17':len(passing-old_pass),'combined_confirmed_topology_patients':len(old_pass|passing),'combined_upper_including_unknown':len(old_pass|old_unknown|passing|unknown),'new_unknown_patient_groups':len(unknown),'old_confirmed_topology_patients':17,'old_upper_including_unknown':19,'new_fully_approved_four_target_patients':0,'formal_launch_permitted':False,'batch_four_target_compilation_started':False,'any_gpu_inference':False,'protocol_sha256':sha(protocol),'source_status_ledger_sha256':sha(ROOT/'source_status_ledger.json'),'capacity_progress_sha256':sha(ROOT/'capacity_final_progress.json'),'topology_progress_sha256':sha(ROOT/'topology_final_progress.json'),'private_first_topology_source_sha256':sha(ROOT/'private_first_new_topology_source_per_patient.json'),'interpretation':'Passing topology is necessary only. First source per patient is provisional topology order, never final four-target approval. Unknowns stay unknown and failed conditions are not zero readouts.'}
  write(ROOT/'qualification_report.json',report);save(status='complete',report_sha256=sha(ROOT/'qualification_report.json'));print(json.dumps(report),flush=True)
 except Exception as exc:save(status='failed',error=type(exc).__name__+': '+str(exc));raise
if __name__=='__main__':main()
