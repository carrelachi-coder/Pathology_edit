import json,csv,time,os,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path('/data1/lyw/pathology_edit_eval/r2_de_terra_six_v4_20260924')
sys.path[:0]=[str(ROOT/'code'),'/home/lyw/wqx-DL/flow-edit/FlowEdit-main']
from measure import measure
STATES=['validated','clarification','review','abstained','failed','timeout','runtime_error','planner_format_error','planner_transport_error']
def dump(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n');tmp.replace(p)
def csvout(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r));tmp=p.with_suffix('.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
 tmp.replace(p)
def summarize():
 cohort=json.loads((ROOT/'frozen_cohort.json').read_text());records=[];metrics=[]
 for row in cohort:
  name=row['record']['case_id'];p=ROOT/'results'/f'{name}.json'
  if not p.exists():continue
  r=json.loads(p.read_text());prpath=ROOT/'runs'/name/name/'program_result.json'
  # CLI returns nonzero for scientific abstention/failure. Classify by the persisted program outcome, not exit code.
  if prpath.exists() and r['outcome']!='timeout':
   pr=json.loads(prpath.read_text());sts={str(s['workflow_status']) for s in pr['steps']};status=pr['status']
   reasons=[z for s in pr['steps'] for z in s.get('reasons',[])]
   if status=='validated' and pr['evaluation'].get('passed'):r['outcome']='validated'
   elif any('CodexCLIQueueError: ValueError:' in z for z in reasons):r['outcome']='planner_format_error'
   elif any('CodexCLIQueueError:' in z for z in reasons):r['outcome']='planner_transport_error'
   elif 'clarification' in status or any('clarification' in x for x in sts):r['outcome']='clarification'
   elif 'review' in status or any('review' in x for x in sts):r['outcome']='review'
   elif 'abstained' in sts:r['outcome']='abstained'
   else:r['outcome']='failed'
   r['reasons']=reasons;r['program_result']=str(prpath)
  records.append(r)
  q=ROOT/'metrics'/f'{name}.json'
  if q.exists():
   m=json.loads(q.read_text())
   if m.get('measurement_version')!='r2de-raster-v2-authorized-protection':
    m=measure(row,m['handoff']);dump(q,m)
   m['request_case_id']=name
   if m.get('joint_within_declared_range') in ('True','False'):m['joint_within_declared_range']=m['joint_within_declared_range']=='True'
   metrics.append(m)
 outcomes=Counter(r['outcome'] for r in records);pending=len(cohort)-len(records)
 report={'updated_at_unix':time.time(),'total_planned':len(cohort),'completed':len(records),'pending':pending,'outcomes':dict(outcomes),'E_measured':len(metrics),'E_raster_audit_failures':[r['request_case_id'] for r in metrics if not r['independent_raster_contract_checks_passed']],'E_unmeasured_validated':outcomes['validated']-len(metrics),'cell_count_basis':'instance-event count, not independent target-instance ground truth','J_definition':'union of independently recomputed tissue label change and exported complete-instance cell footprint change; raw label-delta union also exported separately','status':'complete' if pending==0 else 'running'}
 for field in ['dataset','level']:
  report['by_'+field]={k:{'planned':sum(x[field]==k for x in cohort),'completed':sum(x[field]==k for x in records),'outcomes':dict(Counter(x['outcome'] for x in records if x[field]==k))} for k in sorted({x[field] for x in cohort})}
 report['workers']=[json.loads(p.read_text()) for p in sorted(ROOT.glob('worker*_status.json'))]
 report['reason_counts']=dict(Counter(z for x in records for z in x.get('reasons',[])))
 dump(ROOT/'summary.json',report)
 csvout(ROOT/'D_request_outcomes.csv',[{k:r.get(k) for k in ['case_id','dataset','primitive_id','level','source_group','outcome','elapsed_seconds','reasons','audit_error','program_result']} for r in records]);csvout(ROOT/'E_mask_measurements.csv',metrics)
 if pending==0:
  (ROOT/'COMPLETED').write_text(str(time.time()));print(json.dumps(report,ensure_ascii=False),flush=True)
 return report
if __name__=='__main__':
 if '--watch' in sys.argv:
  while True:
   s=summarize()
   if s['pending']==0:break
   time.sleep(30)
 else:print(json.dumps(summarize(),ensure_ascii=False,indent=2))
