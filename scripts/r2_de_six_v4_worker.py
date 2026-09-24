import os,sys,json,time,signal,subprocess,hashlib,traceback
from pathlib import Path
ROOT=Path('/data1/lyw/pathology_edit_eval/r2_de_terra_six_v4_20260924');CODE=ROOT/'code'
worker=int(sys.argv[1]);gpus=[0,2,6,7]
os.environ.update(CUDA_VISIBLE_DEVICES=str(gpus[worker]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(CODE)+':/home/lyw/wqx-DL/flow-edit/FlowEdit-main',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path.insert(0,str(CODE))
from measure import measure

def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str)+'\n');temp.replace(p)
protocol=json.loads((ROOT/'protocol.json').read_text());assert hashlib.sha256((ROOT/'frozen_cohort.json').read_bytes()).hexdigest()==protocol['cohort_sha256']
rows=json.loads((ROOT/'frozen_cohort.json').read_text());lock=ROOT/f'worker{worker}.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
for row in rows[worker::4]:
 name=row['record']['case_id'];resultfile=ROOT/'results'/f'{name}.json'
 if resultfile.exists():continue
 manifest=ROOT/'inputs'/f'{name}.json';dump(manifest,[row['record']]);out=ROOT/'runs'/name;start=time.time()
 dump(ROOT/f'worker{worker}_status.json',{'status':'running','pid':os.getpid(),'case_id':name,'started_at_unix':start,'gpu':gpus[worker]})
 base={'case_id':name,'index':row['index'],'dataset':row['dataset'],'primitive_id':row['primitive_id'],'level':row['level'],'source_group':row['source_group'],'worker':worker,'gpu':gpus[worker],'started_at_unix':start}
 cmd=[sys.executable,'-u','-c','from phase3_joint_edit_refine.program_cli import main; raise SystemExit(main())','--manifest',str(manifest),'--output-root',str(out),'--semantic-parser','prebound','--agent-mode','cli-queue','--planner-queue-root',str(ROOT/'terra_queue'),'--model','gpt-5.6-terra','--reasoning-effort','medium','--cell-executor','mature','--probnet-checkpoint','/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt','--nuclei-instance-library','/home/lyw/wqx-DL/flow-edit/FlowEdit-main/nuclei_library/'+('GlaS' if row['dataset']=='GLAS' else row['dataset']),'--probnet-dataset','GlaS' if row['dataset']=='GLAS' else row['dataset'],'--device','cuda','--meta-eval']
 (ROOT/'logs').mkdir(exist_ok=True)
 try:
  with (ROOT/'logs'/f'{name}.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=proc.wait(timeout=protocol['per_request_timeout_seconds'])
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    base.update(outcome='timeout',reason=f"prespecified_{protocol['per_request_timeout_seconds']}_second_wall_time_limit");rc=None
  if rc is not None:
   result=out/name/'program_result.json';base['return_code']=rc
   if rc!=0 or not result.exists():base.update(outcome='runtime_error',reason='nonzero_exit_or_missing_program_result')
   else:
    pr=json.loads(result.read_text());base.update(program_status=pr['status'],steps=pr['steps'],program_result=str(result),program_evaluation=pr['evaluation'])
    statuses={x['workflow_status'] for x in pr['steps']};reasons=[z for x in pr['steps'] for z in x.get('reasons',[])];base['reasons']=reasons
    if pr['status']=='validated' and pr['evaluation'].get('passed'):
     handoffs=list((out/name).rglob('generation_handoff/manifest.json'));assert len(handoffs)==1
     m=json.loads(handoffs[0].read_text());reports=json.loads((handoffs[0].parent.parent/'joint_gate_reports.json').read_text());selected=next(x for x in reports if x['candidate_id']==m['candidate_id']);assert selected['passed']
     base.update(outcome='validated',selected_gate_passed=True,handoff=str(handoffs[0]));metrics=measure(row,handoffs[0]);dump(ROOT/'metrics'/f'{name}.json',metrics);base['independent_raster_contract_checks_passed']=metrics['independent_raster_contract_checks_passed']
    elif 'clarification' in pr['status'] or any('clarification' in str(x) for x in statuses):base['outcome']='clarification'
    elif 'review' in pr['status'] or any('review' in str(x) for x in statuses):base['outcome']='review'
    elif 'abstained' in statuses:base['outcome']='abstained'
    else:base['outcome']='failed'
 except Exception as exc:
  # An audit failure is distinct from execution failure; retain validated state if established.
  base.update(audit_error=f'{type(exc).__name__}: {exc}',audit_traceback=traceback.format_exc())
  if base.get('outcome')!='validated':base['outcome']='runtime_error'
 base['elapsed_seconds']=round(time.time()-start,2);dump(resultfile,base);print(json.dumps({k:base.get(k) for k in ['case_id','outcome','elapsed_seconds','audit_error']},ensure_ascii=False),flush=True)
dump(ROOT/f'worker{worker}_status.json',{'status':'complete','pid':os.getpid(),'completed_at_unix':time.time(),'gpu':gpus[worker]})
lock.unlink()
