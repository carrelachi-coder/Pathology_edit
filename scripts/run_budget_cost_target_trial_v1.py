"""Manifest-bound isolated synthetic profile trial. Never invokes pathology.

Producer transport reuses the reviewed v2 bounded process-group gate. Its
60-second cap includes child startup, profiling, output writes and normal exit.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def check_approval(manifest,approval):
    raw=Path(manifest).read_bytes();m=json.loads(raw);a=json.loads(Path(approval).read_bytes())
    if not (a.get('approved') is True and a.get('kind')==m['kind'] and a.get('manifest_sha256')==hashlib.sha256(raw).hexdigest()):raise ValueError('manifest-bound approval required')
    if m['kind']!='isolated-synthetic-budget-cost-proposal-v1':raise ValueError('wrong scope')
    return m


def check_report(r,variant):
    if r.get('candidate_budget') is not variant or r.get('events')!=100000 or r.get('logical_events_written')!=100000 or r.get('global_rng_unchanged') is not True or r.get('capture_complete') is not False:raise ValueError('incomplete synthetic profile')
    h=r.get('logical_stream_sha256')
    if type(h) is not str or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('stream digest')


def compare_reports(reports):
    if len(reports)!=2:raise ValueError('two completed variants required')
    for variant,r in zip([False,True],reports):
        check_report(r,variant)
    if reports[0]['logical_stream_sha256']!=reports[1]['logical_stream_sha256']:raise ValueError('logical bytes mismatch')


def run(manifest,approval):
    start=time.monotonic();m=check_approval(manifest,approval);root=Path(m['output_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'authorization_consumed.json').write_text(json.dumps({'manifest_sha256':sha(manifest),'consumed':True}))
    result={'status':'HOLD','automatic_retries':0,'S07_calls':0,'HE_calls':0,'outer_acceptance_published':False,'profiles':[]}
    previous_alarm=signal.getsignal(signal.SIGALRM)
    def setup_timeout(signum,frame):raise TimeoutError("setup hard deadline")
    signal.signal(signal.SIGALRM,setup_timeout)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,60-(time.monotonic()-start)))
    try:
        deadline=start+60
        def check():
            if time.monotonic()>=deadline:raise TimeoutError('setup')
        here=Path(__file__).resolve().parents[1]
        for p,h in m['code_files_sha256'].items():check();assert sha(here/p)==h
        r=m['runtime'];check();assert sha(sys.executable)==r['executable_sha256'] and sys.version==r['python'] and platform.platform()==r['platform']
        for d in r['packages']:
            check();dist=importlib.metadata.distribution(d['name']);assert dist.version==d['version'];record=dist.read_text('RECORD');digest=hashlib.sha256(record.encode()).hexdigest() if record is not None else None;assert digest==d['record_sha256']
        for k,v in r['environment'].items():check();assert os.environ.get(k)==v
        assert shutil.disk_usage(root).free>=12*1024**3
        pending=root/'hardlink.pending';final=root/'hardlink.final';pending.open('xb').write(b'original');os.link(pending,final)
        try:os.link(pending,final)
        except FileExistsError:pass
        else:raise ValueError('no-replace failure')
        assert final.read_bytes()==b'original';check();result['setup_seconds']=time.monotonic()-start
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_alarm)
        from diagnostic_storage.validator_process_gate_v2 import _run_process
        reports=[]
        for i,variant in enumerate(m['variants']):
            if variant['label']!=['original','candidate'][i]:raise ValueError('variant order')
            expected=['/usr/bin/timeout','--signal=INT','--kill-after=5s','60s',r['executable'],str(here/'scripts/profile_adapter_synthetic_cost_v1.py'),'--events','100000','--output',str(root/variant['label'])]+(['--candidate-budget'] if i else [])
            if variant['argv']!=expected:raise ValueError('argv scope mismatch')
            if time.monotonic()-start>=180:raise TimeoutError('whole trial')
            transport=_run_process(expected,b'{}',60,kind='compare')
            public={k:v for k,v in transport.items() if k!='stdout'};result['profiles'].append({'label':variant['label'],'transport':public})
            if not transport['transport_ok']:raise RuntimeError('producer HOLD')
            report=root/variant['label']/'report.json'
            if report.stat().st_size>1024**2:raise ValueError('report limit')
            reports.append(json.loads(report.read_bytes()))
            (root/(variant['label']+'_transport.json')).open('x').write(json.dumps(public,indent=2))
            # Fail before candidate when the original report is already invalid.
            check_report(reports[-1],bool(i))
        compare_reports(reports)
        if time.monotonic()-start>=180:raise TimeoutError('whole trial')
        result.update(status='synthetic_profiles_complete_stream_hash_equal',seconds=[r['elapsed_seconds'] for r in reports],interpretation='Profiled observations only. No real-source timing, complete capture or eligibility conclusion.')
    except Exception as exc:result['failure_type']=type(exc).__name__
    signal.setitimer(signal.ITIMER_REAL,0)
    signal.signal(signal.SIGALRM,previous_alarm)
    result['elapsed_seconds']=time.monotonic()-start
    (root/'trial_result.json').open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return 0 if result['status']!='HOLD' else 78


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--approval',type=Path,required=True);a=p.parse_args();raise SystemExit(run(a.manifest,a.approval))
