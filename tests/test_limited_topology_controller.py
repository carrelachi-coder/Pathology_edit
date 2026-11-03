"""Synthetic controller/startup checks, never the BCSS probe."""
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from scripts.run_limited_topology_controller import approval_gate,bounded_fork,sequence,run,write_new
from scripts.limited_topology_startup import capture,digest,restore_controlled


def _plan():
    return {'execution_permitted':True,'rng_decision_approved':True,'preflight_complete':True,
            'rng_decision':{'sha256':'decision'},'rng_policy':'common_parent_explicit_postfork_restore_v1',
            'limits':{'source_processes':3,'witness_calls':6,'seconds_per_source_pair':1200,'retries':0}}


@pytest.mark.parametrize('gate',['execution_permitted','rng_decision_approved','preflight_complete'])
def test_closed_gate_blocks_before_identity_or_launch(gate,tmp_path):
    plan=_plan();plan[gate]=False
    with pytest.raises(PermissionError):run(plan,'manifest',{'approved':True,'manifest_sha256':'manifest','rng_decision_sha256':'decision'})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change',['approved','manifest_sha256','rng_decision_sha256'])
def test_approval_must_cover_manifest_and_decision(change):
    record={'approved':True,'manifest_sha256':'manifest','rng_decision_sha256':'decision'}
    record[change]=False if change=='approved' else 'different'
    with pytest.raises(PermissionError):approval_gate(_plan(),'manifest',record)


@pytest.mark.parametrize('change',['source_processes','witness_calls','seconds_per_source_pair','retries'])
def test_frozen_limits_cannot_expand(change):
    plan=_plan();plan['limits'][change]+=1
    with pytest.raises(ValueError):approval_gate(plan,'manifest',{'approved':True,'manifest_sha256':'manifest','rng_decision_sha256':'decision'})


@pytest.mark.parametrize('failed_arm',['A','B','C'])
@pytest.mark.parametrize('failure',['exit','timeout','signal','mismatch','evidence_invalid','assessment_exception'])
def test_real_fork_stop_conditions_and_no_retry(tmp_path,failed_arm,failure):
    emitted=[];started=[]
    def execute(arm,timeout):
        started.append(arm)
        def child():
            (tmp_path/(arm+'.started')).write_text('synthetic child; no business module')
            if arm==failed_arm:
                if failure=='exit':raise RuntimeError('synthetic exit')
                if failure=='timeout':time.sleep(2)
                if failure=='signal':os.kill(os.getpid(),signal.SIGKILL)
        return bounded_fork(child,timeout=.04 if failure=='timeout' and arm==failed_arm else 2)
    def assess(arm):
        if arm==failed_arm:
            if failure=='assessment_exception':raise ValueError('truncated evidence')
            if failure in ('mismatch','evidence_invalid'):return False
        return True
    result=sequence(execute,assess,emit=emitted.append)
    expected=list('ABC')[:list('ABC').index(failed_arm)+1]
    assert started==expected==result['launched']
    assert result['status']=='mismatch_hold' and result['reserved_witness_calls']==2*len(expected)
    assert len([x for x in emitted if x['event']=='launch_reserved'])==len(expected)
    assert result['reserved_witness_calls']<=6


def test_success_exact_order_and_six_reserved_calls(tmp_path):
    calls=[]
    def execute(arm,timeout):
        calls.append((arm,timeout));return bounded_fork(lambda:(tmp_path/arm).write_text(arm),timeout=2)
    result=sequence(execute,lambda arm:True,emit=lambda e:None)
    assert calls==[('A',1200),('B',1200),('C',1200)]
    assert result['reserved_witness_calls']==6 and result['status']=='diagnostic_comparison_complete'
    assert result['qualification_updates_permitted'] is False


def test_explicit_restore_preserves_snapshot_and_local_generator():
    import random,numpy as np
    original=capture();local=np.random.default_rng(741);local_before=json.dumps(local.bit_generator.state)
    try:
        random.random();np.random.random()
        assert digest(capture())!=digest(original)
        after=restore_controlled(original)
        assert digest(after)==digest(original)
        assert json.dumps(local.bit_generator.state)==local_before
        assert after['hash_fingerprint']==original['hash_fingerprint']
    finally:restore_controlled(original)


def test_exclusive_identity_records_do_not_overwrite(tmp_path):
    p=tmp_path/'identity.json';write_new(p,{'first':1})
    before=p.read_bytes()
    with pytest.raises(FileExistsError):write_new(p,{'second':2})
    assert p.read_bytes()==before


def test_assessment_exception_stops_before_next_child():
    called=[]
    def assess(arm):raise OSError('partial comparator receipt')
    result=sequence(lambda arm,t:(called.append(arm) or {'exit_code':0,'timed_out':False}),assess,emit=lambda e:None)
    assert called==['A'] and result['reserved_witness_calls']==2


def test_external_cancellation_reaps_child(tmp_path):
    # A signal exit is reaped and cannot be accepted as a valid diagnosis.
    result=bounded_fork(lambda:os.kill(os.getpid(),signal.SIGTERM),timeout=2)
    assert result['exit_code']==-signal.SIGTERM and result['timed_out'] is False


@pytest.mark.parametrize('mutation',['limit','start_index','probe','mode'])
def test_actual_launch_scope_cannot_expand(tmp_path,mutation):
    from scripts.run_limited_topology_controller import scope_gate
    subset=tmp_path/'subset.json';subset.write_text(json.dumps({'candidate_sources':[{'reference_id':str(i)} for i in range(8)]}))
    expected=['--source-freeze',str(subset),'--cases','templates','--start-index','7','--limit','1','--topology-witness']
    plan={'source_subset':{'path':str(subset)},'case_templates':{'path':'templates'},'reference_id':'7',
          'arms':{k:{'probe':'scripts/probe_manuscript_joint_capacity.py','probe_args':list(expected),'mode':mode} for k,mode in [('A',None),('B','hash_only'),('C','full_capture')]}}
    scope_gate(plan)
    if mutation=='limit':plan['arms']['B']['probe_args'][7]='2'
    elif mutation=='start_index':plan['arms']['A']['probe_args'][5]='0'
    elif mutation=='probe':plan['arms']['C']['probe']='another.py'
    else:plan['arms']['B']['mode']='full_capture'
    with pytest.raises(ValueError):scope_gate(plan)


def test_parent_interrupt_kills_and_reaps_child(monkeypatch):
    import scripts.run_limited_topology_controller as controller
    original_fork=os.fork;original_sleep=time.sleep;pids=[]
    def traced_fork():
        pid=original_fork()
        if pid:pids.append(pid)
        return pid
    def interrupt(_seconds):raise KeyboardInterrupt('synthetic controller cancellation')
    monkeypatch.setattr(controller.os,'fork',traced_fork)
    monkeypatch.setattr(controller.time,'sleep',interrupt)
    with pytest.raises(KeyboardInterrupt):
        bounded_fork(lambda:original_sleep(2),timeout=5)
    assert len(pids)==1
    with pytest.raises(ChildProcessError):os.waitpid(pids[0],os.WNOHANG)
