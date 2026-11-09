import json
import hashlib
import pytest
from scripts.run_budget_cost_target_trial_v1 import check_approval,compare_reports


def test_approval_bound_to_exact_manifest_bytes(tmp_path):
    m=tmp_path/'manifest';a=tmp_path/'approval';m.write_text(json.dumps({'kind':'isolated-synthetic-budget-cost-proposal-v1'}));a.write_text(json.dumps({'approved':True,'kind':'isolated-synthetic-budget-cost-proposal-v1','manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest()}));assert check_approval(m,a)
    m.write_text(m.read_text()+'\n')
    with pytest.raises(ValueError):check_approval(m,a)


@pytest.mark.parametrize('mutation',['hash','count','rng','mode','missing'])
def test_compare_refuses_incomplete_or_changed_streams(mutation):
    rows=[{'candidate_budget':v,'events':100000,'logical_events_written':100000,'global_rng_unchanged':True,'capture_complete':False,'logical_stream_sha256':'a'*64} for v in [False,True]]
    compare_reports(rows)
    if mutation=='hash':rows[1]['logical_stream_sha256']='b'*64
    elif mutation=='count':rows[1]['logical_events_written']-=1
    elif mutation=='rng':rows[1]['global_rng_unchanged']=False
    elif mutation=='mode':rows[1]['candidate_budget']=False
    else:rows.pop()
    with pytest.raises(ValueError):compare_reports(rows)


def test_setup_alarm_holds_before_any_producer(tmp_path,monkeypatch):
    import signal
    import time
    from scripts import run_budget_cost_target_trial_v1 as runner
    output=tmp_path/'output';manifest=tmp_path/'m';manifest.write_text('{}')
    monkeypatch.setattr(runner,'check_approval',lambda *args:{'output_root':str(output),'code_files_sha256':{'dummy.py':'hash'}})
    def slow_hash(path):
        if str(path).endswith('dummy.py'):time.sleep(.2)
        return 'hash'
    monkeypatch.setattr(runner,'sha',slow_hash)
    real_timer=signal.setitimer
    monkeypatch.setattr(runner.signal,'setitimer',lambda kind,seconds:real_timer(kind,.01 if seconds else 0))
    assert runner.run(manifest,tmp_path/'a')==78
    result=json.loads((output/'trial_result.json').read_text())
    assert result['failure_type']=='TimeoutError' and result['profiles']==[]
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0
