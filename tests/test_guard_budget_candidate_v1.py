"""Differential acceptance of an unintegrated budget implementation only."""
from pathlib import Path
import json
import numpy as np
import pytest
from diagnostic_storage import guard_blocks_v2 as codec
from diagnostic_storage.guard_budget_candidate_v1 import CandidateBudget, attach_for_independent_trial


def state(b):
    return b.total, dict(b.charges), dict(b.sizes)


@pytest.mark.parametrize('limit', [262144, 1048576, 4*1024**3])
def test_identical_budget_decisions_and_state(limit, tmp_path):
    old, new = codec.Budget(limit), CandidateBudget(limit)
    rng = np.random.default_rng(193)
    paths = [tmp_path/'blocks.index.pending', tmp_path/'arrays.index.pending', tmp_path/'block.bin.pending', 'relative/file.bin']
    for i in range(2000):
        path = paths[int(rng.integers(len(paths)))];size=int(rng.integers(0, 1500000));control=bool(rng.integers(2))
        method=['claim','check','seal','claim_many'][i % 4]
        errors=[]
        for budget in [old,new]:
            try:
                if method=='seal':budget.seal(path)
                elif method=='claim_many':budget.claim_many([(path,size),(paths[(paths.index(path)+1)%4],size//2)])
                else:getattr(budget,method)(path,size,control)
            except Exception as exc:errors.append((type(exc),str(exc)))
            else:errors.append(None)
        assert errors[0] == errors[1]
        assert state(old)==state(new)


@pytest.mark.parametrize('mode', ['hash_only','full_capture'])
def test_identical_complete_stream_and_files(mode,tmp_path):
    roots=[]
    for candidate in [False,True]:
        root=tmp_path/str(candidate);w=codec.Writer(root,mode=mode,target_block_bytes=350);roots.append(root)
        if candidate:attach_for_independent_trial(w)
        for i in range(120):
            event='guard.stage';fields={}
            if i in [0,60]:event='qualification.begin';fields['operation_id']='U1' if i==0 else 'U2'
            if i in [59,119]:event='qualification.complete';fields['operation_id']='U1' if i==59 else 'U2'
            if i%17==3:fields['mask']=w.add_array(np.arange(12,dtype='>i4').reshape(3,4))
            w.write_event(codec.dump({'sequence':i,'event':event,**fields})+b'\n')
        w.finish(normal_end=True,receipt={'normal_end':True,'operation_ids':['U1','U2']})
    assert codec.compare(*roots)['strict_match']
    files=[{p.name:p.read_bytes() for p in root.iterdir() if p.is_file()} for root in roots]
    assert files[0]==files[1]


def test_observed_allocation_preserved_and_fresh_writer_gate(tmp_path):
    p=tmp_path/'f';p.write_bytes(b'x'*10000)
    a,b=codec.Budget(1048576),CandidateBudget(1048576)
    for budget in [a,b]:budget.claim(p,0);budget.actual(p)
    assert state(a)==state(b)
    w=codec.Writer(tmp_path/'writer');w.write_event(codec.dump({'sequence':0,'event':'guard.stage'})+b'\n')
    with pytest.raises(ValueError):attach_for_independent_trial(w)
    w.finish(normal_end=False,receipt={})
