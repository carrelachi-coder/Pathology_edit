"""Offline byte conservation and fail-closed checks; no source/witness imports."""
import hashlib
import io
import json
import tarfile

import pytest

from scripts import analyze_saved_guard_trace_capacity as analysis


def fixture_archive(path, *, gap=False, bad_chain=False):
    events=[{'sequence':0,'event':'grow.begin','values':{'mapping':[]}}]
    for i in range(1,9):
        events.append({'sequence':i,'event':'guard.stage','values':{'mapping':[['stage','source_present'],['row',2],['col',3],['payload',{'text':'不删字段\\n','nonfinite':'+inf','ordered':[1,None,True]}]]}})
    events.append({'sequence':9,'event':'grow.end','values':{'mapping':[]}})
    if gap:events[3]['sequence']=90
    # Preserve deliberately different whitespace/Unicode spelling byte-for-byte.
    lines=[(json.dumps(e,ensure_ascii=(i%2==0),separators=(',',':') if i%2==0 else (', ',': '))+'\n').encode() for i,e in enumerate(events)]
    raw=b''.join(lines);chain='0'*64
    for line in lines:chain=hashlib.sha256(chain.encode()+line).hexdigest()
    status={'status':{'events':len(events),'semantic_chain_sha256':'bad' if bad_chain else chain,'valid':False,'interpretation_permitted':False}}
    with tarfile.open(path,'w:gz') as archive:
        for name,data in [('run/B/observer/semantic.jsonl',raw),('run/B/observer/observer_status.json',json.dumps(status).encode())]:
            item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
    return raw,events


def test_every_byte_and_event_survives_small_blocks(tmp_path,monkeypatch):
    archive=tmp_path/'source.tar.gz';raw,events=fixture_archive(archive)
    monkeypatch.setattr(analysis,'BLOCK_BYTES',150)
    output=tmp_path/'offline';analysis.analyze(archive,output)
    report=json.loads((output/'capacity_report.json').read_text())
    assert report['input']['logical_events']==len(events)
    assert report['input']['trace_sha256']==hashlib.sha256(raw).hexdigest()
    for value in report['lossless_offline_experiments'].values():
        assert value['byte_exact_roundtrip'] and value['roundtrip_events']==len(events)
        assert value['decoded_stream_sha256']==hashlib.sha256(raw).hexdigest()
        assert len(value['blocks'])>1
    assert report['input']['source_observer_valid'] is False
    assert report['input']['source_completion_not_recovered'] is True
    assert not (output/'observer_complete.json').exists()
    assert report['prefix_boundary']['full_U1_U2_volume_estimate'] is None


@pytest.mark.parametrize('failure',['gap','chain','codec'])
def test_invalid_input_or_corrupted_codec_cannot_produce_receipt(tmp_path,monkeypatch,failure):
    archive=tmp_path/'source.tar.gz';fixture_archive(archive,gap=failure=='gap',bad_chain=failure=='chain')
    if failure=='codec':monkeypatch.setattr(analysis.zlib,'compress',lambda *args:b'corrupt')
    output=tmp_path/'offline'
    with pytest.raises((ValueError,analysis.zlib.error)):
        analysis.analyze(archive,output)
    assert not (output/'capacity_report.json').exists()
