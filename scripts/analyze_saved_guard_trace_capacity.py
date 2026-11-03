#!/usr/bin/env python3
"""Offline capacity/byte-lossless codec experiment on an EXISTING saved prefix.

No pathology imports, no recorder/controller integration, no source execution.
Round-trip success applies ONLY to supplied bytes, never to the missing run tail.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import tarfile
import time
import zlib

BLOCK_BYTES=4*1024**2
MAX_RECORD_BYTES=8*1024**2
GUARDS={'guard.stage','guard.local_result','guard.accept','guard.reject'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def percentile(hist,numerator,denominator):
    target=(sum(hist.values())*numerator+denominator-1)//denominator;total=0
    for value,count in sorted(hist.items()):
        total+=count
        if total>=target:return value


def analyze(archive_path, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    stats={};lengths=collections.Counter();field_names=collections.Counter();stage_names=collections.Counter();guard_schema=collections.Counter()
    guards=0;guard_bytes=0;guard_envelope=0;coords={'row_min':None,'row_max':None,'col_min':None,'col_max':None}
    block=bytearray();block_first=0;block_events=0;count=0;raw_sha=hashlib.sha256();logical_chain='0'*64;grow_calls=[];active=None
    experiments={level:{'blocks':[],'previous':'0'*64,'compressed_bytes':0,'compression_seconds':0.0} for level in (1,6)}
    def flush():
        nonlocal block,block_first,block_events
        if not block:return
        decoded=bytes(block);raw_digest=hashlib.sha256(decoded).hexdigest()
        for level,data in experiments.items():
            started=time.perf_counter();encoded=zlib.compress(decoded,level);data['compression_seconds']+=time.perf_counter()-started
            path=output/f'level_{level}_block_{len(data["blocks"]):05d}.zlib'
            with path.open('xb') as stream:stream.write(encoded)
            entry={'index':len(data['blocks']),'first_sequence':block_first,'last_sequence':block_first+block_events-1,'logical_events':block_events,'decoded_bytes':len(decoded),'encoded_bytes':len(encoded),'decoded_sha256':raw_digest,'encoded_sha256':hashlib.sha256(encoded).hexdigest(),'previous_entry_sha256':data['previous'],'file':path.name}
            entry['entry_sha256']=hashlib.sha256(canonical(entry)).hexdigest();data['previous']=entry['entry_sha256'];data['blocks'].append(entry);data['compressed_bytes']+=len(encoded)
        block=bytearray();block_first=count;block_events=0
    with tarfile.open(archive_path) as archive:
        members=archive.getmembers();trace=[m for m in members if m.name.endswith('/B/observer/semantic.jsonl')];status_files=[m for m in members if m.name.endswith('/B/observer/observer_status.json')]
        if len(trace)!=1 or len(status_files)!=1:raise ValueError('exactly one saved B trace/status required')
        status=json.load(archive.extractfile(status_files[0]))['status'];stream=archive.extractfile(trace[0])
        started=time.perf_counter()
        for line in stream:
            if len(line)>MAX_RECORD_BYTES:raise ValueError('saved record exceeds offline experiment record cap')
            event=json.loads(line);name=event['event'];size=len(line)
            if event['sequence']!=count:raise ValueError('noncontiguous saved sequence')
            if not line.endswith(b'\n'):raise ValueError('unterminated event')
            raw_sha.update(line);logical_chain=hashlib.sha256(logical_chain.encode()+line).hexdigest()
            item=stats.setdefault(name,{'events':0,'bytes':0,'min_record_bytes':size,'max_record_bytes':size})
            item['events']+=1;item['bytes']+=size;item['min_record_bytes']=min(item['min_record_bytes'],size);item['max_record_bytes']=max(item['max_record_bytes'],size);lengths[size]+=1
            if name=='grow.begin':
                if active is not None:raise ValueError('nested grow not represented by current schema')
                active={'ordinal':len(grow_calls),'first_sequence':count,'events':0,'bytes':0,'guard_events':0,'closed':False};grow_calls.append(active)
            if active is not None:
                active['events']+=1;active['bytes']+=size;active['guard_events']+=name in GUARDS
            if name in GUARDS:
                guards+=1;guard_bytes+=size
                values=event['values']['mapping'];guard_schema[(name,tuple(k for k,_ in values))]+=1
                skeleton={'sequence':event['sequence'],'event':name,'values':{'mapping':[]}}
                guard_envelope+=len(json.dumps(skeleton,ensure_ascii=True,allow_nan=False,separators=(',',':')).encode())+1
                for key,value in values:
                    field_names[key]+=1
                    if key=='stage':stage_names[value]+=1
                    if key in ('row','col'):
                        lo=key+'_min';hi=key+'_max';coords[lo]=value if coords[lo] is None else min(coords[lo],value);coords[hi]=value if coords[hi] is None else max(coords[hi],value)
            if name=='grow.end':
                if active is None:raise ValueError('orphan grow end')
                active['closed']=True;active['last_sequence']=count;active=None
            if block and len(block)+size>BLOCK_BYTES:flush()
            if not block:block_first=count
            block.extend(line);block_events+=1;count+=1
        flush();analysis_seconds=time.perf_counter()-started
    if count!=status['events'] or logical_chain!=status['semantic_chain_sha256']:raise ValueError('saved status chain/count mismatch')
    for level,data in experiments.items():
        roundtrip=hashlib.sha256();events_checked=0;previous='0'*64;started=time.perf_counter()
        for entry in data['blocks']:
            if entry['previous_entry_sha256']!=previous:raise ValueError('block chain broken')
            hashed={k:v for k,v in entry.items() if k!='entry_sha256'}
            if hashlib.sha256(canonical(hashed)).hexdigest()!=entry['entry_sha256']:raise ValueError('entry metadata altered')
            encoded=(output/entry['file']).read_bytes()
            if hashlib.sha256(encoded).hexdigest()!=entry['encoded_sha256']:raise ValueError('encoded block corrupted')
            decoded=zlib.decompress(encoded)
            if len(decoded)!=entry['decoded_bytes'] or hashlib.sha256(decoded).hexdigest()!=entry['decoded_sha256']:raise ValueError('decoded block changed')
            lines=decoded.splitlines(keepends=True)
            if len(lines)!=entry['logical_events']:raise ValueError('block event count mismatch')
            for line in lines:
                if json.loads(line)['sequence']!=events_checked:raise ValueError('roundtrip sequence changed')
                roundtrip.update(line);events_checked+=1
            previous=entry['entry_sha256']
        if events_checked!=count or roundtrip.hexdigest()!=raw_sha.hexdigest():raise ValueError('prefix not byte-lossless')
        data['roundtrip_seconds']=time.perf_counter()-started;data['roundtrip_events']=events_checked;data['decoded_stream_sha256']=roundtrip.hexdigest();data['byte_exact_roundtrip']=True
        data['ratio_encoded_over_raw']=data['compressed_bytes']/sum(x['bytes'] for x in stats.values())
    total_bytes=sum(x['bytes'] for x in stats.values())
    for s in stats.values():s['mean_record_bytes']=s['bytes']/s['events'];s['event_fraction']=s['events']/count;s['byte_fraction']=s['bytes']/total_bytes
    report={'scope':'OFFLINE saved-prefix capacity analysis only; not a recorder acceptance, run completion or source qualification.',
            'execution_permitted':False,'S07_rerun':False,'witness_calls':0,
            'input':{'archive_sha256':sha(archive_path),'trace_sha256':raw_sha.hexdigest(),'trace_bytes':total_bytes,'logical_events':count,'source_observer_valid':status['valid'],'source_interpretation_permitted':status['interpretation_permitted'],'source_completion_not_recovered':True,'saved_chain_matches':True},
            'event_types':stats,'record_bytes':{'min':min(lengths),'p50':percentile(lengths,1,2),'p95':percentile(lengths,95,100),'p99':percentile(lengths,99,100),'max':max(lengths),'mean':total_bytes/count},
            'guards':{'events':guards,'bytes':guard_bytes,'event_fraction':guards/count,'byte_fraction':guard_bytes/total_bytes,'repeated_envelope_bytes':guard_envelope,'envelope_definition':'Exact serialized per-record sequence/event/values/mapping-empty envelope size for guard records. Descriptive overhead only; all bytes remain preserved in codec experiment.','mapping_field_name_occurrences':dict(field_names),'stage_value_occurrences':dict(stage_names),'schema_variants':[{'event':key[0],'ordered_mapping_keys':list(key[1]),'events':value} for key,value in guard_schema.items()],'observed_coordinate_range':coords},
            'grow_contexts':grow_calls,'prefix_boundary':{'grow_begins':len(grow_calls),'grow_ends':sum(g['closed'] for g in grow_calls),'open_grow_at_end':active['ordinal'] if active else None,'qualification_begins':stats.get('qualification.begin',{}).get('events',0),'qualification_completions':stats.get('qualification.complete',{}).get('events',0),'unseen_tail_size':None,'full_U1_U2_volume_estimate':None},
            'lossless_offline_experiments':{str(level):data for level,data in experiments.items()},
            'experiment_parameters':{'block_target_bytes':BLOCK_BYTES,'max_record_bytes':MAX_RECORD_BYTES,'levels_predeclared':[1,6],'sampling':False,'event_deletion':False,'field_exclusion':False,'event_reordering':False},
            'timing_scope':{'offline_host_analysis_seconds':analysis_seconds,'interpretation':'Compression timings on this offline host are not live business latency/observer overhead estimates; no full-run extrapolation.'},
            'validity_boundary':'Roundtrip validates all supplied2M events and exact bytes only. Missing U1/U2 tail remains missing; original B observation invalidity and terminal authorization are unchanged.'}
    with (output/'capacity_report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'events':count,'bytes':total_bytes,'guard_event_fraction':guards/count,'guard_byte_fraction':guard_bytes/total_bytes,'levels':{str(k):{'bytes':v['compressed_bytes'],'ratio':v['ratio_encoded_over_raw'],'roundtrip_events':v['roundtrip_events']} for k,v in experiments.items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.archive,a.output)
