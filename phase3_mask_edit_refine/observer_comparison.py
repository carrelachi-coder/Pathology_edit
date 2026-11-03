"""Strict offline evidence checks. No model call, retry or qualification update."""
import hashlib
import json
from pathlib import Path

import numpy as np


def _load(directory):
    root = Path(directory)
    saved = json.loads((root / 'observer_status.json').read_text())
    status = saved['status']
    if not (status['valid'] and status['closed'] and status['interpretation_permitted']):
        raise ValueError('invalid or incomplete observer evidence')
    lines = (root / 'semantic.jsonl').read_text().splitlines(keepends=True)
    chain = '0' * 64
    events = []
    for i, line in enumerate(lines):
        event = json.loads(line)
        if event['sequence'] != i:
            raise ValueError('event sequence gap')
        events.append(event)
        chain = hashlib.sha256((chain + line).encode('utf-8')).hexdigest()
    if not events or len(events) != status['events'] or chain != status['semantic_chain_sha256']:
        raise ValueError('semantic evidence truncated or modified')
    arrays = []
    def walk(value):
        if type(value) is dict:
            if list(value) == ['array']:
                arrays.append(value['array'])
            else:
                for nested in value.values():
                    walk(nested)
        elif type(value) is list:
            for nested in value:
                walk(nested)
    for event in events:
        walk(event)
    if status['mode'] == 'full_capture' and len(arrays) != len(saved['artifacts']):
        raise ValueError('full capture raster inventory incomplete')
    if status['mode'] == 'hash_only' and saved['artifacts']:
        raise ValueError('hash-only evidence contains unexpected raster artifacts')
    for index, item in enumerate(saved['artifacts']):
        name = item['file']
        if Path(name).name != name:
            raise ValueError('artifact path escapes evidence directory')
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('raster artifact modified or missing')
        array = np.load(root / name, allow_pickle=False)
        expected = arrays[index]
        if array.dtype.str != expected['dtype'] or list(array.shape) != expected['shape'] or hashlib.sha256(array.tobytes(order='C')).hexdigest() != expected['content_sha256']:
            raise ValueError('raster content does not match semantic transcript')
    return events, status


def compare_observer_runs(hash_directory, full_directory):
    """No post-hoc field exclusions: compare every semantic value and its order."""
    try:
        left, ls = _load(hash_directory)
        right, rs = _load(full_directory)
        if ls['mode'] != 'hash_only' or rs['mode'] != 'full_capture':
            raise ValueError('comparison requires B hash_only and C full_capture')
        if left != right:
            mismatch = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            return {'status': 'mismatch_hold', 'first_different_event': mismatch,
                    'qualification_updates_permitted': False}
        return {'status': 'semantic_match', 'events': len(left), 'excluded_semantic_fields': [],
                'qualification_updates_permitted': False,
                'scope': 'B/C semantic transcript only; original A and historical comparison require separately retained business receipts'}
    except Exception as exc:
        return {'status': 'mismatch_hold', 'reason': type(exc).__name__ + ': ' + str(exc),
                'qualification_updates_permitted': False}


def rejection_pixel_sets(events):
    """Offline unique sets per grow invocation, preserving cross-guard overlap.

    Coordinates retain repeated-visit counts separately. No causal loss estimate.
    """
    output = []
    current = None
    for event in events:
        if event['event'] == 'grow.begin':
            current = {'begin_sequence': event['sequence'], 'rejections': {}, 'event_counts': {}}
            output.append(current)
        elif event['event'] in ('guard.reject', 'guard.local_result') and current is not None:
            values = dict(event['values']['mapping'])
            reason = values['reason']
            if reason is not None:
                current['rejections'].setdefault(reason, set()).add((int(values['row']), int(values['col'])))
                current['event_counts'][reason] = current['event_counts'].get(reason, 0) + 1
    return output
