"""Opt-in, fail-isolated evidence sink. No runner, model call or qualification gate.

Activate explicitly with ``observation_session`` in a diagnostic harness. Business
code ignores every observe() return value. Failures invalidate evidence in memory;
they never propagate into the compiler's retry/qualification exception handlers.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import sys

import numpy as np

_ACTIVE = ContextVar('passive_topology_observer', default=None)
_CHECKPOINTS = frozenset({'qualification.begin', 'compiler.attempt', 'planner.ready',
                         'execution.begin', 'probe.complete', 'grow.begin', 'grow.end',
                         'compiler.exception', 'qualification.exception'})


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


HASH_CHUNK_BYTES = 1024 * 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    # Caller supplies a C-contiguous detached array or read-only mmap. No tobytes().
    digest = hashlib.sha256()
    if array.nbytes:
        data = memoryview(array).cast('B')
        for start in range(0, len(data), HASH_CHUNK_BYTES):
            digest.update(data[start:start + HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


class PassiveObserver:
    """Synchronous detached capture, identical semantic stream in both modes.

    No arbitrary object callbacks, repr(), to_metadata(), generator consumption,
    worker thread, random file names, cache lookup, or policy decisions.
    """
    def __init__(self, directory, *, mode, max_events=2_000_000,
                 max_bytes=4 * 1024**3, max_array_bytes=64 * 1024**2,
                 max_artifacts=100_000, max_event_metadata_bytes=8 * 1024**2):
        if mode not in ('hash_only', 'full_capture'):
            raise ValueError('mode must be hash_only or full_capture')
        self.directory = Path(directory)
        self.mode = mode
        self.max_events, self.max_bytes = max_events, max_bytes
        self.max_array_bytes, self.max_artifacts = max_array_bytes, max_artifacts
        self.max_event_metadata_bytes = max_event_metadata_bytes
        self._metadata_used = 0
        self.valid = True
        self.faults = []
        self.events = 0
        self.bytes_written = 0
        self.artifacts = []
        self._arrays = 0
        self._queue = queue.Queue(maxsize=1)
        self._stream = None
        self._chain = '0' * 64
        self._closed = False
        self._owns_directory = False
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._owns_directory = True
            self._stream = (self.directory / 'semantic.jsonl').open('x', encoding='utf-8')
        except Exception as exc:
            self.invalidate('initialize', exc)

    def invalidate(self, stage, exc):
        # Do not call the exception's __str__ or use business exception objects.
        self.valid = False
        if not self.faults:
            self.faults.append({'stage': stage, 'type': type(exc).__name__})

    def _copy_array(self, value):
        if value.dtype.hasobject:
            raise TypeError('object arrays are not passive evidence')
        return np.array(value, copy=True, order='C', subok=False)

    def _write_raster(self, array):
        filename = f'array_{self._arrays:08d}.npy'
        path = self.directory / filename
        # Lossless detached bytes; the file digest is outside semantic comparison.
        with path.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        self.bytes_written += path.stat().st_size
        self.artifacts.append({'file': filename, 'sha256': file_sha256(path)})

    def _reserve_metadata(self, value):
        # Conservative payload accounting, not a claim to cap Python/process RSS.
        charge = 512
        if type(value) is str:
            charge += 12 * len(value)  # ensure_ascii can use two Unicode escapes.
        elif type(value) is int:
            charge += value.bit_length() // 3 + 2
        self._metadata_used += charge
        if self._metadata_used > self.max_event_metadata_bytes:
            raise MemoryError('diagnostic event metadata limit')

    def _encode(self, value):
        self._reserve_metadata(value)
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if math.isfinite(value):
                return value
            return {'nonfinite': 'nan' if math.isnan(value) else ('+inf' if value > 0 else '-inf')}
        if isinstance(value, np.generic):
            return self._encode(value.item())
        if type(value) is np.ndarray:
            # Check the logical size even for a huge strided/broadcast view BEFORE
            # allocating a detached copy. Storage and buffer limits are separate.
            if value.dtype.hasobject:
                raise TypeError('object arrays are not passive evidence')
            if value.nbytes > self.max_array_bytes:
                raise MemoryError('diagnostic detached-array limit')
            if self._arrays >= self.max_artifacts:
                raise MemoryError('diagnostic artifact inventory limit')
            if self.bytes_written + value.nbytes + 4096 > self.max_bytes:
                raise OSError('diagnostic storage limit before copy')
            array = self._copy_array(value)
            self._arrays += 1
            encoded = {'array': {'dtype': array.dtype.str, 'shape': list(array.shape),
                                 'content_sha256': array_sha256(array)}}
            if self.mode == 'full_capture':
                self._write_raster(array)
            return encoded
        if type(value) is np.random.Generator:
            return {'local_numpy_generator_state': self._encode(value.bit_generator.state)}
        if type(value) in (list, tuple, set, frozenset):
            return {'sequence_type': type(value).__name__, 'items': [self._encode(v) for v in value]}
        if type(value) is dict:
            # Encode insertion order and key types instead of sorting business data.
            return {'mapping': [[self._encode(k), self._encode(v)] for k, v in value.items()]}
        if isinstance(value, BaseException):
            return {'exception_type': type(value).__module__ + '.' + type(value).__qualname__,
                    'args': self._encode(value.args), 'attributes': self._encode(vars(value))}
        if is_dataclass(value) and not isinstance(value, type):
            return {'dataclass': type(value).__module__ + '.' + type(value).__qualname__,
                    'fields': [[f.name, self._encode(object.__getattribute__(value, f.name))]
                               for f in fields(value)]}
        # Never exhaust arbitrary iterables or invoke custom serialization code.
        raise TypeError('unsupported passive evidence type')

    def _rng(self):
        random_module = sys.modules.get('random')
        python_state = random_module.getstate() if random_module is not None else {'unavailable': 'random not already loaded'}
        states = {'python': python_state, 'numpy_legacy': np.random.get_state(),
                  'torch_cpu': {'unavailable': 'torch not already loaded'},
                  'torch_cuda': {'unavailable': 'not queried; observer never initializes CUDA'},
                  'unpassed_local_generators': {'unavailable': 'no generator discovery or initialization'}}
        torch = sys.modules.get('torch')
        if torch is not None:
            generator = vars(torch).get('default_generator')
            if generator is not None:
                states['torch_cpu'] = generator.get_state().numpy()
        return states

    def record(self, event, values):
        if not self.valid:
            return
        if self._closed or self.events >= self.max_events:
            raise RuntimeError('closed observer or event limit')
        self._metadata_used = 0
        self._reserve_metadata(event)
        record = {'sequence': self.events, 'event': event, 'values': self._encode(values)}
        if event in _CHECKPOINTS:
            # Read existing states without draws; buffers use the same lossless encoding.
            record['rng'] = self._encode(self._rng())
        self._queue.put_nowait(record)
        pending = self._queue.get_nowait()
        line = _json(pending) + '\n'
        if self.bytes_written + len(line.encode('utf-8')) > self.max_bytes:
            raise OSError('diagnostic storage limit')
        self._write_line(line)
        self.bytes_written += len(line.encode('utf-8'))
        self._chain = hashlib.sha256((self._chain + line).encode('utf-8')).hexdigest()
        self.events += 1

    def _write_line(self, line):
        self._stream.write(line)

    def status(self):
        return {'schema_version': 2, 'valid': self.valid, 'interpretation_permitted': self.valid and self._closed and self.events > 0,
                'mode': self.mode, 'events': self.events, 'semantic_chain_sha256': self._chain,
                'faults': [dict(f) for f in self.faults], 'closed': self._closed,
                'bytes_written': self.bytes_written, 'arrays': self._arrays,
                'resource_limits': {'max_array_bytes': self.max_array_bytes,
                                    'max_artifacts': self.max_artifacts,
                                    'max_event_metadata_bytes': self.max_event_metadata_bytes,
                                    'hash_chunk_bytes': HASH_CHUNK_BYTES},
                'qualification_updates_permitted': False}

    def close(self):
        if self._closed:
            return
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception as exc:
            self.invalidate('close_stream', exc)
        self._closed = True
        if not self._owns_directory:
            return
        try:
            # Refuse existing summary files as well; output is never a resume source.
            with (self.directory / 'observer_status.json').open('x', encoding='utf-8') as out:
                out.write(_json({'status': self.status(), 'artifacts': self.artifacts}) + '\n')
            if self.valid and self.events:
                self._publish_completion()
        except Exception as exc:
            self.invalidate('close_summary', exc)
        # status() remains available even if no file could be written.


    def _publish_completion(self):
        # Only publish after summary close succeeds. A failed pending-marker close
        # cannot leave an apparently complete run. No random names or overwrite.
        pending = self.directory / 'observer_complete.pending'
        final = self.directory / 'observer_complete.json'
        with pending.open('x', encoding='utf-8') as stream:
            stream.write(_json({'summary_sha256': file_sha256(self.directory / 'observer_status.json')}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if final.exists():
            raise FileExistsError('completion marker already exists')
        pending.rename(final)


def observe(event, **values):
    """No business return data; all sink failures stay outside business handlers."""
    observer = _ACTIVE.get()
    if observer is not None:
        try:
            observer.record(event, values)
        except Exception as exc:
            observer.invalidate('record', exc)
    return None


@contextmanager
def observation_session(observer):
    if _ACTIVE.get() is not None:
        raise RuntimeError('nested observation sessions are not supported')
    token = _ACTIVE.set(observer)
    try:
        yield observer
    except BaseException as exc:
        observer.invalidate('session_aborted', exc)
        raise
    finally:
        _ACTIVE.reset(token)
        observer.close()
