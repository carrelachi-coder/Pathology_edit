"""Content identities and fail-closed reuse for research artifacts (no ML imports)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_payload(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def asset_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    if path.is_file():
        return {'path': str(path), 'sha256': sha256_file(path)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    index = path / 'model_index.json'
    if index.is_file():
        # Diffusers loads its declared component folders, not duplicate monolithic
        # exports or preview images stored beside them.
        definition = json.loads(index.read_text())
        components = [path / name for name, value in definition.items()
                      if not name.startswith('_') and isinstance(value, list) and any(value)]
        if any(not component.is_dir() for component in components):
            raise FileNotFoundError('missing model_index component directory')
        candidates = [index] + [p for component in components for p in component.rglob('*')]
    else:
        candidates = path.rglob('*')
    files = {str(p.relative_to(path)): sha256_file(p) for p in sorted(candidates)
             if p.is_file() and not any(part in {'.git', '.cache', '__pycache__'} for part in p.relative_to(path).parts)}
    if not files:
        raise ValueError(f'empty model directory: {path}')
    return {'path': str(path), 'sha256': digest_payload(files), 'files': files}


def validate_resume(record_path: Path, expected: dict, outputs: list[Path]) -> bool:
    """Reject partial/legacy/drifted runs without overwriting their artifacts."""
    occupied = record_path.exists() or any(path.exists() for path in outputs) or (record_path.parent.is_dir() and any(record_path.parent.iterdir()))
    if not occupied:
        return False
    if not record_path.is_file():
        raise ValueError('existing artifacts lack experiment identity; use a new output root')
    saved = json.loads(record_path.read_text())
    if saved.get('identity') != expected:
        raise ValueError('experiment identity drift; preserve old artifacts and use a new output root')
    hashes = saved.get('outputs', {})
    if not outputs or any(not p.is_file() or hashes.get(p.name) != sha256_file(p) for p in outputs):
        raise ValueError('existing run is incomplete or output content changed; use a new output root')
    return True
