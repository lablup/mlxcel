"""Fail-closed identities and complete Objects output contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical() -> dict[str, Any]:
    return json.loads((HERE / 'provenance.json').read_text())


def verify_source_runtime(source: Path, mx: Any) -> dict[str, Any]:
    expected = canonical()
    for name, digest in expected['source_sha256'].items():
        require(sha(source / name) == digest, f'Source mismatch: {name}')
    binding = Path(mx.__file__).resolve()
    require(mx.__version__ == expected['runtime']['version'], 'MLX version mismatch')
    require(sha(binding) == expected['runtime']['binding_sha256'], 'Binding mismatch')
    mapped = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines() if line.split()[-1].endswith('/libmlx.so')}
    require(len(mapped) == 1, 'Expected exactly one mapped native MLX library')
    native = mapped.pop()
    require(sha(native) == expected['runtime']['native_sha256'], 'Mapped native mismatch')
    return {'version': mx.__version__, 'binding_path': str(binding), 'native_path': str(native), **expected['runtime']}


def verify_bundle(bundle: Path) -> None:
    for name, row in canonical()['checkpoint']['files'].items():
        path = bundle / name
        require(path.is_file() and path.stat().st_size == row['bytes'], f'Checkpoint size mismatch: {name}')
        require(sha(path) == row['sha256'], f'Checkpoint hash mismatch: {name}')


def verify_input(path: Path, name: str) -> None:
    row = canonical()['inputs'][name]
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], f'Input mismatch: {name}')


def validate_result(flat: dict[str, Any], formats: list[str], conditioned: bool) -> dict[str, Any]:
    required = {'coords', 'latents', 'pose.rotation', 'pose.rotation_matrix', 'pose.translation', 'pose.scale'}
    geometry = {'mesh': ('vertices', 'faces', 'vertex_colors'), 'gaussian': ('positions', 'sh_dc', 'scales', 'rotations', 'opacities'), 'gaussian_4': ('positions', 'sh_dc', 'scales', 'rotations', 'opacities')}
    for kind in formats:
        required.update(f'{kind}.{key}' for key in geometry[kind])
    require(set(flat) == required, 'Incomplete or unexpected result tensor keys')
    stats = {}
    for name, value in flat.items():
        array = np.asarray(value)
        require(array.size > 0 and bool(np.isfinite(array).all()), f'Empty/nonfinite output: {name}')
        stats[name] = {'shape': list(array.shape), 'dtype': str(value.dtype), 'elements': array.size, 'finite': int(np.isfinite(array).sum())}
    coords = np.asarray(flat['coords'])
    require(coords.ndim == 2 and coords.shape[1] == 4 and coords.dtype.kind in 'iu', 'Invalid coordinates')
    require(bool((coords[:, 0] == 0).all()) and bool(((coords[:, 1:] >= 0) & (coords[:, 1:] < 64)).all()), 'Coordinate range mismatch')
    require(np.asarray(flat['latents']).shape == (len(coords), 8), 'Latent shape mismatch')
    for name, shape in {'rotation': (4,), 'rotation_matrix': (3, 3), 'translation': (3,), 'scale': (3,)}.items():
        require(np.asarray(flat['pose.' + name]).shape == shape, 'Pose shape mismatch')
    require(bool((np.asarray(flat['pose.scale']) > 0).all()), 'Invalid pose scale')
    if 'mesh' in formats:
        vertices, faces, colors = (np.asarray(flat['mesh.' + key]) for key in geometry['mesh'])
        require(vertices.ndim == 2 and vertices.shape[1] == 3 and colors.shape == (len(vertices), 6), 'Mesh vertex schema mismatch')
        require(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in 'iu', 'Face schema mismatch')
        require(bool(((faces >= 0) & (faces < len(vertices))).all()), 'Invalid face index')
    for kind in set(formats) - {'mesh'}:
        count = np.asarray(flat[kind + '.positions']).shape[0]
        for name, width in {'positions': 3, 'sh_dc': 3, 'scales': 3, 'rotations': 4, 'opacities': 1}.items():
            require(np.asarray(flat[kind + '.' + name]).shape == (count, width), 'Gaussian shape mismatch')
        require(bool((np.asarray(flat[kind + '.scales']) > 0).all()), 'Nonpositive Gaussian scale')
        opacity = np.asarray(flat[kind + '.opacities'])
        require(bool(((opacity >= 0) & (opacity <= 1)).all()), 'Invalid opacity')
    require(type(conditioned) is bool, 'Conditioning metadata must be bool')
    return stats
