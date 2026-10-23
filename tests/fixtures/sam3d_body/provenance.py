# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed identities for unchanged SAM 3D Body reference execution."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f'Duplicate JSON key: {key}')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def pin():
    return read_json(Path(__file__).with_name('provenance.json'))


def verify_dependencies():
    expected = pin()['runtime']['dependency_files']
    for module, record in expected.items():
        spec = importlib.util.find_spec(module)
        require(spec is not None and spec.origin is not None, f'Missing dependency: {module}')
        require(str(Path(spec.origin).resolve()) == record['origin'], f'Dependency origin differs: {module}')
        for filename, digest in record['files'].items():
            require(sha(filename) == digest, f'Dependency file differs: {filename}')
    interpreter = pin()['runtime']['interpreter']
    require(str(Path(sys.executable).resolve()) == interpreter['path'] and sha(sys.executable) == interpreter['sha256'] and sys.version == interpreter['version'], 'Interpreter differs')
    return {'interpreter': interpreter, 'dependency_files': expected,
            'trust_limit': 'Pins relevant package Python/extension files and interpreter; OS libraries and startup configuration remain trusted.'}


def verify_source_runtime(source, mx):
    p = pin()
    source = Path(source).resolve()
    require({x.name for x in source.glob('*.py')} == set(p['source_files']), 'Source inventory differs')
    for name, expected in p['source_files'].items():
        require(sha(source / name) == expected, f'Source differs: {name}')
    import mlx.core
    import cv2
    import numpy
    import safetensors
    import ml_dtypes
    import importlib.metadata
    expected = p['runtime']
    for name, value in [('opencv', cv2.__version__), ('numpy', numpy.__version__),
                        ('safetensors', safetensors.__version__), ('ml_dtypes', ml_dtypes.__version__)]:
        require(value == expected[name], f'Dependency differs: {name}')
    version = getattr(mx, '__version__', importlib.metadata.version('mlx'))
    require(version == expected['version'], 'MLX version differs')
    binding = Path(mlx.core.__file__).resolve()
    require(sha(binding) == expected['binding_sha256'], 'Binding differs')
    for name, digest in expected['python_files'].items():
        require(sha(binding.parents[2] / name) == digest, f'MLX Python module differs: {name}')
    mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if line.split()[-1].endswith('/libmlx.so')}
    require(len(mapped) == 1, 'Ambiguous mapped native MLX')
    native = Path(mapped.pop()).resolve()
    require(sha(native) == expected['native_sha256'], 'Native runtime differs')
    require(os.environ.get('MLX_ENABLE_TF32') == '0', 'TF32 must be explicitly disabled')
    return {'version': version, 'binding_path': str(binding), 'native_path': str(native),
            'binding_sha256': sha(binding), 'native_sha256': sha(native),
            'dependencies': {k: expected[k] for k in ('opencv', 'numpy', 'safetensors', 'ml_dtypes')}}


def load_upstream(source):
    source = Path(source).resolve()
    name = 'pinned_sam3d_body'
    require(name not in sys.modules, 'Body package already imported')
    spec = importlib.util.spec_from_file_location(name, source / '__init__.py',
                                                submodule_search_locations=[str(source)])
    require(spec is not None and spec.loader is not None, 'Cannot locate Body package')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    p = pin()['checkpoint']
    for name, record in p['files'].items():
        path = bundle / name
        require(path.is_file() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
                f'Checkpoint differs: {name}')
    import struct
    with (bundle / 'model.safetensors').open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop('__metadata__', None)
    counts = {}
    for value in header.values():
        counts[value['dtype']] = counts.get(value['dtype'], 0) + 1
    require(len(header) == p['tensor_count'] and counts == p['dtype_counts'], 'Checkpoint dtype/schema differs')
    index = read_json(bundle / 'model.safetensors.index.json')['weight_map']
    require(set(index) == set(header) and set(index.values()) == {'model.safetensors'}, 'Index differs')
    return header


def verify_image(path):
    import cv2
    import numpy as np
    p = pin()['input']
    require(sha(path) == p['canonical_sha256'], 'Canonical image differs')
    image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    require(list(image.shape) == p['shape'] and image.dtype == np.uint8, 'Image schema differs')
    require(hashlib.sha256(image.tobytes()).hexdigest() == p['rgb_sha256'], 'Decoded RGB differs')
    return image


def authenticate_helpers(manifest_path, expected):
    require(sha(manifest_path) == expected, 'Unapproved helper manifest')
    manifest = read_json(manifest_path)
    require(set(manifest) == {'generate_reference.py', 'provenance.py', 'provenance.json'}, 'Helper inventory differs')
    root = Path(manifest_path).parent
    for name, digest in manifest.items():
        require(sha(root / name) == digest, f'Executed helper differs: {name}')
    return manifest
