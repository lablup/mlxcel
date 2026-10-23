"""Regenerate small authentic CPU writer and FlexiCubes fixtures."""
from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
from provenance import canonical, sha, verify_source_runtime
from upstream_loader import load_upstream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    runtime = verify_source_runtime(args.source, mx)
    load_upstream(args.source)
    from mlx_vlm.models.sam3d_objects.generate import write_gaussians, write_obj
    from mlx_vlm.models.sam3d_objects.mesh import CORNERS, extract_mesh
    from mlx_vlm.models.sam3d_objects.sparse import Grid
    gaussian = {
        'positions': [[0, 1, 2], [-1, 0.5, 3], [4, -2, 0]],
        'sh_dc': [[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6], [0.7, -0.8, 0.9]],
        'scales': [[1, 2, 0.5], [0.25, 4, 3], [2, 0.125, 8]],
        'rotations': [[1, 0, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]],
        'opacities': [[0], [0.5], [1]],
    }
    triangle = {'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0]], 'faces': [[0, 1, 2]], 'vertex_colors': [[1, 0, 0, 9, 8, 7], [0, 1, 0, 6, 5, 4], [0, 0, 1, 3, 2, 1]]}
    with tempfile.TemporaryDirectory() as scratch:
        ply = Path(scratch) / 'three.ply'; obj = Path(scratch) / 'triangle.obj'
        write_gaussians(ply, {k: mx.array(v, mx.float32) for k, v in gaussian.items()})
        write_obj(obj, {k: mx.array(v, mx.int32 if k == 'faces' else mx.float32) for k, v in triangle.items()})
        header, binary = ply.read_bytes().split(b'end_header\n', 1)
        writers = {'gaussian_input': gaussian, 'ply_header': (header + b'end_header\n').decode(), 'ply_rows': np.frombuffer(binary, '<f4').reshape(3, 17).tolist(), 'mesh_input': triangle, 'obj_text': obj.read_text()}
    r = 8
    xyz = np.stack(np.meshgrid(np.arange(r), np.arange(r), np.arange(r), indexing='ij'), -1).reshape(-1, 3).astype(np.int32)
    coords = np.concatenate([np.zeros((len(xyz), 1), np.int32), xyz], axis=1)
    corners = np.array(CORNERS, np.float32)
    positions = (xyz[:, None].astype(np.float32) + corners[None]) / np.float32(r) - np.float32(0.5)
    sdf = np.sqrt(np.sum(positions * positions, axis=-1)) - np.float32(0.3)
    features = np.zeros((len(xyz), 101), np.float32)
    features[:, :8] = sdf + np.float32(1 / r)
    mesh = extract_mesh(mx.array(features), Grid(mx.array(coords), r))
    mx.eval(mesh)
    sphere = {'resolution': r, 'radius': 0.3, 'coords': coords.tolist(), 'features': features.tolist(), 'mesh': {k: np.asarray(v).tolist() for k, v in mesh.items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('writers.json', writers), ('sphere.json', sphere)]:
        (args.output / name).write_text(json.dumps(value, separators=(',', ':')) + '\n')
    provenance = {'source_revision': canonical()['source_revision'], 'source_sha256': canonical()['source_sha256'], 'runtime': runtime, 'device': 'cpu', 'generator_sha256': sha(Path(__file__)), 'files': {name: sha(args.output / name) for name in ('writers.json', 'sphere.json')}}
    (args.output / 'small-oracle-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'vertices': mesh['vertices'].shape[0], 'faces': mesh['faces'].shape[0], 'files': provenance['files']}))


if __name__ == '__main__':
    main()
