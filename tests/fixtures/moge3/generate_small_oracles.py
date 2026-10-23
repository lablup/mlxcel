"""Regenerate tiny authentic CPU sparse, geometry and searchsorted literals."""
import argparse
import json
import pathlib
import mlx.core as mx
from upstream_loader import load_upstream
from provenance import canonical, require, sha, verify_source_runtime
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path(__file__).parent)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
mx.set_default_device(mx.cpu)
runtime_receipt = verify_source_runtime(args.source, mx)
load_upstream(args.source)
from mlx_vlm.models.moge3.sparse import SubmanifoldConv3d, sparse_pool2x_mean, sparse_upsample2x_nearest
coords = mx.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 2, 2, 2], [0, 4, 4, 4]], dtype=mx.int32)
shape = (1, 5, 5, 5)
feats = mx.arange(12, dtype=mx.float32).reshape(6, 2) / 8
conv = SubmanifoldConv3d(2, 3)
conv.weight = mx.arange(162, dtype=mx.float32).reshape(3, 3, 3, 3, 2) / 128
conv.bias = mx.array([0.25, -0.5, 1.0])
out, nmap = conv(feats, coords, shape)
pf, pc, ps, parent = sparse_pool2x_mean(feats, coords, shape)
up, _, _ = sparse_upsample2x_nearest(pf, parent, coords, shape)
obj = {'source_revision': 'abd3b2d7b728cb5372d4e092260da325e5ded209', 'device': 'cpu', 'shape': shape, 'coords': coords.tolist(), 'features': feats.tolist(), 'weight': conv.weight.tolist(), 'bias': conv.bias.tolist(), 'neighbors': nmap.tolist(), 'conv': out.tolist(), 'pool_features': pf.tolist(), 'pool_coords': pc.tolist(), 'pool_shape': ps, 'parents': parent.tolist(), 'upsample': up.tolist()}
(args.output / 'sparse-oracle.json').write_text(json.dumps(obj, indent=2) + '\n')
print('six-voxel authentic sparse oracle complete')
from mlx_vlm.models.moge3.geometry import normalized_view_plane_uv, recover_focal_shift
uv = normalized_view_plane_uv(16, 12)
z = 2.0 + uv[..., 0] * 0.4 + uv[..., 1] * 0.3
points = mx.concatenate([uv * z[..., None] / 1.7, z[..., None]], axis=-1)
focal, shift = recover_focal_shift(points)
fixed_focal, fixed_shift = recover_focal_shift(points, focal=mx.array(1.7))
empty_focal, empty_shift = recover_focal_shift(points, mask=mx.zeros((12, 16), dtype=mx.bool_))
sorted_values = mx.array([-(1 << 40), -3, 0, 0, (1 << 24) + 1, 1 << 40], dtype=mx.int64)
queries = mx.array([-(1 << 41), -(1 << 40), -2, 0, 1, 1 << 24, (1 << 24) + 1, 1 << 40, 1 << 41], dtype=mx.int64)
small = {'geometry': {'focal': focal.item(), 'shift': shift.item(), 'fixed_focal': fixed_focal.item(), 'fixed_shift': fixed_shift.item(), 'empty_focal': empty_focal.item(), 'empty_shift': empty_shift.item(), 'known_focal': 1.7, 'shape': [12, 16]}, 'searchsorted': {'sorted': sorted_values.tolist(), 'queries': queries.tolist(), 'left': mx.searchsorted(sorted_values, queries, side='left').tolist(), 'right': mx.searchsorted(sorted_values, queries, side='right').tolist(), 'empty': mx.searchsorted(mx.array([], dtype=mx.int64), queries).tolist()}}
shifted_focal, shifted_z = recover_focal_shift(points - mx.array([0.0, 0.0, 0.35]))
small["geometry"]["shifted_focal"] = shifted_focal.item()
small["geometry"]["shifted_z"] = shifted_z.item()
small["geometry"]["known_shift"] = 0.35
require(abs(shifted_focal.item() - 1.7) <= 1e-4, "Shifted focal oracle exceeds tolerance")
require(abs(shifted_z.item() - 0.35) <= 1e-4, "Shift oracle exceeds tolerance")
require(abs(small['geometry']['focal'] - 1.7) <= 1e-4, 'Focal oracle exceeds tolerance')
(args.output / 'geometry-search-oracle.json').write_text(json.dumps(small, indent=2) + '\n')
print(json.dumps(small))

receipt = {
    "source_revision": canonical()["source_revision"],
    "source_sha256": canonical()["source_sha256"],
    "runtime_receipt": runtime_receipt,
    "device": str(mx.default_device()),
    "generator_sha256": sha(pathlib.Path(__file__)),
    "outputs": {name: sha(args.output / name) for name in ("sparse-oracle.json", "geometry-search-oracle.json")},
}
(args.output / "small-oracle-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
