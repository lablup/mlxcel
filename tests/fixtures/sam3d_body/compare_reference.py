# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strict all-vertex Body comparison after independent capture authentication."""
from pathlib import Path
import argparse
import json
import hashlib
import importlib.util
import re
import sys


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def bootstrap(path, digest):
    require(sha(path) == digest, 'Unapproved capture plan')
    plan = read_json(path)
    own = plan['helpers']['compare_reference.py']
    require(Path(own['path']).resolve() == Path(__file__).resolve() and sha(Path(__file__)) == own['sha256'], 'Executing comparator differs')
    helper = plan['helpers']['capture_identity.py']
    hp = Path(helper['path'])
    require(hp.is_absolute() and hp.resolve() == hp and sha(hp) == helper['sha256'], 'Validator identity differs')
    spec = importlib.util.spec_from_file_location('body_pinned_capture_identity', hp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies(canonical):
    interpreter = canonical['runtime']['interpreter']
    require(str(Path(sys.executable).resolve()) == interpreter['path'] and sha(sys.executable) == interpreter['sha256'] and sys.version == interpreter['version'], 'Comparator interpreter differs')
    for name in ('numpy','safetensors'):
        record = canonical['runtime']['dependency_files'][name]
        spec = importlib.util.find_spec(name)
        require(spec is not None and str(Path(spec.origin).resolve()) == record['origin'], 'Comparator dependency origin differs')
        for filename, digest in record['files'].items():
            require(sha(filename) == digest, 'Comparator dependency bytes differ')


def lineage(root, manifest, receipt, canonical, stem, plan):
    records = manifest['_lineage']
    for name, record in records.items():
        require(sha(root/name) == record['sha256'] and (root/name).stat().st_size == record['bytes'], 'Reference lineage differs')
    row = read_json(root/'reference-source-lineage.json')
    archive = root/'executed-reference-source'
    require({p.name for p in archive.iterdir()} == set(row['executed_source']), 'Executed helper inventory differs')
    for name, digest in row['executed_source'].items():
        require(sha(archive/name) == digest, 'Executed helper differs')
    reviewed = read_json(archive/'reviewed-manifest.json.txt')
    require(receipt['helpers'] == reviewed and set(reviewed) == {'generate_reference.py','provenance.py','provenance.json'}, 'Reference helper binding differs')
    for name, digest in reviewed.items(): require(sha(archive/(name+'.txt')) == digest, 'Executed reviewed helper differs')
    require(read_json(archive/'provenance.json.txt') == canonical, 'Executed reference provenance differs')
    require(receipt['dependency_identity']['interpreter'] == canonical['runtime']['interpreter'] and receipt['dependency_identity']['dependency_files'] == canonical['runtime']['dependency_files'], 'Reference dependency binding differs')
    unused = {'body_model.parameter_limits','body_model.pmi','body_model.texcoord_faces','body_model.texcoords'}
    require(set(receipt['loader_compatibility']['filtered_head_keys']) == unused and receipt['loader_compatibility']['forward_and_predict_unchanged'] is True, 'Loader compatibility differs')
    resource = read_json(root/(stem+'.resources.json'))
    require(resource['child_exit'] == 0 and resource['stopped_for_memory'] is False and resource['stop_reason'] is None, 'Reference resources indicate failure')
    require(resource['monitor_sha256'] == sha(archive/'run_reference.py.txt'), 'Reference monitor differs')
    require(receipt['output_sha256'] == manifest[plan['case']]['files'][stem+'.safetensors']['sha256'], 'Receipt tensor hash differs')
    import struct
    with (Path(plan['checkpoint_dir'])/'model.safetensors').open('rb') as f:
        header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
    header.pop('__metadata__', None)
    excluded = ('decoder_hand.','head_pose_hand.','head_camera_hand.','init_pose_hand.','init_camera_hand.','init_to_token_mhr_hand.','prev_to_token_mhr_hand.','keypoint_embedding_hand.','keypoint3d_embedding_hand.','keypoint_posemb_linear_hand.','keypoint3d_posemb_linear_hand.','keypoint_feat_linear_hand.','ray_cond_emb_hand.','prompt_encoder.mask_downscaling.')
    bare = {'init_camera_hand','init_pose_hand','keypoint3d_embedding_hand','keypoint_embedding_hand'}
    keys = {k for k in header if not k.startswith(excluded) and k not in bare and k not in {'head_pose.'+n for n in unused}}
    require(len(keys) == 958 and set(receipt['loaded_parameters']) == keys, 'Loaded parameter inventory differs')
    dtype = {'BF16':'mlx.core.bfloat16','F32':'mlx.core.float32','I32':'mlx.core.int32','U8':'mlx.core.uint8'}
    for key in keys:
        value = receipt['loaded_parameters'][key]
        require(value['shape'] == header[key]['shape'] and value['dtype'] == dtype[header[key]['dtype']] and re.fullmatch('[0-9a-f]{64}',value['sha256']) is not None, 'Loaded parameter schema/digest differs')



def compare(reference, actual, mesh_path):
    import numpy as np
    expected = {'vertices': ('pred_vertices', (18439, 3)),
                'keypoints_3d': ('pred_keypoints_3d', (70, 3)),
                'joint_coords': ('pred_joint_coords', (127, 3)),
                'camera': ('pred_camera', (3,))}
    require(set(actual) == set(expected), 'Actual tensor inventory differs')
    require(set(reference) == {v[0] for v in expected.values()} | {'bbox', 'faces'}, 'Reference inventory differs')
    metrics = {}; failures = []
    for name, (ref_name, shape) in expected.items():
        x, y = reference[ref_name], actual[name]
        require(x.shape == y.shape == shape and x.dtype == y.dtype == np.float32, f'Schema differs: {name}')
        require(bool(np.isfinite(x).all()) and bool(np.isfinite(y).all()), f'Nonfinite tensor: {name}')
        delta = y.astype(np.float64) - x.astype(np.float64)
        maximum = float(np.max(np.abs(delta)))
        relative = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(x.astype(np.float64).ravel()), 1e-12))
        metrics[name] = {'max_abs': maximum, 'relative_l2': relative, 'elements': int(x.size)}
        if name in ('vertices', 'keypoints_3d') and not maximum < 1e-3:
            failures.append(f'{name}: max_abs {maximum} must be <1e-3 metres')
        if name == 'camera' and not relative < 1e-3:
            failures.append(f'camera: relative L2 {relative} must be <1e-3')
    vertices = []; faces = []
    for line in mesh_path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('#'): continue
        require(fields[0] in ('v', 'f') and len(fields) == 4, 'OBJ must contain colorless vertices and triangles only')
        if fields[0] == 'v':
            require(all(re.fullmatch(r'-?[0-9]+\.[0-9]{6}',v) is not None for v in fields[1:]), 'OBJ vertex precision must be six fixed decimals')
            vertices.append([float(v) for v in fields[1:]])
        else: faces.append([int(v)-1 for v in fields[1:]])
    v = np.asarray(vertices, np.float64); f = np.asarray(faces, np.int32)
    require(v.shape == (18439, 3) and f.shape == (36874, 3), 'OBJ topology dimensions differ')
    require(bool(np.isfinite(v).all()) and np.array_equal(f, reference['faces']), 'OBJ topology or finite support differs')
    require(float(np.max(np.abs(v-actual['vertices']))) <= 0.00000051, 'OBJ vertices differ from tensor beyond six-decimal serialization')
    require(float(np.max(np.ptp(v,axis=0))) > 0, 'Collapsed mesh')
    metrics['mesh'] = {'vertices': len(vertices), 'faces': len(faces), 'faces_exact': True,
                       'tensor_serialization_max_abs': float(np.max(np.abs(v-actual['vertices'])))}
    return metrics, failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    require(not a.report.exists(), 'Refuse to overwrite comparison report')
    validator = bootstrap(a.plan, a.plan_sha256)
    plan, canonical, _ = validator.authenticate_plan(a.plan, a.plan_sha256)
    capture = validator.validate_capture(a.plan, a.plan_sha256)
    dependencies(canonical)
    root = Path(__file__).parent
    manifest = read_json(root/'reference-manifest.json')
    for name, record in manifest[plan['case']]['files'].items():
        path = root/name
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], f'Reference artifact differs: {name}')
    stem = 'reference-'+plan['case']
    receipt = read_json(root/(stem+'.json'))
    lineage(root, manifest, receipt, canonical, stem, plan)
    require(receipt['case'] == plan['case'] and receipt['checkpoint'] == canonical['checkpoint'] and receipt['input'] == canonical['input'], 'Reference identity differs')
    require(receipt['options'] == canonical['cases'][plan['case']] and receipt['tf32'] == '0' and 'gpu' in receipt['device'].lower(), 'Reference options/backend differ')
    require(receipt['source_files'] == canonical['source_files'] and receipt['source_revision'] == canonical['source_revision'], 'Reference upstream source differs')
    for key in ('version','native_sha256','binding_sha256'):
        require(receipt['runtime'][key] == canonical['runtime'][key], f'Reference runtime differs: {key}')
    for suffix in ('child.exit','wrapper.exit','guard.exit'):
        require((root/(stem+'.'+suffix)).read_text() == '0\n', 'Reference execution failed')
    out = Path(plan['run_dir'])/'output'
    from safetensors.numpy import load_file
    reference = load_file(str(root/(stem+'.safetensors')))
    require(set(receipt['outputs']) == set(reference), 'Reference output receipt inventory differs')
    for name, array in reference.items():
        require(receipt['outputs'][name] == {'shape':list(array.shape),'dtype':str(array.dtype),'finite':True}, 'Reference output receipt schema differs')
    metrics, failures = compare(reference, load_file(str(out/'result.safetensors')), out/'mesh.obj')
    report = {'case': plan['case'], 'plan_sha256': a.plan_sha256, 'comparator_sha256': sha(Path(__file__)),
              'reference_sha256': receipt['output_sha256'], 'capture': capture, 'metrics': metrics, 'failures': failures,
              'thresholds': {'vertices_max_abs_metres': 1e-3, 'keypoints_max_abs_metres': 1e-3, 'camera_relative_l2': 1e-3}}
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    require(not failures, 'Strict Body parity failed')


if __name__ == '__main__': main()
