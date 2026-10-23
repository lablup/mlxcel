# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture the unchanged public SAM3DPredictor on the authenticated checkpoint."""
from pathlib import Path
import argparse
import json
import os
import time
from provenance import (authenticate_helpers, load_upstream, pin, require, sha,
                        verify_bundle, verify_image, verify_source_runtime, verify_dependencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('boxed', 'default'), default='boxed')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--reviewed-manifest', type=Path, required=True)
    parser.add_argument('--reviewed-manifest-sha256', required=True)
    parser.add_argument('--imports-only', action='store_true')
    args = parser.parse_args()
    helpers = authenticate_helpers(args.reviewed_manifest, args.reviewed_manifest_sha256)
    dependency_identity = verify_dependencies()
    import mlx.core as mx
    import numpy as np
    import ml_dtypes  # Register NumPy's BF16 dtype before the unchanged loader uses it.
    from safetensors.numpy import save_file
    mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
    runtime = verify_source_runtime(args.source, mx)
    require(args.device in str(mx.default_device()).lower(), 'Resolved device differs')
    module = load_upstream(args.source)
    from pinned_sam3d_body.generate import SAM3DPredictor
    config = module.SAM3DConfig.load(args.bundle / 'config.json')
    require(list(config.image_size) == pin()['released_image_size'], 'Released image size differs')
    require(list(module.SAM3DConfig().image_size) == pin()['dataclass_default_image_size'], 'Dataclass defaults differ')
    require(not args.output.exists(), 'Refuse to overwrite reference')
    if args.imports_only:
        require(sha(args.bundle / 'config.json') == pin()['checkpoint']['files']['config.json']['sha256'], 'Config differs')
        print(json.dumps({'imports_only': True, 'runtime': runtime, 'image_size': config.image_size, 'helpers': helpers}))
        return
    header = verify_bundle(args.bundle)
    image = verify_image(args.image)
    options = pin()['cases'][args.case]
    started = time.monotonic()
    # Released sanitized checkpoint retains four buffers absent from the pinned head.
    # Its raw-key loader explicitly skips pmi/texture data; MHR forward explicitly
    # does not apply parameter limits. Filter only these proven-unused load entries.
    from pinned_sam3d_body.mhr_head import MHRHead
    unused = {'body_model.parameter_limits', 'body_model.pmi',
              'body_model.texcoord_faces', 'body_model.texcoords'}
    original_load = MHRHead.load_weights
    require('load_weights' not in MHRHead.__dict__, 'Unexpected head override')
    def compatible_load(self, weights, *positional, **keywords):
        require(isinstance(weights, list), 'Unexpected load interface')
        require({name for name, _ in weights if name in unused} == unused, 'Unused buffer inventory differs')
        return original_load(self, [(name, value) for name, value in weights if name not in unused], *positional, **keywords)
    MHRHead.load_weights = compatible_load
    try:
        predictor = SAM3DPredictor.from_pretrained(str(args.bundle))
    finally:
        del MHRHead.load_weights
    loader_compatibility = {'reason': 'Canonical public load fails strict head load on four unused released buffers',
                            'filtered_head_keys': sorted(unused), 'forward_and_predict_unchanged': True}

    from mlx.utils import tree_flatten
    params = dict(tree_flatten(predictor.model.parameters()))
    loaded = {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in params.items()}
    # Exact retained key inventory follows the pinned public body-only loader.
    excluded = ('decoder_hand.', 'head_pose_hand.', 'head_camera_hand.', 'init_pose_hand.',
                'init_camera_hand.', 'init_to_token_mhr_hand.', 'prev_to_token_mhr_hand.',
                'keypoint_embedding_hand.', 'keypoint3d_embedding_hand.', 'keypoint_posemb_linear_hand.',
                'keypoint3d_posemb_linear_hand.', 'keypoint_feat_linear_hand.', 'ray_cond_emb_hand.',
                'prompt_encoder.mask_downscaling.')
    unused_bare_hand = {'init_camera_hand', 'init_pose_hand', 'keypoint3d_embedding_hand', 'keypoint_embedding_hand'}
    expected = {k: k for k in header if not k.startswith(excluded) and k not in unused_bare_hand and
                k not in {'head_pose.'+name for name in unused} and
                not (k.startswith('backbone.') and ('bias_mask' in k or 'k_proj.bias' in k))}
    require(not any(k.startswith('mhr.') for k in header), 'Released schema unexpectedly needs MHR remapping')
    require(set(params) == set(expected), f'Loaded key mismatch missing={sorted(set(expected)-set(params))} extra={sorted(set(params)-set(expected))}')
    dtype_names = {'BF16': 'mlx.core.bfloat16', 'F32': 'mlx.core.float32', 'I32': 'mlx.core.int32', 'U8': 'mlx.core.uint8'}
    import hashlib, mmap, struct
    with (args.bundle / 'model.safetensors').open('rb') as weight_file:
        header_size = struct.unpack('<Q', weight_file.read(8))[0]
        with mmap.mmap(weight_file.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            for name, value in params.items():
                record = header[expected[name]]
                require(list(value.shape) == record['shape'] and str(value.dtype) == dtype_names[record['dtype']], f'Loaded tensor schema differs: {name}')
                flat = value.reshape(-1); itemsize = {'BF16':2,'F32':4,'I32':4,'U8':1}[record['dtype']]
                offset = 8 + header_size + record['data_offsets'][0]
                digest = hashlib.sha256()
                for start in range(0, flat.size, 1048576):
                    chunk = flat[start:start+1048576]
                    actual = np.asarray(chunk.view(mx.uint16) if record['dtype'] == 'BF16' else chunk).tobytes()
                    require(actual == raw[offset+start*itemsize:offset+start*itemsize+len(actual)], f'Loaded value differs: {name} at {start}')
                    digest.update(actual)
                loaded[name]['sha256'] = digest.hexdigest()
    result = predictor.predict(image, bbox=options['bbox'], cam_int=None)
    require(set(result) == {'pred_vertices', 'pred_keypoints_3d', 'pred_joint_coords', 'pred_camera', 'bbox'}, 'Public output schema differs')
    shapes = {'pred_vertices': (18439, 3), 'pred_keypoints_3d': (70, 3), 'pred_joint_coords': (127, 3), 'pred_camera': (3,)}
    tensors = {}
    for name, shape in shapes.items():
        value = result[name]
        require(value.shape == shape and value.dtype == np.float32 and bool(np.isfinite(value).all()), f'Invalid output: {name}')
        tensors[name] = value
    for name in ('pred_vertices', 'pred_keypoints_3d', 'pred_joint_coords'):
        require(float(np.max(np.ptp(tensors[name], axis=0))) > 0.0, f'Collapsed output: {name}')
    tensors['bbox'] = np.asarray(result['bbox'], dtype=np.float32)
    expected_box = options['bbox'] if options['bbox'] is not None else [0, 0, image.shape[1], image.shape[0]]
    require(np.array_equal(tensors['bbox'], np.asarray(expected_box, np.float32)), 'Returned bbox differs')
    faces = np.asarray(predictor.model.head_pose.faces)
    require(faces.shape == (36874, 3) and faces.dtype == np.int32 and faces.min() >= 0 and faces.max() < 18439, 'Invalid mesh faces')
    tensors['faces'] = faces
    args.output.mkdir(parents=True)
    save_file(tensors, str(args.output / 'reference.safetensors'))
    receipt = {'schema_version': 1, 'api': pin()['api'], 'case': args.case, 'options': options,
               'actual_bbox': expected_box, 'actual_cam_int': pin()['input']['default_cam_int'],
               'image_size': list(config.image_size), 'device': str(mx.default_device()), 'tf32': os.environ['MLX_ENABLE_TF32'],
               'runtime': runtime, 'dependency_identity': dependency_identity, 'checkpoint': pin()['checkpoint'], 'source_revision': pin()['source_revision'],
               'source_files': pin()['source_files'], 'input': pin()['input'], 'helpers': helpers,
               'loader_compatibility': loader_compatibility, 'loaded_parameters': loaded, 'output_sha256': sha(args.output / 'reference.safetensors'),
               'outputs': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'finite': bool(np.isfinite(v).all())} for k, v in tensors.items()},
               'elapsed_seconds_diagnostic_only': time.monotonic() - started}
    (args.output / 'reference.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'output_sha256': receipt['output_sha256'], 'receipt_sha256': sha(args.output / 'reference.json'), 'outputs': receipt['outputs']}))


if __name__ == '__main__':
    main()
