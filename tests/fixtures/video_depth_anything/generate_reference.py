#!/usr/bin/env python3
"""Regenerate complete VDA outputs using pinned, unmodified upstream model math."""
import argparse
import ast
import hashlib
import inspect
import json
import resource
import os
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from safetensors.numpy import save_file

PIN = 'abd3b2d7b728cb5372d4e092260da325e5ded209'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def import_upstream(source):
    # Namespace packages avoid importing unrelated multimodal architectures.
    for name, path in [('mlx_vlm', source), ('mlx_vlm.models', source/'models'),
                       ('mlx_vlm.models.dinov2', source/'models/dinov2'),
                       ('mlx_vlm.models.video_depth_anything', source/'models/video_depth_anything')]:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    # Execute exact upstream dependency definitions, not substitute implementations.
    base_path = source/'models/base.py'
    base = types.ModuleType('mlx_vlm.models.base')
    base.inspect, base.dataclass = inspect, dataclass
    tree = ast.parse(base_path.read_text())
    wanted = {'BaseModelConfig', 'check_array_shape', 'install_auto_processor_patch'}
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(base_path), 'exec'), base.__dict__)
            found.add(node.name)
    require(found == wanted, "Pinned upstream support definitions are incomplete")
    sys.modules[base.__name__] = base
    from mlx_vlm.models.video_depth_anything.config import ModelConfig
    from mlx_vlm.models.video_depth_anything.video_depth_anything import Model
    from mlx_vlm.models.video_depth_anything.processing_video_depth_anything import VideoDepthProcessor
    from mlx_vlm.models.video_depth_anything.generate import VideoDepthPredictor
    return ModelConfig, Model, VideoDepthProcessor, VideoDepthPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--kind', choices=['relative', 'metric'], required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--verify-only', action='store_true', help='Validate provenance/runtime without loading a model')
    parser.add_argument('--runtime-dir', default='/home/devops/Development/mlxcel/.cache/epic-2382/python-exact-core')
    parser.add_argument('--stage-timing', action='store_true', help='Evaluate unchanged backbone/head outputs to time native CPU stages')
    parser.add_argument('--ffmpeg', default='/tmp/mlxcel-ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg')
    args = parser.parse_args()
    # Validate all executable source/checkpoint/native bytes before importing MLX.
    manifest_dir = Path(__file__).resolve().parent
    expected = json.loads((manifest_dir/'source-manifest.json').read_text())
    for item in expected:
        path = args.cache/item['path']
        require(path.is_file() and digest(path) == item['sha256'], f'Unverified cached source/checkpoint: {path}')
    native = json.loads((manifest_dir/'native-runtime-provenance.json').read_text())
    runtime = Path(args.runtime_dir)
    for recorded, sha in native['runtime_files'].items():
        if '/python-exact-core/' not in recorded:
            continue
        suffix = recorded.split('/python-exact-core/', 1)[1]
        require(digest(runtime/suffix) == sha, f'Unverified native runtime: {suffix}')
    for suffix, sha in native['python_module_sha256'].items():
        require(digest(runtime/suffix) == sha, f'Unverified MLX Python module: {suffix}')
    contract = json.loads((manifest_dir/'reference-contract.json').read_text())
    require(digest(args.fixtures/'davis_64.mp4') == contract['clip_sha256'], 'Unverified input clip')
    require(os.environ.get('MLX_ENABLE_TF32', '0') == '0', 'Reference requires MLX_ENABLE_TF32=0')
    os.environ['MLX_ENABLE_TF32'] = '0'
    import mlx.core as mx
    require(Path(mx.__file__).resolve().is_relative_to(runtime.resolve()), mx.__file__)
    mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libmlx.so' in line}
    require(mapped == {str((runtime/'libmlx.so').resolve())}, mapped)
    require(mx.__version__ == contract['mlx_version'], 'Reference contract mismatch')
    require(cv2.__version__ == contract['opencv_version'], 'Reference contract mismatch')
    require(np.__version__ == contract['numpy_version'], 'Reference contract mismatch')
    mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
    require(mx.default_device() == (mx.cpu if args.device == 'cpu' else mx.gpu), 'Reference contract mismatch')
    if args.verify_only:
        print('PASS provenance and exact native runtime; no model loaded', flush=True)
        return
    source = args.cache/'upstream/mlx_vlm'
    ModelConfig, Model, Processor, Predictor = import_upstream(source)
    checkpoint = args.cache/'checkpoints'/args.kind
    cfg = ModelConfig.from_dict(json.loads((checkpoint/'config.json').read_text()))
    model = Model(cfg)
    weights = Model.sanitize(mx.load(str(checkpoint/'model.safetensors')))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    from mlx.utils import tree_flatten
    loaded = {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name,value in tree_flatten(model.parameters())}
    require(loaded and {v['dtype'] for v in loaded.values()} == {'mlx.core.float32'}, 'Actual loaded parameters must all be float32')
    print('LOADED', args.kind, args.device, len(loaded), 'all float32', flush=True)
    print('BINDING', mx.__file__, 'NATIVE', sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if 'libmlx.so' in l}), flush=True)
    clip = args.fixtures/'davis_64.mp4'
    raw = subprocess.check_output([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(clip), '-frames:v', '40', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(40,96,128,3).copy()
    require(hashlib.sha256(frames.tobytes()).hexdigest() == contract['decoded_rgb_sha256'], 'Decoder changed reference pixels')
    if args.device == 'cpu' and args.kind == 'relative':
        save_file({'frames_nhwc_u8': frames}, str(args.fixtures/'davis_40_frames.safetensors'))
    processor = Processor.from_pretrained(checkpoint)
    processed = processor.preprocess_frame(frames[0])
    require(processed.shape == (518,686,3) and processed.dtype == np.float32, 'Unexpected processor output shape/dtype')
    require(hashlib.sha256(processed.tobytes()).hexdigest() == contract['processor_sha256'], 'Reference contract mismatch')
    print('PROCESSOR', processed.shape, processed.dtype, 'sha256', hashlib.sha256(processed.tobytes()).hexdigest(), flush=True)
    class TimedModel:
        config = model.config
        calls = 0
        def __call__(self, x):
            self.calls += 1
            print('WINDOW_START', self.calls, x.shape, x.dtype, flush=True)
            begin = time.monotonic()
            depth = model(x)
            mx.eval(depth)
            print('WINDOW_DONE', self.calls, round(time.monotonic()-begin,3), 'seconds', 'peak_rss_kib',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
            return depth
    if args.stage_timing:
        # Diagnostics only: call the original methods and evaluate their unchanged
        # returned arrays. No replacement operators, weights, or model code.
        original_backbone = model.pretrained.get_intermediate_layers
        def timed_backbone(*positional, **keywords):
            print('STAGE_START backbone', flush=True)
            begin = time.monotonic()
            result = original_backbone(*positional, **keywords)
            mx.eval(result)
            print('STAGE_DONE backbone', time.monotonic()-begin, flush=True)
            return result
        object.__setattr__(model.pretrained, 'get_intermediate_layers', timed_backbone)
        head_type = type(model.head)
        original_head = head_type.__call__
        def timed_head(self, *positional, **keywords):
            print('STAGE_START temporal_head', flush=True)
            begin = time.monotonic()
            result = original_head(self, *positional, **keywords)
            mx.eval(result)
            print('STAGE_DONE temporal_head', time.monotonic()-begin, flush=True)
            return result
        head_type.__call__ = timed_head
    timed = TimedModel()
    start = time.monotonic()
    depth = Predictor(timed, processor).infer(frames, progress=False)
    require(timed.calls == 2 and depth.shape == (40,96,128) and depth.dtype == np.float32, 'Unexpected window count or output shape/dtype')
    require(np.isfinite(depth).all(), 'Nonfinite reference depth')
    directory = args.fixtures if args.device == 'cpu' else args.cache/'gpu-references'
    directory.mkdir(parents=True, exist_ok=True)
    result = directory/f'small_{args.kind}_40.safetensors'
    save_file({'depth': np.ascontiguousarray(depth)}, str(result), metadata={'upstream_revision': PIN, 'device': args.device, 'checkpoint_dtype': 'float32'})
    metadata = {'upstream_revision':PIN,'checkpoint_config':cfg.to_dict(),'checkpoint_sha256':digest(checkpoint/'model.safetensors'), 'clip_sha256':digest(clip),'decoded_rgb_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),'loaded_parameters':loaded,'processor_shape':list(processed.shape),'processor_sha256':hashlib.sha256(processed.tobytes()).hexdigest(),'output_shape':list(depth.shape),'output_dtype':str(depth.dtype),'output_sha256':digest(result),'device':args.device,'stage_timing':args.stage_timing,'mlx_enable_tf32':os.environ['MLX_ENABLE_TF32'],'source_manifest_sha256':digest(manifest_dir/'source-manifest.json'),'runtime_provenance_sha256':digest(manifest_dir/'native-runtime-provenance.json'),'mlx_version':mx.__version__,'opencv_version':cv2.__version__,'numpy_version':np.__version__,'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (directory/f'small_{args.kind}_40.provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('PASS', result, metadata['seconds'], 'seconds', float(depth.min()),float(depth.max()),flush=True)

if __name__ == '__main__':
    main()
