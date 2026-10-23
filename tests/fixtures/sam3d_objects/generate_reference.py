"""Capture unchanged pinned Objects Pipeline with full default sampling."""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

from provenance import canonical, require, sha, validate_result, verify_bundle, verify_input, verify_source_runtime
from upstream_loader import load_upstream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--case', choices=['default', 'rgba-dropout'], default='default')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--imports-only', action='store_true')
    args = parser.parse_args()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.set_default_device(mx.gpu if args.device == 'gpu' else mx.cpu)
    runtime = verify_source_runtime(args.source, mx)
    require(os.environ.get('MLX_ENABLE_TF32') == '0', 'TF32 must be disabled')
    verify_bundle(args.bundle)
    image_name = 'object.png' if args.case == 'default' else 'object-rgba.png'
    fixture = Path(__file__).resolve().parent
    require(sha(fixture / 'input-provenance.json') == canonical()['input_provenance_sha256'], 'Input provenance changed')
    consumed_names = [image_name, 'mask.png'] if args.case == 'default' else [image_name]
    for name in consumed_names:
        verify_input(args.input_dir / name, name)
    load_upstream(args.source)
    from mlx_vlm.models.sam3d_objects.generate import read_image, read_mask, write_obj, write_gaussians
    from mlx_vlm.models.sam3d_objects.pipeline import Pipeline, Request
    from mlx_vlm.models.sam3d_objects.config import ModelConfig
    parsed = ModelConfig.from_dict(json.loads((args.bundle / "config.json").read_text()))
    loaded = ModelConfig.load(args.bundle / "config.json")
    require(parsed.depth_model is not None and loaded.depth_model is not None, "Published depth configuration is missing")
    require(parsed.ss_steps == loaded.ss_steps == 25 and parsed.slat_steps == loaded.slat_steps == 25, "Published sampling defaults changed")
    require(str(mx.default_device()) == ("Device(gpu, 0)" if args.device == "gpu" else "Device(cpu, 0)"), "Import changed the requested runtime device")
    print(json.dumps({'stage': 'authenticated', 'runtime': runtime, 'device': str(mx.default_device())}), flush=True)
    if args.imports_only:
        return
    require(not args.output.exists(), 'Refuse to overwrite a previous reference')
    args.output.mkdir(parents=True)
    start = time.monotonic()
    pipeline = Pipeline.from_pretrained(args.bundle)
    mx.eval(pipeline.model.parameters())
    params = dict(tree_flatten(pipeline.model.parameters()))
    counts = collections.Counter(str(v.dtype) for v in params.values())
    require(all(str(v.dtype) == ('mlx.core.float32' if k.startswith('depth_model.') else 'mlx.core.bfloat16') for k, v in params.items()), 'Loaded dtype policy mismatch')
    require(pipeline.model.shared_backbone is True, 'Released duplicate DINO values did not match')
    del params
    options = dict(canonical()['default_options'])
    options['estimate_depth'] = args.case == 'default'
    image = read_image(args.input_dir / image_name)
    mask = read_mask(args.input_dir / 'mask.png') if args.case == 'default' else None
    request = Request(image, mask, seed=42, ss_steps=25, slat_steps=25, formats=tuple(options['formats']), estimate_depth=options['estimate_depth'])
    print(json.dumps({'stage': 'loaded', 'dtype_counts': counts, 'shared_backbone': True, 'options': options}), flush=True)
    result = None
    for event in pipeline.stream(request):
        mx.eval(event.data)
        stage = {'stage': event.stage, 'step': event.step, 'total_steps': event.total_steps, 'elapsed_seconds': time.monotonic() - start, 'active_bytes': mx.get_active_memory(), 'peak_bytes': mx.get_peak_memory(), 'cache_bytes': mx.get_cache_memory()}
        if 'coords' in event.data:
            stage['coords'] = event.data['coords'].shape[0]
        print(json.dumps(stage), flush=True)
        if event.stage == 'depth':
            mx.save_safetensors(str(args.output / 'pointmap.safetensors'), {'points': event.data['pointmap']})
        if event.stage == 'complete':
            result = event.data
    require(result is not None, 'Pipeline did not emit complete output')
    flat = {k: v for k, v in tree_flatten(result) if isinstance(v, mx.array)}
    conditioned = result['pointmap_conditioned']
    require(conditioned is (args.case == 'default'), 'Conditioning route mismatch')
    stats = validate_result(flat, options['formats'], conditioned)
    path = args.output / 'reference.safetensors'
    mx.save_safetensors(str(path), flat)
    write_obj(args.output / 'mesh.obj', result['mesh'])
    write_gaussians(args.output / 'gaussian.ply', result['gaussian'])
    receipt = {'source_revision': canonical()['source_revision'], 'source_sha256': canonical()['source_sha256'], 'checkpoint': canonical()['checkpoint'], 'runtime': runtime, 'device': str(mx.default_device()), 'tf32': '0', 'case': args.case, 'options': options, 'input_files': {name: canonical()['inputs'][name] for name in consumed_names}, 'pointmap_conditioned': conditioned, 'outputs': stats, 'output_sha256': sha(path), 'generator_sha256': sha(Path(__file__)), 'provenance_sha256': sha(fixture / 'provenance.json'), 'dtype_counts': counts, 'shared_backbone': True, 'elapsed_seconds': time.monotonic() - start}
    receipt['consumed_input_roles'] = {'image': image_name, 'mask': 'mask.png' if args.case == 'default' else 'image alpha', 'pointmap': 'automatic depth' if args.case == 'default' else 'none'}
    receipt['executed_helpers_sha256'] = {name: sha(fixture / name) for name in ('generate_reference.py', 'provenance.py', 'upstream_loader.py')}
    receipt['input_provenance_sha256'] = sha(fixture / 'input-provenance.json')
    artifact_names = ['reference.safetensors', 'mesh.obj', 'gaussian.ply'] + (['pointmap.safetensors'] if args.case == 'default' else [])
    receipt['artifact_sha256'] = {name: sha(args.output / name) for name in artifact_names}
    (args.output / 'reference.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'stage': 'saved', 'output_sha256': receipt['output_sha256'], 'elapsed_seconds': receipt['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
