"""Authenticated diagnostic capture of unchanged upstream conditioning and step one."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

from provenance import canonical, require, sha, verify_bundle, verify_input, verify_source_runtime
from upstream_loader import load_upstream


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'bundle', 'output', 'input-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--case', choices=['rgba-dropout'], required=True)
    p.add_argument('--device', choices=['gpu'], required=True)
    a = p.parse_args()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.set_default_device(mx.gpu)
    runtime = verify_source_runtime(a.source, mx)
    require(os.environ.get('MLX_ENABLE_TF32') == '0', 'Disable TF32')
    verify_bundle(a.bundle)
    here = Path(__file__).resolve().parent
    require(sha(here / 'input-provenance.json') == canonical()['input_provenance_sha256'], 'Input provenance changed')
    verify_input(a.input_dir / 'object-rgba.png', 'object-rgba.png')
    load_upstream(a.source)
    from mlx_vlm.models.sam3d_objects.generate import read_image
    from mlx_vlm.models.sam3d_objects.pipeline import Pipeline, LATENTS, prepare_inputs, sample
    from mlx_vlm.models.interpolate import resize_bilinear_nhwc
    require(str(mx.default_device()) == 'Device(gpu, 0)', 'Device changed')
    require(not a.output.exists(), 'Output already exists')
    a.output.mkdir(parents=True)
    pipeline = Pipeline.from_pretrained(a.bundle)
    model, flat = pipeline.model, {}
    mx.eval(model.parameters())
    parameters = dict(tree_flatten(model.parameters()))
    counts = collections.Counter(str(v.dtype) for v in parameters.values())
    require(counts == {'mlx.core.bfloat16': 3288, 'mlx.core.float32': 607}, 'Parameter count/dtype mismatch')
    require(all(str(v.dtype) == ('mlx.core.float32' if k.startswith('depth_model.') else 'mlx.core.bfloat16') for k, v in parameters.items()), 'Parameter dtype placement mismatch')
    require(model.shared_backbone is True, 'DINO sharing mismatch')
    del parameters
    config = model.config
    require(config.ss_steps == 25 and config.slat_steps == 25, 'Sampling defaults changed')

    def put(name, value):
        require(name not in flat, 'Duplicate stage')
        mx.eval(value)
        require(value.dtype in (mx.float32, mx.bfloat16), 'Unexpected stage dtype')
        require(bool(mx.all(mx.isfinite(value)).item()), 'Nonfinite stage ' + name)
        flat[name] = value

    inputs, metadata = prepare_inputs(read_image(a.input_dir / 'object-rgba.png'), None, None, size=config.image_size)
    require(metadata['pointmap_conditioned'] is False, 'Unexpected pointmap')
    for name, value in inputs.items():
        if value is not None:
            put('input.' + name, value)
    dino = model.ss_condition_embedder.module_list[0]
    pixels = resize_bilinear_nhwc(inputs['image'], (dino._size, dino._size))
    pixels = ((pixels - mx.array([0.485, 0.456, 0.406])) / mx.array([0.229, 0.224, 0.225])).astype(dino.backbone.cls_token.dtype)
    put('dino.image.normalized_pixels', pixels)
    tokens = dino.backbone.prepare_tokens(pixels)
    put('dino.image.prepared_tokens', tokens)
    block = dino.backbone.blocks[0]
    norm = block.norm1(tokens)
    put('dino.image.norm1', norm)
    put('dino.image.qkv_fused', block.attn.qkv(norm))
    put('dino.image.qkv_separate', norm @ block.attn.qkv.weight.T + block.attn.qkv.bias)
    put('dino.image.block0', block(tokens))
    features = {}
    condition = model.ss_condition_embedder(inputs, cache=features)
    put('structure.condition', condition)
    require(set(features) == {'image', 'rgb_image', 'mask', 'rgb_image_mask'}, 'Unexpected DINO cache')
    for name, value in features.items():
        module = model.ss_condition_embedder.module_list[0 if name in ('image', 'rgb_image') else 1]
        put('dino.' + name + '.prenorm', value)
        put('dino.' + name + '.norm', module.backbone.norm(value))
    conditions = pipeline._conditions(model.ss_generator, condition, config.ss_guidance)
    require(len(conditions) == 2, 'Expected classifier-free guidance')
    keys = mx.random.split(mx.random.key(42), len(LATENTS) + 1)
    state = {name: mx.random.normal((1, config.latent_resolution**3 if name == 'shape' else 1, channels), key=keys[i]) for i, (name, channels) in enumerate(LATENTS.items())}
    velocity, uncond = model.ss_generator.guided(state, mx.array([0.0], mx.float32), *conditions, mx.zeros(1))
    # Invoke the actual upstream sampler for the update; do not reconstruct its arithmetic.
    step, next_state = next(sample(model.ss_generator, state, config.ss_steps, config.ss_rescale_t, config.ss_guidance, *conditions))
    require(step == 1, 'Unexpected sampler step')
    for name, value in state.items():
        put('initial.' + name, value)
        put('velocity.conditioned.' + name, velocity[name])
        put('velocity.unconditioned.' + name, uncond[name])
        put('step1.' + name, next_state[name])
    out = a.output / 'stages.safetensors'
    mx.save_safetensors(str(out), flat)
    receipt = dict(purpose='diagnostic-only', case=a.case, seed=42, structure_steps=25, captured_step=1,
                   source_revision=canonical()['source_revision'], source_sha256=canonical()['source_sha256'],
                   checkpoint=canonical()['checkpoint'], runtime=runtime, device=str(mx.default_device()), tf32='0',
                   input_files={'object-rgba.png': canonical()['inputs']['object-rgba.png']},
                   consumed_input_roles={'image': 'object-rgba.png', 'mask': 'image alpha', 'pointmap': 'none'},
                   helpers_sha256={n: sha(here / n) for n in (Path(__file__).name, 'provenance.py', 'upstream_loader.py', 'provenance.json', 'input-provenance.json')},
                   output_sha256=sha(out), outputs={n: {'shape': list(v.shape), 'dtype': str(v.dtype)} for n, v in flat.items()},
                   dtype_counts=counts, shared_backbone=True)
    (a.output / 'stages.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'stage': 'saved', 'tensor_count': len(flat), 'output_sha256': receipt['output_sha256']}), flush=True)


if __name__ == '__main__':
    main()
