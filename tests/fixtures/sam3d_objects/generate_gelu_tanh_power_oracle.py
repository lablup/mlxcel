"""Reproduce the bounded tanh GELU probe with the authenticated MLX runtime.

Requires sibling provenance.py, provenance.json and the original probe receipt.
Outputs <output-stem>.safetensors and <output-stem>.json; refuses overwrite.
"""
import argparse
import inspect
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Pinned upstream mlx-vlm checkout root')
    parser.add_argument('--output-stem', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    args = parser.parse_args()
    output = Path(str(args.output_stem) + '.safetensors')
    receipt = Path(str(args.output_stem) + '.json')
    if output.exists() or receipt.exists():
        raise RuntimeError('output or receipt already exists')

    # Resolve helpers beside this script; no machine-specific loader paths.
    from provenance import require, sha, verify_source_runtime
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    receipt_name = ('gelu-tanh-power-cpu-provenance.json' if args.device == 'cpu'
                    else 'gelu-tanh-power-provenance.json')
    original = json.loads(Path(__file__).with_name(receipt_name).read_text())
    runtime = verify_source_runtime(args.source, mx)
    source = Path(inspect.getfile(nn.gelu_approx)).resolve()
    require(sha(source) == original['activation_source_sha256'],
            'Activation source mismatch')
    require(runtime['native_sha256'] == original['native_sha256'],
            'Original native runtime mismatch')
    mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
    require(str(mx.default_device()) == original['device'], 'Requested device mismatch')

    @mx.compile
    def power(x):
        return 0.5*x*(1+mx.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3)))

    @mx.compile
    def multiply(x):
        return x*(0.5*(1+mx.tanh(math.sqrt(2/math.pi)*(x+0.044715*((x*x)*x)))))

    arrays, results = {}, {}
    inputs = [
        ('probe', mx.array([.90625, -.828125, -.90625, -2.125], mx.bfloat16)),
        ('bf16_grid', (mx.arange(-4096, 4096, dtype=mx.float32)/1024).astype(mx.bfloat16)),
        ('f32_grid', mx.arange(-4096, 4096, dtype=mx.float32)/1024),
    ]
    for name, x in inputs:
        y, p, m = nn.GELU(approx='tanh')(x), power(x), multiply(x)
        mx.eval(x, y, p, m)
        for key, value in [('x', x), ('upstream', y), ('power', p), ('multiply', m)]:
            arrays[name + '_' + key] = value
        yy, pp, mm = (np.array(a.astype(mx.float32)) for a in (y, p, m))
        require(bool(np.array_equal(yy, pp)), name + ': Power graph mismatch')
        results[name] = {
            'dtype': str(x.dtype), 'shape': list(x.shape), 'power_exact': True,
            'multiply_differing': int(np.count_nonzero(yy != mm)),
            'multiply_maxabs': float(np.max(np.abs(yy-mm))),
        }
        if name == 'probe':
            results[name].update(upstream=yy.tolist(), multiply=mm.tolist())
    require(results == original['results'], 'Original oracle statistics mismatch')
    mx.save_safetensors(str(output), arrays)
    require(sha(output) == original['output_sha256'], 'Original oracle bytes mismatch')
    result = {
        'results': results, 'script_sha256': sha(Path(__file__)), 'runtime': runtime,
        'activation_source': str(source), 'activation_source_sha256': sha(source),
        'output_sha256': sha(output), 'native_sha256': runtime['native_sha256'],
        'device': str(mx.default_device()),
        'original_capture_script_sha256': original['script_sha256'],
    }
    with receipt.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
