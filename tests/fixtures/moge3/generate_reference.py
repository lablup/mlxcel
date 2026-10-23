"""Run unchanged pinned upstream MoGe3 with an isolated package loader."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from upstream_loader import load_upstream
from provenance import canonical, options, require, sha, validate_outputs, verify_inputs, verify_source_runtime
p = argparse.ArgumentParser()
p.add_argument('--source', type=pathlib.Path, required=True)
p.add_argument('--model', type=pathlib.Path, required=True)
p.add_argument('--image', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
p.add_argument('--opts', default='{}')
p.add_argument('--imports-only', action='store_true')
a = p.parse_args()
import mlx.core as mx
import numpy as np
from PIL import Image
mx.set_default_device(mx.cpu if a.device == 'cpu' else mx.gpu)
runtime_receipt = verify_source_runtime(a.source, mx)
checkpoint_size = verify_inputs(a.model, a.image)
opts = options(json.loads(a.opts))
require(os.environ.get("MLX_ENABLE_TF32") == "0", "Set MLX_ENABLE_TF32=0 before process start")
load_upstream(a.source)
from mlx_vlm.models.moge3.config import ModelConfig
from mlx_vlm.models.moge3.moge3 import Model
from mlx.utils import tree_flatten
print('runtime', mx.__version__, mx.__file__, 'device', mx.default_device(), flush=True)
if a.imports_only:
    sys.exit(0)
start = time.monotonic()
config = ModelConfig.from_dict(json.loads((a.model / 'config.json').read_text()))
model = Model(config)
weights = mx.load(str(a.model / 'model.safetensors'))
model.load_weights(list(model.sanitize(weights).items()), strict=True)
model.eval()
mx.eval(model.parameters())
del weights
image = np.asarray(Image.open(a.image).convert('RGB'), dtype=np.float32) / np.float32(255.0)
pixels = mx.array(image)
params = dict(tree_flatten(model.parameters()))
dtypes = {k: str(v.dtype) for k, v in params.items()}
require(set(dtypes.values()) == {'mlx.core.float32'}, 'Loaded checkpoint did not preserve F32')
require(list(image.shape) == canonical()['image_shape'], 'Image dimensions mismatch')
print('loaded', len(params), 'dtypes', sorted(set(dtypes.values())), flush=True)
result = model.infer(pixels, **opts)
flat = {}
for key, value in result.items():
    if isinstance(value, list):
        for i, item in enumerate(value):
            flat[f"{key.removesuffix('_per_step')}_step{i}"] = item
    else:
        flat[key] = value
mx.eval(*flat.values())
validate_outputs(flat, opts, list(image.shape), reference=True)
a.output.parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(str(a.output), flat)

meta = {'elapsed_seconds': time.monotonic() - start, 'options': opts, 'tf32': os.environ.get('MLX_ENABLE_TF32'), 'runtime': mx.__version__, 'module': mx.__file__, 'device': str(mx.default_device()), 'source_revision': 'abd3b2d7b728cb5372d4e092260da325e5ded209', 'image_sha256': sha(a.image), 'checkpoint_sha256': sha(a.model / 'model.safetensors'), 'output_sha256': sha(a.output), 'loaded_parameter_dtypes': dtypes, 'input_dtype': str(pixels.dtype), 'outputs': {}, 'mapped_native_libraries': [line.strip() for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'libmlx' in line]}
source_files = list((a.source / "mlx_vlm/models/moge3").glob("*.py"))
source_files += list((a.source / "mlx_vlm/models/dinov2").glob("*.py"))
source_files += [a.source / "mlx_vlm/models/interpolate.py", a.source / "mlx_vlm/models/base.py"]
meta["source_sha256"] = {str(path.relative_to(a.source)): sha(path) for path in sorted(source_files)}
meta["generator_sha256"] = sha(pathlib.Path(__file__))
meta["runtime_receipt"] = runtime_receipt
meta["checkpoint_size"] = checkpoint_size
meta["checkpoint_files"] = {name: data["sha256"] for name, data in canonical()["checkpoints"][checkpoint_size]["files"].items()}
meta["image_shape"] = list(image.shape)
meta["provenance_sha256"] = sha(pathlib.Path(__file__).with_name("provenance.json"))
for key, v in flat.items():
    arr = np.asarray(v)
    meta['outputs'][key] = {'shape': list(v.shape), 'dtype': str(v.dtype), 'finite': int(np.isfinite(arr).sum()), 'elements': arr.size}
a.output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({k: v for k, v in meta.items() if k not in ('loaded_parameter_dtypes', 'mapped_native_libraries')}, indent=2), flush=True)
