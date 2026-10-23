"""Regenerate tiny upstream LM states from the committed authenticated sampled input."""
from pathlib import Path
import argparse
import hashlib
import json

EXECUTED_HELPERS = {'provenance.py': '062ca141346c59988ead974a2426bdc5912b4b17a309536ad76b15f4ff9f2829', 'provenance.json': 'cfccfb008b154edc6a67ec45ed4233de929f7a0ab74f1aafc679d8019fc83641', 'upstream_loader.py': '27f4d0384fcdcce9a2461e30af84cd6e70cf4d4f7a133f35fc96647fab45956f'}
for _name, _expected in EXECUTED_HELPERS.items():
    if hashlib.sha256((Path(__file__).parent / _name).read_bytes()).hexdigest() != _expected:
        raise ValueError(f"Unapproved reproduction helper: {_name}")

import mlx.core as mx
from provenance import require, verify_source_runtime
from upstream_loader import load_upstream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    args = parser.parse_args()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    require(digest(args.inputs) == 'd1be5971cd37960e5b5eb31b2fee95169af8e711cb82d0fcd4b5d3c6ec65ba9d', 'Sampled inputs changed')
    require(not args.output.exists(), 'Refuse to replace oracle')
    runtime = verify_source_runtime(args.source, mx)
    load_upstream(args.source)
    from mlx_vlm.models.moge3.geometry import _lm_evaluate, _lm_step, _solve_focal_shift
    mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
    inputs = mx.load(str(args.inputs))
    require(set(inputs) == {'uv', 'xyz'}, 'Unexpected sampled inputs')
    uv, xyz = inputs['uv'], inputs['xyz']
    require(uv.shape == (1,4096,2) and xyz.shape == (1,4096,3) and uv.dtype == xyz.dtype == mx.float32, 'Invalid input schema')
    xy, z = xyz[..., :2], xyz[..., 2]
    shift = mx.zeros((1,1), mx.float32)
    state = (shift, mx.full((1,),1e-3,mx.float32), mx.zeros((1,),mx.bool_), *_lm_evaluate(shift,uv,xy,z,None))
    states = [state]
    for _ in range(30):
        state = _lm_step(state,uv,xy,z,None,mx.array(1e-3,mx.float32))
        states.append(state)
    shift, focal = _solve_focal_shift(uv,xyz)
    output = {'states': mx.stack([mx.stack([v.reshape(1).astype(mx.float32) for v in st],axis=-1) for st in states],axis=0), 'focal_shift':mx.stack([focal,shift],axis=-1)}
    mx.eval(*output.values())
    require(bool(mx.all(state[0].reshape(-1) == shift).item()) and bool(mx.all(state[-1] == focal).item()), 'Trace changed final solve')
    args.output.mkdir(parents=True)
    mx.save_safetensors(str(args.output/'states.safetensors'),output)
    receipt = {'purpose':'tiny exact upstream solver oracle; not model acceptance','device':args.device,'runtime':runtime,'executed_helpers':EXECUTED_HELPERS,'script_sha256':digest(Path(__file__)),'inputs_sha256':digest(args.inputs),'states_sha256':digest(args.output/'states.safetensors'),'state_fields':['shift','lambda','done','cost','gradient','hessian','focal'],'state_count':31}
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
