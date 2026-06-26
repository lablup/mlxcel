"""Characterize the exported StableHLO: on-device argmax sampling (the
recommended design) vs returning full logits, and the op inventory."""
import collections
import re
import jax
import jax.numpy as jnp
from jax import export as jexport

import model_jax as M

MODEL = "models/Llama-3.2-1B-Instruct"
MAX_SEQ = 256
SDS = jax.ShapeDtypeStruct


def main():
    cfg = M.load_config(MODEL)
    L, nkv, d = cfg.n_layers, cfg.n_kv, cfg.head_dim
    params = M.load_params(MODEL, cfg)
    _, decode_step = M.make_fns(cfg, MAX_SEQ)
    params_aval = jax.tree.map(lambda a: SDS(a.shape, a.dtype), params)

    def decode_argmax(params, token, pos, clen, kc, vc):
        logits, kc, vc = decode_step(params, token, pos, clen, kc, vc)
        return jnp.argmax(logits).astype(jnp.int32), kc, vc

    avals = (params_aval, SDS((), jnp.int32), SDS((), jnp.int32), SDS((), jnp.int32),
             SDS((L, MAX_SEQ, nkv, d), jnp.float32), SDS((L, MAX_SEQ, nkv, d), jnp.float32))
    exp_logits = jexport.export(jax.jit(decode_step))(*avals)
    exp_argmax = jexport.export(jax.jit(decode_argmax))(*avals)

    def out_bytes(exp):
        a = exp.out_avals[0]
        import numpy as np
        return int(np.prod(a.shape)) * a.dtype.itemsize, a.shape, a.dtype

    lb, ls, ld = out_bytes(exp_logits)
    ab, ash, ad = out_bytes(exp_argmax)
    print("=== on-device sampling: per-token device->host copy ===")
    print(f"  return logits : out[0] = {ls} {ld}  -> {lb:>7d} bytes/token")
    print(f"  return argmax : out[0] = {ash} {ad}  -> {ab:>7d} bytes/token")
    print(f"  reduction     : {lb/ab:.0f}x smaller host transfer with on-device argmax")

    # op inventory of the logits-returning decode graph
    mlir = exp_logits.mlir_module()
    ops = collections.Counter(re.findall(r"\b(?:stablehlo|chlo)\.([a-z_0-9]+)", mlir))
    print("\n=== StableHLO op inventory (decode_step) ===")
    for name, cnt in ops.most_common(14):
        print(f"  {cnt:5d}  {name}")
    print(f"  custom_call present: {'custom_call' in ops}")
    print(f"  distinct op kinds  : {len(ops)}")


if __name__ == "__main__":
    main()
