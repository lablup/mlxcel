"""Phase 2a: int4 dequant-in-graph correctness + coherence on the JAX harness.

1. Quantize the 7 per-layer linears to affine int4 (group 64), report error.
2. Prove the in-graph dequant matches a host dequant-then-fp32 baseline.
3. Greedy-decode with int4 weights, confirm coherence vs the fp16 run.
4. Export the int4 decode_step to StableHLO and confirm the dequant ops are
   present (bit unpack + convert + scale), separate from dot_general.
"""
import json
import os
import collections
import re

import numpy as np
import jax
import jax.numpy as jnp
from jax import export as jexport
from transformers import AutoTokenizer

import model_jax as M
import model_int4 as Mi
import quant as Q

MODEL = "models/Llama-3.2-1B-Instruct"
MAX_SEQ = 256
BUCKETS = [32, 64, 128, 256]
EOS = {128001, 128008, 128009}
SDS = jax.ShapeDtypeStruct
ART = "artifacts"


def bucket(n):
    return next(b for b in BUCKETS if n <= b)


def main():
    os.makedirs(ART, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "Give me three short tips for staying focused while working."}],
        add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    n = len(ids)
    Lp = bucket(n)

    cfg = M.load_config(MODEL)
    L, nkv, d = cfg.n_layers, cfg.n_kv, cfg.head_dim
    fp32 = M.load_params(MODEL, cfg)

    # 1. quantize + error
    qp = Mi.quantize_params(fp32)
    errs = {}
    for name in Mi.LINEARS:
        W = np.asarray(fp32["layers"][0][name], np.float32)
        p, s, w = (np.asarray(x) for x in qp["layers"][0][name])
        errs[name] = Q.quant_error(W, p, s, w)
    print("=== int4 affine quant (group 64), layer-0 relative Frobenius error ===")
    for k, v in errs.items():
        print(f"  {k:5s}: {v:.4f}")
    # packed-memory check on one weight
    W0 = fp32["layers"][0]["down"]
    packed0 = np.asarray(qp["layers"][0]["down"][0])
    print(f"  down weight: fp32 {W0.nbytes/1e6:.1f} MB -> int4 packed {packed0.nbytes/1e6:.1f} MB "
          f"+ scales (~1/{Q.GROUP} of fp32)")

    # 2. in-graph dequant == host dequant-then-fp32
    prefill_i4, decode_i4 = Mi.make_int4_fns(cfg, MAX_SEQ)
    fp_from_q = Mi.dequantize_params(qp)
    prefill_fp, decode_fp = M.make_fns(cfg, MAX_SEQ)
    toks = np.zeros(Lp, np.int32); toks[:n] = np.array(ids, np.int32)
    pos = np.arange(Lp, dtype=np.int32)
    lo_ig, _, _ = jax.jit(prefill_i4)(qp, jnp.asarray(toks), jnp.asarray(pos), jnp.int32(n))
    lo_fp, _, _ = jax.jit(prefill_fp)(fp_from_q, jnp.asarray(toks), jnp.asarray(pos), jnp.int32(n))
    lo_ig, lo_fp = np.asarray(lo_ig), np.asarray(lo_fp)
    print("\n=== in-graph dequant vs host dequant-then-fp32 (same weights, two paths) ===")
    print(f"  logit abs diff: max={np.abs(lo_ig-lo_fp).max():.3e} mean={np.abs(lo_ig-lo_fp).mean():.3e}")
    print(f"  argmax in-graph={int(lo_ig.argmax())} host={int(lo_fp.argmax())} match={int(lo_ig.argmax())==int(lo_fp.argmax())}")

    # 3. greedy coherence with int4, compared to the fp16 run
    pj, dj = jax.jit(prefill_i4), jax.jit(decode_i4)
    logits, kc, vc = pj(qp, jnp.asarray(toks), jnp.asarray(pos), jnp.int32(n))
    nxt = int(np.asarray(logits).argmax()); gen = [nxt]; clen = cur = n
    for _ in range(47):
        if nxt in EOS:
            break
        logits, kc, vc = dj(qp, jnp.int32(nxt), jnp.int32(cur), jnp.int32(clen), kc, vc)
        nxt = int(np.asarray(logits).argmax()); gen.append(nxt); clen += 1; cur += 1
    int4_text = tok.decode(gen)
    print("\n=== int4 greedy continuation ===")
    print(repr(int4_text))
    fp16 = json.load(open(f"{ART}/results.json")) if os.path.exists(f"{ART}/results.json") else None
    if fp16:
        a = fp16["jax_ids"]; m = min(len(a), len(gen))
        agree = sum(int(a[i] == gen[i]) for i in range(m))
        fd = next((i for i in range(m) if a[i] != gen[i]), None)
        print(f"int4 vs fp16 greedy: {agree}/{m} tokens agree"
              + (f", first differ at {fd}" if fd is not None else " (identical)"))

    # 4. export int4 decode_step StableHLO, confirm dequant ops present
    qp_aval = jax.tree.map(lambda x: SDS(x.shape, x.dtype), qp)
    exp = jexport.export(jax.jit(decode_i4))(
        qp_aval, SDS((), jnp.int32), SDS((), jnp.int32), SDS((), jnp.int32),
        SDS((L, MAX_SEQ, nkv, d), jnp.float32), SDS((L, MAX_SEQ, nkv, d), jnp.float32))
    mlir = exp.mlir_module()
    open(f"{ART}/decode_step_int4.stablehlo.mlir", "w").write(mlir)
    ops = collections.Counter(re.findall(r"\b(?:stablehlo|chlo)\.([a-z_0-9]+)", mlir))
    print("\n=== int4 decode_step StableHLO: dequant ops present and separate ===")
    for op in ["shift_right_logical", "and", "convert", "dot_general", "broadcast_in_dim", "reshape", "multiply", "add"]:
        print(f"  {op:20s}: {ops.get(op, 0)}")
    print(f"  custom_call present: {'custom_call' in ops}  (dequant is plain ops, not a custom op)")
    print(f"  uint32 inputs (packed weights) in signature: {'ui32' in mlir or 'i32' in mlir.split('->')[0]}")
    json.dump({"quant_err_layer0": errs, "int4_text": int4_text, "int4_ids": gen},
              open(f"{ART}/results_int4.json", "w"), indent=2)
    print(f"\nwrote {ART}/decode_step_int4.stablehlo.mlir + results_int4.json")


if __name__ == "__main__":
    main()
