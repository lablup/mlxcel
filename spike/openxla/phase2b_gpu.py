"""Phase 2b: real GPU decode throughput on the GB10.

Fixes the Phase 2a harness artifacts that made the GPU look slow:
  - weights uploaded to the device ONCE (resident), not re-passed every step,
  - on-device argmax (decode returns the next token id, so only 4 bytes cross
    per step instead of a 513 KB logits copy),
  - KV cache kept resident on the device across steps.
Measures steady-state decode tok/s for int4 and fp32, on CUDA (GB10) and on CPU
through the same IREE harness, prefilling on CPU to seed the cache.

Run: JAX_PLATFORMS=cpu .venv/bin/python phase2b_gpu.py
"""
import json
import os
import subprocess
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import export as jexport
from transformers import AutoTokenizer

import model_jax as M
import model_int4 as Mi

MODEL = "models/Llama-3.2-1B-Instruct"
MAX_SEQ = 256
BUCKETS = [32, 64, 128, 256]
EOS = {128001, 128008, 128009}
SDS = jax.ShapeDtypeStruct
ART = "artifacts"
IREE = ".venv/bin/iree-compile"
STEPS = 40


def bucket(n):
    return next(b for b in BUCKETS if n <= b)


def compile_iree(mlir, vmfb, driver):
    backend = "--iree-hal-target-device=cuda" if driver == "cuda" else "--iree-hal-target-backends=llvm-cpu"
    subprocess.run([IREE, "--iree-input-type=stablehlo", backend, mlir, "-o", vmfb],
                   check=True, capture_output=True, text=True)


def load_iree(vmfb, driver):
    import iree.runtime as rt
    config = rt.Config(driver)
    ctx = rt.SystemContext(config=config)
    vm = rt.VmModule.from_flatbuffer(ctx.instance, open(vmfb, "rb").read())
    ctx.add_vm_module(vm)
    fn = getattr(ctx.modules, vm.name).main
    return fn, config.device, rt


def decode_argmax_fn(decode_step):
    def da(params, token, pos, clen, kc, vc):
        logits, kc, vc = decode_step(params, token, pos, clen, kc, vc)
        return jnp.argmax(logits).astype(jnp.int32), kc, vc
    return da


def run_variant(name, params, prefill, decode_step, cfg, toks, n, Lp, driver):
    """Compile decode (on-device argmax) for `driver`, run with resident weights."""
    L, nkv, d = cfg.n_layers, cfg.n_kv, cfg.head_dim
    da = decode_argmax_fn(decode_step)
    p_aval = jax.tree.map(lambda x: SDS(x.shape, x.dtype), params)
    exp = jexport.export(jax.jit(da))(
        p_aval, SDS((), jnp.int32), SDS((), jnp.int32), SDS((), jnp.int32),
        SDS((L, MAX_SEQ, nkv, d), jnp.float32), SDS((L, MAX_SEQ, nkv, d), jnp.float32))
    mlir = f"{ART}/{name}_decode_argmax.stablehlo.mlir"
    vmfb = f"{ART}/{name}_decode_argmax.{driver}.vmfb"
    open(mlir, "w").write(exp.mlir_module())
    compile_iree(mlir, vmfb, driver)

    # seed the cache with a CPU prefill (one-time), get the first token
    logits, kc, vc = jax.jit(prefill)(params, jnp.asarray(toks), jnp.arange(Lp, dtype=jnp.int32), jnp.int32(n))
    first = int(np.asarray(logits).argmax())
    kc, vc = np.asarray(kc), np.asarray(vc)

    fn, dev, rt = load_iree(vmfb, driver)
    leaves = [rt.asdevicearray(dev, np.asarray(x)) for x in jax.tree_util.tree_flatten(params)[0]]
    kc_d, vc_d = rt.asdevicearray(dev, kc), rt.asdevicearray(dev, vc)

    tok, clen, pos, gen, times = first, n, n, [first], []
    for step in range(STEPS + 3):  # 3 warmup
        t0 = time.time()
        out = fn(*leaves, np.int32(tok), np.int32(pos), np.int32(clen), kc_d, vc_d)
        tok = int(np.asarray(out[0].to_host()))   # 4-byte token, the only host copy
        kc_d, vc_d = out[1], out[2]                # cache stays resident
        if step >= 3:
            times.append(time.time() - t0)
        gen.append(tok); clen += 1; pos += 1
        if tok in EOS:
            break
    tps = 1.0 / np.mean(times) if times else float("nan")
    return tps, np.mean(times) * 1e3, gen


def main():
    os.makedirs(ART, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "Give me three short tips for staying focused while working."}],
        add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    n = len(ids); Lp = bucket(n)
    toks = np.zeros(Lp, np.int32); toks[:n] = np.array(ids, np.int32)

    cfg = M.load_config(MODEL)
    fp32 = M.load_params(MODEL, cfg)
    qp = Mi.quantize_params(fp32)
    prefill_fp, decode_fp = M.make_fns(cfg, MAX_SEQ)
    prefill_i4, decode_i4 = Mi.make_int4_fns(cfg, MAX_SEQ)

    variants = [
        ("int4", qp, prefill_i4, decode_i4),
        ("fp32", fp32, prefill_fp, decode_fp),
    ]
    rows = []
    for name, params, pf, ds in variants:
        for driver in ("cuda", "local-task"):
            tps, ms, gen = run_variant(name, params, pf, ds, cfg, toks, n, Lp, driver)
            where = "GB10" if driver == "cuda" else "CPU"
            print(f"{name:5s} {where:4s}: {tps:6.1f} tok/s ({ms:6.1f} ms/tok)  "
                  f"sample: {tok.decode(gen[:8])!r}")
            rows.append({"variant": name, "device": where, "tok_s": round(tps, 1), "ms_tok": round(ms, 1)})
    json.dump(rows, open(f"{ART}/phase2b_perf.json", "w"), indent=2)
    print(f"\nresident weights + on-device argmax. wrote {ART}/phase2b_perf.json")


if __name__ == "__main__":
    main()
