"""Phase 2a (GPU): compile the int4 dequant-in-graph graphs to CUDA (sm_121) with
IREE and run greedy decode on the GB10. Verifies the GPU path is numerically
correct (token-match to the CPU int4 run) and times decode. Acceleration test
enabled by CUDA being available on this box.
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
SM = "sm_121"


def bucket(n):
    return next(b for b in BUCKETS if n <= b)


def compile_cuda(mlir, vmfb):
    # Default cuda device target; PTX is JIT'd to the GB10 (sm_121) by the driver.
    # Explicit --iree-cuda-target=sm_121 is rejected by this IREE build's device syntax.
    cmd = [IREE, "--iree-input-type=stablehlo", "--iree-hal-target-device=cuda", mlir, "-o", vmfb]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


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
    qp = Mi.quantize_params(fp32)
    prefill_i4, decode_i4 = Mi.make_int4_fns(cfg, MAX_SEQ)

    qp_aval = jax.tree.map(lambda x: SDS(x.shape, x.dtype), qp)
    exp_p = jexport.export(jax.jit(prefill_i4))(
        qp_aval, SDS((Lp,), jnp.int32), SDS((Lp,), jnp.int32), SDS((), jnp.int32))
    exp_d = jexport.export(jax.jit(decode_i4))(
        qp_aval, SDS((), jnp.int32), SDS((), jnp.int32), SDS((), jnp.int32),
        SDS((L, MAX_SEQ, nkv, d), jnp.float32), SDS((L, MAX_SEQ, nkv, d), jnp.float32))
    open(f"{ART}/prefill_int4.stablehlo.mlir", "w").write(exp_p.mlir_module())
    open(f"{ART}/decode_step_int4.stablehlo.mlir", "w").write(exp_d.mlir_module())

    print("compiling int4 graphs to CUDA (GB10, sm_121 via driver PTX-JIT) ...")
    compile_cuda(f"{ART}/prefill_int4.stablehlo.mlir", f"{ART}/prefill_int4.cuda.vmfb")
    compile_cuda(f"{ART}/decode_step_int4.stablehlo.mlir", f"{ART}/decode_step_int4.cuda.vmfb")
    print(f"  prefill.cuda.vmfb {os.path.getsize(f'{ART}/prefill_int4.cuda.vmfb')/1e3:.0f} KB | "
          f"decode.cuda.vmfb {os.path.getsize(f'{ART}/decode_step_int4.cuda.vmfb')/1e3:.0f} KB")

    from iree.runtime import load_vm_flatbuffer_file
    mp = load_vm_flatbuffer_file(f"{ART}/prefill_int4.cuda.vmfb", driver="cuda")
    md = load_vm_flatbuffer_file(f"{ART}/decode_step_int4.cuda.vmfb", driver="cuda")

    leaves = [np.asarray(x) for x in jax.tree_util.tree_flatten(qp)[0]]
    toks = np.zeros(Lp, np.int32); toks[:n] = np.array(ids, np.int32)
    pos = np.arange(Lp, dtype=np.int32)

    t0 = time.time()
    out = mp.main(*leaves, toks, pos, np.asarray(n, np.int32))
    logits = np.asarray(out[0].to_host()); kc, vc = out[1], out[2]
    prefill_ms = (time.time() - t0) * 1e3

    nxt = int(logits.argmax()); gen = [nxt]; clen = cur = n; tdec = []
    for _ in range(47):
        if nxt in EOS:
            break
        t0 = time.time()
        out = md.main(*leaves, np.asarray(nxt, np.int32), np.asarray(cur, np.int32),
                      np.asarray(clen, np.int32), kc, vc)
        logits = np.asarray(out[0].to_host()); kc, vc = out[1], out[2]
        tdec.append(time.time() - t0)
        nxt = int(logits.argmax()); gen.append(nxt); clen += 1; cur += 1

    steady = tdec[1:] if len(tdec) > 1 else tdec
    gpu_text = tok.decode(gen)
    print("\n=== int4 greedy on GB10 (IREE CUDA) ===")
    print(repr(gpu_text))
    cpu = json.load(open(f"{ART}/results_int4.json")) if os.path.exists(f"{ART}/results_int4.json") else None
    if cpu:
        a = cpu["int4_ids"]; m = min(len(a), len(gen))
        agree = sum(int(a[i] == gen[i]) for i in range(m))
        print(f"GPU int4 vs CPU int4 token match: {agree}/{m}"
              + ("  (exact)" if agree == m else f"  first differ {next(i for i in range(m) if a[i]!=gen[i])}"))
    print(f"GPU prefill: {prefill_ms:.0f} ms (incl. first-call JIT) | "
          f"GPU decode: {np.mean(steady)*1e3:.1f} ms/tok over {len(steady)} steps")
    print("  NOTE: not a throughput number. This functional harness re-uploads all "
          "weights as graph\n  inputs every step and syncs logits to host per token; "
          "real GPU perf needs resident\n  weights + batching + on-device sampling (Phase 2b). "
          "GPU correctness is the result here.")
    json.dump({"gpu_text": gpu_text, "gpu_ids": gen,
               "gpu_decode_tok_s": float(1.0/np.mean(steady)) if steady else None},
              open(f"{ART}/results_int4_gpu.json", "w"), indent=2)


if __name__ == "__main__":
    main()
