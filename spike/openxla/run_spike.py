"""Phase 1 spike harness: export Llama-3.2-1B prefill + decode-step to StableHLO,
run greedy on PJRT from the *serialized* exported artifact, and check the
continuation against an HF transformers temp-0 reference.

Run:  JAX_PLATFORMS=cpu .venv/bin/python run_spike.py
"""
import json
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax import export as jexport
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import model_jax as M

MODEL = "models/Llama-3.2-1B-Instruct"
MAX_SEQ = 256
BUCKETS = [32, 64, 128, 256]
EOS = {128001, 128008, 128009}
SDS = jax.ShapeDtypeStruct
ARTIFACTS = "artifacts"


def pick_bucket(n):
    for b in BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"prompt {n} exceeds max bucket {BUCKETS[-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Give me three short tips for staying focused while working.")
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()

    import os
    os.makedirs(ARTIFACTS, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = int(prompt_ids.shape[0])
    Lp = pick_bucket(n)
    print(f"prompt tokens = {n}  ->  prefill bucket Lp = {Lp}  (MAX_SEQ = {MAX_SEQ})")

    cfg = M.load_config(MODEL)
    L, nkv, d = cfg.n_layers, cfg.n_kv, cfg.head_dim
    print(f"config: L={L} H={cfg.hidden} n_q={cfg.n_q} n_kv={cfg.n_kv} d={d} vocab={cfg.vocab}")

    t0 = time.time()
    params = M.load_params(MODEL, cfg)
    prefill, decode_step = M.make_fns(cfg, MAX_SEQ)
    print(f"loaded params + built fns in {time.time()-t0:.1f}s")

    # ---- export both graphs to StableHLO ----
    params_aval = jax.tree.map(lambda a: SDS(a.shape, a.dtype), params)
    exp_prefill = jexport.export(jax.jit(prefill))(
        params_aval, SDS((Lp,), jnp.int32), SDS((Lp,), jnp.int32), SDS((), jnp.int32))
    exp_decode = jexport.export(jax.jit(decode_step))(
        params_aval, SDS((), jnp.int32), SDS((), jnp.int32), SDS((), jnp.int32),
        SDS((L, MAX_SEQ, nkv, d), jnp.float32), SDS((L, MAX_SEQ, nkv, d), jnp.float32))

    for name, exp in [("prefill", exp_prefill), ("decode_step", exp_decode)]:
        blob = exp.serialize()
        open(f"{ARTIFACTS}/{name}.exported.bin", "wb").write(blob)
        mlir = exp.mlir_module()
        open(f"{ARTIFACTS}/{name}.stablehlo.mlir", "w").write(mlir)
        n_ops = mlir.count("stablehlo.")
        print(f"  exported {name:11s}: serialized {len(blob)/1e3:7.1f} KB | "
              f"StableHLO {len(mlir)/1e3:7.1f} KB, ~{n_ops} stablehlo ops | "
              f"in={len(exp.in_avals)} out={len(exp.out_avals)}")

    # ---- reload the serialized artifacts and drive greedy on PJRT ----
    re_prefill = jexport.deserialize(open(f"{ARTIFACTS}/prefill.exported.bin", "rb").read())
    re_decode = jexport.deserialize(open(f"{ARTIFACTS}/decode_step.exported.bin", "rb").read())

    toks = np.zeros(Lp, np.int32)
    toks[:n] = prompt_ids.numpy().astype(np.int32)
    pos = np.arange(Lp, dtype=np.int32)

    t0 = time.time()
    logits, kc, vc = re_prefill.call(params, jnp.asarray(toks), jnp.asarray(pos), jnp.int32(n))
    jax.block_until_ready((logits, kc, vc))
    prefill_ms = (time.time() - t0) * 1e3

    gen = []
    next_tok = int(np.asarray(logits).argmax())
    gen.append(next_tok)
    clen, cur_pos = n, n
    step_times = []
    for _ in range(args.max_new - 1):
        if next_tok in EOS:
            break
        t0 = time.time()
        logits, kc, vc = re_decode.call(
            params, jnp.int32(next_tok), jnp.int32(cur_pos), jnp.int32(clen), kc, vc)
        jax.block_until_ready(logits)
        step_times.append(time.time() - t0)
        next_tok = int(np.asarray(logits).argmax())
        gen.append(next_tok)
        clen += 1
        cur_pos += 1
    jax_ids = gen

    # warm steady-state decode rate (drop the first compiled call)
    steady = step_times[1:] if len(step_times) > 1 else step_times
    tok_s = (1.0 / np.mean(steady)) if steady else float("nan")

    # ---- HF reference: temp-0 greedy ----
    hf = AutoModelForCausalLM.from_pretrained(MODEL).float().eval()
    with torch.no_grad():
        out = hf.generate(prompt_ids[None], attention_mask=torch.ones_like(prompt_ids[None]),
                          do_sample=False, max_new_tokens=args.max_new,
                          eos_token_id=list(EOS), pad_token_id=128001)
    hf_ids = out[0, n:].tolist()

    # ---- compare ----
    m = min(len(jax_ids), len(hf_ids))
    matches = sum(int(jax_ids[i] == hf_ids[i]) for i in range(m))
    first_div = next((i for i in range(m) if jax_ids[i] != hf_ids[i]), None)

    print("\n=== greedy continuation (exported StableHLO on PJRT) ===")
    print(repr(tok.decode(jax_ids)))
    print("\n=== HF transformers temp-0 reference ===")
    print(repr(tok.decode(hf_ids)))
    print(f"\nprefill: {prefill_ms:.0f} ms (bucket {Lp}) | decode steady: {tok_s:.1f} tok/s "
          f"({np.mean(steady)*1e3:.0f} ms/tok) over {len(steady)} steps")
    print(f"token match: {matches}/{m}"
          + (f"  first divergence at step {first_div}" if first_div is not None else "  (exact)"))

    json.dump({
        "prompt_tokens": n, "bucket": Lp, "max_seq": MAX_SEQ,
        "jax_ids": jax_ids, "hf_ids": hf_ids,
        "token_match": f"{matches}/{m}", "first_divergence": first_div,
        "prefill_ms": prefill_ms, "decode_tok_s": tok_s,
        "jax_text": tok.decode(jax_ids), "hf_text": tok.decode(hf_ids),
    }, open(f"{ARTIFACTS}/results.json", "w"), indent=2)
    print(f"\nwrote {ARTIFACTS}/results.json + prefill/decode_step .stablehlo.mlir + .exported.bin")


if __name__ == "__main__":
    main()
