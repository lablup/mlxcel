"""Validate the Rust-emitted decode_step StableHLO end to end.

Loads the real bf16 Llama-3.2-1B weights (upcast fp32), feeds them as graph
inputs, and drives a greedy loop entirely through decode_step: the prompt is
streamed one token at a time (cache_len = i for prompt token i), then generation
continues. Because decode masks keys with `iota <= cache_len`, streaming the
prompt through decode is mathematically identical to a batched prefill, so this
fully exercises decode_step and reaches the same tokens.

Token target: spike/openxla/artifacts/results.json `hf_ids` (HF temp-0, 48 tok).

Run via validate.sh, or directly:
  .venv/bin/python run_decode.py --mlir decode.mlir
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
import iree.runtime as rt

REF = "/home/inureyes/Development/mlxcel/spike/openxla"
MODEL = f"{REF}/models/Llama-3.2-1B-Instruct"
IREE_COMPILE = f"{REF}/.venv/bin/iree-compile"
RESULTS = f"{REF}/artifacts/results.json"

N_LAYERS, MAX_SEQ, N_KV, HEAD_DIM = 16, 256, 8, 64
EOS = {128001, 128008, 128009}
PROMPT = "Give me three short tips for staying focused while working."
MAX_NEW = 48


def load_weights():
    """Return the 146 weight arrays in the emitter's exact arg order:
    embed, final_norm, then per layer [down, gate, in_ln, post_ln, up, wk, wo,
    wq, wv]. Weights stay [out, in] (bf16 -> fp32), tied embedding."""
    raw = {}
    with safe_open(f"{MODEL}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k).float().numpy().astype(np.float32)
    args = [raw["model.embed_tokens.weight"], raw["model.norm.weight"]]
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        args += [
            raw[p + "mlp.down_proj.weight"],
            raw[p + "mlp.gate_proj.weight"],
            raw[p + "input_layernorm.weight"],
            raw[p + "post_attention_layernorm.weight"],
            raw[p + "mlp.up_proj.weight"],
            raw[p + "self_attn.k_proj.weight"],
            raw[p + "self_attn.o_proj.weight"],
            raw[p + "self_attn.q_proj.weight"],
            raw[p + "self_attn.v_proj.weight"],
        ]
    return args


def compile_module(mlir_path, vmfb_path):
    subprocess.run(
        [IREE_COMPILE, "--iree-input-type=stablehlo",
         "--iree-hal-target-backends=llvm-cpu", mlir_path, "-o", vmfb_path],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", required=True)
    ap.add_argument("--vmfb", default=None)
    args = ap.parse_args()
    vmfb = args.vmfb or (os.path.splitext(args.mlir)[0] + ".vmfb")

    if not args.vmfb or not os.path.exists(vmfb):
        print(f"compiling {args.mlir} -> {vmfb}")
        compile_module(args.mlir, vmfb)

    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(text, add_special_tokens=False).input_ids
    n = len(prompt_ids)
    print(f"prompt tokens = {n}")

    t0 = time.time()
    weights = load_weights()
    print(f"loaded {len(weights)} weight tensors in {time.time()-t0:.1f}s")

    cfg = rt.Config("local-task")
    ctx = rt.SystemContext(config=cfg)
    with open(vmfb, "rb") as f:
        vm = rt.VmModule.from_flatbuffer(ctx.instance, f.read())
    ctx.add_vm_module(vm)
    main_fn = getattr(ctx.modules, vm.name).main

    # Persist weights as device arrays once (the embedding alone is ~1 GB fp32;
    # re-importing it every step dominates otherwise). KV cache also stays on
    # device and is threaded back each step without a host round-trip.
    device = cfg.device
    dev_weights = [rt.asdevicearray(device, w) for w in weights]
    kcache = rt.asdevicearray(
        device, np.zeros((N_LAYERS, MAX_SEQ, N_KV, HEAD_DIM), np.float32))
    vcache = rt.asdevicearray(
        device, np.zeros((N_LAYERS, MAX_SEQ, N_KV, HEAD_DIM), np.float32))

    def step(token, pos, clen, kc, vc):
        outs = main_fn(*dev_weights,
                       np.array(token, np.int32), np.array(pos, np.int32),
                       np.array(clen, np.int32), kc, vc)
        return outs[0], outs[1], outs[2]

    # stream the prompt through decode (cache_len = i for prompt token i)
    t0 = time.time()
    logits = None
    for i in range(n):
        logits, kcache, vcache = step(prompt_ids[i], i, i, kcache, vcache)
    prompt_ms = (time.time() - t0) * 1e3

    # greedy generation
    gen = []
    next_tok = int(np.asarray(logits).argmax())
    gen.append(next_tok)
    clen = n
    step_ms = []
    for _ in range(MAX_NEW - 1):
        if next_tok in EOS:
            break
        t0 = time.time()
        logits, kcache, vcache = step(next_tok, clen, clen, kcache, vcache)
        step_ms.append((time.time() - t0) * 1e3)
        next_tok = int(np.asarray(logits).argmax())
        gen.append(next_tok)
        clen += 1

    hf_ids = json.load(open(RESULTS))["hf_ids"]
    m = min(len(gen), len(hf_ids))
    matches = sum(int(gen[i] == hf_ids[i]) for i in range(m))
    first_div = next((i for i in range(m) if gen[i] != hf_ids[i]), None)

    print("\n=== Rust-emitted decode_step greedy continuation ===")
    print(repr(tok.decode(gen)))
    print(f"\nprompt stream: {prompt_ms:.0f} ms ({n} tok) | "
          f"decode steady: {np.mean(step_ms[1:]) if len(step_ms) > 1 else float('nan'):.0f} ms/tok")
    print(f"token match vs HF temp-0: {matches}/{m}"
          + (f"  first divergence at {first_div}" if first_div is not None else "  (EXACT)"))
    print("RESULT:", "TOKEN-EXACT PASS" if (matches == m == len(hf_ids)) else "MISMATCH")
    sys.exit(0 if matches == m == len(hf_ids) else 1)


if __name__ == "__main__":
    main()
