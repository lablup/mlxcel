"""Validate the Rust-emitted prefill + decode_step StableHLO end to end.

Drives the autoregressive loop the way a real runtime would: the bucketed
`prefill` graph processes the whole padded prompt at once and produces the first
token, then `decode_step` continues greedily, threading the KV cache prefill
populated. This exercises the Rust-emitted prefill graph specifically (the
multi-token embedding `stablehlo.gather`, the [Lp,Lp] causal mask, the [Lp] KV
block write, and the real_len-1 last-logit slice), not just decode streaming.

Token target: spike/openxla/artifacts/results.json `hf_ids` (HF temp-0, 48 tok).

Run via validate.sh, or directly:
  .venv/bin/python run_prefill.py --prefill prefill.mlir --decode decode.mlir
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
LP = 64  # prefill bucket (matches results.json `bucket`)
EOS = {128001, 128008, 128009}
PROMPT = "Give me three short tips for staying focused while working."
MAX_NEW = 48


def load_weights():
    """Return the 146 weight arrays in the emitter's exact arg order:
    embed, final_norm, then per layer [down, gate, in_ln, post_ln, up, wk, wo,
    wq, wv]. Weights stay [out, in] (bf16 -> fp32), tied embedding. Identical to
    run_decode.load_weights; prefill and decode share the same weight schema."""
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


def load_fn(ctx, vmfb_path):
    with open(vmfb_path, "rb") as f:
        vm = rt.VmModule.from_flatbuffer(ctx.instance, f.read())
    ctx.add_vm_module(vm)
    return getattr(ctx.modules, vm.name).main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", required=True, help="prefill .mlir")
    ap.add_argument("--decode", required=True, help="decode_step .mlir")
    ap.add_argument("--prefill-vmfb", default=None)
    ap.add_argument("--decode-vmfb", default=None)
    args = ap.parse_args()
    pre_vmfb = args.prefill_vmfb or (os.path.splitext(args.prefill)[0] + ".vmfb")
    dec_vmfb = args.decode_vmfb or (os.path.splitext(args.decode)[0] + ".vmfb")

    for mlir, vmfb in [(args.prefill, pre_vmfb), (args.decode, dec_vmfb)]:
        if not os.path.exists(vmfb):
            print(f"compiling {mlir} -> {vmfb}")
            compile_module(mlir, vmfb)

    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(text, add_special_tokens=False).input_ids
    n = len(prompt_ids)
    print(f"prompt tokens = {n} (bucket Lp = {LP})")
    if n > LP:
        sys.exit(f"prompt ({n}) exceeds bucket ({LP})")

    t0 = time.time()
    weights = load_weights()
    print(f"loaded {len(weights)} weight tensors in {time.time()-t0:.1f}s")

    cfg = rt.Config("local-task")
    ctx = rt.SystemContext(config=cfg)
    prefill_fn = load_fn(ctx, pre_vmfb)
    decode_fn = load_fn(ctx, dec_vmfb)

    device = cfg.device
    dev_weights = [rt.asdevicearray(device, w) for w in weights]

    # --- prefill: whole padded prompt in one shot ---
    tokens = np.zeros(LP, np.int32)
    tokens[:n] = prompt_ids                       # pad tail with token 0 (masked out)
    positions = np.arange(LP, dtype=np.int32)     # 0..Lp-1; pads are causally masked
    t0 = time.time()
    outs = prefill_fn(*dev_weights, tokens, positions, np.array(n, np.int32))
    logits, kcache, vcache = outs[0], outs[1], outs[2]
    prefill_ms = (time.time() - t0) * 1e3

    gen = [int(np.asarray(logits).argmax())]      # first token from prefill (pos n-1)
    clen = n

    # --- decode: continue greedily, threading prefill's KV cache ---
    step_ms = []
    next_tok = gen[0]
    for _ in range(MAX_NEW - 1):
        if next_tok in EOS:
            break
        t0 = time.time()
        outs = decode_fn(*dev_weights,
                         np.array(next_tok, np.int32), np.array(clen, np.int32),
                         np.array(clen, np.int32), kcache, vcache)
        logits, kcache, vcache = outs[0], outs[1], outs[2]
        step_ms.append((time.time() - t0) * 1e3)
        next_tok = int(np.asarray(logits).argmax())
        gen.append(next_tok)
        clen += 1

    hf_ids = json.load(open(RESULTS))["hf_ids"]
    m = min(len(gen), len(hf_ids))
    matches = sum(int(gen[i] == hf_ids[i]) for i in range(m))
    first_div = next((i for i in range(m) if gen[i] != hf_ids[i]), None)

    print("\n=== Rust-emitted prefill + decode greedy continuation ===")
    print(repr(tok.decode(gen)))
    print(f"\nprefill: {prefill_ms:.0f} ms ({n} tok, bucket {LP}) | "
          f"decode steady: {np.mean(step_ms[1:]) if len(step_ms) > 1 else float('nan'):.0f} ms/tok")
    print(f"token match vs HF temp-0: {matches}/{m}"
          + (f"  first divergence at {first_div}" if first_div is not None else "  (EXACT)"))
    ok = matches == m == len(hf_ids)
    print("RESULT:", "TOKEN-EXACT PASS" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
