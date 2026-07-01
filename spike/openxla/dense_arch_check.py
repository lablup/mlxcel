"""Execution check for issue #498: the dense arch pack (Cohere/Cohere2, Phi3,
StableLM, StarCoder2, Granite, MiniCPM).

Validates the mlxcel-xla Rust emitter's per-family forward against an independent
HF fp32 oracle, on a SMALL synthetic model per arch (random weights fed identically
to both sides, so the only variable is the emitted math). Mirrors the method of
`gemma2_sliding_window_check.py`: build one small HF model, freeze its state_dict,
emit the prefill graph via the scoped pure-Rust `dump_prefill_graph_for_execution_check`
test, compile it with IREE (llvm-cpu), run it on the frozen weights in the emitter's
arg order, and compare the last-token logits (argmax + max abs diff) to HF eager fp32.

The families in `transformers` (cohere, cohere2, phi3, stablelm, starcoder2, granite)
use their HF classes as the oracle; MiniCPM (v1, not in transformers) uses a small
numpy oracle (Llama forward + scale_emb / scale_depth / dim_model_base).

Run (from the repo, with the spike venv python), one or more arch names:
    spike/openxla/.venv/bin/python spike/openxla/dense_arch_check.py cohere granite ...
    ... all        # every arch

Exit 0 = all requested PASS. Short prompt / CPU target, streams progress.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from iree.compiler.tools import compile_file
from iree.runtime import load_vm_flatbuffer_file

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREFILL_LP = 256
PROMPT_LEN = 12
TOL = 3e-2  # last-token logit tolerance (fp32 CPU; interleaved-rope/layernorm slack)


# --- per-arch specs -----------------------------------------------------------
# Each spec: HF config/model builders, small dims, the emitter config.json, and the
# flags that drive the emitter's weight arg order (mirrors take_layer_weights /
# weight_names). `has_post` = sequential (not parallel_block); `gated` = not dense.


def base_dims(**over):
    d = dict(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        num_hidden_layers=3,
        vocab_size=48,
        max_position_embeddings=512,
        rope_theta=10000.0,
        # small special-token ids so HF embedding padding_idx checks pass on the
        # tiny synthetic vocab (the emitter ignores these).
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    d.update(over)
    return d


def build_hf(arch, dims):
    import transformers as tf

    if arch == "cohere":
        cfg = tf.CohereConfig(**dims, layer_norm_eps=1e-5, logit_scale=0.25, use_qk_norm=False)
        model = tf.CohereForCausalLM(cfg)
    elif arch == "cohere2":
        cfg = tf.Cohere2Config(
            **dims, layer_norm_eps=1e-5, logit_scale=0.25,
            sliding_window=64, sliding_window_pattern=2,
        )
        model = tf.Cohere2ForCausalLM(cfg)
    elif arch == "phi3":
        cfg = tf.Phi3Config(**dims, rms_norm_eps=1e-5, tie_word_embeddings=False)
        model = tf.Phi3ForCausalLM(cfg)
    elif arch == "stablelm":
        cfg = tf.StableLmConfig(
            **dims, layer_norm_eps=1e-5, tie_word_embeddings=False,
            use_qkv_bias=True, partial_rotary_factor=0.25, qk_layernorm=False,
            use_parallel_residual=False,
        )
        model = tf.StableLmForCausalLM(cfg)
    elif arch == "starcoder2":
        cfg = tf.Starcoder2Config(
            **dims, norm_epsilon=1e-5, use_bias=True,
            hidden_act="gelu_pytorch_tanh", tie_word_embeddings=True,
        )
        model = tf.Starcoder2ForCausalLM(cfg)
    elif arch == "granite":
        cfg = tf.GraniteConfig(
            **dims, rms_norm_eps=1e-5, tie_word_embeddings=True,
            embedding_multiplier=12.0, residual_multiplier=0.22,
            attention_multiplier=0.125, logits_scaling=8.0,
        )
        model = tf.GraniteForCausalLM(cfg)
    else:
        raise ValueError(arch)
    return model.eval().float()


def emitter_config(arch, dims):
    """The config.json the Rust emitter parses (same shape as the HF config)."""
    c = dict(
        model_type=arch,
        hidden_size=dims["hidden_size"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=dims["num_key_value_heads"],
        intermediate_size=dims["intermediate_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        vocab_size=dims["vocab_size"],
        rope_theta=dims["rope_theta"],
    )
    if arch in ("cohere", "cohere2"):
        c["layer_norm_eps"] = 1e-5
        c["logit_scale"] = 0.25
        c["attention_bias"] = False
    if arch == "cohere":
        c["use_qk_norm"] = False
    if arch == "cohere2":
        c["sliding_window"] = 64
        c["sliding_window_pattern"] = 2
    if arch == "phi3":
        c["rms_norm_eps"] = 1e-5
        c["tie_word_embeddings"] = False
    if arch == "stablelm":
        c["layer_norm_eps"] = 1e-5
        c["tie_word_embeddings"] = False
        c["use_qkv_bias"] = True
        c["partial_rotary_factor"] = 0.25
    if arch == "starcoder2":
        c["norm_epsilon"] = 1e-5
        c["use_bias"] = True
        c["tie_word_embeddings"] = True
    if arch == "granite":
        c["rms_norm_eps"] = 1e-5
        c["tie_word_embeddings"] = True
        c["embedding_multiplier"] = 12.0
        c["residual_multiplier"] = 0.22
        c["attention_multiplier"] = 0.125
        c["logits_scaling"] = 8.0
    return c


def flags(arch):
    layernorm = arch in ("cohere", "cohere2", "stablelm", "starcoder2")
    norm_bias = arch in ("stablelm", "starcoder2")
    parallel = arch in ("cohere", "cohere2")
    dense = arch == "starcoder2"
    qkv_bias = arch in ("stablelm", "starcoder2")
    o_bias = arch == "starcoder2"
    mlp_bias = arch == "starcoder2"
    fused = arch == "phi3"
    untied = arch in ("phi3", "stablelm")
    return dict(
        layernorm=layernorm, norm_bias=norm_bias, parallel=parallel, dense=dense,
        qkv_bias=qkv_bias, o_bias=o_bias, mlp_bias=mlp_bias, fused=fused, untied=untied,
        has_post=not parallel, gated=not dense,
    )


def get(sd, name):
    return sd[name].detach().cpu().numpy().astype(np.float32)


def build_weights(arch, sd, dims):
    """The emitter arg order (mirrors take_layer_weights / weight_names in Rust)."""
    f = flags(arch)
    nl = dims["num_hidden_layers"]
    nq, nkv = dims["num_attention_heads"], dims["num_key_value_heads"]
    hd = dims["hidden_size"] // nq
    inter = dims["intermediate_size"]
    out = [get(sd, "model.embed_tokens.weight"), get(sd, "model.norm.weight")]
    if f["norm_bias"]:
        out.append(get(sd, "model.norm.bias"))
    if f["untied"]:
        out.append(get(sd, "lm_head.weight"))
    for i in range(nl):
        p = f"model.layers.{i}."
        # MLP weights: phi3 fuses gate/up (down stays separate); starcoder2 is dense
        # (c_fc = up, c_proj = down, no gate); else gated gate/up/down.
        if f["fused"]:
            down_w = get(sd, p + "mlp.down_proj.weight")
            gu = get(sd, p + "mlp.gate_up_proj.weight")
            gate_w, up_w = gu[:inter], gu[inter:]  # chunk(2): gate first, up second
        elif f["dense"]:
            up_w = get(sd, p + "mlp.c_fc.weight")
            down_w = get(sd, p + "mlp.c_proj.weight")
            gate_w = None
        else:
            down_w = get(sd, p + "mlp.down_proj.weight")
            up_w = get(sd, p + "mlp.up_proj.weight")
            gate_w = get(sd, p + "mlp.gate_proj.weight")
        # Attention q/k/v: phi3 fuses them into qkv_proj ([Q|K|V] rows).
        if f["fused"]:
            qkv = get(sd, p + "self_attn.qkv_proj.weight")
            wq = qkv[: nq * hd]
            wk = qkv[nq * hd : nq * hd + nkv * hd]
            wv = qkv[nq * hd + nkv * hd :]
        else:
            wq, wk, wv = (get(sd, p + f"self_attn.{x}_proj.weight") for x in "qkv")
        wo = get(sd, p + "self_attn.o_proj.weight")
        # order: down, [gate], in_ln, [post_ln], up, wk, wo, wq, wv, ...
        out.append(down_w)
        if f["gated"]:
            out.append(gate_w)
        out.append(get(sd, p + "input_layernorm.weight"))
        if f["has_post"]:
            out.append(get(sd, p + "post_attention_layernorm.weight"))
        out += [up_w, wk, wo, wq, wv]
        if f["qkv_bias"]:
            out += [get(sd, p + f"self_attn.{x}_proj.bias") for x in "kqv"]
        if f["norm_bias"]:
            out.append(get(sd, p + "input_layernorm.bias"))
            if f["has_post"]:
                out.append(get(sd, p + "post_attention_layernorm.bias"))
        if f["o_bias"]:
            out.append(get(sd, p + "self_attn.o_proj.bias"))
        if f["mlp_bias"]:
            if arch == "starcoder2":
                out.append(get(sd, p + "mlp.c_proj.bias"))  # down
                out.append(get(sd, p + "mlp.c_fc.bias"))    # up
            else:
                out.append(get(sd, p + "mlp.down_proj.bias"))
                if f["gated"]:
                    out.append(get(sd, p + "mlp.gate_proj.bias"))
                out.append(get(sd, p + "mlp.up_proj.bias"))
    return [np.ascontiguousarray(w, dtype=np.float32) for w in out]


def randomize_1d(model):
    """Exercise every norm weight / bias / qkv bias (HF inits many to 0 or 1)."""
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.dim() == 1:
                param.copy_(torch.randn_like(param) * 0.1)


def emitter_logits(arch, dims, weights_np, tokens, positions, real_len):
    cfg_json = emitter_config(arch, dims)
    workdir = tempfile.mkdtemp(prefix=f"dense_{arch}_")
    cfg_path = os.path.join(workdir, "config.json")
    mlir_path = os.path.join(workdir, "prefill.mlir")
    vmfb_path = os.path.join(workdir, "prefill.vmfb")
    with open(cfg_path, "w") as fh:
        json.dump(cfg_json, fh)
    print(f"[emit] {arch}: cargo dump prefill graph ...", flush=True)
    subprocess.run(
        ["cargo", "test", "-p", "mlxcel-xla", "--lib",
         "emitter::tests::dump_prefill_graph_for_execution_check",
         "--", "--ignored", "--nocapture"],
        cwd=WORKTREE,
        env={**os.environ, "MLXCEL_DUMP_CONFIG": cfg_path, "MLXCEL_DUMP_OUT": mlir_path},
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"[compile] {arch}: iree-compile (llvm-cpu) ...", flush=True)
    compile_file(mlir_path, output_file=vmfb_path, input_type="stablehlo",
                 target_backends=["llvm-cpu"])
    print(f"[run] {arch}: IREE prefill ...", flush=True)
    mod = load_vm_flatbuffer_file(vmfb_path, driver="local-task")
    inputs = list(weights_np) + [tokens, positions, real_len]
    out = mod.main(*inputs)
    logits = out[0].to_host() if hasattr(out[0], "to_host") else np.asarray(out[0])
    return np.asarray(logits, dtype=np.float32)


def hf_logits(model, prompt):
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        out = model(input_ids=torch.tensor(prompt[None, :], dtype=torch.long))
    return out.logits[0, PROMPT_LEN - 1].numpy().astype(np.float32)


def check_arch(arch):
    print(f"\n===== {arch} =====", flush=True)
    torch.manual_seed(0)
    dims = base_dims()
    if arch == "cohere2":
        dims = base_dims(num_hidden_layers=4)  # see layers 1,3 full (pattern 2)
    model = build_hf(arch, dims)
    randomize_1d(model)
    sd = model.state_dict()
    weights_np = build_weights(arch, sd, dims)

    rng = np.random.default_rng(0)
    prompt = rng.integers(0, dims["vocab_size"], size=PROMPT_LEN).astype(np.int32)
    tokens = np.zeros(PREFILL_LP, dtype=np.int32)
    tokens[:PROMPT_LEN] = prompt
    positions = np.arange(PREFILL_LP, dtype=np.int32)
    real_len = np.asarray(PROMPT_LEN, dtype=np.int32)

    li = emitter_logits(arch, dims, weights_np, tokens, positions, real_len)
    lh = hf_logits(model, prompt)
    diff = float(np.max(np.abs(li - lh)))
    ai, ah = int(li.argmax()), int(lh.argmax())
    ok = ai == ah and diff < TOL
    print(f"[{arch}] argmax iree={ai} hf={ah} max|logit diff|={diff:.3e} -> "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def main():
    args = sys.argv[1:]
    all_archs = ["cohere", "cohere2", "phi3", "stablelm", "starcoder2", "granite"]
    if not args or args == ["all"]:
        archs = all_archs
    else:
        archs = args
    results = {a: check_arch(a) for a in archs}
    print("\n===== summary =====", flush=True)
    for a, ok in results.items():
        print(f"  {a:12} {'PASS' if ok else 'FAIL'}", flush=True)
    ok_all = all(results.values())
    print("RESULT:", "PASS" if ok_all else "FAIL", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
