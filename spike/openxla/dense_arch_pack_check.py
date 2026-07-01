"""Execution check for issue #497: the dense arch pack (Qwen3, Gemma1, Gemma3,
SmolLM3, OLMo2).

Proves the mlxcel-xla Rust emitter reproduces each family's attention/MLP math by
comparing its prefill last-token logits to an independent HF fp32 oracle, on a
SMALL SYNTHETIC model per family (random weights, tiny dims). The same weights are
fed to both sides, so the only variable is the emitted graph: a logit match proves
the family delta (per-head q/k norm for Qwen3 / Gemma3, flat q/k norm for OLMo2, the
Gemma embed-scale / (1+w) / GeGLU split, the OLMo reordered post-norm, SmolLM3 NoPE,
and Gemma3's dual local/global RoPE) is correct. This needs no real checkpoint and
no `xla-iree` cargo feature: it emits via the scoped pure-Rust
`dump_prefill_graph_for_execution_check` test and runs the graph with IREE's Python
llvm-cpu backend. Short prompt / CPU, streams progress (watchdog-safe).

Method (mirrors spike/openxla/gemma2_sliding_window_check.py):
  1. Build one small HF model of the family (random weights; the 1-D norm weights,
     which HF inits to 0/1, are randomized so the norms and q/k norms are exercised).
  2. Emit the matching emitter `config.json`'s prefill graph via the pure-Rust dump
     test, compile with IREE (llvm-cpu), run on the frozen weights in the emitter's
     arg order.
  3. Run HF (eager, fp32) on the same tokens; compare last-token logits (argmax +
     max abs diff).

Run (from the repo, with the spike venv's python):
    spike/openxla/.venv/bin/python spike/openxla/dense_arch_pack_check.py
    ... --family qwen3      # one family only

Exit 0 = every requested family PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREFILL_LP = 256  # emitter MAX_SEQ / prefill bucket
PROMPT_LEN = 12
TOL = 2e-2  # last-token logit tolerance (loose for the Gemma tanh GeGLU near-ties)

# Small dims shared by the HF config and the emitter config.json. head_dim*n_q ==
# hidden (square o_proj); n_kv < n_q exercises GQA; the flat OLMo2 q/k norm is then
# [n_q*head_dim] = 16 and [n_kv*head_dim] = 8.
HIDDEN = 16
N_Q = 4
N_KV = 2
HEAD_DIM = 4
INTER = 32
N_LAYERS = 4
VOCAB = 64
EPS = 1e-6
ROPE_THETA = 10000.0


def arg_names(*, has_input_norm, qkv_bias, qk_norm, has_pre_ff, has_post_ff, untied):
    """The emitter's weight arg order (mirrors `weight_names` in iree.rs exactly)."""
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    if untied:
        names.append("lm_head.weight")
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        names.append(p + "mlp.down_proj.weight")
        names.append(p + "mlp.gate_proj.weight")
        if has_input_norm:
            names.append(p + "input_layernorm.weight")
        for suf in (
            "post_attention_layernorm.weight",
            "mlp.up_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ):
            names.append(p + suf)
        if qkv_bias:
            for suf in ("k_proj.bias", "q_proj.bias", "v_proj.bias"):
                names.append(p + "self_attn." + suf)
        if qk_norm:
            names.append(p + "self_attn.q_norm.weight")
            names.append(p + "self_attn.k_norm.weight")
        if has_pre_ff:
            names.append(p + "pre_feedforward_layernorm.weight")
        if has_post_ff:
            names.append(p + "post_feedforward_layernorm.weight")
    return names


def base_dims():
    return dict(
        hidden_size=HIDDEN,
        num_attention_heads=N_Q,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        vocab_size=VOCAB,
        rms_norm_eps=EPS,
        rope_theta=ROPE_THETA,
        max_position_embeddings=512,
        attention_bias=False,
    )


def base_emitter_cfg(model_type):
    return dict(
        model_type=model_type,
        hidden_size=HIDDEN,
        num_attention_heads=N_Q,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        vocab_size=VOCAB,
        rms_norm_eps=EPS,
        rope_theta=ROPE_THETA,
        attention_bias=False,
        tie_word_embeddings=True,
    )


def spec_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(**base_dims(), tie_word_embeddings=True)
    return dict(
        hf=(Qwen3Config, Qwen3ForCausalLM, cfg),
        emitter_cfg=base_emitter_cfg("qwen3"),
        arg_flags=dict(
            has_input_norm=True, qkv_bias=False, qk_norm=True,
            has_pre_ff=False, has_post_ff=False, untied=False,
        ),
    )


def spec_gemma1():
    from transformers import GemmaConfig, GemmaForCausalLM

    cfg = GemmaConfig(
        **base_dims(), tie_word_embeddings=True,
        hidden_act="gelu_pytorch_tanh", hidden_activation="gelu_pytorch_tanh",
    )
    ec = base_emitter_cfg("gemma")
    ec["hidden_activation"] = "gelu_pytorch_tanh"
    return dict(
        hf=(GemmaConfig, GemmaForCausalLM, cfg),
        emitter_cfg=ec,
        arg_flags=dict(
            has_input_norm=True, qkv_bias=False, qk_norm=False,
            has_pre_ff=False, has_post_ff=False, untied=False,
        ),
    )


def spec_gemma3():
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM

    # Distinct local RoPE base (100) vs global (ROPE_THETA=10000) so the dual-RoPE is
    # exercised; an inert sliding window (>= prompt) keeps the mask a no-op while the
    # sliding layers still rotate on the local base (as HF does).
    cfg = Gemma3TextConfig(
        **base_dims(), tie_word_embeddings=True,
        query_pre_attn_scalar=HEAD_DIM, rope_local_base_freq=100.0,
        sliding_window=64, sliding_window_pattern=3,
        attn_logit_softcapping=None, final_logit_softcapping=None,
        hidden_activation="gelu_pytorch_tanh",
    )
    ec = base_emitter_cfg("gemma3_text")
    ec.update(
        rope_local_base_freq=100.0, sliding_window=64, sliding_window_pattern=3,
        query_pre_attn_scalar=HEAD_DIM, hidden_activation="gelu_pytorch_tanh",
    )
    return dict(
        hf=(Gemma3TextConfig, Gemma3ForCausalLM, cfg),
        emitter_cfg=ec,
        arg_flags=dict(
            has_input_norm=True, qkv_bias=False, qk_norm=True,
            has_pre_ff=True, has_post_ff=True, untied=False,
        ),
    )


def spec_smollm3():
    from transformers import SmolLM3Config, SmolLM3ForCausalLM

    no_rope = [1, 1, 1, 0]  # layer 3 NoPE
    # SmolLM3Config's real default pad/bos/eos ids exceed the synthetic vocab; pin
    # them inside it so the padding-idx embedding is valid.
    cfg = SmolLM3Config(
        **base_dims(), tie_word_embeddings=True, no_rope_layers=no_rope,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    ec = base_emitter_cfg("smollm3")
    ec["no_rope_layers"] = no_rope
    return dict(
        hf=(SmolLM3Config, SmolLM3ForCausalLM, cfg),
        emitter_cfg=ec,
        arg_flags=dict(
            has_input_norm=True, qkv_bias=False, qk_norm=False,
            has_pre_ff=False, has_post_ff=False, untied=False,
        ),
    )


def spec_olmo2():
    from transformers import Olmo2Config, Olmo2ForCausalLM

    cfg = Olmo2Config(**base_dims(), tie_word_embeddings=False)
    ec = base_emitter_cfg("olmo2")
    ec["tie_word_embeddings"] = False
    return dict(
        hf=(Olmo2Config, Olmo2ForCausalLM, cfg),
        emitter_cfg=ec,
        arg_flags=dict(
            has_input_norm=False, qkv_bias=False, qk_norm=True,
            has_pre_ff=False, has_post_ff=True, untied=True,
        ),
    )


SPECS = {
    "qwen3": spec_qwen3,
    "gemma1": spec_gemma1,
    "gemma3": spec_gemma3,
    "smollm3": spec_smollm3,
    "olmo2": spec_olmo2,
}


def build_reference_weights(model_cls, cfg):
    torch.manual_seed(0)
    model = model_cls(cfg).eval().float()
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.dim() == 1:  # RMSNorm weights (HF inits them to 0 or 1)
                param.copy_(torch.randn_like(param) * 0.1)
    return model


def emitter_logits(emitter_cfg, weights_np, tokens, positions, real_len, tag):
    workdir = tempfile.mkdtemp(prefix=f"dense_{tag}_")
    cfg_path = os.path.join(workdir, "config.json")
    mlir_path = os.path.join(workdir, "prefill.mlir")
    vmfb_path = os.path.join(workdir, "prefill.vmfb")
    with open(cfg_path, "w") as fh:
        json.dump(emitter_cfg, fh)

    print(f"[emit] {tag}: cargo dump prefill graph ...", flush=True)
    subprocess.run(
        [
            "cargo", "test", "-p", "mlxcel-xla", "--lib",
            "emitter::tests::dump_prefill_graph_for_execution_check",
            "--", "--ignored", "--nocapture",
        ],
        cwd=WORKTREE,
        env={**os.environ, "MLXCEL_DUMP_CONFIG": cfg_path, "MLXCEL_DUMP_OUT": mlir_path},
        check=True,
        stdout=subprocess.DEVNULL,
    )

    from iree.compiler.tools import compile_file
    from iree.runtime import load_vm_flatbuffer_file

    print(f"[compile] {tag}: iree-compile (llvm-cpu) ...", flush=True)
    compile_file(
        mlir_path, output_file=vmfb_path,
        input_type="stablehlo", target_backends=["llvm-cpu"],
    )
    print(f"[run] {tag}: IREE prefill ...", flush=True)
    mod = load_vm_flatbuffer_file(vmfb_path, driver="local-task")
    out = mod.main(*(list(weights_np) + [tokens, positions, real_len]))
    logits = out[0].to_host() if hasattr(out[0], "to_host") else np.asarray(out[0])
    return np.asarray(logits, dtype=np.float32)


def hf_logits(model, prompt):
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        out = model(input_ids=torch.tensor(prompt[None, :], dtype=torch.long))
    return out.logits[0, PROMPT_LEN - 1].numpy().astype(np.float32)


def run_family(name):
    print(f"\n===== {name} =====", flush=True)
    spec = SPECS[name]()
    _, model_cls, cfg = spec["hf"]
    model = build_reference_weights(model_cls, cfg)
    state = model.state_dict()

    names = arg_names(**spec["arg_flags"])
    missing = [n for n in names if n not in state]
    if missing:
        print(f"[{name}] MISSING weights in HF state_dict: {missing[:4]} ...", flush=True)
        return False
    weights_np = [np.ascontiguousarray(state[n].numpy(), dtype=np.float32) for n in names]

    rng = np.random.default_rng(1)
    prompt = rng.integers(0, VOCAB, size=PROMPT_LEN).astype(np.int32)
    tokens = np.zeros(PREFILL_LP, dtype=np.int32)
    tokens[:PROMPT_LEN] = prompt
    positions = np.arange(PREFILL_LP, dtype=np.int32)
    real_len = np.asarray(PROMPT_LEN, dtype=np.int32)

    li = emitter_logits(spec["emitter_cfg"], weights_np, tokens, positions, real_len, name)
    lh = hf_logits(model, prompt)
    diff = float(np.max(np.abs(li - lh)))
    ai, ah = int(li.argmax()), int(lh.argmax())
    ok = ai == ah and diff < TOL
    print(
        f"[{name}] argmax iree={ai} hf={ah}  max|logit diff|={diff:.3e}  "
        f"-> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=list(SPECS), help="one family (default: all)")
    args = ap.parse_args()
    families = [args.family] if args.family else list(SPECS)
    results = {}
    for fam in families:
        try:
            results[fam] = run_family(fam)
        except Exception as e:  # noqa: BLE001 - report and continue to the next family
            print(f"[{fam}] ERROR: {type(e).__name__}: {e}", flush=True)
            results[fam] = False
    print("\n===== summary =====", flush=True)
    for fam, ok in results.items():
        print(f"  {fam:9s}: {'PASS' if ok else 'FAIL'}", flush=True)
    all_ok = all(results.values())
    print("RESULT:", "PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
