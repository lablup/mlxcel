#!/usr/bin/env python3
"""Synthetic HF-vs-emitter forward-parity check for the issue #499 dense pack.

Some dense-pack families ship only in large or custom-code checkpoints that are
impractical to run in-agent, but have an in-tree HF config class. For those, this
builds a TINY random HF model (real modeling code, no download, no dequant),
freezes its weights, feeds them to the mlxcel-xla emitter's prefill graph compiled
with IREE (llvm-cpu), and compares last-token logits. The weights are identical on
both sides, so the only variable is the forward, and HF's real modeling code is the
oracle. This catches exactly the class of delta config inspection misses (e.g.
ERNIE-4.5's interleaved RoPE), which is why it complements the byte-exact structural
gate and the config/weight-name unit tests.

Currently validates Seed-OSS (native `SeedOssConfig`; the untied + q/k/v-bias
Qwen2 forward with `rope_type = default`). It does NOT build the Rust `xla-iree`
feature (only the pure-Rust dump test), so it is watchdog-safe.

Run (from the repo, using the spike venv's python):
    spike/openxla/.venv/bin/python spike/openxla/synthetic_arch_check.py

Exit 0 = PASS (argmax equal and last-token logits within tolerance).
"""

from __future__ import annotations

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
PREFILL_LP = 256  # emitter MAX_SEQ / prefill bucket
PROMPT_LEN = 24
TOL = 2e-2


def arg_order_names(n_layers: int, tied: bool, qkv_bias: bool) -> list[str]:
    """Emitter weight arg order (Llama scheme) mirroring `weight_names.rs`."""
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    if not tied:
        names.append("lm_head.weight")
    for i in range(n_layers):
        p = f"model.layers.{i}."
        for suf in (
            "mlp.down_proj.weight",
            "mlp.gate_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.up_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ):
            names.append(p + suf)
        if qkv_bias:
            for suf in (
                "self_attn.k_proj.bias",
                "self_attn.q_proj.bias",
                "self_attn.v_proj.bias",
            ):
                names.append(p + suf)
    return names


def emitter_last_logits(cfg_json: dict, weights_np: list[np.ndarray], prompt: np.ndarray) -> np.ndarray:
    work = tempfile.mkdtemp(prefix="synth_arch_")
    cfg_path = os.path.join(work, "config.json")
    with open(cfg_path, "w") as fh:
        json.dump(cfg_json, fh)
    print("[emit] cargo dump prefill graph ...", flush=True)
    subprocess.run(
        [
            "cargo", "test", "-p", "mlxcel-xla", "--lib",
            "emitter::tests::dump_dense_pack_graphs_for_execution_check",
            "--", "--ignored", "--nocapture",
        ],
        cwd=WORKTREE,
        env={**os.environ, "MLXCEL_DUMP_CONFIG": cfg_path, "MLXCEL_DUMP_DIR": work},
        check=True,
    )
    vmfb = os.path.join(work, "prefill.vmfb")
    print("[compile] iree-compile (llvm-cpu) ...", flush=True)
    compile_file(
        os.path.join(work, "prefill_logits.mlir"),
        output_file=vmfb,
        input_type="stablehlo",
        target_backends=["llvm-cpu"],
    )
    tokens = np.zeros(PREFILL_LP, dtype=np.int32)
    tokens[:PROMPT_LEN] = prompt
    positions = np.arange(PREFILL_LP, dtype=np.int32)
    print("[run] IREE prefill ...", flush=True)
    mod = load_vm_flatbuffer_file(vmfb, driver="local-task")
    out = mod.main(*weights_np, tokens, positions, np.asarray(PROMPT_LEN, dtype=np.int32))
    logits = out[0].to_host() if hasattr(out[0], "to_host") else np.asarray(out[0])
    return np.asarray(logits, dtype=np.float32)


def check_seed_oss() -> bool:
    from transformers import SeedOssConfig, SeedOssForCausalLM

    dims = dict(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,  # n_q*head_dim=32 != hidden=16 -> non-square o_proj, as in real Seed-OSS
        intermediate_size=32,
        num_hidden_layers=3,
        vocab_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        attention_bias=True,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    hf_cfg = SeedOssConfig(**dims)
    hf_cfg._attn_implementation = "eager"
    model = SeedOssForCausalLM(hf_cfg).eval().float()
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.dim() == 1:  # RMSNorm weights / biases: randomize so they are exercised
                p.copy_(torch.randn_like(p) * 0.1)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    names = arg_order_names(dims["num_hidden_layers"], tied=False, qkv_bias=True)
    weights = [np.ascontiguousarray(state[n].numpy(), dtype=np.float32) for n in names]

    rng = np.random.default_rng(0)
    prompt = rng.integers(0, dims["vocab_size"], size=PROMPT_LEN).astype(np.int32)

    emit_cfg = dict(
        model_type="seed_oss",
        hidden_size=dims["hidden_size"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=dims["num_key_value_heads"],
        head_dim=dims["head_dim"],
        intermediate_size=dims["intermediate_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        vocab_size=dims["vocab_size"],
        rms_norm_eps=dims["rms_norm_eps"],
        rope_theta=dims["rope_theta"],
        attention_bias=True,
        tie_word_embeddings=False,
        rope_scaling={"rope_type": "default"},
    )
    emit_logits = emitter_last_logits(emit_cfg, weights, prompt)

    with torch.no_grad():
        hf_logits = (
            model(input_ids=torch.tensor(prompt[None, :], dtype=torch.long))
            .logits[0, PROMPT_LEN - 1]
            .numpy()
            .astype(np.float32)
        )
    diff = float(np.max(np.abs(emit_logits - hf_logits)))
    ai, ah = int(emit_logits.argmax()), int(hf_logits.argmax())
    ok = ai == ah and diff < TOL
    print(
        f"[seed_oss] argmax emitter={ai} hf={ah} max|logit diff|={diff:.3e} -> "
        f"{'OK' if ok else 'MISMATCH'}",
        flush=True,
    )
    return ok


def main() -> int:
    ok = check_seed_oss()
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
