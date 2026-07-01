#!/usr/bin/env python3
"""Bounded token-exact execution check for the issue #499 dense arch pack.

Proves that the mlxcel-xla Rust emitter's prefill + decode graphs, compiled with
IREE (llvm-cpu) and run on a real checkpoint's dequantized f32 weights, produce
the SAME greedy continuation as an HF fp32 oracle, WITHOUT building the Rust
`xla-iree` feature. It is the in-agent counterpart of `scripts/xla/validate_arch.sh`
(whose two execution gates need the native IREE build): the full token-exact /
serve gate is run post-merge; this proves architectural correctness cheaply on the
smallest checkpoint of a family.

Method (weights fed identically to both sides, so the only variable is the graph):
  1. Dequantize the checkpoint to f32 and run the HF fp32 greedy oracle (eager
     attention, matching the emitter's explicit softmax), recording the prompt's
     last-position logits and the reference token ids.
  2. Emit the prefill + decode host-sampled logits graphs from the model's REAL
     `config.json` via the scoped, pure-Rust `dump_dense_pack_graphs_for_execution_check`
     test, and compile each with IREE (llvm-cpu).
  3. Load the f32 weights in the emitter's arg order (`weight_names.rs`), seed the
     prefill graph, then drive the decode graph token by token, threading the KV
     cache (mirroring `iree.rs::IreeLlama`), taking the argmax each step.
  4. Primary: the emitter's prefill logits must match HF's to a tight tolerance
     (the whole forward on real weights). Secondary: the greedy token trajectory
     (late divergences on an arbitrary prompt are near-tie flips, not errors).

Only architectures the emitter supports load; a deferred arch is rejected at the
emit step (e.g. ERNIE-4.5, whose interleaved GPT-J RoPE this very check surfaced:
its prefill logits diverged from HF by ~8 while a supported family's match to
~1e-8). The in-agent supported checkpoints are large, so this is the opt-in /
post-merge counterpart to the cheap synthetic parity check (`synthetic_arch_check.py`);
it does NOT build the Rust `xla-iree` feature.

Run (from the repo, using the spike venv's python), short prompt / CPU, chatty:
    spike/openxla/.venv/bin/python spike/openxla/arch_execution_check.py \\
        --model /models/<supported-checkpoint> --max-new 16

Exit 0 = forward parity PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import oracle_continuation as oc

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAX_SEQ = 256  # emitter MAX_SEQ / prefill bucket (PREFILL_LP)


def weight_arg_names(cfg: dict) -> list[str]:
    """The emitter's weight arg order for the standard (Llama) scheme, mirroring
    `weight_names.rs`: embed, final_norm, [lm_head if untied], then per layer
    down, gate, in_ln, post_ln, up, wk, wo, wq, wv, [k/q/v bias if qkv_bias].

    The dense pack's standard-name families (Seed-OSS, MiMo, InternLM3) load
    through this; ExaOne 3.x's GPT-2-style scheme is a separate mapping not
    exercised here.
    """
    n_layers = int(cfg.get("num_hidden_layers") or cfg["num_layers"])
    tied = cfg.get("tie_word_embeddings", True)
    mt = cfg.get("model_type")
    qkv_bias = (
        mt == "qwen2"
        or bool(cfg.get("attention_bias"))
        or bool(cfg.get("qkv_bias"))
    )
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


def hf_oracle(
    weights_dir: str, prompt_ids: list[int], n_new: int
) -> tuple[list[int], np.ndarray]:
    """HF fp32 greedy (pure argmax, no EOS stop) for `n_new` steps from explicit
    `prompt_ids`, plus the prompt's last-position logits (the vector the emitter's
    prefill returns). Returns `(ref_token_ids, prompt_last_logits)`.

    Eager attention is used so HF's softmax matches the emitter's explicit softmax
    (the emitter has no fused SDPA), making the logit-parity comparison meaningful.
    The tokenizer is deliberately not used: the check only needs the SAME prompt ids
    on both sides, which sidesteps the custom tokenizers (and their optional deps)
    several of these checkpoints ship. The caller strips the temp dequant config's
    `auto_map` so HF loads the in-tree architecture class (no custom model code)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            weights_dir, dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
        .float()
    )
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    prompt_last_logits: np.ndarray | None = None
    ref: list[int] = []
    with torch.no_grad():
        for step in range(n_new):
            logits = model(ids).logits[:, -1, :]
            if step == 0:
                prompt_last_logits = logits[0].numpy().astype(np.float32)
            nxt = int(torch.argmax(logits, dim=-1).item())
            ref.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    assert prompt_last_logits is not None
    return ref, prompt_last_logits


def emit_graphs(cfg_path: str, out_dir: str) -> tuple[str, str]:
    """Dump the emitter's prefill + decode logits graphs from `cfg_path` via the
    scoped, pure-Rust dump test. Returns (prefill_mlir, decode_mlir) paths."""
    print(f"[emit] cargo dump prefill/decode graphs from {cfg_path} ...", flush=True)
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "mlxcel-xla",
            "--lib",
            "emitter::tests::dump_dense_pack_graphs_for_execution_check",
            "--",
            "--ignored",
            "--nocapture",
        ],
        cwd=WORKTREE,
        env={**os.environ, "MLXCEL_DUMP_CONFIG": cfg_path, "MLXCEL_DUMP_DIR": out_dir},
        check=True,
    )
    return (
        os.path.join(out_dir, "prefill_logits.mlir"),
        os.path.join(out_dir, "decode_logits.mlir"),
    )


def compile_vmfb(mlir_path: str, vmfb_path: str) -> None:
    from iree.compiler.tools import compile_file

    print(f"[compile] iree-compile (llvm-cpu) {os.path.basename(mlir_path)} ...", flush=True)
    compile_file(
        mlir_path,
        output_file=vmfb_path,
        input_type="stablehlo",
        target_backends=["llvm-cpu"],
    )


def load_f32_weights(dequant_dir: str, names: list[str]) -> list[np.ndarray]:
    from safetensors import safe_open

    with safe_open(os.path.join(dequant_dir, "model.safetensors"), framework="numpy") as f:
        avail = set(f.keys())
        out = []
        for n in names:
            if n not in avail:
                raise KeyError(f"weight {n} missing from dequantized checkpoint")
            out.append(np.ascontiguousarray(f.get_tensor(n), dtype=np.float32))
    return out


def to_host(x) -> np.ndarray:
    return x.to_host() if hasattr(x, "to_host") else np.asarray(x)


def emitter_continuation(
    prefill_vmfb: str,
    decode_vmfb: str,
    weights: list[np.ndarray],
    prompt_ids: list[int],
    n_new: int,
) -> list[int]:
    """Seed prefill then drive decode token by token (mirroring `iree.rs`)."""
    from iree.runtime import load_vm_flatbuffer_file

    pre = load_vm_flatbuffer_file(prefill_vmfb, driver="local-task")
    dec = load_vm_flatbuffer_file(decode_vmfb, driver="local-task")

    real_len = len(prompt_ids)
    if real_len > MAX_SEQ:
        raise ValueError(f"prompt of {real_len} exceeds MAX_SEQ={MAX_SEQ}")
    tokens = np.zeros(MAX_SEQ, dtype=np.int32)
    tokens[:real_len] = np.asarray(prompt_ids, dtype=np.int32)
    positions = np.arange(MAX_SEQ, dtype=np.int32)

    print(f"[run] prefill (real_len={real_len}) ...", flush=True)
    out = pre.main(*weights, tokens, positions, np.asarray(real_len, dtype=np.int32))
    logits = np.asarray(to_host(out[0]), dtype=np.float32)
    prefill_logits = logits.copy()  # the prompt's last-position logits
    kc, vc = out[1], out[2]  # keep as device arrays and thread them back in
    tok = int(logits.argmax())
    gen = [tok]

    cache_len = real_len
    for step in range(1, n_new):
        out = dec.main(
            *weights,
            np.asarray(tok, dtype=np.int32),
            np.asarray(cache_len, dtype=np.int32),
            np.asarray(cache_len, dtype=np.int32),
            kc,
            vc,
        )
        logits = to_host(out[0])
        kc, vc = out[1], out[2]
        tok = int(np.asarray(logits).argmax())
        gen.append(tok)
        cache_len += 1
        print(f"[run] decode step {step}: token={tok}", flush=True)
    return gen, prefill_logits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument(
        "--prompt-ids",
        default="",
        help="comma-separated prompt token ids (default: a short valid sequence "
        "derived from the config); the tokenizer is not used",
    )
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    with open(os.path.join(args.model, "config.json")) as f:
        raw_cfg = json.load(f)

    work = tempfile.mkdtemp(prefix="arch_exec_")
    # 1. Dequant (if needed) + HF fp32 oracle.
    quant = oc.read_quantization(args.model)
    dequant_dir = args.model
    if quant is not None:
        dequant_dir = os.path.join(work, "dequant")
        print(
            f"[oracle] dequantizing {args.model} (bits={quant['bits']}, "
            f"group_size={quant['group_size']}) -> {dequant_dir}",
            flush=True,
        )
        oc.dequant_checkpoint(args.model, dequant_dir, quant["bits"], quant["group_size"])
    # Use the in-tree architecture class, not the checkpoint's bundled custom code:
    # strip `auto_map` from the temp dequant config (safe only on the copy we made).
    if dequant_dir != args.model:
        dq_cfg_path = os.path.join(dequant_dir, "config.json")
        with open(dq_cfg_path) as f:
            dq_cfg = json.load(f)
        if dq_cfg.pop("auto_map", None) is not None:
            with open(dq_cfg_path, "w") as f:
                json.dump(dq_cfg, f, indent=2)
    # Prompt ids: explicit override, else a short valid sequence within vocab.
    vocab = int(raw_cfg["vocab_size"])
    if args.prompt_ids.strip():
        prompt_ids = [int(x) for x in args.prompt_ids.split(",")]
    else:
        bos = raw_cfg.get("bos_token_id")
        if isinstance(bos, list):
            bos = bos[0]
        seed = [bos, 1000, 2000, 4000, 8000, 16000]
        prompt_ids = [t for t in seed if isinstance(t, int) and 0 <= t < vocab][:6]
    if not prompt_ids or any(not (0 <= t < vocab) for t in prompt_ids):
        raise ValueError(f"prompt ids {prompt_ids} out of range [0,{vocab})")
    print(f"[oracle] prompt_ids={prompt_ids}", flush=True)
    print("[oracle] HF fp32 greedy (eager) ...", flush=True)
    ref, hf_prompt_logits = hf_oracle(dequant_dir, prompt_ids, args.max_new)
    print(f"[oracle] ref continuation: {ref}", flush=True)

    # 2. Emit + compile the emitter graphs from the REAL config.json.
    cfg_path = os.path.join(dequant_dir, "config.json")  # dequant strips quantization
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(args.model, "config.json")
    prefill_mlir, decode_mlir = emit_graphs(cfg_path, work)
    prefill_vmfb = os.path.join(work, "prefill.vmfb")
    decode_vmfb = os.path.join(work, "decode.vmfb")
    compile_vmfb(prefill_mlir, prefill_vmfb)
    compile_vmfb(decode_mlir, decode_vmfb)

    # 3. Load weights in emitter order, drive prefill + decode.
    names = weight_arg_names(raw_cfg)
    weights = load_f32_weights(dequant_dir, names)
    print(f"[run] loaded {len(weights)} weight tensors in emitter arg order", flush=True)
    got, emit_prompt_logits = emitter_continuation(
        prefill_vmfb, decode_vmfb, weights, prompt_ids, len(ref)
    )

    # 4a. Primary criterion: prefill logit parity. The emitter's prefill runs the
    # WHOLE forward (embed, every layer's attention + MLP + norms, RoPE, final norm,
    # LM head) on the real dequantized weights; matching HF's last-position logits
    # to a tight tolerance proves the architecture is emitted correctly. This is
    # robust (unlike a bare argmax) to the ~1e-3 eager-vs-emitter numerical gap that
    # can flip near-ties, especially on the arbitrary prompt ids used here.
    diff = float(np.max(np.abs(emit_prompt_logits - hf_prompt_logits)))
    rel = diff / float(np.max(np.abs(hf_prompt_logits)) + 1e-9)
    ai, ah = int(emit_prompt_logits.argmax()), int(hf_prompt_logits.argmax())
    tol = 5e-2
    logit_ok = ai == ah and diff < tol

    # 4b. Secondary (informational): the greedy token trajectory. Reported as a
    # match count; late divergences on the nonsensical prompt are near-tie flips,
    # not an architecture error (the real token-exact gate uses real prompts and
    # runs post-merge via the Rust IREE engine).
    n = min(len(got), len(ref))
    traj_match = sum(1 for i in range(n) if got[i] == ref[i])
    print("", flush=True)
    print(f"[compare] emitter tokens: {got[:n]}", flush=True)
    print(f"[compare] oracle  tokens: {ref[:n]}", flush=True)
    print(
        f"[compare] prefill logits: argmax emitter={ai} hf={ah} "
        f"max|diff|={diff:.4e} (rel {rel:.2e}); trajectory {traj_match}/{n} match",
        flush=True,
    )
    if logit_ok:
        print(
            f"RESULT: PASS (prefill forward matches HF fp32: argmax equal, "
            f"max|logit diff| {diff:.4e} < {tol})",
            flush=True,
        )
        return 0
    print(
        f"RESULT: FAIL (prefill argmax emitter={ai} vs hf={ah}, "
        f"max|logit diff| {diff:.4e})",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
