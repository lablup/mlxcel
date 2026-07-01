"""Execution check for issue #495: Gemma2 sliding-window (local/global) attention.

Validates the mlxcel-xla Rust emitter's Gemma2 sliding-window mask against an
independent HF fp32 Gemma2 oracle, on a small synthetic model so the 4096-token
production window can be shrunk to bite inside the emitter's 256-slot cache
(`MAX_SEQ`). transformers 5.x Gemma2 uses the same schedule the emitter does:
`layer_types = [sliding, full, sliding, full, ...]` (even layers local), a
per-local-layer `config.sliding_window`, and `(1 + weight)` RMSNorm; so a matched
config lets us compare last-token logits directly.

Method (weights fed identically to both sides, so the ONLY variable is the mask):
  1. Build one small HF Gemma2 (random weights; the 1-D norm weights, which HF
     inits to zero, are randomized so the norms are exercised). Freeze its
     state_dict.
  2. For each window W in {inert (>= prompt length), biting (< prompt length)}:
       - emit the prefill graph for a matched Gemma2 config (window W) via the
         scoped, pure-Rust `dump_prefill_graph_for_execution_check` test,
       - compile it with IREE (llvm-cpu) and run it on the frozen weights,
       - run HF (eager, fp32) with `sliding_window = W` on the same tokens,
       - compare last-token logits (argmax + max abs diff).
  3. Confirm the biting window actually changes HF's output vs the inert window
     (so the "token-exact at the biting window" result is not vacuous).

Run (from the repo, using the spike venv's python):
    MLXCEL spike venv python  spike/openxla/gemma2_sliding_window_check.py

Exit 0 = PASS. Short prompt / CPU target, streams progress (watchdog-safe).
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
from transformers import Gemma2Config, Gemma2ForCausalLM

WORKTREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREFILL_LP = 256  # emitter MAX_SEQ / prefill bucket
PROMPT_LEN = 24  # > biting window, < inert window and < PREFILL_LP
TOL = 2e-2  # last-token logit tolerance (Gemma2 tanh soft-cap near-ties)

# Small Gemma2 dims shared by the emitter config and the HF config. head_dim is
# deliberately not hidden/heads (n_q*head_dim = 32 != hidden = 16) to exercise the
# non-square o_proj, as in real Gemma2.
DIMS = dict(
    hidden_size=16,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    intermediate_size=32,
    num_hidden_layers=4,
    vocab_size=64,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=512,
    query_pre_attn_scalar=8,
    attn_logit_softcapping=50.0,
    final_logit_softcapping=30.0,
    hidden_activation="gelu_pytorch_tanh",
)
WINDOWS = {"inert": 64, "biting": 8}  # 64 >= PROMPT_LEN (no-op); 8 < PROMPT_LEN


def arg_order_names(n_layers):
    """The emitter's weight arg order for a tied, bias-free Gemma2 (matches
    `weight_names` in iree.rs: embed, final_norm, then per layer down, gate,
    in_ln, post_ln, up, wk, wo, wq, wv, pre_ff, post_ff)."""
    names = ["model.embed_tokens.weight", "model.norm.weight"]
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
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            names.append(p + suf)
    return names


def build_reference_weights():
    torch.manual_seed(0)
    cfg = Gemma2Config(**DIMS, sliding_window=WINDOWS["inert"])
    model = Gemma2ForCausalLM(cfg).eval().float()
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.dim() == 1:  # RMSNorm weights (HF inits them to zero)
                param.copy_(torch.randn_like(param) * 0.1)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def emitter_logits(window, weights_np, tokens, positions, real_len):
    cfg_json = dict(
        model_type="gemma2",
        hidden_size=DIMS["hidden_size"],
        num_attention_heads=DIMS["num_attention_heads"],
        num_key_value_heads=DIMS["num_key_value_heads"],
        head_dim=DIMS["head_dim"],
        intermediate_size=DIMS["intermediate_size"],
        num_hidden_layers=DIMS["num_hidden_layers"],
        vocab_size=DIMS["vocab_size"],
        rms_norm_eps=DIMS["rms_norm_eps"],
        rope_theta=DIMS["rope_theta"],
        query_pre_attn_scalar=DIMS["query_pre_attn_scalar"],
        attn_logit_softcapping=DIMS["attn_logit_softcapping"],
        final_logit_softcapping=DIMS["final_logit_softcapping"],
        sliding_window=window,
        tie_word_embeddings=True,
    )
    workdir = tempfile.mkdtemp(prefix=f"gemma2_sw_{window}_")
    cfg_path = os.path.join(workdir, "config.json")
    mlir_path = os.path.join(workdir, "prefill.mlir")
    vmfb_path = os.path.join(workdir, "prefill.vmfb")
    with open(cfg_path, "w") as fh:
        json.dump(cfg_json, fh)

    print(f"[emit] window={window}: cargo dump prefill graph ...", flush=True)
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "mlxcel-xla",
            "--lib",
            "emitter::tests::dump_prefill_graph_for_execution_check",
            "--",
            "--ignored",
            "--nocapture",
        ],
        cwd=WORKTREE,
        env={**os.environ, "MLXCEL_DUMP_CONFIG": cfg_path, "MLXCEL_DUMP_OUT": mlir_path},
        check=True,
    )

    print(f"[compile] window={window}: iree-compile (llvm-cpu) ...", flush=True)
    compile_file(
        mlir_path,
        output_file=vmfb_path,
        input_type="stablehlo",
        target_backends=["llvm-cpu"],
    )

    print(f"[run] window={window}: IREE prefill ...", flush=True)
    mod = load_vm_flatbuffer_file(vmfb_path, driver="local-task")
    inputs = list(weights_np) + [tokens, positions, real_len]
    out = mod.main(*inputs)
    logits = out[0].to_host() if hasattr(out[0], "to_host") else np.asarray(out[0])
    return np.asarray(logits, dtype=np.float32)


def hf_logits(window, state_dict, prompt):
    cfg = Gemma2Config(**DIMS, sliding_window=window)
    cfg._attn_implementation = "eager"  # eager applies Gemma2 logit soft-capping
    model = Gemma2ForCausalLM(cfg).eval().float()
    model.load_state_dict(state_dict)
    with torch.no_grad():
        out = model(input_ids=torch.tensor(prompt[None, :], dtype=torch.long))
    return out.logits[0, PROMPT_LEN - 1].numpy().astype(np.float32)


def main():
    state_dict = build_reference_weights()
    names = arg_order_names(DIMS["num_hidden_layers"])
    weights_np = [np.ascontiguousarray(state_dict[n].numpy(), dtype=np.float32) for n in names]

    rng = np.random.default_rng(0)
    prompt = rng.integers(0, DIMS["vocab_size"], size=PROMPT_LEN).astype(np.int32)
    tokens = np.zeros(PREFILL_LP, dtype=np.int32)
    tokens[:PROMPT_LEN] = prompt
    positions = np.arange(PREFILL_LP, dtype=np.int32)
    real_len = np.asarray(PROMPT_LEN, dtype=np.int32)

    ok = True
    hf_by_window = {}
    for tag, window in WINDOWS.items():
        li = emitter_logits(window, weights_np, tokens, positions, real_len)
        lh = hf_logits(window, state_dict, prompt)
        hf_by_window[tag] = lh
        diff = float(np.max(np.abs(li - lh)))
        ai, ah = int(li.argmax()), int(lh.argmax())
        match = ai == ah and diff < TOL
        ok = ok and match
        print(
            f"[{tag}] window={window} argmax iree={ai} hf={ah} "
            f"max|logit diff|={diff:.3e} -> {'OK' if match else 'MISMATCH'}",
            flush=True,
        )

    # The biting window must actually change HF's output vs the inert window, else
    # "token-exact at the biting window" would be a vacuous no-op.
    bite = float(np.max(np.abs(hf_by_window["inert"] - hf_by_window["biting"])))
    bit = bite > TOL
    ok = ok and bit
    print(
        f"[bite] HF max|logit diff| inert-vs-biting = {bite:.3e} -> "
        f"{'window bites' if bit else 'NO EFFECT (window did not bite)'}",
        flush=True,
    )

    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
