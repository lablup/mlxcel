"""Execution check for issue #500: the shared MoE FFN graph primitive.

Validates the mlxcel-xla Rust emitter's MoE FFN block against a genuine HF fp32
MoE block, on a small synthetic model so the routing / dispatch math is the ONLY
variable. The emitter's standalone MoE-block probe (`emit_moe_probe`) runs just the
router + top-k expert dispatch + weighted combine (+ shared expert), with no
attention, no pre-norm, and no residual, so it lines up exactly with an HF MoE
block's forward and can be compared directly.

Families checked (issues #501 / #3345 extend #500's Qwen2-MoE / Mixtral):
  - Qwen2-MoE: gated shared expert, softmax-before-top-k, norm_topk_prob.
  - Mixtral: no shared expert, always renormalized.
  - Qwen3-MoE: no shared expert, norm_topk_prob (its qk-norm attention is validated
    separately; the FFN block is identical routing to Mixtral).
  - OLMoE: no shared expert, norm_topk_prob, experts use `intermediate_size`.
  - Nemotron-H: correction-biased grouped sigmoid top-k, relu2 routed and shared
    experts, with combine weights gathered from the uncorrected sigmoid scores.
The first four share the exact softmax-over-all-experts -> top-k -> renorm router;
Nemotron-H exercises the independent sigmoid router and relu2 projection schema.

Method (identical weights fed to both sides):
  1. Build one small HF MoE block (Qwen2-MoE with a gated shared expert; Mixtral /
     Qwen3-MoE / OLMoE with no shared expert), random weights, fp32, eval.
  2. Run the HF block on a random hidden `hn` [N, H] -> reference output [N, H].
  3. Map the HF (fused `gate_up_proj` / `down_proj`) weights to the emitter's probe
     arg layout: router [E, H]; stacked expert gate/up [E, I, H] (the two halves of
     `gate_up_proj`), down [E, H, I]; and, for Qwen2-MoE, the shared SwiGLU + its
     sigmoid gate.
  4. Emit the probe graph for a matched config via the scoped, pure-Rust
     `dump_moe_probe_for_execution_check` test, compile it with IREE (llvm-cpu),
     and run it on those weights + `hn`.
  5. Compare the IREE block output to the HF block output (max abs diff).

The HF block for each case is the independent routing and expert oracle. A close
fp32 match proves both selection semantics and the emitted expert projection schema.

Run (from the repo, using the spike venv's python):
    spike/openxla/.venv/bin/python spike/openxla/moe_oracle.py

ROCm (after `eval "$(scripts/iree/setup-rocm.sh --env)"`, under the host guard):
    spike/openxla/.venv/bin/python spike/openxla/moe_oracle.py \
        --device hip --rocm-target gfx1151

Exit 0 = PASS. Tiny dims, CPU target, streams progress (watchdog-safe).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from iree.compiler.tools import compile_file
from iree.runtime import load_vm_flatbuffer_file

WORKTREE = os.environ.get(
    "MLXCEL_WORKTREE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
TOL = 5e-5  # both sides fp32; only softmax / combine reduction-order noise differs
SEED = 0
DEVICE = "local-task"
ROCM_TARGET = None


def randomize(module):
    """Give every parameter a small, reproducible random value (a standalone block
    is otherwise left at raw-tensor init), so the router logits are distinct (no
    top-k ties) and every projection is exercised."""
    torch.manual_seed(SEED)
    with torch.no_grad():
        for p in module.parameters():
            nn.init.normal_(p, std=0.1)


def emit_probe(cfg_json, n, tag):
    """Dump the emitter's MoE probe graph for `cfg_json`, compile it with IREE
    (llvm-cpu), and return the loaded module."""
    workdir = tempfile.mkdtemp(prefix=f"moe_{tag}_")
    cfg_path = os.path.join(workdir, "config.json")
    mlir_path = os.path.join(workdir, "moe_probe.mlir")
    vmfb_path = os.path.join(workdir, "moe_probe.vmfb")
    with open(cfg_path, "w") as fh:
        json.dump(cfg_json, fh)

    print(f"[emit] {tag}: cargo dump MoE probe graph ...", flush=True)
    subprocess.run(
        [
            "cargo",
            "test",
            "--profile",
            "test-fast",
            "-p",
            "mlxcel-xla",
            "--lib",
            "emitter::tests::dump_moe_probe_for_execution_check",
            "--",
            "--ignored",
            "--nocapture",
        ],
        cwd=WORKTREE,
        env={
            **os.environ,
            "MLXCEL_DUMP_CONFIG": cfg_path,
            "MLXCEL_DUMP_OUT": mlir_path,
            "MLXCEL_MOE_N": str(n),
        },
        check=True,
    )

    if DEVICE == "hip":
        if not ROCM_TARGET:
            raise RuntimeError(
                "--device hip requires --rocm-target or MLXCEL_XLA_ROCM_TARGET"
            )
        backend = "rocm"
        extra_args = [
            "--iree-hal-target-device=hip",
            f"--iree-rocm-target={ROCM_TARGET}",
        ]
    else:
        backend = "llvm-cpu"
        extra_args = [
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
        ]
    print(f"[compile] {tag}: iree-compile ({backend}, driver={DEVICE}) ...", flush=True)
    compile_file(
        mlir_path,
        output_file=vmfb_path,
        input_type="stablehlo",
        target_backends=[backend],
        extra_args=extra_args,
    )
    return load_vm_flatbuffer_file(vmfb_path, driver=DEVICE)


def np32(t):
    return np.ascontiguousarray(t.detach().numpy(), dtype=np.float32)


def run_probe(mod, args):
    out = mod.main(*args)
    out = out[0] if isinstance(out, (list, tuple)) else out
    host = out.to_host() if hasattr(out, "to_host") else np.asarray(out)
    return np.asarray(host, dtype=np.float32)


def split_fused_experts(sd, e, i):
    """The emitter's stacked expert args from HF's fused params. HF stores
    `gate_up_proj` [E, 2I, H] (F.linear out-major, chunked into gate|up on the
    output axis) and `down_proj` [E, H, I]; the emitter takes gate [E, I, H], up
    [E, I, H], down [E, H, I]."""
    gup = sd["experts.gate_up_proj"]  # [E, 2I, H]
    gate = gup[:, :i, :].contiguous()  # [E, I, H]
    up = gup[:, i : 2 * i, :].contiguous()  # [E, I, H]
    down = sd["experts.down_proj"].contiguous()  # [E, H, I]
    assert gate.shape == (e, i, gup.shape[2]), gate.shape
    return gate, up, down


def check_qwen2_moe():
    from transformers import Qwen2MoeConfig
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

    h, i, ish, e, k, n = 16, 6, 10, 4, 2, 5
    cfg = Qwen2MoeConfig(
        hidden_size=h,
        num_experts=e,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        moe_intermediate_size=i,
        shared_expert_intermediate_size=ish,
    )
    block = Qwen2MoeSparseMoeBlock(cfg).float().eval()
    randomize(block)
    sd = block.state_dict()

    torch.manual_seed(SEED + 1)
    hn = torch.randn(1, n, h)
    with torch.no_grad():
        ref = block(hn).reshape(n, h)

    gate, up, down = split_fused_experts(sd, e, i)
    # probe arg order: router, gate, up, down, shared gate/up/down, shared gate, hn
    args = [
        np32(sd["gate.weight"]),  # [E, H]
        np32(gate),
        np32(up),
        np32(down),
        np32(sd["shared_expert.gate_proj.weight"]),  # [Is, H]
        np32(sd["shared_expert.up_proj.weight"]),  # [Is, H]
        np32(sd["shared_expert.down_proj.weight"]),  # [H, Is]
        np32(sd["shared_expert_gate.weight"]),  # [1, H]
        np32(hn.reshape(n, h)),  # [N, H]
    ]
    cfg_json = dict(
        model_type="qwen2_moe",
        hidden_size=h,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=16,
        moe_intermediate_size=i,
        shared_expert_intermediate_size=ish,
        num_hidden_layers=1,
        num_experts=e,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        vocab_size=10,
        tie_word_embeddings=False,
    )
    mod = emit_probe(cfg_json, n, "qwen2_moe")
    out = run_probe(mod, args)
    return compare("qwen2_moe (shared+gate, norm_topk_prob)", out, np32(ref))


def check_mixtral():
    from transformers import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

    h, i, e, k, n = 16, 6, 4, 2, 5
    cfg = MixtralConfig(
        hidden_size=h,
        num_local_experts=e,
        num_experts_per_tok=k,
        intermediate_size=i,
    )
    block = MixtralSparseMoeBlock(cfg).float().eval()
    randomize(block)
    sd = block.state_dict()

    torch.manual_seed(SEED + 2)
    hn = torch.randn(1, n, h)
    with torch.no_grad():
        ref = block(hn).reshape(n, h)

    gate, up, down = split_fused_experts(sd, e, i)
    args = [
        np32(sd["gate.weight"]),  # [E, H]
        np32(gate),
        np32(up),
        np32(down),
        np32(hn.reshape(n, h)),  # [N, H]
    ]
    cfg_json = dict(
        model_type="mixtral",
        hidden_size=h,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=i,
        num_hidden_layers=1,
        num_local_experts=e,
        num_experts_per_tok=k,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        vocab_size=10,
    )
    mod = emit_probe(cfg_json, n, "mixtral")
    out = run_probe(mod, args)
    return compare("mixtral (no shared, always renorm)", out, np32(ref))


def check_qwen3_moe():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    h, i, e, k, n = 16, 6, 4, 2, 5
    cfg = Qwen3MoeConfig(
        hidden_size=h,
        num_experts=e,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        moe_intermediate_size=i,
    )
    block = Qwen3MoeSparseMoeBlock(cfg).float().eval()
    randomize(block)
    sd = block.state_dict()

    torch.manual_seed(SEED + 3)
    hn = torch.randn(1, n, h)
    with torch.no_grad():
        ref = block(hn).reshape(n, h)

    gate, up, down = split_fused_experts(sd, e, i)
    # probe arg order (no shared expert): router, gate, up, down, hn
    args = [
        np32(sd["gate.weight"]),  # [E, H]
        np32(gate),
        np32(up),
        np32(down),
        np32(hn.reshape(n, h)),  # [N, H]
    ]
    cfg_json = dict(
        model_type="qwen3_moe",
        hidden_size=h,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        moe_intermediate_size=i,
        num_hidden_layers=1,
        num_experts=e,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        vocab_size=10,
        tie_word_embeddings=False,
    )
    mod = emit_probe(cfg_json, n, "qwen3_moe")
    out = run_probe(mod, args)
    return compare("qwen3_moe (no shared, norm_topk_prob)", out, np32(ref))


def check_olmoe():
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

    # OLMoE's experts use `intermediate_size` (no `moe_intermediate_size`).
    h, i, e, k, n = 16, 6, 4, 2, 5
    cfg = OlmoeConfig(
        hidden_size=h,
        num_experts=e,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        intermediate_size=i,
    )
    block = OlmoeSparseMoeBlock(cfg).float().eval()
    randomize(block)
    sd = block.state_dict()

    torch.manual_seed(SEED + 4)
    hn = torch.randn(1, n, h)
    with torch.no_grad():
        ref = block(hn).reshape(n, h)

    gate, up, down = split_fused_experts(sd, e, i)
    args = [
        np32(sd["gate.weight"]),  # [E, H]
        np32(gate),
        np32(up),
        np32(down),
        np32(hn.reshape(n, h)),  # [N, H]
    ]
    cfg_json = dict(
        model_type="olmoe",
        hidden_size=h,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=i,
        num_hidden_layers=1,
        num_experts=e,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        rope_theta=500000.0,
        vocab_size=10,
        tie_word_embeddings=False,
    )
    mod = emit_probe(cfg_json, n, "olmoe")
    out = run_probe(mod, args)
    return compare("olmoe (no shared, norm_topk_prob, expert intermediate_size)", out, np32(ref))


def check_nemotron_h(k=2, norm_topk_prob=True, tag="nemotron_h"):
    from transformers import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMoE

    h, i, ish, e, n = 16, 6, 10, 8, 5
    cfg = NemotronHConfig(
        hidden_size=h,
        intermediate_size=32,
        layers_block_type=["moe"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        n_routed_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=i,
        n_shared_experts=1,
        moe_shared_expert_intermediate_size=ish,
        n_group=2,
        topk_group=1,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=1.25,
        mlp_hidden_act="relu2",
        use_mamba_kernels=False,
    )
    block = NemotronHMoE(cfg).float().eval()
    randomize(block)
    # The correction is a registered buffer, not a parameter, so `randomize`
    # intentionally does not touch it. Make it nonzero and non-symmetric to prove
    # it changes selection while combine weights still come from raw sigmoid.
    with torch.no_grad():
        block.gate.e_score_correction_bias.copy_(
            torch.tensor([0.07, -0.03, 0.02, -0.06, -0.01, 0.05, -0.04, 0.03])
        )
    sd = block.state_dict()

    torch.manual_seed(SEED + 5)
    hn = torch.randn(1, n, h)
    with torch.no_grad():
        ref = block(hn).reshape(n, h)

    # Probe arg order: router, correction bias, routed up/down, shared up/down, hn.
    args = [
        np32(sd["gate.weight"]),
        np32(sd["gate.e_score_correction_bias"]),
        np32(sd["experts.up_proj"]),
        np32(sd["experts.down_proj"]),
        np32(sd["shared_experts.up_proj.weight"]),
        np32(sd["shared_experts.down_proj.weight"]),
        np32(hn.reshape(n, h)),
    ]
    cfg_json = dict(
        model_type="nemotron_h",
        hidden_size=h,
        intermediate_size=32,
        hybrid_override_pattern="E",
        num_hidden_layers=1,
        vocab_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_head_dim=8,
        mamba_num_heads=4,
        mamba_head_dim=4,
        ssm_state_size=4,
        n_groups=1,
        conv_kernel=4,
        n_routed_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=i,
        n_shared_experts=1,
        moe_shared_expert_intermediate_size=ish,
        n_group=2,
        topk_group=1,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=1.25,
        mlp_hidden_act="relu2",
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
    )
    mod = emit_probe(cfg_json, n, tag)
    out = run_probe(mod, args)
    return compare(
        f"{tag} (grouped sigmoid, top-{k}, norm={norm_topk_prob}, relu2+shared)",
        out,
        np32(ref),
    )


def compare(name, iree_out, hf_out):
    diff = float(np.max(np.abs(iree_out - hf_out)))
    rel = diff / (float(np.max(np.abs(hf_out))) + 1e-12)
    ok = diff < TOL
    print(
        f"[{name}] max|block diff|={diff:.3e} (rel {rel:.3e}) -> {'OK' if ok else 'MISMATCH'}",
        flush=True,
    )
    return ok


def main():
    global DEVICE, ROCM_TARGET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("local-task", "hip"),
        default=os.environ.get("MLXCEL_XLA_DEVICE", "local-task"),
    )
    parser.add_argument(
        "--rocm-target",
        default=os.environ.get("MLXCEL_XLA_ROCM_TARGET"),
        help="AMD architecture for --device hip, for example gfx1151",
    )
    args = parser.parse_args()
    DEVICE = args.device
    ROCM_TARGET = args.rocm_target
    ok = True
    ok = check_qwen2_moe() and ok
    ok = check_mixtral() and ok
    ok = check_qwen3_moe() and ok
    ok = check_olmoe() and ok
    ok = check_nemotron_h() and ok
    # HF normalizes whenever the flag is true, including top-k=1. The paired
    # unnormalized case prevents the probe from accidentally treating the flag as
    # a no-op for one selected expert (MLX eager currently differs here).
    ok = check_nemotron_h(1, True, "nemotron_h_top1_norm") and ok
    ok = check_nemotron_h(1, False, "nemotron_h_top1_raw") and ok
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
