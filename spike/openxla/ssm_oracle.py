"""HF fp32 oracle for the single-layer Mamba2 mixer probe (issue #3337).

Validates the mlxcel-xla SSM primitives (`emitter/ssm.rs`: causal conv1d,
selective scan, gated norm) against ONE genuine HF `Mamba2Mixer` run through its
slow path `torch_forward` (no `mamba_ssm`, no CUDA kernels), in fp32, with
seeded random weights. Modeled on `moe_oracle.py`, but split into two steps so
the XLA side runs through the real IREE shim (`examples/xla_ssm_probe.rs`):

    $PY spike/openxla/ssm_oracle.py --emit-inputs DIR --mode prefill --real-len 100
    scripts/rocm_gpu_guard.sh --hold -- cargo run --release --features xla-iree \
        --example xla_ssm_probe -- --inputs DIR --device hip --out DIR
    $PY spike/openxla/ssm_oracle.py --check DIR

`--emit-inputs` writes the weights, input rows, starting state, and the HF
expected `y`, final `h` and conv tail as `.npy` plus `meta.json`. Prefill pads
the input to `--capacity` rows with random (not zero) rows past `real_len`, so
the check proves padding does not advance the state. HF prefill always starts
from a zero state, so prefill uses `h0 = conv0 = 0` unless `--init-state`, in
which case the reference is HF's decode path run token by token from that state.
Decode feeds a random `h0` / conv tail through a cache stub that exposes them
to HF's own decode branch. `--check` compares an XLA result directory (max-abs)
and exits non-zero above `--tol` (default 5e-5).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

TOL = 5e-5
PRESETS = {
    # Small, but with G > 1 and G < H so the group broadcast is exercised.
    "tiny": dict(hidden_size=32, expand=2, head_dim=16, state_size=16, n_groups=2),
    # The state-spaces/mamba2-130m mixer dimensions.
    "mamba2-130m": dict(hidden_size=768, expand=2, head_dim=64, state_size=128, n_groups=1),
}


class StateStub:
    """The slice of the HF `Cache` API `Mamba2Mixer.torch_forward` touches, for
    one layer. `previous=True` routes HF into its decode branch from the given
    state; `previous=False` lets prefill record its final state and conv window."""

    class _Layer:
        def __init__(self, recurrent):
            self.recurrent_states = recurrent
            self.device = torch.device("cpu")

    def __init__(self, conv, recurrent, previous):
        self.conv = conv  # [B, C, K]
        self.layers = [self._Layer(recurrent)]
        self.previous = previous

    def has_previous_state(self, layer_idx):
        return self.previous

    def update_conv_state(self, new, layer_idx):
        if self.previous:
            rolled = self.conv.roll(shifts=-new.shape[-1], dims=-1)
            rolled[:, :, -new.shape[-1] :] = new
            self.conv = rolled
        else:
            self.conv = new.clone()
        return self.conv

    def update_recurrent_state(self, state, layer_idx):
        self.layers[0].recurrent_states = state
        return state


def build_mixer(preset, conv_kernel, use_bias, seed, hf_chunk):
    from transformers import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

    dims = PRESETS[preset]
    d_inner = dims["expand"] * dims["hidden_size"]
    cfg = Mamba2Config(
        num_heads=d_inner // dims["head_dim"],
        conv_kernel=conv_kernel,
        use_bias=use_bias,
        use_conv_bias=True,
        chunk_size=hf_chunk,
        num_hidden_layers=1,
        **dims,
    )
    torch.manual_seed(seed)
    mixer = Mamba2Mixer(cfg, layer_idx=0).float().eval()
    with torch.no_grad():
        # Keep HF's A_log / dt_bias init (realistic decay rates); perturb the
        # rest so every term (D skip, norm weight, biases) is exercised.
        mixer.D.copy_(1.0 + 0.5 * torch.randn_like(mixer.D))
        mixer.norm.weight.copy_(1.0 + 0.1 * torch.randn_like(mixer.norm.weight))
        mixer.A_log.add_(0.1 * torch.randn_like(mixer.A_log))
        mixer.dt_bias.add_(0.1 * torch.randn_like(mixer.dt_bias))
        if mixer.conv1d.bias is not None:
            mixer.conv1d.bias.copy_(0.1 * torch.randn_like(mixer.conv1d.bias))
    return cfg, mixer


def weights_of(mixer, use_bias):
    """The probe's resident weights in `SsmProbeSpec::weight_layout` order."""
    out = [("in_proj", mixer.in_proj.weight)]
    if use_bias:
        out.append(("in_proj_b", mixer.in_proj.bias))
    out.append(("conv_w", mixer.conv1d.weight))  # [C, 1, K]
    out.append(("conv_b", mixer.conv1d.bias))
    out += [
        ("dt_bias", mixer.dt_bias),
        ("a_log", mixer.A_log),
        ("d", mixer.D),
        ("norm_w", mixer.norm.weight),
        ("out_proj", mixer.out_proj.weight),
    ]
    if use_bias:
        out.append(("out_proj_b", mixer.out_proj.bias))
    return out


def np32(t):
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)


def decode_steps(mixer, rows, h0, conv0):
    """HF's own decode branch over `rows [B, L, hidden]`, from `h0 [B,H,P,N]`
    and the last K-1 pre-conv inputs `conv0 [B, K-1, C]`."""
    batch, _, channels = conv0.shape
    window = torch.cat([torch.zeros(batch, 1, channels), conv0], dim=1).transpose(1, 2)
    stub = StateStub(window.contiguous(), h0.clone(), previous=True)
    ys = []
    for t in range(rows.shape[1]):
        ys.append(mixer.torch_forward(rows[:, t : t + 1], cache_params=stub))
    y = torch.cat(ys, dim=1) if ys else rows.new_zeros(batch, 0, mixer.hidden_size)
    return y, stub.layers[0].recurrent_states, stub.conv[:, :, 1:].transpose(1, 2)


def emit_inputs(args):
    os.makedirs(args.emit_inputs, exist_ok=True)
    hf_chunk = args.hf_chunk or (32 if args.dims == "mamba2-130m" else 256)
    cfg, mixer = build_mixer(args.dims, args.conv_kernel, args.use_bias, args.seed, hf_chunk)
    h, p, n = mixer.num_heads, mixer.head_dim, mixer.ssm_state_size
    c, k, hidden = mixer.conv_dim, mixer.conv_kernel_size, mixer.hidden_size
    gen = torch.Generator().manual_seed(args.seed + 1)

    with torch.no_grad():
        if args.mode == "decode":
            b = args.batch
            x = torch.randn(b, 1, hidden, generator=gen)
            h0 = 0.5 * torch.randn(b, h, p, n, generator=gen)
            conv0 = torch.randn(b, k - 1, c, generator=gen)
            y, h_t, tail = decode_steps(mixer, x, h0, conv0)
            x, y = x[:, 0], y[:, 0]
            rows, real_len = b, 1
        else:
            rows, real_len = args.capacity, args.real_len
            if not 0 <= real_len <= rows:
                sys.exit(f"--real-len must be in 0..={rows}")
            x = torch.randn(rows, hidden, generator=gen)
            if args.init_state:
                h0 = 0.5 * torch.randn(h, p, n, generator=gen)
                conv0 = torch.randn(k - 1, c, generator=gen)
                y, h_t, tail = decode_steps(mixer, x[None, :real_len], h0[None], conv0[None])
                y, h_t, tail = y[0], h_t[0], tail[0]
            else:
                h0, conv0 = torch.zeros(h, p, n), torch.zeros(k - 1, c)
                if real_len == 0:
                    y, h_t, tail = torch.zeros(0, hidden), h0, conv0
                else:
                    stub = StateStub(None, None, previous=False)
                    y = mixer.torch_forward(x[None, :real_len], cache_params=stub)[0]
                    h_t = stub.layers[0].recurrent_states[0]
                    tail = stub.conv[0, :, 1:].transpose(0, 1)

    for name, tensor in weights_of(mixer, args.use_bias):
        np.save(os.path.join(args.emit_inputs, f"{name}.npy"), np32(tensor))
    for name, tensor in [("x", x), ("h0", h0), ("conv0", conv0)]:
        np.save(os.path.join(args.emit_inputs, f"{name}.npy"), np32(tensor))
    for name, tensor in [("expected_y", y), ("expected_h", h_t), ("expected_conv", tail)]:
        np.save(os.path.join(args.emit_inputs, f"{name}.npy"), np32(tensor))
    lo, hi = cfg.time_step_limit
    meta = dict(
        mode=args.mode,
        rows=rows,
        real_len=real_len,
        dims=args.dims,
        seed=args.seed,
        init_state=bool(args.init_state),
        hidden=hidden,
        n_heads=h,
        head_dim=p,
        state_size=n,
        n_groups=mixer.n_groups,
        conv_kernel=k,
        use_conv_bias=True,
        use_bias=bool(args.use_bias),
        time_step_limit=[float(lo), None if hi == float("inf") else float(hi)],
        norm_eps=float(mixer.layer_norm_epsilon),
        norm_group_size=mixer.intermediate_size,
        chunk_size=args.chunk_size,
    )
    with open(os.path.join(args.emit_inputs, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(
        f"[emit] {args.mode} dims={args.dims} rows={rows} real_len={real_len} "
        f"init_state={bool(args.init_state)} -> {args.emit_inputs}",
        flush=True,
    )
    return 0


def check(args):
    out = args.check
    run = {}
    run_path = os.path.join(out, "run.json")
    if os.path.exists(run_path):
        with open(run_path) as fh:
            run = json.load(fh)
    src = out if os.path.exists(os.path.join(out, "meta.json")) else run.get("inputs", out)
    with open(os.path.join(src, "meta.json")) as fh:
        meta = json.load(fh)
    live = meta["real_len"] if meta["mode"] == "prefill" else meta["rows"]
    pairs = [("y", "expected_y"), ("h", "expected_h"), ("conv", "expected_conv")]
    report, ok = {}, True
    for got_name, want_name in pairs:
        got = np.load(os.path.join(out, f"{got_name}.npy"))
        want = np.load(os.path.join(src, f"{want_name}.npy"))
        if got_name == "y":
            got = got[:live]
        if got.shape != want.shape:
            print(f"[{got_name}] shape {got.shape} != expected {want.shape}", flush=True)
            return 1
        finite = bool(np.all(np.isfinite(got)))
        diff = float(np.max(np.abs(got - want))) if got.size else 0.0
        scale = float(np.max(np.abs(want))) if want.size else 0.0
        passed = finite and diff <= args.tol
        ok = ok and passed
        report[got_name] = dict(max_abs=diff, ref_max_abs=scale, finite=finite)
        print(
            f"[{got_name}] max|diff|={diff:.3e} (ref max {scale:.3e}) -> "
            f"{'OK' if passed else 'MISMATCH'}",
            flush=True,
        )
    report.update(
        mode=meta["mode"],
        rows=meta["rows"],
        real_len=meta["real_len"],
        device=run.get("device"),
        form=run.get("form"),
        tol=args.tol,
        passed=ok,
    )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emit-inputs", metavar="DIR")
    ap.add_argument("--check", metavar="DIR")
    ap.add_argument("--mode", choices=["decode", "prefill"], default="prefill")
    ap.add_argument("--real-len", type=int, default=0)
    ap.add_argument("--capacity", type=int, default=256, help="padded prefill rows")
    ap.add_argument("--batch", type=int, default=1, help="decode rows")
    ap.add_argument("--dims", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--conv-kernel", type=int, default=4)
    ap.add_argument("--use-bias", action="store_true")
    ap.add_argument("--init-state", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=128, help="XLA prefill chunk")
    ap.add_argument("--hf-chunk", type=int, default=0, help="HF torch_forward chunk")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--json", metavar="FILE", help="write the --check report here")
    args = ap.parse_args()
    if bool(args.emit_inputs) == bool(args.check):
        ap.error("pass exactly one of --emit-inputs DIR or --check DIR")
    torch.set_grad_enabled(False)
    return emit_inputs(args) if args.emit_inputs else check(args)


if __name__ == "__main__":
    sys.exit(main())
