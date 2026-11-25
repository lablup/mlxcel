"""HF fp32 oracle for single-layer Mamba2 and Mamba1 mixer probes.

Validates the mlxcel-xla SSM primitives (`emitter/ssm.rs`: causal conv1d and
selective scan) against a genuine HF `Mamba2Mixer`, `MambaMixer`, or
`FalconMambaMixer` slow path (no custom SSM or CUDA kernels), in fp32, with
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
MAMBA2_PRESETS = {
    # Small, but with G > 1 and G < H so the group broadcast is exercised.
    "tiny": dict(hidden_size=32, expand=2, head_dim=16, state_size=16, n_groups=2),
    # The state-spaces/mamba2-130m mixer dimensions.
    "mamba2-130m": dict(hidden_size=768, expand=2, head_dim=64, state_size=128, n_groups=1),
}
MAMBA1_PRESETS = {
    "tiny": dict(hidden_size=16, expand=2, state_size=4, time_step_rank=1),
    # state-spaces/mamba-130m-hf.
    "mamba1-130m": dict(hidden_size=768, expand=2, state_size=16, time_step_rank=48),
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


def build_mixer(kind, preset, conv_kernel, use_bias, seed, hf_chunk, time_step_limit=None):
    if kind in ("mamba1", "falcon_mamba"):
        if preset not in MAMBA1_PRESETS:
            raise ValueError(f"{preset} is not a Mamba1 preset")
        if kind == "falcon_mamba":
            from transformers import FalconMambaConfig
            from transformers.models.falcon_mamba.modeling_falcon_mamba import FalconMambaMixer

            config_cls, mixer_cls = FalconMambaConfig, FalconMambaMixer
        else:
            from transformers import MambaConfig
            from transformers.models.mamba.modeling_mamba import MambaMixer

            config_cls, mixer_cls = MambaConfig, MambaMixer
        dims = MAMBA1_PRESETS[preset]
        cfg = config_cls(
            conv_kernel=conv_kernel,
            use_bias=use_bias,
            use_conv_bias=True,
            num_hidden_layers=1,
            **dims,
        )
        torch.manual_seed(seed)
        # Transformers 5.12.1 initializes Falcon-Mamba's non-persistent RMS
        # buffers before registering them. Constructing without initialization
        # and then calling the same upstream initializer avoids that ordering
        # bug without changing the independent reference implementation.
        initialize = kind != "falcon_mamba"
        mixer = mixer_cls(cfg, layer_idx=0, initialize_mixer_weights=initialize).float().eval()
        if not initialize:
            mixer.init_falcon_mamba_weights()
        with torch.no_grad():
            mixer.D.copy_(1.0 + 0.5 * torch.randn_like(mixer.D))
            mixer.A_log.add_(0.1 * torch.randn_like(mixer.A_log))
            mixer.dt_proj.bias.add_(0.1 * torch.randn_like(mixer.dt_proj.bias))
            if mixer.conv1d.bias is not None:
                mixer.conv1d.bias.copy_(0.1 * torch.randn_like(mixer.conv1d.bias))
        return cfg, mixer

    from transformers import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

    if preset not in MAMBA2_PRESETS:
        raise ValueError(f"{preset} is not a Mamba2 preset")
    dims = MAMBA2_PRESETS[preset]
    d_inner = dims["expand"] * dims["hidden_size"]
    cfg = Mamba2Config(
        num_heads=d_inner // dims["head_dim"],
        conv_kernel=conv_kernel,
        use_bias=use_bias,
        use_conv_bias=True,
        chunk_size=hf_chunk,
        num_hidden_layers=1,
        **({"time_step_limit": tuple(time_step_limit)} if time_step_limit else {}),
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


def weights_of(mixer, use_bias, kind):
    """The probe's resident weights in `SsmProbeSpec::weight_layout` order."""
    out = [("in_proj", mixer.in_proj.weight)]
    if use_bias:
        out.append(("in_proj_b", mixer.in_proj.bias))
    out.append(("conv_w", mixer.conv1d.weight))  # [C, 1, K]
    out.append(("conv_b", mixer.conv1d.bias))
    if kind == "mamba2":
        out += [
            ("dt_bias", mixer.dt_bias),
            ("a_log", mixer.A_log),
            ("d", mixer.D),
            ("norm_w", mixer.norm.weight),
        ]
    else:
        out += [
            ("x_proj", mixer.x_proj.weight),
            ("dt_proj", mixer.dt_proj.weight),
            ("dt_proj_b", mixer.dt_proj.bias),
            ("a_log", mixer.A_log),
            ("d", mixer.D),
        ]
    out.append(("out_proj", mixer.out_proj.weight))
    if use_bias:
        out.append(("out_proj_b", mixer.out_proj.bias))
    return out


def np32(t):
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)


def reference_forward(mixer, rows, cache):
    forward = mixer.torch_forward if hasattr(mixer, "torch_forward") else mixer.slow_forward
    return forward(rows, cache_params=cache)


def decode_steps(mixer, rows, h0, conv0):
    """HF's decode branch over `rows [B,L,hidden]` from recurrent state and
    the last K-1 pre-convolution inputs `conv0 [B,K-1,C]`."""
    batch, _, channels = conv0.shape
    window = torch.cat([torch.zeros(batch, 1, channels), conv0], dim=1).transpose(1, 2)
    stub = StateStub(window.contiguous(), h0.clone(), previous=True)
    ys = []
    for t in range(rows.shape[1]):
        ys.append(reference_forward(mixer, rows[:, t : t + 1], stub))
    y = torch.cat(ys, dim=1) if ys else rows.new_zeros(batch, 0, mixer.hidden_size)
    return y, stub.layers[0].recurrent_states, stub.conv[:, :, 1:].transpose(1, 2)


def emit_inputs(args):
    os.makedirs(args.emit_inputs, exist_ok=True)
    hf_chunk = args.hf_chunk or (32 if args.dims == "mamba2-130m" else 256)
    cfg, mixer = build_mixer(
        args.kind,
        args.dims,
        args.conv_kernel,
        args.use_bias,
        args.seed,
        hf_chunk,
        args.time_step_limit,
    )
    mamba1 = args.kind != "mamba2"
    if mamba1:
        h, p = 1, mixer.intermediate_size
        c = mixer.intermediate_size
        state_shape = (p, mixer.ssm_state_size)
    else:
        h, p = mixer.num_heads, mixer.head_dim
        c = mixer.conv_dim
        state_shape = (h, p, mixer.ssm_state_size)
    n, k, hidden = mixer.ssm_state_size, mixer.conv_kernel_size, mixer.hidden_size
    gen = torch.Generator().manual_seed(args.seed + 1)

    with torch.no_grad():
        if args.mode == "decode":
            b = args.batch
            x = torch.randn(b, 1, hidden, generator=gen)
            h0 = 0.5 * torch.randn((b, *state_shape), generator=gen)
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
                h0 = 0.5 * torch.randn(state_shape, generator=gen)
                conv0 = torch.randn(k - 1, c, generator=gen)
                y, h_t, tail = decode_steps(mixer, x[None, :real_len], h0[None], conv0[None])
                y, h_t, tail = y[0], h_t[0], tail[0]
            else:
                h0, conv0 = torch.zeros(state_shape), torch.zeros(k - 1, c)
                if real_len == 0:
                    y, h_t, tail = torch.zeros(0, hidden), h0, conv0
                else:
                    stub = StateStub(None, None, previous=False)
                    y = reference_forward(mixer, x[None, :real_len], stub)[0]
                    h_t = stub.layers[0].recurrent_states[0]
                    tail = stub.conv[0, :, 1:].transpose(0, 1)

    for name, tensor in weights_of(mixer, args.use_bias, args.kind):
        np.save(os.path.join(args.emit_inputs, f"{name}.npy"), np32(tensor))
    for name, tensor in [("x", x), ("h0", h0), ("conv0", conv0)]:
        np.save(os.path.join(args.emit_inputs, f"{name}.npy"), np32(tensor))
    for name, tensor in [("expected_y", y), ("expected_h", h_t), ("expected_conv", tail)]:
        np.save(os.path.join(args.emit_inputs, f"{name}.npy"), np32(tensor))
    lo, hi = cfg.time_step_limit if not mamba1 else (0.0, float("inf"))
    norm_eps = mixer.rms_eps if args.kind == "falcon_mamba" else cfg.layer_norm_epsilon
    meta = dict(
        mode=args.mode,
        rows=rows,
        real_len=real_len,
        dims=args.dims,
        family=args.kind,
        kind="mamba1" if mamba1 else "mamba2",
        seed=args.seed,
        init_state=bool(args.init_state),
        hidden=hidden,
        n_heads=h,
        head_dim=p,
        state_size=n,
        n_groups=1 if mamba1 else mixer.n_groups,
        conv_kernel=k,
        use_conv_bias=True,
        use_bias=bool(args.use_bias),
        time_step_limit=[float(lo), None if hi == float("inf") else float(hi)],
        norm_eps=float(norm_eps),
        norm_group_size=mixer.intermediate_size,
        chunk_size=args.chunk_size,
        dt_rank=mixer.time_step_rank if mamba1 else 0,
        bcdt_norm=args.kind == "falcon_mamba",
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
    ap.add_argument(
        "--kind", choices=["mamba2", "mamba1", "falcon_mamba"], default="mamba2"
    )
    ap.add_argument(
        "--dims", choices=sorted(MAMBA1_PRESETS.keys() | MAMBA2_PRESETS.keys()), default="tiny"
    )
    ap.add_argument("--conv-kernel", type=int, default=4)
    ap.add_argument("--use-bias", action="store_true")
    ap.add_argument("--init-state", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=128, help="XLA prefill chunk")
    ap.add_argument("--hf-chunk", type=int, default=0, help="HF torch_forward chunk")
    ap.add_argument(
        "--time-step-limit",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="HF time_step_limit (dt clamp after softplus); default (0, inf)",
    )
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--json", metavar="FILE", help="write the --check report here")
    args = ap.parse_args()
    if bool(args.emit_inputs) == bool(args.check):
        ap.error("pass exactly one of --emit-inputs DIR or --check DIR")
    torch.set_grad_enabled(False)
    return emit_inputs(args) if args.emit_inputs else check(args)


if __name__ == "__main__":
    sys.exit(main())
