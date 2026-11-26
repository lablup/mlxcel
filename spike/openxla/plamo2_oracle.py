"""Independent pure-PyTorch PLaMo 2 reference and XLA fixtures (issue #3347).

The HF remote code (`modeling_plamo.py`) routes its CPU scan and conv through
`mamba_ssm` / `causal_conv1d` reference functions, which need a CUDA build to
install. This script re-implements the HF forward in float64 from the HF
formulas (cited inline) without importing either package, then writes tiny
synthetic checkpoints under literal HF tensor names together with the greedy
tokens and per-step logits the Rust gates in `tests/xla_plamo2.rs` compare.

Two fixture families:

* `hybrid/`: Mamba, local attention, Mamba, full attention. Short prompts that
  cross the SSD chunk and a small attention window, distinct norm weights and
  offsets, head-specific Q/K scales, and dt-clamp-active heads. The scan uses
  MLX eager's `clamp(softplus(dt + dt_bias), 0.001, 100)` (HF does not clamp).
* `window/`: the 2048-token sliding window. An attention-only PLaMo 2
  (`mamba_enabled: false`) with one local and one full layer, prompts of
  2047..2051 tokens, and decode steps that move the boundary key across age
  2048. HF keeps keys with `age <= attention_window_size` (`swa_mask`,
  `triu(diagonal=-window)`), so the reference mask is inclusive. The same
  weights are also written as `window_minus/` and `window_plus/` controls
  (`attention_window_size` 2047 and 2049), and the reference records the
  trajectories of the exclusive (`age < W`), one-wider (`age <= W + 1`) and
  unwindowed masks so the gate proves it can tell them apart.

Run with the shared oracle venv (torch 2.12.1+cpu):
    python spike/openxla/plamo2_oracle.py --out /tmp/plamo2-fixtures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

torch.set_default_dtype(torch.float64)
DT_LIMIT = (0.001, 100.0)


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    # HF `_rms_norm`: x * rsqrt(mean(x^2) + eps), weight applied by the caller.
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def rope(x: torch.Tensor, base: float, positions: torch.Tensor) -> torch.Tensor:
    # HF `RotaryEmbedding` + `_rotary_pos_emb` (split-half, non-interleaved).
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2) / d))
    freqs = positions[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return x * emb.cos()[:, None, :] + rotate_half(x) * emb.sin()[:, None, :]


def is_mamba(cfg: dict, i: int) -> bool:
    # HF `is_mamba`.
    if not cfg.get("mamba_enabled", True):
        return False
    step = cfg.get("mamba_step", 2)
    n = cfg["num_hidden_layers"]
    if n <= step // 2:
        return i != n - 1
    return i % step != step // 2


class Plamo2:
    """float64 PLaMo 2 forward over one token sequence (no cache)."""

    def __init__(self, cfg: dict, w: dict[str, torch.Tensor]):
        self.cfg, self.w = cfg, w
        self.eps = cfg["rms_norm_eps"]
        self.d = cfg["hidden_size_per_head"]
        self.dt_dim = max(64, cfg["hidden_size"] // 16)

    def norm(self, x, name, offset):
        # HF `RMSNorm(offset=...)`: (offset + weight) * rmsnorm(x).
        return (offset + self.w[name]) * rms(x, self.eps)

    def attention(self, li: int, x: torch.Tensor, mask_mode: str) -> torch.Tensor:
        c, p, d = self.cfg, f"model.layers.layers.{li}.mixer.", self.d
        nq, nkv, t = c["num_attention_heads"], c["num_key_value_heads"], x.shape[0]
        qkv = x @ self.w[p + "qkv_proj.weight"].T
        q, k, v = torch.split(qkv, [nq * d, nkv * d, nkv * d], dim=-1)
        q, k, v = q.view(t, nq, d), k.view(t, nkv, d), v.view(t, nkv, d)
        # HF: _rms_norm(., None, 1e-6) * q_weight[None, :, None]
        q = rms(q, 1e-6) * self.w[p + "q_weight"][None]
        k = rms(k, 1e-6) * self.w[p + "k_weight"][None]
        full = li in c.get("full_attention_idx", [])
        base = c.get("rope_theta", 10000) if full else c.get("rope_local_theta", 10000)
        pos = torch.arange(t, dtype=torch.float64)
        q, k = rope(q, base, pos), rope(k, base, pos)
        k = k.repeat_interleave(nq // nkv, dim=1)
        v = v.repeat_interleave(nq // nkv, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(d)
        age = pos[:, None] - pos[None, :]
        keep = age >= 0
        window = c.get("attention_window_size", 2048)
        if not full and mask_mode != "none":
            bound = {"inclusive": window, "exclusive": window - 1, "plus": window + 1}[mask_mode]
            keep = keep & (age <= bound)
        scores = scores.masked_fill(~keep[None], float("-inf"))
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(t, nq * d)
        return out @ self.w[p + "o_proj.weight"].T

    def mamba(self, li: int, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        c, p = self.cfg, f"model.layers.layers.{li}.mixer."
        h_n, d, n, k = c["mamba_num_heads"], self.d, c["mamba_d_state"], c["mamba_d_conv"]
        t, di = x.shape[0], h_n * d
        zx = (x @ self.w[p + "in_proj.weight"].T).view(t, h_n, 2 * d)
        z, xs = zx[..., :d].reshape(t, di), zx[..., d:].reshape(t, di)
        cw = self.w[p + "conv1d.weight"][:, 0, :]  # [di, K]
        xp = torch.cat([torch.zeros(k - 1, di), xs], 0)
        conv = sum(cw[:, j][None] * xp[j : j + t] for j in range(k))
        xs = conv * torch.sigmoid(conv)  # causal_conv1d(activation="silu")
        bcdt = xs @ self.w[p + "bcdt_proj.weight"].T
        bm, cm, dt = torch.split(bcdt, [n, n, self.dt_dim], dim=-1)
        dt = rms(dt, self.eps) * self.w[p + "dt_norm_weight"]
        bm = rms(bm, self.eps) * self.w[p + "B_norm_weight"]
        cm = rms(cm, self.eps) * self.w[p + "C_norm_weight"]
        dt = torch.nn.functional.softplus(dt @ self.w[p + "dt_proj.weight"].T + self.w[p + "dt_bias"])
        if clamp:
            dt = dt.clamp(*DT_LIMIT)
        a = -torch.exp(self.w[p + "A_log"])
        xh = xs.view(t, h_n, d)
        state = torch.zeros(h_n, d, n)
        ys = []
        for i in range(t):
            decay = torch.exp(dt[i] * a)[:, None, None]
            state = state * decay + (dt[i][:, None] * xh[i])[:, :, None] * bm[i][None, None, :]
            ys.append(state @ cm[i] + self.w[p + "D"][:, None] * xh[i])
        y = torch.stack(ys).reshape(t, di)
        return (y * z * torch.sigmoid(z)) @ self.w[p + "out_proj.weight"].T

    def logits(self, tokens: list[int], mask_mode: str = "inclusive", clamp: bool = True):
        c = self.cfg
        x = self.w["model.embed_tokens.weight"][torch.tensor(tokens)]
        for li in range(c["num_hidden_layers"]):
            p = f"model.layers.layers.{li}."
            h = self.norm(x, p + "pre_mixer_norm.weight", 1.0)
            if is_mamba(c, li):
                h = self.mamba(li, h, clamp)
            else:
                h = self.attention(li, h, mask_mode)
            x = x + self.norm(h, p + "post_mixer_norm.weight", 1.0 / 5)
            h = self.norm(x, p + "pre_mlp_norm.weight", 1.0)
            gu = h @ self.w[p + "mlp.gate_up_proj.weight"].T
            g, u = gu.chunk(2, dim=-1)
            h = (g * torch.sigmoid(g) * u) @ self.w[p + "mlp.down_proj.weight"].T
            x = x + self.norm(h, p + "post_mlp_norm.weight", 1.0 / (5**1.5))
        x = self.norm(x[-1], "model.norm.weight", 1.0)
        return x @ self.w["model.embed_tokens.weight"].T

    def greedy(self, prompt: list[int], steps: int, **kw):
        tokens, logits = list(prompt), []
        for _ in range(steps):
            row = self.logits(tokens, **kw)
            logits.append(row)
            tokens.append(int(row.argmax()))
        return tokens[len(prompt) :], logits


def random_weights(cfg: dict, gen: torch.Generator) -> dict[str, torch.Tensor]:
    def r(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen) * scale

    def u(*shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=gen)

    hid, d, inter = cfg["hidden_size"], cfg["hidden_size_per_head"], cfg["intermediate_size"]
    nq, nkv, v = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["vocab_size"]
    # A small embedding keeps the tied head from echoing the last token, so
    # greedy trajectories vary instead of repeating one id.
    w = {"model.embed_tokens.weight": r(v, hid, scale=0.1), "model.norm.weight": r(hid, scale=0.3)}
    for li in range(cfg["num_hidden_layers"]):
        p = f"model.layers.layers.{li}."
        # Distinct, non-trivial weights around the offsets (zeros would hide a
        # missing offset; ones would hide a missing weight).
        for name in ["pre_mixer_norm", "post_mixer_norm", "pre_mlp_norm", "post_mlp_norm"]:
            w[p + name + ".weight"] = r(hid, scale=0.3)
        w[p + "mlp.gate_up_proj.weight"] = r(2 * inter, hid, scale=hid**-0.5)
        w[p + "mlp.down_proj.weight"] = r(hid, inter, scale=inter**-0.5)
        m = p + "mixer."
        if is_mamba(cfg, li):
            hn, n = cfg["mamba_num_heads"], cfg["mamba_d_state"]
            di, rk, k = hn * d, max(64, hid // 16), cfg["mamba_d_conv"]
            w[m + "in_proj.weight"] = r(2 * di, hid, scale=hid**-0.5)
            w[m + "conv1d.weight"] = r(di, 1, k, scale=0.5)
            w[m + "bcdt_proj.weight"] = r(2 * n + rk, di, scale=di**-0.5)
            w[m + "dt_proj.weight"] = r(hn, rk, scale=rk**-0.5)
            w[m + "dt_norm_weight"] = u(rk, lo=0.5, hi=1.5)
            w[m + "B_norm_weight"] = u(n, lo=0.5, hi=1.5)
            w[m + "C_norm_weight"] = u(n, lo=0.5, hi=1.5)
            w[m + "A_log"] = u(hn, lo=0.5, hi=3.0)
            w[m + "D"] = r(hn, scale=0.5)
            # Half the heads sit far below the 0.001 floor, so the clamp is live.
            bias = u(hn, lo=-1.0, hi=1.0)
            bias[::2] = -9.0
            w[m + "dt_bias"] = bias
            w[m + "out_proj.weight"] = r(hid, di, scale=di**-0.5)
        else:
            w[m + "qkv_proj.weight"] = r((nq + 2 * nkv) * d, hid, scale=hid**-0.5)
            w[m + "o_proj.weight"] = r(hid, nq * d, scale=(nq * d) ** -0.5)
            w[m + "q_weight"] = u(nq, d, lo=0.5, hi=2.0)
            w[m + "k_weight"] = u(nkv, d, lo=0.5, hi=2.0)
    return w


def write_fixture(out: Path, cfg: dict, w: dict, cases: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    save_file({k: v.to(torch.float32).contiguous() for k, v in w.items()}, str(out / "model.safetensors"))
    (out / "oracle.json").write_text(json.dumps(cases) + "\n")


def as_list(rows):
    return [row.to(torch.float32).tolist() for row in rows]


def hybrid(out: Path, gen: torch.Generator) -> None:
    cfg = {
        "model_type": "plamo2",
        "architectures": ["Plamo2ForCausalLM"],
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 4,
        "vocab_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size_per_head": 8,
        "mamba_num_heads": 4,
        "mamba_d_state": 8,
        "mamba_d_conv": 4,
        "mamba_step": 2,
        "mamba_chunk_size": 8,
        "attention_window_size": 6,
        "full_attention_idx": [3],
        "rope_theta": 10000,
        "rope_local_theta": 5000,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    }
    w = random_weights(cfg, gen)
    model = Plamo2(cfg, w)
    cases = []
    for length in [3, 9, 17, 12]:
        prompt = torch.randint(0, cfg["vocab_size"], (length,), generator=gen).tolist()
        generated, logits = model.greedy(prompt, 8)
        # Diagnostics: how far the dt clamp and the window move this case.
        _, unclamped = model.greedy(prompt, 1, clamp=False)
        _, unwindowed = model.greedy(prompt, 1, mask_mode="none")
        cases.append(
            {
                "prompt": prompt,
                "generated": generated,
                "logits": as_list(logits),
                "clamp_effect": float((unclamped[0] - logits[0]).abs().max()),
                "window_effect": float((unwindowed[0] - logits[0]).abs().max()),
            }
        )
        print(f"hybrid len={length} tokens={generated} clamp_effect={cases[-1]['clamp_effect']:.3e} "
              f"window_effect={cases[-1]['window_effect']:.3e}", flush=True)
    write_fixture(out / "hybrid", cfg, w, cases)


WINDOW = 2048
TOKEN_A, TOKEN_B, TOKEN_C = 5, 7, 9


def window(out: Path, seed: int, steps: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    hid = 32
    cfg = {
        "model_type": "plamo2",
        "architectures": ["Plamo2ForCausalLM"],
        "hidden_size": hid,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size_per_head": 8,
        "mamba_enabled": False,
        "attention_window_size": WINDOW,
        "full_attention_idx": [1],
        "rope_theta": 10000,
        "rope_local_theta": 5000,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    }
    w = random_weights(cfg, gen)
    # Make the window boundary observable: token A's embedding lives on the
    # first half of the hidden lanes and B / C on the second half, and layer
    # 0's V projection reads A's lanes with a large gain. A key at the boundary
    # age then moves the local layer's output by far more than f32 noise, so an
    # off-by-one window changes logits (and usually tokens).
    emb = w["model.embed_tokens.weight"]
    half = hid // 2
    emb[TOKEN_A, half:] = 0.0
    emb[TOKEN_A, :half] = emb[TOKEN_A, :half].abs() + 0.05
    for tok in (TOKEN_B, TOKEN_C):
        emb[tok, :half] = 0.0
    nq, nkv, d = 4, 2, 8
    qkv = w["model.layers.layers.0.mixer.qkv_proj.weight"]
    v_rows = slice((nq + nkv) * d, (nq + 2 * nkv) * d)
    direction = torch.randn(nkv * d, generator=gen)
    qkv[v_rows, :half] += 1500.0 * direction[:, None] / half
    model = Plamo2(cfg, w)
    modes = ["inclusive", "exclusive", "plus", "none"]
    cases = []
    for length in [2047, 2048, 2049, 2050, 2051]:
        a_pos = max(0, length - 1 - WINDOW)
        prompt = [TOKEN_C] * a_pos + [TOKEN_A] + [TOKEN_B] * (length - 1 - a_pos)
        case = {"prompt": prompt[:0], "prompt_len": length, "a_pos": a_pos}
        for mode in modes:
            generated, logits = model.greedy(prompt, steps, mask_mode=mode)
            case[mode] = {"generated": generated, "logits": as_list(logits)}
        ref = torch.tensor(case["inclusive"]["logits"])
        top2 = ref.topk(2, dim=-1).values
        case["min_margin"] = float((top2[:, 0] - top2[:, 1]).min())
        diffs = {m: float((torch.tensor(case[m]["logits"]) - ref).abs().max(-1).values.max()) for m in modes[1:]}
        print(f"window len={length} A@{a_pos} inclusive={case['inclusive']['generated']} "
              f"exclusive={case['exclusive']['generated']} plus={case['plus']['generated']} "
              f"none={case['none']['generated']} max|variant-inclusive|={diffs} "
              f"min top-2 margin={case['min_margin']:.3e}", flush=True)
        case["prompt"] = prompt
        cases.append(case)
    write_fixture(out / "window", cfg, w, cases)
    for name, size in [("window_minus", WINDOW - 1), ("window_plus", WINDOW + 1)]:
        control = dict(cfg, attention_window_size=size)
        write_fixture(out / name, control, w, [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3347)
    parser.add_argument("--window-steps", type=int, default=6)
    # Seed 2 is the first whose greedy tokens tell every mask variant apart
    # (searched over seeds 1-3 with the V gain below); later seeds may not.
    parser.add_argument("--window-seed", type=int, default=2)
    parser.add_argument("--only", choices=["hybrid", "window"])
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(args.seed)
    if args.only != "window":
        hybrid(args.out, gen)
    if args.only != "hybrid":
        window(args.out, args.window_seed, args.window_steps)


if __name__ == "__main__":
    main()
