"""JAX Llama-3.2-1B forward for static-shape StableHLO export.

Two graphs drive the autoregressive loop on a graph compiler:

  prefill(params, tokens[Lp], positions[Lp], real_len)
      -> (last_logits[V], kcache, vcache)
  decode_step(params, token, pos, cache_len, kcache, vcache)
      -> (logits[V], kcache, vcache)

The KV cache is a pair of fixed-capacity tensors [L, MAX_SEQ, n_kv, d]. The host
drives the token loop and passes the cache back in each step (donated buffers).
Shapes are fully static: prefill is bucketed to a padded prompt length Lp, and
decode-step is single-token. RoPE (llama3 scaling), the causal/padding mask, and
GQA head repeat are implemented to match HF transformers numerically.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open


@dataclass(frozen=True)
class Config:
    hidden: int
    inter: int
    n_layers: int
    n_q: int
    n_kv: int
    head_dim: int
    eps: float
    rope_theta: float
    vocab: int
    # llama3 rope scaling
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    orig_ctx: int

    @property
    def group(self) -> int:
        return self.n_q // self.n_kv

    @property
    def scale(self) -> float:
        return self.head_dim ** -0.5


def load_config(model_dir: str) -> Config:
    c = json.load(open(f"{model_dir}/config.json"))
    rs = c["rope_scaling"]
    return Config(
        hidden=c["hidden_size"],
        inter=c["intermediate_size"],
        n_layers=c["num_hidden_layers"],
        n_q=c["num_attention_heads"],
        n_kv=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        eps=c["rms_norm_eps"],
        rope_theta=c["rope_theta"],
        vocab=c["vocab_size"],
        factor=rs["factor"],
        low_freq_factor=rs["low_freq_factor"],
        high_freq_factor=rs["high_freq_factor"],
        orig_ctx=rs["original_max_position_embeddings"],
    )


def load_params(model_dir: str, cfg: Config) -> dict:
    """Load bf16 safetensors, upcast to fp32 jnp arrays. Weights stay [out, in]
    so a linear is ``x @ W.T``. Embedding is tied (logits use embed.T)."""
    raw = {}
    with safe_open(f"{model_dir}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k).float().numpy()  # bf16 -> fp32 numpy

    def g(name):
        return jnp.asarray(raw[name], dtype=jnp.float32)

    params = {
        "embed": g("model.embed_tokens.weight"),
        "final_norm": g("model.norm.weight"),
        "layers": [],
    }
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        params["layers"].append({
            "in_ln": g(p + "input_layernorm.weight"),
            "post_ln": g(p + "post_attention_layernorm.weight"),
            "wq": g(p + "self_attn.q_proj.weight"),
            "wk": g(p + "self_attn.k_proj.weight"),
            "wv": g(p + "self_attn.v_proj.weight"),
            "wo": g(p + "self_attn.o_proj.weight"),
            "gate": g(p + "mlp.gate_proj.weight"),
            "up": g(p + "mlp.up_proj.weight"),
            "down": g(p + "mlp.down_proj.weight"),
        })
    return params


# --- RoPE: llama3 scaling, byte-for-byte with HF _compute_llama3_parameters ---
def llama3_inv_freq(cfg: Config) -> np.ndarray:
    base = 1.0 / (cfg.rope_theta ** (np.arange(0, cfg.head_dim, 2, dtype=np.float64) / cfg.head_dim))
    low_wl = cfg.orig_ctx / cfg.low_freq_factor
    high_wl = cfg.orig_ctx / cfg.high_freq_factor
    wavelen = 2 * math.pi / base
    inv = np.where(wavelen > low_wl, base / cfg.factor, base)
    smooth = (cfg.orig_ctx / wavelen - cfg.low_freq_factor) / (cfg.high_freq_factor - cfg.low_freq_factor)
    smoothed = (1 - smooth) * inv / cfg.factor + smooth * inv
    is_medium = ~(wavelen < high_wl) & ~(wavelen > low_wl)
    inv = np.where(is_medium, smoothed, inv)
    return inv  # [head_dim/2]


def rope_tables(cfg: Config, max_seq: int):
    """Precompute cos/sin tables [max_seq, head_dim] (fp32 constants)."""
    inv = llama3_inv_freq(cfg)                       # [d/2]
    pos = np.arange(max_seq, dtype=np.float64)       # [S]
    freqs = np.outer(pos, inv)                       # [S, d/2]
    emb = np.concatenate([freqs, freqs], axis=-1)    # [S, d]
    return jnp.asarray(np.cos(emb), jnp.float32), jnp.asarray(np.sin(emb), jnp.float32)


def _rotate_half(x):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rope(x, cos, sin):
    # x: [T, heads, d]; cos/sin: [T, d] -> broadcast over head axis
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return x * cos + _rotate_half(x) * sin


def _rms_norm(x, w, eps):
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(var + eps)) * w


def _silu(x):
    return x * jax.nn.sigmoid(x)


def _mlp(x, lp):
    return (_silu(x @ lp["gate"].T) * (x @ lp["up"].T)) @ lp["down"].T


def make_fns(cfg: Config, max_seq: int):
    cos_t, sin_t = rope_tables(cfg, max_seq)
    g, nq, nkv, d, H = cfg.group, cfg.n_q, cfg.n_kv, cfg.head_dim, cfg.hidden
    NEG = jnp.float32(-1e30)

    def prefill(params, tokens, positions, real_len):
        Lp = tokens.shape[0]
        x = params["embed"][tokens]                              # [Lp, H]
        cos = cos_t[positions]                                   # [Lp, d]
        sin = sin_t[positions]
        # causal mask [Lp, Lp]: query i attends key j iff j <= i
        i = jnp.arange(Lp)[:, None]
        j = jnp.arange(Lp)[None, :]
        cmask = jnp.where(j <= i, 0.0, NEG).astype(jnp.float32)  # [Lp, Lp]

        kcache = jnp.zeros((cfg.n_layers, max_seq, nkv, d), jnp.float32)
        vcache = jnp.zeros((cfg.n_layers, max_seq, nkv, d), jnp.float32)

        for li, lp in enumerate(params["layers"]):
            h = _rms_norm(x, lp["in_ln"], cfg.eps)
            q = (h @ lp["wq"].T).reshape(Lp, nq, d)
            k = (h @ lp["wk"].T).reshape(Lp, nkv, d)
            v = (h @ lp["wv"].T).reshape(Lp, nkv, d)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            kcache = kcache.at[li, :Lp].set(k)
            vcache = vcache.at[li, :Lp].set(v)
            k_rep = jnp.repeat(k, g, axis=1)                     # [Lp, nq, d]
            v_rep = jnp.repeat(v, g, axis=1)
            scores = jnp.einsum("ihd,jhd->hij", q, k_rep) * cfg.scale
            scores = scores + cmask[None]
            attn = jax.nn.softmax(scores, axis=-1)
            o = jnp.einsum("hij,jhd->ihd", attn, v_rep).reshape(Lp, nq * d)
            x = x + o @ lp["wo"].T
            x = x + _mlp(_rms_norm(x, lp["post_ln"], cfg.eps), lp)

        xf = _rms_norm(x, params["final_norm"], cfg.eps)
        last = jax.lax.dynamic_slice_in_dim(xf, real_len - 1, 1, axis=0)[0]  # [H]
        logits = last @ params["embed"].T                        # [V]
        return logits, kcache, vcache

    def decode_step(params, token, pos, cache_len, kcache, vcache):
        x = params["embed"][token]                               # [H]
        cos = cos_t[pos][None]                                   # [1, d]
        sin = sin_t[pos][None]
        valid = (jnp.arange(max_seq) <= cache_len)               # [S] keys 0..cache_len
        kmask = jnp.where(valid, 0.0, NEG).astype(jnp.float32)

        for li, lp in enumerate(params["layers"]):
            h = _rms_norm(x, lp["in_ln"], cfg.eps)
            q = (h @ lp["wq"].T).reshape(1, nq, d)
            k = (h @ lp["wk"].T).reshape(1, nkv, d)
            v = (h @ lp["wv"].T).reshape(1, nkv, d)
            q = _apply_rope(q, cos, sin)[0]                       # [nq, d]
            k = _apply_rope(k, cos, sin)[0]                       # [nkv, d]
            v = v[0]
            kcache = kcache.at[li, cache_len].set(k)  # dynamic_update_slice at scalar idx
            vcache = vcache.at[li, cache_len].set(v)
            kl = kcache[li]                                       # [S, nkv, d]
            vl = vcache[li]
            k_rep = jnp.repeat(kl, g, axis=1)                    # [S, nq, d]
            v_rep = jnp.repeat(vl, g, axis=1)
            scores = jnp.einsum("hd,shd->hs", q, k_rep) * cfg.scale  # [nq, S]
            scores = scores + kmask[None]
            attn = jax.nn.softmax(scores, axis=-1)
            o = jnp.einsum("hs,shd->hd", attn, v_rep).reshape(nq * d)
            x = x + o @ lp["wo"].T
            x = x + _mlp(_rms_norm(x, lp["post_ln"], cfg.eps), lp)

        xf = _rms_norm(x, params["final_norm"], cfg.eps)
        logits = xf @ params["embed"].T                          # [V]
        return logits, kcache, vcache

    return prefill, decode_step
