"""int4 dequant-in-graph variant of the Llama-3.2-1B decode/prefill graphs.

The 7 per-layer linear weights (q,k,v,o,gate,up,down) are stored as packed int4
plus per-group scale and min. Each weight is unpacked (shift + mask), converted
to fp32, and dequantized (codes * scale + min) inside the graph, immediately
before its `dot_general`. Embedding and norms stay fp32. This is the
correctness-first int4 route; the spike characterizes whether XLA/IREE fuse the
unpack/convert/scale into the matmul or materialize a full fp32 weight first.

Reuses the structural pieces (RoPE, RMSNorm, mask, GQA) from model_jax so the
only change versus the fp32 graph is the weight source.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import model_jax as M
import quant as Q

LINEARS = ("wq", "wk", "wv", "wo", "gate", "up", "down")


def quantize_params(params, group=Q.GROUP):
    """fp32 params -> int4 params: quantize the 7 linears, keep embed/norms fp32.
    Each quantized weight becomes (packed, scale, wmin) as jnp arrays."""
    import numpy as np
    qp = {"embed": params["embed"], "final_norm": params["final_norm"], "layers": []}
    for lp in params["layers"]:
        ql = {"in_ln": lp["in_ln"], "post_ln": lp["post_ln"]}
        for name in LINEARS:
            W = np.asarray(lp[name], dtype=np.float32)
            packed, scale, wmin = Q.quantize_affine_int4(W, group)
            ql[name] = (jnp.asarray(packed), jnp.asarray(scale), jnp.asarray(wmin))
        qp["layers"].append(ql)
    return qp


def dequantize_params(qp, group=Q.GROUP):
    """int4 params -> fp32 params (the dequant-then-fp32 baseline; reuses the
    plain model_jax graph). Mirrors the in-graph dequant exactly on the host."""
    fp = {"embed": qp["embed"], "final_norm": qp["final_norm"], "layers": []}
    for ql in qp["layers"]:
        lp = {"in_ln": ql["in_ln"], "post_ln": ql["post_ln"]}
        for name in LINEARS:
            packed, scale, wmin = ql[name]
            lp[name] = jnp.asarray(Q.dequantize_ref(
                __import__("numpy").asarray(packed), __import__("numpy").asarray(scale),
                __import__("numpy").asarray(wmin), group))
        fp["layers"].append(lp)
    return fp


def _dequant_ig(qw, group):
    """in-graph dequant of one packed int4 weight -> fp32 [out, in]."""
    packed, scale, wmin = qw
    out, ip8 = packed.shape
    in_dim = ip8 * 8
    shifts = jnp.arange(8, dtype=jnp.uint32) * 4
    codes = (packed[:, :, None] >> shifts) & jnp.uint32(0xF)   # [out, in/8, 8]
    codes = codes.reshape(out, in_dim).astype(jnp.float32)
    scale_b = jnp.repeat(scale, group, axis=1)                 # [out, in]
    wmin_b = jnp.repeat(wmin, group, axis=1)
    return codes * scale_b + wmin_b


def _lin(x, qw, group):
    return x @ _dequant_ig(qw, group).T


def make_int4_fns(cfg: M.Config, max_seq: int, group: int = Q.GROUP):
    cos_t, sin_t = M.rope_tables(cfg, max_seq)
    g, nq, nkv, d = cfg.group, cfg.n_q, cfg.n_kv, cfg.head_dim
    NEG = jnp.float32(-1e30)

    def mlp(h, ql):
        return _lin(M._silu(_lin(h, ql["gate"], group)) * _lin(h, ql["up"], group), ql["down"], group)

    def prefill(params, tokens, positions, real_len):
        Lp = tokens.shape[0]
        x = params["embed"][tokens]
        cos, sin = cos_t[positions], sin_t[positions]
        i = jnp.arange(Lp)[:, None]; j = jnp.arange(Lp)[None, :]
        cmask = jnp.where(j <= i, 0.0, NEG).astype(jnp.float32)
        pad = max_seq - Lp
        kc_layers, vc_layers = [], []
        for li, ql in enumerate(params["layers"]):
            h = M._rms_norm(x, ql["in_ln"], cfg.eps)
            q = _lin(h, ql["wq"], group).reshape(Lp, nq, d)
            k = _lin(h, ql["wk"], group).reshape(Lp, nkv, d)
            v = _lin(h, ql["wv"], group).reshape(Lp, nkv, d)
            q = M._apply_rope(q, cos, sin); k = M._apply_rope(k, cos, sin)
            # pad to MAX_SEQ + stack across layers (avoids a range-slice scatter
            # that IREE's CUDA backend does not lower; dynamic_update_slice/pad do).
            kc_layers.append(jnp.pad(k, ((0, pad), (0, 0), (0, 0))))
            vc_layers.append(jnp.pad(v, ((0, pad), (0, 0), (0, 0))))
            k_rep = jnp.repeat(k, g, axis=1); v_rep = jnp.repeat(v, g, axis=1)
            scores = jnp.einsum("ihd,jhd->hij", q, k_rep) * cfg.scale + cmask[None]
            attn = jax.nn.softmax(scores, axis=-1)
            o = jnp.einsum("hij,jhd->ihd", attn, v_rep).reshape(Lp, nq * d)
            x = x + _lin(o, ql["wo"], group)
            x = x + mlp(M._rms_norm(x, ql["post_ln"], cfg.eps), ql)
        xf = M._rms_norm(x, params["final_norm"], cfg.eps)
        last = jax.lax.dynamic_slice_in_dim(xf, real_len - 1, 1, axis=0)[0]
        return last @ params["embed"].T, jnp.stack(kc_layers, 0), jnp.stack(vc_layers, 0)

    def decode_step(params, token, pos, cache_len, kcache, vcache):
        x = params["embed"][token]
        cos, sin = cos_t[pos][None], sin_t[pos][None]
        valid = jnp.arange(max_seq) <= cache_len
        kmask = jnp.where(valid, 0.0, NEG).astype(jnp.float32)
        for li, ql in enumerate(params["layers"]):
            h = M._rms_norm(x, ql["in_ln"], cfg.eps)
            q = _lin(h, ql["wq"], group).reshape(1, nq, d)
            k = _lin(h, ql["wk"], group).reshape(1, nkv, d)
            v = _lin(h, ql["wv"], group).reshape(1, nkv, d)
            q = M._apply_rope(q, cos, sin)[0]; k = M._apply_rope(k, cos, sin)[0]; v = v[0]
            kcache = kcache.at[li, cache_len].set(k); vcache = vcache.at[li, cache_len].set(v)
            k_rep = jnp.repeat(kcache[li], g, axis=1); v_rep = jnp.repeat(vcache[li], g, axis=1)
            scores = jnp.einsum("hd,shd->hs", q, k_rep) * cfg.scale + kmask[None]
            attn = jax.nn.softmax(scores, axis=-1)
            o = jnp.einsum("hs,shd->hd", attn, v_rep).reshape(nq * d)
            x = x + _lin(o, ql["wo"], group)
            x = x + mlp(M._rms_norm(x, ql["post_ln"], cfg.eps), ql)
        xf = M._rms_norm(x, params["final_norm"], cfg.eps)
        return xf @ params["embed"].T, kcache, vcache

    return prefill, decode_step
