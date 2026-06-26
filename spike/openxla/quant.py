"""Affine asymmetric int4 group quantization (RTN), packed 8 nibbles per uint32.

Representative of the mlx-community affine 4-bit scheme (group along the
contraction dim; per-group scale + min). Exact format alignment with mlxcel's
loader is a Phase 3 concern; this characterizes the dequant-in-graph route.
"""
import numpy as np

GROUP = 64


def quantize_affine_int4(W, group=GROUP):
    """W: float32 [out, in] (in divisible by group and by 8).
    Returns (packed uint32[out, in/8], scale f32[out, in/group], wmin f32[out, in/group])."""
    out, ind = W.shape
    ng = ind // group
    Wg = W.reshape(out, ng, group)
    wmin = Wg.min(axis=2)
    wmax = Wg.max(axis=2)
    scale = (wmax - wmin) / 15.0
    scale = np.where(scale == 0.0, 1.0, scale)
    codes = np.round((Wg - wmin[:, :, None]) / scale[:, :, None])
    codes = np.clip(codes, 0, 15).astype(np.uint32).reshape(out, ind)
    shifts = np.arange(8, dtype=np.uint32) * 4
    packed = (codes.reshape(out, ind // 8, 8) << shifts).sum(axis=2).astype(np.uint32)
    return packed, scale.astype(np.float32), wmin.astype(np.float32)


def dequantize_ref(packed, scale, wmin, group=GROUP):
    """numpy reference dequant (ground truth for the in-graph version)."""
    out, ip8 = packed.shape
    in_dim = ip8 * 8
    shifts = np.arange(8, dtype=np.uint32) * 4
    codes = (packed[:, :, None] >> shifts) & np.uint32(0xF)
    codes = codes.reshape(out, in_dim).astype(np.float32)
    return codes * np.repeat(scale, group, axis=1) + np.repeat(wmin, group, axis=1)


def quant_error(W, packed, scale, wmin, group=GROUP):
    Wq = dequantize_ref(packed, scale, wmin, group)
    num = np.linalg.norm(W - Wq)
    den = np.linalg.norm(W) + 1e-12
    return float(num / den)  # relative Frobenius error
