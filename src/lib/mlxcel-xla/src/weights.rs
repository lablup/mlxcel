// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Widen safetensors weight bytes to f32 (the dtype the emitted StableHLO graphs
//! take), for the IREE loader (issue #449 M3 Stage 2d). bf16 and f16 are the
//! common checkpoint dtypes; f32 is a passthrough. Every conversion is exact
//! (f32 represents every bf16/f16 value), so the widened weights match HF's own
//! f32 cast, which the token-exact oracle gate depends on.
//!
//! This module also owns [`weight_specs`], the per-architecture checkpoint-weight
//! order the IREE loader (`iree.rs`) reads (issue #498), kept here (not in the
//! feature-gated `iree.rs`) so it is unit-tested without the IREE runtime and stays
//! in lock-step with the emitter's arg order (`emitter::model::take_layer_weights`).

use crate::emitter::{Config, quant_in_graph};

/// One checkpoint weight the loader reads, in the emitter's arg order. Most are a
/// whole safetensors tensor; a Phi3 checkpoint fuses q/k/v into one `qkv_proj` and
/// gate/up into one `gate_up_proj`, so the loader takes a row-slice of the fused
/// tensor for each of the emitter's separate `wq`/`wk`/`wv`/`gate`/`up` args.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WeightSpec {
    /// Load the whole checkpoint tensor `name` (widened to f32, an MLX-quantized
    /// U32 tensor dequantized to f32).
    Whole(String),
    /// Load the whole checkpoint tensor `name` for a linear projection the emitter
    /// took via `take_weight` (issue #572). Widened to f32 like [`WeightSpec::Whole`],
    /// but the loader packs it to an f16 resident buffer when the f16 GPU path is
    /// active (`Config::supports_f16_resident` + f16 precision), matching the emitter's
    /// f16 weight arg; otherwise it uploads f32, byte-identical to the old `Whole`.
    /// Kept distinct from `Whole` (embed / norm / lm_head), which stays f32-resident.
    Proj(String),
    /// Load rows `[start, end)` of the checkpoint tensor `name` (a fused Phi3
    /// projection, split into an emitter arg). Row-major `[out, in]`, so this is
    /// the `[start, end)` slice of the `out` axis.
    Rows {
        name: String,
        start: usize,
        end: usize,
    },
    /// Upload one part of an MLX affine-quantized projection as RAW bytes without
    /// dequantizing (issue #516 packed path): the packed `[out, in_packed]` U32
    /// weight, or its `[out, in/group_size]` f16 `scales` / `biases`. The graph
    /// dequants it in-place ([`Builder::dequant_affine`]). `name` is the exact
    /// tensor (`X.weight` / `X.scales` / `X.biases`). Emitted as three consecutive
    /// specs per projection, matching the emitter's packed / scales / biases args.
    QuantRaw { name: String, part: QuantPart },
}

/// Which part of an MLX affine-quantized projection a [`WeightSpec::QuantRaw`]
/// uploads (issue #516): the packed U32 weight, or its f16 scales / biases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantPart {
    Packed,
    Scales,
    Biases,
}

impl WeightSpec {
    /// The checkpoint tensor this spec reads from (whole, sliced, or a quant part).
    pub(crate) fn tensor_name(&self) -> &str {
        match self {
            WeightSpec::Whole(n) => n,
            WeightSpec::Proj(n) => n,
            WeightSpec::Rows { name, .. } => name,
            WeightSpec::QuantRaw { name, .. } => name,
        }
    }
}

/// The checkpoint weights the loader reads, in the emitter's exact arg order
/// (`take_lm_head` / `take_final_norm_bias` / `take_layer_weights` in
/// `emitter/model.rs`): `embed`, `norm` (+ `norm.bias` for a LayerNorm arch), then
/// — for an untied checkpoint — the LM head, then per layer `down`, `gate` (gated
/// only), `in_ln` (unless the OLMo reordered post-norm drops it), `post_ln`
/// (sequential only), `up`, `wk`, `wo`, `wq`, `wv`, then the q/k/v biases, the q/k
/// norms (Qwen3 / Gemma3 / OLMo2/3), the Gemma2/3 and OLMo2/3 feed-forward norms,
/// and finally the #498 LayerNorm biases, the `o_proj` bias, and the MLP biases.
///
/// The base tensor names come from [`weight_names::scheme_names`](crate::weight_names)
/// (issue #499): the standard HF Llama layout, or ExaOne 3.x's GPT-2-style names.
/// A dense (StarCoder2) MLP uses `c_fc`/`c_proj` and has no gate; a fused (Phi3)
/// checkpoint row-slices `qkv_proj` / `gate_up_proj` (both dense/fused deltas are
/// Llama-scheme). Byte-identical order to the pre-dense-pack loader for the Llama
/// family, and it mirrors `take_layer_weights` so the loaded buffers line up with
/// the emitted graph's args exactly.
pub(crate) fn weight_specs(cfg: &Config) -> Vec<WeightSpec> {
    weight_specs_q(cfg, quant_in_graph() && cfg.supports_packed_quant())
}

/// Push a projection weight `name` (`X.weight`) in the emitter's arg order: three
/// [`WeightSpec::QuantRaw`] parts (packed / scales / biases, from `X.weight` /
/// `X.scales` / `X.biases`) when the issue #516 packed path is active (`quant`),
/// else the single `Whole` f32 tensor (dequantized / widened at load). Mirrors
/// `take_weight` in the emitter so the loaded buffers line up with the graph's
/// 1-or-3 args per projection.
fn push_proj(out: &mut Vec<WeightSpec>, name: String, quant: bool) {
    if quant {
        let stem = name.strip_suffix(".weight").unwrap_or(&name).to_string();
        out.push(WeightSpec::QuantRaw {
            name,
            part: QuantPart::Packed,
        });
        out.push(WeightSpec::QuantRaw {
            name: format!("{stem}.scales"),
            part: QuantPart::Scales,
        });
        out.push(WeightSpec::QuantRaw {
            name: format!("{stem}.biases"),
            part: QuantPart::Biases,
        });
    } else {
        out.push(WeightSpec::Proj(name));
    }
}

/// [`weight_specs`] with the packed-path decision passed explicitly (so it is
/// testable without the `MLXCEL_XLA_QUANT` env). `quant` already folds in
/// [`Config::supports_packed_quant`], so it is `true` only for the standard Llama
/// layout; the fused / dense / MoE branches below are therefore never quantized.
fn weight_specs_q(cfg: &Config, quant: bool) -> Vec<WeightSpec> {
    let s = crate::weight_names::scheme_names(cfg.weight_scheme);
    let hd = cfg.head_dim;
    let nq = cfg.n_q * hd;
    let nkv = cfg.n_kv * hd;
    let inter = cfg.inter;
    let gated = !cfg.dense_mlp;
    let has_post = !cfg.parallel_block;

    let mut out: Vec<WeightSpec> = Vec::new();
    out.push(WeightSpec::Whole(s.embed.to_string()));
    out.push(WeightSpec::Whole(s.final_norm.to_string()));
    // #498 final-norm affine bias (LayerNorm archs only; Llama scheme).
    if cfg.norm_bias {
        out.push(WeightSpec::Whole("model.norm.bias".to_string()));
    }
    if !cfg.tie_word_embeddings {
        out.push(WeightSpec::Whole(s.lm_head.to_string()));
    }
    for i in 0..cfg.n_layers {
        let p = format!("{}{i}.", s.layer_stem);
        // A MoE layer (issue #500) takes no dense MLP weights (down/gate/up); its
        // expert bank is appended after the attention weights / norms instead.
        let moe_layer = cfg.is_moe_layer(i);
        // down (dense StarCoder2 uses c_proj; else the scheme's down projection).
        if !moe_layer {
            let name = if cfg.dense_mlp {
                format!("{p}mlp.c_proj.weight")
            } else {
                format!("{p}{}", s.down)
            };
            push_proj(&mut out, name, quant);
        }
        // gate (gated MLP only; the first half of gate_up_proj for a fused Phi3).
        if gated && !moe_layer {
            if cfg.fused_gate_up {
                out.push(WeightSpec::Rows {
                    name: format!("{p}mlp.gate_up_proj.weight"),
                    start: 0,
                    end: inter,
                });
            } else {
                push_proj(&mut out, format!("{p}{}", s.gate), quant);
            }
        }
        // input_layernorm: present unless the OLMo reordered post-norm drops it.
        if cfg.has_input_norm() {
            out.push(WeightSpec::Whole(format!("{p}{}", s.input_layernorm)));
        }
        // post_attention_layernorm: dropped for a parallel-block arch (Cohere).
        if has_post {
            out.push(WeightSpec::Whole(format!(
                "{p}{}",
                s.post_attention_layernorm
            )));
        }
        // up (c_fc for dense; the second half of gate_up_proj for a fused Phi3).
        // Skipped on a MoE layer (issue #500), which has no dense up projection.
        if !moe_layer {
            if cfg.fused_gate_up {
                out.push(WeightSpec::Rows {
                    name: format!("{p}mlp.gate_up_proj.weight"),
                    start: inter,
                    end: 2 * inter,
                });
            } else if cfg.dense_mlp {
                out.push(WeightSpec::Whole(format!("{p}mlp.c_fc.weight")));
            } else {
                push_proj(&mut out, format!("{p}{}", s.up), quant);
            }
        }
        // wk, wo, wq, wv (JAX-alphabetical; a fused Phi3 qkv_proj is [Q|K|V] rows).
        if cfg.fused_qkv {
            let qkv = format!("{p}self_attn.qkv_proj.weight");
            out.push(WeightSpec::Rows {
                name: qkv.clone(),
                start: nq,
                end: nq + nkv,
            }); // wk
            out.push(WeightSpec::Whole(format!("{p}{}", s.o_proj))); // wo
            out.push(WeightSpec::Rows {
                name: qkv.clone(),
                start: 0,
                end: nq,
            }); // wq
            out.push(WeightSpec::Rows {
                name: qkv,
                start: nq + nkv,
                end: nq + 2 * nkv,
            }); // wv
        } else {
            push_proj(&mut out, format!("{p}{}", s.k_proj), quant);
            push_proj(&mut out, format!("{p}{}", s.o_proj), quant);
            push_proj(&mut out, format!("{p}{}", s.q_proj), quant);
            push_proj(&mut out, format!("{p}{}", s.v_proj), quant);
        }
        // q/k/v projection biases (k, q, v order).
        if cfg.qkv_bias {
            out.push(WeightSpec::Whole(format!("{p}{}", s.k_bias)));
            out.push(WeightSpec::Whole(format!("{p}{}", s.q_bias)));
            out.push(WeightSpec::Whole(format!("{p}{}", s.v_bias)));
        }
        // q/k norms (Qwen3 / Gemma3 per-head, OLMo2/3 flat), q then k.
        if cfg.qk_norm.is_some() {
            out.push(WeightSpec::Whole(format!("{p}{}", s.q_norm)));
            out.push(WeightSpec::Whole(format!("{p}{}", s.k_norm)));
        }
        // Feed-forward norms: Gemma2/3 add pre AND post; OLMo2/3 add post only.
        if cfg.has_pre_ff_norm() {
            out.push(WeightSpec::Whole(format!("{p}{}", s.pre_ff_norm)));
        }
        if cfg.has_post_ff_norm() {
            out.push(WeightSpec::Whole(format!("{p}{}", s.post_ff_norm)));
        }
        // #498 LayerNorm biases, the o_proj bias, and the MLP biases (Llama scheme).
        if cfg.norm_bias {
            out.push(WeightSpec::Whole(format!("{p}input_layernorm.bias")));
            if has_post {
                out.push(WeightSpec::Whole(format!(
                    "{p}post_attention_layernorm.bias"
                )));
            }
        }
        if cfg.attn_o_bias {
            out.push(WeightSpec::Whole(format!("{p}self_attn.o_proj.bias")));
        }
        if cfg.mlp_bias {
            out.push(WeightSpec::Whole(if cfg.dense_mlp {
                format!("{p}mlp.c_proj.bias")
            } else {
                format!("{p}mlp.down_proj.bias")
            }));
            if gated {
                out.push(WeightSpec::Whole(format!("{p}mlp.gate_proj.bias")));
            }
            out.push(WeightSpec::Whole(if cfg.dense_mlp {
                format!("{p}mlp.c_fc.bias")
            } else {
                format!("{p}mlp.up_proj.bias")
            }));
        }
        // MoE expert bank (issue #500), last in the layer: the router, the stacked
        // mlx-lm `switch_mlp` gate/up/down, then the optional shared expert (and its
        // sigmoid gate). `weight_prefix` is `mlp` (Qwen2-MoE) or `block_sparse_moe`
        // (Mixtral); the order mirrors `take_moe_weights` in `emitter/model.rs` and
        // the emitted args, so the loaded buffers line up with the graph.
        if moe_layer {
            let m = cfg.moe.as_ref().expect("a MoE layer has a MoeConfig");
            let mp = m.weight_prefix;
            out.push(WeightSpec::Whole(format!("{p}{mp}.gate.weight")));
            out.push(WeightSpec::Whole(format!(
                "{p}{mp}.switch_mlp.gate_proj.weight"
            )));
            out.push(WeightSpec::Whole(format!(
                "{p}{mp}.switch_mlp.up_proj.weight"
            )));
            out.push(WeightSpec::Whole(format!(
                "{p}{mp}.switch_mlp.down_proj.weight"
            )));
            if let Some(sh) = m.shared {
                out.push(WeightSpec::Whole(format!(
                    "{p}{mp}.shared_expert.gate_proj.weight"
                )));
                out.push(WeightSpec::Whole(format!(
                    "{p}{mp}.shared_expert.up_proj.weight"
                )));
                out.push(WeightSpec::Whole(format!(
                    "{p}{mp}.shared_expert.down_proj.weight"
                )));
                if sh.gated {
                    out.push(WeightSpec::Whole(format!(
                        "{p}{mp}.shared_expert_gate.weight"
                    )));
                }
            }
        }
    }
    out
}

/// Slice rows `[start, end)` of a row-major `[out, in]` f32 buffer into a
/// `[end - start, in]` buffer (the Phi3 fused-projection split at load).
pub(crate) fn slice_rows(
    buf: &[f32],
    out: usize,
    start: usize,
    end: usize,
) -> Result<Vec<f32>, String> {
    if out == 0 || !buf.len().is_multiple_of(out) {
        return Err(format!(
            "cannot row-slice a {} element buffer as [{out}, in]",
            buf.len()
        ));
    }
    let in_ = buf.len() / out;
    if start > end || end > out {
        return Err(format!(
            "row slice [{start}, {end}) out of range for {out} rows"
        ));
    }
    Ok(buf[start * in_..end * in_].to_vec())
}

/// bf16 little-endian bytes -> f32 (bf16 is the high 16 bits of f32).
pub(crate) fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// One IEEE 754 half (f16) -> f32. The arithmetic forms are exact: a normal's
/// `1 + mant/1024` is a dyadic with denominator 2^10 and the `2^(exp-15)` / `2^-24`
/// scales are exact powers of two, so the widening is bit-for-bit.
pub(crate) fn half_to_f32(h: u16) -> f32 {
    let sign = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as f32;
    match exp {
        0 => sign * mant * 2f32.powi(-24),           // zero / subnormal
        0x1f if mant == 0.0 => sign * f32::INFINITY, // +/- inf
        0x1f => f32::NAN,                            // nan
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp as i32 - 15), // normal
    }
}

/// f16 little-endian bytes -> f32, via a 65536-entry `u16 -> f32` lookup table.
/// The table is built once (every f16 bit pattern, exact) and then each element
/// is a single index, so widening a multi-GB checkpoint is memory-bound rather
/// than arithmetic-bound (an 8B-param checkpoint otherwise spends minutes in
/// per-element `powi`).
pub(crate) fn f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let table: Vec<f32> = (0..=u16::MAX).map(half_to_f32).collect();
    bytes
        .chunks_exact(2)
        .map(|c| table[u16::from_le_bytes([c[0], c[1]]) as usize])
        .collect()
}

/// f32 little-endian bytes -> f32 (a plain reinterpret, for f32 checkpoints).
pub(crate) fn f32_le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One f32 -> IEEE 754 half (f16) bit pattern, round-to-nearest, ties-to-even (the
/// IEEE default), matching a `stablehlo.convert` f32 -> f16. So a projection weight
/// packed here (issue #572, f16-resident) is bit-identical to demoting the same f32
/// weight inside the graph, and the contraction sees the same f16 operand and stays
/// token-exact. It is the exact inverse of [`half_to_f32`]:
/// `f32_to_f16_bits(half_to_f32(h)) == h` for every finite, non-NaN f16 `h`.
pub(crate) fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff;

    // NaN / Inf (f32 exponent all ones): NaN -> a canonical quiet f16 NaN, Inf -> f16 Inf.
    if abs >= 0x7f80_0000 {
        return sign | if abs > 0x7f80_0000 { 0x7e00 } else { 0x7c00 };
    }

    // f16 biased exponent = (f32 biased exponent - 127) + 15.
    let e = (abs >> 23) as i32 - 127 + 15;

    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> Inf
    }

    if e <= 0 {
        // Subnormal f16, or underflow to a signed zero.
        if e < -10 {
            return sign; // below half the smallest subnormal -> +/- 0
        }
        // 24-bit significand (implicit leading 1), shifted into the subnormal range
        // and rounded to nearest, ties to even.
        let mant = (abs & 0x007f_ffff) | 0x0080_0000;
        let shift = (14 - e) as u32; // e in [-10, 0] -> shift in [14, 24]
        let q = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let round = u32::from(rem > half || (rem == half && q & 1 == 1));
        // q + round may reach 0x400, which is exactly the smallest normal (correct).
        return sign | (q + round) as u16;
    }

    // Normal f16: keep the top 10 mantissa bits, round to nearest even on bit 12. A
    // mantissa carry rolls into the exponent, an exponent carry into 0x7c00 (Inf) --
    // both the correct results.
    let mant = abs & 0x007f_ffff;
    let base = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff; // the 13 dropped low bits
    let half = 0x1000u32; // 1 << 12
    let round = u32::from(rem > half || (rem == half && base & 1 == 1));
    sign | (base + round) as u16
}

/// Pack a row-major f32 weight to its little-endian f16 bit pattern for an
/// f16-resident device upload (issue #572), via [`f32_to_f16_bits`] (RNE, matching
/// the in-graph demotion). The `u16` values are native-endian; the shim copies the
/// raw bytes, so on a little-endian host they land as the f16 buffer IREE expects.
pub(crate) fn pack_f16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f32_to_f16_bits(x)).collect()
}

/// Dequantize one MLX affine-quantized weight to row-major `[out, in]` f32.
///
/// `packed` is the row-major `[out, in_packed]` u32 weight (little-endian bytes,
/// `in_packed = in * bits / 32`); `scales` / `biases` are the row-major
/// `[out, in/group_size]` f16 buffers. Each weight is recovered as
/// `w[o,i] = q[o,i] * scale[o, i/group_size] + bias[o, i/group_size]`, where `q`
/// is the `bits`-wide value unpacked low-order-first from `packed[o, i/(32/bits)]`
/// (the MLX affine layout). The graph runs in f32, so the packed weights are
/// widened here once at load.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequantize_affine(
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    out: usize,
    in_packed: usize,
    bits: usize,
    group_size: usize,
    scales_bf16: bool,
) -> Result<Vec<f32>, String> {
    if !(bits == 4 || bits == 8) {
        return Err(format!(
            "unsupported quantization bits {bits} (expected 4 or 8)"
        ));
    }
    let per_u32 = 32 / bits; // values packed per u32
    let in_ = in_packed * per_u32;
    if group_size == 0 || !in_.is_multiple_of(group_size) {
        return Err(format!(
            "quantization group_size {group_size} does not divide in dimension {in_}"
        ));
    }
    let n_groups = in_ / group_size;
    if packed.len() != out * in_packed * 4 {
        return Err(format!(
            "packed weight is {} bytes, expected {} ([{out}, {in_packed}] u32)",
            packed.len(),
            out * in_packed * 4
        ));
    }
    // mlx-lm stores the affine scale/bias in either f16 or bf16; widen the
    // matching 16-bit format to f32 (both are exact in f32).
    let (scales, biases) = if scales_bf16 {
        (bf16_to_f32(scales), bf16_to_f32(biases))
    } else {
        (f16_to_f32(scales), f16_to_f32(biases))
    };
    if scales.len() != out * n_groups || biases.len() != out * n_groups {
        return Err(format!(
            "scales/biases have {}/{} elements, expected {} ([{out}, {n_groups}])",
            scales.len(),
            biases.len(),
            out * n_groups
        ));
    }
    let mask: u32 = (1u32 << bits) - 1;
    let mut w = vec![0f32; out * in_];
    for o in 0..out {
        let row = &packed[o * in_packed * 4..(o + 1) * in_packed * 4];
        let grow = o * n_groups;
        let wrow = o * in_;
        for p in 0..in_packed {
            let u =
                u32::from_le_bytes([row[p * 4], row[p * 4 + 1], row[p * 4 + 2], row[p * 4 + 3]]);
            for j in 0..per_u32 {
                let i = p * per_u32 + j;
                let q = ((u >> (bits * j)) & mask) as f32;
                let g = i / group_size;
                w[wrow + i] = q * scales[grow + g] + biases[grow + g];
            }
        }
    }
    Ok(w)
}

/// Dequantize a STACKED mlx-lm affine-quantized expert weight (issue #500) to
/// row-major `[experts, out, in]` f32. The MoE `switch_mlp` projections pack all
/// `experts` into one `[experts, out, in_packed]` U32 tensor with companion
/// `[experts, out, in/group_size]` f16 `scales` / `biases`; this dequantizes each
/// expert's `[out, in_packed]` slab with [`dequantize_affine`] and concatenates
/// them, so the loader hands the emitter's `[E, out, in]` expert arg one f32
/// buffer. Byte-for-byte identical to dequantizing each expert separately.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequantize_affine_stacked(
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    experts: usize,
    out: usize,
    in_packed: usize,
    bits: usize,
    group_size: usize,
    scales_bf16: bool,
) -> Result<Vec<f32>, String> {
    if experts == 0 {
        return Err("stacked expert weight has 0 experts".to_string());
    }
    if !(bits == 4 || bits == 8) {
        return Err(format!(
            "unsupported quantization bits {bits} (expected 4 or 8)"
        ));
    }
    let per_u32 = 32 / bits;
    let in_ = in_packed * per_u32;
    if group_size == 0 || !in_.is_multiple_of(group_size) {
        return Err(format!(
            "quantization group_size {group_size} does not divide in dimension {in_}"
        ));
    }
    let n_groups = in_ / group_size;
    // Per-expert strides: the U32 weight is 4 bytes/element, the f16 scales/biases
    // 2 bytes/element. `dequantize_affine` re-validates each slab's exact sizes.
    let packed_stride = out * in_packed * 4;
    let sb_stride = out * n_groups * 2;
    if packed.len() != experts * packed_stride {
        return Err(format!(
            "stacked packed weight is {} bytes, expected {} ([{experts}, {out}, {in_packed}] u32)",
            packed.len(),
            experts * packed_stride
        ));
    }
    if scales.len() != experts * sb_stride || biases.len() != experts * sb_stride {
        return Err(format!(
            "stacked scales/biases have {}/{} bytes, expected {} ([{experts}, {out}, {n_groups}] 16-bit)",
            scales.len(),
            biases.len(),
            experts * sb_stride
        ));
    }
    let mut w = Vec::with_capacity(experts * out * in_);
    for e in 0..experts {
        let p = &packed[e * packed_stride..(e + 1) * packed_stride];
        let s = &scales[e * sb_stride..(e + 1) * sb_stride];
        let bi = &biases[e * sb_stride..(e + 1) * sb_stride];
        let slab = dequantize_affine(p, s, bi, out, in_packed, bits, group_size, scales_bf16)?;
        w.extend_from_slice(&slab);
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Llama family weight order is embed + norm (f32-resident `Whole`) then, per
    /// layer, the 7 linear projections as `Proj` (f16-resident-capable, issue #572)
    /// and the 2 norms as `Whole`, in the fixed order below. Names / order unchanged.
    #[test]
    fn weight_specs_llama_projections_are_proj_norms_are_whole() {
        let c = Config::llama_3_2_1b();
        let specs = weight_specs(&c);
        // Every projection weight (`*_proj.weight`) is `Proj`; embed / norm / the
        // per-layer norms are `Whole`. No fused rows or quant parts for plain Llama.
        for s in &specs {
            let n = s.tensor_name();
            match s {
                WeightSpec::Proj(_) => assert!(n.ends_with("_proj.weight"), "Proj {n}"),
                WeightSpec::Whole(_) => assert!(!n.ends_with("_proj.weight"), "Whole {n}"),
                other => panic!("unexpected spec {other:?}"),
            }
        }
        assert_eq!(specs.len(), 2 + 9 * c.n_layers);
        let names: Vec<&str> = specs.iter().map(WeightSpec::tensor_name).collect();
        assert_eq!(names[0], "model.embed_tokens.weight");
        assert_eq!(names[1], "model.norm.weight");
        assert_eq!(
            &names[2..11],
            &[
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ]
        );
    }

    #[test]
    fn f32_to_f16_round_trips_every_finite_half() {
        // half_to_f32(h) is exactly representable, so packing it back must recover h
        // for every finite, non-NaN f16 pattern (signed zeros, subnormals, normals).
        for h in 0u16..=u16::MAX {
            if (h >> 10) & 0x1f == 0x1f {
                continue; // skip Inf / NaN (NaN is non-canonical); Inf covered below
            }
            assert_eq!(
                f32_to_f16_bits(half_to_f32(h)),
                h,
                "round-trip failed for f16 bits {h:#06x}"
            );
        }
    }

    #[test]
    fn f32_to_f16_rounds_ties_to_even_and_saturates() {
        // Exact tie between 1.0 (0x3c00, even mantissa) and its successor 0x3c01
        // rounds down to the even neighbour; the tie one step up rounds to 0x3c02.
        assert_eq!(
            f32_to_f16_bits((half_to_f32(0x3c00) + half_to_f32(0x3c01)) / 2.0),
            0x3c00
        );
        assert_eq!(
            f32_to_f16_bits((half_to_f32(0x3c01) + half_to_f32(0x3c02)) / 2.0),
            0x3c02
        );
        // Above the f16 max (65504) saturates to +/-Inf; f16 max itself is exact.
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(70000.0), 0x7c00);
        assert_eq!(f32_to_f16_bits(-70000.0), 0xfc00);
        // Signed zero.
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    }

    /// Phi3 row-slices the fused `qkv_proj` ([Q|K|V]) and `gate_up_proj` (gate then
    /// up) into the emitter's separate args, and is untied (`lm_head` after norm).
    #[test]
    fn weight_specs_phi3_row_slices_the_fused_projections() {
        let json = r#"{"model_type":"phi3","hidden_size":32,"num_attention_heads":4,
            "num_key_value_heads":2,"intermediate_size":64,"num_hidden_layers":1,
            "rms_norm_eps":1e-5,"rope_theta":1e4,"vocab_size":48,"tie_word_embeddings":false}"#;
        let c = Config::from_json_str(json).expect("phi3 parses");
        let specs = weight_specs(&c);
        assert_eq!(specs[2], WeightSpec::Whole("lm_head.weight".to_string()));
        let gu = "model.layers.0.mlp.gate_up_proj.weight".to_string();
        assert!(specs.contains(&WeightSpec::Rows {
            name: gu.clone(),
            start: 0,
            end: 64
        }));
        assert!(specs.contains(&WeightSpec::Rows {
            name: gu,
            start: 64,
            end: 128
        }));
        // qkv: q rows [0,32), k [32,48), v [48,64) (nq=4*8, nkv=2*8).
        let qkv = "model.layers.0.self_attn.qkv_proj.weight".to_string();
        assert!(specs.contains(&WeightSpec::Rows {
            name: qkv.clone(),
            start: 0,
            end: 32
        }));
        assert!(specs.contains(&WeightSpec::Rows {
            name: qkv.clone(),
            start: 32,
            end: 48
        }));
        assert!(specs.contains(&WeightSpec::Rows {
            name: qkv,
            start: 48,
            end: 64
        }));
        assert!(!specs.iter().any(|s| s.tensor_name().contains("q_proj")));
    }

    /// StarCoder2 uses the dense `c_fc`/`c_proj` MLP (no gate) and carries biases on
    /// the norms and every projection.
    #[test]
    fn weight_specs_starcoder2_dense_mlp_and_biases() {
        let json = r#"{"model_type":"starcoder2","hidden_size":32,"num_attention_heads":4,
            "num_key_value_heads":2,"intermediate_size":64,"num_hidden_layers":1,
            "norm_epsilon":1e-5,"rope_theta":1e4,"vocab_size":48,"use_bias":true}"#;
        let c = Config::from_json_str(json).expect("starcoder2 parses");
        let names: Vec<String> = weight_specs(&c)
            .iter()
            .map(|s| s.tensor_name().to_string())
            .collect();
        assert!(names.iter().any(|n| n.ends_with("mlp.c_fc.weight")));
        assert!(names.iter().any(|n| n.ends_with("mlp.c_proj.weight")));
        assert!(!names.iter().any(|n| n.ends_with("mlp.gate_proj.weight")));
        assert!(names.iter().any(|n| n == "model.norm.bias"));
        assert!(names.iter().any(|n| n.ends_with("input_layernorm.bias")));
        assert!(names.iter().any(|n| n.ends_with("self_attn.o_proj.bias")));
        assert!(names.iter().any(|n| n.ends_with("mlp.c_fc.bias")));
        assert!(names.iter().any(|n| n.ends_with("mlp.c_proj.bias")));
    }

    /// Cohere is tied, LayerNorm (no norm bias), parallel (no post-attn norm).
    #[test]
    fn weight_specs_cohere_is_tied_parallel_no_post_norm() {
        let json = r#"{"model_type":"cohere","hidden_size":32,"num_attention_heads":4,
            "num_key_value_heads":2,"intermediate_size":64,"num_hidden_layers":1,
            "layer_norm_eps":1e-5,"rope_theta":1e4,"vocab_size":48,"logit_scale":0.25}"#;
        let c = Config::from_json_str(json).expect("cohere parses");
        let names: Vec<String> = weight_specs(&c)
            .iter()
            .map(|s| s.tensor_name().to_string())
            .collect();
        assert!(!names.iter().any(|n| n == "lm_head.weight"), "tied");
        assert!(
            !names.iter().any(|n| n.contains("post_attention_layernorm")),
            "parallel"
        );
        assert!(
            !names.iter().any(|n| n == "model.norm.bias"),
            "no norm bias"
        );
        assert!(
            names.iter().any(|n| n.ends_with("mlp.gate_proj.weight")),
            "gated"
        );
    }

    /// Qwen3-MoE / OLMoE (issue #501) name their per-layer expert bank on the `mlp`
    /// prefix, matching the mlx-lm `switch_mlp` stacking a real checkpoint uses
    /// (`model.layers.0.mlp.gate.weight`, `model.layers.0.mlp.switch_mlp.*_proj.weight`),
    /// alongside the q/k norms; they add no shared-expert tensor, no dense
    /// `mlp.{gate,up,down}_proj`, and no q/k/v bias.
    #[test]
    fn weight_specs_qwen3_moe_and_olmoe_name_the_expert_bank() {
        let qwen3 = r#"{"model_type":"qwen3_moe","hidden_size":8,"num_attention_heads":3,
            "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,"moe_intermediate_size":12,
            "num_hidden_layers":1,"num_experts":4,"num_experts_per_tok":2,"norm_topk_prob":true,
            "rms_norm_eps":1e-6,"rope_theta":1e6,"vocab_size":10,"tie_word_embeddings":false}"#;
        let olmoe = r#"{"model_type":"olmoe","hidden_size":8,"num_attention_heads":2,
            "num_key_value_heads":1,"head_dim":4,"intermediate_size":12,"num_hidden_layers":1,
            "num_experts":4,"num_experts_per_tok":2,"norm_topk_prob":true,"rms_norm_eps":1e-6,
            "rope_theta":5e5,"vocab_size":10,"tie_word_embeddings":false}"#;
        for (name, json) in [("qwen3_moe", qwen3), ("olmoe", olmoe)] {
            let c = Config::from_json_str(json).unwrap_or_else(|e| panic!("{name}: {e}"));
            let names: Vec<String> = weight_specs(&c)
                .iter()
                .map(|s| s.tensor_name().to_string())
                .collect();
            let has = |n: &str| names.iter().any(|x| x == n);
            assert!(has("model.layers.0.mlp.gate.weight"), "{name}: router");
            assert!(
                has("model.layers.0.mlp.switch_mlp.gate_proj.weight")
                    && has("model.layers.0.mlp.switch_mlp.up_proj.weight")
                    && has("model.layers.0.mlp.switch_mlp.down_proj.weight"),
                "{name}: stacked switch_mlp experts"
            );
            assert!(
                has("model.layers.0.self_attn.q_norm.weight")
                    && has("model.layers.0.self_attn.k_norm.weight"),
                "{name}: q/k norms"
            );
            assert!(
                !names.iter().any(|n| n.contains("shared_expert")),
                "{name}: no shared expert"
            );
            assert!(
                !has("model.layers.0.mlp.gate_proj.weight")
                    && !has("model.layers.0.mlp.up_proj.weight")
                    && !has("model.layers.0.mlp.down_proj.weight"),
                "{name}: no dense MLP tensor on a MoE layer"
            );
            assert!(
                !names.iter().any(|n| n.contains("q_proj.bias")),
                "{name}: no q/k/v bias"
            );
        }
    }

    /// `slice_rows` extracts a row band of a row-major `[out, in]` buffer and
    /// rejects an out-of-range band or a non-divisible length.
    #[test]
    fn slice_rows_extracts_the_row_band() {
        let buf: Vec<f32> = (0..8).map(|x| x as f32).collect(); // 4 rows x 2 cols
        assert_eq!(slice_rows(&buf, 4, 1, 3).unwrap(), vec![2.0, 3.0, 4.0, 5.0]);
        assert!(slice_rows(&buf, 4, 2, 5).is_err());
        assert!(slice_rows(&buf, 3, 0, 1).is_err());
    }

    /// f16 widening is exact against `f32 as` for representative values: zero, one,
    /// a fraction, a negative, the max normal, and a subnormal.
    #[test]
    fn half_to_f32_matches_reference_values() {
        // (f16 bits, expected f32) pairs.
        let cases: [(u16, f32); 7] = [
            (0x0000, 0.0),            // +0
            (0x8000, -0.0),           // -0
            (0x3c00, 1.0),            // 1.0
            (0x3800, 0.5),            // 0.5
            (0xc000, -2.0),           // -2.0
            (0x7bff, 65504.0),        // max normal f16
            (0x0001, 2f32.powi(-24)), // smallest positive subnormal
        ];
        for (bits, want) in cases {
            let got = half_to_f32(bits);
            assert_eq!(got, want, "f16 {bits:#06x} -> {got} != {want}");
        }
    }

    /// inf / nan f16 encodings widen to f32 inf / nan.
    #[test]
    fn half_to_f32_handles_inf_and_nan() {
        assert!(half_to_f32(0x7c00).is_infinite() && half_to_f32(0x7c00) > 0.0);
        assert!(half_to_f32(0xfc00).is_infinite() && half_to_f32(0xfc00) < 0.0);
        assert!(half_to_f32(0x7e00).is_nan());
    }

    /// The byte converters round-trip a little-endian buffer of two values.
    #[test]
    fn f16_byte_buffer_widens_both_lanes() {
        // 1.0 (0x3c00) then -2.0 (0xc000), little-endian.
        let bytes = [0x00, 0x3c, 0x00, 0xc0];
        assert_eq!(f16_to_f32(&bytes), vec![1.0, -2.0]);
    }

    /// bf16 widening keeps the high 16 bits (1.0 -> 0x3f80).
    #[test]
    fn bf16_byte_buffer_widens() {
        let bytes = [0x80, 0x3f]; // bf16 1.0, little-endian
        assert_eq!(bf16_to_f32(&bytes), vec![1.0]);
    }

    /// f32 passthrough reinterprets 4-byte lanes.
    #[test]
    fn f32_passthrough_reinterprets() {
        let bytes = 1.5f32.to_le_bytes();
        assert_eq!(f32_le_to_f32(&bytes), vec![1.5]);
    }

    /// 4-bit affine dequant on a hand-built row: one u32 packs eight nibbles
    /// 1..=8 (low-order first), two groups of 4 with scale/bias (2.0, +10) and
    /// (0.5, -1), so `q*scale + bias` is exact.
    #[test]
    fn dequantize_affine_recovers_hand_example() {
        // u32 = 0x8765_4321 -> nibbles [1,2,3,4,5,6,7,8] low-order first.
        let packed = [0x21u8, 0x43, 0x65, 0x87];
        let scales = [0x00u8, 0x40, 0x00, 0x38]; // f16 [2.0, 0.5]
        let biases = [0x00u8, 0x49, 0x00, 0xBC]; // f16 [10.0, -1.0]
        let w = dequantize_affine(&packed, &scales, &biases, 1, 1, 4, 4, false).unwrap();
        assert_eq!(w, vec![12.0, 14.0, 16.0, 18.0, 1.5, 2.0, 2.5, 3.0]);
    }

    /// BF16 scales/biases (as emitted by e.g. Qwen3 / Qwen3-MoE MLX 4-bit
    /// checkpoints) dequantize identically to the F16 hand example: the loader
    /// accepts either 16-bit float format for the affine scale/bias. The values
    /// 2.0 / 0.5 / 10.0 / -1.0 are exact in bf16, so the recovered row is unchanged.
    #[test]
    fn dequantize_affine_accepts_bf16_scales() {
        let packed = [0x21u8, 0x43, 0x65, 0x87];
        let scales = [0x00u8, 0x40, 0x00, 0x3F]; // bf16 [2.0, 0.5]
        let biases = [0x20u8, 0x41, 0x80, 0xBF]; // bf16 [10.0, -1.0]
        let w = dequantize_affine(&packed, &scales, &biases, 1, 1, 4, 4, true).unwrap();
        assert_eq!(w, vec![12.0, 14.0, 16.0, 18.0, 1.5, 2.0, 2.5, 3.0]);
    }

    /// 8-bit affine dequant: one u32 packs four bytes 10/20/30/40 (low-order
    /// first), two groups of 2 with scale/bias (2.0, +10) and (0.5, -1), so
    /// `q*scale + bias` is exact. Exercises the `bits = 8` (`per_u32 = 4`) path.
    #[test]
    fn dequantize_affine_8bit_recovers_hand_example() {
        // u32 = 0x281E_140A -> bytes [10, 20, 30, 40] low-order first.
        let packed = [0x0Au8, 0x14, 0x1E, 0x28];
        let scales = [0x00u8, 0x40, 0x00, 0x38]; // f16 [2.0, 0.5]
        let biases = [0x00u8, 0x49, 0x00, 0xBC]; // f16 [10.0, -1.0]
        let w = dequantize_affine(&packed, &scales, &biases, 1, 1, 8, 2, false).unwrap();
        assert_eq!(w, vec![30.0, 50.0, 14.0, 19.0]);
    }

    /// A packed buffer whose size disagrees with `[out, in_packed]` is rejected.
    #[test]
    fn dequantize_affine_rejects_size_mismatch() {
        let packed = [0u8; 4];
        let sb = [0u8; 4];
        assert!(dequantize_affine(&packed, &sb, &sb, 2, 1, 4, 4, false).is_err());
    }

    /// The stacked (rank-3) expert dequant is exactly the per-expert dequant
    /// concatenated: two experts, each the 4-bit hand example, yield that row
    /// twice. Exercises the `[experts, out, in_packed]` slab strides (issue #500).
    #[test]
    fn dequantize_affine_stacked_concatenates_expert_slabs() {
        // One expert's inputs (the `dequantize_affine_recovers_hand_example` row).
        let packed1 = [0x21u8, 0x43, 0x65, 0x87];
        let scales1 = [0x00u8, 0x40, 0x00, 0x38]; // f16 [2.0, 0.5]
        let biases1 = [0x00u8, 0x49, 0x00, 0xBC]; // f16 [10.0, -1.0]
        let one = dequantize_affine(&packed1, &scales1, &biases1, 1, 1, 4, 4, false).unwrap();

        // Stack two identical experts.
        let packed: Vec<u8> = packed1.iter().chain(&packed1).copied().collect();
        let scales: Vec<u8> = scales1.iter().chain(&scales1).copied().collect();
        let biases: Vec<u8> = biases1.iter().chain(&biases1).copied().collect();
        let stacked =
            dequantize_affine_stacked(&packed, &scales, &biases, 2, 1, 1, 4, 4, false).unwrap();

        let mut expected = one.clone();
        expected.extend_from_slice(&one);
        assert_eq!(stacked, expected);
        assert_eq!(stacked.len(), 2 * 8, "two experts x eight recovered values");
    }

    /// A stacked buffer whose size disagrees with `[experts, out, in_packed]` is
    /// rejected, so a mis-shaped expert bank fails loudly rather than mis-loading.
    #[test]
    fn dequantize_affine_stacked_rejects_size_mismatch() {
        let packed = [0u8; 4]; // one expert's worth, but experts = 2 declared
        let sb = [0u8; 4];
        assert!(dequantize_affine_stacked(&packed, &sb, &sb, 2, 1, 1, 4, 4, false).is_err());
    }

    /// The issue #516 packed path expands each of the 7 per-layer projections into
    /// three consecutive `QuantRaw` specs (packed `.weight`, then `.scales`, then
    /// `.biases`), while embed / norms stay single `Whole` specs. This is the loader
    /// contract that must mirror the emitter's `take_weight` 3-args-per-projection so
    /// the uploaded buffers line up with the graph args. `quant = false` is the
    /// legacy all-`Whole` order (byte-identical to before), so the packed path is
    /// purely additive.
    #[test]
    fn weight_specs_q_packed_expands_projections_to_triples() {
        let c = Config::llama_3_2_1b();
        let specs = weight_specs_q(&c, true);
        // embed + final_norm remain single f32 tensors (not quantized in the v1 path).
        assert_eq!(
            specs[0],
            WeightSpec::Whole("model.embed_tokens.weight".into())
        );
        assert_eq!(specs[1], WeightSpec::Whole("model.norm.weight".into()));
        let packed = specs
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    WeightSpec::QuantRaw {
                        part: QuantPart::Packed,
                        ..
                    }
                )
            })
            .count();
        let scales = specs
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    WeightSpec::QuantRaw {
                        part: QuantPart::Scales,
                        ..
                    }
                )
            })
            .count();
        let biases = specs
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    WeightSpec::QuantRaw {
                        part: QuantPart::Biases,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(packed, 7 * c.n_layers, "7 packed projections per layer");
        assert_eq!(scales, packed, "one scales part per packed weight");
        assert_eq!(biases, packed, "one biases part per packed weight");
        // The three parts of a projection are consecutive and share the tensor stem.
        let i = specs
            .iter()
            .position(|s| matches!(s, WeightSpec::QuantRaw { name, part: QuantPart::Packed } if name.ends_with("q_proj.weight")))
            .expect("a packed q_proj part");
        assert!(
            matches!(&specs[i + 1], WeightSpec::QuantRaw { name, part: QuantPart::Scales } if name.ends_with("q_proj.scales"))
        );
        assert!(
            matches!(&specs[i + 2], WeightSpec::QuantRaw { name, part: QuantPart::Biases } if name.ends_with("q_proj.biases"))
        );
        // quant = false is the unquantized layout: projections are `Proj` (issue
        // #572), everything else `Whole`, and there are no packed parts (the packed
        // path is purely additive).
        assert!(
            weight_specs_q(&c, false)
                .iter()
                .all(|s| matches!(s, WeightSpec::Whole(_) | WeightSpec::Proj(_))),
            "unquantized layout has no packed parts"
        );
    }
}
