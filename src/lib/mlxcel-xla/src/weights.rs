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

use crate::emitter::Config;

/// One checkpoint weight the loader reads, in the emitter's arg order. Most are a
/// whole safetensors tensor; a Phi3 checkpoint fuses q/k/v into one `qkv_proj` and
/// gate/up into one `gate_up_proj`, so the loader takes a row-slice of the fused
/// tensor for each of the emitter's separate `wq`/`wk`/`wv`/`gate`/`up` args.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WeightSpec {
    /// Load the whole checkpoint tensor `name`.
    Whole(String),
    /// Load rows `[start, end)` of the checkpoint tensor `name` (a fused Phi3
    /// projection, split into an emitter arg). Row-major `[out, in]`, so this is
    /// the `[start, end)` slice of the `out` axis.
    Rows {
        name: String,
        start: usize,
        end: usize,
    },
}

impl WeightSpec {
    /// The checkpoint tensor this spec reads from (whole or sliced).
    pub(crate) fn tensor_name(&self) -> &str {
        match self {
            WeightSpec::Whole(n) => n,
            WeightSpec::Rows { name, .. } => name,
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
        // down (dense StarCoder2 uses c_proj; else the scheme's down projection).
        out.push(WeightSpec::Whole(if cfg.dense_mlp {
            format!("{p}mlp.c_proj.weight")
        } else {
            format!("{p}{}", s.down)
        }));
        // gate (gated MLP only; the first half of gate_up_proj for a fused Phi3).
        if gated {
            if cfg.fused_gate_up {
                out.push(WeightSpec::Rows {
                    name: format!("{p}mlp.gate_up_proj.weight"),
                    start: 0,
                    end: inter,
                });
            } else {
                out.push(WeightSpec::Whole(format!("{p}{}", s.gate)));
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
        if cfg.fused_gate_up {
            out.push(WeightSpec::Rows {
                name: format!("{p}mlp.gate_up_proj.weight"),
                start: inter,
                end: 2 * inter,
            });
        } else if cfg.dense_mlp {
            out.push(WeightSpec::Whole(format!("{p}mlp.c_fc.weight")));
        } else {
            out.push(WeightSpec::Whole(format!("{p}{}", s.up)));
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
            out.push(WeightSpec::Whole(format!("{p}{}", s.k_proj)));
            out.push(WeightSpec::Whole(format!("{p}{}", s.o_proj)));
            out.push(WeightSpec::Whole(format!("{p}{}", s.q_proj)));
            out.push(WeightSpec::Whole(format!("{p}{}", s.v_proj)));
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

/// Dequantize one MLX affine-quantized weight to row-major `[out, in]` f32.
///
/// `packed` is the row-major `[out, in_packed]` u32 weight (little-endian bytes,
/// `in_packed = in * bits / 32`); `scales` / `biases` are the row-major
/// `[out, in/group_size]` f16 buffers. Each weight is recovered as
/// `w[o,i] = q[o,i] * scale[o, i/group_size] + bias[o, i/group_size]`, where `q`
/// is the `bits`-wide value unpacked low-order-first from `packed[o, i/(32/bits)]`
/// (the MLX affine layout). The graph runs in f32, so the packed weights are
/// widened here once at load.
pub(crate) fn dequantize_affine(
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    out: usize,
    in_packed: usize,
    bits: usize,
    group_size: usize,
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
    let scales = f16_to_f32(scales);
    let biases = f16_to_f32(biases);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The Llama family weight order is the legacy all-`Whole` sequence (embed,
    /// norm, then 9 per layer), so the #498 spec loader is byte-identical for it.
    #[test]
    fn weight_specs_llama_is_the_legacy_whole_order() {
        let c = Config::llama_3_2_1b();
        let specs = weight_specs(&c);
        assert!(specs.iter().all(|s| matches!(s, WeightSpec::Whole(_))));
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
        let w = dequantize_affine(&packed, &scales, &biases, 1, 1, 4, 4).unwrap();
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
        let w = dequantize_affine(&packed, &scales, &biases, 1, 1, 8, 2).unwrap();
        assert_eq!(w, vec![30.0, 50.0, 14.0, 19.0]);
    }

    /// A packed buffer whose size disagrees with `[out, in_packed]` is rejected.
    #[test]
    fn dequantize_affine_rejects_size_mismatch() {
        let packed = [0u8; 4];
        let sb = [0u8; 4];
        assert!(dequantize_affine(&packed, &sb, &sb, 2, 1, 4, 4).is_err());
    }
}
