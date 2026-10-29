// Copyright 2025-2026 Lablup Inc.
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

//! LLaDA-Image's arithmetic for the LLaDA-2 MoE decoder (issue #2346).
//!
//! A port of `LLaDAImageTextEncoder` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llada_image/text_encoder.py>,
//! which subclasses upstream's LLaDA-2 MoE decoder and changes its numerics.
//! Every difference from the text model's [`Llada2Numerics::Fused`] path is
//! deliberate and reproduced here, because the image prior's discrete tokens
//! follow the MoE routing, and the routing flips on one-ulp differences:
//!
//! - RMSNorm is `rms_norm(x, None, eps) * weight`: the normalized activation is
//!   rounded to bf16 before the gain (the fused kernel applies the gain in f32).
//! - RoPE uses explicit cos/sin tables computed in f32 and rounded to the
//!   activation dtype before the rotation, over per-row `position_ids` (the
//!   VQ prior's unconditional CFG row is shifted by the pad length). The fused
//!   kernel keeps the tables in f32.
//! - The dense MLP and shared expert are `down(silu(gate(x)) * up(x))`.
//! - The router is upstream's `LLaDA2MoeGate`: group-limited top-k whose
//!   expert order is sorted by descending score (the fused gate keeps the
//!   argpartition order, which changes the f32 summation order below).
//! - The routed experts weight the activation by the f32 routing scores
//!   BEFORE the down projection (`TextMoE`), always on the expert-sorted
//!   gather path, and sum the expert outputs in f32.
//!
//! The entry point is [`Llada2MoeModel::forward_embeds`]: input embeddings
//! (word embeddings plus QueryFormer queries, or plain token embeddings for
//! the VQ prior), a boolean attention mask and per-row positions, no KV cache.
//!
//! Used by: LLaDAImage (`crate::imagegen::llada_image`).

use mlxcel_core::layers::RMSNorm;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::{Attention, DenseMLP, FFN, Layer, Llada2MoeModel, Llada2Numerics, MoEBlock, MoEGate};

/// `mx.fast.rms_norm(x, None, eps) * weight` (`TextRMSNorm`).
fn rms_then_gain(norm: &RMSNorm, x: &MlxArray) -> UniquePtr<MlxArray> {
    let normalized = mlxcel_core::fast_rms_norm_no_weight(x, norm.eps);
    mlxcel_core::multiply(&normalized, &norm.weight)
}

/// RoPE tables of `TextAttention._apply_rope`: `theta ** (-arange(0, d, 2) /
/// d)` and the angles in f32 on the device (the same ops upstream issues, so
/// the f32 values match bit for bit), then cos and sin rounded to `dt`.
/// `positions` is `[B, L]`; the tables are `[B, 1, L, rope_dim]`.
fn rope_tables(
    positions: &MlxArray,
    rope_dim: i32,
    theta: f32,
    dt: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let f32_scalar = |v: f32| mlxcel_core::full_f32(&[1], v, dtype::FLOAT32);
    let steps = mlxcel_core::arange_f32(0.0, rope_dim as f32, 2.0);
    let exponent =
        mlxcel_core::divide(&mlxcel_core::negative(&steps), &f32_scalar(rope_dim as f32));
    let frequencies = mlxcel_core::power(&f32_scalar(theta), &exponent);
    let positions = mlxcel_core::astype(positions, dtype::FLOAT32);
    let angles = mlxcel_core::multiply(&mlxcel_core::expand_dims(&positions, -1), &frequencies);
    let angles = mlxcel_core::concatenate(&angles, &angles, -1);
    let table = |t: UniquePtr<MlxArray>| mlxcel_core::astype(&mlxcel_core::expand_dims(&t, 1), dt);
    (
        table(mlxcel_core::cos(&angles)),
        table(mlxcel_core::sin(&angles)),
    )
}

/// `rotary * cos + concat(-second, first) * sin` over the first `rope_dim`
/// channels of `x` (`[B, H, L, head_dim]`); the rest pass through.
fn rotate(x: &MlxArray, cos: &MlxArray, sin: &MlxArray, rope_dim: i32) -> UniquePtr<MlxArray> {
    let head_dim = *mlxcel_core::array_shape(x).last().unwrap_or(&0);
    let rotary = slice_axis(x, -1, 0, rope_dim);
    let first = slice_axis(&rotary, -1, 0, rope_dim / 2);
    let second = slice_axis(&rotary, -1, rope_dim / 2, rope_dim);
    let swapped = mlxcel_core::concatenate(&mlxcel_core::negative(&second), &first, -1);
    let rotated = mlxcel_core::add(
        &mlxcel_core::multiply(&rotary, cos),
        &mlxcel_core::multiply(&swapped, sin),
    );
    // A no-op with upstream's activation-dtype tables; with f32 tables (the
    // parity tests' wrong-port control) it rounds back as `mx.fast.rope` does.
    let rotated = mlxcel_core::astype(&rotated, mlxcel_core::array_dtype(x));
    if rope_dim >= head_dim {
        return rotated;
    }
    let rest = slice_axis(x, -1, rope_dim, head_dim);
    mlxcel_core::concatenate(&rotated, &rest, -1)
}

impl Attention {
    /// `TextAttention.__call__` without a cache.
    fn forward_image(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let total_heads = self.num_heads + 2 * self.num_kv_heads;
        let qkv = mlxcel_core::reshape(
            &self.qkv_proj.forward(x),
            &[b, l, total_heads, self.head_dim],
        );
        let mut q = slice_axis(&qkv, 2, 0, self.num_heads);
        let mut k = slice_axis(&qkv, 2, self.num_heads, self.num_heads + self.num_kv_heads);
        let v = slice_axis(&qkv, 2, self.num_heads + self.num_kv_heads, total_heads);
        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            q = rms_then_gain(qn, &q);
            k = rms_then_gain(kn, &k);
        }
        let heads = |t: &MlxArray| mlxcel_core::transpose_axes(t, &[0, 2, 1, 3]);
        let (mut q, mut k, v) = (heads(&q), heads(&k), heads(&v));
        if self.rotary_dim > 0 {
            q = rotate(&q, cos, sin, self.rotary_dim);
            k = rotate(&k, cos, sin, self.rotary_dim);
        }
        // `head_dim ** -0.5` in double, as Python computes the scale.
        let scale = (f64::from(self.head_dim)).powf(-0.5) as f32;
        let out = mlxcel_core::layers::attention(&q, &k, &v, scale, mask, 0.0, 0);
        let out = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]),
            &[b, l, self.num_heads * self.head_dim],
        );
        self.dense.forward(&out)
    }
}

impl DenseMLP {
    /// `TextMLP`: `down(silu(gate(x)) * up(x))`.
    fn forward_image(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = mlxcel_core::silu(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(&mlxcel_core::multiply(&gate, &up))
    }
}

/// Upstream `_topk`: the `k` largest values of `x` along the last axis and
/// their indices, both ordered by descending value.
fn topk_sorted(x: &MlxArray, k: i32) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let partitioned = mlxcel_core::argpartition(&mlxcel_core::negative(x), k - 1, -1);
    let indices = slice_axis(&partitioned, -1, 0, k);
    let values = mlxcel_core::take_along_axis(x, &indices, -1);
    let order = mlxcel_core::argsort(&mlxcel_core::negative(&values), -1);
    (
        mlxcel_core::take_along_axis(&values, &order, -1),
        mlxcel_core::take_along_axis(&indices, &order, -1),
    )
}

impl MoEGate {
    /// `LLaDA2MoeGate.__call__` plus `TextMoE`'s score recomputation: the
    /// selected experts `[.., top_k]` (int32, descending biased score) and
    /// their f32 routing weights.
    fn route_image(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let x_f32 = mlxcel_core::astype(x, dtype::FLOAT32);
        let logits = mlxcel_core::matmul(&x_f32, &mlxcel_core::transpose(&self.weight));
        let scores = mlxcel_core::sigmoid(&logits);
        let biased = mlxcel_core::add(&scores, &self.expert_bias);

        // `_group_limited_topk`: keep the `topk_group` groups with the largest
        // top-2 sums, fill the rest with the f32 minimum, then sorted top-k.
        let shape = mlxcel_core::array_shape(&biased);
        let rank = shape.len();
        let num_experts = shape[rank - 1];
        let group_size = num_experts / self.n_group;
        let mut grouped_shape = shape[..rank - 1].to_vec();
        grouped_shape.extend([self.n_group, group_size]);
        let grouped = mlxcel_core::reshape(&biased, &grouped_shape);
        let (top2, _) = topk_sorted(&grouped, 2);
        let group_scores = mlxcel_core::sum_axis_native_dtype(&top2, -1, false);
        let (_, group_idx) = topk_sorted(&group_scores, self.topk_group);
        // `arange(E) // group_size == group_idx`, in int32 (exact integers).
        let group_of = mlxcel_core::floor_divide(
            &mlxcel_core::arange_i32(0, num_experts, 1),
            &mlxcel_core::from_slice_i32(&[group_size], &[1]),
        );
        let group_idx = mlxcel_core::astype(&group_idx, dtype::INT32);
        let hit = mlxcel_core::equal(&group_of, &mlxcel_core::expand_dims(&group_idx, -1));
        let keep = mlxcel_core::any_axis(&hit, -2, false);
        let floor = mlxcel_core::full_f32(&[1], f32::MIN, dtype::FLOAT32);
        let masked = mlxcel_core::where_cond(&keep, &biased, &floor);
        let (_, indices) = topk_sorted(&masked, self.top_k);
        let indices = mlxcel_core::astype(&indices, dtype::INT32);

        let mut weights = mlxcel_core::take_along_axis(&scores, &indices, -1);
        if self.top_k > 1 {
            let sum = mlxcel_core::sum_axis_native_dtype(&weights, -1, true);
            let eps = mlxcel_core::full_f32(&[1], 1e-20, dtype::FLOAT32);
            weights = mlxcel_core::divide(&weights, &mlxcel_core::add(&sum, &eps));
        }
        let scale = mlxcel_core::full_f32(&[1], self.routed_scaling_factor, dtype::FLOAT32);
        (indices, mlxcel_core::multiply(&weights, &scale))
    }
}

impl MoEBlock {
    /// `TextMoE.__call__` on `x` (`[B, L, D]`).
    fn forward_image(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let hidden = shape[shape.len() - 1];
        let (indices, scores) = self.gate.route_image(x);
        let top_k = *mlxcel_core::array_shape(&indices).last().unwrap_or(&1);
        let x_flat = mlxcel_core::reshape(x, &[-1, hidden]);
        let routed = self.switch_mlp.forward_prescaled_down(
            &x_flat,
            &mlxcel_core::reshape(&indices, &[-1]),
            &mlxcel_core::reshape(&scores, &[-1]),
            top_k,
        );
        let mut out_shape = mlxcel_core::array_shape(&indices);
        out_shape.push(hidden);
        let routed = mlxcel_core::reshape(&routed, &out_shape);
        let summed = mlxcel_core::sum_axis_native_dtype(
            &mlxcel_core::astype(&routed, dtype::FLOAT32),
            -2,
            false,
        );
        let output = mlxcel_core::astype(&summed, mlxcel_core::array_dtype(x));
        match &self.shared_experts {
            Some(shared) => mlxcel_core::add(&output, &shared.forward_image(x)),
            None => output,
        }
    }
}

impl Layer {
    fn forward_image(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let normed = rms_then_gain(&self.input_layernorm, x);
        let h = mlxcel_core::add(x, &self.attn.forward_image(&normed, mask, cos, sin));
        let normed = rms_then_gain(&self.post_attention_layernorm, &h);
        let ffn_out = match &self.ffn {
            FFN::Dense(mlp) => mlp.forward_image(&normed),
            FFN::MoE(moe) => moe.forward_image(&normed),
        };
        mlxcel_core::add(&h, &ffn_out)
    }
}

impl Llada2MoeModel {
    /// The word embeddings of `ids` (`model.word_embeddings`).
    ///
    /// Used by: LLaDAImage (prompt and VQ prior inputs).
    pub fn embed(&self, ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(ids)
    }

    /// The decoder over `embeds` (`[B, L, hidden]`) with the boolean
    /// attention `mask` (`[B or 1, 1, L, L]`, `true` = attend) and `positions`
    /// (`[B or 1, L]`), no KV cache; returns the final normed hidden state.
    /// Requires [`Llada2Numerics::LladaImage`]: the fused text path has no
    /// per-row positions.
    ///
    /// Used by: LLaDAImage (prompt encoding and the VQ prior).
    pub fn forward_embeds(
        &self,
        embeds: &MlxArray,
        mask: Option<&MlxArray>,
        positions: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        if self.numerics != Llada2Numerics::LladaImage {
            return Err("forward_embeds requires the LLaDA-Image numerics".to_string());
        }
        let shape = mlxcel_core::array_shape(embeds);
        let pos_shape = mlxcel_core::array_shape(positions);
        if shape.len() != 3 || pos_shape.len() != 2 || pos_shape[1] != shape[1] {
            return Err(format!(
                "forward_embeds expects [B, L, D] embeddings and [B, L] positions, got {shape:?} and {pos_shape:?}"
            ));
        }
        let (cos, sin) = self.image_rope_tables(positions, mlxcel_core::array_dtype(embeds));
        let mut h = mlxcel_core::copy(embeds);
        for layer in &self.layers {
            h = layer.forward_image(&h, mask, &cos, &sin);
        }
        Ok(rms_then_gain(&self.norm, &h))
    }

    /// The RoPE tables of every layer (they share one rotary dim and theta),
    /// rounded to `table_dtype`.
    fn image_rope_tables(
        &self,
        positions: &MlxArray,
        table_dtype: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let (rope_dim, theta) = self
            .layers
            .first()
            .map(|l| (l.attn.rotary_dim, l.attn.rope_theta))
            .unwrap_or((0, 1.0));
        rope_tables(positions, rope_dim, theta, table_dtype)
    }

    /// Decoder layer `index` alone on `h` (`[B, L, D]`), with the same mask
    /// and positions as [`Self::forward_embeds`]; no final norm. The parity
    /// tests feed it upstream's own layer inputs, so each layer is compared
    /// without the drift of the layers before it (the MoE routing amplifies
    /// one ulp into a 0.17 change of the final hidden state). `table_dtype`
    /// overrides the RoPE table dtype for a wrong-port control (`None` = the
    /// activation dtype, as upstream rounds them).
    ///
    /// Used by: LLaDAImage (parity tests).
    #[cfg(test)]
    pub(crate) fn forward_embeds_layer(
        &self,
        index: usize,
        h: &MlxArray,
        mask: Option<&MlxArray>,
        positions: &MlxArray,
        table_dtype: Option<i32>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        if self.numerics != Llada2Numerics::LladaImage {
            return Err("forward_embeds_layer requires the LLaDA-Image numerics".to_string());
        }
        let layer = self
            .layers
            .get(index)
            .ok_or_else(|| format!("no decoder layer {index} of {}", self.layers.len()))?;
        let dt = table_dtype.unwrap_or_else(|| mlxcel_core::array_dtype(h));
        let (cos, sin) = self.image_rope_tables(positions, dt);
        Ok(layer.forward_image(h, mask, &cos, &sin))
    }
}

#[cfg(test)]
#[path = "image_numerics_tests.rs"]
mod tests;
