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

//! Layers of the OpenAI Privacy Filter encoder: banded bidirectional
//! attention with sinks and the f32 top-4 MoE block. See
//! [`super::openai_privacy_filter`] for the model and the upstream reference.

use mlxcel_core::dtype;
use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gpt_oss_common::{ExpertLinear, gpt_oss_swiglu};
use crate::models::switch_layers::{gather_sort, moe_weighted_sum, scatter_unsort};

use super::openai_privacy_filter::PrivacyFilterArgs;

/// Boolean band mask `|q - k| <= radius` for query positions
/// `[query_start, query_end)` against key positions `[key_start, key_end)`,
/// shaped `[1, 1, Lq, Lk]` for SDPA.
pub(crate) fn band_mask(
    query_start: i32,
    query_end: i32,
    key_start: i32,
    key_end: i32,
    radius: i32,
) -> UniquePtr<MlxArray> {
    let queries = mlxcel_core::arange_i32(query_start, query_end, 1);
    let queries = mlxcel_core::reshape(&queries, &[query_end - query_start, 1]);
    let keys = mlxcel_core::arange_i32(key_start, key_end, 1);
    let keys = mlxcel_core::reshape(&keys, &[1, key_end - key_start]);
    let distance = mlxcel_core::abs(&mlxcel_core::subtract(&queries, &keys));
    let radius = mlxcel_core::from_slice_i32(&[radius], &[1]);
    let mask = mlxcel_core::less_equal(&distance, &radius);
    mlxcel_core::reshape(&mask, &[1, 1, query_end - query_start, key_end - key_start])
}

/// Symmetric local attention in query chunks, as upstream's
/// `_bidirectional_local_attention`: each chunk of `chunk_size` queries
/// attends to the keys `[start - radius, end + radius)` under the band mask,
/// so no `T x T` mask is ever built. `q`, `k`, `v` are `[B, heads, T, D]`
/// (GQA allowed), `sinks` is cast to the attention dtype at the kernel, and
/// the scale is 1 because q and k already carry `head_dim^-0.25` each.
///
/// The pipeline scores one unpadded window at a time, so there is no padding
/// mask to intersect with the band.
pub(crate) fn banded_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    sinks: &MlxArray,
    radius: i32,
    chunk_size: i32,
) -> UniquePtr<MlxArray> {
    let q_shape = mlxcel_core::array_shape(q);
    let k_shape = mlxcel_core::array_shape(k);
    let (batch, heads, seq_len, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let kv_heads = k_shape[1];
    let sinks = mlxcel_core::astype(sinks, mlxcel_core::array_dtype(q));
    let sinks_ptr = sinks.as_ref().unwrap() as *const MlxArray;

    let mut chunks: Vec<UniquePtr<MlxArray>> = Vec::new();
    let mut query_start = 0;
    while query_start < seq_len {
        let query_end = (query_start + chunk_size).min(seq_len);
        let key_start = (query_start - radius).max(0);
        let key_end = (query_end + radius).min(seq_len);
        let mask = band_mask(query_start, query_end, key_start, key_end, radius);
        let q_chunk = mlxcel_core::slice(
            q,
            &[0, 0, query_start, 0],
            &[batch, heads, query_end, head_dim],
        );
        let k_chunk = mlxcel_core::slice(
            k,
            &[0, 0, key_start, 0],
            &[batch, kv_heads, key_end, head_dim],
        );
        let v_chunk = mlxcel_core::slice(
            v,
            &[0, 0, key_start, 0],
            &[batch, kv_heads, key_end, head_dim],
        );
        let mask_ptr = mask.as_ref().unwrap() as *const MlxArray;
        // SAFETY: `mask` and `sinks` outlive the call and point at live arrays.
        let out = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention_with_sinks(
                &q_chunk, &k_chunk, &v_chunk, 1.0, mask_ptr, sinks_ptr,
            )
        };
        chunks.push(out);
        query_start = query_end;
    }
    if chunks.len() == 1 {
        return chunks.pop().expect("one chunk");
    }
    let refs: Vec<&MlxArray> = chunks.iter().map(|chunk| chunk.as_ref().unwrap()).collect();
    mlxcel_core::concatenate_many(&refs, 2)
}

/// Biased GQA attention with interleaved YaRN RoPE, per-head sinks and the
/// symmetric sliding-window band.
pub(crate) struct PrivacyFilterAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    /// `[num_attention_heads]`, kept in its stored dtype.
    sinks: UniquePtr<MlxArray>,
    rope_freqs: UniquePtr<MlxArray>,
    /// `1 + 0.1 ln(factor)`, multiplied into q and k before rotation.
    attention_factor: f32,
    /// `head_dim^-0.25`, applied to both q and k so SDPA runs at scale 1.
    qk_scale: f32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    radius: i32,
    chunk_size: i32,
}

impl PrivacyFilterAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &PrivacyFilterArgs,
        prefix: &str,
        rope_freqs: &MlxArray,
        attention_factor: f32,
    ) -> Result<Self, String> {
        let linear = |name: &str| -> Result<UnifiedLinear, String> {
            let full = format!("{prefix}.{name}");
            let (group_size, bits, mode) = args.quant_for(&full);
            UnifiedLinear::from_weights_with_mode(weights, &full, group_size, bits, &mode)
        };
        let sinks = weights
            .get(&format!("{prefix}.sinks"))
            .map(|w| mlxcel_core::copy(w))
            .unwrap_or_else(|| {
                mlxcel_core::zeros(&[args.num_attention_heads as i32], dtype::FLOAT32)
            });
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            o_proj: linear("o_proj")?,
            sinks,
            rope_freqs: mlxcel_core::copy(rope_freqs),
            attention_factor,
            qk_scale: (args.head_dim as f32).powf(-0.25),
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim: args.head_dim as i32,
            radius: args.sliding_window as i32,
            chunk_size: args.attention_chunk_size as i32,
        })
    }

    /// `PrivacyFilterRoPE` then the `head_dim^-0.25` scale. Both scalars are
    /// materialized in the activation dtype, as upstream's
    /// `mx.array(factor, dtype=x.dtype)` and weak Python float do.
    fn rope(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let scaled = if self.attention_factor != 1.0 {
            mlxcel_core::multiply_scalar(x, self.attention_factor)
        } else {
            mlxcel_core::copy(x)
        };
        let rotated = mlxcel_core::fast_rope_with_freqs(
            &scaled,
            self.head_dim,
            true,
            1.0,
            0,
            &self.rope_freqs,
        );
        mlxcel_core::multiply_scalar(&rotated, self.qk_scale)
    }

    /// `x` is `[B, T, hidden]`.
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);
        let q = self.rope(&q);
        let k = self.rope(&k);
        let out = banded_attention(&q, &k, &v, &self.sinks, self.radius, self.chunk_size);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }
}

/// Top-k router plus biased experts, all in f32 as the reference model.
///
/// Routed rows are sorted by expert before the gathers (`gather_sort` +
/// `sorted = true`) whenever a chunk has at least 64 of them, the
/// `_gather_sort` threshold upstream uses, on every plane. Dense planes used
/// to gather unsorted: on CUDA the sorted `gather_mm` branch
/// (`gather_mm_rhs`) read the stride of the activations' size-1 axis as the
/// GEMM's leading dimension, and MLX leaves that stride at whatever the
/// producer had. The per-expert bias add hands down a `[rows, 1, N]` with a
/// unit stride on that axis (the bias is `expand_dims(take(bias, ids), -2)`,
/// and which operand the add inherits its strides from depends on which one
/// the allocator let it donate), so every row after the first of an expert
/// group read shifted activations: PR #2460's "allocation-history-dependent"
/// drift. Issue #2462 fixed the overlay to take the batch stride, so the
/// dense checkpoints get the sorted kernel back (about 1.5x faster on a
/// 7.5k-token text).
pub(crate) struct PrivacyFilterMoe {
    router: UnifiedLinear,
    gate_proj: ExpertLinear,
    up_proj: ExpertLinear,
    down_proj: ExpertLinear,
    top_k: i32,
    chunk_size: i32,
}

impl PrivacyFilterMoe {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &PrivacyFilterArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let router_prefix = format!("{prefix}.router");
        let (group_size, bits, mode) = args.quant_for(&router_prefix);
        let router = UnifiedLinear::from_weights_with_mode(
            weights,
            &router_prefix,
            group_size,
            bits,
            &mode,
        )?;
        let (group_size, bits, mode) = args.expert_quantization();
        let expert = |name: &str| {
            ExpertLinear::from_weights(
                weights,
                &format!("{prefix}.experts.{name}"),
                group_size,
                bits,
                &mode,
            )
        };
        // The router's output width is the number of experts the gathers index;
        // a plane with fewer experts would be read out of bounds on the GPU.
        let leading = |key: &str| -> Result<usize, String> {
            weights
                .get(key)
                .map(|w| mlxcel_core::array_shape(w)[0] as usize)
                .ok_or_else(|| format!("Missing weight: {key}"))
        };
        let router_rows = leading(&format!("{router_prefix}.weight"))?;
        for name in ["gate_proj", "up_proj", "down_proj"] {
            let experts = leading(&format!("{prefix}.experts.{name}.weight"))?;
            if experts != router_rows || experts != args.num_local_experts {
                return Err(format!(
                    "{prefix}: router has {router_rows} outputs, experts.{name} has {experts} \
                     experts, config says {}",
                    args.num_local_experts
                ));
            }
        }
        let (gate_proj, up_proj, down_proj) = (
            expert("gate_proj")?,
            expert("up_proj")?,
            expert("down_proj")?,
        );
        Ok(Self {
            router,
            gate_proj,
            up_proj,
            down_proj,
            top_k: args.num_experts_per_tok as i32,
            chunk_size: args.moe_chunk_size as i32,
        })
    }

    /// Router logits in f32, top-k by logit sorted descending, softmax weights.
    fn route(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let logits = self.router.forward(x);
        let logits = mlxcel_core::astype(&logits, dtype::FLOAT32);
        let shape = mlxcel_core::array_shape(&logits);
        let (tokens, experts) = (shape[0], shape[1]);
        let kth = experts - self.top_k;
        let partitioned = mlxcel_core::argpartition(&logits, kth, -1);
        let indices = mlxcel_core::slice(&partitioned, &[0, kth], &[tokens, experts]);
        let top_logits = mlxcel_core::take_along_axis(&logits, &indices, -1);
        let order = mlxcel_core::argsort(&mlxcel_core::negative(&top_logits), -1);
        let indices = mlxcel_core::take_along_axis(&indices, &order, -1);
        let top_logits = mlxcel_core::take_along_axis(&top_logits, &order, -1);
        // Upstream divides by top_k here and multiplies the sum back; the two
        // power-of-two scalings cancel exactly in f32.
        let weights = mlxcel_core::softmax_precise(&top_logits, -1);
        (indices, weights)
    }

    /// `x` is `[tokens, hidden]` f32; returns `[tokens, top_k, hidden]` f32.
    fn experts(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let index_shape = mlxcel_core::array_shape(indices);
        let (tokens, top_k) = (index_shape[0], index_shape[1]);
        let hidden = mlxcel_core::array_shape(x)[1];
        let x = mlxcel_core::reshape(x, &[tokens, 1, 1, hidden]);
        let f32 = dtype::FLOAT32;
        if tokens * top_k >= 64 {
            let (sorted_x, sorted_indices, inverse) = gather_sort(&x, indices);
            let gate = mlxcel_core::astype(
                &self.gate_proj.forward(&sorted_x, &sorted_indices, true),
                f32,
            );
            let up =
                mlxcel_core::astype(&self.up_proj.forward(&sorted_x, &sorted_indices, true), f32);
            let activated = gpt_oss_swiglu(&up, &gate, 7.0);
            let out = self.down_proj.forward(&activated, &sorted_indices, true);
            scatter_unsort(&mlxcel_core::astype(&out, f32), &inverse, &index_shape)
        } else {
            let gate = mlxcel_core::astype(&self.gate_proj.forward(&x, indices, false), f32);
            let up = mlxcel_core::astype(&self.up_proj.forward(&x, indices, false), f32);
            let activated = gpt_oss_swiglu(&up, &gate, 7.0);
            let out = self.down_proj.forward(&activated, indices, false);
            mlxcel_core::squeeze_axis(&mlxcel_core::astype(&out, f32), -2)
        }
    }

    fn forward_chunk(&self, x: &MlxArray, output_dtype: i32) -> UniquePtr<MlxArray> {
        let x = mlxcel_core::astype(x, dtype::FLOAT32);
        let (indices, weights) = self.route(&x);
        let expert_out = self.experts(&x, &indices);
        moe_weighted_sum(&expert_out, &weights, output_dtype)
    }

    /// `x` is `[B, T, hidden]`; tokens are routed in chunks of `moe_chunk_size`.
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let hidden = shape[shape.len() - 1];
        let tokens: i32 = shape[..shape.len() - 1].iter().product();
        let flat = mlxcel_core::reshape(x, &[tokens, hidden]);
        let output_dtype = mlxcel_core::array_dtype(x);
        let mut chunks: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut start = 0;
        while start < tokens {
            let end = (start + self.chunk_size).min(tokens);
            let rows = mlxcel_core::slice(&flat, &[start, 0], &[end, hidden]);
            chunks.push(self.forward_chunk(&rows, output_dtype));
            start = end;
        }
        let out = if chunks.len() == 1 {
            chunks.pop().expect("one chunk")
        } else {
            let refs: Vec<&MlxArray> = chunks.iter().map(|chunk| chunk.as_ref().unwrap()).collect();
            mlxcel_core::concatenate_many(&refs, 0)
        };
        mlxcel_core::reshape(&out, &shape)
    }
}

/// One pre-norm encoder layer.
pub(crate) struct PrivacyFilterLayer {
    self_attn: PrivacyFilterAttention,
    mlp: PrivacyFilterMoe,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl PrivacyFilterLayer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &PrivacyFilterArgs,
        index: usize,
        rope_freqs: &MlxArray,
        attention_factor: f32,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{index}");
        let norm = |name: &str| -> Result<RMSNorm, String> {
            let key = format!("{prefix}.{name}.weight");
            let weight = weights
                .get(&key)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Missing weight: {key}"))?;
            Ok(RMSNorm::new(weight, args.rms_norm_eps))
        };
        Ok(Self {
            self_attn: PrivacyFilterAttention::from_weights(
                weights,
                args,
                &format!("{prefix}.self_attn"),
                rope_freqs,
                attention_factor,
            )?,
            mlp: PrivacyFilterMoe::from_weights(weights, args, &format!("{prefix}.mlp"))?,
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let attn = self.self_attn.forward(&self.input_layernorm.forward(x));
        let h = mlxcel_core::add(x, &attn);
        let mlp = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &mlp)
    }
}
