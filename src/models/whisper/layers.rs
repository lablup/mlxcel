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

//! Shared transformer building blocks for the Whisper-style encoder-decoder.
//!
//! The encoder stacks self-attention-only blocks; the decoder stacks blocks
//! that add a cross-attention sublayer attending to the encoder output. Both
//! reuse the same [`MultiHeadAttention`] and [`ResidualAttentionBlock`] here,
//! differing only in whether the cross-attention sublayer is present.

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Per-sublayer key/value cache. Holds `[batch, length, n_state]` tensors that
/// are appended to (self-attention) or computed once (cross-attention).
pub(crate) struct KvCache {
    pub k: UniquePtr<MlxArray>,
    pub v: UniquePtr<MlxArray>,
}

fn linear(weights: &WeightMap, prefix: &str) -> Result<Linear, String> {
    Linear::from_weights(weights, prefix)
}

fn layer_norm(weights: &WeightMap, prefix: &str) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Whisper weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    // Whisper uses the standard LayerNorm epsilon.
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

/// Build a `[length, length]` additive causal mask in `dtype` (0 on and below
/// the diagonal, `-inf` above). Used only for the initial multi-token prefill;
/// incremental single-token steps need no mask.
pub(crate) fn additive_causal_mask(length: i32, dtype: i32) -> UniquePtr<MlxArray> {
    let l = length as usize;
    let mut data = vec![0.0f32; l * l];
    for i in 0..l {
        for j in (i + 1)..l {
            data[i * l + j] = f32::NEG_INFINITY;
        }
    }
    let mask = mlxcel_core::from_slice_f32(&data, &[length, length]);
    mlxcel_core::astype(&mask, dtype)
}

/// Scaled dot-product attention over `[batch, len, n_state]` projections.
///
/// `mask`, when present, is added to the `[lq, lk]` logits before softmax.
fn qkv_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    n_head: i32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let q_shape = mlxcel_core::array_shape(q);
    let k_shape = mlxcel_core::array_shape(k);
    let batch = q_shape[0];
    let lq = q_shape[1];
    let n_state = q_shape[2];
    let lk = k_shape[1];
    let head_dim = n_state / n_head;
    // The reference scales queries and keys each by head_dim^-0.25 so the
    // product is divided by sqrt(head_dim).
    let scale = (head_dim as f32).powf(-0.25);

    // q -> [b, h, lq, hd]
    let q = mlxcel_core::reshape(q, &[batch, lq, n_head, head_dim]);
    let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
    let q = mlxcel_core::multiply_scalar(&q, scale);
    // k -> [b, h, hd, lk]
    let k = mlxcel_core::reshape(k, &[batch, lk, n_head, head_dim]);
    let k = mlxcel_core::transpose_axes(&k, &[0, 2, 3, 1]);
    let k = mlxcel_core::multiply_scalar(&k, scale);
    // v -> [b, h, lk, hd]
    let v = mlxcel_core::reshape(v, &[batch, lk, n_head, head_dim]);
    let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

    let mut qk = mlxcel_core::matmul(&q, &k);
    if let Some(mask) = mask {
        qk = mlxcel_core::add(&qk, mask);
    }
    let weights = mlxcel_core::softmax_precise(&qk, -1);
    let out = mlxcel_core::matmul(&weights, &v);
    let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
    mlxcel_core::reshape(&out, &[batch, lq, n_state])
}

/// Multi-head attention with separate query/key/value/output projections.
/// `key` has no bias in the Whisper architecture; the loader picks that up
/// automatically because [`Linear`] treats `.bias` as optional.
pub(crate) struct MultiHeadAttention {
    n_head: i32,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
}

impl MultiHeadAttention {
    fn from_weights(weights: &WeightMap, prefix: &str, n_head: i32) -> Result<Self, String> {
        Ok(Self {
            n_head,
            query: linear(weights, &format!("{prefix}.query"))?,
            key: linear(weights, &format!("{prefix}.key"))?,
            value: linear(weights, &format!("{prefix}.value"))?,
            out: linear(weights, &format!("{prefix}.out"))?,
        })
    }

    /// Self-attention. When `cache` is present, the freshly projected key/value
    /// are appended to the cached history before attending.
    fn self_attention(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Option<KvCache>,
    ) -> UniquePtr<MlxArray> {
        let q = self.query.forward(x);
        let k_new = self.key.forward(x);
        let v_new = self.value.forward(x);
        let (k, v) = match cache.take() {
            Some(prev) => (
                mlxcel_core::concatenate(&prev.k, &k_new, 1),
                mlxcel_core::concatenate(&prev.v, &v_new, 1),
            ),
            None => (k_new, v_new),
        };
        let out = qkv_attention(&q, &k, &v, self.n_head, mask);
        *cache = Some(KvCache { k, v });
        self.out.forward(&out)
    }

    /// Cross-attention against the encoder output `xa`. Key/value are computed
    /// once and reused across decode steps via `cache`.
    fn cross_attention(
        &self,
        x: &MlxArray,
        xa: &MlxArray,
        cache: &mut Option<KvCache>,
    ) -> UniquePtr<MlxArray> {
        let q = self.query.forward(x);
        let (k, v) = match cache.take() {
            Some(prev) => (prev.k, prev.v),
            None => (self.key.forward(xa), self.value.forward(xa)),
        };
        let out = qkv_attention(&q, &k, &v, self.n_head, None);
        *cache = Some(KvCache { k, v });
        self.out.forward(&out)
    }
}

/// One pre-norm transformer block. Encoder blocks omit cross-attention; decoder
/// blocks include it.
pub(crate) struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<MultiHeadAttention>,
    cross_attn_ln: Option<LayerNorm>,
    mlp1: Linear,
    mlp2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        n_head: i32,
        cross_attention: bool,
    ) -> Result<Self, String> {
        let (cross_attn, cross_attn_ln) = if cross_attention {
            (
                Some(MultiHeadAttention::from_weights(
                    weights,
                    &format!("{prefix}.cross_attn"),
                    n_head,
                )?),
                Some(layer_norm(weights, &format!("{prefix}.cross_attn_ln"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            attn: MultiHeadAttention::from_weights(weights, &format!("{prefix}.attn"), n_head)?,
            attn_ln: layer_norm(weights, &format!("{prefix}.attn_ln"))?,
            cross_attn,
            cross_attn_ln,
            mlp1: linear(weights, &format!("{prefix}.mlp1"))?,
            mlp2: linear(weights, &format!("{prefix}.mlp2"))?,
            mlp_ln: layer_norm(weights, &format!("{prefix}.mlp_ln"))?,
        })
    }

    /// Forward pass.
    ///
    /// `self_cache` / `cross_cache` carry per-block decode state; pass `&mut None`
    /// for the encoder (cacheless) path.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        xa: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        self_cache: &mut Option<KvCache>,
        cross_cache: &mut Option<KvCache>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.attn_ln.forward(x);
        let y = self.attn.self_attention(&normed, mask, self_cache);
        let mut x = mlxcel_core::add(x, &y);

        if let (Some(cross_attn), Some(cross_attn_ln)) = (&self.cross_attn, &self.cross_attn_ln) {
            let xa = xa.expect("decoder cross-attention requires encoder features");
            let normed = cross_attn_ln.forward(&x);
            let y = cross_attn.cross_attention(&normed, xa, cross_cache);
            x = mlxcel_core::add(&x, &y);
        }

        let h = self.mlp_ln.forward(&x);
        let h = self.mlp1.forward(&h);
        let h = mlxcel_core::gelu(&h);
        let h = self.mlp2.forward(&h);
        mlxcel_core::add(&x, &h)
    }
}
