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

//! BART-style post-norm transformer building blocks for the Florence-2
//! seq2seq engine.
//!
//! Unlike the pre-norm Whisper blocks in `crate::models::whisper::layers`,
//! BART applies LayerNorm *after* each residual add (post-norm ordering):
//! `x = LN(x + Sublayer(x))`. Encoder blocks are bidirectional
//! self-attention only; decoder blocks add causal masking and a
//! cross-attention sublayer over the encoder output with its own LayerNorm.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/language.py

use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Florence2Quantization;

/// Per-sublayer key/value cache holding `[batch, length, d_model]` tensors
/// (projected but not yet split into heads). Self-attention caches append per
/// decode step; cross-attention caches are computed once from the encoder
/// output and reused unchanged on every subsequent step.
pub(crate) struct KvCache {
    pub k: UniquePtr<MlxArray>,
    pub v: UniquePtr<MlxArray>,
}

/// Dual per-layer decode cache: the growing self-attention K/V plus the
/// fixed cross-attention K/V. Together these are what makes the seq2seq
/// decode loop O(1) per step instead of re-encoding history.
#[derive(Default)]
pub(crate) struct Florence2LayerCache {
    pub(crate) self_kv: Option<KvCache>,
    pub(crate) cross_kv: Option<KvCache>,
}

/// Build one projection.
///
/// [`UnifiedLinear`] decides per prefix: it takes the quantized path when the
/// weight map holds `{prefix}.scales` and falls back to a dense `Linear`
/// otherwise, so this is the single call for both a bf16 export and a
/// `-3bit` / `-4bit` / `-6bit` / `-8bit` one. `{prefix}.bias`, which every
/// BART projection ships, is picked up on both arms.
fn linear(
    weights: &WeightMap,
    prefix: &str,
    quantization: Florence2Quantization,
) -> Result<UnifiedLinear, String> {
    UnifiedLinear::from_weights(weights, prefix, quantization.group_size, quantization.bits)
}

/// Check a loaded (possibly quantized) embedding table against the row count
/// that bounds lookups into it and the width it must have, and return its
/// actual row count.
///
/// Thin adapter over the shared `validate_embedding_table` guard. The reason
/// Florence-2 needs it at all is that a packed table's `shape[1]` is a
/// function of the bit depth rather than the model width, so the dense
/// `cols == expected` comparison the family used before is wrong for a
/// quantized checkpoint; the guard's quantized arm reconstructs the width MLX
/// will compute from `scales` and the reconciled group size instead.
///
/// `claimed_rows` is the largest index the forward path can produce plus one
/// (`POSITION_OFFSET + max_position_embeddings` for the BART position tables,
/// `vocab_size` for the shared token table). Pass `1` for a table whose only
/// bound is its own row count, which is how the fusion-stage 2-D position
/// tables work.
pub(crate) fn embedding_table_rows(
    table: &UnifiedEmbedding,
    key: &str,
    claimed_rows: i64,
    claimed_rows_field: &str,
    expected_cols: i32,
    width_field: &str,
) -> Result<i32, String> {
    let claimed_rows = usize::try_from(claimed_rows)
        .map_err(|_| format!("Florence-2 {key}: negative row bound {claimed_rows}"))?;
    let expected_cols = usize::try_from(expected_cols)
        .map_err(|_| format!("Florence-2 {key}: negative expected width {expected_cols}"))?;
    let rows = crate::models::gpt2::validate_embedding_table(
        table,
        key,
        claimed_rows,
        claimed_rows_field,
        expected_cols,
        width_field,
    )
    .map_err(|e| format!("Florence-2 {e}"))?;
    i32::try_from(rows).map_err(|_| format!("Florence-2 {key}: {rows} rows does not fit in i32"))
}

pub(crate) fn layer_norm(weights: &WeightMap, prefix: &str) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Florence-2 weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    // BART uses the standard LayerNorm epsilon.
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

/// Build a `[length, offset + length]` additive causal mask in `dtype`
/// (0 where query position `i` may attend key position `j`, i.e.
/// `j <= offset + i`, and `-inf` above). Only a multi-token decoder call
/// needs it; an incremental single-token step attends to the whole cached
/// history unmasked.
pub(crate) fn additive_causal_mask(length: i32, offset: i32, dtype: i32) -> UniquePtr<MlxArray> {
    let l = length as usize;
    let total = (offset + length) as usize;
    let mut data = vec![0.0f32; l * total];
    for i in 0..l {
        for j in (offset as usize + i + 1)..total {
            data[i * total + j] = f32::NEG_INFINITY;
        }
    }
    let mask = mlxcel_core::from_slice_f32(&data, &[length, offset + length]);
    mlxcel_core::astype(&mask, dtype)
}

/// Scaled dot-product attention over `[batch, len, d_model]` projections.
///
/// Queries are pre-scaled by `head_dim^-0.5` (the reference passes the same
/// scale into `mx.fast.scaled_dot_product_attention`). `mask`, when present,
/// is added to the `[lq, lk]` logits before softmax.
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
    let d_model = q_shape[2];
    let lk = k_shape[1];
    let head_dim = d_model / n_head;
    let scale = (head_dim as f32).powf(-0.5);

    // q -> [b, h, lq, hd], scaled.
    let q = mlxcel_core::reshape(q, &[batch, lq, n_head, head_dim]);
    let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
    let q = mlxcel_core::multiply_scalar(&q, scale);
    // k -> [b, h, hd, lk]
    let k = mlxcel_core::reshape(k, &[batch, lk, n_head, head_dim]);
    let k = mlxcel_core::transpose_axes(&k, &[0, 2, 3, 1]);
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
    mlxcel_core::reshape(&out, &[batch, lq, d_model])
}

/// Multi-head attention with separate q/k/v/out projections (all biased in
/// BART checkpoints; [`Linear`] treats `.bias` as optional so unbiased
/// exports also load).
pub(crate) struct Florence2Attention {
    n_head: i32,
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
}

impl Florence2Attention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        n_head: i32,
        quantization: Florence2Quantization,
    ) -> Result<Self, String> {
        Ok(Self {
            n_head,
            q_proj: linear(weights, &format!("{prefix}.q_proj"), quantization)?,
            k_proj: linear(weights, &format!("{prefix}.k_proj"), quantization)?,
            v_proj: linear(weights, &format!("{prefix}.v_proj"), quantization)?,
            out_proj: linear(weights, &format!("{prefix}.out_proj"), quantization)?,
        })
    }

    /// Self-attention. When `cache` is present (decoder), the freshly
    /// projected key/value are appended to the cached history before
    /// attending; the encoder passes `&mut None`.
    pub(crate) fn self_attention(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Option<KvCache>,
    ) -> UniquePtr<MlxArray> {
        let q = self.q_proj.forward(x);
        let k_new = self.k_proj.forward(x);
        let v_new = self.v_proj.forward(x);
        let (k, v) = match cache.take() {
            Some(prev) => (
                mlxcel_core::concatenate(&prev.k, &k_new, 1),
                mlxcel_core::concatenate(&prev.v, &v_new, 1),
            ),
            None => (k_new, v_new),
        };
        let out = qkv_attention(&q, &k, &v, self.n_head, mask);
        *cache = Some(KvCache { k, v });
        self.out_proj.forward(&out)
    }

    /// Cross-attention against the encoder output `xa`. Key/value are
    /// projected once and reused across decode steps via `cache`.
    pub(crate) fn cross_attention(
        &self,
        x: &MlxArray,
        xa: &MlxArray,
        cache: &mut Option<KvCache>,
    ) -> UniquePtr<MlxArray> {
        let q = self.q_proj.forward(x);
        let (k, v) = match cache.take() {
            Some(prev) => (prev.k, prev.v),
            None => (self.k_proj.forward(xa), self.v_proj.forward(xa)),
        };
        let out = qkv_attention(&q, &k, &v, self.n_head, None);
        *cache = Some(KvCache { k, v });
        self.out_proj.forward(&out)
    }
}

/// One bidirectional encoder block, BART post-norm ordering:
/// `x = LN(x + SelfAttn(x)); x = LN(x + FFN(x))`.
pub(crate) struct Florence2EncoderLayer {
    self_attn: Florence2Attention,
    self_attn_layer_norm: LayerNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    final_layer_norm: LayerNorm,
}

impl Florence2EncoderLayer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        n_head: i32,
        quantization: Florence2Quantization,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: Florence2Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                n_head,
                quantization,
            )?,
            self_attn_layer_norm: layer_norm(weights, &format!("{prefix}.self_attn_layer_norm"))?,
            fc1: linear(weights, &format!("{prefix}.fc1"), quantization)?,
            fc2: linear(weights, &format!("{prefix}.fc2"), quantization)?,
            final_layer_norm: layer_norm(weights, &format!("{prefix}.final_layer_norm"))?,
        })
    }

    /// `mask`, when present, is the additive attention mask broadcast over
    /// the `[batch, head, query, key]` logits (`0` for a real key, `-inf` for
    /// a padded one). The fused vision + prompt path builds it from the joint
    /// attention mask; a plain text encode passes `None`.
    pub(crate) fn forward(&self, x: &MlxArray, mask: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let y = self.self_attn.self_attention(x, mask, &mut None);
        let x = mlxcel_core::add(x, &y);
        let x = self.self_attn_layer_norm.forward(&x);

        let h = self.fc1.forward(&x);
        let h = mlxcel_core::gelu(&h);
        let h = self.fc2.forward(&h);
        let x = mlxcel_core::add(&x, &h);
        self.final_layer_norm.forward(&x)
    }
}

/// One decoder block: causal self-attention, encoder cross-attention, and
/// GELU FFN, each followed by its own post-norm LayerNorm.
pub(crate) struct Florence2DecoderLayer {
    self_attn: Florence2Attention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: Florence2Attention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    final_layer_norm: LayerNorm,
}

impl Florence2DecoderLayer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        n_head: i32,
        quantization: Florence2Quantization,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: Florence2Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                n_head,
                quantization,
            )?,
            self_attn_layer_norm: layer_norm(weights, &format!("{prefix}.self_attn_layer_norm"))?,
            encoder_attn: Florence2Attention::from_weights(
                weights,
                &format!("{prefix}.encoder_attn"),
                n_head,
                quantization,
            )?,
            encoder_attn_layer_norm: layer_norm(
                weights,
                &format!("{prefix}.encoder_attn_layer_norm"),
            )?,
            fc1: linear(weights, &format!("{prefix}.fc1"), quantization)?,
            fc2: linear(weights, &format!("{prefix}.fc2"), quantization)?,
            final_layer_norm: layer_norm(weights, &format!("{prefix}.final_layer_norm"))?,
        })
    }

    /// Forward one decoder block. `xa` is the encoder output attended by the
    /// cross-attention sublayer; `cache` carries this layer's dual decode
    /// state (self-attention history + one-shot cross-attention K/V).
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        xa: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Florence2LayerCache,
    ) -> UniquePtr<MlxArray> {
        let y = self.self_attn.self_attention(x, mask, &mut cache.self_kv);
        let x = mlxcel_core::add(x, &y);
        let x = self.self_attn_layer_norm.forward(&x);

        let y = self
            .encoder_attn
            .cross_attention(&x, xa, &mut cache.cross_kv);
        let x = mlxcel_core::add(&x, &y);
        let x = self.encoder_attn_layer_norm.forward(&x);

        let h = self.fc1.forward(&x);
        let h = mlxcel_core::gelu(&h);
        let h = self.fc2.forward(&h);
        let x = mlxcel_core::add(&x, &h);
        self.final_layer_norm.forward(&x)
    }
}
