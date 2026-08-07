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

//! Positional embeddings applied to DaViT image features before they are
//! projected into the text embedding space.
//!
//! Both live between the vision tower and the BART encoder, and neither is
//! part of either half: the backbone emits raw `[B, H*W, dim_embed[-1]]`
//! tokens with no notion of where a token sat in the grid, and the text
//! encoder adds its own 1-D BART position table on top of the fused
//! sequence. These two supply the *image-side* geometry.
//!
//! - [`LearnedPositionEmbedding2D`] adds a learned row + column embedding to
//!   each token of a square feature map (`image_pos_embed.*` in the
//!   checkpoint, `{"type": "learned_abs_2d", "max_pos_embeddings": 50}`).
//! - [`PositionalEmbeddingCosine1D`] adds a fixed sinusoidal embedding over
//!   the *frame* axis (`visual_temporal_embed.pos_idx_to_embed`,
//!   `{"type": "COSINE", "max_temporal_embeddings": 100}`). Still frames use
//!   a single frame, so this contributes row 0 (`sin(0), cos(0), ...`), which
//!   is not the zero vector and must be applied for parity.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/florence2.py

use mlxcel_core::layers::Embedding;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Learned 2-D absolute position embedding over a `H x W` feature grid.
///
/// The row and column tables are each half of the feature width; a token at
/// `(y, x)` gets `concat(column[x], row[y])`. Column comes first, matching
/// the reference (and the HuggingFace `Florence2` implementation it was
/// ported from), so swapping the halves would silently mis-place every
/// image token.
pub(crate) struct LearnedPositionEmbedding2D {
    row_embeddings: Embedding,
    column_embeddings: Embedding,
    /// Rows in each table, i.e. the largest grid side that can be embedded.
    num_pos: i32,
    row_dim: i32,
    column_dim: i32,
}

impl LearnedPositionEmbedding2D {
    /// Load `{prefix}.row_embeddings.weight` and
    /// `{prefix}.column_embeddings.weight`, checking that the two halves add
    /// up to `embedding_dim` (the vision tower's output width).
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        embedding_dim: i32,
    ) -> Result<Self, String> {
        let row_embeddings = Embedding::from_weights(weights, &format!("{prefix}.row_embeddings"))?;
        let column_embeddings =
            Embedding::from_weights(weights, &format!("{prefix}.column_embeddings"))?;
        let row_shape = mlxcel_core::array_shape(&row_embeddings.weight);
        let column_shape = mlxcel_core::array_shape(&column_embeddings.weight);
        if row_shape.len() != 2 || column_shape.len() != 2 {
            return Err(format!(
                "Florence-2 {prefix}: position tables must be 2-D, got {row_shape:?} / {column_shape:?}"
            ));
        }
        if row_shape[0] != column_shape[0] {
            return Err(format!(
                "Florence-2 {prefix}: row table has {} positions but column table has {}",
                row_shape[0], column_shape[0]
            ));
        }
        // Upstream splits as `embedding_dim // 2` rows and the remainder
        // columns; both halves are 512 for the 1024-wide base-ft tower.
        if row_shape[1] + column_shape[1] != embedding_dim {
            return Err(format!(
                "Florence-2 {prefix}: row width {} + column width {} != image feature width {embedding_dim}",
                row_shape[1], column_shape[1]
            ));
        }
        Ok(Self {
            row_embeddings,
            column_embeddings,
            num_pos: row_shape[0],
            row_dim: row_shape[1],
            column_dim: column_shape[1],
        })
    }

    /// Build the `[1, height, width, embedding_dim]` position tensor for a
    /// grid of this size. Broadcasting over the batch axis is left to the
    /// caller's `add`, which is what the reference's explicit
    /// `broadcast_to(..., (batch, h, w, c))` amounts to.
    pub(crate) fn forward(&self, height: i32, width: i32) -> Result<UniquePtr<MlxArray>, String> {
        if height < 1 || width < 1 {
            return Err(format!(
                "Florence-2 image position embedding: grid {height}x{width} must be positive"
            ));
        }
        if height > self.num_pos || width > self.num_pos {
            return Err(format!(
                "Florence-2 image position embedding: grid {height}x{width} exceeds max_pos_embeddings {}",
                self.num_pos
            ));
        }

        let column_pos = mlxcel_core::arange_i32(0, width, 1);
        let row_pos = mlxcel_core::arange_i32(0, height, 1);
        let x_emb = self.column_embeddings.forward(&column_pos);
        let y_emb = self.row_embeddings.forward(&row_pos);

        // column: [W, Dc] -> [1, W, Dc] -> [H, W, Dc]
        let x_emb = mlxcel_core::reshape(&x_emb, &[1, width, self.column_dim]);
        let x_emb = mlxcel_core::broadcast_to(&x_emb, &[height, width, self.column_dim]);
        // row: [H, Dr] -> [H, 1, Dr] -> [H, W, Dr]
        let y_emb = mlxcel_core::reshape(&y_emb, &[height, 1, self.row_dim]);
        let y_emb = mlxcel_core::broadcast_to(&y_emb, &[height, width, self.row_dim]);

        let pos = mlxcel_core::concatenate(&x_emb, &y_emb, 2);
        Ok(mlxcel_core::reshape(
            &pos,
            &[1, height, width, self.row_dim + self.column_dim],
        ))
    }
}

/// Fixed sinusoidal position embedding over the temporal (frame) axis.
///
/// The table is a materialized checkpoint tensor
/// (`visual_temporal_embed.pos_idx_to_embed`), because upstream registers the
/// precomputed buffer as a module parameter and therefore loads it from the
/// checkpoint rather than recomputing it. [`Self::compute`] reproduces the
/// same closed form for exports that omit the buffer.
pub(crate) struct PositionalEmbeddingCosine1D {
    pos_idx_to_embed: UniquePtr<MlxArray>,
    max_seq_len: i32,
    embed_dim: i32,
}

impl PositionalEmbeddingCosine1D {
    /// Use `{prefix}.pos_idx_to_embed` from the checkpoint when present,
    /// otherwise synthesize it. Either way the result is validated against
    /// `(max_seq_len, embed_dim)` so a mismatched export fails here with the
    /// shapes named instead of inside MLX.
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        embed_dim: i32,
        max_seq_len: i32,
    ) -> Result<Self, String> {
        if embed_dim < 2 || embed_dim % 2 != 0 {
            return Err(format!(
                "Florence-2 {prefix}: temporal embedding width {embed_dim} must be a positive even number"
            ));
        }
        if max_seq_len < 1 {
            return Err(format!(
                "Florence-2 {prefix}: max_temporal_embeddings {max_seq_len} must be positive"
            ));
        }
        let key = format!("{prefix}.pos_idx_to_embed");
        let pos_idx_to_embed = match weights.get(&key) {
            Some(table) => {
                let shape = mlxcel_core::array_shape(table);
                if shape != vec![max_seq_len, embed_dim] {
                    return Err(format!(
                        "Florence-2 {key}: expected shape [{max_seq_len}, {embed_dim}], got {shape:?}"
                    ));
                }
                mlxcel_core::copy(table)
            }
            None => Self::compute(embed_dim, max_seq_len),
        };
        Ok(Self {
            pos_idx_to_embed,
            max_seq_len,
            embed_dim,
        })
    }

    /// `embed[t, 2i] = sin(t * exp(-ln(10000) * i / embed_dim))`,
    /// `embed[t, 2i+1] = cos(...)` (sin/cos interleaved, not concatenated).
    fn compute(embed_dim: i32, max_seq_len: i32) -> UniquePtr<MlxArray> {
        let half = (embed_dim / 2) as usize;
        let rows = max_seq_len as usize;
        let width = embed_dim as usize;
        let factor = (10000.0f64).ln();
        let mut data = vec![0.0f32; rows * width];
        for (i, chunk) in data.chunks_exact_mut(width).enumerate() {
            let t = i as f64;
            for j in 0..half {
                let denominator = (-factor * j as f64 / embed_dim as f64).exp();
                let angle = t * denominator;
                chunk[2 * j] = angle.sin() as f32;
                chunk[2 * j + 1] = angle.cos() as f32;
            }
        }
        mlxcel_core::from_slice_f32(&data, &[max_seq_len, embed_dim])
    }

    /// The first `seq_len` rows, shaped `[1, seq_len, embed_dim]`.
    pub(crate) fn forward(&self, seq_len: i32) -> Result<UniquePtr<MlxArray>, String> {
        if seq_len < 1 || seq_len > self.max_seq_len {
            return Err(format!(
                "Florence-2 temporal embedding: sequence length {seq_len} outside 1..={}",
                self.max_seq_len
            ));
        }
        let rows = mlxcel_core::slice(&self.pos_idx_to_embed, &[0, 0], &[seq_len, self.embed_dim]);
        Ok(mlxcel_core::reshape(&rows, &[1, seq_len, self.embed_dim]))
    }
}

/// Turn a `[batch, seq]` 0/1 attention mask into the `[batch, 1, 1, seq]`
/// additive mask the encoder's attention adds to its logits.
///
/// `log(1) = 0` and `log(0) = -inf`, so a fully valid mask contributes an
/// exact zero to every logit and a masked-out key is driven to zero weight by
/// the softmax. The reshape puts the key axis last so the tensor broadcasts
/// across batch, head, and query axes of the `[b, h, lq, lk]` logits.
pub(crate) fn additive_attention_mask(
    attention_mask: &MlxArray,
    dtype: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(attention_mask);
    let as_f32 = mlxcel_core::astype(attention_mask, mlxcel_core::dtype::FLOAT32);
    let additive = mlxcel_core::log(&as_f32);
    let additive = mlxcel_core::astype(&additive, dtype);
    mlxcel_core::reshape(&additive, &[shape[0], 1, 1, shape[1]])
}
