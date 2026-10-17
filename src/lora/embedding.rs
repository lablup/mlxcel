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

//! LoRA pairs on an input embedding (issue #2421).
//!
//! mlx-lm's `LoRAEmbedding` (`mlx_lm/tuner/lora.py`) stores the pair as
//! `lora_a = [vocab, rank]`, `lora_b = [rank, hidden]`. Its lookup adds
//! `scale * lora_a[x] @ lora_b`, its `as_linear` (the tied head) adds
//! `scale * (x @ lora_b.T) @ lora_a.T`, and `fuse` writes
//! `W + (scale * lora_a @ lora_b).astype(W.dtype)`, so both ends of a tied
//! projection read one adapted matrix. That orientation is the transpose of
//! the one every linear pair uses, so an embedding target cannot go through
//! the linear delta: this module reads its own layouts and canonicalizes all of
//! them to the `LoRAEmbedding` one, `delta [vocab, hidden] = a @ b`.
//!
//! Three on-disk layouts are accepted, matched against the embedding's row
//! count (its vocabulary), in this order:
//!
//! 1. mlx-lm `LoRAEmbedding`: `a = [vocab, rank]`, `b = [rank, hidden]`.
//! 2. A linear-shaped pair written for the `[hidden] -> [vocab]` projection in
//!    the mlx-lm linear orientation: `a = [hidden, rank]`, `b = [rank, vocab]`.
//! 3. The same projection in the PEFT orientation: `a = [rank, hidden]`,
//!    `b = [vocab, rank]`.
//!
//! The last two are what the fused path already folded into the matrix before
//! this module existed, so they keep their delta. When `vocab == hidden` the
//! first two both match and the mlx-lm reading wins, because that is the
//! layout `mlx_lm.lora` writes.
//!
//! Used by: `loader::fuse_lora_weights_into`, `runtime::stage_runtime_adapters`

use anyhow::Result;
use mlxcel_core::{MlxArray, UniquePtr};

/// Whether a resolved base weight is an input embedding table: its stem ends
/// in an `embed_tokens` path segment (`model.embed_tokens.weight`,
/// `language_model.model.embed_tokens.weight`, ...).
pub(super) fn is_input_embedding_weight(base_weight_name: &str) -> bool {
    let stem = base_weight_name
        .strip_suffix(".weight")
        .unwrap_or(base_weight_name);
    stem == "embed_tokens" || stem.ends_with(".embed_tokens")
}

/// The on-disk layout of one embedding pair (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EmbeddingPairLayout {
    /// `a = [vocab, rank]`, `b = [rank, hidden]`.
    MlxLmEmbedding,
    /// `a = [hidden, rank]`, `b = [rank, vocab]`.
    LinearMlxLm,
    /// `a = [rank, hidden]`, `b = [vocab, rank]`.
    LinearPeft,
}

/// Detect an embedding pair's layout against the table's `vocab` rows and
/// return it with the hidden width the pair produces.
///
/// # Errors
///
/// Returns an error naming the layer when the pair is not two 2-D matrices or
/// matches none of the accepted layouts for `vocab` rows.
pub(super) fn detect_embedding_layout(
    layer: &str,
    a_shape: &[i32],
    b_shape: &[i32],
    vocab: i32,
) -> Result<(EmbeddingPairLayout, i32)> {
    let ([a0, a1], [b0, b1]) = (a_shape, b_shape) else {
        anyhow::bail!(
            "LoRA adapter layer {layer}: expected 2-D matrices, got A {a_shape:?}, B {b_shape:?}"
        );
    };
    if *a0 == vocab && a1 == b0 {
        Ok((EmbeddingPairLayout::MlxLmEmbedding, *b1))
    } else if *b1 == vocab && a1 == b0 {
        Ok((EmbeddingPairLayout::LinearMlxLm, *a0))
    } else if *b0 == vocab && a0 == b1 {
        Ok((EmbeddingPairLayout::LinearPeft, *a1))
    } else {
        anyhow::bail!(
            "LoRA adapter layer {layer}: incompatible shapes A {a_shape:?}, B {b_shape:?} for an \
             input embedding of {vocab} rows (expected mlx-lm's LoRAEmbedding layout, lora_a \
             [vocab, rank] and lora_b [rank, hidden])"
        );
    }
}

/// The pair's factors in the `LoRAEmbedding` layout, `(a [vocab, rank],
/// b [rank, hidden])`, so that `a @ b` is the table delta.
pub(super) fn canonical_embedding_factors(
    lora_a: &MlxArray,
    lora_b: &MlxArray,
    layout: EmbeddingPairLayout,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    match layout {
        EmbeddingPairLayout::MlxLmEmbedding => {
            (mlxcel_core::copy(lora_a), mlxcel_core::copy(lora_b))
        }
        // delta = (a @ b)ᵀ = bᵀ @ aᵀ
        EmbeddingPairLayout::LinearMlxLm => (
            mlxcel_core::transpose(lora_b),
            mlxcel_core::transpose(lora_a),
        ),
        // delta = b @ a
        EmbeddingPairLayout::LinearPeft => (mlxcel_core::copy(lora_b), mlxcel_core::copy(lora_a)),
    }
}

/// The fused table for one embedding pair: `base + (scale * a @ b)` with the
/// delta computed in `float32` and cast to the base dtype before the add, which
/// is what `mlx_lm.fuse` writes. Unlike a fused linear, the table keeps its
/// dtype, so the hidden states the lookup produces keep theirs too.
///
/// # Errors
///
/// Returns an error when the pair matches no accepted layout or its delta does
/// not have the base table's shape (a quantized table, whose packed columns
/// hide the hidden width, lands here: fusing stays dense-base-only).
pub(super) fn fuse_embedding_pair(
    layer: &str,
    base_weight_name: &str,
    base: &MlxArray,
    lora_a: &MlxArray,
    lora_b: &MlxArray,
    scale: f32,
) -> Result<UniquePtr<MlxArray>> {
    let base_shape = mlxcel_core::array_shape(base);
    let a_shape = mlxcel_core::array_shape(lora_a);
    let b_shape = mlxcel_core::array_shape(lora_b);
    let vocab = base_shape.first().copied().unwrap_or(0);
    let (layout, hidden) = detect_embedding_layout(layer, &a_shape, &b_shape, vocab)?;
    if base_shape.len() != 2 || base_shape[1] != hidden {
        anyhow::bail!(
            "LoRA shape mismatch fusing adapter layer {layer} into base weight \
             {base_weight_name}: base is {base_shape:?}, delta is {:?}.",
            [vocab, hidden]
        );
    }
    let (a, b) = canonical_embedding_factors(lora_a, lora_b, layout);
    let a = mlxcel_core::astype(&a, mlxcel_core::dtype::FLOAT32);
    let b = mlxcel_core::astype(&b, mlxcel_core::dtype::FLOAT32);
    let delta = mlxcel_core::multiply_scalar(&mlxcel_core::matmul(&a, &b), scale);
    let delta = mlxcel_core::astype(&delta, mlxcel_core::array_dtype(base));
    Ok(mlxcel_core::add(base, &delta))
}

#[cfg(test)]
#[path = "embedding_tests.rs"]
mod tests;
