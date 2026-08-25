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

//! Weight-key normalization shared by the decoder-backbone embedding
//! families served through `/v1/embeddings`.
//!
//! A sentence-transformers export of a decoder backbone differs from the
//! generator checkpoint of the same architecture in three mechanical ways,
//! and every family that reuses a generator backbone has to undo all three
//! before the backbone constructor can find its tensors:
//!
//! 1. The backbone roots are stored bare (`embed_tokens.weight`,
//!    `layers.0.…`, `norm.weight`) because the export saves the inner
//!    `…Model` rather than the `…ForCausalLM` wrapper, while the mlx
//!    conversions of the same checkpoints keep the `model.` prefix.
//! 2. A generation head (`lm_head.*`, `head.*`) may still be present and is
//!    never used by an embedder.
//! 3. Post-pooling `Dense` modules live in numbered subfolders
//!    (`2_Dense/model.safetensors` with a `linear.weight` inside), which
//!    `load_weights_from_dir_with_subfolders` surfaces as
//!    `2_Dense.linear.weight`, while an mlx conversion folds them into the
//!    main shards as `dense.0.*`.
//!
//! Used by: `crate::models::gemma3_embedding`, `crate::models::qwen3_embedding`.

use mlxcel_core::weights::WeightMap;

/// Backbone roots a decoder `…Model` export stores without the `model.`
/// prefix that the generator constructors expect.
const BACKBONE_ROOTS: &[&str] = &["embed_tokens.", "layers.", "norm."];

/// Generation-head roots an embedder never reads.
const HEAD_ROOTS: &[&str] = &["lm_head.", "head."];

/// Tensor suffixes a sentence-transformers `Dense` module can carry.
const DENSE_SUFFIXES: &[&str] = &["weight", "bias", "scales", "biases"];

/// Split `{N}_Dense.linear.{suffix}` into `(N, suffix)`.
///
/// Returns `None` for every other key, including a `{N}_Dense.` key whose
/// inner module is not the `linear` sentence-transformers `Dense` uses, so
/// an unknown module is left in place rather than silently renamed.
fn dense_module_key(key: &str) -> Option<(u32, &str)> {
    let (folder, rest) = key.split_once('.')?;
    let index: u32 = folder.strip_suffix("_Dense")?.parse().ok()?;
    let suffix = rest.strip_prefix("linear.")?;
    DENSE_SUFFIXES.contains(&suffix).then_some((index, suffix))
}

/// Rename `{N}_Dense.linear.*` tensors to `dense.{k}.*`, where `k` numbers
/// the folders by `N` ascending.
///
/// The folder numbers are sentence-transformers module positions (`2_Dense`
/// after `1_Pooling`), not projection indices, so they are ranked rather
/// than used directly. Returns the number of distinct `Dense` folders found;
/// `0` means the checkpoint already stores the projections as `dense.{k}.*`
/// (the mlx conversion layout) or has none.
pub(crate) fn fold_dense_modules(weights: &mut WeightMap) -> usize {
    let mut folders: Vec<u32> = weights
        .keys()
        .filter_map(|key| dense_module_key(key).map(|(index, _)| index))
        .collect();
    folders.sort_unstable();
    folders.dedup();
    if folders.is_empty() {
        return 0;
    }

    let renames: Vec<(String, String)> = weights
        .keys()
        .filter_map(|key| {
            let (index, suffix) = dense_module_key(key)?;
            let rank = folders.iter().position(|&f| f == index)?;
            Some((key.clone(), format!("dense.{rank}.{suffix}")))
        })
        .collect();
    for (from, to) in renames {
        if let Some(tensor) = weights.remove(&from) {
            weights.insert(to, tensor);
        }
    }
    folders.len()
}

/// Drop every generation-head tensor.
pub(crate) fn drop_generation_head(weights: &mut WeightMap) {
    weights.retain(|key, _| !HEAD_ROOTS.iter().any(|root| key.starts_with(root)));
}

/// Prefix bare backbone roots with `model.`.
///
/// A key that already starts with `model.` is left alone, so this is a no-op
/// on an mlx conversion and idempotent when applied twice.
pub(crate) fn prefix_backbone_roots(weights: &mut WeightMap) {
    let renames: Vec<(String, String)> = weights
        .keys()
        .filter(|key| BACKBONE_ROOTS.iter().any(|root| key.starts_with(root)))
        .map(|key| (key.clone(), format!("model.{key}")))
        .collect();
    for (from, to) in renames {
        if let Some(tensor) = weights.remove(&from) {
            weights.insert(to, tensor);
        }
    }
}

/// Apply all three normalizations in the order the layouts require: fold the
/// `Dense` subfolders first (their keys are not backbone roots), then drop
/// the head, then prefix the backbone.
///
/// Returns the number of `Dense` folders folded.
pub(crate) fn sanitize_decoder_embedding_weights(weights: &mut WeightMap) -> usize {
    let folded = fold_dense_modules(weights);
    drop_generation_head(weights);
    prefix_backbone_roots(weights);
    folded
}

/// Logical `(out_features, in_features)` of a linear stored at `prefix`.
///
/// A quantized weight is packed along the input axis (`[out, in * bits /
/// 32]`), so the input width is read from the `scales` grouping
/// (`[out, in / group_size]`) instead of the packed row. `None` when the
/// tensor is absent or is not rank 2.
pub(crate) fn linear_features(
    weights: &WeightMap,
    prefix: &str,
    group_size: i32,
) -> Option<(i32, i32)> {
    let shape = mlxcel_core::array_shape(weights.get(&format!("{prefix}.weight"))?);
    if shape.len() != 2 {
        return None;
    }
    let in_features = match weights.get(&format!("{prefix}.scales")) {
        Some(scales) => {
            let scales_shape = mlxcel_core::array_shape(scales);
            if scales_shape.len() != 2 {
                return None;
            }
            scales_shape[1].checked_mul(group_size)?
        }
        None => shape[1],
    };
    Some((shape[0], in_features))
}

#[cfg(test)]
#[path = "embedding_sanitize_tests.rs"]
mod embedding_sanitize_tests;
