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

//! Pieces shared by the late-interaction (ColBERT-style) visual retrievers
//! `ColIdefics3` and `ColQwen2.5`.
//!
//! Both families are the same recipe over two different VLM backbones: run
//! the causal stack to the final norm, project every token's hidden state
//! to 128 dimensions with one `Linear`, L2-normalize each token vector and
//! zero the rows of padding tokens. What differs is only the backbone, the
//! projection's checkpoint key and the prompt format, so the projection
//! math, the `embedding_dim` reader, the `1_Dense` override and the
//! LoRA-only rejection live here rather than twice.

use std::path::Path;

use anyhow::{Result, bail};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::Value;

/// Width of one token vector when `config.json` declares no
/// `embedding_dim`. Both published families train 128.
pub const DEFAULT_EMBEDDING_DIM: usize = 128;

/// Epsilon guarding the per-token L2 denominator, matching
/// `crate::embeddings::pooling::POOLING_EPS`.
const NORM_EPS: f32 = 1e-9;

/// Number of augmentation tokens appended to a query, per the reference
/// processors of both families.
pub const QUERY_AUGMENTATION_TOKENS: usize = 10;

/// Module folder a sentence-transformers export keeps the trained
/// projection in. `load_weights_from_dir_with_subfolders` surfaces its
/// tensors as `1_Dense.linear.*`.
const DENSE_MODULE_PREFIX: &str = "1_Dense.linear.";

/// Tensor suffixes the projection can carry (dense or quantized).
const PROJECTION_SUFFIXES: &[&str] = &["weight", "bias", "scales", "biases"];

/// `config.json` `embedding_dim`, defaulting to [`DEFAULT_EMBEDDING_DIM`].
///
/// The base checkpoints of both families omit the key; the native
/// `colqwen2` layout carries it explicitly.
#[must_use]
pub fn embedding_dim(config: &Value) -> usize {
    config
        .get("embedding_dim")
        .and_then(Value::as_u64)
        .filter(|&v| v > 0)
        .map_or(DEFAULT_EMBEDDING_DIM, |v| v as usize)
}

/// Reject a repository that ships only a PEFT adapter.
///
/// `vidore/colSmol-256M` and `vidore/colqwen2.5-v0.2` are LoRA adapters on
/// top of their `-base` repositories: they carry `adapter_model.safetensors`
/// and the trained `1_Dense/` projection, but none of the backbone weights.
/// Merging an adapter is out of scope for mlxcel, and loading the base
/// alone would silently serve an untrained retriever, so this is a load
/// error naming the fix.
pub fn reject_lora_only_checkpoint(model_dir: &Path) -> Result<()> {
    if !model_dir.join("adapter_model.safetensors").is_file() {
        return Ok(());
    }
    let has_base_weights = std::fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".safetensors") && name != "adapter_model.safetensors"
        });
    if has_base_weights {
        return Ok(());
    }
    bail!(
        "{} holds only a LoRA adapter (adapter_model.safetensors with no base shard); \
         merge the adapter into the base checkpoint first and point -m at the merged \
         directory (mlxcel does not merge adapters)",
        model_dir.display()
    )
}

/// Promote the trained `1_Dense/` projection over the one stored at the
/// checkpoint root.
///
/// A merged export keeps both: the base repository's untrained
/// `linear.*` / `custom_text_proj.*` in the main shard and the trained
/// projection in `1_Dense/model.safetensors`. sentence-transformers applies
/// the module folder, so the folder wins. Returns `true` when an override
/// was applied.
pub fn apply_dense_projection_override(weights: &mut WeightMap, target_prefix: &str) -> bool {
    let mut applied = false;
    for suffix in PROJECTION_SUFFIXES {
        let source = format!("{DENSE_MODULE_PREFIX}{suffix}");
        let Some(tensor) = weights.remove(&source) else {
            continue;
        };
        weights.insert(format!("{target_prefix}.{suffix}"), tensor);
        applied = true;
    }
    if applied {
        // A dense `1_Dense` folder next to a quantized root projection would
        // otherwise leave the root's `scales` / `biases` behind and make the
        // linear look quantized. Drop whatever the folder did not supply.
        for suffix in PROJECTION_SUFFIXES {
            let key = format!("{target_prefix}.{suffix}");
            if !weights.contains_key(&format!("{DENSE_MODULE_PREFIX}{suffix}"))
                && matches!(*suffix, "scales" | "biases")
            {
                weights.remove(&key);
            }
        }
    }
    applied
}

/// Project `hidden: [B, L, H]` to `[B, L, D]`, L2-normalize every token
/// vector and zero the rows of padding tokens.
///
/// `attention_mask` is the engine's `[B, L]` int32 mask (`1` real, `0`
/// padding). The normalization runs in f32 whatever the activation dtype
/// is, so a f16 backbone cannot round a unit row away from 1.0 by more
/// than the projection's own error, and a zero row stays exactly zero
/// through the engine's second (idempotent) normalization.
#[must_use]
pub fn project_and_normalize(
    hidden: &MlxArray,
    projection: &UnifiedLinear,
    attention_mask: &MlxArray,
) -> UniquePtr<MlxArray> {
    let projected = mlxcel_core::astype(&projection.forward(hidden), dtype::FLOAT32);
    let shape = mlxcel_core::array_shape(&projected);
    debug_assert_eq!(
        shape.len(),
        3,
        "project_and_normalize expects [B, L, D], got {shape:?}"
    );
    let (batch, length) = (shape[0], shape[1]);

    let norm = mlxcel_core::linalg_norm(&projected, -1, true);
    let eps = mlxcel_core::from_slice_f32(&[NORM_EPS], &[1]);
    let unit = mlxcel_core::divide(&projected, &mlxcel_core::maximum(&norm, &eps));

    let zero_i32 = mlxcel_core::from_slice_i32(&[0], &[1, 1]);
    let real = mlxcel_core::astype(
        &mlxcel_core::not_equal(attention_mask, &zero_i32),
        dtype::FLOAT32,
    );
    let real = mlxcel_core::reshape(&real, &[batch, length, 1]);
    mlxcel_core::multiply(&unit, &real)
}

/// `Query: {text}` followed by `QUERY_AUGMENTATION_TOKENS` copies of the
/// family's augmentation token.
///
/// Both reference processors pad a query with a fixed run of the
/// checkpoint's padding token so short queries carry enough vectors for
/// MaxSim to discriminate. The run is part of the text, not of the padding
/// the engine adds, so it survives batching.
#[must_use]
pub fn format_query(text: &str, augmentation_token: &str) -> String {
    let mut out = String::with_capacity(
        "Query: ".len() + text.len() + augmentation_token.len() * QUERY_AUGMENTATION_TOKENS,
    );
    out.push_str("Query: ");
    out.push_str(text);
    for _ in 0..QUERY_AUGMENTATION_TOKENS {
        out.push_str(augmentation_token);
    }
    out
}

#[cfg(test)]
#[path = "col_late_interaction_tests.rs"]
mod col_late_interaction_tests;
