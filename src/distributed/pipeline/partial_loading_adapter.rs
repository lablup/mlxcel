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

//! Stage-local LoRA adapter loading for pipeline parallelism.
//!
//! This split-out module keeps the core `partial_loading` file focused on
//! base-weight classification and partial-load validation, while the
//! adapter-specific helpers live here. They are still re-exported from
//! `partial_loading.rs` so callers can import a single namespace.
//!
//! Used by: `lora::apply_stage_lora_adapter`, family stage executors

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mlxcel_core::weights::WeightMap;

use super::partial_loading::{LayerFilter, should_load_key};

/// Decide whether an adapter tensor should be loaded for the given stage
/// filter.
///
/// Adapter tensor names mirror base-model layer paths (for example
/// `model.layers.5.self_attn.q_proj.lora_a`), so we reuse
/// [`super::partial_loading::classify_weight_key`] to determine the owning
/// layer/embedding/lm_head. This is intentionally the same policy as
/// [`should_load_key`]:
/// - Layer-scoped adapter tensors are kept only when the layer index falls
///   inside `filter.layer_range`.
/// - Embedding / lm_head / norm-scoped adapter tensors follow the same
///   stage-ownership rules as their base counterparts.
/// - Unclassifiable adapter tensors (e.g. bookkeeping scalars) are kept on
///   the first stage by default, which matches how the fusion path and the
///   non-PP adapter loader treat them.
///
/// Used by: PP stage initialization (single-adapter composition, v1)
pub fn should_load_adapter_key(key: &str, filter: &LayerFilter) -> bool {
    should_load_key(key, filter)
}

/// Filter an already-loaded adapter [`WeightMap`] in place, keeping only the
/// tensors relevant to `filter`. Returns the number of keys removed.
///
/// Used for testability when the full adapter has already been loaded and we
/// want to drop out-of-range tensors before fusion. Production PP code paths
/// should prefer [`load_stage_adapter_weights`], which avoids even handing
/// out-of-range tensors to Rust.
pub fn filter_adapter_weights(weights: &mut WeightMap, filter: &LayerFilter) -> usize {
    let keys_to_remove: Vec<String> = weights
        .keys()
        .filter(|k| !should_load_adapter_key(k, filter))
        .cloned()
        .collect();
    let count = keys_to_remove.len();
    for key in keys_to_remove {
        weights.remove(&key);
    }
    count
}

/// Resolve the path of the adapter safetensors file inside an adapter
/// directory. Mirrors the search order used by the single-process adapter
/// loader (`src/lora/loader.rs`): first `adapters.safetensors`, then the
/// HuggingFace PEFT name `adapter_model.safetensors`.
pub fn resolve_adapter_weights_path(adapter_dir: &Path) -> Result<PathBuf> {
    let primary = adapter_dir.join("adapters.safetensors");
    if primary.exists() {
        return Ok(primary);
    }
    let alt = adapter_dir.join("adapter_model.safetensors");
    if alt.exists() {
        return Ok(alt);
    }
    anyhow::bail!(
        "No adapter weights found. Expected adapters.safetensors or \
         adapter_model.safetensors in {}",
        adapter_dir.display()
    )
}

/// Load only the adapter tensors that belong to this stage.
///
/// This walks the adapter safetensors file's tensor index via the mlxcel-core
/// FFI, decides for each tensor whether its layer falls inside the stage's
/// range via [`should_load_adapter_key`], and skips the rest. Tensors outside
/// the range are never taken into the Rust [`WeightMap`]; the MLX-side lazy
/// arrays backing them are dropped when the loader handle is released.
///
/// This is the entry point PP stages use for LoRA composition.
///
/// Used by: `LoadedStageExecutor::load_with_adapter`, family stage executors
pub fn load_stage_adapter_weights(adapter_dir: &Path, filter: &LayerFilter) -> Result<WeightMap> {
    let weights_path = resolve_adapter_weights_path(adapter_dir)?;
    mlxcel_core::weights::load_safetensors_filtered(&weights_path, |name| {
        should_load_adapter_key(name, filter)
    })
    .map_err(|err| {
        anyhow::anyhow!(
            "failed to load stage-filtered adapter weights from {}: {err}",
            weights_path.display(),
        )
    })
    .with_context(|| {
        format!(
            "stage filter: layers {}..{} (embedding={}, lm_head={})",
            filter.layer_range.start,
            filter.layer_range.end,
            filter.has_embedding,
            filter.has_lm_head,
        )
    })
}

#[cfg(test)]
#[path = "partial_loading_adapter_tests.rs"]
mod tests;
