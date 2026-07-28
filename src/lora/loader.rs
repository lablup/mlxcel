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

//! LoRA adapter loading and weight fusion
//!
//! This module handles loading LoRA adapter weights and fusing them
//! with base model weights for efficient inference.
//!
//! Uses mlxcel-core types (WeightMap, UniquePtr<MlxArray>) for compatibility
//! with the rest of the codebase.

use anyhow::Result;
use mlxcel_core::MlxArray;
use mlxcel_core::UniquePtr;
use mlxcel_core::weights::WeightMap;
use std::path::Path;

use super::config::AdapterConfig;

/// Load adapter weights from a safetensors file
fn load_adapter_weights(adapter_path: &Path) -> Result<WeightMap> {
    let weights_path = adapter_path.join("adapters.safetensors");

    // Try adapters.safetensors first, then adapter_model.safetensors (HuggingFace format)
    let weights_path = if weights_path.exists() {
        weights_path
    } else {
        let alt_path = adapter_path.join("adapter_model.safetensors");
        if alt_path.exists() {
            alt_path
        } else {
            anyhow::bail!(
                "No adapter weights found. Expected adapters.safetensors or adapter_model.safetensors in {:?}",
                adapter_path
            );
        }
    };

    let weights = mlxcel_core::weights::load_safetensors(&weights_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load adapter weights from {:?}: {}",
            weights_path,
            e
        )
    })?;

    Ok(weights)
}

/// Fuse LoRA weights into base model weights
///
/// LoRA formula: W_fused = W_base + scale * (lora_b @ lora_a)
///
/// Where:
/// - W_base: original weight matrix (out_features, in_features)
/// - lora_a: low-rank matrix A (rank, in_features)
/// - lora_b: low-rank matrix B (out_features, rank)
/// - scale: scaling factor (often alpha / rank)
///
/// Returns a new HashMap with fused weights
pub fn fuse_lora_weights(
    base_weights: &WeightMap,
    adapter_weights: &WeightMap,
    scale: f32,
) -> Result<WeightMap> {
    let mut fused_weights: WeightMap = base_weights
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
        .collect();
    fuse_lora_weights_into(&mut fused_weights, adapter_weights, scale)?;
    Ok(fused_weights)
}

/// Fuse LoRA adapter weights into an existing base [`WeightMap`] in place.
///
/// This is the composition primitive shared by both the non-PP adapter path
/// (which first clones the base weight map and then calls this helper on the
/// clone) and the pipeline-parallel stage-local adapter path (which operates
/// directly on the stage's in-memory base weight map to avoid the extra
/// allocation of a second weight map).
///
/// If the adapter weight map is stage-filtered, only the layers covered by
/// the filter will produce fusion updates — `base_weights` entries that are
/// not referenced by any `lora_a` / `lora_b` pair are left untouched.
///
/// # Layout
///
/// Every delta this produces is in `mlxcel_core` `Linear` layout,
/// `[out_features, in_features]`. The base weight has to be in the same layout
/// for the element-wise add to mean anything, and nothing upstream guarantees
/// that: `load_model_with_adapter` fuses into the raw safetensors map and only
/// afterwards constructs the model, which is where a family such as GPT-2
/// transposes its on-disk `Conv1D` projections. Both mismatch cases are checked
/// here, before any operand reaches MLX. See
/// [`reject_conv1d_layout_fusion`] and the shape guard in the fusion loop.
///
/// # Errors
///
/// Returns an error, leaving `base_weights` untouched, when the base weight map
/// stores its projections transposed, or when a resolved base weight and its
/// delta disagree on shape.
///
/// Used by: [`fuse_lora_weights`], [`apply_stage_lora_adapter`]
pub fn fuse_lora_weights_into(
    base_weights: &mut WeightMap,
    adapter_weights: &WeightMap,
    scale: f32,
) -> Result<()> {
    // Group adapter weights by their base layer name
    // LoRA weights are typically named like:
    // - layers.0.self_attn.q_proj.lora_a (rank, in_features)
    // - layers.0.self_attn.q_proj.lora_b (out_features, rank)
    let mut lora_pairs: std::collections::HashMap<
        String,
        (Option<UniquePtr<MlxArray>>, Option<UniquePtr<MlxArray>>),
    > = std::collections::HashMap::new();

    for (name, weight) in adapter_weights {
        if name.ends_with(".lora_a") {
            let base_name = name.trim_end_matches(".lora_a").to_string();
            lora_pairs.entry(base_name).or_insert((None, None)).0 = Some(mlxcel_core::copy(weight));
        } else if name.ends_with(".lora_b") {
            let base_name = name.trim_end_matches(".lora_b").to_string();
            lora_pairs.entry(base_name).or_insert((None, None)).1 = Some(mlxcel_core::copy(weight));
        }
        // Ignore other weights (like scales for DoRA)
    }

    // Refuse up front if the base weight map still stores its projections
    // transposed. This has to be a whole-map verdict taken before the first
    // `add`, because the square `attn.c_proj` is precisely the case where the
    // two shapes agree and the corruption would be silent.
    if !lora_pairs.is_empty() {
        let adapter_layers: Vec<&str> = lora_pairs.keys().map(String::as_str).collect();
        reject_conv1d_layout_fusion(base_weights, &adapter_layers)?;
    }

    // Fuse each LoRA pair with the corresponding base weight
    for (base_name, (lora_a_opt, lora_b_opt)) in lora_pairs {
        let (Some(lora_a), Some(lora_b)) = (lora_a_opt, lora_b_opt) else {
            tracing::warn!(
                "Incomplete LoRA pair for {}: missing lora_a or lora_b",
                base_name
            );
            continue;
        };

        // Find the corresponding base weight
        let base_weight_name = find_base_weight_name(&base_name, base_weights)?;

        let Some(base_weight) = base_weights.get(&base_weight_name) else {
            tracing::warn!(
                "Base weight not found for LoRA layer {}: tried {}",
                base_name,
                base_weight_name
            );
            continue;
        };

        // Compute the LoRA delta: scale * (lora_b @ lora_a)
        let delta = compute_lora_delta(&lora_a, &lora_b, scale)?;

        // Never hand mismatched operands to `mlxcel_core::add`. MLX broadcasts,
        // so a mismatch is not reliably an error: a base weight that is one
        // broadcastable step away from the delta fuses cleanly and produces a
        // silently wrong tensor. And where broadcasting genuinely fails, `add`
        // is not declared fallible in the cxx bridge, so the MLX C++ exception
        // is `std::terminate` and takes the process down instead of returning
        // an error the caller can report.
        let base_shape = mlxcel_core::array_shape(base_weight);
        let delta_shape = mlxcel_core::array_shape(&delta);
        if base_shape != delta_shape {
            let is_transpose = base_shape.len() == 2
                && delta_shape.len() == 2
                && base_shape[0] == delta_shape[1]
                && base_shape[1] == delta_shape[0];
            let hint = if is_transpose {
                " The two are exact transposes of each other, so this base weight is stored \
                 transposed relative to the [out_features, in_features] layout every LoRA delta \
                 uses. Fusion does not transpose an operand to make the shapes agree; convert the \
                 checkpoint to [out_features, in_features] instead."
            } else {
                ""
            };
            anyhow::bail!(
                "LoRA shape mismatch fusing adapter layer {base_name} into base weight \
                 {base_weight_name}: base is {base_shape:?}, delta is {delta_shape:?}.{hint}"
            );
        }

        // Fuse: W_fused = W_base + delta
        let fused = mlxcel_core::add(base_weight, &delta);

        base_weights.insert(base_weight_name, fused);
    }

    Ok(())
}

/// Suffixes of the transformer-block projections that a HuggingFace GPT-2
/// export stores as a `Conv1D`, i.e. `[in_features, out_features]`, transposed
/// relative to the `[out_features, in_features]` layout `Linear` weights and
/// every LoRA delta use.
///
/// GPT-2 is the only family in this tree that ships checkpoints in that layout.
/// `Gpt2Layout` (`src/models/gpt2.rs`) transposes them, but it does so at model
/// construction, which runs *after* fusion on the `load_model_with_adapter`
/// path, so fusion has to recognise the on-disk layout itself.
const CONV1D_PROJECTION_SUFFIXES: [&str; 4] = [
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
];

/// Whether `key` names one of the [`CONV1D_PROJECTION_SUFFIXES`] projections of
/// a transformer block, returning the suffix that matched.
///
/// The `h.<N>.` block index is required, not just the suffix: GPT-BigCode uses
/// the same `c_attn` / `c_proj` / `c_fc` names while storing them `[out, in]`
/// (`src/models/gpt_bigcode.rs`), and an unrelated `...c_proj.weight` elsewhere
/// in a checkpoint must never be mistaken for a Conv1D projection.
fn conv1d_projection_suffix(key: &str) -> Option<&'static str> {
    CONV1D_PROJECTION_SUFFIXES.into_iter().find(|suffix| {
        let Some(rest) = key.strip_suffix(suffix) else {
            return false;
        };
        let Some((head, index)) = rest.rsplit_once("h.") else {
            return false;
        };
        // `h.` has to start a path segment, so that `blah.0.attn.c_proj.weight`
        // (which also contains the substring `h.`) is not read as block 0.
        (head.is_empty() || head.ends_with('.'))
            && !index.is_empty()
            && index.bytes().all(|b| b.is_ascii_digit())
    })
}

/// Evidence that a base weight map still stores its transformer-block
/// projections in the HuggingFace `Conv1D` layout.
struct Conv1dLayoutEvidence {
    /// The fused QKV projection key the verdict was taken from.
    key: String,
    /// Its `[n_embd, 3 * n_embd]` shape.
    shape: Vec<i32>,
}

/// Detect whether `base_weights` is still in the HuggingFace `Conv1D` layout.
///
/// The only usable signal is the fused QKV projection: `h.<N>.attn.c_attn.weight`
/// is `[n_embd, 3 * n_embd]` on disk and `[3 * n_embd, n_embd]` once transposed,
/// and the two can never be confused. That is what makes a whole-map verdict
/// necessary as well as possible: the square `attn.c_proj` carries no layout
/// signal of its own, so only its `c_attn` sibling can say which way round it is
/// stored. This is the same probe `Gpt2Layout::detect` uses, minus the config,
/// which fusion does not have.
///
/// Returns the lexicographically smallest matching key so the diagnostic is
/// stable from run to run ([`WeightMap`] is a `HashMap`).
fn detect_conv1d_projection_layout(base_weights: &WeightMap) -> Option<Conv1dLayoutEvidence> {
    let mut evidence: Option<Conv1dLayoutEvidence> = None;

    for (key, weight) in base_weights {
        let Some(block) = key.strip_suffix(".attn.c_attn.weight") else {
            continue;
        };
        if conv1d_projection_suffix(key).is_none() {
            continue;
        }
        // A quantized projection is packed, so its stored shape matches neither
        // float layout and carries no layout signal. `Gpt2Layout::detect`
        // short-circuits on exactly this check.
        if base_weights.contains_key(&format!("{block}.attn.c_attn.scales")) {
            continue;
        }
        let shape = mlxcel_core::array_shape(weight);
        let [rows, cols] = shape.as_slice() else {
            continue;
        };
        // `i64` because `rows * 3` overflows `i32` for a hostile shape.
        if *rows <= 0 || i64::from(*cols) != i64::from(*rows) * 3 {
            continue;
        }
        if evidence.as_ref().is_none_or(|found| *key < found.key) {
            evidence = Some(Conv1dLayoutEvidence {
                key: key.clone(),
                shape,
            });
        }
    }

    evidence
}

/// Refuse to fuse into a base weight map whose projections are still stored
/// transposed, before any operand reaches MLX.
///
/// `adapter_layers` are the layer paths the adapter targets, i.e. the
/// `lora_a` / `lora_b` key stems. Only targets that resolve to a projection
/// actually stored in `Conv1D` layout are reported; an adapter that touches
/// nothing transposed (an embedding-only adapter, say) fuses as before.
///
/// This deliberately does not repair anything. Transposing the delta would fix
/// the non-square projections and leave the square `attn.c_proj` just as wrong,
/// because a square weight gives no evidence about which orientation was
/// intended. Reporting is the only honest outcome.
fn reject_conv1d_layout_fusion(base_weights: &WeightMap, adapter_layers: &[&str]) -> Result<()> {
    let Some(evidence) = detect_conv1d_projection_layout(base_weights) else {
        return Ok(());
    };

    let mut affected: Vec<String> = Vec::new();
    for layer in adapter_layers {
        let resolved = find_base_weight_name(layer, base_weights)?;
        if conv1d_projection_suffix(&resolved).is_some() && base_weights.contains_key(&resolved) {
            affected.push(resolved);
        }
    }
    if affected.is_empty() {
        return Ok(());
    }
    affected.sort();

    let Conv1dLayoutEvidence { key, shape } = evidence;
    anyhow::bail!(
        "LoRA fusion refused: this checkpoint still stores its transformer-block projections in \
         the HuggingFace Conv1D layout [in_features, out_features], while every LoRA delta is \
         [out_features, in_features]. Detected from {key} with shape {shape:?}, which is \
         [n_embd, 3 * n_embd]. Adapter targets landing on a Conv1D-stored projection: {}. \
         Adding the delta as it stands is not safe: the non-square projections cannot broadcast \
         and abort the process inside MLX, and the square attention c_proj broadcasts cleanly \
         while accumulating the transpose of the intended update, with no error anywhere. \
         Transposing an operand is not a repair either, because a square projection carries no \
         signal about which orientation was meant. Use a checkpoint whose projections are already \
         stored [out_features, in_features], such as an mlx-community GPT-2 conversion, or load \
         the model without the adapter.",
        affected.join(", "),
    );
}

/// Find the base weight name that corresponds to a LoRA layer name
fn find_base_weight_name(lora_name: &str, base_weights: &WeightMap) -> Result<String> {
    // Common patterns to try:
    // 1. Direct match with .weight suffix
    // 2. Replace specific LoRA naming conventions
    let candidates = vec![
        format!("{}.weight", lora_name),
        lora_name.to_string(),
        // HuggingFace PEFT format uses base_layer
        lora_name.replace(".base_layer", ".weight"),
    ];

    for candidate in &candidates {
        if base_weights.contains_key(candidate) {
            return Ok(candidate.clone());
        }
    }

    // If no direct match, return the most likely candidate
    Ok(format!("{}.weight", lora_name))
}

/// Compute the LoRA delta: scale * (lora_b @ lora_a)
///
/// Handles different matrix orientations based on shapes
fn compute_lora_delta(
    lora_a: &MlxArray,
    lora_b: &MlxArray,
    scale: f32,
) -> Result<UniquePtr<MlxArray>> {
    let a_shape = mlxcel_core::array_shape(lora_a);
    let b_shape = mlxcel_core::array_shape(lora_b);

    // Determine orientation based on shapes
    // We need: delta shape = (out_features, in_features) for Linear weight
    //
    // mlx-lm convention:
    // - lora_a: (in_features, rank)
    // - lora_b: (rank, out_features)
    // - delta = (lora_a @ lora_b).T = lora_b.T @ lora_a.T
    //
    // Standard convention (HuggingFace PEFT):
    // - lora_a: (rank, in_features)
    // - lora_b: (out_features, rank)
    // - delta = lora_b @ lora_a

    let delta = if a_shape.len() == 2 && b_shape.len() == 2 {
        // Check if shapes are compatible for either convention
        if a_shape[1] == b_shape[0] {
            // mlx-lm: a=(in, rank), b=(rank, out) -> need transpose result
            let product = mlxcel_core::matmul(lora_a, lora_b);
            mlxcel_core::transpose_axes(&product, &[1, 0])
        } else if a_shape[0] == b_shape[1] {
            // Standard PEFT: a=(rank, in), b=(out, rank) -> b @ a
            mlxcel_core::matmul(lora_b, lora_a)
        } else {
            anyhow::bail!(
                "Incompatible LoRA shapes: lora_a={:?}, lora_b={:?}",
                a_shape,
                b_shape
            );
        }
    } else {
        anyhow::bail!(
            "Expected 2D LoRA matrices, got lora_a={:?}, lora_b={:?}",
            a_shape,
            b_shape
        );
    };

    // Scale the delta
    let scale_arr = mlxcel_core::full_f32(&[1], scale, mlxcel_core::dtype::FLOAT32);
    let scaled_delta = mlxcel_core::multiply(&delta, &scale_arr);

    Ok(scaled_delta)
}

/// Apply LoRA adapters to base model weights by fusion
///
/// This function loads the adapter configuration and weights,
/// then fuses the LoRA weights with the base model weights.
///
/// # Arguments
///
/// * `base_weights` - The base model weights to modify
/// * `adapter_path` - Path to the adapter directory containing adapter_config.json and adapters.safetensors
///
/// # Returns
///
/// A new HashMap containing the fused weights
pub fn apply_lora_adapters(base_weights: &WeightMap, adapter_path: &Path) -> Result<WeightMap> {
    // Load adapter configuration
    let config = AdapterConfig::load(adapter_path)?;

    tracing::info!(
        "Loading LoRA adapter: rank={}, scale={:.2}, type={:?}",
        config.rank(),
        config.effective_scale(),
        config.fine_tune_type
    );

    if !config.is_lora() {
        anyhow::bail!(
            "Adapter is not LoRA type: {:?}. Full fine-tuning adapters should be loaded directly.",
            config.fine_tune_type
        );
    }

    // Load adapter weights
    let adapter_weights = load_adapter_weights(adapter_path)?;

    tracing::info!("Loaded {} adapter weight tensors", adapter_weights.len());

    // Fuse weights
    let fused = fuse_lora_weights(base_weights, &adapter_weights, config.effective_scale())?;

    // Count how many weights were modified
    let modified_count = adapter_weights
        .keys()
        .filter(|k| k.ends_with(".lora_a"))
        .count();

    tracing::info!("Fused LoRA adapters into {} layers", modified_count);

    Ok(fused)
}

/// Apply a stage-local LoRA adapter to a pipeline stage's base weight map
/// in place.
///
/// This is the pipeline-parallel counterpart of [`apply_lora_adapters`]:
/// it reuses the same adapter directory layout (`adapter_config.json` plus
/// `adapters.safetensors` / `adapter_model.safetensors`) and the same rank
/// / scaling semantics, but loads only the adapter tensors that belong to
/// the stage's layer range, fuses them into the stage's base weights in
/// place, and never materializes per-layer deltas for other stages.
///
/// The caller is expected to pass the stage's [`LayerFilter`]. The
/// adapter configuration is validated exactly as in the non-PP path, and so is
/// the base weight layout: this path never goes through
/// `load_model_with_adapter`, but it shares [`fuse_lora_weights_into`], so it
/// inherits the same layout and shape guards.
///
/// Used by: pipeline stage initialization (family stage executors)
pub fn apply_stage_lora_adapter(
    base_weights: &mut WeightMap,
    adapter_path: &Path,
    filter: &crate::distributed::pipeline::LayerFilter,
) -> Result<()> {
    let config = AdapterConfig::load(adapter_path)?;

    tracing::info!(
        "Loading stage-local LoRA adapter: rank={}, scale={:.2}, type={:?}, stage_layers={}..{}",
        config.rank(),
        config.effective_scale(),
        config.fine_tune_type,
        filter.layer_range.start,
        filter.layer_range.end,
    );

    if !config.is_lora() {
        anyhow::bail!(
            "Adapter is not LoRA type: {:?}. Full fine-tuning adapters should be loaded directly.",
            config.fine_tune_type
        );
    }

    let adapter_weights =
        crate::distributed::pipeline::load_stage_adapter_weights(adapter_path, filter)?;

    let modified_count = adapter_weights
        .keys()
        .filter(|k| k.ends_with(".lora_a"))
        .count();

    tracing::info!(
        "Loaded {} adapter tensors for stage (skipped out-of-range adapter layers)",
        adapter_weights.len(),
    );

    fuse_lora_weights_into(base_weights, &adapter_weights, config.effective_scale())?;

    tracing::info!(
        "Fused stage-local LoRA adapters into {} layers",
        modified_count,
    );

    Ok(())
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
