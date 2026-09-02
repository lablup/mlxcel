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
use std::collections::HashMap;
use std::path::Path;

use super::config::{AdapterConfig, FineTuneType};

/// Load adapter weights from a safetensors file
pub(crate) fn load_adapter_weights(adapter_path: &Path) -> Result<WeightMap> {
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

/// Refuse an adapter whose `fine_tune_type` this build cannot apply.
///
/// DoRA is refused rather than treated as LoRA. A DoRA checkpoint carries a
/// per-output-row magnitude vector alongside its low-rank pair, and applying
/// only the pair produces weights that match neither the base model nor the
/// fine-tune. Accepting it used to be silent because `is_lora()` returned true
/// for it and the magnitude tensors were dropped as "other weights".
///
/// Used by: [`apply_lora_adapters_scaled`], [`apply_stage_lora_adapter`],
/// [`super::runtime::RuntimeLoraSet::from_specs`]
pub(super) fn reject_unsupported_fine_tune_type(
    config: &AdapterConfig,
    adapter_path: &Path,
) -> Result<()> {
    if config.is_fusable_lora() {
        return Ok(());
    }
    if config.fine_tune_type == FineTuneType::DoRA {
        anyhow::bail!(
            "DoRA adapters are not supported: {} declares fine_tune_type dora; fuse it with its \
             own tooling first",
            adapter_path.display()
        );
    }
    anyhow::bail!(
        "Adapter {} is not LoRA type: {:?}. Full fine-tuning adapters should be loaded directly.",
        adapter_path.display(),
        config.fine_tune_type
    );
}

/// One adapter tensor pair that has been checked against the base weights and
/// can be applied to them.
///
/// The adapter's own layer path is carried alongside the resolved base weight
/// key because the two differ (`...q_proj` against `...q_proj.weight`, and the
/// HuggingFace PEFT `...q_proj.base_layer` against that same `.weight`), and
/// the fusion diagnostics name both.
pub(super) struct FusablePair {
    /// The `.lora_a` / `.lora_b` key stem, i.e. the adapter's own layer path.
    pub(super) layer: String,
    /// The base weight key this pair's delta is applied to.
    pub(super) base_weight_name: String,
    pub(super) lora_a: UniquePtr<MlxArray>,
    pub(super) lora_b: UniquePtr<MlxArray>,
}

/// The base weight keys a LoRA layer path can resolve to, most specific first.
///
/// Used by: [`find_base_weight_name`], [`validate_adapter_tensors`]
fn base_weight_candidates(lora_name: &str) -> Vec<String> {
    let mut candidates = vec![format!("{lora_name}.weight"), lora_name.to_string()];
    // HuggingFace PEFT wraps the frozen projection in a `.base_layer`, which
    // resolves to the same `.weight` the other conventions name directly.
    let peft = lora_name.replace(".base_layer", ".weight");
    if !candidates.contains(&peft) {
        candidates.push(peft);
    }
    candidates
}

/// Find the base weight name a LoRA layer name resolves to, or `None` when the
/// checkpoint holds none of the candidates.
///
/// This deliberately has no fallback. It used to return `"{name}.weight"` when
/// nothing matched, which made "resolved" and "exists" indistinguishable to the
/// caller: that is how an adapter trained for a different architecture reached
/// the fusion loop, got skipped with a `warn!`, and left the model serving base
/// weights while every log line said the adapter was loaded (issue #1328).
///
/// Used by: [`validate_adapter_tensors`]
fn find_base_weight_name(lora_name: &str, base_weights: &WeightMap) -> Option<String> {
    base_weight_candidates(lora_name)
        .into_iter()
        .find(|candidate| base_weights.contains_key(candidate))
}

/// Pair up the adapter's tensors and check every one of them against the base
/// weights, returning the pairs that can be applied or one error listing every
/// tensor that cannot.
///
/// Three rules, each of which used to be a `warn!` and a `continue` that left
/// the base weight in place:
///
/// 1. Every tensor is one half of a `<layer>.lora_a` / `<layer>.lora_b` pair.
///    A DoRA magnitude vector, a stray `.weight`, and the HuggingFace PEFT
///    `.lora_A.weight` spelling this build does not read all land here.
/// 2. Both halves of every pair are present.
/// 3. Every pair resolves to a base weight this checkpoint actually holds.
///
/// Rank compatibility and the delta-versus-base shape check stay where they
/// already were, in [`compute_lora_delta`] and the fusion loop's shape guard,
/// because both need the computed delta.
///
/// Violations are collected rather than reported one at a time: an adapter
/// built for the wrong architecture fails on every layer, and reporting the
/// first one would need as many load attempts as the model has layers. Both
/// the violation list and the returned pairs are sorted, because [`WeightMap`]
/// is a `HashMap` and neither the diagnostic nor the fusion order may depend on
/// its iteration order.
///
/// Used by: [`fuse_lora_weights_into`], [`super::runtime::stage_runtime_adapters`]
pub(super) fn validate_adapter_tensors(
    base_weights: &WeightMap,
    adapter_weights: &WeightMap,
) -> Result<Vec<FusablePair>> {
    type Halves<'a> = (
        Option<&'a UniquePtr<MlxArray>>,
        Option<&'a UniquePtr<MlxArray>>,
    );

    let mut halves: HashMap<&str, Halves<'_>> = HashMap::new();
    let mut violations: Vec<String> = Vec::new();

    for (name, weight) in adapter_weights {
        if let Some(layer) = name.strip_suffix(".lora_a") {
            halves.entry(layer).or_default().0 = Some(weight);
        } else if let Some(layer) = name.strip_suffix(".lora_b") {
            halves.entry(layer).or_default().1 = Some(weight);
        } else {
            violations.push(format!(
                "{name}: not a LoRA tensor (expected .lora_a or .lora_b)"
            ));
        }
    }

    let mut pairs: Vec<FusablePair> = Vec::with_capacity(halves.len());
    for (layer, (lora_a, lora_b)) in halves {
        let (Some(lora_a), Some(lora_b)) = (lora_a, lora_b) else {
            let (present, missing) = if lora_a.is_some() {
                ("lora_a", "lora_b")
            } else {
                ("lora_b", "lora_a")
            };
            violations.push(format!(
                "{layer}.{present}: incomplete pair (missing {layer}.{missing})"
            ));
            continue;
        };
        let Some(base_weight_name) = find_base_weight_name(layer, base_weights) else {
            violations.push(format!(
                "{layer}.lora_a: no base weight (tried {})",
                base_weight_candidates(layer).join(", ")
            ));
            continue;
        };
        pairs.push(FusablePair {
            layer: layer.to_string(),
            base_weight_name,
            lora_a: mlxcel_core::copy(lora_a),
            lora_b: mlxcel_core::copy(lora_b),
        });
    }

    if !violations.is_empty() {
        violations.sort();
        let count = violations.len();
        let noun = if count == 1 { "tensor" } else { "tensors" };
        anyhow::bail!(
            "{count} adapter {noun} cannot be applied to this model:\n  {}\n\
             Every tensor in a fusable adapter has to be one half of a <layer>.lora_a / \
             <layer>.lora_b pair whose base weight this checkpoint holds; skipping the rest \
             would serve weights that match neither the base model nor the fine-tune.",
            violations.join("\n  "),
        );
    }

    pairs.sort_by(|a, b| {
        a.base_weight_name
            .cmp(&b.base_weight_name)
            .then_with(|| a.layer.cmp(&b.layer))
    });
    Ok(pairs)
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
    let (fused_weights, _) = fuse_lora_weights_counted(base_weights, adapter_weights, scale)?;
    Ok(fused_weights)
}

/// [`fuse_lora_weights`] with the number of pairs that were actually applied.
///
/// Only [`apply_lora_adapters_scaled`] needs the count, and it needs it to be
/// the fused total rather than an adapter key count, so this stays private
/// instead of widening the public surface.
///
/// Used by: [`fuse_lora_weights`], [`apply_lora_adapters_scaled`]
fn fuse_lora_weights_counted(
    base_weights: &WeightMap,
    adapter_weights: &WeightMap,
    scale: f32,
) -> Result<(WeightMap, usize)> {
    let mut fused_weights: WeightMap = base_weights
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
        .collect();
    let fused_count = fuse_lora_weights_into(&mut fused_weights, adapter_weights, scale)?;
    Ok((fused_weights, fused_count))
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
/// the filter will produce fusion updates. `base_weights` entries that are
/// not referenced by any `lora_a` / `lora_b` pair are left untouched.
///
/// Returns the number of pairs that were applied. That is the count callers
/// must log: counting `.lora_a` keys instead reported tensors that had been
/// skipped as if they had been fused (issue #1328).
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
/// Returns an error when any adapter tensor cannot be applied to this base
/// weight map (see [`validate_adapter_tensors`]) or when the base weight map
/// stores its projections transposed. Both verdicts are taken before the first
/// write, so `base_weights` is untouched. A per-pair shape disagreement is
/// found later, with the computed delta in hand, so it can leave the pairs
/// sorted ahead of it applied.
///
/// Used by: [`fuse_lora_weights_counted`], [`apply_stage_lora_adapter`]
pub fn fuse_lora_weights_into(
    base_weights: &mut WeightMap,
    adapter_weights: &WeightMap,
    scale: f32,
) -> Result<usize> {
    // Pair the adapter tensors up and resolve each pair's base weight before
    // anything is written. Adapter tensor names look like
    // `layers.0.self_attn.q_proj.lora_a` / `.lora_b`, and anything that is not
    // one half of such a pair, or that resolves to no base weight, fails the
    // whole load here rather than being skipped.
    let pairs = validate_adapter_tensors(base_weights, adapter_weights)?;

    // Refuse up front if the base weight map still stores its projections
    // transposed. This has to be a whole-map verdict taken before the first
    // `add`, because the square `attn.c_proj` is precisely the case where the
    // two shapes agree and the corruption would be silent.
    reject_conv1d_layout_fusion(base_weights, &pairs)?;

    // Fuse each LoRA pair with the corresponding base weight
    let mut fused_count = 0usize;
    for pair in &pairs {
        let FusablePair {
            layer: base_name,
            base_weight_name,
            lora_a,
            lora_b,
        } = pair;

        let Some(base_weight) = base_weights.get(base_weight_name) else {
            anyhow::bail!(
                "internal error: base weight {base_weight_name} for adapter layer {base_name} \
                 was resolved during validation but is no longer in the weight map"
            );
        };

        // Compute the LoRA delta: scale * (lora_b @ lora_a)
        let delta = compute_lora_delta(lora_a, lora_b, scale)?;

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

        base_weights.insert(base_weight_name.clone(), fused);
        fused_count += 1;
    }

    Ok(fused_count)
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
/// `pairs` are the adapter pairs [`validate_adapter_tensors`] already resolved
/// against this map, so every `base_weight_name` is known to exist. Only
/// targets that land on a projection actually stored in `Conv1D` layout are
/// reported; an adapter that touches nothing transposed (an embedding-only
/// adapter, say) fuses as before.
///
/// This deliberately does not repair anything. Transposing the delta would fix
/// the non-square projections and leave the square `attn.c_proj` just as wrong,
/// because a square weight gives no evidence about which orientation was
/// intended. Reporting is the only honest outcome.
fn reject_conv1d_layout_fusion(base_weights: &WeightMap, pairs: &[FusablePair]) -> Result<()> {
    if pairs.is_empty() {
        return Ok(());
    }
    let Some(evidence) = detect_conv1d_projection_layout(base_weights) else {
        return Ok(());
    };

    let mut affected: Vec<&str> = pairs
        .iter()
        .filter(|pair| conv1d_projection_suffix(&pair.base_weight_name).is_some())
        .map(|pair| pair.base_weight_name.as_str())
        .collect();
    if affected.is_empty() {
        return Ok(());
    }
    // Two adapter layers can resolve to one base weight (a `.base_layer`
    // spelling next to a plain one), so the list is deduplicated as well.
    affected.sort_unstable();
    affected.dedup();

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
    apply_lora_adapters_scaled(base_weights, adapter_path, 1.0)
}

/// [`apply_lora_adapters`] with a b10621 `--lora-scaled` user scale
/// multiplied into the adapter's own `alpha / r`, exactly as upstream
/// multiplies its per-adapter scale into the applied delta (issue #1439).
pub fn apply_lora_adapters_scaled(
    base_weights: &WeightMap,
    adapter_path: &Path,
    user_scale: f32,
) -> Result<WeightMap> {
    // Load adapter configuration
    let config = AdapterConfig::load(adapter_path)?;

    tracing::info!(
        "Loading LoRA adapter: rank={}, scale={:.2}, user_scale={:.2}, type={:?}",
        config.rank(),
        config.effective_scale(),
        user_scale,
        config.fine_tune_type
    );

    reject_unsupported_fine_tune_type(&config, adapter_path)?;

    // Load adapter weights
    let adapter_weights = load_adapter_weights(adapter_path)?;

    tracing::info!("Loaded {} adapter weight tensors", adapter_weights.len());

    // Fuse weights
    let (fused, fused_count) = fuse_lora_weights_counted(
        base_weights,
        &adapter_weights,
        config.effective_scale() * user_scale,
    )
    .map_err(|err| anyhow::anyhow!("Adapter at {}: {err}", adapter_path.display()))?;

    // An adapter that applies nothing is the failure this path exists to
    // report: the model would load and serve the base weights unchanged while
    // every log line said the adapter was in place. The pipeline-stage entry
    // point deliberately does not take this check, because a stage that owns
    // none of the adapter's layers applies zero pairs by design.
    if fused_count == 0 {
        anyhow::bail!(
            "Adapter at {} applied no tensors to this model: it holds no lora_a / lora_b pair",
            adapter_path.display()
        );
    }

    tracing::info!("Fused LoRA adapters into {fused_count} layers");

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
/// Returns the number of pairs this stage applied. Zero is a valid result and
/// is never an error here: an adapter may target a subset of the model's
/// layers, and a stage that owns none of them has nothing to apply. The
/// whole-model entry point ([`apply_lora_adapters_scaled`]) is the one that
/// treats zero as a failed load.
///
/// Used by: pipeline stage initialization (family stage executors)
pub fn apply_stage_lora_adapter(
    base_weights: &mut WeightMap,
    adapter_path: &Path,
    filter: &crate::distributed::pipeline::LayerFilter,
) -> Result<usize> {
    let config = AdapterConfig::load(adapter_path)?;

    tracing::info!(
        "Loading stage-local LoRA adapter: rank={}, scale={:.2}, type={:?}, stage_layers={}..{}",
        config.rank(),
        config.effective_scale(),
        config.fine_tune_type,
        filter.layer_range.start,
        filter.layer_range.end,
    );

    reject_unsupported_fine_tune_type(&config, adapter_path)?;

    let adapter_weights =
        crate::distributed::pipeline::load_stage_adapter_weights(adapter_path, filter)?;

    tracing::info!(
        "Loaded {} adapter tensors for stage (skipped out-of-range adapter layers)",
        adapter_weights.len(),
    );

    let fused_count =
        fuse_lora_weights_into(base_weights, &adapter_weights, config.effective_scale())
            .map_err(|err| anyhow::anyhow!("Adapter at {}: {err}", adapter_path.display()))?;

    tracing::info!("Fused stage-local LoRA adapters into {fused_count} layers");

    Ok(fused_count)
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
