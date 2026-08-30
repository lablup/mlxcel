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

//! Runtime (unfused) LoRA serving state (llama-server b10621, issue #1439).
//!
//! b10621 keeps adapters as runtime-swappable layers: `POST /lora-adapters`
//! changes the server-wide adapter scales, the native `lora` request field
//! selects per-request scales, and `--lora-init-without-apply` loads
//! adapters at scale 0.0 to be applied later. This module is the serving
//! half of mlxcel's unfused path: [`RuntimeLoraSet`] owns one shared scale
//! handle per adapter (the layers hold clones and read them per forward, see
//! [`mlxcel_core::runtime_lora`]), the b10621 "server default" scale vector
//! the routes read and write, and the staging step that turns adapter
//! safetensors into per-layer pending terms before model construction.
//!
//! Scale semantics mirror upstream exactly: each adapter's *user* scale is
//! what the routes see (1.0 from `--lora`, the given value from
//! `--lora-scaled`, 0.0 under `--lora-init-without-apply`), and the layers
//! multiply it by the adapter's own `alpha / rank` at forward time, the same
//! product the fused path bakes in.
//!
//! Application timing also mirrors upstream: a request snapshots the server
//! default (or its own `lora` field) at admission, batches only ever contain
//! one snapshot, and the executing worker writes the handles per batch, so a
//! `POST /lora-adapters` never changes a generation already in flight.

use std::sync::RwLock;

use anyhow::Result;
use mlxcel_core::runtime_lora::{PendingLoraTerm, SharedLoraScale};
use mlxcel_core::weights::WeightMap;

use super::config::AdapterConfig;
use super::loader::{find_base_weight_name, load_adapter_weights};
use super::multi::LoraAdapterSpec;

/// One adapter's serving state.
pub struct RuntimeLoraAdapter {
    pub spec: LoraAdapterSpec,
    /// The adapter's own `alpha / rank`, from its `adapter_config.json`.
    pub base_scale: f32,
    /// The live user scale every layer term of this adapter reads.
    pub handle: SharedLoraScale,
}

/// The server's runtime-LoRA state: per-adapter shared handles plus the
/// b10621 server-default scale vector.
pub struct RuntimeLoraSet {
    pub adapters: Vec<RuntimeLoraAdapter>,
    /// The b10621 `params_base.lora_adapters` equivalent: what
    /// `GET /lora-adapters` reports and `POST /lora-adapters` replaces.
    /// Requests snapshot it at admission; the worker applies a snapshot to
    /// the handles per batch.
    server_scales: RwLock<Vec<f32>>,
}

impl std::fmt::Debug for RuntimeLoraSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeLoraSet")
            .field("adapters", &self.adapters.len())
            .field("server_scales", &self.server_scales())
            .finish()
    }
}

impl RuntimeLoraSet {
    /// Build the serving state from the parsed CLI specification, reading
    /// each adapter's `adapter_config.json` for its `alpha / rank`. Fails on
    /// a missing or non-LoRA adapter, exactly like the fused path's load.
    pub fn from_specs(specs: &[LoraAdapterSpec]) -> Result<Self> {
        let mut adapters = Vec::with_capacity(specs.len());
        for spec in specs {
            let config = AdapterConfig::load(&spec.path).map_err(|e| {
                anyhow::anyhow!("LoRA adapter {} failed to load: {e}", spec.path.display())
            })?;
            if !config.is_lora() {
                anyhow::bail!(
                    "Adapter {} is not LoRA type: {:?}. Full fine-tuning adapters should be \
                     loaded directly.",
                    spec.path.display(),
                    config.fine_tune_type
                );
            }
            adapters.push(RuntimeLoraAdapter {
                base_scale: config.effective_scale(),
                handle: SharedLoraScale::new(spec.reported_scale()),
                spec: spec.clone(),
            });
        }
        let server_scales = adapters
            .iter()
            .map(|a| a.spec.reported_scale())
            .collect::<Vec<f32>>();
        Ok(Self {
            adapters,
            server_scales: RwLock::new(server_scales),
        })
    }

    /// A disk-free set for route and scheduler tests: one adapter per user
    /// scale, with dummy paths and `alpha / rank == 1.0`.
    #[cfg(test)]
    pub(crate) fn stub(user_scales: &[f32]) -> Self {
        let adapters = user_scales
            .iter()
            .enumerate()
            .map(|(idx, scale)| RuntimeLoraAdapter {
                spec: LoraAdapterSpec {
                    path: std::path::PathBuf::from(format!("/adapters/stub-{idx}")),
                    scale: *scale,
                    apply: true,
                },
                base_scale: 1.0,
                handle: SharedLoraScale::new(*scale),
            })
            .collect::<Vec<_>>();
        let server_scales = user_scales.to_vec();
        Self {
            adapters,
            server_scales: RwLock::new(server_scales),
        }
    }

    /// The current server-default user scales (b10621 `GET /lora-adapters`).
    #[must_use]
    pub fn server_scales(&self) -> Vec<f32> {
        self.server_scales
            .read()
            .map(|s| s.clone())
            .unwrap_or_else(|_| vec![0.0; self.adapters.len()])
    }

    /// Replace the server-default scales (b10621 `POST /lora-adapters`).
    /// Affects requests admitted afterwards; in-flight generations keep the
    /// snapshot they were admitted with, exactly like upstream's slots.
    pub fn set_server_scales(&self, scales: Vec<f32>) {
        if let Ok(mut guard) = self.server_scales.write() {
            *guard = scales;
        }
    }

    /// Write `scales` into the live handles the layers read. Called by the
    /// executing worker before running a batch whose snapshot differs from
    /// what is currently applied.
    pub fn apply_scales(&self, scales: &[f32]) {
        for (adapter, scale) in self.adapters.iter().zip(scales.iter()) {
            adapter.handle.set(*scale);
        }
    }

    /// A stable identifier for one scale configuration, used as the prompt
    /// cache's `lora_id` component so cache entries never cross adapter
    /// configurations.
    #[must_use]
    pub fn scales_digest(scales: &[f32]) -> String {
        let mut out = String::from("lora");
        for scale in scales {
            out.push(':');
            out.push_str(&format!("{:.6}", scale));
        }
        out
    }
}

/// The detected on-disk orientation of one adapter pair.
///
/// Both conventions in the wild are accepted, with the same shape heuristic
/// and precedence as the fused path's `compute_lora_delta`: mlx-lm stores
/// `a = [in, rank]`, `b = [rank, out]` (checked first), HuggingFace PEFT
/// stores `a = [rank, in]`, `b = [out, rank]`. A square projection with a
/// symmetric rank is ambiguous and resolves to mlx-lm, exactly as fusing
/// does, so the two paths cannot disagree on the same file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairOrientation {
    MlxLm,
    Peft,
}

/// Detect the pair's orientation and its `(in, out)`, or fail with the
/// fused path's incompatible-shapes posture.
fn detect_orientation(
    layer: &str,
    a_shape: &[i32],
    b_shape: &[i32],
) -> Result<(PairOrientation, i32, i32)> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        anyhow::bail!(
            "LoRA adapter layer {layer}: expected 2-D matrices, got A {a_shape:?}, B {b_shape:?}"
        );
    }
    if a_shape[1] == b_shape[0] {
        // mlx-lm: a = [in, rank], b = [rank, out].
        Ok((PairOrientation::MlxLm, a_shape[0], b_shape[1]))
    } else if a_shape[0] == b_shape[1] {
        // PEFT: a = [rank, in], b = [out, rank].
        Ok((PairOrientation::Peft, a_shape[1], b_shape[0]))
    } else {
        anyhow::bail!(
            "LoRA adapter layer {layer}: incompatible shapes A {a_shape:?}, B {b_shape:?}"
        );
    }
}

/// Validate one adapter pair's resolved `(in, out)` against its base weight
/// so a mismatched adapter fails the load instead of aborting at the first
/// forward (an MLX shape throw crosses the cxx bridge as `std::terminate`).
///
/// For a dense base, `in_features` is checked exactly. For a quantized base
/// the packed plane hides `in_features`, so the check pins it through the
/// scales plane instead: the input width must be a whole multiple of the
/// scales' group count with a plausible group size. The GPT-2 `Conv1D`
/// transposed layout fails these checks loudly, matching the fused path's
/// refusal of that layout.
fn validate_pair_shapes(
    layer: &str,
    base_weights: &WeightMap,
    base_weight_name: &str,
    in_features: i32,
    out_features: i32,
) -> Result<()> {
    let Some(base) = base_weights.get(base_weight_name) else {
        return Ok(()); // caller already warned; nothing to validate against
    };
    let base_shape = mlxcel_core::array_shape(base);
    if base_shape.len() != 2 {
        anyhow::bail!(
            "LoRA adapter layer {layer}: base weight {base_weight_name} has shape {base_shape:?}, \
             not a 2-D projection"
        );
    }
    if out_features != base_shape[0] {
        anyhow::bail!(
            "LoRA adapter layer {layer}: the pair produces {out_features} output features but \
             the base weight {base_weight_name} has {} rows",
            base_shape[0]
        );
    }
    let scales_name = format!("{}.scales", base_weight_name.trim_end_matches(".weight"));
    if let Some(scales) = base_weights.get(&scales_name) {
        // Quantized base: in_features is hidden by the packing; pin it via
        // the group count.
        let s_shape = mlxcel_core::array_shape(scales);
        let groups = s_shape.get(1).copied().unwrap_or(0);
        let plausible_group = groups > 0
            && in_features % groups == 0
            && matches!(in_features / groups, 16 | 32 | 64 | 128);
        if !plausible_group {
            anyhow::bail!(
                "LoRA adapter layer {layer}: the pair consumes {in_features} input features, \
                 which does not match the quantized base weight {base_weight_name} (scales \
                 shape {s_shape:?})"
            );
        }
    } else if in_features != base_shape[1] {
        anyhow::bail!(
            "LoRA adapter layer {layer}: the pair consumes {in_features} input features but the \
             base weight {base_weight_name} has {} columns",
            base_shape[1]
        );
    }
    Ok(())
}

/// Load every adapter in `set` and stage its per-layer terms for the model
/// construction about to run on this thread (see
/// [`mlxcel_core::runtime_lora::stage`]). Unmatched tensors warn with the
/// same posture as the fused path (#1328 owns making both strict); shape
/// mismatches are hard errors, as they are when fusing.
pub fn stage_runtime_adapters(base_weights: &WeightMap, set: &RuntimeLoraSet) -> Result<()> {
    use std::collections::HashMap;

    let mut pending: HashMap<String, Vec<PendingLoraTerm>> = HashMap::new();
    for adapter in &set.adapters {
        let adapter_weights = load_adapter_weights(&adapter.spec.path)?;
        // Group `.lora_a` / `.lora_b` tensors by their base layer name,
        // the same grouping the fused path applies.
        let mut pairs: HashMap<String, (Option<&_>, Option<&_>)> = HashMap::new();
        for (name, weight) in &adapter_weights {
            if let Some(base) = name.strip_suffix(".lora_a") {
                pairs.entry(base.to_string()).or_default().0 = Some(weight);
            } else if let Some(base) = name.strip_suffix(".lora_b") {
                pairs.entry(base.to_string()).or_default().1 = Some(weight);
            }
        }
        let mut staged_layers = 0usize;
        for (layer, (a, b)) in pairs {
            let (Some(a), Some(b)) = (a, b) else {
                tracing::warn!(
                    "Incomplete LoRA pair for {layer}: missing lora_a or lora_b (adapter {})",
                    adapter.spec.path.display()
                );
                continue;
            };
            let base_weight_name = find_base_weight_name(&layer, base_weights)?;
            if !base_weights.contains_key(&base_weight_name) {
                tracing::warn!(
                    "Base weight not found for LoRA layer {layer}: tried {base_weight_name} \
                     (adapter {})",
                    adapter.spec.path.display()
                );
                continue;
            }
            let a_shape = mlxcel_core::array_shape(a);
            let b_shape = mlxcel_core::array_shape(b);
            let (orientation, in_features, out_features) =
                detect_orientation(&layer, &a_shape, &b_shape)?;
            validate_pair_shapes(
                &layer,
                base_weights,
                &base_weight_name,
                in_features,
                out_features,
            )?;
            let prefix = base_weight_name.trim_end_matches(".weight").to_string();
            // Normalize both on-disk orientations into the forward
            // orientation the layers consume: a_t = [in, rank],
            // b_t = [rank, out].
            let (a_t, b_t) = match orientation {
                PairOrientation::MlxLm => (mlxcel_core::copy(a), mlxcel_core::copy(b)),
                PairOrientation::Peft => (mlxcel_core::transpose(a), mlxcel_core::transpose(b)),
            };
            pending.entry(prefix).or_default().push(PendingLoraTerm {
                handle: adapter.handle.clone(),
                base_scale: adapter.base_scale,
                a_t,
                b_t,
            });
            staged_layers += 1;
        }
        tracing::info!(
            "Staged runtime LoRA adapter {} for {} layers (rank scale {:.3}, user scale {:.2})",
            adapter.spec.path.display(),
            staged_layers,
            adapter.base_scale,
            adapter.handle.get(),
        );
    }
    mlxcel_core::runtime_lora::stage(pending);
    Ok(())
}

/// Report (and clear) whatever model construction did not claim.
pub fn finish_runtime_staging() {
    for layer in mlxcel_core::runtime_lora::drain_unclaimed() {
        tracing::warn!(
            "Runtime LoRA terms for layer {layer} were not claimed by the model's constructors; \
             the adapter does not apply to that layer (the model builds it outside the \
             UnifiedLinear/Linear/FusedQKV constructors)"
        );
    }
}
